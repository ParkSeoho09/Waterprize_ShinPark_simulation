"""반복 실행 도구: 병렬 실행, 격자·시간 간격 독립성 점검, 라틴 하이퍼큐브 몬테카를로(불확실성·민감도),
실험 CSV 보정(scipy), 사진 비교(IoU·도달거리). 모두 같은 엔진(engine.run_scenario)을 반복해서 부른다.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ───────────────────────────── 병렬 실행 ─────────────────────────────────
def _worker(args):
    import sys
    import time as _t
    sys.path.insert(0, HERE)
    from engine import StopRequested, run_scenario
    from params import Common
    cdict, sc, ev = args[:3]
    idx, shared = (args[3], args[4]) if len(args) >= 5 else (None, None)
    c = Common.from_dict(cdict)
    for k, v in cdict.items():          # Common.from_dict가 모르는 추가 속성(예: heterogeneity)
        if not hasattr(c, k):
            setattr(c, k, v)
    prog = stop = None
    if shared is not None:
        last = [0.0]

        def prog(f, msg):               # 작업별 진행률을 부모에게(0.5초에 한 번만 — 프로세스 간 통신 비용)
            now = _t.time()
            if now - last[0] > 0.5 or f >= 1.0:
                last[0] = now
                try:
                    shared[idx] = (float(f), str(msg))
                except Exception:       # noqa: BLE001
                    pass

        def stop():
            try:
                return bool(shared.get("stop", False))
            except Exception:           # noqa: BLE001
                return False
    try:
        return run_scenario(c, sc, prog, stop, ev)
    except StopRequested:
        return None
    except Exception as e:      # noqa: BLE001
        return {"status": "ERROR", "fail_reason": str(e), "name": getattr(sc, "name", "대조군")}


def mem_available_mb():
    """지금 쓸 수 있는 메모리(MB)와 전체 메모리(MB). 알 수 없으면 (None, None). psutil 없이 Windows·Linux 지원."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / 2 ** 20, vm.total / 2 ** 20
    except Exception:
        pass
    try:
        import sys
        if sys.platform.startswith("win"):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            # 32비트 파이썬은 가상 주소 공간(약 2 GB)이 실제 한도
            return min(m.ullAvailPhys, m.ullAvailVirtual) / 2 ** 20, m.ullTotalPhys / 2 ** 20
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = float(v.split()[0]) / 1024
        return info.get("MemAvailable"), info.get("MemTotal")
    except Exception:
        return None, None


def run_mem_mb(n_cells):
    """계산 하나(대조군 또는 비교군)가 쓰는 메모리 추정(MB). 시험: 보통 격자(5.3만 셀) 약 270 MB,
    새 프로세스 시작(넘파이·사이파이 등)에 약 100 MB. 결과(스냅숏·프레임)와 여유를 더해 셀당 6 KB로 잡는다."""
    return 120.0 + 6.0 * n_cells / 1000.0


def safe_cores(n_tasks, n_cells, want=None):
    """메모리를 넘지 않는 병렬 프로세스 수. want(사용자 지정)가 있으면 그것을 넘지 않게만 줄인다."""
    ncpu = os.cpu_count() or 1
    k = max(1, min(ncpu - 1, 4, n_tasks)) if want is None else max(1, int(want))
    avail, _ = mem_available_mb()
    if avail:
        k = max(1, min(k, int(0.6 * avail // run_mem_mb(n_cells))))
    return k


def run_many(tasks, cores=1, progress: Optional[Callable] = None, stop: Optional[Callable] = None):
    """tasks: [(Common, Scenario|None, eval_start)] → 결과 목록(같은 순서). cores>1이면 프로세스 병렬."""
    payload = [(t[0].to_dict() if hasattr(t[0], "to_dict") else t[0], t[1], t[2]) for t in tasks]
    for (c, _, _), p in zip(tasks, payload):
        if hasattr(c, "heterogeneity"):
            p[0]["heterogeneity"] = c.heterogeneity
    out = [None] * len(tasks)
    if cores <= 1 or len(tasks) == 1:
        for i, p in enumerate(payload):
            if stop and stop():
                break
            out[i] = _worker(p)
            if progress:
                progress((i + 1) / len(tasks), f"{i + 1}/{len(tasks)}")
        return out
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    ctx = mp.get_context("spawn")
    # 프로세스 여러 개가 각자 PHREEQC 스레드를 띄우면 코어보다 많아진다 → 프로세스당 스레드 수를 나눠 준다(자식이 물려받음)
    old_thr = os.environ.get("WP_CHEM_THREADS")
    # 작업 프로세스는 제출할 때 생기므로 실행이 끝날 때까지 값을 유지했다가 되돌린다
    os.environ["WP_CHEM_THREADS"] = str(max(1, min(4, ((os.cpu_count() or 1) - 1) // max(1, int(cores)))))
    names = [getattr(t[1], "name", None) or "대조군" for t in tasks]
    mgr = None
    shared = None
    try:
        try:                            # 작업별 진행률·중단 신호를 주고받는 공유 사전(실패하면 완료 개수만 표시)
            mgr = ctx.Manager()
            shared = mgr.dict()
            shared["stop"] = False
        except Exception:               # noqa: BLE001
            mgr = shared = None
        with ProcessPoolExecutor(max_workers=int(cores), mp_context=ctx) as ex:
            futs = {ex.submit(_worker, (p[0], p[1], p[2], i, shared) if shared is not None else p): i
                    for i, p in enumerate(payload)}
            pending = set(futs)
            done = 0
            while pending:
                fin, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in fin:
                    out[futs[f]] = f.result()
                    done += 1
                if stop and stop():
                    if shared is not None:
                        shared["stop"] = True
                    for g in pending:
                        g.cancel()
                    for f in pending:       # 이미 돌던 작업은 중단 신호를 받고 곧 끝난다
                        try:
                            out[futs[f]] = f.result(timeout=60)
                        except Exception:   # noqa: BLE001
                            pass
                    break
                if progress:
                    parts, tot = [], 0.0
                    for f, i in futs.items():
                        if f.done():
                            fr = 1.0
                        else:
                            try:
                                fr, msg = shared.get(i, (0.0, "대기")) if shared is not None else (0.0, "")
                            except Exception:   # noqa: BLE001
                                fr = 0.0
                        tot += fr
                        parts.append(f"{names[i]} {100 * fr:.0f}%")
                    progress(tot / len(tasks), f"{done}/{len(tasks)} 완료 · " + " · ".join(parts))
    finally:
        if mgr is not None:
            try:
                mgr.shutdown()
            except Exception:   # noqa: BLE001
                pass
        if old_thr is None:
            os.environ.pop("WP_CHEM_THREADS", None)
        else:
            os.environ["WP_CHEM_THREADS"] = old_thr
    return out


# ───────────────────────────── 요약 값 ───────────────────────────────────
def key_quantities(res, ctrl=None, c=None, ev=None):
    """독립성·불확실성 비교용 핵심량."""
    import metrics as M
    D = res["daily"]
    q = {}
    if res.get("status") not in ("OK",):
        return {"status": res.get("status")}
    last = res["snaps"][-1] if res.get("snaps") else None
    g = res["grid"]
    if last is not None:
        act = res["active"]
        h = np.where(act, last["h"], np.nan) if "h" in last else None
        q["h_mean_m"] = float(np.nanmean(h)) if h is not None else np.nan
        prof = np.nanmean(np.where(act, last["NO3"], np.nan).reshape(-1, g["nz"]), axis=0)
        q["NO3_profile"] = prof.tolist()
    q["vmax_mday"] = float(res.get("vmax", np.nan)) * 86400 if res.get("vmax") is not None else np.nan
    w = slice(-7, None)
    q["Cout_last7"] = float(np.nanmean(D["Cout_mgN"][w])) if len(D["Cout_mgN"]) else np.nan
    q["SO4_last7"] = float(np.nanmean(D["SO4_out_mg"][w])) if len(D["SO4_out_mg"]) else np.nan
    q["pH_min"] = float(np.nanmin(D["pH_min"])) if len(D.get("pH_min", [])) and np.isfinite(D["pH_min"]).any() else np.nan
    q["Kreac_end"] = float(D["Kreac"][-1]) if len(D["Kreac"]) else np.nan
    q["Keff_end"] = float(D["Keff"][-1]) if len(D["Keff"]) else np.nan
    q["N_to_gw_g"] = float(D["N_to_gw_g"][-1]) if len(D["N_to_gw_g"]) else np.nan
    mb = res.get("mass_balance", {})
    q["mb_max"] = float(max([abs(v) for k, v in mb.items() if isinstance(v, (int, float)) and k != "W_err_rel"] or [np.nan]))
    if ctrl is not None and c is not None:
        m = M.compute(res, ctrl, c, ev)
        for k in ("eta_pre", "eta_mass", "dpH_below", "r_cm", "V_eff_cm3", "Kreac_ret", "R", "SO4_ratio", "gw_improve_pct"):
            q[k] = m.get(k)
    return q


# ───────────────────────────── 격자·시간 간격 독립성 ───────────────────────
def independence(c, sc, factors=(1.0, 1.5, 2.0), dt_half=True, cores=1, progress=None, stop=None, out_dir=None):
    from engine import eval_start_from
    levels = []
    tasks = []
    for f in factors:
        cc = copy.deepcopy(c)
        cc.nx, cc.ny, cc.nz = int(round(c.nx * f)), int(round(c.ny * f)), int(round(c.nz * f))
        levels.append((f"격자 ×{f:g} ({cc.nx}×{cc.ny}×{cc.nz})", cc))
    if dt_half:
        cc = copy.deepcopy(c)
        cc.dt_factor = c.dt_factor * 0.5
        cc.flow_dt_max_h = c.flow_dt_max_h * 0.5
        cc.chem_dt_h = c.chem_dt_h * 0.5
        levels.append(("시간 간격 ×0.5 (기준 격자)", cc))
    for name, cc in levels:
        tasks.append((cc, None, None))
        if sc is not None:
            tasks.append((cc, sc, None))
    res = run_many(tasks, cores, progress, stop)
    table = []
    raw = {}
    i = 0
    for name, cc in levels:
        ctrl = res[i]; i += 1
        r = res[i] if sc is not None else ctrl
        if sc is not None:
            i += 1
        ev = eval_start_from(ctrl, cc.NO3_in_mgN) if ctrl and ctrl.get("status") == "OK" else None
        q = key_quantities(r, ctrl if sc is not None else None, cc, ev) if r and r.get("status") == "OK" else \
            {"status": (r or {}).get("status", "?"), "reason": (r or {}).get("fail_reason", "")}
        q["level"] = name
        q["cells"] = cc.nx * cc.ny * cc.nz
        q["wall_s"] = (r or {}).get("wall_s")
        table.append(q)
        raw[name] = {"daily": {k: np.asarray(v).tolist() for k, v in (r or {}).get("daily", {}).items()},
                     "status": (r or {}).get("status"), "grid": (r or {}).get("grid")}
    keys = ["Cout_last7", "SO4_last7", "pH_min", "Kreac_end", "Keff_end", "h_mean_m", "vmax_mday", "N_to_gw_g",
            "eta_pre", "eta_mass", "mb_max"]
    verdict = {}
    grid_rows = [t for t in table if t["level"].startswith("격자")]
    if len(grid_rows) >= 3:
        for k in keys:
            a, b, cc_ = (grid_rows[0].get(k), grid_rows[1].get(k), grid_rows[2].get(k))
            try:
                d1 = abs(b - a); d2 = abs(cc_ - b)
                scale = max(abs(cc_), 1e-12)
                verdict[k] = {"변화(×1→×1.5)": d1 / scale, "변화(×1.5→×2)": d2 / scale,
                              "판정": "수렴(변화 <5%, 감소)" if (d2 / scale < 0.05 and d2 <= d1 * 1.05) else
                                     ("변화 작음(<5%)" if d2 / scale < 0.05 else "미수렴 — 더 세밀한 격자 필요")}
            except Exception:
                verdict[k] = {"판정": "판정 불가"}
    trow = [t for t in table if t["level"].startswith("시간")]
    if trow and grid_rows:
        for k in keys:
            try:
                a, b = grid_rows[0].get(k), trow[0].get(k)
                verdict.setdefault(k, {})["시간 간격 ½ 변화"] = abs(b - a) / max(abs(a), 1e-12)
            except Exception:
                pass
    out = {"table": table, "verdict": verdict, "when": time.strftime("%Y-%m-%d %H:%M"), "raw": raw,
           "scenario": getattr(sc, "name", "대조군")}
    if out_dir:
        with open(os.path.join(out_dir, f"independence_{int(time.time())}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, default=float)
    return out


# ───────────────────────────── 불확실성(LHS 몬테카를로) ───────────────────
UQ_PARAMS = {
    # 키: (설명, 기본 하한, 기본 상한, 로그 여부)
    "k_half_a": ("표면 반-차수 상수 k½a ((mg/L)^½·m/h) — Sierra-Alvarez 2007 ~ Koenig & Liu 2001", 1.5e-3, 4.7e-3, True),
    "wfps_thr": ("탈질 수분 함수 문턱 포화도 (-) — 0.6(Grundmann & Rolston) ~ 0.8", 0.55, 0.8, False),
    "k_ox_a": ("호기성 황 산화 표면 속도 (mol S/m²/s) — Watkinson 1989: 48–76 µg/cm²/일 ±", 1.0e-7, 3.0e-7, True),
    "S_d_media": ("충전재 황 입경 (m)", 2.8e-3, 5.6e-3, True),
    "alpha_L": ("종분산 길이 (m)", 0.003, 0.03, True),
    "lambda_filt": ("입자 포획 계수 λ (1/m)", 300.0, 3000.0, True),
    "phi_reac": ("반응재 초기 공극률 (-)", 0.32, 0.48, False),
    "Ks_mult": ("토양 Ks 배율 (-)", 0.5, 2.0, True),
    "bio_density": ("생물막 밀도 (kg/m³)", 10.0, 100.0, True),
}


def apply_sample(c, key, val):
    if key == "phi_reac":
        c.phi_reac_override = float(val)
    elif key == "Ks_mult":
        from params import SOIL_TEXTURES
        base = c.Ks_override or SOIL_TEXTURES[c.texture]["Ks"]
        c.Ks_override = float(base * val)
    else:
        c.params[key] = float(val)


def lhs_mc(c, sc, ranges: Dict[str, tuple], n=20, cores=1, seed=0, progress=None, stop=None):
    from scipy.stats import qmc, spearmanr
    from engine import eval_start_from
    keys = list(ranges)
    U = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    samples = []
    for u in U:
        s = {}
        for j, k in enumerate(keys):
            lo, hi, lg = ranges[k]
            s[k] = float(np.exp(np.log(lo) + u[j] * (np.log(hi) - np.log(lo)))) if lg else float(lo + u[j] * (hi - lo))
        samples.append(s)
    tasks = []
    for s in samples:
        cc = copy.deepcopy(c)
        for k, v in s.items():
            apply_sample(cc, k, v)
        tasks.append((cc, None, None))
        tasks.append((cc, sc, None))
    res = run_many(tasks, cores, progress, stop)
    import metrics as M
    rows = []
    for i, s in enumerate(samples):
        ctrl, r = res[2 * i], res[2 * i + 1]
        if not ctrl or not r or ctrl.get("status") != "OK" or r.get("status") != "OK":
            rows.append(dict(s, status=(r or {}).get("status", "?")))
            continue
        cc = copy.deepcopy(c)
        for k, v in s.items():
            apply_sample(cc, k, v)
        ev = eval_start_from(ctrl, cc.NO3_in_mgN)
        m = M.compute(r, ctrl, cc, ev)
        row = dict(s, status="OK")
        for _, key, _ in M.SUMMARY_ROWS:
            v = m.get(key)
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                row[key] = float(v)
        rows.append(row)
    ok = [r for r in rows if r.get("status") == "OK"]
    mkeys = [k for k in (ok[0].keys() if ok else []) if k not in keys and k != "status"]
    stats = {}
    sens = {}
    for mk in mkeys:
        v = np.array([r.get(mk, np.nan) for r in ok], float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        se = sd / np.sqrt(len(v))
        stats[mk] = {"평균": mean, "표준편차": sd, "95% 구간(분포 2.5~97.5%)": (lo, hi),
                     "평균의 95% 신뢰구간": (mean - 1.96 * se, mean + 1.96 * se), "n": int(len(v))}
        sens[mk] = {}
        for k in keys:
            x = np.array([r[k] for r in ok], float)
            y = np.array([r.get(mk, np.nan) for r in ok], float)
            m_ = np.isfinite(y)
            if m_.sum() >= 4 and np.std(y[m_]) > 0:
                rho, p = spearmanr(x[m_], y[m_])
                sens[mk][k] = {"순위상관 ρ": float(rho), "p": float(p)}
    return {"samples": rows, "stats": stats, "sensitivity": sens, "params": {k: ranges[k] for k in keys},
            "n_ok": len(ok), "n": n, "method": "라틴 하이퍼큐브 + Spearman 순위상관", "seed": seed}


# ───────────────────────────── 실험 데이터 보정 ───────────────────────────
OBS_VARS = {"유출 NO3-N (mg/L)": "Cout_mgN", "유출 SO4 (mg/L)": "SO4_out_mg", "유출 pH": "pH_out",
            "수분 θ (깊이별)": "theta", "압력수두 h (m, 깊이별)": "h", "유효 K (m/s)": "Keff"}


def model_values(res, obs):
    """obs DataFrame 행마다 같은 시점·깊이의 모델 값."""
    D = res["daily"]
    out = []
    g = res["grid"]
    for _, r in obs.iterrows():
        var = OBS_VARS.get(r["변수"], r["변수"])
        t = float(r["시간(일)"])
        if var in ("theta", "h"):
            snaps = res.get("snaps", [])
            if not snaps:
                out.append(np.nan); continue
            ts = np.array([s["op_day"] for s in snaps])
            k = int(np.clip(round(float(r.get("깊이(m)", 0.0)) / g["dz"] - 0.5), 0, g["nz"] - 1))
            vals = np.array([float(np.nanmean(np.where(res["active"][:, :, k], s[var][:, :, k], np.nan))) for s in snaps])
            out.append(float(np.interp(t, ts, vals)))
        else:
            out.append(float(np.interp(t, D["day"], D[var])) if len(D["day"]) else np.nan)
    return np.array(out)


def fit_stats(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    if not m.any():
        return {"n": 0}
    o, s = obs[m], sim[m]
    e = s - o
    nse = 1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2) if np.sum((o - o.mean()) ** 2) > 0 else np.nan
    return {"n": int(m.sum()), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "편향": float(np.mean(e)), "NSE": float(nse)}


def calibrate(c, sc, obs, params: Dict[str, tuple], max_nfev=12, progress=None, stop=None):
    """obs: 열 [시간(일), 변수, 깊이(m), 값, 구분(보정/검증)]. params: {키: (하한, 상한, 로그)}.
    최소제곱(scipy.optimize.least_squares, 변수별 표준편차로 정규화). 반환: 최적 파라미터·통계."""
    from scipy.optimize import least_squares
    from engine import run_scenario
    cal = obs[obs["구분"].astype(str).str.contains("보정")]
    val = obs[obs["구분"].astype(str).str.contains("검증")]
    keys = list(params)
    lo = np.array([np.log(params[k][0]) if params[k][2] else params[k][0] for k in keys])
    hi = np.array([np.log(params[k][1]) if params[k][2] else params[k][1] for k in keys])
    x0 = 0.5 * (lo + hi)
    for j, k in enumerate(keys):
        cur = c.params.get(k) if k in c.params else None
        if cur is not None:
            x0[j] = np.clip(np.log(cur) if params[k][2] else cur, lo[j], hi[j])
    scale = cal.groupby("변수")["값"].transform(lambda v: max(float(np.std(v)), 1e-6 * max(abs(float(np.mean(v))), 1e-12)) or 1.0).values
    hist = []

    def cfg(x):
        cc = copy.deepcopy(c)
        for j, k in enumerate(keys):
            apply_sample(cc, k, float(np.exp(x[j]) if params[k][2] else x[j]))
        return cc

    def resid(x):
        if stop and stop():
            raise RuntimeError("중단")
        r = run_scenario(cfg(x), sc)
        if r.get("status") != "OK":
            return np.full(len(cal), 1e3)
        sim = model_values(r, cal)
        e = (sim - cal["값"].values) / scale
        e = np.where(np.isfinite(e), e, 1e3)
        hist.append({"x": [float(v) for v in x], "cost": float(0.5 * np.sum(e ** 2))})
        if progress:
            progress(min(1.0, len(hist) / max_nfev), f"평가 {len(hist)}회, 비용 {hist[-1]['cost']:.3g}")
        return e
    sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=max_nfev, diff_step=0.05)
    best = {k: float(np.exp(sol.x[j]) if params[k][2] else sol.x[j]) for j, k in enumerate(keys)}
    r = run_scenario(cfg(sol.x), sc)
    out = {"best": best, "success": bool(sol.success), "message": str(sol.message), "nfev": int(sol.nfev),
           "history": hist, "label": "보정"}
    for name, part in (("보정", cal), ("검증", val)):
        if len(part):
            sim = model_values(r, part)
            out[f"stats_{name}"] = {v: fit_stats(part["값"].values[part["변수"].values == v],
                                                 sim[part["변수"].values == v]) for v in part["변수"].unique()}
    out["result"] = r
    return out


# ───────────────────────────── 사진 비교 ─────────────────────────────────
def photo_mask(img_bytes, calib, threshold, channel="밝기"):
    """사진 → 농도 → 이진 마스크. calib: [(픽셀값, 농도), ...] 두 개 이상(선형 보정선).
    channel: '밝기' | 'R' | 'G' | 'B'."""
    from PIL import Image
    import io
    im = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB")).astype(float)
    ch = {"R": im[..., 0], "G": im[..., 1], "B": im[..., 2]}.get(channel, im.mean(axis=2))
    p = np.array([a for a, _ in calib], float); cval = np.array([b for _, b in calib], float)
    A = np.vstack([p, np.ones_like(p)]).T
    slope, icpt = np.linalg.lstsq(A, cval, rcond=None)[0]
    conc = slope * ch + icpt
    return conc, conc >= threshold, (float(slope), float(icpt))


def photo_compare(img_bytes, px_per_cm, origin_px, calib, threshold, model_section, dx, dz, channel="밝기"):
    """model_section: (nx, nz) 시뮬 단면(같은 변수·단위). origin_px: 사진에서 흙수조 왼쪽 지표 모서리 (x, y) 픽셀.
    반환: IoU, 도달 깊이(사진·모델), 오차(cm)."""
    conc, mask, line = photo_mask(img_bytes, calib, threshold, channel)
    H, W = mask.shape
    nx, nz = model_section.shape
    # 모델 셀 중심 → 사진 픽셀
    xs = (np.arange(nx) + 0.5) * dx * 100 * px_per_cm + origin_px[0]
    zs = (np.arange(nz) + 0.5) * dz * 100 * px_per_cm + origin_px[1]
    XI = np.clip(np.round(xs).astype(int), 0, W - 1)
    ZI = np.clip(np.round(zs).astype(int), 0, H - 1)
    inside = (xs[:, None] >= 0) & (xs[:, None] < W) & (zs[None, :] >= 0) & (zs[None, :] < H)
    photo_on_grid = mask[ZI[None, :], XI[:, None]] & inside
    model_mask = (model_section >= threshold) & inside
    inter = (photo_on_grid & model_mask).sum()
    union = (photo_on_grid | model_mask).sum()
    iou = float(inter / union) if union else float("nan")

    def reach(m):
        k = np.where(m.any(axis=0))[0]
        return float((k.max() + 1) * dz * 100) if len(k) else 0.0
    rp, rm = reach(photo_on_grid), reach(model_mask)
    return {"IoU": iou, "도달 깊이 사진(cm)": rp, "도달 깊이 모델(cm)": rm, "도달 깊이 오차(cm)": rm - rp,
            "보정선(농도=a·값+b)": line, "photo_mask": photo_on_grid, "model_mask": model_mask}


# ───────────────────────────── v4.1 조건 비교(스윕)·최적화 ─────────────────────────────
SWEEP_VARS = {
    "황 투입량 (g)": ("S_dose_g", "200, 400, 800, 1600"),
    "반응 구역 포화도 (지정, 0~1)": ("moist_set", "0.4, 0.6, 0.7, 0.8, 0.9, 1.0"),
    "황 입경 (mm)": ("S_d_mm", "0.05, 0.2, 1, 4"),
    "황:물 배합 (황 1 g당 물 g)": ("water_per_S", "1, 2.5, 5, 10"),
    "수분 유지 재료 비율 (건조 고형분 w/w)": ("ret_frac", "0, 0.02, 0.05, 0.09"),
}
OPT_METRICS = {"장치 제거율 (%)": "dev_removal", "지하수 유입 감소율 (%, 대조군 대비)": "sys_removal"}


def _strip(r):
    """스윕 결과는 지표 계산에 필요한 것만 남긴다(장면·프레임 제외 — 메모리)."""
    if r is None:
        return None
    keep = ("status", "fail_reason", "name", "is_control", "daily", "ledger", "ledger0", "S0_g", "diag", "refills",
            "design", "warnings")
    return {k: r.get(k) for k in keep}


def sweep(common, scen_objs, var_label, values, retainer=None, cores=1, horizons=(90.0, 180.0),
          progress=None, stop=None):
    """조건 하나를 여러 값으로 바꾸며 비교군마다 계산. 지하수 흐름·지층·유입 농도(공통 조건)는 그대로(통제변수).
    대조군은 반응재가 없어 모든 조건에서 같으므로 한 번만 계산. 반환: {'rows': [...], 'ctrl_ok': bool}."""
    import design as DZ
    from metrics import core_metrics
    key = SWEEP_VARS[var_label][0]
    base = copy.deepcopy(common)
    base.light = True
    base.n_snapshots = 2
    tasks = [(base, None, None)]
    labels = [("대조군", None)]
    for v in values:
        for so in scen_objs:
            so2 = copy.copy(so)
            rg = dict(so.reagent or {})
            d = dict(DZ.default_design(rg.get("form")))
            d.update({k_: v_ for k_, v_ in (rg.get("design") or {}).items() if v_ is not None})
            d["on"] = True
            c2 = base
            if key == "moist_set":
                c2 = copy.deepcopy(base)
                c2.moist_mode = "반응 구역 포화도 지정"
                c2.moist_set = float(v)
            else:
                d[key] = float(v)
                if key == "ret_frac" and retainer:
                    d["retainer"] = retainer
            rg["design"] = d
            so2.reagent = rg
            tasks.append((c2, so2, None))
            labels.append((so.name, float(v)))
    res = run_many(tasks, cores, progress, stop)
    res = [_strip(r) for r in res]
    ctrl = res[0]
    rows = []
    for (nm, v), r in zip(labels[1:], res[1:]):
        row = {"비교군": nm, "값": v, "상태": (r or {}).get("status", "중단"),
               "실패 이유": (r or {}).get("fail_reason", "")}
        if r is not None and r.get("status") == "OK":
            for hz in list(horizons) + [None]:
                m = core_metrics(r, ctrl, common, hz)
                if m is None:
                    continue
                tag = f"{m['day']:.0f}일" if hz else "끝"
                for kk in ("dev_removal", "sys_removal", "dev_Cin", "dev_Cout", "out_C", "S0_g", "S_used_g", "S_ox_g",
                           "S_left_g", "N_removed_g", "N_per_kgS", "N_per_kgS_used", "sat_mean", "bypass_pct"):
                    row[f"{kk}@{tag}"] = m.get(kk)
        rows.append(row)
    return {"rows": rows, "ctrl_ok": bool(ctrl and ctrl.get("status") == "OK"), "var": var_label, "key": key,
            "values": list(values), "horizons": list(horizons)}


def score_rows(rows, metric_key, tag, w_rem=0.5, w_eff=0.3, w_dose=0.2):
    """다목적 점수: 제거율(높을수록) · 황 투입량 대비 제거 N(높을수록) · 황 투입량(적을수록).
    각 항목은 같은 표 안의 최댓값으로 나눠 0~1로 맞춘 뒤 가중합. 파레토(제거율↑, 황 투입량↓) 표시."""
    import math
    def g(r, k):
        v = r.get(f"{k}@{tag}")
        return v if (v is not None and isinstance(v, (int, float)) and math.isfinite(v)) else None
    ok = [r for r in rows if g(r, metric_key) is not None]
    if not ok:
        return rows
    mr = max(max(g(r, metric_key) for r in ok), 1e-9)
    me = max(max((g(r, "N_per_kgS") or 0.0) for r in ok), 1e-9)
    md = max(max((g(r, "S0_g") or 0.0) for r in ok), 1e-9)
    for r in ok:
        rem = max(g(r, metric_key), 0.0) / mr
        eff = max(g(r, "N_per_kgS") or 0.0, 0.0) / me
        dose = (g(r, "S0_g") or 0.0) / md
        r["점수"] = w_rem * rem + w_eff * eff - w_dose * dose
    for r in ok:
        rr, dd = g(r, metric_key), g(r, "S0_g") or 0.0
        r["파레토"] = not any((g(o, metric_key) >= rr and (g(o, "S0_g") or 0.0) <= dd) and
                            (g(o, metric_key) > rr or (g(o, "S0_g") or 0.0) < dd) for o in ok if o is not r)
    return rows
