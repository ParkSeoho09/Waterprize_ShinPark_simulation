"""불포화 탈질의 수분 의존(F9) — 서로 배타적인 세 모드와 보정.

포화도 정의(코드 전체 통일, F9-4):
  S_wfps = θ/φ              (= WFPS, 물이 찬 공극 비율). NEMIS 계열 임계값은 이 기준.
  Se     = (θ − θr)/(θs − θr)  (유효포화도, van Genuchten 곡선용). 탈질 수분 함수에는 쓰지 않는다.

속도식(F9-1, 표면적 반-차수):
  r = k½a · a_s · f_moist(S_wfps) · f_O2(C_O2) · f_pH · f_T · √C        [mg N/L/h, 물 부피 기준이면 /θ]

모드(F9-2, 같은 억제를 두 번 세지 않도록 서로 배타):
  A 기계론(기본): f_moist = f_w(S_wfps), f_O2 = K_O2/(K_O2 + C_O2)
       f_w = S^p — 황 표면 중 물막이 덮은 면적 분율. 유도: 반-차수 식은 생물막 단위 면적당 속도이고,
       펜듈러 링의 부피 V에 대해 접촉 면적은 V^(2/3)로 커진다 → p = 2/3(기본). 물막이 입자 전면을 덮으면 p → 1/3,
       부피 정비례면 p → 1. 값은 흙수조 보정으로 정한다(가정·유도).
       대안: 젖음각 θc의 기하 관계(단극 (1−cosθc)/2, 양극 1−cosθc, 적도 cosθc) — 상수 분율.
  B 경험식: f_moist = NEMIS 계열(임계 포화도 s_t 이하 0). f_O2는 탈질에 곱하지 않는다
       (NEMIS는 산소 제한을 수분으로 대신 나타낸 경험식). 호기성 황 산화의 O2 소모는 그대로 계산.
       멱함수형 ((S − s_t)/(1 − s_t))^w, 지수형 (e^{β(S−s_t)} − 1)/(e^{β(1−s_t)} − 1).
  C 상한: f_moist = 1, f_O2 = 1.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

MODES = ["A 기계론(f_w·f_O2)", "B 경험식(NEMIS 계열)", "C 상한(억제 없음)", "D 혐기 미소부위(Smith 1980, 옵션)"]
FW_FORMS = ["S^p", "젖음각-단극", "젖음각-양극", "젖음각-적도"]
NEMIS_FORMS = ["멱함수", "지수형"]


def mode_letter(mode: str) -> str:
    return (mode or "A")[0].upper()


def S_wfps(theta, phi):
    """물이 찬 공극 비율 S = θ/φ (WFPS). NEMIS 임계값은 이 기준."""
    return np.clip(np.asarray(theta, float) / np.maximum(np.asarray(phi, float), 1e-12), 0.0, 1.0)


def Se_vg(theta, theta_r, theta_s):
    """유효포화도 Se = (θ − θr)/(θs − θr) — van Genuchten 곡선용(탈질 수분 함수에는 쓰지 않음)."""
    return np.clip((np.asarray(theta, float) - theta_r) / np.maximum(theta_s - theta_r, 1e-12), 0.0, 1.0)


def f_w(S, p=2.0 / 3.0, form="S^p", theta_c_deg=None):
    """황 표면의 물막 접촉 분율(모드 A)."""
    S = np.clip(np.asarray(S, float), 0.0, 1.0)
    if form == "S^p":
        return S ** float(p)
    if theta_c_deg is None:
        raise ValueError("젖음각 환산을 쓰려면 젖음각(°)을 입력해야 합니다.")
    c = np.cos(np.radians(float(theta_c_deg)))
    val = {"젖음각-단극": (1 - c) / 2, "젖음각-양극": 1 - c, "젖음각-적도": c}[form]
    return np.full_like(S, float(np.clip(val, 0.0, 1.0)))


def f_nemis(S, thr=0.70, w=1.74, form="멱함수", beta=None):
    """NEMIS 계열 수분 함수(모드 B). S = WFPS."""
    S = np.clip(np.asarray(S, float), 0.0, 1.0)
    x = np.clip((S - thr) / max(1.0 - thr, 1e-12), 0.0, 1.0)
    if form == "멱함수":
        return x ** float(w)
    if beta is None:
        raise ValueError("지수형 NEMIS를 쓰려면 β를 입력해야 합니다(근거 있는 기본값 없음).")
    b = float(beta)
    if abs(b) < 1e-12:
        return x
    d = S - thr
    return np.where(d > 0, np.expm1(b * np.maximum(d, 0)) / np.expm1(b * (1 - thr)), 0.0)


def f_O2(C_O2, K):
    """Monod 억제 K/(K + C). 단위는 C와 K가 같아야 함(mol/m³)."""
    return K / (K + np.maximum(np.asarray(C_O2, float), 0.0))


def factors(mode: str, S, C_O2, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(f_moist, f_O2_탈질) — 모드별로 정확히 하나의 억제만. cfg: p, fw_form, theta_c, thr, w, nemis_form, beta, K_O2."""
    m = mode_letter(mode)
    S = np.asarray(S, float)
    one = np.ones_like(S)
    if m == "A":
        fm = f_w(S, cfg.get("p", 2 / 3), cfg.get("fw_form", "S^p"), cfg.get("theta_c"))
        fo = f_O2(C_O2, cfg["K_O2"]) if C_O2 is not None else one
        return fm, fo
    if m == "B":
        return f_nemis(S, cfg.get("thr", 0.70), cfg.get("w", 1.74), cfg.get("nemis_form", "멱함수"), cfg.get("beta")), one
    if m == "D":
        return f_anoxic(S, cfg), one
    return one, one


def f_anoxic(S, cfg):
    """모드 D(F8 옵션): 흙덩이(반지름 a) 안 혐기 부피 분율 (r_c/a)³. 구 확산–0차 소비 정상해(Smith 1980 형, 원문 확인 필요):
    겉면 O2 C_s, 흙덩이 안 유효 확산 D_e = D_w·(θ/φ)·φ^{4/3}(Millington–Quirk 수용액형, 가정), 소비 R.
    혐기 핵이 생기는 조건 C_s < R·a²/(6 D_e). 핵 반지름 x = r_c/a: 1 − 3x² + 2x³ = 6·D_e·C_s/(R·a²).
    탈질은 혐기 부피에서만 일어난다고 보고 f_moist = x³ 로 둔다(O2 억제는 이 분율에 이미 포함 → f_O2 곱하지 않음)."""
    a, R = cfg.get("agg_a"), cfg.get("agg_R")
    if a is None or R is None:
        raise ValueError("모드 D는 흙덩이 반지름과 O2 소비율 입력이 필요합니다.")
    S = np.clip(np.asarray(S, float), 0.0, 1.0)
    phi = float(cfg.get("phi", 0.4)); Dw = 2.1e-9; Cs = float(cfg.get("C_O2_air", 0.27))   # mol/m³(공기 포화, 20 °C)
    De = Dw * S * phi ** (4.0 / 3.0)
    rhs = np.clip(6 * De * Cs / (R * a * a), 0.0, None)
    x = np.zeros_like(S)
    ok = rhs < 1.0
    # 1 − 3x² + 2x³ = rhs 를 x∈[0,1]에서 이분법(단조 감소)
    lo = np.zeros(ok.sum()); hi = np.ones(ok.sum()); t = rhs[ok]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f = 1 - 3 * mid ** 2 + 2 * mid ** 3 - t
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    x[ok] = 0.5 * (lo + hi)
    return x ** 3


# ───────────────────────────── F9-6 필요 체류시간·두께 ─────────────────────────────
def residence_time_h(C0, target, k_eff):
    """반-차수 플러그 흐름: √C = √C0 − k·t/2 → t = 2√C0(1 − √(1 − 목표 제거율))/k. [h], k in (mg/L)^½/h."""
    if k_eff <= 0:
        return float("inf")
    return 2.0 * np.sqrt(C0) * (1.0 - np.sqrt(1.0 - target)) / k_eff


# ───────────────────────────── F9-7 포화도별 제거율로 보정 ─────────────────────────────
def removal_model(S, Cin, Q_L_h, V_L, k_a_a, mode, cfg, param, value):
    """보정용 모형: 균질 반응 구역(체적 V, 비표면적 a)을 지나는 플러그 흐름.
    물 부피 기준 속도 k_w = k½a·a·f/θ, 체류시간 τ = θV/Q → √Cin − √Cout = k½a·a·f·V/(2Q) (θ가 소거됨).
    k_a_a = k½a·a [(mg/L)^½/h, 벌크 부피 기준]. param ∈ {"p", "thr"} 를 value로 바꿔 계산."""
    c = dict(cfg); c[param] = value
    fm, _ = factors(mode, np.asarray(S, float), None, c)
    drop = k_a_a * fm * np.asarray(V_L, float) / (2.0 * np.asarray(Q_L_h, float))
    return np.maximum(np.sqrt(np.asarray(Cin, float)) - drop, 0.0) ** 2


def calibrate(rows: List[dict], k_a_a: float, mode: str, cfg: dict) -> Dict:
    """포화도별 제거율 표로 p(모드 A) 또는 임계 포화도(모드 B)를 최소제곱 추정.
    rows: dict(S, Cin, Q_L_h, V_L, Cout, use='보정'|'확인'). 보정 행이 3개 미만이면 추정하지 않는다(R6: 보정·확인 분리)."""
    m = mode_letter(mode)
    if m == "C":
        return {"ok": False, "reason": "모드 C(억제 없음)에는 추정할 수분 파라미터가 없습니다."}
    param, lo, hi = ("p", 1e-3, 3.0) if m == "A" else ("thr", 0.0, 0.99)
    cal = [r for r in rows if str(r.get("use", "보정")).startswith("보정")]
    chk = [r for r in rows if not str(r.get("use", "보정")).startswith("보정")]
    for r in rows:
        for k in ("S", "Cin", "Q_L_h", "V_L", "Cout"):
            if r.get(k) is None or not np.isfinite(float(r[k])):
                raise ValueError(f"보정 표에 빈 값이 있습니다({k}). 빈칸을 채우거나 그 행을 지우세요.")
    if len(cal) < 3:
        return {"ok": False, "reason": f"보정 행이 {len(cal)}개라 추정하지 않습니다(3개 이상 필요).",
                "n_cal": len(cal), "n_check": len(chk)}

    def sse(v):
        pred = removal_model([r["S"] for r in cal], [r["Cin"] for r in cal], [r["Q_L_h"] for r in cal],
                             [r["V_L"] for r in cal], k_a_a, mode, cfg, param, v)
        return float(np.sum((pred - np.array([r["Cout"] for r in cal], float)) ** 2))
    grid = np.linspace(lo, hi, 600)
    vals = np.array([sse(v) for v in grid])
    i = int(np.argmin(vals))
    a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(80):                                       # 황금분할로 다듬기
        m1 = b_ - (b_ - a_) / 1.618034
        m2 = a_ + (b_ - a_) / 1.618034
        if sse(m1) < sse(m2):
            b_ = m2
        else:
            a_ = m1
    best = 0.5 * (a_ + b_)
    out = {"ok": True, "param": param, "value": best, "sse": sse(best), "n_cal": len(cal), "n_check": len(chk)}
    if chk:
        pred = removal_model([r["S"] for r in chk], [r["Cin"] for r in chk], [r["Q_L_h"] for r in chk],
                             [r["V_L"] for r in chk], k_a_a, mode, cfg, param, best)
        obs = np.array([r["Cout"] for r in chk], float)
        out["check_rmse"] = float(np.sqrt(np.mean((pred - obs) ** 2)))
        out["check_pred"] = pred.tolist()
    return out


# ───────────────────────────── S3 Moon 외(2004) 칼럼 재현 ─────────────────────────────
# 문헌 값: 유입 30·40·60 mg N/L, 반-차수 상수 31.7·33.3·36.4e-3 mg^½ L^-½ min^-1 (×60 → (mg/L)^½/h), 제거율 > 90 %
# (Environ. Pollut. 129:499 — 값은 요약에서 옮김, 원문 확인 필요)
MOON2004 = [(30.0, 31.7e-3 * 60), (40.0, 33.3e-3 * 60), (60.0, 36.4e-3 * 60)]


def moon2004_check(col: dict, k_half_a: float, factor_ok: float = 1.5) -> dict:
    """칼럼 조건(col_L_m, col_D_m, col_Q_mL_min, col_phi, col_dS_mm, col_S_vfrac)으로 앱의 포화 칼럼 반-차수 상수
    k_w = k½a·a_S/φ 를 계산해 문헌 상수와 비교. 30 mg/L로 배율 하나를 보정하고(보정), 40·60 mg/L로 확인(R6).
    합격: 보정 전 비 1/1.5~1.5 안(예측), 보정 후 확인 농도 비 1/1.5~1.5 안, 확인 농도 예측 제거율 > 90 %."""
    keys = ["col_L_m", "col_D_m", "col_Q_mL_min", "col_phi", "col_dS_mm", "col_S_vfrac"]
    miss = [k for k in keys if col.get(k) is None]
    if miss:
        return {"done": False, "reason": "칼럼 조건 미입력: " + ", ".join(miss)}
    phi = float(col["col_phi"])
    aS = 6.0 * (1 - phi) * float(col["col_S_vfrac"]) / (float(col["col_dS_mm"]) * 1e-3)
    k_app = k_half_a * aS / phi                                       # (mg/L)^½/h, 포화
    A = np.pi * float(col["col_D_m"]) ** 2 / 4
    Q = float(col["col_Q_mL_min"]) * 1e-6 * 60                        # m³/h
    tau = phi * A * float(col["col_L_m"]) / Q                          # h (공극 체류시간)
    rows = []
    scale = MOON2004[0][1] / k_app                                     # 30 mg/L로 보정
    for i, (C0, k_lit) in enumerate(MOON2004):
        kc = k_app * scale
        Cout = max(np.sqrt(C0) - kc * tau / 2, 0.0) ** 2
        rows.append({"C0": C0, "k_lit": k_lit, "k_app": k_app, "ratio_raw": k_app / k_lit,
                     "k_cal": kc, "ratio_cal": kc / k_lit, "removal_pred": (1 - Cout / C0) * 100,
                     "use": "보정" if i == 0 else "확인"})
    chk = [r for r in rows if r["use"] == "확인"]
    ok_raw = 1 / factor_ok <= rows[0]["ratio_raw"] <= factor_ok
    ok_cal = all(1 / factor_ok <= r["ratio_cal"] <= factor_ok and r["removal_pred"] > 90 for r in chk)
    return {"done": True, "rows": rows, "tau_h": tau, "scale": scale, "ok_raw": ok_raw, "ok_cal": ok_cal}
