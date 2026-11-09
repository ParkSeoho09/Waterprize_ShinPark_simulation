"""앱 내장 '자체 점검'(외부 프로그램과 비교하지 않음): 방정식을 맞게 푸는지 해석해·준해석해·보존 법칙으로 확인.

  S1  1D 침투 Richards vs 준해석해(ODE)                    (V1 재사용)
  S2  1D 포화 이송 TVD vs 3종 경계 해석해                    (V2 재사용)
  S3  SLAD 반-차수 해석해·양론                              (V3 재사용)
  E1a 3D 대각(경사) 포화 흐름: 선형 수두장 정확 재현(이방성 포함)
  E1b 경사진 불투수 장벽(계단식 면 장벽) 누수 없음
  E1c 지표 포화 전환: q = 2Ks 일 때 정상 침투 = Ks
  E1d 3D 격자(5×5)에서 1D 해와 같음(측면 항 영향 없음)
  E1e 직렬 저항 장벽: Q = ΔH/(L/K + R) 정확
  E2a 불포화(θ<θs) 1D 이송 vs 3종 해석해
  E2b 이류-분산 + 1차 감쇠 정상해(TVD 수치 분산 오차 점검)
  E3  pH 보정 켠 SLAD + 석회석 완충 회분: 알칼리도·황·칼슘 수지, PHREEQC 엔진과 내장 화학의 같은 계산
  E5  θ가 크게 다른 조건(건조 0.15 / 습윤 0.9)에서 주입 입자 질량 보존
결과: selftest/selfcheck_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "verify"))

from flow_richards import FlowBC, RichardsFV, vg_kr, vg_theta  # noqa: E402
from params import MW, REGISTRY, SOIL_TEXTURES, VM, Common  # noqa: E402
from transport_reaction import LEDGER_KEYS, ChemState, ReactParams, Transport, alk_from_pH, react  # noqa: E402
from voxel import Grid, face_crossings  # noqa: E402

SL = SOIL_TEXTURES["사양토(Sandy loam)"]


def _fv(g, Kx, Kz, soil=SL, barrier=None, active=None):
    o = np.ones(g.shape)
    nx, ny, nz = g.shape
    bar = barrier or (np.zeros((nx - 1, ny, nz)), np.zeros((nx, ny - 1, nz)), np.zeros((nx, ny, nz - 1)), np.zeros((nx, ny)))
    return RichardsFV(g, np.ones(g.shape, bool) if active is None else active, Kx * o, Kz * o, soil["theta_r"] * o,
                      soil["theta_s"] * o, soil["alpha"] * o, soil["n"] * o, bar)


def row(id_, item, crit, result, ok):
    """판정: True → 통과, False → 미통과(R7), None → 기록(합격 기준 없는 시험)."""
    ok = None if ok is None else bool(ok)          # numpy bool도 같은 판정
    return dict(id=id_, 항목=item, 기준=crit, 결과=result, 판정="통과" if ok is True else ("기록" if ok is None else "미통과"))


def E1a_diagonal():
    g = Grid(16, 3, 20, 0.8, 0.15, 1.0, 1.0)
    errs = []
    for Kx, Kz in ((1e-5, 1e-5), (1e-5, 2e-6)):
        fv = _fv(g, Kx, Kz)
        X = np.broadcast_to(g.xc[:, None, None], g.shape)
        Z = g.H - np.broadcast_to(g.dc[None, None, :], g.shape)
        Hex = 2.0 + 0.3 * X - 0.2 * Z          # 대각 방향 동수경사
        hex_ = Hex - Z                          # 압력수두(모두 >0 → 포화)
        fixed = np.full(g.shape, np.nan)
        b = np.zeros(g.shape, bool)
        b[0] = b[-1] = True; b[:, :, 0] = b[:, :, -1] = True
        fixed[b] = hex_[b]
        st = fv.solve_steady(hex_ * 0 + 1.0, FlowBC(q_top=0.0, bottom="자유배수", fixed_h=fixed))
        errs.append(float(np.abs(st.h - hex_).max()))
    ok = max(errs) < 1e-8
    return row("E1a", "3D 대각 포화 흐름(선형 수두장, 등방·이방 Kz/Kx=0.2)", "최대 수두 오차 < 1e-8 m",
               f"등방 {errs[0]:.1e} m, 이방 {errs[1]:.1e} m", ok)


def _plane_mesh(p0, n, size):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    a = np.cross(n, [0, 1, 0]); a /= np.linalg.norm(a)
    b = np.cross(n, a)
    t = 1e-4                                   # 얇은 판(0.1 mm)
    corners = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            corners.append(p0 + s1 * size * a + s2 * size * b)
    c = np.array(corners)
    V = np.vstack([c - t * n, c + t * n])
    T = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 4, 5], [0, 5, 1], [1, 5, 7], [1, 7, 3],
                  [3, 7, 6], [3, 6, 2], [2, 6, 4], [2, 4, 0]])
    return V, T


def E1b_inclined_barrier():
    g = Grid(24, 4, 24, 1.0, 0.2, 1.0, 1.0)
    V, T = _plane_mesh(np.array([0.5, 0.1, 0.5]), [1.0, 0.0, 0.6], 2.0)     # x축에 비스듬한 판(약 31°)
    bx, by, bz, bt = face_crossings(V, T, g)
    R = (np.where(bx, np.inf, 0.0), np.where(by, np.inf, 0.0), np.where(bz, np.inf, 0.0), np.where(bt, np.inf, 0.0))
    fv = _fv(g, 1e-5, 1e-5, barrier=R)
    fixed = np.full(g.shape, np.nan)
    Z = g.H - np.broadcast_to(g.dc[None, None, :], g.shape)
    fixed[0] = 0.9 - Z[0]; fixed[-1] = 0.5 - Z[-1]          # 왼쪽 H=0.9, 오른쪽 H=0.5 (지표 아래, 일부 불포화)
    bc = FlowBC(q_top=0.0, bottom="누수(배수 저항)", leak_R=1e30, fixed_h=fixed)   # 바닥 사실상 무유동
    st = fv.solve_steady(np.ones(g.shape) * -0.2, bc)
    Q_left = float(np.abs(st.Fsrc[0]).sum())
    # 장벽이 없을 때
    fv0 = _fv(g, 1e-5, 1e-5)
    st0 = fv0.solve_steady(np.ones(g.shape) * -0.2, bc)
    Q0 = float(np.abs(st0.Fsrc[0]).sum())
    ratio = Q_left / Q0
    return row("E1b", "경사진 불투수 장벽(31° 판, 계단식 면 장벽) 누수", "장벽 통과 유량 / 장벽 없을 때 < 1e-8",
               f"비 {ratio:.1e} (장벽 면 {int(bx.sum() + by.sum() + bz.sum())}개)", ratio < 1e-8)


def E1c_ponding():
    g = Grid(1, 1, 80, 0.1, 0.1, 2.0, 2.0)
    fv = _fv(g, SL["Ks"], SL["Ks"])
    st = fv.solve_steady(None, FlowBC(q_top=2 * SL["Ks"], bottom="자유배수"))
    infil = float(st.Ftop.sum()) / (g.dx * g.dy)
    err = abs(infil / SL["Ks"] - 1)
    return row("E1c", "지표 포화 전환(q = 2Ks, 자유배수)", "정상 침투 = Ks (상대오차 < 1e-6), 나머지는 지표 유출",
               f"침투/Ks = {infil / SL['Ks']:.8f}, 유출 = {st.runoff / (g.dx * g.dy) / SL['Ks']:.4f} Ks, 포화 {bool(st.ponded.any())}",
               err < 1e-6 and bool(st.ponded.any()))


def E1d_3d_vs_1d():
    q = 5e-3 / 86400
    g1 = Grid(1, 1, 60, 0.1, 0.1, 1.2, 1.2)
    g3 = Grid(5, 5, 60, 0.5, 0.5, 1.2, 1.2)
    s1 = _fv(g1, SL["Ks"], SL["Ks"]).solve_steady(None, FlowBC(q_top=q, bottom="지하수 수위", z_wt=0.0))
    s3 = _fv(g3, SL["Ks"], SL["Ks"]).solve_steady(None, FlowBC(q_top=q, bottom="지하수 수위", z_wt=0.0))
    d = float(np.abs(s3.h - s1.h[0, 0][None, None, :]).max())
    lat = float(max(np.abs(s3.Fx).max(), np.abs(s3.Fy).max()))
    return row("E1d", "3D 격자(5×5)와 1D 해 일치(측면 항)", "수두 차 < 1e-8 m, 측면 유량 ≈ 0",
               f"최대 수두 차 {d:.1e} m, 최대 측면 유량 {lat:.1e} m3/s", d < 1e-8 and lat < 1e-15)


def E1e_series_barrier():
    g = Grid(20, 1, 1, 1.0, 0.1, 0.1, 0.1)
    K = 1e-5; Rb = 5e4
    bx = np.zeros((19, 1, 1)); bx[9] = Rb
    bar = (bx, np.zeros((20, 0, 1)), np.zeros((20, 1, 0)), np.full((20, 1), np.inf))
    fv = _fv(g, K, K, barrier=bar)
    fixed = np.full(g.shape, np.nan)
    z = g.H - g.dc[0]
    fixed[0] = 2.0 - z; fixed[-1] = 1.0 - z
    st = fv.solve_steady(np.ones(g.shape), FlowBC(q_top=0.0, bottom="누수(배수 저항)", leak_R=1e30, fixed_h=fixed))
    A = g.dy * g.dz
    L = (g.nx - 1) * g.dx
    Qex = 1.0 / (L / (K * A) + Rb / A)
    Q = float(st.Fx[0, 0, 0])
    return row("E1e", "직렬 저항 장벽(얇은 실링) 1D", "Q = ΔH/(L/K + R) 상대오차 < 1e-10",
               f"상대오차 {abs(Q / Qex - 1):.1e}", abs(Q / Qex - 1) < 1e-10)


def E2a_unsat_transport():
    from scipy.special import erfc, erfcx
    q = 0.2 * SL["Ks"]
    res = {}
    for nz in (100, 200, 400):
        g = Grid(1, 1, nz, 0.1, 0.1, 1.0, 1.0)
        fv = _fv(g, SL["Ks"], SL["Ks"])
        st = fv.solve_steady(None, FlowBC(q_top=q, bottom="자유배수"))
        th = float(st.theta.ravel()[nz // 2])
        aL, Dm = 0.01, 1.7e-9
        tr = Transport(g, fv, dict(alpha_L=aL, alpha_T_ratio=0.1, Dm=Dm, phi_cells=np.full(g.N, SL["theta_s"])))
        tr.set_flow(st)
        c = {"NO3": np.zeros(g.N)}
        L = {k: 0.0 for k in LEDGER_KEYS}
        t, tend = 0.0, 0.3 * 1.0 * th / q
        while t < tend - 1e-9:
            dt = min(tr.dt_max, tend - t)
            tr.step(c, dt, {"top": {"NO3": 1.0}}, L)
            t += dt
        v = q / th
        D = aL * v + Dm * th ** (7 / 3) / SL["theta_s"] ** 2
        x = g.dc
        a1 = (x - v * t) / (2 * np.sqrt(D * t)); a2 = (x + v * t) / (2 * np.sqrt(D * t))
        third = 0.5 * erfc(a1) + np.sqrt(v * v * t / (np.pi * D)) * np.exp(-a1 ** 2) \
            - 0.5 * (1 + v * x / D + v * v * t / D) * np.exp(v * x / D - a2 ** 2) * erfcx(a2)
        thv = st.theta.ravel()
        mass = ((c["NO3"] * thv * g.vcell).sum() - (L["N_in"] - L["N_out"])) / max(L["N_in"], 1e-30)
        res[nz] = (float(np.abs(c["NO3"] - third).max()), float(mass), th, float(thv.std()))
    ok = res[400][0] < 0.01 and abs(res[400][1]) < 1e-10
    return row("E2a", "불포화 1D 이송(q = 0.2 Ks, θ < θs) vs 3종 해석해",
               "최대오차 < 0.01, 질량오차 < 1e-10", "최대오차 " + ", ".join(f"nz={k}: {v[0]:.4f}" for k, v in res.items())
               + f" · θ={res[400][2]:.3f} (θs={SL['theta_s']}) · 질량오차 {res[400][1]:.1e}", ok)


def E2b_decay():
    q = 0.2 * SL["Ks"]
    out = {}
    for nz in (50, 100, 200):
        g = Grid(1, 1, nz, 0.1, 0.1, 1.0, 1.0)
        fv = _fv(g, SL["Ks"], SL["Ks"])
        st = fv.solve_steady(None, FlowBC(q_top=q, bottom="자유배수"))
        aL, Dm = 0.01, 0.0
        tr = Transport(g, fv, dict(alpha_L=aL, alpha_T_ratio=0.1, Dm=Dm, phi_cells=np.full(g.N, SL["theta_s"])))
        tr.set_flow(st)
        th = float(st.theta.ravel()[nz // 2]); v = q / th; D = aL * v
        k = 2.0 * v / 1.0                     # 1 m 에서 e^-2 정도 감쇠
        c = {"NO3": np.zeros(g.N)}
        L = {kk: 0.0 for kk in LEDGER_KEYS}
        t, tend = 0.0, 3.0 * 1.0 / v
        while t < tend - 1e-9:
            dt = min(tr.dt_max, tend - t)
            tr.step(c, dt, {"top": {"NO3": 1.0}}, L)
            c["NO3"] *= np.exp(-k * dt)
            t += dt
        x = g.dc
        # 3종(유량가중) 유입 경계 반무한 정상해: c = c0·2/(1+β)·exp((v−w)x/2D), w = sqrt(v²+4Dk), β=w/v
        w = np.sqrt(v * v + 4 * D * k)
        ex = 2.0 / (1.0 + w / v) * np.exp((v - w) * x / (2 * D))
        sel = x < 0.8
        out[nz] = float(np.abs(c["NO3"][sel] - ex[sel]).max() / ex[sel].max())
    ok = out[200] < 0.03
    return row("E2b", "이류-분산 + 1차 감쇠 정상해(수치 분산 점검)", "상대오차 < 3% (nz=200)",
               ", ".join(f"nz={k}: {v * 100:.2f}%" for k, v in out.items()), ok)


def E3_ph_buffer():
    th = np.array([0.4])
    phi = 0.4; r = 3.0
    S0 = (1 - phi) * r / (r + 1) / VM["S"]; C0 = (1 - phi) / (r + 1) / VM["CaCO3"]
    pCO2 = -2.0
    alk0 = alk_from_pH(7.5, pCO2)

    def state(cal=True):
        return ChemState({"NO3": np.array([50 / 14.007]), "SO4": np.array([0.1]), "Ca": np.array([0.5]),
                          "ALK": np.array([alk0]), "P": np.zeros(1)}, np.array([S0]), np.array([C0 if cal else 0.0]),
                         np.zeros(1), np.zeros(1), np.zeros(1), np.array([6 / 2e-3]), np.array([1.0]),
                         {k: 0.0 for k in LEDGER_KEYS},
                         aS=np.array([6 * np.cbrt(S0 * VM["S"]) / REGISTRY["S_d_media"][0]]), age=np.array([1e3]))
    # 표면적 기반 속도식(v4, 예전 k=5.0 대체)
    rp = ReactParams(k_half=0.0, S_ref=REGISTRY["S_ref_frac"][0], pH_low=6.2, pH_high=9.0, theta_T=1.07, T=22, T_ref=22,
                     stoich_SO4=1.10, stoich_H=1.28, stoich_S=1.10, pCO2=pCO2, I_bg=0.005, bio_vol=3.6e-4,
                     bio_decay=0.05, calcite_mode="속도론", lambda_filt=1000, ratio=3,
                     kin="area", k_a=REGISTRY["k_half_a"][0], wfps_thr=0.0, wfps_exp=0.0, lag_d=0.0)
    from transport_reaction import pH_of
    res = {}
    for cal in (True, False):
        cs = state(cal)
        pHmin = 99
        for _ in range(180):
            react(cs, th, 1e-3, 60.0, rp)
            pHmin = min(pHmin, float(pH_of(cs.c["ALK"], cs.c["Ca"], cs.c["SO4"], cs.c["NO3"], pCO2, 0.005)[0]))
        L = cs.ledger
        dA = (cs.c["ALK"][0] - alk0) * 0.4 * 1e-3
        budA = (L["ALK_cal"] - L["ALK_denit"])
        res[cal] = (pHmin, abs(dA - budA) / max(abs(budA), 1e-30), (50 / 14.007 - cs.c["NO3"][0]) * 14.007)
    msg = (f"석회석 있음: pH 최저 {res[True][0]:.2f}, 탈질 {res[True][2]:.1f} mg/L, 알칼리도 수지 {res[True][1]:.1e}; "
           f"석회석 없음: pH 최저 {res[False][0]:.2f}, 탈질 {res[False][2]:.1f} mg/L")
    ok = res[True][1] < 1e-9 and res[True][0] > res[False][0]
    # PHREEQC 엔진과 내장 화학 비교(같은 회분, 2시간)
    try:
        import phreeqc_engine as PE
        if PE.available():
            cells = {"NO3": np.array([50 / 14.007e3]), "SO4": np.array([1e-4]), "Ca": np.array([5e-4]),
                     "ALK": np.array([alk0 * 1e-3]), "Ssol": np.array([S0 / 0.4 * 1e-3]),
                     "Cal": np.array([C0 / 0.4 * 1e-3]), "Gyp": np.zeros(1), "theta": th, "aCal": np.array([3000.0])}
            o = PE.react_cells(cells, 7200.0, rp, np.array([pCO2]), "속도론")
            cs = state(True)
            for _ in range(120):
                react(cs, th, 1e-3, 60.0, rp)
            dn_int = (50 / 14.007 - cs.c["NO3"][0]) * 14.007
            dn_ph = o["dN"][0] * 1e3 * 14.007
            msg += f" · 2시간 탈질량 PHREEQC {dn_ph:.2f} vs 내장 {dn_int:.2f} mg/L (pH PHREEQC {o['pH'][0]:.2f})"
        else:
            msg += " · PHREEQC 엔진 없음(비교 생략)"
    except Exception as e:     # noqa: BLE001
        msg += f" · PHREEQC 비교 오류 {str(e)[:120]}"
    msg += " · pH 보정 함수 모양(pH_low 6.2, pH_high 9.0)은 '가정'"
    return row("E3", "pH 보정 켠 SLAD + 석회석 완충(회분)", "알칼리도 수지 < 1e-9, 석회석이 pH 최저값을 높임", msg, ok)


def E5_particle_theta():
    from engine import Scenario, run_scenario
    from io_3mf import load_3mf
    from params import GRID_PRESETS, suggest_role
    from voxel import Placement
    import install_events as ie
    with open(os.path.join(APP, "samples", "injection_well_scale0.05.3mf"), "rb") as fh:
        d = load_3mf(fh.read(), "w")
    roles = {o.name: {"role": suggest_role(o.name)} for o in d["objects"]}
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 10.0}),
          dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 0.8, "고형분비(v/v)": 0.1, "유량(L/min)": 0.2}),
          dict(ie.empty_row("정치(대기)"), **{"소요(분)": 30.0})]
    out = []
    for sat in (0.25, 0.9):
        c = Common(); c.nx, c.ny, c.nz = GRID_PRESETS["가벼움"]; c.days = 0.5; c.light = True; c.n_snapshots = 2
        c.init_saturation = sat; c.straining = False; c.chem_engine = "내장"
        sc = Scenario(name="w", objects=d["objects"], roles=roles, placement=Placement(0.25, 0.25, 0, 0, 0, 1), events=ev,
                      reagent={"form": "(c) 입자 슬러리(포획)"})
        r = run_scenario(c, sc)
        L = r["ledger"]
        srcin = L.get("P_srcin", 0.0)
        Vt = 0.0
        clos = (srcin - L.get("P_dep", 0) - L.get("P_out", 0)) / max(srcin, 1e-30)
        out.append((sat, r["status"], srcin, L.get("P_dep", 0), clos, r["mass_balance"].get("P_err_rel", np.nan)))
    ok = all(o[1] == "OK" and abs(o[5]) < 1e-6 for o in out)
    return row("E5", "θ가 크게 다른 조건(초기 포화도 0.25 / 0.9)에서 주입 입자 질량 보존(포획 + 유출 + 부유)",
               "상대 오차 < 1e-6", "; ".join(f"포화도 {o[0]}: {o[1]}, 주입 {o[2] * 1e6:.1f} cm³, 포획 {o[3] * 1e6:.1f} cm³, "
                                           f"수지 오차 {o[5]:.1e}" for o in out), ok)


def B3_splitting():
    """연산자 분리 오차: Lie(이송→반응) vs Strang(이송½→반응→이송½) vs 기준(Lie, Δt/20). 석회석 평형 모드(급변)."""
    from transport_reaction import ChemState
    q = 20e-3 / 86400
    g = Grid(1, 1, 60, 0.1, 0.1, 0.6, 0.6)
    fv = _fv(g, SL["Ks"], SL["Ks"])
    st = fv.solve_steady(None, FlowBC(q_top=q, bottom="자유배수"))
    th = st.theta.ravel()
    phi = 0.4; r = 3.0
    S0 = (1 - phi) * r / (r + 1) / VM["S"]; C0 = (1 - phi) / (r + 1) / VM["CaCO3"]
    alk0 = alk_from_pH(7.0, -2.0)

    def run(div, mode):
        tr = Transport(g, fv, dict(alpha_L=0.01, alpha_T_ratio=0.1, Dm=1.7e-9, phi_cells=np.full(g.N, SL["theta_s"])))
        tr.set_flow(st)
        N = g.N
        S = np.zeros(N); C = np.zeros(N); S[10:20] = 0.05 * S0; C[10:20] = 0.05 * C0
        cs = ChemState({"NO3": np.zeros(N), "SO4": np.full(N, 0.1), "Ca": np.full(N, 0.5), "ALK": np.full(N, alk0),
                        "P": np.zeros(N)}, S, C, np.zeros(N), np.zeros(N), np.zeros(N), np.full(N, 3000.0),
                       np.full(N, SL["theta_s"]), {k: 0.0 for k in LEDGER_KEYS},
                       aS=6 * np.cbrt(S * VM["S"]) / REGISTRY["S_d_media"][0], age=np.full(N, 1e3))
        rp = ReactParams(k_half=0.0, S_ref=0.45, pH_low=6.2, pH_high=9.0, theta_T=1.07, T=22, T_ref=22, stoich_SO4=1.1,
                         stoich_H=1.28, stoich_S=1.1, pCO2=-2.0, I_bg=0.005, bio_vol=3.6e-4, bio_decay=0.05,
                         calcite_mode="평형", lambda_filt=1000, ratio=3,
                         kin="area", k_a=REGISTRY["k_half_a"][0], wfps_thr=0.0, wfps_exp=0.0, lag_d=0.0)
        bcc = {"top": {"NO3": 50 / 14.007, "SO4": 0.1, "Ca": 0.5, "ALK": alk0}}
        t, tend = 0.0, 1.5 * 86400
        dt0 = tr.dt_max / div
        while t < tend - 1e-9:
            dt = min(dt0, tend - t)
            if mode == "Strang":
                tr.step(cs.c, 0.5 * dt, bcc, cs.ledger); react(cs, th, g.vcell, dt, rp); tr.step(cs.c, 0.5 * dt, bcc, cs.ledger)
            else:
                tr.step(cs.c, dt, bcc, cs.ledger); react(cs, th, g.vcell, dt, rp)
            t += dt
        return cs.c["NO3"].copy() * 14.007, cs.c["ALK"].copy()
    ref = run(20, "Lie")
    out = {}
    for mode in ("Lie", "Strang"):
        n, a = run(1, mode)
        out[mode] = (float(np.abs(n - ref[0]).max()), float(np.abs(a - ref[1]).max() / max(np.abs(ref[1]).max(), 1e-12)))
    return row("B3", "연산자 분리 오차(석회석 평형 모드, 1.5일 칼럼)", "Δt/20 기준해 대비 NO₃ 최대오차 < 1 mg N/L, 알칼리도 < 5% (두 방식 모두)",
               f"Lie: NO₃ {out['Lie'][0]:.3f} mg/L, ALK {out['Lie'][1] * 100:.2f}% · Strang: NO₃ {out['Strang'][0]:.3f} mg/L, "
               f"ALK {out['Strang'][1] * 100:.2f}% (이 조건에서는 Strang이 더 낫지 않음 → 기본 Lie 유지)",
               all(v[0] < 1.0 and v[1] < 0.05 for v in out.values()))


def reuse_verify():
    import run_verification as RV
    figs = {}
    out = [RV.v1_richards(figs, HERE), RV.v2_ogata(figs, HERE)] + RV.v3_slad(figs, HERE)
    for r in out:
        r["id"] = {"V1": "S1", "V2": "S2", "V3a": "S3a", "V3b": "S3b", "V3c": "S3c"}.get(r["id"], r["id"])
    return out, figs


# ───────────────────────────── v4: 문헌 기반 속도식 검증 ─────────────────────────────
def _rp_v4(**kw):
    P = {k: v[0] for k, v in REGISTRY.items()}
    d = dict(k_half=5.0, S_ref=P["S_ref_frac"], pH_low=P["pH_low"], pH_high=P["pH_high"], theta_T=1.07, T=22.0,
             T_ref=22.0, stoich_SO4=1.10, stoich_H=1.28, stoich_S=1.10, pCO2=-2.0, I_bg=0.005, bio_vol=0.0,
             bio_decay=0.0, calcite_mode="속도론", lambda_filt=1000, ratio=3, kin="area", k_a=P["k_half_a"],
             wfps_thr=P["wfps_thr"], wfps_exp=P["wfps_exp"], k_ox_a=P["k_ox_a"], Ea_ox=P["Ea_ox"],
             s_opt_ox=P["s_opt_ox"], lag_d=0.0)
    d.update(kw)
    return ReactParams(**d)


def L1_koenig_size():
    """Koenig & Liu (2001) 입경별 반-차수 상수를 표면적 식(k½v = k½a·a, a = 6(1−ε)/d)이 재현하는가 — 같은 자료로 맞춘 값이라 '보정 재현'."""
    ka = REGISTRY["k_half_a"][0]
    rep = [((2.8, 5.6), (2.94, 3.60)), ((5.6, 11.2), (1.47, 2.04)), ((11.2, 16.0), (1.12, 1.29))]
    out, worst = [], 0.0
    for (d1, d2), (k1, k2) in rep:
        d = np.sqrt(d1 * d2) * 1e-3
        kv = ka * 6 * (1 - 0.4) / d
        mid = 0.5 * (k1 + k2)
        err = (kv - mid) / mid
        worst = max(worst, abs(err))
        out.append(f"{d1}–{d2} mm: 모델 {kv:.2f} / 보고 {k1}–{k2}")
    return row("L1", "황 입경별 반-차수 상수(Koenig & Liu 2001 칼럼)", "보고 범위 중간값 대비 ±25% (같은 자료로 k½a를 맞춤 = 보정 재현)",
               " · ".join(out) + f" · 최대 차이 {worst * 100:.0f}% (ε=0.4 가정)", worst < 0.25)


def L2_sierra_alvarez():
    """독립 자료: Sierra-Alvarez et al. (2007) 표면 기준 탈질 속도 26.4 mmol N/m²/d (질산염 7.3 mM)."""
    ka = REGISTRY["k_half_a"][0]
    C = 7.3 * 14.007                                    # mg N/L
    flux = ka * np.sqrt(C)                              # g N/m²/h  ((mg/L)^½·m/h × (mg/L)^½ = g/m²/h)
    pred = flux * 24 / 14.007 * 1000                    # mmol N/m²/d
    ratio = pred / 26.4
    return row("L2", "독립 자료와 비교: 표면 탈질 속도(Sierra-Alvarez et al. 2007)",
               "같은 자릿수(0.33~3배) — 온도·배양 조건이 다른 독립 실험", f"모델 {pred:.1f} vs 보고 26.4 mmol N/m²/일 (비 {ratio:.2f}). "
               "기본값은 칼럼(Koenig & Liu) 쪽 — 이 자료에 맞추면 k½a ≈ 1.5e-3(불확실성 분석 하한)", 1 / 3 <= ratio <= 3)


def L3_area_halforder_cell():
    """구현 확인: 포화 충전층 한 셀의 탈질이 표면적 반-차수 해석해 √C = √C0 − k½a·a/(2θ)·t 와 같은가."""
    th = np.array([0.4]); phi_m = 1.0
    d = 4e-3; phiS = 0.45
    S0 = phiS / VM["S"]
    alk0 = alk_from_pH(7.5, -2.0)
    cs = ChemState({"NO3": np.array([50 / 14.007]), "SO4": np.array([0.1]), "Ca": np.array([2.0]),
                    "ALK": np.array([alk0]), "P": np.zeros(1)}, np.array([S0]), np.zeros(1), np.zeros(1), np.zeros(1),
                   np.array([0.15]), np.array([3000.0]), np.array([0.4 + phiS + 0.15]), {k: 0.0 for k in LEDGER_KEYS},
                   aS=np.array([6 * np.cbrt(phiS) / d]), age=np.array([100.0]))
    rp = _rp_v4(pH_low=-50, pH_high=50)                  # pH 보정 없이(해석해와 같은 조건)
    a = 6 * phiS / d
    kw = rp.k_a * a / 0.4 / 3600                         # (mg/L)^½/s, 물 부피 기준
    err = 0.0
    t = 0.0
    for _ in range(60):
        react(cs, th, 1e-3, 60.0, rp)
        t += 60
        ana = max(np.sqrt(50) - 0.5 * kw * t, 0) ** 2
        err = max(err, abs(cs.c["NO3"][0] * 14.007 - ana))
    return row("L3", "표면적 반-차수 탈질 구현(회분 한 셀)", "해석해 대비 최대오차 < 0.05 mg N/L",
               f"a = {a:.0f} m²/m³, 1시간 최대오차 {err:.2e} mg N/L, 남은 NO₃-N {cs.c['NO3'][0] * 14.007:.2f} mg/L", err < 0.05)


def L4_sulfur_shrinking():
    """호기성 황 산화: 표면 속도가 일정하면 남은 질량 = (1 − k·t/(ρ_m·r0))³ (Watkinson 1989의 구형 입자 식)."""
    th = np.array([0.3]); phiS = 0.01; d = 2e-4
    S0 = phiS / VM["S"]
    alk0 = alk_from_pH(7.5, -2.0)
    cs = ChemState({"NO3": np.zeros(1), "SO4": np.array([0.1]), "Ca": np.array([2.0]), "ALK": np.array([alk0]),
                    "P": np.zeros(1), "O2": np.array([0.25])}, np.array([S0]), np.array([50.0]), np.zeros(1),
                   np.zeros(1), np.zeros(1), np.array([3000.0]), np.array([0.45]), {k: 0.0 for k in LEDGER_KEYS},
                   O2g=np.array([1e6]), aS=np.array([6 * np.cbrt(phiS) / d]), age=np.array([100.0]))
    rp = _rp_v4(o2_on=True, K_O2=1e-9, T=25.0, s_opt_ox=0.3 / 0.45 if False else (0.3 / cs.porosity()[0]))
    k = rp.k_ox_a                                       # mol/m²/s
    rho_m = 1.0 / VM["S"]
    r0 = d / 2
    err, t = 0.0, 0.0
    for _ in range(40):
        react(cs, th, 1e-3, 86400.0, rp)
        t += 86400.0
        ana = max(1 - k * t / (rho_m * r0), 0) ** 3
        err = max(err, abs(cs.S[0] / S0 - ana))
    days_full = rho_m * r0 / k / 86400
    return row("L4", "호기성 황 산화의 입자 축소(구형, 표면 속도 일정)", "남은 질량 비율 최대오차 < 0.005",
               f"0.2 mm 입자, 25°C·최적 수분: 40일 뒤 남은 비율 {cs.S[0] / S0:.3f} (해석해 {max(1 - k * t / (rho_m * r0), 0) ** 3:.3f}), "
               f"최대오차 {err:.1e}; 다 녹는 시간 {days_full:.0f}일 (Watkinson & Blair 1993: 50 µg S/cm²/일)", err < 0.005)


def L5_o2_gas_diffusion():
    """토양 공기 O2 확산(Millington–Quirk) 구현: 1D 기둥 과도 확산 vs 해석해 erfc (반응 없음)."""
    from transport_reaction import o2_gas_diffuse, KHO2, R_ATM
    from scipy.special import erfc
    nz = 200; H = 1.0
    g = Grid(1, 1, nz, 0.01, 0.01, H, H)
    fv = _fv(g, 1e-5, 1e-5)
    tr = Transport(g, fv, {"alpha_L": 0.01, "alpha_T_ratio": 0.1, "Dm": 1e-9})
    th = np.full(nz, 0.15); phi = 0.45
    T = 20.0
    cs = ChemState({"O2": np.zeros(nz)}, np.zeros(nz), np.zeros(nz), np.zeros(nz), np.zeros(nz), np.zeros(nz),
                   np.zeros(nz), np.full(nz, phi), {k: 0.0 for k in LEDGER_KEYS}, O2g=np.zeros(nz))
    D0 = REGISTRY["D_O2_air"][0]
    ea = phi - 0.15
    Deff = D0 * ea ** (10 / 3) / phi ** 2
    Hd = KHO2 * R_ATM * (T + 273.15)
    Dapp = Deff / (ea + 0.15 * Hd)
    Catm = 0.21 / (R_ATM * (T + 273.15))
    dt = 60.0; t = 0.0
    for _ in range(6 * 60):                      # 6시간
        o2_gas_diffuse(cs, tr, th, dt, T, D0)
        t += dt
    z = (np.arange(nz) + 0.5) * H / nz
    Cg = cs.O2g / ea
    ana = Catm * erfc(z / (2 * np.sqrt(Dapp * t)))
    m = z < 0.5
    err = float(np.max(np.abs(Cg[m] - ana[m])) / Catm)
    return row("L5", "토양 공기 O₂ 확산(Millington & Quirk 1961) 과도 1D", "해석해 erfc 대비 최대오차 < 2% (대기 농도 기준)",
               f"ε_a = {ea:.2f}, D_eff = {Deff:.2e} m²/s, 6시간 침투 깊이 2√(Dt) = {2 * np.sqrt(Dapp * t) * 100:.1f} cm, 최대오차 {err * 100:.2f}%",
               err < 0.02)


def L6_moisture_function():
    """NEMIS 수분 함수: 포화도 0.62 이하 0, 1에서 1, 0.8에서 ((0.8−0.62)/0.38)^1.74."""
    th = REGISTRY["wfps_thr"][0]; w = REGISTRY["wfps_exp"][0]
    f = lambda s: np.clip((s - th) / (1 - th), 0, 1) ** w      # noqa: E731
    vals = {s: float(f(s)) for s in (0.5, 0.62, 0.7, 0.8, 0.9, 1.0)}
    ok = vals[0.5] == 0 and vals[1.0] == 1 and abs(vals[0.8] - ((0.18 / 0.38) ** 1.74)) < 1e-12
    return row("L6", "불포화 탈질 수분 함수(NEMIS, Hénault & Germon 2000)", "문헌 식과 같은 값",
               " · ".join(f"s={k}: {v:.3f}" for k, v in vals.items()) + " — 토양 탈질 식을 황 탈질에 유추 적용(직접 측정 자료 없음)", ok)


# ───────────────────────────── v4.1: 설계 변수(투입량·배합·수분 유지 재료) ─────────────────────────────
def D1_retainer_literature():
    """수분 유지 재료 효과가 문헌 값과 같게 들어가는가: 벤토나이트 0/3/9 % 투수계수(Chalermyanont & Arrykul 2005),
    바이오차 5 % 공극률 +8.4 %·Ks +25.2 %·유효수분 +15.1 %(Omondi et al. 2016)."""
    import design as DZ
    sand = SOIL_TEXTURES["모래(Sand)"]; clay = SOIL_TEXTURES["식토(Clay)"]
    base = dict(Ks=sand["Ks"], phi=0.40, alpha=sand["alpha"], n=sand["n"], thr=sand["theta_r"])
    km = {x: DZ.zone_hydraulics(base, {"retainer": "벤토나이트", "ret_frac": x}, clay)["Ks_mult"] for x in (0.0, 0.03, 0.09)}
    ref = {0.0: 1.0, 0.03: 1e-9 / 3.6e-7, 0.09: 4.13e-11 / 3.6e-7}
    eb = max(abs(np.log10(km[x] / ref[x])) for x in km)
    zb = DZ.zone_hydraulics(base, {"retainer": "바이오차", "ret_frac": 0.05}, clay)
    awc0 = DZ._awc(base["thr"], base["phi"], base["alpha"], base["n"])
    awc1 = DZ._awc(base["thr"], zb["phi"], zb["alpha"], zb["n"])
    ok = eb < 1e-9 and abs(zb["phi"] / 0.40 - 1.084) < 1e-9 and abs(zb["Ks_mult"] - 1.252) < 1e-9 and abs(awc1 / awc0 - 1.151) < 0.01
    return row("D1", "수분 유지 재료 효과 = 문헌 값(벤토나이트 Ks, 바이오차 공극률·Ks·유효수분)",
               "문헌 점과 같음(Ks 로그 차 < 1e-9, 바이오차 ±1 %)",
               f"벤토나이트 Ks 배율 0 %: {km[0.0]:.3g}, 3 %: {km[0.03]:.3g}, 9 %: {km[0.09]:.3g} (문헌 비 1, 2.8e-3, 1.1e-4) · "
               f"바이오차 5 %: 공극률 ×{zb['phi'] / 0.4:.3f}, Ks ×{zb['Ks_mult']:.3f}, 유효수분 ×{awc1 / awc0:.3f} — "
               "보수성(벤토나이트)·투입률 환산(바이오차)은 가정", ok)


def D2_slurry_mass():
    """황:물 배합 → 슬러리 부피·고형분비 계산이 질량 보존을 지키는가(황 투입량 = 슬러리 속 황)."""
    import design as DZ
    worst = 0.0
    for wps in (0.5, 2.5, 10.0):
        for ret, f in (("없음", 0.0), ("벤토나이트", 0.05), ("바이오차", 0.1)):
            d = {"water_per_S": wps, "retainer": ret, "ret_frac": f}
            sp = DZ.slurry_props(3.0, d)
            V = 500.0 * sp["L_per_gS"]                        # 황 500 g
            Vs = V * sp["fs_vv"]                              # 고형분 L
            S_g = Vs * 1000 * sp["vS"] * DZ.RHO["S"]          # cm3 × g/cm3
            worst = max(worst, abs(S_g - 500.0) / 500.0)
    return row("D2", "슬러리 배합 질량 보존(황 투입량 → 주입 부피·고형분비)", "상대오차 < 1e-12",
               f"물/황 0.5·2.5·10, 재료 없음·벤토나이트 5 %·바이오차 10 %: 최대 상대오차 {worst:.1e}", worst < 1e-12)


# ───────────────────────────── F9 수분 의존 분해 시험(S14~S17, 정의 점검) ─────────────────────────────
def S14_fw_limits():
    """f_w = S^p 극한·단조·기대값. 기대값(계산 전 확정, 요청서): S=0.16 → 0.54·0.29·0.16, S=0.22 → 0.60·0.36·0.22 (p=1/3·2/3·1).
    요청서 값이 소수 둘째 자리라 '구현값을 둘째 자리로 반올림 = 요청서 값'과 '구현값 = S^p (1e-12)'를 함께 본다."""
    import moisture as MO
    exp_ = {0.16: (0.54, 0.29, 0.16), 0.22: (0.60, 0.36, 0.22)}
    ps = (1 / 3, 2 / 3, 1.0)
    ok = True; parts = []
    for p_ in ps:
        ok &= abs(float(MO.f_w(1.0, p_)) - 1) < 1e-12 and abs(float(MO.f_w(0.0, p_))) < 1e-12
        ss_ = np.linspace(0, 1, 201)
        ok &= bool(np.all(np.diff(MO.f_w(ss_, p_)) >= -1e-15))
    for S_, ev in exp_.items():
        got = [float(MO.f_w(S_, p_)) for p_ in ps]
        ok &= all(abs(g - S_ ** p_) < 1e-12 for g, p_ in zip(got, ps))
        ok &= all(round(g, 2) == e for g, e in zip(got, ev))
        parts.append(f"S={S_}: " + "·".join(f"{g:.3f}" for g in got) + f" (요청서 {'·'.join(f'{e:.2f}' for e in ev)})")
    return row("S14", "f_w = S^p 극한·단조·기대값(F9-1)", "S=1→1, S=0→0, 단조 증가, S=0.16·0.22 값이 요청서 값과 같음(둘째 자리)",
               " · ".join(parts), ok)


def S15_no_double_count():
    """모드 A·B·C의 함수 조합이 설계대로인가: 한 셀 회분(짧은 시간, 기상 O₂ 저장소 큼)에서
    탈질량 비 B/C = f_NEMIS(S), A/C = f_w(S)·f_O2(O₂)(±2 %), 모드 B에서도 호기성 황 산화의 O₂ 소모가 있음."""
    import moisture as MO
    S_ = 0.9; th = np.array([0.36]); phi = 0.40
    d = 4e-3; phiS = 0.0                       # 셀 공극률을 정확히 0.40으로 두려고 황 부피는 phi_matrix로 맞춤
    S0 = 0.05 / VM["S"]
    alk0 = alk_from_pH(7.5, -2.0)
    O2c = 0.25                                 # mol/m3 ≈ 8 mg/L
    from transport_reaction import KHO2, R_ATM
    O2g0 = (phi - th[0]) * O2c / (KHO2 * R_ATM * (22.0 + 273.15))    # 공기 공극의 기상 O2(용존과 헨리 평형)

    def run(mode):
        cs = ChemState({"NO3": np.array([50 / 14.007]), "SO4": np.array([0.1]), "Ca": np.array([2.0]),
                        "ALK": np.array([alk0]), "P": np.zeros(1), "O2": np.array([O2c])}, np.array([S0]), np.zeros(1),
                       np.zeros(1), np.zeros(1), np.zeros(1), np.array([3000.0]), np.array([phi + 0.05]),
                       {k: 0.0 for k in LEDGER_KEYS}, O2g=np.array([O2g0]),
                       aS=np.array([6 * np.cbrt(0.05) / d]), age=np.array([1e3]))
        cfg = {"p": 2 / 3, "fw_form": "S^p", "thr": 0.70, "w": 1.74, "nemis_form": "멱함수", "K_O2": 0.2 / 32}
        rp = _rp_v4(o2_on=True, K_I_O2=0.2 / 32, pH_low=-50, pH_high=50, moist_mode=mode, moist_cfg=cfg)
        for _ in range(5):
            react(cs, th, 1e-3, 60.0, rp)
        return cs.ledger["N_denit"], cs.ledger.get("S_ox_O2", 0.0), cs
    nA, oA, csA = run("A"); nB, oB, _ = run("B"); nC, oC, _ = run("C")
    O2_now = float(csA.c["O2"][0])
    eA = float(MO.f_w(S_, 2 / 3) * MO.f_O2(O2_now, 0.2 / 32)); eB = float(MO.f_nemis(S_, 0.70, 1.74))
    rA, rB = nA / nC, nB / nC
    ok = abs(rA / eA - 1) < 0.02 and abs(rB / eB - 1) < 0.02 and oB > 0 and oA > 0 and oC > 0
    return row("S15", "모드별 함수 조합(중복 계산 방지, F9-2)", "A/C = f_w·f_O2, B/C = f_NEMIS (±2 %), 모드 B에도 O₂ 소모 있음",
               f"S={S_}, O₂ {O2_now * 32:.2f} mg/L: A/C {rA:.4f} (기대 {eA:.4f}), B/C {rB:.4f} (기대 {eB:.4f}), "
               f"호기성 황 산화 A·B·C {oA:.2e}·{oB:.2e}·{oC:.2e} mol", ok)


def S17_calib_recover():
    """합성 자료(p = 2/3로 생성, 포화도 0.3~1.0 다섯 점, 잡음 없음)를 F9-7 보정에 넣어 p를 되찾는가(오차 10 % 이내).
    네 점 보정·한 점 확인으로 R6(보정·확인 분리)도 확인."""
    import moisture as MO
    k_a_a = 4.0e-3 * 6 * 0.3 / 4e-3            # (mg/L)^½/h, 벌크 기준(φ_S 0.3, 4 mm)
    cfg = {"p": 2 / 3, "fw_form": "S^p"}
    rows = []
    for i, S_ in enumerate((0.3, 0.45, 0.6, 0.8, 1.0)):
        Cout = float(MO.removal_model([S_], [50.0], [0.5], [1.0], k_a_a, "A", cfg, "p", 2 / 3)[0])
        rows.append({"S": S_, "Cin": 50.0, "Q_L_h": 0.5, "V_L": 1.0, "Cout": Cout, "use": "확인" if i == 2 else "보정"})
    out = MO.calibrate(rows, k_a_a, "A", {"fw_form": "S^p"})
    few = MO.calibrate(rows[:2], k_a_a, "A", {"fw_form": "S^p"})
    ok = out.get("ok") and abs(out["value"] / (2 / 3) - 1) < 0.10 and not few.get("ok")
    return row("S17", "합성 자료로 p 역추정(F9-7)", "되찾은 p가 2/3의 ±10 %, 보정 행 2개면 추정하지 않음",
               f"p = {out.get('value', float('nan')):.4f} (보정 {out.get('n_cal')}점, 확인 {out.get('n_check')}점, 확인 RMSE "
               f"{out.get('check_rmse', float('nan')):.2e} mg/L) · 2점: {few.get('reason', '')}", ok)


def F94_saturation_defs():
    """포화도 정의 통일(F9-4): 반응 코드는 S = θ/φ(WFPS)를 쓰고 유효포화도 Se와 구분되는가(코드·값 점검)."""
    import inspect
    import moisture as MO
    import transport_reaction as TRm
    src = inspect.getsource(TRm.react)
    uses_wfps = "S_wfps = np.clip(th / phi_" in src and "Se" not in src.split("S_wfps")[1][:200]
    th_, phi_, thr_, ths_ = 0.20, 0.40, 0.05, 0.40
    S1 = float(MO.S_wfps(th_, phi_)); S2 = float(MO.Se_vg(th_, thr_, ths_))
    ok = uses_wfps and abs(S1 - 0.5) < 1e-12 and abs(S2 - (0.15 / 0.35)) < 1e-12
    return row("F9-4", "포화도 정의 통일: S = θ/φ (WFPS) ≠ Se", "반응 코드가 θ/φ 사용, 예 θ 0.20·φ 0.40·θr 0.05 → S 0.500, Se 0.429",
               f"반응 코드 S = θ/φ 사용: {uses_wfps}, S = {S1:.3f}, Se = {S2:.3f}", ok)


def S16_mode_rank():
    """모드 간 민감도(F9-5): 같은 입력(예시 두 비교군, 가벼움 격자, 20일, 내장 화학)에서 모드 A·B·C의 장치 제거율과
    비교군 순위를 내고 순위 일치 여부를 '기록'한다(합격 기준 없음). 느린 시험 — 환경변수 WP_SLOW=1일 때만 계산."""
    if os.environ.get("WP_SLOW") != "1":
        prev = None
        try:
            with open(os.path.join(APP, "selftest", "selfcheck_results.json"), encoding="utf-8") as fh:
                prev = next((r for r in json.load(fh)["table"] if r["id"] == "S16"), None)
        except Exception:      # noqa: BLE001
            prev = None
        if prev:
            return prev
        return row("S16", "모드 A·B·C 민감도·순위(F9-5)", "기록(합격 기준 없음)", "미수행(느린 시험: WP_SLOW=1)", None)
    import copy as _cp
    from engine import Scenario, run_scenario
    from io_3mf import load_3mf
    from params import GRID_PRESETS, suggest_role
    from voxel import Placement
    import install_events as ie
    import studies
    c = Common(); c.nx, c.ny, c.nz = GRID_PRESETS["가벼움"]; c.days = 20.0; c.chem_engine = "내장"; c.light = True
    c.n_snapshots = 2
    objs = []
    d = load_3mf(open(os.path.join(APP, "samples/injection_well_scale0.05.3mf"), "rb").read(), "w")
    objs.append(Scenario(name="영구 주입공", objects=d["objects"], roles={o.name: {"role": suggest_role(o.name)} for o in d["objects"]},
                         placement=Placement(0.25, 0.25, 0, 0, 0, 1),
                         events=[dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}), dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
                                 dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
                                 dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0})],
                         reagent={"form": "(c) 입자 슬러리(포획)"}))
    # v4.3: 비교 대상을 본 연구의 케이슨(불포화대 하향, caisson_down)으로 바꿈(합격 기준 없음 — '기록' 시험)
    import geometry_design as GD
    cpath = os.path.join(APP, "samples/caisson_down.3mf")
    d = load_3mf(open(cpath, "rb").read(), "w")
    objs.append(Scenario(name="케이슨형(하향)", objects=d["objects"], roles=GD.sample_roles(cpath, d["objects"]),
                         placement=Placement(0.25, 0.25, 0, 0, 0, 1), events=[dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0})],
                         reagent={"form": "(b) 고체 카트리지 교체"}))
    ctrl = run_scenario(c, None)
    main = [run_scenario(c, so) for so in objs]
    out = studies.mode_compare(c, objs, ctrl, main, cores=1)
    parts = []
    for L in ("A", "B", "C"):
        vals = []
        for nm, m in out["by_mode"].get(L, {}).items():
            mm = m.get(None) if isinstance(m, dict) and None in m else None
            vals.append(f"{nm} {mm['dev_removal']:.1f} %" if mm and mm.get("dev_removal") == mm.get("dev_removal") else f"{nm} –")
        parts.append(f"{L}: " + ", ".join(vals))
    v = out["rank"]["dev_removal"]["verdict"]
    return row("S16", "모드 A·B·C 민감도·순위(F9-5)", "기록(합격 기준 없음)",
               "장치 제거율(20일) " + " | ".join(parts) + f" → {v}", None)


# ───────────────────────────── Part 3·6 추가 시험(S8·S10·S11·S12·S18·S19) — 기준은 계산 전에 고정(R1) ─────────────────────────────
S8_CASES = [dict(a=1e-3, t_p=3e-3, K_s=1.2e-5, K_r=5e-4, K_h=5e-4),
            dict(a=1e-3, t_p=3e-3, K_s=1e-4, K_r=1e-4, K_h=1e-4),
            dict(a=2e-3, t_p=1e-3, K_s=1.2e-5, K_r=5e-4, K_h=1e-3)]


def S8_hole_resistance():
    import plate as PL
    errs = []
    for cs in S8_CASES:
        Rf = PL.hole_resistance(cs["a"], cs["t_p"], cs["K_s"], cs["K_r"], cs["K_h"])
        Rn, _ = PL.axisym_hole_conductance(cs["a"], cs["t_p"], cs["K_s"], cs["K_r"], cs["K_h"])
        errs.append(abs(Rf / Rn - 1))
    a, Ks, Kr = 1e-3, 1.2e-5, 5e-4
    lim = PL.hole_resistance(a, 1e-12, Ks, Kr, 1e30)
    lim_err = abs(lim / (1 / (4 * Ks * a) + 1 / (4 * Kr * a)) - 1)
    r0, _ = PL.plate_resistance_per_area(1.0, 0, a, 3e-3, Ks, Kr, Kr)
    dh = 0.01
    orf = PL.hole_resistance(a, 3e-3, Ks, Kr, fill="빈 공간", Cd=0.6, dh_ref=dh) - (1 / (4 * Ks * a) + 1 / (4 * Kr * a))
    orf_ex = dh / (0.6 * np.pi * a * a * np.sqrt(2 * 9.81 * dh))
    orf_err = abs(orf / orf_ex - 1)
    ok = max(errs) <= 0.05 and lim_err < 1e-9 and np.isinf(r0) and orf_err < 1e-9
    return row("S8", "구멍 접근저항 R_h vs 축대칭 세밀 격자(셀 ≤ a/8), 극한·오리피스",
               "세 경우 상대오차 ≤ 5 %; K_h→∞·t_p→0 극한 일치(1e-9); 구멍 0개 = ∞; 오리피스 Δh/Q 일치(1e-9)",
               "상대오차 " + ", ".join(f"{e * 100:.2f} %" for e in errs) + f"; 극한 {lim_err:.1e}; 구멍 0개 R={r0}; 오리피스 {orf_err:.1e}", ok)


def S10_capillary_barrier():
    """흙(사양토) 0.3 m 아래 모래 0.3 m, 처음 h = −2 m, 지표 50 mm/일. 흙 쪽 경계면 수두가 모래 진입 수두(−1/α)에 닿기 전까지
    경계면 통과 누적량이 같은 기간 지표 침투 누적량의 5 % 미만인지(모세관 장벽)."""
    sand = SOIL_TEXTURES["모래(Sand)"]
    nz = 60
    g = Grid(1, 1, nz, 0.1, 0.1, 0.6, 0.6)
    k_int = 30
    def arr(key):
        a = np.full(g.shape, SL[key], float); a[:, :, k_int:] = sand[key]; return a
    bar = (np.zeros((0, 1, nz)), np.zeros((1, 0, nz)), np.zeros((1, 1, nz - 1)), np.zeros((1, 1)))
    fv = RichardsFV(g, np.ones(g.shape, bool), arr("Ks"), arr("Ks"), arr("theta_r"), arr("theta_s"), arr("alpha"), arr("n"), bar)
    bc = FlowBC(q_top=50e-3 / 86400, bottom="자유배수")
    h = np.full(g.shape, -2.0)
    he = -1.0 / sand["alpha"]
    dt = 60.0
    cum_top = cum_int = 0.0
    reached = False
    pond = None
    t = 0.0
    for it in range(20000):
        hn, conv, nit, pn = fv.step(h, dt, bc, pond)
        if not conv:
            dt *= 0.5
            if dt < 1e-2:
                return row("S10", "모세관 장벽 1D", "수렴", f"{t / 86400:.3f}일에서 Δt {dt:.1e} s로도 수렴 실패", False)
            continue
        pond = pn
        st = fv.fluxes(hn, bc, pond)
        t += dt
        dt = min(dt * (1.5 if nit <= 6 else 1.0), 3600.0)
        h = hn
        if float(h[0, 0, k_int - 1]) >= he:
            reached = True
            break
        cum_top += float(st.Ftop.sum()) * dt
        cum_int += float(st.Fz[0, 0, k_int - 1]) * dt
    frac = cum_int / max(cum_top, 1e-30)
    return row("S10", "모세관 장벽 1D(흙 위·모래 아래)", "흙 경계면 수두가 −1/α(모래)에 닿기 전 통과량 < 지표 침투량의 5 %",
               f"진입 수두 도달 {'예' if reached else '아니오'}({t / 86400:.2f}일), 그전 통과/침투 = {frac * 100:.3f} %",
               reached and frac < 0.05)


def S11_mass_balance():
    """영구 주입공 5일(가벼움): N·S·Ca·알칼리도·입자 ≤ 1e-6, 물 ≤ 뉴턴 수렴 기준에서 유도한 상한(R4)."""
    from engine import run_scenario
    from run_verification import _scenario
    import install_events as ie
    c = Common(); c.nx, c.ny, c.nz = 22, 22, 36; c.days = 5.0; c.chem_engine = "내장"; c.light = True; c.n_snapshots = 2
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}), dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
          dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
          dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0})]
    sc, _ = _scenario(os.path.join(APP, "samples/injection_well_scale0.05.3mf"), ev)
    sc.reagent = {"form": "(c) 입자 슬러리(포획)"}
    r = run_scenario(c, sc)
    mb = r.get("mass_balance") or {}
    sol = {k: mb.get(k) for k in ("N_err_rel", "S_err_rel", "Ca_err_rel", "ALK_err_rel", "P_err_rel")}
    ok_s = all(v is not None and abs(v) <= 1e-6 for v in sol.values())
    w, wb = mb.get("W_err_rel"), mb.get("W_tol_bound")
    ok_w = w is not None and wb is not None and abs(w) <= wb
    return row("S11", "질량수지 전체 성분(R4)", "용질·입자 ≤ 1e-6, 물 ≤ Σ tol_h·C(h)·V / 유입(뉴턴 기준 유도 상한)",
               ", ".join(f"{k} {v:.1e}" for k, v in sol.items() if v is not None) + f"; 물 {w:.2e} (상한 {wb:.2e}); 상태 {r['status']}",
               r["status"] == "OK" and ok_s and ok_w)


def S12_thickness_residence():
    import geometry_design as GD
    from engine import moist_cfg
    c = Common()
    inp = dict(A_tank_m2=0.25, zone_len_m=0.5, mass_g=800.0, porosity=0.4, S_mass_frac=0.7, C0_mgN=50.0, q_mm_day=5.0,
               T_days=90.0, SF=2.0, target=0.9, S_media=0.8, d_S_m=4e-3, K_soil_m_s=1.2e-5)
    out = GD.caisson_calc(inp, moist_cfg(c), t_min_m=0.0, C_O2_molm3=REGISTRY["K_I_O2"][0])
    errs = []
    for rr in out["rows"]:
        if rr.get("상태") == "OK":
            errs.append(abs(rr["체류시간 점검(불포화, θ·체적/유량) h"] / (rr["필요 체류시간 h"] * inp["SF"]) - 1))
    return row("S12", "두께 식 z = V·t·SF 와 체류시간 θ·체적/유량 일치", "모드별 상대오차 < 1e-9",
               "상대오차 " + ", ".join(f"{e:.1e}" for e in errs), bool(errs) and max(errs) < 1e-9)


def S18_capture_fraction():
    import geometry_design as GD
    from io_3mf import load_3mf
    d = load_3mf(open(os.path.join(APP, "samples/monkstown_saturated_reference.3mf"), "rb").read(), "m")
    byname = {o.name: o for o in d["objects"]}
    A = 0.5 * 0.5
    ch = GD.hull_area_xy(byname["reactive_media_22ft"].vertices_m) / A * 100
    outer = GD.hull_area_xy(byname["outer_shell"].vertices_m) / A * 100
    ok = abs(ch - 1.2) <= 0.1 and abs(outer - 3.6) <= 0.1
    return row("S18", "포획 분율 K2(반응재·바깥 단면 / 흙수조 단면, 기존 3MF)", "챔버 1.2 ± 0.1 %, 바깥 3.6 ± 0.1 %",
               f"챔버 {ch:.2f} %, 바깥 {outer:.2f} %", ok)


def S19_resolution_log():
    import geometry_design as GD
    bad = GD.resolution_check(0.05, 0.5 / 22, gap_m=0.01)
    good = GD.resolution_check(0.124, 0.5 / 22, gap_m=0.05)
    ok = len(bad["log"]) == 2 and len(good["log"]) == 0
    return row("S19", "격자 해상 검사 K4(챔버 ≥ 4칸, 물길 ≥ 2칸) 로그", "부족 입력 → 두 항목 모두 로그, 충분 입력 → 로그 없음",
               f"부족: {bad['log']}; 충분: {good['log']}", ok)



def S3_moon2004():
    """칼럼 조건은 ⑥ 실험 보정 탭(또는 selftest/s3_column.json)에서 입력. 없으면 '미수행'."""
    import moisture as MO
    fp = os.path.join(HERE, "s3_column.json")
    col = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}
    out = MO.moon2004_check(col, float(col.get("k_half_a") or REGISTRY["k_half_a"][0]))
    item = "Moon 외(2004) 칼럼: 30·40·60 mg N/L, 반-차수 상수 31.7·33.3·36.4e-3 mg^½L^-½min^-1 (원문 확인 필요)"
    crit = "30 mg/L 보정 → 40·60 mg/L 확인: 상수 비 1/1.5~1.5, 예측 제거율 > 90 % (보정 전 비도 기록)"
    if not out["done"]:
        return dict(id="S3", 항목=item, 기준=crit, 결과="원문 칼럼 조건 미입력 — " + out["reason"], 판정="미수행")
    res = (f"보정 전 앱/문헌 비 {out['rows'][0]['ratio_raw']:.2f} ({'1.5배 안' if out['ok_raw'] else '1.5배 밖'}); 배율 {out['scale']:.3f}; "
           + "; ".join(f"{r['C0']:.0f} mg/L 비 {r['ratio_cal']:.2f}, 예측 제거 {r['removal_pred']:.1f} %" for r in out["rows"][1:])
           + f"; 체류 {out['tau_h']:.2f} h")
    return row("S3", item, crit, res, out["ok_cal"])


def NOT_DONE_rows():
    """원문이 없어 수행하지 못한 문헌 재현 시험 — 숫자를 지어내지 않고 '미수행'으로 기록."""
    return [
        dict(id="S4", 항목="Moon 외(2006) 알칼리도: 필요량 이론의 약 2배, 300 mg CaCO3/L 억제, 석회석 용해 38 mg/L/day",
             기준="방향·배수 1.5 이내", 결과="원문 미확보", 판정="미수행"),
        dict(id="S9", 항목="불포화 1D 침투 Celia 외(1990)·Haverkamp 외(1977) 기준 문제",
             기준="상대오차 1~5 %", 결과="원문(기준 문제 매개변수·해) 미확보 — V1(준해석해 ODE) 비교로 대신함", 판정="미수행"),
    ]



# ───────────────────────────── v4.3 시험 — 기준은 구현 전에 확정(R1, E4) ─────────────────────────────
# 변경 이력: 2026-11-09 v4.3에서 T-grid, PLATE-grid, CAIS-grid, D2, S5, S6, S13, D6 추가. 기준은 아래 문자열 그대로.
TGRID_CRIT = "22→33→44 격자: 첫 주입 전달량·개구부 최대 압력·개구부 평균 유량의 격자 간 상대 차이 ≤ 10 % (압력은 두 값이 모두 1 kPa 미만이면 절대 차이 ≤ 1 kPa)"
GRID_CRIT = "22→33→44 격자: 통과 유량(장치 유입 물)·장치 제거율 상대 차이 ≤ 10 %"
GRIDS = [(22, 22, 36), (33, 33, 54), (44, 44, 72)]


def _prev_row(id_):
    try:
        with open(os.path.join(APP, "selftest", "selfcheck_results.json"), encoding="utf-8") as fh:
            return next((r for r in json.load(fh)["table"] if r["id"] == id_), None)
    except Exception:      # noqa: BLE001
        return None


def _rel_spread(vals, abs_floor=None):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < len(vals):
        return np.inf
    if abs_floor is not None and np.all(np.abs(v) < abs_floor):
        return 0.0 if np.ptp(v) <= abs_floor else np.inf
    return float(np.ptp(v) / max(np.min(np.abs(v)), 1e-30))


def _well_scenario(dp=None):
    from engine import Scenario
    from io_3mf import load_3mf
    from params import suggest_role
    from voxel import Placement
    import install_events as ie
    d = load_3mf(open(os.path.join(APP, "samples/injection_well_scale0.05.3mf"), "rb").read(), "w")
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}), dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
          dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
          dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0})]
    return Scenario(name="영구 주입공", objects=d["objects"], roles={o.name: {"role": suggest_role(o.name)} for o in d["objects"]},
                    placement=Placement(0.25, 0.25, 0, 0, 0, 1), events=ev, reagent={"form": "(c) 입자 슬러리(포획)"})


def T_grid():
    """A4: 영구 주입공 주입(입자 200 µm = 체거름, 10 µm = 흙으로 들어감) — 격자 22/33/44. 느린 시험(WP_SLOW=1)."""
    if os.environ.get("WP_SLOW") != "1":
        return _prev_row("T-grid") or row("T-grid", "주입량 격자 의존(A4)", TGRID_CRIT, "미수행(느린 시험: WP_SLOW=1)", None)
    from engine import run_scenario
    res, ok_all = [], True
    for dp in (2e-4, 1e-5):
        vals = {"del": [], "P": [], "Q": []}
        for nx, ny, nz in GRIDS:
            c = Common(); c.nx, c.ny, c.nz = nx, ny, nz; c.days = 0.01; c.chem_engine = "내장"; c.light = True
            c.n_snapshots = 2; c.params["inj_d_p"] = dp
            r = run_scenario(c, _well_scenario())
            e = next((x for x in r.get("events_log", []) if str(x.get("event", "")).startswith("시약")), {})
            if r["status"] != "OK":
                vals["del"].append(None); vals["P"].append(None); vals["Q"].append(None)
                res.append(f"d={dp * 1e6:.0f}µm {nx}: {r['status']} {str(r.get('fail_reason'))[:80]}")
                continue
            vals["del"].append(e.get("delivered_L")); vals["P"].append(e.get("max_pressure_kPa")); vals["Q"].append(e.get("Q_mean_Lmin"))
        sd, sp_, sq = _rel_spread(vals["del"]), _rel_spread(vals["P"], 1.0), _rel_spread(vals["Q"])
        ok = max(sd, sp_, sq) <= 0.10
        ok_all &= ok
        fmt = lambda a: "/".join("–" if x is None else f"{x:.3g}" for x in a)
        res.append(f"d={dp * 1e6:.0f} µm: 전달 {fmt(vals['del'])} L (차 {sd * 100:.1f} %), 최대 압력 {fmt(vals['P'])} kPa "
                   f"(차 {sp_ * 100 if np.isfinite(sp_) else float('inf'):.1f} %), 평균 유량 {fmt(vals['Q'])} L/min (차 {sq * 100:.1f} %)")
    return row("T-grid", "주입량 격자 의존(A4) — 우물 모델", TGRID_CRIT, "; ".join(res), ok_all)


def _grid_device(name, scen_fn, days):
    from engine import run_scenario
    from metrics import core_metrics
    W, E, notes = [], [], []
    for nx, ny, nz in GRIDS:
        c = Common(); c.nx, c.ny, c.nz = nx, ny, nz; c.days = days; c.chem_engine = "내장"; c.light = True; c.n_snapshots = 2
        ctrl = run_scenario(c, None)
        r = run_scenario(c, scen_fn())
        if r["status"] != "OK" or ctrl["status"] != "OK":
            W.append(None); E.append(None); notes.append(f"{nx}: {r['status']}"); continue
        m = core_metrics(r, ctrl, c, None)
        W.append(m.get("dev_W_in_L")); E.append(m.get("dev_removal"))
    return W, E, notes


def _caisson_scenario():
    import geometry_design as GD
    from engine import Scenario
    from io_3mf import load_3mf
    from voxel import Placement
    import install_events as ie
    cpath = os.path.join(APP, "samples/caisson_down.3mf")
    d = load_3mf(open(cpath, "rb").read(), "c")
    return Scenario(name="케이슨형(하향)", objects=d["objects"], roles=GD.sample_roles(cpath, d["objects"]),
                    placement=Placement(0.25, 0.25, 0, 0, 0, 1), events=[dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0})],
                    reagent={"form": "(b) 고체 카트리지 교체"})


def CAIS_grid():
    """A5·B3: 케이슨(구멍 판 + 반응재층) 통과 유량과 장치 제거율의 격자 안정성(10일). 느린 시험."""
    if os.environ.get("WP_SLOW") != "1":
        return _prev_row("CAIS-grid") or row("CAIS-grid", "케이슨·구멍 판 격자 안정(A5·B3)", GRID_CRIT, "미수행(느린 시험: WP_SLOW=1)", None)
    W, E, notes = _grid_device("케이슨", _caisson_scenario, 10.0)
    sw, se = _rel_spread(W), _rel_spread(E)
    fmt = lambda a: "/".join("–" if x is None else f"{x:.4g}" for x in a)
    return row("CAIS-grid", "케이슨·구멍 판 격자 안정(A5·B3, 10일)", GRID_CRIT,
               f"통과 물 {fmt(W)} L (차 {sw * 100:.1f} %), 장치 제거율 {fmt(E)} % (차 {se * 100:.1f} %) {' '.join(notes)}",
               max(sw, se) <= 0.10)


def D2_acid():
    """D2(F7): SLAD 산 생성 — 내장 화학 1.28 eq/mol N(Batchelor & Lawrence 1978, 생체 합성 시 NH4⁺ 흡수 포함)과
    PHREEQC 반응식(전하 중성, 생체 없음) 2s−1 = 1.20 eq/mol(s = 1.10)을 같은 탈질량에서 비교. 기록 시험(차이의 원인 문서화)."""
    from transport_reaction import ChemState, ReactParams
    rp_H = REGISTRY["stoich_H"][0]
    s_ = REGISTRY["stoich_S"][0] if "stoich_S" in REGISTRY else 1.10
    phq = 2 * s_ - 1
    res = (f"내장 {rp_H:.2f} eq/mol N, PHREEQC {phq:.2f} eq/mol N, 차이 {rp_H - phq:+.2f} eq/mol "
           "— 원인: 내장 식은 Batchelor & Lawrence(1978) 양론(생체 C5H7O2N 합성, NH4⁺ 0.08 mol/mol N 흡수 → H⁺ 추가)을, "
           "PHREEQC 식은 전하 중성 반응(HNit −1, H2SO4 +s, 생체 없음)을 씀")
    return row("D2", "SLAD 산 생성 두 엔진 비교(F7)", "기록(차이와 원인 문서화)", res, None)


def S5_ratio():
    """S5: Zhang & Lampe(1999) 황:석회석 비 정성 비교(원문 확인 필요). 같은 체적의 반응재 칼럼 배치 반응에서
    석회석 비율이 클수록 최저 pH가 높고, 황 비율이 클수록 탈질량이 크다(방향만 비교)."""
    from transport_reaction import ChemState, ReactParams, react
    import engine as EN
    out = []
    for ratio in (1.0, 2.0, 3.0):
        c = Common(); c.ratio_S_CaCO3 = ratio
        r = _batch_ratio(c)
        out.append((ratio, r["pH_min"], r["denit"]))
    pHs = [x[1] for x in out]; dns = [x[2] for x in out]
    ok = (pHs[0] >= pHs[1] >= pHs[2]) and (dns[0] <= dns[1] <= dns[2])
    return row("S5", "Zhang & Lampe(1999) 황:석회석 비 정성(원문 확인 필요)",
               "석회석 비율↑ → 최저 pH↑, 황 비율↑ → 탈질량↑ (1:1·2:1·3:1에서 단조)",
               "; ".join(f"{r_:g}:1 최저 pH {p_:.2f}, 탈질 {d_ * 1e3:.3f} mmol" for r_, p_, d_ in out), ok)


def _batch_ratio(c):
    """1 L 배치(공극률 0.4, 반응재 체적분율 고정)를 2시간 내장 화학으로 반응 — S5용."""
    from transport_reaction import ChemState, ReactParams, react
    from engine import Sim
    sim = Sim(c, None)
    N = 1
    cs = sim.cs
    cs_ = ChemState(c={k: np.array([v[0]]) for k, v in cs.c.items()}, S=np.zeros(N), Cal=np.zeros(N), Gyp=np.zeros(N),
                    Bio=np.zeros(N), Inert=np.zeros(N), aCal=np.full(N, cs.aCal[0]), phi_matrix=np.array([0.4]),
                    ledger={k: 0.0 for k in cs.ledger}, phi_min=np.full(N, 0.01),
                    aS=np.zeros(N) if cs.aS is not None else None, age=np.full(N, 1e9))
    vsol = 0.6
    r = c.ratio_S_CaCO3
    cs_.S[:] = vsol * r / (r + 1) / VM["S"]
    cs_.Cal[:] = vsol / (r + 1) / VM["CaCO3"]
    if cs_.aS is not None:
        cs_.aS[:] = 6.0 * cs_.S * VM["S"] / c.p("S_d_media")
    cs_.c["NO3"][:] = 50.0 / 14.007
    theta = np.array([0.4])
    pHmin = 14.0
    tot = 0.0
    from transport_reaction import speciate_full, _pc
    for _ in range(12):                      # 2시간(10분 × 12) — 질산염이 다 없어지기 전 구간에서 비교
        tot += react(cs_, theta, 1e-3, 600.0, sim.rp)
        x_, *_ = speciate_full(cs_.c["ALK"], cs_.c["Ca"], cs_.c["SO4"], cs_.c["NO3"], _pc(sim.rp, np.array([0])), sim.rp.I_bg, None)
        pHmin = min(pHmin, float(-np.log10(x_[0])))
    return {"pH_min": pHmin, "denit": tot}


def S6_biofilm_short_circuit():
    """S6: Flere & Zhang(1998) 생물막 축적 → 단락 흐름(방향만, 원문 미확보). 2D 단면(흙 안 반응재 블록)에서 생물막 부피 분율이
    늘면(Kozeny–Carman으로 K 감소) 블록을 지나는 유량 분율이 단조 감소하는지."""
    import scipy.sparse as sp_
    import scipy.sparse.linalg as spl
    nx, nz = 30, 30
    phi0 = 0.4
    fr = []
    for fb in (0.0, 0.05, 0.10, 0.20):
        K = np.full((nx, nz), 1e-5)
        blk = np.zeros((nx, nz), bool); blk[12:18, 10:20] = True
        phi = phi0 - fb
        K[blk] = 1e-4 * (phi ** 3 / (1 - phi) ** 2) / (phi0 ** 3 / (1 - phi0) ** 2)
        N = nx * nz
        idx = np.arange(N).reshape(nx, nz)
        rows, cols, vals = [], [], []
        diag = np.zeros(N); rhs = np.zeros(N)
        for (a, b) in ((idx[:-1, :], idx[1:, :]), (idx[:, :-1], idx[:, 1:])):
            Ka, Kb = K.ravel()[a.ravel()], K.ravel()[b.ravel()]
            T = 2 * Ka * Kb / (Ka + Kb)
            rows += [a.ravel(), b.ravel()]; cols += [b.ravel(), a.ravel()]; vals += [-T, -T]
            np.add.at(diag, a.ravel(), T); np.add.at(diag, b.ravel(), T)
        top, bot = idx[:, 0], idx[:, -1]
        Tt = 2 * K[:, 0]; Tb_ = 2 * K[:, -1]
        diag[top] += Tt; rhs[top] += Tt * 1.0
        diag[bot] += Tb_
        A = sp_.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)) + sp_.diags(diag)
        H = spl.spsolve(A.tocsc(), rhs).reshape(nx, nz)
        q_blk = float(np.sum(2 * K[12:18, 0] * 0 + (H[12:18, 9] - H[12:18, 10]) * 2 * K[12:18, 9] * K[12:18, 10] / (K[12:18, 9] + K[12:18, 10])))
        q_tot = float(np.sum(Tt * (1.0 - H[:, 0])))
        fr.append(q_blk / q_tot)
    ok = all(fr[i] > fr[i + 1] for i in range(len(fr) - 1))
    return row("S6", "Flere & Zhang(1998) 생물막 단락 흐름 방향(원문 미확보)", "생물막 부피 분율 0→0.2에서 반응재 통과 유량 분율이 단조 감소",
               "통과 분율 " + " → ".join(f"{x:.3f}" for x in fr), ok)


def S13_synthetic_calib():
    """S13: 합성 보정 — 알려진 k½a로 만든 합성 관측(장치 하류 NO3)을 보정에 넣어 k½a를 되찾는지(±10 %). 느린 시험."""
    if os.environ.get("WP_SLOW") != "1":
        return _prev_row("S13") or row("S13", "합성 보정(k½a 역추정)", "참값 대비 ±10 %", "미수행(느린 시험: WP_SLOW=1)", None)
    import pandas as pd
    import studies
    from engine import run_scenario
    c = Common(); c.nx, c.ny, c.nz = 22, 22, 36; c.days = 8.0; c.chem_engine = "내장"; c.light = True; c.n_snapshots = 2
    k_true = 3.0e-3
    c.params["k_half_a"] = k_true
    so = _caisson_scenario()
    r = run_scenario(c, so)
    D = r["daily"]
    obs = []
    for d_, v_ in zip(D["day"], D["dev_Cout_mgN"]):
        if d_ >= 2 and np.isfinite(v_):
            obs.append({"시간(일)": d_, "변수": "dev_Cout_mgN", "깊이(m)": None, "값": v_, "구분": "보정" if int(d_) % 2 == 0 else "검증"})
    obs = pd.DataFrame(obs)
    c2 = Common(); c2.nx, c2.ny, c2.nz = 22, 22, 36; c2.days = 8.0; c2.chem_engine = "내장"; c2.light = True; c2.n_snapshots = 2
    CR = studies.calibrate(c2, so, obs, {"k_half_a": (1.5e-3, 6e-3, True)}, max_nfev=10)
    k_est = CR["best"]["k_half_a"]
    err = abs(k_est / k_true - 1)
    return row("S13", "합성 보정(k½a 역추정, 케이슨 8일)", "참값 대비 ±10 %",
               f"참값 {k_true:.2e}, 추정 {k_est:.2e} (오차 {err * 100:.1f} %)", err <= 0.10)


def D6_slurry_inputs():
    """D6(F10): 슬러리 값(고형분비·입경·포획 계수·주입량·유량·압력·첨가재 투입률)이 직접 입력되고 ⑧에서 스윕되는지,
    입력에 최솟값·최댓값 제한이 없는지(병렬 코어 수 제외) — 화면 코드 정적 점검."""
    import glob, re
    import studies
    keys_needed = ["event:고형분비(v/v)", "param:inj_d_p_um", "param:lambda_filt", "event:부피(L)", "event:유량(L/min)",
                   "event:압력(kPa)", "ret_frac"]
    have = {v[0] for v in studies.SWEEP_VARS.values()}
    miss = [k for k in keys_needed if k not in have]
    lim = []
    for f in [os.path.join(APP, "app_ui.py")] + glob.glob(os.path.join(APP, "ui_tabs", "*.py")):
        for i, ln in enumerate(open(f, encoding="utf-8").read().split("\n")):
            if re.search(r"number_input\(.*(min_value|max_value)", ln) and "코어" not in ln and "cores" not in ln:
                lim.append(f"{os.path.basename(f)}:{i + 1}")
    return row("D6", "슬러리 입력·스윕·제한 없음(F10)", "필요한 7개 스윕 항목 모두 있음, number_input 최솟값·최댓값 제한 0개(코어 수 제외)",
               f"빠진 스윕: {miss or '없음'}; 제한 있는 입력: {lim or '없음'}", not miss and not lim)


TESTS = [E1a_diagonal, E1b_inclined_barrier, E1c_ponding, E1d_3d_vs_1d, E1e_series_barrier, E2a_unsat_transport,
         E2b_decay, E3_ph_buffer, B3_splitting, E5_particle_theta,
         L1_koenig_size, L2_sierra_alvarez, L3_area_halforder_cell, L4_sulfur_shrinking, L5_o2_gas_diffusion,
         L6_moisture_function, D1_retainer_literature, D2_slurry_mass, S14_fw_limits, S15_no_double_count,
         S17_calib_recover, F94_saturation_defs, S16_mode_rank, S8_hole_resistance, S10_capillary_barrier,
         S11_mass_balance, S12_thickness_residence, S18_capture_fraction, S19_resolution_log,
         D2_acid, S5_ratio, S6_biofilm_short_circuit, D6_slurry_inputs, T_grid, CAIS_grid, S13_synthetic_calib]


def run_all(include_slow=True, progress=None):
    rows = []
    t0 = time.time()
    try:
        r0, figs = reuse_verify()
        rows += r0
    except Exception as e:      # noqa: BLE001
        rows.append(row("S1-3", "V1~V3 재사용", "", f"오류 {e}", False)); figs = {}
    for i, f in enumerate(TESTS):
        if not include_slow and f.__name__.startswith("E5"):
            continue
        if progress:
            progress(i / len(TESTS), f.__name__)
        try:
            rows.append(f())
        except Exception as e:      # noqa: BLE001
            import traceback
            rows.append(row(f.__name__.split("_")[0], f.__name__, "", f"오류 {e} {traceback.format_exc()[-300:]}", False))
    rows.append(S3_moon2004())
    rows += NOT_DONE_rows()
    out = {"table": rows, "figures": figs, "when": time.strftime("%Y-%m-%d %H:%M"), "sec": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "selfcheck_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    res = run_all(include_slow="--fast" not in sys.argv)
    for r in res["table"]:
        print(f"[{r['판정']}] {r['id']} {r['항목']}\n      {r['결과']}", flush=True)
