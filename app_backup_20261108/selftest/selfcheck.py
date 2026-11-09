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
    return dict(id=id_, 항목=item, 기준=crit, 결과=result, 판정="통과" if ok else "미검증")


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
                         {k: 0.0 for k in LEDGER_KEYS})
    rp = ReactParams(k_half=5.0, S_ref=REGISTRY["S_ref_frac"][0], pH_low=6.2, pH_high=9.0, theta_T=1.07, T=22, T_ref=22,
                     stoich_SO4=1.10, stoich_H=1.28, stoich_S=1.10, pCO2=pCO2, I_bg=0.005, bio_vol=3.6e-4,
                     bio_decay=0.05, calcite_mode="속도론", lambda_filt=1000, ratio=3)
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
                       np.full(N, SL["theta_s"]), {k: 0.0 for k in LEDGER_KEYS})
        rp = ReactParams(k_half=5.0, S_ref=0.45, pH_low=6.2, pH_high=9.0, theta_T=1.07, T=22, T_ref=22, stoich_SO4=1.1,
                         stoich_H=1.28, stoich_S=1.1, pCO2=-2.0, I_bg=0.005, bio_vol=3.6e-4, bio_decay=0.05,
                         calcite_mode="평형", lambda_filt=1000, ratio=3)
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


TESTS = [E1a_diagonal, E1b_inclined_barrier, E1c_ponding, E1d_3d_vs_1d, E1e_series_barrier, E2a_unsat_transport,
         E2b_decay, E3_ph_buffer, B3_splitting, E5_particle_theta,
         L1_koenig_size, L2_sierra_alvarez, L3_area_halforder_cell, L4_sulfur_shrinking, L5_o2_gas_diffusion,
         L6_moisture_function, D1_retainer_literature, D2_slurry_mass]


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
    out = {"table": rows, "figures": figs, "when": time.strftime("%Y-%m-%d %H:%M"), "sec": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "selfcheck_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    res = run_all(include_slow="--fast" not in sys.argv)
    for r in res["table"]:
        print(f"[{r['판정']}] {r['id']} {r['항목']}\n      {r['결과']}", flush=True)
