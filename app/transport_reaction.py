"""반응성 이송: 질산염·황산염·칼슘·알칼리도·주입 입자 + SLAD/석회석/석고 + 공극 막힘.

이송   : 셀 중심 유한체적, 흐름 해석의 면 유량을 그대로 사용(질량 보존).
         이류 = TVD(van Leer 제한자) 2차, 분산 = 종/횡 분산 + Millington–Quirk 확산.
         안정 조건(쿠랑/확산 수)으로 Δt 자동 분할.
반응   : 연산자 분할. 반응재가 있는 셀만 계산.
  SLAD  NO3⁻ + 1.10 S + … → 0.50 N2 + 1.10 SO4²⁻ + 1.28 H⁺ (Batchelor & Lawrence 1978)
        속도: 반-차수 dC/dt = −k·C^½ (구간 해석해 적분) × 황량·pH·온도 보정
  석회석 CaCO3 + CO2 + H2O → Ca²⁺ + 2HCO3⁻ : PWP 속도식(Plummer et al. 1978, PHREEQC 형식) 또는 평형
  석고  CaSO4·2H2O 평형 침전/용해
  pH    알칼리도·고정 pCO2·Davies 활동도에서 전하 균형으로 계산
막힘   : 공극률 = 기질 공극률 − (황 + 석회석 + 석고 + 생물막 + 주입 불활성분) 부피 → Kozeny–Carman K
수지   : 질소·황·칼슘·알칼리도(전하) 장부를 매 단계 누적.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import moisture as MO
import scipy.sparse as sp

from params import LOGK, VM, MW

SPECIES = ("NO3", "SO4", "Ca", "ALK", "P")   # mol/m3, mol/m3, mol/m3, eq/m3, m3고체/m3물


# ───────────────────────────── 화학 평형 계산 ─────────────────────────────
I_DAVIES_MAX = 0.5     # Davies 식이 쓸 만한 이온강도 상한(mol/L). 넘으면 −0.3I 항 때문에 γ가 다시 커져 포화도가 엉터리가 됨


def davies(I):
    """Davies 활동도 계수 log γ (z=1). 이온강도는 I_DAVIES_MAX에서 자른다 — 매우 마른 셀(물 몇 %)에서
    황 산화 산·석회석 용해로 농도가 극단으로 갈 때 포화도 판정이 뒤집혀 석회석이 끝없이 녹던 수치 폭주 방지(v4)."""
    I = np.minimum(np.maximum(I, 0.0), I_DAVIES_MAX)
    sI = np.sqrt(I)
    return -0.509 * (sI / (1.0 + sI) - 0.3 * I)


ION_PAIRS = {"on": True}      # 내장 화학에서 CaSO4°, CaCO3°, CaHCO3+ 이온쌍 사용(B6). PHREEQC 엔진은 DB 전체 사용.


def speciate_full(ALK, Ca, SO4, NO3, pCO2_log, I_bg, x0=None, tol=1e-9, pairs=None):
    """알칼리도(eq/m3)·총농도(mol/m3) → {H+} 활동도 x, γ1, γ2, CO2(aq), 자유 Ca·SO4(mol/m3).
    알칼리도 정의에 이온쌍(CaHCO3+, CaCO3°)을 포함한 전하/알칼리도 식
        A = (HCO3 + OH + CaHCO3) + 2(CO3 + CaCO3°) − H
    을 로그 공간 뉴턴으로 풀고, 자유 이온은 질량수지 고정점 반복으로 구한다. 상수는 phreeqc.dat 값."""
    pairs = ION_PAIRS["on"] if pairs is None else pairs
    K1, K2, KH, Kw = (10 ** LOGK[k] for k in ("K1", "K2", "KH", "Kw"))
    kS, kC, kHC = (10 ** LOGK[k] for k in ("CaSO4", "CaCO3", "CaHCO3"))
    CO2 = KH * 10 ** np.asarray(pCO2_log, float)
    A = np.asarray(ALK, float) * 1e-3
    CaT = np.maximum(np.asarray(Ca, float), 0) * 1e-3
    ST = np.maximum(np.asarray(SO4, float), 0) * 1e-3
    N = np.maximum(np.asarray(NO3, float), 0) * 1e-3
    cf, sf = CaT.copy(), ST.copy()
    x = None if x0 is None else np.asarray(x0, float).copy()
    for outer in range(5 if pairs else 1):
        I = I_bg + 0.5 * (4 * cf + 4 * sf + N + np.abs(A) + (CaT - cf - (ST - sf)) * (1 if pairs else 0))
        lg = davies(I)
        g1 = 10 ** lg
        g2 = 10 ** (4 * lg)
        a = (K1 * CO2 + Kw) / g1
        b = 2 * K1 * K2 * CO2 / g2
        if pairs:
            a = a + kHC * g2 * cf * K1 * CO2 / g1
            b = b + 2 * kC * g2 * cf * K1 * K2 * CO2
        if x is None:
            x = np.where(A > 1e-7, a / np.maximum(A, 1e-7), np.maximum(-A * g1, 1e-7))
        x = np.minimum(np.maximum(x, 1e-13), 1.0)
        u = np.log(x)
        for _ in range(30):
            x = np.exp(u)
            f = a / x + b / (x * x) - x / g1 - A
            fu = -a / x - 2 * b / (x * x) - x / g1
            du = np.clip(-f / fu, -2.3, 2.3)
            u = u + du
            if np.max(np.abs(du)) < tol:
                break
        x = np.minimum(np.maximum(np.exp(u), 1e-13), 1.0)
        if not pairs:
            break
        rC = kC * g2 * K1 * K2 * CO2 / (x * x)          # CaCO3°/Ca_free
        rHC = kHC * g2 * K1 * CO2 / (g1 * x)              # CaHCO3+/Ca_free
        cf_new = CaT / (1.0 + kS * g2 * g2 * sf + rC + rHC)
        sf_new = ST / (1.0 + kS * g2 * g2 * cf_new)
        dmax = float(np.max(np.abs(cf_new - cf) / np.maximum(CaT, 1e-12))) if np.size(cf) else 0.0
        cf, sf = 0.5 * (cf + cf_new), 0.5 * (sf + sf_new)
        if dmax < 1e-6:
            cf, sf = cf_new, sf_new
            break
    return x, g1, g2, CO2, cf * 1e3, sf * 1e3


def speciate(ALK, Ca, SO4, NO3, pCO2_log, I_bg, x0=None, tol=1e-9):
    x, g1, g2, CO2, _, _ = speciate_full(ALK, Ca, SO4, NO3, pCO2_log, I_bg, x0, tol)
    return x, g1, g2, CO2


def pH_of(ALK, Ca, SO4, NO3, pCO2_log, I_bg):
    x, *_ = speciate(ALK, Ca, SO4, NO3, pCO2_log, I_bg)
    return -np.log10(x)


def alk_from_pH(pH, pCO2_log, I_bg=0.005):
    """초기 토양수/유입수 pH → 알칼리도(eq/m3) (Ca·SO4·이온쌍 영향 무시한 첫 근사)."""
    x = 10.0 ** (-pH)
    lg = davies(I_bg); g1 = 10 ** lg; g2 = 10 ** (4 * lg)
    K1, K2, KH, Kw = (10 ** LOGK[k] for k in ("K1", "K2", "KH", "Kw"))
    CO2 = KH * 10 ** pCO2_log
    A = (K1 * CO2 + Kw) / (g1 * x) + 2 * K1 * K2 * CO2 / (g2 * x * x) - x / g1
    return A * 1e3


def calcite_SR(ALK, Ca, SO4, NO3, pCO2_log, I_bg, x0=None):
    """석회석 포화비 SR = γ2[Ca]free·{CO3}/Ksp (이온쌍 반영)."""
    x, g1, g2, CO2, cf, sf = speciate_full(ALK, Ca, SO4, NO3, pCO2_log, I_bg, x0)
    K1, K2 = 10 ** LOGK["K1"], 10 ** LOGK["K2"]
    aCO3 = K1 * K2 * CO2 / x ** 2
    return g2 * cf * 1e-3 * aCO3 / 10 ** LOGK["Ksp_cal"], x, CO2


# ───────────────────────────── 상태 ─────────────────────────────────────
@dataclass
class ChemState:
    c: Dict[str, np.ndarray]          # 수용액 (셀 평탄 배열)
    S: np.ndarray                     # mol/m3 벌크
    Cal: np.ndarray
    Gyp: np.ndarray
    Bio: np.ndarray                   # m3/m3
    Inert: np.ndarray                 # 주입 불활성 고형분 m3/m3
    aCal: np.ndarray                  # 석회석 비표면적 m2/m3(석회석)
    phi_matrix: np.ndarray            # 첨가 고형분이 없을 때의 공극률
    ledger: Dict[str, float] = field(default_factory=dict)
    O2g: Optional[np.ndarray] = None  # 기상 O2 mol/m3 벌크(B5 옵션)
    phi_min: Optional[np.ndarray] = None   # 셀별 포획 하한 공극률(토양 0.05, 개구부·관 내부 = 충전층 공극률)
    aS: Optional[np.ndarray] = None   # 황 표면적 계수 cA = a_S/φ_S^(2/3) [m2/m3 벌크] — 표면적 기반 속도식(v4)
    age: Optional[np.ndarray] = None  # 셀에 황이 생긴 뒤 지난 시간(일) — 미생물 적응 지연(v4)

    def solid_volume(self):
        return self.S * VM["S"] + self.Cal * VM["CaCO3"] + self.Gyp * VM["gyp"] + self.Bio + self.Inert

    def porosity(self):
        return np.clip(self.phi_matrix - self.solid_volume(), 0.02, 0.99)

    def copy(self):
        return ChemState({k: v.copy() for k, v in self.c.items()}, self.S.copy(), self.Cal.copy(),
                         self.Gyp.copy(), self.Bio.copy(), self.Inert.copy(), self.aCal.copy(),
                         self.phi_matrix.copy(), dict(self.ledger),
                         None if self.O2g is None else self.O2g.copy(),
                         None if self.phi_min is None else self.phi_min.copy(),
                         None if self.aS is None else self.aS.copy(),
                         None if self.age is None else self.age.copy())


LEDGER_KEYS = [
    "N_in", "N_out", "N_denit", "N_store0",
    "S_in", "S_out", "S_solid0", "S_inj", "SO4_store0", "Gyp0",
    "Ca_in", "Ca_out", "Ca_store0", "Cal_diss", "Gyp_prec",
    "ALK_in", "ALK_out", "ALK_store0", "ALK_denit", "ALK_cal",
    "P_inj", "P_dep", "P_out", "PC_inj", "PC_dep", "PC_out",
    "W_in", "W_out", "W_swap",
    "IF_N", "IF_W", "IF_SO4",     # 경계면(토양층 바닥) 누적
]


def add_sulfur(cs: "ChemState", dS, d_p, idx=None):
    """황 고체 dS(mol/m3 벌크)를 지름 d_p(m) 입자로 더한다. 표면적 계수 cA = a_S/φ_S^(2/3)를 갱신:
    기존 입자 표면(φ_old^(2/3)·cA) + 새 입자 표면 6·dφ/d_p. 이후 입자가 녹을 때 표면은 φ^(2/3)에 비례(구형 축소,
    Watkinson 1989). 표면적 기반이 아니면(cs.aS None) 양만 더한다."""
    sel = slice(None) if idx is None else idx
    dS = np.asarray(dS, float)
    if cs.aS is not None:
        phi_old = cs.S[sel] * VM["S"]
        dphi = np.maximum(dS, 0.0) * VM["S"]
        a_old = cs.aS[sel] * np.power(np.maximum(phi_old, 0.0), 2.0 / 3.0)
        a_new = a_old + 6.0 * dphi / max(float(d_p), 1e-9)
        phi_new = phi_old + dphi
        cs.aS[sel] = np.where(phi_new > 0, a_new / np.power(np.maximum(phi_new, 1e-30), 2.0 / 3.0), cs.aS[sel])
        if cs.age is not None:
            cs.age[sel] = np.where(phi_old > 0, cs.age[sel], 0.0)      # 새로 황이 생긴 셀은 적응 시계를 0부터
    cs.S[sel] = cs.S[sel] + dS


KHO2 = 1.3              # mol/m3/atm (O2 헨리 상수 ≈1.3e-3 mol/L/atm, 25°C, Sander 2015)
R_ATM = 8.206e-5        # m3·atm/(mol·K)


def o2_gas_diffuse(cs: "ChemState", tr: "Transport", theta, dt, T_c, D0, active=None):
    """토양 공기 중 O2 확산(지표 = 대기 21%) — 암시적 오일러 한 단계. Millington & Quirk (1961):
    D_eff = D0·ε_a^(10/3)/φ². 셀 안에서 기상·용존은 헨리 평형(H = K_H·R·T). 반환: 대기에서 들어온 O2 mol."""
    import scipy.sparse.linalg as spl
    g = tr.g
    N = tr.N
    V = g.vcell
    th = np.maximum(np.asarray(theta, float), 0.0)
    phi = cs.porosity()
    ea = np.clip(phi - th, 0.0, None)
    H = KHO2 * R_ATM * (T_c + 273.15)                   # 용존/기상 농도비(무차원)
    Deff = D0 * ea ** (10.0 / 3.0) / np.maximum(phi, 1e-3) ** 2
    capv = ea + th * H                                   # 기상 농도 1 mol/m3당 벌크 O2 [mol/m3]
    ok = capv > 1e-9
    if active is not None:
        ok &= active
    if cs.O2g is None:
        cs.O2g = np.zeros(N)
    O2 = cs.c["O2"]
    tot = cs.O2g + O2 * th
    Cg = np.where(ok, tot / np.where(ok, capv, 1.0), 0.0)
    rows, cols, vals = [], [], []
    diag = np.where(ok, capv * V / dt, 1.0)
    for f in tr.faces:
        a, b = f["a"], f["b"]
        Gf = f["A"] / f["d"] * 2 * Deff[a] * Deff[b] / np.maximum(Deff[a] + Deff[b], 1e-40)
        Gf = np.where(f["seal"] | ~ok[a] | ~ok[b], 0.0, Gf)
        m = Gf > 0
        a, b, Gf = a[m], b[m], Gf[m]
        rows += [a, b]; cols += [b, a]; vals += [-Gf, -Gf]
        diag += np.bincount(a, weights=Gf, minlength=N) + np.bincount(b, weights=Gf, minlength=N)
    Catm = 0.21 / (R_ATM * (T_c + 273.15))
    rhs = np.where(ok, capv * V / dt * Cg, 0.0)
    top = tr.top_cells
    Gt = np.where(ok[top], g.dx * g.dy / (0.5 * g.dz) * Deff[top], 0.0)
    diag[top] += Gt
    rhs[top] += Gt * Catm
    A = sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0),
                       (np.concatenate(rows) if rows else np.zeros(0, int),
                        np.concatenate(cols) if cols else np.zeros(0, int))), shape=(N, N))
    A = A + sp.diags(diag)
    M = sp.diags(1.0 / diag)
    x, info = spl.cg(A, rhs, x0=Cg, rtol=1e-10, atol=0.0, maxiter=2000, M=M)
    if info != 0:
        x = spl.spsolve(A.tocsc(), rhs)
    x = np.where(ok, np.maximum(x, 0.0), Cg)
    q_in = float((Gt * (Catm - x[top])).sum() * dt)
    cs.O2g = np.where(ok, ea * x, cs.O2g)
    cs.c["O2"] = np.where(ok & (th > 0), H * x, O2)
    cs.ledger["O2_atm_in"] = cs.ledger.get("O2_atm_in", 0.0) + q_in
    return q_in


# ───────────────────────────── 이송 연산자 ───────────────────────────────
class Transport:
    def __init__(self, g, fv, params):
        """fv: RichardsFV(위상 재사용). params: dict(alpha_L, alpha_T_ratio, Dm)."""
        self.g = g
        self.fv = fv
        self.N = g.N
        self.p = params
        nx, ny, nz = g.shape
        idx = np.arange(g.N).reshape(g.shape)
        self.faces = []
        for ax, f in enumerate(fv.faces):
            a, b = f["a"], f["b"]                        # 전체 인덱스
            stride = [ny * nz, nz, 1][ax]
            n_ax = g.shape[ax]
            pos_a = np.unravel_index(a, g.shape)[ax]
            pos_b = np.unravel_index(b, g.shape)[ax]
            am = np.where(pos_a > 0, a - stride, -1)
            bp = np.where(pos_b < n_ax - 1, b + stride, -1)
            act = fv.active.ravel()
            am = np.where((am >= 0) & act[np.maximum(am, 0)], am, -1)
            bp = np.where((bp >= 0) & act[np.maximum(bp, 0)], bp, -1)
            seal = f["R"] > 0
            self.faces.append(dict(a=a, b=b, am=am, bp=bp, fid=f["fid"], A=f["A"], d=f["d"], ax=ax,
                                   seal=seal))
        self.k_int = g.k_interface
        self.top_cells = idx[:, :, 0].ravel()
        self.bot_cells = idx[:, :, -1].ravel()
        self.left_cells = idx[0, :, :].ravel()
        self.right_cells = idx[-1, :, :].ravel()
        self.set_flow(None)

    def set_flow(self, st, theta=None):
        self.st = st
        if st is None:
            return
        g = self.g
        self.theta = st.theta.ravel().copy() if theta is None else theta
        self.Q = []
        for f, F in zip(self.faces, (st.Fx, st.Fy, st.Fz)):
            self.Q.append(F.ravel()[f["fid"]])
        # 셀 유속(공극 유속) 벡터
        V = g.vcell
        th = np.maximum(self.theta, 1e-6)
        vcomp = []
        for ax, (F, A) in enumerate(zip((st.Fx, st.Fy, st.Fz), (g.dy * g.dz, g.dx * g.dz, g.dx * g.dy))):
            pad = [(0, 0)] * 3
            pad[ax] = (1, 0); Fm = np.pad(F, pad)       # 셀 앞면
            pad[ax] = (0, 1); Fp = np.pad(F, pad)       # 셀 뒷면
            if ax == 2:
                Fm[:, :, 0] = st.Ftop
                Fp[:, :, -1] = st.Fbot
            if ax == 0:
                Fm[0, :, :] = st.Fleft
                Fp[-1, :, :] = st.Fright
            vcomp.append((0.5 * (Fm + Fp) / A).ravel() / th)
        self.v = np.stack(vcomp)                          # (3, N) 공극 유속 m/s
        self.vmag = np.sqrt((self.v ** 2).sum(0)) + 1e-30
        aL = self.p["alpha_L"]; aT = aL * self.p["alpha_T_ratio"]
        Dm = self.p["Dm"]
        phi = np.maximum(self.p.get("phi_cells", np.full(self.N, 0.4)), 1e-3)
        Ddiff = Dm * np.maximum(self.theta, 0) ** (10.0 / 3.0) / phi ** 2      # 벌크 유효 확산
        self.G = []
        out = np.zeros(self.N)
        for ax, f in enumerate(self.faces):
            a, b = f["a"], f["b"]
            Dmech_c = aT * self.vmag + (aL - aT) * self.v[ax] ** 2 / self.vmag
            th_f = 0.5 * (self.theta[a] + self.theta[b])
            Dm_f = 0.5 * (Dmech_c[a] + Dmech_c[b]) * th_f
            Dd_f = 2 * Ddiff[a] * Ddiff[b] / np.maximum(Ddiff[a] + Ddiff[b], 1e-40)
            G = f["A"] / f["d"] * (Dm_f + Dd_f)
            G = np.where(f["seal"], 0.0, G)
            self.G.append(G)
            Q = self.Q[ax]
            out += np.bincount(a, weights=np.maximum(Q, 0) + G, minlength=self.N)
            out += np.bincount(b, weights=np.maximum(-Q, 0) + G, minlength=self.N)
        # 경계 유출
        out[self.top_cells] += np.maximum(-st.Ftop.ravel(), 0)
        out[self.bot_cells] += np.maximum(st.Fbot.ravel(), 0)
        out[self.left_cells] += np.maximum(-st.Fleft.ravel(), 0)
        out[self.right_cells] += np.maximum(st.Fright.ravel(), 0)
        out += np.maximum(-st.Fsrc.ravel(), 0)
        cap = self.theta * g.vcell
        ok = (cap > 0) & (out > 0)
        self.dt_max = float(np.min(0.45 * cap[ok] / out[ok])) if ok.any() else 1e9
        self._build_operator()

    def _build_operator(self):
        """면 유량이 고정된 동안 쓸 선형 연산자: dM/dt = L·c (1차 풍상 이류 + 분산) + Dv·(제한자 보정)."""
        N = self.N
        A_, B_, UP, DN, UU, HAS, Q_, G_, IF = [], [], [], [], [], [], [], [], []
        kif = self.k_int
        for ax, f in enumerate(self.faces):
            Q = self.Q[ax]
            a, b = f["a"], f["b"]
            pos = Q > 0
            up = np.where(pos, a, b); dn = np.where(pos, b, a)
            iuu = np.where(pos, f["am"], f["bp"])
            has = iuu >= 0
            A_.append(a); B_.append(b); UP.append(up); DN.append(dn); UU.append(np.where(has, iuu, up))
            HAS.append(has); Q_.append(Q); G_.append(self.G[ax])
            if ax == 2 and kif < self.g.nz:
                IF.append(np.unravel_index(b, self.g.shape)[2] == kif)
            else:
                IF.append(np.zeros(len(a), bool))
        a = np.concatenate(A_); b = np.concatenate(B_)
        self.fa_all, self.fb_all = a, b
        self.up_all = np.concatenate(UP); self.dn_all = np.concatenate(DN); self.uu_all = np.concatenate(UU)
        self.has_all = np.concatenate(HAS); self.Q_all = np.concatenate(Q_); self.G_all = np.concatenate(G_)
        self.if_all = np.concatenate(IF)
        nf = len(a)
        fidx = np.arange(nf)
        # Dv: 면 유량 → 셀 질량 변화 (b에 +, a에 −)
        self.Dv = sp.csr_matrix((np.concatenate([np.ones(nf), -np.ones(nf)]),
                                 (np.concatenate([b, a]), np.concatenate([fidx, fidx]))), shape=(N, nf))
        # 면 플럭스 = Q·c[up] + G·(c[a] − c[b])  (1차 부분)
        Fm = sp.csr_matrix((np.concatenate([self.Q_all, self.G_all, -self.G_all]),
                            (np.concatenate([fidx, fidx, fidx]), np.concatenate([self.up_all, a, b]))), shape=(nf, N))
        self.L = (self.Dv @ Fm).tocsr()
        self.Fm = Fm

    def step(self, c: Dict[str, np.ndarray], dt: float, bc_conc: Dict[str, Dict[str, float]],
             ledger: Dict[str, float], src_conc: Optional[Dict[str, float]] = None, iface=True,
             retard: Optional[Dict[str, float]] = None):
        """질량 M=R·θVc 갱신(이류+분산+경계). bc_conc: {'top':{종:농도}, 'gw':{...}}.
        retard: 종별 지연계수 R(흡착, 선형 평형) — 액체 공여체용. R=1이면 보통 이송."""
        g = self.g
        st = self.st
        V = g.vcell
        th = self.theta
        cap = th * V
        PART = ("P", "PC")
        names = [k for k in c.keys() if k not in PART or np.any(c[k] > 0) or (src_conc or {}).get(k, 0) > 0]
        dM = {k: np.zeros(self.N) for k in names}
        kif = self.k_int
        Q = self.Q_all
        has = self.has_all
        ifm = self.if_all
        for k in names:
            ck = c[k]
            cu = ck[self.up_all]; cd = ck[self.dn_all]; cuu = ck[self.uu_all]
            den = cd - cu
            big = np.abs(den) > 1e-30
            r = np.where(big, (cu - cuu) / np.where(big, den, 1.0), 0.0)
            ar = np.abs(r)
            psi = np.where(has, (r + ar) / (1.0 + ar), 0.0)
            corr = Q * 0.5 * psi * den
            dM[k] += dt * (self.L @ ck + self.Dv @ corr)
            if iface and kif < g.nz:
                Fi = (self.Fm[ifm] @ ck + corr[ifm]).sum() * dt
                if k == "NO3":
                    ledger["IF_N"] += float(Fi)
                elif k == "SO4":
                    ledger["IF_SO4"] += float(Fi)
        if iface and kif < g.nz:
            ledger["IF_W"] += float(Q[ifm].sum()) * dt
        # 경계
        top_c = bc_conc.get("top", {})
        gw_c = bc_conc.get("gw", {})
        Ft = st.Ftop.ravel()
        Fb = st.Fbot.ravel()
        Fl = st.Fleft.ravel()
        Fr = st.Fright.ravel()
        Fs = st.Fsrc.ravel()
        tc, bcl, lc, rc = self.top_cells, self.bot_cells, self.left_cells, self.right_cells
        for k in names:
            ck = c[k]
            cin_top = top_c.get(k, 0.0)
            cgw = gw_c.get(k, 0.0)
            # 지표 유출(Ft<0)은 증발·지표 유출수: 용질은 남긴다(증발 농축, 가정)
            fin_t = np.where(Ft > 0, Ft * cin_top, 0.0) * dt
            dM[k][tc] += fin_t
            if k == "NO3":
                ledger["N_top_in"] = ledger.get("N_top_in", 0.0) + float(fin_t.sum())
            fout_b = np.where(Fb > 0, Fb * ck[bcl], Fb * cgw) * dt
            dM[k][bcl] -= fout_b
            fin_l = np.where(Fl > 0, Fl * cgw, Fl * ck[lc]) * dt
            dM[k][lc] += fin_l
            fout_r = np.where(Fr > 0, Fr * ck[rc], Fr * cgw) * dt
            dM[k][rc] -= fout_r
            sc = (src_conc or {}).get(k, 0.0)
            fs = np.where(Fs > 0, Fs * sc, Fs * ck) * dt
            dM[k] += fs
            inflow = fin_t.clip(min=0).sum() + (-fout_b).clip(min=0).sum() + fin_l.clip(min=0).sum() \
                + (-fout_r).clip(min=0).sum() + fs.clip(min=0).sum()
            outflow = (-fin_t).clip(min=0).sum() + fout_b.clip(min=0).sum() + (-fin_l).clip(min=0).sum() \
                + fout_r.clip(min=0).sum() + (-fs).clip(min=0).sum()
            key = {"NO3": "N", "SO4": "S", "Ca": "Ca", "ALK": "ALK"}.get(k, k)
            if k in PART:
                ledger[f"{k}_out"] = ledger.get(f"{k}_out", 0.0) + float(outflow)
                ledger[f"{k}_srcin"] = ledger.get(f"{k}_srcin", 0.0) + float(inflow)
            else:
                ledger[f"{key}_in"] = ledger.get(f"{key}_in", 0.0) + float(inflow)
                ledger[f"{key}_out"] = ledger.get(f"{key}_out", 0.0) + float(outflow)
            if iface and kif >= g.nz and k in ("NO3", "SO4"):   # 지하수층 없음: 바닥 유출 = 경계면
                ledger["IF_N" if k == "NO3" else "IF_SO4"] += float(fout_b.sum())
        win = (Ft.clip(min=0).sum() + (-Fb).clip(min=0).sum() + Fl.clip(min=0).sum()
               + (-Fr).clip(min=0).sum() + Fs.clip(min=0).sum()) * dt
        wout = ((-Ft).clip(min=0).sum() + Fb.clip(min=0).sum() + (-Fl).clip(min=0).sum()
                + Fr.clip(min=0).sum() + (-Fs).clip(min=0).sum()) * dt
        ledger["W_in"] += float(win); ledger["W_out"] += float(wout)
        ledger["W_top_in"] = ledger.get("W_top_in", 0.0) + float(Ft.clip(min=0).sum() * dt)
        if iface and kif >= g.nz:
            ledger["IF_W"] += float(Fb.sum() * dt)
        ok = cap > 0
        for k in names:
            R = (retard or {}).get(k, 1.0)
            M = c[k] * cap * R + dM[k]
            c[k] = np.where(ok, M / np.where(ok, cap * R, 1.0), 0.0)
            if k != "ALK":
                c[k] = np.maximum(c[k], 0.0)
        return c


# ───────────────────────────── 반응 ─────────────────────────────────────
@dataclass
class ReactParams:
    k_half: float
    S_ref: float
    pH_low: float
    pH_high: float
    theta_T: float
    T: float
    T_ref: float
    stoich_SO4: float
    stoich_H: float
    stoich_S: float
    pCO2: object             # 스칼라 또는 셀별 배열(log10 atm) — B4
    I_bg: float
    bio_vol: float
    bio_decay: float
    calcite_mode: str
    lambda_filt: float
    ratio: float              # 황:석회석 부피비
    bio_det: float = 0.0      # 생물막 유속 탈착 계수 [1/m] (가정)
    o2_on: bool = False       # 초기 O2 소비 단계(B5)
    k_O2: float = 0.0         # 황 표면 산화 속도 [mol O2/m3물/s] (fS=1 기준, 가정)
    K_O2: float = 0.0         # 반포화 [mol/m3]
    K_I_O2: float = 0.0       # 탈질 억제 상수 [mol/m3]
    donor: Optional[dict] = None   # 액체 공여체 반응 정의(yD, ySO4, yH, k, KN, KD) — 미정이면 None
    chem: str = "SLAD"        # "SLAD" | "추적자(반응 없음)" | "1차 감쇠"
    k1_decay: float = 0.0     # 대체품 모드 1차 감쇠 [1/s]
    # ── v4 표면적 기반 속도식(문헌) ─────────────────────────────
    kin: str = "volume"       # "area"(v4: Koenig & Liu 2001 표면적 반-차수) | "volume"(v3: 황 부피비^(2/3))
    k_a: float = 0.0          # 표면 반-차수 상수 [(mg/L)^½·m/h] — 체적 상수 = k_a × 비표면적(m2/m3)
    wfps_thr: float = 0.62    # 탈질 수분 함수 문턱 포화도(NEMIS, Hénault & Germon 2000)
    wfps_exp: float = 1.74    # 수분 함수 지수(NEMIS)
    k_ox_a: float = 0.0       # 호기성 황 산화 표면 속도 [mol S/m2/s] (Watkinson & Blair 1993, 25°C)
    Ea_ox: float = 85e3       # 호기성 황 산화 활성화 에너지 [J/mol] (Watkinson & Blair 1993)
    s_opt_ox: float = 0.6     # 호기성 황 산화가 가장 빠른 포화도(포장용수량 근처, 같은 문헌의 정성 기술 — 가정)
    lag_d: float = 0.0        # 미생물 적응 지연 시정수 [일] (0이면 없음)
    s_set: Optional[float] = None   # 반응 구역 포화도 지정(조작 변수). None이면 Richards 계산값
    moist_mode: str = "A"     # F9 수분 의존 모드: "A" 기계론(f_w·f_O2) | "B" 경험식(NEMIS 계열) | "C" 상한
    moist_cfg: Optional[dict] = None   # moisture.factors 설정(p, fw_form, theta_c, thr, w, nemis_form, beta)
    nitrite: bool = False              # D4: NO3→NO2→N2 2단계
    k_NO2_ratio: float = 1.0           # D4: 2단계 상수 / 1단계 상수
    het_k: float = 0.0                 # D1: 종속영양 탈질 1차 상수 1/s (토양층 셀 전체)
    het_mask: Optional[np.ndarray] = None


def _fpH(pH, rp: ReactParams):
    f = 1.0 / (1.0 + 10 ** (rp.pH_low - pH)) / (1.0 + 10 ** (pH - rp.pH_high))
    f0 = 1.0 / (1.0 + 10 ** (rp.pH_low - 7.5)) / (1.0 + 10 ** (7.5 - rp.pH_high))
    return f / f0


def _pc(rp, ii=None):
    p = rp.pCO2
    if np.ndim(p) == 0:
        return float(p)
    return np.asarray(p)[ii] if ii is not None else np.asarray(p)


def _calcite_eq_delta(ALK, Ca, SO4, NO3, lo, hi, rp, x0=None, pc=None):
    """SR(Δ)=1 이 되는 석회석 용해량 Δ (mol/m3 물). log SR(Δ)는 Δ에 대해 증가 → Illinois 가위치법."""
    pc = rp.pCO2 if pc is None else pc
    pcv = np.broadcast_to(np.asarray(pc, float), np.shape(ALK)).copy()
    a, b = lo.copy(), hi.copy()
    fa = np.log(np.maximum(calcite_SR(ALK + 2 * a, Ca + a, SO4, NO3, pcv, rp.I_bg, x0)[0], 1e-300))
    fb = np.log(np.maximum(calcite_SR(ALK + 2 * b, Ca + b, SO4, NO3, pcv, rp.I_bg, x0)[0], 1e-300))
    out = np.where(fa >= 0, a, np.where(fb <= 0, b, 0.5 * (a + b)))
    act = (fa < 0) & (fb > 0)
    if not act.any():
        return out
    a, b, fa, fb = a[act], b[act], fa[act], fb[act]
    args = [v[act] for v in (ALK, Ca, SO4, NO3)]
    pca = pcv[act]
    side = np.zeros(len(a))
    m = 0.5 * (a + b)
    for _ in range(30):
        m = (a * fb - b * fa) / (fb - fa)
        m = np.where(np.isfinite(m), m, 0.5 * (a + b))
        fm = np.log(np.maximum(calcite_SR(args[0] + 2 * m, args[1] + m, args[2], args[3], pca, rp.I_bg)[0], 1e-300))
        left = fm < 0
        fb = np.where(left & (side == -1), fb * 0.5, fb)
        fa = np.where(~left & (side == 1), fa * 0.5, fa)
        a = np.where(left, m, a); fa = np.where(left, fm, fa)
        b = np.where(left, b, m); fb = np.where(left, fb, fm)
        side = np.where(left, -1, 1)
        if np.max(np.abs(fm)) < 1e-6 or np.max(b - a) < 1e-12:
            break
    out[act] = m
    return out


_K3_SHIFT = (-5.86 - 317.0 / 298.15) - (-1.1 - 1737.0 / 298.15)


def _pwp_k(Tc):
    """PWP 속도상수(Plummer et al. 1978; PHREEQC 형식). k3의 25°C 위 식을 25°C에서 아래 식과 같아지도록
    이동시켜 연속으로 만들었다(원식은 25°C에서 0.6% 불연속 — 알려진 문제 A5)."""
    T = Tc + 273.15
    k1 = 10 ** (0.198 - 444.0 / T)
    k2 = 10 ** (2.84 - 2177.0 / T)
    k3 = 10 ** (-5.86 - 317.0 / T) if T <= 298.15 else 10 ** (-1.1 - 1737.0 / T + _K3_SHIFT)
    return 10 * k1, 10 * k2, 10 * k3     # mmol/cm2/s → mol/m2/s


ALK_STEP_MAX = 0.1      # 하위 단계당 알칼리도 변화 상한 eq/m3 (= meq/L) — A5에서 0.3 → 0.1
OX_ALK_STEP = 10.0      # 호기성 황 산화 산의 하위 단계당 상한 eq/m3 (pH와 무관한 속도라 크게, v4). 시험: 0.1 대비 황 산화량 +0.09%, 유출 SO4 +0.02%


def _het_denit(cs, theta, V, dt, rp):
    """D1 옵션: 토양 유기탄소를 쓰는 종속영양 탈질(1차, 토양층 셀). 5CH2O + 4NO3⁻ → 2N2 + 4HCO3⁻ + CO2 + 3H2O
    → 알칼리도 +1 eq/mol N. 수분 의존은 선택한 수분 모드의 f_moist를 그대로 쓴다. 상수는 사용자 입력(기본 꺼짐)."""
    m = (theta > 1e-6) & (rp.het_mask if rp.het_mask is not None else True)
    if not np.any(m):
        return
    ii = np.where(m)[0]
    th = theta[ii]
    phi_ = np.maximum(cs.porosity()[ii], 1e-6)
    fW, _ = MO.factors(rp.moist_mode, np.clip(th / phi_, 0, 1), None, rp.moist_cfg or {})
    NO3 = cs.c["NO3"][ii]
    dn = NO3 * (1 - np.exp(-rp.het_k * fW * dt))
    cs.c["NO3"][ii] = NO3 - dn
    cs.c["ALK"][ii] += dn
    dmol = float((dn * th).sum() * V)
    cs.ledger["N_denit"] += dmol
    cs.ledger["N_denit_het"] = cs.ledger.get("N_denit_het", 0.0) + dmol
    cs.ledger["ALK_denit"] -= dmol          # 산 생성 장부에 음수(= 알칼리도 생성)


def react(cs: ChemState, theta: np.ndarray, V: float, dt: float, rp: ReactParams,
          cell_mask: Optional[np.ndarray] = None, vmag: Optional[np.ndarray] = None, _split=False):
    """반응재·석고·생물막(또는 액체 공여체·O2)이 있는 셀만 반응. 장부(ledger) 갱신. 내장 화학 엔진."""
    L = cs.ledger
    c_ = cs.c
    if rp.chem == "추적자(반응 없음)":
        return 0.0
    if rp.chem == "1차 감쇠":
        f = np.exp(-rp.k1_decay * dt)
        m = theta > 1e-9
        lost = float(((1 - f) * c_["NO3"][m] * theta[m]).sum() * V)
        c_["NO3"][m] *= f
        L["N_denit"] += lost
        return lost
    th_all = theta
    if cell_mask is None:
        cell_mask = (cs.S > 0) | (cs.Cal > 0) | (cs.Gyp > 0)
        if rp.donor is not None and "D" in c_:
            cell_mask = cell_mask | (c_["D"] > 0)
    cell_mask = cell_mask & (th_all > 1e-6)
    if rp.het_k > 0:
        _het_denit(cs, theta, V, dt, rp)
    if not cell_mask.any():
        if np.any(cs.Bio > 0):
            cs.Bio *= np.exp(-rp.bio_decay * dt / 86400.0)
        return 0.0
    ii = np.where(cell_mask)[0]
    th = th_all[ii]
    pc = _pc(rp, ii)
    NO3 = c_["NO3"][ii].copy(); SO4 = c_["SO4"][ii].copy()
    NO2 = c_["NO2"][ii].copy() if (rp.nitrite and "NO2" in c_) else None
    Ca = c_["Ca"][ii].copy(); ALK = c_["ALK"][ii].copy()
    O2 = c_["O2"][ii].copy() if (rp.o2_on and "O2" in c_) else None
    O2g = cs.O2g[ii].copy() if (rp.o2_on and getattr(cs, "O2g", None) is not None) else None
    D = c_["D"][ii].copy() if (rp.donor is not None and "D" in c_) else None
    S = cs.S[ii].copy(); Cal = cs.Cal[ii].copy(); Gyp = cs.Gyp[ii].copy(); Bio = cs.Bio[ii].copy()
    aCal = cs.aCal[ii]
    fT = rp.theta_T ** (rp.T - rp.T_ref)
    k1, k2, k3 = _pwp_k(rp.T)
    Ksp_g = 10 ** LOGK["Ksp_gyp"]
    area = rp.kin == "area" and cs.aS is not None
    if area:
        # v4: 체적 속도상수 = k_a × 황 비표면적 a_S (Koenig & Liu 2001), 물 부피당으로 나눔(/θ)
        cA = cs.aS[ii]
        phi_ = np.maximum(cs.porosity()[ii], 1e-6)
        S_wfps = np.clip(th / phi_, 0.0, 1.0)                  # S = θ/φ (WFPS) — 흐름 계산값
        sat_r = S_wfps if rp.s_set is None else np.full_like(S_wfps, rp.s_set)   # 반응 계산용 S(지정 모드면 지정값)
        fW, _ = MO.factors(rp.moist_mode, sat_r, None, rp.moist_cfg or {})       # F9: 모드별 수분 함수(O2 억제는 아래)
        age = cs.age[ii] if cs.age is not None else np.full(len(ii), 1e9)
        flag = 1.0 - np.exp(-age / rp.lag_d) if rp.lag_d > 0 else 1.0
        kA = rp.k_a * cA / np.maximum(th, 1e-9) * flag      # (mg/L)^½/h per φ_S^(2/3)
        fTox = np.exp(-rp.Ea_ox / 8.314 * (1.0 / (rp.T + 273.15) - 1.0 / 298.15))
        fm_ox = np.clip(1.0 - ((sat_r - rp.s_opt_ox) / max(rp.s_opt_ox, 1e-6)) ** 2, 0.0, 1.0)

        def _fS(S_):
            return np.power(np.maximum(S_ * VM["S"], 0.0), 2.0 / 3.0)
        kfac = kA * fW
    else:
        def _fS(S_):
            return np.where(S_ > 0, (S_ * VM["S"] / rp.S_ref) ** (2.0 / 3.0), 0.0)
        kfac = rp.k_half
    # 하위 단계 수: 알칼리도 소비가 단계당 ALK_STEP_MAX를 넘지 않게
    Cmg = NO3 * MW["N"]
    fS0 = _fS(S)
    kp0 = kfac / 3600.0 * fS0 * fT
    dn_est = np.minimum(Cmg, 0.5 * kp0 * dt * np.sqrt(np.maximum(Cmg, 0)) + 0.25 * (kp0 * dt) ** 2) / MW["N"]
    need = rp.stoich_H * dn_est / ALK_STEP_MAX
    if area and O2 is not None:
        # 호기성 황 산화의 산 생성(2 eq/mol S)도 하위 단계 수에 반영 — 물이 적은 셀은 물 부피당 산이 크다.
        # 이 속도는 pH와 무관하므로(석회석 반응만 pH에 의존, 넘침은 평형으로 잘라 냄) 단계당 허용 폭을 OX_ALK_STEP로 크게
        rS0 = rp.k_ox_a * cA * fS0 / np.maximum(th, 1e-9) * fTox * fm_ox * flag
        o2_avail = O2 + (O2g / np.maximum(th, 1e-9) if O2g is not None else 0.0)
        need = need + 2.0 * np.minimum(rS0 * dt, o2_avail / 1.5) / OX_ALK_STEP
    need = np.ceil(need)
    nsub = int(np.clip(np.max(need) if len(dn_est) else 1, 1, 400))
    if nsub > 8 and not _split:
        # 하위 단계가 많이 필요한 셀(물이 적고 산이 많이 생기는 셀)만 따로 잘게 — 나머지 셀까지 같이 잘게 돌리지 않는다(속도)
        hard = need > 8
        if hard.sum() <= 0.5 * len(ii):
            Nn = len(cs.S)
            m_h = np.zeros(Nn, bool); m_h[ii[hard]] = True
            m_e = np.zeros(Nn, bool); m_e[ii[~hard]] = True
            tot_ = react(cs, theta, V, dt, rp, cell_mask=m_e, vmag=vmag, _split=True) if m_e.any() else 0.0
            return tot_ + react(cs, theta, V, dt, rp, cell_mask=m_h, vmag=vmag, _split=True)
    h = dt / nsub
    denit_tot = 0.0
    x_prev = None
    air_all = np.maximum(cs.porosity()[ii] - th, 0.0) if O2 is not None else None
    for _ in range(nsub):
        x_prev, g1_, g2_, CO2_, cf_, sf_ = speciate_full(ALK, Ca, SO4, NO3, pc, rp.I_bg, x_prev)
        pH = -np.log10(x_prev)
        fS = _fS(S)
        # ── O2: 황 표면 산화 S + 1.5 O2 + H2O → SO4 + 2H+, 탈질 억제
        finh = 1.0
        if O2 is not None:
            if O2g is not None:          # 기상 O2와 평형(공기 공극 → 용존)
                tot = O2 * th + O2g
                cap = th + air_all / (KHO2 * 8.206e-5 * (rp.T + 273.15))   # 기상 O2 = 공기부피 × p/(RT)
                O2 = tot / np.maximum(cap, 1e-12)
                O2g = tot - O2 * th
            if area:
                # v4: 표면 속도(Watkinson & Blair 1993, 25°C 최적 수분) × 비표면적 × O2 Monod × 수분 × 온도(Ea) × 적응
                rS = rp.k_ox_a * cA * fS / np.maximum(th, 1e-9) * O2 / (rp.K_O2 + O2 + 1e-30) * fTox * fm_ox * flag
                availS = O2 + (O2g / np.maximum(th, 1e-9) if O2g is not None else 0.0)     # 물 부피당 쓸 수 있는 O2
                dO = np.minimum(1.5 * rS * h, availS)
                if O2g is not None:
                    tot = O2 * th + O2g - dO * th
                    O2 = np.maximum(tot, 0.0) / np.maximum(cap, 1e-12)
                    O2g = np.maximum(tot - O2 * th, 0.0)
                else:
                    O2 = O2 - dO
            else:
                rO2 = rp.k_O2 * fS * O2 / (rp.K_O2 + O2 + 1e-30) * fT
                dO = np.minimum(rO2 * h, O2)
                O2 -= dO
            ox = dO / 1.5
            ox = np.minimum(ox, S / np.maximum(th, 1e-9))
            S -= ox * th; SO4 += ox; ALK -= 2 * ox
            L["S_ox_O2"] = L.get("S_ox_O2", 0.0) + float((ox * th).sum() * V)
            L["ALK_O2"] = L.get("ALK_O2", 0.0) + float((2 * ox * th).sum() * V)
            finh = rp.K_I_O2 / (rp.K_I_O2 + O2)
            if area and rp.moist_mode != "A":
                # F9-2: 모드 B(NEMIS는 O2 제한을 수분으로 대신 나타냄)·C(상한)는 탈질에 f_O2를 곱하지 않는다.
                # 호기성 황 산화의 O2 소모(위)는 모든 모드에서 그대로.
                finh = 1.0
        # ── SLAD (반-차수, 구간 해석해)
        kp = kfac / 3600.0 * fS * _fpH(pH, rp) * fT * finh
        C = NO3 * MW["N"]
        sq = np.maximum(np.sqrt(np.maximum(C, 0)) - 0.5 * kp * h, 0.0)
        dC = C - sq ** 2
        dn = dC / MW["N"]
        if rp.nitrite:
            # D4: 1단계 NO3→NO2(전자 2개) — 전체 양론의 2/5, 2단계 NO2→N2(전자 3개) — 3/5 (전자 비 가정)
            f1 = 0.4
            dn = np.minimum(dn, S / (f1 * rp.stoich_S * th)); dn = np.maximum(dn, 0.0)
            NO3 -= dn; NO2 += dn
            SO4 += f1 * rp.stoich_SO4 * dn; ALK -= f1 * rp.stoich_H * dn; S -= f1 * rp.stoich_S * dn * th
            C2 = NO2 * MW["N"]
            sq2 = np.maximum(np.sqrt(np.maximum(C2, 0)) - 0.5 * rp.k_NO2_ratio * kp * h, 0.0)
            dn2 = (C2 - sq2 ** 2) / MW["N"]
            dn2 = np.maximum(np.minimum(dn2, S / ((1 - f1) * rp.stoich_S * th)), 0.0)
            NO2 -= dn2
            SO4 += (1 - f1) * rp.stoich_SO4 * dn2; ALK -= (1 - f1) * rp.stoich_H * dn2
            S -= (1 - f1) * rp.stoich_S * dn2 * th
            S = np.maximum(S, 0.0)
            Bio += rp.bio_vol * dn2 * th
            dmol = float((dn2 * th).sum() * V)
            L["N_denit"] += dmol
            L["ALK_denit"] += rp.stoich_H * (f1 * float((dn * th).sum() * V) + (1 - f1) * dmol)
            denit_tot += dmol
        else:
            dn = np.minimum(dn, S / (rp.stoich_S * th))
            dn = np.maximum(dn, 0.0)
            NO3 -= dn
            SO4 += rp.stoich_SO4 * dn
            ALK -= rp.stoich_H * dn
            S -= rp.stoich_S * dn * th
            S = np.maximum(S, 0.0)
            Bio += rp.bio_vol * dn * th
            dmol = float((dn * th).sum() * V)
            denit_tot += dmol
            L["N_denit"] += dmol
            L["ALK_denit"] += rp.stoich_H * dmol
        # ── 액체 공여체(시약 형태 d): Monod 이중 제한, 양론은 사용자 입력
        if D is not None:
            dp = rp.donor
            r = dp["k"] * NO3 / (dp["KN"] + NO3 + 1e-30) * D / (dp["KD"] + D + 1e-30) * fT * finh
            dnd = np.minimum(r * h, np.minimum(NO3, D / max(dp["yD"], 1e-12)))
            dnd = np.maximum(dnd, 0.0)
            NO3 -= dnd; D -= dp["yD"] * dnd; SO4 += dp["ySO4"] * dnd; ALK -= dp["yH"] * dnd
            Bio += rp.bio_vol * dnd * th
            dmd = float((dnd * th).sum() * V)
            denit_tot += dmd
            L["N_denit"] += dmd
            L["ALK_denit"] += dp["yH"] * dmd
            L["D_used"] = L.get("D_used", 0.0) + dp["yD"] * dmd
            L["S_from_D"] = L.get("S_from_D", 0.0) + dp["ySO4"] * dmd
        # ── 석회석
        cal_on = (Cal > 0)
        SR, x, CO2 = calcite_SR(ALK, Ca, SO4, NO3, pc, rp.I_bg, x_prev)
        if cal_on.any() or np.any(SR > 1.0):
            hi = np.where(cal_on, Cal / th, 0.0)
            lo = -np.minimum(Ca, np.maximum(ALK, 0) / 2.0)
            lo = np.where(cal_on, lo, 0.0)          # 석회석이 있는 셀에서만 침전(핵생성 가정)
            need = cal_on & (np.abs(SR - 1.0) > 1e-6)
            delta = np.zeros_like(Cal)
            if need.any():
                j = np.where(need)[0]
                pcj = pc[j] if np.ndim(pc) else pc
                if rp.calcite_mode == "평형":
                    delta[j] = _calcite_eq_delta(ALK[j], Ca[j], SO4[j], NO3[j], lo[j], hi[j], rp, x[j], pcj)
                else:
                    Vc = Cal[j] * VM["CaCO3"]
                    As = aCal[j] * Vc
                    CO2j = CO2[j] if np.ndim(CO2) else CO2
                    rate = (k1 * x[j] + k2 * CO2j + k3) * (1.0 - SR[j] ** (2.0 / 3.0))
                    dk = rate * As * h / th[j]
                    dk = np.minimum(np.maximum(dk, lo[j]), hi[j])
                    SRk = calcite_SR(ALK[j] + 2 * dk, Ca[j] + dk, SO4[j], NO3[j], pcj, rp.I_bg, x[j])[0]
                    over = ((SR[j] < 1) & (SRk > 1)) | ((SR[j] > 1) & (SRk < 1))
                    dd = dk.copy()
                    if over.any():
                        jj = np.where(over)[0]
                        dd[jj] = _calcite_eq_delta(ALK[j][jj], Ca[j][jj], SO4[j][jj], NO3[j][jj],
                                                   np.minimum(0.0, dk[jj]), np.maximum(0.0, dk[jj]), rp, x[j][jj],
                                                   pcj[jj] if np.ndim(pcj) else pcj)
                    delta[j] = dd
            Ca += delta
            ALK += 2 * delta
            Cal -= delta * th
            Cal = np.maximum(Cal, 0.0)
            dmol = float((delta * th).sum() * V)
            L["Cal_diss"] += dmol
            L["ALK_cal"] += 2 * dmol
        # ── 석고(평형, 이온쌍으로 자유 이온 분율 보정)
        _, _, g2, _, cf, sf = speciate_full(ALK, Ca, SO4, NO3, pc, rp.I_bg, x)
        fCa = np.where(Ca > 0, cf / np.maximum(Ca, 1e-30), 1.0)
        fSO = np.where(SO4 > 0, sf / np.maximum(SO4, 1e-30), 1.0)
        Kc = Ksp_g / (g2 ** 2 * np.maximum(fCa * fSO, 1e-6)) * 1e6
        prod = Ca * SO4
        chk = (prod > Kc) | (Gyp > 0)
        if chk.any():
            s_ = Ca + SO4
            disc = np.maximum(s_ ** 2 - 4 * (prod - Kc), 0.0)
            y = 0.5 * (s_ - np.sqrt(disc))
            y = np.where(chk, y, 0.0)
            y = np.maximum(y, -Gyp / th)
            Ca -= y; SO4 -= y
            Gyp += y * th
            dmol = float((y * th).sum() * V)
            L["Gyp_prec"] += dmol
    Bio *= np.exp(-rp.bio_decay * dt / 86400.0)
    if rp.bio_det > 0 and vmag is not None:
        Bio *= np.exp(-rp.bio_det * vmag[ii] * dt)        # 유속 탈착(B8, 가정): 공극수가 1 m 흐를 때 비율 bio_det
    c_["NO3"][ii] = NO3; c_["SO4"][ii] = SO4; c_["Ca"][ii] = Ca; c_["ALK"][ii] = ALK
    if NO2 is not None:
        c_["NO2"][ii] = NO2
    if O2 is not None:
        c_["O2"][ii] = O2
        if O2g is not None:
            cs.O2g[ii] = O2g
    if D is not None:
        c_["D"][ii] = D
    cs.S[ii] = S; cs.Cal[ii] = Cal; cs.Gyp[ii] = Gyp; cs.Bio[ii] = Bio
    return denit_tot


PHI_MIN_DEP = 0.05     # 이 공극률 아래로는 더 쌓이지 않음(차단 함수, 가정)


def deposit_particles(cs: ChemState, theta, vmag, dt, rp: ReactParams, V: float, all_now=False,
                      inert_frac=0.0, d_p=2e-4, species="P", lam=None, frac_S=None, d_area=None):
    """주입 입자 포획: dP/dt = −λ|v|B(φ)P (구간 해석해). B = 남은 공극 비율(차단 함수, 가정).

    all_now=True (정치 종료 등): '남은 공극이 허용하는 만큼' 즉시 포획한다. 공극이 다 찬 셀의 나머지 입자는
    부유 상태로 남아 다음 흐름에 실려 이동·유출된다(전량 강제 포획으로 공극률이 하한 아래로 내려가던 v2 동작 수정, A16).
    단위: P = 물 1 m3당 입자 고체 부피(m3/m3 물). 장부 P_inj/P_dep/P_out 모두 '고체 m3'.
    frac_S: 포획 고체 중 황 부피분율(슬러리 = r/(r+1), 콜로이드 황 = 1)."""
    if species not in cs.c:
        return 0.0
    P = cs.c[species]
    if not np.any(P > 0):
        return 0.0
    lam = rp.lambda_filt if lam is None else lam
    phi = cs.porosity()
    pmin = PHI_MIN_DEP if cs.phi_min is None else cs.phi_min
    room = np.maximum(phi - pmin, 0.0)                          # 더 쌓을 수 있는 부피 m3/m3 벌크
    B = np.clip(room / np.maximum(cs.phi_matrix - pmin, 1e-6), 0.0, 1.0)
    if all_now:
        frac = np.ones_like(P)
    else:
        frac = 1.0 - np.exp(-lam * vmag * B * dt)
    th = np.maximum(theta, 1e-12)
    dep_w = np.minimum(P * frac, room / th)                    # 공극 한도(정치 포함 항상 적용)
    dep_w = np.where(theta > 1e-9, dep_w, 0.0)
    dep_b = dep_w * theta                 # m3고체 / m3벌크
    cs.c[species] = P - dep_w
    r = rp.ratio
    fS = (r / (r + 1.0)) if frac_S is None else float(frac_S)
    vs = dep_b * (1 - inert_frac)
    dS = vs * fS / VM["S"]
    dC = vs * (1.0 - fS) / VM["CaCO3"]
    newC = cs.Cal + dC
    a_new = 6.0 / d_p
    cs.aCal = np.where(newC > 0, (cs.aCal * cs.Cal + a_new * dC) / np.maximum(newC, 1e-30), cs.aCal)
    add_sulfur(cs, dS, d_area or d_p)
    cs.Cal += dC
    cs.Inert += dep_b * inert_frac
    cs.ledger["S_inj"] += float(dS.sum() * V)
    tot = float(dep_b.sum() * V)
    cs.ledger[f"{species}_dep"] = cs.ledger.get(f"{species}_dep", 0.0) + tot
    return tot


def kozeny_carman_ratio(phi, phi_ref):
    phi = np.clip(phi, 1e-3, 0.999)
    phi_ref = np.clip(phi_ref, 1e-3, 0.999)
    return (phi ** 3 / (1 - phi) ** 2) / (phi_ref ** 3 / (1 - phi_ref) ** 2)
