"""투수성 판(상판·하판)의 구멍 흐름(Part 3).

구멍 하나의 저항(얇은 판의 원형 구멍, 양쪽 반무한 다공질):
    R_h = 1/(4·K_s·a) + 1/(4·K_r·a) + t_p/(K_h·π·a²)        [s/m²]
  앞 두 항 = 원판 접촉의 라플라스 해(반무한 매질에 반지름 a 원판 등전위면의 퍼짐 저항 1/(4Ka)).
  Hall (1975) 유추 — 원문 확인 필요. 셋째 항 = 구멍 안 채움재(길이 t_p) 다공 저항.
판 전체: C_face = n/R_h (병렬) [m²/s]. 판 단위 면적당 직렬 저항 r = A_p/C_face [s] — 면 장벽(t/K와 같은 단위)으로 넣는다.
불포화: r/Kr(상류 셀). 빈 공간 구멍은 오리피스 Q = C_d·A·√(2 g Δh)을 기준 수두차 Δh_ref에서 선형화해 직렬로 더한다(가정).
구멍 간격이 지름의 5배보다 작으면 구멍끼리 간섭해 저항이 이 식보다 커진다 — 배열 보정식은 문헌 미확보(원문 확인 필요),
면 전도도는 판이 없을 때 이하로만 제한된다(직렬 저항 ≥ 0).
"""
from __future__ import annotations

import numpy as np

G = 9.81
FILLS = ["반응재/흙", "메시", "빈 공간"]


def hole_resistance(a, t_p, K_s, K_r, K_h=None, fill="반응재/흙", Cd=0.6, dh_ref=0.01):
    """구멍 하나 저항 R_h [s/m²]. a = 구멍 반지름 m, t_p = 판 두께 m, K = m/s."""
    if a <= 0:
        raise ValueError("구멍 지름은 0보다 커야 합니다.")
    R = 1.0 / (4.0 * K_s * a) + 1.0 / (4.0 * K_r * a)
    if fill == "빈 공간":
        A = np.pi * a * a
        R += np.sqrt(max(dh_ref, 1e-9)) / (Cd * A * np.sqrt(2 * G))       # Δh/Q at Δh_ref (선형화, 가정)
    else:
        if K_h is None or not np.isfinite(K_h) or K_h <= 0:
            raise ValueError("구멍 채움재 투수계수 K_h를 입력하세요(메시·채움재).")
        R += t_p / (K_h * np.pi * a * a)
    return float(R)


def plate_resistance_per_area(A_p, n_holes, a, t_p, K_s, K_r, K_h=None, fill="반응재/흙", Cd=0.6, dh_ref=0.01):
    """판 단위 면적당 직렬 저항 r = A_p·R_h/n [s] 와 진단."""
    if n_holes <= 0:
        return float("inf"), {"note": "구멍 0개 — 불투수 판"}
    Rh = hole_resistance(a, t_p, K_s, K_r, K_h, fill, Cd, dh_ref)
    C = n_holes / Rh
    pitch = np.sqrt(A_p / n_holes)
    info = {"R_h": Rh, "C_face": C, "pitch_m": pitch, "open_ratio": n_holes * np.pi * a * a / A_p,
            "interference": pitch < 5 * 2 * a}
    return float(A_p / C), info


# ───────────────────────────── S8: 구멍 하나를 축대칭 세밀 격자로 직접 풀기 ─────────────────────────────
def axisym_hole_conductance(a, t_p, K_s, K_r, K_h, Rd_over_a=60.0, Hz_over_a=60.0, n_per_a=8, stretch=1.06):
    """원통 영역(반지름 Rd, 판 위·아래 높이 Hz)의 정상 포화 흐름을 축대칭 유한체적으로 풀어
    구멍의 저항 R_num = (판 있는 전체 저항) − (판 없는 전체 저항, 판 두께만큼 흙·반응재로 채움) + t_p/(K·πRd²)을 낸다.
    반지름 방향은 구멍 근처 a/n_per_a, 바깥으로 등비 증가. 위 경계 H=1, 아래 경계 H=0."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    h0 = a / n_per_a
    r_edges = [0.0]
    while r_edges[-1] < a - 1e-15:
        r_edges.append(min(r_edges[-1] + h0, a))
    dr = h0
    while r_edges[-1] < Rd_over_a * a:
        dr *= stretch
        r_edges.append(r_edges[-1] + dr)
    r_edges = np.array(r_edges)
    nt = max(1, int(round(t_p / h0)))
    dzp = t_p / nt

    def zgrid(H):
        e = [0.0]; d = h0
        while e[-1] < H:
            e.append(e[-1] + d); d *= stretch
        return np.array(e)
    ze_up = zgrid(Hz_over_a * a)
    ze_dn = zgrid(Hz_over_a * a)
    # z: 아래 영역(반응재 K_r) | 판(두께 t_p) | 위 영역(흙 K_s), z는 아래에서 위로
    z_edges = np.concatenate([-(ze_dn[::-1]) - 0.0, np.arange(1, nt + 1) * dzp, t_p + ze_up[1:]])
    z_edges = np.unique(z_edges)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1]); dz = np.diff(z_edges)
    rc = 0.5 * (r_edges[1:] + r_edges[:-1]); drs = np.diff(r_edges)
    nr, nz = len(rc), len(zc)

    def solve(with_plate):
        K = np.where(zc[None, :] < 0, K_r, K_s) * np.ones((nr, 1))
        in_plate = (zc > 0) & (zc < t_p)
        if with_plate:
            K[:, in_plate] = np.where(rc[:, None] < a, K_h, 0.0)
        else:
            K[:, in_plate] = K_r                     # 판 두께 구간을 반응재로 채운 기준
        N = nr * nz
        idx = np.arange(N).reshape(nr, nz)
        rows, cols, vals = [], [], []
        diag = np.zeros(N); rhs = np.zeros(N)
        # 반지름 방향 면
        for i in range(nr - 1):
            re = r_edges[i + 1]
            A = 2 * np.pi * re * dz
            k1, k2 = K[i], K[i + 1]
            d1, d2 = 0.5 * drs[i], 0.5 * drs[i + 1]
            with np.errstate(divide="ignore"):
                T = np.where((k1 > 0) & (k2 > 0), A / (d1 / np.maximum(k1, 1e-300) + d2 / np.maximum(k2, 1e-300)), 0.0)
            a_, b_ = idx[i], idx[i + 1]
            rows += [a_, b_]; cols += [b_, a_]; vals += [-T, -T]
            diag[a_] += T; diag[b_] += T
        # 연직 면
        Az = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        for j in range(nz - 1):
            k1, k2 = K[:, j], K[:, j + 1]
            d1, d2 = 0.5 * dz[j], 0.5 * dz[j + 1]
            T = np.where((k1 > 0) & (k2 > 0), Az / (d1 / np.maximum(k1, 1e-300) + d2 / np.maximum(k2, 1e-300)), 0.0)
            a_, b_ = idx[:, j], idx[:, j + 1]
            rows += [a_, b_]; cols += [b_, a_]; vals += [-T, -T]
            diag[a_] += T; diag[b_] += T
        # 위(H=1)·아래(H=0) 디리클레
        Tt = Az * K[:, -1] / (0.5 * dz[-1]); Tb = Az * K[:, 0] / (0.5 * dz[0])
        diag[idx[:, -1]] += Tt; rhs[idx[:, -1]] += Tt * 1.0
        diag[idx[:, 0]] += Tb
        dead = diag <= 0
        diag[dead] = 1.0
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)) + sp.diags(diag)
        H = spl.spsolve(M.tocsc(), rhs)
        Q = float(np.sum(Tb * H[idx[:, 0]]))                # 아래로 나가는 유량 (ΔH = 1)
        return 1.0 / Q                                       # 전체 저항 [s/m²]
    R_with = solve(True)
    R_without = solve(False)
    A_dom = np.pi * r_edges[-1] ** 2
    R_num = R_with - R_without + t_p / (K_r * A_dom)        # 기준에서 뺀 판 구간(반응재) 저항을 되돌림
    return float(R_num), {"nr": nr, "nz": nz, "Rd": float(r_edges[-1]), "Hz": float(ze_up[-1])}
