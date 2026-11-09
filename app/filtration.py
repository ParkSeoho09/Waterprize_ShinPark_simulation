"""입자 포획 계수 λ — 콜로이드 여과 이론(F4).

λ = 3(1−φ)/(2 d_c) · α · η0                          [1/m]   (Yao et al. 1971)
η0: Tufenkji & Elimelech (2004) Environ. Sci. Technol. 38:529 상관식(확산·차단·중력 항, 반데르발스 포함) — 원문 확인 필요.
α(부착 효율)는 근거 있는 기본값이 없어 사용자 입력 필수(R3). Hamaker 상수 기본 1e-20 J(전형값, 가정).
Tosco & Sethi (2010) 이중 부위 모델 항목 중 이 앱에 있는 것/없는 것:
  있음 — 침착(λ, 1차), 체거름(straining: 입자/토양 지름비 > 0.154이면 개구부 필터 케이크), 공극 막힘(Kozeny–Carman 되먹임)
  없음 — 탈착(가역 부착), blocking(포화 부착량에 따른 감소, Langmuir형), ripening(부착 입자 위 추가 부착)
"""
from __future__ import annotations

import numpy as np

KB = 1.380649e-23        # J/K
G = 9.81


def te2004_eta0(d_p, d_c, U, phi, T_C=20.0, rho_p=2070.0, rho_f=998.0, mu=1.0e-3, A_H=1.0e-20):
    """단일 집진체 접촉 효율 η0 (Tufenkji & Elimelech 2004). 단위 SI. U = 다르시 유속 m/s."""
    T = T_C + 273.15
    gam = (1 - phi) ** (1 / 3)
    As = 2 * (1 - gam ** 5) / (2 - 3 * gam + 3 * gam ** 5 - 2 * gam ** 6)
    a_p = d_p / 2
    Dinf = KB * T / (3 * np.pi * mu * d_p)
    NR = d_p / d_c
    NPe = U * d_c / Dinf
    NvdW = A_H / (KB * T)
    NA = A_H / (12 * np.pi * mu * a_p ** 2 * U)
    NG = 2.0 / 9.0 * a_p ** 2 * (rho_p - rho_f) * G / (mu * U)
    etaD = 2.4 * As ** (1 / 3) * NR ** -0.081 * NPe ** -0.715 * NvdW ** 0.052
    etaI = 0.55 * As * NR ** 1.675 * NA ** 0.125
    etaG = 0.22 * NR ** -0.24 * NG ** 1.11 * NvdW ** 0.053
    eta = float(etaD + etaI + etaG)
    note = "" if eta <= 1 else "η0 > 1 — 상관식 적용 범위 밖(큰 입자: 체거름 영역), 1로 제한"
    return min(eta, 1.0), {"diffusion": etaD, "interception": etaI, "gravity": etaG, "As": As, "note": note}


def lambda_filtration(d_p, d_c, U, phi, alpha, **kw):
    if alpha is None:
        raise ValueError("여과 이론 λ를 쓰려면 부착 효율 α를 입력하세요(근거 있는 기본값 없음).")
    eta0, parts = te2004_eta0(d_p, d_c, U, phi, **kw)
    lam = 3 * (1 - phi) / (2 * d_c) * float(alpha) * eta0
    return float(lam), dict(parts, eta0=eta0)
