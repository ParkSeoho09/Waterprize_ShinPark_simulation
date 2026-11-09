"""공통 상수·기본값 레지스트리.

모든 물리 파라미터에 출처 또는 '가정' 라벨을 붙인다(정직성 규칙).
라벨: "문헌" = 출처 있음 / "가정" = 임의 설정 / "보정" = 데이터에 맞춰 조정한 값.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ───────────────────────────── 물리 상수(25°C) ─────────────────────────────
G = 9.81
RHO_W = 1000.0
MW = {"N": 14.007, "S": 32.06, "SO4": 96.06, "Ca": 40.078, "CaCO3": 100.09, "gyp": 172.17}
RHO_SOLID = {"S": 2070.0, "CaCO3": 2710.0, "gyp": 2320.0}          # kg/m3 (화학 편람값)
# 단위 환산(R2: 환산은 이 파일에서만)
GAL_US_M3 = 3.785411784e-3          # 1 US gal = 3.785 L
FT_M = 0.3048
GPM_PER_FT_TO_M2_S = GAL_US_M3 / 60.0 / FT_M      # gpm/ft → m³/s per m
EPA_SCREEN_CAP_GPM_FT = 2.0          # EPA 2013 p.44 사례: 스크린 단위 길이당 전송 용량 2 gpm/ft (문헌)
PSI_KPA = 6.894757
FRACTURE_PSI = (100.0, 500.0)        # EPA 2013 pp.24–25: 100~500 psi에서 흙 파쇄 가능 (문헌)
MM_DAY_TO_M_H = 1e-3 / 24.0
S_PER_N_G = 1.10 * 32.06 / 14.007    # 탈질 화학량론 1.10 mol SO4/mol N → 2.52 g S / g N
VM = {k: MW[k if k != "gyp" else "gyp"] * 1e-3 / RHO_SOLID[k] for k in ("S", "CaCO3", "gyp")}  # m3/mol

LOGK = {   # PHREEQC phreeqc.dat (Parkhurst & Appelo 2013), 25°C
    "K1": -6.351, "K2": -10.329, "KH": -1.468, "Kw": -14.0,
    "Ksp_cal": -8.48, "Ksp_gyp": -4.58,
    # 이온쌍(phreeqc.dat 3.7.x): Ca+2 + SO4-2 = CaSO4 ; Ca+2 + CO3-2 = CaCO3 ; Ca+2 + HCO3- = CaHCO3+ (11.435−10.329)
    "CaSO4": 2.25, "CaCO3": 3.224, "CaHCO3": 1.106,
}

# ───────────────────────────── 파라미터 레지스트리 ─────────────────────────
# key: (값, 단위, 라벨, 설명/출처)
REGISTRY: Dict[str, tuple] = {
    "stoich_SO4": (1.10, "mol SO4/mol NO3", "문헌",
                   "Batchelor & Lawrence (1978) 황 독립영양 탈질 양론. NO3-N 1 mg당 SO4 7.54 mg(지표 문서 이론값)과 일치"),
    "stoich_H": (1.28, "eq 산/mol NO3", "문헌",
                 "Batchelor & Lawrence (1978). NO3-N 1 mg당 알칼리도 4.57 mg as CaCO3 소비와 일치"),
    "stoich_S": (1.10, "mol S/mol NO3", "문헌", "Batchelor & Lawrence (1978)"),
    "k_half": (5.0, "mg-N^½/L^½/h", "문헌",
               "반-차수 속도상수. 선행 SLAD 칼럼(알칼리도 850 mg/L) 보고 범위 4.36~6.91의 중간값 — 참고 시뮬 A 문서 인용, 원 논문 대조 필요"),
    # ── v4 표면적 기반 SLAD 속도식(문헌) ──
    "biofilm_rho": (30.0, "kg/m3", "가정", "생물막 밀도 X_f (D5 스윕 10~100). 반-차수 상수 ∝ √X_f (Harremoës 1978 — 원문 확인 필요)"),
    "biofilm_rho_ref": (30.0, "kg/m3", "가정", "k½a 기준값이 대응한다고 보는 생물막 밀도(가정)"),
    "k_half_a": (4.0e-3, "(mg/L)^½·m/h", "문헌",
                 "표면 반-차수 상수: 체적 상수 k½v = k½a × 황 비표면적 a(m²/m³). Koenig & Liu (2001, Water Res. 35:1969) "
                 "황 충전층 입경별 k½v(2.8–5.6 mm: 2.94–3.60, 5.6–11.2 mm: 1.47–2.04, 11.2–16 mm: 1.12–1.29 mg^½L^-½h^-1)을 "
                 "a = 6(1−ε)/d(ε=0.4 가정, 기하평균 입경)로 나눈 값의 평균(3.6~4.5e-3). 낮은 쪽 근거: Sierra-Alvarez et al. "
                 "(2007, Water Res. 41:1253) 표면 기준 속도 26.4 mmol N/m²/d → 1.5e-3. 불확실 범위 1.5e-3~4.7e-3"),
    "S_d_media": (4.0e-3, "m", "문헌",
                  "사전 배치 충전재의 황 입경. 상용 입상 황 2.8–5.6 mm 등급(Koenig & Liu 2001의 가장 작은 등급, 기하평균 4.0 mm)"),
    "wfps_thr": (0.62, "-", "문헌",
                 "탈질 수분 함수 f = ((s − s_t)/(1 − s_t))^w 의 문턱 포화도 s_t. NEMIS (Hénault & Germon 2000) 0.62. "
                 "포화도 60% 아래에서는 탈질이 거의 없음(Grundmann & Rolston 1987). 공기 중 O2가 있는 불포화 흙의 무산소 미소 부위를 나타냄"),
    "wfps_exp": (1.74, "-", "문헌", "탈질 수분 함수 지수 w. NEMIS (Hénault & Germon 2000). Heinen (2006, Geoderma 133:444) 정리"),
    "k_ox_a": (1.8e-7, "mol S/m²/s", "문헌",
               "호기성 황 산화(티오바실루스) 표면 속도. Watkinson & Blair (1993, Fert. Res. 35:115): 25°C·최적 수분에서 약 "
               "50 µg S/cm²/일. Watkinson (1989, Aust. J. Soil Res. 27:365): 48–76 µg/cm²/일"),
    "Ea_ox": (85e3, "J/mol", "문헌", "호기성 황 산화 활성화 에너지(Watkinson & Blair 1993)"),
    "s_opt_ox": (0.6, "-", "가정",
                 "호기성 황 산화가 가장 빠른 포화도. Watkinson & Blair (1993): 속도는 수분에 대해 2차식, 포장용수량 근처 최대 — "
                 "정확한 계수는 없어 포화도 0.6을 꼭짓점으로 둠"),
    "lag_days": (3.0, "일", "가정",
                 "황이 놓인 뒤 미생물이 붙어 반응이 본격화되는 적응 시정수. Watkinson & Blair (1993): 티오바실루스 '수 일 지연'"),
    "D_O2_air": (2.0e-5, "m²/s", "문헌", "공기 중 O2 분자 확산계수(20°C). 토양 유효 확산 = D0·ε_a^(10/3)/φ² (Millington & Quirk 1961)"),
    "S_ref_frac": (0.45, "m3 S/m3 충전층", "가정",
                   "속도상수를 측정한 충전층의 황 부피분율(공극 0.4, 황:석회석 3:1 가정). 반응 속도를 황 함량 비로 환산할 때 기준"),
    "pH_low": (6.2, "-", "가정", "이 pH 아래에서 탈질 속도가 절반 이하로 떨어짐(Sengupta & Ergas 계열 경향 참고)"),
    "pH_high": (9.0, "-", "가정", "이 pH 위에서 탈질 속도 감소"),
    "theta_T": (1.07, "-", "가정", "온도 보정 계수(아레니우스형), 기준 22°C"),
    "T_ref": (22.0, "°C", "가정", "속도상수 기준 온도"),
    "cal_d_p": (2.0e-3, "m", "가정", "충전 석회석 입경(비표면적 6/d 계산)"),
    "inj_d_p": (2.0e-4, "m", "가정", "주입 슬러리 입자 입경"),
    "pCO2": (-2.0, "log10 atm", "가정", "토양 공기 CO2 분압(일반 범위 10^-3 ~ 10^-1.5). 깊이별 모드에서는 지표 값"),
    "pCO2_bottom": (-1.7, "log10 atm", "가정", "깊이별 pCO2 모드: 흙수조 바닥 값(지표와 선형 보간, B4)"),
    "pCO2_reac": (-1.5, "log10 atm", "가정", "깊이별 pCO2 모드: 반응재 셀 값(미생물 호흡·탈질로 높음 가정)"),
    "I_bg": (0.005, "mol/L", "가정", "배경 이온강도(Davies 활동도 계산용)"),
    "bio_density": (25.0, "kg/m3", "가정",
                    "생물막 건조밀도. 탈질 1 mol당 생물막 부피 = 0.08 mol 세포 × 113 g/mol ÷ 밀도. 불확실 범위 10~100 kg/m3"),
    "bio_decay": (0.05, "1/day", "가정", "생물막 감쇠율"),
    "bio_det": (0.02, "1/m", "가정",
                "생물막 유속 탈착 계수: 공극수가 1 m 흐를 때 떨어져 나가는 비율(B8). 기질 확산 제한은 미구현"),
    "lambda_filt": (1000.0, "1/m", "가정",
                    "주입 입자 포획(여과) 계수. 콜로이드 여과 이론(Yao et al. 1971) λ=3(1−φ)ηα/(2d_c) 규모 추정. "
                    "v2 기본 30 1/m은 낙관적(거의 안 걸러짐) — B7. 불확실 범위 300~3000"),
    "strain_ratio": (0.154, "-", "문헌",
                     "입자/토양입자 지름비가 이 값을 넘으면 입자가 토양 공극에 못 들어가 개구부에 필터 케이크로 쌓임. "
                     "Herzig et al. (1970), McDowell-Boyer et al. (1986) 정리값 — 원 논문 대조 필요"),
    "K_open": (1.0e-2, "m/s", "가정",
               "주입 개구부(스크린·관) 셀의 투수계수. v2는 토양 K를 그대로 써서 주입이 실제보다 어렵거나 쉽게 나올 수 있었음(A15)"),
    "colloid_d_p": (1.0e-6, "m", "가정", "콜로이드·나노 황 입경(시약 형태 e)"),
    "lambda_colloid": (50.0, "1/m", "가정", "콜로이드 포획 계수(시약 형태 e) — 입자 슬러리보다 작게 가정"),
    "k_O2": (1.0e-6, "mol/m3/s", "가정", "O2에 의한 황 표면 산화 속도(황 부피분율 기준값일 때). 초기 O2 단계(B5)"),
    "K_O2": (0.01, "mol/m3", "가정", "황 산화 O2 반포화 농도(≈0.3 mg/L)"),
    "K_I_O2": (0.2 / 32.0, "mol/m3", "가정",
               "모드 A 탈질의 O₂ 억제 상수 K_O2(0.2 mg O₂/L, 스윕 0.1~0.5). 문헌에서 찾은 AOA 0.038~0.131·혼합 AOB 0.25~1.0 mg/L는 "
               "암모니아 산화균 값이라 쓰지 않음"),
    "fw_p": (2.0 / 3.0, "-", "가정",
             "모드 A 물막 접촉 분율 f_w = S^p의 지수(유도: 펜듈러 링 접촉 면적 ∝ V^(2/3)). 범위 1/3~1, 흙수조로 보정"),
    "nemis_thr": (0.70, "-", "원문 확인 필요",
                  "모드 B NEMIS 계열 임계 WFPS. HESS 2004(하천변 토양): 탈질이 WFPS 60~80 %에서 급증, 임계 약 70 % 권장(초록 확인, "
                  "본문 원문 확인 필요). 스윕 0.60~0.80. NEMIS 원식(Hénault & Germon 2000)은 0.62"),
    "nemis_w": (1.74, "-", "문헌", "모드 B 멱함수형 지수(NEMIS, Hénault & Germon 2000)"),
    "DO_in": (8.0, "mg/L", "가정", "유입수 용존산소(공기 포화 근사)"),
    "alpha_L": (0.01, "m", "가정", "종분산 길이(흙수조 규모)"),
    "alpha_T_ratio": (0.1, "-", "가정", "횡분산 길이 / 종분산 길이"),
    "Dm": (1.7e-9, "m2/s", "문헌", "질산염 자유수 확산계수(25°C, Li & Gregory 1974)"),
    "pH_ok_window": (0.5, "-", "문헌", "지표 문서: |ΔpH|<0.5 = 적정 완충"),
    "SO4_increase_thr": (0.1, "mol/m3", "가정", "V_eff 계산 시 'SO4 증가 확인' 판정 문턱(0.1 mM ≈ 9.6 mg/L)"),
    "reactive_thr_frac": (0.01, "-", "가정", "확산반경 r 판정: 황 부피분율이 충전층 대비 1% 이상인 셀"),
}


def reg(key):
    return REGISTRY[key][0]


# 토성별 대표 입경 d50 [m] — 입자 막힘(straining) 판정용. 모두 '가정'(USDA 토성 삼각도 중앙 조성에서 어림).
D50_TEXTURE = {
    "모래(Sand)": 3.0e-4, "양질사토(Loamy sand)": 2.0e-4, "사양토(Sandy loam)": 1.0e-4, "양토(Loam)": 4.0e-5,
    "미사질양토(Silt loam)": 2.5e-5, "미사(Silt)": 1.5e-5, "사질식양토(Sandy clay loam)": 6.0e-5,
    "식양토(Clay loam)": 2.0e-5, "미사질식양토(Silty clay loam)": 1.0e-5, "사질식토(Sandy clay)": 3.0e-5,
    "미사질식토(Silty clay)": 5.0e-6, "식토(Clay)": 3.0e-6,
}


# ───────────────────────────── 토성(Carsel & Parrish 1988) ─────────────────
# θr, θs, α[1/cm], n, Ks[cm/day]
_CP = {
    "모래(Sand)": (0.045, 0.43, 0.145, 2.68, 712.8),
    "양질사토(Loamy sand)": (0.057, 0.41, 0.124, 2.28, 350.2),
    "사양토(Sandy loam)": (0.065, 0.41, 0.075, 1.89, 106.1),
    "양토(Loam)": (0.078, 0.43, 0.036, 1.56, 24.96),
    "미사질양토(Silt loam)": (0.067, 0.45, 0.020, 1.41, 10.80),
    "미사(Silt)": (0.034, 0.46, 0.016, 1.37, 6.00),
    "사질식양토(Sandy clay loam)": (0.100, 0.39, 0.059, 1.48, 31.44),
    "식양토(Clay loam)": (0.095, 0.41, 0.019, 1.31, 6.24),
    "미사질식양토(Silty clay loam)": (0.089, 0.43, 0.010, 1.23, 1.68),
    "사질식토(Sandy clay)": (0.100, 0.38, 0.027, 1.23, 2.88),
    "미사질식토(Silty clay)": (0.070, 0.36, 0.005, 1.09, 0.48),
    "식토(Clay)": (0.068, 0.38, 0.008, 1.09, 4.80),
}
SOIL_TEXTURES = {k: {"theta_r": v[0], "theta_s": v[1], "alpha": v[2] * 100.0, "n": v[3],
                     "Ks": v[4] / 100.0 / 86400.0} for k, v in _CP.items()}
SOIL_SOURCE = "Carsel & Parrish (1988) Water Resour. Res. 24(5) 토성별 평균값"


# ───────────────────────────── 객체 역할 ─────────────────────────────────
ROLES = ["반응재(고체)", "주입 개구부", "관 내부(빈 공간)", "불투수", "저투수 실링", "투수 충전재", "토양", "투수성 판(구멍)"]
PLATE_ROLE = "투수성 판(구멍)"

ROLE_DEFAULTS = {
    #               K[m/s], 공극률, 반응, α[1/m], n,   우선순위, 색
    "반응재(고체)":     dict(K=5e-4, phi=0.40, reactive=True, alpha=14.5, n=2.68, prio=4, color="#E3B505"),
    "주입 개구부":      dict(K=1e-2, phi=0.90, reactive=False, alpha=50.0, n=3.0, prio=7, color="#D7263D"),
    "관 내부(빈 공간)": dict(K=1e-2, phi=0.95, reactive=False, alpha=50.0, n=3.0, prio=2, color="#A9D6E5"),
    "불투수":          dict(K=0.0, phi=0.0, reactive=False, alpha=1.0, n=1.1, prio=6, color="#4A4E57"),
    "저투수 실링":      dict(K=1e-10, phi=0.45, reactive=False, alpha=0.5, n=1.09, prio=5, color="#7B5E57"),
    "투수 충전재":      dict(K=1e-4, phi=0.38, reactive=False, alpha=14.5, n=2.68, prio=3, color="#C9B79C"),
    "토양":            dict(K=None, phi=None, reactive=False, alpha=None, n=None, prio=1, color="#8B6F47"),
    # 상판·하판: 셀을 차지하지 않고 셀 사이 면의 직렬 저항(구멍 지름·개수·판 두께·채움에서 계산, plate.py)
    "투수성 판(구멍)":  dict(K=None, phi=None, reactive=False, alpha=None, n=None, prio=6, color="#5DA9E9"),
}
ROLE_SOURCE = ("역할별 K·공극률·vG 값은 가정(충전층·벤토나이트·모래 일반 범위). "
               "불투수는 계산에서 제외(흐름·물질 이동 없음). 주입 개구부는 기본 K=1e-2 m/s, 공극률 0.9(스크린·관 내부를 큰 K의 "
               "다공질로 근사 — 가정, A15). 토양 역할은 빈칸이면 토양 값을 그대로 씀. 관 내부 자유수(압력·유량 연속 조건)는 미구현.")

_KEYWORDS = [
    ("주입 개구부", ["screen", "slot", "port", "opening", "inlet", "nozzle", "개구", "포트", "스크린", "주입구"]),
    ("반응재(고체)", ["reactive", "media", "slad", "sulfur", "sulphur", "limestone", "cartridge",
                    "반응", "황", "석회", "카트리지"]),
    ("저투수 실링", ["bentonite", "seal", "grout", "clay", "벤토", "실링", "그라우트", "점토"]),
    ("관 내부(빈 공간)", ["void", "interior", "bore_space", "lumen", "빈공간", "관내부", "내부공간"]),
    ("토양", ["soil", "backfill", "surface_fill", "native", "토양", "되메움"]),
    ("투수 충전재", ["sand", "gravel", "pack", "filter", "모래", "자갈", "충전"]),
    ("투수성 판(구멍)", ["perforated", "perf_plate", "타공", "구멍판"]),
    ("불투수", ["shell", "casing", "pipe", "wall", "floor", "pad", "concrete", "leg", "point", "cap",
               "plate", "steel", "pvc", "케이싱", "벽", "바닥", "콘크리트", "관", "다리"]),
]


def suggest_role(name: str) -> str:
    """이름 키워드로 역할 제안. 영문은 단어(토큰) 단위로 비교해 'support'가 'port'로 잡히는 오류를 막는다."""
    low = name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9가-힣]+", low) if t]
    for role, keys in _KEYWORDS:
        for k in keys:
            if re.search(r"[가-힣]", k):
                if k in low:
                    return role
            elif "_" in k:
                if k in low:
                    return role
            elif any(t == k or (t.startswith(k) and len(k) >= 4) for t in tokens):
                return role
    return "불투수"


# ───────────────────────────── 공통 조건 ─────────────────────────────────
@dataclass
class SoilLayer:
    top: float          # 깊이 m
    bottom: float
    theta_r: float
    theta_s: float
    alpha: float        # 1/m
    n: float
    Ks: float           # m/s (수평)
    aniso: float = 1.0  # Kz/Kx


@dataclass
class Common:
    # 흙수조
    Lx: float = 0.50
    Ly: float = 0.50
    H_soil: float = 0.70         # 토양층(불포화대) 깊이 m
    H_aq: float = 0.15           # 아래 포화대(지하수층) 두께 m (0이면 없음)
    wt_depth: Optional[float] = None   # 지하수면 깊이 m (None이면 토양층 바닥)
    wt_margin_m: Optional[float] = None   # 반응 구간 바닥 ~ 지하수면 여유 거리 m (None이면 모세관대 1/α 자동, 가정)
    gw_velocity: float = 0.0     # 지하수 다르시 유속 m/day (+x 방향)
    bottom_bc: str = "지하수 수위"   # "지하수 수위" | "누수(배수 저항)" | "자유배수"
    leak_thick: float = 0.05         # 누수 하단: 배수층 두께 m (가정)
    leak_K: float = 1.0e-4           # 누수 하단: 배수층 K m/s (가정)
    grid_preset: str = "보통"
    nx: int = 34
    ny: int = 34
    nz: int = 56
    # 토양
    soil_mode: str = "간단"
    texture: str = "사양토(Sandy loam)"
    Ks_override: Optional[float] = None    # m/s
    init_saturation: float = 0.5           # 초기 포화도(0~1)
    layers: List[SoilLayer] = field(default_factory=list)
    aniso: float = 1.0
    aq_texture: str = "모래(Sand)"
    # 유입수
    q_mm_day: float = 5.0
    rain_series: List[List[float]] = field(default_factory=list)   # [[시작일, mm/일], ...] 계단형. 비면 q_mm_day 일정
    et_mm_day: float = 0.0           # 잠재 증발산 mm/일(지표에서 빠짐, 용질은 남음)
    h_crit: float = -100.0           # 증발 한계 압력수두 m
    NO3_in_mgN: float = 50.0
    nitrite_on: bool = False             # D4 (F11): NO3→NO2→N2 2단계(기본 꺼짐)
    k_NO2_ratio: Optional[float] = None  # D4: 2단계 반-차수 상수 / 1단계 상수 (입력 필수)
    het_k_d: Optional[float] = None      # D1: 토양 유기탄소 종속영양 탈질 1차 상수 1/일(입력 시 켜짐, 토양층 전체)
    agg_radius_m: Optional[float] = None # D3 (F8) 모드 D: 흙덩이 반지름 m (입력 필수)
    agg_R_O2: Optional[float] = None     # D3: 흙덩이 안 O2 소비율 mol/m³/s (입력 필수)
    well_model: bool = True          # v4.3 A1: 가는 스크린을 우물 모델(방사 보정 면 전도도)로 — 끄면 v4.2 셀 개구부
    lambda_mode: str = "직접 입력"     # F4: "직접 입력"(REGISTRY lambda_filt) | "여과 이론(TE 2004)"
    attach_alpha: Optional[float] = None   # F4 부착 효율 α(여과 이론일 때 필수, 기본값 없음)
    filt_U_m_s: Optional[float] = None     # F4 여과 계산 다르시 유속 m/s(여과 이론일 때 필수)
    # 사용자 입력(빈칸 = 미입력, 숨은 기본값 없음 R3) — 프로젝트 파일에 함께 저장
    field_unsat_m: Optional[float] = None          # v4.3 B2 현장 불포화대 두께 m
    field_screen_m: Optional[float] = None         # 현장 스크린 길이 m
    field_casing_d_m: Optional[float] = None       # 현장 케이싱 지름 m
    casing_frac: Optional[float] = None            # 케이싱 지름 / 흙수조 폭 비
    target_S_g: Optional[float] = None             # v4.3 B1 두 비교군 목표 황 질량 g(빈칸 = 영구 주입공 수렴 격자 전달량)
    target_S_src: str = ""                         # 목표 황 질량의 출처(격자·실행)
    printer_min_wall_m: Optional[float] = None     # 3D 프린터 최소 벽 두께 m
    printer_min_hole_m: Optional[float] = None     # 3D 프린터 최소 구멍 지름 m
    col_L_m: Optional[float] = None                # S3 문헌 칼럼(Moon 외 2004) 충전 길이 m
    col_D_m: Optional[float] = None                # 칼럼 안지름 m
    col_Q_mL_min: Optional[float] = None           # 칼럼 유량 mL/min
    col_phi: Optional[float] = None                # 칼럼 충전층 공극률
    col_dS_mm: Optional[float] = None              # 칼럼 황 입경 mm
    col_S_vfrac: Optional[float] = None            # 칼럼 고형분 중 황 부피분율(황:석회석 3:1이면 0.75)
    moist_calib_rows: List[dict] = field(default_factory=list)   # F9-7 포화도별 제거율 표(보정/확인)
    refill_ok_pct: float = 90.0      # X2: 재보충 '성공' = 계획 대비 전달률 ≥ 이 값(%)
    eta_R_min_pct: float = 1.0       # X4: 재보충 전 η가 이 값(%)보다 작으면 R 계산 안 함
    SO4_in_mg: float = 10.0
    Ca_in_mg: float = 20.0
    alk_in_mgCaCO3: float = 50.0
    gw_NO3_mgN: float = 0.0
    soil_NO3_mgN: float = 0.0
    soil_pH0: float = 7.0
    # 반응재
    ratio_S_CaCO3: float = 3.0
    packed_porosity: float = 0.40
    temperature: float = 22.0
    # 기간·수치
    days: float = 365.0
    flow_mode: str = "과도"         # "과도"(기본, 매 단계 Richards) | "준정상"(빠른 근사)
    flow_update_days: float = 10.0  # 준정상 재계산 주기
    flow_dt_max_h: float = 6.0      # 과도 흐름 최대 시간 간격(시간)
    flow_dt_min_s: float = 0.5      # 이보다 작아져도 수렴 못하면 FAILED
    calcite_mode: str = "속도론"    # or "평형"
    chem_engine: str = "PHREEQC"    # "PHREEQC"(엔진 사용) | "내장"(빠름, 이온쌍 포함)
    chem_dt_h: float = 2.0          # PHREEQC 반응 계산 간격(시간)
    splitting: str = "Lie"          # "Lie" | "Strang"(B3)
    pCO2_mode: str = "균일"          # "균일" | "깊이별"(B4)
    o2_stage: bool = True           # O2 추적(용존·기상, 호기성 황 산화, 탈질 억제) — v4 기본 켬
    o2_gas: bool = True             # 토양 공기 O2 확산(지표 대기와 교환, Millington–Quirk) — v4
    port_thin_as_soil: bool = True  # v4.4: 격자보다 얇은 주입 개구부(셀이 메시 체적의 3배 초과)는 빈 공간이 아니라 흙 물성(점 주입원)
    kinetics: str = "표면적(문헌, v4)"   # "표면적(문헌, v4)" | "부피(v3)"
    denit_moisture: str = "NEMIS(문헌)"   # (v4.1 호환용, v4.2부터 moist_model 사용)
    moist_model: str = "A 기계론(f_w·f_O2)"   # F9 수분 의존: A 기계론 | B 경험식(NEMIS 계열) | C 상한 — 서로 배타
    fw_form: str = "S^p"            # 모드 A f_w 형태: S^p | 젖음각-단극 | 젖음각-양극 | 젖음각-적도
    fw_theta_c: Optional[float] = None   # 젖음각(°) — 젖음각 환산 때만, 기본값 없음
    nemis_form: str = "멱함수"       # 모드 B 곡선: 멱함수 | 지수형
    nemis_beta: Optional[float] = None   # 지수형 β — 기본값 없음(입력 필요)
    mode3: bool = True              # 결과 표에 모드 A·B·C를 항상 함께 계산(F9-5)
    moist_thr: float = 0.62         # 수분 함수 문턱 포화도(사용자 지정일 때) — NEMIS 기본값 0.62는 토양 탈질 문헌값
    moist_exp: float = 1.74         # 수분 함수 지수(사용자 지정일 때)
    moist_mode: str = "계산값(Richards)"   # "계산값(Richards)" | "반응 구역 포화도 지정"(조작 변수)
    moist_set: float = 0.8          # 반응 구역 포화도 지정값(0~1) — 반응·호기성 산화 계산에 쓰고 흐름은 Richards 그대로
    chem_mode: str = "SLAD"         # "SLAD" | "추적자(반응 없음)" | "1차 감쇠"(대체품 모드)
    k1_decay_day: float = 0.1       # 1차 감쇠 모드 속도상수 1/일
    inject_model: str = "과도 Richards"   # 또는 "중첩 근사(빠름)"(B1)
    straining: bool = True          # 입자/토양 입경비로 필터 케이크 판정(B7)
    dt_factor: float = 1.0          # 이송 시간 간격 배율(시간 간격 독립성 점검용, 0.5 = 절반)
    heterogeneity: Dict[str, float] = field(default_factory=dict)   # {"sigma_lnK", "corr_len", "seed"} 토양 불균질성
    phi_reac_override: Optional[float] = None    # 반응재 초기 공극률 일괄 지정(불확실성 분석용)
    calibrated: Dict[str, float] = field(default_factory=dict)      # 실험 보정으로 정한 파라미터('보정' 라벨)
    n_snapshots: int = 30
    light: bool = False
    redistribute_runoff: bool = True   # 구조물 덮개 등에서 생긴 지표 유출을 열린 지표로 재분배(가정)
    disturb_thick: float = 0.0     # 삽입 교란대 두께 m
    disturb_Kmult: float = 1.0     # 교란대 K 배율
    params: Dict[str, float] = field(default_factory=dict)   # REGISTRY 덮어쓰기

    def p(self, key):
        return self.params.get(key, REGISTRY[key][0])

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["layers"] = [SoilLayer(**l) for l in d.get("layers", [])]
        if "moist_model" not in d and "denit_moisture" in d:        # v4.1 저장 파일 호환
            dm = str(d["denit_moisture"])
            d["moist_model"] = "C 상한(억제 없음)" if dm.startswith("끔") else "B 경험식(NEMIS 계열)"
        return Common(**{k: v for k, v in d.items() if k in Common.__dataclass_fields__})


GRID_PRESETS = {"가벼움": (22, 22, 36), "보통": (32, 32, 52), "정밀": (46, 46, 76)}
