"""황 슬러리 설계 변수(v4.1): 황 투입량·입경·비표면적·황:물 배합·수분 유지 재료.

연구 설계(사용자 지정):
  - 독립변수 = 황 투입량, 수분 조건, 황 입경, 슬러리 배합(황:물, 수분 유지 재료 종류·비율)
  - 통제변수 = 흙수조의 지중 수리 조건(지하수 유속·수위·지층 K), 유입·초기 질산염 농도
  - 지하수 흐름 자체는 최적화 대상이 아님(장치가 기존 흐름에서 질산염을 얼마나 제거하는지 계산)

수분 유지 재료의 효과는 문헌 값이 있는 것만 문헌 값으로 넣고, 없는 항목은 '가정'으로 표시하며 사용자가 덮어쓸 수 있다.
근거(요약은 README):
  - 벤토나이트: 다짐 모래-벤토나이트 혼합의 투수계수 0% 3.60e-5 cm/s → 3% 약 1e-7 cm/s → 9% 4.13e-9 cm/s
    (Chalermyanont & Arrykul 2005, Songklanakarin J. Sci. Technol. 27(2):313–323). 보수성(수분특성) 변화의 정량값은
    찾지 못함 → 점토 쪽으로의 보간은 가정. 팽윤은 계산하지 않음.
  - 바이오차: 메타분석 평균 공극률 +8.4 %, 유효수분량 +15.1 %, 포화 투수계수 +25.2 %
    (Omondi et al. 2016, Geoderma 274:28–34). 거친 흙에서 효과가 큼. 평균값이 어떤 투입률에 해당하는지는
    논문 요약에 없어 기준 투입률 5 % w/w로 선형 환산(가정).
  - 고흡수성 수지(하이드로겔): 사양토에서 0.7 % w/w까지, 건조 중 평균 수분량 +88 %, 용적밀도 −12.5 %, 포화 투수계수 증가
    (Aslan & Canbolat 2025, Soil & Environment 44(2):145–155). 투수계수 증가 크기는 요약에 없음 → 배율 1(가정).
  - 황 입자의 소수성(물에 잘 젖지 않음)과 분산제는 계산하지 않음(정량 문헌 없음).
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np

RHO = {"S": 2.07, "CaCO3": 2.71}          # g/cm3 고체 밀도

RETAINERS: Dict[str, dict] = {
    "없음": dict(rho=2.65, src="", label="—"),
    "벤토나이트": dict(rho=2.65, x_max=0.09, label="문헌(투수계수) + 가정(보수성)",
                   src="Chalermyanont & Arrykul (2005): 모래-벤토나이트 0–9 % w/w 투수계수. 보수성 보간·팽윤 없음은 가정"),
    "바이오차": dict(rho=1.5, x_ref=0.05, dphi=0.084, dK=0.252, dAWC=0.151, label="문헌 평균(메타분석) + 가정(투입률 환산)",
                 src="Omondi et al. (2016) 메타분석 평균: 공극률 +8.4 %, 유효수분 +15.1 %, Ks +25.2 % (기준 5 % w/w로 선형 환산 — 가정)"),
    "고흡수성 수지(SAP)": dict(rho=1.4, x_ref=0.007, dWC=0.88, dBD=-0.125, label="문헌(사양토 시험) + 가정(Ks 배율 1)",
                          src="Aslan & Canbolat (2025): 0.7 % w/w에서 건조 중 수분 +88 %, 용적밀도 −12.5 %. Ks 증가 크기 미상 → 1"),
    "사용자 정의": dict(rho=2.0, label="사용자 입력", src="Ks 배율·공극률·수분특성(α, n)을 직접 입력"),
}
BENT_PTS = (np.array([0.0, 0.03, 0.09]), np.log10(np.array([3.60e-7, 1.0e-9, 4.13e-11])))   # w/w, log10 K(m/s)


def default_design(form: Optional[str] = None) -> dict:
    """설계 변수 기본값(꺼짐). 켜면 비교군의 설치 표·역할표 대신 이 값으로 황 양·배합을 정한다."""
    slurry = bool(form) and form.startswith("(c)")
    return {"on": False, "S_dose_g": None, "S_d_mm": 0.2 if slurry else 4.0, "S_ssa_m2g": None,
            "water_per_S": 2.5, "retainer": "없음", "ret_frac": 0.0,
            "Ks_mult": None, "phi_set": None, "alpha_set": None, "n_set": None}


def get_design(reagent: Optional[dict]) -> Optional[dict]:
    d = (reagent or {}).get("design")
    if not d or not d.get("on"):
        return None
    out = default_design((reagent or {}).get("form"))
    out.update({k: v for k, v in d.items() if v is not None or k in ("S_dose_g", "S_ssa_m2g")})
    return out


def solids_composition(ratio_S_CaCO3: float, retainer: str, ret_frac: float) -> dict:
    """건조 고형분 조성. ratio = 황:석회석 부피비(공통 조건), ret_frac = 수분 유지 재료 질량분율(건조 고형분 기준).
    반환: 부피분율 vS, vC, vR(합 1)과 황 1 g당 고형분 질량·부피."""
    r = max(float(ratio_S_CaCO3), 1e-9)
    VS = 1.0 / RHO["S"]                 # cm3 per g S
    VC = VS / r
    mC = VC * RHO["CaCO3"]
    f = float(np.clip(ret_frac or 0.0, 0.0, 0.95)) if retainer != "없음" else 0.0
    mR = f / (1 - f) * (1.0 + mC)
    VR = mR / RETAINERS.get(retainer, RETAINERS["사용자 정의"])["rho"]
    Vs = VS + VC + VR
    return {"vS": VS / Vs, "vC": VC / Vs, "vR": VR / Vs, "m_solid_per_gS": 1.0 + mC + mR, "V_solid_per_gS": Vs,
            "mC_per_gS": mC, "mR_per_gS": mR}


def slurry_props(ratio_S_CaCO3: float, d: dict) -> dict:
    """황:물 배합에서 슬러리 성질. water_per_S = 황 1 g당 물 g."""
    comp = solids_composition(ratio_S_CaCO3, d.get("retainer", "없음"), d.get("ret_frac", 0.0))
    mW = max(float(d.get("water_per_S") or 0.0), 0.0)
    Vtot = comp["V_solid_per_gS"] + mW                       # cm3 per g S
    mtot = comp["m_solid_per_gS"] + mW
    return dict(comp, fs_vv=comp["V_solid_per_gS"] / Vtot, L_per_gS=Vtot * 1e-3,
                S_mass_frac=1.0 / mtot, water_mass_frac=mW / mtot, solids_mass_frac=comp["m_solid_per_gS"] / mtot)


def area_diameter(d: dict, fallback_m: float) -> float:
    """표면적 계산에 쓰는 등가 입경(m). 비표면적(m²/g)을 주면 d = 6/(ρ·SSA)."""
    ssa = d.get("S_ssa_m2g") if d else None
    if ssa:
        return 6.0 / (RHO["S"] * 1e6 * float(ssa))
    dm = d.get("S_d_mm") if d else None
    return float(dm) * 1e-3 if dm else fallback_m


def _vg_theta(h, thr, ths, a, n):
    m = 1 - 1 / n
    return thr + (ths - thr) / (1 + (a * np.abs(h)) ** n) ** m


def _awc(thr, ths, a, n):
    return _vg_theta(3.3, thr, ths, a, n) - _vg_theta(150.0, thr, ths, a, n)    # 포장용수량(−33 kPa) − 위조점(−1.5 MPa)


def _alpha_for_awc(thr, ths, a, n, target):
    """유효수분량이 target이 되는 α (log 이분법). 늘릴 수 없으면 가장 가까운 값."""
    # 유효수분은 α에 대해 단조가 아니다(한 점에서 최대). 기준 α에서 출발해 α를 줄이거나 키우며 처음 목표에 닿는 값을 찾는다.
    if _awc(thr, ths, a, n) >= target:
        return float(a)
    best_a, best_v = a, _awc(thr, ths, a, n)
    for direction in (-1.0, 1.0):
        a_prev = a
        for k in range(1, 401):
            a2 = a * 10 ** (direction * 0.01 * k)
            v = _awc(thr, ths, a2, n)
            if v >= target:
                lo, hi = np.log(a_prev), np.log(a2)          # 목표를 넘은 구간 안에서 이분법으로 정확히
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if _awc(thr, ths, np.exp(mid), n) >= target:
                        hi = mid
                    else:
                        lo = mid
                return float(np.exp(hi))
            if v > best_v:
                best_a, best_v = a2, v
            a_prev = a2
    return float(best_a)


def zone_hydraulics(base: dict, d: Optional[dict], clay: dict) -> dict:
    """반응 구역(충전재 또는 필터 케이크) 수리 성질에 수분 유지 재료 효과를 적용.
    base: dict(Ks, phi, alpha, n, thr). clay: 점토 vG(α, n) — 벤토나이트 보간용(가정).
    반환: dict(Ks_mult, phi, alpha, n, notes[])."""
    out = {"Ks_mult": 1.0, "phi": base["phi"], "alpha": base["alpha"], "n": base["n"], "notes": []}
    if not d:
        return out
    kind = d.get("retainer", "없음")
    x = float(d.get("ret_frac") or 0.0)
    R = RETAINERS.get(kind, RETAINERS["사용자 정의"])
    if kind == "벤토나이트" and x > 0:
        xm = R["x_max"]
        if x > xm:
            out["notes"].append(f"벤토나이트 {x * 100:.1f} %는 문헌 범위(0–9 %) 밖 — 9 % 값으로 자름")
        xx = min(x, xm)
        lk = float(np.interp(xx, *BENT_PTS))
        out["Ks_mult"] = 10 ** (lk - BENT_PTS[1][0])
        w = xx / xm
        out["alpha"] = float(base["alpha"] * (clay["alpha"] / base["alpha"]) ** w)
        out["n"] = float(1.0 + (base["n"] - 1.0) * ((clay["n"] - 1.0) / (base["n"] - 1.0)) ** w)
        out["notes"].append("벤토나이트: 투수계수 = 문헌 보간, 수분특성 = 점토 쪽 보간(가정), 팽윤 없음(가정)")
    elif kind == "바이오차" and x > 0:
        s = x / R["x_ref"]
        out["phi"] = float(min(base["phi"] * (1 + R["dphi"] * s), 0.9))
        out["Ks_mult"] = 1 + R["dK"] * s
        awc0 = _awc(base["thr"], base["phi"], base["alpha"], base["n"])
        out["alpha"] = _alpha_for_awc(base["thr"], out["phi"], base["alpha"], base["n"], awc0 * (1 + R["dAWC"] * s))
        out["notes"].append(f"바이오차: 메타분석 평균을 {x * 100:.1f} % w/w로 선형 환산(기준 5 %, 가정)")
    elif kind == "고흡수성 수지(SAP)" and x > 0:
        s = x / R["x_ref"]
        if s > 1:
            out["notes"].append(f"SAP {x * 100:.2f} %는 시험 범위(≤0.7 %) 밖 — 0.7 % 값으로 자름")
            s = 1.0
        out["phi"] = float(min(1 - (1 - base["phi"]) * (1 + R["dBD"] * s), 0.9))
        awc0 = _awc(base["thr"], base["phi"], base["alpha"], base["n"])
        out["alpha"] = _alpha_for_awc(base["thr"], out["phi"], base["alpha"], base["n"], awc0 * (1 + R["dWC"] * s))
        out["notes"].append("SAP: 건조 중 수분 증가율을 유효수분 증가로 근사(가정), Ks 배율 1(가정), 팽윤 부피 없음")
    # 사용자 덮어쓰기(모든 재료)
    if d.get("Ks_mult") not in (None, ""):
        out["Ks_mult"] = float(d["Ks_mult"])
    if d.get("phi_set") not in (None, ""):
        out["phi"] = float(d["phi_set"])
    if d.get("alpha_set") not in (None, ""):
        out["alpha"] = float(d["alpha_set"])
    if d.get("n_set") not in (None, ""):
        out["n"] = float(d["n_set"])
    return out


def apply_to_scenario(sc, c):
    """설계 변수를 비교군 설치 표에 반영한 복사본: 슬러리 주입 행의 부피·고형분비를 황 투입량·배합에서 계산.
    사전 배치 충전재의 양은 엔진이 media_scale로 맞춘다. 원본은 바꾸지 않는다."""
    d = get_design(getattr(sc, "reagent", None))
    if d is None or sc is None:
        return sc
    import install_events as ie
    sc2 = copy.copy(sc)
    sc2.reagent = dict(sc.reagent)
    rows = [dict(r) for r in sc.events]
    sp = slurry_props(c.ratio_S_CaCO3, d)
    inj = [r for r in rows if r.get("이벤트") == "시약 주입" and (r.get("부피(L)") or 0) > 0]
    form = (sc.reagent or {}).get("form") or ""
    if inj and (form.startswith("(c)") or form.startswith("(e)")):
        for r in rows:
            if r.get("이벤트") in ("시약 주입", "재보충") and r.get("부피(L)"):
                r["고형분비(v/v)"] = sp["fs_vv"]
        if d.get("S_dose_g"):
            tot = sum(float(r["부피(L)"]) for r in inj)
            need = float(d["S_dose_g"]) * sp["L_per_gS"]
            for r in inj:
                r["부피(L)"] = float(r["부피(L)"]) / tot * need
    sc2.events = rows
    _ = ie
    return sc2
