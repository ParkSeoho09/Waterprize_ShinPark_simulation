"""실험 형상·크기(Part 1)와 케이슨 치수 계산기(Part 2), 매개변수형 3MF 생성.

원칙: 임의 상수 금지 — 빠진 입력은 ValueError로 멈춘다(R3). 단위 환산은 params.py 상수만 쓴다(R2).
계산기는 '제안'만 하고, 사용자가 고친 값이 우선한다.

문헌 근거(쪽수는 lit/ 추출본 기준):
  EPA (2013) In situ remediation — p.44: 스크린 단위 길이당 전송 용량 사례 2 gpm/ft, 5~10 ft 짧은 구간 여러 개 권장;
                                    pp.24–25: 100~500 psi에서 흙 파쇄 가능.
  ITRC (2011) PRB-5 — §5.1.3 필요 두께 b = V·t_res(·SF), §5.1.5 SF 2~3 (계산 두께의 2~3배가 적절할 수 있음);
                      케이슨 설치 공법(깊이 20 m 초과, 지름 최대 4.6 m) — 설치 교란(경계면 도말·매질 침하) 언급.
  Courcelles (2014) — 식 26 체류시간 = 공극률·반응재 체적/유량 (식 원문은 이미지, 원문 확인 필요);
                      반응재/토양 K 비 6·10 사용, 10 이상이면 수리 문제 없음(차수벽 길이 > 10·Yp 조건).
  Moon 외 (2004) — 반-차수 속도 약 2.0 (mg/L)^½/h (점검용, min→h 환산 원문 확인 필요).
"""
from __future__ import annotations

import io
import zipfile
from typing import Dict, List, Optional

import numpy as np

import moisture as MO
from params import (EPA_SCREEN_CAP_GPM_FT, GPM_PER_FT_TO_M2_S, MM_DAY_TO_M_H, MW, REGISTRY, RHO_SOLID, S_PER_N_G,
                    SOIL_TEXTURES)

MOON2004_K = 2.0          # (mg/L)^½/h, 점검용(원문 확인 필요)
NAMES = {"caisson_down": "케이슨형 — 불포화대 적용형(수직 하향 침투, caisson_down)",
         "monkstown_saturated_reference": "Monkstown형 포화대 참고(상향 흐름, 불포화대에서는 동작 안 함)"}


def _need(d: dict, keys):
    miss = [k for k in keys if d.get(k) is None]
    if miss:
        raise ValueError("입력이 비어 있습니다: " + ", ".join(miss) + " — 값을 채우세요(숨은 기본값 없음, R3).")


# ───────────────────────────── Part 1-1 깊이 구간 ─────────────────────────────
def water_table(c) -> Dict:
    """지하수면 깊이(지표 기준 m)와 근거. wt_depth가 비면 바닥 경계 조건과 흙 두께로 계산해 기록한다."""
    if c.wt_depth is not None:
        return {"wt_m": float(c.wt_depth), "basis": "사용자 입력 wt_depth"}
    if c.bottom_bc == "지하수 수위":
        return {"wt_m": float(c.H_soil), "basis": f"바닥 경계 '지하수 수위': 토양층 바닥(포화대 윗면) = H_soil {c.H_soil:g} m"}
    return {"wt_m": float(c.H_soil + c.H_aq),
            "basis": f"바닥 경계 '{c.bottom_bc}': 영역 안에 지하수면 없음 → 영역 바닥(H_soil + H_aq = {c.H_soil + c.H_aq:g} m)을 상한으로 씀"}


def depth_zone(c, top_m: float = 0.0) -> Dict:
    """두 비교군 공통 깊이 구간: top(지표 또는 주입 시작 깊이) ~ 지하수면 − 여유 거리(기본 1/α, 가정)."""
    from engine import wt_margin
    wt = water_table(c)
    margin, src = wt_margin(c)
    bot = wt["wt_m"] - margin
    if bot <= top_m + 1e-9:
        raise ValueError(f"깊이 구간이 없습니다: 지하수면 {wt['wt_m']:.3f} m − 여유 {margin:.3f} m = {bot:.3f} m ≤ 시작 깊이 {top_m:.3f} m.")
    return {"top_m": float(top_m), "bot_m": float(bot), "length_m": float(bot - top_m), "wt_m": wt["wt_m"],
            "wt_basis": wt["basis"], "margin_m": float(margin), "margin_basis": src}


# ───────────────────────────── Part 1-2 크기 ─────────────────────────────
def scale_ratio(c, field_dims: Dict) -> Dict:
    """기하 상사: 흙수조(가로·세로·깊이) / 현장 원형 치수. field_dims: Lx, Ly, H (m)."""
    _need(field_dims, ["Lx", "Ly", "H"])
    r = {"x": c.Lx / field_dims["Lx"], "y": c.Ly / field_dims["Ly"], "z": c.H_soil / field_dims["H"]}
    r["min"] = min(r.values())
    return r


def printer_check(dims: Dict, printer: Dict) -> List[str]:
    """3D 프린터 최소 벽 두께·구멍 지름 점검. 위반은 오류 문자열 목록."""
    _need(printer, ["min_wall_m", "min_hole_m"])
    errs = []
    if dims.get("wall_m") is not None and dims["wall_m"] < printer["min_wall_m"] - 1e-12:
        errs.append(f"벽 두께 {dims['wall_m'] * 1e3:.2f} mm < 프린터 최소 {printer['min_wall_m'] * 1e3:.2f} mm")
    if dims.get("hole_d_m") is not None and dims["hole_d_m"] < printer["min_hole_m"] - 1e-12:
        errs.append(f"구멍 지름 {dims['hole_d_m'] * 1e3:.2f} mm < 프린터 최소 {printer['min_hole_m'] * 1e3:.2f} mm")
    return errs


# ───────────────────────────── Part 1-3 스크린 길이 ─────────────────────────────
def screen_design(Q_Lmin: float, zone: Dict, cap_gpm_ft: float = EPA_SCREEN_CAP_GPM_FT,
                  fracs=(0.25, 0.5, 0.75, 1.0), n_segments=(1, 2, 3)) -> Dict:
    """L_s = 설계 주입률 / 단위 길이 전송 용량(EPA 2013 p.44, 2 gpm/ft). 가용 구간 배수·구간 수 스윕 표."""
    if Q_Lmin is None or Q_Lmin <= 0:
        raise ValueError("설계 주입률(L/min)을 입력하세요.")
    cap = cap_gpm_ft * GPM_PER_FT_TO_M2_S * 60000.0          # L/min per m
    Ls = Q_Lmin / cap
    avail = zone["length_m"]
    rows = []
    for f in fracs:
        for n in n_segments:
            L = f * avail
            seg = L / n
            gap = (avail - L) / max(n - 1, 1) if n > 1 else 0.0
            rows.append({"가용 구간 배수": f, "구간 수": n, "스크린 총길이 m": L, "구간 길이 m": seg, "구간 간격 m": gap,
                         "용량 대비 L_s 비": L / Ls if Ls > 0 else float("nan"),
                         "비고": ("L_s보다 짧음 — 용량 초과" if L < Ls else "")})
    return {"cap_L_min_m": cap, "L_s_m": Ls, "avail_m": avail, "rows": rows,
            "basis": f"L_s = {Q_Lmin:g} L/min ÷ {cap:.1f} L/min/m (EPA 2013 p.44, {cap_gpm_ft:g} gpm/ft) = {Ls * 1000:.1f} mm"}


def field_reference_Ls(Q_Lmin: float, cap_gpm_ft: float = EPA_SCREEN_CAP_GPM_FT) -> Dict:
    """현장 규모 참고값(흙수조 적용 불가, v4.3 B2): L_s = 주입률 / 2 gpm/ft(EPA 2013 p.44). 기본 계산에서 쓰지 않는다."""
    cap = cap_gpm_ft * GPM_PER_FT_TO_M2_S * 60000.0
    return {"L_s_m": Q_Lmin / cap, "label": "현장 규모 참고값(흙수조 적용 불가)",
            "basis": f"{Q_Lmin:g} L/min ÷ {cap:.1f} L/min/m (EPA 2013 p.44, {cap_gpm_ft:g} gpm/ft)"}


def scaled_well(field: Dict, zone: Dict, c, printer_min_m=None, fracs=(0.25, 0.5, 0.75, 1.0), n_segments=(1, 2, 3)) -> Dict:
    """기하 상사 축척(v4.3 B2). field: 현장 불포화대 두께 unsat_m, 현장 스크린 길이 screen_m, 현장 케이싱 지름 casing_d_m,
    케이싱 지름/흙수조 폭 비 casing_frac(입력). 축척 계수 s = 흙수조 불포화대(깊이 구간) / 현장 불포화대.
    스크린 길이 = 현장 스크린·s (참고) 및 가용 구간 0.25~1.0배 × 단일/다중 구간 스윕. 케이싱 지름 = max(비율·폭, 프린터 최소)."""
    _need(field, ["unsat_m", "screen_m", "casing_d_m", "casing_frac"])
    s = zone["length_m"] / float(field["unsat_m"])
    L_sim = float(field["screen_m"]) * s
    d_frac = float(field["casing_frac"]) * min(c.Lx, c.Ly)
    d_sim = float(field["casing_d_m"]) * s
    d_use = max(d_frac, printer_min_m or 0.0)
    rows = []
    avail = zone["length_m"]
    for f in fracs:
        for n in n_segments:
            L = f * avail
            rows.append({"가용 구간 배수": f, "구간 수": n, "스크린 총길이 m": L, "구간 길이 m": L / n,
                         "구간 간격 m": (avail - L) / max(n - 1, 1) if n > 1 else 0.0,
                         "현장 축척 길이 대비": L / L_sim if L_sim > 0 else float("nan")})
    return {"scale": s, "screen_scaled_m": min(L_sim, avail), "casing_d_m": d_use, "casing_d_scaled_m": d_sim,
            "rows": rows, "basis": (f"축척 s = 흙수조 불포화대 {avail:.3f} m / 현장 {field['unsat_m']:g} m = {s:.4f}; "
                                    f"스크린 {field['screen_m']:g} m × s = {L_sim * 1e3:.1f} mm; "
                                    f"케이싱 지름 = max(폭 비 {field['casing_frac']:g} × {min(c.Lx, c.Ly):g} m, 프린터 최소 "
                                    f"{(printer_min_m or 0) * 1e3:.1f} mm) = {d_use * 1e3:.1f} mm")}


# ───────────────────────────── Part 2 케이슨 치수 계산기 ─────────────────────────────
CALC_KEYS = ["A_tank_m2", "zone_len_m", "mass_g", "porosity", "S_mass_frac", "C0_mgN", "q_mm_day", "T_days",
             "SF", "target", "S_media", "d_S_m", "K_soil_m_s"]


def caisson_calc(inp: Dict, moist_cfg: Dict, t_min_m: float = 0.0, C_O2_molm3: Optional[float] = None,
                 SF_sweep=(1.0, 2.0, 3.0), K_ratios=(1.0, 3.0, 6.0, 10.0), k_half_a=None, k_ox_a=None) -> Dict:
    """inp: A_tank_m2, zone_len_m, mass_g(총 반응재), porosity, S_mass_frac(고형분 중 황 질량분율), C0_mgN, q_mm_day,
    T_days(목표 운영 기간), SF, target(목표 제거율), S_media(반응재 포화도 WFPS), d_S_m(황 입경), K_soil_m_s.
    모드별 필요 두께 표(F9-6), SF 스윕, 황 수명(호기성 산화 손실 포함), 단면적·체류시간 점검, K 비 스윕 값."""
    _need(inp, CALC_KEYS)
    phi = float(inp["porosity"]); fS = float(inp["S_mass_frac"])
    rho_mix = 1.0 / (fS / RHO_SOLID["S"] + (1 - fS) / RHO_SOLID["CaCO3"])          # kg/m3 고체
    rho_b = (1 - phi) * rho_mix                                                   # kg/m3 벌크
    phiS = (1 - phi) * (fS / RHO_SOLID["S"]) * rho_mix                            # 벌크 중 황 부피분율
    aS = 6.0 * phiS / float(inp["d_S_m"])                                        # m²/m³ 벌크
    kha = float(k_half_a if k_half_a is not None else REGISTRY["k_half_a"][0])   # (mg/L)^½·m/h
    S = float(inp["S_media"]); theta = S * phi
    if theta <= 0:
        raise ValueError("반응재 포화도가 0입니다 — 물이 없으면 탈질 체류시간을 정의할 수 없습니다.")
    q_m_h = float(inp["q_mm_day"]) * MM_DAY_TO_M_H
    V = q_m_h / theta                                                            # 공극 유속 m/h (ITRC 5.1.3)
    k_w = kha * aS / theta                                                       # 물 부피 기준 (mg/L)^½/h
    rows = []
    for full in MO.MODES[:3]:
        L = MO.mode_letter(full)
        if L == "A" and C_O2_molm3 is None:
            rows.append({"모드": full, "상태": "계산 안 함", "이유": "모드 A는 반응재 O₂ 농도(C_O2) 입력이 필요합니다"})
            continue
        fm, fo = MO.factors(full, np.array([S]), None if C_O2_molm3 is None else np.array([C_O2_molm3]), moist_cfg)
        keff = float(k_w * fm[0] * fo[0])
        t_h = MO.residence_time_h(float(inp["C0_mgN"]), float(inp["target"]), keff)
        z = V * t_h * float(inp["SF"])
        thk = max(z, t_min_m)
        row = {"모드": full, "상태": "OK", "f_moist": float(fm[0]), "f_O2": float(fo[0]), "k_eff (mg/L)^½/h": keff,
               "필요 체류시간 h": t_h, "V m/h": V, "z = V·t·SF m": z, "두께 m": thk,
               "두께 결정": "구조 최소" if t_min_m > z else "체류시간"}
        for sf in SF_sweep:
            row[f"z(SF={sf:g}) m"] = V * t_h * sf
        if not np.isfinite(thk):
            row["상태"] = "도달 불가"
            row["이유"] = f"이 포화도({S:.2f})에서 f_moist·f_O2 = {fm[0] * fo[0]:.3g} → 목표 제거율에 필요한 두께가 무한대"
        if np.isfinite(thk) and thk > 0:
            A = float(inp["mass_g"]) / 1000.0 / (rho_b * thk)
            row["단면적 m²"] = A
            row["지름 m(원형)"] = 2 * np.sqrt(A / np.pi)
            row["포획 분율 %"] = A / float(inp["A_tank_m2"]) * 100
            Q = q_m_h * A                                                        # m³/h (바로 위 면적만 받음)
            row["체류시간 점검(포화, 공극률·체적/유량) h"] = phi * A * thk / Q if Q > 0 else float("nan")
            row["체류시간 점검(불포화, θ·체적/유량) h"] = theta * A * thk / Q if Q > 0 else float("nan")
            if thk > float(inp["zone_len_m"]):
                row["상태"] = "오류"; row["이유"] = f"두께 {thk:.3f} m가 깊이 구간 {inp['zone_len_m']:.3f} m보다 깁니다"
            if A > float(inp["A_tank_m2"]):
                row["상태"] = "오류"; row["이유"] = "단면적이 흙수조보다 큽니다"
            # 황 수명(항목 3): 2.52 g S/g N × 기간 N 부하 × SF + 호기성 산화 손실(상한: O2 무제한, k_ox_a·a_S·체적·기간)
            N_load = float(inp["C0_mgN"]) * float(inp["q_mm_day"]) * 1e-3 * A * float(inp["T_days"])   # g N (mg/L × m/d × m² × d = g)
            S_need_dn = S_PER_N_G * N_load * float(inp["SF"])
            kox = float(k_ox_a if k_ox_a is not None else REGISTRY["k_ox_a"][0])
            S_aer = kox * aS * A * thk * float(inp["T_days"]) * 86400.0 * MW["S"]                    # g S (상한)
            S_have = float(inp["mass_g"]) * fS
            row["기간 N 부하 g"] = N_load
            row["황 필요(탈질×SF) g"] = S_need_dn
            row["황 호기성 손실 상한 g"] = S_aer
            row["황 보유 g"] = S_have
            row["황 수명 판정"] = "충분" if S_need_dn + S_aer <= S_have else ("탈질만은 충분(호기성 손실 포함 시 부족)"
                                                                         if S_need_dn <= S_have else "부족")
        rows.append(row)
    Krows = [{"K 비(반응재/토양)": kr, "K_media m/s": kr * float(inp["K_soil_m_s"]),
              "비고": ("Courcelles 2014: 10 이상이면 수리 문제 없음(차수벽 > 10·Yp 조건)" if kr >= 10 else
                     ("Courcelles 2014 예제 값" if kr == 6 else ""))} for kr in K_ratios]
    check = {"Moon 2004 점검값 (mg/L)^½/h": MOON2004_K, "앱 포화 k_w (mg/L)^½/h": kha * aS / phi,
             "비(앱/Moon)": (kha * aS / phi) / MOON2004_K, "라벨": "원문 확인 필요(min→h 환산)"}
    return {"rho_bulk_kg_m3": rho_b, "a_S_m2_m3": aS, "theta": theta, "V_m_h": V, "k_w": k_w, "rows": rows,
            "K_sweep": Krows, "moon_check": check,
            "basis": ("t = 2√C0(1−√(1−목표))/k_eff, k_eff = k½a·a_S/θ·f_moist·f_O2; z = V·t·SF, V = q/θ (ITRC 2011 §5.1.3); "
                      "SF 기본 2, 스윕 1·2·3 (ITRC §5.1.5); 두께 = max(z, 구조 최소); 단면적 = 질량/(ρ_b·두께); "
                      "체류시간 점검 = 공극률·체적/유량 (Courcelles 2014 식 26, 원문 확인 필요)")}


def resolution_check(diam_m: float, dx: float, gap_m: Optional[float] = None) -> Dict:
    """K4: 반응 챔버 가로 ≥ 4칸, 물길(틈) ≥ 2칸인지. 부족하면 면 전도도(Part 3)로 처리했다고 기록."""
    nc = diam_m / dx
    out = {"chamber_cells": nc, "chamber_ok": nc >= 4}
    if gap_m is not None:
        out["gap_cells"] = gap_m / dx; out["gap_ok"] = gap_m / dx >= 2
    msgs = []
    if not out["chamber_ok"]:
        msgs.append(f"반응 챔버 가로 {nc:.1f}칸 < 4칸 — 격자를 세분하거나 큰 격자 결과는 해상 부족으로 기록")
    if gap_m is not None and not out["gap_ok"]:
        msgs.append(f"물길 {out['gap_cells']:.1f}칸 < 2칸 — 투수성 판 면 전도도(Part 3)로 처리")
    out["log"] = msgs
    return out


# ───────────────────────────── 3MF 생성 ─────────────────────────────
def _cyl(r, z0, z1, n=40, cx=0.0, cy=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    V = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)], [[cx, cy, z0], [cx, cy, z1]]])
    T = []
    b0, t0 = 2 * n, 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        T += [[i, j, n + j], [i, n + j, n + i], [b0, j, i], [t0, n + i, n + j]]
    return V, np.array(T)


def _tube(r_in, r_out, z0, z1, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    V = np.vstack([np.c_[r_out * c, r_out * s, np.full(n, z0)], np.c_[r_out * c, r_out * s, np.full(n, z1)],
                   np.c_[r_in * c, r_in * s, np.full(n, z0)], np.c_[r_in * c, r_in * s, np.full(n, z1)]])
    T = []
    for i in range(n):
        j = (i + 1) % n
        oi, oj, Oi, Oj = i, j, n + i, n + j            # 바깥 아래·위
        ii, ij, Ii, Ij = 2 * n + i, 2 * n + j, 3 * n + i, 3 * n + j
        T += [[oi, oj, Oj], [oi, Oj, Oi]]               # 바깥 면(밖으로)
        T += [[ii, Ij, ij], [ii, Ii, Ij]]               # 안쪽 면(축 쪽으로)
        T += [[oi, ii, ij], [oi, ij, oj]]               # 아랫면
        T += [[Oi, Oj, Ij], [Oi, Ij, Ii]]               # 윗면
    return V, np.array(T)


def write_3mf(objects: List[Dict], title: str, description: str) -> bytes:
    """objects: [{name, V (m, z 위쪽, 지표 z=0), T}] → 3MF(mm)."""
    xs = ['<?xml version="1.0" encoding="UTF-8"?>',
          '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
          f'<metadata name="Title">{title}</metadata>', f'<metadata name="Description">{description}</metadata>', '<resources>']
    for i, o in enumerate(objects, start=1):
        xs.append(f'<object id="{i}" name="{o["name"]}" type="model"><mesh><vertices>')
        xs += [f'<vertex x="{v[0] * 1e3:.4f}" y="{v[1] * 1e3:.4f}" z="{v[2] * 1e3:.4f}"/>' for v in o["V"]]
        xs.append('</vertices><triangles>')
        xs += [f'<triangle v1="{t[0]}" v2="{t[1]}" v3="{t[2]}"/>' for t in o["T"]]
        xs.append('</triangles></mesh></object>')
    xs.append('</resources><build>')
    xs += [f'<item objectid="{i}"/>' for i in range(1, len(objects) + 1)]
    xs.append('</build></model>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
                   'relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/'
                   '3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", "\n".join(xs))
    return buf.getvalue()


def caisson_down(diam_m, media_thk_m, top_depth_m, wall_m, plate_t_m, zone: Dict, n=40) -> Dict:
    """케이슨 — 불포화대 적용형(K1): 불투수 측벽 + 투수성 상판(유입) + 반응재층 + 투수성 하판(유출).
    z는 지표 0, 아래로 음수. 유출(하판) z가 유입(상판)보다 낮지 않으면 오류. 반환 {bytes, objects, dims}."""
    for k, v in dict(diam_m=diam_m, media_thk_m=media_thk_m, top_depth_m=top_depth_m, wall_m=wall_m, plate_t_m=plate_t_m).items():
        if v is None or not np.isfinite(v) or v < 0 or (k != "top_depth_m" and v <= 0):
            raise ValueError(f"케이슨 치수 {k}가 비었거나 0 이하입니다.")
    r = diam_m / 2
    z_top_plate_hi = -top_depth_m
    z_top_plate_lo = z_top_plate_hi - plate_t_m
    z_media_lo = z_top_plate_lo - media_thk_m
    z_bot_plate_lo = z_media_lo - plate_t_m
    inlet_z, outlet_z = 0.5 * (z_top_plate_hi + z_top_plate_lo), 0.5 * (z_media_lo + z_bot_plate_lo)
    if not outlet_z < inlet_z:
        raise ValueError("유출구 z가 유입구보다 낮지 않습니다(하향 흐름 구성이 아님).")
    if -z_bot_plate_lo > zone["bot_m"] + 1e-9:
        raise ValueError(f"케이슨 바닥 {-z_bot_plate_lo:.3f} m가 깊이 구간 바닥 {zone['bot_m']:.3f} m보다 깊습니다(K6).")
    objs = []
    V, T = _tube(r, r + wall_m, z_bot_plate_lo, 0.0, n); objs.append({"name": "caisson_side_wall", "V": V, "T": T})
    V, T = _cyl(r, z_top_plate_lo, z_top_plate_hi, n); objs.append({"name": "perforated_plate_top", "V": V, "T": T})
    V, T = _cyl(r, z_media_lo, z_top_plate_lo, n); objs.append({"name": "reactive_media", "V": V, "T": T})
    V, T = _cyl(r, z_bot_plate_lo, z_media_lo, n); objs.append({"name": "perforated_plate_bottom", "V": V, "T": T})
    if top_depth_m > 0:
        V, T = _cyl(r, z_top_plate_hi, 0.0, n); objs.append({"name": "surface_fill_above", "V": V, "T": T})
    dims = dict(diam_m=diam_m, media_thk_m=media_thk_m, top_depth_m=top_depth_m, wall_m=wall_m, plate_t_m=plate_t_m,
                inlet_z=inlet_z, outlet_z=outlet_z, bottom_depth_m=-z_bot_plate_lo)
    desc = (f"caisson_down: unsaturated-zone downward-flow caisson. impermeable side wall + perforated top plate (inlet) + reactive media "
            f"+ perforated bottom plate (outlet). D={diam_m * 1e3:.1f} mm, media {media_thk_m * 1e3:.1f} mm, from Part 2 calculator. m→mm, z=0 ground.")
    return {"bytes": write_3mf(objs, "caisson_down", desc), "objects": objs, "dims": dims}


def injection_well(well_d_m, wall_m, zone: Dict, screen_frac=1.0, n_seg=1, seg_gap_m=None, n=24) -> Dict:
    """영구 주입공(원래 형태 유지, 구간만 불포화대로 한정): 케이싱(불투수) + 스크린 구간(주입 개구부)."""
    if well_d_m is None or wall_m is None:
        raise ValueError("주입공 지름·벽 두께를 입력하세요.")
    if not (0 < screen_frac <= 1):
        raise ValueError("스크린 길이 배수는 0~1입니다.")
    r = well_d_m / 2
    L = zone["length_m"] * screen_frac
    gap = seg_gap_m if seg_gap_m is not None else ((zone["length_m"] - L) / max(n_seg - 1, 1) if n_seg > 1 else 0.0)
    seg = L / n_seg
    objs = []
    z = -zone["bot_m"]
    tops = []
    for i in range(n_seg):                       # 아래에서 위로 구간 배치
        zlo, zhi = z, z + seg
        V, T = _cyl(r, zlo, zhi, n); objs.append({"name": f"screen_{i + 1}", "V": V, "T": T})
        tops.append((zlo, zhi)); z = zhi + gap
    # 케이싱: 맨 위 스크린 위 ~ 지표, 스크린 사이 구간
    zt = tops[-1][1]
    if zt < -1e-9:
        V, T = _tube(r, r + wall_m, zt, 0.0, n); objs.append({"name": "casing_blank_upper", "V": V, "T": T})
    for i in range(n_seg - 1):
        V, T = _tube(r, r + wall_m, tops[i][1], tops[i + 1][0], n); objs.append({"name": f"casing_blank_between_{i + 1}", "V": V, "T": T})
    V, T = _cyl(r + wall_m, -zone["bot_m"] - wall_m, -zone["bot_m"], n); objs.append({"name": "pvc_point", "V": V, "T": T})
    dims = dict(well_d_m=well_d_m, wall_m=wall_m, screen_total_m=L, n_seg=n_seg, seg_m=seg, gap_m=gap,
                screen_top_depth_m=-zt, screen_bot_depth_m=zone["bot_m"])
    desc = f"injection_well (unsaturated zone): {n_seg} screen segment(s), total {L * 1e3:.0f} mm, D={well_d_m * 1e3:.1f} mm. z=0 ground."
    return {"bytes": write_3mf(objs, "injection_well_unsat", desc), "objects": objs, "dims": dims}


def capture_fraction(res: dict, c) -> Dict:
    """K2: 포획 분율 = 반응재 단면적(위에서 본 투영) / 흙수조 단면적. 결과(res)의 반응재 셀에서 계산."""
    reac = res.get("reac0")
    g = res.get("grid") or {}
    if reac is None or not np.any(reac):
        return {"capture_pct": float("nan")}
    A = float(np.asarray(reac).any(axis=2).sum() * g["dx"] * g["dy"])
    return {"capture_pct": A / (c.Lx * c.Ly) * 100, "A_reac_m2": A}


def sample_roles(path_3mf: str, objects) -> Dict[str, dict]:
    """예시 3MF 옆 json의 역할·구멍 변수(사용자 수정 가능). 없으면 이름 키워드로 역할만 제안."""
    import json
    import os
    from params import PLATE_ROLE, suggest_role
    roles = {o.name: {"role": suggest_role(o.name)} for o in objects}
    js = os.path.splitext(path_3mf)[0] + ".json"
    if os.path.exists(js):
        info = json.load(open(js, encoding="utf-8"))
        for nm, rr in roles.items():
            if rr["role"] == PLATE_ROLE and info.get("plate"):
                rr.update(info["plate"])
            if rr["role"] == "반응재(고체)" and info.get("media_replaceable", True):
                rr["accessible"] = True                       # K7: 교체 가능(재보충 = 매질 교체)
    return roles


def hull_area_xy(V) -> float:
    """메시 꼭짓점의 xy 볼록 껍질 면적 m²(원통·상자 같은 볼록 반응재의 투영 단면적)."""
    from scipy.spatial import ConvexHull
    P = np.unique(np.round(np.asarray(V)[:, :2], 9), axis=0)
    return float(ConvexHull(P).volume) if len(P) >= 3 else 0.0
