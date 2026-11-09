"""SLAD 반응재 배치 방식 비교 시뮬레이터 — Streamlit 앱 (한국어 UI)

실행: run_app.bat 더블클릭  (또는  python -m streamlit run app.py)
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import install_events as ie  # noqa: E402
import metrics as M  # noqa: E402
import texts as TX  # noqa: E402
import viz  # noqa: E402
from engine import (LIQUID_MATERIALS, REAGENT_FORMS, THIOSULFATE_STOICH, Scenario, StopRequested,  # noqa: E402
                    build_geometry, control_breakthrough_day, eval_start_from, moist_cfg, planned_masses, run_scenario,
                    validate_moist, validate_reagent)
import phreeqc_engine as PE  # noqa: E402
import studies  # noqa: E402
import design as DZ  # noqa: E402
import moisture as MO  # noqa: E402
import runjob  # noqa: E402
from io_3mf import UNIT_LABEL, UNIT_TO_M, load_mesh_file, rescale_objects  # noqa: E402
from params import (GRID_PRESETS, REGISTRY, ROLE_DEFAULTS, ROLES, SOIL_SOURCE, SOIL_TEXTURES,  # noqa: E402
                    PLATE_ROLE, Common, SoilLayer, suggest_role)
from voxel import Placement, make_grid  # noqa: E402

st.set_page_config(page_title="SLAD 배치 방식 비교 시뮬레이터", layout="wide", page_icon="💧")
import struct as _struct  # noqa: E402
IS_32BIT = _struct.calcsize("P") == 4
ST_MIN = (1, 36)      # 지원 Streamlit 최소 버전(D6): width="stretch"·popover·toggle 사용. 낮으면 호환 래퍼로 대체.
try:
    _stv = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _stv = (0, 0)


def _compat(fn):
    """Streamlit 버전 호환: width="stretch"를 모르는 버전이면 use_container_width=True로 바꿔 다시 호출."""
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except TypeError as e:
            if "width" in k and "width" in str(e):
                k.pop("width")
                k["use_container_width"] = True
                return fn(*a, **k)
            raise
    return w


for _n in ("button", "data_editor", "dataframe", "download_button"):
    if hasattr(st, _n):
        setattr(st, _n, _compat(getattr(st, _n)))


# ───────────────────────────── 공통 도우미 ───────────────────────────────
def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_html(html, height):
    import streamlit.components.v1 as components
    components.html(html, height=height, scrolling=False)


def num(label, value, key, help=None, fmt="%.4g", step=None):
    """최솟값·최댓값 제한 없는 숫자 입력."""
    kw = dict(value=value, key=key, help=help, format=fmt)
    if step is not None:
        kw["step"] = step
    return st.number_input(label, **kw)


UNIT_KEYS = {"mm": "millimeter", "cm": "centimeter", "m": "meter"}
UNIT_FROM = {v: k for k, v in UNIT_KEYS.items()}

SAMPLES = {
    "영구 주입공 (injection_well)": ("samples/injection_well_scale0.05.3mf",
        "주입관 스크린을 통해 황:석회석 슬러리를 토양에 주입한다. 주입된 입자가 스크린 주변 공극에 걸려 반응재 층을 만든다. "
        "재보충은 같은 주입관으로 다시 주입한다(주입관 막힘이 누적될 수 있음).",
        "1) 흙수조 중앙에 주입정을 30분 동안 지표에서 수직으로 삽입한다. 2) 개구부를 1분 열어 둔다. "
        "3) 고형분 20% 슬러리 2.84 L를 펌프로 0.2 L/min 주입한다. 4) 60분 정치한다. "
        "5) 효율이 50% 미만이면 1 L를 펌프로 재보충, 7일 간격, 최대 5회, 작업 20분.",
        "(c) 입자 슬러리(포획)",
        [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
         dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
         dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
         dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0}),
         dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2,
                                       "트리거 η(%)": 50.0, "간격(일)": 7.0, "최대 회차": 5.0, "소요(분)": 20.0})]),
    "케이슨형 (caisson_down)": ("samples/caisson_down.3mf",
        "케이슨 — 불포화대 적용형(수직 하향 침투). 불투수 측벽 안에 황:석회석 반응재층을 두고, 위의 투수성 상판으로 빗물·관개수가 "
        "들어와 아래 투수성 하판으로 나간다. 깔때기가 없어 바로 위 면적의 물만 받는다(포획 분율). 반응재는 꺼내 바꿀 수 있다"
        "(재보충 = 매질 교체). 치수는 Part 2 계산기 제안값(samples/caisson_down.json).",
        "1) 반응재를 채운 케이슨을 흙수조 중앙에 30분 동안 삽입한다. 2) 효율이 50% 미만이면 반응재를 새것으로 교체한다"
        "(7일 간격, 최대 5회, 작업 20분).",
        "(b) 고체 카트리지 교체",
        [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
         dict(ie.empty_row("재보충"), **{"방식": "카트리지 교체", "트리거 η(%)": 50.0, "간격(일)": 7.0,
                                       "최대 회차": 5.0, "소요(분)": 20.0})]),
    "Monkstown형 포화대 참고 (monkstown_saturated_reference)": ("samples/monkstown_saturated_reference.3mf",
        "ITRC(2011) 상자 F의 Monkstown형(바깥 7 ft·챔버 4 ft·매질 22 ft, 상향 흐름)을 0.05배로 줄인 포화대 전용 참고 형상. "
        "상향 흐름은 양의 수두가 필요한 포화대 구성이라 불포화대 흙수조에서는 물이 반응재를 거의 지나지 않는다(우회). "
        "벽 두께·바닥 틈·챔버 높이는 가정. 비교 기준용으로만 남긴다.",
        "1) 반응재를 채운 케이슨을 흙수조 중앙에 30분 동안 삽입한다. 2) 재보충 불가.",
        "(a) 고체 사전 배치",
        [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0})]),
}


def ss():
    return st.session_state


def init_state():
    if "common" not in ss():
        ss().common = Common()
        ss().scen = []
        ss().cur = 0
        ss().results = None
        ss().running = False
        ss().ctrl_cache = {}
    if "sid" not in ss():
        import uuid
        ss().sid = uuid.uuid4().hex


init_state()


PLATE_COLS = ["구멍 지름(mm)", "구멍 개수", "개구율", "판 두께(mm)", "구멍 채움", "K_h (m/s)"]


def role_df_for(objects, old=None, preset=None):
    """preset: {객체 이름: roles dict} — 예시 3MF 옆 json(구멍 변수·교체 가능)."""
    rows = []
    oldmap = {} if old is None else {r["객체"]: r for r in old.to_dict("records")}
    preset = preset or {}
    for o in objects:
        if o.name in oldmap:
            rows.append(oldmap[o.name])
            continue
        pr = preset.get(o.name, {})
        role = pr.get("role") or suggest_role(o.name)
        d = ROLE_DEFAULTS[role]
        rows.append({"객체": o.name, "역할": role,
                     "K (m/s)": d["K"], "공극률": d["phi"], "반응": bool(d["reactive"]),
                     "우선순위": d["prio"], "교체 가능": bool(pr.get("accessible", False)), "얇은 객체": "자동",
                     "벽 두께(mm)": round(o.min_thickness_m * 1000, 3) if np.isfinite(o.min_thickness_m) else None,
                     "구멍 지름(mm)": pr["hole_d_m"] * 1e3 if pr.get("hole_d_m") is not None else None,
                     "구멍 개수": pr.get("n_holes"), "개구율": pr.get("open_ratio"),
                     "판 두께(mm)": pr["t_p_m"] * 1e3 if pr.get("t_p_m") is not None else None,
                     "구멍 채움": pr.get("fill") or ("반응재/흙" if role == PLATE_ROLE else None),
                     "K_h (m/s)": pr.get("K_h"),
                     "체적(cm³)": round(o.volume_m3 * 1e6, 3), "닫힘": o.closed})
    return pd.DataFrame(rows)


def roles_from_df(df):
    out = {}
    for r in df.to_dict("records"):
        role = r["역할"] if r["역할"] in ROLES else "불투수"
        d = ROLE_DEFAULTS[role]
        def val(x, default):
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return default
                return float(x)
            except (TypeError, ValueError):
                return default
        out[r["객체"]] = {"role": role, "K": val(r.get("K (m/s)"), d["K"]), "phi": val(r.get("공극률"), d["phi"]),
                         "reactive": bool(r.get("반응", d["reactive"])), "prio": int(val(r.get("우선순위"), d["prio"])),
                         "accessible": bool(r.get("교체 가능", False)), "thin_mode": r.get("얇은 객체") or "자동",
                         "thickness_m": val(r.get("벽 두께(mm)"), 2.0) / 1000.0,
                         "alpha": d["alpha"], "n": d["n"]}
        if role == PLATE_ROLE:                       # 투수성 판 구멍 변수(Part 3) — 빈칸은 엔진이 오류로 알림(R3)
            hd = val(r.get("구멍 지름(mm)"), None)
            tp = val(r.get("판 두께(mm)"), None)
            out[r["객체"]].update({"hole_d_m": hd / 1000.0 if hd is not None else None,
                                   "n_holes": val(r.get("구멍 개수"), None), "open_ratio": val(r.get("개구율"), None),
                                   "t_p_m": tp / 1000.0 if tp is not None else None,
                                   "fill": r.get("구멍 채움") or "반응재/흙", "K_h": val(r.get("K_h (m/s)"), None)})
    return out


def new_scenario(name="비교군"):
    return {"name": name, "mechanism": "", "install_text": "", "file_name": "", "file_bytes": None,
            "unit_src": "millimeter", "unit_sel": "millimeter", "objects": [],
            "title": "", "description": "", "role_df": pd.DataFrame(),
            "reagent": {"form": None, "material": "선택 안 함"}, "file_hash": "",
            "placement": dict(x=None, y=None, top_depth=0.0, rot_deg=0.0, tilt_deg=0.0, scale=1.0),
            "events_df": pd.DataFrame([ie.empty_row("구조물 삽입")]), "allow_refill_wo_opening": False,
            "mass_correction": True, "geom_report": None}


def load_file_into(sc, data, fname, preset=None):
    d = load_mesh_file(data, fname)
    sc["file_bytes"] = data
    sc["file_name"] = fname
    sc["unit_src"] = d["unit"]
    sc["unit_sel"] = d["unit"]
    sc["objects"] = d["objects"]
    import hashlib
    sc["file_hash"] = hashlib.sha256(data).hexdigest()
    sc["title"] = d["title"]
    sc["description"] = d["description"]
    sc["role_df"] = role_df_for(d["objects"], preset=preset)
    sc["geom_report"] = None


def scenario_obj(sc, c: Common) -> Scenario:
    pl = dict(sc["placement"])
    pl["x"] = c.Lx / 2 if pl.get("x") is None else pl["x"]
    pl["y"] = c.Ly / 2 if pl.get("y") is None else pl["y"]
    return Scenario(name=sc["name"], mechanism=sc["mechanism"], install_text=sc["install_text"],
                    file_name=sc["file_name"], unit=sc["unit_sel"], objects=sc["objects"],
                    roles=roles_from_df(sc["role_df"]) if len(sc["role_df"]) else {},
                    placement=Placement(**pl), events=sc["events_df"].to_dict("records"),
                    allow_refill_wo_opening=sc["allow_refill_wo_opening"],
                    mass_correction=sc["mass_correction"], reagent=dict(sc.get("reagent") or {}),
                    file_hash=sc.get("file_hash", ""))


def design_editor(sc, c, k):
    """비교군별 설계 변수(독립변수): 황 투입량·입경·비표면적·황:물 배합·수분 유지 재료."""
    rg = sc.setdefault("reagent", {"form": None, "material": "선택 안 함"})
    form = rg.get("form") or ""
    if not (form.startswith("(a)") or form.startswith("(b)") or form.startswith("(c)")):
        return
    d = rg.setdefault("design", DZ.default_design(form))
    with st.expander("🧪 황 투입량·입경·슬러리 배합 (연구 독립변수)", expanded=bool(d.get("on"))):
        d["on"] = st.toggle("설계 변수 사용(끄면 설치 표·역할표 값 그대로)", bool(d.get("on")), key=k + "dz_on")
        if not d["on"]:
            st.caption("켜면 아래 값으로 황 양과 배합을 정합니다. 슬러리 주입 비교군은 '시약 주입' 행의 부피·고형분비가, "
                       "사전 배치(케이슨) 비교군은 충전재의 황 양이 이 값으로 바뀝니다. 지하수 흐름·지층·유입 농도는 ② 공통 조건(통제변수) 그대로.")
            return
        slurry = form.startswith("(c)")
        try:
            so_ = scenario_obj(sc, c)
            pm = planned_masses(so_, c, make_grid(c), build_geometry(so_, c, make_grid(c))) if so_.objects else None
            S_now = (pm["S_equiv_mol"] * 32.06) if pm else None
        except Exception:       # noqa: BLE001
            S_now = None
        a1, a2, a3 = st.columns(3)
        dose = a1.number_input("황 총 투입량 (g)", value=float(d["S_dose_g"]) if d.get("S_dose_g") else
                               (round(S_now, 1) if S_now else 400.0), key=k + "dz_dose",
                               help="독립변수. 비우면 지금 설치 표·충전재의 황 양" + (f"(약 {S_now:.0f} g)" if S_now else ""))
        d["S_dose_g"] = float(dose)
        d["S_d_mm"] = float(a2.number_input("황 입경 (mm)", value=float(d.get("S_d_mm") or (0.2 if slurry else 4.0)),
                                            key=k + "dz_d", format="%.3g",
                                            help="표면적(반응 속도)·입자 포획·케이크 투수계수에 씀. 주입 슬러리 기본 0.2 mm, 입상 황 4 mm"))
        ssa = a3.number_input("황 비표면적 (m²/g, 선택)", value=d.get("S_ssa_m2g"), key=k + "dz_ssa", format="%.4g",
                              help="측정값이 있으면 입력(BET 등). 비우면 구형 입자 6/(ρd)로 계산")
        d["S_ssa_m2g"] = float(ssa) if ssa else None
        d_area = DZ.area_diameter(d, d["S_d_mm"] * 1e-3)
        st.caption(f"표면적 계산 등가 입경 {d_area * 1000:.3g} mm → 비표면적 {6 / (DZ.RHO['S'] * 1e6 * d_area):.4g} m²/g "
                   f"(반응 속도 ∝ 비표면적, Koenig & Liu 2001)")
        if slurry:
            b1, b2, b3 = st.columns(3)
            d["water_per_S"] = float(b1.number_input("황:물 배합 (황 1 g당 물 g)", value=float(d.get("water_per_S") or 2.5),
                                                     key=k + "dz_w", format="%.3g",
                                                     help="물만 늘리면 주입 부피와 처음 젖는 정도가 커지지만, 물은 흙의 수분특성대로 빠집니다"))
        else:
            b2, b3 = st.columns(2)
        rlist = list(DZ.RETAINERS.keys())
        d["retainer"] = b2.selectbox("수분 유지 재료", rlist, index=rlist.index(d.get("retainer", "없음")) if d.get("retainer") in rlist else 0,
                                     key=k + "dz_ret", help="반응 구역(케이크·충전재)의 수분특성·공극률·투수계수를 바꿉니다. 근거는 아래 표시")
        d["ret_frac"] = float(b3.number_input("수분 유지 재료 비율 (건조 고형분 w/w)", value=float(d.get("ret_frac") or 0.0),
                                              key=k + "dz_rf", format="%.3g"))
        R = DZ.RETAINERS[d["retainer"]]
        if d["retainer"] != "없음":
            st.caption(f"근거: {R['src']} — 라벨: {R['label']}")
        with st.popover("반응 구역 수리 성질 직접 입력(선택)") if hasattr(st, "popover") else st.expander("반응 구역 수리 성질 직접 입력(선택)"):
            st.caption("문헌 근거가 없는 배합이면 실험값을 넣으세요. 비우면 위 재료의 문헌·가정값을 씀.")
            for kk, lab, fmt in (("Ks_mult", "투수계수 배율 (-)", "%.4g"), ("phi_set", "공극률 (-)", "%.3g"),
                                 ("alpha_set", "van Genuchten α (1/m)", "%.4g"), ("n_set", "van Genuchten n (-)", "%.4g")):
                v_ = st.number_input(lab, value=d.get(kk), key=k + "dz_" + kk, format=fmt)
                d[kk] = float(v_) if v_ not in (None, "") else None
        sp = DZ.slurry_props(c.ratio_S_CaCO3, d)
        rd = ROLE_DEFAULTS["반응재(고체)"]
        sand = SOIL_TEXTURES["모래(Sand)"]
        base = (dict(Ks=sand["Ks"], phi=c.packed_porosity, alpha=sand["alpha"], n=sand["n"], thr=sand["theta_r"]) if slurry else
                dict(Ks=rd["K"], phi=rd["phi"], alpha=rd["alpha"], n=rd["n"], thr=min(0.03, 0.1 * rd["phi"])))
        zh = DZ.zone_hydraulics(base, d, SOIL_TEXTURES["식토(Clay)"])
        rows_ = [{"항목": "황 투입량", "값": f"{d['S_dose_g']:.1f} g"},
                 {"항목": "고형분 조성(부피) 황 / 석회석 / 수분 유지 재료", "값": f"{sp['vS']:.2f} / {sp['vC']:.2f} / {sp['vR']:.3f}"}]
        if slurry:
            rows_ += [{"항목": "슬러리 부피", "값": f"{d['S_dose_g'] * sp['L_per_gS']:.2f} L"},
                      {"항목": "고형분비(v/v)", "값": f"{sp['fs_vv']:.3f}"},
                      {"항목": "슬러리 내 황 질량분율 / 물 질량분율", "값": f"{sp['S_mass_frac']:.3f} / {sp['water_mass_frac']:.3f}"}]
        rows_ += [{"항목": "반응 구역 투수계수 배율", "값": f"{zh['Ks_mult']:.3g}"},
                  {"항목": "반응 구역 공극률", "값": f"{zh['phi']:.3f}"},
                  {"항목": "반응 구역 수분특성 α / n", "값": f"{zh['alpha']:.3g} 1/m / {zh['n']:.3g}"}]
        st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch")
        for nt in zh["notes"]:
            st.caption("· " + nt)
        st.caption("물만 더한 경우와 수분 유지 재료를 섞은 경우의 차이: 물은 주입 부피(처음 젖는 정도)만 바꾸고 흙의 수분특성대로 빠지며, "
                   "수분 유지 재료는 반응 구역의 수분특성(α, n)·공극률·투수계수 자체를 바꿔 물을 붙잡습니다.")


def validate_common(c: Common):
    errs, warns = list(validate_moist(c)), []
    def E(cond, msg):
        if cond:
            errs.append(msg)
    lm_ = str(getattr(c, "lambda_mode", "")).startswith("여과")
    E(lm_ and c.attach_alpha is None, "여과 이론 λ: 부착 효율 α를 입력하세요(근거 있는 기본값 없음).")
    E(lm_ and c.filt_U_m_s is None, "여과 이론 λ: 여과 계산 다르시 유속(m/s)을 입력하세요.")
    def W(cond, msg):
        if cond:
            warns.append(msg)
    E(c.Lx <= 0 or c.Ly <= 0 or c.H_soil <= 0, "흙수조 크기는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.H_aq < 0, "지하수층 두께가 음수입니다 — 이 값은 계산이 불가능합니다.")
    E(min(c.nx, c.ny, c.nz) < 3, "격자 수는 각 방향 3 이상이어야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.q_mm_day < 0, "침투 플럭스가 음수입니다 — 이 값은 계산이 불가능합니다.")
    E(min(c.NO3_in_mgN, c.SO4_in_mg, c.Ca_in_mg, c.gw_NO3_mgN, c.soil_NO3_mgN) < 0, "농도에 음수가 있습니다 — 이 값은 계산이 불가능합니다.")
    E(c.days <= 0, "운영 기간은 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.ratio_S_CaCO3 <= 0, "황:CaCO₃ 비는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(not (0 < c.init_saturation <= 1), "초기 포화도는 0~1 사이여야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.flow_update_days <= 0, "흐름 재계산 주기는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.flow_dt_max_h <= 0 or c.flow_dt_min_s <= 0, "흐름 시간 간격은 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.chem_dt_h <= 0, "반응 계산 간격은 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.bottom_bc.startswith("누수") and (c.leak_K <= 0 or c.leak_thick < 0), "누수 하단의 배수층 K·두께가 계산할 수 없는 값입니다.")
    E(c.H_aq > 0 and c.H_aq < (c.H_soil + c.H_aq) / max(c.nz, 1) * 0.5,
      "지하수층이 격자 한 칸의 절반보다 얇습니다 — 지하수층을 표현할 수 없어 계산하지 않습니다(두께를 늘리거나 nz를 늘리세요).")
    for d0, mm in (c.rain_series or []):
        E(mm is None or d0 is None, "강우 시계열에 빈 값이 있습니다 — 이 값은 계산이 불가능합니다.")
    W(c.flow_mode == "준정상", "흐름이 '준정상(빠른 근사)'입니다. 강우 변동·저류 변화가 계단식으로만 반영됩니다.")
    W(c.chem_engine == "PHREEQC" and not PE.available(), "PHREEQC 엔진을 찾지 못해 내장 화학으로 계산합니다: " + PE.status().get("error", "")[:160])
    if c.wt_depth is not None:
        E(c.wt_depth < 0 or c.wt_depth > c.H_soil + c.H_aq, "지하수면 깊이가 흙수조 밖입니다 — 이 값은 계산이 불가능합니다.")
    if c.Ks_override is not None:
        E(c.Ks_override <= 0, "포화 투수계수는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    for L in c.layers:
        E(L.theta_s <= L.theta_r or L.n <= 1 or L.alpha <= 0 or L.Ks <= 0,
          f"층 {L.top}~{L.bottom} m: θs>θr, n>1, α>0, Ks>0 이어야 합니다 — 이 값은 계산이 불가능합니다.")
    W(c.q_mm_day > 100, f"침투 {c.q_mm_day:g} mm/일은 매우 큰 값입니다(일반 관개 1~20 mm/일).")
    W(c.NO3_in_mgN > 300, f"유입 NO₃-N {c.NO3_in_mgN:g} mg/L는 비현실적으로 높습니다.")
    W(not (5 <= c.temperature <= 40), f"온도 {c.temperature:g}°C는 미생물 탈질 범위를 벗어납니다.")
    W(not (0.5 <= c.ratio_S_CaCO3 <= 6), f"황:CaCO₃ = {c.ratio_S_CaCO3:g}:1은 선행연구 범위(1:1~3:1)를 벗어납니다.")
    W(c.nx * c.ny * c.nz > 250000, f"격자 {c.nx * c.ny * c.nz:,}칸: 계산이 매우 오래 걸립니다.")
    W(c.days > 1500, "운영 기간이 깁니다: 계산이 오래 걸립니다.")
    W(c.init_saturation > 0.95, "초기 포화도가 거의 포화입니다.")
    W(max(c.Lx, c.Ly, c.H_soil) > 5, "흙수조가 5 m를 넘습니다(현장 규모).")
    for k, v in c.params.items():
        if k in REGISTRY and v is not None and v < 0 and k not in ("pCO2",):
            errs.append(f"고급 파라미터 {k} = {v}: 이 값은 계산이 불가능합니다(음수).")
    return errs, warns


def preview_fig(sc, c, annotations=None):
    s = scenario_obj(sc, c)
    g = make_grid(c)
    return viz.structure_preview(s.objects, s.roles, s.placement, g, wt_depth=c.wt_depth, annotations=annotations)


# ───────────────────────────── 머리말 ─────────────────────────────────────
import ui_style  # noqa: E402
ui_style.inject(st)


def trust_strip(R, all_res):
    """결과 화면 맨 위: 신뢰성 네 갈래를 따로 보여 준다(서로의 근거가 되지 않음)."""
    chip = ui_style.chip
    parts = []
    bad = [r["name"] for r in all_res if r.get("status") != "OK"]
    parts.append(("이번 계산 수렴", chip("FAILED " + ", ".join(bad), "fail") if bad else chip("모두 수렴", "ok")))
    mb = [r["name"] for r in all_res if r.get("mass_balance_bad")]
    parts.append(("질량수지", chip("허용오차 초과 " + ", ".join(mb), "warn") if mb else chip("허용오차 안", "ok")))
    try:
        with open(os.path.join(HERE, "selftest", "selfcheck_results.json"), encoding="utf-8") as fh:
            tb = json.load(fh)["table"]
        judged = [t for t in tb if t["판정"] in ("통과", "미통과")]
        npass = sum(t["판정"] == "통과" for t in judged)
        parts.append(("자체 점검", chip(f"{npass}/{len(judged)} 통과", "ok" if npass == len(judged) else "warn")))
    except Exception:      # noqa: BLE001
        parts.append(("자체 점검", chip("기록 없음 — ⑥ 탭에서 실행", "warn")))
    cal = ss().get("calib")
    parts.append(("실험 보정", chip("보정됨", "ok") if cal else chip("보정 자료 없음", "warn")))
    parts.append(("불확실성", chip("분석함", "ok") if ss().get("uq") else chip("아직 안 함", "warn")))
    cells = "".join(f'<div class="trust-cell"><span>{k}</span>{v}</div>' for k, v in parts)
    return f'<div class="trust" role="group" aria-label="결과 신뢰성 상태">{cells}</div>'

if IS_32BIT:
    st.warning("32비트 파이썬으로 실행 중입니다. 메모리를 약 2 GB까지만 쓸 수 있어 '보통' 이상 격자에서 메모리 오류가 날 수 있습니다. "
               "python.org에서 64비트(Windows installer 64-bit) 파이썬을 설치해 쓰세요.")
status_area = st.container()     # 실행 진행률(백그라운드 계산) — 아래에서 채운다
def _tab(name):
    """탭 화면 모듈(ui_tabs/<name>.py)을 이 파일의 전역 이름 공간에서 실행 — 분리 전과 같은 동작."""
    path = os.path.join(HERE, "ui_tabs", name + ".py")
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), globals())


tabs = st.tabs(["① 비교군 입력", "② 공통 조건", "③ 실행", "④ 결과·애니메이션", "⑤ 지표·메커니즘 설명",
                "⑥ 신뢰성(점검·보정·불확실성)", "⑦ 내보내기", "⑧ 조건 비교·최적화"])
if _stv < ST_MIN:
    st.warning(f"Streamlit {st.__version__} — 권장 {ST_MIN[0]}.{ST_MIN[1]} 이상. 일부 화면이 호환 모드로 표시됩니다.")
c: Common = ss().common

# ═════════════════════════════ ① 비교군 입력 ══════════════════════════════
with tabs[0]:
    _tab("tab1_input")

# ═════════════════════════════ ② 공통 조건 ════════════════════════════════
with tabs[1]:
    _tab("tab2_common")

# ═════════════════════════════ ③ 실행 ═════════════════════════════════════
def control_key(common: Common):
    """대조군 결과를 다시 쓸 수 있는지 판단하는 열쇠: 공통 조건 전체 + 코드 파일 해시."""
    import hashlib
    from engine import code_hashes
    blob = json.dumps({"c": common.to_dict(), "het": repr(getattr(common, "heterogeneity", None)),
                       "code": code_hashes()}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prepare_runs(common: Common, scen_list):
    """실행 가능한 비교군만 골라 Scenario 객체로(화면 쪽에서, 계산 시작 전에). → (runnable[(i, Scenario)], skipped[(이름, 이유)])"""
    runnable, skipped = [], []
    for i, sc in enumerate(scen_list):
        roles_of = {r["객체"]: r["역할"] for r in sc["role_df"].to_dict("records")} if len(sc["role_df"]) else {}
        e2, _ = ie.validate(sc["events_df"].to_dict("records"), roles_of)
        if not sc["objects"]:
            skipped.append((sc["name"], "3MF가 없습니다.")); continue
        if not ie.clean_rows(sc["events_df"].to_dict("records")):
            skipped.append((sc["name"], "설치 방법을 입력하세요(표가 비어 있어 실행하지 않음).")); continue
        if e2:
            skipped.append((sc["name"], "설치 방법을 입력하세요. " + " / ".join(e2))); continue
        so = copy.deepcopy(scenario_obj(sc, common))
        er = validate_reagent(so)
        if er:
            skipped.append((sc["name"], " / ".join(er))); continue
        runnable.append((i, so))
    return runnable, skipped


def run_all(common: Common, runnable, skipped, cached_ctrl=None, cores=1, progress=None, stop=None):
    """대조군 + 비교군 실행(백그라운드 스레드에서 호출 — 여기서는 st·세션 상태를 건드리지 않는다).
    중단은 stop 콜백으로 엔진 안까지 전달(A1). cached_ctrl: 같은 공통 조건으로 이미 계산한 대조군."""
    progress = progress or (lambda f, m: None)
    stop = stop or (lambda: False)
    out = {"control": None, "scen": [], "skipped": list(skipped), "common": common.to_dict(), "t": time.time()}
    t0 = time.time()
    progress(0.0, "대조군(구조물 없음) 계산 중…")
    if cached_ctrl is not None:
        progress(0.05, "대조군: 같은 조건으로 이미 계산한 결과를 다시 씁니다")
    if cores > 1 and len(runnable) >= 1:
        tasks = ([] if cached_ctrl is not None else [(common, None, None)]) + [(common, so, None) for _, so in runnable]
        res = studies.run_many(tasks, cores, lambda f, m: progress(min(0.99, f), f"병렬 계산 {m}"), stop)
        if stop():
            raise StopRequested()
        if cached_ctrl is None:
            ctrl, res = res[0], res[1:]
        else:
            ctrl = cached_ctrl
        if ctrl is None or ctrl.get("status") == "ERROR":
            raise RuntimeError("대조군 계산 실패: " + (ctrl or {}).get("fail_reason", "중단됨"))
        out["control"] = ctrl
        for (i, so), r in zip(runnable, res):
            if r is None:
                out["skipped"].append((so.name, "중단됨")); continue
            if r.get("status") == "ERROR":
                out["skipped"].append((so.name, r.get("fail_reason", ""))); continue
            r["color"] = viz.SCEN_COLORS[i % len(viz.SCEN_COLORS)]
            out["scen"].append(r)
    else:
        n_all = len(runnable) + (0 if cached_ctrl is not None else 1)
        w = 1.0 / max(1, n_all)

        def prog_ctrl(f, msg):
            progress(min(0.99, w * f), f"대조군 {msg}")
        ctrl = cached_ctrl if cached_ctrl is not None else run_scenario(common, None, prog_ctrl, stop)
        out["control"] = ctrl
        base = 0.0 if cached_ctrl is not None else w
        for j, (i, so) in enumerate(runnable):
            def prog(f, msg, j=j, nm=so.name):
                progress(min(0.99, base + w * (j + f)), f"{nm} ({j + 1}/{len(runnable)}): {msg}")
            try:
                r = run_scenario(common, so, prog, stop)
                r["color"] = viz.SCEN_COLORS[i % len(viz.SCEN_COLORS)]
                out["scen"].append(r)
            except StopRequested:
                raise
            except MemoryError:
                raise
            except ValueError as e:
                out["skipped"].append((so.name, str(e)))
            except Exception as e:
                out["skipped"].append((so.name, f"계산 오류: {e}\n{traceback.format_exc()[-800:]}"))
    bt = eval_start_from(ctrl, common.NO3_in_mgN) if ctrl and ctrl.get("status") == "OK" else None
    out["eval_start"] = bt
    # F9-5: 수분 의존 모드 A·B·C를 나란히 계산하고 비교군 순위가 같은지 판정(선택한 모드는 이미 계산한 결과)
    if getattr(common, "mode3", True) and common.kinetics.startswith("표면적") and out["scen"] and not stop():
        try:
            ok_objs = [so for _, so in runnable if so.name in {r["name"] for r in out["scen"]}]
            out["modes"] = studies.mode_compare(common, ok_objs, ctrl, out["scen"], cores,
                                                lambda f, m: progress(0.9 + 0.09 * f, f"수분 모드 비교 {m}"), stop)
        except StopRequested:
            raise
        except Exception as e:      # noqa: BLE001
            out["modes"] = {"error": str(e)}
    out["metrics"] = [M.compute(r, ctrl, common, bt) for r in out["scen"]] + [M.compute(ctrl, None, common, bt)]
    out["failed"] = [r["name"] for r in out["scen"] + [ctrl] if r.get("status") != "OK"]
    out["elapsed_s"] = time.time() - t0
    progress(1.0, f"완료 · {time.time() - t0:.0f}초")
    return out


QUICK_DAYS = 90.0     # 가벼움 격자·기본 조건에서 대조군 돌파 ≈ 47일 → 그 뒤 40일 이상을 η 평가 구간으로


def quick_common(c: Common) -> Common:
    """빠른 확인 실행: 가벼운 격자 · 최대 30일 · 내장 화학 · 가벼운 모드(나머지 조건은 ②와 같음)."""
    q = copy.deepcopy(c)
    q.nx, q.ny, q.nz = GRID_PRESETS["가벼움"]
    q.grid_preset = "가벼움"
    q.days = min(float(c.days), float(ss().get("quick_days", QUICK_DAYS)))
    q.chem_engine = "내장"
    q.light = True
    q.n_snapshots = min(int(c.n_snapshots), 12)
    return q


def start_run(c: Common, cores: int, quick: bool = False):
    """실행 버튼: 화면에서 입력을 정리해 백그라운드 작업으로 넘긴다. 반환: 시작했으면 True."""
    if runjob.running(ss().sid):
        return False
    errs, _ = validate_common(c)
    if errs:
        st.error("공통 조건에 계산할 수 없는 값이 있어 실행하지 않습니다.")
        return False
    common = quick_common(c) if quick else copy.deepcopy(c)
    runnable, skipped = prepare_runs(common, ss().scen)
    # 새 계산 전에 지난 결과·내보내기 파일을 놓아 메모리를 비운다(두 실행 결과를 동시에 들고 있지 않게).
    ss().results = None
    for k_ in ("xlsx", "pngzip", "html_cmp", "html_inst"):
        ss().pop(k_, None)
    import gc
    gc.collect()
    n_cells = int(common.nx * common.ny * common.nz)
    n_tasks = 1 + len(runnable)
    cores_run = max(1, studies.safe_cores(n_tasks, n_cells, want=cores))
    ckey = control_key(common)
    cached = ss().ctrl_cache.get(ckey)
    ss().run_meta = {"ckey": ckey, "cores": cores, "cores_run": cores_run, "n_scen": len(runnable), "quick": quick,
                     "names": [so.name for _, so in runnable], "skipped_pre": list(skipped)}
    runjob.start(ss().sid, run_all, common, runnable, skipped, cached, cores_run, label="실행")
    return True


def collect_run():
    """끝난 백그라운드 작업의 결과를 세션으로 옮기고 화면에 띄울 알림을 만든다(매 화면 실행 맨 앞에서)."""
    j = runjob.get(ss().sid)
    if not j or not j["done"]:
        return
    runjob.pop(ss().sid)
    meta = ss().get("run_meta", {}) or {}
    msgs = []
    if j["error"] is None and j["result"] is not None:
        res = j["result"]
        res["quick"] = bool(meta.get("quick"))
        ss().results = res
        ctrl = res.get("control")
        if ctrl is not None and ctrl.get("status") == "OK" and meta.get("ckey"):
            ss().ctrl_cache = {meta["ckey"]: ctrl}          # 최근 조건 하나만 보관(메모리)
        if res["failed"]:
            msgs.append(("error", "FAILED — 수렴하지 못한 계산이 있습니다: " + ", ".join(res["failed"]) +
                         ". 해당 결과는 '비신뢰'이며 내보내기가 기본 차단됩니다. 시간 간격·격자를 줄여 다시 실행하세요."))
        else:
            msgs.append(("success", f"계산을 마쳤습니다: 대조군 + 비교군 {len(res['scen'])}개 · "
                                    f"{res.get('elapsed_s', 0) / 60:.1f}분. ④ 결과·애니메이션 탭을 보세요."))
        for nme, why in res["skipped"]:
            msgs.append(("error", f"'{nme}' 실행 안 함: {why}"))
    else:
        et = j["error_type"]
        if et == "StopRequested":
            msgs.append(("warning", "계산을 중단했습니다(결과는 저장하지 않음)."))
        elif et == "MemoryError":
            msgs.append(("error", "메모리가 부족해 계산을 멈췄습니다(결과는 저장하지 않음). 이렇게 해 보세요: "
                         "① ③의 '병렬 코어 수'를 1로, ② ②에서 격자를 '가벼움'으로, ③ 다른 프로그램·브라우저 탭을 닫고 "
                         "앱 창(검은 창)을 닫았다가 run_app.bat으로 다시 시작."
                         + (" 지금 파이썬이 32비트라 메모리를 약 2 GB까지만 쓸 수 있습니다 — 64비트 파이썬 설치를 권합니다."
                            if IS_32BIT else "")))
        elif et == "BrokenProcessPool":
            msgs.append(("error", "병렬 계산 프로세스가 비정상 종료했습니다(대개 메모리 부족). "
                                  "③의 '병렬 코어 수'를 1로 줄여 다시 실행하세요."))
        else:
            msgs.append(("error", f"계산 중 오류: {j['error']}\n\n{j['trace']}"))
    ss().run_msgs = msgs


def _show_msgs():
    for kind, m in ss().get("run_msgs", []) or []:
        getattr(st, kind)(m)


def _job_status_body():
    """진행 중이면 진행률·경과 시간·중단 버튼. 끝나면 화면 전체를 다시 그려 결과를 띄운다."""
    j = runjob.get(ss().sid)
    if not j:
        return
    if j["done"]:
        st.rerun()
        return
    el = time.time() - j["t0"]
    eta = ""
    if j["frac"] > 0.03:
        rem = el * (1 - j["frac"]) / j["frac"]
        eta = f" · 남은 시간 약 {rem / 60:.0f}분" if rem >= 90 else f" · 남은 시간 약 {rem:.0f}초"
    meta = ss().get("run_meta", {}) or {}
    st.progress(j["frac"], text=f"⏳ 계산 중 ({el / 60:.1f}분 경과{eta}) — {j['msg']}")
    a1, a2 = st.columns([4, 1])
    a1.caption(f"대조군 + 비교군 {meta.get('n_scen', '?')}개({', '.join(meta.get('names', []))}) · 병렬 {meta.get('cores_run', 1)}개. "
               "계산은 백그라운드에서 계속되니 다른 탭을 보거나 버튼을 눌러도 끊기지 않습니다. 끝나면 결과가 자동으로 나타납니다.")
    if a2.button("⏹ 중단", key="job_stop", width="stretch"):
        runjob.request_stop(ss().sid)
        st.toast("중단 요청을 보냈습니다. 진행 중인 단계가 끝나면 멈춥니다.") if hasattr(st, "toast") else None


_frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
job_status = _frag(run_every=2)(_job_status_body) if _frag else _job_status_body


collect_run()
with status_area:
    job_status()

with tabs[2]:
    _tab("tab3_run")

# ═════════════════════════════ ④ 결과·애니메이션 ══════════════════════════
@st.cache_data(show_spinner="기본 대조군 계산 중(꽂지 않았을 때)…", max_entries=4)
def baseline_control(cdict):
    cc = Common.from_dict(cdict)
    cc.nx, cc.ny, cc.nz = GRID_PRESETS["가벼움"]
    cc.light = True
    cc.flow_mode = "준정상"          # 미리보기는 빠른 근사(실행 결과의 대조군은 ②의 흐름 방식 그대로)
    cc.chem_engine = "내장"
    cc.n_snapshots = min(cc.n_snapshots, 12)
    r = run_scenario(cc, None)
    bt = eval_start_from(r, cc.NO3_in_mgN)
    return r, M.compute(r, None, cc, bt), bt


with tabs[3]:
    _tab("tab4_results")

# ═════════════════════════════ ⑤ 지표·메커니즘 설명 ═══════════════════════
with tabs[4]:
    _tab("tab5_docs")

# ═════════════════════════════ ⑥ 신뢰성 ════════════════════════════════════
def _scen_choice(label, key):
    opts = ["대조군(구조물 없음)"] + [s_["name"] for s_ in ss().scen]
    nm = st.selectbox(label, opts, key=key)
    if nm == opts[0]:
        return None, nm
    s_ = next(x for x in ss().scen if x["name"] == nm)
    so = scenario_obj(s_, c)
    errs_ = validate_reagent(so) + (build_geometry(so, c, make_grid(c))["errors"] if so.objects else ["3MF 없음"])
    if errs_:
        st.error(" / ".join(errs_))
        return "ERR", nm
    return so, nm


with tabs[5]:
    _tab("tab6_reliability")

# ═════════════════════════════ ⑦ 내보내기 ═════════════════════════════════
with tabs[6]:
    _tab("tab7_export")


# ═════════════════════════════ ⑧ 조건 비교·최적화 ═════════════════════════════
def _sweep_key():
    return ss().sid + ":sweep"


def _sweep_status_body():
    j = runjob.get(_sweep_key())
    if not j:
        return
    if j["done"]:
        st.rerun()
        return
    el = time.time() - j["t0"]
    st.progress(j["frac"], text=f"⏳ 조건 비교 계산 중 ({el / 60:.1f}분) — {j['msg']}")
    if st.button("⏹ 조건 비교 중단", key="sw_stop"):
        runjob.request_stop(_sweep_key())


sweep_status = _frag(run_every=2)(_sweep_status_body) if _frag else _sweep_status_body

with tabs[7]:
    _tab("tab8_sweep")

# 프래그먼트가 없는 옛 Streamlit: 계산 중이면 2초마다 화면을 다시 그려 진행률·결과를 갱신
if _frag is None and (runjob.running(ss().sid) or runjob.running(_sweep_key())):
    time.sleep(2.0)
    st.rerun()
