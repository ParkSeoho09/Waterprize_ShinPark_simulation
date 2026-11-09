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
                    build_geometry, control_breakthrough_day, eval_start_from, planned_masses, run_scenario,
                    validate_reagent)
import phreeqc_engine as PE  # noqa: E402
import studies  # noqa: E402
import design as DZ  # noqa: E402
import runjob  # noqa: E402
from io_3mf import UNIT_LABEL, UNIT_TO_M, load_mesh_file, rescale_objects  # noqa: E402
from params import (GRID_PRESETS, REGISTRY, ROLE_DEFAULTS, ROLES, SOIL_SOURCE, SOIL_TEXTURES,  # noqa: E402
                    Common, SoilLayer, suggest_role)
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
    "케이슨형 (caisson_gate)": ("samples/caisson_gate_scale0.05.3mf",
        "강철 케이슨 안 내부 챔버에 황:석회석 충전층을 미리 채워 넣는다. 설치 후 반응재를 꺼내거나 다시 채울 수 없다(재보충 불가).",
        "1) 반응재를 채운 케이슨을 흙수조 중앙에 30분 동안 삽입한다. 2) 효율이 50% 미만이면 재보충을 시도한다(구조상 불가).",
        "(a) 고체 사전 배치",
        [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
         dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "트리거 η(%)": 50.0, "간격(일)": 7.0,
                                       "최대 회차": 5.0, "소요(분)": 20.0})]),
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


def role_df_for(objects, old=None):
    rows = []
    oldmap = {} if old is None else {r["객체"]: r for r in old.to_dict("records")}
    for o in objects:
        if o.name in oldmap:
            rows.append(oldmap[o.name])
            continue
        role = suggest_role(o.name)
        d = ROLE_DEFAULTS[role]
        rows.append({"객체": o.name, "역할": role,
                     "K (m/s)": d["K"], "공극률": d["phi"], "반응": bool(d["reactive"]),
                     "우선순위": d["prio"], "교체 가능": False, "얇은 객체": "자동",
                     "벽 두께(mm)": round(o.min_thickness_m * 1000, 3) if np.isfinite(o.min_thickness_m) else None,
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
    return out


def new_scenario(name="비교군"):
    return {"name": name, "mechanism": "", "install_text": "", "file_name": "", "file_bytes": None,
            "unit_src": "millimeter", "unit_sel": "millimeter", "objects": [],
            "title": "", "description": "", "role_df": pd.DataFrame(),
            "reagent": {"form": None, "material": "선택 안 함"}, "file_hash": "",
            "placement": dict(x=None, y=None, top_depth=0.0, rot_deg=0.0, tilt_deg=0.0, scale=1.0),
            "events_df": pd.DataFrame([ie.empty_row("구조물 삽입")]), "allow_refill_wo_opening": False,
            "mass_correction": True, "geom_report": None}


def load_file_into(sc, data, fname):
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
    sc["role_df"] = role_df_for(d["objects"])
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
                               (round(S_now, 1) if S_now else 400.0), min_value=0.1, key=k + "dz_dose",
                               help="독립변수. 비우면 지금 설치 표·충전재의 황 양" + (f"(약 {S_now:.0f} g)" if S_now else ""))
        d["S_dose_g"] = float(dose)
        d["S_d_mm"] = float(a2.number_input("황 입경 (mm)", value=float(d.get("S_d_mm") or (0.2 if slurry else 4.0)),
                                            min_value=0.001, key=k + "dz_d", format="%.3g",
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
                                                     min_value=0.0, key=k + "dz_w", format="%.3g",
                                                     help="물만 늘리면 주입 부피와 처음 젖는 정도가 커지지만, 물은 흙의 수분특성대로 빠집니다"))
        else:
            b2, b3 = st.columns(2)
        rlist = list(DZ.RETAINERS.keys())
        d["retainer"] = b2.selectbox("수분 유지 재료", rlist, index=rlist.index(d.get("retainer", "없음")) if d.get("retainer") in rlist else 0,
                                     key=k + "dz_ret", help="반응 구역(케이크·충전재)의 수분특성·공극률·투수계수를 바꿉니다. 근거는 아래 표시")
        d["ret_frac"] = float(b3.number_input("수분 유지 재료 비율 (건조 고형분 w/w)", value=float(d.get("ret_frac") or 0.0),
                                              min_value=0.0, max_value=0.9, key=k + "dz_rf", format="%.3g"))
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
                   "수분 유지 재료는 반응 구역의 수분특성(α, n)·공극률·투수계수 자체를 바꿔 물을 붙잡습니다. "
                   "황 입자의 소수성·분산제 효과와 재료의 팽윤은 계산하지 않습니다(정량 문헌 없음).")


def validate_common(c: Common):
    errs, warns = [], []
    def E(cond, msg):
        if cond:
            errs.append(msg)
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
    W(not c.o2_stage, "O₂ 추적을 끈 상태입니다: 공기 중 O₂에 의한 황 소모와 탈질 억제가 빠져 탈질이 실제보다 좋게 나올 수 있습니다(결과 화면에도 표시).")
    W(c.denit_moisture.startswith("끔"), "불포화 탈질 수분 함수를 껐습니다: 포화 칼럼에서 잰 속도를 마른 흙에도 그대로 써 낙관적인 상한입니다.")
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
        npass = sum(t["판정"] == "통과" for t in tb)
        parts.append(("자체 점검", chip(f"{npass}/{len(tb)} 통과", "ok" if npass == len(tb) else "warn")))
    except Exception:      # noqa: BLE001
        parts.append(("자체 점검", chip("기록 없음 — ⑥ 탭에서 실행", "warn")))
    cal = ss().get("calib")
    parts.append(("실험 보정", chip("보정됨", "ok") if cal else chip("미검증 — 실험 데이터 없음", "warn")))
    parts.append(("불확실성", chip("분석함", "ok") if ss().get("uq") else chip("아직 안 함", "warn")))
    cells = "".join(f'<div class="trust-cell"><span>{k}</span>{v}</div>' for k, v in parts)
    return f'<div class="trust" role="group" aria-label="결과 신뢰성 상태">{cells}</div>'

if IS_32BIT:
    st.warning("32비트 파이썬으로 실행 중입니다. 메모리를 약 2 GB까지만 쓸 수 있어 '보통' 이상 격자에서 메모리 오류가 날 수 있습니다. "
               "python.org에서 64비트(Windows installer 64-bit) 파이썬을 설치해 쓰세요.")
status_area = st.container()     # 실행 진행률(백그라운드 계산) — 아래에서 채운다
tabs = st.tabs(["① 비교군 입력", "② 공통 조건", "③ 실행", "④ 결과·애니메이션", "⑤ 지표·메커니즘 설명",
                "⑥ 신뢰성(점검·보정·불확실성)", "⑦ 내보내기", "⑧ 조건 비교·최적화"])
if _stv < ST_MIN:
    st.warning(f"Streamlit {st.__version__} — 권장 {ST_MIN[0]}.{ST_MIN[1]} 이상. 일부 화면이 호환 모드로 표시됩니다.")
c: Common = ss().common

# ═════════════════════════════ ① 비교군 입력 ══════════════════════════════
with tabs[0]:
    top = st.columns([3, 2, 2, 2])
    with top[0]:
        names = [s["name"] for s in ss().scen]
        if names:
            ss().cur = min(ss().cur, len(names) - 1)
            ss().cur = st.radio("비교군 선택", range(len(names)), format_func=lambda i: names[i],
                                index=ss().cur, horizontal=True, key="scen_radio")
        else:
            st.info("비교군이 없습니다. 오른쪽에서 추가하거나 예시를 불러오세요. 비교군이 없어도 ④ 결과 탭에서 기본 대조군(꽂지 않았을 때)은 볼 수 있습니다.")
    with top[1]:
        if st.button("➕ 빈 비교군 추가", width="stretch"):
            ss().scen.append(new_scenario(f"비교군 {len(ss().scen) + 1}"))
            ss().cur = len(ss().scen) - 1
            st.rerun()
    with top[2]:
        ALL_EX = "예시 전부 (영구 주입공 + 케이슨형)"
        ex = st.selectbox("예시 불러오기", ["—", ALL_EX] + list(SAMPLES.keys()), key="ex_sel",
                          help="고른 뒤 아래 '예시 추가'를 눌러야 비교군 목록에 들어갑니다.")
        if ex != "—" and st.button("예시 추가", width="stretch", type="primary"):
            have = {s_["name"] for s_ in ss().scen}
            for exk in (list(SAMPLES.keys()) if ex == ALL_EX else [ex]):
                nm_ = exk.split(" (")[0]
                if ex == ALL_EX and nm_ in have:
                    continue
                path, mech, text, form, rows = SAMPLES[exk]
                sc_new = new_scenario(nm_)
                with open(os.path.join(HERE, path), "rb") as fh:
                    load_file_into(sc_new, fh.read(), os.path.basename(path))
                sc_new["mechanism"] = mech
                sc_new["install_text"] = text
                sc_new["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                sc_new["reagent"] = {"form": form, "material": "선택 안 함"}
                ss().scen.append(sc_new)
            ss().cur = len(ss().scen) - 1
            st.rerun()
    with top[3]:
        if ss().scen:
            m1, m2, m3, m4 = st.columns(4)
            if m1.button("⧉", help="선택 비교군 복제", width="stretch"):
                src = ss().scen[ss().cur]
                dup = {k: (v.copy() if isinstance(v, (pd.DataFrame, dict)) else (list(v) if isinstance(v, list) else v))
                       for k, v in src.items()}
                dup["placement"] = dict(src["placement"]); dup["reagent"] = dict(src.get("reagent") or {})
                dup["name"] = src["name"] + " (복제)"
                ss().scen.insert(ss().cur + 1, dup)
                ss().cur += 1
                st.rerun()
            if m2.button("▲", help="위로", width="stretch") and ss().cur > 0:
                i = ss().cur; ss().scen[i - 1], ss().scen[i] = ss().scen[i], ss().scen[i - 1]; ss().cur -= 1
                st.rerun()
            if m3.button("▼", help="아래로", width="stretch") and ss().cur < len(ss().scen) - 1:
                i = ss().cur; ss().scen[i + 1], ss().scen[i] = ss().scen[i], ss().scen[i + 1]; ss().cur += 1
                st.rerun()
            if m4.button("🗑", help="선택 비교군 삭제", width="stretch"):
                ss().scen.pop(ss().cur)
                ss().cur = max(0, ss().cur - 1)
                st.rerun()

    if ss().scen:
        sc = ss().scen[ss().cur]
        k = f"s{id(sc)}"
        left, right = st.columns([1.05, 1])
        with left:
            sc["name"] = st.text_input("비교군 이름", sc["name"], key=k + "name")
            up = st.file_uploader("구조물 3MF (또는 STL)", type=["3mf", "stl"], key=k + "up")
            if up is not None and up.name != sc["file_name"]:
                try:
                    load_file_into(sc, up.getvalue(), up.name)
                    st.success(f"{up.name}: 객체 {len(sc['objects'])}개, 단위 {UNIT_LABEL.get(sc['unit_src'])}")
                except Exception as e:
                    st.error(f"3MF를 읽지 못했습니다: {e}")
            if sc["file_name"]:
                st.caption(f"파일: **{sc['file_name']}** · 3MF 단위 속성: {UNIT_LABEL.get(sc['unit_src'], sc['unit_src'])}"
                           + (f" · 제목: {sc['title']}" if sc["title"] else ""))
                if sc["description"]:
                    with st.expander("3MF에 적힌 설명"):
                        st.write(sc["description"])
            sc["mechanism"] = st.text_area("작용 메커니즘 설명 (결과·내보내기에 그대로 표시)", sc["mechanism"], height=90,
                                           key=k + "mech")
            u1, u2 = st.columns(2)
            with u1:
                cur_u = UNIT_FROM.get(sc["unit_sel"], "mm")
                newu = st.selectbox("좌표 단위", ["mm", "cm", "m"], index=["mm", "cm", "m"].index(cur_u),
                                    help="기본은 3MF의 unit 속성입니다. 파일 단위가 틀렸을 때만 바꾸세요. 자동 축소는 하지 않습니다.",
                                    key=k + "unit")
                if UNIT_KEYS[newu] != sc["unit_sel"] and sc["objects"]:
                    sc["objects"] = rescale_objects(sc["objects"], sc["unit_sel"], UNIT_KEYS[newu])
                    sc["unit_sel"] = UNIT_KEYS[newu]
                    sc["geom_report"] = None
            with u2:
                sc["placement"]["scale"] = num("배율(사용자가 정할 때만)", float(sc["placement"]["scale"]), k + "scale",
                                               help="1이면 원래 크기. 축소가 필요하면 직접 비율을 적으세요.")
            p1, p2, p3 = st.columns(3)
            with p1:
                sc["placement"]["x"] = num("위치 x (m)", sc["placement"]["x"] if sc["placement"]["x"] is not None else c.Lx / 2,
                                           k + "px", help="3MF 원점(보통 구조물 축)이 놓일 흙수조 x 좌표")
                sc["placement"]["rot_deg"] = num("회전 (°)", float(sc["placement"]["rot_deg"]), k + "rot")
            with p2:
                sc["placement"]["y"] = num("위치 y (m)", sc["placement"]["y"] if sc["placement"]["y"] is not None else c.Ly / 2,
                                           k + "py")
                sc["placement"]["tilt_deg"] = num("기울기 (°)", float(sc["placement"]["tilt_deg"]), k + "tilt")
            with p3:
                sc["placement"]["top_depth"] = num("3MF z=0 깊이 (m)", float(sc["placement"]["top_depth"]), k + "pd",
                                                   help="0이면 3MF의 z=0 면이 지표에 놓입니다.")
            if len(sc["role_df"]):
                st.markdown("**객체별 역할표** — 이름 키워드로 제안된 값입니다. 바꿀 수 있습니다.")
                cfg = {
                    "객체": st.column_config.TextColumn(disabled=True),
                    "역할": st.column_config.SelectboxColumn(options=ROLES, required=True),
                    "K (m/s)": st.column_config.NumberColumn(format="%.2e", help="빈칸 = 토양 값"),
                    "공극률": st.column_config.NumberColumn(format="%.3f", help="빈칸 = 토양 값"),
                    "반응": st.column_config.CheckboxColumn(help="반응재(고체)일 때 SLAD 반응 여부"),
                    "우선순위": st.column_config.NumberColumn(format="%d", help="겹칠 때 큰 값이 이깁니다"),
                    "교체 가능": st.column_config.CheckboxColumn(help="카트리지처럼 꺼내 새것으로 바꿀 수 있는 반응재"),
                    "얇은 객체": st.column_config.SelectboxColumn(options=["자동", "면 래스터화", "체적만"]),
                    "벽 두께(mm)": st.column_config.NumberColumn(format="%.3f", help="자동 추정 최소 두께. 실링 장벽 저항 계산에 씀"),
                    "체적(cm³)": st.column_config.NumberColumn(disabled=True, format="%.2f"),
                    "닫힘": st.column_config.CheckboxColumn(disabled=True, help="닫힌 메시인지(아니면 내부 판정 부정확)"),
                }
                sc["role_df"] = st.data_editor(sc["role_df"], column_config=cfg, hide_index=True, key=k + "roles",
                                               width="stretch")
                st.caption("역할 기본값: " + "; ".join(f"{r} K={ROLE_DEFAULTS[r]['K']} φ={ROLE_DEFAULTS[r]['phi']}" for r in ROLES[:6])
                           + " — 모두 '가정' 값입니다.")
                if st.button("복셀 검사(격자에 올려 보기)", key=k + "vox"):
                    g = make_grid(c)
                    try:
                        geom = build_geometry(scenario_obj(sc, c), c, g)
                        pm = planned_masses(scenario_obj(sc, c), c, g, geom)
                        sc["geom_report"] = {"counts": geom["comp"]["counts"], "warnings": geom["warnings"],
                                             "errors": geom["errors"], "masses": pm,
                                             "faces": {ov.name: ov.n_faces for ov in geom["ov"]}}
                    except Exception as e:
                        sc["geom_report"] = {"errors": [str(e)], "warnings": [], "counts": {}, "masses": {}, "faces": {}}
                rep = sc["geom_report"]
                if rep:
                    for e in rep["errors"]:
                        st.error(e)
                    df = pd.DataFrame([{"객체": n, "셀 수": v, "장벽 면 수": rep["faces"].get(n, 0)} for n, v in rep["counts"].items()])
                    st.dataframe(df, hide_index=True, width="stretch")
                    for w in rep["warnings"]:
                        st.warning(w)
                    if rep["masses"]:
                        m = rep["masses"]
                        st.caption(f"반응재 질량: 사전 배치 {m['pre_mesh_g']:.0f} g(메시 체적) / {m['pre_voxel_g']:.0f} g(복셀) · "
                                   f"설치 주입 계획 {m['injection_g']:.0f} g · 합계 {m['total_g']:.0f} g")
            sc["mass_correction"] = st.checkbox("반응재 질량을 메시 체적 기준으로 보정(복셀 오차 제거)", sc["mass_correction"], key=k + "mc")
            sc["allow_refill_wo_opening"] = st.checkbox("개구부 없이도 재보충 허용(가정 — 반응재 객체를 교체하는 것으로 봄)",
                                                        sc["allow_refill_wo_opening"], key=k + "arw")
        with right:
            if sc["objects"]:
                show_fig(preview_fig(sc, c), key=k + "prev")
            else:
                st.info("3MF를 올리면 객체별 색으로 3D 미리보기가 나타납니다.")

        st.markdown("---")
        st.subheader("시약 형태 — 필수")
        rg = sc.setdefault("reagent", {"form": None, "material": "선택 안 함"})
        r1, r2 = st.columns([1, 2])
        with r1:
            opts = ["(선택하세요)"] + REAGENT_FORMS
            cur_f = rg.get("form") if rg.get("form") in REAGENT_FORMS else "(선택하세요)"
            f_ = st.selectbox("시약 형태", opts, index=opts.index(cur_f), key=k + "rform",
                              help="(a) 미리 넣어 둔 고체 (b) 꺼내 바꾸는 고체 카트리지 (c) 입자 슬러리를 주입해 흙에 걸리게 함 "
                                   "(d) 녹아 있는 전자공여체 용액 (e) 콜로이드·나노 현탁액")
            rg["form"] = f_ if f_ in REAGENT_FORMS else None
        with r2:
            if rg.get("form", "") and rg["form"].startswith("(d)"):
                mats = LIQUID_MATERIALS
                rg["material"] = st.selectbox("액체 시약 물질", mats, index=mats.index(rg.get("material", "선택 안 함"))
                                              if rg.get("material") in mats else 0, key=k + "rmat")
                if rg["material"] == "티오황산염(S2O3²⁻)" and not rg.get("_thio_filled"):
                    rg.update({kk: vv for kk, vv in THIOSULFATE_STOICH.items()}); rg["_thio_filled"] = True
                st.caption("양론(NO₃ 1 mol당): 공여체 소비 yD, SO₄ 생성 ySO4, 산 생성 yH, 공여체 1 mol의 황 sD. "
                           "속도: Monod 이중 제한 k·[NO₃]/(KN+[NO₃])·[D]/(KD+[D]). **빈칸 = 미정 → 계산하지 않음**(값을 지어내지 않음). "
                           "티오황산염 양론은 문헌 참고값(5 S₂O₃²⁻ + 8 NO₃⁻ → 10 SO₄²⁻ + 4 N₂ + 2 H⁺), 속도는 미정.")
                q1, q2, q3 = st.columns(3)
                for col, keys in ((q1, ("yD", "ySO4", "yH")), (q2, ("sD", "k", "KN")), (q3, ("KD", "conc_molL", "Rf"))):
                    with col:
                        for kk in keys:
                            lab = {"yD": "yD (mol/mol)", "ySO4": "ySO4 (mol/mol)", "yH": "yH (eq/mol)", "sD": "sD (mol S/mol)",
                                   "k": "k (mol/m³/s)", "KN": "KN (mol/m³)", "KD": "KD (mol/m³)",
                                   "conc_molL": "주입 농도 (mol/L)", "Rf": "지연계수 R (흡착)"}[kk]
                            v = st.number_input(lab, value=rg.get(kk), key=k + "rg" + kk, format="%.4g")
                            rg[kk] = v
            elif rg.get("form", "") and rg["form"].startswith("(e)"):
                rg["lambda_colloid"] = st.number_input("콜로이드 포획 계수 λ (1/m)", value=rg.get("lambda_colloid"),
                                                       key=k + "rgl", format="%.4g", help="빈칸 = 미정(계산 안 함). 기본 가정값 50")
                rg["d_p"] = st.number_input("콜로이드 입경 (m)", value=rg.get("d_p"), key=k + "rgd", format="%.3e",
                                            help="빈칸 = 미정. 필터 케이크(막힘) 판정에도 씀")
            elif rg.get("form"):
                st.caption({"(a)": "반응재 역할 객체에 미리 채워진 황:석회석. 재보충은 개구부가 있을 때만 가능.",
                            "(b)": "역할표에서 '교체 가능'을 체크한 반응재 객체를 재보충 때 새것으로 바꿉니다(설치 표 방식 = 카트리지 교체).",
                            "(c)": "설치 표의 '시약 주입'으로 슬러리를 넣습니다. 입자는 여과 포획되고, 토양 입경에 비해 크면 개구부에 필터 케이크로 쌓입니다."
                            }.get(rg["form"][:3], ""))
        for e_ in validate_reagent(scenario_obj(sc, c)):
            st.error(e_)
        design_editor(sc, c, k)
        st.subheader("A1. 설치(꽂는) 방법 — 필수")
        st.caption("어떻게 꽂고, 어떻게 시약을 넣는지 적으세요. 시뮬레이터는 이 표대로 구조물을 넣고 시약을 주입하는 과정을 재현합니다.")
        a, b = st.columns([1, 1.4])
        with a:
            sc["install_text"] = st.text_area("설치 과정 설명(자유 텍스트)", sc["install_text"], height=180, key=k + "itxt",
                                              placeholder="예) 1) 케이싱을 30 cm 깊이까지 20분 동안 삽입한다. 2) 포트를 연다. "
                                                          "3) 고형분 20% 슬러리 2 L를 펌프로 0.2 L/min 주입한다. 4) 60분 정치. "
                                                          "5) 효율 50% 미만이면 카트리지 교체, 최대 8회, 15분.")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("텍스트 → 표 (규칙 기반)", key=k + "parse"):
                    rows = ie.parse_text(sc["install_text"], [o.name for o in sc["objects"]])
                    if rows:
                        sc["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                        st.success(f"{len(rows)}개 단계를 제안했습니다. 표에서 확인·수정하세요.")
                    else:
                        st.warning("텍스트에서 단계를 찾지 못했습니다. 표에 직접 입력하세요.")
            with b2:
                with st.popover("LLM으로 변환(선택)") if hasattr(st, "popover") else st.expander("LLM으로 변환(선택)"):
                    key_in = st.text_input("Anthropic API 키(저장 안 함)", type="password", key=k + "apikey")
                    model = st.text_input("모델", "claude-sonnet-5-5", key=k + "model")
                    if st.button("변환", key=k + "llm") and key_in:
                        try:
                            rows = ie.parse_text_llm(sc["install_text"], [o.name for o in sc["objects"]], key_in, model)
                            sc["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                            st.success(f"{len(rows)}개 단계를 받았습니다.")
                        except Exception as e:
                            st.error(f"LLM 변환 실패: {e} — 규칙 기반 변환이나 직접 입력을 쓰세요.")
        with b:
            ecfg = {
                "이벤트": st.column_config.SelectboxColumn(options=ie.EVENT_TYPES, required=True),
                "방식": st.column_config.SelectboxColumn(options=ie.METHODS),
                "시약 형태": st.column_config.SelectboxColumn(options=ie.REAGENT_FORM_OPTIONS,
                                                          help="시약 주입 행에서 비교군 기본 형태 대신 쓸 형태(비우면 비교군 형태)"),
                "대상 객체": st.column_config.TextColumn(help="쉼표로 여러 개, '전체' = 해당 역할 객체 모두"),
            }
            for col in ie.NUMERIC:
                ecfg[col] = st.column_config.NumberColumn(format="%.4g")
            df = sc["events_df"]
            for col in ie.COLUMNS:
                if col not in df.columns:
                    df[col] = None
            sc["events_df"] = st.data_editor(df[ie.COLUMNS], num_rows="dynamic", column_config=ecfg, hide_index=True,
                                             key=k + "events", width="stretch")
            roles_of = {r["객체"]: r["역할"] for r in sc["role_df"].to_dict("records")} if len(sc["role_df"]) else {}
            errs, warns = ie.validate(sc["events_df"].to_dict("records"), roles_of)
            for e in errs:
                st.error(e)
            for w in warns:
                st.warning(w)
            sched = ie.schedule(sc["events_df"].to_dict("records"))
            tr = [r for r in sched if r["이벤트"] == "재보충"]
            if sched:
                st.caption("타임라인: " + " → ".join(
                    f"{r['이벤트']}({r['t_start'] * 1440:.0f}분)" if r["t_start"] is not None else f"{r['이벤트']}(운영 중)"
                    for r in sched))
            # 질량 맞추기 도우미
            with st.expander("반응재 총 질량 맞추기 도우미"):
                ref = st.number_input("목표 반응재 질량 (g)", value=1265.0, key=k + "mtgt")
                fs = st.number_input("슬러리 고형분비 (v/v)", value=0.2, key=k + "mfs")
                r_ = c.ratio_S_CaCO3
                rho = (r_ * 2.07 + 2.71) / (r_ + 1)
                st.write(f"필요 슬러리 부피 ≈ **{ref / (max(fs, 1e-9) * rho * 1000):.3f} L** (고체 혼합 밀도 {rho:.2f} g/cm³)")

# ═════════════════════════════ ② 공통 조건 ════════════════════════════════
with tabs[1]:
    st.caption("아래 조건은 모든 비교군과 대조군에 똑같이 적용됩니다(지표 문서의 통제변수). 숫자 입력에는 최솟값·최댓값 제한이 없습니다.")
    H = TX.PARAM_HELP
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**흙수조·지하수층**")
        c.Lx = num("가로 Lx (m)", c.Lx, "Lx", H["Lx"])
        c.Ly = num("세로 Ly (m)", c.Ly, "Ly", H["Ly"])
        c.H_soil = num("토양층 깊이 (m)", c.H_soil, "Hs", H["H_soil"])
        c.H_aq = num("지하수층 두께 (m)", c.H_aq, "Ha", H["H_aq"])
        c.wt_depth = st.number_input("지하수면 깊이 (m, 비우면 토양층 바닥)", value=c.wt_depth, key="wt", help=H["wt_depth"],
                                     format="%.4g")
        c.gw_velocity = num("지하수 유속 (m/일)", c.gw_velocity, "gwv", H["gw_velocity"])
        bopts = ["지하수 수위", "누수(배수 저항)", "자유배수"]
        c.bottom_bc = st.selectbox("바닥 조건", bopts, index=bopts.index(c.bottom_bc) if c.bottom_bc in bopts else 0,
                                   help=H["bottom_bc"])
        if c.bottom_bc.startswith("누수"):
            c.leak_thick = num("배수층 두께 (m)", c.leak_thick, "lth", "바닥과 배수구 사이 층 두께. 저항 = 두께/K(가정)")
            c.leak_K = num("배수층 K (m/s)", c.leak_K, "lK", "작을수록 물이 잘 안 빠져 지하수면이 오르내림(주입·강우 시 수위 상승)")
        c.aq_texture = st.selectbox("지하수층 토성", list(SOIL_TEXTURES), index=list(SOIL_TEXTURES).index(c.aq_texture))
    with c2:
        st.markdown("**토양(불포화대)**")
        c.soil_mode = st.radio("입력 방식", ["간단", "상세"], index=["간단", "상세"].index(c.soil_mode), horizontal=True)
        c.texture = st.selectbox("토성", list(SOIL_TEXTURES), index=list(SOIL_TEXTURES).index(c.texture), help=H["texture"])
        tx = SOIL_TEXTURES[c.texture]
        c.Ks_override = st.number_input("포화 투수계수 Ks (m/s, 비우면 토성 값)", value=c.Ks_override, key="Ks",
                                        help=H["Ks"], format="%.3e")
        st.caption(f"토성 기본값: θr {tx['theta_r']}, θs {tx['theta_s']}, α {tx['alpha']:.1f} 1/m, n {tx['n']}, "
                   f"Ks {tx['Ks']:.2e} m/s — {SOIL_SOURCE}")
        c.init_saturation = num("초기 포화도 (0~1)", c.init_saturation, "isat", H["init_saturation"])
        if c.soil_mode == "상세":
            c.aniso = num("이방성 Kz/Kx", c.aniso, "aniso", H["aniso"])
            hc1, hc2, hc3 = st.columns(3)
            het = dict(c.heterogeneity or {})
            het["sigma_lnK"] = hc1.number_input("불균질 σ(lnK)", value=float(het.get("sigma_lnK", 0.0)), key="sig",
                                                help="0이면 균질. 토양 Ks에 로그정규 무작위장을 곱함(가정)")
            het["corr_len"] = hc2.number_input("상관 길이 (m)", value=float(het.get("corr_len", 0.05)), key="clen")
            het["seed"] = hc3.number_input("난수 시드", value=int(het.get("seed", 0)), key="seed", step=1)
            c.heterogeneity = het
            st.markdown("층상 구조(위에서부터, 빈 구간은 위 토성 사용)")
            ldf = pd.DataFrame([asdict(L) for L in c.layers]) if c.layers else pd.DataFrame(
                [dict(top=0.0, bottom=c.H_soil, theta_r=tx["theta_r"], theta_s=tx["theta_s"], alpha=tx["alpha"],
                      n=tx["n"], Ks=tx["Ks"], aniso=1.0)])
            ldf = st.data_editor(ldf, num_rows="dynamic", key="layers", width="stretch",
                                 column_config={"top": st.column_config.NumberColumn("상단(m)", help=None),
                                                "bottom": st.column_config.NumberColumn("하단(m)"),
                                                "theta_r": st.column_config.NumberColumn("θr", help=H["theta_r"]),
                                                "theta_s": st.column_config.NumberColumn("θs", help=H["theta_s"]),
                                                "alpha": st.column_config.NumberColumn("α (1/m)", help=H["alpha"]),
                                                "n": st.column_config.NumberColumn("n", help=H["n"]),
                                                "Ks": st.column_config.NumberColumn("Ks (m/s)", format="%.2e", help=H["Ks"]),
                                                "aniso": st.column_config.NumberColumn("Kz/Kx", help=H["aniso"])})
            c.layers = [SoilLayer(**{k: float(v) for k, v in r.items()}) for r in ldf.dropna().to_dict("records")]
        st.markdown("**유입수·초기 상태**")
        c.q_mm_day = num("침투 플럭스 q (mm/일)", c.q_mm_day, "q", H["q"])
        st.caption(f"= 흙수조 전체 {c.q_mm_day * c.Lx * c.Ly:.3f} L/일")
        with st.expander("강우·관개 시계열 / 증발산"):
            st.caption("계단형: 각 행의 '시작일'부터 다음 행까지 그 값(mm/일). 비우면 위의 q로 일정. CSV 열: 시작일, mm/일")
            upr = st.file_uploader("강우 CSV", type=["csv"], key="raincsv")
            if upr is not None:
                try:
                    rdf = pd.read_csv(upr)
                    c.rain_series = [[float(a), float(b)] for a, b in rdf.iloc[:, :2].values]
                except Exception as e:
                    st.error(f"CSV를 읽지 못했습니다: {e}")
            rdf = pd.DataFrame(c.rain_series or [], columns=["시작일", "mm/일"])
            rdf = st.data_editor(rdf, num_rows="dynamic", key="rain_ed", hide_index=True)
            c.rain_series = [[float(a), float(b)] for a, b in rdf.dropna().values] if len(rdf) else []
            c.et_mm_day = num("잠재 증발산 (mm/일)", c.et_mm_day, "et", "지표에서 빠지는 물. 용질은 남아 농축됩니다(가정)")
            c.h_crit = num("증발 한계 압력수두 (m)", c.h_crit, "hcrit", "지표 셀이 이보다 마르면 증발량이 줄어듭니다")
        c.NO3_in_mgN = num("유입 NO₃-N (mg/L)", c.NO3_in_mgN, "no3", H["NO3_in"])
        c.SO4_in_mg = num("유입 SO₄ (mg/L)", c.SO4_in_mg, "so4", H["SO4_in"])
        c.Ca_in_mg = num("유입 Ca (mg/L)", c.Ca_in_mg, "ca", H["Ca_in"])
        c.alk_in_mgCaCO3 = num("유입 알칼리도 (mg/L as CaCO₃)", c.alk_in_mgCaCO3, "alk", H["alk_in"])
        c.soil_pH0 = num("초기 토양 pH", c.soil_pH0, "ph0", H["soil_pH0"])
        c.soil_NO3_mgN = num("초기 토양수 NO₃-N (mg/L)", c.soil_NO3_mgN, "sno3")
        c.gw_NO3_mgN = num("지하수 배경 NO₃-N (mg/L)", c.gw_NO3_mgN, "gno3")
    with c3:
        st.markdown("**반응재(모든 비교군 동일)**")
        c.ratio_S_CaCO3 = num("황 : CaCO₃ 부피비 (x:1)", c.ratio_S_CaCO3, "ratio", H["ratio"])
        c.temperature = num("온도 (°C)", c.temperature, "temp", H["temperature"])
        st.markdown("**기간·계산**")
        c.days = num("운영 기간 (일)", c.days, "days", H["days"])
        pre = st.selectbox("격자 프리셋", list(GRID_PRESETS) + ["직접 입력"],
                           index=(list(GRID_PRESETS) + ["직접 입력"]).index(c.grid_preset), help=H["grid"])
        c.grid_preset = pre
        if pre != "직접 입력":
            c.nx, c.ny, c.nz = GRID_PRESETS[pre]
        g1, g2, g3 = st.columns(3)
        c.nx = int(g1.number_input("nx", value=int(c.nx), key="nx", step=1))
        c.ny = int(g2.number_input("ny", value=int(c.ny), key="ny", step=1))
        c.nz = int(g3.number_input("nz", value=int(c.nz), key="nz", step=1))
        g = make_grid(c)
        st.caption(f"셀 크기 {g.dx * 1000:.1f} × {g.dy * 1000:.1f} × {g.dz * 1000:.1f} mm · {g.N:,}칸")
        c.flow_mode = st.radio("흐름 해석", ["과도", "준정상"], index=["과도", "준정상"].index(c.flow_mode), horizontal=True,
                               help="과도(기본) = 매 단계 Richards 방정식. 준정상 = 빠른 근사(정해진 주기마다 정상해)")
        if c.flow_mode == "준정상":
            c.flow_update_days = num("흐름 재계산 주기 (일)", c.flow_update_days, "fud", H["flow_update_days"])
        else:
            f1, f2 = st.columns(2)
            c.flow_dt_max_h = f1.number_input("흐름 Δt 최대 (시간)", value=float(c.flow_dt_max_h), key="fdtm", format="%.3g")
            c.flow_dt_min_s = f2.number_input("Δt 하한 (초) — 넘어서도 수렴 못하면 FAILED", value=float(c.flow_dt_min_s),
                                              key="fdtn", format="%.3g")
        c.calcite_mode = st.radio("석회석 용해", ["속도론", "평형"], index=["속도론", "평형"].index(c.calcite_mode),
                                  horizontal=True, help=H["calcite_mode"])
        st.markdown("**화학 엔진**")
        pst = PE.status()
        c.chem_engine = st.radio("지화학 계산", ["PHREEQC", "내장"], index=["PHREEQC", "내장"].index(c.chem_engine),
                                 horizontal=True, help="PHREEQC(USGS) 라이브러리로 탄산계·이온쌍·석회석·석고와 SLAD 속도식을 계산. "
                                                       "내장 = 같은 식을 파이썬으로(빠름, 이온쌍 CaSO4°·CaCO3°·CaHCO3⁺ 포함)")
        st.caption(("PHREEQC 연결됨 " + pst.get("version", "") + f" · {os.path.basename(pst.get('lib', ''))}") if pst["ok"] == "예"
                   else "PHREEQC 없음 → 내장 화학 사용 (" + pst.get("error", "")[:120] + ")")
        x1, x2 = st.columns(2)
        c.chem_dt_h = x1.number_input("PHREEQC 계산 간격 (시간)", value=float(c.chem_dt_h), key="cdt", format="%.3g")
        c.splitting = x2.selectbox("연산자 분리", ["Lie", "Strang"], index=["Lie", "Strang"].index(c.splitting),
                                   help="Strang = 이송 ½ → 반응 → 이송 ½ (분리 오차가 작음, 느림)")
        c.pCO2_mode = st.radio("pCO₂", ["균일", "깊이별"], index=["균일", "깊이별"].index(c.pCO2_mode), horizontal=True,
                               help="깊이별 = 지표 값과 바닥 값 선형 보간(고급 파라미터 pCO2, pCO2_bottom). 모두 가정 값")
        st.markdown("**반응 속도식(v4, 문헌 기반)**")
        kin_opts = ["표면적(문헌, v4)", "부피(v3)"]
        c.kinetics = st.radio("SLAD 속도식", kin_opts, index=kin_opts.index(c.kinetics) if c.kinetics in kin_opts else 0,
                              horizontal=True,
                              help="표면적 = 체적 속도상수가 황 비표면적에 비례(Koenig & Liu 2001: 입경별 반-차수 상수). "
                                   "고운 슬러리 입자는 빠르고 굵은 입상 황은 느림. 부피(v3) = 예전 식(입경 무시, 비교용)")
        st.markdown("**수분 조건(독립변수)**")
        mo_opts = ["NEMIS(문헌)", "사용자 지정", "끔(포화 칼럼 속도)"]
        cur_mo = c.denit_moisture if c.denit_moisture in mo_opts else "NEMIS(문헌)"
        c.denit_moisture = st.radio("탈질 수분 함수 f(s) = ((s − s_t)/(1 − s_t))^w", mo_opts, index=mo_opts.index(cur_mo),
                                    horizontal=True,
                                    help="NEMIS = s_t 0.62, w 1.74 (Hénault & Germon 2000, 토양 종속영양 탈질). 60 % 문턱은 Grundmann & "
                                         "Rolston (1987). 불포화 황 충전층·황 슬러리에서 직접 잰 함수는 찾지 못했음 → 이 값은 "
                                         "'문헌 기반 기본 시나리오'이지 검증된 임계값이 아닙니다. 사용자 지정 = 실험·가정 값. "
                                         "끔 = 포화 칼럼에서 잰 속도를 그대로(낙관적 상한)")
        if c.denit_moisture == "사용자 지정":
            u1, u2 = st.columns(2)
            c.moist_thr = float(u1.number_input("문턱 포화도 s_t (0~1)", value=float(c.moist_thr), min_value=0.0, max_value=0.99,
                                                key="m_thr", format="%.3g"))
            c.moist_exp = float(u2.number_input("지수 w", value=float(c.moist_exp), min_value=0.0, key="m_exp", format="%.3g"))
        mm_opts = ["계산값(Richards)", "반응 구역 포화도 지정"]
        c.moist_mode = st.radio("반응 구역 수분", mm_opts, index=mm_opts.index(c.moist_mode) if c.moist_mode in mm_opts else 0,
                                horizontal=True,
                                help="계산값 = 흙·반응재 수분특성과 관개·지하수로 Richards가 계산한 포화도(수분 유지 재료 효과 포함). "
                                     "지정 = 반응 구역이 이 포화도로 유지된다고 보고 반응(탈질 수분 함수·호기성 황 산화)에 씀. "
                                     "물의 흐름·이송·공기 확산은 Richards 계산값 그대로(통제변수)")
        if c.moist_mode.startswith("반응 구역"):
            c.moist_set = float(st.slider("반응 구역 포화도 s (0~1)", 0.0, 1.0, float(c.moist_set), 0.01, key="m_set"))
        th_ = REGISTRY["wfps_thr"][0] if c.denit_moisture.startswith("NEMIS") else (c.moist_thr if c.denit_moisture == "사용자 지정" else 0.0)
        w_ = REGISTRY["wfps_exp"][0] if c.denit_moisture.startswith("NEMIS") else (c.moist_exp if c.denit_moisture == "사용자 지정" else 0.0)
        ss_ = np.linspace(0, 1, 101)
        ff_ = np.clip((ss_ - th_) / max(1 - th_, 1e-6), 0, 1) ** w_
        try:
            import plotly.graph_objects as go
            fig_ = go.Figure()
            for nm_, (t2, w2) in {"NEMIS (0.62, 1.74)": (0.62, 1.74), "문턱 0.5, 지수 1": (0.5, 1.0), "문턱 0.8, 지수 2": (0.8, 2.0)}.items():
                fig_.add_scatter(x=ss_, y=np.clip((ss_ - t2) / (1 - t2), 0, 1) ** w2, mode="lines", name=nm_,
                                 line=dict(dash="dot", width=1))
            fig_.add_scatter(x=ss_, y=ff_, mode="lines", name="지금 설정", line=dict(width=3))
            if c.moist_mode.startswith("반응 구역"):
                fig_.add_vline(x=c.moist_set, line_dash="dash")
            fig_.update_layout(height=230, margin=dict(l=40, r=10, t=10, b=30), xaxis_title="포화도 s", yaxis_title="탈질 속도 배율 f(s)",
                               legend=dict(orientation="h", y=-0.35))
            show_fig(fig_, key="moistcurve")
        except Exception:       # noqa: BLE001
            pass
        c.o2_stage = st.checkbox("O₂ 추적(호기성 황 산화·탈질 억제)", c.o2_stage,
                                 help="공기·유입수 O₂가 황을 산화(황 소모·산 생성, Watkinson & Blair 1993 표면 속도)하고 포화 셀에서 탈질 억제")
        c.o2_gas = st.checkbox("토양 공기 O₂ 확산(지표 대기와 교환)", c.o2_gas, disabled=not c.o2_stage,
                               help="Millington & Quirk (1961) 유효 확산. 끄면 처음 공극 공기의 O₂만 씀(v3 방식)")
        cm_opts = ["SLAD", "추적자(반응 없음)", "1차 감쇠"]
        c.chem_mode = st.selectbox("반응 모드", cm_opts, index=cm_opts.index(c.chem_mode),
                                   help="대체품 모드: 염료·추적자 실험(반응 없음) 또는 1차 감쇠로 바꿔 실험과 비교")
        if c.chem_mode == "1차 감쇠":
            c.k1_decay_day = num("1차 감쇠 상수 (1/일)", c.k1_decay_day, "k1d")
        st.markdown("**주입 계산**")
        c.inject_model = st.radio("주입 흐름", ["과도 Richards", "중첩 근사(빠름)"],
                                  index=["과도 Richards", "중첩 근사(빠름)"].index(c.inject_model), horizontal=True,
                                  help="과도 Richards = 원천항을 넣어 불포화 흐름을 그대로 풂(기본). 중첩 근사 = 빠르지만 낙관적")
        c.straining = st.checkbox("입자 크기로 필터 케이크 판정", c.straining,
                                  help="입자 지름/토양 d50 > 0.154 이면 입자가 흙에 못 들어가고 개구부에 쌓인다고 봄(문헌 기준, 토양 d50은 가정)")
        c.n_snapshots = int(num("저장할 3D 장면 수", float(c.n_snapshots), "nsnap"))
        c.light = st.toggle("가벼운 모드(프레임·해상도 축소)", c.light)
        c.redistribute_runoff = st.checkbox("지표 유출 재분배(가정, 기본 켬)", c.redistribute_runoff,
                                            help="구조물 덮개나 포화된 지표에서 못 들어간 물을 열린 흙 표면으로 고르게 보냅니다. "
                                                 "끄면 그 물(과 질산염)은 흙수조 밖으로 사라진 것으로 봅니다.")
        st.markdown("**삽입 교란(가정, 기본 없음)**")
        c.disturb_thick = num("교란대 두께 (m)", c.disturb_thick, "dth", H["disturb"])
        c.disturb_Kmult = num("교란대 K 배율", c.disturb_Kmult, "dkm", H["disturb"])
    with st.expander("고급 파라미터(반응·수송) — 출처/가정 라벨"):
        rows = [{"키": k, "값": c.params.get(k, v[0]), "단위": v[1], "라벨": TX.STATUS_LABELS.get(v[2], v[2]), "설명": v[3]}
                for k, v in REGISTRY.items()]
        pdf = st.data_editor(pd.DataFrame(rows), disabled=["키", "단위", "라벨", "설명"], hide_index=True, key="adv",
                             width="stretch", column_config={"값": st.column_config.NumberColumn(format="%.4g")})
        for r in pdf.to_dict("records"):
            if r["값"] is not None and r["값"] != REGISTRY[r["키"]][0]:
                c.params[r["키"]] = float(r["값"])
            elif r["키"] in c.params and r["값"] == REGISTRY[r["키"]][0]:
                c.params.pop(r["키"])
    errs, warns = validate_common(c)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)
    if ss().scen:
        st.markdown("**비교군별 반응재 총 질량(통제변수: 모두 같아야 함)**")
        rows = []
        for s_ in ss().scen:
            try:
                so = scenario_obj(s_, c)
                gm = build_geometry(so, c, g) if so.objects else None
                pm = planned_masses(so, c, g, gm) if gm else {"total_g": 0, "pre_mesh_g": 0, "injection_g": 0}
                rows.append({"비교군": s_["name"], "사전 배치(g)": round(pm["pre_mesh_g"], 1), "주입 계획(g)": round(pm["injection_g"], 1),
                             "합계(g)": round(pm["total_g"], 1)})
            except Exception as e:
                rows.append({"비교군": s_["name"], "사전 배치(g)": None, "주입 계획(g)": None, "합계(g)": None})
        mdf = pd.DataFrame(rows)
        st.dataframe(mdf, hide_index=True, width="stretch")
        tot = [r["합계(g)"] for r in rows if r["합계(g)"]]
        if len(tot) > 1 and (max(tot) - min(tot)) / max(max(tot), 1e-9) > 0.05:
            st.warning("비교군별 반응재 총 질량이 5% 넘게 다릅니다. 지표 문서: 질량이 다르면 '구조의 차이'가 아니라 '양의 차이'를 재게 됩니다. "
                       "① 탭의 '질량 맞추기 도우미'로 주입량을 조정하세요.")

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
    st.markdown("실행하면 **기본 대조군(반응재를 꽂지 않은 같은 흙수조)** 과 ①의 **비교군 전부**를 같은 조건·같은 격자로 계산합니다.")
    errs, warns = validate_common(c)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)
    plan = []
    for s_ in ss().scen:
        roles_of = {r["객체"]: r["역할"] for r in s_["role_df"].to_dict("records")} if len(s_["role_df"]) else {}
        e2, w2 = ie.validate(s_["events_df"].to_dict("records"), roles_of)
        status = "준비됨" if (s_["objects"] and not e2) else ("3MF 없음" if not s_["objects"] else "설치 방법을 입력하세요")
        plan.append({"비교군": s_["name"], "파일": s_["file_name"], "객체 수": len(s_["objects"]),
                     "설치 단계": len(ie.clean_rows(s_["events_df"].to_dict("records"))), "상태": status})
    if plan:
        st.dataframe(pd.DataFrame(plan), hide_index=True, width="stretch")
    else:
        st.warning("비교군이 없습니다. 지금 실행하면 대조군만 계산합니다. ① 탭에서 '예시 불러오기'로 예시를 고르고 **예시 추가**를 누르거나 3MF를 올리세요.")
    st.caption(f"흐름: {c.flow_mode} ({'Richards 정상해를 ' + str(c.flow_update_days) + '일마다·투수성 5% 변화 시 재계산 — 준정상 근사' if c.flow_mode == '준정상' else '매 단계 과도 Richards'}) · "
               f"석회석: {c.calcite_mode} · 격자 {c.nx}×{c.ny}×{c.nz}")
    is_running = runjob.running(ss().sid)
    st.caption(f"정식 실행 조건: {c.days:g}일 · 격자 {c.nx}×{c.ny}×{c.nz} · 흐름 {c.flow_mode} · 화학 {c.chem_engine}. "
               "기본값(365일·보통 격자·PHREEQC)은 컴퓨터에 따라 수십 분~1시간 넘게 걸립니다. "
               f"먼저 결과 모양을 보려면 **빠른 확인 실행**(가벼움 격자 · 최대 {QUICK_DAYS:g}일 · 내장 화학, 보통 몇 분)을 쓰세요.")
    r1, r2, r3 = st.columns([2, 1.4, 0.7])
    go_run = r1.button("▶ 실행 (대조군 + 비교군 전체)", type="primary", width="stretch",
                       disabled=bool(errs) or is_running, key="go_run")
    qd_opts = [90.0, 180.0]
    ss().quick_days = float(r2.selectbox("빠른 확인 기간", qd_opts, index=qd_opts.index(ss().get("quick_days", QUICK_DAYS))
                                         if ss().get("quick_days", QUICK_DAYS) in qd_opts else 0,
                                         format_func=lambda x: f"{x:.0f}일", key="qdays",
                                         help="180일이면 결과에서 90일·180일 지표를 나란히 비교합니다(② 운영 기간보다 길게는 안 됨)"))
    go_quick = r2.button(f"⚡ 빠른 확인 실행 ({ss().quick_days:.0f}일·가벼움)", width="stretch",
                         disabled=bool(errs) or is_running, key="go_quick")
    ncpu = os.cpu_count() or 1
    n_tasks = 1 + sum(p_["상태"] == "준비됨" for p_ in plan)
    n_cells = int(c.nx * c.ny * c.nz)
    avail_mb, total_mb = studies.mem_available_mb()
    # 기본값: 대조군·비교군을 동시에 계산(코어 하나는 화면용으로 남김, 최대 4) — 단, 남은 메모리 안에서만.
    cores = int(r3.number_input("병렬 코어 수", value=studies.safe_cores(n_tasks, n_cells), min_value=1,
                                max_value=max(1, ncpu), key="cores", step=1,
                                help=f"이 컴퓨터 {ncpu}코어"
                                     + (f", 지금 남은 메모리 약 {avail_mb / 1024:.1f} GB" if avail_mb else "")
                                     + f". 계산 하나가 약 {studies.run_mem_mb(n_cells):.0f} MB를 씁니다(현재 격자). "
                                       "2 이상이면 대조군·비교군을 동시에 계산해 더 빨리 끝남. "
                                       "1이면 하나씩 차례로 계산하고 메모리를 가장 적게 씁니다."))
    if avail_mb and avail_mb < 1.5 * studies.run_mem_mb(n_cells):
        st.warning(f"남은 메모리가 약 {avail_mb:.0f} MB로 적습니다(계산 하나에 약 {studies.run_mem_mb(n_cells):.0f} MB). "
                   "다른 프로그램·브라우저 탭을 닫거나, ②에서 격자를 '가벼움'으로 바꾸세요.")
    if (go_run or go_quick) and start_run(c, cores, quick=bool(go_quick and not go_run)):
        meta = ss().run_meta
        if meta["cores_run"] < cores:
            st.info(f"남은 메모리에 맞춰 병렬 코어를 {cores} → {meta['cores_run']}개로 줄여 계산합니다.")
        ss().run_msgs = []
        st.rerun()
    if runjob.running(ss().sid):
        st.info("계산 중입니다. 진행률은 화면 맨 위에 표시됩니다.")
    _show_msgs()
    if ss().results:
        R = ss().results
        st.caption(f"마지막 실행: 대조군 + {len(R['scen'])}개 비교군 · 대조군 돌파일(η 평가 시작) "
                   f"{R['eval_start'] if R['eval_start'] is not None else '없음(모사 기간 안에 도달 안 함)'}일")

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
    st.info("**모델 한계(항상 표시)** — " + " · ".join(f"{a}: {b}" for a, b in TX.LIMITS[:3]) +
            " … 전체는 ⑤ 탭. 🟡 가정 값은 실험으로 보정 전입니다.")
    R = ss().results
    if R is None and runjob.running(ss().sid):
        st.info("⏳ 대조군 + 비교군을 계산하고 있습니다(진행률은 화면 맨 위). 끝나면 이 탭에 비교군별 지표 표·설치 애니메이션·"
                "결과 3D·다중 비교·그래프가 자동으로 나타납니다. 그동안 다른 탭을 보거나 버튼을 눌러도 계산은 끊기지 않습니다.")
    elif R is None:
        if ss().scen:
            st.warning(f"**비교군 {len(ss().scen)}개({', '.join(s_['name'] for s_ in ss().scen)})는 아직 계산하지 않았습니다.** "
                       "비교군 결과·애니메이션은 실행을 해야 만들어집니다. 아래 버튼(또는 ③ 실행 탭의 ▶ 실행)을 누르세요.")
            b4a, b4b = st.columns(2)
            go4 = b4a.button("▶ 지금 실행 (②의 조건 그대로)", type="primary", key="go_run4", width="stretch",
                             disabled=bool(validate_common(c)[0]))
            go4q = b4b.button(f"⚡ 빠른 확인 실행 ({QUICK_DAYS:g}일·가벼움 격자·내장 화학)", key="go_quick4", width="stretch",
                              disabled=bool(validate_common(c)[0]))
            if go4 or go4q:
                n_cells_ = int(c.nx * c.ny * c.nz) if go4 else int(np.prod(GRID_PRESETS["가벼움"]))
                if start_run(c, studies.safe_cores(1 + len(ss().scen), n_cells_), quick=bool(go4q and not go4)):
                    ss().run_msgs = []
                    st.rerun()
        st.subheader("기본 대조군 미리보기 — 반응재를 꽂지 않았을 때의 피해")
        st.caption("실행 전에도 볼 수 있는 대조군 미리보기입니다(가벼운 격자, 현재 공통 조건, 빠른 근사 = 준정상). "
                   "비교군 결과가 아니며, 실행 결과의 대조군과는 격자·흐름 방식이 다릅니다.")
        ok_c = not validate_common(c)[0]
        if ok_c and st.toggle("기본 대조군 미리보기 보기", value=not ss().scen):
            cr, cm, bt = baseline_control(c.to_dict())
            k1, k2, k3 = st.columns(3)
            k1.metric("지하수 도달 NO₃-N 누적", f"{cm['N_to_gw_g']:.2f} g")
            k2.metric("먹는물 기준(10 mg/L) 도달", f"{cm['arrival_10mg_day']:.0f}일" if np.isfinite(cm['arrival_10mg_day']) else "모사 기간 내 없음")
            k3.metric("유출 농도 50% 도달(돌파)", f"{bt:.0f}일" if bt else "없음")
            show_fig(viz.baseline_damage(cr, c), key="base_dmg")
            ann = viz.param_annotations(c, None, cr, cr["grid"])
            show_fig(viz.results_scene(cr, c, ["NO₃ 오염 플룸", "지하수층 NO₃"], light=True, annotations=ann, height=560),
                     key="base_anim")
    else:
        C_ = Common.from_dict(R["common"])
        all_res = R["scen"] + [R["control"]]
        mets = R["metrics"]
        sc_by_name = {s_["name"]: s_ for s_ in ss().scen}
        st.warning("**모델 가정에 따른 비교 결과입니다.** 실험 데이터로 보정·검증하기 전의 정량값이며, 🟡 가정 값에 따라 달라집니다. "
                   "신뢰성은 ⑥ 탭의 자체 점검(방정식을 맞게 푸는가)·보정(실험을 재현하는가)·불확실성(얼마나 믿을 수 있는가)으로 따로 확인하세요.")
        if R.get("eval_start") is None and mets and mets[-1].get("eta_unavailable"):
            st.warning("**η(제거 효율)는 이번 결과에서 평가하지 않았습니다.** " + mets[-1]["eta_unavailable"])
        if R.get("quick"):
            st.info(f"⚡ **빠른 확인 결과**입니다(가벼움 격자 · {C_.days:g}일 · 내장 화학). 결과의 모양을 보는 용도이며, "
                    "보고서용 값은 ③에서 ▶ 실행(②의 조건 그대로)으로 다시 계산하세요.")
        st.markdown(trust_strip(R, all_res), unsafe_allow_html=True)
        if not C_.o2_stage:
            st.info("O₂ 추적이 꺼져 있습니다: 공기에 의한 황 소모·탈질 억제가 빠져 탈질이 좋게 나올 수 있습니다(② 탭에서 켤 수 있음).")
        if R.get("failed"):
            st.error("FAILED(수렴 실패): " + ", ".join(R["failed"]) + " — 이 결과는 비신뢰 결과입니다.")
        st.subheader("연구 핵심 지표 — 90일 · 180일 비교")
        st.caption("통제변수(②: 지하수 흐름·지층·유입 농도)는 모든 비교군이 같습니다. 장치 유입부·하류 농도는 운영 시작 때의 반응 구역 "
                   "경계를 드나든 물의 유량 가중 평균, 장치 제거율 = 반응 구역 안에서 탈질된 N ÷ 반응 구역으로 들어간 N. "
                   "우회 비율 = 1 − (반응재 통과 유량 ÷ 반응재가 없을 때 같은 자리를 지나는 유량), 같은 흙·경계 조건으로 계산.")
        try:
            hc, hr = M.core_table(R["scen"], R["control"], C_)
            st.dataframe(pd.DataFrame(hr, columns=hc), hide_index=True, width="stretch", height=min(760, 36 * (len(hr) + 1)))
            if C_.days < 180:
                st.caption(f"운영 기간이 {C_.days:g}일이라 180일 칸은 없습니다(② 운영 기간 또는 ③ 빠른 확인 기간을 180일로).")
        except Exception as e_:      # noqa: BLE001
            st.warning(f"핵심 지표 표를 만들지 못했습니다: {e_}")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            figc = make_subplots(rows=2, cols=2, subplot_titles=("장치 유입부 vs 하류 NO₃-N (mg/L)", "흙수조 바닥 유출 NO₃-N (mg/L)",
                                                                 "잔여 황 (g)", "반응 구역 포화도 s"))
            for r_ in R["scen"] + [R["control"]]:
                Dd = r_["daily"]
                col_ = r_.get("color", "#666")
                nm_ = r_["name"]
                if not r_.get("is_control"):
                    figc.add_scatter(x=Dd["day"], y=Dd.get("dev_Cin_mgN", []), name=f"{nm_} 유입부", line=dict(color=col_, dash="dot"),
                                     row=1, col=1)
                    figc.add_scatter(x=Dd["day"], y=Dd.get("dev_Cout_mgN", []), name=f"{nm_} 하류", line=dict(color=col_), row=1, col=1)
                    figc.add_scatter(x=Dd["day"], y=Dd.get("S_g", []), name=f"{nm_} 황", line=dict(color=col_), row=2, col=1,
                                     showlegend=False)
                    figc.add_scatter(x=Dd["day"], y=Dd.get("sat_reac", []), name=f"{nm_} s", line=dict(color=col_), row=2, col=2,
                                     showlegend=False)
                figc.add_scatter(x=Dd["day"], y=Dd["Cout_mgN"], name=f"{nm_} 바닥 유출", line=dict(color=col_ if not r_.get("is_control") else "#888"),
                                 row=1, col=2, showlegend=False)
            if C_.moist_mode.startswith("반응 구역"):
                figc.add_hline(y=C_.moist_set, line_dash="dash", row=2, col=2)
            figc.update_layout(height=560, margin=dict(l=40, r=10, t=40, b=30), legend=dict(orientation="h", y=-0.12))
            show_fig(figc, key="corecharts")
        except Exception as e_:      # noqa: BLE001
            st.caption(f"(시계열 그림 생략: {e_})")
        st.subheader("비교군 × 지표")
        head, rows = M.summary_table(mets)
        st.dataframe(pd.DataFrame(rows, columns=head), hide_index=True, width="stretch", height=min(900, 36 * (len(rows) + 1)))
        st.caption(f"η 평가 구간: 대조군 유출 농도가 유입의 90%에 도달한 {R['eval_start']}일 이후(도달 지연 착시 제외). '대조군 보정 제거율'은 같은 날 대조군 유출과 비교한 값. 재보충 전·후 구간 η는 아래 표.")
        with st.expander("재보충 전·후 구간별 η, 비교군 메커니즘 설명, 경고"):
            for r, m in zip(R["scen"], mets):
                st.markdown(f"**{r['name']}** — {r['mechanism'] or '(메커니즘 설명 없음)'}")
                st.dataframe(pd.DataFrame([{"구간 시작(일)": s_["from"], "구간 끝(일)": s_["to"], "η (%)": round(s_["eta"], 1)}
                                           for s_ in m["eta_segments"]]), hide_index=True)
                d = r.get("diag", {}) or {}
                if d.get("through_verdict"):
                    st.caption(f"반응재 통과 진단: {d['through_verdict']} (반응재 통과 유량 / 반응재 없을 때 같은 위치 유량 = "
                               f"{d.get('through_ratio', float('nan')):.3f})")
                if m.get("eval_note"):
                    st.warning(m["eval_note"])
                for w in r["warnings"][:12]:
                    st.caption("⚠ " + w)
        with st.expander("수렴·질량수지 보고서(매 실행 자동 점검)"):
            st.caption("질량수지가 맞는다는 것은 '계산이 물질을 잃지 않았다'는 뜻이지 물리적으로 옳다는 증거는 아닙니다.")
            rep_rows = []
            for r in all_res:
                sv = r.get("solver", {})
                mb = r.get("mass_balance", {})
                row_ = {"실행": r["name"], "상태": r.get("status"), "화학 엔진": r.get("chem_engine"),
                        "흐름 단계(채택/거부)": f"{int(sv.get('flow_accepted', 0))}/{int(sv.get('flow_rejected', 0))}",
                        "뉴턴 반복": int(sv.get("newton_iters", 0)), "Δt 축소": int(sv.get("dt_cuts", 0)),
                        "Δt 범위(초)": f"{sv.get('dt_min_s', float('nan')):.3g}~{sv.get('dt_max_s', float('nan')):.3g}",
                        "선형계 CG/BiCGSTAB/LGMRES/GMRES/직접/정칙화": "/".join(str(int(sv.get(k, 0))) for k in (
                            "lin_cg", "lin_bicgstab", "lin_lgmres", "lin_gmres", "lin_direct", "lin_regularized")),
                        "최대 상대잔차": f"{sv.get('lin_max_relres', 0):.1e}", "PHREEQC 호출": int(sv.get("phreeqc_calls", 0))}
                for k_, v_ in mb.items():
                    if isinstance(v_, (int, float)):
                        row_["수지 " + k_.replace("_err_rel", "")] = f"{v_:.1e}"
                row_["허용 초과"] = ", ".join(r.get("mass_balance_bad", {}).keys()) or "없음"
                rep_rows.append(row_)
            st.dataframe(pd.DataFrame(rep_rows), hide_index=True, width="stretch")
            for r in all_res:
                if r.get("mass_balance", {}).get("W_note"):
                    st.caption(f"{r['name']} 물 수지: {r['mass_balance']['W_note']}")
                if r.get("fail_reason"):
                    st.error(f"{r['name']}: {r['fail_reason']}")
        view = st.radio("보기", ["설치 과정 애니메이션", "결과 3D(레이어)", "다중 비교(시간 연동)", "그래프", "재보충 이벤트"],
                        horizontal=True)
        show_ann = st.toggle("파라미터 표시(구석 캡션·마우스 툴팁)", value=True)
        light = st.toggle("가벼운 모드", value=C_.light, key="light_view")
        if view == "설치 과정 애니메이션":
            nm = st.selectbox("비교군", [r["name"] for r in R["scen"]])
            r = next(x for x in R["scen"] if x["name"] == nm)
            so = scenario_obj(sc_by_name[nm], C_) if nm in sc_by_name else None
            if so is None:
                st.warning("이 결과의 비교군 입력이 지금 목록에 없어 구조물 모양을 그릴 수 없습니다.")
            else:
                from voxel import Placement as _P
                pl = _P(**r["placement"]) if r.get("placement") else so.placement
                ann = viz.param_annotations(C_, so, r, r["grid"]) if show_ann else None
                show_fig(viz.install_animation(r, so.objects, so.roles, pl, wt_depth=C_.wt_depth, light=light, annotations=ann),
                         key="inst")
                st.caption("단계: " + " → ".join(dict.fromkeys(f["label"].split(" (")[0] for f in r["frames"])))
        elif view == "결과 3D(레이어)":
            a1, a2 = st.columns([2, 1])
            nm = a2.selectbox("비교군/대조군", [r["name"] for r in all_res])
            r = next(x for x in all_res if x["name"] == nm)
            layers = a1.multiselect("레이어", list(viz.LAYER_DEF), default=["반응재 분포", "NO₃ 오염 플룸", "pH 단면", "지하수층 NO₃"])
            cut = a2.slider("단면 자르기(x 위치, 1 = 자르지 않음)", 0.05, 1.0, 1.0, 0.05)
            so = scenario_obj(sc_by_name[nm], C_) if nm in sc_by_name else None
            from voxel import Placement as _P
            pl = _P(**r["placement"]) if r.get("placement") else None
            ann = viz.param_annotations(C_, so, r, r["grid"]) if show_ann else None
            show_fig(viz.results_scene(r, C_, layers, objects=so.objects if so else None, roles=so.roles if so else None,
                                       placement=pl, cut_x=None if cut >= 0.999 else cut * C_.Lx, light=light,
                                       annotations=ann, wt_depth=C_.wt_depth), key="res3d")
        elif view == "다중 비교(시간 연동)":
            b1, b2, b3 = st.columns([2, 1, 1])
            layers = b1.multiselect("레이어", list(viz.LAYER_DEF), default=["반응재 분포", "NO₃ 오염 플룸"])
            metric = b2.selectbox("그래프", ["Cout_mgN", "eta", "SO4_out_mg", "pH_out", "N_to_gw_g"],
                                  format_func=lambda x: {"Cout_mgN": "유출 NO₃-N", "eta": "η", "SO4_out_mg": "유출 SO₄",
                                                         "pH_out": "유출 pH", "N_to_gw_g": "지하수 도달 누적"}[x])
            sync = b3.checkbox("카메라 동기화", value=True, help="한 화면을 돌리면 다른 화면도 같이 돕니다.")
            objs = {}
            for r in R["scen"]:
                if r["name"] in sc_by_name:
                    so = scenario_obj(sc_by_name[r["name"]], C_)
                    from voxel import Placement as _P
                    objs[r["name"]] = (so.objects, so.roles, _P(**r["placement"]))
            fig = viz.compare_view(all_res, C_, layers, metric=metric, light=light, scen_objects=objs)
            if sync:
                show_html(viz.to_html(fig, sync_cameras=True), height=860)
            else:
                show_fig(fig, key="cmp")
        elif view == "그래프":
            bt = R["eval_start"]
            show_fig(viz.timeseries(all_res, "eta", "제거효율 η (%)", 100, eval_start=bt), key="g_eta")
            show_fig(viz.timeseries(all_res, "Cout_mgN", "유출 NO₃-N (mg/L)", hlines=[(10, "먹는물 기준 10")], eval_start=bt), key="g_c")
            show_fig(viz.timeseries(all_res, "N_to_gw_g", "지하수 도달 NO₃-N 누적 (g)"), key="g_gw")
            g1, g2 = st.columns(2)
            with g1:
                show_fig(viz.timeseries(all_res, "pH_below", "pH (반응재 아래 0~10 cm)", hlines=[(C_.soil_pH0 - 0.5, "−0.5"), (C_.soil_pH0 + 0.5, "+0.5")]), key="g_ph")
                show_fig(viz.timeseries(all_res, "SO4_out_mg", "유출 SO₄ (mg/L)", hlines=[(250, "250 mg/L")]), key="g_so4")
            with g2:
                show_fig(viz.timeseries(all_res, "Kreac", "반응재 영역 K (m/s)"), key="g_k")
                show_fig(viz.timeseries(all_res, "Keff", "도메인 유효 K (m/s, 정수두 계산)"), key="g_keff")
            st.markdown("**깊이 프로파일(마지막 시점)**")
            p1, p2, p3 = st.columns(3)
            with p1:
                show_fig(viz.depth_profiles(all_res, mets[:-1] + [mets[-1]], "pH", "pH"), key="dp_ph")
            with p2:
                show_fig(viz.depth_profiles(all_res, mets, "NO3", "NO₃-N (mg/L)"), key="dp_n")
            with p3:
                show_fig(viz.depth_profiles(all_res, mets, "SO4", "SO₄ (mg/L)"), key="dp_s")
        else:
            for r, m in zip(R["scen"], mets):
                st.markdown(f"**{r['name']}** — {m['refill_note']}")
                if m["refill_table"]:
                    st.dataframe(pd.DataFrame([{"회차": x.get("n"), "운영 일": round(x["day"], 1), "방식": x.get("method"),
                                                "성공": x.get("ok"), "η 직전(%)": round((x.get("eta_before") or 0) * 100, 1),
                                                "R 24h(%)": x.get("R_24h"), "R 이동시간 보정(%, 근사)": x.get("R_travel"),
                                                "시작(분)": round(x.get("t_start_min", 0), 1), "끝(분)": round(x.get("t_end_min", 0) or 0, 1),
                                                "주입량(L)": round(x.get("delivered_L", 0) or 0, 2), "T_r(분)": x.get("duration_min"),
                                                "사유": x.get("why", "")} for x in m["refill_table"]]), hide_index=True)

# ═════════════════════════════ ⑤ 지표·메커니즘 설명 ═══════════════════════
with tabs[4]:
    st.subheader("평가지표 (지표 문서 정의)")
    st.caption("지표 이름을 누르면 자세한 설명이 펼쳐집니다.")
    for name, d in TX.METRICS.items():
        with st.expander(f"**{name}** — {d['one']}"):
            st.markdown(f"**무엇을 재나요?** {d['what']}")
            st.markdown(f"**읽는 법** {d['read']}")
            st.markdown(f"**좋은 값의 기준** {d['good']}")
            st.markdown(f"**이 시뮬레이션에서 계산하는 방법** {d['how']}")
            if st.toggle("계산 방법 보기", key="f_" + name):
                st.code(d["formula"], language=None)
    st.subheader("메커니즘 설명")
    for name, d in TX.MECHANISMS.items():
        with st.expander(f"**{name}** — {d['one']}"):
            st.markdown(d["detail"])
            if d["formula"] and st.toggle("계산 방법 보기", key="m_" + name):
                st.code(d["formula"], language=None)
    st.subheader("모델 한계")
    for a, b in TX.LIMITS:
        with st.expander(f"**{a}**"):
            st.markdown(b)

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
    st.caption("신뢰성은 세 가지를 구분해 봅니다: ① 자체 점검 = 방정식을 맞게 푸는가(해석해·보존 법칙), "
               "② 보정·검증 = 실험을 재현하는가, ③ 불확실성 = 가정 값이 바뀌면 결과가 얼마나 흔들리는가. "
               "외부 프로그램과 결과를 비교하는 검증은 하지 않습니다.")
    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["자체 점검", "격자·시간 독립성", "실험 보정", "사진 비교", "불확실성",
                                          "알려진 문제 처리", "엔진·환경"])
    with t1:
        sp = os.path.join(HERE, "selftest", "selfcheck_results.json")
        fast = st.checkbox("빠르게(E5 입자 시험 제외)", value=False, key="sc_fast")
        if st.button("🔎 자체 점검 실행", key="sc_run"):
            sys.path.insert(0, os.path.join(HERE, "selftest"))
            import selfcheck as SC
            pb = st.progress(0.0)
            with st.spinner("해석해·보존 시험 중…"):
                SC.run_all(include_slow=not fast, progress=lambda f, m: pb.progress(min(1.0, f), text=m))
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as fh:
                SCR = json.load(fh)
            tb_ = SCR["table"]
            np_ = sum(t_.get("판정") == "통과" for t_ in tb_)
            miss_ = [t_["id"] for t_ in tb_ if t_.get("판정") != "통과"]
            st.markdown(ui_style.chip(f"{np_}/{len(tb_)} 통과", "ok" if not miss_ else "warn") +
                        (f" &nbsp;미검증: {', '.join(miss_)}" if miss_ else ""), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(tb_), hide_index=True, width="stretch")
            st.caption(f"실행 {SCR.get('when', '')} · {SCR.get('sec', '')}초 · 통과 못 한 항목은 '미검증'")
            for k_, img in (SCR.get("figures") or {}).items():
                p_ = os.path.join(HERE, "selftest", img)
                if os.path.exists(p_):
                    st.image(p_, caption=k_)
    with t2:
        st.markdown("격자를 1.5배·2배로 세분하고 시간 간격을 절반으로 줄여 다시 풀어, 수두·유속(국부 최대)·NO₃ 유출·SO₄·최저 pH·K·"
                    "질량수지가 얼마나 바뀌는지 봅니다. 2배 세분 결과도 저장합니다. **오래 걸립니다**(2배 = 셀 8배).")
        so, nm = _scen_choice("대상", "ind_sc")
        a_, b_ = st.columns(2)
        dd = a_.number_input("이 점검의 운영 기간(일)", value=float(min(c.days, 30.0)), key="ind_days")
        cores_i = int(b_.number_input("병렬 코어", value=1, key="ind_cores", step=1))
        if so != "ERR" and st.button("격자·시간 독립성 점검 실행", key="ind_run"):
            cc = copy.deepcopy(c); cc.days = dd; cc.light = True
            pb = st.progress(0.0)
            ss().stop_flag = False
            ss().indep = studies.independence(cc, so, cores=max(1, cores_i), out_dir=os.path.join(HERE, "selftest"),
                                              progress=lambda f, m: pb.progress(min(1.0, f), text=m),
                                              stop=lambda: bool(ss().get("stop_flag")))
        I_ = ss().get("indep")
        if not I_:
            # 이번 세션에서 돌리지 않았으면, 함께 배포한(또는 이전에 저장한) 가장 최근 결과를 보여 준다
            import glob
            fs_ = sorted(glob.glob(os.path.join(HERE, "selftest", "independence_*.json")))
            if fs_:
                try:
                    with open(fs_[-1], encoding="utf-8") as fh:
                        I_ = json.load(fh)
                    st.info(f"저장된 결과를 보여 줍니다: {os.path.basename(fs_[-1])} "
                            f"(대상 {I_.get('scenario', '?')}, {I_.get('when', '')}). 다시 돌리면 이번 결과로 바뀝니다.")
                except Exception:      # noqa: BLE001
                    I_ = None
        if I_:
            st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k != "NO3_profile"} for t in I_["table"]]),
                         hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(I_["verdict"]).T, width="stretch")
            st.download_button("결과 저장(.json, 2배 세분 포함)", json.dumps(I_, ensure_ascii=False, default=float),
                               file_name="격자시간독립성.json", key="ind_dl")
        import glob as _glob
        gs_ = sorted(_glob.glob(os.path.join(HERE, "selftest", "grid_study_superposition_*.json")))
        if gs_:
            with st.expander("함께 배포한 점검: 주입 모델 '중첩 근사(빠름)'으로 ×1·×1.5·×2 모두 완료한 결과"):
                try:
                    with open(gs_[-1], encoding="utf-8") as fh:
                        G_ = json.load(fh)
                    st.caption("기본 주입 모델(과도 Richards)은 ×2 격자 주입 사례가 수렴하지 않아, 주입만 중첩 근사로 바꿔 "
                               "흐름·이송·반응 부분의 격자 의존성을 끝까지 비교한 것입니다(README '격자·시간 독립성').")
                    st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k != "NO3_profile"} for t in G_["table"]]),
                                 hide_index=True, width="stretch")
                    st.dataframe(pd.DataFrame(G_["verdict"]).T, width="stretch")
                except Exception as e_:      # noqa: BLE001
                    st.caption(f"읽지 못함: {e_}")
    with t3:
        st.markdown("실험 CSV(열: **시간(일), 변수, 깊이(m), 값, 구분**)를 올리면 '구분'이 *보정*인 행으로 파라미터를 맞추고 "
                    "*검증* 행으로 RMSE·MAE·편향·NSE를 냅니다. 맞춘 값에는 자동으로 **'보정'** 라벨이 붙습니다.")
        st.caption("변수 이름: " + ", ".join(studies.OBS_VARS))
        upc = st.file_uploader("실험 CSV", type=["csv"], key="calcsv")
        so, nm = _scen_choice("이 실험에 해당하는 비교군", "cal_sc")
        prow = [{"키": k_, "설명": v_[0], "하한": v_[1], "상한": v_[2], "로그": v_[3], "보정": k_ in ("k_half",)}
                for k_, v_ in studies.UQ_PARAMS.items()]
        pdf = st.data_editor(pd.DataFrame(prow), hide_index=True, key="cal_par", disabled=["키", "설명"])
        nfev = int(st.number_input("최대 모델 실행 수", value=10, key="cal_nfev", step=1))
        if upc is not None and so != "ERR" and st.button("보정 실행", key="cal_run"):
            obs = pd.read_csv(upc)
            pars = {r["키"]: (float(r["하한"]), float(r["상한"]), bool(r["로그"])) for r in pdf.to_dict("records") if r["보정"]}
            cc = copy.deepcopy(c); cc.light = True
            pb = st.progress(0.0)
            try:
                CR = studies.calibrate(cc, so, obs, pars, max_nfev=nfev,
                                       progress=lambda f, m: pb.progress(min(1.0, f), text=m))
                ss().calib = {k_: v_ for k_, v_ in CR.items() if k_ != "result"}
            except Exception as e:
                st.error(f"보정 실패: {e}")
        CR = ss().get("calib")
        if CR:
            st.write("최적 값(라벨: 보정)", CR["best"])
            for k_ in ("stats_보정", "stats_검증"):
                if k_ in CR:
                    st.markdown(f"**{k_.split('_')[1]} 자료 적합도**")
                    st.dataframe(pd.DataFrame(CR[k_]).T, width="stretch")
            if st.button("보정 값을 공통 조건에 적용(라벨 '보정')", key="cal_apply"):
                for k_, v_ in CR["best"].items():
                    studies.apply_sample(c, k_, v_)
                    c.calibrated[k_] = v_
                st.success("적용했습니다. 고급 파라미터 표의 라벨이 '보정'으로 바뀝니다.")
    with t4:
        st.markdown("흙수조 단면 사진(염료·지시약)을 올리고 눈금자 스케일과 색→농도 보정선을 주면, 같은 시점 시뮬 단면과 "
                    "**IoU**(겹침 비율)와 **도달 깊이 오차**를 냅니다.")
        R = ss().results
        img = st.file_uploader("사진", type=["png", "jpg", "jpeg"], key="photo")
        if img is not None and R:
            st.image(img, width=360)
            a_, b_, d_ = st.columns(3)
            ppc = a_.number_input("눈금자: 1 cm = 몇 픽셀", value=10.0, key="ppc")
            ox = b_.number_input("흙수조 왼쪽 지표 모서리 x(픽셀)", value=0.0, key="ox")
            oy = d_.number_input("… y(픽셀)", value=0.0, key="oy")
            chn = a_.selectbox("색 채널", ["밝기", "R", "G", "B"], key="pch")
            cal_df = st.data_editor(pd.DataFrame([{"픽셀값": 30.0, "농도": 50.0}, {"픽셀값": 220.0, "농도": 0.0}]),
                                    num_rows="dynamic", key="pcal")
            thr = b_.number_input("판정 농도 문턱", value=10.0, key="pthr")
            all_r = R["scen"] + [R["control"]]
            nm = d_.selectbox("비교할 결과", [r["name"] for r in all_r], key="pres")
            r_ = next(x for x in all_r if x["name"] == nm)
            var = a_.selectbox("변수", ["NO3", "SO4", "pH", "theta"], key="pvar")
            si = int(b_.number_input("장면 번호(0 = 운영 시작)", value=len(r_["snaps"]) - 1, key="psi", step=1))
            yj = int(d_.number_input("y 단면 칸", value=r_["grid"]["ny"] // 2, key="pyj", step=1))
            if st.button("비교", key="pgo"):
                sec = r_["snaps"][min(max(si, 0), len(r_["snaps"]) - 1)][var][:, min(max(yj, 0), r_["grid"]["ny"] - 1), :]
                outp = studies.photo_compare(img.getvalue(), ppc, (ox, oy), [tuple(x) for x in cal_df.dropna().values],
                                             thr, np.nan_to_num(sec), r_["grid"]["dx"], r_["grid"]["dz"], chn)
                st.write({k_: v_ for k_, v_ in outp.items() if not k_.endswith("mask")})
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_heatmap(z=outp["photo_mask"].T.astype(int) + 2 * outp["model_mask"].T.astype(int),
                                colorscale=[[0, "white"], [0.33, "#4c78a8"], [0.66, "#f58518"], [1, "#54a24b"]],
                                showscale=False)
                fig.update_yaxes(autorange="reversed", title="깊이 칸"); fig.update_xaxes(title="x 칸")
                fig.update_layout(title="파랑 = 사진만, 주황 = 모델만, 초록 = 겹침", height=380)
                show_fig(fig, key="photo_fig")
        elif img is not None:
            st.info("먼저 ③에서 실행해 결과를 만드세요.")
    with t5:
        st.markdown("불확실한 파라미터의 범위를 주면 **라틴 하이퍼큐브 몬테카를로**로 반복 실행해 지표마다 평균·표준편차·95% 구간과 "
                    "**순위 상관(Spearman) 민감도**를 냅니다. 각 표본마다 대조군도 같이 다시 계산합니다.")
        so, nm = _scen_choice("대상 비교군", "uq_sc")
        rows_u = [{"키": k_, "설명": v_[0], "하한": v_[1], "상한": v_[2], "로그 균등": v_[3], "포함": True}
                  for k_, v_ in studies.UQ_PARAMS.items()]
        udf = st.data_editor(pd.DataFrame(rows_u), hide_index=True, key="uq_par", disabled=["키", "설명"])
        a_, b_, d_ = st.columns(3)
        nN = int(a_.number_input("표본 수", value=12, key="uq_n", step=1))
        ud = b_.number_input("운영 기간(일)", value=float(min(c.days, 60.0)), key="uq_days")
        uc = int(d_.number_input("병렬 코어", value=max(1, (os.cpu_count() or 2) - 1), key="uq_cores", step=1))
        if so not in (None, "ERR") and st.button("불확실성 분석 실행", key="uq_run"):
            rng_ = {r["키"]: (float(r["하한"]), float(r["상한"]), bool(r["로그 균등"])) for r in udf.to_dict("records") if r["포함"]}
            cc = copy.deepcopy(c); cc.days = ud; cc.light = True
            pb = st.progress(0.0)
            ss().stop_flag = False
            ss().uq = studies.lhs_mc(cc, so, rng_, n=nN, cores=uc, progress=lambda f, m: pb.progress(min(1.0, f), text=m),
                                     stop=lambda: bool(ss().get("stop_flag")))
        U_ = ss().get("uq")
        if U_:
            st.caption(f"{U_['method']} · 성공 {U_['n_ok']}/{U_['n']} 표본")
            srows = [{"지표": k_, "평균": v_["평균"], "표준편차": v_["표준편차"], "95% 구간": str(tuple(round(x, 3) for x in v_["95% 구간(분포 2.5~97.5%)"])),
                      "평균의 95% 신뢰구간": str(tuple(round(x, 3) for x in v_["평균의 95% 신뢰구간"]))} for k_, v_ in U_["stats"].items()]
            st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch")
            sens = pd.DataFrame({mk: {pk: v["순위상관 ρ"] for pk, v in d.items()} for mk, d in U_["sensitivity"].items()}).T
            st.markdown("**민감도(순위 상관 ρ, 지표 × 파라미터)**")
            st.dataframe(sens.round(2), width="stretch")
    with t6:
        kp2 = os.path.join(HERE, "selftest", "known_issues_v2.json")
        kp3 = os.path.join(HERE, "selftest", "known_issues_v3.json")
        if os.path.exists(kp2) and os.path.exists(kp3):
            with open(kp2, encoding="utf-8") as fh:
                k2 = {x["test"]: x for x in json.load(fh)}
            with open(kp3, encoding="utf-8") as fh:
                k3 = {x["test"]: x for x in json.load(fh)}
            rows_k = []
            for t_, x in k3.items():
                b_ = k2.get(t_, {})
                rows_k.append({"번호": x["id"], "v2 시험": ("실패(문제 사실)" if not b_.get("pass", True) else "통과(문제 아님)") if b_ else "–",
                               "v2 내용": b_.get("detail", ""), "v3 시험": "통과" if x["pass"] else "실패", "v3 내용": x["detail"]})
            st.dataframe(pd.DataFrame(rows_k), hide_index=True, width="stretch")
            st.caption("전체 처리 표(수정 내용·해당 없음 근거 포함)는 README '알려진 문제 처리 표'에 있습니다.")
        if st.button("알려진 문제 시험 다시 실행(현재 코드)", key="ki_run"):
            sys.path.insert(0, os.path.join(HERE, "selftest"))
            import known_issues as KI
            with st.spinner("시험 중…"):
                st.dataframe(pd.DataFrame(KI.run("v3")), hide_index=True, width="stretch")
    with t7:
        pst = PE.status()
        st.markdown(f"**PHREEQC(IPhreeqc)**: 연결 {pst['ok']} · 버전 {pst.get('version', '')} · 라이브러리 `{pst.get('lib', '')}` · "
                    f"DB `{os.path.basename(pst.get('db', ''))}`")
        if pst["error"]:
            st.error(pst["error"])
        ep = os.path.join(HERE, "engine_check.json")
        if os.path.exists(ep):
            with open(ep, encoding="utf-8") as fh:
                st.json(json.load(fh))
        if st.button("엔진 설치 상태 다시 확인", key="eng_chk"):
            import engine_check
            with st.spinner("확인 중…"):
                st.json(engine_check.check(write=True))
        vp = os.path.join(HERE, "verify", "results.json")
        if os.path.exists(vp):
            with st.expander("v2 개발 때 기록한 검증 표(참고)"):
                with open(vp, encoding="utf-8") as fh:
                    V = json.load(fh)
                st.dataframe(pd.DataFrame(V["table"]), hide_index=True, width="stretch")

# ═════════════════════════════ ⑦ 내보내기 ═════════════════════════════════
with tabs[6]:
    import export_xlsx as X
    st.markdown("**프로젝트 저장·불러오기** (입력 전체 + 3MF 원본, zip)")
    if st.button("프로젝트 파일 만들기", key="mkproj"):
        ss().projzip = X.project_to_zip(c, ss().scen)
    if ss().get("projzip"):
        st.download_button("💾 프로젝트 저장(.zip)", ss().projzip, file_name="slad_project.zip", mime="application/zip")
    upj = st.file_uploader("프로젝트 불러오기(.zip 또는 이전 판 .json)", type=["zip", "json"], key="projup")
    if upj is not None and st.button("불러오기 적용"):
        try:
            if upj.name.lower().endswith(".zip"):
                cc, scs = X.project_from_zip(upj.getvalue(), new_scenario, load_file_into)
            else:
                cc, scs = X.project_from_json(upj.getvalue().decode("utf-8"), new_scenario, load_file_into)
            ss().common = cc
            ss().scen = scs
            ss().cur = 0
            st.success("불러왔습니다. 같은 입력과 같은 코드(해시)로 다시 실행하면 같은 결과가 나옵니다.")
            st.rerun()
        except Exception as e:
            st.error(f"불러오기 실패: {e}")
    R = ss().results
    if R is None:
        st.info("결과 내보내기는 ③ 실행 후에 쓸 수 있습니다.")
    else:
        C_ = Common.from_dict(R["common"])
        st.markdown("**결과 내보내기**")
        unrel = bool(R.get("failed"))
        allow = True
        if unrel:
            st.error("수렴 실패(FAILED) 결과가 있어 Excel·CSV 내보내기를 기본 차단했습니다: " + ", ".join(R["failed"]))
            allow = st.checkbox("그래도 내보내기(모든 파일에 '비신뢰 결과' 표시)", value=False, key="allow_unrel")
        extras = {}
        if ss().get("uq"):
            extras["uq"] = ss().uq
        if ss().get("calib"):
            extras["calib"] = ss().calib
        if ss().get("indep"):
            extras["indep"] = ss().indep
        else:
            import glob
            fs_ = sorted(glob.glob(os.path.join(HERE, "selftest", "independence_*.json")))
            if fs_:
                try:
                    with open(fs_[-1], encoding="utf-8") as fh:
                        extras["indep"] = json.load(fh)
                except Exception:      # noqa: BLE001
                    pass
        sp_ = os.path.join(HERE, "selftest", "selfcheck_results.json")
        if os.path.exists(sp_):
            with open(sp_, encoding="utf-8") as fh:
                extras["selfcheck"] = json.load(fh)["table"]
        kp_ = os.path.join(HERE, "selftest", "known_issues_table.json")
        if os.path.exists(kp_):
            with open(kp_, encoding="utf-8") as fh:
                extras["known"] = json.load(fh)
        e1, e2, e3 = st.columns(3)
        with e1:
            if allow and st.button("Excel 만들기(.xlsx)"):
                ss().xlsx = X.build_xlsx(R, C_, ss().scen, unreliable=unrel, extras=extras)
            if allow and ss().get("xlsx"):
                st.download_button("📊 Excel 내려받기", ss().xlsx, file_name="slad_결과.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if allow:
                head, rows = M.summary_table(R["metrics"])
                csv = pd.DataFrame(rows, columns=head).to_csv(index=False)
                if unrel:
                    csv = "# 비신뢰 결과(수렴 실패 포함)\n" + csv
                st.download_button("지표표 CSV", csv.encode("utf-8-sig"), file_name="slad_지표.csv", mime="text/csv")
        with e2:
            if allow:
                st.download_button("결과 저장(.npz)", X.results_npz(R), file_name="slad_결과.npz")
                st.download_button("결과 요약(.json)", X.results_json(R, unreliable=unrel), file_name="slad_결과.json",
                                   mime="application/json")
                if st.button("그림 PNG 묶음 만들기(.zip)"):
                    ss().pngzip, fw = X.png_zip(R, C_, unreliable=unrel)
                    if fw:
                        st.warning(fw)
                if ss().get("pngzip"):
                    st.download_button("🖼 PNG 묶음", ss().pngzip, file_name="slad_그림.zip")
        with e3:
            if st.button("애니메이션 HTML 만들기"):
                all_res = R["scen"] + [R["control"]]
                objs = {}
                for r in R["scen"]:
                    s_ = next((x for x in ss().scen if x["name"] == r["name"]), None)
                    if s_:
                        so = scenario_obj(s_, C_)
                        objs[r["name"]] = (so.objects, so.roles, Placement(**r["placement"]))
                fig = viz.compare_view(all_res, C_, ["반응재 분포", "NO₃ 오염 플룸"], light=True, scen_objects=objs)
                ss().html_cmp = viz.to_html(fig, sync_cameras=True).encode("utf-8")
                parts = []
                for r in R["scen"]:
                    if r["name"] in objs:
                        o, rl, pl = objs[r["name"]]
                        parts.append(viz.install_animation(r, o, rl, pl, light=True))
                ss().html_inst = [viz.to_html(f).encode("utf-8") for f in parts]
            if ss().get("html_cmp"):
                st.download_button("🎬 다중 비교 애니메이션.html", ss().html_cmp, file_name="slad_비교_애니메이션.html", mime="text/html")
            for i, h in enumerate(ss().get("html_inst", []) or []):
                st.download_button(f"🎬 설치 과정 {i + 1}.html", h, file_name=f"slad_설치과정_{i + 1}.html", mime="text/html",
                                   key=f"hi{i}")


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
    st.markdown("황 **투입량 · 반응 구역 수분 · 입경 · 황:물 배합 · 수분 유지 재료 비율** 중 하나를 여러 값으로 바꿔 가며 "
                "선택한 장치(비교군)를 같은 지하수 환경(② 공통 조건 = 통제변수)에서 계산하고, 질산염 제거율을 비교합니다. "
                "지하수 흐름은 바꾸지 않습니다. 대조군은 반응재가 없어 모든 조건에서 같으므로 한 번만 계산합니다.")
    jsw = runjob.get(_sweep_key())
    if jsw and jsw["done"]:
        runjob.pop(_sweep_key())
        if jsw["error"] is None:
            ss().sweep_res = jsw["result"]
            st.success("조건 비교 계산을 마쳤습니다.")
        elif jsw["error_type"] == "StopRequested":
            st.warning("조건 비교를 중단했습니다.")
        else:
            st.error(f"조건 비교 오류: {jsw['error']}\n\n{jsw['trace']}")
    names_ = [s_["name"] for s_ in ss().scen]
    if not names_:
        st.info("① 탭에서 비교군(예: '예시 전부')을 먼저 추가하세요.")
    else:
        pick = st.multiselect("비교할 장치(비교군)", names_, default=names_, key="sw_pick")
        var = st.selectbox("바꿀 조건(독립변수)", list(studies.SWEEP_VARS.keys()), key="sw_var")
        vtxt = st.text_input("값 (쉼표로 구분)", studies.SWEEP_VARS[var][1], key="sw_vals_" + studies.SWEEP_VARS[var][0])
        try:
            vals = [float(x) for x in vtxt.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            vals = []
            st.error("값은 숫자를 쉼표로 구분해 적으세요.")
        ret_ = None
        if studies.SWEEP_VARS[var][0] == "ret_frac":
            ret_ = st.selectbox("수분 유지 재료", [k_ for k_ in DZ.RETAINERS if k_ != "없음"], key="sw_ret")
        if studies.SWEEP_VARS[var][0] == "moist_set" and c.denit_moisture.startswith("끔"):
            st.warning("② 탭에서 탈질 수분 함수가 '끔'이라 포화도를 바꿔도 탈질 속도는 그대로입니다(호기성 황 산화만 바뀜).")
        s1, s2, s3, s4 = st.columns(4)
        sw_days = float(s1.selectbox("기간", [90.0, 180.0], index=1, format_func=lambda x: f"{x:.0f}일", key="sw_days"))
        sw_grid = s2.selectbox("격자", list(GRID_PRESETS.keys()), index=0, key="sw_grid")
        sw_chem = s3.selectbox("화학", ["내장", "PHREEQC"], index=0, key="sw_chem", help="내장 = 빠름(PHREEQC와 탈질량 차이 0.3 % 안팎, 자체 점검)")
        n_runs = 1 + len(vals) * len(pick)
        sw_cores = int(s4.number_input("병렬 코어", value=studies.safe_cores(n_runs, int(np.prod(GRID_PRESETS[sw_grid]))),
                                       min_value=1, max_value=max(1, os.cpu_count() or 1), key="sw_cores"))
        st.caption(f"계산 {n_runs}개(대조군 1 + 값 {len(vals)} × 장치 {len(pick)}). 가벼움 격자·내장 화학·{sw_days:.0f}일이면 "
                   "계산 하나에 보통 1~5분, 병렬 코어 수만큼 동시에 돕니다. 비교군의 설계 변수(①)가 꺼져 있으면 그 비교군의 지금 값을 출발점으로 씁니다.")
        busy = runjob.running(_sweep_key()) or runjob.running(ss().sid)
        if st.button("▶ 조건 비교 실행", type="primary", disabled=busy or not pick or not vals, key="sw_go"):
            cq = copy.deepcopy(c)
            cq.days = sw_days
            cq.nx, cq.ny, cq.nz = GRID_PRESETS[sw_grid]
            cq.grid_preset = sw_grid
            cq.chem_engine = sw_chem
            runnable_, skipped_ = prepare_runs(cq, [s_ for s_ in ss().scen if s_["name"] in pick])
            for nm_, why_ in skipped_:
                st.error(f"'{nm_}' 제외: {why_}")
            if runnable_:
                runjob.start(_sweep_key(), studies.sweep, cq, [so for _, so in runnable_], var, vals, ret_,
                             max(1, sw_cores), (90.0, 180.0), label="조건 비교")
                st.rerun()
        sweep_status()
    SR = ss().get("sweep_res")
    if SR:
        st.markdown(f"#### 결과: {SR['var']}")
        tags = [f"{h:.0f}일" for h in SR["horizons"]]
        tag = st.radio("기간", tags + ["끝"], horizontal=True, key="sw_tag")
        mlabel = st.radio("최적화 기준 제거율", list(studies.OPT_METRICS.keys()), horizontal=True, key="sw_metric")
        mkey = studies.OPT_METRICS[mlabel]
        w1, w2, w3 = st.columns(3)
        wr = w1.slider("가중치: 제거율", 0.0, 1.0, 0.5, 0.05, key="sw_wr")
        we = w2.slider("가중치: 황 투입량 대비 제거 N", 0.0, 1.0, 0.3, 0.05, key="sw_we")
        wd = w3.slider("가중치: 황 투입량(적을수록 좋음)", 0.0, 1.0, 0.2, 0.05, key="sw_wd")
        rows_ = studies.score_rows([dict(r_) for r_ in SR["rows"]], mkey, tag, wr, we, wd)
        cols_show = ["비교군", "값", "상태", f"{mkey}@{tag}", f"dev_removal@{tag}", f"sys_removal@{tag}", f"dev_Cin@{tag}",
                     f"dev_Cout@{tag}", f"S0_g@{tag}", f"S_used_g@{tag}", f"S_ox_g@{tag}", f"N_removed_g@{tag}",
                     f"N_per_kgS@{tag}", f"sat_mean@{tag}", f"bypass_pct@{tag}", "점수", "파레토"]
        nice = {f"dev_removal@{tag}": "장치 제거율 %", f"sys_removal@{tag}": "지하수 유입 감소율 %", f"dev_Cin@{tag}": "장치 유입 NO₃-N",
                f"dev_Cout@{tag}": "장치 하류 NO₃-N", f"S0_g@{tag}": "황 투입 g", f"S_used_g@{tag}": "황 소비 g",
                f"S_ox_g@{tag}": "공기 산화 g", f"N_removed_g@{tag}": "제거 N g", f"N_per_kgS@{tag}": "g N/kg S",
                f"sat_mean@{tag}": "평균 포화도", f"bypass_pct@{tag}": "우회 %", f"{mkey}@{tag}": "기준 제거율 %"}
        df_ = pd.DataFrame([{nice.get(k_, k_): r_.get(k_) for k_ in dict.fromkeys(cols_show)} for r_ in rows_])
        st.dataframe(df_.round(4), hide_index=True, width="stretch")
        for r_ in rows_:
            if r_.get("상태") != "OK":
                st.error(f"{r_['비교군']} · 값 {r_['값']}: {r_.get('상태')} {str(r_.get('실패 이유', ''))[:200]}")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            figs_ = make_subplots(rows=1, cols=3, subplot_titles=(f"{mlabel}", "황 투입량 대비 제거 N (g N/kg S)", "황 소비 (g)"))
            pal = viz.SCEN_COLORS
            for i_, nm_ in enumerate(dict.fromkeys(r_["비교군"] for r_ in rows_)):
                rr_ = sorted([r_ for r_ in rows_ if r_["비교군"] == nm_], key=lambda x: x["값"])
                xs = [r_["값"] for r_ in rr_]
                for col_i, kk in enumerate((mkey, "N_per_kgS", "S_used_g")):
                    figs_.add_scatter(x=xs, y=[r_.get(f"{kk}@{tag}") for r_ in rr_], mode="lines+markers", name=nm_,
                                      line=dict(color=pal[i_ % len(pal)]), showlegend=(col_i == 0), row=1, col=col_i + 1)
            figs_.update_xaxes(title_text=SR["var"])
            figs_.update_layout(height=380, margin=dict(l=40, r=10, t=40, b=40), legend=dict(orientation="h", y=-0.3))
            show_fig(figs_, key="sweepfig")
        except Exception as e_:      # noqa: BLE001
            st.caption(f"(그림 생략: {e_})")
        scored = [r_ for r_ in rows_ if r_.get("점수") is not None]
        if scored:
            for nm_ in dict.fromkeys(r_["비교군"] for r_ in scored):
                best = max([r_ for r_ in scored if r_["비교군"] == nm_], key=lambda x: x["점수"])
                st.success(f"**{nm_}** 가장 좋은 조건: {SR['var']} = {best['값']:g} · 점수 {best['점수']:.2f} · "
                           f"{mlabel} {best.get(f'{mkey}@{tag}', float('nan')):.1f} · 황 투입 {best.get(f'S0_g@{tag}', float('nan')):.0f} g · "
                           f"황 소비 {best.get(f'S_used_g@{tag}', float('nan')):.2f} g")
            st.caption("점수 = 가중치×(제거율/최댓값) + 가중치×(황 투입량 대비 제거 N/최댓값) − 가중치×(황 투입량/최댓값). "
                       "파레토 = 제거율이 같거나 높으면서 황을 덜 쓰는 다른 조건이 없는 조건. 모델 가정(⑤·⑥ 탭)에 따른 비교이며 "
                       "수분 함수·수분 유지 재료 효과의 불확실성이 큽니다.")
        st.download_button("조건 비교 결과 CSV", pd.DataFrame(rows_).to_csv(index=False).encode("utf-8-sig"),
                           file_name="slad_조건비교.csv", key="sw_csv")

# 프래그먼트가 없는 옛 Streamlit: 계산 중이면 2초마다 화면을 다시 그려 진행률·결과를 갱신
if _frag is None and (runjob.running(ss().sid) or runjob.running(_sweep_key())):
    time.sleep(2.0)
    st.rerun()
