"""세 비교군(새 배치·영구 주입공·케이슨) + 대조군 비교 실행 — app_run_prompt.md 값 그대로.

  python runs/compare_run.py --run B --days 365 --grid 33x33x44 --out runs/cmp/B_main [--refill-json refill.json]
  --run A : 설계서 원안 반응재(슬러리 0.224 L, 케이슨 D44·T4)   --run B : 확대(26.5 L, 케이슨 D150·T50)
입력 3MF는 runs/inputs/ 에 둔다. 모든 입력값은 아래 PROMPT_* 상수(프롬프트 §2~§3)에 적어 두었고 라벨은 주석.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)

import export_xlsx as X                       # noqa: E402
import install_events as ie                   # noqa: E402
import metrics as M                           # noqa: E402
from engine import Scenario, eval_start_from, run_scenario   # noqa: E402
from io_3mf import load_mesh_file             # noqa: E402
from params import Common, suggest_role       # noqa: E402
from voxel import Placement                   # noqa: E402
import geometry_design as GD                  # noqa: E402

IN = os.path.join(HERE, "inputs")
PLATE = {"hole_d_m": 0.002, "open_ratio": 0.20, "t_p_m": 0.003, "fill": "반응재/흙"}   # [가정] 프롬프트 §3 케이슨 판 변수


def make_common(days, grid, d_p=5.0e-6, d_p_label=""):
    c = Common()
    c.Lx = c.Ly = 2.4                          # [설계서]
    c.H_soil = 2.2                             # [설계서]
    c.H_aq = 0.0                               # [가정]
    c.wt_depth = None                          # [앱]
    c.bottom_bc = "자유배수"                    # [가정]
    c.wt_margin_m = 0.30                       # [설계서]
    c.gw_velocity = 0.0
    c.soil_mode = "간단"
    c.texture = "양토(Loam)"                    # [사용자]
    c.init_saturation = 0.5
    c.q_mm_day = 5.0
    c.NO3_in_mgN = 50.0
    c.SO4_in_mg, c.Ca_in_mg, c.alk_in_mgCaCO3 = 10.0, 20.0, 50.0
    c.ratio_S_CaCO3 = 3.0                      # [설계서]
    c.temperature = 22.0
    c.days = float(days)
    nx, ny, nz = (int(v) for v in grid.split("x"))
    c.nx, c.ny, c.nz = nx, ny, nz
    c.flow_mode = "과도"
    c.chem_engine = "PHREEQC"                  # 라이브러리가 없으면 엔진이 내장으로 바꾸고 결과에 기록
    c.kinetics = "표면적(문헌, v4)"
    c.moist_model = "A 기계론(f_w·f_O2)"
    c.mode3 = True
    c.o2_stage = True
    c.o2_gas = True
    c.chem_mode = "SLAD"
    c.inject_model = "과도 Richards"
    c.straining = True
    c.well_model = True
    c.lambda_mode = "직접 입력"
    c.refill_ok_pct = 90.0
    c.eta_R_min_pct = 1.0
    c.redistribute_runoff = True
    c.params["inj_d_p"] = float(d_p)           # [사용자] 5 µm
    return c


def _load(fname):
    d = load_mesh_file(open(os.path.join(IN, fname), "rb").read(), fname)
    return d["objects"]


def rows_layout(V_tot_L, refill=None):
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 120.0, "메모": "관 4개 × 30분 [가정]"}),
          dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0, "메모": "열린 포트만 객체로 존재"})]
    for d in ("1p75", "1p45", "1p15"):
        names = ",".join(f"port_open_d{d}_p{i}" for i in (1, 2, 3, 4))
        ev.append(dict(ie.empty_row("시약 주입"), **{"대상 객체": names, "방식": "펌프", "부피(L)": V_tot_L / 3.0,
                                                  "고형분비(v/v)": 0.02, "유량(L/min)": 0.2, "메모": "아래 깊이부터 [가정]"}))
    ev.append(dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0, "메모": "[가정]"}))
    if refill:
        ev.append(dict(ie.empty_row("재보충"), **refill))
    return ev


def rows_well(V_tot_L, refill=None):
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
          dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
          dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": V_tot_L, "고형분비(v/v)": 0.02, "유량(L/min)": 0.2}),
          dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0})]
    if refill:
        ev.append(dict(ie.empty_row("재보충"), **refill))
    return ev


def rows_caisson(refill=None):
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0})]
    if refill:
        ev.append(dict(ie.empty_row("재보충"), **refill))
    return ev


def build_scenarios(c, run, refill=None):
    """refill: {'layout':{...}, 'well':{...}, 'caisson':{...}} 재보충 행(없으면 1차 계산)"""
    refill = refill or {}
    V_tot = 0.224 if run == "A" else 26.5
    cx = cy = c.Lx / 2
    scs = []
    # 1 새 배치
    obj = _load("layout_4pipe_rhombus.3mf")
    roles = {o.name: {"role": suggest_role(o.name)} for o in obj}
    scs.append(Scenario(name="신박 4관 마름모", mechanism="동축 이중관을 영구 서비스 관으로 남기고, 깊이 1.15·1.45·1.75 m 출구만 열어 슬러리를 층별 주입한다. 닫힌 출구(0.15·0.45·0.75 m)와 관 끝은 막혀 있다. 부족한 깊이만 재주입한다.",
                        objects=obj, roles=roles, placement=Placement(cx, cy, 0.0, 0.0, 0.0, 1.0),
                        events=rows_layout(V_tot, refill.get("layout")), reagent={"form": "(c) 입자 슬러리(포획)"}))
    # 2 영구 주입공
    obj = _load("injection_well_1x1.3mf")
    roles = {o.name: {"role": suggest_role(o.name)} for o in obj}
    scs.append(Scenario(name="영구 주입공", objects=obj, roles=roles, placement=Placement(cx, cy, 0.0, 0.0, 0.0, 1.0),
                        events=rows_well(V_tot, refill.get("well")), reagent={"form": "(c) 입자 슬러리(포획)"}))
    # 3 케이슨
    fn = "caisson_D44_T4.3mf" if run == "A" else "caisson_D150_T50.3mf"
    obj = _load(fn)
    roles = {}
    for o in obj:
        r = suggest_role(o.name)
        rr = {"role": r}
        if r == "투수성 판(구멍)":
            rr.update(PLATE)
        if r == "반응재(고체)":
            rr["accessible"] = True
        roles[o.name] = rr
    scs.append(Scenario(name="케이슨(하향)", objects=obj, roles=roles, placement=Placement(cx, cy, 0.0, 0.0, 0.0, 1.0),
                        events=rows_caisson(refill.get("caisson")), reagent={"form": "(b) 고체 카트리지 교체"}))
    return scs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--grid", default="33x33x44")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="콤마로 비교군 번호(0 새 배치, 1 영구 주입공, 2 케이슨); 기본 전부")
    ap.add_argument("--no-ctrl", action="store_true")
    ap.add_argument("--refill-json", default=None)
    ap.add_argument("--d-p", type=float, default=5.0e-6)
    ap.add_argument("--eval-start", type=float, default=None)
    ap.add_argument("--no-modes", action="store_true", help="수분 모드 B·C 비교 계산 생략")
    ap.add_argument("--o2-zero", action="store_true", help="[가정] 참고 시뮬(01_Final_Simulation)과 같이 용존 O₂ 0·토양 공기 O₂ 교환 끔")
    ap.add_argument("--port-void", action="store_true", help="v4.3 방식: 얇은 개구부도 빈 공간 물성")
    ap.add_argument("--save-full", action="store_true", help="스냅숏·일별 시계열·설치 프레임 전체를 <out>.pkl로 저장(그림·애니메이션용)")
    ap.add_argument("--snaps", type=int, default=None, help="저장할 스냅숏 개수(기본 앱 값 30)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    c = make_common(a.days, a.grid, a.d_p)
    if a.o2_zero:
        c.params["DO_in"] = 0.0
        c.o2_gas = False
    if a.port_void:
        c.port_thin_as_soil = False
    if a.snaps:
        c.n_snapshots = int(a.snaps)
    refill = json.load(open(a.refill_json, encoding="utf-8")) if a.refill_json else None
    scs = build_scenarios(c, a.run, refill)
    if a.only:
        scs = [scs[int(i)] for i in a.only.split(",")]
    tm = {}
    prog = lambda f, m: print(f"\r{f:5.1%} {m[:60]}      ", end="", flush=True)
    t0 = time.time()
    ctrl = None
    if not a.no_ctrl:
        print("대조군 계산…", flush=True)
        ctrl = run_scenario(c, None, prog); print()
        tm["control_s"] = time.time() - t0
    bt = a.eval_start if a.eval_start is not None else (eval_start_from(ctrl, c.NO3_in_mgN) if ctrl else None)
    res = []
    for s in scs:
        t1 = time.time()
        print(f"{s.name} 계산…", flush=True)
        res.append(run_scenario(c, s, prog, eval_start_day=bt)); print()
        tm[s.name] = time.time() - t1
    R = {"control": ctrl, "scen": res, "skipped": [], "common": c.to_dict(), "eval_start": bt}
    R["metrics"] = [M.compute(r, ctrl, c, bt) for r in res] + ([M.compute(ctrl, None, c, bt)] if ctrl else [])
    if a.save_full:
        import pickle
        pickle.dump({"control": ctrl, "scen": res, "metrics": R["metrics"], "eval_start": bt, "common": R["common"],
                     "args": vars(a)}, open(a.out + ".pkl", "wb"), protocol=4)
    head, rows = M.summary_table(R["metrics"])
    import pandas as pd
    print(pd.DataFrame(rows, columns=head).to_string(index=False))
    if ctrl is not None:
        open(a.out + ".xlsx", "wb").write(X.build_xlsx(R, c, []))
    else:
        R["runs"] = [r for r in res]
    try:
        open(a.out + ".json", "wb").write(X.results_json(R))
    except Exception as e:                      # 대조군 없는 진단 실행
        json.dump({"runs": [{"name": r.get("name"), "diag": r.get("diag"), "S0_g": r.get("S0_g"), "status": r.get("status"),
                             "warnings": r.get("warnings")} for r in res], "err": str(e)},
                  open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, default=float)
    json.dump({"timing_s": tm, "total_s": time.time() - t0, "days": a.days, "grid": a.grid, "run": a.run,
               "status": [r.get("status") for r in res], "eval_start": bt}, open(a.out + ".timing.json", "w"), indent=1)
    print("저장:", a.out, "총", round(time.time() - t0), "초", flush=True)
    if not a.no_modes and res and all(r.get("status") == "OK" for r in res):
        import studies
        t2 = time.time()
        print("수분 모드 B·C 비교 계산…", flush=True)
        md = studies.mode_compare(c, scs, ctrl, res, 1, lambda f, m: print(f"\r모드 {f:5.1%} {m[:50]}   ", end="", flush=True), None)
        print()
        def _j(o):
            if isinstance(o, dict):
                return {str(k): _j(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_j(v) for v in o]
            return o
        json.dump(_j(md), open(a.out + ".modes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        tm["modes_s"] = time.time() - t2
        json.dump({"timing_s": tm, "total_s": time.time() - t0, "days": a.days, "grid": a.grid, "run": a.run,
                   "status": [r.get("status") for r in res], "eval_start": bt}, open(a.out + ".timing.json", "w"), indent=1)
        print("모드 비교 저장", round(time.time() - t2), "초")


if __name__ == "__main__":
    main()
