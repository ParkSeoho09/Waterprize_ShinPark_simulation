"""수정 전후 비교(C2·C3) 재현 스크립트. 사용: python runs/x11_baseline.py <출력.json> [일수] [주입 입자 지름 m]
수정 전 기준선과 같은 입력: 영구 주입공·케이슨, 사양토, NO3 50 mg N/L, 5 mm/일, 22x22x36, 내장 화학, 시드 12345, 모드 A·B·C."""
import sys, os, time, json, copy
try:
    import resource
except ImportError:          # Windows
    resource = None
APP=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.abspath(sys.argv[1])
DAYS=float(sys.argv[2]) if len(sys.argv)>2 else 90; DP=float(sys.argv[3]) if len(sys.argv)>3 else None
sys.path.insert(0,APP); os.chdir(APP)
import numpy as np
from params import Common, suggest_role
from engine import run_scenario, Scenario
import metrics as M
from io_3mf import load_3mf
from voxel import Placement
import install_events as ie
np.random.seed(12345)
c=Common(); c.nx,c.ny,c.nz=22,22,36; c.days=DAYS; c.chem_engine="내장"; c.light=True; c.n_snapshots=4
c.texture="사양토(Sandy loam)"; c.NO3_in_mgN=50.0; c.q_mm_day=5.0
if DP: c.params["inj_d_p"]=DP
def well():
    d=load_3mf(open("samples/injection_well_scale0.05.3mf","rb").read(),"w")
    roles={o.name:{"role":suggest_role(o.name)} for o in d["objects"]}
    ev=[dict(ie.empty_row("구조물 삽입"),**{"소요(분)":30.0}),dict(ie.empty_row("개구부 열기"),**{"소요(분)":1.0}),
        dict(ie.empty_row("시약 주입"),**{"방식":"펌프","부피(L)":2.84,"고형분비(v/v)":0.2,"유량(L/min)":0.2}),
        dict(ie.empty_row("정치(대기)"),**{"소요(분)":60.0})]
    return Scenario(name="영구 주입공",objects=d["objects"],roles=roles,placement=Placement(0.25,0.25,0,0,0,1),events=ev,reagent={"form":"(c) 입자 슬러리(포획)"})
def cais():
    import geometry_design as GD
    p="samples/caisson_down.3mf"; d=load_3mf(open(p,"rb").read(),"c")
    return Scenario(name="케이슨형(하향)",objects=d["objects"],roles=GD.sample_roles(p,d["objects"]),placement=Placement(0.25,0.25,0,0,0,1),
       events=[dict(ie.empty_row("구조물 삽입"),**{"소요(분)":30.0})],reagent={"form":"(b) 고체 카트리지 교체"})
def f(x):
    try: x=float(x); return None if x!=x else x
    except Exception: return str(x)
t=time.time(); ctrl=run_scenario(c,None); tc=time.time()-t
import engine as _E
out={"t_ctrl_s":tc,"runs":[],"code_hash":_E.code_hash(),"engine":_E.ENGINE_VERSION,"inj_d_p":c.p("inj_d_p"),"days":DAYS}
for mode in ("A 기계론(f_w·f_O2)","B 경험식(NEMIS 계열)","C 상한(억제 없음)"):
    for mk in (well,cais):
        c2=copy.deepcopy(c); c2.moist_model=mode
        t=time.time(); r=run_scenario(c2,mk()); tw=time.time()-t
        row={"mode":mode[0],"name":r["name"],"status":r["status"],"t_s":tw}
        if r["status"]=="OK":
            m=M.compute(r,ctrl,c2,None); cm=M.core_metrics(r,ctrl,c2,None)
            import studies as _ST
            row.update({k_:(f(v_) if not isinstance(v_,bool) else v_) for k_,v_ in _ST.injection_summary(r).items()})
            for k in ("r_cm","delivery_pct","S_plan_g","S0_g","reactive_g","denit_g","denit_g_eval","N_to_gw_g","S_denit_g","S_aerobic_g","S_aerobic_pct","eta_denit","eta_flow_storage","eta_ctrl_mass","eta_vs_ctrl","dpH_out_ctrl","V_eff_cm3","V_eff_S_cm3"):
                row[k]=f(m.get(k))
            for k in ("dev_removal","sys_removal","capture_pct","sat_mean","sat_min","sat_max","bypass_pct"):
                row["core_"+k]=f(cm.get(k))
            row["mb_bad"]=list((r.get("mass_balance_bad") or {}).keys()); row["W_err"]=f((r.get("mass_balance") or {}).get("W_err_rel"))
            row["code_hash"]=r["provenance"].get("code_hash")
        else: row["reason"]=r.get("fail_reason")
        out["runs"].append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
out["maxrss_MB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource else None; out["seed"]=12345
json.dump(out,open(OUT,'w'),ensure_ascii=False,indent=1)
