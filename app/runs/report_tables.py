"""compare_run.py 결과(JSON)에서 보고서 표 1·2·3을 만든다.  python runs/report_tables.py runs/cmp/B_main22 [runs/cmp/A_main22]"""
import json
import math
import sys

NAMES = ["신박 4관 마름모", "영구 주입공", "케이슨(하향)", "대조군(구조물 없음)"]


def f(x, n=3):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    try:
        if math.isnan(x):
            return "–"
    except TypeError:
        return str(x)
    return f"{x:.{n}g}"


def load(prefix):
    d = json.load(open(prefix + ".json", encoding="utf-8"))
    try:
        mo = json.load(open(prefix + ".modes.json", encoding="utf-8"))
    except Exception:
        mo = None
    try:
        tm = json.load(open(prefix + ".timing.json", encoding="utf-8"))
    except Exception:
        tm = {}
    return d, mo, tm


def table1(d):
    ms = {m["name"]: m for m in d["metrics"]}
    runs = {r["name"]: r for r in d["runs"]}
    rows = [
        ("질산염 제거효율 η, 평가 구간(대조군 보정 포함) %", lambda m: m.get("eta_mass")),
        ("η 반응 기여(평가 구간 탈질/유입 N) %", lambda m: m.get("eta_denit")),
        ("η 흐름·저장 기여 %p", lambda m: m.get("eta_flow_storage")),
        ("지표1 탈질 질량 누적 g N", lambda m: m.get("denit_g")),
        ("퍼짐 반경 r (cm)", lambda m: m.get("r_cm")),
        ("유효 부피 V_eff — SO₄ 증가 기준 (cm³)", lambda m: m.get("V_eff_cm3")),
        ("유효 부피 V_eff — 황 소모 셀 기준 (cm³)", lambda m: m.get("V_eff_S_cm3")),
        ("전달률 = 전달/계획 (%)", lambda m: m.get("delivery_pct")),
        ("투수성 유지율 (반응재 영역 K, %)", lambda m: m.get("Kreac_ret")),
        ("투수성 유지율 (도메인 유효 K, %)", lambda m: m.get("Keff_vs_soil")),
        ("ΔpH 반응재 아래 (대조군 대비)", lambda m: m.get("dpH_below_ctrl")),
        ("ΔpH 반응재 주변부 (대조군 대비)", lambda m: m.get("dpH_around_ctrl")),
        ("ΔpH 바닥 유출부 (대조군 대비)", lambda m: m.get("dpH_out_ctrl")),
        ("pH 판정", lambda m: m.get("pH_verdict")),
        ("황산이온 이론비 (측정/이론)", lambda m: m.get("SO4_ratio")),
        ("황 사용 — 탈질 g S", lambda m: m.get("S_denit_g")),
        ("황 사용 — 호기성 산화 g S", lambda m: m.get("S_aerobic_g")),
        ("지하수(바닥) 도달 N 누적 g", lambda m: m.get("N_to_gw_g")),
        ("지하수 10 mg/L 도달일", lambda m: m.get("arrival_10mg_day")),
        ("η_stable 이후 η가 0.8× 아래로 내려간 날(수명)", lambda m: m.get("eta_life_day")),
    ]
    out = ["| 지표 | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    out.append("| 대조군 돌파일 / η 평가 시작일 (일) | " + " | ".join([f(d.get("eval_start"))] * len(NAMES)) + " |")
    for lab, fn in rows:
        out.append(f"| {lab} | " + " | ".join(f(fn(ms[n])) if n in ms else "–" for n in NAMES) + " |")
    pts = []
    for n in NAMES[:3]:
        r = runs.get(n)
        dg = (r or {}).get("diag") or {}
        pts.append(f(dg.get("through_ratio")) + " (" + str(dg.get("through_verdict", "–")) + ")")
    out.append("| 반응재 통과 유량 비 / 판정 | " + " | ".join(pts) + " | – |")
    out.append("| 계산 시간 (초) | " + " | ".join(f((runs.get(n) or {}).get("wall_s"), 4) for n in NAMES) + " |")
    return "\n".join(out)


def table2(mo):
    if not mo:
        return "(모드 비교 결과 없음)"
    out = []
    for key, lab in (("dev_removal", "장치 제거율(η) 순위"), ("sys_removal", "도메인 η 순위")):
        r = (mo.get("rank") or {}).get(key, {})
        out.append(f"**{lab}**: {r.get('verdict', '–')}")
        for L, o in (r.get("orders") or {}).items():
            out.append(f"- 모드 {L}: " + " > ".join(o))
    return "\n".join(out)


def table_modes_metrics(mo, names):
    if not mo:
        return ""
    out = ["| 모드 | 비교군 | 장치 제거율 % | 도메인 η % | 황 호기성 비율 % | 황 전달 g S |", "|---|---|---|---|---|---|"]
    for L, d in (mo.get("by_mode") or {}).items():
        for nm in names:
            v = d.get(nm)
            if not isinstance(v, dict):
                continue
            m = v.get("None") or v.get("null")
            if not m:
                out.append(f"| {L} | {nm} | {v.get('status', '–')} | | | |")
                continue
            out.append(f"| {L} | {nm} | {f(m.get('dev_removal'))} | {f(m.get('sys_removal'))} | {f(m.get('S_aerobic_pct'))} | {f(m.get('S0_g'))} |")
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d, mo, tm = load(p)
        print(f"## {p}\n")
        print(table1(d))
        print("\n" + table2(mo))
        print("\n" + table_modes_metrics(mo, NAMES[:3]))
        print("\n시간(초):", json.dumps(tm.get("timing_s", {}), ensure_ascii=False), "총", f(tm.get("total_s"), 5))
