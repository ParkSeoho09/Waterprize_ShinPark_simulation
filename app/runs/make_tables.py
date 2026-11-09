"""README 표 생성: 수정 전후(X11·C2·C3), 주입 격자 시험(T-grid), 격자별 시간·메모리. 사용: python runs/make_tables.py"""
import json, os, glob
R = os.path.dirname(os.path.abspath(__file__))
def L(p):
    try: return json.load(open(os.path.join(R, p), encoding="utf-8"))
    except Exception: return None
def f(x, n=3):
    if x is None: return "–"
    if isinstance(x, bool): return "예" if x else "아니오"
    if isinstance(x, str): return x
    return f"{x:.{n}g}"
out = []
b = L("slad_baseline_v4.1_before.json")
v42 = L("slad_v4.2_after.json"); v43 = L("slad_v4.3_after.json"); v43d = L("slad_v4.3_after_d10um.json")
out.append("| 영구 주입공 (90일, 22×22×36) | v4.1 수정 전 | v4.2 A / B / C | v4.3 A / B / C (200 µm) | v4.3 A / B / C (10 µm) |")
out.append("|---|---|---|---|---|")
def trio(d, key, name="영구 주입공", n=3):
    if not d: return "–"
    vals = []
    for m in "ABC":
        r = next((x for x in d["runs"] if x["mode"] == m and x["name"].startswith(name)), None)
        vals.append(f(r.get(key) if r else None, n))
    return " / ".join(vals)
bc = (b or {}).get("core", {})
rows = [("확산 반경 r (cm)", "r_cm", None), ("반응 구역 등가 반경 (cm)", "반응 구역 등가 반경 r cm", None),
        ("전달률 (황, %)", "delivery_pct", None), ("주입 전달률 (부피, %)", "주입 전달률 %(부피)", None),
        ("반응재 질량 (g)", "reactive_g", None), ("황 전달 (g S)", "S0_g", "S0_g"), ("denit_g (g N)", "denit_g", "denit_g"),
        ("η 반응 기여 (%)", "eta_denit", None), ("η 흐름·저장 기여 (%p)", "eta_flow_storage", None),
        ("지하수 도달 N (g)", "N_to_gw_g", None), ("황 사용 — 탈질 (g)", "S_denit_g", None), ("황 사용 — 공기 산화 (g)", "S_aerobic_g", None),
        ("장치 제거율 (%)", "core_dev_removal", None), ("도메인 η (%)", "core_sys_removal", None), ("포획 분율 (%)", "core_capture_pct", None)]
for lab, k, bk in rows:
    bv = (b or {}).get(bk) if bk else None
    out.append(f"| {lab} | {f(bv)} | {trio(v42, k)} | {trio(v43, k)} | {trio(v43d, k)} |")
out.append("")
out.append("| 케이슨(하향) 같은 지표 | v4.3 A / B / C (200 µm 기준 목표 황) |")
out.append("|---|---|")
for lab, k, _ in rows:
    out.append(f"| {lab} | {trio(v43, k, '케이슨')} |")
out.append("")
if v43: out.append(f"v4.3 코드 해시 `{v43.get('code_hash')}`, 엔진 {v43.get('engine')}, 시드 {v43.get('seed')}")

out.append("")
out.append("**T-grid — 영구 주입공 첫 주입(계획 2.84 L), 주입 입자별·격자별**")
out.append("")
out.append("| 입자 | 격자 | 전달 (L) | 주입 시간 (분) | 최대 압력 (kPa) | 최대 유량 (L/min) | 막힘 | 황 전달 (g S) | 계산 시간 (초) | 최대 메모리 (MB) |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for dp, lab in (("2e-4", "200 µm"), ("1e-5", "10 µm")):
    for g in ("22x22x36", "33x33x54", "44x44x72"):
        d = L(os.path.join("..", "..", "runs", "p7", f"tg_{dp}_{g}.json"))
        if not d:
            out.append(f"| {lab} | {g.replace('x','×')} | 미완료(실행 중단) | – | – | – | – | – | – | – |"); continue
        a = lambda k: (d.get(k) or [None])[0] if isinstance(d.get(k), list) else d.get(k)
        out.append(f"| {lab} | {g.replace('x','×')} | {f(a('inj_delivered_L'),4)} | {f(a('inj_duration_min'),3)} | {f(a('inj_maxP_kPa'),3)} | {f(a('inj_Qmax_Lmin'),3)} | {f(a('blocked'))} | {f(d.get('S0_g'),4)} | {f(d.get('t_s'),4)} | {f(d.get('maxrss_MB'),4)} |")
print("\n".join(out))
