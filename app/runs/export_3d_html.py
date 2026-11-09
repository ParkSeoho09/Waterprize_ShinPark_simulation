"""앱 결과 탭과 같은 3D 애니메이션(plotly) HTML을 저장된 결과(.pkl)에서 만든다. plotly가 설치된 PC(앱이 돌아가는 환경)에서 실행.

  python runs/export_3d_html.py runs/cmp/B44_o2z_full.pkl runs/fig/B44_o2z

만드는 것(앱 탭 7 '애니메이션 HTML 만들기'와 같은 함수)
  3d_비교_애니메이션.html      viz.compare_view: 대조군·비교군 3D 나란히 + 아래 지표 그래프, 시간 슬라이더 하나로 연동
  3d_설치과정_<비교군>.html    viz.install_animation: 구조물 삽입 → 개구부 열기 → 주입 → 정치, ▶ 재생 버튼
  3d_결과_<비교군>.html        viz.results_scene: 운영 기간 스냅숏(반응재·NO₃ 플룸·pH) 슬라이더
"""
from __future__ import annotations

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import viz                                    # noqa: E402
from compare_run import build_scenarios, make_common   # noqa: E402


def main(pkl, out):
    os.makedirs(out, exist_ok=True)
    import gzip
    R = pickle.load(gzip.open(pkl, "rb") if pkl.endswith(".gz") else open(pkl, "rb"))
    a = R.get("args", {})
    c = make_common(a.get("days", 365), a.get("grid", "22x22x44"), a.get("d_p", 5e-6))
    if a.get("o2_zero"):
        c.params["DO_in"] = 0.0; c.o2_gas = False
    scs = {s.name: s for s in build_scenarios(c, a.get("run", "B"))}
    objs = {r["name"]: (scs[r["name"]].objects, scs[r["name"]].roles, scs[r["name"]].placement)
            for r in R["scen"] if r["name"] in scs}
    all_res = list(R["scen"]) + ([R["control"]] if R.get("control") else [])
    fig = viz.compare_view(all_res, c, ["반응재 분포", "NO₃ 오염 플룸"], light=True, scen_objects=objs)
    open(os.path.join(out, "3d_비교_애니메이션.html"), "w", encoding="utf-8").write(viz.to_html(fig, sync_cameras=True))
    for r in R["scen"]:
        if r["name"] not in objs:
            continue
        o, rl, pl = objs[r["name"]]
        f = viz.install_animation(r, o, rl, pl, light=True)
        open(os.path.join(out, f"3d_설치과정_{r['name']}.html"), "w", encoding="utf-8").write(viz.to_html(f))
        f = viz.results_scene(r, c, ["반응재 분포", "NO₃ 오염 플룸"], objects=o, roles=rl, placement=pl, light=True)
        open(os.path.join(out, f"3d_결과_{r['name']}.html"), "w", encoding="utf-8").write(viz.to_html(f))
    print("저장:", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
