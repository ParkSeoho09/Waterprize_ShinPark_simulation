"""compare_run.py --save-full 로 저장한 <out>.pkl 에서 그림(PNG)과 애니메이션(GIF)을 만든다.

  python runs/make_figures.py runs/cmp/B44_o2z_full.pkl runs/fig/B44_o2z

만드는 것
  fig1_구조_배치.png        비교군별 구조물·반응재 위치(평면도 + 단면도)
  fig2_시계열.png           바닥 유출 NO3 농도, 일별 η, 누적 탈질, 지하수 도달 N
  fig3_최종상태.png         운영 마지막 날 단면: NO3 / pH / SO4 / 남은 황
  fig4_깊이별_황잔량.png     깊이별 남은 황(초기 대비 %) — 재보충 깊이 선택용
  anim1_설치_<비교군>.gif    설치 단계(삽입→개구부 열기→주입→정치) 단면: 반응재 고체·부유 슬러리
  anim2_NO3.gif             운영 기간 NO3 단면(대조군 + 3 비교군)
  anim3_pH.gif              운영 기간 pH 단면
  anim4_황잔량.gif           운영 기간 남은 황 단면(초기 대비)
단면은 비교군마다 반응재 고체가 가장 많은 y 위치의 x–깊이 면이다(제목에 y 표시).
"""
from __future__ import annotations

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib import font_manager                   # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402

for f in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

COL = {"대조군(구조물 없음)": "#6b6b6b", "신박 4관 마름모": "#1f6fb2", "영구 주입공": "#d4781c", "케이슨(하향)": "#2f9a5a"}


def _g(res):
    g = res["grid"]
    return g["nx"], g["ny"], g["nz"], g["Lx"], g["Ly"], g["H"]


def _slice_j(res):
    """반응재 고체가 가장 많은 y 인덱스(대조군은 가운데)."""
    nx, ny, nz, *_ = _g(res)
    sn = res.get("snaps") or []
    if res.get("is_control") or not sn:
        return ny // 2
    s = np.nan_to_num(sn[0]["solid"]).sum(axis=(0, 2))
    return int(np.argmax(s)) if s.max() > 0 else ny // 2


def _xz(arr, j, active=None):
    a = np.array(arr[:, j, :], float)
    if active is not None:
        a[~active[:, j, :]] = np.nan
    return a.T                                           # (nz, nx): 행 = 깊이


def _ext(res):
    nx, ny, nz, Lx, Ly, H = _g(res)
    return [0, Lx, H, 0]


def order(R):
    runs = ([R["control"]] if R.get("control") else []) + list(R["scen"])
    return runs


def fig_structure(R, out):
    sc = R["scen"]
    fig, ax = plt.subplots(2, len(sc), figsize=(4.2 * len(sc), 7.6), squeeze=False)
    for i, r in enumerate(sc):
        nx, ny, nz, Lx, Ly, H = _g(r)
        obj = np.asarray(r["obj_cell"]); reac = np.asarray(r["reac0"])
        st = obj >= 0
        top = np.where(reac.any(axis=2), 2, np.where(st.any(axis=2), 1, 0)).T
        a = ax[0, i]
        a.imshow(top, origin="lower", extent=[0, Lx, 0, Ly], cmap=matplotlib.colors.ListedColormap(["#f3efe6", "#8a8a8a", COL.get(r["name"], "#c33")]), vmin=0, vmax=2)
        j = _slice_j(r)
        a.axhline((j + 0.5) * Ly / ny, color="k", lw=0.8, ls="--")
        a.set_title(f"{r['name']} — 평면(지표에서 내려다봄)", fontsize=10)
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
        side = np.where(reac[:, j, :], 2, np.where(st[:, j, :], 1, 0)).T
        b = ax[1, i]
        b.imshow(side, extent=_ext(r), aspect="auto", cmap=matplotlib.colors.ListedColormap(["#f3efe6", "#8a8a8a", COL.get(r["name"], "#c33")]), vmin=0, vmax=2)
        b.set_title(f"단면 y = {(j + 0.5) * Ly / ny:.2f} m (점선 위치)", fontsize=10)
        b.set_xlabel("x (m)"); b.set_ylabel("깊이 (m)")
    fig.text(0.5, 0.005, "회색 = 구조물 셀, 색 = 운영 시작 때 반응재가 있는 셀", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(os.path.join(out, "fig1_구조_배치.png")); plt.close(fig)


def fig_timeseries(R, out):
    es = R.get("eval_start")
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for r in order(R):
        D = r["daily"]; d = np.asarray(D["day"], float)
        if len(d) == 0:
            continue
        d = d - d[0]
        col = COL.get(r["name"], None)
        ax[0, 0].plot(d, D["Cout_mgN"], color=col, lw=1.3, label=r["name"])
        ax[0, 1].plot(d, np.asarray(D["eta"], float) * 100, color=col, lw=1.0, label=r["name"])
        den = np.asarray(D["denit_g"], float); ax[1, 0].plot(d, den - den[0], color=col, lw=1.3, label=r["name"])
        ngw = np.asarray(D["N_to_gw_g"], float); ax[1, 1].plot(d, ngw - ngw[0], color=col, lw=1.3, label=r["name"])
    ax[0, 0].set_ylabel("바닥 유출 NO3 (mg N/L)"); ax[0, 0].set_title("바닥(지하수 쪽) 유출 농도")
    ax[0, 1].set_ylabel("일별 η = 1 − 유출/유입 농도 (%)"); ax[0, 1].set_title("일별 제거효율(돌파 전 값은 의미 없음)")
    ax[0, 1].set_ylim(-20, 100)
    ax[1, 0].set_ylabel("누적 탈질 (g N)"); ax[1, 0].set_title("누적 탈질량")
    ax[1, 1].set_ylabel("누적 지하수 도달 N (g)"); ax[1, 1].set_title("지하수로 내려간 질소")
    for a in ax.ravel():
        a.set_xlabel("운영 일수 (일)"); a.grid(alpha=0.3)
        if es is not None:
            a.axvline(es, color="k", ls=":", lw=0.9)
    ax[0, 0].legend(fontsize=8)
    fig.text(0.5, 0.005, f"점선 = 대조군 돌파일(η 평가 시작, {es:.0f}일)" if es is not None else "", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(os.path.join(out, "fig2_시계열.png")); plt.close(fig)


def _s_rel(r, sn):
    s0 = np.asarray(r["snaps"][0]["S_mol"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(s0 > s0.max() * 1e-4, np.asarray(sn["S_mol"], float) / s0 * 100, np.nan)
    return rel


def fig_end(R, out):
    runs = order(R)
    rows = [("NO3", "NO3 (mg N/L)", "viridis", 0, 60), ("pH", "pH", "RdYlBu", 4, 9),
            ("SO4", "SO4 (mg/L)", "magma", 0, None), ("Srel", "남은 황 (초기 대비 %)", "YlOrBr", 0, 100)]
    smax = max(float(np.nanpercentile(r["snaps"][-1]["SO4"], 99.5)) for r in runs if r.get("snaps"))
    fig, ax = plt.subplots(len(rows), len(runs), figsize=(3.6 * len(runs), 2.9 * len(rows)), squeeze=False)
    for i, r in enumerate(runs):
        if not r.get("snaps"):
            continue
        sn = r["snaps"][-1]; j = _slice_j(r); act = np.asarray(r["active"])
        for k, (key, lab, cm, lo, hi) in enumerate(rows):
            a = ax[k, i]
            if key == "Srel":
                if r.get("is_control"):
                    a.axis("off"); continue
                arr = _s_rel(r, sn)
            else:
                arr = sn[key]
            im = a.imshow(_xz(arr, j, act), extent=_ext(r), aspect="auto", cmap=cm, vmin=lo, vmax=(hi if hi is not None else smax))
            if k == 0:
                a.set_title(f"{r['name']}\n(y = {(j + 0.5) * r['grid']['Ly'] / r['grid']['ny']:.2f} m)", fontsize=9)
            a.set_xlabel("x (m)", fontsize=8); a.set_ylabel("깊이 (m)", fontsize=8); a.tick_params(labelsize=7)
            if i == len(runs) - 1 or (key == "Srel" and i == len(runs) - 1):
                fig.colorbar(im, ax=a, fraction=0.05).set_label(lab, fontsize=8)
    fig.suptitle(f"운영 {runs[0]['snaps'][-1]['op_day']:.0f}일째 단면", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_최종상태.png")); plt.close(fig)


def fig_depth(R, out):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    rows = []
    for r in R["scen"]:
        if not r.get("snaps"):
            continue
        nx, ny, nz, Lx, Ly, H = _g(r)
        z = (np.arange(nz) + 0.5) * H / nz
        s0 = np.asarray(r["snaps"][0]["S_mol"], float).sum(axis=(0, 1))
        s1 = np.asarray(r["snaps"][-1]["S_mol"], float).sum(axis=(0, 1))
        m = s0 > s0.max() * 1e-3
        rel = np.where(m, s1 / np.where(m, s0, 1) * 100, np.nan)
        ax.plot(rel, z, "o-", color=COL.get(r["name"]), label=r["name"], ms=3)
        for k in np.where(m)[0]:
            vc = Lx / nx * Ly / ny * H / nz                     # S_mol은 mol/m³ → 셀 부피를 곱해 g
            rows.append((r["name"], float(z[k]), float(s0[k] * vc * 32.06), float(s1[k] * vc * 32.06), float(rel[k])))
    ax.invert_yaxis(); ax.set_xlabel("남은 황 (운영 시작 대비 %)"); ax.set_ylabel("깊이 (m)")
    ax.axvline(25, color="r", ls=":", lw=0.9); ax.text(26, ax.get_ylim()[0] * 0.98, "재보충 기준 25 %", color="r", fontsize=8, va="bottom")
    ax.set_xlim(0, 105); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"깊이별 남은 황 — 운영 {R['scen'][0]['snaps'][-1]['op_day']:.0f}일째", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_깊이별_황잔량.png")); plt.close(fig)
    with open(os.path.join(out, "fig4_깊이별_황잔량.csv"), "w", encoding="utf-8-sig") as f:
        f.write("비교군,깊이_m,시작_gS,마지막_gS,남은비율_%\n")
        for row in rows:
            f.write("{},{:.3f},{:.2f},{:.2f},{:.1f}\n".format(*row))


def anim_install(r, out, fps=3):
    fr = r.get("frames") or []
    if not fr:
        return
    j = _slice_j(r); act = np.asarray(r["active"])
    smax = max(float(np.nanmax(f["solid"])) for f in fr) or 1.0
    pmax = max(float(np.nanmax(f["P"])) for f in fr) or 1.0
    obj = np.asarray(r["obj_cell"]) >= 0
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ims = []
    for a, key, cm, vmax, lab in ((ax[0], "solid", "YlOrBr", smax, "반응재 고체 (부피 분율)"),
                                  (ax[1], "P", "PuBu", pmax, "부유 슬러리 입자 (mol/m³)")):
        im = a.imshow(_xz(fr[0][key], j, act), extent=_ext(r), aspect="auto", cmap=cm, vmin=0, vmax=vmax)
        a.contour(np.linspace(0, r["grid"]["Lx"], r["grid"]["nx"]), np.linspace(0, r["grid"]["H"], r["grid"]["nz"]),
                  obj[:, j, :].T.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        a.set_xlabel("x (m)"); a.set_ylabel("깊이 (m)")
        fig.colorbar(im, ax=a, fraction=0.046).set_label(lab, fontsize=8)
        ims.append((im, key))
    ttl = fig.suptitle("")

    def up(k):
        f = fr[k]
        for im, key in ims:
            im.set_data(_xz(f[key], j, act))
        t_min = (f["t"] - fr[0]["t"]) * 1440
        ttl.set_text(f"{r['name']} 설치 단계 {k + 1}/{len(fr)}: {f['label']}  (경과 {t_min:.0f}분)")
        return [im for im, _ in ims] + [ttl]
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    FuncAnimation(fig, up, frames=len(fr), blit=False).save(os.path.join(out, f"anim1_설치_{r['name']}.gif"), writer=PillowWriter(fps=fps))
    plt.close(fig)


def anim_field(R, out, key, fname, lab, cm, lo, hi, fps=6, scen_only=False):
    runs = R["scen"] if scen_only else order(R)
    runs = [r for r in runs if r.get("snaps")]
    n = min(len(r["snaps"]) for r in runs)
    fig, ax = plt.subplots(1, len(runs), figsize=(3.8 * len(runs), 4.0), squeeze=False)
    ims = []
    for i, r in enumerate(runs):
        j = _slice_j(r); act = np.asarray(r["active"])
        sn = r["snaps"][0]
        arr = _s_rel(r, sn) if key == "Srel" else sn[key]
        im = ax[0, i].imshow(_xz(arr, j, act), extent=_ext(r), aspect="auto", cmap=cm, vmin=lo, vmax=hi)
        ax[0, i].set_title(r["name"], fontsize=9); ax[0, i].set_xlabel("x (m)", fontsize=8)
        ax[0, i].tick_params(labelsize=7)
        ims.append((im, r, j, act))
    ax[0, 0].set_ylabel("깊이 (m)")
    fig.colorbar(ims[-1][0], ax=ax[0, -1], fraction=0.05).set_label(lab, fontsize=8)
    ttl = fig.suptitle("")

    def up(k):
        for im, r, j, act in ims:
            sn = r["snaps"][k]
            im.set_data(_xz(_s_rel(r, sn) if key == "Srel" else sn[key], j, act))
        ttl.set_text(f"{lab} — 운영 {runs[0]['snaps'][k]['op_day']:.0f}일째")
        return [x[0] for x in ims] + [ttl]
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    FuncAnimation(fig, up, frames=n, blit=False).save(os.path.join(out, fname), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main(pkl, out):
    os.makedirs(out, exist_ok=True)
    R = pickle.load(open(pkl, "rb"))
    fig_structure(R, out); print("fig1")
    fig_timeseries(R, out); print("fig2")
    fig_end(R, out); print("fig3")
    fig_depth(R, out); print("fig4")
    for r in R["scen"]:
        anim_install(r, out); print("anim1", r["name"])
    anim_field(R, out, "NO3", "anim2_NO3.gif", "NO3 (mg N/L)", "viridis", 0, 60); print("anim2")
    anim_field(R, out, "pH", "anim3_pH.gif", "pH", "RdYlBu", 4, 9); print("anim3")
    anim_field(R, out, "Srel", "anim4_황잔량.gif", "남은 황 (초기 대비 %)", "YlOrBr", 0, 100, scen_only=True); print("anim4")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
