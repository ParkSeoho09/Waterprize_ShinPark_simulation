"""가는 스크린·개구부의 우물 모델(v4.3 Part A) — 격자에 무관한 주입.

스크린을 지나는 셀 i 마다 Peaceman 우물 지수
    WI_i = 2π·K_i·Δz_i / (ln(r_o/r_w) + s)                         [m²/s],  q_i = WI_i·(H_w − H_i)
  r_o = 0.28·√(√(ky/kx)·dx² + √(kx/ky)·dy²) / ((ky/kx)^¼ + (kx/ky)^¼)   (Peaceman 1983; 등방 정사각 격자에서 ≈ 0.198·dx,
        Peaceman 1978 — 원문 확인 필요)
  r_w = 스크린 실제 반지름(메시에서), Δz_i = 스크린과 셀의 연직 겹침 길이, K_i = 셀의 수평 Ks(주입이면 상류 = 우물, kr = 1).
  s = 스킨(직렬 저항, 무차원):  충전재 고리 s_pack = (K/K_pack − 1)·ln(r_pack/r_w)
                                필터 케이크 s_cake = (K/K_cake − 1)·ln(r_c/r_in)   (Hawkins 형, 가정)
                                스크린 슬롯   s_slot = 2π·K·L·R_slot   (plate.py 구멍 저항 R_h를 병렬로 묶은 값)
스크린 표면적은 셀 면적이 아니라 실제 원통 표면적 2π·r_w·L이다(슬롯 저항 계산에 씀).
펌프: 유량 지정이면 H_w를 Σq_i = Q 로 정하고(셀 수두에 대한 1차 관계라 야코비안에 밀집 블록으로 들어감),
      압력 제한이면 H_w = 지표 수두 + p/(ρg) 로 고정.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from voxel import to_tank

RHO_G = 998.0 * 9.81


def kozeny_carman_K(d_m, phi, T_C=20.0):
    """구형 입자 충전층 K = (ρg/μ)·d²·φ³/(180(1−φ)²) [m/s]."""
    mu = 2.414e-5 * 10 ** (247.8 / (T_C + 273.15 - 140.0))
    return RHO_G / mu * d_m ** 2 * phi ** 3 / (180.0 * (1 - phi) ** 2)


def peaceman_ro(dx, dy, kx=1.0, ky=1.0):
    a = np.sqrt(ky / kx); b = np.sqrt(kx / ky)
    return 0.28 * np.sqrt(a * dx * dx + b * dy * dy) / ((ky / kx) ** 0.25 + (kx / ky) ** 0.25)


def _radius_about(Vt, cx, cy):
    return float(np.max(np.hypot(Vt[:, 0] - cx, Vt[:, 1] - cy)))


def find_wells(objects, roles: Dict[str, dict], placement, open_role="주입 개구부") -> List[dict]:
    """주입 개구부 객체들을 연직 우물(축·r_w·깊이 구간)로 묶는다. 같은 축(중심 거리 < r_w)에 있는 개구부는 한 우물의 여러 스크린.
    같은 축의 가는 완결 객체(충전재·실링·케이싱·관 끝·좁은 되메움, 반지름 ≤ 2·r_pack)는 우물 모델이 대신하므로
    복셀에서 뺀다('group'). 충전재(투수 충전재) 반지름·K는 스킨에 쓴다."""
    tanks = {o.name: to_tank(o.vertices_m, placement) for o in objects}
    screens = [o for o in objects if roles.get(o.name, {}).get("role") == open_role]
    wells: List[dict] = []
    for o in screens:
        Vt = tanks[o.name]
        lo, hi = Vt.min(axis=0), Vt.max(axis=0)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        rw = _radius_about(Vt, cx, cy)
        seg = {"name": o.name, "z0": float(lo[2]), "z1": float(hi[2])}
        for w in wells:
            if np.hypot(w["x"] - cx, w["y"] - cy) < max(w["r_w"], rw):
                w["screens"].append(seg); break
        else:
            wells.append({"x": cx, "y": cy, "r_w": rw, "screens": [seg]})
    for w in wells:
        co = []
        for o in objects:
            Vt = tanks[o.name]
            lo, hi = Vt.min(axis=0), Vt.max(axis=0)
            cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
            if np.hypot(cx - w["x"], cy - w["y"]) < w["r_w"]:
                co.append((o, _radius_about(Vt, w["x"], w["y"])))
        packs = [(o, r) for o, r in co if roles.get(o.name, {}).get("role") in ("투수 충전재", "저투수 실링")]
        r_pack = max([r for _, r in packs], default=w["r_w"])
        w["r_pack"] = r_pack
        sand = [(o, r) for o, r in packs if roles.get(o.name, {}).get("role") == "투수 충전재"]
        w["pack_names"] = [o.name for o, _ in sand]
        w["group"] = [o.name for o, r in co if r <= 2.0 * r_pack + 1e-9]
        w["L"] = float(sum(s["z1"] - s["z0"] for s in w["screens"]))
    return wells


def well_cells(w, g, active=None):
    """우물 축이 지나는 셀 열에서 스크린과 겹치는 셀과 겹침 길이 Δz (m). 반환 (전체 인덱스 배열, Δz 배열)."""
    ix = int(np.clip(np.floor(w["x"] / g.dx), 0, g.nx - 1))
    iy = int(np.clip(np.floor(w["y"] / g.dy), 0, g.ny - 1))
    cells, dzs = [], []
    for k in range(g.nz):
        top, bot = k * g.dz, (k + 1) * g.dz
        ov = 0.0
        for s in w["screens"]:
            ov += max(0.0, min(bot, s["z1"]) - max(top, s["z0"]))
        if ov > 1e-12:
            idx = (ix * g.ny + iy) * g.nz + k
            if active is None or active[idx]:
                cells.append(idx); dzs.append(ov)
    return np.array(cells, int), np.array(dzs, float)


def well_index(w, g, cells, dzs, K_soil_cells, K_pack=None, K_cake=None, V_cake=0.0, phi_cake=0.4, R_slot=None):
    """셀별 우물 지수 WI [m²/s]와 진단. 우물에서 블록 등가 반지름 r_o까지의 방사 저항을 층별로 더한다:
        R = ln(min(r_o, r_pack)/r_w)/K_pack + ln(max(r_o, r_pack)/r_pack)/K + (1/K_cake − 1/K)·ln(r_c/r_pack) + R_slot'
    (r_pack까지 충전재, 그 바깥 흙, 케이크는 충전재 바깥 면에서 흙 쪽으로 자람 — 가정). WI = 2π·Δz / R.
    r_o > r_pack이면 스킨 표현 s_pack = (K/K_pack − 1)·ln(r_pack/r_w)와 같다."""
    ro = peaceman_ro(g.dx, g.dy)
    rw, rp = w["r_w"], w["r_pack"]
    if ro <= rw:
        raise ValueError(f"격자 칸이 너무 작아 우물 모델을 쓸 수 없습니다: r_o = {ro * 1e3:.2f} mm ≤ 스크린 반지름 {rw * 1e3:.2f} mm "
                         "(국소 세분화 또는 더 큰 격자 필요).")
    K = np.asarray(K_soil_cells, float)
    R = np.zeros_like(K)
    if K_pack and rp > rw:
        R += np.log(min(ro, rp) / rw) / K_pack + (np.log(ro / rp) / K if ro > rp else 0.0)
    else:
        R += np.log(ro / rw) / K
    r_c = rp
    if V_cake > 0 and K_cake:
        r_c = float(np.sqrt(rp ** 2 + V_cake / (np.pi * w["L"] * (1 - phi_cake))))
        R += (1.0 / K_cake - 1.0 / K) * np.log(r_c / rp)
    if R_slot:
        R += R_slot * 2 * np.pi * w["L"]
    R = np.maximum(R, 1e-12)
    WI = 2 * np.pi * dzs / R
    s_eq = K * R - np.log(ro / rw)                 # 등가 스킨(진단)
    return WI, {"r_o": ro, "r_cake": r_c, "skin_mean": float(np.mean(s_eq))}


def capacity_solids(w, phi_pack, phi_cake):
    """필터 케이크(체거름)가 쌓일 수 있는 고형분 부피 m³: 충전재 고리 공극(r_w~r_pack) + 스크린 안(r < r_w), 케이크 공극률 φ_c.
    체거름 입자는 흙 공극에 못 들어가므로 이보다 많이 넣으면 우물이 막힌다(격자와 무관한 물리 한계)."""
    ann = np.pi * (w["r_pack"] ** 2 - w["r_w"] ** 2) * w["L"] * (phi_pack if w["r_pack"] > w["r_w"] else 0.0)
    inner = np.pi * w["r_w"] ** 2 * w["L"]
    return float((ann + inner) * (1 - phi_cake))


def radial_rho(w, r_out, K_soil, K_pack=None, K_cake=None, fill=0.0):
    """우물(r_w)에서 반지름 r_out까지의 방사 저항 ρ = Σ ln(r2/r1)/K  [s/m]. 케이크는 충전재 고리를 바깥(흙 쪽)에서
    안쪽으로 채운다(fill = 고리 채움 비율 0~1, 가정)."""
    rw, rp = w["r_w"], w["r_pack"]
    rho = 0.0
    if rp > rw and K_pack:
        r_ci = np.sqrt(rp ** 2 - fill * (rp ** 2 - rw ** 2))
        rho += np.log(max(r_ci, rw) / rw) / K_pack
        if fill > 0 and K_cake:
            rho += np.log(rp / max(r_ci, rw)) / K_cake
    rho += np.log(r_out / rp) / K_soil
    return rho


def face_R(w, g, cells, dzs, K_open, K_soil, K_pack=None, K_cake=None, fill=0.0):
    """우물 셀(개구부)의 수평 면 4개 직렬 저항 R [s] — 우물에서 이웃 셀 중심까지 방사 해와 같은 전도도가 되도록:
        C_k = 2π·Δz_k/ρ(r_n),  r_n = √(dx·dy) (이웃 중심 거리),  면 하나 T = C_k/4 = A/(d/(2K_open) + d/(2K_s) + R)
    → R = A·4/C_k − d/(2K_open) − d/(2K_s). 음수가 되면(격자가 충전재보다 가늘거나 비슷) 0으로 두고 경고."""
    rn = np.sqrt(g.dx * g.dy)
    if rn <= w["r_pack"]:
        raise ValueError(f"격자 칸 {rn * 1e3:.1f} mm가 우물 완결부 반지름 {w['r_pack'] * 1e3:.1f} mm 이하라 우물 모델을 쓸 수 없습니다.")
    Ks = np.asarray(K_soil, float)
    rho = np.array([radial_rho(w, rn, k, K_pack, K_cake, fill) for k in Ks])
    C = 2 * np.pi * dzs / rho
    out = {}
    clipped = 0
    for ax, (A, d) in enumerate(((g.dy * g.dz, g.dx), (g.dx * g.dz, g.dy))):
        R = A * 4.0 / C - d / (2 * K_open) - d / (2 * Ks)
        clipped += int((R < 0).sum())
        out[ax] = np.maximum(R, 0.0)
    return out, {"C_total": float(C.sum()), "clipped": clipped, "r_n": rn}
