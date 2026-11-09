"""예시 3MF 생성(K1·K5·항목 6) — 재현 가능하게 스크립트로 남긴다.

caisson_down.3mf: 케이슨 — 불포화대 적용형(하향 침투). 치수는 Part 2 계산기(geometry_design.caisson_calc)가 정한다.
  반응재 질량 = 목표 황 질량(B1). 기본 = 영구 주입공이 가장 촘촘한 수렴 격자(44×44×72)에서 실제로 전달한 황 질량
  (v4.3 우물 모델, 입자 200 µm — 체거름으로 충전재 고리가 차서 6.18 g S에서 막힘; 22·33·44 격자 모두 같음).
  두께 = max(체류시간 계산값, 3D 프린터 최소 두께) — 격자 칸은 쓰지 않는다(B3). 상판은 지표면에 맞춤.
  반응재 O2 농도(모드 A의 f_O2) = K_O2와 같게 둠(f_O2 = 0.5, 가정) — 실제 값은 시뮬레이션 결과로 바꾼다.
  투수성 판 구멍: 지름 = 프린터 최소 구멍, 개구율 20 %, 판 두께 3 mm, 채움 = 반응재/흙 (모두 가정, 사용자 수정).
실행: python make_samples.py  → samples/caisson_down.3mf, samples/caisson_down.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import design as DZ                         # noqa: E402
import geometry_design as GD                # noqa: E402
from engine import moist_cfg                # noqa: E402
from params import GRID_PRESETS, REGISTRY, SOIL_TEXTURES, Common   # noqa: E402

PRINTER = {"min_wall_m": 0.002, "min_hole_m": 0.002}                 # 사용자 입력(예시 값, 가정)
MEDIA_MIN_THK_M = 0.04      # 반응재 층 최소 두께 입력(가정, 사용자 수정): 4 cm — 체류시간 값(약 5 mm)만으로 정하면 지름 50 cm의 얇은 판이 되어 케이슨이 아니게 됨
PLATE = {"hole_d_m": 0.002, "open_ratio": 0.20, "t_p_m": 0.003, "fill": "반응재/흙"}


TARGET_S_G = 851.0        # 목표 황 질량 g S (B1) — 출처: runs/p7 T-grid 10 µm 결과(22 격자 847.2, 33 격자 855.1 g S의 평균; 44 격자는 미완료)
TARGET_S_SRC = "영구 주입공 전달 황 질량, 주입 입자 10 µm(계획 2.84 L 전량 전달), 22·33 격자 평균, v4.3 우물 모델. 200 µm 입자는 충전재 고리 용량 6.18 g S에서 막혀 케이슨이 격자에 안 잡히는 크기(지름 4 cm)가 되므로 기본값으로 쓰지 않음"


def build(S_delivered_g=TARGET_S_G, c=None):
    c = c or Common()
    nx, ny, nz = GRID_PRESETS["가벼움"]
    dz = (c.H_soil + c.H_aq) / nz
    zone = GD.depth_zone(c)
    comp = DZ.solids_composition(c.ratio_S_CaCO3, "없음", 0.0)
    fS = 1.0 / comp["m_solid_per_gS"]                                 # 고형분 중 황 질량분율
    mass_g = S_delivered_g / fS
    inp = dict(A_tank_m2=c.Lx * c.Ly, zone_len_m=zone["length_m"], mass_g=mass_g, porosity=c.packed_porosity,
               S_mass_frac=fS, C0_mgN=c.NO3_in_mgN, q_mm_day=c.q_mm_day, T_days=180.0, SF=2.0, target=0.9,
               S_media=0.5, d_S_m=REGISTRY["S_d_media"][0], K_soil_m_s=SOIL_TEXTURES[c.texture]["Ks"])
    t_min = max(PRINTER["min_wall_m"], MEDIA_MIN_THK_M)                  # B3: max(체류시간 값, 최소 두께 입력) — 격자 칸 아님
    cal = GD.caisson_calc(inp, moist_cfg(c), t_min_m=t_min, C_O2_molm3=REGISTRY["K_I_O2"][0])
    rowA = next(r for r in cal["rows"] if r["모드"].startswith("A"))
    if rowA.get("상태") != "OK":
        raise ValueError(f"계산기 모드 A 결과가 쓸 수 없습니다: {rowA}")
    D, thk = float(rowA["지름 m(원형)"]), float(rowA["두께 m"])
    top_depth = 0.0                                                     # 상판 = 지표(B3: 격자와 무관)
    cd = GD.caisson_down(D, thk, top_depth, PRINTER["min_wall_m"], PLATE["t_p_m"], zone)
    res = GD.resolution_check(D, c.Lx / nx)
    info = {"zone": zone, "calc_input": inp, "t_min_m": t_min, "calc_rows": cal["rows"], "basis": cal["basis"],
            "dims": cd["dims"], "plate": PLATE, "printer": PRINTER, "resolution_light": res,
            "capture_pct": 3.14159 * D * D / 4 / (c.Lx * c.Ly) * 100, "S_delivered_g": S_delivered_g,
            "target_S_src": TARGET_S_SRC,
            "note": "Part 2 계산기 제안값(모드 A, SF 2). 사용자가 고친 값이 우선."}
    return cd, info


if __name__ == "__main__":
    cd, info = build()
    with open(os.path.join(HERE, "samples", "caisson_down.3mf"), "wb") as f:
        f.write(cd["bytes"])
    with open(os.path.join(HERE, "samples", "caisson_down.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(info["dims"], indent=1), info["capture_pct"], info["resolution_light"])
