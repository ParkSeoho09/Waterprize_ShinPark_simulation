# CHANGELOG

## v4.4 (2026-11-09) — 얇은 개구부 처리, 계산 정체 수정, 실행·그림 스크립트

### 엔진
- 격자보다 얇은 주입 개구부(셀 체적이 메시 체적의 3배 초과)를 '빈 공간'(K 1e-2, α 50, n 3) 대신 흙 물성 그대로의 주입 출구로 둔다
  (`Common.port_thin_as_soil`, 기본 켬, 탭 ②에 체크박스). 수정 전: 10 mm 포트가 109 mm 셀로 부풀려져 흡입 −0.39 m에서
  θ≈0.03으로 말라 모세관 장벽이 됨 → 신박 4관 마름모 반응재 통과 유량 비 1.3e-6(우회). 수정 후 0.48(부분 우회).
  참고: 단계형 시뮬 01_Final_Simulation 의 점 출구 처리. 이전 방식은 `--port-void`로 재현.
- 이 출구 셀은 필터 케이크가 차도 수분특성을 흙 그대로 둔다. `phi_min` 대입을 `np.minimum`으로(배열 대입 오류).
- 주입 뒤 압력 해소 구간도 주입과 같은 Δt 하한·반복 허용을 쓰고, Δt가 10분 미만이면 반복 8~12회일 때도 Δt를 ×1.25씩 키움(Δt가 1초 아래에 갇히던 정체).
- 반복법이 두 번 넘게 LGMRES까지 간 중간 크기 계(3만~10만 셀)는 직접 분해를 먼저 씀(33×33×44 케이슨 11일째 정체 원인).
- 통과 유량 진단용 정상해에 300초 한도. 못 풀면 진단만 '판정 불가'.
- 재보충 기록에 계획 부피(`planned_L`)를 넣음. 없을 때 전달률이 NaN이라 4 L/26.5 L 주입도 '성공'으로 집계되던 문제.

### 실행·그림 (runs/)
- `compare_run.py`: 프롬프트(app_run_prompt.md) 값으로 세 비교군 + 대조군 실행. `--o2-zero`([가정] 용존 O₂ 0·토양 공기 O₂ 교환 끔,
  참고 시뮬과 같은 조건), `--port-void`, `--save-full`(스냅숏·일별 시계열·설치 프레임 전체를 .pkl로), `--snaps N`.
- `report_tables.py`: 표 1·수분 모드 순위표 출력.
- `make_figures.py`: .pkl → PNG 4종·GIF 4종(matplotlib). `export_3d_html.py`: 앱과 같은 plotly 3D HTML(plotly 있는 PC에서 실행).
- `FIGURES.md`: 그림·애니메이션을 만드는 경로 설명.

### 알려진 문제(고치지 않음)
- 장치 제거율(dev_removal) = 도메인 전체 탈질 ÷ 운영 시작 때 '황이 기준 이상인 셀'로 들어간 N. 분자와 분모의 범위가 달라
  얇게 퍼진 슬러리에서 100 %를 넘는다(산소 0 조건 279 %, 857 %). 순위는 도메인 η로 판단.

## v4.3 (2026-11-09) — v4.2 남은 문제 수정

규칙 R1~R9·화면 정책 유지. v4.2 전체는 git 태그 `v4.2`와 사용자 PC의 `app_backup_v4.2_20261109\`(연결되면 복사)에 보관.
시험 기준은 구현 전에 `selftest/selfcheck.py`에 적었다(TGRID_CRIT, GRID_CRIT 등). 사후 변경: S5는 기준(방향)은 그대로 두고
배치 시간을 24 h → 2 h로 바꿨다(24 h에서는 세 배합 모두 질산염이 다 없어져 비교가 안 됨 — 설정 오류 수정).

### Part A 주입량의 격자 의존 제거
- `well.py`(새 파일): 가는 스크린을 우물로 처리. 우물 축이 지나는 셀 열 = 개구부 셀, 그 수평 면 4개에 방사 해와 같은 전도도가 되는
  직렬 저항을 넣음(우물 r_w → 충전재 r_pack → 흙 → 이웃 셀 중심, Peaceman형 방사 보정). 스크린 끝 연직 면은 막음(케이싱·관 끝).
  같은 축의 가는 완결 객체(충전재·실링·케이싱·관 끝·좁은 되메움)는 복셀에서 빼고 우물 모델이 대신함.
- Peaceman 우물 지수 WI = 2πKΔz/ln(r_o/r_w)는 이 앱의 격자(22·33·44)에서 r_o(0.2·dx)가 충전재 반지름(3.8 mm)보다 작아져
  음의 지수가 되므로, 같은 방사 해를 "우물 셀–이웃 셀 면 저항"으로 넣는 방식으로 구현했다(원리 동일, 적용 범위: 격자 칸 > 완결부 지름).
- 체거름 입자(입자/토양 d50 > 0.154)의 필터 케이크는 셀 공극이 아니라 실제 용량(충전재 고리 공극 + 스크린 안)에 쌓이고, 차면
  우물이 막힌다. 케이크 투수계수는 Kozeny–Carman(입경·공극률), 고리를 채운 비율로 면 저항을 매 단계 갱신.
- 큰 격자에서만 타던 "주입 압력 사전 추정 → 처음부터 압력 제한" 경로를 우물 모델에서는 끔(격자마다 다른 경로를 타면 결과가 갈라짐).
- 주입 기록에 평균·최대 유량(L/min)과 막힘 여부 추가.

### Part B 공정성 기준을 격자와 분리
- 목표 황 질량 입력(`target_S_g`, 출처 `target_S_src`). 예시 케이슨은 영구 주입공(10 µm, 계획 2.84 L 전달)의 황 질량 851 g S에 맞추고 층 최소 두께 4 cm(입력, 가정) → 지름 17 cm. 200 µm 입자의 6.18 g S는 케이슨이 4 cm로 줄어 격자에 안 잡혀 기본값에서 뺌.
- 스크린 길이: 2 gpm/ft 규칙은 "현장 규모 참고값(흙수조 적용 불가)"으로만 표시. 기본 계산 = 기하 상사(현장 불포화대·스크린·
  케이싱 지름 입력 → 축척 계수), 가용 구간 0.25~1.0배 × 1~3구간 스윕, 케이싱 지름 = max(폭 비율, 프린터 최소).
- 케이슨 두께 = max(체류시간 계산값, 3D 프린터 최소 두께). "격자 2칸" 구조 최소 삭제, 상판 깊이도 격자와 무관(지표).

### Part C
- ⑧에 주입 입자 지름 inj_d_p(µm, 황 충전재 입경과 별개) 스윕: 5·10·15·20·50·200 µm, 그래프에 통과 한계 0.154·d50 세로선.
  각 값에서 주입 전달률·최대 압력·막힘·반응 구역 등가 반경 r·부피·황 질량·필터 케이크 두께를 표로 냄.
  0.154 = 2/√3 − 1: 정사면체로 맞닿은 같은 구 세 개 사이 틈을 지나는 가장 큰 구의 지름비(Herzig 외 1970 — 원문 확인 필요, 기하 값은 검산함).
- 수정 전후 비교 스크립트·결과: `runs/x11_baseline.py`, `runs/slad_*.json`(시드 12345, 코드 해시 기록).

### Part D
- D1(F6): PHREEQC 모드는 SOLUTION -temp로 입력 온도의 log K(T)를 이미 적용(확인). 옵션: 토양 유기탄소 종속영양 탈질(1차, 입력 시 켜짐,
  알칼리도 +1 eq/mol N). 양이온 교환(CEC)·토양 탄산염 완충·암모늄은 미구현(PHREEQC EXCHANGE 상태를 셀마다 저장해야 함).
- D2(F7): 산 생성 비교 시험(내장 1.28 vs PHREEQC 1.20 eq/mol N, 원인 = 생체 합성의 NH4⁺ 흡수).
- D3(F8): 수분 모드 D "혐기 미소부위"(흙덩이 반지름·O2 소비율 입력, 구 확산–0차 소비 정상해, Smith 1980형 — 원문 확인 필요).
- D4(F11): 아질산 2단계(NO3→NO2 전자 2개, NO2→N2 전자 3개로 양론 2:3 분할, 2단계/1단계 속도 비 입력). N 수지에 NO2 포함.
- D5(F12): 생물막 밀도 X_f 스윕(10~100 kg/m³). k½a ∝ √X_f (반-차수 = 생물막 내부 확산 제한 영역, Harremoës 1978 — 원문 확인 필요).
  탈착 계수는 미구현: 반-차수 영역에서 속도는 생물막 두께와 무관하고, 탈착이 영향을 주려면 두께가 침투 깊이보다 얇아지는 판정에
  수율·최대 비성장 속도 입력이 필요.
- D6(F10): 슬러리 고형분비·주입 부피·유량·펌프 압력·λ를 ⑧ 스윕에 추가, 설계 편집기 입력의 최솟값·최댓값 제한 제거, 정적 시험 D6.

### Part E 시험
- 추가: T-grid(A4), CAIS-grid(A5·B3), D2, S5, S6, S13, D6. S16은 비교 대상을 caisson_down으로 바꿔 WP_SLOW=1로 끝까지 실행.

### Part F 최적화
- cProfile(영구 주입공 22×22×36, 0.2일, 내장 화학, `runs/p7/grid.py … prof`): 총 84.8 s 중 `scipy…_superlu.gstrf`(완전 LU 분해) 52회가 63.8 s(75 %),
  `csc_matvec` 3.9 s, `_system`(야코비안 조립) 5.3 s, CG 6.7 s. 병목 = 뉴턴이 정체한 단계의 직접법 분해(1.2 s/회).
- 조치: 분해를 대칭 패턴 모드(`SymmetricMode`, 대각 우선 피벗)로 먼저 시도하고, 실제 잔차 ≤ 1e-6 기준을 못 맞추면 v4.2 분해로 되돌아감
  (`WP_FAST_LU=0`이면 항상 v4.2 방식). 채움량은 같고 분해 시간 약 2.5배 감소(시험 행렬 1.28 s → 0.48 s).
- 결과 지표 비교(같은 시나리오): 계산 시간 84.8 → 48.6 s(두 번 모두 다른 계산과 동시 실행), 뉴턴 반복 433 = 433, 전달 황 6.178181 g S 동일,
  유입수·질소수지·황수지 상대 오차 모두 1e-12 이내로 동일 → 지표 변화 없음.
- numba: 사용자 PC가 연결되지 않아 설치 가능 여부를 확인하지 못했고 이 작업 환경에서도 PyPI 접근이 막혀 있어 채택하지 않음(numpy·scipy 유지).

## v4.2 (2026-11-08)

Rules R1–R9 applied. Test criteria were fixed before implementation and were not changed afterwards.

### Screen policy
- Limitation sentences, model-assumption warnings, the O₂-off notice and the evaluation-note warning were removed from the screen. They now live in `limitations.md`, README and the Excel "한계·가정" sheet.
- Grid-resolution diagnostics (thin structures handled as face barriers, chamber cell count K4) go to the run log and Excel only.
- The test-status chip counts only 통과/미통과. "기록" (no pass criterion) and "미수행" (original paper not obtained) are listed separately.

### Reproducibility (R9)
- Each result file records the engine version (`v4.2`), the chemistry engine and its version (`PHREEQC 3.7.3…` or `내장 v4.2`), and the full code hash (`code_hash`, a single sha256 over all per-file hashes).
- ⑥ → 엔진·환경 has a "두 화학 엔진 비교" (`studies.engine_compare`). It also appears as the Excel sheet "화학 엔진 비교".
- Fixed random seed: 12345 in the Part 0 / X11 scripts (`runs/p0/x11.py`).

### Moisture dependence (F9)
- `moisture.py` defines three mutually exclusive modes: A mechanistic f_w·f_O2, B NEMIS family, C upper bound. Saturation is unified as S = θ/φ (WFPS).
- Mode A·B·C comparison table and rank-agreement verdict. Calibration of p or the threshold from removal-vs-saturation data (F9-7).

### Geometry and size (Part 1·2, K1–K8)
- `geometry_design.py`:
  - Depth zone: surface down to the water table minus a margin (default 1/α). If wt_depth is empty, the water table is computed from the bottom BC and the basis is recorded.
  - Screen length: L_s = Q / 2 gpm/ft (EPA 2013 p.44), with a sweep table.
  - Caisson calculator: t, z = V·t·SF, sulfur life including the aerobic-loss upper bound, cross-section, residence-time check, K-ratio sweep.
  - Parametric 3MF generator.
- The new sample `samples/caisson_down.3mf` (+ `.json`) is the unsaturated-zone downward-flow caisson: impermeable side wall + perforated top plate + media + perforated bottom plate.
  - Dimensions come from the calculator (mode A, SF 2), with media mass matched to the injection well's delivered sulfur.
  - The media is replaceable, so refill = media replacement (K7).
- The old `caisson_gate_scale0.05.3mf` was renamed to `samples/monkstown_saturated_reference.3mf` (saturated-zone reference only).
- Capture fraction (K2), chamber cell count (K4), K8 disturbance-zone sweep, and the reactive/soil K-ratio sweep were added.

### Perforated plates (Part 3)
- `plate.py`: hole resistance R_h, orifice, interference flag, and an axisymmetric verification solver.
- The "투수성 판(구멍)" role is treated as a face series resistance divided by upstream Kr. Hole variables can be edited in the role table and swept in ⑧.

### Metric corrections (Part 5)
- X2: planned vs delivered sulfur mass and delivery ratio. A refill counts as "successful" only if delivery ≥ 90 % (input).
- X3: η split into a reaction contribution (eval-window denitrification / inflow N) and a flow·storage contribution. denit_g reported both as cumulative and eval-window values.
- X4: R not computed when the pre-refill η < 1 %; the %p increase is reported instead.
- X5: ΔpH against the control in the same cells; a verdict is given only when a difference exists.
- X6: V_eff by two definitions (SO₄ increase / sulfur-consuming cells).
- X7: sulfur use split into denitrification (2.52 g S/g N) and aerobic oxidation.
- X8: initial η separated from the control.
- X10: reactive-zone saturation time series (mean/min/max) always stored in JSON and Excel.

### Remaining limitations (Part 4)
- F2: Hagen–Poiseuille equivalent K inside pipes (when a pipe diameter is given).
- F3: "파쇄 가능, 결과 무효" flag when injection pressure exceeds 100 psi.
- F4: filtration-theory λ option (Tufenkji & Elimelech 2004; α is a required input).
- F14: R travel-time correction using a conservative tracer in the same flow field.
- R4: the water-balance tolerance is now an upper bound derived from the Newton convergence criterion, computed per run.

### Tests
S8, S10, S11, S12, S18 and S19 added. S3, S4 and S9 are recorded as "미수행" because the original papers were not obtained.

### Part 7 (optimization)
- Profiling and grid timings: see the table below and README "v4.2".
- numba: wheel installation could not be checked in this workspace (PyPI blocked), and the user PC was not connected → not adopted (numpy path kept).
- `app_ui.py` split into tab modules `ui_tabs/tab*.py`. They run in the same global namespace, so behaviour is unchanged (same before/after mock-UI test results).
