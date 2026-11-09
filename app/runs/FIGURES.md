# 그림·애니메이션을 만드는 방법

시뮬레이션 결과에서 그림(PNG)과 애니메이션(GIF·HTML)이 나오기까지의 경로를 정리한다.

## 1. 계산 중에 저장되는 것 (engine.py)

| 무엇 | 언제 | 담긴 값 | 코드 |
|---|---|---|---|
| 설치 프레임 `frames` | 설치 표의 행마다(구조물 삽입 → 개구부 열기 → 시약 주입 진행률 → 정치) | 반응재 고체 부피 분율 `solid`, 부유 슬러리 입자 `P`, 수분 `theta`, 삽입된 객체·열린 개구부 목록, 시각 `t` | `Engine._frame()` |
| 운영 스냅숏 `snaps` | 운영 기간을 `n_snapshots`등분(기본 30, `--snaps 53`이면 약 7일 간격) | NO₃(mg N/L), SO₄(mg/L), pH, 반응재 고체, 황 몰수 `S_mol`, 투수성 비 K/K₀, 수분 θ, 압력수두 h | `Engine._snapshot()` |
| 일별 시계열 `daily` | 매일 | 바닥 유출 NO₃ 농도, 일별 η, 누적 탈질, 지하수 도달 N, pH(반응재 아래·주변·유출), SO₄, K 등 | `Engine._record_day()` |

배열은 모두 격자 모양 (nx, ny, nz)이고, z 인덱스 0이 지표, 클수록 깊다.

## 2. 결과 저장 (runs/compare_run.py)

```
python runs/compare_run.py --run B --days 365 --grid 22x22x44 --out runs/cmp/B44_o2z_full --o2-zero --no-modes --save-full --snaps 53
```

- `--save-full`: 위의 frames·snaps·daily를 포함한 결과 전체를 `<out>.pkl`로 저장한다. 이 옵션이 없으면 표(xlsx·json)만 남는다.
- `--snaps N`: 스냅숏 개수.

## 3. 정지 그림·GIF (runs/make_figures.py, matplotlib)

```
python runs/make_figures.py runs/cmp/B44_o2z_full.pkl runs/fig/B44_o2z
```

| 파일 | 내용 | 만드는 방법 |
|---|---|---|
| fig1_구조_배치.png | 비교군별 평면도·단면도 | `obj_cell`(구조물 셀)과 `reac0`(운영 시작 때 반응재 셀)을 z 방향으로 겹쳐 평면도를, 선택한 y 위치에서 잘라 단면도를 그림 |
| fig2_시계열.png | 바닥 유출 NO₃, 일별 η, 누적 탈질, 지하수 도달 N | `daily` 시계열. 점선은 대조군 돌파일(η 평가 시작) |
| fig3_최종상태.png | 마지막 날 단면 4종(NO₃·pH·SO₄·남은 황) | 마지막 스냅숏의 x–깊이 단면. 남은 황 = `S_mol` ÷ 첫 스냅숏 `S_mol` × 100 |
| fig4_깊이별_황잔량.png / .csv | 깊이별 남은 황(%) | `S_mol`을 x·y로 합해 첫 스냅숏 대비 비율. 재보충 깊이(25 % 미만) 판단용 |
| anim1_설치_<비교군>.gif | 설치 단계 단면 | `frames`를 차례로 그림(왼쪽 반응재 고체, 오른쪽 부유 슬러리, 검은 선 = 구조물 윤곽). 3 프레임/초 |
| anim2_NO3.gif | 운영 기간 NO₃ 단면(대조군+3 비교군) | `snaps`를 차례로. 색 범위 0–60 mg N/L 고정 |
| anim3_pH.gif | 운영 기간 pH 단면 | 색 범위 pH 4–9 고정 |
| anim4_황잔량.gif | 운영 기간 남은 황 단면 | 첫 스냅숏 대비 % |

- **단면 위치:** 비교군마다 반응재 고체가 가장 많은 y 줄을 고른다. 신박 4관 마름모는 앞줄 두 관을 지나는 면이고, 뒷줄 두 관은 이 면에 보이지 않는다. 대조군은 가운데 줄이다.
- **GIF 저장:** matplotlib `FuncAnimation` + `PillowWriter`를 쓴다.

## 4. 3D 인터랙티브 HTML (runs/export_3d_html.py, plotly)

앱의 결과 탭(④)과 내보내기 탭(⑦ "애니메이션 HTML 만들기")에 쓰는 함수를 그대로 쓴다. plotly가 설치된 PC(앱 실행 환경)에서 돌린다.

```
python runs/export_3d_html.py runs/cmp/B44_o2z_full.pkl runs/fig/B44_o2z
```

| 파일 | 함수 | 내용 |
|---|---|---|
| 3d_비교_애니메이션.html | `viz.compare_view` | 대조군·비교군 3D를 나란히 놓고, 아래 지표 그래프와 시간 슬라이더 하나로 연동. 반응재 등치면 + NO₃ 플룸 |
| 3d_설치과정_<비교군>.html | `viz.install_animation` | 3MF 메시가 위에서 내려와 박히고, 개구부가 열리고, 슬러리가 퍼지는 과정. ▶ 재생·느리게·카메라 버튼 |
| 3d_결과_<비교군>.html | `viz.results_scene` | 운영 스냅숏 슬라이더(반응재 분포·NO₃ 플룸) |

- **그리는 방법:** 3D는 `go.Isosurface`(등치면)를 쓰고, 셀이 많으면 `go.Surface` 단면으로 바꾼다. 3MF 메시는 `go.Mesh3d`로 그린다.
- **프레임:** 프레임마다 `go.Frame`을 만들고, 재생 버튼과 슬라이더를 붙인다.
- **저장:** `viz.to_html`로 저장한다. plotly.js를 HTML에 넣어 오프라인에서도 열린다.
- **이 스크립트의 실행 상태:** 이 스크립트는 클라우드 작업 환경에 plotly가 없어 거기서 실행해 보지 못했다. 앱과 같은 함수·인자를 쓰지만, 처음 실행할 때 오류가 나면 그대로 알려 달라.
