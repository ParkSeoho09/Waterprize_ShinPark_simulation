"""tab8_sweep — app_ui.py의 탭 8 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
st.markdown("황 **투입량 · 반응 구역 수분 · 입경 · 황:물 배합 · 수분 유지 재료 비율** 중 하나를 여러 값으로 바꿔 가며 "
            "선택한 장치(비교군)를 같은 지하수 환경(② 공통 조건 = 통제변수)에서 계산하고, 질산염 제거율을 비교합니다. "
            "지하수 흐름은 바꾸지 않습니다. 대조군은 반응재가 없어 모든 조건에서 같으므로 한 번만 계산합니다.")
jsw = runjob.get(_sweep_key())
if jsw and jsw["done"]:
    runjob.pop(_sweep_key())
    if jsw["error"] is None:
        ss().sweep_res = jsw["result"]
        st.success("조건 비교 계산을 마쳤습니다.")
    elif jsw["error_type"] == "StopRequested":
        st.warning("조건 비교를 중단했습니다.")
    else:
        st.error(f"조건 비교 오류: {jsw['error']}\n\n{jsw['trace']}")
names_ = [s_["name"] for s_ in ss().scen]
if not names_:
    st.info("① 탭에서 비교군(예: '예시 전부')을 먼저 추가하세요.")
else:
    pick = st.multiselect("비교할 장치(비교군)", names_, default=names_, key="sw_pick")
    var = st.selectbox("바꿀 조건(독립변수)", list(studies.SWEEP_VARS.keys()), key="sw_var")
    vtxt = st.text_input("값 (쉼표로 구분)", studies.SWEEP_VARS[var][1], key="sw_vals_" + studies.SWEEP_VARS[var][0])
    try:
        vals = [float(x) for x in vtxt.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        vals = []
        st.error("값은 숫자를 쉼표로 구분해 적으세요.")
    ret_ = None
    if studies.SWEEP_VARS[var][0] == "ret_frac":
        ret_ = st.selectbox("수분 유지 재료", [k_ for k_ in DZ.RETAINERS if k_ != "없음"], key="sw_ret")
    if studies.SWEEP_VARS[var][0] == "moist_set" and MO.mode_letter(c.moist_model) == "C":
        st.warning("② 탭에서 탈질 수분 함수가 '끔'이라 포화도를 바꿔도 탈질 속도는 그대로입니다(호기성 황 산화만 바뀜).")
    s1, s2, s3, s4 = st.columns(4)
    sw_days = float(s1.selectbox("기간", [90.0, 180.0], index=1, format_func=lambda x: f"{x:.0f}일", key="sw_days"))
    sw_grid = s2.selectbox("격자", list(GRID_PRESETS.keys()), index=0, key="sw_grid")
    sw_chem = s3.selectbox("화학", ["내장", "PHREEQC"], index=0, key="sw_chem", help="내장 = 빠름(PHREEQC와 탈질량 차이 0.3 % 안팎, 자체 점검)")
    n_runs = 1 + len(vals) * len(pick)
    sw_cores = int(s4.number_input("병렬 코어", value=studies.safe_cores(n_runs, int(np.prod(GRID_PRESETS[sw_grid]))),
                                   min_value=1, max_value=max(1, os.cpu_count() or 1), key="sw_cores"))
    st.caption(f"계산 {n_runs}개(대조군 1 + 값 {len(vals)} × 장치 {len(pick)}). 가벼움 격자·내장 화학·{sw_days:.0f}일이면 "
               "계산 하나에 보통 1~5분, 병렬 코어 수만큼 동시에 돕니다. 비교군의 설계 변수(①)가 꺼져 있으면 그 비교군의 지금 값을 출발점으로 씁니다.")
    busy = runjob.running(_sweep_key()) or runjob.running(ss().sid)
    if st.button("▶ 조건 비교 실행", type="primary", disabled=busy or not pick or not vals, key="sw_go"):
        cq = copy.deepcopy(c)
        cq.days = sw_days
        cq.nx, cq.ny, cq.nz = GRID_PRESETS[sw_grid]
        cq.grid_preset = sw_grid
        cq.chem_engine = sw_chem
        runnable_, skipped_ = prepare_runs(cq, [s_ for s_ in ss().scen if s_["name"] in pick])
        for nm_, why_ in skipped_:
            st.error(f"'{nm_}' 제외: {why_}")
        if runnable_:
            runjob.start(_sweep_key(), studies.sweep, cq, [so for _, so in runnable_], var, vals, ret_,
                         max(1, sw_cores), (90.0, 180.0), label="조건 비교")
            st.rerun()
    sweep_status()
SR = ss().get("sweep_res")
if SR:
    st.markdown(f"#### 결과: {SR['var']}")
    tags = [f"{h:.0f}일" for h in SR["horizons"]]
    tag = st.radio("기간", tags + ["끝"], horizontal=True, key="sw_tag")
    mlabel = st.radio("최적화 기준 제거율", list(studies.OPT_METRICS.keys()), horizontal=True, key="sw_metric")
    mkey = studies.OPT_METRICS[mlabel]
    w1, w2, w3 = st.columns(3)
    wr = w1.slider("가중치: 제거율", 0.0, 1.0, 0.5, 0.05, key="sw_wr")
    we = w2.slider("가중치: 황 투입량 대비 제거 N", 0.0, 1.0, 0.3, 0.05, key="sw_we")
    wd = w3.slider("가중치: 황 투입량(적을수록 좋음)", 0.0, 1.0, 0.2, 0.05, key="sw_wd")
    rows_ = studies.score_rows([dict(r_) for r_ in SR["rows"]], mkey, tag, wr, we, wd)
    cols_show = ["비교군", "값", "상태", f"{mkey}@{tag}", f"dev_removal@{tag}", f"sys_removal@{tag}", f"dev_Cin@{tag}",
                 f"dev_Cout@{tag}", f"S0_g@{tag}", f"S_used_g@{tag}", f"S_ox_g@{tag}", f"N_removed_g@{tag}",
                 f"N_per_kgS@{tag}", f"sat_mean@{tag}", f"bypass_pct@{tag}", "주입 전달률 %(부피)", "주입 최대 압력 kPa",
                 "주입 막힘", "반응 구역 등가 반경 r cm", "반응 구역 부피 cm³", "반응재 황 질량 g", "필터 케이크 두께 mm",
                 "점수", "파레토"]
    nice = {f"dev_removal@{tag}": "장치 제거율 %", f"sys_removal@{tag}": "지하수 유입 감소율 %", f"dev_Cin@{tag}": "장치 유입 NO₃-N",
            f"dev_Cout@{tag}": "장치 하류 NO₃-N", f"S0_g@{tag}": "황 투입 g", f"S_used_g@{tag}": "황 소비 g",
            f"S_ox_g@{tag}": "공기 산화 g", f"N_removed_g@{tag}": "제거 N g", f"N_per_kgS@{tag}": "g N/kg S",
            f"sat_mean@{tag}": "평균 포화도", f"bypass_pct@{tag}": "우회 %", f"{mkey}@{tag}": "기준 제거율 %"}
    df_ = pd.DataFrame([{nice.get(k_, k_): r_.get(k_) for k_ in dict.fromkeys(cols_show)} for r_ in rows_])
    st.dataframe(df_.round(4), hide_index=True, width="stretch")
    for r_ in rows_:
        if r_.get("상태") != "OK":
            st.error(f"{r_['비교군']} · 값 {r_['값']}: {r_.get('상태')} {str(r_.get('실패 이유', ''))[:200]}")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        figs_ = make_subplots(rows=1, cols=3, subplot_titles=(f"{mlabel}", "황 투입량 대비 제거 N (g N/kg S)", "황 소비 (g)"))
        pal = viz.SCEN_COLORS
        for i_, nm_ in enumerate(dict.fromkeys(r_["비교군"] for r_ in rows_)):
            rr_ = sorted([r_ for r_ in rows_ if r_["비교군"] == nm_], key=lambda x: x["값"])
            xs = [r_["값"] for r_ in rr_]
            for col_i, kk in enumerate((mkey, "N_per_kgS", "S_used_g")):
                figs_.add_scatter(x=xs, y=[r_.get(f"{kk}@{tag}") for r_ in rr_], mode="lines+markers", name=nm_,
                                  line=dict(color=pal[i_ % len(pal)]), showlegend=(col_i == 0), row=1, col=col_i + 1)
        if SR.get("key") == "param:inj_d_p_um":
            # C1: 선택한 토양 d50에서 계산한 통과 한계 0.154·d50 (µm) — 이보다 큰 입자는 흙 공극에 못 들어감(체거름)
            from params import D50_TEXTURE
            lim_um = c.p("strain_ratio") * D50_TEXTURE.get(c.texture, 1e-4) * 1e6
            for col_i in (1, 2, 3):
                figs_.add_vline(x=lim_um, line_dash="dash", line_color="#888", row=1, col=col_i)
            figs_.add_annotation(x=lim_um, y=1.02, yref="paper", text=f"통과 한계 {lim_um:.1f} µm", showarrow=False,
                                 row=1, col=1)
            figs_.update_xaxes(type="log")
        figs_.update_xaxes(title_text=SR["var"])
        figs_.update_layout(height=380, margin=dict(l=40, r=10, t=40, b=40), legend=dict(orientation="h", y=-0.3))
        show_fig(figs_, key="sweepfig")
    except Exception as e_:      # noqa: BLE001
        st.caption(f"(그림 생략: {e_})")
    scored = [r_ for r_ in rows_ if r_.get("점수") is not None]
    if scored:
        for nm_ in dict.fromkeys(r_["비교군"] for r_ in scored):
            best = max([r_ for r_ in scored if r_["비교군"] == nm_], key=lambda x: x["점수"])
            st.success(f"**{nm_}** 가장 좋은 조건: {SR['var']} = {best['값']:g} · 점수 {best['점수']:.2f} · "
                       f"{mlabel} {best.get(f'{mkey}@{tag}', float('nan')):.1f} · 황 투입 {best.get(f'S0_g@{tag}', float('nan')):.0f} g · "
                       f"황 소비 {best.get(f'S_used_g@{tag}', float('nan')):.2f} g")
        st.caption("점수 = 가중치×(제거율/최댓값) + 가중치×(황 투입량 대비 제거 N/최댓값) − 가중치×(황 투입량/최댓값). "
                   "파레토 = 제거율이 같거나 높으면서 황을 덜 쓰는 다른 조건이 없는 조건.")
    st.download_button("조건 비교 결과 CSV", pd.DataFrame(rows_).to_csv(index=False).encode("utf-8-sig"),
                       file_name="slad_조건비교.csv", key="sw_csv")
