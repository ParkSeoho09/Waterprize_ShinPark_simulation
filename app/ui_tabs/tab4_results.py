"""tab4_results — app_ui.py의 탭 4 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
R = ss().results
if R is None and runjob.running(ss().sid):
    st.info("⏳ 대조군 + 비교군을 계산하고 있습니다(진행률은 화면 맨 위). 끝나면 이 탭에 비교군별 지표 표·설치 애니메이션·"
            "결과 3D·다중 비교·그래프가 자동으로 나타납니다. 그동안 다른 탭을 보거나 버튼을 눌러도 계산은 끊기지 않습니다.")
elif R is None:
    if ss().scen:
        st.warning(f"**비교군 {len(ss().scen)}개({', '.join(s_['name'] for s_ in ss().scen)})는 아직 계산하지 않았습니다.** "
                   "비교군 결과·애니메이션은 실행을 해야 만들어집니다. 아래 버튼(또는 ③ 실행 탭의 ▶ 실행)을 누르세요.")
        b4a, b4b = st.columns(2)
        go4 = b4a.button("▶ 지금 실행 (②의 조건 그대로)", type="primary", key="go_run4", width="stretch",
                         disabled=bool(validate_common(c)[0]))
        go4q = b4b.button(f"⚡ 빠른 확인 실행 ({QUICK_DAYS:g}일·가벼움 격자·내장 화학)", key="go_quick4", width="stretch",
                          disabled=bool(validate_common(c)[0]))
        if go4 or go4q:
            n_cells_ = int(c.nx * c.ny * c.nz) if go4 else int(np.prod(GRID_PRESETS["가벼움"]))
            if start_run(c, studies.safe_cores(1 + len(ss().scen), n_cells_), quick=bool(go4q and not go4)):
                ss().run_msgs = []
                st.rerun()
    st.subheader("기본 대조군 미리보기 — 반응재를 꽂지 않았을 때의 피해")
    st.caption("실행 전에도 볼 수 있는 대조군 미리보기입니다(가벼운 격자, 현재 공통 조건, 빠른 근사 = 준정상). "
               "비교군 결과가 아니며, 실행 결과의 대조군과는 격자·흐름 방식이 다릅니다.")
    ok_c = not validate_common(c)[0]
    if ok_c and st.toggle("기본 대조군 미리보기 보기", value=not ss().scen):
        cr, cm, bt = baseline_control(c.to_dict())
        k1, k2, k3 = st.columns(3)
        k1.metric("지하수 도달 NO₃-N 누적", f"{cm['N_to_gw_g']:.2f} g")
        k2.metric("먹는물 기준(10 mg/L) 도달", f"{cm['arrival_10mg_day']:.0f}일" if np.isfinite(cm['arrival_10mg_day']) else "모사 기간 내 없음")
        k3.metric("유출 농도 50% 도달(돌파)", f"{bt:.0f}일" if bt else "없음")
        show_fig(viz.baseline_damage(cr, c), key="base_dmg")
        ann = viz.param_annotations(c, None, cr, cr["grid"])
        show_fig(viz.results_scene(cr, c, ["NO₃ 오염 플룸", "지하수층 NO₃"], light=True, annotations=ann, height=560),
                 key="base_anim")
else:
    C_ = Common.from_dict(R["common"])
    all_res = R["scen"] + [R["control"]]
    mets = R["metrics"]
    sc_by_name = {s_["name"]: s_ for s_ in ss().scen}
    if R.get("eval_start") is None and mets and mets[-1].get("eta_unavailable"):
        st.warning("**η(제거 효율)는 이번 결과에서 평가하지 않았습니다.** " + mets[-1]["eta_unavailable"])
    if R.get("quick"):
        st.caption(f"⚡ 빠른 확인 실행: 가벼움 격자 · {C_.days:g}일 · 내장 화학")
    st.markdown(trust_strip(R, all_res), unsafe_allow_html=True)
    if R.get("failed"):
        st.error("FAILED(수렴 실패): " + ", ".join(R["failed"]) + " — 이 결과는 비신뢰 결과입니다.")
    st.subheader("연구 핵심 지표 — 90일 · 180일 비교")
    st.caption("통제변수(②: 지하수 흐름·지층·유입 농도)는 모든 비교군이 같습니다. 장치 유입부·하류 농도는 운영 시작 때의 반응 구역 "
               "경계를 드나든 물의 유량 가중 평균, 장치 제거율 = 반응 구역 안에서 탈질된 N ÷ 반응 구역으로 들어간 N. "
               "우회 비율 = 1 − (반응재 통과 유량 ÷ 반응재가 없을 때 같은 자리를 지나는 유량), 같은 흙·경계 조건으로 계산.")
    try:
        hc, hr = M.core_table(R["scen"], R["control"], C_)
        st.dataframe(pd.DataFrame(hr, columns=hc), hide_index=True, width="stretch", height=min(760, 36 * (len(hr) + 1)))
        if C_.days < 180:
            st.caption(f"운영 기간이 {C_.days:g}일이라 180일 칸은 없습니다(② 운영 기간 또는 ③ 빠른 확인 기간을 180일로).")
    except Exception as e_:      # noqa: BLE001
        st.warning(f"핵심 지표 표를 만들지 못했습니다: {e_}")
    MD = R.get("modes")
    if MD and MD.get("by_mode"):
        st.subheader("수분 의존 모드 A·B·C 비교")
        rows_m = []
        for L_ in ("A", "B", "C"):
            for nm_, mm_ in (MD["by_mode"].get(L_) or {}).items():
                row_ = {"모드": L_ + (" (선택)" if L_ == MD.get("current") else ""), "비교군": nm_}
                if isinstance(mm_, dict) and "status" in mm_ and None not in mm_:
                    row_["상태"] = f"{mm_['status']} {str(mm_.get('reason', ''))[:60]}"
                else:
                    for hz_, tg_ in ((90.0, "90일"), (180.0, "180일"), (None, "끝")):
                        m3 = mm_.get(hz_) if isinstance(mm_, dict) else None
                        if m3:
                            row_[f"장치 제거율 % ({tg_})"] = round(m3.get("dev_removal", float("nan")), 2)
                            row_[f"지하수 유입 감소율 % ({tg_})"] = round(m3.get("sys_removal", float("nan")), 2)
                            row_[f"제거 N g ({tg_})"] = round(m3.get("N_removed_g", float("nan")), 4)
                rows_m.append(row_)
        st.dataframe(pd.DataFrame(rows_m), hide_index=True, width="stretch")
        rk = MD.get("rank", {})
        st.dataframe(pd.DataFrame([{"기준": {"dev_removal": "장치 제거율", "sys_removal": "지하수 유입 감소율"}[k_],
                                    "판정": v_["verdict"], "순위(모드별)": " | ".join(f"{L_}: {' > '.join(o_)}" for L_, o_ in v_["orders"].items())}
                                   for k_, v_ in rk.items()]), hide_index=True, width="stretch")
    elif MD and MD.get("error"):
        st.error(f"모드 비교 계산 오류: {MD['error']}")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        figc = make_subplots(rows=2, cols=2, subplot_titles=("장치 유입부 vs 하류 NO₃-N (mg/L)", "흙수조 바닥 유출 NO₃-N (mg/L)",
                                                             "잔여 황 (g)", "반응 구역 포화도 s"))
        for r_ in R["scen"] + [R["control"]]:
            Dd = r_["daily"]
            col_ = r_.get("color", "#666")
            nm_ = r_["name"]
            if not r_.get("is_control"):
                figc.add_scatter(x=Dd["day"], y=Dd.get("dev_Cin_mgN", []), name=f"{nm_} 유입부", line=dict(color=col_, dash="dot"),
                                 row=1, col=1)
                figc.add_scatter(x=Dd["day"], y=Dd.get("dev_Cout_mgN", []), name=f"{nm_} 하류", line=dict(color=col_), row=1, col=1)
                figc.add_scatter(x=Dd["day"], y=Dd.get("S_g", []), name=f"{nm_} 황", line=dict(color=col_), row=2, col=1,
                                 showlegend=False)
                figc.add_scatter(x=Dd["day"], y=Dd.get("sat_reac", []), name=f"{nm_} s", line=dict(color=col_), row=2, col=2,
                                 showlegend=False)
            figc.add_scatter(x=Dd["day"], y=Dd["Cout_mgN"], name=f"{nm_} 바닥 유출", line=dict(color=col_ if not r_.get("is_control") else "#888"),
                             row=1, col=2, showlegend=False)
        if C_.moist_mode.startswith("반응 구역"):
            figc.add_hline(y=C_.moist_set, line_dash="dash", row=2, col=2)
        figc.update_layout(height=560, margin=dict(l=40, r=10, t=40, b=30), legend=dict(orientation="h", y=-0.12))
        show_fig(figc, key="corecharts")
    except Exception as e_:      # noqa: BLE001
        st.caption(f"(시계열 그림 생략: {e_})")
    st.subheader("비교군 × 지표")
    head, rows = M.summary_table(mets)
    st.dataframe(pd.DataFrame(rows, columns=head), hide_index=True, width="stretch", height=min(900, 36 * (len(rows) + 1)))
    st.caption(f"η 평가 구간: 대조군 유출 농도가 유입의 90%에 도달한 {R['eval_start']}일 이후(도달 지연 착시 제외). '대조군 보정 제거율'은 같은 날 대조군 유출과 비교한 값. 재보충 전·후 구간 η는 아래 표.")
    with st.expander("재보충 전·후 구간별 η, 비교군 메커니즘 설명, 경고"):
        for r, m in zip(R["scen"], mets):
            st.markdown(f"**{r['name']}** — {r['mechanism'] or '(메커니즘 설명 없음)'}")
            st.dataframe(pd.DataFrame([{"구간 시작(일)": s_["from"], "구간 끝(일)": s_["to"], "η (%)": round(s_["eta"], 1)}
                                       for s_ in m["eta_segments"]]), hide_index=True)
            d = r.get("diag", {}) or {}
            if d.get("through_verdict"):
                st.caption(f"반응재 통과 진단: {d['through_verdict']} (반응재 통과 유량 / 반응재 없을 때 같은 위치 유량 = "
                           f"{d.get('through_ratio', float('nan')):.3f})")
            for w in r["warnings"][:12]:
                st.caption("⚠ " + w)
    with st.expander("수렴·질량수지 보고서(매 실행 자동 점검)"):
        st.caption("질량수지가 맞는다는 것은 '계산이 물질을 잃지 않았다'는 뜻이지 물리적으로 옳다는 증거는 아닙니다.")
        rep_rows = []
        for r in all_res:
            sv = r.get("solver", {})
            mb = r.get("mass_balance", {})
            row_ = {"실행": r["name"], "상태": r.get("status"), "화학 엔진": r.get("chem_engine"),
                    "흐름 단계(채택/거부)": f"{int(sv.get('flow_accepted', 0))}/{int(sv.get('flow_rejected', 0))}",
                    "뉴턴 반복": int(sv.get("newton_iters", 0)), "Δt 축소": int(sv.get("dt_cuts", 0)),
                    "Δt 범위(초)": f"{sv.get('dt_min_s', float('nan')):.3g}~{sv.get('dt_max_s', float('nan')):.3g}",
                    "선형계 CG/BiCGSTAB/LGMRES/GMRES/직접/정칙화": "/".join(str(int(sv.get(k, 0))) for k in (
                        "lin_cg", "lin_bicgstab", "lin_lgmres", "lin_gmres", "lin_direct", "lin_regularized")),
                    "최대 상대잔차": f"{sv.get('lin_max_relres', 0):.1e}", "PHREEQC 호출": int(sv.get("phreeqc_calls", 0))}
            for k_, v_ in mb.items():
                if isinstance(v_, (int, float)):
                    row_["수지 " + k_.replace("_err_rel", "")] = f"{v_:.1e}"
            row_["허용 초과"] = ", ".join(r.get("mass_balance_bad", {}).keys()) or "없음"
            rep_rows.append(row_)
        st.dataframe(pd.DataFrame(rep_rows), hide_index=True, width="stretch")
        for r in all_res:
            if r.get("mass_balance", {}).get("W_note"):
                st.caption(f"{r['name']} 물 수지: {r['mass_balance']['W_note']}")
            if r.get("fail_reason"):
                st.error(f"{r['name']}: {r['fail_reason']}")
    view = st.radio("보기", ["설치 과정 애니메이션", "결과 3D(레이어)", "다중 비교(시간 연동)", "그래프", "재보충 이벤트"],
                    horizontal=True)
    show_ann = st.toggle("파라미터 표시(구석 캡션·마우스 툴팁)", value=True)
    light = st.toggle("가벼운 모드", value=C_.light, key="light_view")
    if view == "설치 과정 애니메이션":
        nm = st.selectbox("비교군", [r["name"] for r in R["scen"]])
        r = next(x for x in R["scen"] if x["name"] == nm)
        so = scenario_obj(sc_by_name[nm], C_) if nm in sc_by_name else None
        if so is None:
            st.warning("이 결과의 비교군 입력이 지금 목록에 없어 구조물 모양을 그릴 수 없습니다.")
        else:
            from voxel import Placement as _P
            pl = _P(**r["placement"]) if r.get("placement") else so.placement
            ann = viz.param_annotations(C_, so, r, r["grid"]) if show_ann else None
            show_fig(viz.install_animation(r, so.objects, so.roles, pl, wt_depth=C_.wt_depth, light=light, annotations=ann),
                     key="inst")
            st.caption("단계: " + " → ".join(dict.fromkeys(f["label"].split(" (")[0] for f in r["frames"])))
    elif view == "결과 3D(레이어)":
        a1, a2 = st.columns([2, 1])
        nm = a2.selectbox("비교군/대조군", [r["name"] for r in all_res])
        r = next(x for x in all_res if x["name"] == nm)
        layers = a1.multiselect("레이어", list(viz.LAYER_DEF), default=["반응재 분포", "NO₃ 오염 플룸", "pH 단면", "지하수층 NO₃"])
        cut = a2.slider("단면 자르기(x 위치, 1 = 자르지 않음)", 0.05, 1.0, 1.0, 0.05)
        so = scenario_obj(sc_by_name[nm], C_) if nm in sc_by_name else None
        from voxel import Placement as _P
        pl = _P(**r["placement"]) if r.get("placement") else None
        ann = viz.param_annotations(C_, so, r, r["grid"]) if show_ann else None
        show_fig(viz.results_scene(r, C_, layers, objects=so.objects if so else None, roles=so.roles if so else None,
                                   placement=pl, cut_x=None if cut >= 0.999 else cut * C_.Lx, light=light,
                                   annotations=ann, wt_depth=C_.wt_depth), key="res3d")
    elif view == "다중 비교(시간 연동)":
        b1, b2, b3 = st.columns([2, 1, 1])
        layers = b1.multiselect("레이어", list(viz.LAYER_DEF), default=["반응재 분포", "NO₃ 오염 플룸"])
        metric = b2.selectbox("그래프", ["Cout_mgN", "eta", "SO4_out_mg", "pH_out", "N_to_gw_g"],
                              format_func=lambda x: {"Cout_mgN": "유출 NO₃-N", "eta": "η", "SO4_out_mg": "유출 SO₄",
                                                     "pH_out": "유출 pH", "N_to_gw_g": "지하수 도달 누적"}[x])
        sync = b3.checkbox("카메라 동기화", value=True, help="한 화면을 돌리면 다른 화면도 같이 돕니다.")
        objs = {}
        for r in R["scen"]:
            if r["name"] in sc_by_name:
                so = scenario_obj(sc_by_name[r["name"]], C_)
                from voxel import Placement as _P
                objs[r["name"]] = (so.objects, so.roles, _P(**r["placement"]))
        fig = viz.compare_view(all_res, C_, layers, metric=metric, light=light, scen_objects=objs)
        if sync:
            show_html(viz.to_html(fig, sync_cameras=True), height=860)
        else:
            show_fig(fig, key="cmp")
    elif view == "그래프":
        bt = R["eval_start"]
        show_fig(viz.timeseries(all_res, "eta", "제거효율 η (%)", 100, eval_start=bt), key="g_eta")
        show_fig(viz.timeseries(all_res, "Cout_mgN", "유출 NO₃-N (mg/L)", hlines=[(10, "먹는물 기준 10")], eval_start=bt), key="g_c")
        show_fig(viz.timeseries(all_res, "N_to_gw_g", "지하수 도달 NO₃-N 누적 (g)"), key="g_gw")
        g1, g2 = st.columns(2)
        with g1:
            show_fig(viz.timeseries(all_res, "pH_below", "pH (반응재 아래 0~10 cm)", hlines=[(C_.soil_pH0 - 0.5, "−0.5"), (C_.soil_pH0 + 0.5, "+0.5")]), key="g_ph")
            show_fig(viz.timeseries(all_res, "SO4_out_mg", "유출 SO₄ (mg/L)", hlines=[(250, "250 mg/L")]), key="g_so4")
        with g2:
            show_fig(viz.timeseries(all_res, "Kreac", "반응재 영역 K (m/s)"), key="g_k")
            show_fig(viz.timeseries(all_res, "Keff", "도메인 유효 K (m/s, 정수두 계산)"), key="g_keff")
        st.markdown("**깊이 프로파일(마지막 시점)**")
        p1, p2, p3 = st.columns(3)
        with p1:
            show_fig(viz.depth_profiles(all_res, mets[:-1] + [mets[-1]], "pH", "pH"), key="dp_ph")
        with p2:
            show_fig(viz.depth_profiles(all_res, mets, "NO3", "NO₃-N (mg/L)"), key="dp_n")
        with p3:
            show_fig(viz.depth_profiles(all_res, mets, "SO4", "SO₄ (mg/L)"), key="dp_s")
    else:
        for r, m in zip(R["scen"], mets):
            st.markdown(f"**{r['name']}** — {m['refill_note']}")
            if m["refill_table"]:
                st.dataframe(pd.DataFrame([{"회차": x.get("n"), "운영 일": round(x["day"], 1), "방식": x.get("method"),
                                            "성공": x.get("ok"), "η 직전(%)": round((x.get("eta_before") or 0) * 100, 1),
                                            "R 24h(%)": x.get("R_24h"), "R 이동시간 보정(%, 근사)": x.get("R_travel"),
                                            "시작(분)": round(x.get("t_start_min", 0), 1), "끝(분)": round(x.get("t_end_min", 0) or 0, 1),
                                            "주입량(L)": round(x.get("delivered_L", 0) or 0, 2), "T_r(분)": x.get("duration_min"),
                                            "사유": x.get("why", "")} for x in m["refill_table"]]), hide_index=True)
