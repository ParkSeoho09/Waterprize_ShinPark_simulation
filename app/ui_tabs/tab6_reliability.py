"""tab6_reliability — app_ui.py의 탭 6 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
st.caption("신뢰성은 세 가지를 구분해 봅니다: ① 자체 점검 = 방정식을 맞게 푸는가(해석해·보존 법칙), "
           "② 보정·검증 = 실험을 재현하는가, ③ 불확실성 = 가정 값이 바뀌면 결과가 얼마나 흔들리는가. "
           "외부 프로그램과 결과를 비교하는 검증은 하지 않습니다.")
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["자체 점검", "격자·시간 독립성", "실험 보정", "사진 비교", "불확실성",
                                      "알려진 문제 처리", "엔진·환경"])
with t1:
    sp = os.path.join(HERE, "selftest", "selfcheck_results.json")
    fast = st.checkbox("빠르게(E5 입자 시험 제외)", value=False, key="sc_fast")
    if st.button("🔎 자체 점검 실행", key="sc_run"):
        sys.path.insert(0, os.path.join(HERE, "selftest"))
        import selfcheck as SC
        pb = st.progress(0.0)
        with st.spinner("해석해·보존 시험 중…"):
            SC.run_all(include_slow=not fast, progress=lambda f, m: pb.progress(min(1.0, f), text=m))
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            SCR = json.load(fh)
        tb_ = SCR["table"]
        jd_ = [t_ for t_ in tb_ if t_.get("판정") in ("통과", "미통과")]
        np_ = sum(t_.get("판정") == "통과" for t_ in jd_)
        miss_ = [t_["id"] for t_ in jd_ if t_.get("판정") == "미통과"]
        st.markdown(ui_style.chip(f"{np_}/{len(jd_)} 통과", "ok" if not miss_ else "warn") +
                    (f" &nbsp;미통과: {', '.join(miss_)}" if miss_ else ""), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(tb_), hide_index=True, width="stretch")
        st.caption(f"실행 {SCR.get('when', '')} · {SCR.get('sec', '')}초 · 판정: 통과 / 미통과 / 기록(합격 기준 없음)")
        for k_, img in (SCR.get("figures") or {}).items():
            p_ = os.path.join(HERE, "selftest", img)
            if os.path.exists(p_):
                st.image(p_, caption=k_)
with t2:
    st.markdown("격자를 1.5배·2배로 세분하고 시간 간격을 절반으로 줄여 다시 풀어, 수두·유속(국부 최대)·NO₃ 유출·SO₄·최저 pH·K·"
                "질량수지가 얼마나 바뀌는지 봅니다. 2배 세분 결과도 저장합니다. **오래 걸립니다**(2배 = 셀 8배).")
    so, nm = _scen_choice("대상", "ind_sc")
    a_, b_ = st.columns(2)
    dd = a_.number_input("이 점검의 운영 기간(일)", value=float(min(c.days, 30.0)), key="ind_days")
    cores_i = int(b_.number_input("병렬 코어", value=1, key="ind_cores", step=1))
    if so != "ERR" and st.button("격자·시간 독립성 점검 실행", key="ind_run"):
        cc = copy.deepcopy(c); cc.days = dd; cc.light = True
        pb = st.progress(0.0)
        ss().stop_flag = False
        ss().indep = studies.independence(cc, so, cores=max(1, cores_i), out_dir=os.path.join(HERE, "selftest"),
                                          progress=lambda f, m: pb.progress(min(1.0, f), text=m),
                                          stop=lambda: bool(ss().get("stop_flag")))
    I_ = ss().get("indep")
    if not I_:
        # 이번 세션에서 돌리지 않았으면, 함께 배포한(또는 이전에 저장한) 가장 최근 결과를 보여 준다
        import glob
        fs_ = sorted(glob.glob(os.path.join(HERE, "selftest", "independence_*.json")))
        if fs_:
            try:
                with open(fs_[-1], encoding="utf-8") as fh:
                    I_ = json.load(fh)
                st.info(f"저장된 결과를 보여 줍니다: {os.path.basename(fs_[-1])} "
                        f"(대상 {I_.get('scenario', '?')}, {I_.get('when', '')}). 다시 돌리면 이번 결과로 바뀝니다.")
            except Exception:      # noqa: BLE001
                I_ = None
    if I_:
        st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k != "NO3_profile"} for t in I_["table"]]),
                     hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(I_["verdict"]).T, width="stretch")
        st.download_button("결과 저장(.json, 2배 세분 포함)", json.dumps(I_, ensure_ascii=False, default=float),
                           file_name="격자시간독립성.json", key="ind_dl")
    import glob as _glob
    gs_ = sorted(_glob.glob(os.path.join(HERE, "selftest", "grid_study_superposition_*.json")))
    if gs_:
        with st.expander("함께 배포한 점검: 주입 모델 '중첩 근사(빠름)'으로 ×1·×1.5·×2 모두 완료한 결과"):
            try:
                with open(gs_[-1], encoding="utf-8") as fh:
                    G_ = json.load(fh)
                st.caption("기본 주입 모델(과도 Richards)은 ×2 격자 주입 사례가 수렴하지 않아, 주입만 중첩 근사로 바꿔 "
                           "흐름·이송·반응 부분의 격자 의존성을 끝까지 비교한 것입니다(README '격자·시간 독립성').")
                st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k != "NO3_profile"} for t in G_["table"]]),
                             hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame(G_["verdict"]).T, width="stretch")
            except Exception as e_:      # noqa: BLE001
                st.caption(f"읽지 못함: {e_}")
with t3:
    st.markdown("실험 CSV(열: **시간(일), 변수, 깊이(m), 값, 구분**)를 올리면 '구분'이 *보정*인 행으로 파라미터를 맞추고 "
                "*검증* 행으로 RMSE·MAE·편향·NSE를 냅니다. 맞춘 값에는 자동으로 **'보정'** 라벨이 붙습니다.")
    st.caption("변수 이름: " + ", ".join(studies.OBS_VARS))
    upc = st.file_uploader("실험 CSV", type=["csv"], key="calcsv")
    so, nm = _scen_choice("이 실험에 해당하는 비교군", "cal_sc")
    prow = [{"키": k_, "설명": v_[0], "하한": v_[1], "상한": v_[2], "로그": v_[3], "보정": k_ in ("k_half",)}
            for k_, v_ in studies.UQ_PARAMS.items()]
    pdf = st.data_editor(pd.DataFrame(prow), hide_index=True, key="cal_par", disabled=["키", "설명"])
    nfev = int(st.number_input("최대 모델 실행 수", value=10, key="cal_nfev", step=1))
    if upc is not None and so != "ERR" and st.button("보정 실행", key="cal_run"):
        obs = pd.read_csv(upc)
        pars = {r["키"]: (float(r["하한"]), float(r["상한"]), bool(r["로그"])) for r in pdf.to_dict("records") if r["보정"]}
        cc = copy.deepcopy(c); cc.light = True
        pb = st.progress(0.0)
        try:
            CR = studies.calibrate(cc, so, obs, pars, max_nfev=nfev,
                                   progress=lambda f, m: pb.progress(min(1.0, f), text=m))
            ss().calib = {k_: v_ for k_, v_ in CR.items() if k_ != "result"}
        except Exception as e:
            st.error(f"보정 실패: {e}")
    CR = ss().get("calib")
    if CR:
        st.write("최적 값(라벨: 보정)", CR["best"])
        for k_ in ("stats_보정", "stats_검증"):
            if k_ in CR:
                st.markdown(f"**{k_.split('_')[1]} 자료 적합도**")
                st.dataframe(pd.DataFrame(CR[k_]).T, width="stretch")
        if st.button("보정 값을 공통 조건에 적용(라벨 '보정')", key="cal_apply"):
            for k_, v_ in CR["best"].items():
                studies.apply_sample(c, k_, v_)
                c.calibrated[k_] = v_
            st.success("적용했습니다. 고급 파라미터 표의 라벨이 '보정'으로 바뀝니다.")
    st.markdown("---")
    st.markdown("#### 문헌 칼럼 재현 S3 — Moon 외(2004) 칼럼 조건 입력")
    st.caption("원문에서 칼럼 조건을 옮겨 적으면 앱의 반-차수 상수를 문헌 값(30·40·60 mg N/L)과 비교합니다. "
               "30 mg/L로 배율을 보정하고 40·60 mg/L로 확인합니다. 빈칸이 있으면 '미수행'으로 남습니다.")
    q1, q2, q3 = st.columns(3)
    c.col_L_m = q1.number_input("충전 길이 (m)", value=c.col_L_m, key="s3_L", format="%.4g")
    c.col_D_m = q2.number_input("칼럼 안지름 (m)", value=c.col_D_m, key="s3_D", format="%.4g")
    c.col_Q_mL_min = q3.number_input("유량 (mL/min)", value=c.col_Q_mL_min, key="s3_Q", format="%.4g")
    c.col_phi = q1.number_input("공극률 (-)", value=c.col_phi, key="s3_phi", format="%.4g")
    c.col_dS_mm = q2.number_input("황 입경 (mm)", value=c.col_dS_mm, key="s3_d", format="%.4g")
    c.col_S_vfrac = q3.number_input("고형분 중 황 부피분율 (-)", value=c.col_S_vfrac, key="s3_vs", format="%.4g",
                                    help="황:석회석 3:1(v/v)이면 0.75")
    if st.button("S3 칼럼 재현 실행", key="s3_go"):
        col_ = {k_: getattr(c, k_) for k_ in ("col_L_m", "col_D_m", "col_Q_mL_min", "col_phi", "col_dS_mm", "col_S_vfrac")}
        col_["k_half_a"] = c.p("k_half_a")
        with open(os.path.join(HERE, "selftest", "s3_column.json"), "w", encoding="utf-8") as fh:
            json.dump(col_, fh, ensure_ascii=False, indent=1)
        sys.path.insert(0, os.path.join(HERE, "selftest"))
        import selfcheck as SC
        r3 = SC.S3_moon2004()
        sp3 = os.path.join(HERE, "selftest", "selfcheck_results.json")
        if os.path.exists(sp3):
            with open(sp3, encoding="utf-8") as fh:
                SCR3 = json.load(fh)
            SCR3["table"] = [t_ for t_ in SCR3["table"] if t_.get("id") != "S3"] + [r3]
            with open(sp3, "w", encoding="utf-8") as fh:
                json.dump(SCR3, fh, ensure_ascii=False, indent=1)
        ss().s3_row = r3
    if ss().get("s3_row"):
        st.dataframe(pd.DataFrame([ss().s3_row]), hide_index=True, width="stretch")

    st.markdown("#### 포화도별 제거율로 수분 파라미터 추정 (F9-7)")
    st.markdown("같은 매질을 여러 포화도에서 돌린 칼럼·흙수조 결과를 넣으면, 모드 A는 **p**(f_w = S^p), 모드 B는 **임계 WFPS**를 "
                "최소제곱으로 추정합니다. 모형: 균질 반응 구역(매질 체적 V)을 지나는 플러그 흐름, √C_in − √C_out = k½a·a·f(S)·V/(2Q). "
                "'구분'이 *보정*인 행만 추정에 쓰고 *확인* 행으로 오차를 냅니다. 보정 행이 3개 미만이면 추정하지 않습니다.")
    mc1, mc2, mc3 = st.columns(3)
    mcal_mode = mc1.selectbox("추정할 모드", [m_ for m_ in MO.MODES[:3] if MO.mode_letter(m_) != "C"], key="mcal_mode")
    mcal_d = mc2.number_input("매질 황 입경 (mm)", value=None, key="mcal_d", format="%.4g", help="입력 필요")
    mcal_phiS = mc3.number_input("매질 황 부피분율 φ_S (-)", value=None, key="mcal_phis", format="%.4g", help="입력 필요")
    base_rows = c.moist_calib_rows or [{"S": None, "Cin": None, "Q_L_h": None, "V_L": None, "Cout": None, "use": "보정"}]
    mdf = st.data_editor(pd.DataFrame(base_rows), num_rows="dynamic", key="mcal_ed", width="stretch",
                         column_config={"S": st.column_config.NumberColumn("포화도 S = θ/φ"),
                                        "Cin": st.column_config.NumberColumn("유입 NO₃-N (mg/L)"),
                                        "Q_L_h": st.column_config.NumberColumn("유량 (L/h)"),
                                        "V_L": st.column_config.NumberColumn("매질 체적 (L)"),
                                        "Cout": st.column_config.NumberColumn("유출 NO₃-N (mg/L)"),
                                        "use": st.column_config.SelectboxColumn("구분", options=["보정", "확인"])}
                         if hasattr(st, "column_config") else None)
    c.moist_calib_rows = [{k_: (None if (isinstance(v_, float) and v_ != v_) else v_) for k_, v_ in r_.items()}
                          for r_ in mdf.to_dict("records")]          # 프로젝트 파일에 함께 저장
    if st.button("수분 파라미터 추정", key="mcal_go"):
        try:
            if mcal_d is None or mcal_phiS is None:
                raise ValueError("매질 황 입경과 황 부피분율을 입력하세요(기본값으로 채우지 않음).")
            rows_ = [r_ for r_ in mdf.to_dict("records") if any(r_.get(k_) is not None and r_.get(k_) == r_.get(k_)
                                                              for k_ in ("S", "Cin", "Q_L_h", "V_L", "Cout"))]
            k_a_a = c.p("k_half_a") * 6.0 * float(mcal_phiS) / (float(mcal_d) * 1e-3)
            out_ = MO.calibrate(rows_, k_a_a, mcal_mode, moist_cfg(c))
            ss().mcal_result = dict(out_, rows=rows_, mode=mcal_mode, k_a_a=k_a_a)
        except ValueError as e_:
            st.error(str(e_))
    mres = ss().get("mcal_result")
    if mres:
        if not mres.get("ok"):
            st.warning(mres.get("reason", "추정하지 않음"))
        else:
            nm_p = {"p": "p (f_w = S^p)", "thr": "임계 WFPS s_t"}[mres["param"]]
            st.success(f"추정 {nm_p} = {mres['value']:.4f} (보정 행 {mres['n_cal']}개, 확인 행 {mres['n_check']}개"
                       + (f", 확인 RMSE {mres['check_rmse']:.3g} mg/L" if mres.get("check_rmse") is not None else "") + ")")
            if st.button("이 값을 ② 공통 조건에 적용(라벨 '보정')", key="mcal_apply"):
                key_ = "fw_p" if mres["param"] == "p" else "nemis_thr"
                c.params[key_] = float(mres["value"])
                c.calibrated = dict(c.calibrated or {}, **{key_: float(mres["value"])})
                st.success("적용했습니다.")

with t4:
    st.markdown("흙수조 단면 사진(염료·지시약)을 올리고 눈금자 스케일과 색→농도 보정선을 주면, 같은 시점 시뮬 단면과 "
                "**IoU**(겹침 비율)와 **도달 깊이 오차**를 냅니다.")
    R = ss().results
    img = st.file_uploader("사진", type=["png", "jpg", "jpeg"], key="photo")
    if img is not None and R:
        st.image(img, width=360)
        a_, b_, d_ = st.columns(3)
        ppc = a_.number_input("눈금자: 1 cm = 몇 픽셀", value=10.0, key="ppc")
        ox = b_.number_input("흙수조 왼쪽 지표 모서리 x(픽셀)", value=0.0, key="ox")
        oy = d_.number_input("… y(픽셀)", value=0.0, key="oy")
        chn = a_.selectbox("색 채널", ["밝기", "R", "G", "B"], key="pch")
        cal_df = st.data_editor(pd.DataFrame([{"픽셀값": 30.0, "농도": 50.0}, {"픽셀값": 220.0, "농도": 0.0}]),
                                num_rows="dynamic", key="pcal")
        thr = b_.number_input("판정 농도 문턱", value=10.0, key="pthr")
        all_r = R["scen"] + [R["control"]]
        nm = d_.selectbox("비교할 결과", [r["name"] for r in all_r], key="pres")
        r_ = next(x for x in all_r if x["name"] == nm)
        var = a_.selectbox("변수", ["NO3", "SO4", "pH", "theta"], key="pvar")
        si = int(b_.number_input("장면 번호(0 = 운영 시작)", value=len(r_["snaps"]) - 1, key="psi", step=1))
        yj = int(d_.number_input("y 단면 칸", value=r_["grid"]["ny"] // 2, key="pyj", step=1))
        if st.button("비교", key="pgo"):
            sec = r_["snaps"][min(max(si, 0), len(r_["snaps"]) - 1)][var][:, min(max(yj, 0), r_["grid"]["ny"] - 1), :]
            outp = studies.photo_compare(img.getvalue(), ppc, (ox, oy), [tuple(x) for x in cal_df.dropna().values],
                                         thr, np.nan_to_num(sec), r_["grid"]["dx"], r_["grid"]["dz"], chn)
            st.write({k_: v_ for k_, v_ in outp.items() if not k_.endswith("mask")})
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_heatmap(z=outp["photo_mask"].T.astype(int) + 2 * outp["model_mask"].T.astype(int),
                            colorscale=[[0, "white"], [0.33, "#4c78a8"], [0.66, "#f58518"], [1, "#54a24b"]],
                            showscale=False)
            fig.update_yaxes(autorange="reversed", title="깊이 칸"); fig.update_xaxes(title="x 칸")
            fig.update_layout(title="파랑 = 사진만, 주황 = 모델만, 초록 = 겹침", height=380)
            show_fig(fig, key="photo_fig")
    elif img is not None:
        st.info("먼저 ③에서 실행해 결과를 만드세요.")
with t5:
    st.markdown("불확실한 파라미터의 범위를 주면 **라틴 하이퍼큐브 몬테카를로**로 반복 실행해 지표마다 평균·표준편차·95% 구간과 "
                "**순위 상관(Spearman) 민감도**를 냅니다. 각 표본마다 대조군도 같이 다시 계산합니다.")
    so, nm = _scen_choice("대상 비교군", "uq_sc")
    rows_u = [{"키": k_, "설명": v_[0], "하한": v_[1], "상한": v_[2], "로그 균등": v_[3], "포함": True}
              for k_, v_ in studies.UQ_PARAMS.items()]
    udf = st.data_editor(pd.DataFrame(rows_u), hide_index=True, key="uq_par", disabled=["키", "설명"])
    a_, b_, d_ = st.columns(3)
    nN = int(a_.number_input("표본 수", value=12, key="uq_n", step=1))
    ud = b_.number_input("운영 기간(일)", value=float(min(c.days, 60.0)), key="uq_days")
    uc = int(d_.number_input("병렬 코어", value=max(1, (os.cpu_count() or 2) - 1), key="uq_cores", step=1))
    if so not in (None, "ERR") and st.button("불확실성 분석 실행", key="uq_run"):
        rng_ = {r["키"]: (float(r["하한"]), float(r["상한"]), bool(r["로그 균등"])) for r in udf.to_dict("records") if r["포함"]}
        cc = copy.deepcopy(c); cc.days = ud; cc.light = True
        pb = st.progress(0.0)
        ss().stop_flag = False
        ss().uq = studies.lhs_mc(cc, so, rng_, n=nN, cores=uc, progress=lambda f, m: pb.progress(min(1.0, f), text=m),
                                 stop=lambda: bool(ss().get("stop_flag")))
    U_ = ss().get("uq")
    if U_:
        st.caption(f"{U_['method']} · 성공 {U_['n_ok']}/{U_['n']} 표본")
        srows = [{"지표": k_, "평균": v_["평균"], "표준편차": v_["표준편차"], "95% 구간": str(tuple(round(x, 3) for x in v_["95% 구간(분포 2.5~97.5%)"])),
                  "평균의 95% 신뢰구간": str(tuple(round(x, 3) for x in v_["평균의 95% 신뢰구간"]))} for k_, v_ in U_["stats"].items()]
        st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch")
        sens = pd.DataFrame({mk: {pk: v["순위상관 ρ"] for pk, v in d.items()} for mk, d in U_["sensitivity"].items()}).T
        st.markdown("**민감도(순위 상관 ρ, 지표 × 파라미터)**")
        st.dataframe(sens.round(2), width="stretch")
with t6:
    kp2 = os.path.join(HERE, "selftest", "known_issues_v2.json")
    kp3 = os.path.join(HERE, "selftest", "known_issues_v3.json")
    if os.path.exists(kp2) and os.path.exists(kp3):
        with open(kp2, encoding="utf-8") as fh:
            k2 = {x["test"]: x for x in json.load(fh)}
        with open(kp3, encoding="utf-8") as fh:
            k3 = {x["test"]: x for x in json.load(fh)}
        rows_k = []
        for t_, x in k3.items():
            b_ = k2.get(t_, {})
            rows_k.append({"번호": x["id"], "v2 시험": ("실패(문제 사실)" if not b_.get("pass", True) else "통과(문제 아님)") if b_ else "–",
                           "v2 내용": b_.get("detail", ""), "v3 시험": "통과" if x["pass"] else "실패", "v3 내용": x["detail"]})
        st.dataframe(pd.DataFrame(rows_k), hide_index=True, width="stretch")
        st.caption("전체 처리 표(수정 내용·해당 없음 근거 포함)는 README '알려진 문제 처리 표'에 있습니다.")
    if st.button("알려진 문제 시험 다시 실행(현재 코드)", key="ki_run"):
        sys.path.insert(0, os.path.join(HERE, "selftest"))
        import known_issues as KI
        with st.spinner("시험 중…"):
            st.dataframe(pd.DataFrame(KI.run("v3")), hide_index=True, width="stretch")
with t7:
    pst = PE.status()
    st.markdown(f"**PHREEQC(IPhreeqc)**: 연결 {pst['ok']} · 버전 {pst.get('version', '')} · 라이브러리 `{pst.get('lib', '')}` · "
                f"DB `{os.path.basename(pst.get('db', ''))}`")
    if pst["error"]:
        st.error(pst["error"])
    ep = os.path.join(HERE, "engine_check.json")
    if os.path.exists(ep):
        with open(ep, encoding="utf-8") as fh:
            st.json(json.load(fh))
    if st.button("엔진 설치 상태 다시 확인", key="eng_chk"):
        import engine_check
        with st.spinner("확인 중…"):
            st.json(engine_check.check(write=True))
    st.markdown("**두 화학 엔진 비교(R9)** — 같은 입력을 내장 엔진과 PHREEQC로 각각 돌려 핵심 지표 차이를 표로 냅니다.")
    ec1, ec2 = st.columns(2)
    ec_days = ec1.number_input("비교 기간(일)", value=float(min(c.days, 20.0)), key="ec_days")
    if ec2.button("엔진 비교 실행", key="ec_run") and ss().scen:
        objs_ = []
        for s_ in ss().scen:
            try:
                objs_.append(scenario_obj(s_, c))
            except Exception as e:      # noqa: BLE001
                st.error(f"{s_['name']}: {e}")
        pb = st.progress(0.0)
        with st.spinner("두 엔진으로 계산 중…"):
            ss().engine_cmp = studies.engine_compare(copy.deepcopy(c), objs_, days=ec_days,
                                                     progress=lambda f, m: pb.progress(min(1.0, f), text=m))
    if ss().get("engine_cmp"):
        st.dataframe(pd.DataFrame(ss().engine_cmp), hide_index=True, width="stretch")
    vp = os.path.join(HERE, "verify", "results.json")
    if os.path.exists(vp):
        with st.expander("v2 개발 때 기록한 검증 표(참고)"):
            with open(vp, encoding="utf-8") as fh:
                V = json.load(fh)
            st.dataframe(pd.DataFrame(V["table"]), hide_index=True, width="stretch")
