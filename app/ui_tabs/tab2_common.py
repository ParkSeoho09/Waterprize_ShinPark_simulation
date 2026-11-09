"""tab2_common — app_ui.py의 탭 2 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
st.caption("아래 조건은 모든 비교군과 대조군에 똑같이 적용됩니다(지표 문서의 통제변수). 숫자 입력에는 최솟값·최댓값 제한이 없습니다.")
H = TX.PARAM_HELP
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**흙수조·지하수층**")
    c.Lx = num("가로 Lx (m)", c.Lx, "Lx", H["Lx"])
    c.Ly = num("세로 Ly (m)", c.Ly, "Ly", H["Ly"])
    c.H_soil = num("토양층 깊이 (m)", c.H_soil, "Hs", H["H_soil"])
    c.H_aq = num("지하수층 두께 (m)", c.H_aq, "Ha", H["H_aq"])
    c.wt_depth = st.number_input("지하수면 깊이 (m, 비우면 토양층 바닥)", value=c.wt_depth, key="wt", help=H["wt_depth"],
                                 format="%.4g")
    c.gw_velocity = num("지하수 유속 (m/일)", c.gw_velocity, "gwv", H["gw_velocity"])
    bopts = ["지하수 수위", "누수(배수 저항)", "자유배수"]
    c.bottom_bc = st.selectbox("바닥 조건", bopts, index=bopts.index(c.bottom_bc) if c.bottom_bc in bopts else 0,
                               help=H["bottom_bc"])
    if c.bottom_bc.startswith("누수"):
        c.leak_thick = num("배수층 두께 (m)", c.leak_thick, "lth", "바닥과 배수구 사이 층 두께. 저항 = 두께/K(가정)")
        c.leak_K = num("배수층 K (m/s)", c.leak_K, "lK", "작을수록 물이 잘 안 빠져 지하수면이 오르내림(주입·강우 시 수위 상승)")
    c.aq_texture = st.selectbox("지하수층 토성", list(SOIL_TEXTURES), index=list(SOIL_TEXTURES).index(c.aq_texture))
with c2:
    st.markdown("**토양(불포화대)**")
    c.soil_mode = st.radio("입력 방식", ["간단", "상세"], index=["간단", "상세"].index(c.soil_mode), horizontal=True)
    c.texture = st.selectbox("토성", list(SOIL_TEXTURES), index=list(SOIL_TEXTURES).index(c.texture), help=H["texture"])
    tx = SOIL_TEXTURES[c.texture]
    c.Ks_override = st.number_input("포화 투수계수 Ks (m/s, 비우면 토성 값)", value=c.Ks_override, key="Ks",
                                    help=H["Ks"], format="%.3e")
    st.caption(f"토성 기본값: θr {tx['theta_r']}, θs {tx['theta_s']}, α {tx['alpha']:.1f} 1/m, n {tx['n']}, "
               f"Ks {tx['Ks']:.2e} m/s — {SOIL_SOURCE}")
    c.init_saturation = num("초기 포화도 (0~1)", c.init_saturation, "isat", H["init_saturation"])
    if c.soil_mode == "상세":
        c.aniso = num("이방성 Kz/Kx", c.aniso, "aniso", H["aniso"])
        hc1, hc2, hc3 = st.columns(3)
        het = dict(c.heterogeneity or {})
        het["sigma_lnK"] = hc1.number_input("불균질 σ(lnK)", value=float(het.get("sigma_lnK", 0.0)), key="sig",
                                            help="0이면 균질. 토양 Ks에 로그정규 무작위장을 곱함(가정)")
        het["corr_len"] = hc2.number_input("상관 길이 (m)", value=float(het.get("corr_len", 0.05)), key="clen")
        het["seed"] = hc3.number_input("난수 시드", value=int(het.get("seed", 0)), key="seed", step=1)
        c.heterogeneity = het
        st.markdown("층상 구조(위에서부터, 빈 구간은 위 토성 사용)")
        ldf = pd.DataFrame([asdict(L) for L in c.layers]) if c.layers else pd.DataFrame(
            [dict(top=0.0, bottom=c.H_soil, theta_r=tx["theta_r"], theta_s=tx["theta_s"], alpha=tx["alpha"],
                  n=tx["n"], Ks=tx["Ks"], aniso=1.0)])
        ldf = st.data_editor(ldf, num_rows="dynamic", key="layers", width="stretch",
                             column_config={"top": st.column_config.NumberColumn("상단(m)", help=None),
                                            "bottom": st.column_config.NumberColumn("하단(m)"),
                                            "theta_r": st.column_config.NumberColumn("θr", help=H["theta_r"]),
                                            "theta_s": st.column_config.NumberColumn("θs", help=H["theta_s"]),
                                            "alpha": st.column_config.NumberColumn("α (1/m)", help=H["alpha"]),
                                            "n": st.column_config.NumberColumn("n", help=H["n"]),
                                            "Ks": st.column_config.NumberColumn("Ks (m/s)", format="%.2e", help=H["Ks"]),
                                            "aniso": st.column_config.NumberColumn("Kz/Kx", help=H["aniso"])})
        c.layers = [SoilLayer(**{k: float(v) for k, v in r.items()}) for r in ldf.dropna().to_dict("records")]
    st.markdown("**유입수·초기 상태**")
    c.q_mm_day = num("침투 플럭스 q (mm/일)", c.q_mm_day, "q", H["q"])
    st.caption(f"= 흙수조 전체 {c.q_mm_day * c.Lx * c.Ly:.3f} L/일")
    with st.expander("강우·관개 시계열 / 증발산"):
        st.caption("계단형: 각 행의 '시작일'부터 다음 행까지 그 값(mm/일). 비우면 위의 q로 일정. CSV 열: 시작일, mm/일")
        upr = st.file_uploader("강우 CSV", type=["csv"], key="raincsv")
        if upr is not None:
            try:
                rdf = pd.read_csv(upr)
                c.rain_series = [[float(a), float(b)] for a, b in rdf.iloc[:, :2].values]
            except Exception as e:
                st.error(f"CSV를 읽지 못했습니다: {e}")
        rdf = pd.DataFrame(c.rain_series or [], columns=["시작일", "mm/일"])
        rdf = st.data_editor(rdf, num_rows="dynamic", key="rain_ed", hide_index=True)
        c.rain_series = [[float(a), float(b)] for a, b in rdf.dropna().values] if len(rdf) else []
        c.et_mm_day = num("잠재 증발산 (mm/일)", c.et_mm_day, "et", "지표에서 빠지는 물. 용질은 남아 농축됩니다(가정)")
        c.h_crit = num("증발 한계 압력수두 (m)", c.h_crit, "hcrit", "지표 셀이 이보다 마르면 증발량이 줄어듭니다")
    c.NO3_in_mgN = num("유입 NO₃-N (mg/L)", c.NO3_in_mgN, "no3", H["NO3_in"])
    c.SO4_in_mg = num("유입 SO₄ (mg/L)", c.SO4_in_mg, "so4", H["SO4_in"])
    c.Ca_in_mg = num("유입 Ca (mg/L)", c.Ca_in_mg, "ca", H["Ca_in"])
    c.alk_in_mgCaCO3 = num("유입 알칼리도 (mg/L as CaCO₃)", c.alk_in_mgCaCO3, "alk", H["alk_in"])
    c.soil_pH0 = num("초기 토양 pH", c.soil_pH0, "ph0", H["soil_pH0"])
    c.soil_NO3_mgN = num("초기 토양수 NO₃-N (mg/L)", c.soil_NO3_mgN, "sno3")
    c.gw_NO3_mgN = num("지하수 배경 NO₃-N (mg/L)", c.gw_NO3_mgN, "gno3")
with c3:
    st.markdown("**반응재(모든 비교군 동일)**")
    c.ratio_S_CaCO3 = num("황 : CaCO₃ 부피비 (x:1)", c.ratio_S_CaCO3, "ratio", H["ratio"])
    c.temperature = num("온도 (°C)", c.temperature, "temp", H["temperature"])
    st.markdown("**기간·계산**")
    c.days = num("운영 기간 (일)", c.days, "days", H["days"])
    pre = st.selectbox("격자 프리셋", list(GRID_PRESETS) + ["직접 입력"],
                       index=(list(GRID_PRESETS) + ["직접 입력"]).index(c.grid_preset), help=H["grid"])
    c.grid_preset = pre
    if pre != "직접 입력":
        c.nx, c.ny, c.nz = GRID_PRESETS[pre]
    g1, g2, g3 = st.columns(3)
    c.nx = int(g1.number_input("nx", value=int(c.nx), key="nx", step=1))
    c.ny = int(g2.number_input("ny", value=int(c.ny), key="ny", step=1))
    c.nz = int(g3.number_input("nz", value=int(c.nz), key="nz", step=1))
    g = make_grid(c)
    st.caption(f"셀 크기 {g.dx * 1000:.1f} × {g.dy * 1000:.1f} × {g.dz * 1000:.1f} mm · {g.N:,}칸")
    c.flow_mode = st.radio("흐름 해석", ["과도", "준정상"], index=["과도", "준정상"].index(c.flow_mode), horizontal=True,
                           help="과도(기본) = 매 단계 Richards 방정식. 준정상 = 빠른 근사(정해진 주기마다 정상해)")
    if c.flow_mode == "준정상":
        c.flow_update_days = num("흐름 재계산 주기 (일)", c.flow_update_days, "fud", H["flow_update_days"])
    else:
        f1, f2 = st.columns(2)
        c.flow_dt_max_h = f1.number_input("흐름 Δt 최대 (시간)", value=float(c.flow_dt_max_h), key="fdtm", format="%.3g")
        c.flow_dt_min_s = f2.number_input("Δt 하한 (초) — 넘어서도 수렴 못하면 FAILED", value=float(c.flow_dt_min_s),
                                          key="fdtn", format="%.3g")
    c.calcite_mode = st.radio("석회석 용해", ["속도론", "평형"], index=["속도론", "평형"].index(c.calcite_mode),
                              horizontal=True, help=H["calcite_mode"])
    st.markdown("**화학 엔진**")
    pst = PE.status()
    c.chem_engine = st.radio("지화학 계산", ["PHREEQC", "내장"], index=["PHREEQC", "내장"].index(c.chem_engine),
                             horizontal=True, help="PHREEQC(USGS) 라이브러리로 탄산계·이온쌍·석회석·석고와 SLAD 속도식을 계산. "
                                                   "내장 = 같은 식을 파이썬으로(빠름, 이온쌍 CaSO4°·CaCO3°·CaHCO3⁺ 포함)")
    st.caption(("PHREEQC 연결됨 " + pst.get("version", "") + f" · {os.path.basename(pst.get('lib', ''))}") if pst["ok"] == "예"
               else "PHREEQC 없음 → 내장 화학 사용 (" + pst.get("error", "")[:120] + ")")
    x1, x2 = st.columns(2)
    c.chem_dt_h = x1.number_input("PHREEQC 계산 간격 (시간)", value=float(c.chem_dt_h), key="cdt", format="%.3g")
    c.splitting = x2.selectbox("연산자 분리", ["Lie", "Strang"], index=["Lie", "Strang"].index(c.splitting),
                               help="Strang = 이송 ½ → 반응 → 이송 ½ (분리 오차가 작음, 느림)")
    c.pCO2_mode = st.radio("pCO₂", ["균일", "깊이별"], index=["균일", "깊이별"].index(c.pCO2_mode), horizontal=True,
                           help="깊이별 = 지표 값과 바닥 값 선형 보간(고급 파라미터 pCO2, pCO2_bottom). 모두 가정 값")
    st.markdown("**반응 속도식(v4, 문헌 기반)**")
    kin_opts = ["표면적(문헌, v4)", "부피(v3)"]
    c.kinetics = st.radio("SLAD 속도식", kin_opts, index=kin_opts.index(c.kinetics) if c.kinetics in kin_opts else 0,
                          horizontal=True,
                          help="표면적 = 체적 속도상수가 황 비표면적에 비례(Koenig & Liu 2001: 입경별 반-차수 상수). "
                               "고운 슬러리 입자는 빠르고 굵은 입상 황은 느림. 부피(v3) = 예전 식(입경 무시, 비교용)")
    st.markdown("**수분 조건(독립변수) — 불포화 탈질의 수분 의존(F9)**")
    mopts = MO.MODES
    c.moist_model = st.radio("수분 의존 모드(서로 배타 — 같은 억제를 두 번 세지 않음)", mopts,
                             index=mopts.index(c.moist_model) if c.moist_model in mopts else 0, key="m_mode",
                             help="A: r ∝ f_w(S)·f_O2(C_O2), f_w = S^p(물막 접촉 분율, 유도), f_O2 = K/(K+C_O2). "
                                  "B: NEMIS 계열 수분 함수만(f_O2는 탈질에 안 곱함, 호기성 황 산화 O₂ 소모는 계산). "
                                  "C: f_w = f_O2 = 1. S = θ/φ (WFPS)")
    pr = c.params
    ML = MO.mode_letter(c.moist_model)
    if ML == "A":
        a1, a2, a3 = st.columns(3)
        c.fw_form = a1.selectbox("f_w 형태", MO.FW_FORMS, index=MO.FW_FORMS.index(c.fw_form) if c.fw_form in MO.FW_FORMS else 0,
                                 key="m_fwf")
        if c.fw_form == "S^p":
            pr["fw_p"] = float(a2.number_input("p (f_w = S^p) — 가정, 범위 1/3~1", value=float(c.p("fw_p")), key="m_p",
                                               format="%.4g"))
        else:
            tc = a2.number_input("젖음각 θc (°) — 입력 필요", value=c.fw_theta_c, key="m_tc", format="%.4g")
            c.fw_theta_c = float(tc) if tc is not None else None
        pr["K_I_O2"] = float(a3.number_input("K_O2 (mg O₂/L) — 가정, 스윕 0.1~0.5", value=float(c.p("K_I_O2")) * 32.0,
                                             key="m_ko2", format="%.4g")) / 32.0
    elif ML == "B":
        b1, b2, b3 = st.columns(3)
        c.nemis_form = b1.selectbox("곡선 형태", MO.NEMIS_FORMS, index=MO.NEMIS_FORMS.index(c.nemis_form)
                                    if c.nemis_form in MO.NEMIS_FORMS else 0, key="m_nf")
        pr["nemis_thr"] = float(b2.number_input("임계 WFPS s_t — 원문 확인 필요, 스윕 0.60~0.80", value=float(c.p("nemis_thr")),
                                                key="m_thr2", format="%.3g"))
        if c.nemis_form == "멱함수":
            pr["nemis_w"] = float(b3.number_input("지수 w (NEMIS 1.74, 문헌)", value=float(c.p("nemis_w")), key="m_w2",
                                                  format="%.3g"))
        else:
            bb = b3.number_input("β (지수형) — 입력 필요", value=c.nemis_beta, key="m_beta", format="%.4g")
            c.nemis_beta = float(bb) if bb is not None else None
    elif ML == "D":
        d1, d2 = st.columns(2)
        c.agg_radius_m = d1.number_input("흙덩이 반지름 a (m) — 입력 필요", value=c.agg_radius_m, key="m_agg", format="%.4g")
        c.agg_R_O2 = d2.number_input("흙덩이 안 O₂ 소비율 R (mol/m³/s) — 입력 필요", value=c.agg_R_O2, key="m_aggR", format="%.4g")
    with st.expander("추가 반응 옵션(기본 꺼짐) — 아질산 중간체(D4)·종속영양 탈질(D1)"):
        o1, o2, o3 = st.columns(3)
        c.nitrite_on = o1.toggle("아질산 2단계 NO₃→NO₂→N₂", bool(c.nitrite_on), key="m_no2")
        if c.nitrite_on:
            c.k_NO2_ratio = o2.number_input("2단계/1단계 속도 비 (입력 필요)", value=c.k_NO2_ratio, key="m_no2r", format="%.4g")
        c.het_k_d = o3.number_input("토양 유기탄소 종속영양 탈질 1차 상수 (1/일, 빈칸 = 끔)", value=c.het_k_d, key="m_het",
                                    format="%.4g")
        st.caption("두 옵션은 내장 화학 엔진에서 계산됩니다. PHREEQC 모드는 온도(log K(T))를 입력 온도로 적용합니다.")
    c.well_model = st.toggle("가는 스크린·개구부를 우물 모델로 계산(v4.3, 격자 무관 주입)", bool(c.well_model), key="m_well",
                             help="끄면 v4.2 방식(개구부를 셀 전체로)")
    c.mode3 = st.toggle("결과에 모드 A·B·C를 항상 함께 계산(순위 일치 판정)", bool(c.mode3), key="m_mode3",
                        help="계산 시간이 약 3배")
    mm_opts = ["계산값(Richards)", "반응 구역 포화도 지정"]
    c.moist_mode = st.radio("반응 구역 수분", mm_opts, index=mm_opts.index(c.moist_mode) if c.moist_mode in mm_opts else 0,
                            horizontal=True,
                            help="계산값 = Richards가 계산한 S = θ/φ. 지정 = 반응 구역 S를 조작변수로 지정해 반응에 씀"
                                 "(물 흐름·이송·공기 확산은 계산값 그대로)")
    if c.moist_mode.startswith("반응 구역"):
        c.moist_set = float(st.slider("반응 구역 포화도 S = θ/φ (0~1)", 0.0, 1.0, float(c.moist_set), 0.01, key="m_set"))
    for e_ in validate_moist(c):
        st.error(e_)
    try:
        import plotly.graph_objects as go
        ss_ = np.linspace(0, 1, 101)
        fig_ = go.Figure()
        for pp in (1 / 3, 2 / 3, 1.0):
            fig_.add_scatter(x=ss_, y=MO.f_w(ss_, pp), mode="lines", name=f"A: S^{pp:.2g}", line=dict(dash="dot", width=1))
        for tt in (0.6, 0.7, 0.8):
            fig_.add_scatter(x=ss_, y=MO.f_nemis(ss_, tt, c.p("nemis_w")), mode="lines", name=f"B: s_t {tt}",
                             line=dict(dash="dash", width=1))
        try:
            cur_f, _ = MO.factors(c.moist_model, ss_, None, moist_cfg(c))
            fig_.add_scatter(x=ss_, y=cur_f, mode="lines", name="지금 설정", line=dict(width=3))
        except ValueError:
            pass
        if c.moist_mode.startswith("반응 구역"):
            fig_.add_vline(x=c.moist_set, line_dash="dash")
        fig_.update_layout(height=240, margin=dict(l=40, r=10, t=10, b=30), xaxis_title="S = θ/φ (WFPS)",
                           yaxis_title="탈질 속도 배율(수분)", legend=dict(orientation="h", y=-0.4))
        show_fig(fig_, key="moistcurve")
    except Exception:       # noqa: BLE001
        pass
    c.o2_stage = st.checkbox("O₂ 추적(호기성 황 산화·탈질 억제)", c.o2_stage,
                             help="공기·유입수 O₂가 황을 산화(황 소모·산 생성, Watkinson & Blair 1993 표면 속도)하고 포화 셀에서 탈질 억제")
    c.o2_gas = st.checkbox("토양 공기 O₂ 확산(지표 대기와 교환)", c.o2_gas, disabled=not c.o2_stage,
                           help="Millington & Quirk (1961) 유효 확산. 끄면 처음 공극 공기의 O₂만 씀(v3 방식)")
    c.port_thin_as_soil = st.checkbox("격자보다 얇은 개구부는 흙 물성 출구로", c.port_thin_as_soil,
                                      help="관 벽의 구멍처럼 셀보다 훨씬 얇은 개구부를 빈 공간(α 50, n 3)이 아니라 주변 흙과 같은 물성의 "
                                           "주입 출구로 둔다. 끄면 v4.3 방식(빈 공간).")
    cm_opts = ["SLAD", "추적자(반응 없음)", "1차 감쇠"]
    c.chem_mode = st.selectbox("반응 모드", cm_opts, index=cm_opts.index(c.chem_mode),
                               help="대체품 모드: 염료·추적자 실험(반응 없음) 또는 1차 감쇠로 바꿔 실험과 비교")
    if c.chem_mode == "1차 감쇠":
        c.k1_decay_day = num("1차 감쇠 상수 (1/일)", c.k1_decay_day, "k1d")
    st.markdown("**주입 계산**")
    c.inject_model = st.radio("주입 흐름", ["과도 Richards", "중첩 근사(빠름)"],
                              index=["과도 Richards", "중첩 근사(빠름)"].index(c.inject_model), horizontal=True,
                              help="과도 Richards = 원천항을 넣어 불포화 흐름을 그대로 풂(기본). 중첩 근사 = 빠르지만 낙관적")
    c.straining = st.checkbox("입자 크기로 필터 케이크 판정", c.straining,
                              help="입자 지름/토양 d50 > 0.154 이면 입자가 흙에 못 들어가고 개구부에 쌓인다고 봄(문헌 기준, 토양 d50은 가정)")
    c.n_snapshots = int(num("저장할 3D 장면 수", float(c.n_snapshots), "nsnap"))
    c.light = st.toggle("가벼운 모드(프레임·해상도 축소)", c.light)
    c.redistribute_runoff = st.checkbox("지표 유출 재분배(가정, 기본 켬)", c.redistribute_runoff,
                                        help="구조물 덮개나 포화된 지표에서 못 들어간 물을 열린 흙 표면으로 고르게 보냅니다. "
                                             "끄면 그 물(과 질산염)은 흙수조 밖으로 사라진 것으로 봅니다.")
    st.markdown("**입자 포획 계수 λ (F4)**")
    LMODES = ["직접 입력", "여과 이론(TE 2004)"]
    c.lambda_mode = st.radio("λ 계산", LMODES, index=LMODES.index(c.lambda_mode if c.lambda_mode in LMODES else LMODES[0]),
                             horizontal=True, key="lam_mode",
                             help="직접 입력 = 고급 파라미터 lambda_filt. 여과 이론 = Tufenkji & Elimelech(2004) η0로 λ = 3(1−φ)αη0/(2d_c)")
    if c.lambda_mode.startswith("여과"):
        c.attach_alpha = num("부착 효율 α (필수)", c.attach_alpha, "lam_a", "근거 있는 기본값이 없어 직접 입력합니다(R3)")
        c.filt_U_m_s = num("여과 계산 다르시 유속 (m/s, 필수)", c.filt_U_m_s, "lam_u", "주입 중 개구부 주변 유속 규모")
    st.markdown("**재보충 판정 (X2·X4)**")
    c.refill_ok_pct = num("재보충 '성공' 기준 전달률 (%)", c.refill_ok_pct, "rf_ok", "계획 대비 실제 전달률이 이 값 이상일 때만 성공")
    c.eta_R_min_pct = num("R 계산 하한 η (%)", c.eta_R_min_pct, "rf_eta", "재보충 전 η가 이보다 작으면 R을 계산하지 않음")
    st.markdown("**삽입 교란(가정, 기본 없음)**")
    c.disturb_thick = num("교란대 두께 (m)", c.disturb_thick, "dth", H["disturb"])
    c.disturb_Kmult = num("교란대 K 배율", c.disturb_Kmult, "dkm", H["disturb"])
with st.expander("고급 파라미터(반응·수송) — 출처/가정 라벨"):
    rows = [{"키": k, "값": c.params.get(k, v[0]), "단위": v[1], "라벨": TX.STATUS_LABELS.get(v[2], v[2]), "설명": v[3]}
            for k, v in REGISTRY.items()]
    pdf = st.data_editor(pd.DataFrame(rows), disabled=["키", "단위", "라벨", "설명"], hide_index=True, key="adv",
                         width="stretch", column_config={"값": st.column_config.NumberColumn(format="%.4g")})
    for r in pdf.to_dict("records"):
        if r["값"] is not None and r["값"] != REGISTRY[r["키"]][0]:
            c.params[r["키"]] = float(r["값"])
        elif r["키"] in c.params and r["값"] == REGISTRY[r["키"]][0]:
            c.params.pop(r["키"])
errs, warns = validate_common(c)
for e in errs:
    st.error(e)
for w in warns:
    st.warning(w)
if ss().scen:
    st.markdown("**비교군별 반응재 총 질량(통제변수: 모두 같아야 함)**")
    rows = []
    for s_ in ss().scen:
        try:
            so = scenario_obj(s_, c)
            gm = build_geometry(so, c, g) if so.objects else None
            pm = planned_masses(so, c, g, gm) if gm else {"total_g": 0, "pre_mesh_g": 0, "injection_g": 0}
            rows.append({"비교군": s_["name"], "사전 배치(g)": round(pm["pre_mesh_g"], 1), "주입 계획(g)": round(pm["injection_g"], 1),
                         "합계(g)": round(pm["total_g"], 1)})
        except Exception as e:
            rows.append({"비교군": s_["name"], "사전 배치(g)": None, "주입 계획(g)": None, "합계(g)": None})
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf, hide_index=True, width="stretch")
    tot = [r["합계(g)"] for r in rows if r["합계(g)"]]
    if len(tot) > 1 and (max(tot) - min(tot)) / max(max(tot), 1e-9) > 0.05:
        st.warning("비교군별 반응재 총 질량이 5% 넘게 다릅니다. 지표 문서: 질량이 다르면 '구조의 차이'가 아니라 '양의 차이'를 재게 됩니다. "
                   "① 탭의 '질량 맞추기 도우미'로 주입량을 조정하세요.")
