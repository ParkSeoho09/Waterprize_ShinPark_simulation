"""tab1_input — app_ui.py의 탭 1 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
top = st.columns([3, 2, 2, 2])
with top[0]:
    names = [s["name"] for s in ss().scen]
    if names:
        ss().cur = min(ss().cur, len(names) - 1)
        ss().cur = st.radio("비교군 선택", range(len(names)), format_func=lambda i: names[i],
                            index=ss().cur, horizontal=True, key="scen_radio")
    else:
        st.info("비교군이 없습니다. 오른쪽에서 추가하거나 예시를 불러오세요. 비교군이 없어도 ④ 결과 탭에서 기본 대조군(꽂지 않았을 때)은 볼 수 있습니다.")
with top[1]:
    if st.button("➕ 빈 비교군 추가", width="stretch"):
        ss().scen.append(new_scenario(f"비교군 {len(ss().scen) + 1}"))
        ss().cur = len(ss().scen) - 1
        st.rerun()
with top[2]:
    ALL_EX = "예시 전부 (영구 주입공 + 케이슨형)"
    ex = st.selectbox("예시 불러오기", ["—", ALL_EX] + list(SAMPLES.keys()), key="ex_sel",
                      help="고른 뒤 아래 '예시 추가'를 눌러야 비교군 목록에 들어갑니다.")
    if ex != "—" and st.button("예시 추가", width="stretch", type="primary"):
        have = {s_["name"] for s_ in ss().scen}
        for exk in (list(SAMPLES.keys())[:2] if ex == ALL_EX else [ex]):
            nm_ = exk.split(" (")[0]
            if ex == ALL_EX and nm_ in have:
                continue
            path, mech, text, form, rows = SAMPLES[exk]
            sc_new = new_scenario(nm_)
            with open(os.path.join(HERE, path), "rb") as fh:
                data_ = fh.read()
            import geometry_design as GD
            objs_ = load_mesh_file(data_, os.path.basename(path))["objects"]
            load_file_into(sc_new, data_, os.path.basename(path), preset=GD.sample_roles(os.path.join(HERE, path), objs_))
            sc_new["mechanism"] = mech
            sc_new["install_text"] = text
            sc_new["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
            sc_new["reagent"] = {"form": form, "material": "선택 안 함"}
            ss().scen.append(sc_new)
        ss().cur = len(ss().scen) - 1
        st.rerun()
with top[3]:
    if ss().scen:
        m1, m2, m3, m4 = st.columns(4)
        if m1.button("⧉", help="선택 비교군 복제", width="stretch"):
            src = ss().scen[ss().cur]
            dup = {k: (v.copy() if isinstance(v, (pd.DataFrame, dict)) else (list(v) if isinstance(v, list) else v))
                   for k, v in src.items()}
            dup["placement"] = dict(src["placement"]); dup["reagent"] = dict(src.get("reagent") or {})
            dup["name"] = src["name"] + " (복제)"
            ss().scen.insert(ss().cur + 1, dup)
            ss().cur += 1
            st.rerun()
        if m2.button("▲", help="위로", width="stretch") and ss().cur > 0:
            i = ss().cur; ss().scen[i - 1], ss().scen[i] = ss().scen[i], ss().scen[i - 1]; ss().cur -= 1
            st.rerun()
        if m3.button("▼", help="아래로", width="stretch") and ss().cur < len(ss().scen) - 1:
            i = ss().cur; ss().scen[i + 1], ss().scen[i] = ss().scen[i], ss().scen[i + 1]; ss().cur += 1
            st.rerun()
        if m4.button("🗑", help="선택 비교군 삭제", width="stretch"):
            ss().scen.pop(ss().cur)
            ss().cur = max(0, ss().cur - 1)
            st.rerun()

if ss().scen:
    sc = ss().scen[ss().cur]
    k = f"s{id(sc)}"
    left, right = st.columns([1.05, 1])
    with left:
        sc["name"] = st.text_input("비교군 이름", sc["name"], key=k + "name")
        up = st.file_uploader("구조물 3MF (또는 STL)", type=["3mf", "stl"], key=k + "up")
        if up is not None and up.name != sc["file_name"]:
            try:
                load_file_into(sc, up.getvalue(), up.name)
                st.success(f"{up.name}: 객체 {len(sc['objects'])}개, 단위 {UNIT_LABEL.get(sc['unit_src'])}")
            except Exception as e:
                st.error(f"3MF를 읽지 못했습니다: {e}")
        if sc["file_name"]:
            st.caption(f"파일: **{sc['file_name']}** · 3MF 단위 속성: {UNIT_LABEL.get(sc['unit_src'], sc['unit_src'])}"
                       + (f" · 제목: {sc['title']}" if sc["title"] else ""))
            if sc["description"]:
                with st.expander("3MF에 적힌 설명"):
                    st.write(sc["description"])
        sc["mechanism"] = st.text_area("작용 메커니즘 설명 (결과·내보내기에 그대로 표시)", sc["mechanism"], height=90,
                                       key=k + "mech")
        u1, u2 = st.columns(2)
        with u1:
            cur_u = UNIT_FROM.get(sc["unit_sel"], "mm")
            newu = st.selectbox("좌표 단위", ["mm", "cm", "m"], index=["mm", "cm", "m"].index(cur_u),
                                help="기본은 3MF의 unit 속성입니다. 파일 단위가 틀렸을 때만 바꾸세요. 자동 축소는 하지 않습니다.",
                                key=k + "unit")
            if UNIT_KEYS[newu] != sc["unit_sel"] and sc["objects"]:
                sc["objects"] = rescale_objects(sc["objects"], sc["unit_sel"], UNIT_KEYS[newu])
                sc["unit_sel"] = UNIT_KEYS[newu]
                sc["geom_report"] = None
        with u2:
            sc["placement"]["scale"] = num("배율(사용자가 정할 때만)", float(sc["placement"]["scale"]), k + "scale",
                                           help="1이면 원래 크기. 축소가 필요하면 직접 비율을 적으세요.")
        p1, p2, p3 = st.columns(3)
        with p1:
            sc["placement"]["x"] = num("위치 x (m)", sc["placement"]["x"] if sc["placement"]["x"] is not None else c.Lx / 2,
                                       k + "px", help="3MF 원점(보통 구조물 축)이 놓일 흙수조 x 좌표")
            sc["placement"]["rot_deg"] = num("회전 (°)", float(sc["placement"]["rot_deg"]), k + "rot")
        with p2:
            sc["placement"]["y"] = num("위치 y (m)", sc["placement"]["y"] if sc["placement"]["y"] is not None else c.Ly / 2,
                                       k + "py")
            sc["placement"]["tilt_deg"] = num("기울기 (°)", float(sc["placement"]["tilt_deg"]), k + "tilt")
        with p3:
            sc["placement"]["top_depth"] = num("3MF z=0 깊이 (m)", float(sc["placement"]["top_depth"]), k + "pd",
                                               help="0이면 3MF의 z=0 면이 지표에 놓입니다.")
        if len(sc["role_df"]):
            st.markdown("**객체별 역할표** — 이름 키워드로 제안된 값입니다. 바꿀 수 있습니다.")
            cfg = {
                "객체": st.column_config.TextColumn(disabled=True),
                "역할": st.column_config.SelectboxColumn(options=ROLES, required=True),
                "K (m/s)": st.column_config.NumberColumn(format="%.2e", help="빈칸 = 토양 값"),
                "공극률": st.column_config.NumberColumn(format="%.3f", help="빈칸 = 토양 값"),
                "반응": st.column_config.CheckboxColumn(help="반응재(고체)일 때 SLAD 반응 여부"),
                "우선순위": st.column_config.NumberColumn(format="%d", help="겹칠 때 큰 값이 이깁니다"),
                "교체 가능": st.column_config.CheckboxColumn(help="카트리지처럼 꺼내 새것으로 바꿀 수 있는 반응재"),
                "얇은 객체": st.column_config.SelectboxColumn(options=["자동", "면 래스터화", "체적만"]),
                "벽 두께(mm)": st.column_config.NumberColumn(format="%.3f", help="자동 추정 최소 두께. 실링 장벽 저항 계산에 씀"),
                "구멍 지름(mm)": st.column_config.NumberColumn(format="%.2f", help="투수성 판(구멍)만: 구멍 지름"),
                "구멍 개수": st.column_config.NumberColumn(format="%.0f", help="투수성 판만: 개수(빈칸이면 개구율로 계산)"),
                "개구율": st.column_config.NumberColumn(format="%.3f", help="투수성 판만: 구멍 면적 / 판 면적"),
                "판 두께(mm)": st.column_config.NumberColumn(format="%.2f", help="투수성 판만: 판 두께 t_p"),
                "구멍 채움": st.column_config.SelectboxColumn(options=["반응재/흙", "메시", "빈 공간"]),
                "K_h (m/s)": st.column_config.NumberColumn(format="%.2e", help="구멍 채움재 투수계수(메시는 필수, 반응재/흙은 빈칸=반응재 K)"),
                "체적(cm³)": st.column_config.NumberColumn(disabled=True, format="%.2f"),
                "닫힘": st.column_config.CheckboxColumn(disabled=True, help="닫힌 메시인지(아니면 내부 판정 부정확)"),
            }
            sc["role_df"] = st.data_editor(sc["role_df"], column_config=cfg, hide_index=True, key=k + "roles",
                                           width="stretch")
            st.caption("역할 기본값: " + "; ".join(f"{r} K={ROLE_DEFAULTS[r]['K']} φ={ROLE_DEFAULTS[r]['phi']}" for r in ROLES[:6])
                       + " — 모두 '가정' 값입니다.")
            if st.button("복셀 검사(격자에 올려 보기)", key=k + "vox"):
                g = make_grid(c)
                try:
                    geom = build_geometry(scenario_obj(sc, c), c, g)
                    pm = planned_masses(scenario_obj(sc, c), c, g, geom)
                    sc["geom_report"] = {"counts": geom["comp"]["counts"], "warnings": geom["warnings"],
                                         "errors": geom["errors"], "masses": pm,
                                         "faces": {ov.name: ov.n_faces for ov in geom["ov"]}}
                except Exception as e:
                    sc["geom_report"] = {"errors": [str(e)], "warnings": [], "counts": {}, "masses": {}, "faces": {}}
            rep = sc["geom_report"]
            if rep:
                for e in rep["errors"]:
                    st.error(e)
                df = pd.DataFrame([{"객체": n, "셀 수": v, "장벽 면 수": rep["faces"].get(n, 0)} for n, v in rep["counts"].items()])
                st.dataframe(df, hide_index=True, width="stretch")
                for w in rep["warnings"]:
                    st.warning(w)
                if rep["masses"]:
                    m = rep["masses"]
                    st.caption(f"반응재 질량: 사전 배치 {m['pre_mesh_g']:.0f} g(메시 체적) / {m['pre_voxel_g']:.0f} g(복셀) · "
                               f"설치 주입 계획 {m['injection_g']:.0f} g · 합계 {m['total_g']:.0f} g")
        sc["mass_correction"] = st.checkbox("반응재 질량을 메시 체적 기준으로 보정(복셀 오차 제거)", sc["mass_correction"], key=k + "mc")
        sc["allow_refill_wo_opening"] = st.checkbox("개구부 없이도 재보충 허용(가정 — 반응재 객체를 교체하는 것으로 봄)",
                                                    sc["allow_refill_wo_opening"], key=k + "arw")
    with right:
        if sc["objects"]:
            show_fig(preview_fig(sc, c), key=k + "prev")
        else:
            st.info("3MF를 올리면 객체별 색으로 3D 미리보기가 나타납니다.")

    st.markdown("---")
    st.subheader("시약 형태 — 필수")
    rg = sc.setdefault("reagent", {"form": None, "material": "선택 안 함"})
    r1, r2 = st.columns([1, 2])
    with r1:
        opts = ["(선택하세요)"] + REAGENT_FORMS
        cur_f = rg.get("form") if rg.get("form") in REAGENT_FORMS else "(선택하세요)"
        f_ = st.selectbox("시약 형태", opts, index=opts.index(cur_f), key=k + "rform",
                          help="(a) 미리 넣어 둔 고체 (b) 꺼내 바꾸는 고체 카트리지 (c) 입자 슬러리를 주입해 흙에 걸리게 함 "
                               "(d) 녹아 있는 전자공여체 용액 (e) 콜로이드·나노 현탁액")
        rg["form"] = f_ if f_ in REAGENT_FORMS else None
    with r2:
        if rg.get("form", "") and rg["form"].startswith("(d)"):
            mats = LIQUID_MATERIALS
            rg["material"] = st.selectbox("액체 시약 물질", mats, index=mats.index(rg.get("material", "선택 안 함"))
                                          if rg.get("material") in mats else 0, key=k + "rmat")
            if rg["material"] == "티오황산염(S2O3²⁻)" and not rg.get("_thio_filled"):
                rg.update({kk: vv for kk, vv in THIOSULFATE_STOICH.items()}); rg["_thio_filled"] = True
            st.caption("양론(NO₃ 1 mol당): 공여체 소비 yD, SO₄ 생성 ySO4, 산 생성 yH, 공여체 1 mol의 황 sD. "
                       "속도: Monod 이중 제한 k·[NO₃]/(KN+[NO₃])·[D]/(KD+[D]). **빈칸 = 미정 → 계산하지 않음**(값을 지어내지 않음). "
                       "티오황산염 양론은 문헌 참고값(5 S₂O₃²⁻ + 8 NO₃⁻ → 10 SO₄²⁻ + 4 N₂ + 2 H⁺), 속도는 미정.")
            q1, q2, q3 = st.columns(3)
            for col, keys in ((q1, ("yD", "ySO4", "yH")), (q2, ("sD", "k", "KN")), (q3, ("KD", "conc_molL", "Rf"))):
                with col:
                    for kk in keys:
                        lab = {"yD": "yD (mol/mol)", "ySO4": "ySO4 (mol/mol)", "yH": "yH (eq/mol)", "sD": "sD (mol S/mol)",
                               "k": "k (mol/m³/s)", "KN": "KN (mol/m³)", "KD": "KD (mol/m³)",
                               "conc_molL": "주입 농도 (mol/L)", "Rf": "지연계수 R (흡착)"}[kk]
                        v = st.number_input(lab, value=rg.get(kk), key=k + "rg" + kk, format="%.4g")
                        rg[kk] = v
        elif rg.get("form", "") and rg["form"].startswith("(e)"):
            rg["lambda_colloid"] = st.number_input("콜로이드 포획 계수 λ (1/m)", value=rg.get("lambda_colloid"),
                                                   key=k + "rgl", format="%.4g", help="빈칸 = 미정(계산 안 함). 기본 가정값 50")
            rg["d_p"] = st.number_input("콜로이드 입경 (m)", value=rg.get("d_p"), key=k + "rgd", format="%.3e",
                                        help="빈칸 = 미정. 필터 케이크(막힘) 판정에도 씀")
        elif rg.get("form"):
            st.caption({"(a)": "반응재 역할 객체에 미리 채워진 황:석회석. 재보충은 개구부가 있을 때만 가능.",
                        "(b)": "역할표에서 '교체 가능'을 체크한 반응재 객체를 재보충 때 새것으로 바꿉니다(설치 표 방식 = 카트리지 교체).",
                        "(c)": "설치 표의 '시약 주입'으로 슬러리를 넣습니다. 입자는 여과 포획되고, 토양 입경에 비해 크면 개구부에 필터 케이크로 쌓입니다."
                        }.get(rg["form"][:3], ""))
    for e_ in validate_reagent(scenario_obj(sc, c)):
        st.error(e_)
    design_editor(sc, c, k)
    st.subheader("A1. 설치(꽂는) 방법 — 필수")
    st.caption("어떻게 꽂고, 어떻게 시약을 넣는지 적으세요. 시뮬레이터는 이 표대로 구조물을 넣고 시약을 주입하는 과정을 재현합니다.")
    a, b = st.columns([1, 1.4])
    with a:
        sc["install_text"] = st.text_area("설치 과정 설명(자유 텍스트)", sc["install_text"], height=180, key=k + "itxt",
                                          placeholder="예) 1) 케이싱을 30 cm 깊이까지 20분 동안 삽입한다. 2) 포트를 연다. "
                                                      "3) 고형분 20% 슬러리 2 L를 펌프로 0.2 L/min 주입한다. 4) 60분 정치. "
                                                      "5) 효율 50% 미만이면 카트리지 교체, 최대 8회, 15분.")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("텍스트 → 표 (규칙 기반)", key=k + "parse"):
                rows = ie.parse_text(sc["install_text"], [o.name for o in sc["objects"]])
                if rows:
                    sc["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                    st.success(f"{len(rows)}개 단계를 제안했습니다. 표에서 확인·수정하세요.")
                else:
                    st.warning("텍스트에서 단계를 찾지 못했습니다. 표에 직접 입력하세요.")
        with b2:
            with st.popover("LLM으로 변환(선택)") if hasattr(st, "popover") else st.expander("LLM으로 변환(선택)"):
                key_in = st.text_input("Anthropic API 키(저장 안 함)", type="password", key=k + "apikey")
                model = st.text_input("모델", "claude-sonnet-5-5", key=k + "model")
                if st.button("변환", key=k + "llm") and key_in:
                    try:
                        rows = ie.parse_text_llm(sc["install_text"], [o.name for o in sc["objects"]], key_in, model)
                        sc["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                        st.success(f"{len(rows)}개 단계를 받았습니다.")
                    except Exception as e:
                        st.error(f"LLM 변환 실패: {e} — 규칙 기반 변환이나 직접 입력을 쓰세요.")
    with b:
        ecfg = {
            "이벤트": st.column_config.SelectboxColumn(options=ie.EVENT_TYPES, required=True),
            "방식": st.column_config.SelectboxColumn(options=ie.METHODS),
            "시약 형태": st.column_config.SelectboxColumn(options=ie.REAGENT_FORM_OPTIONS,
                                                      help="시약 주입 행에서 비교군 기본 형태 대신 쓸 형태(비우면 비교군 형태)"),
            "대상 객체": st.column_config.TextColumn(help="쉼표로 여러 개, '전체' = 해당 역할 객체 모두"),
        }
        for col in ie.NUMERIC:
            ecfg[col] = st.column_config.NumberColumn(format="%.4g")
        df = sc["events_df"]
        for col in ie.COLUMNS:
            if col not in df.columns:
                df[col] = None
        sc["events_df"] = st.data_editor(df[ie.COLUMNS], num_rows="dynamic", column_config=ecfg, hide_index=True,
                                         key=k + "events", width="stretch")
        roles_of = {r["객체"]: r["역할"] for r in sc["role_df"].to_dict("records")} if len(sc["role_df"]) else {}
        errs, warns = ie.validate(sc["events_df"].to_dict("records"), roles_of)
        for e in errs:
            st.error(e)
        for w in warns:
            st.warning(w)
        sched = ie.schedule(sc["events_df"].to_dict("records"))
        tr = [r for r in sched if r["이벤트"] == "재보충"]
        if sched:
            st.caption("타임라인: " + " → ".join(
                f"{r['이벤트']}({r['t_start'] * 1440:.0f}분)" if r["t_start"] is not None else f"{r['이벤트']}(운영 중)"
                for r in sched))
        # 질량 맞추기 도우미
        with st.expander("반응재 총 질량 맞추기 도우미"):
            ref = st.number_input("목표 반응재 질량 (g)", value=1265.0, key=k + "mtgt")
            fs = st.number_input("슬러리 고형분비 (v/v)", value=0.2, key=k + "mfs")
            r_ = c.ratio_S_CaCO3
            rho = (r_ * 2.07 + 2.71) / (r_ + 1)
            st.write(f"필요 슬러리 부피 ≈ **{ref / (max(fs, 1e-9) * rho * 1000):.3f} L** (고체 혼합 밀도 {rho:.2f} g/cm³)")

# ───── Part 1·2 형상·크기 계산기 ─────
with st.expander("형상·크기 계산기 — 깊이 구간 · 스크린 길이 · 케이슨 치수 (Part 1·2)"):
    import geometry_design as GD
    try:
        zone = GD.depth_zone(c)
        st.write(f"깊이 구간: 지표 ~ **{zone['bot_m']:.3f} m** (지하수면 {zone['wt_m']:.3f} m — {zone['wt_basis']}; "
                 f"여유 {zone['margin_m']:.3f} m — {zone['margin_basis']})")
    except ValueError as e:
        st.error(str(e)); zone = None
    if zone:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**영구 주입공 — 기하 상사 축척 (B2)**")
            c.field_unsat_m = num("현장 불포화대 두께 (m)", c.field_unsat_m, "gd_fu")
            c.field_screen_m = num("현장 스크린 길이 (m)", c.field_screen_m, "gd_fs")
            c.field_casing_d_m = num("현장 케이싱 지름 (m)", c.field_casing_d_m, "gd_fd")
            c.casing_frac = num("케이싱 지름 / 흙수조 폭 비", c.casing_frac, "gd_cf")
            wd_def = None
            try:
                sw = GD.scaled_well({"unsat_m": c.field_unsat_m, "screen_m": c.field_screen_m,
                                     "casing_d_m": c.field_casing_d_m, "casing_frac": c.casing_frac}, zone, c,
                                    printer_min_m=c.printer_min_hole_m)
                st.caption(sw["basis"])
                st.dataframe(pd.DataFrame(sw["rows"]), hide_index=True, width="stretch")
                wd_def = sw["casing_d_m"]
            except ValueError as e:
                st.warning(str(e))
            Qd = num("설계 주입률 (L/min)", 0.2, "gd_q")
            rf_ = GD.field_reference_Ls(Qd)
            st.caption(f"{rf_['label']}: L_s = {rf_['L_s_m'] * 1e3:.1f} mm ({rf_['basis']}) — 기본 계산에 쓰지 않음")
            wd = num("주입공 지름 (m)", wd_def, "gd_wd")
            ww = num("케이싱 벽 두께 (m)", 0.002, "gd_ww")
            fr_ = st.selectbox("스크린 길이 배수(가용 구간 대비)", [0.25, 0.5, 0.75, 1.0], index=3, key="gd_fr")
            ns_ = st.selectbox("스크린 구간 수", [1, 2, 3], index=0, key="gd_ns")
            if st.button("이 주입공을 비교군으로 추가", key="gd_addw"):
                try:
                    w_ = GD.injection_well(wd, ww, zone, fr_, ns_)
                    sc_new = new_scenario(f"주입공(스크린 {fr_:g}×, {ns_}구간)")
                    load_file_into(sc_new, w_["bytes"], "injection_well_unsat.3mf")
                    sc_new["events_df"] = pd.DataFrame(SAMPLES["영구 주입공 (injection_well)"][4], columns=ie.COLUMNS)
                    sc_new["reagent"] = {"form": "(c) 입자 슬러리(포획)", "material": "선택 안 함"}
                    ss().scen.append(sc_new); ss().cur = len(ss().scen) - 1
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        with g2:
            st.markdown("**케이슨 — 불포화대 적용형 치수 (Part 2)**")
            c.target_S_g = num("목표 황 질량 (g S) — 두 비교군 공통(B1)", c.target_S_g, "gd_tS",
                               "빈칸이면 영구 주입공이 가장 촘촘한 수렴 격자에서 실제로 전달한 황 질량을 넣으세요")
            if c.target_S_src:
                st.caption("출처: " + c.target_S_src)
            import design as DZ0_
            fS0_ = 1.0 / DZ0_.solids_composition(c.ratio_S_CaCO3, "없음", 0.0)["m_solid_per_gS"]
            mass_ = (c.target_S_g / fS0_) if c.target_S_g else None
            if mass_ is None:
                st.warning("목표 황 질량을 입력하세요.")
                mass_ = 0.0
            st.caption(f"총 반응재 질량 = {mass_:.1f} g (황 질량분율 {fS0_:.3f})")
            Smed = num("반응재 포화도 S (WFPS)", 0.5, "gd_s")
            Tday = num("목표 운영 기간 (일)", 180.0, "gd_T")
            SF_ = num("안전계수 SF (ITRC 2~3)", 2.0, "gd_sf")
            tgt = num("목표 제거율", 0.9, "gd_tg")
            cO2 = num("반응재 O₂ (mg/L, 모드 A)", 0.2, "gd_o2")
            tmin = 0.0      # B3: 구조 최소 = 3D 프린터 최소 두께(아래 입력). 격자 칸은 쓰지 않는다.
            c.printer_min_wall_m = num("프린터 최소 벽 두께 (m, 필수)", c.printer_min_wall_m, "gd_pw", "사용할 3D 프린터의 값")
            c.printer_min_hole_m = num("프린터 최소 구멍 지름 (m, 필수)", c.printer_min_hole_m, "gd_ph", "사용할 3D 프린터의 값")
            pw, ph = c.printer_min_wall_m, c.printer_min_hole_m
            if pw is None or ph is None:
                st.warning("3D 프린터 최소 벽 두께·구멍 지름을 입력하세요(케이슨 치수·구멍 판의 하한).")
                pw = ph = 0.0
            import design as DZ_
            fS_ = 1.0 / DZ_.solids_composition(c.ratio_S_CaCO3, "없음", 0.0)["m_solid_per_gS"]
            inp = dict(A_tank_m2=c.Lx * c.Ly, zone_len_m=zone["length_m"], mass_g=mass_, porosity=c.packed_porosity,
                       S_mass_frac=fS_, C0_mgN=c.NO3_in_mgN, q_mm_day=c.q_mm_day, T_days=Tday, SF=SF_, target=tgt,
                       S_media=Smed, d_S_m=c.p("S_d_media"), K_soil_m_s=float(c.Ks_override or SOIL_TEXTURES[c.texture]["Ks"]))
            try:
                cal = GD.caisson_calc(inp, moist_cfg(c), t_min_m=max(tmin, pw), C_O2_molm3=cO2 / 32.0,
                                      k_half_a=c.p("k_half_a"), k_ox_a=c.p("k_ox_a"))
                st.caption(cal["basis"])
                st.dataframe(pd.DataFrame(cal["rows"]), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame(cal["K_sweep"]), hide_index=True, width="stretch")
                st.caption(f"점검: Moon 2004 {cal['moon_check']['Moon 2004 점검값 (mg/L)^½/h']} vs 앱 포화 k_w "
                           f"{cal['moon_check']['앱 포화 k_w (mg/L)^½/h']:.2f} (mg/L)^½/h — 원문 확인 필요")
                okrows = [r_ for r_ in cal["rows"] if r_.get("상태") == "OK"]
                if okrows:
                    pick = st.selectbox("3MF로 만들 모드", [r_["모드"] for r_ in okrows], key="gd_pick")
                    rr_ = next(r_ for r_ in okrows if r_["모드"] == pick)
                    D_ = num("지름 (m) — 계산값, 수정 가능", float(rr_["지름 m(원형)"]), "gd_D")
                    T_ = num("반응재 두께 (m) — 계산값, 수정 가능", float(rr_["두께 m"]), "gd_thk")
                    hd = num("판 구멍 지름 (m)", ph or None, "gd_hd"); orr = num("판 개구율", 0.2, "gd_or"); tp = num("판 두께 (m)", 0.003, "gd_tp")
                    for e_ in GD.printer_check({"wall_m": pw or None, "hole_d_m": hd},
                                               {"min_wall_m": c.printer_min_wall_m or 0.0, "min_hole_m": c.printer_min_hole_m or 0.0}):
                        st.error("프린터 한계: " + e_)
                    res_ = GD.resolution_check(D_, c.Lx / c.nx)
                    for m_ in res_["log"]:
                        st.caption("K4: " + m_)
                    if st.button("이 케이슨을 비교군으로 추가", key="gd_addc"):
                        try:
                            cd_ = GD.caisson_down(D_, T_, 0.0, pw, tp, zone)
                            pre = {o["name"]: ({"role": PLATE_ROLE, "hole_d_m": hd, "open_ratio": orr, "t_p_m": tp,
                                                "fill": "반응재/흙"} if o["name"].startswith("perforated") else
                                               ({"accessible": True} if o["name"] == "reactive_media" else {}))
                                   for o in cd_["objects"]}
                            sc_new = new_scenario(f"케이슨(하향, D {D_ * 100:.1f} cm)")
                            load_file_into(sc_new, cd_["bytes"], "caisson_down.3mf", preset=pre)
                            sc_new["events_df"] = pd.DataFrame(SAMPLES["케이슨형 (caisson_down)"][4], columns=ie.COLUMNS)
                            sc_new["reagent"] = {"form": "(b) 고체 카트리지 교체", "material": "선택 안 함"}
                            ss().scen.append(sc_new); ss().cur = len(ss().scen) - 1
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
            except ValueError as e:
                st.error(str(e))
