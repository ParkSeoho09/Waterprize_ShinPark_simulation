"""tab7_export — app_ui.py의 탭 7 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
import export_xlsx as X
st.markdown("**프로젝트 저장·불러오기** (입력 전체 + 3MF 원본, zip)")
if st.button("프로젝트 파일 만들기", key="mkproj"):
    ss().projzip = X.project_to_zip(c, ss().scen)
if ss().get("projzip"):
    st.download_button("💾 프로젝트 저장(.zip)", ss().projzip, file_name="slad_project.zip", mime="application/zip")
upj = st.file_uploader("프로젝트 불러오기(.zip 또는 이전 판 .json)", type=["zip", "json"], key="projup")
if upj is not None and st.button("불러오기 적용"):
    try:
        if upj.name.lower().endswith(".zip"):
            cc, scs = X.project_from_zip(upj.getvalue(), new_scenario, load_file_into)
        else:
            cc, scs = X.project_from_json(upj.getvalue().decode("utf-8"), new_scenario, load_file_into)
        ss().common = cc
        ss().scen = scs
        ss().cur = 0
        st.success("불러왔습니다. 같은 입력과 같은 코드(해시)로 다시 실행하면 같은 결과가 나옵니다.")
        st.rerun()
    except Exception as e:
        st.error(f"불러오기 실패: {e}")
R = ss().results
if R is None:
    st.info("결과 내보내기는 ③ 실행 후에 쓸 수 있습니다.")
else:
    C_ = Common.from_dict(R["common"])
    st.markdown("**결과 내보내기**")
    unrel = bool(R.get("failed"))
    allow = True
    if unrel:
        st.error("수렴 실패(FAILED) 결과가 있어 Excel·CSV 내보내기를 기본 차단했습니다: " + ", ".join(R["failed"]))
        allow = st.checkbox("그래도 내보내기(모든 파일에 '비신뢰 결과' 표시)", value=False, key="allow_unrel")
    extras = {}
    if ss().get("uq"):
        extras["uq"] = ss().uq
    if ss().get("calib"):
        extras["calib"] = ss().calib
    if ss().get("indep"):
        extras["indep"] = ss().indep
    else:
        import glob
        fs_ = sorted(glob.glob(os.path.join(HERE, "selftest", "independence_*.json")))
        if fs_:
            try:
                with open(fs_[-1], encoding="utf-8") as fh:
                    extras["indep"] = json.load(fh)
            except Exception:      # noqa: BLE001
                pass
    sp_ = os.path.join(HERE, "selftest", "selfcheck_results.json")
    if os.path.exists(sp_):
        with open(sp_, encoding="utf-8") as fh:
            extras["selfcheck"] = json.load(fh)["table"]
    kp_ = os.path.join(HERE, "selftest", "known_issues_table.json")
    if os.path.exists(kp_):
        with open(kp_, encoding="utf-8") as fh:
            extras["known"] = json.load(fh)
    if ss().get("engine_cmp"):
        extras["engine_cmp"] = ss().engine_cmp
    e1, e2, e3 = st.columns(3)
    with e1:
        if allow and st.button("Excel 만들기(.xlsx)"):
            ss().xlsx = X.build_xlsx(R, C_, ss().scen, unreliable=unrel, extras=extras)
        if allow and ss().get("xlsx"):
            st.download_button("📊 Excel 내려받기", ss().xlsx, file_name="slad_결과.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if allow:
            head, rows = M.summary_table(R["metrics"])
            csv = pd.DataFrame(rows, columns=head).to_csv(index=False)
            if unrel:
                csv = "# 비신뢰 결과(수렴 실패 포함)\n" + csv
            st.download_button("지표표 CSV", csv.encode("utf-8-sig"), file_name="slad_지표.csv", mime="text/csv")
    with e2:
        if allow:
            st.download_button("결과 저장(.npz)", X.results_npz(R), file_name="slad_결과.npz")
            st.download_button("결과 요약(.json)", X.results_json(R, unreliable=unrel), file_name="slad_결과.json",
                               mime="application/json")
            if st.button("그림 PNG 묶음 만들기(.zip)"):
                ss().pngzip, fw = X.png_zip(R, C_, unreliable=unrel)
                if fw:
                    st.warning(fw)
            if ss().get("pngzip"):
                st.download_button("🖼 PNG 묶음", ss().pngzip, file_name="slad_그림.zip")
    with e3:
        if st.button("애니메이션 HTML 만들기"):
            all_res = R["scen"] + [R["control"]]
            objs = {}
            for r in R["scen"]:
                s_ = next((x for x in ss().scen if x["name"] == r["name"]), None)
                if s_:
                    so = scenario_obj(s_, C_)
                    objs[r["name"]] = (so.objects, so.roles, Placement(**r["placement"]))
            fig = viz.compare_view(all_res, C_, ["반응재 분포", "NO₃ 오염 플룸"], light=True, scen_objects=objs)
            ss().html_cmp = viz.to_html(fig, sync_cameras=True).encode("utf-8")
            parts = []
            for r in R["scen"]:
                if r["name"] in objs:
                    o, rl, pl = objs[r["name"]]
                    parts.append(viz.install_animation(r, o, rl, pl, light=True))
            ss().html_inst = [viz.to_html(f).encode("utf-8") for f in parts]
        if ss().get("html_cmp"):
            st.download_button("🎬 다중 비교 애니메이션.html", ss().html_cmp, file_name="slad_비교_애니메이션.html", mime="text/html")
        for i, h in enumerate(ss().get("html_inst", []) or []):
            st.download_button(f"🎬 설치 과정 {i + 1}.html", h, file_name=f"slad_설치과정_{i + 1}.html", mime="text/html",
                               key=f"hi{i}")
