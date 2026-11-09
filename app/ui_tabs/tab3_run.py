"""tab3_run — app_ui.py의 탭 3 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
st.markdown("실행하면 **기본 대조군(반응재를 꽂지 않은 같은 흙수조)** 과 ①의 **비교군 전부**를 같은 조건·같은 격자로 계산합니다.")
errs, warns = validate_common(c)
for e in errs:
    st.error(e)
for w in warns:
    st.warning(w)
plan = []
for s_ in ss().scen:
    roles_of = {r["객체"]: r["역할"] for r in s_["role_df"].to_dict("records")} if len(s_["role_df"]) else {}
    e2, w2 = ie.validate(s_["events_df"].to_dict("records"), roles_of)
    status = "준비됨" if (s_["objects"] and not e2) else ("3MF 없음" if not s_["objects"] else "설치 방법을 입력하세요")
    plan.append({"비교군": s_["name"], "파일": s_["file_name"], "객체 수": len(s_["objects"]),
                 "설치 단계": len(ie.clean_rows(s_["events_df"].to_dict("records"))), "상태": status})
if plan:
    st.dataframe(pd.DataFrame(plan), hide_index=True, width="stretch")
else:
    st.warning("비교군이 없습니다. 지금 실행하면 대조군만 계산합니다. ① 탭에서 '예시 불러오기'로 예시를 고르고 **예시 추가**를 누르거나 3MF를 올리세요.")
st.caption(f"흐름: {c.flow_mode} ({'Richards 정상해를 ' + str(c.flow_update_days) + '일마다·투수성 5% 변화 시 재계산 — 준정상 근사' if c.flow_mode == '준정상' else '매 단계 과도 Richards'}) · "
           f"석회석: {c.calcite_mode} · 격자 {c.nx}×{c.ny}×{c.nz}")
is_running = runjob.running(ss().sid)
st.caption(f"정식 실행 조건: {c.days:g}일 · 격자 {c.nx}×{c.ny}×{c.nz} · 흐름 {c.flow_mode} · 화학 {c.chem_engine}. "
           "기본값(365일·보통 격자·PHREEQC)은 컴퓨터에 따라 수십 분~1시간 넘게 걸립니다. "
           f"먼저 결과 모양을 보려면 **빠른 확인 실행**(가벼움 격자 · 최대 {QUICK_DAYS:g}일 · 내장 화학, 보통 몇 분)을 쓰세요.")
r1, r2, r3 = st.columns([2, 1.4, 0.7])
go_run = r1.button("▶ 실행 (대조군 + 비교군 전체)", type="primary", width="stretch",
                   disabled=bool(errs) or is_running, key="go_run")
qd_opts = [90.0, 180.0]
ss().quick_days = float(r2.selectbox("빠른 확인 기간", qd_opts, index=qd_opts.index(ss().get("quick_days", QUICK_DAYS))
                                     if ss().get("quick_days", QUICK_DAYS) in qd_opts else 0,
                                     format_func=lambda x: f"{x:.0f}일", key="qdays",
                                     help="180일이면 결과에서 90일·180일 지표를 나란히 비교합니다(② 운영 기간보다 길게는 안 됨)"))
go_quick = r2.button(f"⚡ 빠른 확인 실행 ({ss().quick_days:.0f}일·가벼움)", width="stretch",
                     disabled=bool(errs) or is_running, key="go_quick")
ncpu = os.cpu_count() or 1
n_tasks = 1 + sum(p_["상태"] == "준비됨" for p_ in plan)
n_cells = int(c.nx * c.ny * c.nz)
avail_mb, total_mb = studies.mem_available_mb()
# 기본값: 대조군·비교군을 동시에 계산(코어 하나는 화면용으로 남김, 최대 4) — 단, 남은 메모리 안에서만.
cores = int(r3.number_input("병렬 코어 수", value=studies.safe_cores(n_tasks, n_cells), min_value=1,
                            max_value=max(1, ncpu), key="cores", step=1,
                            help=f"이 컴퓨터 {ncpu}코어"
                                 + (f", 지금 남은 메모리 약 {avail_mb / 1024:.1f} GB" if avail_mb else "")
                                 + f". 계산 하나가 약 {studies.run_mem_mb(n_cells):.0f} MB를 씁니다(현재 격자). "
                                   "2 이상이면 대조군·비교군을 동시에 계산해 더 빨리 끝남. "
                                   "1이면 하나씩 차례로 계산하고 메모리를 가장 적게 씁니다."))
if avail_mb and avail_mb < 1.5 * studies.run_mem_mb(n_cells):
    st.warning(f"남은 메모리가 약 {avail_mb:.0f} MB로 적습니다(계산 하나에 약 {studies.run_mem_mb(n_cells):.0f} MB). "
               "다른 프로그램·브라우저 탭을 닫거나, ②에서 격자를 '가벼움'으로 바꾸세요.")
if (go_run or go_quick) and start_run(c, cores, quick=bool(go_quick and not go_run)):
    meta = ss().run_meta
    if meta["cores_run"] < cores:
        st.info(f"남은 메모리에 맞춰 병렬 코어를 {cores} → {meta['cores_run']}개로 줄여 계산합니다.")
    ss().run_msgs = []
    st.rerun()
if runjob.running(ss().sid):
    st.info("계산 중입니다. 진행률은 화면 맨 위에 표시됩니다.")
_show_msgs()
if ss().results:
    R = ss().results
    st.caption(f"마지막 실행: 대조군 + {len(R['scen'])}개 비교군 · 대조군 돌파일(η 평가 시작) "
               f"{R['eval_start'] if R['eval_start'] is not None else '없음(모사 기간 안에 도달 안 함)'}일")
