"""tab5_docs — app_ui.py의 탭 5 화면(동작 그대로 분리, Part 7). app_ui의 전역 이름 공간에서 실행된다."""
# flake8: noqa
st.subheader("평가지표 (지표 문서 정의)")
st.caption("지표 이름을 누르면 자세한 설명이 펼쳐집니다.")
for name, d in TX.METRICS.items():
    with st.expander(f"**{name}** — {d['one']}"):
        st.markdown(f"**무엇을 재나요?** {d['what']}")
        st.markdown(f"**읽는 법** {d['read']}")
        st.markdown(f"**좋은 값의 기준** {d['good']}")
        st.markdown(f"**이 시뮬레이션에서 계산하는 방법** {d['how']}")
        if st.toggle("계산 방법 보기", key="f_" + name):
            st.code(d["formula"], language=None)
st.subheader("메커니즘 설명")
for name, d in TX.MECHANISMS.items():
    with st.expander(f"**{name}** — {d['one']}"):
        st.markdown(d["detail"])
        if d["formula"] and st.toggle("계산 방법 보기", key="m_" + name):
            st.code(d["formula"], language=None)
st.caption("모델의 가정·한계는 README.md와 limitations.md, 엑셀 내보내기의 '한계·가정' 시트에 정리되어 있습니다.")
