"""SLAD 반응재 배치 방식 비교 시뮬레이터 — Streamlit 앱 시작 파일 (화면 코드는 app_ui.py)

실행: run_app.bat 더블클릭  (또는  python -m streamlit run app.py)

이 파일을 얇게 둔 이유: Streamlit은 화면 스크립트를 '__main__' 모듈로 실행한다. Windows에서 병렬 계산 프로세스(spawn)를
띄우면 자식 프로세스가 '__main__'의 파일을 '__mp_main__'으로 다시 실행하므로, 화면 코드가 여기 있으면 작업 프로세스마다
앱 화면 전체(대조군 미리보기 계산 포함)가 한 번씩 더 돌아 시간·메모리를 낭비했다. 자식에서는 아무것도 하지 않는다.
"""
import os as _os

if __name__ != "__mp_main__":
    _ui = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "app_ui.py")
    with open(_ui, encoding="utf-8") as _fh:
        exec(compile(_fh.read(), _ui, "exec"))
