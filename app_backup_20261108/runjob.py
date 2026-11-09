"""백그라운드 계산 작업 — Streamlit 화면이 다시 그려져도(버튼·탭 조작) 계산이 끊기지 않게 별도 스레드에서 돌린다.

Streamlit은 위젯을 건드릴 때마다 app.py 전체를 처음부터 다시 실행하고, 그때 실행 중이던 계산도 끊는다.
그래서 실행 버튼 안에서 몇 분짜리 계산을 돌리면 그 사이 무엇이든 누르면 비교군 계산이 조용히 사라졌다.
여기서는 계산을 세션별 작업으로 등록해 스레드에서 돌리고, 화면은 진행률만 읽어 간다.
이 모듈은 한 번만 import되므로(JOBS는 화면 재실행 사이에도 유지) 작업 상태를 보관할 수 있다.
"""
from __future__ import annotations

import threading
import time
import traceback

JOBS: dict = {}
_LOCK = threading.Lock()


def get(sid):
    with _LOCK:
        return JOBS.get(sid)


def pop(sid):
    with _LOCK:
        return JOBS.pop(sid, None)


def running(sid):
    j = get(sid)
    return bool(j) and not j["done"]


def start(sid, fn, *args, label="계산", **kw):
    """fn(*args, progress=cb, stop=cb, **kw)를 스레드에서 실행. 같은 세션에 진행 중인 작업이 있으면 시작하지 않는다."""
    with _LOCK:
        j = JOBS.get(sid)
        if j and not j["done"]:
            return j
        j = {"label": label, "frac": 0.0, "msg": "시작", "t0": time.time(), "t1": None, "done": False,
             "stop": False, "result": None, "error": None, "error_type": None, "trace": ""}
        JOBS[sid] = j

    def prog(f, msg):
        j["frac"] = max(0.0, min(1.0, float(f)))
        j["msg"] = str(msg)

    def stop():
        return bool(j["stop"])

    def body():
        try:
            j["result"] = fn(*args, progress=prog, stop=stop, **kw)
        except BaseException as e:      # noqa: BLE001 — 화면에 그대로 알린다
            j["error"] = e
            j["error_type"] = type(e).__name__
            j["trace"] = traceback.format_exc()[-1500:]
        finally:
            j["t1"] = time.time()
            j["done"] = True

    th = threading.Thread(target=body, name=f"wp-{label}", daemon=True)
    j["thread"] = th
    th.start()
    return j


def request_stop(sid):
    j = get(sid)
    if j:
        j["stop"] = True
