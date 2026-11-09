"""느린 시험 하나를 돌려 결과 행을 runs/slow_<id>.json에 저장. 사용: WP_SLOW=1 python runs/run_slow_test.py <함수 이름>"""
import os, sys, json, time
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p_ in (APP, os.path.join(APP, "selftest"), os.path.join(APP, "verify")):
    sys.path.insert(0, p_)
os.chdir(APP)
os.environ["WP_SLOW"] = "1"
import selfcheck as SC
t = time.time()
r = getattr(SC, sys.argv[1])()
r["sec"] = round(time.time() - t)
json.dump(r, open(os.path.join(APP, "runs", f"slow_{r['id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False))
