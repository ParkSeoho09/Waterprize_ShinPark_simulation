cd /tmp/claude-0/v3/app
python3 runs/x11_baseline.py runs/slad_v4.3_after.json 90 > runs/x11_v43.log 2>&1
python3 runs/x11_baseline.py runs/slad_v4.3_after_d10um.json 90 1e-5 > runs/x11_v43_d10.log 2>&1
echo done > runs/queue2_done.txt
