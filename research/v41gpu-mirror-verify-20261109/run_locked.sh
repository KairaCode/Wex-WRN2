#!/bin/sh
# one host-style v41gpu launch (WNR2 U: + C: mirror) under heavy.lock (lab/wnr3-5090-20261107/lock.sh protocol)
. '<KAIRA>'/lab/wnr3-5090-20261107/lock.sh
take_lock "v41gpu mirror verify: 1 host-style engine launch, ~10 min (subagent)" || exit 3
cd "$(dirname "$0")"
if tasklist | grep -qi "KAIRA.exe\|deepseek_v4"; then echo "KAIRA or an engine is running: abort"; exit 4; fi
echo "== temps before $(date +%T)"; C:/Python314/python.exe -I ../v41-gpu-20261107/ab/nvme_temp.py 3=C: 5=U:
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
C:/Python314/python.exe -I host_launch.py "$1"
rc=$?
sleep 5
echo "engine processes after stop:"; tasklist | grep -i "deepseek\|colibri" || echo "  none"
echo "== temps after $(date +%T)"; C:/Python314/python.exe -I ../v41-gpu-20261107/ab/nvme_temp.py 3=C: 5=U:
exit $rc
