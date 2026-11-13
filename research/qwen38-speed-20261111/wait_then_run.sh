#!/usr/bin/env bash
# Wait (at most WAIT_MIN minutes, default 12) for a free machine, then run q38_ab_temp.sh once.
# Free = no Unity shader compile, VRAM in use <= 1500 MiB, available RAM >= 15.6 GB (so both arms get --ram 13),
# commit headroom >= 51 GB (the harness's own gate), CPU load <= 20 %: twice in a row, 20 s apart.
set -u
here="$(dirname "$0")"
deadline=$((SECONDS + ${WAIT_MIN:-12} * 60)); ok=0
while [ $SECONDS -lt $deadline ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
  read -r avail head cpu <<<"$(powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; \$c=(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average; '{0} {1} {2}' -f [int]\$m.AvailableMBytes, [int]((\$m.CommitLimit-\$m.CommittedBytes)/1MB), [int]\$c" | tr -d '\r')"
  unity=$(tasklist | grep -c -i "UnityShaderCompiler")
  # commit headroom: the harness only warns below its job cap (50.7e9 B = 48,352 MiB); kept here as the floor
  if [ "$unity" = 0 ] && [ "${used:-99999}" -le 1600 ] && [ "${avail:-0}" -ge 15600 ] && [ "${head:-0}" -ge 48400 ] && [ "${cpu:-100}" -le 20 ]; then ok=$((ok + 1)); else ok=0; fi
  echo "$(date +%T) vram=${used}MiB avail=${avail}MB commit_headroom=${head}MB cpu=${cpu}% unity_compilers=$unity ok=$ok"
  [ $ok -ge 2 ] && break
  sleep 20
done
[ $ok -ge 2 ] || { echo "NOT STARTED: the machine was not free within ${WAIT_MIN:-12} minutes"; exit 3; }
bash "$here/q38_ab_temp.sh"; rc=$?
echo "driver rc=$rc"
C:/Python314/python.exe -I "$here/q38_ab_compare.py"
exit $rc
