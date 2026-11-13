# Phase 3 (under heavy.lock): task 2b rule-g encode/decode (CPU), then task 3 ceilbench on sm_120 (GPU).
LAB='<KAIRA>'/lab/wnr3-5090-20261107
SC='<TMP>'/data
. "$LAB/lock.sh"
take_lock "wnr3-5090 phase3 (rule-g encode + ceilbench sm_120)" || exit 3
cd "$LAB"
echo "== t2b rule g $(date +%T)"; python -I g0/t2b_rule_g.py "$LAB/g0" > g0/t2b_rule_g.log 2>&1; echo "rc=$?"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,memory.used,utilization.gpu,temperature.gpu --format=csv > phase3_gpu_state.txt
echo "== ceilbench $(date +%T)"
CEIL_ARCH=sm_120 ./venv/Scripts/python.exe -I ceil-gpu/src/ceilbench.py "$SC/enc" ceil-gpu/bench_5090.json "${ROUNDS:-41}" 32,64,128,256 lb,out4 A259,D,CF0,CF5,CFD5,DB2,CFDB2 > ceil-gpu/bench_5090.log 2>&1; echo "rc=$?"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,memory.used,utilization.gpu,temperature.gpu --format=csv >> phase3_gpu_state.txt
echo "== done $(date +%T)"
