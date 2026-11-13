# Phase 2 (GPU, under heavy.lock): task 2 GPU decode of the g0/normal bins with the UNCHANGED engine decoder
# (kaira_wex_cuda.dll + test_decode_batch.exe from lab/v41-gpu-20261107/w1/build, clean src 79aff2b, wex_cuda.cuh
# sha b6a6563f = HEAD 88ddb07), then task 3 ceilbench on sm_120.
LAB='<KAIRA>'/lab/wnr3-5090-20261107
SC='<TMP>'/data
W1='<KAIRA>'/lab/v41-gpu-20261107/w1/build
. "$LAB/lock.sh"
take_lock "wnr3-5090 phase2 GPU (g0 GPU decode + ceilbench sm_120)" || exit 3
cd "$LAB"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,memory.used,utilization.gpu,temperature.gpu --format=csv > phase2_gpu_state.txt
for k in A B; do
  echo "== test_decode_batch $k $(date +%T)"
  "$W1/test_decode_batch.exe" "$W1/kaira_wex_cuda.dll" "$SC/g0bins/records_$k.bin" > "g0/gpu_decode_$k.log" 2>&1; echo "rc=$?" >> "g0/gpu_decode_$k.log"
done
echo "== ceilbench $(date +%T)"
CEIL_ARCH=sm_120 ./venv/Scripts/python.exe -I ceil-gpu/src/ceilbench.py "$SC/enc" ceil-gpu/bench_5090.json "${ROUNDS:-41}" 32,64,128,256 lb,out4 A259,D,CF0,CF5,CFD5,DB2,CFDB2 > ceil-gpu/bench_5090.log 2>&1; echo "rc=$?" >> ceil-gpu/bench_5090.log
nvidia-smi --query-gpu=name,driver_version,clocks.sm,memory.used,utilization.gpu,temperature.gpu --format=csv >> phase2_gpu_state.txt
echo "== done $(date +%T)"
