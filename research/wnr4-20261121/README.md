# WNR4 research (2026-11-21)

The measurements behind [docs/wnr4-research.md](../../docs/wnr4-research.md) and [docs/WNR4-SPEC.md](../../docs/WNR4-SPEC.md). Codec: `wnr4/wnr4.h` (CPU), `wnr4/wnr4_cuda.cuh` (CUDA), `wnr4/tools/wnr4_tools.c` (ctypes tool API), `wnr4/tools/wnr4_cuda_bridge.cu` (prototype helper DLL), `wnr4/tests/test_wnr4.c` (CPU tests and fuzz).

## Files

| file | what |
|---|---|
| `data.py` | read-only access to the three containers (WEX4 reader), coded domain (canonical zero, WNR3 ceil form), train/test split |
| `stats/wnr4_stats.c`, `stats/run_stats.py` | held-out cross-entropy of weight and scale context models (first study) |
| `stats/ctx3.py` | held-out cross-entropy of further candidates (position, per-layer, scale contexts) |
| `stats/ctx2.py`, `stats/colrow.py`, `stats/colshare.py`, `stats/colprof.py` | in-sample bounds: position, column/row classes, column profiles shared across experts |
| `measure.py` | WNR4 vs WNR3 (same training records) on 2048 held-out records, every record decoded back and compared; `--pack` writes records for the GPU bench |
| `run_measure.sh` | the full ratio runs (results in `results/measure-*.json`) |
| `gpu/bench_gpu.cu` | GPU: bit-exact check of every pack record, single-bit flips, kernel-only GB/s by batch, end-to-end from page-locked host, WNR3 production helper and kernels on the same records |
| `gpu/test_helper.cu` | the helper API: batches from host into device slabs, status per record, corrupted record, unknown model, wrong owner |
| `gpu/bench_cpu.c` | CPU decode speed (reference decoder, OpenMP over tiles) and WNR3's CPU reference |
| `run_gpu_round.sh`, `run_cpu_round.sh`, `run_round8.sh`, `run_verify.sh` | GPU and CPU rounds; LOCKSH (optional) names a shell file with `take_lock LABEL` / `release_lock` for a machine-wide GPU lock |
| `literature.md` | literature and the transfer model |
| `results/` | logs and JSON of every run quoted in the docs |

## Rerun

Paths: set `WNR4_Q38MX`, `WNR4_V41`, `WNR4_Q38NV` (containers) and `WNR4_NV_TREE` (a checkout of the `nvfp4-wnr3` branch, for its NVFP4-aware `wex4.py` and `c/tools/wex_nv_tools.c`; that branch, commit a96c4ff, is not on the public remote yet, so the NVFP4 runs and the WNR3-NV baseline cannot be reproduced from this repository alone), or put them in `out/paths.json`. Builds (MSYS2 UCRT64 gcc, CUDA 13 with MSVC through `nvcc.cmd`):

```
gcc -O3 -march=haswell -fopenmp -static -shared -o out/wnr4_tools.dll ../../wnr4/tools/wnr4_tools.c
gcc -O3 -march=haswell -static -shared -o out/wex_nr3_tools.dll ../../c/tools/wex_nr3_tools.c -I../../c
gcc -O3 -march=haswell -fopenmp -static -shared -o out/wex_nv_tools.dll $WNR4_NV_TREE/c/tools/wex_nv_tools.c -I$WNR4_NV_TREE/c
gcc -O2 -march=native -shared -o out/wnr4_stats.dll stats/wnr4_stats.c -lm
nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc -o out/bench_gpu.exe gpu/bench_gpu.cu
nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc -o out/test_helper.exe gpu/test_helper.cu
gcc -O2 -fopenmp -static -o out/test_wnr4.exe ../../wnr4/tests/test_wnr4.c && out/test_wnr4.exe 3000
python -I measure.py q38mx results/measure-q38mx-cz.json --cz 1 --pb 5 --G 32,64,128,256
python -I measure.py q38mx out/pack.json --train 256 --test 128 --G 128 --pb 5 --pack out/packs --pack-n 128 --pack-G 128
out/bench_gpu.exe out/packs/q38mx.pack 5
```

Generated data (`out/`: DLLs, packs of real records, binaries) is not tracked.
