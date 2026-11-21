# WNR4 research (2026-11-21)

The measurements behind [docs/wnr4-research.md](../../docs/wnr4-research.md) and [docs/WNR4-SPEC.md](../../docs/WNR4-SPEC.md). Codec: `wnr4/wnr4.h` (CPU), `wnr4/wnr4_cuda.cuh` (CUDA), `wnr4/tools/wnr4_tools.c` (ctypes tool API), `wnr4/tools/wnr4_cuda_bridge.cu` (prototype helper DLL), `wnr4/tests/test_wnr4.c` (CPU tests and fuzz).

## Files

| file | what |
|---|---|
| `data.py` | read-only access to the three containers (WEX4 reader), coded domain (canonical zero, WNR3 ceil form), train/test split |
| `stats/wnr4_stats.c`, `stats/run_stats.py` | held-out cross-entropy of weight and scale context models (first study) |
| `stats/ctx3.py` | held-out cross-entropy of further candidates (position, per-layer, scale contexts); prints only, its runs are copied in `results/ctx3-console.txt` |
| `stats/ctx2.py`, `stats/colrow.py`, `stats/colshare.py`, `stats/colprof.py` | in-sample bounds: position, column/row classes, column profiles shared across experts |
| `measure.py` | WNR4 vs WNR3 (same training records) on 2048 held-out records, every record decoded back and compared; `--pack` writes records for the GPU bench |
| `run_measure.sh` | the full ratio runs (results in `results/measure-*.json`) |
| `gpu/bench_gpu.cu` | GPU: bit-exact check of every pack record, single-bit flips, kernel-only GB/s by batch, end-to-end from page-locked host, WNR3 production helper and kernels on the same records |
| `gpu/test_helper.cu` | the helper API: batches from host into device slabs, status per record, corrupted record, unknown model, wrong owner |
| `gpu/bench_cpu.c` | CPU decode speed (reference decoder, OpenMP over tiles) and WNR3's CPU reference |
| `run_gpu_round.sh`, `run_cpu_round.sh`, `run_round8.sh`, `run_verify.sh` | GPU and CPU rounds; LOCKSH (optional) names a shell file with `take_lock LABEL` / `release_lock` for a machine-wide GPU lock |
| `literature.md` | literature and the transfer model |
| `results/` | logs and JSON of the runs quoted in the docs (the WNR3 columns of the 64-expert sample table come from a study that is not in this repository) |

## Rerun

Paths: set `WNR4_Q38MX`, `WNR4_V41`, `WNR4_Q38NV` (containers) and `WNR4_NV_TREE` (a checkout of the `nvfp4-wnr3` branch, for its NVFP4-aware `wex4.py` and `c/tools/wex_nv_tools.c`; that branch, commit a96c4ff, is not on the public remote yet, so the NVFP4 runs and the WNR3-NV baseline cannot be reproduced from this repository alone), or put them in `out/paths.json`. Builds (MSYS2 UCRT64 gcc, CUDA 13 with MSVC through `nvcc.cmd`):

```
gcc -O3 -march=haswell -fopenmp -static -shared -o out/wnr4_tools.dll ../../wnr4/tools/wnr4_tools.c
gcc -O3 -march=haswell -static -shared -o out/wex_nr3_tools.dll ../../c/tools/wex_nr3_tools.c -I../../c
gcc -O3 -march=haswell -fopenmp -static -shared -o out/wex_nv_tools.dll $WNR4_NV_TREE/c/tools/wex_nv_tools.c -I$WNR4_NV_TREE/c
gcc -O2 -march=native -shared -o out/wnr4_stats.dll stats/wnr4_stats.c -lm
nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc -o out/bench_gpu9.exe gpu/bench_gpu.cu
nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc -o out/test_helper9.exe gpu/test_helper.cu
gcc -O2 -march=haswell -fopenmp -static -o out/bench_cpu.exe gpu/bench_cpu.c
gcc -O2 -fopenmp -static -o out/test_wnr4.exe ../../wnr4/tests/test_wnr4.c && out/test_wnr4.exe 3000
python -I measure.py q38mx results/measure-q38mx-cz.json --cz 1 --pb 5 --G 32,64,128,256
# the three packs of the GPU and CPU rounds (out/packs3; ratios in results/pack3-*.json)
python -I measure.py q38mx out/pack3-q38mx.json --train 256 --test 128 --G 128 --pb 5 --cz 1 --pack out/packs3 --pack-n 128 --pack-G 128
python -I measure.py q38nv out/pack3-q38nv.json --train 256 --test 128 --G 128 --pb 5 --cz 0 --pack out/packs3 --pack-n 128 --pack-G 128
python -I measure.py v41 out/pack3-v41.json --train 256 --test 32 --G 128 --pb 5 --cz 1 --pack out/packs3 --pack-n 32 --pack-G 128
bash run_verify.sh        # bench9-* and helper9-* logs
bash run_cpu_round.sh     # test_wnr4.log and cpu-* logs
```

The round scripts call numbered binaries, one number per round of the work: `run_verify.sh` the final source as built above (`bench_gpu9.exe`, `test_helper9.exe`); `run_gpu_round.sh` (the bench8-*, helper8-* and ncu8-* logs) `bench_gpu8.exe` and `test_helper8.exe`, built from the source before the review fixes (those added checks, one of them in the kernel, and changed the helper and the tests; the bench9 logs measure the kernel again after them). To rerun that round on the current source, build it under those names. The earlier bench3/bench6 and helper/helper6 logs came from earlier versions of the kernel and are kept as history only.

Generated data (`out/`: DLLs, packs of real records, binaries) is not tracked.
