# WNR4 CUDA decoders, rerun with tracked logs (2026-11-27)

**Question.** The GPU figures published on 2026-11-26 and 2026-11-27 for the v2 kernels of WNR4 kind I4 (347.2 GB/s at G = 32, 0.027 ms for one record at G = 4) and of kinds MX and NV (322-366 and 257-264 GB/s) had no log in this repository: the only tracked I4 GPU log was the first kernel's (156.9 GB/s, [wnr4-i4-20261122](../wnr4-i4-20261122/README.md)), and the MX figure rested on the baseline arm of another study ([wnr4-swar-scales-20261127](../wnr4-swar-scales-20261127/README.md)). Do the two GPU tests of this repository, built from `main` and run as their headers say, give those figures?

**Answer.** Yes for the kernels: within about 1% at batch 32 and for one record, except kind I4 at G = 64 (4% lower at batch 32 in two of three runs; one outlier for one record). Not rerun, because nothing in this repository produces them: the end-to-end figures of kind I4 (70.3-72.0 GB/s, 0.122 ms and 0.069 ms for one record) and every engine figure (tokens per second, warm start, miss counts).

## Setup

- **Machine.** Windows 11, Ryzen 9 7950X3D, RTX 5090 (driver 621.40; 170 SMs, 96 MiB of L2), CUDA 13.4 (nvcc 13.4.92) with the MSVC host compiler of Visual Studio 18 Build Tools. The desktop was in use: 1.1 to 1.4 GB of VRAM held by other programs and CPU builds running; no other GPU compute job.
- **Code.** `main` at 0631f75. The commit that adds this folder changes nothing under `wnr4/`.
- **Build**, from the repository root, the commands of the two test headers:

  ```
  nvcc -O3 -std=c++17 -arch=sm_120 wnr4/tests/test_wnr4_i4_gpu.cu -o test_wnr4_i4_gpu
  nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler=/O2,/EHsc wnr4/tests/test_wnr4_gpu.cu -o test_wnr4_gpu_v2
  ```

- **Kind I4.** `test_wnr4_i4_gpu 640x2560,640x2560,2560x640 planar SAMPLE_DIR G`, three times: G = 32, 8 and 4. Each run also measures G = 16 and G = 64 on its first 32 records. `SAMPLE_DIR` holds the 288 sampled records of the Qwen3.8-Flash-Next int4-g64 export of [wnr4-i4-20261122](../wnr4-i4-20261122/README.md), the same files (32 `train_*`, 256 `test_*`; model weights, not in the repository).
- **Kinds MX and NV.** `test_wnr4_gpu_v2` once, without arguments: its correctness matrix, then its benchmark on **synthetic** records, which the test draws from the statistics of the tracked WNM2 fixture it loads (log line `V4.1 ceil WNM2 loaded: yes`), in the shapes of Qwen3.8 MXFP4, Qwen3.8 NVFP4 and DeepSeek V4.1 REAP-192 records. No real MXFP4 or NVFP4 record was decoded.
- **Timing**, as the tests print it. Kernel only: records resident in VRAM, L2 flushed before each launch, median of 20 launches after one warm-up, GB/s of decoded bytes. "Bridge": end to end from page-locked host memory through `wnr4/tools/wnr4_cuda_bridge.cu`, median of 15.
- **Runs.** One run of each command. The three I4 runs repeat G = 16 and G = 64, the only repetition there is.

## Results

**Kind I4, real records, kernel only.** Where three values are given they are those of the runs at G = 32, 8 and 4, in that order.

| G | stored / raw (the 32 records of the bench) | one record | batch 8 | batch 16 | batch 32 | published 2026-11-26 (one record; batch 32) |
|---:|---|---|---|---|---|---|
| 4 | 0.79222 | 0.027 ms (103.8 GB/s) | 0.073 ms (303.0 GB/s) | 0.140 ms (315.5 GB/s) | 0.260 ms, **339.9 GB/s** | 0.027 ms; 341.7 GB/s |
| 8 | 0.78662 | 0.046 ms (60.3 GB/s) | 0.069 ms (321.0 GB/s) | 0.132 ms (334.2 GB/s) | 0.259 ms, **341.1 GB/s** | 0.044 ms; 339.1 GB/s |
| 16 | 0.78384 | 0.077 / 0.079 / 0.079 ms | 0.095 ms (233.0 / 232.7 / 232.6 GB/s) | 0.130 ms (340.3 / 340.7 / 339.8 GB/s) | **345.5 / 346.6 / 344.2 GB/s** | 0.077 ms; 344.6 GB/s |
| 32 | 0.78264 | 0.146 ms (18.9 GB/s) | 0.157 ms (140.9 GB/s) | 0.181 ms (244.3 GB/s) | 0.255 ms, **347.0 GB/s** | 0.147 ms; 347.2 GB/s |
| 64 | 0.78181 | 0.471 / 0.290 / 0.290 ms | 0.304 / 0.310 / 0.310 ms (72.7 / 71.3 / 71.3 GB/s) | 0.306 / 0.314 / 0.314 ms (144.3 / 140.8 / 140.7 GB/s) | **247.6 / 237.8 / 236.6 GB/s** | 0.285 ms; 247.4 GB/s |

In each of the three runs: all 288 records coded and decoded on the CPU, then bit-exact through the helper in synchronous batches of 32 and in asynchronous batches of 16 with 2 in flight; one payload bit flipped in each odd record, 144 of 144 refused and the 144 others bit-exact; the synthetic groups at 4, 8 and 16 warps per block; the refusals of the helper; last line `ALL OK`. Stored / raw of all 288 records: 0.78336 (G = 32), 0.78755 (G = 8), 0.79316 (G = 4).

**Kinds MX and NV, synthetic records, kernel only**, record-level direct launch (the job-list launch of the same batch is within 2% in every row of the log), GB/s:

| pack (64 records; V4.1: 16) | stored / raw | one record | batch 8 | batch 16 | batch 32 | batch 64 | bridge, async 2 in flight, largest batch |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3.8 MX, G = 32 | 0.85176 | 0.085 ms (30.6) | 210.8 | 326.9 | 353.1 | **356.9** | 65.3 |
| Qwen3.8 MX, G = 128 | 0.85005 | 0.316 ms (8.3) | 65.1 | 127.2 | 218.6 | **339.3** | 64.9 |
| Qwen3.8 NV, G = 32 | 0.89187 | 0.140 ms (19.7) | 145.1 | 255.8 | 259.6 | **263.9** | 60.9 |
| Qwen3.8 NV, G = 128 | 0.88479 | 0.517 ms (5.3) | 42.4 | 82.6 | 149.2 | **257.6** | 61.9 |
| V4.1 MX, G = 32 | 0.85089 | 0.105 ms (178.6) | 301.0 | **364.8** (job list 366.6) | - | - | 65.7 |
| V4.1 MX, G = 128 | 0.84880 | 0.329 ms (57.2) | 289.2 | **323.1** | - | - | 65.7 |

Published on 2026-11-27 for these kernels: MX 322-366 GB/s, NV 257-264 GB/s, one record 0.085-0.107 ms at G = 32 and 0.318-0.331 ms at G = 128, bridge 65.6-65.7 GB/s. This run: MX 323.1 to 366.6 GB/s and NV 257.6 to 263.9 GB/s at the largest batch of each pack; one MX record 0.085 and 0.105 ms at G = 32, 0.316 and 0.329 ms at G = 128; bridge 60.9 to 65.7 GB/s. The correctness matrix passed (`=== Correctness Matrix PASSED ===`), and so did the fault isolation of every pack.

## Limits

- One run per command, on a desktop in use. The one-record time of kind I4 at G = 64 was 0.471 ms in the first run and 0.290 ms in the two others; the cause was not looked for.
- MX and NV were measured on synthetic records only. The figures on real records in this repository are those of the first kernel (188-209 and 140-149 GB/s, [wnr4-20261121](../wnr4-20261121/README.md)).
- Not rerun: the end-to-end path of kind I4 (no bench of it is in the repository; the I4 test checks that path for correctness, not for time), any engine, any whole-model container.
- The I4 model is trained inside the test, on the 32 training records and half of its synthetic records, as on 2026-11-22.

## Raw files

- `results/i4-G32.log`, `results/i4-G8.log`, `results/i4-G4.log`: the output of the three I4 runs.
- `results/mxnv.log`: the output of the MX and NV test.

Each log starts with one `nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv` and ends with the exit code of the test.
