# WNR4 MXFP4 Prototypes: Per-Record Anchor-Relative Scales (`s_rec16`) and 63-Context Carry-Free 64-Bit SWAR 8-Magnitude Codec (`WNR4-SWAR-8MAG`) (2026-11-27)

## Verdict

- **Prototype 3a (`s_rec16` per-record 16-symbol anchor-relative UE8M0 scale table): ADOPT.**
  Eliminates the anchor-shifted scale blowup on DeepSeek V4.1 `L39 E120`-class outliers (**-5.36% of raw**, **-1.01 MB/record**, cutting scale payload **8.1×** from `1,149,632 B` to `142,108 B` with a `24 B` per-record header), improves typical experts by **-0.012% to -0.020% of raw** (`20 B` header, zero floor-1 waste on absent scales), increases `V4WexCrc` fixed-slot RAM cache capacity by **+6.32%** (**+64 slots** on 16 GiB DDR5, **+128 slots** on 32 GiB DDR5), stays 100% lossless across all 256 byte values (`0..255`) via an 8-bit uniform rANS escape (`f = 64`), and fits Phase 1 in `2 × uint4` (`32 B/matrix`) of GPU registers with zero scale SMEM and zero change to `WNR4-MX v2`'s fast Phase-2 loop.
- **Prototype 3b (`WNR4-SWAR-8MAG`, 63-ctx `1,008 B` SMEM + carry-free 64-bit SWAR + warp-coalesced aligned backward 32-bit raw sign stream + `MAGT` 480-bit tail + `s_rec16` + 4-slot `cp.async` ring): NOT ADOPTED FOR GPU.**
  100% bit-exact CPU == GPU across all `W ∈ {1, 2, 4, 8, 16}`, **0 stack bytes / 0 register spills / 64 regs** on `sm_120`, **66.7% occupancy** (`32/48` warps/SM at `W = 2..16`, `24/48` at `W = 1`), and compression within **+0.012% to +0.022%** of `WNR4-MX pb=5` on typical V4.1 (and **-5.36%** on `L39 E120` via `s_rec16`). However, on RTX 5090 (`sm_120`, L2 flushed, median of 20), `WNR4-SWAR-8MAG` peaks at **230.1 GB/s** (V4.1 `G=32`, `b=16`) and **216.4 GB/s** (Q38 `G=32`, `b=64`), losing **1.49–1.82× in throughput** against `WNR4-MX v2 pb=5` (**320.5–365.7 GB/s**, and **342.9–401.1 GB/s** at `pb=1`) and **1.62–1.90× in 1-record latency** (`0.079–0.331 ms` for `WNR4-MX v2` vs `0.147–0.569 ms` for `WNR4-SWAR-8MAG`). Even after double-buffering the sign stream in warp registers (`sw0`/`sw1` + 2-cycle `__shfl_sync`) and packing the 64-bit SWAR sign masks into a single `__popc`, splitting magnitudes from raw signs imposes a second per-nibble `VOTE.ANY` + `2× POPC` collective, a ~85%-taken `if (sbits < cnt)` `SHFL.IDX` + `BSYNC.RECONVERGENT` + 64-bit `SHF.R.U64` path, and 64-bit SWAR arithmetic versus `WNR4-MX v2`'s single-stream 32-bit 4-way pre-swapped lookup (which also achieves `0` spills and `66.7%` occupancy at `W=16` on `sm_120`).

---

## Question

1. **Item 3a (`s_rec16`):** How much worst-case record size and fixed-slot `V4WexCrc` RAM-cache waste does a per-record 16-symbol anchor-relative UE8M0 scale table (quantized over present symbols only, plus lossless escape) remove on DeepSeek V4.1 `L39 E120`-class anchor-shifted experts and on typical MXFP4 experts?
2. **Item 3b (`WNR4-SWAR-8MAG`):** On real RTX 5090 (`sm_120`) hardware, does splitting MXFP4 weights into 8 magnitudes (63 contexts = `1,008 B` SMEM/matrix, carry-free 64-bit SWAR search) + warp-coalesced aligned 32-bit raw sign stream + `MAGT` 480-bit tail beat or lose to `WNR4-MX v2`'s 16-symbol pre-swapped 4-way lookup in GPU throughput, 1-record latency, occupancy, and compression ratio?

---

## Setup

- **Hardware:** Windows 11 desktop, NVIDIA GeForce RTX 5090 (`sm_120`, 170 SMs, 96 MiB L2), x86-64 AVX2 CPU.
- **Compilers:**
  - CPU tests: MSYS2 UCRT64 `gcc -O3 -march=haswell -fopenmp -Wall -Wextra` (0 warnings).
  - GPU benchmark: CUDA `nvcc` via `research/wnr4-20261121/nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler=/O2,/EHsc,/W3 --ptxas-options=-v` (0 warnings in all prototype files, 0 stack frame, 0 spills across all `W ∈ {1, 2, 4, 8, 16}`).
- **Baseline:** Current `wnr4/wnr4.h` + `wnr4/wnr4_cuda.cuh` (`WNR4-MX v2`, `pb=1` and `pb=5`, pre-swapped 16-symbol 4-way search, 4-slot `cp.async` SMEM ring, `W ∈ {1, 2, 4, 8, 16}`).
- **Datasets:** Sampled from tracked `research/wnr3-5090-20261107/ceil-kaira/model_CFG_rule_g.wnm2` (authentic ~28% ceil-form blocks + canonical zero):
  1. **DeepSeek V4.1 typical MXFP4:** `scount={368640,368640,368640}`, `scols={160,72,160}`, `raw = 18,800,640 B`, coded anchors `(120, 120, 120)`, `G ∈ {32, 64, 128}`, `n = 16` records (`train = 4`).
  2. **DeepSeek V4.1 `L39 E120` anchor-shifted outlier:** exact V4.1 geometry (`raw = 18,800,640 B`), encoded with the typical V4.1 trained model; coded anchors `(121, 122, 121)` (`w1`/`w3` shifted +1; `w2` coded histogram ~65% `123`, ~27% `122`, ~4% `121`, ~4% `124`), `G ∈ {32, 64, 128}`, `n = 8` records.
  3. **All-256 escape stress:** V4.1 geometry with all 256 byte values `0..255` present in every matrix (`n = 2`, verified bit-exact on GPU).
  4. **Qwen3.8 MXFP4:** `scount={51200,51200,51200}`, `scols={80,20,80}`, `raw = 2,611,200 B`, `G ∈ {32, 128}`, `n = 64` records (`train = 8`).
- **Timing method:** Kernel-only CUDA events around each launch; before EVERY timed iteration, `cudaMemsetAsync` flushes `2 × L2 + 1 MiB = 193 MiB`; 1 warmup + 20 timed iterations, median reported; full sweep `W ∈ {1, 2, 4, 8, 16}` for both `WNR4-MX v2` (`pb=5` & `pb=1`) and `WNR4-SWAR-8MAG`. One run of the benchmark binary (`bench_gpu.log`).

### Commands

```bash
# CPU tests (unit SWAR, s_rec16 edge cases, round trips G=1..128, exhaustive sign-stream bit flips, 8,500 fuzz trials)
gcc -O3 -march=haswell -fopenmp -Wall -Wextra \
    research/wnr4-swar-scales-20261127/test_wnr4_swar.c -o out/test_wnr4_swar.exe
out/test_wnr4_swar.exe 2000

# GPU build, verify, and L2-flushed head-to-head benchmark
research/wnr4-20261121/nvcc.cmd -O3 -std=c++17 -arch=sm_120 -Xcompiler=/O2,/EHsc,/W3 --ptxas-options=-v \
    research/wnr4-swar-scales-20261127/bench_wnr4_swar_gpu.cu -o out/bench_wnr4_swar_gpu.exe
out/bench_wnr4_swar_gpu.exe
```

---

## Results

### 1. Compression Ratios & Scale Payload (4096-B aligned records, 100% CPU & GPU bit-exact)

| Suite (`raw`) | `G` | `WNR4-MX pb=1` | `WNR4-MX pb=5` (scale B) | `3a: +s_rec16 pb=1` | `3a: +s_rec16 pb=5` (scale B / hdr) | `3b: SWAR-8MAG` (Δ vs `pb=5` / vs `3a pb=5`) |
|---|---:|---:|---:|---:|---:|---:|
| **V4.1 typical** (`18,800,640 B`, `n=16`) | 32 | `0.85133` (`16,005,632 B`) | `0.85087` (`15,996,928 B`, `126,729 B`) | `0.85120` (`-0.0136%`) | **`0.85075`** (`-0.0123%`, `123,653 B` / `20 B`) | `0.85099` (`+0.0123%` / `+0.0245%`) |
| **V4.1 typical** (`18,800,640 B`, `n=16`) | 64 | `0.84993` (`15,979,264 B`) | `0.84946` (`15,970,304 B`, `126,712 B`) | `0.84977` (`-0.0163%`) | **`0.84929`** (`-0.0163%`, `123,679 B` / `20 B`) | `0.84966` (`+0.0204%` / `+0.0368%`) |
| **V4.1 typical** (`18,800,640 B`, `n=16`) | 128 | `0.84924` (`15,966,208 B`) | `0.84880` (`15,958,016 B`, `126,733 B`) | `0.84909` (`-0.0150%`) | **`0.84860`** (`-0.0204%`, `123,674 B` / `20 B`) | `0.84902` (`+0.0218%` / `+0.0422%`) |
| **V4.1 `L39 E120` shifted** (`18,800,640 B`, `n=8`) | 32 | `0.90496` (`17,013,760 B`) | `0.90452` (`17,005,568 B`, `1,149,898 B`) | `0.85125` (`-5.3704%`) | **`0.85082`** (`-5.3704%`, `142,062 B` / `24 B`) | `0.85082` (`-5.3704%` / `+0.0000%`) |
| **V4.1 `L39 E120` shifted** (`18,800,640 B`, `n=8`) | 64 | `0.90351` (`16,986,624 B`) | `0.90308` (`16,978,432 B`, `1,149,850 B`) | `0.84986` (`-5.3649%`) | **`0.84943`** (`-5.3649%`, `142,084 B` / `24 B`) | `0.84954` (`-5.3540%` / `+0.0109%`) |
| **V4.1 `L39 E120` shifted** (`18,800,640 B`, `n=8`) | 128 | `0.90283` (`16,973,824 B`) | `0.90240` (`16,965,632 B`, `1,149,632 B`) | `0.84924` (`-5.3595%`) | **`0.84880`** (`-5.3595%`, `142,108 B` / `24 B`) | `0.84883` (`-5.3568%` / `+0.0027%`) |
| **V4.1 escapes `0..255`** (`18,800,640 B`, `n=2`) | 128 | `0.84902` (`15,962,112 B`) | `0.85011` (`15,982,592 B`, `127,843 B`) | `0.84902` (`+0.0000%`) | `0.85011` (`+0.0000%`, `125,577 B` / `96 B`) | `0.84880` (`-0.1307%` / `-0.1307%`) |
| **Qwen3.8 MXFP4** (`2,611,200 B`, `n=64`) | 32 | `0.85176` (`2,224,128 B`) | `0.85176` (`2,224,128 B`, `17,587 B`) | `0.85176` (`+0.0000%`) | `0.85176` (`+0.0000%`, `17,176 B` / `20 B`) | `0.85176` (`+0.0000%` / `+0.0000%`) |
| **Qwen3.8 MXFP4** (`2,611,200 B`, `n=64`) | 128 | `0.85020` (`2,220,032 B`) | `0.85007` (`2,219,712 B`, `17,601 B`) | `0.85020` (`+0.0000%`) | **`0.84956`** (`-0.0515%`, `17,192 B` / `20 B`) | `0.85020` (`+0.0123%` / `+0.0637%`) |

`WNR4-SWAR-8MAG` per-record byte breakdown on V4.1 typical `G=128`: `s_rec16` header `20 B`, scale rANS `123,657 B`, magnitude rANS `11,907,817 B`, aligned backward raw signs `3,892,824 B`, `pad2` `270 B`, `MAGT` saved `129,600 bits` (`16,200 B`; at `G=32`: `518,400 bits` = `64,800 B`). Model blob size: `3,434 B` (`1,008 B` SMEM/matrix) vs `45,866 B` (`pb=1`) / `227,306 B` (`pb=5`).

### 2. Worst-Case Record Size & `V4WexCrc` Fixed-Slot RAM Cache Gain (V4.1 `G=128`, `L39 E120` outlier)

| Codec | Worst-Case Record | Scale Payload (`L39 E120`) | 16 GiB DDR5 Slots (`Δ`) | 32 GiB DDR5 Slots (`Δ`) |
|---|---:|---:|---:|---:|
| `WNR4-MX pb=5` (baseline) | `16,965,632 B` (`16.97 MB`) | `1,149,632 B` (`1.15 MB`) | `1,012` (`+0`, `+0.00%`) | `2,025` (`+0`, `+0.00%`) |
| **3a: `WNR4-MX + s_rec16 pb=5`** | **`15,958,016 B`** (`15.96 MB`) | **`142,108 B`** (`142 KB`) | **`1,076`** (**`+64`**, **`+6.32%`**) | **`2,153`** (**`+128`**, **`+6.32%`**) |
| **3b: `WNR4-SWAR-8MAG`** | `15,962,112 B` (`15.96 MB`) | `142,122 B` (`142 KB`) | `1,076` (`+64`, `+6.32%`) | `2,152` (`+127`, `+6.27%`) |

### 3. `sm_120` Kernel Resource Usage (`--ptxas-options=-v`) & Occupancy

| Kernel | `W` | Regs | Stack / Spills (st/ld) | Static SMEM | Barriers | RTX 5090 Occupancy |
|---|---:|---:|---:|---:|---:|---:|
| `wnr4_swar_cuda_kernel` | 1 | **64** | **0 B / 0 B / 0 B** | `2,048 B` | 0 | `24 blk = 24/48` warps/SM (`50.0%`) |
| `wnr4_swar_cuda_kernel` | 2 | **64** | **0 B / 0 B / 0 B** | `3,072 B` | 1 | `16 blk = 32/48` warps/SM (`66.7%`) |
| `wnr4_swar_cuda_kernel` | 4 | **64** | **0 B / 0 B / 0 B** | `5,120 B` | 1 | `8 blk = 32/48` warps/SM (`66.7%`) |
| `wnr4_swar_cuda_kernel` | 8 | **64** | **0 B / 0 B / 0 B** | `9,216 B` | 1 | `4 blk = 32/48` warps/SM (`66.7%`) |
| `wnr4_swar_cuda_kernel` | 16 | **64** | **0 B / 0 B / 0 B** | `17,408 B` | 1 | `2 blk = 32/48` warps/SM (`66.7%`) |
| `wnr4_cuda_kernel` (`MX v2 pb=1/5`) | 1 | 79–87 | **0 B / 0 B / 0 B** | `15,280 B` | 0 | `6 blk = 6/48` warps/SM (`12.5%`) |
| `wnr4_cuda_kernel` (`MX v2 pb=1/5`) | 4 | 79–87 | **0 B / 0 B / 0 B** | `21,424 B` | 1 | `4 blk = 16/48` warps/SM (`33.3%`) |
| `wnr4_cuda_kernel` (`MX v2 pb=1/5`) | 8 | 79–85 | **0 B / 0 B / 0 B** | `21,424 B` | 1 | `3/2 blk = 24/16` warps/SM (`50.0%`/`33.3%`) |
| `wnr4_cuda_kernel` (`MX v2 pb=1/5`) | 16 | **62–64** | **0 B / 0 B / 0 B** | `29,616 B` | 1 | `2 blk = 32/48` warps/SM (`66.7%`) |

### 4. RTX 5090 (`sm_120`) Kernel-Only Throughput & Latency (96 MiB L2 Flushed, Median of 20, Best `W ∈ {1,2,4,8,16}`)

| Dataset & `G` | Batch `b` (`tiles`) | `WNR4-MX v2 pb=5` | `WNR4-MX v2 pb=1` | `WNR4-SWAR-8MAG` | `SWAR / v2 pb=5` |
|---|---:|---:|---:|---:|---:|
| **V4.1 typical `G=32`** | `1` (`1,080`) | `0.099 ms` (`189.8 GB/s`, `W=4`) | `0.089 ms` (`211.7 GB/s`, `W=4`) | `0.169 ms` (`111.4 GB/s`, `W=1`) | **0.59×** (`1.71×` slower) |
| **V4.1 typical `G=32`** | `4` (`4,320`) | `0.232 ms` (`323.8 GB/s`, `W=4`) | `0.216 ms` (`348.3 GB/s`, `W=4`) | `0.406 ms` (`185.5 GB/s`, `W=1`) | **0.57×** |
| **V4.1 typical `G=32`** | `8` (`8,640`) | `0.453 ms` (`331.7 GB/s`, `W=4`) | `0.417 ms` (`361.1 GB/s`, `W=8`) | `0.687 ms` (`219.0 GB/s`, `W=1`) | **0.66×** |
| **V4.1 typical `G=32`** | `16` (`17,280`) | **`0.823 ms` (`365.7 GB/s`, `W=16`)** | **`0.750 ms` (`401.1 GB/s`, `W=8`)** | `1.307 ms` (`230.1 GB/s`, `W=1`) | **0.63×** (`1.59×` slower) |
| **V4.1 typical `G=64`** | `1` (`540`) | `0.169 ms` (`111.1 GB/s`, `W=4`) | `0.154 ms` (`121.8 GB/s`, `W=4`) | `0.292 ms` (`64.4 GB/s`, `W=1`) | **0.58×** |
| **V4.1 typical `G=64`** | `16` (`8,640`) | `0.916 ms` (`328.3 GB/s`, `W=4`) | `0.831 ms` (`362.1 GB/s`, `W=8`) | `1.364 ms` (`220.5 GB/s`, `W=1`) | **0.67×** (`1.49×` slower) |
| **V4.1 typical `G=128`** | `1` (`270`) | `0.331 ms` (`56.8 GB/s`, `W=4`) | `0.300 ms` (`62.7 GB/s`, `W=2`) | `0.569 ms` (`33.1 GB/s`, `W=1`) | **0.58×** (`1.72×` slower) |
| **V4.1 typical `G=128`** | `4` (`1,080`) | `0.380 ms` (`197.8 GB/s`, `W=4`) | `0.341 ms` (`220.6 GB/s`, `W=4`) | `0.638 ms` (`117.8 GB/s`, `W=1`) | **0.60×** |
| **V4.1 typical `G=128`** | `8` (`2,160`) | `0.498 ms` (`301.8 GB/s`, `W=16`) | `0.460 ms` (`327.3 GB/s`, `W=16`) | `0.890 ms` (`169.1 GB/s`, `W=1`) | **0.56×** |
| **V4.1 typical `G=128`** | `16` (`4,320`) | **`0.939 ms` (`320.5 GB/s`, `W=16`)** | **`0.877 ms` (`342.9 GB/s`, `W=16`)** | `1.613 ms` (`186.5 GB/s`, `W=1`) | **0.58×** (`1.72×` slower) |
| **V4.1 `L39 E120` `G=128`** | `1` (`270`) | `0.339 ms` (`55.5 GB/s`, `W=2`) | `0.300 ms` (`62.6 GB/s`, `W=2`) | `0.571 ms` (`33.0 GB/s`, `W=1`) | **0.59×** |
| **V4.1 `L39 E120` `G=128`** | `8` (`2,160`) | `0.503 ms` (`299.0 GB/s`, `W=16`) | `0.464 ms` (`324.1 GB/s`, `W=16`) | `0.890 ms` (`169.0 GB/s`, `W=1`) | **0.57×** |
| **Qwen3.8 `G=32`** | `1` (`150`) | `0.091 ms` (`28.8 GB/s`, `W=4`) | `0.079 ms` (`33.2 GB/s`, `W=2`) | `0.147 ms` (`17.8 GB/s`, `W=1`) | **0.62×** (`1.62×` slower) |
| **Qwen3.8 `G=32`** | `16` (`2,400`) | `0.126 ms` (`330.9 GB/s`, `W=16`) | `0.118 ms` (`355.4 GB/s`, `W=16`) | `0.230 ms` (`181.5 GB/s`, `W=1`) | **0.55×** (`1.82×` slower) |
| **Qwen3.8 `G=32`** | `64` (`9,600`) | **`0.472 ms` (`354.2 GB/s`, `W=16`)** | **`0.447 ms` (`373.7 GB/s`, `W=16`)** | `0.772 ms` (`216.4 GB/s`, `W=1`) | **0.61×** (`1.64×` slower) |
| **Qwen3.8 `G=128`** | `1` (`39`) | `0.326 ms` (`8.0 GB/s`, `W=1`) | `0.300 ms` (`8.7 GB/s`, `W=1`) | `0.569 ms` (`4.6 GB/s`, `W=1`) | **0.58×** (`1.75×` slower) |
| **Qwen3.8 `G=128`** | `64` (`2,496`) | **`0.499 ms` (`334.9 GB/s`, `W=16`)** | **`0.468 ms` (`356.9 GB/s`, `W=16`)** | `0.897 ms` (`186.3 GB/s`, `W=1`) | **0.56×** (`1.80×` slower) |

*Note on sign-window ablation:* Adding the double-buffered warp register sign window (`sw0`/`sw1` + 2-cycle `__shfl_sync`) and 1-`__popc` carry-free 64-bit SWAR pack cut `WNR4-SWAR-8MAG` 1-record latency by **15–18%** (`0.676 → 0.569 ms` on V4.1 `G=128`, `0.668 → 0.569 ms` on Q38 `G=128`) and improved batch throughput by **+3–6%** (`224.1 → 230.1 GB/s`), confirming that the remaining **1.49–1.82×** gap against zero-spill `WNR4-MX v2 pb=5` (and up to **1.92×** against `pb=1`) is intrinsic to decoding a second raw-sign bitstream per nibble (`2× VOTE.ANY` + `5× POPC` + `~85%`-taken `SHFL.IDX`/`BSYNC` + 64-bit `sbuf` shifts + 64-bit SWAR subtract).

### 5. Verification Summary

- **CPU (`test_wnr4_swar.c`, `test_cpu.log`):**
  - Carry-free 64-bit SWAR `wnr4_swar_find8`: 200 random CDFs × 16,384 slots = **3,276,800 / 3,276,800 exact**.
  - `s_rec16` edge cases (`K=1`, boundaries `0`/`255`, negative masks/pads/underflows): **OK**.
  - Round trips across `G ∈ {1, 3, 7, 32, 64, 128}` on normal ceil, `L39 E120` shifted, all-256 escapes, constant `K=1` + sparse `MAGT`, and tiny sub-warp matrices (`1, 2, 17` cells): **100% bit-exact** for both 3a and 3b.
  - Exhaustive single-bit flips in raw sign streams: **1,024 / 1,024 caught** (`0` missed; used sign bits caught by 32-bit MurmurHash3, unused high pad bits caught by `sbuf != 0`).
  - Random 1–3 bit mutation fuzzing across all used bytes (header, `s_rec16`, offsets, states, rANS words, `pad2`, sign stream): **8,500 / 8,500 refused**, `0` wrong accepts.
- **GPU (`bench_wnr4_swar_gpu.cu`, `bench_gpu.log`):**
  - **100% bit-exact CPU == GPU** across all `W ∈ {1, 2, 4, 8, 16}` on every V4.1 typical, `L39 E120` outlier, all-256 escaped, and Qwen3.8 record.
  - GPU fault isolation & single-bit sign-stream flips: **100% of flipped odd records rejected**, **100% of clean even records decoded bit-exact**.

---

## Limits

- Datasets are sampled from `research/wnr3-5090-20261107/ceil-kaira/model_CFG_rule_g.wnm2` rather than read from full multi-hundred-GB WEX4 containers; real DeepSeek V4.1 `L39 E120` also has minor per-column weight variation beyond the anchor shift.
- Measurements come from one RTX 5090 run (median of 20 L2-flushed kernel iterations per configuration); `WNR4-MX v2` numbers matched a prior idle run within ~1%.
- CPU SIMD decode throughput of `WNR4-SWAR-8MAG` was not benchmarked (see `research/wnr3-improve-20261117/` for AVX2/AVX-512 WNX-MAGT CPU speedups).
- Prototype 3a (`WNR4-MX + s_rec16`) was verified and sized on CPU; its GPU Phase-1 register kernel (`2 × uint4`) was verified inside `wnr4_swar_cuda_kernel`, while standalone `WNR4-MX v2 + s_rec16` GPU timing was not separately benchmarked (Phase 1 is 3.1% of symbols; Phase 2 is identical to `WNR4-MX v2`).

---

## Raw Files

| File | Description |
|---|---|
| `wnr4_swar.h` | Header-only C99 reference trainer, encoder, and decoder for Prototype 3a (`wnr4_srec16_*`) and Prototype 3b (`wnr4_swar_*`) |
| `wnr4_swar_cuda.cuh` | `sm_80+` / `sm_120` CUDA decoder for `WNR4-SWAR-8MAG` (`W ∈ {1, 2, 4, 8, 16}`, 4-slot `cp.async` ring, carry-free 64-bit SWAR, double-buffered warp register sign window, `s_rec16` register Phase 1, `MAGT`, 32-bit MurmurHash3) |
| `test_wnr4_swar.c` | CPU unit, round-trip, exhaustive sign-stream flip, negative, and fuzz test suite |
| `bench_wnr4_swar_gpu.cu` | GPU bit-exact verification, fault isolation, occupancy, ratio/RAM-cache, and L2-flushed head-to-head benchmark vs `WNR4-MX v2` |
| `test_cpu.log` | Raw stdout of `out/test_wnr4_swar.exe 2000` |
| `bench_gpu.log` | Raw stdout of `out/bench_wnr4_swar_gpu.exe` on RTX 5090 |
