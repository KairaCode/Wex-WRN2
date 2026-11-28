# WNR4 kind I4 on int4-g64 experts (2026-11-22)

*Note, 2026-11-27. The page below is as it was written on 2026-11-22; three of its statements are no longer current. (1) "Not integrated in any engine", "No engine run": since 2026-11-26 the maintainer's `qwen38` engine reads kind I4 records ([STATUS.md](../../STATUS.md)); its figures are reported there without a log in this repository. (2) 156.9 GB/s is the speed of the first CUDA kernel; the kernel of 2026-11-26 decodes the same 288 records at 347.0 GB/s (G = 32, batch 32): [wnr4-gpu-rerun-20261127](../wnr4-gpu-rerun-20261127/README.md). (3) The answer for Inkling-2-Mini is still no, and a whole-model kind I4 container of that export has been published all the same (`inkling-2-mini/inkling2_mini_int4g64.wnr4i4.wex`, 12,109,963,264 B, in the Hugging Face catalogue of [STATUS.md](../../STATUS.md)); nothing in this repository measures it.*

**Question.** Can WNR4 tiles store int4-g64 experts (signed int4 codes stored as nibble v + 8, one f32 scale per 64 inputs along the row) in fewer bytes than zstd, byte-exact, with a GPU decoder as fast as the other WNR4 kinds'?

**Answer.** Yes for an int4 export of an FP8 block-scaled checkpoint: on 256 held-out records of Qwen3.8-Flash-Next, kind I4 ([docs/WNR4-SPEC.md](../../docs/WNR4-SPEC.md) section 6) takes **0.78333** of raw at G = 32, against 0.8074 for zstd -19, every record byte-exact, and its CUDA decoder reaches 156.9 GB/s of decoded records at batch 32 on an RTX 5090. No for an int4 export of an NVFP4 checkpoint: 90% of the Inkling-2-Mini scales miss the lattice and escape, 0.88139 of raw, worse than zstd -19 (0.8406). Not integrated in any engine.

## Data

- **Qwen3.8-Flash-Next, int4-g64.** A RAW WEX4 container of the routed experts of an int4-g64 export of the FP8 release (24,576 records = 48 layers x 512 experts, 2,764,800 bytes each: gate and up 640 x 2560, down 2560 x 640, planar packing, the codes of the three matrices then their scales). The int4 converter writes each group's scale as amax / 7 of the FP8 values `E4M3 x B` (B the release's scale of the 128 x 128 block), so every scale lies on the lattice of WNR4-SPEC 6.1. Sample: 288 records spread over the 48 layers (expert k of the sample in layer k mod 48, a random expert, seed 20261121), 1 in 9 for training: 32 training and 256 held-out records, read with `wex4.Reader.record` (SHA-256 checked).
- **Inkling-2-Mini, int4-g64.** An int4-g64 conversion of the NVFP4 release (3 matrices: gate and up 1152 x 2304, down 2304 x 1152, planar); the experts were cut from its per-layer safetensors files in the same record layout, with the same sampling over its 24 layers (32 training, 256 held-out). Its scales come from NVFP4 block scales, not from amax / 7 of an FP8 block, and mostly miss the lattice.

The sample files themselves are not in this repository (they are weights of the models), and neither are the two sampling scripts, which read local copies of the checkpoints. `zstd -19` figures: zstandard level 19 on each held-out record (unpadded), from an earlier run of a prototype of this kind on the same records.

## Results

CPU, `bench_i4.c` (this folder): Ryzen 9 7950X3D, Windows 11, MSYS2 UCRT64 gcc 16.2, `-O2 -march=haswell -fopenmp`; one run each; ratio = records padded to 4096 bytes over raw; decode on 1 and on 8 threads, encode on 1 thread. Logs: [results/cpu-qwen38.log](results/cpu-qwen38.log), [results/cpu-inkling.log](results/cpu-inkling.log).

| data | G | stored / raw | zstd -19 | escaped scales per record | decode, 1 thread | decode, 8 threads | encode, 1 thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next, 256 held-out | 16 | 0.78474 | 0.8074 | 0 | 81 MB/s | 406 MB/s | 88 MB/s |
| Qwen3.8-Flash-Next, 256 held-out | 32 | **0.78333** | 0.8074 | 0 | 84 MB/s | 447 MB/s | 93 MB/s |
| Qwen3.8-Flash-Next, 256 held-out | 64 | 0.78270 | 0.8074 | 0 | 87 MB/s | 409 MB/s | 94 MB/s |
| Inkling-2-Mini, 256 held-out | 32 | 0.88139 | 0.8406 | 111,969 of 124,416 | 74 MB/s | 313 MB/s | 31 MB/s |

Every record decoded byte-exact on 1 and on 8 threads, and every record with one payload bit flipped was refused or decoded to other bytes (the bench counts both as noticed; the CPU tests require refusal).

GPU, `wnr4/tests/test_wnr4_i4_gpu.cu`: RTX 5090 (driver 621.24), CUDA 13.4, `nvcc -O3 -std=c++17 -arch=sm_120` with the MSVC host compiler of Visual Studio 18 Build Tools; the 288 Qwen3.8 records (the model trained on the 32 training records and on half of the test's synthetic records), coded at G = 32 and decoded through the helper (synchronous batches of 32, asynchronous batches of 16 with 2 in flight, one payload bit flipped in each odd record: all 144 refused, the others bit-exact), plus two synthetic groups (G 3 with 3% escapes; every scale escaped, G 1) at 4, 8 and 16 warps per block, and the refusals of the helper. Kernel only, records resident in VRAM, L2 (96 MiB) flushed before each run, median of 20, GB/s of decoded records. Log: [results/gpu-test.log](results/gpu-test.log).

| records per batch | G = 16 | G = 32 | G = 64 |
|---:|---|---|---|
| 1 | 0.241 ms, 11.5 GB/s | 0.474 ms, 5.8 GB/s | 1.035 ms, 2.7 GB/s |
| 8 | 0.253 ms, 87.4 GB/s | 0.474 ms, 46.7 GB/s | 1.047 ms, 21.1 GB/s |
| 16 | 0.284 ms, 155.8 GB/s | 0.494 ms, 89.5 GB/s | 1.042 ms, 42.5 GB/s |
| 32 | 0.458 ms, 193.2 GB/s | 0.564 ms, 156.9 GB/s | 1.109 ms, 79.8 GB/s |

Whole model, `wex4/wnr4i4.py` on the RAW container of all 24,576 records (67,995,492,352 bytes), on a second Windows desktop (Ryzen 7 9700X, 16 GB RAM, Python 3.14, the tools DLL built with gcc 16.2 `-O3 -march=haswell -fopenmp -static`): `train --every 8` (3,072 records, 58 s), `pack` at G = 32 (17 min, 64 MB/s; every record decoded back before it was written), `verify --source` (4 min 42 s, 242 MB/s on 4 threads). Result: 53,246,078,976 of 67,947,724,800 bytes of records, **0.7836** of raw; container 53,294,047,232 bytes against 67,995,492,352 RAW, 14.7 GB (21.6%) less; every record decoded identical to the RAW container. Log: [results/full-model/pack-verify.log](results/full-model/pack-verify.log), script [results/full-model/run.cmd](results/full-model/run.cmd). Both containers were deleted after the run.

`wex4/wnr4i4.py` packs at G = 32 by default, as `wex4/wnr4gg.py` does: G = 16 would decode one record twice as fast on the GPU for 0.14 points of ratio.

## Not done

- Sage-S2.1-Flash (pairs packing; 0.7855 in the prototype) was not run with kind I4.
- No engine run: no engine reads kind I4 records.

## Raw files

- `bench_i4.c`: the CPU bench (train, code, decode on 1 and N threads, compare, one bit flipped).
- `results/cpu-qwen38.log`: G 32, 64 and 16 on the Qwen3.8 sample; `results/cpu-inkling.log`: G 32 on the Inkling sample.
- `results/gpu-test.log`: the output of `wnr4/tests/test_wnr4_i4_gpu.cu` on the 288 Qwen3.8 records.
- `results/full-model/run.cmd` and `pack-verify.log`: the whole-model train, pack and verify.

All runs used the code of this repository as committed with this folder (branch `wnr4-int4g64`).

## Rerun

From the repository root, with the sampled records in `SAMPLE_DIR/train_*.rec` and `SAMPLE_DIR/test_*.rec`:

```
gcc -O2 -march=haswell -fopenmp research/wnr4-i4-20261122/bench_i4.c -o bench_i4
./bench_i4 640x2560,640x2560,2560x640 planar 32 8 SAMPLE_DIR/train_*.rec -- SAMPLE_DIR/test_*.rec
nvcc -O3 -std=c++17 -arch=sm_120 wnr4/tests/test_wnr4_i4_gpu.cu -o test_wnr4_i4_gpu
./test_wnr4_i4_gpu 640x2560,640x2560,2560x640 planar SAMPLE_DIR 32
```
