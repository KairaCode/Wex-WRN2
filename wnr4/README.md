# wnr4/

**Draft, research prototype.** WNR4, a lossless record codec built so that a GPU can decode many records in one launch: the C and CUDA code, the tool APIs and the tests. Specification: [docs/WNR4-SPEC.md](../docs/WNR4-SPEC.md) (DRAFT 0.1). [docs/WEX4-SPEC.md](../docs/WEX4-SPEC.md) gives it codec 8 and blob kind 3 as a draft (section 7.3), for records that are neither EXPERT nor VALUE_EXACT. No production container, engine or production helper uses it, and the readers `c/wex4.h` and `wex4/wex4.py` do not decode codec 8. What runs today: [STATUS.md](../STATUS.md). Measurements: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md), the dated report [docs/reports/wnr4-research-20261121.md](../docs/reports/wnr4-research-20261121.md) and the runs [research/wnr4-20261121](../research/wnr4-20261121/README.md) (kinds MX and NV) and [research/wnr4-gg-20261121](../research/wnr4-gg-20261121/README.md) (kind GG).

| File | What it is |
|---|---|
| `wnr4.h` | kinds 0 (MX, MXFP4 slabs) and 1 (NV, NVFP4 slabs): header-only C99 trainer, encoder, reference decoder (`wnr4_decode`) and batch decoder (`wnr4_decode_batch`) |
| `wnr4_gg.h` | kind 2 (GG): the 18-byte blocks of the GGUF Q2_0 (ggml type 42) and IQ4_NL (20) slices of a GGUF routed-expert record in WNR4 tiles; every other slice is stored verbatim. Needs `wnr4.h` |
| `wnr4_cuda.cuh` | CUDA decoder of kinds MX and NV: one warp per tile, one launch per batch, output straight into the caller's device destinations |
| `wnr4_gg_cuda.cuh` | CUDA decoder of kind GG, bit-exact with the CPU decoder; the verbatim slices are copied from the record's device copy |
| `tools/wnr4_tools.c` | ctypes API of `wnr4.h`, used by the research scripts |
| `tools/wnr4_gg_tools.c` | ctypes API of `wnr4_gg.h`, used by `wex4/wnr4gg.py` and its tests |
| `tools/wnr4_cuda_bridge.cu` | prototype CUDA helper DLL for all three kinds (below) |
| `tests/test_wnr4.c` | CPU tests of MX and NV: round trips over G 1..256, partial tiles, ceil form, canonical zero, refusals, a mutation fuzz |
| `tests/test_wnr4_gg.c` | CPU tests of GG: round trips, refusals, a mutation fuzz, known answers that pin the format bytes |
| `tests/test_wnr4_gg_gpu.cu` | GPU test of GG through the helper on sampled GGUF experts, with kernel-only GB/s |
| `tests/sample_gg_experts.py` | writes the input of the GPU test from a GGUF part |

## Build and test

From the repository root. CPU tests of all three kinds (MSYS2 UCRT64 gcc on Windows; `CC`, `WEX_OPENMP`, `WEX_CFLAGS` and `WEX4_MARCH` as the script's header says):

```
bash scripts/build_wnr4_check.sh          # last line on success: WNR4 CPU check passed
```

The same tests by hand, and the two tool libraries (`-march=haswell`, x86-64 only, builds the AVX2 table search of `wnr4.h`; `-static` gives a Windows DLL without MinGW runtime dependencies):

```
gcc -O2 -fopenmp wnr4/tests/test_wnr4.c -o test_wnr4 && ./test_wnr4 [fuzz_iterations]
gcc -O2 -fopenmp wnr4/tests/test_wnr4_gg.c -o test_wnr4_gg && ./test_wnr4_gg [fuzz_iterations]
mkdir -p research/wnr4-20261121/out
gcc -O3 -march=haswell -fopenmp -static -shared -o research/wnr4-20261121/out/wnr4_tools.dll wnr4/tools/wnr4_tools.c
gcc -O3 -march=haswell -fopenmp -static -shared -o wnr4_gg_tools.dll wnr4/tools/wnr4_gg_tools.c
```

The helper and the GPU test of GG (CUDA 13, built for `sm_120` and run on an RTX 5090 only). On Windows nvcc needs the MSVC host compiler: [research/wnr4-20261121/nvcc.cmd](../research/wnr4-20261121/nvcc.cmd) runs `vcvars64.bat` first, and the research runs added `-Xcompiler /O2,/EHsc`.

```
nvcc -O3 -std=c++17 -arch=sm_120 -shared wnr4/tools/wnr4_cuda_bridge.cu -o wnr4_cuda.dll
nvcc -O3 -std=c++17 -arch=sm_120 wnr4/tests/test_wnr4_gg_gpu.cu -o test_wnr4_gg_gpu
python -I wnr4/tests/sample_gg_experts.py PART.gguf SAMPLE_DIR 0,6,12,24,36,47 0-15
./test_wnr4_gg_gpu SAMPLE_DIR [G]         # last line on success: all GG checks passed
```

The GPU bench and helper test of kinds MX and NV, and the CPU bench, are in the research folder and read packs of real records that `research/wnr4-20261121/measure.py --pack` writes (that folder's README, "Rerun"; `measure.py` loads `wnr4_tools.dll` and `wex_nr3_tools.dll` from `research/wnr4-20261121/out/`, and that README builds the second):

```
nvcc -O3 -std=c++17 -arch=sm_120 -o bench_gpu research/wnr4-20261121/gpu/bench_gpu.cu      # bench_gpu PACK [iterations]
nvcc -O3 -std=c++17 -arch=sm_120 -o test_helper research/wnr4-20261121/gpu/test_helper.cu  # test_helper PACK
gcc -O2 -march=haswell -fopenmp -static -o bench_cpu research/wnr4-20261121/gpu/bench_cpu.c  # bench_cpu PACK [records] [max_threads]
```

## The helper (ABI 5)

`tools/wnr4_cuda_bridge.cu` is a prototype, separate from the production helper `c/tools/wex_cuda_bridge.cu`. `wnr4_cuda_abi()` returns 5 since the GG entry points were added (it was 4; the MX and NV entry points are unchanged). Records come from page-locked host memory and decode straight into caller-owned device memory; a record that fails reports -1 and its destination may hold partial data, so the caller must not use that slot.

| Entry point | Kinds | What it does |
|---|---|---|
| `wnr4_cuda_abi()` | | returns 5 |
| `wnr4_cuda_register_model(blob, bytes, device)` | MX, NV | checks a WNM4 blob and uploads its tables: 0 ok (also when already registered), -1 invalid blob, 1 CUDA error |
| `wnr4_cuda_batch_submit(n, geo, records, totals, owners, model_id, device, dst_scale, dst_weight, stream)` | MX, NV | queues a batch of one geometry; returns a ticket, or -1 bad arguments, -2 model not registered, -3 CUDA or memory error, -4 the slot still holds a batch not waited for |
| `wnr4_cuda_batch_wait(ticket, status)` | all | status per record: 0 decoded, -1 invalid record or failed integrity check, 1 CUDA error; returns the number of records with status != 0, or -1 for an unknown ticket |
| `wnr4_cuda_decode_batch(n, geo, records, totals, owners, model_id, device, dst_scale, dst_weight, status)` | MX, NV | submit and wait in one call |
| `wnr4_cuda_gg_register_model(blob, bytes, device, id)` | GG | as `wnr4_cuda_register_model`, and `*id` = the model id |
| `wnr4_cuda_gg_batch_submit(n, nseg, types, bytes, records, totals, owners, model_id, device, dst, stream)` | GG | one geometry (ggml types and byte sizes of the slices) per batch; each record decodes to its whole slab at `dst[i]`, verbatim slices included; waited for with `wnr4_cuda_batch_wait` |
| `wnr4_cuda_gg_decode_batch(n, nseg, types, bytes, records, totals, owners, model_id, device, dst, status)` | GG | submit and wait in one call |
| `wnr4_cuda_release_thread()` | all | frees this thread's slots on every device, after waiting for batches in flight |

Up to 4 batches may be in flight per calling thread and device; a fifth submit before a wait gets -4, so no verdict is dropped. Every ticket is waited for on the thread that submitted it, and the record buffers stay unchanged until then. With `stream` NULL the helper uses its own non-blocking streams, which are not ordered after work on the legacy default stream. Every call leaves the caller's current CUDA device as it found it. The file's comments give the arguments in full.

## Packing and verifying

- **Kind GG:** `wex4/gguf2wex4.py pack RAW.wex4 MODEL.gguf` writes a RAW container (with `--experts-only`, of the routed experts alone); `wex4/wnr4gg.py` then trains a model (`train RAW.wex4 MODEL.wnm4 --dll D [--every K]`), packs a codec-8 container (`pack RAW.wex4 OUT.wex4 --model MODEL.wnm4 --dll D`, G = 32 unless `--G` says otherwise), decodes and compares every record (`verify OUT.wex4 --source RAW.wex4 --dll D`) and rebuilds the GGUF parts from a container that is not experts-only (`unpack`). D is `wnr4_gg_tools.dll` from above. Tests: `python -I wex4/test_wnr4gg.py`, which builds the library itself with `$CC`. See [wex4/README.md](../wex4/README.md).
- **Kinds MX and NV:** there is no container writer. `research/wnr4-20261121/measure.py` trains a model on 256 records of an existing WEX4 container, encodes other records, decodes every one back and compares it, and gives the ratio against WNR3; `--pack` writes records for the GPU and CPU benches. The NVFP4 runs need the `nvfp4-wnr3` branch, which is not public yet.

## Not in CI

`scripts/check_all.sh`, which CI runs, runs `scripts/build_wnr4_check.sh` and `wex4/test_wnr4gg.py`. Nothing that needs a GPU is in CI: the helper, `tests/test_wnr4_gg_gpu.cu` and the benches of `research/wnr4-20261121/gpu/`; nor is any run on a real model. The CUDA code was built for `sm_120` and run on one RTX 5090 only.
