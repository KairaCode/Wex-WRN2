# WNR4 kind GG research (2026-11-21)

The measurements behind kind 2, "GG", of [docs/WNR4-SPEC.md](../../docs/WNR4-SPEC.md) (section 5), section 8.2 of [docs/reports/wnr4-research-20261121.md](../../docs/reports/wnr4-research-20261121.md#82-kind-gg-gguf-2-bit-experts) and the GGUF mapping of [docs/WEX4-SPEC.md](../../docs/WEX4-SPEC.md) (section 14.1). Codec: `wnr4/wnr4_gg.h` (CPU), `wnr4/wnr4_gg_cuda.cuh` (CUDA), `wnr4/tools/wnr4_gg_tools.c` (ctypes tool API), `wnr4/tools/wnr4_cuda_bridge.cu` (prototype helper DLL, GG entry points), `wnr4/tests/test_wnr4_gg.c` (CPU tests), `wnr4/tests/test_wnr4_gg_gpu.cu` (GPU test). Container tools: `wex4/gguf2wex4.py` (GGUF to WEX4), `wex4/wnr4gg.py` (WNR4-GG records in such a container). The MXFP4 and NVFP4 kinds were measured in [../wnr4-20261121/](../wnr4-20261121/README.md).

## Question

Can WNR4 tiles store the 2-bit GGUF routed experts (the Q2_0 and IQ4_NL slices of IQ1_M-class quantizations) in fewer bytes, byte-exact, and decode them fast enough on the GPU and the CPU to pay off in an engine that reads its experts from disk?

## Setup

- What GG codes: only the slices of ggml type 42 (Q2_0) and 20 (IQ4_NL), 18-byte blocks (f16 scale, 16 code bytes). Every other slice of the record (here the gate and up slices) is stored verbatim in the record. Why only these two types: see "Pre-study" below.
- Model A: Qwen3.8-Flash-Next UD-IQ1_M (GGUF by unsloth, 3 parts), 48 layers x 512 experts = 24,576 routed-expert records, each gate | up | down; down Q2_0 (ggml 42) in 46 layers and IQ4_NL (20) in 2; gate and up IQ2_XXS (16), IQ2_S (22) or IQ2_XS (17); 34,812,723,200 B of experts.
- Model B, "the second pack": a second GGUF quantization of a Qwen3.8-Flash-Next variant with 256 experts per layer, labelled IQ1_M; 48 layers x 256 = 12,288 experts, 25.146 GB; hidden size 2560, expert FFN size 640. Gate and up are IQ3_XXS (18) in 17 layers, IQ2_S (22) in 20, IQ3_S (21) in 10, IQ4_XS (23) in 1; down is IQ4_NL (20) in 39 layers and Q2_0 (42) in 9. Gate and up are about 59% of the pack's bytes (from the slice sizes in `results/second-pack/sample.json`), all verbatim under GG. It was read from an expert pack in the layout `sample_pack.py` describes (`experts.bin` + `native_experts.txt`), not from a WEX4 container.
- First check: 96 experts of model A sampled by `wnr4/tests/sample_gg_experts.py` (layers 0, 6, 12, 24, 36, 47, experts 0-15). `measure_sample.py` trains a WNM4 kind-2 model on the 48 even-numbered experts, codes the 48 odd-numbered ones and decodes them back on 8 threads. Stored/raw is against the raw record aligned to 4096 B.
- GPU test: the same 96 records through `wnr4/tests/test_wnr4_gg_gpu.cu` on an RTX 5090 (CUDA 13.4, `sm_120`). Records resident in VRAM, L2 (96 MiB) flushed before each run, kernel time only, median of 20; 460,800 coded + 844,800 verbatim B per record.
- Whole model: model A on a second Windows desktop (Ryzen 7 9700X, 16 GB RAM, Python 3.14): `gguf2wex4.py pack --experts-only` to a RAW container, `check_vs_gguf.py`, `wnr4gg.py train --every 8`, `wnr4gg.py pack` at G = 32 and `wnr4gg.py verify --source` (CPU decode, 4 threads).
- Second pack: `sample_pack.py` takes 10 experts per layer (`random.Random(20261121)`): the first 4 train (192 records), the other 6 are held out (288 records); 480 records, 982,272,000 B. `measure_pack.py` trains on the 192, encodes every record on one thread at G = 32, decodes it on 1 or 16 threads and compares. Speed on a Ryzen 9 7950X3D, DLL built with gcc 16.2 `-O3 -march=native -fopenmp`.
- No engine build and no engine run with GG records. Commands: section "Rerun".

## Results

### First check, model A sample (`results/first-check.log`)

48 held-out records coded, 0 failures at both G. Model id 0x911e2cd3, blob 36,644 B. Only the down slice (Q2_0, 460,800 B) is coded: in these six layers no down slice is IQ4_NL.

| G | 48 records | 40 records, types 16+16+42 (1,305,600 B each) | 8 records, types 22+22+42 (1,510,400 B each) |
|---:|---:|---:|---:|
| 128 | 0.9546 | 0.9531 | 0.9610 |
| 32 | 0.9553 | 0.9540 | 0.9610 |

G = 32 costs 0.0007 of ratio (0.07 points) against G = 128 on these records.

### GPU test, the same 96 records (`results/gpu-test.log`)

- Every real record bit-exact through the helper: 80/80 and 16/16 in synchronous batches of 32, and in asynchronous batches of 16 with 2 in flight (failures 0). One payload bit flipped in each odd record: refused 40/40 and 8/8, the even ones bit-exact with status 0.
- Synthetic groups cut from these records (partial tiles at G 3, tiles of one block at G 1), at 4, 8 and 16 warps per block: all bit-exact, flipped records refused. A tile with one unused word, a wrong owner, an unknown model and a bad geometry are refused. Last line: `all GG checks passed`.
- At G = 32 the CPU encode and decode of all 96 records (training records included) gives 0.9542 on the 80 with type-16 gate and up and 0.9612 on the 16 with type 22.

Kernel only, median of 20 (GB/s of coded slices; in brackets the time with the copy of the verbatim slices and GB/s of whole records):

| batch | tiles at G = 32 | G = 32 | tiles at G = 128 | G = 128 |
|---:|---:|---|---:|---|
| 1 | 25 | 0.224 ms, 2.1 GB/s (0.234 ms, 5.6 GB/s) | 7 | 0.861 ms, 0.5 GB/s (0.872 ms, 1.5 GB/s) |
| 8 | 200 | 0.227 ms, 16.3 GB/s (0.247 ms, 42.3 GB/s) | 56 | 0.869 ms, 4.2 GB/s (0.890 ms, 11.7 GB/s) |
| 16 | 400 | 0.234 ms, 31.5 GB/s (0.265 ms, 78.8 GB/s) | 112 | 0.874 ms, 8.4 GB/s (0.906 ms, 23.0 GB/s) |
| 32 | 800 | 0.259 ms, 57.0 GB/s (0.314 ms, 133.0 GB/s) | 224 | 0.875 ms, 16.8 GB/s (0.931 ms, 44.9 GB/s) |

At batch 32, G = 32 decodes the coded slices 3.4 times as fast as G = 128 (57.0 / 16.8 GB/s), for 0.07 points of ratio. One record has 25 tiles at G = 32 and 7 at G = 128, so G = 128 gives the GPU fewer tiles to run in parallel. This is why `wex4/wnr4gg.py` packs at G = 32 by default.

### Whole model A (`results/full-model/`)

| step | result | log |
|---|---|---|
| `gguf2wex4.py pack --experts-only` | 24,576 records (48 expert layers); RAW container 34,862,370,816 B, root `837198d1b8adab49a04077e129e516f7bb45089668177e5238a545a49f646f1a` | `pack_raw.log` (root; the file size is not in the log) |
| `check_vs_gguf.py` | every record equal to its GGUF slices and to its SHA-256: 24,576 records, 0 bad, 110 s, same root | `check_raw.log` |
| `wnr4gg.py train --every 8` | 3,072 records (4.35 GB), model 0xa7b23c09, 36,644 B | `train.log` |
| `wnr4gg.py pack` (G = 32) | all 24,576 records coded: 33,354,760,192 B against 34,839,986,176 B RAW (0.9574); 144-176 MB/s; container 33,377,181,696 B, root `0a9bc529ef90c0ddde78c93e7285f37f48733e674c6a18238addc8dec24fd93d` | `pack_wnr4.log` (the container file size is not in the log) |
| `wnr4gg.py verify --source RAW` | all 24,576 records (all codec 8) decoded on the CPU (4 threads), identical to the RAW container; 390 MB/s | `verify_wnr4.log` |

The WNR4-GG container is 1.49 GB (4.3%) smaller on disk than the RAW one. Only the down slices are coded; the IQ2 gate and up slices are stored verbatim.

### Second pack (`results/second-pack/`)

One model trained on the 192 training records, G = 32. Ratio = coded record bytes (padded to 4096 B) / raw bytes. "Down slice" counts the gate and up slices as stored verbatim and gives the rest of the record to the down slice.

| records | n | raw B | coded B | stored/raw | down slice |
|---|---:|---:|---:|---:|---:|
| held out, all | 288 | 589,363,200 | 575,422,464 | **0.9763** | 0.9420 |
| held out, down IQ4_NL (20) | 234 | 501,964,800 | 490,917,888 | 0.9780 | 0.9488 |
| held out, down Q2_0 (42) | 54 | 87,398,400 | 84,504,576 | 0.9669 | 0.8837 |
| held out, gate/up IQ3_XXS (18) | 102 | 213,657,600 | 208,723,968 | 0.9769 | 0.9424 |
| held out, gate/up IQ3_S (21) | 60 | 137,011,200 | 134,098,944 | 0.9787 | 0.9446 |
| held out, gate/up IQ2_S (22) | 120 | 222,720,000 | 216,911,872 | 0.9739 | 0.9400 |
| held out, gate/up IQ4_XS (23) | 6 | 15,974,400 | 15,687,680 | 0.9821 | 0.9481 |
| training | 192 | 392,908,800 | 383,651,840 | 0.9764 | 0.9423 |

The training records (0.9764) and the held-out ones (0.9763) agree: no overfit. All 480 records decoded byte-identical at 1 and at 16 decode threads.

Speed, 288 held-out records, MB/s of raw record bytes (Ryzen 9 7950X3D):

| decode threads | byte-identical failures | encode, one thread | decode | log |
|---:|---:|---:|---:|---|
| 1 | 0 / 480 | 323.9 MB/s | 223.7 MB/s | `measure-t1.log` |
| 16 | 0 / 480 | 330.4 MB/s | 1690.8 MB/s | `measure-t16.log` |

An earlier run of the same sample, with the codec at an earlier commit (same format, same ratio 0.9763), measured 218 MB/s and 1.88 GB/s; its log is not in this folder. So the CPU decode on 16 threads is 1.7-1.9 GB/s. Whole pack, estimated from the held-out ratio: about 24.55 GB (0.9763 of 25.146 GB), about 0.6 GB saved.

**Disk-bound reading.** The engine that runs this pack reads expert records from NVMe at about 4.7 GB/s on one read thread (an engine measurement, not in this repository). A CPU decode at 1.7-1.9 GB/s on 16 threads is slower than the read it would replace. With a free decode, the 2.4% fewer bytes would give at most about +2% tok/s, below that engine's A/B noise (1.4-2.8% per-arm spread). Verdict: no gain when the experts are read from disk; not integrated.

### Pre-study: per-type coding of UD-Q2_0 (no scripts or logs here)

An earlier prototype coder in the WNR4 style (not the GG codec) measured each block type on 192 records of Qwen3.8-Flash-Next UD-Q2_0, sampled by HTTP Range requests; UD-Q2_0 has the same block types as UD-IQ1_M. Tables were trained on 2 experts per layer and tested on 2 others. Its scripts and logs are not in this repository, and the test sets are small (Q2_0 on 20 records, IQ4_NL on 12).

| ggml type | stored/raw | zstd -19 |
|---|---:|---:|
| Q2_0 (42) | 0.862 | 0.905 |
| IQ4_NL (20) | 0.946 | 0.981 |
| IQ2_XXS (16) | 0.958 | 0.998 |
| IQ2_S (22) | 0.949 | 1.000 |
| IQ2_XS (17) | not measured | not measured |

The four Q2_0 code values have the frequencies 0.275 / 0.428 / 0.235 / 0.062. The IQ2 codes are grid indices and signs, near maximum entropy; richer contexts (previous symbol, row above, class of d) moved them less than 0.3 points. The estimate for UD-IQ1_M was about 32.2 GB with every type coded (-7.5%) and about 33.3 GB with Q2_0 and IQ4_NL only (-4.5%); the measured container is 33.38 GB (whole model A above). This is why GG codes only Q2_0 and IQ4_NL: coding the IQ2 types as well would save about 1.1 GB more on UD-IQ1_M and needs a layout per type. Left for later.

## Conclusions

Kind GG codes only the Q2_0 and IQ4_NL slices of a GGUF routed-expert record; the other slices stay verbatim. The whole Qwen3.8-Flash-Next UD-IQ1_M model (24,576 records) packs to 0.9574 of its RAW records at G = 32, 1.49 GB (4.3%) less on disk, every record decoded identical; on 288 held-out records of a second IQ1_M pack, whose down slices are mostly IQ4_NL (which codes less well than Q2_0), it is 0.9763. The CUDA decoder reaches 57.0 GB/s of coded slices at G = 32 (batch 32, RTX 5090) against 16.8 GB/s at G = 128. The CPU decoder (1.7-1.9 GB/s on 16 threads) is slower than the NVMe read of the engine that runs the second pack (about 4.7 GB/s): no gain when the experts are read from disk. Not integrated in any engine.

The three GG ratios are all valid, for different data: 0.9546 / 0.9553 on the 48 held-out records of the first check, 0.9574 on the whole model A, 0.9763 on the held-out records of the second pack.

## Limits

- One run of each figure. The GPU figures are medians of 20 within that one run; the ratios, the pack, train and verify speeds and the second-pack speeds come from one run each. The earlier second-pack run (218 MB/s, 1.88 GB/s) has no log here.
- One GPU (RTX 5090). The whole-model speeds come from a Ryzen 7 9700X and the second-pack speeds from a Ryzen 9 7950X3D: do not compare them with each other. The codec commit of each run and the machine of the first check are not recorded.
- The first check samples only six layers whose down slices are all Q2_0; IQ4_NL down slices and IQ2_XS gate and up slices are not in it.
- No engine run with GG records and no tok/s. The GPU path (decode into VRAM inside an engine) is not measured. The disk-read figure (about 4.7 GB/s) and the A/B noise of the disk-bound reading come from engine runs that are not in this repository.
- The second pack and the engine that runs it are not public, so `sample_pack.py` cannot be rerun from this repository alone (its docstring describes the `experts.bin` + `native_experts.txt` layout it reads). `sample.bin` (982,272,000 B of model weights) is not tracked; `results/second-pack/sample.json` names every sampled record.
- The pre-study has no scripts or logs in this repository and few test records.
- The model GGUF files and the containers are not tracked.

## Files

| file | what |
|---|---|
| `measure_sample.py` | first check: trains on the even experts of a `sample_gg_experts.py` sample, codes and decodes the odd ones, prints stored/raw by gate/up/down types |
| `check_vs_gguf.py` | every record of an experts-only WEX4 against the GGUF bytes it came from, and its SHA-256 |
| `sample_pack.py` | the seeded 480-record sample of the second pack (`sample.bin` + `sample.json`) |
| `measure_pack.py` | second pack: trains on the training records, encodes and decodes every record at G = 32, byte check, ratio and MB/s by group |
| `results/first-check.log` | output of `measure_sample.py` at G = 128 and G = 32 |
| `results/gpu-test.log` | output of `test_wnr4_gg_gpu` on the 96 records |
| `results/full-model/pack_raw.log`, `check_raw.log`, `train.log`, `pack_wnr4.log`, `verify_wnr4.log` | the whole-model steps, in that order |
| `results/second-pack/sample.log`, `sample.json` | output of `sample_pack.py` and the list of sampled records |
| `results/second-pack/measure-t1.log`, `measure-t16.log` | output of `measure_pack.py` with 1 and 16 decode threads |

## Rerun

From the repository root. Builds: MSYS2 UCRT64 gcc for the DLL (the second-pack speeds used `-O3 -march=native -fopenmp` with gcc 16.2), CUDA 13 for the GPU test (on Windows nvcc needs the MSVC host compiler, see [../wnr4-20261121/nvcc.cmd](../wnr4-20261121/nvcc.cmd)). `PART.gguf` is the GGUF part that holds the sampled layers' expert tensors, `MODEL-00001-of-00003.gguf` the first part of model A.

```
bash scripts/build_wnr4_check.sh      # CPU tests of all WNR4 kinds, GG included
gcc -O3 -march=haswell -fopenmp -static -shared -o wnr4_gg_tools.dll wnr4/tools/wnr4_gg_tools.c

# first check and GPU test (96 experts of model A)
python -I wnr4/tests/sample_gg_experts.py PART.gguf SAMPLE_DIR 0,6,12,24,36,47 0-15
python -I research/wnr4-gg-20261121/measure_sample.py SAMPLE_DIR wnr4_gg_tools.dll 128
python -I research/wnr4-gg-20261121/measure_sample.py SAMPLE_DIR wnr4_gg_tools.dll 32
nvcc -O3 -std=c++17 -arch=sm_120 wnr4/tests/test_wnr4_gg_gpu.cu -o test_wnr4_gg_gpu
./test_wnr4_gg_gpu SAMPLE_DIR

# whole model A
python -I wex4/gguf2wex4.py pack RAW.wex4 MODEL-00001-of-00003.gguf --experts-only
python -I research/wnr4-gg-20261121/check_vs_gguf.py RAW.wex4 MODEL-00001-of-00003.gguf
python -I wex4/wnr4gg.py train RAW.wex4 MODEL.wnm4 --dll wnr4_gg_tools.dll --every 8
python -I wex4/wnr4gg.py pack RAW.wex4 GG.wex4 --model MODEL.wnm4 --dll wnr4_gg_tools.dll
python -I wex4/wnr4gg.py verify GG.wex4 --source RAW.wex4 --dll wnr4_gg_tools.dll

# second pack (PACK_DIR: experts.bin + native_experts.txt)
python -I research/wnr4-gg-20261121/sample_pack.py PACK_DIR SAMPLE2_DIR
python -I research/wnr4-gg-20261121/measure_pack.py SAMPLE2_DIR wnr4_gg_tools.dll 1
python -I research/wnr4-gg-20261121/measure_pack.py SAMPLE2_DIR wnr4_gg_tools.dll 16
```

`wnr4gg.py pack` uses G = 32 unless `--G` says otherwise. Generated data (the DLL, samples, containers, binaries) is not tracked.
