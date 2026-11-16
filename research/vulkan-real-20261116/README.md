# The WNR3 Vulkan prototype on a real container, on an NVIDIA RTX 3070 (2026-11-16)

The first runs of [`wex4/research/vulkan-decode/`](../../wex4/research/vulkan-decode/README.md) on a
container of a real model: the published DeepSeek V4.1 Flash REAP-192 container, 7,680 WNR3-CZ records
(codec 5, flag byte 0x0f: canonical zero and ceil form in all three matrices). It is "the test that counts"
of section 6 of [docs/HELP-WANTED.md](../../docs/HELP-WANTED.md), and the speed of `wnr3_vk bench` on real
records against the CPU decoders of the same machine. One desktop, one GPU, one driver, one day.

**Answer.** Every one of the 7,680 records decoded on the GPU to the CPU reference's bytes (0 differences);
120 of them also with the other shader form and the validation layer on (0 messages), and 1,800 mutated
copies of 80 records got the reference's verdict (all refused, 0 differences). On 16 of them the decode took
1.056 ms of device time per record against 24.1 ms for the fast CPU decoder on 8 threads.

## Question

1. Does the shader of today give the bytes and the verdicts of the CPU reference `wex_nr2_decode` on the
   records of a real container, with the validation layer on?
2. How fast is it there, against the CPU reference and the fast CPU decoder on the same machine?

## Setup

- **Container:** `v41-reap/v4_flash_experts.nr3.wex` of <https://huggingface.co/Kaira666/Kaira-WEX4-Experts>,
  the production DeepSeek V4.1 Flash REAP-192 container: WEX4 4.0, 7,680 WNR3-CZ records, grid 40 x 192,
  123,312,910,336 bytes, 0.854 of its decoded size, root `f0f78a84d040a3b336f3abef3cf8b0a36c649deb38586c813e94d35dea3beeb1`.
  Before these runs `c/tools/wex4_verify` (8 threads) found every blob and record equal to its SHA-256 in
  the index, 123.31 GB in about 262 s. Paths in the logs are `<MODELS>/` for the folder of the file.
- **Machine** (`machine.txt`): NVIDIA GeForce RTX 3070, 8 GB, driver 621.01 (Vulkan 1.4.357; the program
  prints 621.1.0.0); Intel Core i7-7700K, 4 cores, 8 threads, AVX2; 32 GB of RAM; Windows 10 (10.0.19045);
  the container on drive H:, a SATA SSD (Crucial BX500, 1 TB).
- **Software:** MSYS2 UCRT64, gcc 16.2.0, glslang 16.6.0, Vulkan loader and validation layers 1.4.363. The
  NVIDIA device was picked by name (`WNR3_VK_DEVICE=NVIDIA`); `VK_LAYER_PATH=/ucrt64/bin`.
- **Code:** this repository at b95de79 (`main` of 2026-11-16): the shader and the host program as in
  `wex4/research/vulkan-decode/` there.
- **What else ran:** the CPU load read just before and after each bench run is in `bench.log` (19% to 48%:
  other work on the desktop). The GPU ran no other job that was seen.
- **Commands:** `run.sh` of this folder, steps `every64`, `bench`, `all` and `mutate` (its header says what
  each runs). `all.log` was written by it (the step `all`, 10:50 to 14:09 UTC, `main` at b95de79 checked
  out). `every64.log` and `bench.log` were written by the same commands run by hand before the script was
  written from them. `mutate.log` holds two runs by hand while the step `all` held the GPU and its build:
  with the build of the bench (the same flags as the script's), 10 copies of one record in 384, then 20
  copies of one record in 96 (the script's defaults since).

## 1. Bytes and verdicts

"Agree" as in the prototype's README: the same verdict as the reference and, for an accepted record, the
same bytes in all six regions (three of scales, three of packed weights).

| Run (log) | Records of the container | Outcome |
|---|---:|---|
| **`wnr3_vk decode` of the whole container**, table form, validation layer off (`all.log`) | **7,680 of 7,680** | **device and reference agree on 7,680 accepted**, flag byte 0x0f, 2,600,253,492 blocks undone by the ceil-form inverse, 0 refused, 0 refused by the host checks, **0 differences**; 549 batches of at most 256 MiB; 11,923 s (3 h 19 min), the CPU reference on one thread being the slow part (about 1.55 s per record with the read) |
| `run.sh` of the prototype, `MUTATIONS=0 WNR3_VK_EVERY=64`, validation layer on (`every64.log`): one WNR3 record in 64, in 9 batches of at most 256 MiB, then again with `-DSEARCH` | 120 of 7,680 | device and reference agree on 120 accepted, flag byte 0x0f, 40,623,950 blocks undone by the ceil-form inverse, 0 refused, 0 differences; the same with the binary search; 0 messages of the validation layer; last line `vulkan-decode: all checks passed`, 377 s |
| the same run, its other checks: selftest, the three valid vectors (one record in 64 of each), the twelve invalid ones, the bench mode on the small vector | as in the prototype's README | every check as there |
| `wnr3_vk mutate 10 20261114`, one WNR3 record in 384, table form, validation layer off (`mutate.log`, first run) | 20 records, 200 mutated copies | 199 refused by the device and by the reference (error bits seen: 0xb), 1 refused by the host checks, 0 accepted, 0 differences; 82 s |
| `wnr3_vk mutate 20 20261114`, one WNR3 record in 96 (`mutate.log`, second run) | 80 records, 1,600 mutated copies | 1,593 refused by both, 7 by the host checks, 0 accepted, 0 differences; 659 s |

`WNR3_VK_EVERY` applies to the conformance vectors of `run.sh` too, so this run took one record of each
valid vector, not all seven; the selftest and the invalid vectors ran whole.

## 2. Speed

`wnr3_vk bench 21` on the first 16 WNR3 records of the container (layer 0, experts 0 to 15) in one batch:
257,089,536 stored and 300,810,240 decoded bytes, 2,160 + 17,280 streams per record, chunks 2048:512, flag
byte 0x0f. The host program was built as `bench.sh` builds it (`-O3 -march=haswell -fopenmp`, with the fast
CPU decoder of `wex4/research/fast-cpu-decode/`). What each part times is in the header comment of `bench`
in `wnr3_vk.c` and in section 2 of [research/vulkan-gpu-20261115](../vulkan-gpu-20261115/README.md). Every
device run had zero error words, and the device's bytes were the reference's in the comparison run of each
form (`0 differences from the reference`).

Default (table) form, the run that also timed the CPU (CPU load 25% just before it); ms per record is the
median of the batch divided by 16:

| Path, RTX 3070 against i7-7700K | ms per record, median of 21 runs | per batch: min to max, ms | decoded MB/s |
|---|---:|---|---:|
| GPU, decode (device timestamps, pass A + pass B) | 1.056 | 16.131 to 46.975 | 17,803 |
| GPU, decode (host clock, 2 submits and waits) | 1.080 | 16.393 to 47.500 | 17,413 |
| GPU, upload + decode (host clock) | 3.386 | 52.765 to 83.681 | 5,553 |
| CPU, fast decoder (AVX2), 8 threads | 24.108 | 378.046 to 420.029 | 780 |
| CPU, fast decoder (AVX2), 1 thread | 99.549 | 1,561.015 to 1,633.459 | 189 |
| CPU, reference decoder, 1 thread (5 runs) | 775.425 | 12,390.649 to 12,469.773 | 24 |

On these records the decode on the GPU took 1/23 of the time of the fast decoder on 8 threads, and the
upload and the decode together 1/7.1 of it. The upload alone took 2.31 ms per record (staged copy, 8.1 GB/s
of decoded-size bytes as the program counts them). The search form, run alone a minute later (CPU load 48%):
1.176 ms per record of device time, 3.521 ms with the upload, and 23.665, 100.690 and 778.270 ms for the three
CPU rows. In pairs in one process (21 pairs, device timestamps), the table form took 0.858 of the search
form's time (median of the ratios; single pairs from 0.167 to 1.629, so some pairs were disturbed); the
medians of the two forms alone in that run were 1.070 and 1.851 ms per record, against 1.176 for the search
form alone in its own process.

The synthetic records of 2026-11-15 (same geometry, flag byte 0x00, 0.772 of their decoded size) took 1.230
ms per record on the same GPU; these real records, with more payload per block (0.854) and both the
canonical-zero check and the ceil-form inverse, took 1.056 ms. Why the real records were faster was not
looked for.

## Limits

- One GPU, one driver (621.01), one operating system, one run of each step.
- Every record was decoded once, in the table form, with the validation layer off; the search form and the
  validation layer ran on one record in 64 only. Mutated copies: 1,800 of 80 records (the 20 of the first
  mutate run are among the 80 of the second), not of all.
- While the step `all` ran, other work used the machine: a local run of `scripts/check_all.sh`, the CUDA test of [research/cuda-real-20261116](../cuda-real-20261116/README.md)
  on the same GPU, and the two mutate runs; the wall time of `all` is therefore not a speed figure.
- The bench batch is the first 16 records of layer 0; other layers were not timed. The CPU was shared with
  other work (load readings in `bench.log`), so the CPU times are upper bounds.
- Not measured: an upload overlapped with the decode, a commit step, an engine. No profiler was run.

## Raw files

| File | What it is |
|---|---|
| `run.sh` | the commands of every run here |
| `machine.txt` | GPU, driver, CPU, RAM, OS, drive and package versions |
| `every64.log` | `run.sh` of the prototype, one record in 64, validation layer on |
| `all.log` | `wnr3_vk decode` of all 7,680 records |
| `mutate.log` | the two mutate runs |
| `bench.log` | `wnr3_vk bench` on 16 records: table form, search form, pairs |
