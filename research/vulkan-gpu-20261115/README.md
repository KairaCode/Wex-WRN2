# The WNR3 Vulkan prototype on an NVIDIA RTX 3070 (2026-11-15)

The first runs of [`wex4/research/vulkan-decode/`](../../wex4/research/vulkan-decode/README.md) on a GPU,
the first speed number of it, and the probe of the llvmpipe crash of issue 21 on that machine. One
desktop, one GPU, one driver, one day. **No container of a real model was on the machine**: every record
decoded here is synthetic, from a tiny test model, or from a conformance vector.

## Question

1. Does the prototype give the bytes and the verdicts of the CPU reference `wex_nr2_decode` on a GPU,
   with the validation layer on?
2. How fast is it there, against the CPU reference and the fast CPU decoder on the same machine, on records
   of the size of a real expert?
3. Do the table lookups of section 5 of [docs/WNR3-VULKAN.md](../../docs/WNR3-VULKAN.md) beat the binary
   search, with the same bytes?
4. Does the llvmpipe crash of the first shader form show on this machine (Mesa of MSYS2, the NVIDIA driver)?

## Setup

- **Machine** (`machine.txt`): NVIDIA GeForce RTX 3070, 8 GB, driver 621.01 (Vulkan 1.4.357; the
  program prints it as 621.1.0.0, NVIDIA's packing of the driverVersion field), PCIe gen 3 x16 at most; Intel Core
  i7-7700K, 4 cores, 8 threads, AVX2; Windows 10 (10.0.19045). RAM and the drive were not recorded; no
  timed part reads a file (the bench reads its records into memory first).
- **Software:** MSYS2 UCRT64, gcc 16.2.0, glslang 16.6.0, SPIRV-Tools v2026.4, Vulkan loader and
  validation layers 1.4.363, Mesa 26.2.6 (llvmpipe with LLVM 23.1.4, the second device of every run; the
  NVIDIA device was chosen by name, `WNR3_VK_DEVICE=NVIDIA`).
- **The machine was shared.** Other jobs ran on the CPU during all the runs: every log starts with the CPU
  load read just before its run (the line in brackets), from 40% to 100%. Before the three runs of round 1
  of the bench (the two that time the CPU decoders among them) the script waited 5 minutes for a quiet CPU
  and did not get one (readings from 11% to 100%). The GPU was used by nothing else that was seen (no other
  `wnr3_vk` process; other GPU users were not looked for).
- **Code:** this repository at the commit that adds this folder; the shader and the host program as in
  `wex4/research/vulkan-decode/` there. The shader of the first version (main at 1af6c0d) for question 1.
- **Command:** `bash research/vulkan-gpu-20261115/run.sh` (`run.sh` of this folder), after
  `bash wex4/build_and_test.sh`. It writes every log of this folder. The steps ran from 12:50 to 13:23 UTC,
  except `pairs` (17:23 to 17:27) and a second run of `run` and `probe` (17:40 to 17:42), after
  `crash_min.c` was added to the probe; the logs of those two steps are from that second run.

## 1. Bytes and verdicts on the RTX 3070

"Agree" as in the prototype's README: the same verdict as the reference and, for an accepted record, the
same bytes in all six regions. One run of each line, validation layer on unless said otherwise.

| Check (log) | Records | RTX 3070 | llvmpipe, same run |
|---|---:|---|---|
| `selftest`: 5 valid synthetic records (flag bytes 0x00, 0x08, 0x07, 0x0f), 7 that must be refused (`nvidia-run.log`) | 12 | 5 accepted with the reference's bytes (534 blocks undone by the ceil-form inverse), 7 refused with the error word each must have | the same |
| `decode`: the conformance vector `tiny2-cz-upgraded.wex4`, `wex4/out/tiny_nr3_cz.wex4`, `wex4/out/synth.wex4` | 2 + 48 + 4 | all accepted, 0 differences | the same |
| the same two modes with `-DSEARCH` (the binary search) | 12 + 54 | the same | the same |
| the three files with a batch limit of 256 KiB | 54 | 1, 3 and 4 batches, 0 differences | the same |
| `reject`: the three invalid vectors | 6 | as the reference: 3 refused by the host checks, 1 by the device, 2 accepted | the same |
| the bench mode on the 2 records of the vector, both forms (no time read) | 2 | 0 differences | the same |
| `mutate`, 20 copies per record, seed 20261114 | 1,080 | 1,075 refused by both, 5 by the host checks, 0 differences | the same |
| `mutate`, 1,000 copies per record, validation layer off (`nvidia-mutate-1000.log`) | 54,000 | 53,769 refused by both, 231 by the host checks, 0 accepted, 0 differences | not run |
| `negative_control.sh`, validation layer off (`nvidia-negative-control.log`) | 11 broken shaders | 11 noticed | not run |
| validation layer messages | | 0 in every run that had it on | 0 |

`llvmpipe-run.log` holds the llvmpipe column. The 5 GiB file of `big_file.py` and the sanitizer build were
not run here (the CI job runs them on lavapipe). Every record above is small: the largest has 14,000
blocks. Records of the size of a V4.1 expert were decoded only by the bench (section 2), and compared with
the reference there.

**The first run found a fault.** The shader of the first version gives other bytes on this driver for the
two valid ceil-form records of the selftest (`nvidia-main-shader.log`, made by a copy of the host program
that also explains each differing byte, `explain_diff.py`):

| Shader of the first version, on the RTX 3070 | Records 3 and 4 of the selftest |
|---|---|
| as it is, two runs | differ: 28 and 50 scale bytes, each a decrement of the ceil-form inverse that is missing; every one of them is a decrement whose 4-byte word got another decrement from a later block of the same stream (28 of 59 and 50 of 102 such decrements were lost); no other byte differs; the same counts in both runs |
| with the direct upload (`WNR3_VK_STAGE=0`) | the same counts |
| with `robustBufferAccess` | the same counts |
| with `outw` declared `coherent` | the reference's bytes |
| with `outw` declared `volatile` | the reference's bytes |
| on llvmpipe | the reference's bytes |

`spirv-val` accepts that module and the validation layer gave no message. The shader read the scale word
from the buffer for every block and stored it with one byte changed; on this driver a read sometimes gave
the word as it was before the same invocation's own store one block earlier. Since then the shader keeps
the word in a register while the stream is in it and never reads back a word it wrote (design note,
section 6), and all of the table above passes. Not looked for: the cause inside the driver, other NVIDIA
drivers, NVIDIA on Linux. Not reported to NVIDIA.

## 2. Speed

**Records.** `bench.sh` with `EXPERTS=16` wrote 16 synthetic experts of 5120 x 2304, three matrices each:
the geometry of a DeepSeek V4.1 expert record (18,800,640 decoded bytes, 1,105,920 blocks, 2,160 scale and
17,280 weight streams with chunks 2048:512). `fixtures/make_synth_mxfp4.py` draws the bytes at random from a
skewed distribution, so they are **synthetic, not weights of a model**; `c/tools/build_nr3.py` packed them as
WNR3 with a model trained on them. Flag byte 0x00: no canonical zero, no ceil form. A record takes
14,516,480 stored bytes, 0.772 of its decoded size; the real V4.1 WNR3-CZ records take 0.8545 of it
([WNR3-VERIFY](../../docs/reports/WNR3-VERIFY.md)), so a real record has more payload per block.

**What is timed** (`wnr3_vk bench`, the header comment of its function in `wnr3_vk.c`). The 16 records are
one batch (batch limit raised to 1 GiB) and are decoded 21 times, after one second of decode passes and one
whole pass that are not timed. The records are in host memory before any clock starts: no file is read.

- *GPU decode, device timestamps*: from a timestamp before the first command to one after the last command
  of the command buffer of pass A (the 16 error words cleared, then 16 dispatches) and of pass B
  (16 dispatches), the two added. No upload, no submit, no wait, no read-back.
- *GPU decode, host clock*: the same two command buffers, each with its submit and its wait for a fence.
- *upload*: the host writes the records, the model blob with its scale table, the plans and the offsets into
  mapped memory, then one command buffer copies them into device-local buffers (the staged upload), with its
  submit and wait.
- *read-back*: the decoded bytes copied to host memory. Only the comparison needs it: it is in no total.
- *CPU*: the reference decoder `wex_nr2_decode` on one thread (5 runs), and the fast decoder of
  `wex4/research/fast-cpu-decode/` (its AVX2 kernels, its tables built before the clock starts) on one
  thread and on 8 threads (OpenMP), 21 runs each, from memory into memory. gcc 16.2.0,
  `-O3 -march=haswell -fopenmp`.

The error words of every device run were zero. The device's bytes were compared with the reference's in
one more run per shader form after the timed runs, from a zeroed buffer and not timed; the bytes of the
timed runs themselves were not compared. The fast decoder's output was compared with the reference's in
every run. 0 differences (`bench.log`).

Round 1 of the default shader form (`bench.log`, "round 1, table"), the run that also timed the CPU; ms per
record is the median of the batch divided by 16:

| Path, RTX 3070 against i7-7700K | ms per record, median of 21 runs | per batch: min to max, ms | decoded MB/s |
|---|---:|---|---:|
| GPU, decode (device timestamps, pass A + pass B) | 1.230 | 19.450 to 44.125 | 15,289 |
| GPU, decode (host clock, 2 submits and waits) | 1.252 | 19.795 to 44.516 | 15,014 |
| GPU, upload + decode (host clock) | 3.637 | 54.291 to 80.849 | 5,169 |
| CPU, fast decoder (AVX2), 8 threads | 24.945 | 379.477 to 438.775 | 754 |
| CPU, fast decoder (AVX2), 1 thread | 133.956 | 1,880.542 to 3,757.892 | 140 |
| CPU, reference decoder, 1 thread (5 runs) | 862.982 | 13,484.670 to 17,555.351 | 22 |

On these records the decode on the GPU took 1/20 of the time of the fast decoder on 8 threads, and the upload
and the decode together 1/6.9 of it. The other run of round 1 (`-DSEARCH`, "round 1, search") timed the CPU
too, a few minutes earlier: 30.103, 185.101 and 903.249 ms per record for the three CPU rows (5% to 38% more
than above, with the same CPU code: the load of the other jobs), and 1.221 ms (decode) and 4.878 ms (upload
and decode) on the GPU. The device timestamps of the default form moved little over the day: 1.221 to 1.230
ms per record in the three rounds of `bench.log`, 1.181 to 1.250 in the runs of `bench-pairs.log`.

Batch size and upload path, default form, CPU not timed, one run each (`bench.log`):

| Batch | pass A, device, ms per record | pass B, device, ms per record | decode, device, ms per record | upload + decode, host clock, ms per record |
|---|---:|---:|---:|---:|
| 1 record | 0.170 | 1.322 | 1.492 | 4.052 |
| 4 records | 0.061 | 1.153 | 1.214 | 3.481 |
| 16 records (three rounds) | 0.058 | 1.162 to 1.171 | 1.221 to 1.230 | 3.515 to 3.667 |
| 16 records, direct upload (`WNR3_VK_STAGE=0`: the shader reads the inputs in host-visible memory) | 0.541 | 8.082 | 8.607 | 9.567 |

The upload takes longer than the decode: 2.1 to 2.4 ms per record with the staged copy, 6.0 to 6.5 GB/s of
stored bytes for the host's copy and the transfer together (the link is PCIe gen 3 x16 at most,
`machine.txt`; the generation in use was not read). Reading the records over the bus from the shader is 7 times slower than the decode after the copy.
Nothing here overlaps the upload of one batch with the decode of another.

## 3. Table lookups against the binary search

Before: the binary search of the first version (`-DSEARCH`: 8 probes per scale symbol, 4 per weight symbol,
then the two CDF values). After: the default form (design note, section 5): the scale symbol from a table of
one byte per slot, built on the host and checked against the search on all 2^14 slots, and the weight symbol
from one compare of the whole context row. Both forms gave the reference's bytes on every record of section 1
and in the comparison run of every bench process (section 2).

**In pairs** (`bench-pairs.log`, 17:23 to 17:27 UTC). In one process and on the batch as it was uploaded, the
two forms ran one after the other, the order alternating, 21 pairs per run; device timestamps only. Each file
ran in both orders, three rounds. The figure is the median over the 21 pairs of (time of the table form) /
(time of the search form); a run with the table form first prints the inverse, which is turned back here.

| Records | pass A | pass B | pass A + pass B: the six runs |
|---|---|---|---|
| 16 synthetic records of the V4.1 geometry (section 2) | 0.99 to 1.01 | 0.93 to 0.96 | 0.944, 0.947, 0.963 (search first); 0.950, 0.955, 0.963 (table first) |
| 48 records of the tiny-model container (`wex4/out/tiny_nr3_cz.wex4`, 13,056 decoded bytes each, flag byte 0x08) | 0.43 | 0.72 | 0.681 in each of the six |

In absolute times, on the large batch the table form took 18.90 to 20.01 ms per batch in these runs
(1.18 to 1.25 ms per record) and the search form 20.07 to 21.05 ms (1.25 to 1.32). So the table lookups take
4% to 6% less device time than the binary search on records of the V4.1 geometry, and 32% less on the small
records; with the same bytes. They are the default since this run. In these batches of 16 large records the
gain is in pass B only: pass A took 0.058 ms per record with either form, while on the small records the
scale table cut pass A to 0.43 of the search. The search form was not run in a smaller batch of large
records (the table form took 0.061 ms per record in pass A in a batch of 4, section 2). Why, was not looked for (no profiler was run).

**Alone**, the same batch, each form in its own process (`bench.log`, three rounds, device timestamps of
pass A + pass B, median of 21 runs): search 19.533, 20.632, 20.716 ms; table 19.675, 19.677, 19.539 ms. Round 1
had the table form 0.7% slower, rounds 2 and 3 4.6% and 5.7% faster; these are runs of separate processes,
not pairs. The pairs of `bench.log` (13:19 to 13:23 UTC, one order only) are not used above: the ratio of
single pairs (pass A + pass B) spread from 0.20 to 1.33 there, and the search form took 29.9 to 30.5 ms per
batch (median) in them against 19.5 to 20.7 ms alone. That did not repeat in the later session of
`bench-pairs.log`; its cause was not looked for.

## 4. The crash probe

`crash_probe.sh` on both devices of this machine (`probe-llvmpipe.log`, `probe-NVIDIA.log`): 0 of 50
runs crashed on each, the first form of the shader, `crash_min.comp` with both hosts, its host of its own
`crash_min.c` in every set layout, and every form of the reduction included; `spirv-val` accepted every
module. Because nothing crashed, the greedy reduction kept every switch on, so the forms with another
invocation stopping were not built here. The crash is a matter of llvmpipe of Mesa 25.2.8 with LLVM 20.1.2 on
the CI runner; its reduction there (to `crash_min.comp` and `crash_min.c`, crashing only with a binding 0 in
the descriptor set layout) is in the prototype's README, section "The llvmpipe crash", from the log of run
43742350289 of the repository's Actions.

## Limits

- One GPU, one driver (621.01), one operating system. No Intel, AMD or Apple device, no NVIDIA driver on
  Linux.
- No real model: no container of a real model was on the machine. The records of the bench are random
  draws from a skewed distribution packed as WNR3 with a model trained on them; flag byte 0x00, no
  canonical zero, no ceil form. Real V4.1 records are WNR3-CZ, with other statistics; how the speed
  changes with them is not known.
- The CPU was shared with other jobs during every run (load readings in the logs). The CPU times are
  therefore upper bounds of what this CPU does alone, and their spread is wide; the device timestamps
  are what the GPU did and moved less (section 2).
- One run of `run.sh` and of the negative control per device in each log; the bench as sections 2 and 3
  say. No interval is given: the bench rounds are runs of one process each, and the pairs of section 3 are
  given as medians of ratios with their runs, not as an interval.
- Not measured: a decode overlapped with the upload of the next batch, several batches in flight, a
  commit step, an engine. No profiler was run on the GPU.

## Raw files

| File | What it is |
|---|---|
| `run.sh` | the script of every run here (steps `machine run mutate control old probe bench pairs`) |
| `explain_diff.py` | writes the copy of the host program that explains the differing bytes of `nvidia-main-shader.log` |
| `machine.txt` | GPU, driver, CPU, OS and tool versions |
| `nvidia-run.log`, `llvmpipe-run.log` | `run.sh` of the prototype on the two devices, validation layer on |
| `nvidia-mutate-1000.log` | the same with `MUTATIONS=1000`, validation layer off |
| `nvidia-negative-control.log` | `negative_control.sh` on the RTX 3070 |
| `nvidia-main-shader.log` | the selftest with the shader of the first version and its variants (section 1) |
| `probe-llvmpipe.log`, `probe-NVIDIA.log` | `crash_probe.sh` on the two devices |
| `bench.log` | the bench runs of section 2, and the runs alone and the first pairs of section 3 |
| `bench-pairs.log` | the pairs of section 3: both orders, both files, three rounds |

Paths in the logs are shortened to `<Wex-WRN2>`, the repository checkout.
