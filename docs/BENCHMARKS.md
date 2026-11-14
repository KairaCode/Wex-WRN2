# Benchmarks

Every measurement behind the formats and the read path: container sizes, integrity, decode speed, start-up time and the drives. What runs in production is decided in [STATUS.md](../STATUS.md); this page holds the numbers. Each table names the report or the run it comes from; the raw output of the runs is under [research/](../research/README.md).

## Method

- **A/B, in pairs.** A switch or a container is measured against the same build without it. Most A/Bs here are ABBA: the arms run in the order A1 B1 B2 A2, each in a fresh process with a cold start, so that a slow drift of the machine does not favour one arm. "One pair" is one A and one B and gives no interval; an interval (± ...) is given only where several pairs stand behind it.
- **Conditions.** Each number carries its model, container form, drives, RAM budget (`--ram` / `--memory-gb`, in GB), prompts x tokens and engine build. The maintainer's application launches the engine through its gateway with a profile; "the `v41gpu` profile" and "the `qwen38` profile" are those settings ([ARCHITECTURE.md](ARCHITECTURE.md)).
- **Identity apart from speed.** "Tokens identical" or "logits bit-identical" is checked and stated separately from tok/s; a switch that changes the tokens says so.
- **What was not run** is stated next to the result. The rules are in [CONTRIBUTING.md](../CONTRIBUTING.md), rule 5.

## Machine

One desktop: NVIDIA RTX 5090 (32 GB, sm_120), two NVMe drives, U: (PCIe 4) and C: (PCIe 5), each holding a full copy of the containers, Windows 11. Engines built with MSYS2 UCRT64 gcc 16.2 and the CUDA toolkit 13.4 (the maintainer's build notes); tools run with Python 3.14. The CPU-only baseline of 2026-11-07 ran on the same machine.

## Reproduce

- **Without a model:** the format sizes and the integrity checks of the tiny fixtures, the corruption audit of WNR2 against WNR3 (`c/tests/test_wex_format.c`, printed by `scripts/build_cpu_check.sh`), and the CPU decode benches of WEX4 ([wex4/README.md](../wex4/README.md), `wex4/research/fast-cpu-decode/`) run from this repository alone.
- **With a checkpoint:** the container sizes are reproduced by building the containers ([COLIBRI-INTEGRATION.md](COLIBRI-INTEGRATION.md)) from DeepSeek V4.1 Flash REAP-192 or Qwen3.8-Flash-Next MXFP4.
- **With an engine:** every tok/s and TTFT number needs the engine builds named in its row, the checkpoint and the drives; they cannot be rerun from this repository. Their logs are linked so that the numbers can be checked against the raw output.

## Where the numbers come from

Rows and paragraphs up to 2026-11-11 are the old README's, regrouped; a correction is marked where it was made. Of those rows that name no report or run, the ones on the WNR3 switch and on the refit predictor gates are from [docs/reports/WNR3-PRODUCTION-20261110.md](reports/WNR3-PRODUCTION-20261110.md). The others (striped reads, GPU prefill bank, demand-read priority, the WEX4 U: against U:+C: pair, the low-RAM fix and its plan table, `V41_ATTN_GPU`, the 2026-11-11 drive figures) were carried over from the README of 2026-11-11, which predates this repository's current history; their raw output is not in this repository, so they cannot be verified here. The rows of 2026-11-12 and 2026-11-13 link their log in [research/wex-rd-20261112/](../research/wex-rd-20261112/README.md), a copy of the maintainer's lab folder `gpu-miss-20261112`.

## DeepSeek V4.1 REAP-192: container sizes

| V4.1 REAP-192, RTX 5090 | REAP safetensors (original) | WEX1 | WNR2 | **WNR3-CZ** |
|---|---|---|---|---|
| routed experts (7,680) | 151,608,360,960 B (141.2 GiB) of the 162,750,843,576 B checkpoint | 144,389,472,256 B | 129,905,360,896 B | **123,306,192,896 B (114.8 GiB)** |
| size vs the safetensors experts | 1.000 | 0.952 | 0.857 | **0.813** (0.854 of WEX1, 0.949 of WNR2) |
| wrong single-bit flips accepted (random, whole real record, CPU) | no integrity check | no integrity check (raw) | 1.8-1.9% | 0 / 3,000 |
| on disk now | U: (the engine still reads dense weights, attention and head from it) | deleted | deleted | inside WEX4 on U: + C: mirror (WNR3 sha256 before the upgrade 490e8208...) |

Note added 2026-11-12: the safetensors column counts 8,064 experts and the three container columns 7,680. The difference, 151,608,360,960 - 7,680 x 18,800,640 = 7,219,445,760 B, is 384 records of 18,800,640 B: the 384 MTP experts, which are not in the containers ([docs/WEX4-SPEC.md](WEX4-SPEC.md), section 12.3). Against the same 7,680 records WNR3-CZ is 0.854 of raw.

Before the full build WNR3 was verified on 103 real V4.1 records: 0.8545 of raw on 40 held-out experts (WNR2 0.8994), value-exact 103/103 on CPU and GPU, 0 wrong single-bit flips accepted (0/2,231,336 GPU, 0/3,000 random CPU; WNR2 accepts 1.8-1.9% of random flips), same GPU decode speed. The end-to-end logits proof of canonical zero (G7) passed ([docs/reports/WNR3-VERIFY.md](reports/WNR3-VERIFY.md)).

## DeepSeek V4.1 REAP-192: containers, drives and decode speed

| measurement (7 prompts x 64 tokens unless noted) | before | after |
|---|---|---|
| mirror effect, one pair ([NOTE-5](../research/wnr2-fastbench-20261109/NOTE-5-mirror-ab-20261110.md)) | WEX1 U: only 2.02 tok/s | WNR2 U:+C: 2.99 tok/s (**+49%**, TTFT -22%), bit-identical |
| pre-switch A/B, WNR3 engine a55a4ce0 + helper 602e3f4d | WNR2 U:+C: 3.05 tok/s | WNR3 U: only 2.15 tok/s (-30%, no mirror yet); logits bit-identical, 58,568 GPU-direct WNR3 decodes, 0 failures |
| production, launch identical to the application's, 1,886-token turn | WNR2 U:+C: TTFT 326 s, 2.86 tok/s | WNR3 U:+C: TTFT 339-352 s, **2.92-2.96 tok/s**, same text |
| production, final engine with GPU required | WNR2 reference logits a14761eb... | WNR3 U:+C: **2.7-3.3 tok/s (mean ~3.0)**, logits bit-identical |
| WEX4 installed engine, first-read SHA-256 on, one pair, 3 prompts x 64 tokens | WEX4 U: only 2.26 tok/s | WEX4 U:+C: **3.01 tok/s** (+34%; WNR3 U:+C: was ~3.0), logits bit-identical, 0 GPU decode failures |

The second drive is the lever; WNR3 makes the container 5.1% smaller than WNR2 and adds a per-stream integrity hash, with per-record GPU decode at parity. WNR3 + mirror against WNR2 + mirror was not A/B-tested in pairs. Neither Windows memory compression (FP4/FP8/BF16 weights and WNR3 records do not compress) nor the compressed-record RAM cache (equal-RAM A/B -0.19% ± 0.72%, not adopted: [RESULT](../research/wnr3-20261110/ram-record-cache/RESULT-20261110.md)) gives a speed gain at this RAM budget.

## DeepSeek V4.1 REAP-192: WEX start-up and read switches (2026-11-12 and 2026-11-13)

Start-up A/Bs: the gateway launched as the application launches it (`v41gpu` profile, `--ram 13`, WEX4 U:+C:), one short user message (a 647-token prompt after the template), 8 tokens; ABBA, one pair, a cold start per arm. "Ready" is the gateway's ready time; "start to first token" runs from the gateway start to the first token, so it covers load and prewarm. `vram_experts` is the number of experts the prewarm placed in VRAM. Logs: [m4-ahead.log](../research/wex-rd-20261112/m4-ahead.log), [m5-pwbatch.log](../research/wex-rd-20261112/m5-pwbatch.log), [m6-shaasync.log](../research/wex-rd-20261112/m6-shaasync.log), from the queue [rd_queue3.log](../research/wex-rd-20261112/rd_queue3.log).

| switch (build) | ready A / B | start to first token A / B | B/A | vram_experts A / B | verdict |
|---|---|---|---|---|---|
| `COLI_V4_TIER_PREWARM_AHEAD=4` (installed `deepseek_v4.exe` 7ac3baf4) | 28, 26 s / 18, 18 s | 140.9, 138.9 s / 129.6, 132.0 s | **-6.5%** | 963, 964 / 961, 960 | **installed** in the `v41gpu` profile, 2026-11-12 16:15 |
| `V4_PREWARM_BATCH=1` (branch build rel3-wexprewarm) | 32, 30.1 s / 12, 12 s | 145.9, 143.5 s / 126.1, 125.9 s | **-12.9%** | 960, 959 / 952, 946 | winner; **installed** 2026-11-13 04:43 with the rel6 build |
| `V4_WEX_SHA_ASYNC=1` (branch build rel3-wexsha) | 18, 20 s / 16, 16 s | 131.4, 135.1 s / 131.6, 134.4 s | -0.2% | not compared | neutral at start-up; its target is decode demand reads |

The prompt's own TTFT is 111.6-118.4 s in every arm and moves little between A and B, so the gains of the first two rows are in the start-up.

Decode A/Bs of 2026-11-13: release build rel6 of branch rd2-integ-20261112 (`deepseek_v4.exe` 6793ea73), `v41gpu` profile, `--ram 15`, ABBA, one pair each, greedy.

| switch | prompt, tokens | A1 / A2 | B1 / B2 | B/A | text | verdict |
|---|---|---|---|---|---|---|
| `V4_WEX_SUBREADS=4` ([log](../research/wex-rd-20261112/r4-r9subreads.log)) | 657-token prompt, 75 tokens | 3.60 / 3.61 tok/s, TTFT 115.2 / 115.5 s | 3.90 / 3.87 tok/s, TTFT 116.2 / 118.7 s | decode **+7.7%**, TTFT +1.8% | identical in all four arms | **installed** 2026-11-13 04:43 |
| `V41_PREFILL_SUFFIX=1 V4_PREWARM_BATCH=1 V4_WEX_SUBREADS=4` together, against the profile of 2026-11-12 ([log](../research/wex-rd-20261112/r4-r9combo.log)) | 657-token prompt, 75 tokens | 3.47 / 3.60 tok/s; ready 18 / 18 s; TTFT 113.6 / 114.1 s | 3.83 / 3.84 tok/s; ready 12 / 12 s; TTFT 109.9 / 109.1 s | decode **+8.4%**, TTFT -3.8% | identical | the three **installed** together (`V41_PREFILL_SUFFIX` is not a WEX switch) |
| `V4_WEX_SHA_ASYNC=1`, with `V4_WEX_SUBREADS=4` and `V41_GPU_MOE=2` on in both arms ([log](../research/wex-rd-20261112/r4-r10shadec.log)) | 2,329-token prompt, 94 tokens | 4.01 / 4.03 tok/s, TTFT 229.8 / 230.8 s | 4.05 / 4.03 tok/s, TTFT 232.7 / 231.9 s | decode +0.5%, TTFT +0.9% | identical | neutral, **not installed** |

A first try of the sub-reads and hash-pool decode A/Bs on 2026-11-12 (release rel5, `deepseek_v4.exe` fc78ad49, `--ram 16`, [rd_queue6.log](../research/wex-rd-20261112/rd_queue6.log)) lost arms to the test driver's RAM guard and stayed incomplete (sub-reads: 3.80 / 3.73 against one B arm at 4.08 tok/s); the runs above replace it. They also close the two combinations that had not been run: sub-reads with the batched prewarm, and the hash pool with sub-reads.

## DeepSeek V4.1 REAP-192: engine changes that are installed

| measurement (7 prompts x 64 tokens unless noted) | before | after |
|---|---|---|
| refit predictor gates, the predictor-gates engine (WNR3 U:+C:) | previous install, 1,886-token turn 2.83 tok/s | **3.06 tok/s**, logits identical; ABBA 14 pairs **+6.55% ± 1.81%** decode, bit-identical |
| low-RAM plan fix, `--memory-gb 13` at CTX 32768, 3 prompts x 64 tokens (WEX4 U:+C:) | dense weights streamed from disk every token: **0.19-0.27 tok/s**, TTFT 13-16.5 s. Same on the previously installed engine; 2.7 TB read in 39 min | dense resident, prefix-checkpoint slots 4 -> 3, GPU head: **2.82-3.31 tok/s**, TTFT 7.5-9.5 s (about 15x) |
| V4.1 decode attention on the GPU (`V41_ATTN_GPU=1`), ABBA, 3 prompts, RAM 13 | CPU attention: 2.75-3.28 tok/s | **+5.16% ± 0.69%**, but token ids differ on 2 of 3 prompts (device numerics). Quality check through the real decode path (forced tokens, 3 texts, 743 rows): NLL gap B-A -0.014 ± 0.018 nats, KL mean 0.012, top-1 agreement 96.2%: PASS. **Installed 2026-11-11 with `V41_ATTN_GPU=1`** (engine 7ac3baf4) |
| `COLI_V4_TIER_PREWARM_AHEAD=4`, 2026-11-12 | see "WEX start-up and read switches" | ready 26-28 -> 18 s, start to first token -6.5%, 3 fewer VRAM experts of about 964. **Installed in the profile** |
| `V4_PREWARM_BATCH=1` and `V4_WEX_SUBREADS=4`, 2026-11-13 | see "WEX start-up and read switches" | batched prewarm: ready 30-32 -> 12 s, start to first token -12.9%; sub-reads: decode +7.7%, text identical; with `V41_PREFILL_SUFFIX=1` too: decode +8.4%, ready 18 -> 12 s. **Installed** with the rel6 build and kept in the build of 15:56 |

## DeepSeek V4.1 REAP-192: low-RAM collapse, found and fixed on 2026-11-11

At CTX 32768 the V4.1 RAM plan needs about 13.03 GiB to keep the dense weights (6.65 GiB) in RAM. The budget breaks down as:
- fixed buffers, including 4 prefix-checkpoint slots of 210 MiB each;
- the dense weights;
- the minimum of 6 expert slots, 4.20 GiB.

Below that, dense was streamed from disk on every token, and decode fell to about 0.2 tok/s. The application host often passes 12-13 GB.

The fix makes the plan give up prefix-checkpoint slots before it streams dense:

| RAM budget | prefix-checkpoint slots | dense |
|---|---|---|
| 13.0 GiB | 3 | in RAM |
| 12.5 GiB | 1 | in RAM |
| 12.25 GiB | 0 | in RAM |
| below 12.25 GiB | unchanged | streamed |

When dense is still streamed, the engine prints a warning for the host. The GPU head (V4_GPU_HEAD with no host copy) frees another 1.23 GiB for experts.

## Qwen3.8-Flash-Next MXFP4

The first table and the A/Bs up to the CPU-side fixes were measured on the WEX1 form of the containers, before the in-place upgrade to WEX4 of 2026-11-11 23:18-23:24. After it, one run on U: + C: gave 17.04 / 10.77 tok/s on the 258-token and 2k prompts at temperature 0, with no A/B ([research/qwen38-speed-20261111/README.md](../research/qwen38-speed-20261111/README.md), section 3). The 2026-11-12 rows at the end were measured on WEX4.

| Qwen3.8 routed experts (24,576), RTX 5090 | FP8 (before) | MXFP4 WEX1 | MXFP4 WNR3 | **MXFP4 WEX1 + C: mirror** |
|---|---|---|---|---|
| container | snapshot shards | 64,224,956,416 B | 56,532,672,512 B (**0.880** of WEX1) | same as WEX1, twice |
| quality vs FP8 (teacher-forced, 9 prompts) | reference | gap +0.018 nats text, +0.013 chat: **PASS** | bit-identical to WEX1 | bit-identical to WEX1 |
| decode tok/s, 258-token prompt / 2k prompt / chat turns (one run each) | 4.39 / 3.07 / ~2.3 | 11.48 / 7.18 / ~5.3 | 9.48 / 5.27 / ~3.9 | **12.45 / 7.44 / ~6.0** |

Without a mirror WNR3 is slower than WEX1 on one drive, as with V4.1 before its mirror; MXFP4 itself is 2.3-2.6x faster than FP8.

**VRAM arena for the expert tier** (`QT_ARENA=1`, in the engine): an MXFP4 expert costs its real 2.51 MB in VRAM instead of a padded slot, so about 26% more experts stay resident (9,049 instead of 7,178 at the same charged budget, by the research's count). **Installed 2026-11-11 with `QT_ARENA=1`.** A/B, WEX1 + C: mirror, RAM 13, ABBA, identical texts:

| decode tok/s | A (`QT_ARENA=0`) | B (`QT_ARENA=1`) | gain |
|---|---|---|---|
| 258-token prompt | 12.93 / 13.26 | 15.52 / 14.49 | +14.6% |
| 2k-token prompt | 7.17 / 8.67 | 10.50 / 9.50 | +26.3% |
| 3-turn chat | 4.6-7.1 | 7.7-8.5 | +23% to +44% |

The same build with the upstream prefix store ported to qwen38 gave 14.09 / 10.41 / 7.5-8.4 tok/s in one control run. The store held up to 1089 MiB and reused 65-75% of later turns' prompts. The prefix store is installed with that build; it keeps the system prompt's state across turns.

**Two CPU-side fixes** (installed 2026-11-11 evening). Neither touches the containers.
- **Sampler.** At temperature above 0 `qwen38.c` sorted the whole vocabulary on every token: a full `qsort` of 248,320 entries, about 24 ms per token standalone. It now samples through `qwen36_sample.h`, which sorts only the nucleus.
- **One OpenMP team for expert reads.** A read wave used a team sized to its record count, which makes libgomp restart pool threads: a 2-record wave restarted 14 threads on the old code (check in `<TREE>/src/colibri/c/tests/test_qwen38_native_weights.c`). `q38_read_team()` in `qwen38_core.h` runs every read wave on the matmuls' team.

A/B on the real model, WEX1 + C: mirror, RAM 13, `QT_ARENA=1`, one pair with the new build run first, identical text in all 5 requests:

| decode tok/s, temperature 0.6 | A (prefix-store build) | B (sampler-fix build) | gain |
|---|---|---|---|
| 258-token prompt | 12.54 | 18.02 | +43.7% |
| 2k-token prompt | 7.80 | 10.85 | +39.1% |
| 3-turn chat | 5.90 / 6.30 / 6.35 | 8.83 / 8.85 / 8.94 | +40% to +50% |

Not repeated. The sampler part shows only at temperature above 0; the application sends 0.6.

**The container format is speed-neutral for Qwen3.8** (research of 2026-11-11, code, logs and CPU microbenches; no engine run). WEX4 around the same raw records costs 0%; WNR3 as the read format costs 19% to 29%. Decode is bound by the experts that are not resident in VRAM and are computed on the CPU, not by the drives: of 80.3 ms per token on the 258-token prompt, the disk is about 16 ms. So the Qwen3.8 containers keep raw records, and the levers are on the engine side.

**Misses sent to the GPU** (`QT_GPU_MISS=2`, installed 2026-11-12). The hits are launched on the GPU at routing; the misses go up as a second group, each record in one copy from page-locked RAM slots. A/B on WEX4 U:+C:, RAM 13, `QT_ARENA=1`, `CUDA_RESERVE_GB=2`, same build in both arms (release rel-q38dma), ABBA, one pair. Log: [rdq38dma.cmp.txt](../research/wex-rd-20261112/rdq38dma.cmp.txt).

| decode tok/s, A1 / A2 against B1 / B2 | A (`QT_GPU_MISS=0`) | B (`QT_GPU_MISS=2`) | B/A |
|---|---|---|---|
| 258-token prompt | 16.40 / 16.41 | 18.76 / 19.24 | +15.8% |
| 2k-token prompt | 11.22 / 10.02 | 14.21 / 13.35 | +29.7% |
| 3-turn chat | 8.99, 9.54, 9.26 / 8.84, 8.24, 8.14 | 11.31, 11.83, 12.06 / 11.27, 11.75, 11.71 | +26.6%, +32.6%, +36.5% |

Texts: the comparison flags the 258-token prompt as a mismatch; the other four requests are identical in all four arms. The run notes say B2 differs from B1 after character 468 of that prompt only (an order flip near a tie), and that a CPU fallback was taken for 0-60 of thousands of misses per request; neither note is verified here.

**Other Qwen3.8 profile changes of 2026-11-12 and 2026-11-13** (log [gpu_queue.log](../research/wex-rd-20261112/gpu_queue.log) unless named):
- `CUDA_RESERVE_GB` 0.8 -> 2: one pair; the 2k prompt's device wait was 40.8 (outlier) and 9.6 s with 0.8, 9.2 and 9.3 s with 2; the 258-token prompt's was 2.7 and 1.2 s against 1.2 and 1.2 s. No downside seen; adopted.
- `Q38_PREFIX_DISK=1`: the prefix store keeps its checkpoints on disk across engine runs. Check on the installed build: first request after a cold start TTFT 61.5 s, the same request after an engine restart 1.8 s, 2,164 of 2,184 prompt tokens from disk ([m2b-rel4-disk.log](../research/wex-rd-20261112/m2b-rel4-disk.log); one run). An earlier run of the check on the same build failed only because available RAM was too low for the store ([rd_queue3.log](../research/wex-rd-20261112/rd_queue3.log)).
- `Q38_KEEP_FREE_GB=0`: the engine keeps no RAM free for others; the host passes all available RAM.
- `Q38_ASYNC_MISS=1` (2026-11-13, with `QT_GPU_MISS=2`: a decode step's misses are read on the loader thread and each goes to the GPU as soon as its read lands; not a WEX switch). Two ABBA pairs on WEX4 U:+C:, RAM 13: decode B/A -0.6% to +3.6% on the five requests ([rd_queue3.log](../research/wex-rd-20261112/rd_queue3.log), M3), then +0.7% to +6.3% ([rd_queue6.log](../research/wex-rd-20261112/rd_queue6.log)); the comparison flags the 258-token prompt in both. Installed 2026-11-13 04:43.

**Containers verified and root pin, 2026-11-13** (logs in [research/wex-rd-20261112/](../research/wex-rd-20261112/README.md)). `wex4_verify` (8 threads) on both copies: every blob and record matches the index; U: 64.22 GB in 10.7 s, C: in 7.0 s; both print the root `db5389f6...c0df` recorded in [STATUS.md](../STATUS.md) ([log](../research/wex-rd-20261112/q38-wex4-verify.log)). Then an A/B of the pin with the installed build (rel7, `qwen38.exe` 9fb4da92), WEX4 U:+C:, RAM 13, `QT_ARENA=1`, `CUDA_RESERVE_GB=2`, ABBA, one pair ([r10q38root.cmp.txt](../research/wex-rd-20261112/r10q38root.cmp.txt), [r4-r10q38root.log](../research/wex-rd-20261112/r4-r10q38root.log)):

| decode tok/s, A1 / A2 against B1 / B2 | A (no root) | B (`COLI_WEX_ROOT` pinned) | B/A |
|---|---|---|---|
| 258-token prompt | 16.59 / 16.38 | 17.13 / 16.01 | +0.5% |
| 2k-token prompt | 11.83 / 11.92 | 10.88 / 11.14 | -7.3% |
| chat turn 1 | 8.88 / 8.85 | 8.24 / 7.88 | -9.1% |
| chat turn 2 | 9.26 / 9.01 | 8.72 / 8.67 | -4.8% |
| chat turn 3 | 8.96 / 8.72 | 8.84 / 8.69 | -0.9% |

Ready 15 s (A) against 12 s (B); TTFT the same. The comparison flags the 258-token prompt as differing, as in the other Qwen3.8 A/Bs of these days; the other four requests are identical. **What this does not measure:** arm B set only `COLI_WEX_ROOT`, which in the qwen38 engine is one check at open; the SHA-256 of each record at its first read needs `COLI_WEX_VERIFY=1`, and the log has no first-read line. The spread of the five requests is as wide as the effect. Decision: the root stays recorded and not pinned.

## Drives

Probe of 2026-11-12, `c/tools/wexio_probe.py` (in the engine since 2026-11-12): unbuffered overlapped reads at random 4 KiB-aligned offsets of the two V4.1 WEX4 copies, 2 GB per cell, with no engine running; one run. GB/s; p50, p99 and mean latency per request are in the log [m1-probe.log](../research/wex-rd-20261112/m1-probe.log).

| request size | drives | QD 1 | QD 2 | QD 4 | QD 8 | QD 16 |
|---|---|---|---|---|---|---|
| 16 MiB | C: | 8.25 | 9.38 | 10.26 | 10.34 | 10.15 |
| 16 MiB | U: | 5.70 | 6.42 | 6.34 | 6.10 | 5.71 |
| 16 MiB | C:+U: | **13.82** | 15.35 | 15.34 | 14.21 | 13.07 |
| 4 MiB | C: | 5.91 | 7.66 | 8.66 | 9.28 | 9.54 |
| 4 MiB | U: | 5.05 | 6.53 | 6.52 | 6.47 | 6.36 |
| 4 MiB | C:+U: | 10.75 | 14.48 | 15.08 | 14.90 | 15.21 |
| 2 MiB | C: | 4.14 | 7.46 | 8.27 | 8.52 | 9.36 |
| 2 MiB | U: | 3.74 | 6.43 | 6.55 | 6.52 | 6.46 |
| 2 MiB | C:+U: | 8.20 | 13.78 | 14.77 | 14.72 | 15.45 |

Both drives together reach 13.8 GB/s at QD1 with 16 MiB requests and 15.3-15.5 GB/s at QD2 and deeper; C: alone goes from 8.25 to 10.3 GB/s at QD4, U: alone from 5.7 to about 6.5. At 16 MiB and QD1 the latency is 2.0 ms on C: and 2.9 ms on U:. The 38/62 split matches the speed ratio of the two drives.

Earlier figures (2026-11-11, old README, raw output not here): C: (PCIe 5) 2.0 ms and 8.1 GB/s, U: (PCIe 4) 2.9 ms and 5.5 GB/s at QD1 with 16 MiB; with 4 reads in flight each read took 6.8-8.1 ms, so the 13 ms demand reads seen in engine logs were queueing behind prefetch reads, not slow drives. The probe above agrees.

## Earlier results

CPU-only baseline (M0, `DSV4_CUDA=0`: dense, attention and MoE on CPU), DeepSeek V4.1 Flash REAP-192 MXFP4, 16 GB RAM, warm (`docs/reports/v41-reap-baseline-m0-20261107.md`):

| Expert source, same drive (U:, PCIe 4 x4) | decode tok/s | TTFT (30 tok) |
|---|---|---|
| safetensors | 0.98-1.01 | 13-14 s |
| **WEX1** | **1.11-1.14** | **11-12 s** |

On the same drive WEX1 beats safetensors by about 13% in decode (12.8-13.7%). The V4.1 WNR2 container was 129.9 GB against 144.4 GB for WEX1 (-10%, lossless; `docs/reports/v41-reap-wnr2-build-20261107.md`). Earlier on DeepSeek V4 the gap was -12.8%; the V4 speed trade-off is in `docs/reports/wex-architecture-20261018.it.md` section 5.2 (see the contamination caveat in `docs/WNR2-REFERENCE.md`).

GPU tiers (W3, `research/v41-w3-20261109/results.md`, same drive U:, 16 GB, ABBA x2): WNR2 -1.80% ± 0.65% vs WEX1 (arm means 1.898 vs 1.860 tok/s), bit-identical logits. On 2026-11-09 the application switched to WNR2 with reads split over U: and C: (`COLI_WEX_WEIGHTS=38,62`), measured +49% against WEX1 on U: only on 2026-11-10 and replaced by WNR3 + mirror the same day ([STATUS.md](../STATUS.md), "In production").

