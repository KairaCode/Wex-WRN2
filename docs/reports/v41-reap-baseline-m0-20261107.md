# M0: V4.1 REAP-192 baseline and drive probe (2026-11-07)

> **Dated report (2026-11-07).** The WEX1 containers it measures were deleted (the D: copy on 2026-11-09, the U: copy on 2026-11-10); its numbers remain the CPU-only baseline. Superseded for what runs today by [STATUS.md](../../STATUS.md).

Package M0 of `PLAN.md` (protocol 3.1), measurement only: no source was edited. Tags as in the plan: **[M]** measured here, **[I]** inferred.

## Setup

- Code: the engine's main branch of 2026-11-07, `src\` clean. `deepseek_v4.exe` built in workspace `ws-m0` (`make -j8 deepseek-v4`), sha256 `e215651b9fc77dbe869cd023e7b865bb4312d02e1b8cfac2da92e7146689d943`. Build log `lab\v41-port-20261106\ws-m0\m0-build.log`.
- heavy.lock held for the whole package (taken 01:33:14).
- Tools written for this package (in `m0\`):
  - `drive_probe.py`: unbuffered (`FILE_FLAG_NO_BUFFERING`) read probe, one handle per thread, 4 KiB aligned offsets uniform over the file.
  - `serve_bench.py`: one `SERVE=1` engine process, prompts in sequence (warm cache as in real use); per prompt it records TTFT (client clock, SUBMIT to first DATA), the `DONE` line, the `PROF` line and `TIERS`.
  - `gen_prompts.py`: writes `ab\prompts.jsonl`.
  - Self-test of `serve_bench.py` on the tiny V4.1 fixture: `m0\runs\tiny-selftest.*` (DONE/PROF/TIERS parsed for 2 prompts).

## 1. Drive probe [M]

`m0\drive_probe.py`, 8 s per test, GB = 1e9 B. Logs: `m0\drive_probe.log`, `m0\drive_probe_D_repeat.log` (D: repeated, 6 s per test: same numbers within 3%).

| Drive | Device | File probed | seq 8 thr x 8 MiB | rand 18.8 MB QD8 | rand 18.8 MB QD1 | rand 16.4 MB QD8 | QD1 p50 latency |
|---|---|---|---|---|---|---|---|
| D: | Sabrent SB-2130-1TB (NVMe, disk 4), 57 GB free | WEX1 `D:\kaira_wex\v41-reap\v4_flash_experts.wex` | 1.58 | **1.61** | 1.50 | 1.58 | 12.2 ms |
| U: | SK hynix HFS001TEJ9X162N (NVMe, disk 5), 258 GB free | `model-00017-of-00048.safetensors` (3.8 GB) | 6.53 (only 0.5 s, file end) | **6.58** | 5.71 | 6.58 | 3.2 ms |
| C: | Crucial T700 CT1000T700SSD3 (PCIe 5 NVMe, disk 3), 96 GB free | engram `model-00047-of-00048.safetensors` | 10.15 | **10.66** | 8.03 | 10.70 | 2.3 ms |
| E: | ADATA SX8200PNP (NVMe, disk 6), 135 GB free | engram `model-00048-of-00048.safetensors` | 2.31 | **2.29** | 2.25 | 2.29 | 8.3 ms |

Every drive letter is its own physical disk (Get-Partition / Get-PhysicalDisk).

Reading:
- **D:, where the WEX1 container lives, is the slowest drive: ~1.6 GB/s, ~85 records of 18.8 MB per second, flat from QD1 to QD8.** The QD1 latency is a constant 12.1-12.3 ms, so the drive is bandwidth-capped, not queue-starved.
- **Cause [M]: the Sabrent's link trained at PCIe 3.0 x2** (PnP `CurrentLinkWidth 2 / MaxLinkWidth 4`, `CurrentLinkSpeed 3 / MaxLinkSpeed 4`, PCI bus 6 = disk 4 = D:). 3.0 x2 is ~1.97 GB/s raw, so 1.6 GB/s is the link, not the flash. The other links: C: bus 2 gen5 x4, U: bus 9 gen4 x4, E: bus 12 gen3 x4 (its maximum). An M.2 slot that shares lanes (chipset/SATA) or a BIOS lane setting would do this [I]; moving the drive or changing the setting is a hardware decision for the user and could make D: ~3-4x faster.
- U: (the safetensors) is 4.1x faster than D: on the engine pattern. C: is 6.6x faster but has only 96 GB free (the WEX1 is 144 GB). E: (the planned WNR2 target, also engram shard 48) is 1.4x D:.
- The QD1 column is the demand path with one loader in flight; QD8 is the engine's 8 lanes.

## 2. REAP CPU baseline (R0a safetensors, R0b WEX1): NOT RUN, machine not quiet

- **Blocked on the precondition** "free RAM >= 18 GB before a REAP run" (and the quiet-machine rule of 3.1).
  - At 01:33: 3.6 GB free / 8.9 GB available (incl. standby) of 31.1 GB. Other applications, not part of this work, were running: two of their processes held ~16 GB private commit each (one at ~3.4 cores of CPU for the whole window), and the GPU sat at 94-99% utilization with 19-24 GB VRAM used (no application, no colibri engine, no other build of this work).
  - Polled every 60 s from 01:39 to 01:58 (`m0\ram_poll.log`): available RAM stayed between 4.0 and 8.8 GB. One of those processes exited and a new one started with the same 16 GB commit.
  - I did not stop any of those processes (not mine), and a run at a lower `--memory-gb` would not be the M0 baseline. heavy.lock released at 01:59:54.
- **Ready to run** once the machine is quiet: `sh lab\v41-gpu-20261107\m0\run_m0.sh [MAX_TOKENS=64] [IDS=w0,code1,chat_it1]` (take heavy.lock first).
  - Same binary (sha above), `DSV4_CUDA=0 COLI_V4_SAVE_USAGE=0 RAM_GB=16 CTX=4096 V4_SKIP_COLD=0 V4_MTP=0`, engram dirs via `COLI_MODEL_DIRS`; R0b adds `COLI_WEX_PATH=D:\kaira_wex\v41-reap\v4_flash_experts.wex`.
  - One serve process per arm: the RAM tier is pre-read before READY (#2109), then the 16-token warm-up, then the prompts, so the numbers are warm.
  - Per prompt it writes `runs\r0*.jsonl`: TTFT, `DONE` (decode tok/s from the first token, turn hit rate %, prompt tokens), `PROF` (wall, disk s summed over lanes, wait s, expert matmul s, attention+dense s, head s, forwards), and the stderr log with `[v4] expert cache ... GiB/s`, `[v4_wex] Container ... active`, and `v4_direct` at exit.
  - Known limits of the existing counters for V4.1 (no source edit in M0): `decprof` never prints for V4.1 (the plan's M0 code item, lane B); serve gives the turn hit rate, not absolute misses, so misses/token = (1 - hit) x 240 is exact only for decode (prefill adds its union requests); `TIERS` under the hot policy reports resident counts, not bytes. Expected wall: R0a ~4 min, R0b ~7 min at 64 tokens [I, from the smoke rate and the D: probe].

## 3. What the probe already says [I unless tagged]

Disk floor of one decode token = misses/token x 18.8 MB / drive GB/s. Using the plan's ~214 misses/token [I] (~4.0 GB/token):

| Expert source | GB/s (QD8 / QD1) [M] | disk floor per token |
|---|---|---|
| safetensors on U: (R0a) | 6.58 / 5.71 | 0.61-0.70 s |
| WEX1 on D: (R0b) | 1.61 / 1.50 | **2.5-2.7 s** |
| WEX1 if it lived on U: (D1) | 6.58 | 0.61 s |
| WNR2 on E: (W1 target, records ~16.4 MB, ~12.8% fewer bytes) | 2.29 | ~1.5 s |

1. **R0b (WEX1 on D:) will be slower than R0a, not faster**: D: is the drive, not the format. The smoke's 1.44 s/token on U: is about 2x the U: disk floor, so ~0.7 s/token of it is probably CPU (dense FP8 + FP4 experts + attention) or I/O not overlapped with compute. That supports G1/G2 also on the CPU side; R0a's PROF split will settle it.
2. **D1 matters more than the plan assumed, and in a different form.** The WEX1 should not stay on D: as the only copy. Options for the user (each needs their OK): copy the WEX1 to U: (258 GB free, 144 GB needed; 4.1x D:), or fix D:'s link (3.0 x2 -> 4.0 x4). D:+U: striping through `COLI_WEX_EXTRA_MIRRORS` would add only ~1.6 GB/s to U:'s 6.6 [I]. C: (10.7 GB/s) would be the fastest but has only 96 GB free.
3. **W3 (3.2) drive confound is already triggered:** E: (2.29) vs D: (1.61) differ by 42% > 10%, so the WEX1-vs-WNR2 A/B needs the same-drive control (the plan proposes U:; that needs the user's OK for the writes). Comparing WEX1-on-D: with WNR2-on-E: would measure the drives.
4. W1 writes 126 GB to E: and reads D: at 1.6 GB/s: ~15 min of reading the source at minimum, which matches the plan's 13-16 min estimate [I].

## 4. Prompt set for later stages

`lab\v41-gpu-20261107\ab\prompts.jsonl`, written by `m0\gen_prompts.py`: 1 warm-up line (`"warmup": true`, 16 tokens, discarded) plus the 8 prompts of 3.1 at 256 tokens: `code1`, `code2` (code), `chat_it1`, `chat_it2` (Italian chat), `reason1`, `reason2` (reasoning), `ctx2k` (a fixed 7,573-char excerpt of `serve_codec.h` at the engine of 2026-11-07 embedded verbatim, ~2k tokens [I], plus a summarization question), `short1` (short). Fields: `id, category, warmup, max_tokens, user`. No system prompt; `serve_bench.py` applies the V4 chat template (`<bos><User>...<Assistant></think>`, the CLI's non-thinking default).

## 5. Not done

- R0a and R0b (all REAP numbers: decode s/token, TTFT, misses/token, bytes/token, hit rate, disk vs CPU split). Reason above.
- The plan's M0 code item (V4.1 `decprof`), which belongs to lane B; this package was measurement only.
- `deepseek-v4-tiny-check`: no source changed, so nothing to regress.

## 6. REAP CPU baseline, measured 03:10-03:22 after the other workload ended (run_m0.sh + R0c)

Same binary as section 1 (ws-m0, sha e215651b...), DSV4_CUDA=0, RAM_GB=16, CTX=4096, lossless, prompts w0 (warm-up, 16 tok), code1 and chat_it1 (64 tok each), one serve process per arm. Raw: m0\runs\r0*.jsonl / .stderr.log, m0\run_m0.out.log. STAT fields: tokens, decode tok/s, turn hit rate %.

| Arm | Expert source | decode tok/s (code1 / chat_it1) | TTFT s (29-30 prompt tok) | hit rate % |
|---|---|---|---|---|
| R0a | safetensors on U: | 0.976 / 1.008 | 14.3 / 12.8 | 34.6 / 37.1 |
| R0b | WEX1 on D: (PCIe 3.0 x2) | 0.468 / 0.480 | 34.1 / 29.4 | 34.6 / 37.1 |
| R0c | WEX1 on U: (copy, sha verified) | **1.110 / 1.137** | **11.8 / 10.7** | 34.9 / 37.1 |

- WEX1 on U: is +13-14% decode vs safetensors on U: and 2.4x vs WEX1 on D:. D: is drive-bound, as the probe predicted.
- Use the U: WEX1 (`U:\kaira_wex\v41-reap\v4_flash_experts.wex`) as the CPU baseline for every later comparison (~1.12 tok/s decode, ~11 s TTFT on ~30-token prompts).
