# WEX read and start-up switches, drives and Qwen3.8 integrity (2026-11-12 and 2026-11-13)

Copies of the KAIRA lab logs behind the 2026-11-12/13 rows of [STATUS.md](../../STATUS.md). The originals are in `<KAIRA>/lab/gpu-miss-20261112/`, which is not in this repository. The files were copied on 2026-11-13; the only change is that profile paths were shortened to `<KAIRA>` and `<HOME>` (rule 2 of [CONTRIBUTING.md](../../CONTRIBUTING.md)).

## Question

1. Which of the new WEX switches of the DeepSeek V4.1 engine pay on this machine: the VRAM tier prewarm reading ahead (`COLI_V4_TIER_PREWARM_AHEAD=4`) or in batches (`V4_PREWARM_BATCH=1`), the first-read SHA-256 on a hash pool (`V4_WEX_SHA_ASYNC=1`), and each drive's part of a record read as overlapped sub-reads (`V4_WEX_SUBREADS=4`)?
2. What do the two drives give alone and together, per request size and queue depth?
3. Are the two Qwen3.8 WEX4 containers intact, and what does pinning their root (`COLI_WEX_ROOT`) cost the qwen38 engine?

## Setup

- Machine: RTX 5090 (32 GB), two NVMe drives (U: PCIe 4, C: PCIe 5), Windows 11. No other engine or KAIRA instance ran during an arm (the drivers wait for that, see `ab.sh`).
- DeepSeek V4.1 REAP-192: KAIRA profile `v41gpu` (WEX4 with WNR3 records, U: plus a full mirror on C:, `COLI_WEX_WEIGHTS=38,62`, root pinned, `COLI_WEX_GPU_DIRECT=2`, `V4_WEX_BATCH=1`). Gateway launched as KAIRA launches it, greedy. "Ready" is the gateway's ready time; "start to first token" runs from the gateway start to the first token.
- Qwen3.8-Flash-Next MXFP4: WEX4 with RAW records on U: plus a full mirror on C:, `--ram 13`, `QT_ARENA=1 Q38_KEEP_FREE_GB=0 CUDA_RESERVE_GB=2`; five requests per arm (258-token prompt, 2k-token prompt, three chat turns), through `verify_gateway.py` (KAIRA lab, not here) driven by `ab.sh` / `ab_exe.sh`.
- Order: ABBA (A1 B1 B2 A2), one pair per switch, a cold start per arm.

| Build | KAIRA commit | Files | Used by |
|---|---|---|---|
| installed V4.1 engine of 2026-11-11 | not recorded | `deepseek_v4.exe` 7ac3baf4 | `m4-ahead.log` |
| rel3-wexprewarm, rel3-wexsha | d53f846, edcfc38 (branch builds) | not hashed in the logs | `m5-pwbatch.log`, `m6-shaasync.log` |
| rel4 | 05d25e3 | `qwen38.exe` b5962010, `coli_cuda.dll` f2dfa589 | `m2b-rel4-disk.log` |
| rel5-v41 | 7c1c45d (branch rd2-integ-20261112) | `deepseek_v4.exe` fc78ad49 | `rd_queue6.log` |
| rel6-v41 | 80dc959 (branch rd2-integ-20261112) | `deepseek_v4.exe` 6793ea73 | `r4-r9subreads.log`, `r4-r9combo.log`, `r4-r10shadec.log`, `rd_queue8.log` |
| rel7, installed 2026-11-13 07:16 | 13b260b | `qwen38.exe` 9fb4da92, `coli_cuda.dll` 5c631cf4 | `r4-r10q38root.log`, `r10q38root.cmp.txt` |
| `tools/wex4_verify` | cb225b3 | built from `c/tools/wex4_verify.c` | `q38-wex4-verify.log` |

The two Qwen3.8 runs of 2026-11-12 in `gpu_queue.log` and `rdq38dma.cmp.txt` (`CUDA_RESERVE_GB`, `QT_GPU_MISS=2`, disk checkpoints) are the source of the rows of that date in STATUS; they are not WEX switches and are not repeated below.

## Results

| Log | Model, prompt, tokens, RAM | A | B | B/A | Text | Verdict |
|---|---|---|---|---|---|---|
| `m4-ahead.log` | V4.1, 647-token prompt, 8 tokens, `--ram 13` | profile without `COLI_V4_TIER_PREWARM_AHEAD`: ready 28, 26 s; start to first token 140.9, 138.9 s | `=4`: ready 18, 18 s; 129.6, 132.0 s | start to first token -6.5% | not compared | installed in the profile on 2026-11-12 16:15 |
| `m5-pwbatch.log` | same | `V4_PREWARM_BATCH=0`: ready 32, 30.1 s; 145.9, 143.5 s | `=1`: ready 12, 12 s; 126.1, 125.9 s | -12.9% | not compared | winner; installed 2026-11-13 04:43 with rel6 |
| `m6-shaasync.log` | same | `V4_WEX_SHA_ASYNC=0`: ready 18, 20 s; 131.4, 135.1 s | `=1`: ready 16, 16 s; 131.6, 134.4 s | -0.2% | not compared | neutral at start-up |
| `rd_queue6.log` (r4subreads) | V4.1 rel5, 2,329-token prompt, 98 tokens, `--ram 16` | 3.80, 3.73 tok/s | B1 killed by the driver's RAM guard, B2 4.08 tok/s | not computed | not identical (B1 has no text) | incomplete, rerun below |
| `rd_queue6.log` (r4shadec) | same | 3.68 tok/s, A2 killed | 3.60 tok/s, B2 killed | not computed | A1 = B1 | incomplete, rerun below |
| `r4-r9subreads.log` | V4.1 rel6, 657-token prompt, 75 tokens, `--ram 15` | `V4_WEX_SUBREADS` unset: 3.60, 3.61 tok/s; TTFT 115.2, 115.5 s | `=4`: 3.90, 3.87 tok/s; TTFT 116.2, 118.7 s | decode +7.7%, TTFT +1.8% | identical in all four arms | winner; installed 2026-11-13 04:43 |
| `r4-r9combo.log` | same | profile of 2026-11-12: 3.47, 3.60 tok/s; ready 18, 18 s; TTFT 113.6, 114.1 s | `V41_PREFILL_SUFFIX=1 V4_PREWARM_BATCH=1 V4_WEX_SUBREADS=4`: 3.83, 3.84 tok/s; ready 12, 12 s; TTFT 109.9, 109.1 s | decode +8.4%, TTFT -3.8% | identical | the three switches installed together on 2026-11-13 04:43 (`V41_PREFILL_SUFFIX` is not a WEX switch) |
| `r4-r10shadec.log` | V4.1 rel6, 2,329-token prompt, 94 tokens, `--ram 15`; the arms also had the profile's `V4_WEX_SUBREADS=4` and `V41_GPU_MOE=2` | `V4_WEX_SHA_ASYNC` unset: 4.01, 4.03 tok/s; TTFT 229.8, 230.8 s | `=1`: 4.05, 4.03 tok/s; TTFT 232.7, 231.9 s | decode +0.5%, TTFT +0.9% | identical | neutral, not installed |
| `q38-wex4-verify.log` | Qwen3.8 containers, read-only | U: copy: every blob and record matches the index, 64.22 GB in 10.7 s (8 threads) | C: copy: the same in 7.0 s | | | both print root `db5389f699abb49a164a9e73d925b5b629b0ce6de430a553388ba3e1cae5c0df` |
| `r4-r10q38root.log`, `r10q38root.cmp.txt` | Qwen3.8 rel7, five requests, `--ram 13` | `COLI_WEX_ROOT` unset | `COLI_WEX_ROOT=db5389f6...c0df` | decode +0.5, -7.3, -9.1, -4.8, -0.9% per request (median -4.8%); TTFT the same; ready 15 -> 12 s | 258-token prompt flagged as differing (the four others identical) | the root stays not pinned |

`m1-probe.log` (2026-11-12, `tools/wexio_probe.py`, no engine running, one run): unbuffered overlapped reads at random 4 KiB-aligned offsets of the two V4.1 WEX4 copies, 2 GB per cell. Both drives together reach 13.82 GB/s at QD1 with 16 MiB requests and 15.3-15.5 GB/s at QD2 and deeper; C: alone 8.25 -> 10.3 GB/s (QD4), U: alone 5.7 -> about 6.5 GB/s. The whole table is in the log and in the section "Drives" of the measurement tables.

## Limits

- One ABBA pair per switch: no interval is given.
- **The Qwen3.8 root-pin run did not measure the SHA-256 cost.** Arm B set only `COLI_WEX_ROOT`. In the qwen38 engine a pinned root is one check at open; the SHA-256 of each record at its first read needs `COLI_WEX_VERIFY=1`, and the log has no "first-read SHA-256" line. The -4.8% median is therefore not the cost of the per-record hash, which is still unmeasured on Qwen3.8; the spread of the five requests (+0.5% to -9.1%) is as wide as the effect.
- The start-up A/Bs compare start-to-first-token of an 8-token answer; the prompt's own TTFT (111.6-118.4 s) barely moves, so the gains are in the start-up.
- `rd_queue8.log` now holds the rerun of 2026-11-13 03:37 after a reboot: an A/B of `V41_GPU_MOE=2` (not a WEX switch) and the header of a sub-reads arm with nothing after it. The sub-reads A/B was then run as `r4-r9subreads.log`. The earlier content of the file (2026-11-12 evening, every arm refused for low available RAM) was overwritten in the lab and is not here.
- `wex4.py verify` (which also decodes every record) was not run on the real Qwen3.8 files; `wex4_verify` hashes only.
- The `rd_queue*.log` files also hold A/Bs of switches that do not touch WEX (`Q38_ASYNC_MISS`, `V41_GPU_MOE`, `V4_NP_*`); they are kept whole and not discussed here.

## Raw files

| File | What it is |
|---|---|
| `m1-probe.log` | drive probe table (GB/s, p50, p99, mean latency) |
| `m2b-rel4-disk.log` | Qwen3.8 disk checkpoints across an engine restart (PASS) |
| `m4-ahead.log`, `m5-pwbatch.log`, `m6-shaasync.log` | the three V4.1 start-up A/Bs, per-arm engine lines |
| `rd_queue3.log` | the queue of 2026-11-12 afternoon that ran M1 to M6 (with the first, failed, disk check M2 and the qwen38 async-miss A/B M3) |
| `rd_queue6.log` | the queue of 2026-11-12 evening: the incomplete sub-reads and SHA-async decode A/Bs, and other A/Bs |
| `rd_queue8.log` | see Limits |
| `r4-r9subreads.log`, `r4-r9combo.log`, `r4-r10shadec.log` | the V4.1 decode A/Bs of 2026-11-13, per-arm engine lines and verdicts |
| `q38-wex4-verify.log` | `wex4_verify` on both Qwen3.8 copies |
| `r4-r10q38root.log`, `r10q38root.cmp.txt` | the Qwen3.8 root-pin A/B: per-arm records and the comparison table |
| `gpu_queue.log`, `rdq38dma.cmp.txt` | the Qwen3.8 queue of 2026-11-12 morning and the `QT_GPU_MISS=2` comparison |
| `ab.sh`, `ab_exe.sh` | the Qwen3.8 A/B drivers (same exe in both arms; one exe per arm) |
