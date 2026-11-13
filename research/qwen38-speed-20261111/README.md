# Qwen3.8 MXFP4 speed work and the WEX4 move (2026-11-11)

Raw results and scripts behind the "After the install" section of [docs/QWEN38-MXFP4-20261111.md](../../docs/reports/QWEN38-MXFP4-20261111.md). Machine: RTX 5090 (Windows 11, WDDM), 32 GB RAM, engine budget RAM 13, experts in WEX containers on U: with a full mirror on C: (read split 38/62), `QT_ARENA=1`. Paths in the files are shortened to `<KAIRA>` (the KAIRA checkout), `<HOME>` and `<TMP>`.

## 1. Where the decode time goes

A read-only study of the installed engine (code, existing logs, CPU microbenchmarks; no engine run) found that qwen38 decode is not bound by the drives or by the container format:

- A decode step on the 258-token prompt was 80.3 ms: 57.0 ms in the host-side routed-expert block, 6.5 ms waiting for the GPU, 16.8 ms outside any timer. Disk reads were about 16 ms of it. GPU utilisation in decode was 23%.
- An expert that is not resident in VRAM is read into RAM and computed on the CPU; qwen38 has no path that computes a miss on the GPU.
- The container format is speed-neutral: WEX4 measured +0.3% on the other model (noise), and WNR3 as the read format costs 17-30% on qwen38 (GPU decode per 2.6 MB record against 12% fewer bytes read).

Two CPU-side defects of the KAIRA copy of the engine came out of that study and were fixed in KAIRA commit 99ccadc (in master since e245f4c):

- **Sampler.** At temperature > 0 `serve_sample` sorted the whole 248,320-entry vocabulary with `qsort` on every token: 24 ms a token measured standalone. `qwen38.c` now includes `qwen36_sample.h` (nucleus only).
- **Read team.** A decode wave of two or more expert reads ran on an OpenMP team sized to its record count; libgomp then ends the other pool threads and starts new ones at the next matmul. Every read wave now runs on the matmuls' team. A test in `tests/test_qwen38_native_weights.c` fails on the old sizing with "a read wave of 2 records restarted 14 OpenMP threads".

Both already existed upstream in another form (colibri b11974c2 and #2593); the KAIRA copy had fallen behind.

## 2. A/B at temperature 0.6

Every earlier benchmark ran at temperature 0, where the sampler cost does not show; the KAIRA host sends 0.6 (top_p 0.9 by the gateway's default).

- A = the build installed until then (`qwen38.exe` cb15298e, KAIRA opt-q38-prefix d5e319e). B = KAIRA master e245f4c (`qwen38.exe` 1ac9fee2), built from a fresh `git archive` with `make CUDA_DLL=1 qwen38.exe`; same two DLLs in both arms. A disassembly comparison found 1,432 of 1,440 functions identical; the eight that differ are the sampler and the expert read path.
- One pair, B first and then A (the first run of a session is the slow one, so the order cannot flatter B). `COLI_LOGIT_DUMP=1` in both arms. VRAM in use by other programs before each arm: 963 and 953 MiB. 9,049 experts resident in VRAM in both.
- Harness: `verify_gateway.py` with one change, the request temperature from `VERIFY_TEMP` ([verify_gateway_temp.diff](verify_gateway_temp.diff)); driver [q38_ab_temp.sh](q38_ab_temp.sh), wait-for-a-free-machine wrapper [wait_then_run.sh](wait_then_run.sh), report [q38_ab_compare.py](q38_ab_compare.py).

| request | tokens | TTFT s, A / B | decode tok/s, A | decode tok/s, B | B/A | text |
|---|---|---|---|---|---|---|
| 258-token prompt | 200 | 10.2 / 10.5 | 12.54 | 18.02 | +43.7% | identical |
| 2k-token prompt | 30 | 49.5 / 47.0 | 7.80 | 10.85 | +39.1% | identical |
| chat turn 1 | 8 | 2.3 / 2.1 | 5.90 | 8.83 | +49.6% | identical |
| chat turn 2 | 8 | 1.7 / 1.5 | 6.30 | 8.85 | +40.5% | identical |
| chat turn 3 | 14 | 1.7 / 1.6 | 6.35 | 8.94 | +40.9% | identical |

Raw records: [ab-temp06-A-installed.json](ab-temp06-A-installed.json), [ab-temp06-B-e245f4c.json](ab-temp06-B-e245f4c.json).

Reading and limits:
- The engine never seeds `rand()`, so the two builds draw the same sequence: identical text in all five requests means the new sampler picked the same tokens as the full sort on the real model.
- The pair was run once. The chat turns generate 8-14 tokens and are noisy; the 200-token request is the one to read.
- The gain is larger than the removed sort alone predicts (about +30%); the two fixes were not measured apart.

B was installed in KAIRA after this run.

## 3. The real containers moved to WEX4

Both copies of `q38_experts.wex` (WEX1, 64,224,956,416 B) were upgraded in place with `wex4_convert.py upgrade FILE --names qwen38`, the C: mirror first, under the lab's lock, by [wex4_move.py](wex4_move.py) (log: [wex4_move.log](wex4_move.log)). The script downgrades whatever it upgraded if any step fails; no step failed.

| step | result |
|---|---|
| upgrade C: (mirror) | 42 s; 64,251,047,936 B (+26,091,520 B, index 26,086,592 B, 24,576 records, codec 0 = raw) |
| engine on the mirror alone | "WEX4 container" in the log, 5 of 5 requests answered ([wex4-mirror-only.json](wex4-mirror-only.json)) |
| upgrade U: (primary) | 39 s; same size, same root `db5389f699abb49a164a9e73d925b5b629b0ce6de430a553388ba3e1cae5c0df` |
| engine on U: + C: | "WEX4 container" and "WEX mirror active", 5 of 5 answered, 17.04 / 10.77 tok/s on the 258-token and 2k prompts at temperature 0 ([wex4-both-drives.json](wex4-both-drives.json)) |

Limits: the two engine runs are one run each and are not an A/B; `wex4.py verify` (every record hashed and decoded) was not run; the root is not pinned in the profile (`COLI_WEX_ROOT`); the qwen38 engine checks header, index and root at open and has no per-record first-read SHA-256. An engine without the WEX4 loader no longer opens these files; `wex4_convert.py downgrade` restores the WEX1 form.

## 4. Open levers, by expected gain per effort (estimates, not measured)

1. Compute VRAM misses on the GPU (pinned ring, transient slot, grouped kernel): about a week of work.
2. One OpenMP region per layer for the experts that stay on the CPU: bit-identical in a standalone bench.
3. VRAM tier policy: warm-start order from the shipped usage profile, first-use admission when the victim was never used.
4. `Q38_SERVE_CTX_TARGET` sized to the context in use (a hard cap on the conversation length).
5. Prefill: one tier call per group of rows instead of one per row.
