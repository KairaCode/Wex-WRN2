# Note 3 from the main session (2026-11-09): WNR2 U:+C: mirror

The user approved going straight to the U:+C: mirror. A main-session script (lab\v41-gpu-20261107\mirror_uc_setup.ps1, under heavy.lock) does three things:
1. deletes the duplicate WEX1 on D:;
2. moves engram shard 47 from C: to D:, leaving a junction at C:\models\DeepSeek-V4.1-Flash-1017-engram, so COLI_MODEL_DIRS is unchanged;
3. copies WNR2 to C:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex, sha256-verified.

After it finishes (see moves.log "WNR2 verified on C:"), validate also this arm with the short ABBA:
- B = WNR2 on U: only (COLI_WEX_PATH=U:\...\v4_flash_experts.nr2.wex);
- M = WNR2 with the C: mirror. Same COLI_WEX_PATH, plus COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap and COLI_WEX_WEIGHTS=38,62 (U: 6.58 vs C: 10.66 GB/s, primary first). COLI_DISK_WEIGHTS does not split a container-only mirror: without COLI_WEX_WEIGHTS every read goes to U:. Check the engine log for the mirror-registered line and for `[v4_wex] read split over 2 drives: 38% 62%`.
- Order B, M, M, B. Report tok/s, TTFT, NVMe wait ms/token and per-drive read share. Check identical tokens.
- Engram shard 47 now sits on D: (QD1 latency 12 ms vs 2.3 ms on C:). Watch for an engram-related slowdown: compare TTFT and the attention/engram part of PROF with the earlier W3 runs.
