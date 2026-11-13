# Note 4 from the main session (2026-11-09 05:30): W3 final, so compare the mirror against WEX1 too

W3 final (lab\v41-gpu-20261107\ab\results.md): on the integrated build at 16 GB, WNR2 is -1.80% +/- 0.65% vs WEX1, with identical tokens. Under the user's rule WEX1 stays the container unless WNR2 is not slower.

The U:+C: mirror now exists ONLY for WNR2: C:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex, sha-verified. WEX1 cannot have a C: mirror, because C: has only 34.5 GB free.

The comparison that decides production is therefore:
- **A = WEX1 on U: only** (today's best WEX1 setup);
- **M = WNR2 on U: + C: mirror** (COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap, COLI_WEX_WEIGHTS=38,62; COLI_DISK_WEIGHTS does not split a container-only mirror, and without COLI_WEX_WEIGHTS every read goes to U:. Check for `[v4_wex] read split over 2 drives: 38% 62%`).
- Optional: **M+R = M plus the compressed WNR2 RAM record cache** (Note 1), with the total host RAM kept equal.

Use the short ABBA (A, M, M, A) with the same ws-ab build and --memory-gb rule, and identical-token check.
- Expert wait is 58% of the token today (312 of 534 ms), so a 2.6x faster disk path could matter a lot. Measure it; do not assume it.
- Report NVMe wait, per-drive read share, U: temperature (lab\v41-gpu-20261107\ab\nvme_temp.py) and any engram slowdown: shard 47 is now on D:.
