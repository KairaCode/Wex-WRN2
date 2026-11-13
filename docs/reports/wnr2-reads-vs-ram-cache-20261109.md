# WNR2 read volume vs RAM cache (W3, 2026-11-09)

> **Dated report (2026-11-09).** Its first "what to test" item was run with WNR3 records on 2026-11-10 (equal-RAM A/B -0.19% ± 0.72%, not adopted): superseded by [RESULT-20261110.md](../../research/wnr3-20261110/ram-record-cache/RESULT-20261110.md).

(Written 2026-11-09 ~04:40; the last two rows were added after W3 finished.)

W3 totals (KAIRA-local logs lab\v41-gpu-20261107\ab\runs\w3-*.stderr.log, not in this repo; last `v4_direct` and `v4_wex_gpu_direct` lines of each run):

| run | host reads | GPU-direct reads | total reads | GB read |
|---|---|---|---|---|
| A1 WEX1 | 335,033 | 0 | 335,033 | 6,299 |
| B1 WNR2 | 104,621 | 266,900 | 371,521 | 6,284 |
| B2 WNR2 | 105,002 | 268,181 | 373,183 | 6,313 |
| A2 WEX1 | 333,851 | 0 | 333,851 | 6,277 |
| A3 WEX1 | 333,116 | 0 | 333,116 | 6,263 |
| B3 WNR2 | 104,567 | 267,019 | 371,586 | 6,285 |
| B4 WNR2 | 104,148 | 264,596 | 368,744 | 6,237 |
| A4 WEX1 | 330,628 | 0 | 330,628 | 6,216 |

WNR2 reads the same bytes as WEX1, because it makes about 11% more reads at 16.91 MB instead of 18.80 MB.

Hypothesis:
- GPU-direct records (about 72% of WNR2 reads) do not pass through the host RAM tier, so a mirror evicted from VRAM is read from disk again. `ram_skips` (about 104k per run) counts the opposite case: GPU-direct candidates that RAM already held or was loading, which took the host path.
- So experts evicted from VRAM must be re-read from disk, while with WEX1 they are still in RAM.
- In the ab-samedrive and W3 setups the compressed WNR2 RAM record cache was turned off, so both arms had the same RAM.

What to test, in the replay and/or the short ABBA:
- **WNR2 + compressed RAM record cache.** Use the same total host-RAM GB as the WEX1 arm. The WNR2 arm then holds compressed records in RAM, about 11% more experts for the same GB. Find the knob: COLI_WEX_RAM_CACHE_GB (`v4_wex_crc`, deepseek_v4.c:14745 at cfb7ea0; it needs COLI_WEX_GPU_DIRECT>=1 and adds to --memory-gb), and keep the RAM accounting equal.
- **GPU-direct off (host path)** for WNR2: COLI_WEX_GPU_DIRECT=0, so every record goes through the host slab and the RAM tier. With 1, demand misses outside RAM still go disk -> VRAM; only the prefetcher's direct loads (mode 2) stop. COLI_WEX_RAM_CACHE_GB works only with COLI_WEX_GPU_DIRECT>=1.

Report total reads, GB read and ms/token for each variant.
