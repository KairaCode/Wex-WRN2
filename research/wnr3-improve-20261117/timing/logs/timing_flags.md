
## 1 thread per record: 9 records (00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120), 9 rounds, 3 timed passes per run (median pass)
load average: ['1.45', '1.99', '2.11'] at start, ['1.00', '1.23', '1.69'] at end; per round: ['1.19', '1.09', '1.04', '1.02', '1.01', '1.00', '1.00', '1.00', '1.00']
| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |
|---|---|---:|---:|---:|---:|---:|
| A: wex_fast_decode unchanged, k1 x4 fused | orig_a2:w3:1,4,1 | 103.86 (89.68..116.58) | 181 (161..210) | 1 | 103.20 | 1 |
| wex_fast_decode unchanged k1x4, -march=native | orig_nat:w3:1,4,1 | 96.57 (90.60..106.30) | 195 (177..208) | 0.98x (0.91..1.13) | 96.52 | 0.98x (0.91..1.13) |
| fastdec kernel 4 x3, x86-64-v4+vbmi2 (main table) | new_a5:w3:4,3,1 | 45.49 (40.79..50.16) | 413 (375..461) | 2.19x (2.10..2.61) | 45.22 | 2.19x (2.09..2.61) |
| fastdec kernel 4 x3, haswell+AVX-512 flags | new_h5:w3:4,3,1 | 49.37 (45.75..52.20) | 381 (360..411) | 2.08x (2.03..2.29) | 49.22 | 2.09x (2.02..2.28) |
| fastdec kernel 4 x3, -march=native | new_nat:w3:4,3,1 | 48.65 (39.64..53.46) | 386 (352..474) | 2.16x (2.05..2.45) | 48.51 | 2.15x (2.04..2.45) |
| WNX MAG AVX-512 G=2, x86-64-v4+vbmi2 (main table) | orig_a5:wnx:mag_2048:2:2 | 22.94 (20.37..24.64) | 819 (763..923) | 4.50x (4.00..5.14) | 22.90 | 4.54x (3.99..5.10) |
| WNX MAG AVX-512 G=2, fastfmt flags (haswell+f/bw/vl/dq) | ffold_a5:wnx:mag_2048:2:2 | 21.69 (17.50..25.92) | 867 (725..1074) | 4.98x (3.87..6.15) | 21.51 | 4.97x (3.89..6.12) |
| WNX MAG AVX-512 G=2, haswell+AVX-512 flags incl. cd/vbmi2 | orig_h5:wnx:mag_2048:2:2 | 18.24 (17.40..21.48) | 1031 (875..1081) | 5.33x (4.99..6.31) | 18.24 | 5.32x (4.94..6.28) |
| WNX MAG AVX-512 G=2, -march=native | orig_nat:wnx:mag_2048:2:2 | 19.30 (17.59..23.85) | 974 (788..1069) | 5.02x (4.44..5.94) | 19.25 | 5.03x (4.45..5.96) |

## 4 threads per record (OpenMP default wait policy): 9 records (00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120), 9 rounds, 3 timed passes per run (median pass)
load average: ['1.00', '1.23', '1.69'] at start, ['3.48', '2.12', '1.96'] at end; per round: ['1.75', '2.09', '2.52', '2.84', '3.02', '3.24', '3.14', '3.33', '3.48']
| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |
|---|---|---:|---:|---:|---:|---:|
| A: wex_fast_decode unchanged, k1 x4 fused | orig_a2:w3:1,4,1 | 26.00 (23.33..29.22) | 723 (643..806) | 1 | 102.92 | 1 |
| wex_fast_decode unchanged k1x4, -march=native | orig_nat:w3:1,4,1 | 25.38 (24.72..28.41) | 741 (662..761) | 1.00x (0.92..1.08) | 101.05 | 1.01x (0.93..1.09) |
| fastdec kernel 4 x3, x86-64-v4+vbmi2 (main table) | new_a5:w3:4,3,1 | 11.47 (11.29..13.35) | 1639 (1408..1666) | 2.19x (1.95..2.39) | 45.95 | 2.18x (1.98..2.33) |
| fastdec kernel 4 x3, haswell+AVX-512 flags | new_h5:w3:4,3,1 | 12.57 (10.47..13.39) | 1495 (1404..1796) | 2.13x (2.03..2.45) | 49.13 | 2.10x (2.03..2.40) |
| fastdec kernel 4 x3, -march=native | new_nat:w3:4,3,1 | 12.45 (11.95..13.38) | 1510 (1405..1574) | 2.11x (1.97..2.18) | 49.08 | 2.12x (1.97..2.20) |
| WNX MAG AVX-512 G=2, x86-64-v4+vbmi2 (main table) | orig_a5:wnx:mag_2048:2:2 | 5.64 (5.03..6.27) | 3332 (2998..3735) | 4.63x (3.83..5.20) | 22.77 | 4.55x (3.93..5.31) |
| WNX MAG AVX-512 G=2, fastfmt flags (haswell+f/bw/vl/dq) | ffold_a5:wnx:mag_2048:2:2 | 5.41 (5.01..6.56) | 3477 (2866..3750) | 4.93x (4.16..5.17) | 21.41 | 4.90x (4.15..5.31) |
| WNX MAG AVX-512 G=2, haswell+AVX-512 flags incl. cd/vbmi2 | orig_h5:wnx:mag_2048:2:2 | 5.31 (4.83..6.17) | 3539 (3050..3889) | 4.81x (4.39..5.91) | 21.27 | 4.79x (4.31..5.84) |
| WNX MAG AVX-512 G=2, -march=native | orig_nat:wnx:mag_2048:2:2 | 5.95 (4.86..6.80) | 3160 (2765..3871) | 4.59x (3.91..5.23) | 23.10 | 4.55x (3.95..5.17) |
