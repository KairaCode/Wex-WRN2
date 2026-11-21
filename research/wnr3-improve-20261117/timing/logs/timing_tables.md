
## 1 thread per record: 9 records (00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120), 7 rounds, 3 timed passes per run (median pass)
load average: ['1.22', '1.52', '2.74'] at start, ['1.00', '1.04', '1.79'] at end; per round: ['1.04', '1.00', '1.04', '1.00', '1.00', '1.00', '1.00']
| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |
|---|---|---:|---:|---:|---:|---:|
| A: wex_fast_decode unchanged, k1 x4 fused | orig_a2:w3:1,4,1 | 102.97 (89.18..115.20) | 183 (163..211) | 1 | 102.35 | 1 |
| WNR3 reference wex_nr2_decode | orig_a2:ref | 898.79 (854.96..927.62) | 21 (20..22) | 0.11x (0.10..0.13) | 893.15 | 0.11x (0.10..0.13) |
| wex_fast_decode unchanged, k1 x4 fused, AVX-512 flags | orig_a5:w3:1,4,1 | 101.58 (87.29..107.77) | 185 (174..215) | 0.98x (0.91..1.26) | 100.85 | 0.98x (0.92..1.26) |
| fastdec kernel 3 (AVX2) x4 fused | new_a2:w3:3,4,1 | 55.81 (48.99..61.49) | 337 (306..384) | 1.91x (1.72..2.13) | 55.23 | 1.89x (1.71..2.13) |
| fastdec kernel 4 (AVX-512) x3 fused | new_a5:w3:4,3,1 | 42.82 (40.86..50.79) | 439 (370..460) | 2.40x (1.97..2.61) | 42.73 | 2.40x (1.98..2.62) |
| WNX MAG 2048, AVX2 kernel G=1 | orig_a2:wnx:mag_2048:1:1 | 30.66 (26.19..32.14) | 613 (585..718) | 3.28x (3.01..3.93) | 30.57 | 3.29x (3.01..3.93) |
| WNX MAG 2048, AVX2 kernel G=2 | orig_a2:wnx:mag_2048:1:2 | 28.79 (27.29..34.31) | 653 (548..689) | 3.48x (2.70..3.90) | 28.73 | 3.47x (2.69..3.88) |
| WNX MAG 2048, AVX-512 kernel G=1 | orig_a5:wnx:mag_2048:2:1 | 30.50 (23.79..31.49) | 616 (597..790) | 3.51x (2.97..4.30) | 30.41 | 3.55x (2.99..4.26) |
| WNX MAG 2048, AVX-512 kernel G=2 | orig_a5:wnx:mag_2048:2:2 | 22.23 (18.41..23.89) | 846 (787..1021) | 4.54x (4.35..5.87) | 22.22 | 4.52x (4.35..5.87) |
| WNX MAG 2048, AVX-512 G=2, fastfmt's own flags | ffold_a5:wnx:mag_2048:2:2 | 18.70 (17.35..22.28) | 1005 (844..1083) | 5.41x (4.65..5.91) | 18.70 | 5.39x (4.67..5.99) |
| WNX MAGT 2048, AVX2 kernel G=1 | orig_a2:wnx:magt_2048:1:1 | 30.46 (26.59..31.75) | 617 (592..707) | 3.37x (3.13..3.89) | 30.32 | 3.37x (3.13..3.89) |
| WNX MAGT 2048, AVX-512 kernel G=2 | orig_a5:wnx:magt_2048:2:2 | 22.62 (18.08..23.86) | 831 (788..1040) | 4.42x (4.27..5.64) | 22.55 | 4.44x (4.27..5.64) |
| WNX JOINT 2048, AVX2 kernel G=1 | orig_a2:wnx:joint_2048:1:1 | 35.21 (29.48..38.80) | 534 (485..638) | 2.98x (2.69..3.36) | 34.98 | 2.99x (2.69..3.38) |
| WNX JOINT 2048, AVX-512 kernel G=2 | orig_a5:wnx:joint_2048:2:2 | 24.14 (19.44..25.10) | 779 (749..967) | 4.40x (3.59..5.34) | 23.81 | 4.39x (3.57..5.32) |

## 4 threads per record (OpenMP default wait policy): 9 records (00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120), 7 rounds, 3 timed passes per run (median pass)
load average: ['1.00', '1.04', '1.79'] at start, ['3.42', '2.00', '2.04'] at end; per round: ['1.44', '2.01', '2.58', '2.98', '3.21', '3.43', '3.42']
| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |
|---|---|---:|---:|---:|---:|---:|
| A: wex_fast_decode unchanged, k1 x4 fused | orig_a2:w3:1,4,1 | 25.67 (22.93..29.19) | 732 (644..820) | 1 | 102.22 | 1 |
| wex_fast_decode unchanged, k1 x4 fused, AVX-512 flags | orig_a5:w3:1,4,1 | 25.80 (24.42..27.00) | 729 (696..770) | 1.04x (0.94..1.13) | 102.07 | 1.04x (0.95..1.13) |
| fastdec kernel 3 (AVX2) x4 fused | new_a2:w3:3,4,1 | 15.29 (13.95..15.78) | 1229 (1192..1348) | 1.71x (1.52..1.95) | 60.57 | 1.69x (1.56..1.97) |
| fastdec kernel 4 (AVX-512) x3 fused | new_a5:w3:4,3,1 | 12.60 (11.79..13.30) | 1493 (1414..1595) | 2.09x (1.96..2.17) | 49.53 | 2.09x (1.97..2.19) |
| WNX MAG 2048, AVX2 kernel G=1 | orig_a2:wnx:mag_2048:1:1 | 7.67 (7.18..8.31) | 2452 (2262..2617) | 3.32x (3.06..3.72) | 30.17 | 3.36x (3.12..3.77) |
| WNX MAG 2048, AVX2 kernel G=2 | orig_a2:wnx:mag_2048:1:2 | 7.89 (7.33..9.92) | 2383 (1895..2567) | 3.19x (2.50..3.47) | 31.62 | 3.26x (2.59..3.50) |
| WNX MAG 2048, AVX-512 kernel G=1 | orig_a5:wnx:mag_2048:2:1 | 6.73 (6.09..7.14) | 2794 (2632..3089) | 3.88x (3.59..4.23) | 26.71 | 3.75x (3.50..4.18) |
| WNX MAG 2048, AVX-512 kernel G=2 | orig_a5:wnx:mag_2048:2:2 | 5.74 (5.15..6.26) | 3275 (3001..3653) | 4.46x (4.05..4.87) | 22.99 | 4.39x (4.12..4.74) |
| WNX MAG 2048, AVX-512 G=2, fastfmt's own flags | ffold_a5:wnx:mag_2048:2:2 | 5.17 (4.78..6.24) | 3639 (3013..3934) | 5.13x (4.37..5.26) | 21.00 | 5.05x (4.39..5.25) |
| WNX MAGT 2048, AVX2 kernel G=1 | orig_a2:wnx:magt_2048:1:1 | 7.48 (6.91..7.70) | 2513 (2443..2720) | 3.58x (3.30..3.97) | 29.92 | 3.57x (3.33..4.01) |
| WNX MAGT 2048, AVX-512 kernel G=2 | orig_a5:wnx:magt_2048:2:2 | 6.06 (4.91..6.77) | 3100 (2779..3830) | 4.13x (3.65..5.24) | 24.67 | 4.22x (3.67..5.25) |
| WNX JOINT 2048, AVX2 kernel G=1 | orig_a2:wnx:joint_2048:1:1 | 8.37 (7.87..9.61) | 2247 (1956..2389) | 3.05x (2.64..3.27) | 33.62 | 3.02x (2.58..3.27) |
| WNX JOINT 2048, AVX-512 kernel G=2 | orig_a5:wnx:joint_2048:2:2 | 5.75 (5.09..6.25) | 3268 (3006..3697) | 4.63x (4.01..5.29) | 22.76 | 4.67x (4.05..5.17) |

## 4 threads per record, OMP_WAIT_POLICY=passive: 9 records (00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120), 7 rounds, 3 timed passes per run (median pass)
load average: ['3.42', '2.00', '2.04'] at start, ['3.40', '2.47', '2.21'] at end; per round: ['3.55', '3.46', '3.18', '3.18', '3.21', '3.16', '3.40']
| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |
|---|---|---:|---:|---:|---:|---:|
| A: wex_fast_decode unchanged, k1 x4 fused | orig_a2:w3:1,4,1 | 26.44 (23.59..29.88) | 711 (629..797) | 1 | 102.83 | 1 |
| wex_fast_decode unchanged, k1 x4 fused, AVX-512 flags | orig_a5:w3:1,4,1 | 26.10 (22.86..28.92) | 720 (650..823) | 0.99x (0.92..1.16) | 102.01 | 0.99x (0.92..1.17) |
| fastdec kernel 3 (AVX2) x4 fused | new_a2:w3:3,4,1 | 14.64 (13.63..15.39) | 1284 (1222..1379) | 1.81x (1.74..1.99) | 55.18 | 1.87x (1.81..2.08) |
| fastdec kernel 4 (AVX-512) x3 fused | new_a5:w3:4,3,1 | 12.86 (12.02..13.99) | 1462 (1344..1564) | 2.07x (1.95..2.23) | 47.51 | 2.19x (2.00..2.34) |
| WNX MAG 2048, AVX2 kernel G=1 | orig_a2:wnx:mag_2048:1:1 | 7.37 (7.04..8.27) | 2553 (2274..2669) | 3.52x (3.13..3.89) | 28.84 | 3.52x (3.14..3.91) |
| WNX MAG 2048, AVX2 kernel G=2 | orig_a2:wnx:mag_2048:1:2 | 7.78 (7.68..8.85) | 2416 (2125..2449) | 3.21x (2.95..3.48) | 30.49 | 3.20x (2.94..3.44) |
| WNX MAG 2048, AVX-512 kernel G=1 | orig_a5:wnx:mag_2048:2:1 | 7.44 (6.60..7.97) | 2528 (2360..2849) | 3.55x (3.28..4.09) | 28.78 | 3.54x (3.29..4.08) |
| WNX MAG 2048, AVX-512 kernel G=2 | orig_a5:wnx:mag_2048:2:2 | 5.46 (4.99..7.04) | 3441 (2672..3765) | 4.92x (3.72..5.32) | 21.29 | 4.91x (3.79..5.30) |
| WNX MAG 2048, AVX-512 G=2, fastfmt's own flags | ffold_a5:wnx:mag_2048:2:2 | 6.01 (4.74..6.31) | 3128 (2978..3970) | 4.51x (4.29..5.47) | 23.45 | 4.51x (4.41..5.38) |
| WNX MAGT 2048, AVX2 kernel G=1 | orig_a2:wnx:magt_2048:1:1 | 7.59 (6.86..8.00) | 2477 (2351..2742) | 3.52x (3.33..3.98) | 29.10 | 3.56x (3.41..4.02) |
| WNX MAGT 2048, AVX-512 kernel G=2 | orig_a5:wnx:magt_2048:2:2 | 6.22 (5.32..6.96) | 3024 (2703..3534) | 4.21x (3.84..5.08) | 23.70 | 4.31x (3.89..5.24) |
| WNX JOINT 2048, AVX2 kernel G=1 | orig_a2:wnx:joint_2048:1:1 | 8.19 (7.89..8.70) | 2295 (2161..2383) | 3.16x (2.90..3.37) | 32.17 | 3.14x (2.88..3.38) |
| WNX JOINT 2048, AVX-512 kernel G=2 | orig_a5:wnx:joint_2048:2:2 | 5.37 (5.17..5.65) | 3500 (3328..3635) | 4.66x (4.44..5.01) | 21.05 | 4.68x (4.45..5.07) |

## 4 records at once, 1 thread each (throughput; ms = wall / records): 9 records (00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120), 7 rounds, 3 timed passes per run (median pass)
load average: ['3.40', '2.47', '2.21'] at start, ['2.74', '2.61', '2.31'] at end; per round: ['3.63', '3.40', '3.25', '2.61', '2.81', '2.74', '2.74']
| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |
|---|---|---:|---:|---:|---:|---:|
| A: wex_fast_decode unchanged, k1 x4 fused | orig_a2:w3:1,4,1 | 35.01 (29.71..39.97) | 537 (470..633) | 1 | 103.14 | 1 |
| wex_fast_decode unchanged, k1 x4 fused, AVX-512 flags | orig_a5:w3:1,4,1 | 32.93 (29.12..36.22) | 571 (519..646) | 1.01x (0.88..1.15) | 100.55 | 1.03x (0.92..1.07) |
| fastdec kernel 3 (AVX2) x4 fused | new_a2:w3:3,4,1 | 17.80 (16.07..19.63) | 1056 (958..1170) | 1.90x (1.63..2.18) | 54.84 | 1.87x (1.80..2.04) |
| fastdec kernel 4 (AVX-512) x3 fused | new_a5:w3:4,3,1 | 14.07 (12.86..16.84) | 1337 (1117..1462) | 2.36x (2.06..2.63) | 45.08 | 2.29x (2.08..2.40) |
| WNX MAG 2048, AVX2 kernel G=1 | orig_a2:wnx:mag_2048:1:1 | 10.10 (8.90..11.33) | 1861 (1659..2112) | 3.49x (2.86..3.86) | 30.58 | 3.30x (3.19..3.86) |
| WNX MAG 2048, AVX2 kernel G=2 | orig_a2:wnx:mag_2048:1:2 | 9.46 (8.96..11.19) | 1987 (1681..2099) | 3.62x (2.92..4.12) | 30.92 | 3.31x (2.97..3.51) |
| WNX MAG 2048, AVX-512 kernel G=1 | orig_a5:wnx:mag_2048:2:1 | 9.02 (7.94..10.30) | 2083 (1825..2367) | 3.63x (3.42..4.63) | 27.23 | 3.85x (3.49..4.16) |
| WNX MAG 2048, AVX-512 kernel G=2 | orig_a5:wnx:mag_2048:2:2 | 6.01 (5.93..8.57) | 3126 (2194..3172) | 5.14x (3.95..6.14) | 22.60 | 5.00x (4.10..5.15) |
| WNX MAG 2048, AVX-512 G=2, fastfmt's own flags | ffold_a5:wnx:mag_2048:2:2 | 6.97 (5.84..9.92) | 2697 (1895..3219) | 4.89x (3.31..6.10) | 21.29 | 4.81x (4.07..5.29) |
| WNX MAGT 2048, AVX2 kernel G=1 | orig_a2:wnx:magt_2048:1:1 | 10.58 (8.52..14.62) | 1778 (1286..2207) | 3.33x (2.43..4.07) | 31.80 | 3.27x (2.82..3.72) |
| WNX MAGT 2048, AVX-512 kernel G=2 | orig_a5:wnx:magt_2048:2:2 | 7.67 (6.00..9.11) | 2451 (2064..3132) | 4.64x (4.05..6.22) | 22.58 | 4.69x (4.05..5.15) |
| WNX JOINT 2048, AVX2 kernel G=1 | orig_a2:wnx:joint_2048:1:1 | 11.77 (11.41..12.16) | 1597 (1545..1648) | 3.07x (2.77..3.24) | 35.08 | 3.14x (3.00..3.19) |
| WNX JOINT 2048, AVX-512 kernel G=2 | orig_a5:wnx:joint_2048:2:2 | 8.04 (6.37..8.35) | 2339 (2251..2950) | 4.40x (4.25..5.36) | 23.59 | 4.45x (3.88..5.27) |
