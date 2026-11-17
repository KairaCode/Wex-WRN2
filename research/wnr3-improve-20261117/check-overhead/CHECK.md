# Check of the "overhead" lane (scales, skeleton, integrity), 2026-11-17
Everything was rerun from copies in this directory; the lane's files were not touched.
- train: model_train.bin and scale_train.bin are byte-identical to the lane's (TRAIN 80 records, disjoint from TEST 41).
- run: 41/41 harness byte-exact (shipped encoder + lane encoder reproduce stored records), 1148/1148 (41 x 28)
  profile round trips exact vs Z(source); out/report.txt and out/final_table.txt are IDENTICAL to the lane's.
- extract.py: scales/ and meta.json identical; baseline 0.85436 of raw (121 records), recomputed independently from file sizes.
- scale_bounds.py 512, scale_adaptive.py, explain_outlier.py: outputs identical. headers.bin: the 121 sample
  headers match the stored records; 38 non-120-anchor records, mean SP 134,790 confirmed.
- indep/est.py (own code, stored production records, own lenc packer and quantiser, ideal scale bits):
  lenc 24,459 B (lane 24,460); sd 2,423 (lane 2,406); per-record table 24.8 B (lane 24.8);
  profile C saving 40,849 B/record = 0.217% (lane 40,471 = 0.215%); tail5 38,706 (lane 38,591, E-C 38,617).
- indep/escape_test.c: the escape path (0 escapes on TEST, so never exercised by the lane) round-trips for 10
  scale profiles incl. C, E, F with injected out-of-window scales on L10 E153.
- myaudit/: reduced audit (types 0-5, no header bits), new seed 777, L20 E107 + L39 E120, 1500 trials/type:
  0/18,000 accepted-wrong for WNR3, A, C, D, E, F; E 51 digest-only catches; control tail5 without digest 60/18,000
  wrong (50/3,000 swaps = 1.7% ~ 1/64). summary.txt.
- not rerun: full audit with header bits (~40+ min), bench (timing phase later).
