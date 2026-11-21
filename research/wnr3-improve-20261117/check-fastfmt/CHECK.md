# Check of the fastfmt (decode-speed format) lane

Everything here was rerun in \<TMP>/wnr3-research/check-fastfmt. The lane's sources were copied unchanged, except that bench.c's two hard-coded paths now point to this directory.

- prep.py: all 121 records decoded with the production reference. The Z/E files are byte-identical to the lane's data/.
- ff train: out_train.counts is byte-identical. ff eval: logs/eval_test.txt is identical.
- All 7 encode configurations rerun (41 TEST records each). All 287 records are byte-identical to the lane's out/, logs/enc_*.txt are identical, and the reference round trip == Z holds 41/41 everywhere. My copies were deleted because the disk was short; their sha equals the lane's files.
- bench equal (both builds, all 7 configurations): 0 mismatches. See logs/equal_*.txt.
- mutate: a new seed (31337, 300 cases, 4 dirs, AVX-512 build). The reference rejected all 300 and the fast kernels agreed.
- Negative control (nc/: reference hash and final-state checks removed): the test caught "reference ACCEPTED A WRONG OUTPUT" at case 4. See logs/mutate_negcontrol.txt.
- Independent bound (indep_bound.py, numpy; the only library call is wex_nr3_prepare): MAG with 7 srel contexts and raw sign gives 3.57661 bit/w exact and 3.57662 bit/w at 12 bit. This matches the lane.
- Same-training-set baseline (wnr3_train_baseline.py): WNR3-CZ re-encoded with a WNM2 model trained on the same 80 TRAIN records, at production chunk sizes 2048/512. Real records; the reference decode == Z on 41/41.
  - The result is -0.0255% of raw against the stored records. Almost all of it is L39: E120 -0.85%, E154 -0.26%.
  - WNX against this baseline (logs/vs_trainwnr3.txt):

    | configuration | vs TRAIN-model WNR3, % of raw | per-record range |
    |---|---:|---|
    | MAG 2048 | -0.246 | |
    | MAGT 2048 | -0.419 | -0.436..-0.349 |
    | MAGT Bs512 | -0.390 | |
    | MAGT 4096 | -0.487 | |
    | JOINT | -0.249 | |
    | MAG 1024 | +0.062 | |

- Short timing sanity run (5 rounds, load 8-10, logs/time_short.txt). Speed-ups are paired CPU-time ratios.

  | build | MAG | MAGT |
  |---|---:|---:|
  | AVX2 | 3.47x | 3.36x |
  | AVX-512 G=2 | 4.56x | 4.62x |
  | -march=native G=2 (both decoders) | 4.48x | 4.33x |

  wex_fast_decode is not faster with -march=native: 100.9 against 100.5 ms CPU.
