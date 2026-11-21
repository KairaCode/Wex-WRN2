# fastfmt: a WNR3 successor coding designed for decode speed (lane: decode-speed format)

Rerun everything (from this directory; needs \<TMP>/wnr3-data and the Wex-WRN2 checkout, gcc with OpenMP):
`python3 prep.py && ./build.sh && ./ff train out_train.counts $(cat train.txt) && ./ff eval out_train.counts $(cat test.txt) > logs/eval_test.txt && ./run_sizes.sh && ./run_equal.sh && ./run_time.sh > logs/time.txt && ./run_time2.sh > logs/time2.txt && ./bench_avx512 mutate 300 20261117 out/mag_2048 out/magt_2048 out/joint_2048 -- $(cat test.txt) && ./bench_avx512 mutate 600 777 out/mag_2048 out/magt_2048 out/joint_2048 out/magt_4096 -- $(cat test.txt) && ./bench_avx2 mutate 300 4242 out/mag_2048 out/magt_2048 out/joint_2048 -- $(cat test.txt) && ./bench_avx512 scal out/mag_2048 -- data/Z_05_080.bin`

Logs: sizes `logs/enc_*.txt`, model bounds `logs/eval_test.txt`, byte equality `logs/equal_*.txt`, timing `logs/time.txt` (run 1) and `logs/time2.txt` (run 2), mutation `logs/mutate_*.txt`, thread probe `logs/scal.txt`.

Files: `wnx.h` (format, model, encoder, reference decoder), `wnx_fast.h` (AVX2 / AVX-512 decoders), `ff.c` (train,
eval, encode + reference round trip), `bench.c` (equality, paired timing against `wex_fast_decode`, mutation test),
`prep.py` (Z(source) of every sample record with the production CPU reference), `train.txt` / `test.txt` (the
suggested split), `logs/` (raw outputs), `RESULTS.md` (findings). `data/` and `out/` are generated (not needed to read).
