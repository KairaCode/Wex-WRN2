# fastfmt (lane, WNX prototype) — rerun

Copied without `data/` (Z(source) slabs), `out/` (encoded WNX records), binaries and `out_train.counts`.

    python3 prep.py && ./build.sh
    ./ff train out_train.counts $(cat train.txt)
    ./ff eval out_train.counts $(cat test.txt) > logs/eval_test.txt
    ./run_sizes.sh                 # logs/enc_*.txt (sizes, ref decode == Z)
    ./run_equal.sh                 # logs/equal_*.txt
    ./run_time.sh > logs/time.txt && ./run_time2.sh > logs/time2.txt
    ./bench_avx512 mutate 300 20261117 out/mag_2048 out/magt_2048 out/joint_2048 -- $(cat test.txt)
    ./bench_avx512 mutate 600 777 out/mag_2048 out/magt_2048 out/joint_2048 out/magt_4096 -- $(cat test.txt)
    ./bench_avx2 mutate 300 4242 out/mag_2048 out/magt_2048 out/joint_2048 -- $(cat test.txt)

Sizes in the report are the checker's like-for-like figures (../check-fastfmt/logs/vs_trainwnr3.txt);
speeds are the timing rerun's (../timing/).
