# check-fastfmt (skeptic) — rerun

Own files only (bench.c with local paths, run_sizes_a/b.sh, wnr3_train_baseline.py, indep_bound.py, nc/ = negative
control build with the reference's final-state/hash checks removed). Copy ../fastfmt/* into a working directory,
overlay these files, fix the hard-coded `cd` in run_sizes_*.sh, then run the lane's prep/build/train and:

    sh run_sizes_a.sh; sh run_sizes_b.sh             # all 7 configurations, compare with the lane's logs
    python3 wnr3_train_baseline.py                   # WNR3-CZ re-encoded with a TRAIN-trained WNM2 -> logs/wnr3_train.txt (-0.0255%)
    python3 vs_trainwnr3.py logs                     # like-for-like table -> logs/vs_trainwnr3.txt (needs enc_*.txt in logs/)
    python3 indep_bound.py > logs/indep_bound.txt    # 3.57662 bit/w
    ./bench_avx512 mutate 300 31337 out/mag_2048 out/magt_2048 out/joint_2048 out/magt_4096 -- $(cat test.txt)

vs_trainwnr3.py was written by the report assembler; it reproduces the checker's logs/vs_trainwnr3.txt exactly.
