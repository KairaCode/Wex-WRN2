set -eu
cd /tmp/wnr3-research/check-overhead
TRAIN=$(python3 -c "print(' '.join(f'{l}:{e}' for l in range(40) for e in sorted({(37*l+11)%192,(37*l+101)%192})))")
date; OMP_NUM_THREADS=2 ./ovh_proto train /tmp/wnr3-data/recs . $TRAIN 2> out/train.log
cmp model_train.bin ../overhead/model_train.bin && echo MODEL_SAME; cmp scale_train.bin ../overhead/scale_train.bin && echo STAB_SAME
TEST=$(python3 -c "print(' '.join([f'{l}:{(53*l+7)%192}' for l in range(40)]+['39:120']))")
date; OMP_NUM_THREADS=2 ./ovh_proto run /tmp/wnr3-data/recs . out/run_test.json $TEST 2> out/run_test.log; echo run_rc=$?
date
