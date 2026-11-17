# floor lane: rerun

Rerun everything (from this directory): `python3 extract.py 2 && gcc -O2 -march=native -o cm cm.c -lm && gcc -O2 -I/workspace/Wex-WRN2/c -o rt rt.c && gcc -O2 -I/workspace/Wex-WRN2/c -o rt_scale rt_scale.c && python3 decomp.py 2 && python3 mi2.py 2 && python3 mi.py weights 0:7,5:80,11:14,16:87,22:21,28:147,33:28,39:154 && python3 mi.py scales 0:7,5:80,11:14,16:87,22:21,28:147,33:28,39:154,39:120 && python3 adjacent.py && python3 run_cm.py out/cm_full_test.jsonl 2 0xffffffff $(cat out/test_list.txt) && python3 run_cm.py out/cm_f0_test.jsonl 2 0 $(cat out/test_list.txt) && python3 run_cm.py out/cm_primed_test.jsonl 1 0xffffffff $(cat out/primed_subset.txt) --prime 20:175 && python3 floor.py && python3 static_sib.py 1 && python3 run_rt.py $(cat out/test_list.txt) && python3 run_rt_v0.py && python3 run_rt_scale.py && python3 scale_table.py && python3 nn.py 2 16 4 && python3 nn.py 0 16 4`

| Script | Output | What |
|---|---|---|
| extract.py | out/coded/, out/manifest.json | decode 121 records (reference), coded domain (wex_nr3_prepare cz=1) |
| decomp.py | out/decomp.jsonl | stored bytes vs ideal cost under the production model: framing decomposition |
| mi2.py / mi.py / adjacent.py | out/mi2.jsonl, out/mi_*.json, out/adjacent.json | mutual information of all named relations |
| cm.c + run_cm.py | out/cm_*_test.jsonl | adaptive context-mixing bound |
| floor.py | out/floor.json | the floor table (best per part) |
| static_sib.py, rt.c, run_rt.py, run_rt_v0.py | out/static_sib.json, out/tables/, out/rt*.jsonl | sibling static tables, real rANS round trips |
| scale_table.py, rt_scale.c, run_rt_scale.py | out/scale_table.jsonl, out/rt_scale.jsonl | per-record scale tables, real round trips |
| nn.py | out/nn_m*.json | MLP predictor check |
