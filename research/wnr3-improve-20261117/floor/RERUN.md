# floor (lane) — rerun

Copied from \<TMP>/wnr3-research/floor: scripts and the small raw outputs in `out/`. Not copied: `out/coded/`
(2.2 GB of coded slabs, rebuilt by `extract.py`), `out/tables/` (rebuilt by `static_sib.py`), `out/adj/` (2 records
fetched by `adjacent.py`), binaries.

Needs \<TMP>/wnr3-data and /workspace/Wex-WRN2. Run from a writable copy of this folder. `run_cm.py` and the
`run_rt*.py` scripts append to their output: delete `out/*.jsonl` before a clean rerun. About 2 h on 2 threads.

    python3 extract.py 2
    gcc -O2 -march=native -o cm cm.c -lm
    gcc -O2 -I/workspace/Wex-WRN2/c -o rt rt.c
    gcc -O2 -I/workspace/Wex-WRN2/c -o rt_scale rt_scale.c
    python3 decomp.py 2                                   # framing split   -> out/decomp.jsonl
    python3 mi2.py 2                                      # in-sample MI    -> out/mi2.jsonl
    python3 mi.py weights 0:7,5:80,11:14,16:87,22:21,28:147,33:28,39:154
    python3 mi.py scales 0:7,5:80,11:14,16:87,22:21,28:147,33:28,39:154,39:120
    python3 adjacent.py
    python3 run_cm.py out/cm_full_test.jsonl 2 0xffffffff $(cat out/test_list.txt)
    python3 run_cm.py out/cm_f0_test.jsonl 2 0 $(cat out/test_list.txt)
    # the report's floor used 14 primed records; primed_subset.txt lists 12, add 0:7 and 1:60 to reproduce 0.84673
    python3 run_cm.py out/cm_primed_test.jsonl 1 0xffffffff 0:7,1:60,$(cat out/primed_subset.txt) --prime 20:175
    python3 floor.py                                      # -> out/floor.json
    python3 static_sib.py 1 && python3 run_rt.py $(cat out/test_list.txt) && python3 run_rt_v0.py   # sibling context
    python3 run_rt_scale.py && python3 scale_table.py     # per-record scale tables
    python3 nn.py 2 16 4 && python3 nn.py 0 16 4          # MLP check (weak evidence)

Numbers backed here: floor bound 0.84673 of raw, framing 0.585%, weight/scale model gaps 0.069/0.037%
(out/floor.json, out/decomp.jsonl); sibling context -0.0316% (out/rt.jsonl); per-record scale tables -0.0265%
(out/rt_scale.jsonl); MI table (out/mi2.jsonl, out/mi_*.json, out/adjacent.json).
