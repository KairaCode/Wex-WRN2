# overhead (lane) — rerun

Copied from /tmp/wnr3-research/overhead without binaries, `scales/`, `headers.bin`, `model_train.bin`,
`scale_train.bin` (all regenerated). Needs /tmp/wnr3-data. `bash run_all.sh` runs everything; step by step:

    gcc -O2 -fopenmp ovh_proto.c -o ovh_proto -lm
    python3 extract.py; python3 fetch_headers.py          # headers.bin = 32-B header of all 7,680 records (HTTP range reads)
    python3 scale_bounds.py 512 > scale_bounds_cs512.txt; python3 scale_adaptive.py > scale_adaptive.txt
    OMP_NUM_THREADS=2 ./ovh_proto train /tmp/wnr3-data/recs . $(python3 -c "print(' '.join(f'{l}:{e}' for l in range(40) for e in sorted({(37*l+11)%192,(37*l+101)%192})))")
    OMP_NUM_THREADS=2 ./ovh_proto run /tmp/wnr3-data/recs . out/run_test.json $(cat test_list.txt)
    python3 report.py out/run_test.json; python3 final_table.py out/run_test.json; python3 explain_outlier.py
    OMP_NUM_THREADS=2 ./ovh_proto audit /tmp/wnr3-data/recs . out/audit.json 2500 20261117 10:153 20:107 30:61 39:120 && python3 audit_report.py out/audit.json
    for R in 10:153 30:61; do OMP_NUM_THREADS=1 ./ovh_proto bench /tmp/wnr3-data/recs . 9 $R; done

Numbers backed here: every skeleton/scale profile (out/report.txt, out/final_table.txt), the audit
(out/audit_report.txt), the L39 E120 explanation (out/explain_outlier.txt), scale context bounds
(scale_bounds_cs512.txt, scale_adaptive.txt).
