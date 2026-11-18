# check-floor (skeptic) — rerun

Only the checker's own files are here (CHECK.md, indep.py, sib_check.py, scale_indep.py, mi_quick.py, logs, outputs
that differ from the lane's). To rerun: copy ../floor/* into a working directory, add these files, then run the
lane's commands (../floor/RERUN.md; the checker used out/primed_subset.txt as is, 12 records -> floor 0.84674) and:

    python3 indep.py 2        # production encoder rebuilds all 121 stored records from the coded slabs; framing 0.5854%
    python3 sib_check.py 2    # own sibling-context code + 2-fold CV inside TRAIN -> out/sib_check.json (-0.03162%)
    python3 scale_indep.py    # entropy bound of per-record scale tables (-0.02652%)
    python3 mi_quick.py       # I(a3;a1|srel3,srel1) on 7 records, matches mi2.jsonl to 1e-5
