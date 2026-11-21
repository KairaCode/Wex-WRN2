# check-overhead (skeptic) — rerun

Own files only. Copy ../overhead/* into a working directory first. `run_main.sh` has its directory hard-coded
(\<TMP>/wnr3-research/check-overhead): edit the `cd` line.

    bash run_main.sh                                   # train (cmp with lane's tables) + run on 41 TEST records
    python3 indep/est.py                               # own estimate from stored records: lenc, sd, profile C (0.217%), tail5
    gcc -O2 -I.. indep/escape_test.c -o escape_test    # escape path round trip with injected out-of-window scales (see file header)
    gcc -O2 -fopenmp myaudit/ovh_audit.c -o ovh_audit -lm   # reduced audit, seed 777 -> myaudit/audit_s777.json, summary.txt
