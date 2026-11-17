# timing (joint rerun on an idle machine) — rerun

Copied without bin/ and wnx/. Needs the fastfmt and fastdec lane directories (or the copies here) and about 2 GB
free disk for eq_all.sh.

    ./build.sh && ./build_flags.sh && ./eq_all.sh
    python3 timing.py 7 3 1 4 4p r4 > logs/timing_tables.md
    FLAGSET=_flags python3 timing.py 9 3 1 4 > logs/timing_flags.md

All speed figures of the report come from logs/timing_tables.md and logs/timing_flags.md (raw samples in logs/raw_*.json).
