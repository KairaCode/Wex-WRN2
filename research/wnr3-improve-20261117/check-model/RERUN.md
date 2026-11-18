# check-model (skeptic) — rerun

Own files only (common.py is modified to write locally and read the lane's coded slabs). Copy ../model/* into a
working directory, overlay these files, then:

    python3 myprep.py                                 # 158 records: decode, re-encode identity, coded slabs == lane's
    python3 ksets2.py 14 16 1 z8 && python3 real.py P14 P14_sesc P14_C16_K1 P14_C16_K1_z8_sesc prod P14_z8_sesc   # runA.log
    SET=extra python3 real.py prod P14 P14_sesc P14_C16_K1 P14_C16_K1_z8_sesc                                       # runC.log
    SPLIT=B python3 ksets2.py 14 16 1 z8 && SPLIT=B python3 real.py prod P14 P14_sesc P14_C16_K1_z8_sesc           # runB.log
    SPLIT=A python3 indep.py; SPLIT=B python3 indep.py   # own numpy bound -> indep_A.json, indep_B.json

New measurement only here: out/real_P14_z8_sesc.json (sesc + z8 alone: -4,224 B per typical record).
