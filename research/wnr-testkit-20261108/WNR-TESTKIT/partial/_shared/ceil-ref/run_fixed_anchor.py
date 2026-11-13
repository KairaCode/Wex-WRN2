"""Variant g0: constant srel anchor (default 120) on ceil-form instead of the per-matrix densest-pair rule (team4 / ceil-ref).
The value is still stored in the record's median byte, so the decoder is unchanged. All 80 training matrices x3 have
G anchor 120 on ceil-form (anchor_diag.py), so the fg models are exactly the models a constant-120 rule would train.
Usage: python -I run_fixed_anchor.py ANCHOR SAMPLE_DIR MODELS_NPZ OUT_JSONL EXPERT_INDICES CONFIGS  (same args as run_ceil.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr_common as W
import run_ceil

A0 = int(sys.argv.pop(1))
W.anchor_pair = lambda sc: A0          # prepare() looks the name up at call time
run_ceil.main()
