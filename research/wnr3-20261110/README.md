# WNR3 verification, 2026-11-10

Scripts and results behind [docs/WNR3-VERIFY.md](../../docs/reports/WNR3-VERIFY.md) (spec: [docs/WNR3-SPEC.md](../../docs/WNR3-SPEC.md)). Copied from the KAIRA lab folder `<KAIRA>\lab\wnr3-impl-20261110`; the large files stay there (`verify/nr3_*.bin` 3.6-3.7 GB, `wnr2_audit.bin`, `model_*_lab80.wnm2`, executables, `g7lab/bin`, `g7lab/runs`, the fresh workspace copies `fresh-fix3` / `fresh-verify`).

Absolute paths into the KAIRA tree were replaced: shell scripts read `$KAIRA` (the KAIRA tree, forward slashes), `.cmd` files `%KAIRA%`; the Python scripts import `build_nr3` from this repo's `nr3/` and read `research/wnr3-5090-20261107` relative to themselves. Drive paths (`U:`, `D:`, `E:`) are the lab machine's, as elsewhere in this repo. Logs are kept as written (scratchpad paths shown as `<SCRATCH>`).

| path | what |
|---|---|
| `verify/select_experts.py`, `select.json`, `pairs.txt` | the 103 experts: 40 held-out of the 5090 study + the 64 largest WNR2 records |
| `verify/encode.log`, `nr3_cz.bin.json`, `nr3_noncz.bin.json` | `build_nr3.py bins` runs (CZ and non-CZ, models trained on the lab's 80 experts) and their per-record reports |
| `verify/analyze.py`, `analyze.json` | sizes vs WNR2 and raw, independent value-exact CPU decode (section 1) |
| `verify/make_wnr2_bin.py` | WNR2 production records of the same experts, for the audits and the microbench |
| `verify/run_gpu_verify.sh`, `gpu_verify.log`, `gpu_audit.json` | GPU decode of the 103 records, ABAB microbench, GPU corruption audit (sections 1-3), under heavy.lock |
| `verify/audit/` | `audit_gpu.cu` (exhaustive flips via the helper batch API), `audit_real.c` + `run_cpu_audit.sh` (CPU decoder, exhaustive + random flips), `cpu_audit.jsonl` |
| `g7lab/run_g7_lab.sh`, `compare_g7.py` | G7-lab: WEX1 run with and without `V4_LAB_CANON_ZERO=1`, per-step logits memcmp; run 2026-11-10: **PASS** (`g7lab/RESULT-20261110.md`) |
| `mutants/` | `mutants.py`, `fix_mutants.py` (CPU/engine/store mutants, all KILLED), `gpu_mutants.py` + `run_gpu_mutants.sh` (GPU kernel mutants), `run_gpu.sh` (G2b/G4), their logs |
| `g3_compare.py`, `g3_result.json` | G3: new encoder with the lab CFG model vs the lab sizes and anchors |
| `full-build/` | the full V4.1 container: `RESULT-20261110.md` (size, model, sha256) and `readback.py` (7,680/7,680 records decoded from disk); in production since 2026-11-10 ([docs/WNR3-PRODUCTION-20261110.md](../../docs/reports/WNR3-PRODUCTION-20261110.md)) |
| `ram-record-cache/` | `RESULT-20261110.md`: the compressed-record RAM cache with 8 size classes (KAIRA branch `wnr3-ramcache-20261110`), equal-RAM A/B -0.19% ± 0.72%, not adopted |
| `build_helper.cmd` | helper + `test_decode_batch.exe` + `test_wex_nr3_gpu.exe` from a fresh copy of the KAIRA worktree |
