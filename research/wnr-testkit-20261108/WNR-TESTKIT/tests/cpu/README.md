# CPU tests (numpy) — WNR2 / WNR3 real-record coder

Four runners on top of a copy of team 4's ceil-ref reference coder (`lib/wnr_common.py`, with team 6's
`set_chunks()`), team 3's coder copy (`lib/wnr_common_t3.py`, second plain-WNR2 decoder) and the shared helpers
(`lib/kitcommon.py`). No GPU, no network. Python 3.10+ and numpy.

Common options (every runner): `--root` (default: two levels above the script), `--data` (default `<root>\data`),
`--out` (default `<root>\results\<test>`), `--quick`, `--sample-dir`, `--extra-dir`, `--cache`
(default `<root>\results\_cache\cpu_counts`, shared count cache), `--jobs` (default: auto, about half the logical
CPUs, at most RAM/2 GB, at most 16; each worker needs about 1 GB).
Data: the 160-expert sample is looked up in `<data>\v41-sample`, `<data>\v41_sample`, `<data>\sample`, `<data>`;
the extra experts (70 routed + 6 MTP) in `<data>\extra`, `<data>\v41-extra`, `<data>\team4-data\extra`. The list
of experts and their sha256 come from the kit's own manifests (`setup\manifests\v41-sample.json`, `extra.json`), so
copying just the `.bin` files is enough; a `manifest.json` in the data dir is optional and is ignored if it differs
from the kit's (it is used only when the kit manifests are missing). The first dir holding every expert wins (else
the one holding most, with a warning). Every expert file is checked against the manifest sha256 when it is read.
`train_curve_mtp` also needs the extra experts (`setup\fetch_data.py --extra`); without them it FAILs.
Each runner writes `summary.json` (environment, checks, tables), `<test>.log` and a `*.jsonl` of per-record rows
into its results dir, and ends with `RESULT <test> PASS|FAIL|DONE <short summary>`. PASS = all hard checks ok.

```powershell
# from the kit root; first check (about 10 minutes in total on a desktop CPU)
python -I tests\cpu\heldout_configs.py --quick
python -I tests\cpu\wnr2_g0.py --quick
python -I tests\cpu\chunk_sweep.py --quick
python -I tests\cpu\train_curve_mtp.py --quick
# full runs
python -I tests\cpu\heldout_configs.py
python -I tests\cpu\wnr2_g0.py
python -I tests\cpu\chunk_sweep.py
python -I tests\cpu\train_curve_mtp.py
```

Single-worker times measured on the source PC (i7-7700K): count one expert about 2 s, encode one record 2.6 s,
decode one record 2.2 s. Full runs below are single-worker estimates; divide by the worker count (roughly).

| test | what | full (1 worker) | quick |
|---|---|---|---|
| heldout_configs | train on 80 even, 40 held-out experts, wnr2 f fg0 fg0d fg0de fg0dt fg0c fg0cde (+ fg0e fg0t fg0cd fg0ce), all decoded | ~35 min | ~3 min (reference models, 4 experts) |
| wnr2_g0 | plain WNR2, anchor rule med / c120 / pair, decoded by 2 unmodified WNR2 decoders, re-derive tests | ~18 min | ~2 min (reference models, the 3 mismatch experts + 1) |
| chunk_sweep | fg0de at 1024/2048/4096/8192 weight chunks (scale 256..2048), fg0dt at 2048/4096 | ~20 min | ~1 min (2 experts) |
| train_curve_mtp | learning curve 5..190 training experts (estimates), MTP storage options (estimates), 61 real records | ~15 min | ~3 min |

## Pass criteria and reference values

- **heldout_configs**: sources sha256; every record decodes byte-exact (c configs value-exact); full mode: models
  trained here equal ceil-ref `even.npz` (wf, scf, etab) and every record size equals ceil-ref `heldout_g0.jsonl`
  (`ref/heldout_ref.json`); quick (reference models): sizes of the 4 experts equal the reference.
  Reference (40 experts, mean total, % of raw vs WNR2 with 95% CI): wnr2 16,909,926.4 (0.89943);
  f -1.761 (-1.948..-1.632); fg0 -1.791; fg0d -1.932; fg0de -1.922 (-2.114..-1.773); fg0dt -2.144 (-2.334..-1.992);
  fg0c -4.532; fg0cde -4.661 (-4.855..-4.512).
- **wnr2_g0**: D1 and D2 byte-exact for every record; header carries the rule's anchors; full: med model equals
  ceil-ref WNR2, c120 / pair equal team 5's models; records identical (sha256) to team 5's.
  Reference: med 16,909,926.4; c120 16,873,369.6 (-0.194 % raw vs med, CI -0.410..-0.026; 6 matrices in 3 experts
  with anchor != lower median; 88% of the gain from those 3); pair = c120 on this sample.
  Soft check: R1 (decoder re-deriving the median) must never produce silently wrong output.
- **chunk_sweep**: fg0de / fg0dt decode byte-exact; 2048:512 rows equal ceil-ref heldout (exact);
  other chunks equal team 6's partial rows where present (`ref/chunk_ref.json`, 11 experts).
- **train_curve_mtp**: sha256, no test leakage, ceil round trip, all real records byte-exact.
  Soft: draws / aggregates / MTP estimates / real spot records equal team 5 (`ref/curve_mtp_ref.json`).
  Reference: fg0 - wnr2 = -1.795 % raw at N = 190 (random draws); MTP (estimates, mean ratio): WNR2 routed 1.1062,
  (i) routed g0 1.0927, (ii) routed pair 1.0520, MTP-own LOO pair 0.8733, LOO 120 0.8694, LOO const mode 0.8724.

## Notes

- The coder is deterministic: same models, same input -> same bytes on any machine. A size mismatch with the
  reference is a real difference (models, data or numpy behaviour), not noise.
- K = 2 for the MTP experts is composed from the real per-matrix stream bytes of two real records (exact for the
  payload; the reference coder implements K only together with the A side symbols).
- Debug options: `--limit-test N`, `--limit-train N`, `--models ref|train`, heldout `--configs`, wnr2_g0 `--rules`,
  chunk_sweep `--plan '1024:256:fg0e;8192:2048:fg0e'` / `--decode-all`, train_curve_mtp `--no-real` / `--real-limit N`.
