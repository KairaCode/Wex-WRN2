# audit: WNR3 T3 corruption audit on real ceil-form records (CPU)

Spec: `docs/WNR3-DRAFT-v2.1.md` section 11 **T3** and the TAIL-profile gate of 5.8. Measures how often a corrupted
ceil-form record is **silently accepted** (decoder says OK, output differs from the source) under each integrity rule,
and, for every silent acceptance, the damage after the spec ceil inverse.

CPU only (Python 3.10+, NumPy). No GPU kernels, no network.

## What it does

1. Self test: H32 = MurmurHash3_x86_32 verification value `0xB0F57EE3` and 4 known vectors.
2. Loads the 8 held-out (odd-index) experts of the 160-expert V4.1 sample: L00_E113, L06_E073, L11_E132, L17_E123,
   L23_E178, L28_E089, L34_E078, L39_E156 (manifest index 1, 25, 47, 71, 95, 113, 137, 159). They are checked with sha256.
3. Encodes each into 5 record kinds with `wnr_common_cr.py`, a byte copy of team 4 ceil-ref's `wnr_common.py`
   (sha256 F311E27A...0586). Model: `models/even.npz` key `fg_a0_K0_c0`, trained on the 80 even-index experts
   (held-out protocol). Anchor g0 is the constant 120.

   | kind | flags | rule | check |
   |---|---|---|---|
   | fg0d | CEIL\|LENC | **R0** | x0 = L, final state == L, exact length (today's rule) |
   | fg0de | CEIL\|LENC\|HASH | **HASH** | ceil-ref feature e: FNV-1a + lowbias32, 23 bits in x0 |
   | fg0dh | CEIL\|LENC\|HASH | **HASH32** | normative spec 5.1 H32 over ident16 \|\| coded bytes, 23 bits in x0 |
   | fg0dt | CEIL\|LENC\|TAIL | **TAIL** | feature t alone: tail symbols in x0, window check |
   | fg0dtD | CEIL\|LENC\|TAIL\|DIGEST | **TAILD** | TAIL + u32 record digest (XOR of per-stream H32, spec 5.8) |

   Each clean record is decoded byte-exact twice: once by the kit's decoder and once by the original ceil-ref
   `decode_record` (fg0d, fg0de, fg0dt). Its sha256 is compared with the research-PC value in
   `reference/records_sha256.json`. That comparison shows whether the coder gives identical bytes on this machine.
4. Runs the cases. Weight-stream trials cover all 3 matrix types (w1, w2, w3):

   | group | cases |
   |---|---|
   | controls | W0 (clean weight streams), S0 (clean scale streams): all must be accepted, identical output |
   | team 2 A-J (`audit_corrupt.py`) | A random bit flip, B last-byte flip, C flip in the 4-B initial state, D 2-byte overwrite, E zero-fill to the end, F length-1, G length+1, H stream at the next index, I next index with identical contexts, J other matrix type's model |
   | tail | T: one magnitude changed among the last 5 symbols, re-encoded, integrity field of the ORIGINAL data. R0 and TAIL accept this by construction |
   | length table | M1 (bit flip in the LENC length section, record level), M2 (weight boundary shift, sum preserving), M3 (scale boundary shift) |
   | swaps | N1 adjacent weight streams, N2 random pair, N3 same stream from another record, N4 adjacent scale streams, N5 scale stream from another record, O whole record in another expert's slot |
   | scale streams | K random bit flip (5 passes over every scale stream), L last-byte flip |
   | truncation | P1 last 4 KB page dropped, P2 torn write (zero to the end), P3 last 1-4 payload bytes zeroed |

   For each accepted corruption the test records: coded symbols and scales differing, output nibbles, weight values
   and scales differing, whole-block M' == 5 decision flips (`block_flips`), the max error in E2M1 units of the
   source block scale and in absolute value, sign-only damage, and NaN scales. Acceptance is counted twice: with the
   spec ceil inverse (M' 0..4 legal) and with ceil-ref's stricter inverse. Rates carry exact 95% CIs (Clopper-Pearson).
5. Cross-check (`validate.py`): a sample of silent and non-silent trials of cases A and K is replayed on the full
   record bytes through the record-level decoder, and a few through the full decoders (the kit's and ceil-ref's).
   The results must agree with the stream-level engine.

Default full sizes, with 8 experts:
- A and W0: every weight stream, 46,080 random flips per rule and matrix type.
- K: 28,800 scale-stream flips per rule and matrix type.
- Other weight cases: 24,000 per matrix.
- M1: 4,000 record patches per rule. P2: 800.

## How to run (PowerShell, from the kit root)

```powershell
# data: the 8 expert .bin files (or all 160) of the v41-sample, in data\v41-sample\ (default) or pass --sample DIR
# first check, a few minutes:
python -I tests\audit\run_audit.py --quick
# full test (the T3 numbers). --jobs defaults to auto = min(4, logical CPUs / 2, 0.7 x RAM / 3 GB): up to the
# 4 decoder kinds in parallel (~3 GB RAM per job); force it with --jobs N (run_all.ps1 uses the auto default)
python -I tests\audit\run_audit.py
# if the experts are not on this PC but the V4.1 safetensors shards are (revision 4ab649b2...), extract them locally:
python -I tests\audit\run_audit.py --quick --shards "<folder with the model-000NN-of-00048.safetensors files>"
```

Options:
- `--root` / `--data` / `--out`: standard kit options. The default output is `results\audit\`.
- `--sample DIR`: folder that holds the expert `.bin` files.
- `--shards DIR`: extract missing experts from local shards. The bytes are checked against the manifest sha256 and
  written to the sample dir.
- `--rules R0,HASH,HASH32,TAIL,TAILD`: TAIL and TAILD always run together.
- `--cases A,B,...`: run only these cases.
- `--experts N`: how many of the 8 experts to use, spread over the layers. Quick uses 2.
- `--per`, `--per-a`, `--per-s`, `--rper`: trial counts.
- `--skip-verify`: skip the byte-exact decode of the clean records. The result is then at most DONE.
- `--validate-n`, `--validate-full`: size of the cross-check.
- `--keep-records`: keep the encoded records in `<out>\records`, about 16.5 MB each (660 MB in total). By default
  they are deleted at the end.

`--quick` uses 2 experts (L00_E113, L39_E156), 300 streams per (expert, matrix) for the structured cases, 1000 for A
and W0, 60 scale streams per pass, M1 50 and P2 20 per expert. All rules and all cases run.

Outputs in `results\audit\`:
- `summary.json`: environment, config, gates, headline per rule, record checks, expected chance acceptances.
- `audit.log`.
- `tables.txt`: one line per case x rule x matrix.
- `damage_tables.json`: damage histograms of the silent acceptances. For case T, block flips by tail position and by
  magnitude change.
- `case_<id>.json`: per-trial damage of every silent trial; rows for the record-level cases.
- `validate.json`.

The last line printed is `RESULT audit PASS|FAIL|DONE ...`.

## Pass rule

**PASS** requires all of these:
- the H32 self test passes;
- the clean records are byte-exact with both decoders;
- the W0 and S0 controls are all accepted with identical output;
- no case errors;
- **0 silent acceptances for HASH, HASH32 and TAILD in every case** (spec T3: HASH part gates R1; TAIL + DIGEST
  gates R2b);
- the cross-check agrees.

R0 and TAIL alone are measured and reported, not gated: acceptance rates, CIs, block flips and max error.

`summary.json -> expected_chance_acceptances` gives n x 2^-23 (HASH) and n x 2^-32 (TAILD). One acceptance at that
level would be a hash collision, not a construction defect, but the spec still counts it.

**FAIL** means one of the checks above failed. **DONE** means nothing failed but a gate was not run, for example
`--skip-verify` or no strong rule selected. `random_flip_target_met` reports whether the 24,000-per-matrix target of
the task was reached for A and K. With 8 experts it is.

## Reference values (research PC, i7-7700K, numpy 2.5.4, Python 3.14.7)

- `reference/records_sha256.json`: sha256 and sizes of all 40 clean records. The record sizes equal ceil-ref's held-out
  rows fgd / fgde / fgdt.
- `reference/team5_partial_cases.json`: team 5's full-size runs, stopped after W0, A, B (C only in the log). On 8
  experts:

  | case | rule | silent | rate |
  |---|---|---|---|
  | A | R0 | 588 / 138,240 (w1 161, w2 227, w3 200) | 0.425% |
  | A | TAIL | 1,174 / 138,240 (w1 376, w2 417, w3 381) | 0.85% |
  | A | HASH, HASH32, TAILD | 0 / 138,240 each | 0 |
  | W0 | all rules | 0 silent, all accepted | 0 |
  | B | R0 and HASH | 0 / 72,000 | 0 |
  | C (log only) | R0 | 142 / 72,000 | |

  For A under TAIL, 330 of the 1,174 silent trials flip at least one block's M' == 5 decision; the max error is 18
  E2M1 units. For A under R0 there are no block flips and the max error is 12 units.
- Team 5 smoke on 1 expert x 80 streams (not in the kit), case B: TAIL accepted 232 / 240 last-byte flips, TAILD 0.
  Case I: R0 and TAIL accept 100%, because the stream is valid and only the identity differs.
- `reference/team2_wnr2_corrupt_audit.json`: team 2's original audit on WNR2 (non-ceil) streams, rule R0, for comparison.

Smoke runs on this PC with 1 expert and tiny trial counts took 9-19 s each. Their results are consistent with the
values above. Runtime of the full and quick runs: **not measured**. My estimate is 1-2 h for the full run with
`--jobs 1` on an i7-7700K, about 3x faster with `--jobs 4`, and 3-10 min for the quick run. With run_all.ps1 give
`-TimeoutMin` at least 240 for a full run (or leave it at 0 = no timeout).

## Notes / known gaps

- Case T is a re-encode with a stale integrity field, not a byte corruption. R0 and TAIL accept it by construction;
  it is reported to count block flips. HASH, HASH32 and TAILD must reject it.
- Cases H, I and J follow team 2's synthetic definitions. In J the stream keeps its own srel contexts.
- The spec-layout records (64-B header, WNM3 K=2, per-matrix gate) are not implemented anywhere. This test uses
  ceil-ref's 48-B record format plus a 4-B digest word (52-B header) for TAILD.
- DEDUP alias and owner patch (DRAFT 0.1 T3 items) are not covered.
