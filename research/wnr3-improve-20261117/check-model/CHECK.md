# Check of the model lane (/tmp/wnr3-research/model), 2026-11-17

Everything here was rerun in /tmp/wnr3-research/check-model. The copied scripts are unmodified, except that
common.py writes here and reads only the coded slabs from the lane. Those slabs were first proven
byte-identical to a fresh reference decode followed by prepare(cz=1) (myprep.py). Logs: myprep.log, runA.log,
runB.log, runC.log. Per-record JSON: out/. Independent bound: indep.py -> indep_A.json, indep_B.json.

- Inputs: 121 + 37 records. Reference decode OK, re-encode with the production model byte-identical 158/158,
  lane coded slabs identical 158/158, flags 0x0f on all. headers.bin agrees with all 158 local record headers;
  38 records in it have anchors != 120. Baseline 0.85436 padded / 0.85426 unpadded (reproduced).
- harness.py: the prototype reproduces the stored records (00_007, 39_120); q_rule == library build_model.
- Real encode + decode + SHA-256 of Z(source), all ok. Per-record sizes are identical to the lane's.
  Deltas on 40 normal TEST records:
  - C16+z8+sesc: A -11,317 (median -7,858); B -13,363 (median -9,498). Every record smaller on both splits.
  - Shifted records: 37 extra -154,099 B mean; with L39E120, 38 records -174,790 B mean.
  - Container: -92.4 MB. That is 0.064% of the raw container; the lane's 0.075% is of the 123.3 GB file.
  - Other variants: C16 -7,927 (37 extra: -111,983); sesc -3,836; P14 -828; P16 -3,480; T80 lib -835;
    lib_self -7,071; K4 -6,574; C8 -7,343.
  - New: sesc+z8 (not encoded by the lane) -4,224 (z8 adds -388 on top of sesc).
- Independent numpy re-derivation (ideal bound, own contexts, classes, quantiser): A -11,322 / B -13,358 per
  normal record. Shifted (38): A -174,774, B -154,529. The real-vs-ideal gap is below 6 B/record.
- Problems found:
  - sesc alone makes 16 of the 37 shifted records bigger (up to +91 KB, 15_024).
  - On top of C16, sesc makes 19/37 shifted records bigger than C16 alone (max +87 KB). Means are still
    negative.
  - L39 E154 alone carries about 2 KB of the 11.3 KB mean. Without layer 39: -9.4 KB (-0.050%).
- Not rerun: timing; the bounds in screen.py / q_opt / scale_ctx; C32, K2, K8, C8xK2/K4; the T40 retrains;
  P15.
