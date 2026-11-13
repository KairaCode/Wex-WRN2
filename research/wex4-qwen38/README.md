# WEX4 for Qwen3.8 MXFP4 (2026-11-11, in progress)

- `next/wex4_convert.py upgrade FILE --names qwen38`: tensor names and dtypes of the Quark checkpoint
  (`model.language_model.layers.L.mlp.experts.E.{gate,down,up}_proj.weight[_scale]`, U8).
- `dryrun.py` / `dryrun.json`: the WEX4 index of the real Qwen container (WEX1, 48 x 512 experts,
  64.2 GB), built in memory from its header page and table, read-only: 24,576 records, index
  26,086,592 B (+0.04 %), validation 70 ms in C, legacy table reproduced byte for byte.
- Engine: KAIRA branch `q38-wex4-20261111`, commit de17198 on `qwen38-mxfp4-20261111` (patch in
  `integration/kaira-q38-wex4-loader-de17198.patch`). The tiny-fixture test `tests/test_qwen38_mxfp4` passes on the WEX1 and WNR3
  fixtures as they are (34 checks; `--nr3`: 9 with the stand-in helper, 6 on the CPU decoder) and gives the same
  counts on copies whose three containers were upgraded with `--names qwen38` (the engine log shows
  "WEX4 container"). Not done: any real-model run, upgrading the real container (it is in use by the Qwen
  quality and speed runs; upgrade the final WNR3 one), a release build of this branch.
