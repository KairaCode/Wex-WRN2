# fixtures/

**Current.** Small inputs for the tests, so that nothing here needs a real model.

- `v41-tiny/`: a 6-layer DeepSeek V4.1 test model (`model.safetensors` 3,118,984 B, with its config, tokenizer and engram description). `scripts/build_cpu_check.sh` and `wex4/build_and_test.sh` build their WEX1 and WNR3-CZ containers from it. The integration guide says it runs on the DeepSeek engine build only.
- `make_synth_mxfp4.py`: writes a synthetic DeepSeek MXFP4 expert checkpoint (experts only, not a runnable model) with skewed codes, so an entropy coder has something to gain: `python -I fixtures/make_synth_mxfp4.py OUT_DIR [--layers 2] [--experts 8]`.
- `widen_experts.py`: copies a tiny checkpoint with a wider MoE intermediate size, because a WNR2 build needs records that are whole 4096-byte pages. Used by the token A/B of section 7.5 of [docs/reports/colibri-integration-20261109.md](../docs/reports/colibri-integration-20261109.md).

Both scripts use the standard library only.
