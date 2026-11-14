# wex4/

**Current.** The WEX4 container outside the engine: reference implementation, converter, C tools, tests, the reviews and the research behind the format. Specification: [docs/WEX4-SPEC.md](../docs/WEX4-SPEC.md); short Italian summary: [docs/reports/wex4-summary-20261111.it.md](../docs/reports/wex4-summary-20261111.it.md). This folder was called `next/` until 2026-11-11.

## What is mirrored and what lives here

The C reader **[`../c/wex4.h`](../c/wex4.h) is the engine-side reader** (`src/colibri/c/wex4.h` in an engine tree). It is the header the engines include, through `deepseek_v4.c` and [`../c/qwen38_wex.h`](../c/qwen38_wex.h). Everything in `wex4/` exists only here:

| File | What it is |
|---|---|
| `wex4.py` | Python reference, standard library only, written independently of `c/wex4.h`: the tests run both on the same bytes and require the same verdict. Command line: `python -I wex4.py info FILE`, `python -I wex4.py verify FILE [--dll out/wex4_tools.dll]` |
| `wex4_convert.py` | `pack` / `unpack` (safetensors to WEX4 and back), `upgrade` / `downgrade` (a WEX1 or WEX3/WNR3 expert container to WEX4 in place and back; the records are not moved). `--names PROFILE` picks the naming profile of the model family ([below](#naming-profiles)) |
| `wex4_store.h` | prototype of the engine loader of spec section 13.2: opens a WEX4 expert container and gives the store the table it keeps today, plus the per-record SHA-256 |
| `wex4_tools.c` | the C side of the tests, loaded through ctypes: validator, codecs, in-process fuzzer. Builds twice, with the reference WNR3 decoder and with the fast one |
| `test_wex4.py` | round trips, C against Python, hostile files, mutation fuzzing, and the check of the conformance vectors |
| `vectors/` | conformance vectors: valid, invalid and legacy files with the verdict, error class and root a reader must give, in `MANIFEST.tsv` ([README](vectors/README.md)) |
| `build_and_test.sh` | builds the two DLLs and the fixtures, then runs `test_wex4.py` |
| `fuzz/` | libFuzzer harnesses for the C reader, the WNR3 decoders (reference against fast) and `wex4_verify`, seeded with the vectors; `bash wex4/fuzz/run.sh` ([README](fuzz/README.md)) |

The engine side is not here: the loaders are in the engine, shown as patches in [`../integration/`](../integration/README.md) (`engine-wex4-loader-75932c1.patch`, `engine-wex4-first-read-4ba5568.patch`, `engine-q38-wex4-loader-de17198.patch`).

## Run the tests

CPU only, no real model, standard-library Python. On Windows: MSYS2 UCRT64 or MINGW64 `gcc` on `PATH`.

```sh
bash wex4/build_and_test.sh            # from the repository root; add --fuzz N to change the mutation count (default 1000)
```

The line before the last checks the conformance vectors; last line on success: `WEX4 prototype: all tests passed`. Everything built goes to `wex4/out/` (git-ignored). The script builds a WEX1 and a WNR3-CZ container from [`../fixtures/v41-tiny`](../fixtures/README.md) and a synthetic one, so it also exercises `c/tools/repack_v4_wex.py`, `c/tools/build_nr3.py` and `c/tools/wex_nr3_tools.c`.

Two decoders are compared on purpose. `c/wex4.h` always calls the reference WNR3 decoder; `wex4_tools.c`, built with `-DWEX4_FAST_NR3`, puts `research/fast-cpu-decode/wex_fast_decode.h` (AVX2 + OpenMP on x86-64; on any other CPU its scalar kernel, still parallel across streams) under that name. The script stops if the two DLLs do not differ at that call site, because a sync of `c/wex4.h` once removed the hook and the comparison then tested the reference against itself.

## Naming profiles

A naming profile says which tensors of a checkpoint are the six tensors of a routed expert (the scale and the weight of the matrices w1, w2, w3), and which layer and expert each one belongs to. `pack --experts raw|nr3 --names PROFILE` uses it to find the experts in the safetensors files. `upgrade --names PROFILE` uses it to name the tensors of a WEX1 or WEX3 container, which holds no names. The default is `deepseek`.

| Profile | Expert tensors (`{L}` layer, `{E}` expert) | Scale, weight | dtypes `upgrade` writes |
|---|---|---|---|
| `deepseek` | `layers.{L}.ffn.experts.{E}.w1\|w2\|w3` | `.scale`, `.weight` | F8_E8M0, I8 |
| `deepseek-quark` | `layers.{L}.ffn.experts.{E}.w1\|w2\|w3` | `.weight_scale`, `.weight` | U8, U8 |
| `qwen38` | `model.language_model.layers.{L}.mlp.experts.{E}.gate_proj\|down_proj\|up_proj` | `.weight_scale`, `.weight` | U8, U8 |
| `qwen3-moe` | `model.layers.{L}.mlp.experts.{E}.gate_proj\|down_proj\|up_proj` | `.weight_scale`, `.weight` | U8, U8 |
| `kimi-k26` | `language_model.model.layers.{L}.mlp.experts.{E}.gate_proj\|down_proj\|up_proj` | `.weight_scale`, `.weight` | U8, U8 |
| `mixtral` | `model.layers.{L}.block_sparse_moe.experts.{E}.w1\|w2\|w3` | `.weight_scale`, `.weight` | U8, U8 |
| `minimax-m3` | `language_model.model.layers.{L}.block_sparse_moe.experts.{E}.w1\|w2\|w3` | `.weight_scale`, `.weight` | U8, U8 |
| `mistral-large4` | `layers.{L}.experts.{E}.w1\|w2\|w3` | `.weight_scale`, `.weight` | U8, U8 |

The slab order is w1, w2, w3, and `gate_proj`, `down_proj`, `up_proj` for the profiles that use those names (the order `c/tools/repack_q38_mxfp4_wex.py` uses for Qwen3.8). A profile is a naming scheme, so several families share one.

**Where the names come from.** Each profile was read in the safetensors header of a published checkpoint on Hugging Face, on 2026-11-14, at the commit given (first 8 hex digits). Only the header of the first shard and of one middle shard was read (the file named below is the one that shows the expert tensors); the other shards and the weights were not read. In every MXFP4 checkpoint below the weight is U8 `[rows, cols/2]` and the scale U8 `[rows, cols/32]`, except DeepSeek's own (I8 and F8_E8M0).

| Profile | Checkpoint, commit, shard | Same names also read in |
|---|---|---|
| `deepseek` | `deepseek-ai/DeepSeek-V4.1-Flash` 314113f8, `model-00025-of-00048.safetensors` | `deepseek-ai/DeepSeek-V4-Flash` 60d8d707 |
| `deepseek-quark` | `amd/DeepSeek-V4.1-Flash-Quark-MXFP4` 8eea1b23, `model-00025-of-00048.safetensors` | |
| `qwen38` | `amd/Qwen3.8-Flash-Next-Quark-MXFP4` b71e28af, `model-00066-of-00131.safetensors` | `amd/GLM-5.3-Flash-Quark-MXFP4` b5688f25, `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` 17159bf7; `amd/Qwen3.5-397B-A17B-MXFP4` edf0958b and `amd/Qwen3.5-35B-A3B-MXFP4` 2e19c657 (the shards read hold `gate_proj` and `up_proj` only) |
| `qwen3-moe` | `amd/Qwen3-235B-A22B-Instruct-2507-MXFP4` 381342aa, `model-00001-of-00026.safetensors` | `amd/Qwen3-Coder-Next-MXFP4` 1e05fd91, `amd/Kimi-K2-Instruct-0905-MXFP4` fd7ab8ec, `amd/GLM-4.7-MXFP4` e01203e8, `amd/GLM-5.1-MXFP4` 74fb7daa, `amd/DeepSeek-R1-0528-MXFP4` 913fc83b, `amd/DeepSeek-V3.2-mxfp4` 84493811 |
| `kimi-k26` | `amd/Kimi-K2.6-MXFP4` 18ecc30a, `model-00033-of-000064.safetensors` | `amd/Kimi-K2.5-MXFP4` 42c6da36 |
| `mixtral` | `amd/MiniMax-M2.1-MXFP4` bbf15c45, `model-00001-of-00026.safetensors` (MXFP4); `mistralai/Mixtral-8x7B-v0.1` fc7ac946, `model-00001-of-00019.safetensors` (the same module names, BF16, no scale tensor) | |
| `minimax-m3` | `amd/MiniMax-M3-MXFP4` b83d14e3, `model-00030-of-00059.safetensors` | `amd/Kimi-K3-Quark-MXFP4-AttnFP8` c5d1de02 |
| `mistral-large4` | `amd/Mistral-Large-4-1T-A52B-Quark-MXFP4` 793c850b, `consolidated-00001-of-00054.safetensors` | |

**What fits.** An EXPERT record is an MXFP4 slab ([WEX4-SPEC](../docs/WEX4-SPEC.md) section 5.5): three matrices, each with a scale tensor (U8 or F8_E8M0, `[rows, cols]`) and a weight tensor (U8 or I8, `[rows, 16 * cols]`), every dimension at most 65,535. The checkpoints above have this form for one expert at a time, so they need no format change. `pack` checks every expert the profile finds. An expert that is not a whole slab stays ordinary tensors (WBP1 or RAW records), is counted in the line `pack` prints, and comes back byte for byte with `unpack` like everything else.

**What does not fit**, and is therefore stored as ordinary tensors, with no expert record and no WNR3:

- **Float and FP8 checkpoints of the same families.** The names match the profile, but there is no MXFP4 slab: `mistralai/Mixtral-8x7B-v0.1` (BF16), `Qwen/Qwen3-30B-A3B` ad44e777 (BF16), `zai-org/GLM-4.5-Air` a24ceef6 (BF16), `moonshotai/Kimi-K2-Instruct-0905` ac6c49f0 (F8_E4M3 weights with an F32 `weight_scale_inv`). An expert record for them would be a new kind of record, which is a format change and is not done here.
- **Mixed checkpoints.** In `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` the experts of layer 0 are BF16 and those of layer 32 are MXFP4; in `amd/GLM-5.3-Flash-Quark-MXFP4` the experts of layer 45 are F8_E4M3. The float and FP8 experts stay ordinary tensors, the others become expert records.
- **gpt-oss.** `openai/gpt-oss-20b` 6cee5e81 holds one tensor per layer for all 32 experts (`model.layers.{L}.mlp.experts.gate_up_proj_blocks`, U8 `[32, 5760, 90, 16]`, with `_scales`, `down_proj_*` and BF16 biases). `amd/gpt-oss-20b-w-mxfp4-a-bf16` 34f332d1 has one group of tensors per expert, but two matrices (`gate_up_proj`, `down_proj`) and biases. An expert record has exactly three matrices, and a tensor lies in one record, so neither form fits. No profile.
- **Shared experts and MTP layers** have other names (`shared_experts`, `shared_expert`, `mtp.layers`) and are not in the expert grid.

**Limits.**

- The profiles were tested on synthetic fixtures only (`fixtures/make_synth_mxfp4.py --names`, 2 layers x 3 experts, hidden 512 x intermediate 256). No real checkpoint of these families was downloaded or converted.
- No MXFP4 checkpoint of Mixtral itself was found: on 2026-11-14 the repository `amd/Auto-Mixed-Precision-Mixtral-8x7B-Instruct-v0.1-Weight-Activation-Mixed-MXFP4-FP8PT-KVFP8` held no safetensors file. The `mixtral` profile is Mixtral's module names in the MXFP4 form read in MiniMax-M2.1.
- `amd/Step-3.5-Flash-MXFP4` 51e8199b has no tensor name with `experts.` in the two shards read; it was not looked at further.
- The engines of this repository read DeepSeek V4 and V4.1 and Qwen3.8 only. For the other families the profiles serve the container tools (`pack`, `unpack`, `upgrade`, `verify`), not an engine. The tools that write a WEX1 container (`c/tools/repack_v4_wex.py`, `c/tools/repack_q38_mxfp4_wex.py`) know the DeepSeek and Qwen3.8 names only, so for the other families the way in is `pack --experts`.

The test is `test_profiles` in `test_wex4.py`. For every profile it packs the fixture with RAW and with WNR3 expert records, checks the file with both readers, compares each decoded record with the slab of the WEX1 container of the same experts, and unpacks to the same bytes. It upgrades a WEX1 and a WNR3 container with the profile's names, reads every tensor of the fixture back by name, and downgrades to the same bytes. It also packs a mixed checkpoint (a first layer without experts, an MXFP4 layer, a float layer), a fixture under every other profile's names, and names that only look like experts.

## Reviews

[`review/REVIEW.md`](review/REVIEW.md): the adversarial review of the spec and the prototype of 2026-11-11, five lenses (`security-c`, `spec-vs-code`, `converters`, `claims`, `design`), each finding with its verification and what was done. The reviewers' scripts and trigger files are in the subfolders of `review/`. The review is frozen in content. Its paths were rewritten to today's layout on 2026-11-12 (`next/` became `wex4/`, `src/` and `next/wex4.h` became `c/`); line numbers are those of the commit it reviewed.

## Research behind the format

[`research/VERIFICATION.md`](research/VERIFICATION.md) is the entry: each study was re-checked by an independent pass, and its corrected numbers are the ones the spec uses.

| Folder | Study | Result file |
|---|---|---|
| `research/dtype-entropy/` | the tensors that are not routed experts (FP8, BF16, F32): how compressible, and with which static model. The source of the WBP1 byte-plane codec | [RESULTS.md](research/dtype-entropy/RESULTS.md) |
| `research/fast-cpu-decode/` | a bit-exact, faster CPU decoder for today's WNR3 records, no format change; holds `wex_fast_decode.h` | [RESULTS.md](research/fast-cpu-decode/RESULTS.md) |
| `research/mxfp4-contexts/` | can a cheap static context model beat WNR3-CZ on MXFP4 weights? The file's answer: no | [RESULTS.md](research/mxfp4-contexts/RESULTS.md) |
| `research/wnr4-record/` | how many bytes a "WNR4" record skeleton would save over WNR3-CZ. The file's answer: at most 0.58% of raw | [RESULTS.md](research/wnr4-record/RESULTS.md) |
| `research/measure-real/` | the prototype on real data (sizes, speeds, exactness): the script and its raw output (`results.json`, `test_run.log`); no result page | none |

Runs on the real containers are with the other dated runs: [`../research/wex4-m3-20261111/RESULT.md`](../research/wex4-m3-20261111/RESULT.md) (DeepSeek V4.1: gates M3 and M4, first-read SHA-256), [`../research/wex4-qwen38/README.md`](../research/wex4-qwen38/README.md) (Qwen3.8 dry run) and section 3 of [`../research/qwen38-speed-20261111/README.md`](../research/qwen38-speed-20261111/README.md) (the Qwen3.8 containers upgraded in place).

## Old name

[`../docs/NEXT-FORMAT-SPEC.md`](../docs/NEXT-FORMAT-SPEC.md) is a stub that points to the spec. It exists because `c/wex4.h` names that path in a comment, which is kept as it is.
