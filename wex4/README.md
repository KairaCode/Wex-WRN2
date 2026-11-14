# wex4/

**Current.** The WEX4 container outside the engine: reference implementation, converter, C tools, tests, the reviews and the research behind the format. Specification: [docs/WEX4-SPEC.md](../docs/WEX4-SPEC.md); short Italian summary: [docs/reports/wex4-summary-20261111.it.md](../docs/reports/wex4-summary-20261111.it.md). This folder was called `next/` until 2026-11-11.

## What is mirrored and what lives here

The C reader **[`../c/wex4.h`](../c/wex4.h) is the engine-side reader** (`src/colibri/c/wex4.h` in an engine tree). It is the header the engines include, through `deepseek_v4.c` and [`../c/qwen38_wex.h`](../c/qwen38_wex.h). Everything in `wex4/` exists only here:

| File | What it is |
|---|---|
| `wex4.py` | Python reference, standard library only, written independently of `c/wex4.h`: the tests run both on the same bytes and require the same verdict. Command line: `python -I wex4.py info FILE`, `python -I wex4.py verify FILE [--dll out/wex4_tools.dll]` |
| `wex4_convert.py` | `pack` / `unpack` (safetensors to WEX4 and back), `upgrade` / `downgrade` (a WEX1 or WEX3/WNR3 expert container to WEX4 in place and back; the records are not moved). `--names deepseek` or `--names qwen38` picks the tensor names |
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
