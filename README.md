<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wex-logo-dark.svg">
    <img alt="WEX" src="assets/wex-logo-light.svg" width="360">
  </picture>
</p>

# WEX

WEX is a family of expert-container formats for streamed Mixture-of-Experts inference: one file of 4096-aligned records, one per routed expert, that the engine reads from NVMe while it generates.

The repository slug says `Wex-WRN2` for historical reasons; the codec is spelled **WNR2** in every constant, file and page.

**Status.** WEX4 (format version 4.0) and the WNR3 record codec (v1) are frozen and in production since 2026-11-10/11, on one machine, for two models. The code here is the reference implementation and the research behind it: research-grade and Windows-first (CI also runs the CPU checks on Linux), one maintainer, no packages, no release yet, and no stability promise for the tools' command lines; the specifications are the stable part. [![CI](https://github.com/KairaCode/Wex-WRN2/actions/workflows/ci.yml/badge.svg)](https://github.com/KairaCode/Wex-WRN2/actions/workflows/ci.yml)

## What WEX is

Some Mixture-of-Experts (MoE) models do not fit in the machine's memory: DeepSeek V4.1 REAP-192 has 7,680 routed experts of 18.8 MB each. The engine keeps some experts resident and reads the others from NVMe when the router picks them. A WEX file is laid out for that read: every record starts on a 4096-byte boundary, is a multiple of 4096 bytes long, and is fetched with one unbuffered read. A record is either the expert's native MXFP4 slab (4-bit FP4 weights plus one E8M0 scale byte per 32 weights) or a lossless rANS coding of it (WNR3 with canonical zero: 0.854 of raw over the whole DeepSeek V4.1 container), which the CUDA helper decodes straight into VRAM. The current container, WEX4, wraps the same records in a SHA-256-hashed header page and index: 32 bytes, the root, identify the whole file, and the DeepSeek engine, with the root pinned, checks each record against it the first time it reads it. WEX was developed for KAIRA, a private local inference application whose engines are builds of the [colibri](https://github.com/JustVugg/colibri) C engine. The engine-side format code is in `c/`, and its integration into colibri is in progress. How the pieces fit together, and what is not in this repository: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Why not safetensors or GGUF

They are general checkpoint formats; WEX is built for one access pattern: fetching one routed expert from NVMe, many times a second, while the model generates. What that changes ([WEX4-SPEC](docs/WEX4-SPEC.md), sections 1.2 and 8):

| Need | safetensors / GGUF | WEX4 |
|---|---|---|
| one read per expert | an expert is several tensors (the weights and scales of three matrices) at offsets that are not page-aligned: safetensors aligns its header to 8 bytes, GGUF its tensor data to 32 bytes by default | the whole expert is one record, 4096-aligned and a multiple of 4096 bytes: one unbuffered read |
| detect a damaged or swapped file | no checksum: a flipped bit in a weight loads silently | SHA-256 of every record and blob, of the index and of the header page; 32 bytes, the root, identify the whole file |
| fewer bytes per read | no entropy coding | lossless WNR3 records: 0.854 of raw over the whole DeepSeek V4.1 container |
| move a 123 GB production file to the new layout | rewrite it | `upgrade` and `downgrade` in place: one page rewritten, 6.7 MB appended, records untouched |
| a small parser for untrusted files | safetensors: up to 100 MB of JSON | fixed-size binary rows, every count capped, validated in place with no JSON parser |

What they do better: they are public standards with many readers, and a tensor can be memory-mapped and used in place (in WEX4 only RAW records can).

## Format family

Two things are numbered separately. The **container** version (WEX1 to WEX4) is the file layout. The **record codec** (0 to 6) is how one record is coded. "A WNR3 container" means a WEX3 or WEX4 file whose records use codec 5.

| Container | Magic (u32 LE), version | What it adds | Header flags | Record codecs | Status (2026-11-13; live page: [STATUS.md](STATUS.md)) | Specified in | Implemented in |
|---|---|---|---|---|---|---|---|
| **WEX1** | `0x57455831`, 1 | a 4096-byte header page with the 40-byte `WexHeader`, one 72-byte `WexEntry` per (layer, expert), raw aligned records | must be 0 | 0 | no production file left: the V4.1 file was deleted on 2026-11-10 and the Qwen3.8 files were upgraded in place to WEX4. Still the input of `build_nr3.py` and of `upgrade` | `c/wex_format.h`; [WNR2-REFERENCE](docs/WNR2-REFERENCE.md) | `c/wex_format.h`, `c/expert_store_wex.c`, `c/qwen38_wex.h`; writers `c/tools/repack_v4_wex.py`, `c/tools/repack_q38_mxfp4_wex.py`; checker `scripts/verify_wex1.py` |
| **WEX2** | `0x57455832`, 2 | E8M0 scales packed in tiles of 256 values; weights unchanged | 1 only | 0, 1 | written and tested; measured on 2 experts on 2026-10-18, never built for a whole model | [wex2-scale-tiles](docs/reports/wex2-scale-tiles-20261018.it.md) (Italian) | `c/wex_format.h`; writer `repack_v4_wex.py --format-version 2` |
| **WEX3** | `0x57455833`, 3 | coded records, aliased records (flag 4), one WNM2 model found through the `WNMD` descriptor at header byte 64 | any of 1, 2, 4, 8, 16, 32, 64; 32 excludes 4, 8 and 16; 64 needs 32 | 0 to 5 | read path kept; no production file left (the V4.1 WNR3 file became WEX4 in place, `downgrade` gives it back) | [WNR2-REFERENCE](docs/WNR2-REFERENCE.md); [WNR3-SPEC](docs/WNR3-SPEC.md) section 2 | `c/wex_format.h`, `c/wex_decode.c`, `c/wex_cuda.cuh`; writers `repack_v4_wex.py` (codecs 1 to 3), `archive/nr2/build_nr2_full.py` (4), `c/tools/build_nr3.py` (5) |
| **WEX4** | `0x57455834`, 4.0 | a SHA-256-hashed header page and index (blob, record and tensor tables, names, metadata), a SHA-256 per record and blob, exact tiling of the file, any tensor and dtype, in-place upgrade and downgrade of WEX1 and WEX3/WNR3 files | `required_features` bit 0 VALUE_EXACT; `optional_features` bit 0 ST_SOURCES | 0, 5, 6 | **in production** for both models | [WEX4-SPEC](docs/WEX4-SPEC.md) | reader `c/wex4.h`; `c/qwen38_wex.h`; the DeepSeek loader is in the engine build and shown here as patches in `integration/`; reference `wex4/wex4.py`; converter `wex4/wex4_convert.py`. Not read by `c/expert_store_wex.c` |

| Id | Record codec | Magic on disk | WEX3 header flag | Payload | Lossless | Check inside the record | Decoders | Status |
|---|---|---|---|---|---|---|---|---|
| 0 | RAW | none | none | the MXFP4 slab, zero-padded to 4096 | identity | none; a WEX4 index holds its SHA-256 | none needed | **production** (Qwen3.8) |
| 1 | scale tiles | `WCS1` | 1 | scales in tiles; weights raw | yes | structure | CPU | never built for a whole model |
| 2 | palette3 | `WCP3` | 2 | 3-bit palette weights, scale tiles | **no, lossy** | structure | CPU | never used in production |
| 3 | WNR1 | `WNR1` | 8 | rANS with the CDF inside each record | yes | structure | CPU, CUDA | retired |
| 4 | WNR2 | `WNR2` | 16 | rANS with the container's WNM2 model: 567 contexts per matrix, 14-bit CDFs, u16 stream lengths | yes | structure only: 1.8-1.9% of random bit flips accepted | CPU, CUDA (helper feature 4) | superseded on 2026-11-10; decoder kept |
| 5 | WNR3 | `WNR3` | 32, plus 64 for canonical zero | the WNR2 skeleton plus ceil form, rule-g anchors, optional canonical zero (FP4 -0 stored as +0) and the record's owner | yes; with canonical zero value-exact, not byte-exact | 23-bit MurmurHash3 per stream, owner | CPU, CUDA (helper feature 16) | **production** (V4.1, canonical zero) |
| 6 | WBP1, WEX4 only | `WBP1` | not applicable | byte-plane order-0 rANS for FP8, BF16, F32 and integers | yes | hash per stream or chunk, owner | CPU | written and fuzzed; no production file, no engine loader |

Container magics are little-endian integers, so a WEX4 file starts with the bytes `34 58 45 57`; record magics read forward. The WNM2 model is a 59,408-byte table of CDFs built from symbol counts; no neural network is involved, although the constants say `NEURAL_RANS` ([WNR2-REFERENCE](docs/WNR2-REFERENCE.md)). The CUDA helper `kaira_wex_cuda.dll` (source `c/tools/wex_cuda_bridge.cu`) reports feature bits 1 decode into VRAM, 2 host register, 4 WNR2, 8 batched decode into VRAM, 16 WNR3, 32 batched decode into host memory. Cells checked against `c/wex_format.h`, `c/wex4.h`, `c/wex_neural_rans.h` and `c/wex_palette.h`; the Status and Decoders cells of WEX2 and of codecs 1 to 3 come from [WNR2-REFERENCE](docs/WNR2-REFERENCE.md) and the integration guide and were not re-verified.

## File layout

A WEX4 file in the "upgraded" layout: what `wex4_convert.py upgrade` leaves, and the form of both production files. All integers are little-endian and every extent starts on a 4096-byte boundary.

```text
offset 0             HEADER PAGE (4096 B)
                        0 magic 0x57455834      4 version_major = 4     6 version_minor
                        8 required_features    16 optional_features    24 file_bytes
                       32 index_offset         40 index_bytes
                       48 blob_count           52 record_count         56 tensor_count
                       60 num_layers           64 num_experts          68 max_decoded
                       72 names_bytes          80 meta_bytes           88 total_decoded
                       96 index_sha256 [32]   128 reserved, zero
                     4064 header_sha256 [32] = SHA-256 of bytes [0, 4064) = the ROOT
offset 4096          BLOB kind 1    WNM2 model, 59,408 B (only in files with WNR3 records)
                     BLOB kind 17   the old WEX1/WEX3 entry table, 72 B per (layer, expert)
                     RECORD 0 .. record_count-1   stored_bytes each (a multiple of 4096), one read
                        codec 0 RAW    the MXFP4 slab: scales w1 w2 w3, FP4 weights w1 w2 w3, zero padding
                        codec 5 WNR3   32 B header (magic, model_id, SP, WP, Cw, Cs, anchor[3], flags, owner),
                                       u16 stream lengths, scale payload, weight payload, zero padding
                     BLOB kind 16   the old header page, 4096 B (what downgrade restores)
offset index_offset  INDEX (index_bytes = 64*blob_count + 72*record_count + 100*tensor_count
                            + names_bytes + meta_bytes)
                        blob table     64 B each: kind, id, offset, bytes, sha256
                        record table   72 B each: offset, stored_bytes, decoded_bytes, codec, flags, blob,
                                       owner, first_tensor, tensor_count, sha256
                        tensor table   96 B each: name_offset, name_bytes, dtype, rank, shape[8], record,
                                       record_count, offset, nbytes
                        name order (4 B per tensor), name pool, metadata
offset file_bytes    end of file: nothing may follow

hash tree:  ROOT -> index_sha256 -> sha256 of each record and blob -> 23-bit hash of each WNR3 stream
```

Field names and offsets are those of [WEX4-SPEC](docs/WEX4-SPEC.md) sections 3 to 5 and of `wex4_parse_header` in `c/wex4.h`. Sizes of the real files, from the reports (the files were not opened for this page): V4.1 has 7,680 WNR3 records, 46,080 tensors, an index of 6,712,992 B and 123,312,910,336 B in all; Qwen3.8 has 24,576 RAW records, an index of 26,086,592 B, no model blob and 64,251,047,936 B in all. A file written from scratch (`wex4.py`, `wex4_convert.py pack`) puts the index right after the header page. WEX1 to WEX3 are the header page, the entry table and the records, with the WNM2 model before the table when the records are WNR2 or WNR3; an all-zero entry means that expert is served from the safetensors.

## Results at a glance

| Model | Result | Conditions | Report |
|---|---|---|---|
| DeepSeek V4.1 REAP-192 | WNR3 with canonical zero: 123,306,192,896 B, **0.854** of the raw WEX1 container and 0.949 of WNR2 | whole container, 7,680 records, built and read back 7,680 of 7,680; value-exact, not byte-exact | [WNR3-PRODUCTION](docs/reports/WNR3-PRODUCTION-20261110.md) |
| DeepSeek V4.1 REAP-192 | WNR3 accepted **0** wrong single-bit flips of 2,231,336 on the GPU and 0 of 3,000 on the CPU; WNR2 accepted 1.8-1.9% of random flips | GPU: every bit of chosen regions of 3 real records; CPU: random flips over a whole real record | [WNR3-VERIFY](docs/reports/WNR3-VERIFY.md) |
| DeepSeek V4.1 REAP-192, RTX 5090 | second drive: 2.02 -> **2.99 tok/s** (+49%), token ids and logits bit-identical | one pair, 7 prompts x 64 tokens; A is WEX1 on U:, B is WNR2 on U: + C: split 38/62, so format and split are not isolated | [NOTE-5](research/wnr2-fastbench-20261109/NOTE-5-mirror-ab-20261110.md) |
| DeepSeek V4.1 REAP-192, RTX 5090 | WEX4 around the same records: +6,717,440 B, 2.077 against 2.070 tok/s (**+0.32%**, inside the run-to-run noise), same tokens and logits | U: only, 7 prompts x 64 tokens, 2 + 2 runs one after the other, not interleaved | [wex4-m3 RESULT](research/wex4-m3-20261111/RESULT.md) |
| Qwen3.8-Flash-Next MXFP4, RTX 5090 | WNR3 is **0.880** of WEX1 on disk but slower to read: 9.48 against 11.48 tok/s (258-token prompt), 5.27 against 7.18 (2k prompt). These containers keep RAW records | one run per arm, U: only, RAM 13 GB; no canonical zero (the export has no FP4 -0) | [QWEN38-MXFP4](docs/reports/QWEN38-MXFP4-20261111.md) |

The V4.1 switch report's own reading of its numbers: on this machine the second drive is the lever, not the format.

## What runs today

Summary of 2026-11-13; [STATUS.md](STATUS.md) is the page of record, with the roots, the switches, the engine builds and what is switched off.

| Model | Container | Records | Bytes per copy | Drives | Integrity |
|---|---|---|---|---|---|
| DeepSeek V4.1 REAP-192 (MXFP4) | WEX4 since 2026-11-11 (WNR3 since 2026-11-10) | 7,680 WNR3 with canonical zero | 123,312,910,336 | U: plus a full mirror on C:, reads split 38/62 | root pinned; SHA-256 of each record at its first read |
| Qwen3.8-Flash-Next (MXFP4 routed experts) | WEX4 since 2026-11-11 | 24,576 RAW | 64,251,047,936 | U: plus a full mirror on C:, reads split 38/62 | both copies verified on 2026-11-13; root recorded, not pinned; header and index checked at open |

Every measurement table: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Format changes by date: [CHANGELOG.md](CHANGELOG.md).

## Try it in ten minutes

No GPU and no model are needed. Prerequisites: an x86-64 CPU with AVX2 (one test builds with `-march=haswell`) or a 64-bit ARM CPU (there that test builds the scalar kernel of the fast decoder instead), `bash`, `gcc` and Python 3 (standard library only).

- **Windows:** [MSYS2](https://www.msys2.org/) with the UCRT64 packages `mingw-w64-ucrt-x86_64-gcc`, `mingw-w64-ucrt-x86_64-libgomp` (two tests link with `-fopenmp`) and `mingw-w64-ucrt-x86_64-python` (or Git Bash with that `gcc` on `PATH`).
- **Linux:** `gcc` and `python3`, and `export PYTHON=python3`. Linux is checked by CI (ubuntu-24.04, on x86-64 and on ARM64); the maintainer's machines are Windows.

From the repository root:

```sh
bash scripts/check_all.sh
```

It runs the repository check, the standard-library tests, the WEX1-3 and WNR3 CPU check (format tests, `wex4_verify`, the encoder DLL, a WNR3 container built from `fixtures/v41-tiny`) and the WEX4 tests (reader, converters, 46 hostile files, mutation fuzzing in C and in Python, the conformance vectors). The last line on success is `all checks passed`, exit code 0; one run took 2 min 5 s on 2026-11-13 (RTX 5090 desktop, gcc 16.2, Python 3.14; the GPU is not used). Output goes to `out/cpu-check/` and `wex4/out/`, which git ignores. To look at a WEX4 file, upgrade a tiny container that the WEX4 tests left behind, and put it back:

```sh
python -I wex4/wex4_convert.py upgrade wex4/out/tiny_cz.wex      # WEX3 with WNR3 records -> WEX4, in place
python -I wex4/wex4.py verify wex4/out/tiny_cz.wex --dll wex4/out/wex4_tools.dll   # root, counts, every hash
python -I wex4/wex4_convert.py downgrade wex4/out/tiny_cz.wex    # the same bytes as before the upgrade
```

## Documentation

The index of every page, with its kind, language and state, is [docs/README.md](docs/README.md); the terms are in [docs/GLOSSARY.md](docs/GLOSSARY.md).

- **Integrate the formats into a colibri tree and run them:** [docs/COLIBRI-INTEGRATION.md](docs/COLIBRI-INTEGRATION.md). `integration/` holds the engine, backend and host excerpts, the read-path patch for stock colibri at 561966ec (`upstream-561966ec-levelA.patch`: WEX1 and WNR2) and the three WEX4 loader patches. `integration\build\build_wex_windows.cmd` builds the CUDA helper and the checkers (MSVC and CUDA needed).
- **Implement a reader or a writer:** [docs/WEX4-SPEC.md](docs/WEX4-SPEC.md) for the container, [docs/WNR3-SPEC.md](docs/WNR3-SPEC.md) for the record codec, [docs/WNR2-REFERENCE.md](docs/WNR2-REFERENCE.md) for WEX1 to WEX3 and WNR2. Reference code: `c/wex4.h` and `wex4/wex4.py` (written independently of each other), `c/wex_format.h`, `c/wex_neural_rans.h`.
- **Check a result:** [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the measurement tables, `docs/reports/` for the dated reports behind them, [research/README.md](research/README.md) for the runs with their scripts and raw output.

## Layout

```text
README.md  STATUS.md  CHANGELOG.md  CONTRIBUTING.md   front page, what runs today, format changes, rules
SECURITY.md  CITATION.cff     how to report a vulnerability, how to cite
LICENSE  NOTICE  THIRD_PARTY_NOTICES.md              Apache-2.0 text, attribution notices, third-party material
check_repo.py  sync-map.tsv    self-check (links, c/ file map, licence headers, privacy) and the map of c/ files to the maintainer's tree
assets/        the logo (light and dark), the mark, and the source image of the social preview (set in the repository settings)
.github/       CI workflow (Linux and Windows CPU checks), issue forms, pull-request checklist
c/             engine-side WEX code in the colibri c/ layout; pull requests may change it
  *.h *.c *.cuh   format, codecs, CPU and CUDA decoders, the standalone store, wex4.h, qwen38_wex.h
  tools/          container writers, the WNR3 builder and its encoder DLL, the CUDA helper, benches
  tests/          format, store, repack, WNR3 and Qwen3.8 tests; several build only inside a full engine tree
wex4/          WEX4, repo-only: Python reference, converter, C tools, tests, reviews, research
  vectors/        conformance vectors: valid and invalid files with the verdict a reader must give
scripts/       repo-only checks: check_all.sh (every CPU check), build_cpu_check.sh, verify_wex1.py and its test
archive/       the WNR2 lab code (encoder, trainer, checks) and its benches, kept for the record
models/        trained WNM2 models: DeepSeek V4 Flash 0731 and V4.1 REAP-192
fixtures/      the tiny V4.1 model and the makers of synthetic experts
integration/   engine, backend and host excerpts, build scripts, patches
docs/          living pages: index, architecture, glossary, benchmarks, integration guide, WNR2 reference, WNR3 and WEX4 specs
  reports/        dated, frozen reports (some in Italian)
research/      dated lab runs, frozen; README.md is the index
docs/NEXT-FORMAT-SPEC.md         stub: a comment in c/wex4.h still names this path
```

## Where the code lives

The 37 files under `c/` are this repository's engine-side code, and pull requests may change them. The same files are also carried in the maintainer's application tree (under `src/colibri/c/`, same relative paths) and are being integrated into colibri's `dev` branch; the maintainer keeps the copies in step, and `sync-map.tsv` maps each file to its path there. `archive/` came from the maintainer's lab folder, `integration/` quotes engine code that is not in this repository, `research/`, `models/` and `fixtures/v41-tiny` are copies of lab results, the rest was written here. To check the links, the file map, the licence headers and the privacy patterns:

```sh
python -I check_repo.py
```

It exits with 1 when a relative Markdown link or its anchor is broken, when a file under `c/` has no row in `sync-map.tsv`, when a source file outside `research/` has no SPDX licence header, or when a tracked file holds a user-profile path, a personal e-mail address or an access token. The maintainer's comparison with the other tree (`--source <TREE>`) and what may be written here: [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Copyright 2026 Ludovico Zanichelli. Everything in this repository (code, specifications, documentation, models, fixtures and research data) is licensed under the [Apache License 2.0](LICENSE), except the third-party material listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which keeps its own licence. Source files start with an SPDX header (`SPDX-License-Identifier: Apache-2.0`); the frozen run folders under `research/` mostly have none and are covered by this paragraph.

- [NOTICE](NOTICE) holds the attribution notices that section 4(d) of the licence asks you to keep when you redistribute this work, including the one of [colibri](https://github.com/JustVugg/colibri) (Apache-2.0, Copyright 2026 Vincenzo Fornaro). The files derived from colibri are listed there, and each one names its colibri source in its header and says that it was modified.
- If you redistribute this repository or a work based on it, include `LICENSE` and `NOTICE`, keep the notices in the files, and mark the files you changed.

**How to cite.** [CITATION.cff](CITATION.cff) has the metadata (GitHub shows it as "Cite this repository"). Name the author, the title, the year and the commit you used:

```text
Ludovico Zanichelli. WEX: expert-container formats for streamed Mixture-of-Experts inference
(WEX1-WEX4 containers, WNR2/WNR3 record codecs). 2026. https://github.com/KairaCode/Wex-WRN2, commit <sha>.
```

## Security

A WEX file is untrusted input for the WEX4 readers. How to report a vulnerability, which code is supported and what the formats do not protect against: [SECURITY.md](SECURITY.md).

## Model weights

This repository ships no model weights, and it does not accept them: real containers (`*.wex`, git-ignored) and checkpoints stay outside it. A WEX container built from a checkpoint holds that checkpoint's weights and is under the checkpoint's licence, not this one. What is here instead:

- `models/*.wnm2` and the lab models under `research/`: symbol-frequency tables computed from the routed experts of DeepSeek V4 Flash 0731 and DeepSeek V4.1 Flash 1017 REAP-192 (both MIT, Copyright (c) 2023 DeepSeek). They are statistics, not weights.
- `fixtures/v41-tiny/`: a synthetic test model with seeded random weights at toy shapes, written by colibri's `tools/make_dsv41_tiny.py`.
- `research/`: measurements (counts, sizes, entropies, hashes) of records of those checkpoints, and, under `research/wnr3-20261107/team3/other-tensors/data/`, copies of DeepSeek inference code, a model card and safetensors headers (MIT; licence text next to them).

The source and licence of each of these is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A fixture or data file derived from real weights is added only together with a row there that names its source and licence.

## Contributing

Contributions are welcome under the Apache License 2.0, inbound = outbound; a contributor's commits carry a `Signed-off-by` line (Developer Certificate of Origin). The rules, including the privacy rules and how `c/` is kept in step with the other trees that carry it, are in [CONTRIBUTING.md](CONTRIBUTING.md). Tasks to pick up: [docs/HELP-WANTED.md](docs/HELP-WANTED.md).

## In breve (italiano)

- **Cos'è.** WEX è una famiglia di formati contenitore per gli expert dei modelli Mixture-of-Experts: un file di record allineati a 4096 byte, uno per expert, che il motore legge da NVMe con una sola lettura mentre genera. WEX4 aggiunge un header e un indice con lo SHA-256 di ogni record; WNR3 è la codifica rANS senza perdita dei record.
- **Cosa gira oggi.** DeepSeek V4.1 REAP-192 legge un WEX4 con record WNR3 (0,854 del formato grezzo), Qwen3.8-Flash-Next un WEX4 con record MXFP4 grezzi; entrambi da due drive NVMe (U: e C:, letture divise 38/62). Stato aggiornato: [STATUS.md](STATUS.md).
- **Come provarlo.** `bash scripts/check_all.sh` compila e lancia tutti i test su CPU, senza GPU e senza modelli; finisce con `all checks passed`.
- **Da dove cominciare.** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) per l'insieme, [docs/WEX4-SPEC.md](docs/WEX4-SPEC.md) per i byte, [docs/COLIBRI-INTEGRATION.md](docs/COLIBRI-INTEGRATION.md) per collegare un motore, [CONTRIBUTING.md](CONTRIBUTING.md) per contribuire, [SECURITY.md](SECURITY.md) per segnalare una vulnerabilità.
- **Licenza.** Apache 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Le pagine sono in inglese; alcuni report datati in `docs/reports/` sono in italiano.
