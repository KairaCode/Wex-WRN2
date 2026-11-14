# Architecture

How the pieces of this repository fit together: which file reads or writes what, the path a record takes from a checkpoint to the GPU, where the trust boundaries are, and which parts are not in this repository. The byte layouts are in the specifications ([WEX4-SPEC](WEX4-SPEC.md), [WNR3-SPEC](WNR3-SPEC.md), [WNR2-REFERENCE](WNR2-REFERENCE.md)); how to wire an engine is in [COLIBRI-INTEGRATION](COLIBRI-INTEGRATION.md). Terms: [GLOSSARY](GLOSSARY.md).

## Components

| Component | Files | Role |
|---|---|---|
| WEX1-WEX3 format and codecs | `c/wex_format.h`, `c/wex_neural_rans.h`, `c/wex_palette.h` | header-only C: the 40-byte `WexHeader` and 72-byte `WexEntry`, their validation, the rANS core (WNR1, WNR2, WNR3), the WNM2 model, palette3. Every other C file includes `wex_format.h` |
| CPU decode and helper loader | `c/wex_decode.c`, `c/wex_decode.h` | decodes a record on the CPU and loads the optional CUDA helper (`coli_wex_*` functions) |
| CUDA helper | `c/wex_cuda.cuh`, `c/tools/wex_cuda_bridge.cu` -> `kaira_wex_cuda.dll` | GPU decoders. Feature bits: 1 decode into VRAM, 2 host register, 4 WNR2, 8 batched decode into VRAM, 16 WNR3, 32 batched decode into host memory |
| WEX4 reader | `c/wex4.h` | header-only C: header page and index validation (WEX4-SPEC section 6), SHA-256 (with a SHA-NI path), the WBP1 codec, record decode through `wex_format.h` |
| WEX4 checker | `c/tools/wex4_verify.c` | hashes every blob and record of a WEX4 file against its index, on 8 threads by default, and prints `COLI_WEX_ROOT=<root>`; exit 0 pass, 1 mismatch, 2 not WEX4. It does not decode |
| Qwen3.8 store | `c/qwen38_wex.h` | the qwen38 engine's reader of WEX1, WNR3 and WEX4 files; included by the engine's `qwen38_core.h`. Its header comment lists its environment variables |
| Standalone store | `c/expert_store_wex.c`, `c/expert_store_wex.h` | an expert store for WEX1-WEX3 behind colibri's `COLI_EXPERT_STORE` interface; it does not read WEX4 |
| Writers | `c/tools/repack_v4_wex.py` (DeepSeek V4 / V4.1), `c/tools/repack_q38_mxfp4_wex.py` (Qwen3.8 MXFP4), with `c/tools/wex_neural_rans.py` and `c/tools/wex_palette.py` for codecs 1-3 | safetensors -> WEX1 (or WEX2 / WEX3 with codecs 1-3) |
| WNR3 builder | `c/tools/build_nr3.py`, `c/tools/wex_nr3_tools.c` -> `wex_nr3_tools.dll` | trains a WNM2 model and writes WNR3 records (`--cz` for canonical zero) into a WEX3 container; checks every record on the CPU |
| WEX4 reference and converter | `wex4/wex4.py`, `wex4/wex4_convert.py` | an independent Python reader and writer (written apart from `c/wex4.h`, so that the two can be compared), and `upgrade` / `downgrade` / `pack` |
| WEX4 test library | `wex4/wex4_tools.c` -> `wex4_tools.dll` | `c/wex4.h` behind a C ABI for the Python tests, with the fast WNR3 CPU decoder and an in-process fuzzer |
| Second WEX4 reader | `impl/go-wex4/` | a reader and verifier in Go (standard library only) written from the specification pages and not from the two readers above: header page, index, every hash, the RAW, WBP1 and WNR3 decoders. It gives the verdicts of the conformance vectors; no engine uses it |
| Checks | `check_repo.py`, `scripts/`, `wex4/build_and_test.sh`, `wex4/vectors/` | the repository self-check, the CPU builds and tests, the conformance vectors |

## Write path

```text
safetensors checkpoint
   |  c/tools/repack_v4_wex.py  (DeepSeek V4, V4.1)      one 4096-aligned RAW record per (layer, expert)
   |  c/tools/repack_q38_mxfp4_wex.py  (Qwen3.8 MXFP4)
   v
WEX1 file ---------------------------------------------------------------+
   |  c/tools/build_nr3.py DLL WEX1 container OUT.wex --cz                  |
   |  (WNM2 model trained on chosen experts, every record CPU-checked)      |
   v                                                                        |
WEX3 file with WNR3 records (codec 5) and the WNM2 model                    |
   |                                                                        |
   +--> wex4/wex4_convert.py upgrade FILE  <--------------------------------+
           in place: appends the old header page, the old entry table and the index,
           rewrites the header page; every record stays where it was
   v
WEX4 file  ---- wex4/wex4_convert.py downgrade FILE ----> the same bytes as before the upgrade
   |
   +--> python -I wex4/wex4.py verify FILE --dll wex4_tools.dll   all hashes, decodes every record
   +--> tools/wex4_verify FILE                                    all hashes, prints COLI_WEX_ROOT=
```

A file written from scratch (`wex4.py`, `wex4_convert.py pack`) puts the index right after the header page and can hold any tensor, not only experts. With canonical zero, WNR3 is value-exact but not byte-exact: the WEX1 bytes cannot be rebuilt from it (an FP4 -0 comes back as +0), only from the checkpoint.

## Read path

This is the DeepSeek engine's path, as implemented in the engine's `deepseek_v4.c` (not in this repository; see [integration/](../integration/README.md)). The Qwen3.8 path in `c/qwen38_wex.h` follows the same steps without GPU-direct decode.

```text
open      header page -> index (SHA-256 checked) -> root compared with COLI_WEX_ROOT when it is pinned
          mirrors: the same file name in each COLI_WEX_EXTRA_MIRRORS folder; equal header pages = same container
             |
prewarm   at start, the VRAM tier is filled with the most used experts: one record at a time,
          COLI_V4_TIER_PREWARM_AHEAD records ahead, or batches of 8 (V4_PREWARM_BATCH=1)
             |
miss      the router picks an expert that is not resident
             |
read      one unbuffered read of [offset, offset + stored_bytes), 4096-aligned, from the drive that
          hash(layer, expert) picks with the shares of COLI_WEX_WEIGHTS (38,62 on the test machine);
          optionally N overlapped sub-reads per drive (V4_WEX_SUBREADS), and the record cut across
          primary and mirror on a 4096-byte page (V4_WEX_STRIPE, V4_WEX_SUBREADS)
             |
first     SHA-256 of the stored bytes against the index, the first time this process reads the record
read      (with a pinned root, or COLI_WEX_VERIFY=1); optionally on a 4-thread pool (V4_WEX_SHA_ASYNC=1).
          A record that fails stays refused for the rest of the process
             |
decode    RAW: used as it is.  WNR3: the CUDA helper decodes it straight into VRAM
          (COLI_WEX_GPU_DIRECT=1/2, a layer's misses in one call with V4_WEX_BATCH=1), or the CPU
          decoder into the RAM slab. Every rANS stream checks its 23-bit hash inside the decoder
             |
compute   the expert's MXFP4 weights feed the MoE matmul (GPU or CPU)
```

Defaults and the file where each variable is read: [COLIBRI-INTEGRATION](COLIBRI-INTEGRATION.md), section 5.

## Trust boundaries

- **The container is untrusted.** Both WEX4 readers validate the header page and the whole index before they use a byte of it (WEX4-SPEC section 6), every record extent is checked against the file, and decoders write into scratch memory that is used only when they return success. The threat model is WEX4-SPEC section 9; how to report a problem is [SECURITY.md](../SECURITY.md).
- **The root is the anchor.** It ties every byte of the file to 32 bytes; it protects only when it is pinned from a place the attacker cannot write (the engine configuration). Without a pin, the hashes detect damage, not substitution.
- **WEX1-WEX3 files are trusted input.** They carry no hashes (WNR3 records carry their per-stream hashes, which detect corruption, not substitution).
- **Code is trusted.** The CUDA helper and the ctypes libraries (`wex4_tools.dll`, `wex_nr3_tools.dll`) are loaded from the path they are given.

## What is not in this repository

WEX was developed for KAIRA, a private local inference application: a Windows host that starts engines built from the maintainer's own tree of [colibri](https://github.com/JustVugg/colibri), a C inference engine for Mixture-of-Experts models. That tree carries the code of `c/` under `src/colibri/c`. Not in this repository:

- the DeepSeek engine (`deepseek_v4.c`, `deepseek_v4_internal.h`) with its expert store, the read path above and the GPU tiers; `integration/` holds excerpts and patches of it;
- the qwen38 engine (`qwen38.c`, `qwen38_core.h`) around `c/qwen38_wex.h`;
- the host (C#) and its profiles. An **engine profile** is a named engine configuration in the host's configuration file: which executable, which container and mirror folders, which environment variables. `v41gpu` (DeepSeek V4.1) and `qwen38` (Qwen3.8) are the two that read WEX files.

The pages name a few things of the test machine: **U:** and **C:** are its two NVMe drives (U: PCIe 4, C: PCIe 5), which hold the two copies of each container. Paths of the form `<TREE>/lab/...` are folders of the maintainer's lab and cannot be opened from here; where a page quotes a lab run, `research/` holds a copy of its scripts and logs, and the page links that copy.

## The `c/` folder

The 37 files under `c/` are this repository's engine-side code. The maintainer's application tree carries the same files without this repository's two- or three-line licence header (`sync-map.tsv` records the last sync; `check_repo.py --source` compares them without the header). Many of them build only inside an engine tree, because they include engine files that are not here.

| Files | Builds or runs here | Needs an engine tree | Needs a GPU |
|---|---|---|---|
| `wex_format.h`, `wex_neural_rans.h`, `wex_palette.h`, `tests/test_wex_format.c` | yes: `scripts/build_cpu_check.sh` | no | no |
| `wex4.h` | yes: `wex4/build_and_test.sh` (through `wex4/wex4_tools.c`) and `tests/test_wex4_verify.c` | no | no |
| `tools/wex4_verify.c`, `tests/test_wex4_verify.c`, `tests/wex4_fixture.h` | yes: `scripts/build_cpu_check.sh` | no (`test_qwen38_mxfp4.c` also includes the fixture) | no |
| `tools/wex_nr3_tools.c`, `tools/build_nr3.py`, `tools/repack_v4_wex.py`, `tools/wex_neural_rans.py`, `tools/wex_palette.py` | yes: both CPU scripts build a WNR3 container from `fixtures/v41-tiny` | no | no |
| `tests/test_repack_v4_wex.py`, `tests/test_build_nr3_picks.py` | yes: `scripts/check_all.sh` | no | no |
| `tests/fake_wex_cuda.c` | compile only: `scripts/build_cpu_check.sh` | its users do (`test_v4_wex_nr3.c`, `test_qwen38_mxfp4.py`) | no |
| `wex_decode.c`, `wex_decode.h` | compiled by `scripts/build_cpu_check.sh` only when it is given an engine tree | no for the file itself | no (the helper is optional) |
| `expert_store_wex.c`, `expert_store_wex.h`, `tests/test_expert_store_wex.c` | `scripts/build_cpu_check.sh <TREE>/src/colibri/c` | yes: `expert_store.h`, `expert_store_registry.c`, `deepseek_v4_internal.h`, `compat.h` | no |
| `qwen38_wex.h`, `tests/test_qwen38_mxfp4.c` | no | yes: `qwen38.c`, `qwen38_core.h`, `compat.h` | no |
| `tests/test_v4_wex_nr3.c`, `tests/test_v4_gpu_priority.c` | no | yes: `tests/test_deepseek_v4.c` and `deepseek_v4.c` | no |
| `tests/test_wex_planned.c` | `integration/build/build_test_wex_planned.sh`, in an engine tree | yes: `deepseek_v4_internal.h` | no |
| `tests/test_v4_mirror_guard_source.py` | no | yes: it reads `deepseek_v4.c` | no |
| `tests/test_qwen38_mxfp4.py` | no | yes: the qwen38 engine build and its test fixtures | no |
| `tests/make_qwen38_tiny_mxfp4_experts.py`, `tools/make_qwen38_tiny_mxfp4.py` | no | yes: the engine's Qwen3.8 test fixtures; they also need numpy, torch and safetensors | no |
| `tools/repack_q38_mxfp4_wex.py` | with numpy and a Qwen3.8 MXFP4 checkpoint | no | no |
| `wex_cuda.cuh`, `tools/wex_cuda_bridge.cu` | `integration/build/build_wex_windows.cmd` (MSVC and the CUDA toolkit) | no | to run |
| `tests/test_wex_nr3_gpu.cu`, `tools/bench_wex_cuda.cu` | with nvcc, by hand | no | yes |
| `tools/bench_wex_decode.c` | Windows only, by hand (`windows.h`); takes two container files | no | no |
| `tools/wexio_probe.py` | Windows only: `--self-test` reads a synthetic temp file; its default inputs are the lab's container paths on U: and C: | no | no |

`scripts/check_all.sh` runs everything in the "yes" rows. The rest is exercised in the maintainer's engine tree.

## How a change to `c/` gets in

Like any other file: by a pull request here ([CONTRIBUTING.md](../CONTRIBUTING.md), rule 1). The maintainer runs the engine tests that cannot run here and carries the change into the application tree and the colibri integration.
