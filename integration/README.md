# integration/

What an engine needs around the files in [`../c/`](../c/): quotes of the engine code that calls them, the patches that add the loaders, and the build scripts. The step-by-step guide is [docs/COLIBRI-INTEGRATION.md](../docs/COLIBRI-INTEGRATION.md); the WEX4 loader is specified in [docs/WEX4-SPEC.md](../docs/WEX4-SPEC.md).

**Most excerpts are older than the current engine.** They quote the maintainer's engine tree as it was on 2026-11-07; one of them, and one range of `build/Makefile.wex-excerpt.mk`, as it was on 2026-11-10; the WEX4 one as it was on 2026-11-13. The files under `c/` are newer than all of them. An excerpt is a frozen quote: it is not compilable, its line numbers are those of the tree on the date in its header, and it is not kept in step with the engine. It was not compared with the current engine for this page. Read it to see where the hooks go, then read the same function in the engine tree you are patching.

Kinds: **excerpt** = quote of engine or host code, with the source file and line ranges in its header; **patch** = a diff to apply; **history** = a diff kept for the record, to apply nowhere; **build script** = runs from this repository.

## Engine: `deepseek_v4.c`

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `engine/deepseek_v4.mode0-store.wex-excerpt.c` | excerpt | maintainer tree, 2026-11-07 | `deepseek_v4.c`, expert-store unit: container open, WNR2 guard, mirrors, drive pick, record read into the RAM slab (level A, enough for WEX1 / WEX2 / WEX3 / WNR2 without GPU-direct) |
| `engine/deepseek_v4.direct-store.wex-excerpt.c` | excerpt | maintainer tree, 2026-11-07 | `deepseek_v4.c`: store side of GPU-direct (`COLI_WEX_GPU_DIRECT=1/2`) and the compressed-record RAM cache (level B; needs the mode 0 excerpt) |
| `engine/deepseek_v4.direct-gpu.wex-excerpt.c` | excerpt | maintainer tree, 2026-11-07 | `deepseek_v4.c`, GPU unit: mirror cache, fill producer, attach and prefetch entry points (level C; uses helpers that upstream lacks, listed in its header) |
| `engine/deepseek_v4_internal.wex-excerpt.h` | excerpt | maintainer tree, 2026-11-07 | `deepseek_v4_internal.h`: declarations of the WEX engine hooks |
| `engine/deepseek_v4.wnr3.wex-excerpt.c` | excerpt | maintainer tree, 2026-11-10 (WNR3 branch) | `deepseek_v4.c` and `deepseek_v4_internal.h`: WNR3 (codec 5) and canonical zero in the engine's store; needs the three excerpts above |
| `engine/deepseek_v4.wex4.wex-excerpt.c` | excerpt | maintainer tree, 2026-11-13 | `deepseek_v4.c`: the WEX4 open and the pinned root, the first-read SHA-256 and its hash pool (`V4_WEX_SHA_ASYNC`), the refusal of a legacy container under a pinned root, sub-reads (`V4_WEX_SUBREADS`) and the batched tier prewarm (`V4_PREWARM_BATCH`). It shows the code after the three `engine-*` patches below |
| `history/wex-gpu-direct.patch` | history | no commit in the file; the guide calls it the October 20 GPU-direct diff against the tree before GPU-direct, 16 files | reference only |

## CUDA backend and loader

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `backend/backend_cuda_dsv4.wex-excerpt.cu` | excerpt | maintainer tree, 2026-11-07 | `backend_cuda_dsv4.cu`: the exports GPU-direct needs, and `DSV4_CUDA_STREAM_PRIORITY` |
| `backend/backend_cuda_dsv4.wex-excerpt.h` | excerpt | maintainer tree, 2026-11-07 | `backend_cuda_dsv4.h`: declarations of those exports |
| `backend/backend_loader_dsv4.wex-excerpt.c` | excerpt | maintainer tree, 2026-11-07 | `backend_loader_dsv4.c`: Windows loader of `coli_cuda_dsv4*.dll`, optional symbols |
| `backend/dsv4.wex-excerpt.def` | excerpt | maintainer tree, 2026-11-07 | `dsv4.def`: the export list of the backend DLL |
| `backend/backend-upload-race-fix.patch` | history | no commit in the file | `backend_cuda_dsv4.cu`, `upload()`. Apply nowhere: the guide of 2026-11-09 ([docs/reports/colibri-integration-20261109.md](../docs/reports/colibri-integration-20261109.md), section 5.4) says the engine tree of 2026-11-07 and upstream 561966ec already hold an equivalent fix |
| `backend/build_backend_dg.cmd` | build script | none | builds `coli_cuda_dsv4_dg.dll` from an engine tree; needs `COLIBRI_C`, `DEEPGEMM_ROOT` and `CUDA_PATH` and says which one is missing |

## Build

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `build/Makefile.deepseek-v4.wex-excerpt.mk` | excerpt | maintainer tree, 2026-11-07 | `Makefile.deepseek-v4`: the rules that link the WEX objects into `deepseek_v4.exe` (Windows) |
| `build/Makefile.wex-excerpt.mk` | excerpt | maintainer tree, 2026-11-07; the WNR3 test rule at the end is from 2026-11-10 | top-level `Makefile`: the WEX unit tests and `wex_decode.o` in the test builds |
| `build/build_wex_windows.cmd` | build script | none | builds from this repository alone: the CUDA helper DLL `kaira_wex_cuda.dll`, `archive\nr2\wex_nr2_tools.dll`, `test_decode_batch.exe`, `check_container.exe`. Overrides: `VCVARS`, `NVCC`, `ARCH` (default sm_120) |
| `build/build_test_wex_planned.sh` | build script | written for a colibri `c/` tree as it was on 2026-11-07 | builds `test_wex_planned.exe` in an engine tree that has this repository's `c/` copied over it (MSYS2 UCRT64) |

## Application host (C#, outside the engine)

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `host/Engine.wex-excerpt.cs` | excerpt | maintainer tree, 2026-11-07 | `src/host/Engine.cs`: how the profile environment and the mirror folder reach the engine process |
| `host/ColibriFamily.wex-excerpt.cs` | excerpt | maintainer tree, 2026-11-07 | `src/host/ColibriFamily.cs`: the V4.1 environment allow-list |
| `host/Config.wex-excerpt.cs` | excerpt | maintainer tree, 2026-11-07 | `src/host/Config.cs`: profile defaults. Its header marks the container path in it as stale, and one path is shortened to `<HOME>` |
| `host/DeepSeekV41Preflight.wex-excerpt.cs` | excerpt | maintainer tree, 2026-11-07 | `src/host/DeepSeekV41Preflight.cs`: a preflight warning that its header marks as stale |

## Patches

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `engine-wex4-loader-75932c1.patch` | patch | engine commit 75932c1 (2026-11-11) | an engine tree with paths under `src/colibri/c/`: `deepseek_v4.c`, `tests/test_v4_wex_nr3.c`, new `wex4.h`. The engine opens WEX4 containers |
| `engine-wex4-first-read-4ba5568.patch` | patch | engine commit 4ba5568 (2026-11-11) | the same tree, on top of the loader patch: `deepseek_v4.c`, `deepseek_v4_internal.h`, `tests/test_v4_wex_nr3.c`, `wex4.h`. SHA-256 of every record on its first read (`COLI_WEX_VERIFY`) |
| `engine-q38-wex4-loader-de17198.patch` | patch | engine commit de17198 (2026-11-11) | an engine tree with paths under `src/colibri/c/`: `qwen38_wex.h`, new `wex4.h`. The qwen38 engine opens WEX4 containers |
| `upstream-561966ec-levelA.patch` | patch | upstream colibri 561966ec; hooks taken from the engine tree of 2026-11-07 | a stock colibri checkout: `c/deepseek_v4.c`, `c/Makefile.deepseek-v4`. Level A only (WEX1 / WNR2 into the RAM slab). Its header says "Verified 2026-11-08" |

Notes on the patches:

- The three `engine-*` patches are `git format-patch` files with paths under `src/colibri/c/`, the layout of the maintainer's application tree; the 7 hex digits in each name are the commit in the file's `From` line. Before 2026-11-14 they were named `kaira-*.patch` with the same suffixes, and older reports under `research/` use those names. The `wex4.h` and `qwen38_wex.h` they add are older than the copies in `../c/`: take the files from `../c/`.
- There is no upstream patch for WNR3 or WEX4. The upstream patch does not mention WNR3, WEX4 or `COLI_WEX_ROOT`; for those, the hooks are the WNR3 excerpt, the three `engine-*` patches and the WEX4 excerpt.
- The header of the upstream patch tells you to copy this repository's `c/*.h` and `c/*.c` ([`../c/`](../c/)) first. On 2026-11-13 `git apply --check` passed on a clean `git archive 561966ec` with today's `c/*.h` and `c/*.c` copied in; the build and the token A/B of 2026-11-08 were not repeated, and `c/` has gained WNR3 and WEX4 since then.
- The engine excerpts of 2026-11-07 and 2026-11-10 were cut before WEX4 existed: they show the WEX3 open path. The WEX4 path, and what came after the three `engine-*` patches, is in `engine/deepseek_v4.wex4.wex-excerpt.c` (2026-11-13).
