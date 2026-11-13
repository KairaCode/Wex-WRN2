# integration/

What an engine needs around the files in [`../c/`](../c/): quotes of the engine code that calls them, the patches that add the loaders, and the build scripts. The step-by-step guide is [docs/COLIBRI-INTEGRATION.md](../docs/COLIBRI-INTEGRATION.md); the WEX4 loader is specified in [docs/WEX4-SPEC.md](../docs/WEX4-SPEC.md).

**Most excerpts are older than KAIRA master.** They quote KAIRA at 88ddb07 (2026-11-07); one of them, and one range of `build/Makefile.wex-excerpt.mk`, at cade51d (2026-11-10); the WEX4 one at 24f7e72 (2026-11-13); [`kaira-sync.tsv`](../kaira-sync.tsv) records master 940275a (2026-11-13) for the mirrored files. An excerpt is a frozen quote: it is not compilable, its line numbers are those of the commit in its header, and it is not kept in step with master. It was not compared with master for this page. Read it to see where the hooks go, then read the same function in the engine tree you are patching.

Kinds: **excerpt** = quote of engine or host code, with the source file and line ranges in its header; **patch** = a diff to apply; **history** = a diff kept for the record, to apply nowhere; **build script** = runs from this repository.

## Engine: `deepseek_v4.c`

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `engine/deepseek_v4.mode0-store.wex-excerpt.c` | excerpt | KAIRA 88ddb07 | `deepseek_v4.c`, expert-store unit: container open, WNR2 guard, mirrors, drive pick, record read into the RAM slab (level A, enough for WEX1 / WEX2 / WEX3 / WNR2 without GPU-direct) |
| `engine/deepseek_v4.direct-store.wex-excerpt.c` | excerpt | KAIRA 88ddb07 | `deepseek_v4.c`: store side of GPU-direct (`COLI_WEX_GPU_DIRECT=1/2`) and the compressed-record RAM cache (level B; needs the mode 0 excerpt) |
| `engine/deepseek_v4.direct-gpu.wex-excerpt.c` | excerpt | KAIRA 88ddb07 | `deepseek_v4.c`, GPU unit: mirror cache, fill producer, attach and prefetch entry points (level C; uses KAIRA-only helpers that upstream lacks, listed in its header) |
| `engine/deepseek_v4_internal.wex-excerpt.h` | excerpt | KAIRA 88ddb07 | `deepseek_v4_internal.h`: declarations of the WEX engine hooks |
| `engine/deepseek_v4.wnr3.wex-excerpt.c` | excerpt | KAIRA cade51d (branch wnr3-20261110, on master 3b604bb) | `deepseek_v4.c` and `deepseek_v4_internal.h`: WNR3 (codec 5) and canonical zero in the engine's store; needs the three excerpts above |
| `engine/deepseek_v4.wex4.wex-excerpt.c` | excerpt | KAIRA 24f7e72 (master, 2026-11-13) | `deepseek_v4.c`: the WEX4 open and the pinned root, the first-read SHA-256 and its hash pool (`V4_WEX_SHA_ASYNC`), the refusal of a legacy container under a pinned root, sub-reads (`V4_WEX_SUBREADS`) and the batched tier prewarm (`V4_PREWARM_BATCH`). It shows the code after the three `kaira-*` patches below |
| `history/wex-gpu-direct.patch` | history | no commit in the file; the guide calls it the October 20 GPU-direct diff against the tree before GPU-direct, 16 files | reference only |

## CUDA backend and loader

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `backend/backend_cuda_dsv4.wex-excerpt.cu` | excerpt | KAIRA 88ddb07 | `backend_cuda_dsv4.cu`: the exports GPU-direct needs, and `DSV4_CUDA_STREAM_PRIORITY` |
| `backend/backend_cuda_dsv4.wex-excerpt.h` | excerpt | KAIRA 88ddb07 | `backend_cuda_dsv4.h`: declarations of those exports |
| `backend/backend_loader_dsv4.wex-excerpt.c` | excerpt | KAIRA 88ddb07 | `backend_loader_dsv4.c`: Windows loader of `coli_cuda_dsv4*.dll`, optional symbols |
| `backend/dsv4.wex-excerpt.def` | excerpt | KAIRA 88ddb07 | `dsv4.def`: the export list of the backend DLL |
| `backend/backend-upload-race-fix.patch` | history | no commit in the file | `backend_cuda_dsv4.cu`, `upload()`. Apply nowhere: the guide of 2026-11-09 ([docs/reports/colibri-integration-20261109.md](../docs/reports/colibri-integration-20261109.md), section 5.4) says KAIRA 88ddb07 and upstream 561966ec already hold an equivalent fix |
| `backend/build_backend_dg.cmd` | build script | none | builds `coli_cuda_dsv4_dg.dll` from an engine tree; needs `COLIBRI_C`, `DEEPGEMM_ROOT` and `CUDA_PATH` and says which one is missing |

## Build

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `build/Makefile.deepseek-v4.wex-excerpt.mk` | excerpt | KAIRA 88ddb07 | `Makefile.deepseek-v4`: the rules that link the WEX objects into `deepseek_v4.exe` (Windows) |
| `build/Makefile.wex-excerpt.mk` | excerpt | KAIRA 88ddb07; the WNR3 test rule at the end is from cade51d | top-level `Makefile`: the WEX unit tests and `wex_decode.o` in the test builds |
| `build/build_wex_windows.cmd` | build script | none | builds from this repository alone: the helper `kaira_wex_cuda.dll`, `archive\nr2\wex_nr2_tools.dll`, `test_decode_batch.exe`, `check_container.exe`. Overrides: `VCVARS`, `NVCC`, `ARCH` (default sm_120) |
| `build/build_test_wex_planned.sh` | build script | written for a KAIRA-style colibri `c/` tree at 88ddb07 | builds `test_wex_planned.exe` in an engine tree that has this repository's `c/` copied over it (MSYS2 UCRT64) |

## KAIRA host (C#, outside the engine)

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `host/Engine.wex-excerpt.cs` | excerpt | KAIRA 88ddb07 | `src/host/Engine.cs`: how the profile environment and the mirror folder reach the engine process |
| `host/ColibriFamily.wex-excerpt.cs` | excerpt | KAIRA 88ddb07 | `src/host/ColibriFamily.cs`: the V4.1 environment allow-list |
| `host/Config.wex-excerpt.cs` | excerpt | KAIRA 88ddb07 | `src/host/Config.cs`: profile defaults. Its header marks the container path in it as stale, and one path is shortened to `<HOME>` |
| `host/DeepSeekV41Preflight.wex-excerpt.cs` | excerpt | KAIRA 88ddb07 | `src/host/DeepSeekV41Preflight.cs`: a preflight warning that its header marks as stale |

## Patches

| File | Kind | Shows | Applies to |
|---|---|---|---|
| `kaira-wex4-loader-75932c1.patch` | patch | KAIRA 75932c1 (2026-11-11) | a KAIRA tree: `src/colibri/c/deepseek_v4.c`, `tests/test_v4_wex_nr3.c`, new `wex4.h`. The engine opens WEX4 containers |
| `kaira-wex4-first-read-4ba5568.patch` | patch | KAIRA 4ba5568 (2026-11-11) | a KAIRA tree, on top of the loader patch: `deepseek_v4.c`, `deepseek_v4_internal.h`, `tests/test_v4_wex_nr3.c`, `wex4.h`. SHA-256 of every record on its first read (`COLI_WEX_VERIFY`) |
| `kaira-q38-wex4-loader-de17198.patch` | patch | KAIRA de17198 (2026-11-11) | a KAIRA tree: `src/colibri/c/qwen38_wex.h`, new `wex4.h`. The qwen38 engine opens WEX4 containers |
| `upstream-561966ec-levelA.patch` | patch | upstream colibri 561966ec; hooks taken from KAIRA 88ddb07 | a stock colibri checkout: `c/deepseek_v4.c`, `c/Makefile.deepseek-v4`. Level A only (WEX1 / WNR2 into the RAM slab). Its header says "Verified 2026-11-08" |

Notes on the patches:

- The three `kaira-*` patches are `git format-patch` files with paths under `src/colibri/c/`. All three commits are ancestors of KAIRA master 940275a (checked on 2026-11-13 with `git merge-base --is-ancestor`), so a current KAIRA tree already has them; they are here for a reader who has no access to KAIRA. The `wex4.h` and `qwen38_wex.h` they add are older than the copies in `../c/`: take the files from `../c/`.
- There is no upstream patch for WNR3 or WEX4. The upstream patch does not mention WNR3, WEX4 or `COLI_WEX_ROOT`; for those, the hooks are the WNR3 excerpt, the three `kaira-*` patches and the WEX4 excerpt.
- The header of the upstream patch tells you to copy this repository's `c/*.h` and `c/*.c` ([`../c/`](../c/)) first. On 2026-11-13 `git apply --check` passed on a clean `git archive 561966ec` with today's `c/*.h` and `c/*.c` copied in; the build and the token A/B of 2026-11-08 were not repeated, and `c/` has gained WNR3 and WEX4 since then.
- The engine excerpts of 88ddb07 and cade51d were cut before WEX4 existed: they show the WEX3 open path. The WEX4 path, and what came after the three `kaira-*` patches, is in `engine/deepseek_v4.wex4.wex-excerpt.c` (24f7e72).
