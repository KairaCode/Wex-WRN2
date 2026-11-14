# WEX / WNR2 in colibri: integration, generation and use

> **Dated report (2026-11-13 note).** This is the integration guide as it stood on 2026-11-09, moved here from `docs/COLIBRI-INTEGRATION.md` on 2026-11-13 with its text and section numbers unchanged; only its links were adjusted to the new folder. Older reports that cite "the guide, section N" mean this page. It supersedes the "Private repo" sentence below: the repository is licensed under the Apache License 2.0 since 2026-11-13. The current guide is [COLIBRI-INTEGRATION.md](../COLIBRI-INTEGRATION.md). On 2026-11-13 `nr2/` and `bench/` moved to `archive/`; the paths below were changed to match.

State of the body: 2026-11-09. Sections 1-11 quote engine code from the engine of 2026-11-07; sections 12-13 describe the engine of 2026-11-09 and the 2026-11-09 deployment; section 14 records the 2026-11-10 switch to WNR3 (WEX1 and WNR2 containers deleted). Private repo: this code is not pushed to the public colibri repo.

> **What changed after 2026-11-09.** Sections 1-11 describe the WEX1 / WEX2 / WNR2 integration as it stood on 2026-11-09, and their WEX1 and WNR2 commands still regenerate those containers. Since then DeepSeek V4.1 moved to WNR3 records in a WEX4 container, Qwen3.8 was added with RAW MXFP4 records in WEX4 containers, and the V4.1 WEX1 and every V4.1 WNR2 copy were deleted. What runs today: [STATUS.md](../../STATUS.md). Terms: [GLOSSARY.md](../GLOSSARY.md).
>
> | Since | What | Described in | What to copy or run |
> |---|---|---|---|
> | 2026-11-10 | **WNR3**: record codec 5 in a WEX3 container, header flag 32 (plus 64 for canonical zero), helper feature 16 | [WNR3-SPEC.md](../WNR3-SPEC.md); section 14 of this guide; results in [WNR3-VERIFY.md](WNR3-VERIFY.md) | The decoders are in the files of section 2 (`c/wex_format.h`, `c/wex_neural_rans.h`, `c/wex_decode.c`, `c/wex_cuda.cuh`, the bridge). Builder: `c/tools/build_nr3.py` with the CPU encoder DLL `c/tools/wex_nr3_tools.c`; the steps are in [WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md), "Rebuilding WNR3 from the safetensors". CPU self-check with no real model and no GPU: `sh scripts/build_cpu_check.sh` |
> | 2026-11-11 | **WEX4**: a hashed header page and a hashed record index around unchanged RAW or WNR3 records | [WEX4-SPEC.md](../WEX4-SPEC.md) | Copy `c/wex4.h` next to `wex_format.h`. Upgrade a WEX1 or WNR3 container in place: `python -I wex4/wex4_convert.py upgrade FILE.wex` (`downgrade` undoes it). CPU tests: `sh wex4/build_and_test.sh`, which also builds `wex4_tools.dll` into `wex4/out/`. Check a container: `python -I wex4/wex4.py verify FILE --dll wex4/out/wex4_tools.dll`. New variables `COLI_WEX_ROOT` and `COLI_WEX_VERIFY`: table in 5.5 |
> | 2026-11-11 | **Qwen3.8-Flash-Next MXFP4**: a second engine reads these containers | [QWEN38-MXFP4-20261111.md](QWEN38-MXFP4-20261111.md) | `c/qwen38_wex.h` (the qwen38 store for WEX1, WNR3 and WEX4), `c/tools/repack_q38_mxfp4_wex.py` (Quark shards to WEX1), tests `c/tests/test_qwen38_mxfp4.c` and `.py`. The MXFP4 compute code of the qwen38 engine is not container code and stays in the engine tree. The FP8 proposal in section 6.6 was not built |
> | 2026-11-11 | **Striped reads**: one record read from two drives at once, cut on a 4096-byte page in the proportion of `COLI_WEX_WEIGHTS` | qwen38: the header comment of `c/qwen38_wex.h`. DeepSeek: the code is in the maintainer's `deepseek_v4.c` only; its test is `c/tests/test_v4_wex_nr3.c` | `V4_WEX_STRIPE=1` (DeepSeek demand batches) and `COLI_WEX_STRIPE=1` (qwen38, a lone decode miss), both off by default. Measured on V4.1: -8.72% ± 2.66% (ABBA, 3 prompts, identical tokens), so it stays off ([STATUS.md](../../STATUS.md), "In the engine, switched off or not measured") |
> | 2026-11-11 | **Helper feature bit 32** (`WEX_CUDA_FEATURE_BATCH_HOST`): `wex_cuda_decode_batch_to_host`, a batch of WNR2 or WNR3 records decoded on the GPU into host buffers | `c/tools/wex_cuda_bridge.cu`, `c/wex_decode.h` | The bridge now returns features 63. The qwen38 store uses bit 32 for a load wave of WNR3 records. Table in 4.3 |
>
> **Which commit each file in `integration/` shows.** None of them shows the engine as it is today. The files the maintainer keeps identical in the engine tree are the 37 under `c/` (`sync-map.tsv`; `python -I check_repo.py --source <TREE>` compares them).
>
> - 14 excerpts quote the engine of 2026-11-07: `engine/deepseek_v4.mode0-store`, `.direct-store` and `.direct-gpu`, `engine/deepseek_v4_internal`, the four in `backend/`, the two `build/*.mk` (`Makefile.wex-excerpt.mk` also has one range from the WNR3 branch head) and the four in `host/`. Sections 1-11 use their line numbers.
> - `engine/deepseek_v4.wnr3.wex-excerpt.c` quotes the engine of 2026-11-10: the WNR3 and canonical-zero hooks, on top of the three engine excerpts.
> - `engine-wex4-loader-75932c1.patch`, `engine-wex4-first-read-4ba5568.patch` and `engine-q38-wex4-loader-de17198.patch` are the engine commits of those names (2026-11-11). Their paths are the maintainer's (`src/colibri/c/...`): they are the reference for a WEX4 loader and do not apply to stock colibri.
> - `upstream-561966ec-levelA.patch` is level A (WEX1 and WNR2, section 5.1) for stock colibri 561966ec, taken from the engine of 2026-11-07 and verified on 2026-11-08 with the `c/` files of that date. It has not been rebuilt with today's `c/` files. No upstream patch exists for WNR3, WEX4 or Qwen3.8.
> - Shown by no file here: the batched decode of the engine of 2026-11-09 (it is in `research/v41-w2-20261109/laneW-w2.patch`), striped reads, and the other engine changes of 2026-11-11.

## Riepilogo (italiano)

Questa guida porta i container WEX1/WEX2/WEX3-WNR2 in un albero colibri "stock" e spiega come generarli, verificarli e usarli.

- **Cosa c'è nel repo.** Formato, decoder CPU e CUDA, helper `kaira_wex_cuda.dll`, store `expert_store_wex.c`, repacker WEX1/2/3, encoder/trainer WNR2, modelli WNM2, test. In `integration/` ci sono gli estratti del codice del motore (deepseek_v4.c, backend CUDA, loader, Makefile, host C#), ciascuno con file e righe di origine. In `fixtures/` ci sono il modello di test `v41-tiny` e due generatori (`make_synth_mxfp4.py`, `widen_experts.py`).
- **Patch per colibri upstream.** `integration/upstream-561966ec-levelA.patch` porta il livello A in colibri stock (JustVugg 561966ec) con `git apply`: verificata il 2026-11-08 (si applica, compila, token identici). Per altri commit, o per i livelli B e C, gli agganci vanno copiati dagli estratti (sezione 5).
- **Livello A (modo 0, slab in RAM)** basta per usare WEX1 e WNR2 su qualsiasi build Windows: ~11 punti in deepseek_v4.c, piu due oggetti nel Makefile.
- **Livelli B e C (GPU-direct, cache RAM dei record)** richiedono anche le estensioni del fork per il mirror-cache GPU e tre export del backend DSV4.
- **Generare:** WEX1 con `c/tools/repack_v4_wex.py` (V4.1: 115 s, 144,4 GB); WNR2 con `archive/nr2/build_nr2_full.py` sulla GPU (V4.1: 723 s, 129,9 GB, -10%).
- **Verificare:** `scripts/verify_wex1.py` (byte per byte contro i safetensors), sha256 + marker `.verified`, `archive/nr2/sample_records.py` + `test_decode_batch.exe`, `check_container.exe`; identità dei token col motore: `--record-oracle` con i safetensors, poi `--oracle` col container (sezione 7.5).
- **Verificato il 2026-11-08 (sezione 7.6):** colibri upstream 561966ec + patch e il motore del 2026-11-07, su CPU, danno gli stessi token con safetensors, WEX1 e WNR2 su modelli piccoli allargati a moe_inter 1024. I fixture piccoli originali (`v41-tiny`, `deepseek_v4_tiny`) hanno record che non sono multipli di 4096: il container viene accettato ma i record si leggono dai safetensors (`flock_reads=0`).
- **Non esiste:** versione Linux del percorso motore. WEX per Qwen3.8 non esisteva al 2026-11-09 (solo la proposta WEX-Q8 in FP8, sezione 6.6, mai realizzata); dall'11 novembre esiste con expert MXFP4 (riquadro in alto, [QWEN38-MXFP4-20261111.md](QWEN38-MXFP4-20261111.md)). V4.1 su GPU con WEX (G2, W2 `V4_WEX_BATCH`) esiste nel motore del 2026-11-09 ed è installato nel profilo `v41gpu` (sezioni 12-13), ma non negli estratti del 2026-11-07.
- **Aggiornamento 2026-11-10:** l'applicazione usa WNR3-CZ su U: con mirror su C:; WEX1 e le copie WNR2 sono cancellate (sezione 14, [WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md)). I comandi WEX1/WNR2 qui sotto restano validi per rigenerarli.
- **Aggiornamento 2026-11-11:** i record WNR3-CZ stanno ora in un container WEX4 ([WEX4-SPEC.md](../WEX4-SPEC.md)); cosa gira oggi: [STATUS.md](../../STATUS.md).
- **Si parte da:** sezione 2 (copia dei file), 4 (build), 5 (agganci), 6-8 (generare, verificare, usare), 7.5 (prova A/B).

## Contents

1. Architecture
2. Files and where they go in a colibri tree
3. Prerequisites
4. Build
5. Engine wiring (deepseek_v4.c, backend, host)
6. Generating containers
7. Verifying
8. Using
9. Troubleshooting
10. Known limits and open work
11. Sources and keeping this repo in sync
12. Update 2026-11-09
13. Application deployment 2026-11-09
14. Update 2026-11-10: WNR3 in production

Deeper references, not repeated here:

- [WNR2-REFERENCE.md](../WNR2-REFERENCE.md): format, codec, encoder, helper API, engine prose, measurements, timeline.
- [wex-architecture-20261018.it.md](wex-architecture-20261018.it.md) (Italian): header/entry structs, WEX1/2/3 matrix, tiles, palette3, dedup, standalone store.
- [wex2-scale-tiles-20261018.it.md](wex2-scale-tiles-20261018.it.md) (Italian): WEX2 scale tiles.
- [v41-reap-wnr2-build-20261107.md](v41-reap-wnr2-build-20261107.md): the V4.1 WNR2 build (W1) and its 64-record GPU check.
- [v41-reap-baseline-m0-20261107.md](v41-reap-baseline-m0-20261107.md): drive probe and WEX1 vs safetensors baseline.
- [wnr2-wex3-when-compression-pays-20261105.it.md](wnr2-wex3-when-compression-pays-20261105.it.md) and [wnr2-cache-decode-cpu-wddm-20261105.it.md](wnr2-cache-decode-cpu-wddm-20261105.it.md) (Italian): where WNR2 pays and where it does not.
- [wex-prod-20261105.it.md](wex-prod-20261105.it.md) (Italian): batch decode, rechunk and the Qwen3.8 WEX-Q8 proposal.

---

## 1. Architecture

### 1.1 Formats

| Format | Magic / version | Record | Codec on entries | Lossless | Decoder |
|---|---|---|---|---|---|
| **WEX1** | `WEX1` v1, flags 0 | native MXFP4 slab: scales w1\|w2\|w3, then weights w1\|w2\|w3, zero-padded to 4096 | 0 (raw) | yes | none (read straight into the slab) |
| **WEX2** | `WEX2` v2, flag 1 | UE8M0 scales packed in tiles, weights unchanged | 0 or 1 (`SCALE_TILES`, page count in bits 8..31) | yes | CPU, in place |
| **WEX3** | `WEX3` v3, flags 1/2/4/8 | tiles + dedup aliases; palette3 (LOSSY, codec 2); WNR1 neural rANS (codec 3, retired) | 0..3 | palette3 is not | CPU; codec 3 also CUDA |
| **WNR2** | `WEX3` v3, header flag 16 (`NEURAL_RANS2`) set; the V4.1 container has flags 21 = SCALE_TILES\|DEDUP\|NEURAL_RANS2 | neural rANS, 4096-symbol streams with u16 lengths, rANS scales; WNMD descriptor at byte 64 of the header page, WNM2 model at 4096, table at 65536 | 4 | yes | CUDA helper (feature 4) or CPU reference |
| **WNR3** | `WEX3` v3, header flag 32 (+64 canonical zero) | WNR2 skeleton + ceil-form/rule-g + per-stream MurmurHash3 | 5 | yes (CZ: value-exact) | CUDA helper (feature 16) or CPU; see WNR3-SPEC.md |
| **WEX4** (since 2026-11-11) | `WEX4`, major 4, minor 0 | the same records, byte for byte (raw or WNR3), behind a hashed header page and a hashed index with a SHA-256 per record | 0 or 5 on expert records (6 = WBP1, for non-expert tensors) | as the record codec | as the record codec; the container is validated with `c/wex4.h`; see WEX4-SPEC.md |

Every WEX1, WEX2 or WEX3 container is a 4096-byte header page (`WexHeader`, 40 B used), a 72-byte `WexEntry` per (layer, expert) at `table_offset`, and the records, contiguous and 4096-aligned. An absent expert has an all-zero entry. The engine reads one record with one unbuffered read (`FILE_FLAG_NO_BUFFERING`). Byte layouts: `c/wex_format.h`, WNR2-REFERENCE.md sections 5-7.

### 1.2 Data paths in the engine (DeepSeek V4/V4.1, `deepseek_v4.c`)

```
                     COLI_WEX_PATH (+ mirrors)          safetensors shards
                               |                                |
            v4_wex_pick (drive) |                                | (fallback on any WEX failure)
                               v                                v
 MODE 0  (default)   ReadFile -> RAM slab + wex_record_read_offset
                     coli_wex_decode_record (in place):
                        codec 0  : nothing to do (WEX1)
                        codec 1  : CPU tile decode (WEX2)
                        codec 3/4: kaira_wex_cuda.dll decode_to_host, else CPU
                     -> RAM tier -> (GPU build) upload to the fp4 VRAM mirror
 MODE 1/2            VRAM-first miss, record codec 3/4, not in RAM:
 (COLI_WEX_GPU_DIRECT) RAM record cache hit or ReadFile -> page-locked staging
                     -> wex_cuda_decode_to_device straight into the mirror's fp4 buffers
                     (mode 2: the next-layer prefetcher does the same into a pool mirror)
```

- WEX1/WEX2 records are never GPU-direct: only codec 3/4 records qualify (`coli_v4_expert_store_wex_entry`).
- WNR2 in mode 0 costs three PCIe trips per record (H2D record, D2H slab, H2D at attach).
- The helper is loaded lazily, the first time a codec 3/4 record or a WNM2 model needs it. WEX1 never loads it.
- This is the picture at the engine of 2026-11-07. Since the engine of 2026-11-10 a WNR3 record (codec 5) is handled like a codec 4 record in it (host decode, GPU-direct, batch), with helper feature 16 and an owner check before every decode (`integration/engine/deepseek_v4.wnr3.wex-excerpt.c`).

### 1.3 Components

| Component | File(s) | Role |
|---|---|---|
| Format | `c/wex_format.h` (+ `wex_palette.h`, `wex_neural_rans.h`) | structs, validation (`wex_validate_header/_table`), CPU decoders, WNR2 model parse, CPU reference encoder (`wex_nr2_count/_model_build/_encode`) |
| CPU decode + helper loader | `c/wex_decode.c/.h` | `coli_wex_decode_record`, WNM2 registry `coli_wex_register_model`, `LoadLibraryExA` of the helper and its wrappers |
| WEX4 container | `c/wex4.h` | header page and index validation, SHA-256, the legacy `WexEntry` rebuilt from the index, WBP1 codec (since 2026-11-11; WEX4-SPEC.md) |
| CUDA kernels | `c/wex_cuda.cuh` | WNR1, WNR2 and WNR3 kernels, commit-on-success |
| Helper DLL | `c/tools/wex_cuda_bridge.cu` -> `kaira_wex_cuda.dll` | ABI 1; decode_to_host/_to_device, host_register, register_model, batch to device, batch to host |
| Qwen3.8 store | `c/qwen38_wex.h` | the qwen38 engine's reader of WEX1, WNR3 and WEX4 containers of MXFP4 experts (since 2026-11-11) |
| Standalone store | `c/expert_store_wex.c/.h` | registry backend `wex`. With an engine it only delegates to the planned store (section 5.6). |
| Engine hooks | the maintainer's `deepseek_v4.c`, `deepseek_v4_internal.h` | container open, mirrors, read path, GPU-direct, RAM record cache (`integration/engine/`) |
| Backend hooks | `backend_cuda_dsv4.cu/.h`, `backend_loader_dsv4.c`, `dsv4.def` | fp4 mirror alloc/buffers export, stream priority (`integration/backend/`) |
| Writers | `c/tools/repack_v4_wex.py` (WEX1/2/3), `archive/nr2/build_nr2_full.py` + `archive/nr2/encode_nr2.cu` (WNR2) | section 6 |
| Writers added later | `c/tools/build_nr3.py` + `c/tools/wex_nr3_tools.c` (WNR3), `wex4/wex4_convert.py` (WEX4), `c/tools/repack_q38_mxfp4_wex.py` (Qwen3.8 WEX1) | the box at the top of this guide |
| Checkers | `scripts/verify_wex1.py`, `archive/nr2/check_container.c`, `archive/nr2/check_all.py`, `archive/nr2/sample_records.py`, `archive/bench/test_decode_batch.cu` | section 7 |
| Fixtures | `fixtures/v41-tiny/`, `fixtures/make_synth_mxfp4.py`, `fixtures/widen_experts.py` | tiny models for the engine A/B and the store tests (section 7.5) |
| Upstream patch | `integration/upstream-561966ec-levelA.patch` | level A for stock colibri 561966ec (section 5.1) |

---

## 2. Files and where they go in a colibri tree

colibri keeps its C sources in one folder (`c/` upstream, `src/colibri/c` in the maintainer's tree), with `tools/` and `tests/` under it. The repo's `c/` holds copies of the maintainer's files from that folder at the same relative paths, so their relative includes (`../wex_cuda.cuh`, `../wex_format.h`) resolve where they are.

| Repo path | Goes to (colibri `c/`) | Notes |
|---|---|---|
| `c/wex_*.h`, `c/wex_decode.c`, `c/wex_cuda.cuh`, `c/expert_store_wex.*` | `c/` | 8 files. Like every file under `c/` they are byte-identical to the maintainer's tree at the commit recorded in `sync-map.tsv` (they were at the engine of 2026-11-07 when sections 1-11 were written and have gained WNR3 since) |
| `c/wex4.h` | `c/` | the WEX4 reader, needed only to open WEX4 containers; includes `wex_format.h` |
| `c/qwen38_wex.h` | `c/`, with a qwen38 engine that has the maintainer's MXFP4 code | the qwen38 store; includes `wex_format.h`, `wex_decode.h`, `wex_decode.c` and `wex4.h` |
| `c/tools/build_nr3.py`, `c/tools/wex_nr3_tools.c`, `c/tools/repack_q38_mxfp4_wex.py`, `c/tools/make_qwen38_tiny_mxfp4.py` | `c/tools/`, or run them from the repo | WNR3 builder and its CPU encoder DLL; Qwen3.8 repacker and tiny-fixture maker |
| `c/tests/test_v4_wex_nr3.c`, `fake_wex_cuda.c`, `test_wex_nr3_gpu.cu`, `test_build_nr3_picks.py`, `test_qwen38_mxfp4.c`, `test_qwen38_mxfp4.py`, `make_qwen38_tiny_mxfp4_experts.py` | `c/tests/` | engine tests: the WNR3 and WEX4 engine store with a CPU stand-in helper (includes the maintainer's `test_deepseek_v4.c`), the CPU/GPU differential of the WNR3 decoder (needs a GPU and the helper), the builder's training picks, and the qwen38 store (includes the maintainer's `qwen38.c`) |
| `wex4/` | stays in the repo | WEX4 Python reference, converter, C tools, tests, reviews and research |
| `c/tools/wex_cuda_bridge.cu`, `c/tools/bench_wex_cuda.cu`, `c/tools/bench_wex_decode.c` | `c/tools/` | the bridge includes `../wex_cuda.cuh` |
| `c/tools/repack_v4_wex.py`, `wex_neural_rans.py`, `wex_palette.py`, `scripts/verify_wex1.py` | `c/tools/`, or run them from the repo | pure Python; `verify_wex1.py` is new, not in the engine tree |
| `c/tests/test_wex_format.c`, `test_expert_store_wex.c`, `test_wex_planned.c` | `c/tests/` | include `../*.h` |
| `c/tests/test_v4_gpu_priority.c`, `test_v4_mirror_guard_source.py` | `c/tests/` | engine tests for the GPU-direct mirror cache; they need the level C code |
| `c/tests/test_repack_v4_wex.py`, `scripts/test_verify_wex1.py` | stay in the repo | run with `python -I` from the repo |
| `archive/nr2/`, `archive/bench/`, `models/` | stay in the repo | encoder, checks, benches, WNM2 models; not part of the engine |
| `integration/engine/*`, `integration/backend/*`, `integration/build/*.mk`, `integration/host/*` | reference only, merged by hand | each file names its engine source and line ranges in its header |
| `integration/upstream-561966ec-levelA.patch` | `git apply` from the root of a stock colibri 561966ec checkout | level A (A1-A11 + Makefile hunks), section 5.1 |
| `integration/backend/backend-upload-race-fix.patch` | history only, apply nowhere | both the engine of 2026-11-07 and upstream 561966ec already hold an equivalent fix (section 5.4) |
| `integration/history/wex-gpu-direct.patch` | reference only | the Oct-20 GPU-direct diff against the pre-GPU-direct tree, 16 files |
| `integration/build/build_wex_windows.cmd` | run from the repo | builds the helper, the encoder DLL and the checkers (section 4.3) |
| `integration/build/build_test_wex_planned.sh` | run from the repo on a colibri `c/` | builds `test_wex_planned.exe` (section 7.4) |
| `fixtures/v41-tiny/` | stays in the repo | the maintainer's 6-layer V4.1 test model (3 MB; copy of `<TREE>\lab\v41-port-20261106\fixtures\v41-tiny` without its `ref*.json`); runs on the maintainer's engine only |
| `fixtures/make_synth_mxfp4.py`, `fixtures/widen_experts.py` | stay in the repo | fixture generators, standard library only (section 7.5) |

Copy (Git Bash, from the repo root; `C` is the colibri `c/` folder):

```sh
C=/path/to/colibri/c
cp c/wex_*.h c/wex_decode.c c/wex_cuda.cuh c/expert_store_wex.* "$C"/
cp c/tools/wex_cuda_bridge.cu c/tools/bench_wex_cuda.cu c/tools/bench_wex_decode.c \
   c/tools/repack_v4_wex.py c/tools/wex_neural_rans.py c/tools/wex_palette.py scripts/verify_wex1.py "$C"/tools/
cp c/tests/test_wex_format.c c/tests/test_expert_store_wex.c c/tests/test_wex_planned.c "$C"/tests/
# GPU-direct only (level C):
cp c/tests/test_v4_gpu_priority.c c/tests/test_v4_mirror_guard_source.py "$C"/tests/
# WEX4 containers only: the reader header (the loader itself is the engine patch named in the box at the top)
cp c/wex4.h "$C"/
# Stock colibri 561966ec only: level A in one step, from the colibri checkout root
git -C "$C/.." apply --check "$PWD/integration/upstream-561966ec-levelA.patch" &&
git -C "$C/.." apply "$PWD/integration/upstream-561966ec-levelA.patch"
```

The repo has `* -text` in `.gitattributes`: files keep their line endings (several C sources are CRLF).

---

## 3. Prerequisites

| Need | Version used | For |
|---|---|---|
| Windows 10/11 x64 | Windows 11 | the engine WEX path is Windows-only (`CreateFileA`, `ReadFile` overlapped, `_aligned_malloc`) |
| MSYS2 ucrt64 gcc + make, run from an MSYS2 UCRT64 login shell (not Git Bash, section 4.1) | gcc 16.2 | engine and C tests |
| CUDA Toolkit (nvcc) | 13.4 | helper, encoder DLL, GPU tests, DSV4 backend |
| MSVC Build Tools (`vcvars64.bat`) | VS 18 BuildTools | nvcc host compiler on Windows; `check_container.c` |
| NVIDIA GPU | RTX 5090, `sm_120` | the helper kernels use `cp.async` on sm_80 or newer. Below sm_80 `c/wex_cuda.cuh` now compiles a plain load and store instead (`__CUDA_ARCH__ < 800`), so the helper builds for Pascal and Turing; that build has not run on such a GPU. The tools are only tested on sm_120. |
| Python + numpy | 3.14, numpy 2.5.3 | `build_nr2_full.py`, nr2 scripts. `repack_v4_wex.py`, `verify_wex1.py` and the `fixtures/` scripts need only the standard library. |
| PyTorch + transformers (optional) | torch 2.14.1+cpu, transformers 5.20.2 | only to generate colibri's `deepseek_v4_tiny` fixture for the upstream A/B (section 7.5) |
| Free disk | about 1x the container | see section 6 |

---

## 4. Build

### 4.1 Engine: `Makefile.deepseek-v4`

Upstream (`JustVugg/colibri` 561966ec) has `V4_WIN_EXTRA_OBJS = backend_loader_dsv4.o` and no WEX rule. Add, as in `integration/build/Makefile.deepseek-v4.wex-excerpt.mk`:

```make
# Windows block
V4_WIN_EXTRA_OBJS = backend_loader_dsv4.o expert_store_wex.o wex_decode.o

expert_store_wex.o: expert_store_wex.c expert_store_wex.h wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h expert_store.h $(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -c expert_store_wex.c -o $@

wex_decode.o: wex_decode.c wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h $(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -c wex_decode.c -o $@
```

- Add `expert_store_wex.h wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h` to the prerequisites of the `$(TARGET_OBJS) $(TEST_UNIT_OBJS)` rule and of `$(V4_HOT_TEST_OBJ)`, because `deepseek_v4.c` includes `expert_store_wex.h`.
- `expert_store_wex.o` registers the `wex` store backend by itself (`__attribute__((constructor))`), so the registry needs no edit.
- On Windows every link of the expert-store unit needs `wex_decode.o`, because the unit calls `coli_wex_decode_record`. That covers the engine, the V4 tests and the segment build.
- On stock 561966ec, `integration/upstream-561966ec-levelA.patch` already carries these Makefile hunks.

Build command (the output is `deepseek_v4.exe` in `c/`):

```sh
MSYSTEM=UCRT64 C:/msys64/usr/bin/bash.exe -lc "cd /c/path/to/colibri/c && make -f Makefile.deepseek-v4 deepseek-v4 -j12"
```

- Run make from an **MSYS2 UCRT64 login shell** (the line above, or the "MSYS2 UCRT64" start-menu shell). From Git Bash with MSYS2 on `PATH`, every gcc fails with `Cannot create temporary file in C:\WINDOWS\` and make stops with Error 127, even with `TMP` set.
- Measured 2026-11-08: the engine of 2026-11-07 copy 17 s, 1,522,381 B; upstream 561966ec + level A patch 21 s, 1,815,173 B (both `-j12`, gcc 16.2).
- The Windows engine resolves its CUDA tier from `coli_cuda_dsv4*.dll` at runtime (4.4). `DSV4_CUDA=0` keeps everything on the CPU, which is all level A needs.

### 4.2 Tests: top-level `Makefile`

From `integration/build/Makefile.wex-excerpt.mk`:

- `tests/test_wex_format$(EXE)`: a single translation unit.
- `tests/test_expert_store_wex$(EXE)`: links `tests/expert_store_registry.o tests/expert_store_wex.o tests/wex_decode.o`.
- Add `wex_decode.o` to `V4_TEST_LINK_OBJS` and `V4_TEST_OBJ_TARGETS`, the WEX headers to the `tests/test_deepseek_v4` prerequisites, and the three test objects to `AUTODEP_OBJS`.
- For a segment build of the V4 engine, add `$(SEGMENT_BUILD_DIR)/wex_decode.o` and its rule.
- `tests/test_v4_gpu_priority$(EXE)` is fork-only (rule 2655-2657). It also uses engine test hooks that upstream lacks (`coli_v4_test_gpu_frequency_victim`, `coli_v4_test_gpu_configure_host_pin`).
- `test_wex_planned` has **no Makefile rule** anywhere: `integration/build/build_test_wex_planned.sh` builds it (section 7.4).
- The level A patch does not touch the top-level `Makefile`. On stock upstream, add the two test rules by hand from the excerpt, or build the format test as one translation unit: `gcc -O2 -march=x86-64-v3 tests/test_wex_format.c -o tests/test_wex_format.exe`.

Commands (the maintainer's tree, MSYS2 UCRT64 shell; verified 2026-11-08, both pass):

```sh
cd /c/path/to/colibri/c
make tests/test_wex_format.exe tests/test_expert_store_wex.exe
./tests/test_wex_format.exe                          # ends with "ALL WEX STANDALONE TESTS PASSED"
COLI_WEX_CUDA_LIB=0 ./tests/test_expert_store_wex.exe
```

### 4.3 Helper, encoder and checkers: `integration/build/build_wex_windows.cmd`

From the repo, no colibri tree needed:

```bat
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe
integration\build\build_wex_windows.cmd [OUT_DIR]
```

- It needs `VCVARS` (default: VS 18 BuildTools `vcvars64.bat`), `NVCC` (default: `nvcc` on PATH) and `ARCH` (default `sm_120`).
- The default `OUT_DIR` is `<repo>\out`, which is git-ignored.
- Outputs:
  - `OUT\kaira_wex_cuda.dll`;
  - `archive\nr2\wex_nr2_tools.dll`, where `nr2lib.py` loads it;
  - `OUT\test_decode_batch.exe`;
  - `OUT\check_container.exe`, built with MSVC.
- Checked 2026-11-08 on this machine: all four built in 15 s. The helper was 387,072 B and reported ABI 1, features 15, the same size as the W1 reference build. The bridge in `c/tools/` today has two more bits (16 and 32, table below) and returns 63.

The underlying helper command, inside a `vcvars64` shell (the bridge must sit in `tools\` next to a parent folder holding `wex_cuda.cuh` and the headers):

```bat
nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2,/EHsc,/arch:AVX2 -shared c\tools\wex_cuda_bridge.cu -o kaira_wex_cuda.dll
```

Helper ABI (`wex_cuda_helper_abi() == 1`, required export `wex_cuda_decode_to_host`) and feature bits (`wex_cuda_helper_features()`):

| bit | enables | engine needs it for |
|---|---|---|
| 1 | `wex_cuda_decode_to_device` | GPU-direct modes 1/2 |
| 2 | `wex_cuda_host_register/_unregister` | page-locked staging and RAM cache (optional; pageable works) |
| 4 | `wex_cuda_register_model`, codec 4 | **any WNR2 container**: without it the engine refuses WNR2 |
| 8 | `wex_cuda_decode_batch_to_device` | `V4_WEX_BATCH=1` (the engine of 2026-11-09 only, default off, needs `COLI_WEX_GPU_DIRECT>=1`): one batched decode of each layer's WNR2 misses; with mode 2 the prefetcher's same-layer guesses too (section 12). The engine of 2026-11-07 and the excerpts have no caller. |
| 16 | codec 5 (WNR3) in `wex_cuda_decode_to_host` / `_to_device` / `_batch_to_device` | any WNR3 container (header flag 32): without it the engine refuses it |
| 32 | `wex_cuda_decode_batch_to_host`: a batch of WNR2 or WNR3 records decoded into caller host buffers | the qwen38 store (`c/qwen38_wex.h`): one helper call per load wave of two or more WNR3 records. Without it each record is decoded on its own |

Full API semantics: WNR2-REFERENCE.md section 10.

### 4.4 DSV4 CUDA backend DLL (GPU-direct only)

- GPU-direct writes into fp4 mirrors that the backend allocates. The backend therefore needs the exports in `integration/backend/`:
  - `dsv4_cuda_tensor_alloc_fp4`, `dsv4_cuda_tensor_fp4_buffers`, plus `dsv4_cuda_wex_direct_available` when linked statically;
  - their two lines in `dsv4.def`;
  - optional `GetProcAddress` resolution in `backend_loader_dsv4.c`, so an older DLL keeps the host slab path.
- `dsv4_cuda_stream_drain` is already upstream.
- The maintainer builds the DLL with `integration/backend/build_backend_dg.cmd`: nvcc, DeepGEMM, `-Xlinker /DEF:dsv4.def`, `-gencode arch=compute_120a,code=sm_120a`, cuBLASLt. The script reads its machine paths from the environment and stops with a message when one is missing: `COLIBRI_C` (your colibri `c/`, which holds `backend_cuda_dsv4.cu` and `dsv4.def`), `DEEPGEMM_ROOT` (a DeepGEMM checkout), `CUDA_PATH` (the maintainer uses v13.4) and optionally `VCVARS` (default: VS 18 BuildTools); the DLL goes to `out\coli_cuda_dsv4_dg.dll` or to the folder given as first argument. The build itself was not rerun from this repo.
- The engine loads `COLI_DSV4_DLL` if set. Otherwise it tries `coli_cuda_dsv4_dg.dll` (DeepGEMM, sm_120a) and then `coli_cuda_dsv4.dll` from the engine's folder. It skips a DLL that does not load, lacks a required symbol, or answers 0 to `dsv4_cuda_backend_arch_ok(0)`.

---

## 5. Engine wiring

`deepseek_v4.c` is an amalgam built once per `COLI_V4_UNIT_*` define. The WEX code sits in three units:

- the expert store, `COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16` (levels A and B: store open, read path, RAM record cache);
- `COLI_V4_UNIT_BLOCK_HYBRID`: the three level C call sites in `v4_job_run`, `v4_np_worker` and `moe_token_pipeline`;
- the GPU tier, `COLI_V4_UNIT_GPU` (mirror cache, GPU-direct).

The excerpts are ~750 lines for the store unit, ~70 for BLOCK_HYBRID and ~780 for the GPU unit, context included. Upstream has 0 WEX lines. Upstream already has the store skeleton the hooks attach to: `V4ExpertStoreState`, `build_record`, `coli_st_mirror_setup`, `coli_st_expert_route`, `v4_read_expert_record`, `aligned_slab`, the `v4_direct_*` counters and their test hooks, and `indexed_expert_slot`. Port in levels. Each row names the excerpt file and the 2026-11-07 line range in its header.

### 5.1 Level A: mode 0 (host slab). Enough to run any container.

Excerpt: `integration/engine/deepseek_v4.mode0-store.wex-excerpt.c`. All of it is inside `#ifdef _WIN32` except the field defaults.

**Stock colibri 561966ec: use the patch.** `integration/upstream-561966ec-levelA.patch` is A1-A11 below plus the 4.1 Makefile hunks (+338/-6 lines in `c/deepseek_v4.c`, +10/-3 in `c/Makefile.deepseek-v4`). Copy the repo's `c/` files into the checkout's `c/` first (section 2), then `git apply` it from the checkout root. Verified 2026-11-08: it applies to a clean `git archive 561966ec c`, the engine builds, and the A/B in 7.5 is token-identical for safetensors, WEX1 and WNR2 (7.6). It differs from the fork's code in three places:

- the fields other than `wex_entries` exist only under `#ifdef _WIN32`, and A11 is guarded (see the porting notes);
- upstream `V4ExpertRecord` has no `layer`/`expert` fields, so A10 derives them from `record - state->records` (both callers pass a pointer into `state->records`);
- `v4_wex_matches_record` returns 0 for a GGUF record, so a container next to a GGUF checkpoint fails the store open instead of dereferencing safetensors fields.

For any other upstream commit, port from the excerpt by hand, row by row:

| # | Hook | Where to add it | 2026-11-07 lines |
|---|---|---|---|
| A1 | `#include "expert_store_wex.h"` | top of the expert-store unit | 12376 |
| A2 | fields `wex_handle`, `wex_mirror`, `wex_mirrors[3]`, `wex_nmir`, `wex_entries` (`wex_fd`/`wex_mirror_fd` off Windows) | end of `V4ExpertStoreState` | 12459-12476. Skip `wex_crc*` and the `v4_wex_crc_free` declaration unless you port level B |
| A3 | `state->wex_entries != NULL \|\|` | `build_record`, per-matrix layout condition | 12546 |
| A4 | `v4_wex_read_exact`, `v4_wex_load_manifest` (validates header and table; for WNR2 it registers the WNM2 and refuses the container without helper feature 4), `v4_wex_matches_record` | before `coli_deepseek_v4_expert_store_open` | 13064-13159 |
| A5 | field defaults (`INVALID_HANDLE_VALUE`, -1, NULL) | store open, after `state->pool_layer = -1` | 13189-13198 |
| A6 | container open: `COLI_WEX_PATH` or `<model_dir>\v4_flash_experts.wex`, `CreateFileA(NO_BUFFERING\|OVERLAPPED\|SEQUENTIAL_SCAN)`, manifest, then up to 3 mirrors from `COLI_MODEL_MIRROR` + `COLI_WEX_EXTRA_MIRRORS` (same basename, byte-identical header page and table) | store open, right after `coli_st_mirror_setup` | 13208-13307 |
| A7 | layout check per expert; a mismatch **fails the store open** | store open, record loop, after `build_record` | 13320-13328 |
| A8 | close handles, `_aligned_free(wex_entries)` | `destroy` and the open fail path | 13045-13058, 13408-13418 |
| A9 | `v4_wex_pick` (drive per expert: dual-SSD route or `COLI_WEX_WEIGHTS` hash) | before `v4_read_expert_record` | 13774-13817 |
| A10 | WEX branch: overlapped `ReadFile` into `slab + wex_record_read_offset`, `coli_wex_decode_record` in place, counters, safetensors fallback on failure | top of `v4_read_expert_record` | 13819-13864 |
| A11 | `v4_ssd_io mode=wex-direct-aligned` log | ROWS16 store open | 15287-15294 |

Porting notes for upstream:

- Upstream `v4_read_expert_record(state, record, slot, rep)` has no `payload_bytes` parameter and starts with a `record->gguf` branch. Drop the two `payload_bytes` lines. Do not open a container for a GGUF checkpoint: `v4_wex_matches_record` dereferences safetensors tensor records.
- A11 reads `state->wex_handle` outside `#ifdef _WIN32` in the fork, so that unit does not compile off Windows. Guard it:
  `int has_wex = 0;` / `#ifdef _WIN32` / `has_wex = state->wex_entries && state->wex_handle != INVALID_HANDLE_VALUE;` / `#endif`.
- Level A needs no change to `deepseek_v4_internal.h`, no GPU-unit change and no backend change.
- **Records must be whole 4096-byte pages to be read from the container.** WEX pads every record to 4096, and A10 serves a record from the container only when that padded size fits in `record_bytes`. Otherwise it skips the container for that record, reads the safetensors and counts **no** fallback. Real V4/V4.1 records are page multiples (13,369,344 and 18,800,640 B). The tiny test models are not (v41-tiny 13,056 B, deepseek_v4_tiny 26,112 B): with them the log says `Container v1 active` but every record comes from the safetensors. Check `flock_reads` (5.7) and widen tiny models first (7.5).

### 5.2 Level B: GPU-direct, store side, and the RAM record cache

- Excerpt: `integration/engine/deepseek_v4.direct-store.wex-excerpt.c`, same unit as level A. Declarations: `integration/engine/deepseek_v4_internal.wex-excerpt.h` (`deepseek_v4_internal.h` 1250-1285 of 2026-11-07).
- 13937-14257:
  - direct counters, `v4_wex_read_at`;
  - `coli_v4_expert_store_wex_entry`: codec 3, or 4 with the helper; it fits; RAM neither holds nor is loading it;
  - `coli_v4_expert_store_wex_read`: aligned read from `v4_wex_pick`, adds `disk_sec`, `WEX_READ_RETRY`;
  - `coli_v4_wex_direct_note`;
  - the compressed-record RAM cache: `v4_wex_crc` (lazy), `coli_v4_wex_crc_get` / `_done` / `_prewarm`, `v4_wex_crc_report`.
  - Dead code you may drop: `coli_v4_expert_store_wex_compressed`, `coli_v4_wex_flock_retry` (no callers).
- 15015-15027: exit lines `v4_wex_gpu_direct ...` and `v4_wex_ram_cache ...`.
- 15235-15243: RAM-cache prewarm in `coli_v4_expert_store_prewarm_tiers`. That function is fork-only, and so is `coli_v4_gpu_mirror_bitmap`. Without them the cache fills lazily.
- Add the `wex_crc`/`wex_crc_tried` fields and the `v4_wex_crc_free` call from A2/A8.

### 5.3 Level C: GPU-direct, GPU side

Excerpt: `integration/engine/deepseek_v4.direct-gpu.wex-excerpt.c`.

| Hook | 2026-11-07 lines |
|---|---|
| `v4_job_run` (decode loader lane): after a missed `coli_v4_gpu_expert_peek_routed`, call `coli_v4_gpu_expert_attach_wex`; 0 = GPU-only view | 6623-6671 |
| `v4_np_worker` (next-layer prefetcher), mode 2: `coli_v4_gpu_wex_direct_prefetch(rank)` then `coli_v4_gpu_expert_prefetch_wex` | 7332-7347 |
| `moe_token_pipeline`, after compute + drain: `coli_v4_gpu_mirror_step(store)` advances the epoch | 9703-9704 |
| `V4GpuExpertMirror` gains `busy`, `epoch`, `pool`; the cache gains `published` (cond), `epoch`, `guard_skips`, `refusals`, `pool_count/fills/hits`; `v4_gpu_mirror_host_guard` | 17259-17307 |
| Victim choice skips busy and pool entries; `v4_gpu_mirror_victim_safe`; `v4_gpu_mirror_pool_slot` (`COLI_WEX_GPU_DIRECT_POOL`) | 17314-17397 |
| Test hook `coli_v4_test_gpu_pool_slot` (inside `#ifdef COLI_V4_TEST_HOOKS`) | 17430-17454 |
| Init/destroy of `published`, exit line `v4_gpu_mirror_epochs` | 17465-17508 |
| `V4GpuMirrorFill`; `v4_gpu_mirror_refill`/`_create` take a `fill` | 17510-17583 |
| `v4_gpu_mirror_fill_unlocked`: reserve busy under the mutex, fill outside it, publish or empty, broadcast | 17944-18006 |
| `v4_gpu_expert_attach_cached_fill`: the attach path with the producer; same-key waiters; refusal = status 2 | 18092-18359 |
| `v4_wex_fill_mirror`, `v4_wex_gpu_direct_enabled`, `coli_v4_gpu_wex_direct_prefetch`, `coli_v4_gpu_expert_attach_wex`/`_prefetch_wex`, `v4_gpu_expert_attach_wex_credit` (per-thread staging, view shapes: `COLI_TENSOR_FP4_NATIVE_BLOCK`, UE8M0, block 1x32, tensor order gate/down/up = w1/w2/w3) | 18361-18510 |
| `coli_v4_gpu_mirror_step` | 18538-18544 |

- Level C relies on fork-only infrastructure that upstream lacks: `v4_vram_first_enabled`, `coli_v4_gpu_expert_peek_routed`, `coli_v4_expert_store_frequency`, `coli_v4_gpu_mirror_bitmap`, `v4_cuda_priority_mark`, the attach profiler `v4ap_*`, the copy worker (`v4_gpu_mirror_refill_post`) and the VRAM growth guard (`v4_gpu_mirror_grown_bytes`).
- Port those first, or replace them: `V4_VRAM_FIRST` defaults to on; the frequency can be 0; the priority marks and profiler can be no-ops.
- Tests: `c/tests/test_v4_gpu_priority.c` (`test_prefetch_pool`); `c/tests/test_v4_mirror_guard_source.py`, a source-level check that runs only inside the colibri tree.

### 5.4 Backend (level C) and the upload race fix

- `integration/backend/backend_cuda_dsv4.wex-excerpt.cu`:
  - `dsv4_cuda_tensor_alloc_fp4`: needs `*pt == NULL`, `O>=1`, `I>=32` and `I%32==0`; `cudaMalloc`s w (`O*I/2`) and scale (`O*I/32`); `fmt=4`.
  - `dsv4_cuda_tensor_fp4_buffers`.
  - `dsv4_cuda_wex_direct_available` (returns 1 when linked statically).
  - `DSV4_CUDA_STREAM_PRIORITY`: `inference_stream_priority()` and the two `cudaStreamCreateWithPriority` calls in `dsv4_cuda_init`.
- `.h`, loader and `.def` excerpts are next to it. The loader's `dsv4_cuda_wex_direct_available` requires all three of `tensor_alloc_fp4`, `tensor_fp4_buffers` and `stream_drain`.
- **`backend-upload-race-fix.patch`: history only, nothing to apply.** It records the October fix: `upload()` and the pageable refill used synchronous `cudaMemcpy` on the legacy stream, while the compute streams are non-blocking, so MoE kernels could read a fresh mirror before its scales landed.
  - Its base commit is unknown. Checked 2026-11-08: it applies, forward or reverse, to neither the engine of 2026-11-07 nor upstream 561966ec, nor to the parents of the last 8 engine commits that touch the file.
  - Both trees already hold an equivalent fix: the engine of 2026-11-07 in a refactored form (the `prof_*` "upload drain" and a stream-ordered refill), 561966ec through its own `h2d()` and a stream-ordered refill.
  - On any other tree, check that `upload()` and the pageable refill are ordered with the compute streams. That is a correctness dependency for any token-identity A/B on the GPU.

### 5.5 Environment variables

| Variable | Default | Effect | Read in |
|---|---|---|---|
| `COLI_WEX_PATH` | `<model_dir>\v4_flash_experts.wex` | container to open; a missing file is **silent** (no log line). The default path is **opt-out**: a container there is opened even with `COLI_EXPERT_STORE` and `COLI_WEX_PATH` both unset (verified 2026-11-08) | store open (A6); `expert_store_wex.c` |
| `COLI_MODEL_MIRROR` | unset | `;`/`,` list of mirror dirs (shards and/or the container under the same basename); also drives the upstream dual-SSD split | `coli_st_mirror_setup`, A6 |
| `COLI_WEX_EXTRA_MIRRORS` | unset | extra dirs that hold only the container | A6 |
| `COLI_DISK_WEIGHTS` | unset: startup probe (`v4_mirror_probe_bw`); standalone store 58/42 | primary/mirror share with one mirror | upstream `coli_st_mirror_setup` |
| `COLI_WEX_WEIGHTS` | unset: equal shares with 2+ mirrors; with 1 mirror, the shard dual-SSD route, which sends everything to the primary unless `COLI_MODEL_MIRROR` shard copies exist | `<primary>,<m1>[,<m2>,<m3>]`; the count must be 1 + mirrors, else ignored with a log line; read once per process | `v4_wex_pick` |
| `WEX_READ_RETRY` | 0 | 1 = one retry after 16 ms on a failed direct read | level B, standalone store |
| `COLI_WEX_CUDA_LIB` | unset: `kaira_wex_cuda.dll` next to the exe | absolute path to a helper; `0` forces the CPU decoder (and refuses WNR2 in the engine) | `wex_decode.c` |
| `COLI_WEX_GPU_DIRECT` | 0 | 1 = decode lanes go disk/RAM-cache -> VRAM; 2 = also the prefetcher. Needs helper feature 1 and the backend exports, else "staying on the host slab path" | `v4_wex_gpu_direct_enabled` |
| `V4_WEX_BATCH` | 0 (off) | 1 = decode a layer's WNR2 misses in one `wex_cuda_decode_batch_to_device` call; with mode 2, the prefetcher's same-layer guesses too. Needs `COLI_WEX_GPU_DIRECT>=1` and helper feature 8, else it logs `V4_WEX_BATCH needs COLI_WEX_GPU_DIRECT>=1 ... WNR2 misses stay on the per-record decode`. the engine of 2026-11-09 only, not in the 2026-11-07 excerpts | `v4_wexb_init` (deepseek_v4.c of 2026-11-09) |
| `COLI_WEX_GPU_DIRECT_PREFETCH_RANK` | 6 | mode 2: prefetch guesses with rank below it go direct | `coli_v4_gpu_wex_direct_prefetch` |
| `COLI_WEX_GPU_DIRECT_POOL` | 32 (0..256) | mode 2: resident mirrors that prefetch loads may take over (no new VRAM) | `v4_gpu_mirror_pool_slot` |
| `COLI_WEX_RAM_CACHE_GB` | 0 | RAM cache of compressed codec 3/4 records; needs `COLI_WEX_GPU_DIRECT>=1` | `v4_wex_crc` |
| `COLI_WEX_RAM_CACHE_PIN` | on | `0` = do not page-lock the arena | `v4_wex_crc` |
| `COLI_WEX_RAM_CACHE_VERIFY` | 0 | 1 = checksum on fill and on every hit (diagnostic) | `v4_wex_crc` |
| `COLI_WEX_ROOT` | unset | WEX4 only: 64 hex digits of the expected root (the SHA-256 of the header page). A container with another root is refused with `WEX4 root ... is not the pinned COLI_WEX_ROOT`. Not in the 2026-11-07 excerpts | the engine of 2026-11-11 `v4_wex4_load_manifest` (`integration/engine-wex4-loader-75932c1.patch`); `c/qwen38_wex.h` |
| `COLI_WEX_VERIFY` | on when `COLI_WEX_ROOT` is set, else off | WEX4 only, DeepSeek engine: `1` = the SHA-256 of each record is checked the first time the process reads it, and a record that fails is refused and served from the safetensors; `0` = off. Not in the 2026-11-07 excerpts | the engine of 2026-11-11 (`integration/engine-wex4-first-read-4ba5568.patch`) |
| `V4_WEX_STRIPE` (DeepSeek), `COLI_WEX_STRIPE` (qwen38) | 0 | 1 = with a replica, a record is read from two drives at once (the box at the top of this guide). In no excerpt or patch of `integration/` | the maintainer's `deepseek_v4.c` (not in this repo); `c/qwen38_wex.h` |
| `COLI_EXPERT_STORE` | `auto` | `wex` is the registry alias. With an engine it is the same as `auto` (section 5.6). | registry |
| `DSV4_CUDA_STREAM_PRIORITY` | 0 | 1 = inference streams at top priority, so helper kernels yield (trims tails only) | backend |

Prerequisites for modes 1 and 2:

- `DSV4_CUDA=1`, a `COLI_V4_GPU_TIER` build;
- `V4_VRAM_FIRST` on (default), `DSV4_HYBRID` off;
- mirror cache capacity >= 8 (`DSV4_CUDA_EXPERT_MIRRORS`);
- `DSV4_CUDA_VRAM_RESERVE_MB` (the engine of 2026-11-07: default 600, or 2800 with the MoE batch; the engine of 2026-11-09: 4096 for V4.1, the 600/2800 fallback for V4) large enough for the helper's thread-local VRAM lanes, which sit outside the backend's accounting (computed from the code: ~287 MiB per decoding thread on V4.1 for the batch path, plus 2 x record size for the single-record path).
- The RAM-cache prewarm runs only with `COLI_V4_TIER_PREWARM=1` and `COLI_V4_PREWARM` not 0.

### 5.6 The standalone store (`expert_store_wex.c`) and `COLI_EXPERT_STORE`

- With an engine (`coli_v4_expert_store_open_wex(engine != NULL)`), the `wex` backend calls `coli_v4_expert_store_open_planned`. The container then comes from the level A hooks.
- **Drop-in trap:** on stock colibri, linking `expert_store_wex.o` and setting `COLI_EXPERT_STORE=wex` without level A silently reads the safetensors.
- Only the legacy CLI path (engine NULL) uses the store's own reader:
  - single mirror, `COLI_DISK_WEIGHTS` default 58/42, 24 slots per layer;
  - no WNR2 helper guard: it falls back to the CPU decoder at ~0.3 s per expert;
  - a missing container returns -1 with an empty error string.

### 5.7 Expected log lines

| When | Line |
|---|---|
| container accepted | `[v4_wex] Container v1 active: <path> (40 layers, 192 experts, 18800640 decoded bytes/expert, direct_io=on)` |
| container accepted, later formats (not in the 2026-11-07 excerpts) | WNR3-CZ: `Container v3 WNR3 value-exact active` (section 14). WEX4: `[v4_wex] WEX4 container (hashed header and index)`, and with the first-read check `[v4_wex] WEX4 first-read SHA-256 of every record: on` |
| store mode | `v4_ssd_io mode=wex-direct-aligned fallback=buffered-pread` |
| mirror accepted / refused | `[v4_wex] Matching mirror container active: <path>` / `[v4_wex] Ignoring mismatched mirror: <path>` |
| custom split | `[v4_wex] read split over N drives: 40% 60%` |
| helper loaded (codec 3/4 or WNM2 only) | `[wex] Neural CUDA reconstruction enabled: <path>` or `[wex] Neural CUDA helper unavailable; using the CPU decoder.` |
| bad container | `[v4_wex] Invalid or incompatible container: <path>; using safetensors` |
| WNR2 without feature 4 | `[v4_wex] WNR2 container needs kaira_wex_cuda.dll with WNR2 support (helper feature 4) next to the engine` |
| wrong checkpoint | store open error `WEX layout does not match native expert: layer=L expert=E` (fatal) |
| GPU-direct on / off | `[v4_wex] GPU-direct Neural-rANS reconstruction enabled: ...` / `[v4_wex] COLI_WEX_GPU_DIRECT needs ... staying on the host slab path` |
| exit (mode 0) | `v4_direct reads=N flock_reads=N fallbacks=0 payload_bytes=B`: `fallbacks` must stay 0. With a container open, `flock_reads == reads` means every record came from the container: the WEX branch counts both, while the safetensors path, forced to per-matrix reads by A3, counts only `reads`. `flock_reads=0` means every record was skipped (not whole pages, 5.1). With WNR2, `payload_bytes` is the compressed size, smaller than the WEX1 run's |
| exit (GPU-direct) | `v4_wex_gpu_direct reads= payload_bytes= ram_skips= failures=0 device_decodes=`, mode 2 also `v4_gpu_mirror_epochs ...` |
| exit (RAM cache) | `v4_wex_ram_cache slots= ... refused= prewarmed= pinned=`, with VERIFY `v4_wex_ram_cache_verify checks= bad=0` |

### 5.8 application host (C#, outside colibri)

Excerpts are in `integration/host/`, from the engine of 2026-11-07.

- `Config.cs` `DefaultEnv` sets `COLI_EXPERT_STORE=wex`, `COLI_WEX_PATH=U:\models\DeepSeek-V4-Flash-0731\v4_flash_experts.wex` and `COLI_DISK_WEIGHTS=58,42`. That path is **stale**: the V4 container was deleted.
- `Engine.cs`:
  - `ApplyProfileEnv` copies the profile `env`. A V4.1 checkpoint on `deepseek_v41.exe` gets only `IsV41Env` names.
  - `COLI_MODEL_MIRROR` comes only from `mirror_dir`, and only when that dir holds `config.json`. Without a mirror, `COLI_DISK_WEIGHTS` is removed.
- `ColibriFamily.cs` `V41EnvNames` contains no WEX name (open item H1). V4.1 on `deepseek_v4.exe` (`COLI_FAMILY_ENGINE=deepseek_v4`) receives the whole env.
- `DeepSeekV41Preflight.cs` (2026-11-07 lines 243-246) warns on `COLI_EXPERT_STORE`/`COLI_WEX_PATH` that no V4.1 WEX container exists yet (Italian message; the code comment says "no V4.1 one exists yet"). **Stale**: the V4.1 WEX1 and WNR2 exist since 2026-11-07.
- On 2026-11-09 the application config profile `v41gpu` set `COLI_FAMILY_ENGINE=deepseek_v4`, `COLI_WEX_PATH`, `COLI_WEX_EXTRA_MIRRORS`, `COLI_WEX_WEIGHTS`, `COLI_WEX_GPU_DIRECT=2` and `V4_WEX_BATCH=1` (section 13); it did not set `COLI_EXPERT_STORE`, and no other profile had a WEX key. Which profiles read containers today and what they set: [STATUS.md](../../STATUS.md).

---

## 6. Generating containers

### 6.1 What the repacker accepts (DeepSeek V4 / V4.1 native MXFP4)

- One `--model-dir` with every `*.safetensors` shard. The index json is not used. Tensors must not repeat or overlap across shards.
- Records come only from `layers.{L}.ffn.experts.{E}.{w1,w2,w3}.{weight,scale}`:
  - weight: `I8`/`U8` (packed FP4, 2 per byte);
  - scale: `F8_E8M0`/`F8_E8M0FNU`, with `scale.cols*32 == weight.cols*2`.
  - `mtp.*` experts are skipped. A layer without experts gets all-zero entries.
- Geometry from `config.json`:
  - lookup: `text_config` if present, else the top level of `config.json`, keys `num_hidden_layers`, `n_routed_experts`, `hidden_size`, `moe_intermediate_size`. Only when one of them is missing is `inference/config.json` read, under `n_layers`, `n_routed_experts`, `dim`, `moe_inter_dim`;
  - defaults 43 x 256. When dims are given, shapes are checked.
- REAP-192 needs only shards 1-46; the engram shards 47/48 are not read.
- Output is written to a unique `.tmp` next to the target, checked by `validate_container` (layout only), then `os.replace`d. `--mirror-dir` publishes a second copy the same way.

### 6.2 WEX1, DeepSeek V4.1 Flash REAP-192 (A/B reference; profile v41gpu read the WNR2 container from 2026-11-09, section 13, and WNR3 since 2026-11-10, section 14; no WEX1 copy is kept)

```powershell
$env:PYTHONUTF8 = '1'
python -I c\tools\repack_v4_wex.py --model-dir U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 `
    --output D:\kaira_wex\v41-reap\v4_flash_experts.wex --format-version 1 --dry-run
python -I c\tools\repack_v4_wex.py --model-dir U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 `
    --output D:\kaira_wex\v41-reap\v4_flash_experts.wex --format-version 1
```

| | |
|---|---|
| Result (2026-11-07) | 40 x 192 = 7680 records of 18,800,640 B; file **144,389,472,256 B** = 4096 header + 552,960 table + 7680 records |
| Time | 115 s, rc 0. U: read at 6.6 GB/s; D: write-bound at ~1.25 GB/s |
| Disk | the full size free on the target drive (the `.tmp` sits next to it) |
| RAM | streams one expert record (~19 MB) at a time: tens of MB. Estimated from the code, not measured. |
| Determinism | a v1 header has no uuid, so a rebuild reproduces sha256 `4C055238D9011CE25D5C53A5EA61EC32BD8D1CF2886247194A13E7F53E7FA549` |
| Copies | none since 2026-11-10: the U: copy was deleted after the switch to WNR3 (section 14); the D: original was deleted on 2026-11-09 at 05:02. A rebuild reproduces the sha256 above. |

### 6.3 WEX1, DeepSeek V4 Flash 0731 (historical)

`python -I c\tools\repack_v4_wex.py --model-dir U:\models\DeepSeek-V4-Flash-0731` (the script's default `--model-dir`, C:\models\DeepSeek-V4-Flash-0731, now holds only config/tokenizer files; default output `<model_dir>\v4_flash_experts.wex`).

- Size: 43 x 256 = 11,008 records of 13,369,344 B; 147,170,537,472 B.
- Time: 258 s (2026-10-17).
- Status: the V4 container is deleted. The V4 weights are on U:\models\DeepSeek-V4-Flash-0731 (profile `v4`); the C: copy of the shards is deleted.

### 6.4 WEX2 and WEX3 (optional)

- `--format-version 2`: scale tiles.
  - A record compresses only when its scale bytes are a multiple of 4096 and the record shrinks.
  - V4/V4.1 records qualify; tiny test models usually do not.
- `--format-version 3` adds exact dedup, plus one of:
  - `--weight-codec palette3 --max-weight-relative-error X`: **lossy**;
  - `--weight-codec neural-rans --neural-cdf ... --neural-encoder-dll ...`: WNR1, codec 3, retired. Its encoder sources (`lab\wex_neural\encode_cuda.cu`, `train_lossless.py`) are not in this repo.
- Production used WEX1 or WNR2; since 2026-11-10 it uses WNR3 (section 14), since 2026-11-11 inside a WEX4 container (the box at the top of this guide).

### 6.5 WNR2 from a WEX1 (GPU)

1. Build `archive\nr2\wex_nr2_tools.dll`: `integration\build\build_wex_windows.cmd` (or `archive\nr2\build_tools.cmd`). Install numpy.
2. Run (the W1 command, from the repo root):

```powershell
python -I archive\nr2\build_nr2_full.py --source U:\kaira_wex\v41-reap\v4_flash_experts.wex `
    --out E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex --weight-chunk 2048 --scale-chunk 512 `
    --verify-workers 12 --min-free-gib 5 --report-dir D:\wnr2-reports\v41
```

- **Source:**
  - must be a raw WEX1: magic V1, version 1, flags 0, absent entries all zero;
  - every present record's total size must equal its raw size, a multiple of 4096. Real V4/V4.1 records are; tiny test models are not until `fixtures/widen_experts.py` widens them (7.5).
- **What it does** (details: WNR2-REFERENCE.md section 8):
  1. GPU counts over every expert, then the WNM2 model. It **always retrains**; id = FNV-1a of the 59,408-B blob.
  2. GPU encode per expert. Each record is verified by the CPU reference decoder before it is written. A record that would not shrink is stored raw (flags 0). Every 43rd is cross-checked against the CPU encoder.
  3. Header/model/table, `wex_nr2_check_container`, a slab-path readback sample, `os.replace` of `<out>.partial`, a whole-file sha256, and a per-record rehash.
- **`--report-dir`:** keep it outside the repo. It receives `build-full-nr2.json`, `model-full-nr2.wnm2` and `counts-full-nr2.npz`. The default is `archive\nr2\`.
- **Chunks:**
  - 2048:512 for V4.1: the single-record decode kernel is -54% for +0.34% bytes;
  - the default 4096:2048 was used for V4.

| | V4.1 REAP (W1, 2026-11-07) | V4 0731 (2026-10-20) |
|---|---|---|
| Output | 129,905,360,896 B, 7680/7680 WNR2, 0 raw fallbacks, 3.781 bits/weight, -10.03% | 128,368,873,472 B, -12.8% |
| Model | id 0xab4261bd = `models/v41-reap192.wnm2` (sha256 f83fee82...5e3e) | 0x49435c70 = `models/v4-flash-0731.wnm2` |
| Time (RTX 5090) | 723 s: training 62, encode+verify 373, readback+hash 288 | 779 s |
| sha256 | 3f703fa4d993ed46d2873676c71be07cf11c11c7baef3e0bce6428c30921eb9d | c0c91294...d388 (deleted) |
| Copies | none since 2026-11-10 (E: original and the U:/C: copies deleted after the WNR3 switch, section 14) | none |

- **Disk:** the builder refuses to start unless free space > 0.90 x source + `--min-free-gib`.
- **RAM:** not measured. Estimated ~1-1.5 GB: 4 queued raw slabs, plus up to 2 x workers pending buffers of ~19 MB, plus CPU verify scratch.
- **Reproducibility:** the header carries a uuid4 (bytes 40..55), so a rebuild has a different whole-file sha256. Compare the table and the records, not the file hash.
- **A shipped model:** any embedded model is lossless; a better-fitted one is just smaller. `archive\nr2\precheck_records.py WEX1 MODEL.wnm2 OUT.bin [count]` encodes a few experts with an existing model (V4 model on V4.1: 0.941-0.950 of raw, against 0.8997 for the V4.1 model). `build_nr2_full.py` has no option to embed an existing model.

### 6.6 Qwen3.8: MXFP4 containers since 2026-11-11; the FP8 proposal below was not built

- **Since 2026-11-11 Qwen3.8-Flash-Next reads its routed experts from these containers**, from an MXFP4 export of the model and not from the FP8 checkpoint: one expert is one V4.1-style record. Store `c/qwen38_wex.h`, repacker `c/tools/repack_q38_mxfp4_wex.py` (Quark shards to WEX1), then `c/tools/build_nr3.py` or `wex4/wex4_convert.py upgrade FILE --names qwen38` as for V4.1. Sizes, quality against FP8, speed and the installed form: [QWEN38-MXFP4-20261111.md](QWEN38-MXFP4-20261111.md).
- The rest of this section is the state of 2026-11-09 for the **FP8** checkpoint and is kept as history. On that date there was no WEX code, container or generator for Qwen3.8: `qwen38.c` and `qwen38_core.h` had 0 WEX references.
- `repack_v4_wex.py` cannot read the FP8 checkpoint: it needs DeepSeek MXFP4 names and dtypes, and those experts are FP8.
- WNR2 does not apply to FP8 either: it is an MXFP4 codec.
- The measured facts and the only FP8 design are in [wex-prod-20261105.it.md](wex-prod-20261105.it.md) (section "Proposta"); the probes are in `research/q38-wex-q8-20261105/`.
  - FP8 entropy is 6.53 bits/byte, so lossless compression would save at most ~18%. At ~3 ms CPU per expert, that is not the lever.
  - The lever is striping across a second fast NVMe: C:+U: alternating reads gave 6.60 GiB/s, against 3.34 for C: alone and 2.23 for U: alone.
  - **WEX-Q8 proposal, never approved and not built (the MXFP4 route replaced it); it needed the owner's OK on C: space:**
    - aligned raw FP8 records of 4,915,200 B (1,200 pages), byte-identical to the shards, scales kept resident;
    - ~65 GiB for a 58% share of the experts, 112.5 GiB for all of them;
    - each miss read from the less loaded drive.
  - Work it would need:
    - a repacker with a byte-for-byte check;
    - a C reader with tests;
    - a hook in `q38_load_native_fp8_ranges` (`qwen38_core.h`) and in the tier path around `st_read_range_raw_cap`;
    - an A/B in tok/s on a quiet machine.

---

## 7. Verifying

### 7.1 WEX1/WEX2: bytes against the checkpoint

```powershell
python -I scripts\verify_wex1.py --model-dir U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 `
    --wex U:\kaira_wex\v41-reap\v4_flash_experts.wex [--stride 37]
```

- It validates the metadata, checks the geometry against `config.json`, then compares every Nth present expert byte for byte: scales after WEX2 tile decode, and weights.
- Exit 0 means identical. It prints `FAIL ...` and exits 1 on the first difference.
- A full V4.1 pass reads both 144 GB sides; use `--stride` for a sample.
- `scripts/test_verify_wex1.py` covers a synthetic WEX1/WEX2, including corrupted copies. Checked by hand on 2026-11-08: `fixtures/v41-tiny` (WEX1 and WEX2, 48/48), the synthetic 2x8 (16/16) and the widened tiny models of 7.5 (12/12 and 48/48). Not yet run on the full V4.1 container.
- The engine checks only shapes and offsets (A7), never bytes.

### 7.2 sha256 and the `.verified` marker

- A copy is usable only when `<file>.verified` exists. Lab scripts read the marker; the engine and the host do not.
- Recommended procedure:

```powershell
$src = 'U:\kaira_wex\v41-reap\v4_flash_experts.wex'; $dstDir = '<target dir>'   # since 2026-11-10 no WEX1 copy is left: regenerate it first (6.2)
robocopy (Split-Path $src) $dstDir (Split-Path $src -Leaf) /COPY:DAT /J /R:1 /W:1
$dst = Join-Path $dstDir (Split-Path $src -Leaf)
$a = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
$b = (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash
if ($a -ne $b) { throw "sha256 differs: $a vs $b" }
[IO.File]::WriteAllText("$dst.verified", "sha256 $b`r`n")   # 73 B, ASCII, no BOM: the WEX1 marker format
```

- The WNR2 markers on U: and C: were written by `Set-Content -Encoding utf8`, so each has a BOM and extra text after the hash. Parse it as "starts with `sha256 `", or rewrite it in the format above.
- Known hashes:
  - V4.1 WEX1 `4C055238...A549`;
  - V4.1 WNR2 `3F703FA4...EB9D`;
  - models `f83fee82...` (V4.1) and `2ddd4220...` (V4).

### 7.3 WNR2: records against the WEX1

- **GPU batch check**, the W1 64-record test. It needs >= 16 records; 64 records write a 2.28 GB `.bin`, so put it outside OneDrive:

```powershell
python -I archive\nr2\sample_records.py U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex U:\kaira_wex\v41-reap\v4_flash_experts.wex <dir outside OneDrive>\records64.bin 64 20261107
out\test_decode_batch.exe out\kaira_wex_cuda.dll <dir outside OneDrive>\records64.bin
```

  - `sample_records.py` asserts the WNR2 header, equal geometry, entry fields 2..21 equal to the WEX1, and the model id.
  - `test_decode_batch` decodes at batch sizes 1,2,3,4,6,8,10,16 and compares every byte. Expected: `ALL OK`. `failed:2` is the two deliberate corruption cases.
- **Independent C checker:** `out\check_container.exe <wnr2.wex> <source-v1.wex> 0 [stride]`. It uses only the engine headers.
- **Every record through the helper:** `python -I archive\nr2\check_all.py out\kaira_wex_cuda.dll <v1.wex> <nr2.wex> [--json out.json]`. On V4: 11008/11008 in 187.7 s. **Never run on V4.1**: the build verified all 7680 records on the CPU reference, and 64 on the GPU, and the V4.1 WNR2 container was deleted on 2026-11-10.
- **Lab-only:** `archive\nr2\test_nr2_tools.py` and `test_nr2_cuda.cu` need the 9.2 GB V4 pilot containers (now in `Q:\kaira-temp\U_kaira_build_wexgd-20261019`). `test_nr2_cuda.cu` includes `../c/wex_cuda.cuh`.

### 7.4 Unit and engine tests

| Test | How | Needs |
|---|---|---|
| `c/tests/test_repack_v4_wex.py` (30 tests), `scripts/test_verify_wex1.py` | `python -I c\tests\test_repack_v4_wex.py`, `python -I scripts\test_verify_wex1.py` from the repo | Python only |
| `c/tests/test_wex_format.c` | `make tests/test_wex_format.exe` in colibri `c/`, run it | level-free |
| `c/tests/test_expert_store_wex.c` | `make tests/test_expert_store_wex.exe`; uses `COLI_WEX_CUDA_LIB=0` | `expert_store_registry.o` |
| `c/tests/test_wex_planned.c` | `integration/build/build_test_wex_planned.sh`, then `test_wex_planned <MODEL_DIR> <sample-vN.wex> [--expect-fallback\|--expect-cuda]` | the maintainer's tree (2026-11-07) and a fixture |
| `c/tests/test_v4_gpu_priority.c`, `c/tests/test_v4_mirror_guard_source.py` | the fork's Makefile rule 2655-2657; `python tests/test_v4_mirror_guard_source.py` inside `c/` | level C + fork hooks |

- **`test_wex_planned`**: the engine's HOT expert store (`COLI_V4_TEST_HOOKS`) reads a 1 x 2 container and compares every scale/weight byte, prefetch and hit with the safetensors. Build and run (verified 2026-11-08 on a copy of the engine of 2026-11-07):

```sh
# fixture: 1 layer x 2 experts, 512 x 256, records of 51 pages; WEX1, WEX2 and WNR2 of it
python -I fixtures/make_synth_mxfp4.py out/planned --layers 1 --experts 2
python -I c/tools/repack_v4_wex.py --model-dir out/planned --output out/planned/sample-v1.wex --format-version 1
python -I c/tools/repack_v4_wex.py --model-dir out/planned --output out/planned/sample-v2.wex --format-version 2
python -I archive/nr2/build_nr2_full.py --source out/planned/sample-v1.wex --out out/planned/sample-nr2.wex \
    --weight-chunk 2048 --scale-chunk 512 --verify-workers 2 --min-free-gib 1 --report-dir out/reports-planned
# build (MSYS2 UCRT64 login shell, which starts in $HOME: cd first): it also runs
# `make deepseek-v4` and `make tests/test_deepseek_v4.exe` in the colibri tree
MSYSTEM=UCRT64 C:/msys64/usr/bin/bash.exe -lc "cd /c/path/to/Wex-WRN2 && sh integration/build/build_test_wex_planned.sh /c/path/to/colibri/c out/wp"
COLI_WEX_CUDA_LIB=0 out/wp/test_wex_planned.exe out/planned out/planned/sample-v1.wex
COLI_WEX_CUDA_LIB=0 out/wp/test_wex_planned.exe out/planned out/planned/sample-v2.wex
COLI_WEX_CUDA_LIB='C:\abs\kaira_wex_cuda.dll' out/wp/test_wex_planned.exe out/planned out/planned/sample-nr2.wex --expect-cuda
COLI_WEX_CUDA_LIB=0 out/wp/test_wex_planned.exe out/planned out/planned/sample-nr2.wex --expect-fallback
```

  - Expected: `WEX HOT planned: container version=1 ... direct_fallbacks=0 ...; all scale/weight bytes, prefetch and hits passed.`; version 2 reads 409,600 of 417,792 B; WNR2 with the helper `version=3 read_bytes=327680 ... cuda_decodes=2`; WNR2 with `COLI_WEX_CUDA_LIB=0` `safetensors-fallback version=3`.
  - The script fixes the old lab recipe (`lab\run_wex_validation.py:150-178`), whose link fails at the engine of 2026-11-07 with undefined `coli_v41_engram_open`/`_close`: since the V4.1 port they live in `COLI_V4_UNIT_ATTENTION`, so the script links the engine's `COLI_V4_UNIT_ATTENTION.o`. A plain `-O2` compile of that unit pulls in the compressor and indexer units; the engine's `-flto` object does not.
  - The test needs engine test hooks; it is not part of the upstream patch.
  - The old fixture (V4 layer 10, experts 0 and 17, 27 MB) came from the deleted V4 checkpoint; the synthetic one replaces it.

### 7.5 Engine A/B: token identity, safetensors vs container

The proof that a container works is the engine's own oracle: record a greedy run on plain safetensors, then replay it with the container and require every token to match.

**Fixtures.** A container is only read when its records are whole 4096-byte pages (5.1), and WNR2 can only be built then (6.5). The tiny engine models are not, so widen them first.

| Fixture | Runs on | Record | Container read? | WNR2? |
|---|---|---|---|---|
| `fixtures/v41-tiny` (V4.1, 6 layers x 8 experts, dim 128, moe_inter 64) | the maintainer's engine | 13,056 B | no: accepted, records skipped (`flock_reads=0`) | no (builder asserts) |
| colibri `deepseek_v4_tiny` (V4, 3 x 4, hidden 128, moe 128), from `python -I <c>/tools/make_deepseek_v4_tiny.py --output DIR` (needs torch + transformers with `DeepseekV4ForCausalLM`; model sha256 `998511e4...8a9f` with torch 2.14.1+cpu, transformers 5.20.2) | upstream and the fork | 26,112 B | no, as above | no |
| either one widened: `python -I fixtures/widen_experts.py --src DIR --out DIR2 --moe-inter 1024` | same engines | 208,896 B (51 pages) | **yes** | **yes** |
| `fixtures/make_synth_mxfp4.py` (experts only, 512 x 256) | none: store tests only (7.3, 7.4) | 208,896 B | n/a | yes |

`widen_experts.py` rewrites only the routed and shared expert matrices (backbone and mtp) to the new moe_inter, drawing bytes from each tensor's own distribution (seeded, deterministic), and updates `config.json`. Every other tensor stays byte-identical. The result is a different model: its tokens differ from the source, and `ref.json` is not copied.

**Recipe** (Git Bash syntax for the variables; build and run the engine as in 4.1; `W` holds absolute Windows paths, as tested):

```sh
W='C:\work\wexab'; X=/c/path/to/colibri/c/deepseek_v4.exe; HELPER='C:\path\to\kaira_wex_cuda.dll'
export PYTHONUTF8=1 DSV4_CUDA=0 CTX=128 V4_SKIP_COLD=0 V4_MTP=0 COLI_V4_SAVE_USAGE=0
unset COLI_EXPERT_STORE COLI_WEX_PATH COLI_WEX_CUDA_LIB
python -I fixtures/widen_experts.py --src fixtures/v41-tiny --out "$W\m" --moe-inter 1024
python -I c/tools/repack_v4_wex.py --model-dir "$W\m" --output "$W\m.v1.wex" --format-version 1
python -I scripts/verify_wex1.py --model-dir "$W\m" --wex "$W\m.v1.wex"
python -I archive/nr2/build_nr2_full.py --source "$W\m.v1.wex" --out "$W\m.nr2.wex" --weight-chunk 2048 --scale-chunk 512 \
    --verify-workers 4 --min-free-gib 1 --report-dir "$W\reports"
python -I archive/nr2/check_all.py "$HELPER" "$W\m.v1.wex" "$W\m.nr2.wex"
# 1. reference: plain safetensors (no WEX variable, no container in the model dir)
"$X" "$W\m" "hello world, this is a test" --raw-prompt --max-tokens 12 --no-dspark --record-oracle "$W\st.json"
# 2. each container replays that recording
COLI_EXPERT_STORE=wex COLI_WEX_PATH="$W\m.v1.wex" "$X" "$W\m" --oracle "$W\st.json" --greedy 12
COLI_EXPERT_STORE=wex COLI_WEX_PATH="$W\m.nr2.wex" COLI_WEX_CUDA_LIB="$HELPER" "$X" "$W\m" --oracle "$W\st.json" --greedy 12
COLI_EXPERT_STORE=wex COLI_WEX_PATH="$W\m.nr2.wex" COLI_WEX_CUDA_LIB=0 "$X" "$W\m" --oracle "$W\st.json" --greedy 12
```

Pass criteria for every `--oracle` run:

- exit code 0, `PREFILL (teacher-forcing) C vs oracle: N/N positions` and `GREEDY C vs oracle: 12/12 tokens`;
- `[v4_wex] Container v1 active` (WEX1) or `Container v3 active` plus `[wex] Neural CUDA reconstruction enabled` (WNR2);
- `v4_direct reads=R flock_reads=R fallbacks=0`. With WNR2, `payload_bytes` is smaller than in the WEX1 run;
- with `COLI_WEX_CUDA_LIB=0`: `WNR2 container needs kaira_wex_cuda.dll ... (helper feature 4)`, then `Invalid or incompatible container ...; using safetensors`, and still 12/12.

Notes:

- `--greedy` must equal the recorded `--max-tokens`. The default is 20, so a 12-token recording fails with `greedy length mismatch expected=12 got=20`, rc 1.
- `deepseek_v4_tiny` has a 128-token vocabulary: use a token-id prompt, `"<t005><t007><t009><t011><t013><t017><t019><t023>"`, as its own test does. A text prompt stops with `V4 prompt must encode to between 1 and 128 tokens`.
- The `--record-oracle` file holds the model path with unescaped backslashes. The engine reads it back; Python `json.load` does not, so read `prompt_ids`/`full_ids` with a regex.
- Run the reference with no container at `<model_dir>\v4_flash_experts.wex` (the default path is opt-out, 5.5).
- The same recipe works on a stock colibri built with the level A patch; use `deepseek_v4_tiny` there, since upstream `deepseek_v4.exe` does not run V4.1.

### 7.6 Verified end to end (2026-11-08)

Windows 11, RTX 5090 (used only by the helper decodes), MSYS2 UCRT64 gcc 16.2, CUDA 13.4, Python 3.14.8 + numpy 2.5.3. Every engine run was CPU-only (`DSV4_CUDA=0`). Builds, encodes and engine runs ran under `<TREE>\lab\v41-gpu-20261107\heavy.lock`. Two passes: a fresh-reader check of this guide (repo files copied as a fresh clone via `git archive`), then the fixes in this revision.

| # | What | How | Result |
|---|---|---|---|
| 1 | Python tests | `c/tests/test_repack_v4_wex.py`, `scripts/test_verify_wex1.py` | 30/30, 1/1 |
| 2 | Excerpts and copies | each `==== file:a-b` section of `integration/` against the engine of 2026-11-07; `src/`, `tools/`, `tests/` (now under `c/`) against the engine of 2026-11-07 | 53/53 sections match; 20 files byte-identical |
| 3 | Helper, encoder DLL, checkers | `integration\build\build_wex_windows.cmd` (4.3) | rc 0, 10 s; helper 387,072 B, ABI 1, features 15 (the bridge of that date; today's returns 63, section 4.3) |
| 4 | the maintainer's engine | 4.1 command on a `git archive` copy of the engine of 2026-11-07 with the repo files copied in | rc 0, 17 s, 1,522,381 B |
| 5 | Unit tests | `make tests/test_wex_format.exe tests/test_expert_store_wex.exe` (4.2) | both pass, WNR2 round trips included |
| 6 | Level A on stock upstream | copy `src/` (now `c/`), `git apply` the patch on `git archive 561966ec c`, 4.1 command | applies clean; rc 0, 21 s, 1,815,173 B |
| 7 | WEX1/WEX2 from v41-tiny | `repack_v4_wex.py` + `verify_wex1.py` | 48 records, 794,624 B, rebuild byte-identical; verify 48/48 for both |
| 8 | WNR2 pipeline, synthetic 2 x 8 | `make_synth_mxfp4.py` -> repack -> `build_nr2_full.py` -> `check_container.exe`, `sample_records.py` + `test_decode_batch.exe`, `check_all.py` | 3,350,528 -> 2,691,072 B (-19.7%), model 0x3cacd9c3; 16/16 exact in every checker; `test_decode_batch` ALL OK |
| 9 | Store test | 7.4, synthetic 1 x 2 | v1, v2, WNR2 with the helper (`cuda_decodes=2`) and WNR2 with `COLI_WEX_CUDA_LIB=0` all pass |
| 10 | **Upstream 561966ec + patch**, `deepseek_v4_tiny` widened to 1024 | 7.5 | WEX1 and WNR2: TF 20/20, greedy 12/12 (ids 24..35), `reads=12 flock_reads=12 fallbacks=0`; WNR2 payload 2,256,896 B vs 2,506,752; WNR2 2,326,528 B vs WEX1 2,514,944 B, 12/12 `check_all`; `COLI_WEX_CUDA_LIB=0`: refused, safetensors, 12/12 |
| 11 | **the engine of 2026-11-07**, v41-tiny widened to 1024 | 7.5 | WEX1 and WNR2: TF 32/32, greedy 12/12, `reads=46 flock_reads=46 fallbacks=0`; WNR2 payload 8,667,136 B vs 9,609,216; WNR2 9,113,600 B vs WEX1 10,035,200 B, model 0xa3818c64, 48/48 `check_all`; refusal path 12/12 |
| 12 | the engine of 2026-11-07, `deepseek_v4_tiny` widened | 7.5 | WEX1 and WNR2: TF 20/20, greedy 12/12, `flock_reads=12` |
| 13 | Container handling (fork, v41-tiny unwidened) | 7.5 with WEX2, a wrong-geometry container, a missing path, and a container at the default path | WEX2 `Container v2 active`; wrong geometry `Invalid or incompatible container ... using safetensors`; missing path silent; default path opened with no WEX variable set; all 12/12. These records were skipped (`flock_reads=0`), so this row proves container handling, not container reads |
| 14 | Upstream + patch, `deepseek_v4_tiny` unwidened | 7.5 | 20/20, 12/12, container active, `flock_reads=0` (skipped, as 5.1 says) |
| 15 | The recipes exactly as written | a fresh copy of the repo (tracked and new files, no build outputs): 4.3 with the default `out\`, then 7.4 and 7.5 line by line | all pass: tools rc 0 in 10 s; the four `test_wex_planned` runs; WEX1 and WNR2 TF 32/32 + greedy 12/12 with `flock_reads=46`, the refusal path 12/12; Python tests OK |

Not verified: GPU-direct (modes 1/2) and the DSV4 backend DLL build; any `DSV4_CUDA=1` run; the real V4.1 containers in an engine built from this repo; levels B/C on upstream; `archive/nr2/precheck_records.py`, `archive/bench/`, `research/`; Linux.

---

## 8. Using

### 8.1 Engine command lines

- One prompt from the command line: `deepseek_v4.exe <model dir> "<prompt>" [--raw-prompt] [--max-tokens N] [--no-dspark]`. The WEX variables go in the environment. `--record-oracle FILE` and `--oracle FILE --greedy N` are the token-identity check of 7.5.
- Server mode is entered only with `SERVE=1`; it then reads `SNAP` (required), `CTX`, `NGEN`, `RAM_GB` (the line protocol that `coli serve` and `lab\v41-gpu-20261107\m0\serve_bench.py` use). Without `SERVE=1` the exe is the one-shot CLI (`deepseek_v4.exe <dir> <prompt> --memory-gb N`).
- Through the colibri launcher: `python coli serve --model <dir> --ram <GB> --ctx <n> ...`. `coli` passes `COLI_EXPERT_STORE`/`COLI_WEX_PATH` through.

**V4.1 REAP, CPU MoE, WEX1 (mode 0)**, the M0 setup. Sh syntax; set the same variables in PowerShell.

```sh
export SNAP='U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
export COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'
export COLI_WEX_PATH='U:\kaira_wex\v41-reap\v4_flash_experts.wex'
export DSV4_CUDA=0 RAM_GB=16 CTX=4096 V4_SKIP_COLD=0 V4_MTP=0 COLI_V4_SAVE_USAGE=0 SERVE=1
deepseek_v4.exe          # one-shot instead: deepseek_v4.exe "$SNAP" "<prompt>" --memory-gb 16 (RAM_GB is read only in SERVE mode); through the host: COLI_FAMILY_ENGINE=deepseek_v4
```

- Measured (README): 1.11-1.14 decode tok/s against 0.98-1.01 for safetensors on the same drive; TTFT 11-12 s against 13-14 s.

**V4.1 REAP, WNR2 (mode 0):**

- Set `COLI_WEX_PATH=U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex`.
- Put a feature-4 `kaira_wex_cuda.dll` next to the exe, or set `COLI_WEX_CUDA_LIB=<path>`.
- **A full mode-0 run on V4.1 is untested**; the host-decode path itself served ~105k WNR2 records per W3 run with 0 fallbacks. In mode 0 every record goes through `decode_to_host` plus a D2H of 18.8 MB.

**V4 on the GPU with GPU-direct (WNR2):**

- On top of the usual `DSV4_CUDA=1` decode settings, set:
  - `COLI_WEX_PATH=<nr2.wex>`, `COLI_WEX_GPU_DIRECT=2`;
  - optionally `COLI_WEX_RAM_CACHE_GB=4`;
  - `DSV4_CUDA_VRAM_RESERVE_MB=4096`, `DSV4_CUDA_STREAM_PRIORITY=1`.
- For an A/B, pin `COLI_DSV4_DLL` and `COLI_WEX_CUDA_LIB` and record their sha256.
- At the engine of 2026-11-07, V4.1 MoE was CPU-only. Since the engine of 2026-11-09, V4.1 runs its MoE on `moe_token_pipeline` with `DSV4_CUDA=1`, so `COLI_WEX_GPU_DIRECT=2` and `V4_WEX_BATCH=1` also work on V4.1 (sections 12-13, profile `v41gpu`).

**Deploy rule:** ship a new engine together with a helper that has feature 4.

- An engine with the WNR2 guard and an old helper refuses WNR2 and falls back to safetensors, with a log line.
- Since the mirror switch of 2026-11-09 06:17, the deployed `app\colibri-optimized\kaira_wex_cuda.dll` is cceb4fb5... (387,072 B). This is the W1 build that passed the V4.1 64-record check and that W3 measured. The previous helper, a7977200..., is kept in `lab\app-before-mirror-20261109-061724`.

### 8.2 Application host profile

- Put the variables in the model profile's `"env"`, for example `"COLI_WEX_PATH": "U:\\kaira_wex\\v41-reap\\v4_flash_experts.wex"`.
- `"mirror_dir"` becomes `COLI_MODEL_MIRROR` only if that folder holds `config.json`. For a container-only copy, use `COLI_WEX_EXTRA_MIRRORS` in `"env"` instead.
- V4.1 needs `COLI_FAMILY_ENGINE=deepseek_v4` (deepseek_v4.exe) to receive WEX variables at all. Expect the stale preflight warning (5.8).
- **Turning WEX off is not just removing the variables.** The engine also opens `<model_dir>\v4_flash_experts.wex` when no WEX variable is set (5.5). A container left in the model folder silently changes the read path, and a container from another checkpoint is either refused (different layer/expert count; safetensors are used) or fails the store open (same count, different shapes, A7). Rename or move it to disable WEX.
- Host fixes that are documented but not made (the engine source is read-only for this work):
  - `Config.cs` `DefaultEnv` path;
  - the preflight text;
  - H1: WEX names for V4.1 on `deepseek_v41.exe`, which does not read them anyway.

### 8.3 Drive placement and striping

M0 drive probe, 2026-11-07: random 18.8 MB reads, QD8.

| Drive | Link | GB/s | QD1 p50 | Holds |
|---|---|---|---|---|
| C: Crucial T700 | gen5 x4 | 10.66 | 2.3 ms | WNR2 mirror (since 2026-11-09); C:\models\...engram is a junction to D: |
| U: SK hynix | gen4 x4 | 6.58 | 3.2 ms | REAP shards 1-46, WEX1 + WNR2 copies (14.6 GB free after the WNR2 copy) |
| E: ADATA SX8200 | gen3 x4 | 2.29 | 8.3 ms | WNR2 original, engram shard 48 |
| D: Sabrent | PCIe 3 x2 | 1.61 | 12.2 ms | engram shard 47 (moved from C: on 2026-11-09); the WEX1 copy was deleted |

- Put the primary container on the fastest drive that does not also serve the hot shards of the same run.
- Each extra replica must be a byte copy under the **same basename**. Verify it (7.2) before use.
- With one replica and no `COLI_WEX_WEIGHTS`, the split follows the shard route (`COLI_DISK_WEIGHTS` or the startup probe), which only exists when `COLI_MODEL_MIRROR` registers shard copies. For a container-only mirror set `COLI_WEX_WEIGHTS`.
- With 2-3 replicas and no `COLI_WEX_WEIGHTS`, the split is equal. Set `COLI_WEX_WEIGHTS=<primary>,<r1>[,...]` for a custom split, and always set it for a single container-only replica. The split is a fixed hash per (layer, expert). Weights proportional to the measured GB/s are a starting point, not a measured optimum.
- Up to 3 replicas.

### 8.4 GPU-direct and the RAM record cache

- **Mode 1** removes the host round trip for VRAM-first misses of codec 3/4 records.
- **Mode 2** also lets the prefetcher load its top guesses into pool mirrors. No new VRAM: the pool takes over the coldest idle resident mirrors.
- **The RAM record cache** keeps compressed records (slot = the largest codec 3/4 record), so the same RAM holds more experts than WEX1 slabs do. For V4.1 (-10.03%) that is about 11% more records at equal bytes; PLAN 3.2 estimated ~14% from V4's -12.8%. That capacity was not tested: W3 (section 12) ran with this cache off in both arms, and GPU-direct records bypass the host RAM tier. The arm that tests it (pilot C: RAM_GB 12 + COLI_WEX_RAM_CACHE_GB 4) has not run yet.
  - Decision rule (owner decision 2026-11-08 21:15, replacing the +3% rule of PLAN 3.2): WNR2 becomes the container if the full ABBA gives identical tokens and WNR2 is not slower. W3 (same drive U:) measured -1.80% +/- 0.65%, so on one drive WNR2 stays a capacity format. The installed `v41gpu` profile still uses WNR2, because only WNR2 fits on C: for the U:+C: mirror (section 13). The ABBA of WNR2 + mirror against WEX1 on U: alone has not been run yet. *(Run 2026-11-10: +49% for WNR2 + mirror, one pair; WNR3 + mirror replaced it, section 14.)*
- Lossless A/B settings (PLAN 3.1):
  - `V4_SKIP_COLD=0` (the engine default is already off. the application host writes 1 only when the V4 UI switch `v4_skip_cold` in the application config is on, which it currently is not, and it always writes 0 for V4.1 on deepseek_v4.exe.), `V4_TOPK` unset, `V4_MTP=0`, temperature 0;
  - fixed binaries; a quiet machine.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| no `[v4_wex]` line at all | `CreateFileA` failed: wrong `COLI_WEX_PATH`, or no file at `<model_dir>\v4_flash_experts.wex` (silent by design) | check the path; the engine reads safetensors |
| `[v4_wex] Container ... active` although no WEX variable is set | a container sits at `<model_dir>\v4_flash_experts.wex`: the default path is opt-out | rename or move it to disable WEX (8.2) |
| `Container ... active` but `v4_direct ... flock_reads=0` | records are not whole 4096-byte pages: every record is read from the safetensors, with no fallback counted (5.1) | expected on `v41-tiny` and `deepseek_v4_tiny`; widen them (7.5). Real V4/V4.1 records are page multiples |
| `Invalid or incompatible container ... using safetensors` | header/table invalid, geometry differs from `config.json`, truncated copy, WNR2 model unreadable or id mismatch | `python -I scripts\verify_wex1.py ...` (WEX1/2) or `check_container.exe` (WNR2); compare sha256 |
| `WNR2 container needs kaira_wex_cuda.dll ... (helper feature 4)` | no helper, an old helper, or `COLI_WEX_CUDA_LIB=0` | deploy a helper from 4.3 next to the exe |
| store open fails: `WEX layout does not match native expert` | container with the same layer/expert count but different expert shapes. A container with a different layer/expert count (e.g. a V4 container, 43 x 256, on V4.1, 40 x 192) is refused instead with `Invalid or incompatible container ...; using safetensors` | unset `COLI_WEX_PATH` or rebuild from this checkpoint |
| `v4_direct ... fallbacks>0` | read or decode failures, served from safetensors | verify the container; check the drive; `WEX_READ_RETRY=1` for transient errors |
| `Ignoring mismatched mirror` | the replica differs in size, header or table | recopy and verify (7.2) |
| `COLI_WEX_WEIGHTS '...' ignored` | wrong count (must be 1 + replicas) or a non-positive value | fix the list |
| `COLI_WEX_GPU_DIRECT needs ... staying on the host slab path` | helper lacks feature 1, or the DSV4 DLL lacks `tensor_alloc_fp4`/`fp4_buffers`/`stream_drain` | rebuild the helper / backend (4.3, 4.4) |
| `v4_wex_gpu_direct` line missing or `reads=0` | container is WEX1/WEX2 (raw records never go direct), experts already in RAM (`ram_skips`), a CPU-MoE run (`DSV4_CUDA=0`, V4.1 on an engine older than the engine of 2026-11-09, or on a DSV4 DLL without `dsv4_cuda_set_act_group`), or `DSV4_HYBRID` | expected; check the container type |
| VRAM out of memory, `kv comp allocation` errors | helper lanes allocate outside the backend's accounting | raise `DSV4_CUDA_VRAM_RESERVE_MB` |
| tokens differ between WEX1 and WNR2 on the GPU | backend upload race (5.4) or a real bug | check that `upload()` and the refill are stream-ordered (5.4); stop the A/B |
| `--oracle` exits 1: `greedy length mismatch expected=12 got=20` | `--greedy` defaults to 20 | pass `--greedy` equal to the recorded `--max-tokens` |
| `V4 prompt must encode to between 1 and 128 tokens` | text prompt on `deepseek_v4_tiny` (128-token vocabulary) | use a token-id prompt such as `"<t005><t007><t009>"` (7.5) |
| Python `json.load` fails on a `--record-oracle` file | the model path is written with unescaped backslashes | read the ids with a regex |
| make from Git Bash: `Cannot create temporary file in C:\WINDOWS\`, Error 127 | gcc started outside an MSYS2 environment | build from an MSYS2 UCRT64 login shell (4.1) |
| `test_wex_planned` link: undefined `coli_v41_engram_open`/`_close` | old recipe without `COLI_V4_UNIT_ATTENTION.o` | use `integration/build/build_test_wex_planned.sh` (7.4) |
| `repack_v4_wex.py` / `verify_wex1.py`: `unsupported WEX header flags` | the file is a WNR2 (flag 16) | use the nr2 checkers |
| `build_nr2_full.py` assertion on the source | records are not 4096-multiples (tiny fixtures) or the source is not raw WEX1 | use a real V4/V4.1 WEX1, or widen the tiny model with `fixtures/widen_experts.py` (7.5) |
| `ModuleNotFoundError: nr2lib` / `OSError` loading `wex_nr2_tools.dll` | DLL not built, or a lab copy of the script | build with 4.3; use the repo scripts (paths fixed 2026-11-08) |
| non-Windows build error on `wex_handle` | A11 outside `#ifdef _WIN32` | guard it (5.1) |
| `vswhere.exe non riconosciuto` while building | harmless vcvars message | ignore |

---

## 10. Known limits and open work

- **Windows only.** Off Windows the store fields exist but are never opened, and every CUDA helper path reports unavailable.
- **V4.1 on the GPU:** G2 and W2 have been in the engine since 2026-11-09 (section 12); they are not in the engine of 2026-11-07, in the excerpts or upstream, and no upstream patch carries them. The engine calls the batch API (feature 8) when `V4_WEX_BATCH=1` (deepseek_v4.c:19701 at the engine of 2026-11-09).
- **WNR2 on V4.1** has run in the engine on the real container with GPU-direct mode 2, on the engine of 2026-11-09. W3 made 4 runs, token- and logit-identical to WEX1, at -1.80% decode; the 2026-11-09 mirror check is in section 13 (41,271 device decodes, 0 failures). In those runs the host-decode path also served about 105k WNR2 records per run, with 0 fallbacks. Not done yet: a full mode-0 run (`DSV4_CUDA=0`) on V4.1, and `check_all.py` on V4.1. The engine path itself is proven on tiny models: full CPU forward passes with WNR2, token-identical to safetensors, on the engine of 2026-11-07 and on upstream + the level A patch (7.6).
- **Upstream patch:** only level A, only for 561966ec. Levels B/C have no upstream patch: they need the fork's infrastructure of 5.3. WNR3, WEX4 and Qwen3.8 have none either (the box at the top of this guide).
- **Silent skip:** a record that is not a whole number of pages is read from the safetensors without a fallback count (5.1). Harmless on real models; it makes the unwidened tiny fixtures useless as WEX tests.
- **`build_nr2_full.py`:**
  - always retrains, with no option to embed an existing model;
  - GPU only: the DLL exports `wex_nr2_count_cpu`/`wex_nr2_encode_cpu`, but the script does not use them;
  - needs 4096-multiple raw records (`hidden x moe_inter` a multiple of 131072): widen tiny models with `fixtures/widen_experts.py`, or use `fixtures/make_synth_mxfp4.py`.
- **Not in this repo:**
  - the WNR1 encoder (retired);
  - the engine helpers that level C borrows from the fork (5.3);
  - DeepGEMM;
  - the qwen38 engine itself (`qwen38.c`, `qwen38_core.h` and its CUDA tier). Its container store `c/qwen38_wex.h`, the repacker and the tests are here since 2026-11-11.
- **Helper:** `cudaSetDevice` is not restored after a decode.
- **Dead code:** `coli_v4_expert_store_wex_compressed`, `coli_v4_wex_flock_retry`.
- **Doc drift:** WNR2-REFERENCE.md section 11 has line numbers from the engine of 2026-11-09. Sections 1-11 of this guide use the engine of 2026-11-07 line numbers. Sections 12-13 refer to the engine of 2026-11-09, where the GPU workflow is committed. the engine's main branch has moved on since (WNR3, WEX4, Qwen3.8 MXFP4): the box at the top of this guide says what no section here shows.

---

## 11. Sources and keeping this repo in sync

- Engine excerpts come from `src/colibri/c/<file>` of the engine of 2026-11-07. At the engine of 2026-11-09 (2026-11-09) the mode-0 store lines are unchanged (open, mirrors, `v4_wex_pick`, record read); the only change there is a new `v4_wex_pick` caller for batch reads. The GPU-direct side changed in the engine of 2026-11-09: `v4_gpu_mirror_fill_unlocked` was split into a `V4GpuMirrorHold` reserve and `v4_gpu_mirror_publish`, `V4_WEX_BATCH` adds batch reads and decode (W2), and the V4.1 VRAM reserve default is 4096 MiB; see section 12 and `research/v41-w2-20261109/laneW-w2.patch`. The direct-gpu excerpt does not cover these changes.
- `src/`, `tools/` (except `verify_wex1.py`) and `tests/` (except `test_verify_wex1.py`), now `c/` and `scripts/`, were byte copies of the maintainer's `src/colibri/c` at the engine of 2026-11-07 and unchanged at the engine of 2026-11-09. `archive/nr2/` and `archive/bench/` are copies of `lab\wex-gpu-direct-20261019\nr2`, `lab\v41-gpu-20261107\w1` and `lab\wex-prod-20261105`.
- Repo-only changes made on 2026-11-08. Keep them when you resync from the lab:
  - `sys.path` in `archive/nr2/sample_records.py`, `archive/nr2/precheck_records.py`, `archive/bench/rechunk.py`, `archive/bench/reencode_bin.py`;
  - the include path in `archive/nr2/build_tools.cmd`.
- The `.cmd` scripts of `archive/nr2/`, `archive/bench/` and `integration/backend/` are owned by this repo, not copies to resync. The nvcc builds (`archive/bench/build.cmd NAME`, which replaces the six per-benchmark scripts, `archive/nr2/build_tools.cmd`, `archive/nr2/build_test_nr2.cmd`) take `VCVARS`, `NVCC` and `ARCH` from the environment like `integration/build/build_wex_windows.cmd`; `archive/nr2/run_test_nr2.cmd` needs `WEX_PILOT_DIR` (and takes `WEX_HELPER`); the backend script needs `COLIBRI_C`, `DEEPGEMM_ROOT` and `CUDA_PATH`. The two lab scripts that built the helper DLL and `test_decode_batch.exe` are gone: `integration/build/build_wex_windows.cmd` builds both.
- New files: `scripts/verify_wex1.py`, `scripts/test_verify_wex1.py`, `integration/` (including `upstream-561966ec-levelA.patch` and `build/build_test_wex_planned.sh`), `fixtures/make_synth_mxfp4.py`, `fixtures/widen_experts.py`, this guide.
- Also edited on 2026-11-08: the docstring of `archive/nr2/build_nr2_full.py` (it described lab paths and a hard-link mirror the code no longer makes) and a usage line in `archive/nr2/check_all.py`.
- `fixtures/v41-tiny/` is a copy of `<TREE>\lab\v41-port-20261106\fixtures\v41-tiny` (`config.json`, `dsv41_engram.json`, `model.safetensors` sha256 `b58c56b8...8062`, `tokenizer.json`). The maintainer builds it with `src/colibri/c/tools/make_dsv41_tiny.py` (torch).
- To refresh the upstream patch on a newer upstream commit: apply it there, fix the rejects by hand from `integration/engine/deepseek_v4.mode0-store.wex-excerpt.c`, rebuild and rerun 7.5 before committing.
- Copied in from the fork: `c/tests/test_v4_gpu_priority.c`, `c/tests/test_v4_mirror_guard_source.py`, `docs/reports/wex-prod-20261105.it.md`, `research/q38-wex-q8-20261105/`.
- Upstream baseline used for the "upstream has / lacks" statements: `<TREE>/src/colibri/modern` = JustVugg/colibri 561966ec.

## 12. Update 2026-11-09: V4.1 on the GPU tiers, batched WNR2, dual-drive mirror

the engine of 2026-11-09 (local) moves DeepSeek V4.1 REAP onto the CUDA tiers of the V4 engine. Three parts concern the WEX/WNR2 integration:

- **V4.1 experts through `moe_token_pipeline` with VRAM mirrors (G2).** WEX GPU-direct works for V4.1 too: `COLI_WEX_GPU_DIRECT=2` decodes WNR2 records straight into a VRAM mirror; on WEX1 it is a no-op. This is the path measured in W3. Evidence: `research/v41-gpu-20261109/REPORT.md` (snapshot of 05:20, before the commit of 2026-11-09, the install and the mirror check; its commit/install/mirror status is superseded by section 13).
- **Per-layer batched WNR2 decode (W2), opt-in with `V4_WEX_BATCH=1`.**
  - Patch: `research/v41-w2-20261109/laneW-w2.patch`.
  - `v4_gpu_mirror_fill_unlocked` is split into reserve and publish. Each layer issues one set of overlapped reads into a shared page-locked ring and makes one `wex_cuda_decode_batch_to_device` call.
  - Result: +0.5% against the per-record decode, with bit-identical logits.
- **W3 result** (`research/v41-w3-20261109/results.md`): integrated build, 16 GB, both containers on U:. WNR2 is -1.80% +/- 0.65% against WEX1, with bit-identical logits.
  - Cause: GPU-direct records bypass the host RAM tier, so WNR2 makes about 11% more reads for the same bytes (`docs/reports/wnr2-reads-vs-ram-cache-20261109.md`).
  - Its rANS decode also shares the GPU and the copy engine with attention.
  - Verdict: on U: alone WEX1 stays the container and WNR2 a capacity format. Section 13 nonetheless installs WNR2 with the U:+C: mirror, which W3 did not test; the ABBA against WEX1 is still to do. *(Done 2026-11-10: +49%, one pair; then WNR3 replaced WNR2, section 14.)*
- **Dual-drive mirror for a container.** `COLI_WEX_EXTRA_MIRRORS` lists folders that hold an identical copy of the container; the engine checks file size, header and entry table byte for byte.
  - Use `COLI_WEX_WEIGHTS=<primary>,<mirror>` for the split (`v4_wex_pick`, deepseek_v4.c). It is a deterministic hash of (layer, expert), so a given expert is always read from the same drive.
  - Without it, one mirror uses the shard dual-SSD route (`coli_st_expert_route`). That route is active only when `COLI_MODEL_MIRROR` registers shard copies (`coli_st_mirror_setup` returns early otherwise). With a container-only mirror, which is what `COLI_WEX_EXTRA_MIRRORS` gives you, every read then goes to the primary, silently. Always set `COLI_WEX_WEIGHTS`, and check that the engine log shows `[v4_wex] read split over N drives`.
  - (the application host also removes `COLI_DISK_WEIGHTS` and `COLI_MODEL_MIRROR` when `mirror_dir` is null, Engine.cs:652-657 at the engine of 2026-11-09 (643-648 in the uncommitted working tree), so `COLI_DISK_WEIGHTS` is not an option there either.)
  - Partial or sparse copies are not possible.
  - Example used on this PC (WNR2 on U: plus C:, Gen4 + Gen5):
  ```
  COLI_EXPERT_STORE=wex
  COLI_WEX_PATH=U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex
  COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap
  COLI_WEX_WEIGHTS=38,62
  ```
  The setup script is in `research/ops-20261108/`. Verified results are in section 13.
- **Fast testing** (`research/wnr2-fastbench-20261109/`, work in progress):
  - a trace-replay benchmark of disk reads plus decode only, a few minutes per arm;
  - a short ABBA protocol (A1 B1 B2 A2) of 3 prompts x 128 tokens, about 20 min;
  - `archive/bench/nvme_temp.py` reads SSD temperatures without admin rights.

## 13. Application deployment, 2026-11-09: profile `v41gpu` with WNR2 and the U:+C: mirror

State at 2026-11-09 17:08: `v41gpu` is installed and selectable, but the active model in the application config is `qwen38`.

**Installed** into `app\colibri-optimized`, built from the engine of 2026-11-09. Hashes and the rollback script are in `research/v41gpu-deploy-20261109/`.
- `deepseek_v4.exe` 7a6f3e3f;
- `coli_cuda_dsv4_dg.dll` b1b7cdb5;
- `coli_cuda_dsv4.dll` 5483b6a7;
- `mcp_server.py`;
- `kaira_wex_cuda.dll` sha256 cceb4fb573b3b3e7b71843bf85a2b87cf9efdd94155bfc5deec717cce949e782. This is the W1 lane build (lab\v41-gpu-20261107\w1\build), the one W3 measured with WNR2, and not a build of 2026-11-09. It was swapped in with the mirror switch, so `install-hashes.txt` still lists the previous helper a7977200 as not replaced.
`coli_cuda.dll` is kept, because the installed one exports symbols the src/colibri/c build lacks.

**Host switch.** No host code change is needed. With `COLI_FAMILY_ENGINE=deepseek_v4` in the profile env, a DeepSeek V4.1 model runs on `deepseek_v4.exe`: `family_registry.engine_family()` maps deepseek_v41 to deepseek_v4, and the profile env is passed through. The host still applies its own rules afterwards: it removes `COLI_DISK_WEIGHTS` and `COLI_MODEL_MIRROR` when `mirror_dir` is null, forces `V4_MTP=0` and `V4_SKIP_COLD=0`, and can replace `COLI_MODEL_DIRS` from a `kaira-shards.json`. Without the switch, the host filters the env through `IsV41Env`, which drops `COLI_EXPERT_STORE`, `COLI_WEX_*`, `DSV4_*` and `V4_*`.

**The profile** (`research/v41gpu-deploy-20261109/v41gpu-profile-mirror.json`), main points:
- write the profile out in full, with `"mirror_dir": null` and no `extends`. A missing field falls back to V4 values (Config.cs), which would point `COLI_MODEL_MIRROR` at the V4 checkpoint;
- `path_prepend` must contain `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64`, because the dsv4 DLLs import `cublasLt64_13.dll`;
- `ram_gib: 0` means the host's automatic budget, sent to the engine as `--ram N` (RAM_GB);
- WEX env: `COLI_WEX_PATH` (WNR2 on U:), `COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap`, `COLI_WEX_WEIGHTS=38,62`, `COLI_WEX_GPU_DIRECT=2`, `V4_WEX_BATCH=1`.

**Verification** (`research/v41gpu-mirror-verify-20261109/`). The launch was made the way the host does it: `coli_proc.py`, then `coli serve --ram 16 --ctx 32768`.
- Engine log lines:
  - `Container v3 active: U:\...nr2.wex`;
  - `Matching mirror container active: C:\...nr2.wex`;
  - `read split over 2 drives: 38% 62%`.
- OS disk counters during the prompts: C: 61.1%, U: 38.9%.
- WNR2 GPU-direct: 41,271 device decodes, 0 failures.
- Token ids are identical to the first 32 (reason1) or 128 (code1, chat_it1) ids of all 8 W3 runs, WEX1 and WNR2.
- Sanity tok/s, which is not an A/B (128 tokens, ctx 32768, 16 OMP threads against W3's 13):
  - code1 3.04 tok/s, against 2.27 for W3 WEX1 on U: only;
  - chat_it1 2.66 tok/s, against 1.67.
  - A short ABBA of WNR2 + mirror against WEX1 on U: only is still to do. *(Done 2026-11-10: +49%, one pair; then WNR3 replaced WNR2, section 14.)*

**Rollback** takes two steps, with the application closed:
1. copy back only `kaira_wex_cuda.dll` from `lab\app-before-mirror-20261109-061724`;
2. run `lab\app-before-v41-gpu-20261109-060501\rollback.ps1`, which restores the engine files. Because the mirror switch edited the v41gpu block, rollback.ps1 will not find it verbatim and will warn. Then delete the `v41gpu` entry from `models` in the current the application config by hand.

Do not copy the 06:05 application config, and do not use `-RestoreKairaJson` (that restores the 11-08 01:17 file): both would undo later changes, including the `qwen38` active model.

The copy of `rollback.ps1` in this repo is for reference only: it expects the backup files next to it.

**Known issues.**
- The host preflight warns on every `v41gpu` start that COLI_WEX_PATH should be removed (DeepSeekV41Preflight.cs:243-246 at the engine of 2026-11-09, 241-244 in the working tree). The text is outdated; ignore it.
- With the automatic 16 GB budget, free system RAM fell to about 1 GB after the prompts.
- U: warms quickly under load: +21-22 C in about 2.6 min (36->57 and 38->60 C). `archive/bench/nvme_temp.py` reads it.
- The `v4` profile shares the new exe and DLLs and has not had a real V4 smoke yet.

## 14. Update 2026-11-10: WNR3 in production

Section 13's profile `v41gpu` now runs a **WNR3-CZ** container instead of WNR2; everything else in sections 12-13 (mirror variables, split, GPU-direct, batch) is unchanged. Full record: [WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md).

- Container `U:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex` (123,306,192,896 B, 0.949 of WNR2, model 0x27e8a1e0), full mirror on `C:\kaira_wex\v41-reap\` (hash-verified), `COLI_WEX_WEIGHTS=38,62`, `COLI_WEX_GPU_DIRECT=2`, `V4_WEX_BATCH=1`.
- Engine `deepseek_v4.exe` from the engine of 2026-11-10 (a55a4ce0) at the switch, replaced the same evening by the 2026-11-10 engine build (2ab584d9; same WEX code, adds `DSV4_CUDA_REQUIRE=1`: GPU mandatory, not WEX code) and later that night by the 2026-11-10 engine build (b7c7b6a2; same WEX code) with refit V4.1 predictor gates (production doc, *Later the same night*); helper `kaira_wex_cuda.dll` 602e3f4d (feature 16 = WNR3). Deploy rule: a WNR3 container needs a feature-16 helper (feature table in section 4).
- Log lines to check: `Container v3 WNR3 value-exact active`, `Matching mirror container active`, decode on the gpu-pipeline.
- Measured: 1,886-token turn 2.92-2.96 tok/s, TTFT 339-352 s (WNR2 the same morning: 2.86 tok/s, 326 s); 7 prompts x 64 tokens 2.7-3.3 tok/s, logits bit-identical to the WNR2 reference.
- **Deleted after the switch:** WEX1 (U:) and every WNR2 copy (U:, C:, E:). The tables of sections 6, 8.3 and 13 that list those copies are history. `c/tools/build_nr3.py` reads WEX1, so a WNR3 rebuild starts with `repack_v4_wex.py` (6.2).
- Rollback (`lab\app-before-wnr3-20261110-165802\ROLLBACK.txt`) restores the engine, helper and the application config, but the WNR2 container it points to must be rebuilt first.
