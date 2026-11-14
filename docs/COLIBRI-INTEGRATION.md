# Integration guide: WEX4 containers in a colibri engine

Older reports cite sections 1 to 14 of this guide as it stood on 2026-11-09 (WEX1, WEX2 and WNR2, engine quotes of that date, the deployment log of that week). That text is now the dated report [reports/colibri-integration-20261109.md](reports/colibri-integration-20261109.md), with the same section numbers. This page is the current guide: WEX4 containers with RAW or WNR3 records, as the maintainer's application runs them on 2026-11-13.

What each piece is and how a record travels from the drive to the GPU: [ARCHITECTURE.md](ARCHITECTURE.md). The bytes: [WEX4-SPEC.md](WEX4-SPEC.md) and [WNR3-SPEC.md](WNR3-SPEC.md).

## 1. What you need

**From `c/`** (copy the files into the colibri `c/` folder at the same relative paths, so that their `../` includes resolve):

| Role | Files |
|---|---|
| format and codecs (always) | `wex_format.h`, `wex_neural_rans.h`, `wex_palette.h` |
| WEX4 reader (always, for WEX4 files) | `wex4.h` |
| CPU decoder and helper loader | `wex_decode.c`, `wex_decode.h` |
| Qwen3.8 store | `qwen38_wex.h` (included by the qwen38 engine) |
| CUDA helper source | `wex_cuda.cuh`, `tools/wex_cuda_bridge.cu` |
| tools | `tools/repack_v4_wex.py`, `tools/repack_q38_mxfp4_wex.py`, `tools/build_nr3.py`, `tools/wex_nr3_tools.c`, `tools/wex4_verify.c` (and `tools/wex_neural_rans.py`, `tools/wex_palette.py`, which `repack_v4_wex.py` loads) |
| tests | `tests/test_wex_format.c`, `tests/test_wex4_verify.c`, `tests/wex4_fixture.h`; the engine tests `tests/test_v4_wex_nr3.c` and `tests/test_qwen38_mxfp4.c` need the engine sources ([ARCHITECTURE.md](ARCHITECTURE.md), section "The `c/` folder") |

`c/expert_store_wex.c` is colibri's standalone expert store for WEX1-WEX3; it does not read WEX4.

**The CUDA helper** `kaira_wex_cuda.dll` is optional for RAW records and required for WNR3 records on the GPU. It reports ABI 1 and its features with `wex_cuda_helper_features()`:

| Bit | What it adds | Who needs it |
|---|---|---|
| 1 | `wex_cuda_decode_to_device`: decode straight into VRAM | `COLI_WEX_GPU_DIRECT=1/2` |
| 2 | `wex_cuda_host_register` / `_unregister` | page-locked staging (optional) |
| 4 | WNR2 records (codec 4) and `wex_cuda_register_model` | any WNR2 container |
| 8 | `wex_cuda_decode_batch_to_device`: a batch of records in one call | `V4_WEX_BATCH=1`, `V4_PREWARM_BATCH=1` |
| 16 | WNR3 records (codec 5) in every decode entry point | any WNR3 container |
| 32 | `wex_cuda_decode_batch_to_host`: a batch decoded on the GPU into host buffers | the qwen38 store's load waves of WNR3 records |

The bridge in `c/tools/` reports 63 (all six). The full helper API is in [WNR2-REFERENCE.md](WNR2-REFERENCE.md), section 10.

## 2. Build

**CPU checks, from this repository.** `bash scripts/check_all.sh` builds and runs every test that needs no engine tree and no GPU, and ends with `all checks passed`. Windows: MSYS2 UCRT64 or MINGW64 `gcc` and Python 3 on `PATH`, run from Git Bash or an MSYS2 shell. Linux: `gcc`, Python 3 and `PYTHON=python3` (not yet run on Linux; the Linux CI job may fail until its first green run). The CPU is x86-64 with AVX2 (one test builds with `-march=haswell`) or, under Linux, 64-bit ARM, where that test builds the scalar kernel of the fast decoder instead (the `linux-arm` CI job).

**The helper and the GPU checkers, from this repository (Windows):**

```bat
integration\build\build_wex_windows.cmd [OUT_DIR]
```

It needs MSVC (found with `vswhere`, or set `VCVARS` to a `vcvars64.bat`), the CUDA toolkit (`NVCC`, default `nvcc` on `PATH`) and a target (`ARCH`, default `sm_120`; set `ARCH=sm_86` for an RTX 30 card, for example). Below sm_80 `wex_cuda.cuh` replaces `cp.async` with a plain load and store; that build has not run on such a GPU. Outputs: `kaira_wex_cuda.dll`, `test_decode_batch.exe`, `check_container.exe` in `OUT_DIR` (default `out\`), and the archived WNR2 encoder DLL (`archive\nr2\wex_nr2_tools.dll`).

**In the engine.** The engine compiles `wex_decode.c` into its executable and includes `wex4.h`; the qwen38 engine includes `qwen38_wex.h`, which includes `wex_decode.c` itself. How the engine's Makefiles do it: `integration/build/*.wex-excerpt.mk`, and section 4 of the [2026-11-09 guide](reports/colibri-integration-20261109.md).

## 3. Make a container

From a checkpoint to a verified WEX4 file. Every command runs from the repository root with `python -I`; the tools need only the standard library, except `repack_q38_mxfp4_wex.py` (numpy).

1. **WEX1 from safetensors.** DeepSeek V4 or V4.1 (MXFP4 routed experts): `python -I c/tools/repack_v4_wex.py --model-dir <checkpoint> --output experts.wex`. Qwen3.8 MXFP4 (AMD Quark checkpoint): `c/tools/repack_q38_mxfp4_wex.py`; its `--help` lists its options. A WEX1 file holds every expert as a RAW record (codec 0), one 4096-aligned slab each.
2. **WNR3 records (optional).** `python -I c/tools/build_nr3.py <wex_nr3_tools.dll> experts.wex container experts.nr3.wex --layers 0,1,... --train L:E,... --cz` trains a WNM2 model on the listed experts and writes every record as WNR3 into a WEX3 file, checking each one on the CPU. `--cz` (canonical zero) stores FP4 -0 as +0: smaller and value-exact, but the WEX1 bytes can then only be rebuilt from the checkpoint. Build the DLL with `gcc -O2 -shared c/tools/wex_nr3_tools.c -o wex_nr3_tools.dll`. Whether WNR3 pays depends on the machine: on the test machine it made the V4.1 container 0.854 of raw, but slowed Qwen3.8 down ([BENCHMARKS.md](BENCHMARKS.md)).
3. **WEX4, in place.** `python -I wex4/wex4_convert.py upgrade experts.wex` appends the index and the old header page and table to the file and rewrites its first 4096 bytes; the records do not move. `--names qwen38` gives Qwen3.8's tensor names. `downgrade` restores the file byte for byte. On the 123 GB V4.1 file the upgrade appended 6,717,440 B.
4. **Verify.** `python -I wex4/wex4.py verify experts.wex --dll wex4/out/wex4_tools.dll` checks the header, the index and every hash, and decodes every record. `c/tools/wex4_verify.c` only hashes, on 8 threads (64 GB in 7-11 s on the test machine), and prints the line to pin:

   ```sh
   gcc -O2 -fopenmp c/tools/wex4_verify.c -o wex4_verify.exe
   ./wex4_verify.exe experts.wex       # exit 0 pass, 1 mismatch, 2 not WEX4; prints COLI_WEX_ROOT=<64 hex>
   ```

5. **Mirror (optional).** Copy the file byte for byte to another drive, under the same file name. Two copies are the same container when their header pages are equal; `wex4_verify` prints the same root for both.

## 4. Wire an engine

- **Qwen3.8.** `c/qwen38_wex.h` is the complete store: WEX1, WNR3 and WEX4, mirrors, split reads, the root pin and the first-read SHA-256. Its header comment documents every variable. The rest of the qwen38 engine (`qwen38.c`, `qwen38_core.h`) is not in this repository.
- **DeepSeek V4 / V4.1.** The store is in the engine's `deepseek_v4.c`, which is not in this repository. What it takes to add WEX4 to it:
  - the WEX1-WEX3 store first: [integration/](../integration/README.md) has the engine excerpts of 2026-11-09 (levels A to C of the [2026-11-09 guide](reports/colibri-integration-20261109.md), section 5) and, for stock colibri 561966ec, `upstream-561966ec-levelA.patch`;
  - then the three WEX4 patches, in order: `engine-wex4-loader-75932c1.patch` (open a WEX4 file, pin the root), `engine-wex4-first-read-4ba5568.patch` (SHA-256 at the first read) and, for qwen38 only, `engine-q38-wex4-loader-de17198.patch`;
  - then what came after them, shown in `integration/engine/deepseek_v4.wex4.wex-excerpt.c`: the refusal of a legacy file under a pinned root, the hash pool, sub-reads and the batched tier prewarm.

  The patches are engine commits with the paths of the maintainer's tree (`src/colibri/c/...`); they apply to a tree with that layout, and are the reference for a port to another one.
- **A new engine.** Open the file with `wex4_parse_header` and `wex4_open` (`c/wex4.h`), read a record's extent from the record table, read it with one aligned read, check its SHA-256 the first time (`wex4_sha256` against the index), and decode it with `wex4_decode_record` or, for WNR3 on the GPU, the helper. `wex4/wex4.py` shows the same steps in Python.

## 5. Environment

Every variable that changes how a WEX file is opened or read, with its default in the engine code of 2026-11-13. "deepseek_v4.c" means the DeepSeek engine, not in this repository.

| Variable | Default | Effect | Read in |
|---|---|---|---|
| `COLI_WEX_PATH` | DeepSeek: `<model_dir>\v4_flash_experts.wex`, opened if it exists; qwen38: unset (no container) | the container | `deepseek_v4.c`; `c/qwen38_wex.h`; `c/expert_store_wex.c` |
| `COLI_WEX_EXTRA_MIRRORS` | unset | `;`-separated folders that hold a byte copy under the same file name; up to three | `deepseek_v4.c`; `c/qwen38_wex.h` |
| `COLI_WEX_WEIGHTS` | equal shares | read split over primary and mirrors, one positive weight per copy (`38,62` on the test machine); an expert always goes to the same drive. A wrong count is ignored with a log line | `deepseek_v4.c`; `c/qwen38_wex.h` |
| `COLI_WEX_ROOT` | unset | 64 hex digits: a WEX4 file with another root is refused; under a pin, a file that is not WEX4 is refused too (DeepSeek falls back to the safetensors, qwen38 stops) | `deepseek_v4.c`; `c/qwen38_wex.h` |
| `COLI_WEX_VERIFY` | DeepSeek: on when `COLI_WEX_ROOT` is set, else off; qwen38: 0, whatever the root | `1` = SHA-256 of each record's stored bytes at its first read; a record that fails stays refused (DeepSeek serves it from the safetensors; qwen38 stops) | `deepseek_v4.c`; `c/qwen38_wex.h` |
| `COLI_WEX_GPU_DIRECT` | 0 | DeepSeek: 1 = misses outside RAM are decoded straight into VRAM; 2 = the next-layer prefetcher too. Needs helper bit 1 and a backend with the fp4 mirror exports, else it logs and stays on the host slab | `deepseek_v4.c` |
| `COLI_WEX_GPU_DIRECT_PREFETCH_RANK` | 6 | mode 2: prefetch guesses below this rank go to VRAM | `deepseek_v4.c` |
| `COLI_WEX_GPU_DIRECT_POOL` | 32 (0 to 256) | mode 2: resident VRAM mirrors the prefetcher may take over | `deepseek_v4.c` |
| `V4_WEX_BATCH` | 0 | 1 = a layer's misses decoded in one helper call (bit 8); needs `COLI_WEX_GPU_DIRECT>=1` | `deepseek_v4.c` |
| `V4_WEX_SUBREADS` | 1 (off), clamped to 1-16 | N = each drive's part of a record read as N overlapped sub-reads; with a mirror, host reads and batches are also cut across both drives | `deepseek_v4.c` |
| `V4_WEX_STRIPE` | 0 | 1 = a demand batch reads each record from both drives at once, cut on a 4096-byte page by `COLI_WEX_WEIGHTS` | `deepseek_v4.c` |
| `COLI_WEX_STRIPE` | 0 | qwen38: the same for a decode step's lone miss | `c/qwen38_wex.h` |
| `V4_WEX_SHA_ASYNC` | 0 | 1 = the first-read SHA-256 runs on 4 hash threads while records decode; a record is published only after its hash passes. Reserves 4 record copies of RAM | `deepseek_v4.c` |
| `COLI_V4_TIER_PREWARM_AHEAD` | 1 (off), at most 8 | the VRAM tier prewarm reads N records ahead on parallel threads. The prewarm itself runs only with `COLI_V4_TIER_PREWARM=1` (default off), and so does `V4_PREWARM_BATCH` | `deepseek_v4.c` |
| `V4_PREWARM_BATCH` | 0 | 1 = the tier prewarm reads batches of 8 records, two batches deep, through the batch decode; needs `V4_WEX_BATCH=1` and `COLI_WEX_GPU_DIRECT>=1`, else it logs and goes one record at a time | `deepseek_v4.c` |
| `COLI_WEX_RAM_CACHE_GB` | 0 | RAM cache of compressed records (needs `COLI_WEX_GPU_DIRECT>=1`); `COLI_WEX_RAM_CACHE_PIN` (default on) and `COLI_WEX_RAM_CACHE_VERIFY` (default 0) tune it | `deepseek_v4.c` |
| `WEX_READ_RETRY` | off | `1` = one retry after 16 ms when a direct read fails | `deepseek_v4.c`; `c/expert_store_wex.c` |
| `COLI_WEX_CUDA_LIB` | `kaira_wex_cuda.dll` next to the executable | another helper; `0` = CPU decoder only | `c/wex_decode.c` |
| `Q38_MXFP4_DIR` | unset | qwen38: read the Quark shards directly, no container (a check, not the fast path) | `c/qwen38_wex.h` |
| `Q38_ASYNC_MISS` | 0 | qwen38 with `QT_GPU_MISS=2`: a decode step's misses are read on the loader thread, and each goes to the GPU as soon as its read lands | engine `qwen38_core.h` |

What the two engine profiles set on 2026-11-13: [STATUS.md](../STATUS.md).

## 6. Mirrors and split reads

- A mirror is a byte copy of the container on another drive, under the same file name, in a folder listed in `COLI_WEX_EXTRA_MIRRORS`. A copy whose header page differs is ignored with a log line.
- Each (layer, expert) goes to one drive, picked by a hash weighted with `COLI_WEX_WEIGHTS`. Weights proportional to the drives' measured speed are a starting point: on the test machine `38,62` matches U: (PCIe 4) against C: (PCIe 5), and the second drive gave V4.1 +49% decode ([BENCHMARKS.md](BENCHMARKS.md)).
- Splitting one record across both drives (`V4_WEX_STRIPE=1`) measured -8.7% on V4.1, so it stays off; sub-reads (`V4_WEX_SUBREADS=4`) measured +7.7% and are on in the `v41gpu` profile.
- Pick the read size and queue depth with `c/tools/wexio_probe.py` (Windows, read-only; `--self-test` first). It refuses to run while the application or an engine is running, and its default inputs are the test machine's container paths: pass `--file` for yours.

## 7. Integrity

- **At open** the header page and the index are hashed and validated; with `COLI_WEX_ROOT` set, the root must match.
- **At the first read of a record** (DeepSeek with a pinned root or `COLI_WEX_VERIFY=1`; qwen38 with `COLI_WEX_VERIFY=1`), its stored bytes are hashed against the index. Whole and split reads are tracked apart, so a record read once whole and once striped is hashed in both layouts.
- **A record that fails** is refused for the rest of the process: DeepSeek serves that expert from the safetensors, qwen38 stops the engine.
- **A legacy file under a pin** (WEX1-WEX3: no root, no record hashes) is refused: DeepSeek falls back to the safetensors, qwen38 stops.
- **Inside every WNR3 record** each rANS stream checks a 23-bit hash as it decodes, on the CPU and on the GPU, whether or not the SHA-256 is on.
- Pin the root from a source the file's writer cannot change. What the hashes do not protect against: [SECURITY.md](../SECURITY.md) and [WEX4-SPEC.md](WEX4-SPEC.md), sections 8.4 and 9.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| no `[v4_wex]` line at all | no file at `COLI_WEX_PATH` (or at the default path); this is silent by design | check the path |
| `Invalid or incompatible container ...; using safetensors` | header or table invalid, layer or expert count different from the model, truncated copy | `wex4.py verify` or `wex4_verify`; compare roots |
| `WEX4 root ... is not the pinned COLI_WEX_ROOT` | another file, or a changed one | run `wex4_verify` on the file and compare the printed root with the pin |
| `a container that is not WEX4 is not the pinned COLI_WEX_ROOT` | a WEX1-WEX3 file under a pin | upgrade it to WEX4, or unset the pin |
| `Ignoring mismatched mirror` | the copy differs from the primary | recopy and verify it |
| `COLI_WEX_WEIGHTS '...' ignored` | the count is not 1 + mirrors, or a weight is not positive | fix the list |
| `COLI_WEX_GPU_DIRECT needs ... staying on the host slab path` | the helper lacks bit 1, or the backend lacks the fp4 mirror exports | rebuild the helper or the backend |
| `V4_PREWARM_BATCH needs V4_WEX_BATCH=1 and COLI_WEX_GPU_DIRECT>=1` | the batch path is off | set both, or drop the switch |
| a WNR3 container is refused, or decodes on the CPU at about 33 MB/s | no helper, or one without bit 16 | put a current `kaira_wex_cuda.dll` next to the engine |

The longer table of 2026-11-09 (WNR2, tiny fixtures, MSYS2 builds) is in section 9 of the [old guide](reports/colibri-integration-20261109.md).

## 9. Where the older material is

- WEX1, WEX2 and WNR2 integration, level by level, with the engine excerpts of that date, the generation and verification commands of that time and the deployment log of 2026-11-09/10: [reports/colibri-integration-20261109.md](reports/colibri-integration-20261109.md).
- The WNR2 encoder and its GPU checks: [archive/](../archive/README.md).
- The record of the WNR2 period: [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md).
