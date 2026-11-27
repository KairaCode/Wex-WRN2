# `ggml-wex`: Dynamic MoE Streaming Plugin for `llama.cpp` & `ik_llama.cpp`

`ggml-wex` (`libggml-wex.{dll,so,dylib}`) is a lightweight C99 dynamic plugin that brings `libwex` single-NVMe MoE expert streaming, L2 compressed-RAM LRU caching, async next-layer prefetch, and SHA-256 integrity verification to `llama.cpp`, `ik_llama.cpp`, and LM Studio custom runtimes.

## Architecture Highlights

- **Zero link-time dependencies:** `ggml_wex.c` dynamically loads `libwex` at runtime via [`wex_loader.h`](../../include/wex_loader.h) (ABI v1 in [`wex_store.h`](../../include/wex_store.h)). No CUDA toolkit, `libwex` import library, or `llama.cpp` source tree is required to build the standalone plugin.
- **Single-NVMe-first defaults:** `ggml_wex_opts_init()` enables `l2_cache_bytes = UINT64_MAX` (automatic L2 compressed-RAM cache up to 25% of free RAM, capped at 16 GiB) and `prefetch_next_layer = 1` so hot routed experts stay in compressed RAM instead of re-reading NVMe.
- **Deduplicated `GGML_OP_MUL_MAT_ID` streaming:** `ggml_wex_mul_mat_id_q2_0_f32()` deduplicates repeated expert IDs across a multi-token batch, fetches only the unique active experts for the layer in a single `libwex` batch call (hitting L2 first, streaming misses via async NVMe + optional `libwex_cuda` GPU decode), computes exact `Q2_0 x F32` dot products (`Q2_0` ggml type 42: 18-byte / 64-weight blocks matching `wnr4/wnr4_gg.h`), and issues a non-blocking prefetch for `layer + 1`.
- **`GGML_BACKEND_DL` probing & registration:** Exports `ggml_wex_backend_score()` unconditionally (returns `1` when `libwex` ABI v1 is loadable, `0` otherwise). When compiled with `-DGGML_WEX_WITH_GGML` and `-DGGML_INCLUDE_DIR=<path>`, also exports `ggml_backend_score()`, `ggml_backend_init()`, and `ggml_backend_wex_reg()`.

## Preparing a WEX4 `WNR4-GG` Container from GGUF

Pack routed MoE experts from a GGUF checkpoint into a byte-exact WEX4 container using [`gguf2wex4.py`](../../../wex4/gguf2wex4.py) and [`wnr4gg.py`](../../../wex4/wnr4gg.py) (specifications: [WEX4-SPEC.md](../../../docs/WEX4-SPEC.md), [WNR4-SPEC.md](../../../docs/WNR4-SPEC.md)):

```sh
python -I wex4/gguf2wex4.py pack experts_raw.wex4 model-00001-of-00003.gguf --experts-only
python -I wex4/wnr4gg.py train experts_raw.wex4 model.wnm4 --dll wnr4_gg_tools.so --every 8
python -I wex4/wnr4gg.py pack experts_raw.wex4 experts_gg.wex4 --model model.wnm4 --dll wnr4_gg_tools.so
```

## Building

### Standalone (zero `llama.cpp` dependency)

See [`CMakeLists.txt`](CMakeLists.txt):

```sh
cmake -S libwex/adapters/ggml-wex -B build-ggml-wex -DCMAKE_BUILD_TYPE=Release
cmake --build build-ggml-wex
```

Or directly with GCC/Clang:

```sh
gcc -std=c99 -O2 -Wall -Wextra -Werror -shared -fPIC \
    -DGGML_WEX_BUILDING_DLL -Ilibwex/include \
    libwex/adapters/ggml-wex/ggml_wex.c \
    -o libggml-wex.so -ldl
```

### Inside `llama.cpp` / `ik_llama.cpp` (`GGML_BACKEND_DL`)

Pass the path to `ggml/include` so `ggml_backend_score` / `ggml_backend_init` are compiled in:

```sh
cmake -S libwex/adapters/ggml-wex -B build-ggml-wex \
    -DGGML_INCLUDE_DIR=/path/to/llama.cpp/ggml/include
cmake --build build-ggml-wex
```

Place `libggml-wex.{dll,so,dylib}`, `libwex.{dll,so,dylib}`, and optionally `libwex_cuda.{dll,so,dylib}` in the runtime's backend plugin directory (or set `WEX_LIB_PATH`).

## API Overview

See [`ggml_wex.h`](ggml_wex.h) and [`ggml_wex.c`](ggml_wex.c):

- `ggml_wex_backend_score()` — runtime probe (`1` if `libwex` loads, `0` otherwise)
- `ggml_wex_open()` / `ggml_wex_info()` / `ggml_wex_stats()` / `ggml_wex_close()` — lifecycle & telemetry
- `ggml_wex_prefetch_layer()` — non-blocking L2 warm for upcoming layer experts
- `ggml_wex_fetch_experts()` — deduplicated batch fetch into caller slabs
- `ggml_wex_dequant_row_q2_0()` — exact `Q2_0` (18 B / 64 weights) row dequantizer
- `ggml_wex_mul_mat_id_q2_0_f32()` — deduplicated streaming `GGML_OP_MUL_MAT_ID`

## Running the Live Verification Test

See [`test_ggml_wex.c`](test_ggml_wex.c):

```sh
./test_ggml_wex --plugin ./libggml-wex.so --lib /path/to/libwex.so --cuda 0
./test_ggml_wex --plugin ./libggml-wex.so --lib /path/to/libwex.so --cuda /path/to/libwex_cuda.so
```
