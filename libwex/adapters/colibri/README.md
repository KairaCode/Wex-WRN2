# colibri Thin Drop-In Adapter for `libwex`

`colibri_wex_shim.h` is a single-header C99/C++11 adapter (~250 lines) that replaces thousands of lines of engine-side WEX4 container parsing, codec dispatch, first-read SHA-256 verification, async I/O batching, mirror routing, and L2 compressed-RAM caching with three headers and zero link-time dependencies.

## What You Copy

Copy only these three headers into `colibri` (or add `libwex/include` and `libwex/adapters/colibri` to your include path):

1. [`wex_store.h`](../../include/wex_store.h) — pure C99 ABI v1 declarations
2. [`wex_loader.h`](../../include/wex_loader.h) — header-only `dlopen` / `LoadLibraryExA` dynamic loader
3. [`colibri_wex_shim.h`](colibri_wex_shim.h) — `COLI_WEX_*` environment mapping and thin convenience wrappers

At runtime, place `libwex.{dll,so,dylib}` (and optionally `libwex_cuda.{dll,so,dylib}`) next to the engine binary or point `COLI_WEX_LIB` / `WEX_LIB_PATH` at it. Neither the engine nor its build scripts need CUDA headers, `wex_decode.c`, `wex4.h`, or `wnr4*.h`.

For background on legacy `colibri` WEX integration and WEX4/WNR4 container formats, see [COLIBRI-INTEGRATION.md](../../../docs/COLIBRI-INTEGRATION.md), [WEX4-SPEC.md](../../../docs/WEX4-SPEC.md), and [WNR4-SPEC.md](../../../docs/WNR4-SPEC.md).

## Environment Variables

| Variable | Maps to (`wex_open_opts_t`) | Default / Behavior |
|---|---|---|
| `COLI_WEX_PATH` | `path` argument | Used when `path_override` is `NULL` or empty |
| `COLI_WEX_ROOT` | `opts.root_hex` | Optional 64-hex-digit WEX4 root SHA-256 pin; mismatches refuse open (`WEX_E_ROOT`) |
| `COLI_WEX_VERIFY` | `opts.verify` | `0`/`off`, `1`/`on`, or `auto` (ON when `COLI_WEX_ROOT` is set) |
| `COLI_MODEL_MIRROR` | `opts.mirrors` / `opts.num_mirrors` | Comma-, semicolon-, or colon-separated mirror file paths (single-NVMe works with it unset) |
| `COLI_WEX_CUDA_LIB` | `opts.gpu_helper_path` | `"0"` forces CPU-only; unset auto-discovers `libwex_cuda` next to `libwex` |
| `COLI_WEX_L2_MB` | `opts.l2_cache_bytes` | `"auto"` -> `UINT64_MAX`, `"0"` -> off, integer MiB -> bytes; defaults to `"auto"` on single-NVMe (`num_mirrors == 0`) |
| `COLI_WEX_L2_PIN` | `opts.l2_pin_gpu` | `1` page-locks the L2 compressed-RAM arena for fast GPU H2D DMA (`0` normal RAM) |
| `COLI_WEX_QD` | `opts.io_queue_depth` | `0` = auto (adaptive single-NVMe / multi-NVMe queue depth) |
| `COLI_WEX_DIRECT` | `opts.io_direct` | `-1` = auto, `0` = buffered, `1` = `O_DIRECT` / `FILE_FLAG_NO_BUFFERING` |
| `COLI_WEX_LIB` / `WEX_LIB_PATH` | dynamic `libwex` path | Checked before sibling and default library search paths |

## Minimal Engine Integration

```c
#include "colibri_wex_shim.h"
#include <stdio.h>

int load_and_stream(const char *path_or_null) {
    coli_libwex_t cw;
    char err[256], summary[256];

    if (coli_libwex_open_env(&cw, path_or_null, err, sizeof(err)) != WEX_OK) {
        fprintf(stderr, "[libwex] open failed: %s\n", err);
        return -1;
    }

    /* Warm L2 compressed-RAM cache asynchronously for upcoming layer */
    int next_eids[4] = { 0, 1, 2, 3 };
    coli_libwex_prefetch(&cw, 1, next_eids, 4);

    /* Fetch active experts on layer 0 into host scratch slabs (or use coli_libwex_fetch_device) */
    int status[4];
    /* coli_libwex_fetch_host(&cw, 0, active_eids, 4, dst_slabs, dst_caps, 1, status); */

    if (coli_libwex_format_stats(&cw, summary, sizeof(summary)) > 0) {
        fprintf(stderr, "%s\n", summary);
    }
    coli_libwex_close(&cw);
    return 0;
}
```

## Building and Running the Test

See [`test_colibri_shim.c`](test_colibri_shim.c):

```sh
gcc -std=c99 -O2 -Wall -Wextra -Werror \
    -Ilibwex/include \
    libwex/adapters/colibri/test_colibri_shim.c \
    -o test_colibri_shim

# CPU-only run
./test_colibri_shim --lib /path/to/libwex.so --cuda 0

# CUDA GPU-accelerated run
./test_colibri_shim --lib /path/to/libwex.so --cuda /path/to/libwex_cuda.so
```
