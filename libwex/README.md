# libwex

`libwex` is a zero-dependency C99 library (ABI v1) for streaming, verifying, caching, and decoding WEX4 neural-weight containers across CPU (AVX2, NEON, scalar) and optional dynamically loaded GPU backends (`libwex_cuda`).

## Single-NVMe-First Architecture

Most workstations and servers hold model weights on a single NVMe drive. `libwex` is engineered to maximize effective decode throughput on one NVMe while scaling seamlessly if mirrors are present:

- **L2 Compressed-RAM Sharded LRU Cache**: Stores verified compressed records in RAM (up to 25% of available memory or an explicit byte budget) with optional page-locking (`l2_pin_gpu = 1`) for fast GPU H2D transfers. Under Zipfian MoE expert routing, hot experts serve straight from compressed RAM, turning single-NVMe streaming into multi-tens-of-GB/s decoded throughput.
- **Adaptive Async I/O**: Uses Windows `OVERLAPPED` batch reads (with a 16 MiB per-transfer cap), Linux raw `io_uring` + `O_DIRECT`, and macOS/POSIX `F_NOCACHE` + `pread`. Optional N-mirror weighted routing activates automatically when replica paths are provided.
- **Unified Lossless Decode**: Supports `RAW`, `WBP1`, `WNR3`, and `WNR4` (`MX`, `NV`, `GG`, `I4`) behind one opaque `wex_store_t` handle.
- **Streaming SHA-256 Verification**: Overlaps chunked 2 MiB SHA-256 hashing (SHA-NI / ARMv8 Crypto) with async disk completions and enforces fail-closed refusal on root or record mismatches.

## Quick Build

### GNU Make (Linux, macOS, Windows MSYS2 UCRT64/MinGW64)

```sh
cd libwex
make                  # builds build/libwex.{so,dylib,dll}
make test             # builds & runs test_libwex_io and test_libwex_c
make OPENMP=1         # optional OpenMP CPU parallelism
make cuda             # optional build/libwex_cuda.{so,dll} via nvcc
make clean
```

Run the full C + I/O + Python CLI/server/MCP verification script:

```sh
bash libwex/scripts/test_libwex.sh
```

### CMake (>= 3.16)

```sh
cmake -S libwex -B build-libwex -DWEX_BUILD_SHARED=ON -DWEX_BUILD_TESTS=ON
cmake --build build-libwex --parallel
ctest --test-dir build-libwex --output-on-failure
```

Pass `-DWEX_BUILD_CUDA=ON` to build the optional `wex_cuda` shared library alongside `wex`.

See [Makefile](Makefile) and [CMakeLists.txt](CMakeLists.txt) for all build variables and install rules.

## C99 ABI v1 Integration (10 Lines)

Inference engines need only two headers — [include/wex_store.h](include/wex_store.h) and [include/wex_loader.h](include/wex_loader.h) — with zero link-time or GPU-toolkit dependencies:

```c
#include "wex_loader.h"

wex_lib_t lib;
wex_open_opts_t opts;
char err[256];
wex_lib_open(&lib, NULL);                 /* discovers WEX_LIB_PATH / sibling / PATH */
wex_open_opts_init(&opts);
opts.l2_cache_bytes = UINT64_MAX;         /* auto L2 compressed-RAM LRU cache */
wex_store_t *s = lib.open("model.wex", &opts, err, sizeof(err));
lib.fetch_host(s, layer, eids, n, dst_slabs, dst_caps, WEX_FETCH_SCRATCH, status);
lib.close(s);
wex_lib_close(&lib);
```

## CLI, Telemetry Server, IPC & MCP

[cli/wex_cli.py](cli/wex_cli.py) uses Python 3 standard library only (zero `pip` dependencies):

```sh
# Inspect container metadata (text or JSON)
python libwex/cli/wex_cli.py inspect model.wex --output json

# Stream and SHA-256 verify every record (optionally pinning root SHA-256)
python libwex/cli/wex_cli.py verify model.wex --root <64-hex-sha256>

# Sweep L2 compressed-RAM cache sizes on single NVMe
python libwex/cli/wex_cli.py bench model.wex --l2-mb 0,256,1024 --waves 64 --batch 4 --pattern zipf

# Resumable HuggingFace download + immediate SHA-256 verification
# (HF_TOKEN, when set, goes to huggingface.co only, not to the host a download is redirected to)
python libwex/cli/wex_cli.py pull <org>/<repo> model.wex --out ./weights --verify

# Threaded HTTP telemetry, prefetch & optional upstream inference proxy server
python libwex/cli/wex_cli.py serve model.wex --host 127.0.0.1 --port 8080 --l2-mb auto
# Endpoints: GET /health, /status, /v1/models, /metrics, /v1/telemetry
#            POST /v1/prefetch, /v1/chat/completions, /v1/responses, /v1/messages

# Newline-delimited JSON stdin/stdout control protocol
python libwex/cli/wex_cli.py ipc model.wex --l2-mb auto
```

### `serve` Access Control

- `serve` binds to `127.0.0.1` by default and **refuses to start on a non-loopback `--host` without a token**.
- Set the token with the `WEX_SERVE_TOKEN` environment variable (preferred, it stays out of the process list) or `--token <secret>`. Every route then requires `Authorization: Bearer <secret>` and answers `401` otherwise. OpenAI-compatible clients send their API key this way, so configure the token as the client's API key. On loopback the token is optional.
- Without a token, a request whose `Origin` header names a non-loopback site is answered with `403`, so a web page of another site open in a browser cannot drive the local server. This does not stop DNS rebinding, which lets such a page read the `GET` routes; a token does.
- The proxy routes never forward the client's `Authorization`, `X-Api-Key`, or `Cookie` headers to `--upstream`. An upstream that needs its own credential is not supported through the proxy.
- The server speaks plain HTTP: on a non-loopback bind the token travels unencrypted, so use a trusted network or a TLS-terminating reverse proxy. Without a token, every local process can reach the server.

```sh
WEX_SERVE_TOKEN=<secret> python libwex/cli/wex_cli.py serve model.wex --host 0.0.0.0 --port 8080
curl -H "Authorization: Bearer <secret>" http://<host>:8080/v1/telemetry
```

### MCP Server

[cli/wex_mcp.py](cli/wex_mcp.py) exposes a spec-compliant MCP JSON-RPC 2.0 stdio server (stdlib only) providing four tools: `wex_inspect`, `wex_verify`, `wex_bench`, and `wex_server_telemetry`.

```sh
python libwex/cli/wex_mcp.py --root-dir /path/to/models
```

What a tool call can reach is fixed when the server starts, not by the client:

- The `libwex` library is chosen by `--lib` / `--cuda` or `WEX_LIB_PATH`. No tool takes a library path, and a call with an undeclared argument is rejected.
- Tools open containers only under the `--root-dir` directories (repeatable; default: the current working directory). Paths are resolved (symlinks, `..`) before the check. On Windows a path on a drive or network share that holds no root is refused without being opened.
- `wex_server_telemetry` queries loopback only. For a token-protected `wex serve` it sends `WEX_SERVE_TOKEN` from the MCP server's own environment.

## Measured Performance (Single NVMe + L2 Compressed RAM)

Measured on Windows (NVIDIA GeForce RTX 5090, single NVMe, `win-overlapped` direct I/O) streaming a real 40-layer x 192-expert WEX4 `WNR4-MX` container (114.43 GiB stored / 134.47 GiB decoded, 17.93 MiB decoded per expert, 4 experts per wave) directly into PyTorch CUDA VRAM (`wex_fetch_device` + in-VRAM E8M0-to-FP32 expansion):

| Workload (1 NVMe) | L2 Cache | Verify (SHA-256 + Root) | L2 Hit Rate | NVMe Read | Wave Latency | Effective Decoded BW |
|---|---:|---:|---:|---:|---:|---:|
| 40-layer Zipfian MoE (steady-state) | 0 (off) | OFF | 0.0% | 24.41 GiB | 11.75 ms | 6.40 GB/s |
| 40-layer Zipfian MoE (steady-state) | 0 (off) | ON | 0.0% | 24.41 GiB | 14.32 ms | 5.25 GB/s |
| 40-layer Zipfian MoE (steady-state) | 4 GiB pinned | ON | 25.6% | 18.17 GiB | 13.71 ms | 5.49 GB/s |
| 40-layer Zipfian MoE (steady-state) | 8 GiB pinned | ON | 45.4% | 13.33 GiB | 11.56 ms | 6.51 GB/s |
| Hot working set (L2 resident) | 4 GiB pinned | ON | 100.0% | 0.00 GiB | **1.55 ms** | **48.51 GB/s** |

With a second NVMe mirror attached (uniform cold misses, L2 off, host-fetch): single NVMe `17.06 ms/wave` (4.41 GB/s) scales automatically to dual NVMe `11.54 ms/wave` (6.52 GB/s).

## Bindings, Adapters & Tests

- Public C99 ABI: [include/wex_store.h](include/wex_store.h), [include/wex_loader.h](include/wex_loader.h)
- Python SDK package: [bindings/python/pyproject.toml](bindings/python/pyproject.toml)
- CLI & MCP tools: [cli/wex_cli.py](cli/wex_cli.py), [cli/wex_mcp.py](cli/wex_mcp.py)
- End-to-end tests: [scripts/test_libwex.sh](scripts/test_libwex.sh), [tests/test_wex_cli.py](tests/test_wex_cli.py)
- Engine & runtime adapters:
  - [Colibri](adapters/colibri/README.md)
  - [ggml-wex](adapters/ggml-wex/README.md)
  - [Strata](adapters/strata/README.md)
  - [Hermes](adapters/hermes/README.md)
  - [LM Studio](adapters/lmstudio/README.md)
