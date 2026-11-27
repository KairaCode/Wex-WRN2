# libwex Adapter for LM Studio

Use `libwex` with **LM Studio** in two practical ways: as a local OpenAI-compatible proxy/telemetry layer via `wex serve`, or directly inside a custom `llama.cpp` runtime via `ggml-wex` for single-NVMe MoE streaming with L2 Compressed-RAM caching.

See also: [libwex README](../../README.md), [C99 ABI (`wex_store.h`)](../../include/wex_store.h), [CLI (`wex_cli.py`)](../../cli/wex_cli.py), [MCP Server (`wex_mcp.py`)](../../cli/wex_mcp.py), and [ggml-wex Adapter](../ggml-wex/README.md).

## Approach A: Local OpenAI-Compatible Proxy (`wex serve`)

When running LM Studio's built-in local server (default `http://127.0.0.1:1234`), start `wex serve` as a lightweight local gateway that proxies OpenAI-compatible traffic while exposing WEX container inspection, L2 cache prefetching (`POST /v1/prefetch`), and live telemetry (`/v1/telemetry`, `/metrics`):

```bash
python libwex/cli/wex_cli.py serve /path/to/model.wex4 \
  --upstream http://127.0.0.1:1234 \
  --l2-mb auto \
  --host 127.0.0.1 --port 8080
```

You can also do the reverse: run an external WEX-enabled inference server behind `wex serve` on `http://127.0.0.1:8080/v1` and attach LM Studio (or any OpenAI-compatible client) to that local endpoint.

If you use MCP tools inside LM Studio, register [`libwex/cli/wex_mcp.py`](../../cli/wex_mcp.py) in LM Studio's `mcp.json`:

```json
{
  "mcpServers": {
    "libwex": {
      "command": "python",
      "args": ["libwex/cli/wex_mcp.py"]
    }
  }
}
```

## Approach B: Custom `llama.cpp` Runtime with `ggml-wex`

For direct single-NVMe MoE expert streaming (WNR4-GG, WNR4-MX/NV, WNR4-I4, WNR3, WBP1, RAW) with zero-copy decode and L2 Compressed-RAM caching:

1. Build `libwex` (`libwex.dll` / `libwex.so` / `libwex.dylib`) and optionally `libwex_cuda` for GPU-accelerated decode.
2. Follow the [`ggml-wex` adapter guide](../ggml-wex/README.md) to drop `wex_store.h` + `wex_loader.h` + `ggml_wex.c` into your custom `llama.cpp` runtime build (zero extra build dependencies — `libwex` is loaded dynamically at runtime via `wex_lib_open()`).
3. Place `libwex` (and optional `libwex_cuda`) alongside the custom runtime binary or set `WEX_LIB_PATH`.

### Single-NVMe Recommended Defaults

| Option (`wex_open_opts_t`) | Recommended Value | Why |
| --- | --- | --- |
| `l2_cache_bytes` | `UINT64_MAX` (`auto`) | Uses up to 25% of free RAM (capped at 16 GiB) to hold compressed hot experts in RAM, cutting single-NVMe reads dramatically. |
| `l2_pin_gpu` | `1` (with CUDA) | Page-locks the L2 compressed arena for fast PCIe H2D transfers into GPU decode kernels. |
| `io_direct` | `-1` (`auto`) | Selects Windows `OVERLAPPED` + `NO_BUFFERING`, Linux `io_uring` + `O_DIRECT`, or macOS `F_NOCACHE` automatically. |
| `verify` | `WEX_VERIFY_AUTO` | Enables first-read streaming 2 MiB SHA-256 verification whenever `root_hex` is pinned. |
