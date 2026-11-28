# libwex Adapter for Hermes

Connect local **Hermes** workflows to a `libwex` WEX4/WNR4 expert store over a local OpenAI-compatible HTTP endpoint (`wex serve`), direct Python tool dispatch (`wex_provider.py`), or MCP (`wex_mcp.py`).

See also: [libwex README](../../README.md), [C99 ABI (`wex_store.h`)](../../include/wex_store.h), [CLI (`wex_cli.py`)](../../cli/wex_cli.py), [MCP Server (`wex_mcp.py`)](../../cli/wex_mcp.py), and [`wex_provider.py`](wex_provider.py).

## 1. Connect Hermes to `wex serve`

Start `wex serve` pointing at your local inference backend and `.wex4` container:

```bash
python libwex/cli/wex_cli.py serve /path/to/model.wex4 \
  --upstream http://127.0.0.1:8081 \
  --l2-mb auto \
  --host 127.0.0.1 --port 8080
```

Then configure your Hermes local model provider with:

```toml
base_url = "http://127.0.0.1:8080/v1"
api_key = "local"
```

If `wex serve` runs with a token (`WEX_SERVE_TOKEN` / `--token`, mandatory on a non-loopback `--host`), set `api_key` to that token. `wex serve` does not forward the key to `--upstream`; see [`serve` Access Control](../../README.md#serve-access-control).

## 2. Register WEX Python Tools in Hermes

Use `get_hermes_tools()` and `dispatch_tool()` from [`wex_provider.py`](wex_provider.py) to let local Hermes workflows inspect container geometry, check L2 Compressed-RAM hit rates, verify individual experts via SHA-256, and trigger asynchronous prefetching:

```python
from wex_provider import WexExpertProvider, dispatch_tool, get_hermes_tools

tools = get_hermes_tools()

with WexExpertProvider(
    "/path/to/model.wex4",
    l2_cache_bytes="auto",
    verify="on",
) as provider:
    info = dispatch_tool("wex_info", provider=provider)
    pf = dispatch_tool("wex_prefetch", {"layer": 0, "eids": [0, 1, 2, 3]}, provider=provider)
    ver = dispatch_tool("wex_verify_expert", {"layer": 0, "eid": 0}, provider=provider)
    stats = dispatch_tool("wex_stats", provider=provider)
    print(info, pf, ver, stats)
```

To query a running `wex serve` instance over HTTP (`/v1/telemetry` and `POST /v1/prefetch`) instead of opening the container directly in-process, initialize `WexExpertProvider(base_url="http://127.0.0.1:8080")`.

## 3. Connect via MCP (`wex_mcp.py`)

If your Hermes setup uses MCP servers over stdio, register [`libwex/cli/wex_mcp.py`](../../cli/wex_mcp.py) as a local tool server:

```json
{
  "mcpServers": {
    "libwex": {
      "command": "python",
      "args": ["libwex/cli/wex_mcp.py", "--root-dir", "/path/to/models"]
    }
  }
}
```

The tools open containers only under `--root-dir` (default: the server's working directory); see [MCP Server](../../README.md#mcp-server).
