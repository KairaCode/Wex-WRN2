# libwex Adapter for Strata

Connect a single-NVMe WEX4/WNR4 expert store (with L2 Compressed-RAM caching and optional CUDA decode) to a local **Strata** inference server (`http://127.0.0.1:8080`).

Strata exposes standard local HTTP endpoints (`/v1/chat/completions`, `/v1/responses`, `/v1/messages` Messages API, `/v1/models`, `/health`, `/status`, `/metrics`, `/v1/telemetry`), plus CLI `--output json`, JSON stdin/stdout IPC, and Python MCP workflows.

See also: [libwex README](../../README.md), [C99 ABI (`wex_store.h`)](../../include/wex_store.h), [CLI (`wex_cli.py`)](../../cli/wex_cli.py), [MCP Server (`wex_mcp.py`)](../../cli/wex_mcp.py), and [`wex_strata_bridge.py`](wex_strata_bridge.py).

## Option A: Unified HTTP Proxy (`wex serve`)

Run `wex serve` in front of a running Strata server so clients get both Strata's inference endpoints and live WEX container telemetry on one local port:

```bash
python libwex/cli/wex_cli.py serve /path/to/model.wex4 \
  --upstream http://127.0.0.1:8080 \
  --l2-mb auto \
  --host 127.0.0.1 --port 8090
```

Clients can keep using `/v1/chat/completions`, `/v1/responses`, `/v1/messages` (Messages API), and `/v1/models` against `http://127.0.0.1:8090` (with `X-WEX-L2-Hit-Rate` headers on proxied responses), while monitoring `/v1/telemetry`, `/metrics`, `/status`, `/health`, and triggering `POST /v1/prefetch`.

## Option B: `wex_strata_bridge.py` (CLI JSON & Stdin/Stdout IPC)

### Merged JSON Telemetry

```bash
python libwex/adapters/strata/wex_strata_bridge.py \
  --container /path/to/model.wex4 \
  --strata-url http://127.0.0.1:8080 \
  --l2-cache auto \
  --output json
```

### Line-Delimited JSON Stdin/Stdout IPC

Start the bridge in `--ipc` mode to drive `info`, `stats`, `telemetry`, `prefetch`, `has_expert`, `strata_health`, `strata_models`, and `quit` over pipes from Node/TypeScript, Rust, or shell scripts:

```bash
printf '{"op":"info"}\n{"op":"prefetch","layer":0,"eids":[0,1,2,3]}\n{"op":"stats"}\n{"op":"strata_health"}\n{"op":"quit"}\n' \
  | python libwex/adapters/strata/wex_strata_bridge.py \
      --container /path/to/model.wex4 \
      --strata-url http://127.0.0.1:8080 \
      --ipc
```

## Client Examples

### TypeScript / Node (`openai` SDK & `fetch`)

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8090/v1",
  apiKey: "local",
});

const reply = await client.chat.completions.create({
  model: "local-moe",
  messages: [{ role: "user", content: "Summarize WEX4 L2 caching in one line." }],
});
console.log(reply.choices[0].message.content);

// Query merged health/telemetry directly with fetch
const health = await fetch("http://127.0.0.1:8080/health").then((r) => r.json());
console.log("Strata health:", health);
```

### Rust (`reqwest` / `async-openai`)

```rust
// Calling Strata's /v1/models and /v1/chat/completions via reqwest
let client = reqwest::Client::new();
let models: serde_json::Value = client
    .get("http://127.0.0.1:8090/v1/models")
    .send()
    .await?
    .json()
    .await?;
println!("models: {models}");
```

### Python (`StrataWexBridge` & `pywex`)

```python
from wex_strata_bridge import StrataWexBridge

with StrataWexBridge(
    "/path/to/model.wex4",
    strata_url="http://127.0.0.1:8080",
    l2_cache_bytes="auto",
) as bridge:
    bridge.prefetch(layer=0, eids=[0, 1, 2, 3])
    snapshot = bridge.telemetry()
    print("L2 hit rate:", snapshot["wex"]["stats"]["l2_hit_rate"])
    print("Strata healthy:", snapshot["strata"]["health"]["ok"])
```

### Python MCP Integration

Register [`libwex/cli/wex_mcp.py`](../../cli/wex_mcp.py) alongside your Strata MCP workflow to expose WEX container inspection, verification, L2 prefetch, and telemetry tools directly to local tool-calling runtimes.
