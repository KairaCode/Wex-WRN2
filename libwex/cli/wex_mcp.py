#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Spec-compliant MCP JSON-RPC 2.0 stdio server for libwex (stdlib only).

Supports both newline-delimited JSON-RPC (MCP stdio default) and
Content-Length framed JSON-RPC messages over stdin/stdout.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

_CLI_DIR = os.path.dirname(os.path.abspath(__file__))
if _CLI_DIR not in sys.path:
    sys.path.insert(0, _CLI_DIR)

import wex_cli  # noqa: E402

SERVER_NAME = "libwex-mcp"
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "wex_inspect",
        "description": "Inspect a local .wex / .wex4 container header, codec, compression ratio, root SHA-256, I/O engine, and decode backend.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the local .wex / .wex4 container file."},
                "cpu_only": {"type": "boolean", "description": "Disable GPU backend discovery.", "default": False},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wex_verify",
        "description": "Verify the root SHA-256 and stream-verify every record in a local .wex / .wex4 container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the local .wex / .wex4 container file."},
                "root_hex": {"type": "string", "description": "Optional 64-char hex root SHA-256 to pin."},
                "l2_mb": {"type": "integer", "description": "L2 cache size in MiB (default 0).", "default": 0, "minimum": 0},
                "cpu_only": {"type": "boolean", "description": "Force CPU-only decode.", "default": False},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wex_bench",
        "description": "Run a single-NVMe + L2 compressed-RAM LRU cache benchmark sweep on a local .wex / .wex4 container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the local .wex / .wex4 container file."},
                "l2_mb": {"type": "string", "description": "Comma-separated L2 sizes in MiB or 'auto' (e.g. '0,64,256').", "default": "0,64"},
                "waves": {"type": "integer", "description": "Number of fetch waves.", "default": 32, "minimum": 1},
                "batch": {"type": "integer", "description": "Records per wave.", "default": 4, "minimum": 1},
                "pattern": {"type": "string", "enum": ["zipf", "uniform", "seq"], "description": "Access pattern.", "default": "zipf"},
                "verify": {"type": "boolean", "description": "Enable first-read SHA-256 verification.", "default": False},
                "cpu_only": {"type": "boolean", "description": "Force CPU-only decode.", "default": False},
                "seed": {"type": "integer", "description": "Deterministic PRNG seed.", "default": 42},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wex_server_telemetry",
        "description": "Query /v1/telemetry, /status, or /health from a running local wex serve HTTP instance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {"type": "integer", "description": "TCP port of wex serve (default 8080).", "default": 8080, "minimum": 1, "maximum": 65535},
                "host": {"type": "string", "description": "Loopback host (127.0.0.1 or localhost).", "default": "127.0.0.1"},
                "endpoint": {"type": "string", "enum": ["telemetry", "status", "health"], "description": "Endpoint to query.", "default": "telemetry"},
                "timeout": {"type": "number", "description": "HTTP timeout in seconds.", "default": 5.0},
            },
            "additionalProperties": False,
        },
    },
]

ENDPOINT_PATHS = {
    "telemetry": "/v1/telemetry",
    "status": "/status",
    "health": "/health",
}


def query_server_telemetry(port=8080, host="127.0.0.1", endpoint="telemetry", timeout=5.0):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("host must be a loopback address (127.0.0.1, localhost, or ::1)")
    port_num = int(port)
    if port_num < 1 or port_num > 65535:
        raise ValueError(f"invalid port: {port}")
    route = ENDPOINT_PATHS.get(endpoint)
    if not route:
        raise ValueError(f"unsupported endpoint: {endpoint!r}")

    host_fmt = f"[{host}]" if ":" in host else host
    url = f"http://{host_fmt}:{port_num}{route}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to connect to {url}: {exc.reason}") from exc


class WexMcpServer:
    """JSON-RPC 2.0 dispatcher for libwex MCP tools."""

    def __init__(self, default_lib_path=None, default_cuda_path=None):
        self.default_lib_path = default_lib_path
        self.default_cuda_path = default_cuda_path

    def _call_tool(self, name, arguments):
        args = arguments if isinstance(arguments, dict) else {}
        lib_path = self.default_lib_path

        if name == "wex_inspect":
            return wex_cli.run_inspect(
                args["path"],
                lib_path=lib_path,
                cuda_path=self.default_cuda_path,
                cpu_only=bool(args.get("cpu_only", False)),
            )

        if name == "wex_verify":
            return wex_cli.run_verify(
                args["path"],
                root_hex=args.get("root_hex"),
                l2_mb=int(args.get("l2_mb", 0)),
                cpu_only=bool(args.get("cpu_only", False)),
                lib_path=lib_path,
                cuda_path=self.default_cuda_path,
            )

        if name == "wex_bench":
            return wex_cli.run_bench(
                args["path"],
                l2_mb_list=str(args.get("l2_mb", "0,64")),
                waves=int(args.get("waves", 32)),
                batch=int(args.get("batch", 4)),
                pattern=str(args.get("pattern", "zipf")),
                verify=bool(args.get("verify", False)),
                cpu_only=bool(args.get("cpu_only", False)),
                seed=int(args.get("seed", 42)),
                lib_path=lib_path,
                cuda_path=self.default_cuda_path,
            )

        if name == "wex_server_telemetry":
            return query_server_telemetry(
                port=int(args.get("port", 8080)),
                host=str(args.get("host", "127.0.0.1")),
                endpoint=str(args.get("endpoint", "telemetry")),
                timeout=float(args.get("timeout", 5.0)),
            )

        raise KeyError(f"Unknown tool: {name}")

    def handle_message(self, msg):
        if not isinstance(msg, dict):
            return self._error(None, -32600, "Invalid Request")

        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # Notifications carry no id and expect no reply
        if "id" not in msg or req_id is None:
            if method and method.startswith("notifications/"):
                return None
            return None

        if not isinstance(method, str):
            return self._error(req_id, -32600, "Invalid Request: missing method")

        if method == "initialize":
            client_proto = params.get("protocolVersion") if isinstance(params, dict) else None
            proto = client_proto if isinstance(client_proto, str) and client_proto else DEFAULT_PROTOCOL_VERSION
            return self._result(
                req_id,
                {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )

        if method == "ping":
            return self._result(req_id, {})

        if method == "tools/list":
            return self._result(req_id, {"tools": TOOLS})

        if method == "tools/call":
            if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                return self._error(req_id, -32602, "Invalid params: expected tool name")
            tname = params["name"]
            targs = params.get("arguments", {})
            try:
                data = self._call_tool(tname, targs)
                return self._result(
                    req_id,
                    {
                        "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
                        "structuredContent": data,
                        "isError": False,
                    },
                )
            except KeyError as exc:
                return self._error(req_id, -32602, str(exc))
            except Exception as exc:
                err_payload = {"ok": False, "error": str(exc)}
                return self._result(
                    req_id,
                    {
                        "content": [{"type": "text", "text": json.dumps(err_payload, indent=2)}],
                        "structuredContent": err_payload,
                        "isError": True,
                    },
                )

        return self._error(req_id, -32601, f"Method not found: {method}")

    @staticmethod
    def _result(req_id, result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id, code, message):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _read_next_message(stream):
    """Read one JSON-RPC message; returns (raw_bytes, framed_bool) or (None, False) on EOF."""
    while True:
        line = stream.readline()
        if not line:
            return None, False
        stripped = line.strip()
        if not stripped:
            continue

        low = stripped.lower()
        if low.startswith(b"content-length:"):
            try:
                length = int(stripped.split(b":", 1)[1].strip())
            except ValueError:
                return b"", True
            # Consume remaining headers until empty line
            while True:
                hdr = stream.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
            body = stream.read(length) if length > 0 else b""
            return body, True

        return stripped, False


def _write_message(stream, obj, framed):
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if framed:
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        stream.write(header + payload)
    else:
        stream.write(payload + b"\n")
    stream.flush()


def serve_stdio(server, in_stream=None, out_stream=None):
    inp = in_stream if in_stream is not None else sys.stdin.buffer
    out = out_stream if out_stream is not None else sys.stdout.buffer

    while True:
        raw, framed = _read_next_message(inp)
        if raw is None:
            break
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _write_message(out, server._error(None, -32700, f"Parse error: {exc}"), framed)
            continue

        reply = server.handle_message(msg)
        if reply is not None:
            _write_message(out, reply, framed)


def main(argv=None):
    parser = argparse.ArgumentParser(description="libwex MCP JSON-RPC 2.0 stdio server")
    parser.add_argument("--lib", dest="lib_path", default=None, help="Explicit path to libwex shared library")
    parser.add_argument("--cuda", dest="cuda_path", default=None, help="Explicit path to libwex_cuda shared library")
    args = parser.parse_args(argv)

    server = WexMcpServer(default_lib_path=args.lib_path, default_cuda_path=args.cuda_path)
    serve_stdio(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
