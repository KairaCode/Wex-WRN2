# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Pure-stdlib bridge attaching a libwex WexStore to a local Strata HTTP server.

Supports:
  - Single-NVMe + L2 Compressed-RAM expert caching via pywex.WexStore
  - Querying Strata local HTTP endpoints (/health, /status, /v1/models, /v1/telemetry)
  - Merged JSON telemetry combining WEX store counters and Strata server health
  - Line-delimited JSON stdin/stdout IPC loop for TypeScript, Rust, or CLI integration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, TextIO, Union


def _import_pywex() -> Any:
    try:
        import pywex

        return pywex
    except ImportError:
        py_root = Path(__file__).resolve().parents[2] / "bindings" / "python"
        if py_root.is_dir() and str(py_root) not in sys.path:
            sys.path.insert(0, str(py_root))
        import pywex

        return pywex


def _parse_size_arg(raw: Union[str, int]) -> Union[int, str]:
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if s.lower() == "auto":
        return "auto"
    suffixes = {
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    low = s.lower()
    for suf, mult in sorted(suffixes.items(), key=lambda kv: -len(kv[0])):
        if low.endswith(suf):
            num = float(low[: -len(suf)].strip())
            return int(num * mult)
    return int(s)


class StrataWexBridge:
    """Attach a libwex container store to a running local Strata HTTP server."""

    def __init__(
        self,
        container_path: Optional[Union[str, os.PathLike]] = None,
        *,
        strata_url: str = "http://127.0.0.1:8080",
        root_hex: Optional[str] = None,
        verify: Union[str, int, bool] = "auto",
        l2_cache_bytes: Union[int, str] = "auto",
        l2_pin_gpu: bool = False,
        io_direct: int = -1,
        io_queue_depth: int = 0,
        cpu_threads: int = 0,
        gpu_device: int = -1,
        gpu_helper_path: Optional[Union[str, os.PathLike]] = None,
        lib_path: Optional[Union[str, os.PathLike]] = None,
        timeout: float = 5.0,
        store: Optional[Any] = None,
    ) -> None:
        self.strata_url = strata_url.rstrip("/")
        self.timeout = float(timeout)
        self._owns_store = False
        self._store: Optional[Any] = store

        if self._store is None and container_path is not None:
            pywex = _import_pywex()
            self._store = pywex.WexStore(
                container_path,
                root_hex=root_hex,
                verify=verify,
                l2_cache_bytes=l2_cache_bytes,
                l2_pin_gpu=l2_pin_gpu,
                io_direct=io_direct,
                io_queue_depth=io_queue_depth,
                cpu_threads=cpu_threads,
                gpu_device=gpu_device,
                gpu_helper_path=gpu_helper_path,
                lib_path=lib_path,
            )
            self._owns_store = True

    @property
    def store(self) -> Optional[Any]:
        return self._store

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None
            self._owns_store = False

    def __enter__(self) -> "StrataWexBridge":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ---- Strata local HTTP queries -----------------------------------------

    def strata_get(self, endpoint: str) -> Dict[str, Any]:
        """Perform an HTTP GET against the local Strata server and return a JSON dict."""
        path = endpoint if endpoint.startswith("/") else "/" + endpoint
        url = f"{self.strata_url}{path}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "libwex-strata-bridge/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = int(getattr(resp, "status", 200))
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    data = {"raw": raw}
                return {"ok": 200 <= status < 300, "status": status, "url": url, "data": data}
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return {
                "ok": False,
                "status": int(exc.code),
                "url": url,
                "error": f"HTTP {exc.code}: {exc.reason}",
                "body": body,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "ok": False,
                "status": None,
                "url": url,
                "error": str(exc),
            }

    def strata_health(self) -> Dict[str, Any]:
        """Query Strata /health (falling back to /status if /health returns 404)."""
        res = self.strata_get("/health")
        if not res["ok"] and res.get("status") == 404:
            res = self.strata_get("/status")
        return res

    def strata_models(self) -> Dict[str, Any]:
        """Query Strata OpenAI-compatible /v1/models endpoint."""
        return self.strata_get("/v1/models")

    def strata_telemetry(self) -> Dict[str, Any]:
        """Query Strata /v1/telemetry endpoint."""
        return self.strata_get("/v1/telemetry")

    # ---- WEX container operations ------------------------------------------

    def wex_info(self) -> Optional[Dict[str, Any]]:
        if self._store is None:
            return None
        return self._store.info().to_dict()

    def wex_stats(self) -> Optional[Dict[str, Any]]:
        if self._store is None:
            return None
        return self._store.stats().to_dict()

    def has_expert(self, layer: int, eid: int) -> bool:
        if self._store is None:
            return False
        return bool(self._store.has_expert(int(layer), int(eid)))

    def prefetch(
        self, layer: int, eids: Iterable[int], *, by_record_id: bool = False
    ) -> int:
        if self._store is None:
            return 0
        return int(self._store.prefetch(int(layer), [int(e) for e in eids], by_record_id=by_record_id))

    def telemetry(self, *, include_models: bool = True, include_strata_telemetry: bool = True) -> Dict[str, Any]:
        """Return merged JSON telemetry combining libwex store state and Strata server health."""
        wex_section: Dict[str, Any] = {
            "attached": self._store is not None,
            "info": self.wex_info(),
            "stats": self.wex_stats(),
        }
        strata_section: Dict[str, Any] = {
            "url": self.strata_url,
            "health": self.strata_health(),
        }
        if include_models:
            strata_section["models"] = self.strata_models()
        if include_strata_telemetry:
            strata_section["telemetry"] = self.strata_telemetry()

        return {
            "ok": True,
            "wex": wex_section,
            "strata": strata_section,
        }

    # ---- JSON stdin/stdout IPC protocol ------------------------------------

    def handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single JSON IPC command dict and return a JSON-serializable response."""
        op = str(cmd.get("op", "")).strip().lower()
        if op == "quit" or op == "exit":
            return {"ok": True, "op": "quit", "bye": True}

        if op == "info":
            if self._store is None:
                return {"ok": False, "op": op, "error": "No WEX container attached"}
            return {"ok": True, "op": op, "info": self.wex_info()}

        if op == "stats":
            if self._store is None:
                return {"ok": False, "op": op, "error": "No WEX container attached"}
            return {"ok": True, "op": op, "stats": self.wex_stats()}

        if op == "telemetry":
            return self.telemetry(
                include_models=bool(cmd.get("include_models", True)),
                include_strata_telemetry=bool(cmd.get("include_strata_telemetry", True)),
            )

        if op == "prefetch":
            if self._store is None:
                return {"ok": False, "op": op, "error": "No WEX container attached"}
            layer = int(cmd.get("layer", 0))
            raw_eids = cmd.get("eids", [])
            if not isinstance(raw_eids, list):
                return {"ok": False, "op": op, "error": "eids must be a JSON list of integers"}
            by_record_id = bool(cmd.get("by_record_id", False))
            queued = self.prefetch(layer, [int(e) for e in raw_eids], by_record_id=by_record_id)
            return {"ok": True, "op": op, "layer": layer, "queued": queued}

        if op == "has_expert":
            if self._store is None:
                return {"ok": False, "op": op, "error": "No WEX container attached"}
            layer = int(cmd.get("layer", 0))
            eid = int(cmd.get("eid", 0))
            return {"ok": True, "op": op, "layer": layer, "eid": eid, "present": self.has_expert(layer, eid)}

        if op == "strata_health":
            res = self.strata_health()
            return {"ok": bool(res.get("ok")), "op": op, "strata": res}

        if op == "strata_models":
            res = self.strata_models()
            return {"ok": bool(res.get("ok")), "op": op, "strata": res}

        return {"ok": False, "op": op, "error": f"Unknown IPC command: {op!r}"}

    def serve_ipc(
        self,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
    ) -> int:
        """Run a line-delimited JSON stdin/stdout IPC loop until EOF or 'quit'."""
        in_stream = stdin if stdin is not None else sys.stdin
        out_stream = stdout if stdout is not None else sys.stdout
        processed = 0

        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                if not isinstance(cmd, dict):
                    resp: Dict[str, Any] = {"ok": False, "error": "Command must be a JSON object"}
                else:
                    resp = self.handle_command(cmd)
            except json.JSONDecodeError as exc:
                resp = {"ok": False, "error": f"Invalid JSON: {exc}"}
            except Exception as exc:
                resp = {"ok": False, "error": str(exc)}

            out_stream.write(json.dumps(resp, separators=(",", ":")) + "\n")
            out_stream.flush()
            processed += 1
            if resp.get("bye"):
                break

        return processed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="libwex bridge for local Strata HTTP servers (telemetry, prefetch, JSON IPC)."
    )
    parser.add_argument("--container", help="Path to a .wex / .wex4 container file")
    parser.add_argument("--strata-url", default="http://127.0.0.1:8080", help="Strata base URL")
    parser.add_argument("--root-hex", default=None, help="Optional 64-char SHA-256 root hash to pin")
    parser.add_argument("--verify", default="auto", choices=["auto", "on", "off"], help="Record verification mode")
    parser.add_argument("--l2-cache", default="auto", help="L2 cache bytes ('auto', '0', or size like '4G')")
    parser.add_argument("--l2-pin-gpu", action="store_true", help="Pin L2 compressed-RAM arena for GPU H2D")
    parser.add_argument("--gpu-device", type=int, default=-1, help="-1=auto, -2=CPU only, >=0=GPU ordinal")
    parser.add_argument("--gpu-helper", default=None, help="Explicit libwex_cuda path or '0' to disable")
    parser.add_argument("--lib-path", default=None, help="Explicit path to libwex shared library")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    parser.add_argument("--output", choices=["json"], default="json", help="Output format")
    parser.add_argument("--ipc", action="store_true", help="Run line-delimited JSON stdin/stdout IPC loop")
    parser.add_argument("--health", action="store_true", help="Query Strata /health and print JSON")
    args = parser.parse_args(argv)

    with StrataWexBridge(
        args.container,
        strata_url=args.strata_url,
        root_hex=args.root_hex,
        verify=args.verify,
        l2_cache_bytes=_parse_size_arg(args.l2_cache),
        l2_pin_gpu=args.l2_pin_gpu,
        gpu_device=args.gpu_device,
        gpu_helper_path=args.gpu_helper,
        lib_path=args.lib_path,
        timeout=args.timeout,
    ) as bridge:
        if args.ipc:
            bridge.serve_ipc()
            return 0
        if args.health:
            print(json.dumps(bridge.strata_health(), indent=2))
            return 0
        print(json.dumps(bridge.telemetry(), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
