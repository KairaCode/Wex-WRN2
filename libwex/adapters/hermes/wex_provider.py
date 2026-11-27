# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Local Hermes provider and tool dispatcher for libwex (.wex / .wex4 containers).

Supports both direct in-process container access via pywex.WexStore and local HTTP
calls against `wex serve` (http://127.0.0.1:8080). Pure Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


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


class WexExpertProvider:
    """Inspect, verify, prefetch, and monitor WEX containers for Hermes workflows."""

    def __init__(
        self,
        container_path: Optional[Union[str, os.PathLike]] = None,
        *,
        base_url: Optional[str] = None,
        root_hex: Optional[str] = None,
        verify: Union[str, int, bool] = "auto",
        l2_cache_bytes: Union[int, str] = "auto",
        l2_pin_gpu: bool = False,
        gpu_device: int = -1,
        gpu_helper_path: Optional[Union[str, os.PathLike]] = None,
        lib_path: Optional[Union[str, os.PathLike]] = None,
        timeout: float = 5.0,
        store: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
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

    def __enter__(self) -> "WexExpertProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ---- HTTP helpers for `wex serve` mode ---------------------------------

    def _http_request(
        self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("No base_url configured for HTTP WEX requests")
        path = endpoint if endpoint.startswith("/") else "/" + endpoint
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json", "User-Agent": "libwex-hermes-provider/0.1"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else {"ok": True}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    # ---- Provider operations -----------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return WEX container metadata via local store or `wex serve` HTTP (/v1/telemetry)."""
        if self._store is not None:
            return self._store.info().to_dict()
        if self.base_url:
            tele = self._http_request("GET", "/v1/telemetry")
            return tele["store"] if isinstance(tele.get("store"), dict) else tele
        raise RuntimeError("WexExpertProvider requires either container_path/store or base_url")

    def stats(self) -> Dict[str, Any]:
        """Return live WEX I/O, L2 cache, and decode telemetry."""
        if self._store is not None:
            return self._store.stats().to_dict()
        if self.base_url:
            tele = self._http_request("GET", "/v1/telemetry")
            return tele["counters"] if isinstance(tele.get("counters"), dict) else tele
        raise RuntimeError("WexExpertProvider requires either container_path/store or base_url")

    def layout(self, layer: int, eid: int, *, by_record_id: bool = False) -> Dict[str, Any]:
        """Return record slab layout, tensor offsets, and expert shapes."""
        if self._store is None:
            raise RuntimeError("layout() requires a local WexStore (container_path or store)")
        return self._store.record_layout(int(layer), int(eid), by_record_id=by_record_id).to_dict()

    def prefetch(
        self, layer: int, eids: Iterable[int], *, by_record_id: bool = False
    ) -> Dict[str, Any]:
        """Warm the L2 Compressed-RAM cache with upcoming experts."""
        eid_list = [int(e) for e in eids]
        if self._store is not None:
            queued = int(
                self._store.prefetch(int(layer), eid_list, by_record_id=by_record_id)
            )
            return {"layer": int(layer), "eids": eid_list, "queued": queued}
        if self.base_url:
            return self._http_request(
                "POST",
                "/v1/prefetch",
                {"layer": int(layer), "eids": eid_list, "by_record_id": bool(by_record_id)},
            )
        raise RuntimeError("WexExpertProvider requires either container_path/store or base_url")

    def verify_expert(
        self, layer: int, eid: int, *, by_record_id: bool = False
    ) -> Dict[str, Any]:
        """Fetch and decode one expert/record to verify SHA-256 and codec integrity."""
        if self._store is None:
            raise RuntimeError("verify_expert() requires a local WexStore (container_path or store)")
        lay = self._store.record_layout(int(layer), int(eid), by_record_id=by_record_id)
        bufs = self._store.fetch_host(
            int(layer),
            [int(eid)],
            scratch=False,
            by_record_id=by_record_id,
        )
        decoded = bytes(bufs[0])
        digest = hashlib.sha256(decoded).hexdigest()
        return {
            "verified": True,
            "layer": int(layer),
            "eid": int(eid),
            "record_index": lay.record_index,
            "codec_name": lay.codec_name,
            "stored_bytes": lay.stored_bytes,
            "decoded_bytes": len(decoded),
            "decoded_sha256": digest,
        }


# ---- Hermes tool schemas & dispatcher --------------------------------------


def get_hermes_tools() -> List[Dict[str, Any]]:
    """Return JSON-Schema tool definitions ready to register in local Hermes workflows."""
    return [
        {
            "type": "function",
            "function": {
                "name": "wex_info",
                "description": (
                    "Inspect a WEX4/WNR4 container: layers, experts, primary codec, "
                    "compression ratio, root SHA-256, active backend, and L2 capacity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_path": {
                            "type": "string",
                            "description": "Path to a local .wex/.wex4 container (optional if provider is bound).",
                        },
                        "base_url": {
                            "type": "string",
                            "description": "Local wex serve base URL, e.g. http://127.0.0.1:8080.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wex_stats",
                "description": (
                    "Read live WEX telemetry: L2 Compressed-RAM hit rate, disk vs RAM fetches, "
                    "GPU/CPU decode counts, bytes transferred, and timings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_path": {"type": "string"},
                        "base_url": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wex_prefetch",
                "description": "Asynchronously warm the WEX L2 Compressed-RAM cache with upcoming experts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "integer", "minimum": 0},
                        "eids": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "by_record_id": {"type": "boolean", "default": False},
                        "container_path": {"type": "string"},
                        "base_url": {"type": "string"},
                    },
                    "required": ["layer", "eids"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wex_verify_expert",
                "description": (
                    "Read, SHA-256 verify, and losslessly decode one expert or flat record, "
                    "returning its byte count and decoded SHA-256 digest."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "integer", "minimum": 0},
                        "eid": {"type": "integer", "minimum": 0},
                        "by_record_id": {"type": "boolean", "default": False},
                        "container_path": {"type": "string"},
                    },
                    "required": ["layer", "eid"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wex_layout",
                "description": "Inspect decoded slab size, tensor offsets, and expert shapes for a record.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "integer", "minimum": 0},
                        "eid": {"type": "integer", "minimum": 0},
                        "by_record_id": {"type": "boolean", "default": False},
                        "container_path": {"type": "string"},
                    },
                    "required": ["layer", "eid"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def dispatch_tool(
    name: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    provider: Optional[WexExpertProvider] = None,
    lib_path: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """Execute a Hermes WEX tool call and return a JSON-serializable result dict."""
    params = dict(args or {})
    tool_name = str(name).strip()
    owned_provider: Optional[WexExpertProvider] = None

    try:
        active = provider
        if active is None:
            container_path = params.get("container_path")
            base_url = params.get("base_url")
            if not container_path and not base_url:
                return {
                    "ok": False,
                    "tool": tool_name,
                    "error": "Provide a bound provider or pass container_path / base_url in args",
                }
            owned_provider = WexExpertProvider(
                container_path,
                base_url=base_url,
                verify="on" if tool_name == "wex_verify_expert" else "auto",
                l2_cache_bytes=0 if tool_name == "wex_verify_expert" else "auto",
                lib_path=lib_path,
            )
            active = owned_provider

        if tool_name == "wex_info":
            return {"ok": True, "tool": tool_name, "result": active.info()}

        if tool_name == "wex_stats":
            return {"ok": True, "tool": tool_name, "result": active.stats()}

        if tool_name == "wex_prefetch":
            layer = int(params["layer"])
            eids = [int(e) for e in params["eids"]]
            by_record_id = bool(params.get("by_record_id", False))
            return {
                "ok": True,
                "tool": tool_name,
                "result": active.prefetch(layer, eids, by_record_id=by_record_id),
            }

        if tool_name == "wex_verify_expert":
            layer = int(params["layer"])
            eid = int(params["eid"])
            by_record_id = bool(params.get("by_record_id", False))
            return {
                "ok": True,
                "tool": tool_name,
                "result": active.verify_expert(layer, eid, by_record_id=by_record_id),
            }

        if tool_name == "wex_layout":
            layer = int(params["layer"])
            eid = int(params["eid"])
            by_record_id = bool(params.get("by_record_id", False))
            return {
                "ok": True,
                "tool": tool_name,
                "result": active.layout(layer, eid, by_record_id=by_record_id),
            }

        return {"ok": False, "tool": tool_name, "error": f"Unknown WEX tool: {tool_name!r}"}
    except Exception as exc:
        return {"ok": False, "tool": tool_name, "error": str(exc)}
    finally:
        if owned_provider is not None:
            owned_provider.close()


__all__ = [
    "WexExpertProvider",
    "get_hermes_tools",
    "dispatch_tool",
]
