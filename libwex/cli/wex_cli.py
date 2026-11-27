#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Zero-dependency CLI, telemetry HTTP server, and IPC driver for libwex.

Uses Python 3 standard library only (ctypes, http.server, http.client,
urllib.request, json, argparse). Loads libwex ABI v1 directly.
"""

import argparse
import ctypes
import http.client
import http.server
import json
import os
import random
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

WEX_STORE_ABI_VERSION = 1
WEX_MAX_RECORD_TENSORS = 16
UINT64_MAX = (1 << 64) - 1

# Status codes (wex_store.h)
WEX_OK = 0
WEX_E_ARGS = -1
WEX_E_IO = -2
WEX_E_FORMAT = -3
WEX_E_ROOT = -4
WEX_E_HASH = -5
WEX_E_DECODE = -6
WEX_E_UNSUPPORTED = -7
WEX_E_OOM = -8
WEX_E_NOT_FOUND = -9
WEX_E_CAP = -10
WEX_E_DEVICE = -11
WEX_E_REFUSED = -12

STATUS_NAMES = {
    WEX_OK: "WEX_OK",
    WEX_E_ARGS: "WEX_E_ARGS",
    WEX_E_IO: "WEX_E_IO",
    WEX_E_FORMAT: "WEX_E_FORMAT",
    WEX_E_ROOT: "WEX_E_ROOT",
    WEX_E_HASH: "WEX_E_HASH",
    WEX_E_DECODE: "WEX_E_DECODE",
    WEX_E_UNSUPPORTED: "WEX_E_UNSUPPORTED",
    WEX_E_OOM: "WEX_E_OOM",
    WEX_E_NOT_FOUND: "WEX_E_NOT_FOUND",
    WEX_E_CAP: "WEX_E_CAP",
    WEX_E_DEVICE: "WEX_E_DEVICE",
    WEX_E_REFUSED: "WEX_E_REFUSED",
}

# Verify modes (wex_verify_mode_t)
WEX_VERIFY_AUTO = 0
WEX_VERIFY_OFF = 1
WEX_VERIFY_ON = 2

VERIFY_MODE_MAP = {
    "auto": WEX_VERIFY_AUTO,
    "off": WEX_VERIFY_OFF,
    "on": WEX_VERIFY_ON,
}

# Codec kinds (wex_codec_kind_t)
CODEC_NAMES = {
    0: "RAW",
    5: "WNR3",
    6: "WBP1",
    8: "WNR4-MX",
    9: "WNR4-NV",
    10: "WNR4-GG",
    11: "WNR4-I4",
}

# Fetch flags
WEX_FETCH_DEFAULT = 0
WEX_FETCH_SCRATCH = 1 << 0
WEX_FETCH_BYPASS_L2 = 1 << 1
WEX_FETCH_BY_RECORD_ID = 1 << 2
WEX_FETCH_EXPAND_E8M0_F32 = 1 << 3

HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
})


class WexOpenOpts(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("root_hex", ctypes.c_char_p),
        ("verify", ctypes.c_int),
        ("mirrors", ctypes.POINTER(ctypes.c_char_p)),
        ("num_mirrors", ctypes.c_int),
        ("l2_cache_bytes", ctypes.c_uint64),
        ("l2_pin_gpu", ctypes.c_int),
        ("io_direct", ctypes.c_int),
        ("io_queue_depth", ctypes.c_int),
        ("cpu_threads", ctypes.c_int),
        ("gpu_device", ctypes.c_int),
        ("gpu_helper_path", ctypes.c_char_p),
        ("disk_weights", ctypes.POINTER(ctypes.c_uint32)),
    ]


class WexInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("layers", ctypes.c_uint32),
        ("experts", ctypes.c_uint32),
        ("num_records", ctypes.c_uint32),
        ("num_tensors", ctypes.c_uint32),
        ("codec_kind", ctypes.c_uint32),
        ("value_exact", ctypes.c_uint32),
        ("verified_root", ctypes.c_uint32),
        ("verify_on", ctypes.c_uint32),
        ("num_replicas", ctypes.c_uint32),
        ("file_bytes", ctypes.c_uint64),
        ("total_decoded_bytes", ctypes.c_uint64),
        ("max_record_decoded_bytes", ctypes.c_uint64),
        ("max_record_stored_bytes", ctypes.c_uint64),
        ("l2_capacity_bytes", ctypes.c_uint64),
        ("root_hex", ctypes.c_char * 65),
        ("backend_name", ctypes.c_char * 32),
        ("io_engine", ctypes.c_char * 32),
    ]


class WexLayout(ctypes.Structure):
    _fields_ = [
        ("record_index", ctypes.c_uint32),
        ("codec_kind", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("tensor_count", ctypes.c_uint32),
        ("file_offset", ctypes.c_uint64),
        ("stored_bytes", ctypes.c_uint64),
        ("decoded_bytes", ctypes.c_uint64),
        ("tensor_offsets", ctypes.c_uint64 * WEX_MAX_RECORD_TENSORS),
        ("tensor_bytes", ctypes.c_uint64 * WEX_MAX_RECORD_TENSORS),
        ("scale_offsets", ctypes.c_uint32 * 3),
        ("weight_offsets", ctypes.c_uint32 * 3),
        ("scale_shapes", (ctypes.c_uint32 * 2) * 3),
        ("weight_shapes", (ctypes.c_uint32 * 2) * 3),
    ]


class WexStats(ctypes.Structure):
    _fields_ = [
        ("fetches", ctypes.c_uint64),
        ("records_requested", ctypes.c_uint64),
        ("records_from_l2", ctypes.c_uint64),
        ("records_from_disk", ctypes.c_uint64),
        ("gpu_decodes", ctypes.c_uint64),
        ("cpu_decodes", ctypes.c_uint64),
        ("prefetch_hits", ctypes.c_uint64),
        ("refused", ctypes.c_uint64),
        ("disk_bytes", ctypes.c_uint64),
        ("decoded_bytes", ctypes.c_uint64),
        ("l2_used_bytes", ctypes.c_uint64),
        ("l2_entries", ctypes.c_uint64),
        ("io_seconds", ctypes.c_double),
        ("decode_seconds", ctypes.c_double),
        ("verify_seconds", ctypes.c_double),
    ]


class WexError(RuntimeError):
    """Error raised when loading or operating on a libwex store."""


def _default_lib_names():
    if sys.platform == "win32":
        return ("libwex.dll", "wex.dll")
    if sys.platform == "darwin":
        return ("libwex.dylib",)
    return ("libwex.so", "libwex.so.1")


def discover_libwex_candidates(explicit_path=None):
    """Return ordered candidate paths/names to probe for libwex."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
        return candidates

    env_path = os.environ.get("WEX_LIB_PATH")
    if env_path:
        candidates.append(env_path)

    cli_dir = os.path.dirname(os.path.abspath(__file__))
    libwex_root = os.path.dirname(cli_dir)
    cwd = os.getcwd()
    names = _default_lib_names()

    search_dirs = [
        os.path.join(libwex_root, "build"),
        cli_dir,
        libwex_root,
        os.path.join(cwd, "build"),
        os.path.join(cwd, "libwex", "build"),
        cwd,
    ]
    for d in search_dirs:
        for name in names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                candidates.append(p)

    # Finally let the OS loader search PATH / LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
    for name in names:
        found = shutil.which(name)
        if found:
            candidates.append(found)
        candidates.append(name)

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


class WexLib:
    """Thin ctypes wrapper around libwex ABI v1."""

    def __init__(self, lib_path=None):
        self.path = None
        self._dll = None
        self._dll_dir_handle = None
        last_err = None

        for cand in discover_libwex_candidates(lib_path):
            try:
                dll_dir = os.path.dirname(os.path.abspath(cand)) if os.path.isfile(cand) else None
                if sys.platform == "win32" and dll_dir and hasattr(os, "add_dll_directory"):
                    self._dll_dir_handle = os.add_dll_directory(dll_dir)
                dll = ctypes.CDLL(cand)
                self._dll = dll
                self.path = cand
                break
            except OSError as exc:
                last_err = exc
                if self._dll_dir_handle is not None:
                    self._dll_dir_handle.close()
                    self._dll_dir_handle = None

        if self._dll is None:
            raise WexError(
                f"Unable to load libwex shared library (pass --lib or set WEX_LIB_PATH): {last_err}"
            )

        self._bind_symbols()
        abi = int(self._dll.wex_abi_version())
        if abi != WEX_STORE_ABI_VERSION:
            raise WexError(
                f"libwex ABI mismatch: expected {WEX_STORE_ABI_VERSION}, got {abi} ({self.path})"
            )

    def _bind_symbols(self):
        d = self._dll
        d.wex_abi_version.argtypes = []
        d.wex_abi_version.restype = ctypes.c_uint32

        d.wex_version_string.argtypes = []
        d.wex_version_string.restype = ctypes.c_char_p

        d.wex_open.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(WexOpenOpts),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        d.wex_open.restype = ctypes.c_void_p

        d.wex_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(WexInfo)]
        d.wex_info.restype = ctypes.c_int

        d.wex_has_expert.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        d.wex_has_expert.restype = ctypes.c_int

        d.wex_record_layout.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.POINTER(WexLayout),
        ]
        d.wex_record_layout.restype = ctypes.c_int

        d.wex_fetch_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int),
        ]
        d.wex_fetch_host.restype = ctypes.c_int

        d.wex_prefetch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        d.wex_prefetch.restype = ctypes.c_int

        d.wex_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(WexStats)]
        d.wex_stats.restype = None

        d.wex_close.argtypes = [ctypes.c_void_p]
        d.wex_close.restype = None

    def abi_version(self):
        return int(self._dll.wex_abi_version())

    def version_string(self):
        raw = self._dll.wex_version_string()
        return raw.decode("utf-8", errors="replace") if raw else ""

    def open_store(
        self,
        path,
        *,
        root_hex=None,
        verify=WEX_VERIFY_AUTO,
        mirrors=None,
        l2_cache_bytes=0,
        l2_pin_gpu=0,
        io_direct=-1,
        io_queue_depth=0,
        cpu_threads=0,
        gpu_device=-1,
        gpu_helper_path=None,
    ):
        opts = WexOpenOpts()
        opts.struct_size = ctypes.sizeof(WexOpenOpts)
        root_b = root_hex.encode("utf-8") if root_hex else None
        opts.root_hex = root_b
        opts.verify = int(verify)

        mirror_bs = [os.fsencode(m) for m in (mirrors or [])]
        mirror_arr = None
        if mirror_bs:
            mirror_arr = (ctypes.c_char_p * len(mirror_bs))(*mirror_bs)
            opts.mirrors = mirror_arr
            opts.num_mirrors = len(mirror_bs)
        else:
            opts.mirrors = None
            opts.num_mirrors = 0

        opts.l2_cache_bytes = int(l2_cache_bytes)
        opts.l2_pin_gpu = int(l2_pin_gpu)
        opts.io_direct = int(io_direct)
        opts.io_queue_depth = int(io_queue_depth)
        opts.cpu_threads = int(cpu_threads)
        opts.gpu_device = int(gpu_device)

        helper_b = os.fsencode(gpu_helper_path) if gpu_helper_path else None
        opts.gpu_helper_path = helper_b
        opts.disk_weights = None

        err_buf = ctypes.create_string_buffer(512)
        handle = self._dll.wex_open(os.fsencode(path), ctypes.byref(opts), err_buf, len(err_buf))
        if not handle:
            msg = err_buf.value.decode("utf-8", errors="replace") or "wex_open failed"
            raise WexError(msg)
        return WexStore(self, handle, path)


class WexStore:
    """Context-managed handle to an open wex_store_t."""

    def __init__(self, lib, handle, path):
        self.lib = lib
        self._handle = handle
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._handle:
            self.lib._dll.wex_close(self._handle)
            self._handle = None

    def info(self):
        out = WexInfo()
        rc = int(self.lib._dll.wex_info(self._handle, ctypes.byref(out)))
        if rc != WEX_OK:
            raise WexError(f"wex_info failed: {STATUS_NAMES.get(rc, rc)}")
        return info_to_dict(out)

    def stats(self):
        out = WexStats()
        self.lib._dll.wex_stats(self._handle, ctypes.byref(out))
        return stats_to_dict(out)

    def has_expert(self, layer, eid):
        return bool(self.lib._dll.wex_has_expert(self._handle, int(layer), int(eid)))

    def record_layout(self, layer, eid, flags=WEX_FETCH_DEFAULT):
        out = WexLayout()
        rc = int(
            self.lib._dll.wex_record_layout(
                self._handle, int(layer), int(eid), int(flags), ctypes.byref(out)
            )
        )
        if rc != WEX_OK:
            raise WexError(f"wex_record_layout failed: {STATUS_NAMES.get(rc, rc)}")
        tcount = min(int(out.tensor_count), WEX_MAX_RECORD_TENSORS)
        return {
            "record_index": int(out.record_index),
            "codec_kind": int(out.codec_kind),
            "codec_name": CODEC_NAMES.get(int(out.codec_kind), f"CODEC_{out.codec_kind}"),
            "flags": int(out.flags),
            "tensor_count": int(out.tensor_count),
            "file_offset": int(out.file_offset),
            "stored_bytes": int(out.stored_bytes),
            "decoded_bytes": int(out.decoded_bytes),
            "tensor_offsets": [int(out.tensor_offsets[i]) for i in range(tcount)],
            "tensor_bytes": [int(out.tensor_bytes[i]) for i in range(tcount)],
            "scale_offsets": [int(out.scale_offsets[i]) for i in range(3)],
            "weight_offsets": [int(out.weight_offsets[i]) for i in range(3)],
            "scale_shapes": [[int(out.scale_shapes[i][0]), int(out.scale_shapes[i][1])] for i in range(3)],
            "weight_shapes": [[int(out.weight_shapes[i][0]), int(out.weight_shapes[i][1])] for i in range(3)],
        }

    def fetch_host(self, layer, eids, dst_ptrs, dst_caps, flags=WEX_FETCH_DEFAULT):
        n = len(eids)
        eid_arr = (ctypes.c_int * n)(*[int(x) for x in eids])
        status_arr = (ctypes.c_int * n)()
        failed = int(
            self.lib._dll.wex_fetch_host(
                self._handle,
                int(layer),
                eid_arr,
                n,
                dst_ptrs,
                dst_caps,
                int(flags),
                status_arr,
            )
        )
        return failed, [int(status_arr[i]) for i in range(n)]

    def prefetch(self, layer, eids, flags=WEX_FETCH_DEFAULT):
        n = len(eids)
        if n <= 0:
            return 0
        eid_arr = (ctypes.c_int * n)(*[int(x) for x in eids])
        return int(self.lib._dll.wex_prefetch(self._handle, int(layer), eid_arr, n, int(flags)))


def info_to_dict(info):
    codec_kind = int(info.codec_kind)
    file_bytes = int(info.file_bytes)
    total_decoded = int(info.total_decoded_bytes)
    ratio = (total_decoded / file_bytes) if file_bytes > 0 else 0.0
    return {
        "abi_version": int(info.abi_version),
        "layers": int(info.layers),
        "experts": int(info.experts),
        "num_records": int(info.num_records),
        "num_tensors": int(info.num_tensors),
        "codec_kind": codec_kind,
        "codec_name": CODEC_NAMES.get(codec_kind, f"CODEC_{codec_kind}"),
        "value_exact": bool(info.value_exact),
        "verified_root": bool(info.verified_root),
        "verify_on": bool(info.verify_on),
        "num_replicas": int(info.num_replicas),
        "file_bytes": file_bytes,
        "total_decoded_bytes": total_decoded,
        "max_record_decoded_bytes": int(info.max_record_decoded_bytes),
        "max_record_stored_bytes": int(info.max_record_stored_bytes),
        "l2_capacity_bytes": int(info.l2_capacity_bytes),
        "compression_ratio": round(ratio, 4),
        "root_hex": info.root_hex.decode("ascii", errors="replace"),
        "backend_name": info.backend_name.decode("utf-8", errors="replace"),
        "io_engine": info.io_engine.decode("utf-8", errors="replace"),
    }


def stats_to_dict(st):
    from_l2 = int(st.records_from_l2)
    from_disk = int(st.records_from_disk)
    total_reads = from_l2 + from_disk
    hit_ratio = (from_l2 / total_reads) if total_reads > 0 else 0.0
    return {
        "fetches": int(st.fetches),
        "records_requested": int(st.records_requested),
        "records_from_l2": from_l2,
        "records_from_disk": from_disk,
        "l2_hit_ratio": round(hit_ratio, 6),
        "l2_hit_pct": round(hit_ratio * 100.0, 2),
        "gpu_decodes": int(st.gpu_decodes),
        "cpu_decodes": int(st.cpu_decodes),
        "prefetch_hits": int(st.prefetch_hits),
        "refused": int(st.refused),
        "disk_bytes": int(st.disk_bytes),
        "decoded_bytes": int(st.decoded_bytes),
        "l2_used_bytes": int(st.l2_used_bytes),
        "l2_entries": int(st.l2_entries),
        "io_seconds": float(st.io_seconds),
        "decode_seconds": float(st.decode_seconds),
        "verify_seconds": float(st.verify_seconds),
    }


def parse_l2_mb_value(raw):
    text = str(raw).strip().lower()
    if text == "auto":
        return UINT64_MAX, "auto"
    val = int(text)
    if val < 0:
        raise ValueError(f"L2 size must be >= 0 or 'auto', got {raw!r}")
    return val * 1024 * 1024, val


def _alloc_host_batch(count, buf_bytes):
    size = max(1, int(buf_bytes))
    bufs = [(ctypes.c_ubyte * size)() for _ in range(count)]
    ptrs = (ctypes.c_void_p * count)(*[ctypes.addressof(b) for b in bufs])
    caps = (ctypes.c_size_t * count)(*([size] * count))
    return bufs, ptrs, caps


# ---- High-level command operations (reused by wex_mcp.py) ------------------


def run_inspect(path, *, lib_path=None, cuda_path=None, cpu_only=False):
    lib = WexLib(lib_path)
    gpu_device = -2 if cpu_only else -1
    with lib.open_store(
        path,
        verify=WEX_VERIFY_AUTO,
        l2_cache_bytes=0,
        gpu_device=gpu_device,
        gpu_helper_path=cuda_path,
    ) as store:
        info = store.info()
        info["path"] = path
        info["version"] = lib.version_string()
        return info


def run_verify(path, *, root_hex=None, l2_mb=0, cpu_only=False, batch=4, lib_path=None, cuda_path=None):
    lib = WexLib(lib_path)
    l2_bytes, _ = parse_l2_mb_value(l2_mb)
    gpu_device = -2 if cpu_only else -1
    t0 = time.perf_counter()

    with lib.open_store(
        path,
        root_hex=root_hex,
        verify=WEX_VERIFY_ON,
        l2_cache_bytes=l2_bytes,
        gpu_device=gpu_device,
        gpu_helper_path=cuda_path,
    ) as store:
        info = store.info()
        nrecs = info["num_records"]
        max_dec = max(1, info["max_record_decoded_bytes"])
        # Keep scratch RAM bounded for very large records
        max_slots = max(1, min(int(batch), nrecs if nrecs > 0 else 1))
        if max_dec * max_slots > (256 << 20):
            max_slots = max(1, (256 << 20) // max_dec)

        bufs, ptrs, caps = _alloc_host_batch(max_slots, max_dec)

        verified = 0
        hash_errors = 0
        decode_errors = 0
        other_errors = 0
        failed_samples = []
        flags = WEX_FETCH_BY_RECORD_ID | WEX_FETCH_SCRATCH | WEX_FETCH_BYPASS_L2

        pos = 0
        while pos < nrecs:
            cur = min(max_slots, nrecs - pos)
            rids = list(range(pos, pos + cur))
            if cur == max_slots:
                cur_ptrs, cur_caps = ptrs, caps
            else:
                cur_ptrs = (ctypes.c_void_p * cur)(*[ctypes.addressof(bufs[i]) for i in range(cur)])
                cur_caps = (ctypes.c_size_t * cur)(*([max_dec] * cur))
            _, statuses = store.fetch_host(0, rids, cur_ptrs, cur_caps, flags)
            for rid, st in zip(rids, statuses):
                if st == WEX_OK:
                    verified += 1
                else:
                    if st in (WEX_E_HASH, WEX_E_REFUSED):
                        hash_errors += 1
                    elif st == WEX_E_DECODE:
                        decode_errors += 1
                    else:
                        other_errors += 1
                    if len(failed_samples) < 16:
                        failed_samples.append({
                            "record_index": rid,
                            "status": st,
                            "status_name": STATUS_NAMES.get(st, str(st)),
                        })
            pos += cur

        elapsed = max(1e-9, time.perf_counter() - t0)
        st = store.stats()
        eff_gib_s = (st["decoded_bytes"] / float(1 << 30)) / elapsed
        total_err = hash_errors + decode_errors + other_errors
        return {
            "ok": (total_err == 0 and verified == nrecs),
            "path": path,
            "root_hex": info["root_hex"],
            "verified_root": info["verified_root"],
            "records_total": nrecs,
            "records_verified": verified,
            "hash_errors": hash_errors,
            "decode_errors": decode_errors,
            "other_errors": other_errors,
            "failed_records": failed_samples,
            "decoded_bytes": st["decoded_bytes"],
            "disk_bytes": st["disk_bytes"],
            "elapsed_seconds": round(elapsed, 6),
            "effective_gib_s": round(eff_gib_s, 4),
            "backend_name": info["backend_name"],
            "io_engine": info["io_engine"],
        }


def _build_bench_waves(layers, experts, num_records, waves, batch, pattern, seed):
    rng = random.Random(seed)
    use_expert_grid = layers > 0 and experts > 0
    domain = experts if use_expert_grid else num_records
    if domain <= 0:
        raise WexError("Container has zero records to benchmark")

    k = max(1, min(int(batch), domain))
    plan = []

    if pattern == "zipf":
        # Per-layer deterministic rank permutation + Zipf power-law weights
        n_groups = layers if use_expert_grid else 1
        perms = []
        weights = [1.0 / ((r + 1.0) ** 1.05) for r in range(domain)]
        for _ in range(n_groups):
            p = list(range(domain))
            rng.shuffle(p)
            perms.append(p)
        for w in range(waves):
            layer = (w % layers) if use_expert_grid else 0
            perm = perms[layer if use_expert_grid else 0]
            chosen = []
            seen = set()
            tries = 0
            while len(chosen) < k and tries < k * 8:
                tries += 1
                idx = rng.choices(perm, weights=weights, k=1)[0]
                if idx not in seen:
                    seen.add(idx)
                    chosen.append(idx)
            if len(chosen) < k:
                for cand in perm:
                    if cand not in seen:
                        seen.add(cand)
                        chosen.append(cand)
                        if len(chosen) == k:
                            break
            plan.append((layer, chosen))
    elif pattern == "uniform":
        for w in range(waves):
            layer = (w % layers) if use_expert_grid else 0
            chosen = rng.sample(range(domain), k)
            plan.append((layer, chosen))
    elif pattern == "seq":
        cursor = 0
        for w in range(waves):
            layer = (w % layers) if use_expert_grid else 0
            chosen = [(cursor + i) % domain for i in range(k)]
            cursor = (cursor + k) % domain
            plan.append((layer, chosen))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return use_expert_grid, k, plan


def run_bench(
    path,
    *,
    l2_mb_list="0,256,1024",
    l2_pin=0,
    waves=64,
    warmup=0,
    batch=4,
    pattern="zipf",
    mirrors=None,
    verify=False,
    cpu_only=False,
    seed=42,
    lib_path=None,
    cuda_path=None,
):
    lib = WexLib(lib_path)
    if isinstance(l2_mb_list, str):
        l2_specs = [parse_l2_mb_value(tok) for tok in l2_mb_list.split(",") if tok.strip()]
    else:
        l2_specs = [parse_l2_mb_value(tok) for tok in l2_mb_list]
    if not l2_specs:
        l2_specs = [(0, 0)]

    gpu_device = -2 if cpu_only else -1
    verify_mode = WEX_VERIFY_ON if verify else WEX_VERIFY_OFF
    warmup_n = max(0, int(warmup))
    total_waves = warmup_n + int(waves)

    with lib.open_store(
        path,
        verify=WEX_VERIFY_OFF,
        l2_cache_bytes=0,
        gpu_device=-2,
        gpu_helper_path="0",
    ) as probe:
        pinfo = probe.info()

    use_grid, k, full_plan = _build_bench_waves(
        pinfo["layers"],
        pinfo["experts"],
        pinfo["num_records"],
        total_waves,
        int(batch),
        pattern,
        int(seed),
    )
    warm_plan = full_plan[:warmup_n]
    plan = full_plan[warmup_n:]

    max_dec = max(1, pinfo["max_record_decoded_bytes"])
    bufs, ptrs, caps = _alloc_host_batch(k, max_dec)
    fetch_flags = WEX_FETCH_SCRATCH | (0 if use_grid else WEX_FETCH_BY_RECORD_ID)

    results = []
    for l2_bytes, l2_label in l2_specs:
        with lib.open_store(
            path,
            verify=verify_mode,
            mirrors=mirrors,
            l2_cache_bytes=l2_bytes,
            l2_pin_gpu=int(l2_pin) if l2_bytes > 0 else 0,
            gpu_device=gpu_device,
            gpu_helper_path=cuda_path,
        ) as store:
            info = store.info()
            failures = 0
            for layer, ids in warm_plan:
                failed, _ = store.fetch_host(layer, ids, ptrs, caps, fetch_flags)
                failures += failed
            st_pre = store.stats()
            t0 = time.perf_counter()
            for layer, ids in plan:
                failed, _ = store.fetch_host(layer, ids, ptrs, caps, fetch_flags)
                failures += failed
            wall_s = max(1e-9, time.perf_counter() - t0)
            st = store.stats()

            req_d = st["records_requested"] - st_pre["records_requested"]
            l2_d = st["records_from_l2"] - st_pre["records_from_l2"]
            disk_d = st["records_from_disk"] - st_pre["records_from_disk"]
            tot_d = l2_d + disk_d
            hit_pct = round((100.0 * l2_d / tot_d) if tot_d > 0 else 0.0, 2)
            disk_b = st["disk_bytes"] - st_pre["disk_bytes"]
            dec_b = st["decoded_bytes"] - st_pre["decoded_bytes"]
            gpu_d = st["gpu_decodes"] - st_pre["gpu_decodes"]
            cpu_d = st["cpu_decodes"] - st_pre["cpu_decodes"]

            disk_mib = disk_b / float(1 << 20)
            decoded_mib = dec_b / float(1 << 20)
            eff_gb_s = (dec_b / 1e9) / wall_s
            avg_wave_ms = (wall_s * 1000.0) / max(1, len(plan))
            results.append({
                "l2_mb": l2_label,
                "l2_pin": int(l2_pin) if l2_bytes > 0 else 0,
                "l2_capacity_bytes": info["l2_capacity_bytes"],
                "warmup_waves": warmup_n,
                "waves": len(plan),
                "batch": k,
                "records_requested": req_d,
                "records_from_l2": l2_d,
                "records_from_disk": disk_d,
                "l2_hit_pct": hit_pct,
                "disk_mib": round(disk_mib, 3),
                "decoded_mib": round(decoded_mib, 3),
                "wall_seconds": round(wall_s, 6),
                "avg_wave_ms": round(avg_wave_ms, 4),
                "effective_decoded_gb_s": round(eff_gb_s, 4),
                "gpu_decodes": gpu_d,
                "cpu_decodes": cpu_d,
                "io_seconds": round(st["io_seconds"] - st_pre["io_seconds"], 6),
                "decode_seconds": round(st["decode_seconds"] - st_pre["decode_seconds"], 6),
                "verify_seconds": round(st["verify_seconds"] - st_pre["verify_seconds"], 6),
                "failures": failures,
                "backend_name": info["backend_name"],
                "io_engine": info["io_engine"],
                "num_replicas": info["num_replicas"],
            })

    # Keep bufs alive until end of sweep
    _ = bufs
    return {
        "path": path,
        "codec_name": pinfo["codec_name"],
        "layers": pinfo["layers"],
        "experts": pinfo["experts"],
        "num_records": pinfo["num_records"],
        "pattern": pattern,
        "waves": len(plan),
        "batch": k,
        "seed": int(seed),
        "verify": bool(verify),
        "mode": "expert_grid" if use_grid else "record_id",
        "results": results,
    }


def run_pull(
    repo_id,
    filename,
    *,
    out_dir=".",
    revision="main",
    root_hex=None,
    verify=False,
    cpu_only=False,
    lib_path=None,
    cuda_path=None,
    quiet=False,
):
    safe_name = os.path.basename(filename.replace("\\", "/"))
    if not safe_name or safe_name in (".", ".."):
        raise WexError(f"Invalid filename: {filename!r}")

    os.makedirs(out_dir, exist_ok=True)
    dest_path = os.path.join(out_dir, safe_name)
    part_path = dest_path + ".part"

    url = (
        "https://huggingface.co/"
        + urllib.parse.quote(repo_id, safe="/")
        + "/resolve/"
        + urllib.parse.quote(revision, safe="")
        + "/"
        + urllib.parse.quote(filename.lstrip("/"), safe="/")
    )

    existing = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
    headers = {"User-Agent": "libwex-cli/0.1"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = "Bearer " + hf_token
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    t0 = time.perf_counter()
    resumed = False
    downloaded = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", 200)
            if status == 206 and existing > 0:
                mode = "ab"
                resumed = True
            else:
                mode = "wb"
                existing = 0

            with open(part_path, mode) as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing > 0:
            # Partial file already complete
            resumed = True
        else:
            raise WexError(f"Download failed (HTTP {exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise WexError(f"Download failed: {exc.reason}") from exc

    os.replace(part_path, dest_path)
    total_bytes = os.path.getsize(dest_path)
    elapsed = max(1e-9, time.perf_counter() - t0)

    result = {
        "url": url,
        "dest_path": dest_path,
        "total_bytes": total_bytes,
        "downloaded_bytes": downloaded,
        "resumed": resumed,
        "elapsed_seconds": round(elapsed, 4),
    }

    if verify or root_hex:
        v_res = run_verify(
            dest_path,
            root_hex=root_hex,
            l2_mb=0,
            cpu_only=cpu_only,
            lib_path=lib_path,
            cuda_path=cuda_path,
        )
        result["verify"] = v_res
        if not v_res["ok"]:
            raise WexError(f"Verification failed after pull for {dest_path}")

    if not quiet:
        pass
    return result


# ---- HTTP Server (serve) ---------------------------------------------------


def format_prometheus_metrics(info, stats):
    lines = [
        "# HELP wex_fetches_total Total number of wex_fetch calls.",
        "# TYPE wex_fetches_total counter",
        f"wex_fetches_total {stats['fetches']}",
        "# HELP wex_l2_hits_total Total records served from L2 compressed-RAM cache.",
        "# TYPE wex_l2_hits_total counter",
        f"wex_l2_hits_total {stats['records_from_l2']}",
        "# HELP wex_disk_reads_total Total records read from NVMe/disk.",
        "# TYPE wex_disk_reads_total counter",
        f"wex_disk_reads_total {stats['records_from_disk']}",
        "# HELP wex_l2_hit_ratio Ratio of record reads served from L2 cache.",
        "# TYPE wex_l2_hit_ratio gauge",
        f"wex_l2_hit_ratio {stats['l2_hit_ratio']:.6f}",
        "# HELP wex_disk_bytes_total Total compressed bytes read from disk.",
        "# TYPE wex_disk_bytes_total counter",
        f"wex_disk_bytes_total {stats['disk_bytes']}",
        "# HELP wex_decoded_bytes_total Total decoded weight bytes produced.",
        "# TYPE wex_decoded_bytes_total counter",
        f"wex_decoded_bytes_total {stats['decoded_bytes']}",
        "# HELP wex_gpu_decodes_total Total records decoded on GPU.",
        "# TYPE wex_gpu_decodes_total counter",
        f"wex_gpu_decodes_total {stats['gpu_decodes']}",
        "# HELP wex_cpu_decodes_total Total records decoded on CPU.",
        "# TYPE wex_cpu_decodes_total counter",
        f"wex_cpu_decodes_total {stats['cpu_decodes']}",
        "# HELP wex_io_seconds_total Cumulative disk I/O time in seconds.",
        "# TYPE wex_io_seconds_total counter",
        f"wex_io_seconds_total {stats['io_seconds']:.6f}",
        "# HELP wex_decode_seconds_total Cumulative decode time in seconds.",
        "# TYPE wex_decode_seconds_total counter",
        f"wex_decode_seconds_total {stats['decode_seconds']:.6f}",
        "# HELP wex_verify_seconds_total Cumulative SHA-256 verification time in seconds.",
        "# TYPE wex_verify_seconds_total counter",
        f"wex_verify_seconds_total {stats['verify_seconds']:.6f}",
        "# HELP wex_l2_used_bytes Current bytes occupied in L2 cache.",
        "# TYPE wex_l2_used_bytes gauge",
        f"wex_l2_used_bytes {stats['l2_used_bytes']}",
        "# HELP wex_l2_capacity_bytes Configured capacity of L2 cache in bytes.",
        "# TYPE wex_l2_capacity_bytes gauge",
        f"wex_l2_capacity_bytes {info['l2_capacity_bytes']}",
        "",
    ]
    return "\n".join(lines)


class WexHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, *, store, model_id, upstream=None):
        self.wex_store = store
        self.wex_model_id = model_id
        self.wex_upstream = upstream
        self.wex_started_at = time.time()
        self.wex_version = store.lib.version_string()
        super().__init__(server_address, handler_cls)


class WexRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Quiet by default to keep stdout/stderr clean in automated runs
        if os.environ.get("WEX_SERVE_LOG"):
            sys.stderr.write(f"[wex-serve] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, status, payload, extra_headers=None):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status, text, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        clen = self.headers.get("Content-Length")
        if not clen:
            return b""
        n = int(clen)
        if n < 0 or n > (64 << 20):
            raise ValueError("Invalid Content-Length")
        return self.rfile.read(n)

    def _hit_rate_header(self):
        st = self.server.wex_store.stats()
        return {"X-Wex-L2-Hit-Rate": f"{st['l2_hit_ratio']:.4f}"}

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        route = parsed.path
        store = self.server.wex_store

        if route == "/health":
            info = store.info()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "abi_version": info["abi_version"],
                    "version": self.server.wex_version,
                    "backend": info["backend_name"],
                    "io_engine": info["io_engine"],
                },
            )
            return

        if route == "/status":
            info = store.info()
            stats = store.stats()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model_id": self.server.wex_model_id,
                    "path": store.path,
                    "uptime_seconds": round(time.time() - self.server.wex_started_at, 3),
                    "upstream": self.server.wex_upstream,
                    "info": info,
                    "stats": stats,
                    "l2_hit_ratio": stats["l2_hit_ratio"],
                },
            )
            return

        if route == "/v1/models":
            info = store.info()
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.wex_model_id,
                            "object": "model",
                            "owned_by": "libwex",
                            "wex": {
                                "abi_version": info["abi_version"],
                                "codec_kind": info["codec_kind"],
                                "codec_name": info["codec_name"],
                                "layers": info["layers"],
                                "experts": info["experts"],
                                "num_records": info["num_records"],
                                "num_tensors": info["num_tensors"],
                                "value_exact": info["value_exact"],
                                "root_hex": info["root_hex"],
                                "file_bytes": info["file_bytes"],
                                "total_decoded_bytes": info["total_decoded_bytes"],
                                "compression_ratio": info["compression_ratio"],
                                "backend_name": info["backend_name"],
                                "io_engine": info["io_engine"],
                            },
                        }
                    ],
                },
            )
            return

        if route == "/metrics":
            info = store.info()
            stats = store.stats()
            text = format_prometheus_metrics(info, stats)
            self._send_text(200, text, "text/plain; version=0.0.4; charset=utf-8")
            return

        if route == "/v1/telemetry":
            info = store.info()
            stats = store.stats()
            uptime = max(1e-9, time.time() - self.server.wex_started_at)
            cap = info["l2_capacity_bytes"]
            used = stats["l2_used_bytes"]
            util = (used / cap) if cap > 0 else 0.0
            active_s = stats["io_seconds"] + stats["decode_seconds"]
            eff_gb_s = (stats["decoded_bytes"] / 1e9 / active_s) if active_s > 1e-9 else 0.0
            self._send_json(
                200,
                {
                    "timestamp": round(time.time(), 3),
                    "uptime_seconds": round(uptime, 3),
                    "model_id": self.server.wex_model_id,
                    "store": info,
                    "counters": stats,
                    "l2": {
                        "used_bytes": used,
                        "capacity_bytes": cap,
                        "entries": stats["l2_entries"],
                        "utilization_ratio": round(util, 6),
                        "hit_ratio": stats["l2_hit_ratio"],
                        "hit_pct": stats["l2_hit_pct"],
                    },
                    "throughput": {
                        "effective_decoded_gb_s": round(eff_gb_s, 4),
                    },
                },
            )
            return

        self._send_json(404, {"error": {"type": "not_found", "message": f"Unknown path: {route}"}})

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        route = parsed.path

        if route == "/v1/prefetch":
            try:
                raw = self._read_body()
                req = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(req, dict):
                    raise ValueError("JSON body must be an object")
                layer = int(req.get("layer", 0))
                eids = req.get("eids", [])
                if not isinstance(eids, list) or not all(isinstance(x, int) for x in eids):
                    raise ValueError("'eids' must be a list of integers")
                flags = WEX_FETCH_BY_RECORD_ID if req.get("by_record_id") else WEX_FETCH_DEFAULT
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": {"type": "bad_request", "message": str(exc)}})
                return

            queued = self.server.wex_store.prefetch(layer, eids, flags)
            self._send_json(
                200,
                {
                    "ok": True,
                    "layer": layer,
                    "requested": len(eids),
                    "queued": queued,
                },
                extra_headers=self._hit_rate_header(),
            )
            return

        if route in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            self._handle_inference_proxy()
            return

        self._send_json(404, {"error": {"type": "not_found", "message": f"Unknown path: {route}"}})

    def _handle_inference_proxy(self):
        upstream = self.server.wex_upstream
        hit_hdr = self._hit_rate_header()
        if not upstream:
            self._send_json(
                503,
                {
                    "error": {
                        "type": "upstream_not_configured",
                        "message": (
                            "No upstream inference engine configured. "
                            "Restart wex serve with --upstream http://127.0.0.1:<port> "
                            "to proxy OpenAI, Responses, or Messages API endpoints."
                        ),
                    }
                },
                extra_headers=hit_hdr,
            )
            return

        try:
            body = self._read_body()
        except ValueError as exc:
            self._send_json(400, {"error": {"type": "bad_request", "message": str(exc)}})
            return

        up = urllib.parse.urlsplit(upstream)
        if up.scheme not in ("http", "https") or not up.hostname:
            self._send_json(
                502,
                {"error": {"type": "invalid_upstream", "message": f"Invalid upstream URL: {upstream}"}},
                extra_headers=hit_hdr,
            )
            return

        target_path = up.path.rstrip("/") + self.path
        fwd_headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in HOP_BY_HOP_HEADERS or lk == "host":
                continue
            fwd_headers[k] = v
        fwd_headers["Content-Length"] = str(len(body))

        conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
        port = up.port or (443 if up.scheme == "https" else 80)
        conn = conn_cls(up.hostname, port, timeout=120)
        try:
            conn.request("POST", target_path, body=body, headers=fwd_headers)
            resp = conn.getresponse()

            self.send_response(resp.status, resp.reason)
            is_sse = False
            has_clen = False
            for k, v in resp.getheaders():
                lk = k.lower()
                if lk in HOP_BY_HOP_HEADERS or lk == "x-wex-l2-hit-rate":
                    continue
                if lk == "content-type" and "text/event-stream" in v.lower():
                    is_sse = True
                if lk == "content-length":
                    has_clen = True
                self.send_header(k, v)
            for k, v in hit_hdr.items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()

            if is_sse or not has_clen:
                while True:
                    chunk = resp.readline() if is_sse else resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                self.wfile.flush()
        except OSError as exc:
            self._send_json(
                502,
                {"error": {"type": "upstream_error", "message": f"Upstream connection failed: {exc}"}},
                extra_headers=hit_hdr,
            )
        finally:
            conn.close()


def create_wex_http_server(
    path,
    *,
    host="127.0.0.1",
    port=8080,
    l2_mb="auto",
    l2_pin=0,
    verify="auto",
    root_hex=None,
    mirrors=None,
    cpu_only=False,
    upstream=None,
    lib_path=None,
    cuda_path=None,
):
    lib = WexLib(lib_path)
    l2_bytes, _ = parse_l2_mb_value(l2_mb)
    verify_mode = VERIFY_MODE_MAP[verify]
    gpu_device = -2 if cpu_only else -1

    store = lib.open_store(
        path,
        root_hex=root_hex,
        verify=verify_mode,
        mirrors=mirrors,
        l2_cache_bytes=l2_bytes,
        l2_pin_gpu=int(l2_pin),
        gpu_device=gpu_device,
        gpu_helper_path=cuda_path,
    )
    model_id = os.path.splitext(os.path.basename(path))[0]
    try:
        server = WexHTTPServer(
            (host, int(port)),
            WexRequestHandler,
            store=store,
            model_id=model_id,
            upstream=upstream,
        )
    except Exception:
        store.close()
        raise
    return server, store


# ---- Newline-delimited JSON IPC (ipc) --------------------------------------


def run_ipc_loop(store, stdin_stream=None, stdout_stream=None):
    inp = stdin_stream if stdin_stream is not None else sys.stdin
    out = stdout_stream if stdout_stream is not None else sys.stdout

    for raw_line in inp:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if not isinstance(msg, dict):
                raise ValueError("IPC message must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            resp = {"ok": False, "error": f"invalid JSON: {exc}"}
            out.write(json.dumps(resp, separators=(",", ":")) + "\n")
            out.flush()
            continue

        cmd = msg.get("cmd")
        try:
            if cmd == "info":
                resp = {"ok": True, "info": store.info()}
            elif cmd == "stats":
                resp = {"ok": True, "stats": store.stats()}
            elif cmd == "has_expert":
                layer = int(msg["layer"])
                eid = int(msg["eid"])
                resp = {
                    "ok": True,
                    "layer": layer,
                    "eid": eid,
                    "present": store.has_expert(layer, eid),
                }
            elif cmd == "prefetch":
                layer = int(msg.get("layer", 0))
                eids = msg.get("eids", [])
                if not isinstance(eids, list) or not all(isinstance(x, int) for x in eids):
                    raise ValueError("'eids' must be a list of integers")
                flags = WEX_FETCH_BY_RECORD_ID if msg.get("by_record_id") else WEX_FETCH_DEFAULT
                queued = store.prefetch(layer, eids, flags)
                resp = {"ok": True, "layer": layer, "queued": queued}
            elif cmd == "layout":
                layer = int(msg.get("layer", 0))
                eid = int(msg["eid"])
                flags = WEX_FETCH_BY_RECORD_ID if msg.get("by_record_id") else WEX_FETCH_DEFAULT
                resp = {"ok": True, "layout": store.record_layout(layer, eid, flags)}
            elif cmd == "quit":
                resp = {"ok": True, "bye": True}
                out.write(json.dumps(resp, separators=(",", ":")) + "\n")
                out.flush()
                break
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except (KeyError, TypeError, ValueError, WexError) as exc:
            resp = {"ok": False, "error": str(exc)}

        out.write(json.dumps(resp, separators=(",", ":")) + "\n")
        out.flush()


# ---- Formatting helpers ----------------------------------------------------


def _fmt_bytes(n):
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            return f"{val:.2f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024.0
    return f"{n} B"


def _print_inspect_text(info):
    rows = [
        ("Path", info["path"]),
        ("Library", f"{info['version']} (ABI v{info['abi_version']})"),
        ("Codec", f"{info['codec_name']} (kind={info['codec_kind']}, value_exact={int(info['value_exact'])})"),
        ("Layout", f"layers={info['layers']}, experts={info['experts']}, records={info['num_records']}, tensors={info['num_tensors']}"),
        ("Root SHA-256", info["root_hex"]),
        ("File Size", f"{_fmt_bytes(info['file_bytes'])} ({info['file_bytes']} B)"),
        ("Decoded Size", f"{_fmt_bytes(info['total_decoded_bytes'])} ({info['total_decoded_bytes']} B)"),
        ("Compression", f"{info['compression_ratio']:.3f}x"),
        ("Max Record", f"stored={_fmt_bytes(info['max_record_stored_bytes'])}, decoded={_fmt_bytes(info['max_record_decoded_bytes'])}"),
        ("I/O Engine", f"{info['io_engine']} (replicas={info['num_replicas']})"),
        ("Backend", info["backend_name"]),
    ]
    for k, v in rows:
        print(f"{k:<14}: {v}")


def _print_verify_text(res):
    status = "OK" if res["ok"] else "FAILED"
    print(f"Verify        : {status}")
    print(f"Path          : {res['path']}")
    print(f"Root SHA-256  : {res['root_hex']} (pinned={int(res['verified_root'])})")
    print(f"Records       : {res['records_verified']} / {res['records_total']} verified")
    print(f"Errors        : hash={res['hash_errors']}, decode={res['decode_errors']}, other={res['other_errors']}")
    print(f"Elapsed       : {res['elapsed_seconds']:.4f} s ({res['effective_gib_s']:.3f} GiB/s decoded)")
    if res["failed_records"]:
        for item in res["failed_records"]:
            print(f"  record {item['record_index']}: {item['status_name']} ({item['status']})")


def _print_bench_text(res):
    print(
        f"Benchmark: {res['path']} | codec={res['codec_name']} | "
        f"pattern={res['pattern']} | waves={res['waves']} | batch={res['batch']} | verify={int(res['verify'])}"
    )
    hdr = (
        f"{'L2(MiB)':>8} {'Hit%':>7} {'Disk(MiB)':>10} {'Dec(MiB)':>10} "
        f"{'Wall(s)':>9} {'Wave(ms)':>9} {'Eff(GB/s)':>10} {'GPU/CPU':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in res["results"]:
        l2_str = str(r["l2_mb"])
        dec_split = f"{r['gpu_decodes']}/{r['cpu_decodes']}"
        print(
            f"{l2_str:>8} {r['l2_hit_pct']:>6.2f}% {r['disk_mib']:>10.2f} {r['decoded_mib']:>10.2f} "
            f"{r['wall_seconds']:>9.4f} {r['avg_wave_ms']:>9.3f} {r['effective_decoded_gb_s']:>10.3f} {dec_split:>11}"
        )


def _print_pull_text(res):
    print(f"Pulled        : {res['dest_path']}")
    print(f"Source        : {res['url']}")
    print(f"Size          : {_fmt_bytes(res['total_bytes'])} (resumed={int(res['resumed'])}, elapsed={res['elapsed_seconds']:.2f}s)")
    if "verify" in res:
        _print_verify_text(res["verify"])


# ---- CLI Parser & Entrypoint -----------------------------------------------


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", choices=("text", "json"), default=argparse.SUPPRESS,
                        help="Output format (text or json)")
    common.add_argument("--lib", dest="lib_path", default=argparse.SUPPRESS,
                        help="Explicit path to libwex shared library")
    common.add_argument("--cuda", dest="cuda_path", default=argparse.SUPPRESS,
                        help="Explicit path to libwex_cuda shared library")

    parser = argparse.ArgumentParser(
        prog="wex_cli.py",
        description="libwex CLI, benchmark, puller, HTTP telemetry server, and JSON IPC tool",
    )
    parser.add_argument("--output", choices=("text", "json"), default="text",
                        help="Output format (text or json)")
    parser.add_argument("--lib", dest="lib_path", default=None,
                        help="Explicit path to libwex shared library")
    parser.add_argument("--cuda", dest="cuda_path", default=None,
                        help="Explicit path to libwex_cuda shared library")

    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_ins = sub.add_parser("inspect", parents=[common], help="Inspect a .wex container header and metadata")
    p_ins.add_argument("file", help="Path to .wex / .wex4 container")
    p_ins.add_argument("--cpu-only", action="store_true", help="Disable GPU backend discovery")

    # verify
    p_ver = sub.add_parser("verify", parents=[common], help="Stream and SHA-256 verify all records")
    p_ver.add_argument("file", help="Path to .wex / .wex4 container")
    p_ver.add_argument("--root", dest="root_hex", default=None, help="Pin 64-char hex root SHA-256")
    p_ver.add_argument("--l2-mb", default="0", help="L2 compressed-RAM cache size in MiB (default: 0)")
    p_ver.add_argument("--batch", type=int, default=4, help="Records per verification batch (default: 4)")
    p_ver.add_argument("--cpu-only", action="store_true", help="Force CPU-only decode")

    # bench
    p_bnc = sub.add_parser("bench", parents=[common], help="Benchmark single-NVMe + L2 cache sweep")
    p_bnc.add_argument("file", help="Path to .wex / .wex4 container")
    p_bnc.add_argument("--l2-mb", default="0,256,1024", help="Comma-separated L2 sizes in MiB or 'auto'")
    p_bnc.add_argument("--l2-pin", type=int, choices=(0, 1), default=0, help="Page-lock L2 arena for GPU H2D (0|1)")
    p_bnc.add_argument("--waves", type=int, default=64, help="Number of timed fetch waves (default: 64)")
    p_bnc.add_argument("--warmup", type=int, default=0, help="Untimed warmup waves before measurement (default: 0)")
    p_bnc.add_argument("--batch", type=int, default=4, help="Records per wave (default: 4)")
    p_bnc.add_argument("--pattern", choices=("zipf", "uniform", "seq"), default="zipf",
                       help="Expert access pattern (default: zipf)")
    p_bnc.add_argument("--mirror", action="append", default=[], help="Optional replica path (repeatable)")
    p_bnc.add_argument("--verify", type=int, choices=(0, 1), default=0, help="Enable SHA-256 verify (0|1)")
    p_bnc.add_argument("--cpu-only", action="store_true", help="Force CPU-only decode")
    p_bnc.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed")

    # pull
    p_pul = sub.add_parser("pull", parents=[common], help="Download a .wex file from HuggingFace")
    p_pul.add_argument("repo_id", help="HuggingFace repo id (e.g. org/model)")
    p_pul.add_argument("filename", help="File path inside the repo")
    p_pul.add_argument("--out", dest="out_dir", default=".", help="Output directory (default: .)")
    p_pul.add_argument("--revision", default="main", help="Git revision/branch (default: main)")
    p_pul.add_argument("--root", dest="root_hex", default=None, help="Pin 64-char hex root SHA-256")
    p_pul.add_argument("--verify", action="store_true", help="Verify all records after download")
    p_pul.add_argument("--cpu-only", action="store_true", help="Force CPU-only decode when verifying")

    # serve
    p_srv = sub.add_parser("serve", parents=[common], help="Start threaded HTTP telemetry/prefetch/proxy server")
    p_srv.add_argument("file", help="Path to .wex / .wex4 container")
    p_srv.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_srv.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080, 0=ephemeral)")
    p_srv.add_argument("--l2-mb", default="auto", help="L2 cache size in MiB or 'auto'")
    p_srv.add_argument("--l2-pin", type=int, choices=(0, 1), default=0, help="Page-lock L2 arena for GPU (0|1)")
    p_srv.add_argument("--verify", choices=("auto", "on", "off"), default="auto", help="Verification mode")
    p_srv.add_argument("--root", dest="root_hex", default=None, help="Pin 64-char hex root SHA-256")
    p_srv.add_argument("--mirror", action="append", default=[], help="Optional replica path (repeatable)")
    p_srv.add_argument("--cpu-only", action="store_true", help="Force CPU-only decode")
    p_srv.add_argument("--upstream", default=None, help="Optional upstream inference URL (e.g. http://127.0.0.1:8081)")

    # ipc
    p_ipc = sub.add_parser("ipc", parents=[common], help="Run newline-delimited JSON control loop on stdin/stdout")
    p_ipc.add_argument("file", help="Path to .wex / .wex4 container")
    p_ipc.add_argument("--l2-mb", default="auto", help="L2 cache size in MiB or 'auto'")
    p_ipc.add_argument("--l2-pin", type=int, choices=(0, 1), default=0, help="Page-lock L2 arena for GPU (0|1)")
    p_ipc.add_argument("--verify", choices=("auto", "on", "off"), default="auto", help="Verification mode")
    p_ipc.add_argument("--root", dest="root_hex", default=None, help="Pin 64-char hex root SHA-256")
    p_ipc.add_argument("--mirror", action="append", default=[], help="Optional replica path (repeatable)")
    p_ipc.add_argument("--cpu-only", action="store_true", help="Force CPU-only decode")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    as_json = args.output == "json"

    try:
        if args.command == "inspect":
            res = run_inspect(
                args.file,
                lib_path=args.lib_path,
                cuda_path=args.cuda_path,
                cpu_only=args.cpu_only,
            )
            if as_json:
                print(json.dumps(res, indent=2))
            else:
                _print_inspect_text(res)
            return 0

        if args.command == "verify":
            res = run_verify(
                args.file,
                root_hex=args.root_hex,
                l2_mb=args.l2_mb,
                cpu_only=args.cpu_only,
                batch=args.batch,
                lib_path=args.lib_path,
                cuda_path=args.cuda_path,
            )
            if as_json:
                print(json.dumps(res, indent=2))
            else:
                _print_verify_text(res)
            return 0 if res["ok"] else 1

        if args.command == "bench":
            res = run_bench(
                args.file,
                l2_mb_list=args.l2_mb,
                l2_pin=args.l2_pin,
                waves=args.waves,
                warmup=args.warmup,
                batch=args.batch,
                pattern=args.pattern,
                mirrors=args.mirror,
                verify=bool(args.verify),
                cpu_only=args.cpu_only,
                seed=args.seed,
                lib_path=args.lib_path,
                cuda_path=args.cuda_path,
            )
            if as_json:
                print(json.dumps(res, indent=2))
            else:
                _print_bench_text(res)
            total_fail = sum(r.get("failures", 0) for r in res["results"])
            if total_fail > 0:
                if not as_json:
                    sys.stderr.write(f"error: {total_fail} record fetch failure(s) during benchmark\n")
                return 1
            return 0

        if args.command == "pull":
            res = run_pull(
                args.repo_id,
                args.filename,
                out_dir=args.out_dir,
                revision=args.revision,
                root_hex=args.root_hex,
                verify=args.verify,
                cpu_only=args.cpu_only,
                lib_path=args.lib_path,
                cuda_path=args.cuda_path,
                quiet=as_json,
            )
            if as_json:
                print(json.dumps(res, indent=2))
            else:
                _print_pull_text(res)
            return 0

        if args.command == "serve":
            server, store = create_wex_http_server(
                args.file,
                host=args.host,
                port=args.port,
                l2_mb=args.l2_mb,
                l2_pin=args.l2_pin,
                verify=args.verify,
                root_hex=args.root_hex,
                mirrors=args.mirror,
                cpu_only=args.cpu_only,
                upstream=args.upstream,
                lib_path=args.lib_path,
                cuda_path=args.cuda_path,
            )
            bind_host, bind_port = server.server_address[:2]
            ready = {
                "event": "ready",
                "host": bind_host,
                "port": bind_port,
                "url": f"http://{bind_host}:{bind_port}",
                "model_id": server.wex_model_id,
            }
            if as_json:
                print(json.dumps(ready, separators=(",", ":")), flush=True)
            else:
                print(f"libwex server listening on http://{bind_host}:{bind_port} ({args.file})", flush=True)
            try:
                server.serve_forever(poll_interval=0.25)
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                store.close()
            return 0

        if args.command == "ipc":
            lib = WexLib(args.lib_path)
            l2_bytes, _ = parse_l2_mb_value(args.l2_mb)
            gpu_device = -2 if args.cpu_only else -1
            with lib.open_store(
                args.file,
                root_hex=args.root_hex,
                verify=VERIFY_MODE_MAP[args.verify],
                mirrors=args.mirror,
                l2_cache_bytes=l2_bytes,
                l2_pin_gpu=int(args.l2_pin),
                gpu_device=gpu_device,
                gpu_helper_path=args.cuda_path,
            ) as store:
                run_ipc_loop(store)
            return 0

    except (WexError, ValueError, OSError) as exc:
        if as_json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            sys.stderr.write(f"error: {exc}\n")
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
