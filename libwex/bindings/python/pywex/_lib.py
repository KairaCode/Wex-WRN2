# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Pure-stdlib ctypes bindings for libwex C99 ABI v1 (wex_store.h)."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---- ABI constants matching libwex/include/wex_store.h ---------------------

WEX_STORE_ABI_VERSION = 1
WEX_MAX_RECORD_TENSORS = 16
UINT64_MAX = (1 << 64) - 1

# Error / status codes
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

ERROR_NAMES: Dict[int, str] = {
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

# Verification modes (wex_verify_mode_t)
WEX_VERIFY_AUTO = 0
WEX_VERIFY_OFF = 1
WEX_VERIFY_ON = 2

# Codec kinds (wex_codec_kind_t)
WEX_CODEC_RAW = 0
WEX_CODEC_WNR3 = 5
WEX_CODEC_WBP1 = 6
WEX_CODEC_WNR4_MX = 8
WEX_CODEC_WNR4_NV = 9
WEX_CODEC_WNR4_GG = 10
WEX_CODEC_WNR4_I4 = 11

CODEC_NAMES: Dict[int, str] = {
    WEX_CODEC_RAW: "RAW",
    WEX_CODEC_WNR3: "WNR3",
    WEX_CODEC_WBP1: "WBP1",
    WEX_CODEC_WNR4_MX: "WNR4-MX",
    WEX_CODEC_WNR4_NV: "WNR4-NV",
    WEX_CODEC_WNR4_GG: "WNR4-GG",
    WEX_CODEC_WNR4_I4: "WNR4-I4",
}

# Fetch flags
WEX_FETCH_DEFAULT = 0
WEX_FETCH_SCRATCH = 1 << 0
WEX_FETCH_BYPASS_L2 = 1 << 1
WEX_FETCH_BY_RECORD_ID = 1 << 2
WEX_FETCH_EXPAND_E8M0_F32 = 1 << 3

# WEX4 record flags (mirrored from wex4.h for layout helpers)
WEX_REC_VALUE_EXACT = 1 << 0
WEX_REC_EXPERT = 1 << 1


# ---- Error type ------------------------------------------------------------


class WexError(RuntimeError):
    """Typed exception raised when libwex reports a non-zero error status."""

    def __init__(self, code: int, message: str = "") -> None:
        self.code = int(code)
        name = ERROR_NAMES.get(self.code, f"WEX_E_{self.code}")
        self.message = message or name
        super().__init__(f"[{name} ({self.code})] {self.message}")


# ---- C ABI v1 ctypes.Structure definitions ---------------------------------


class _CWexOpenOpts(ctypes.Structure):
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


class _CWexInfo(ctypes.Structure):
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


class _CWexLayout(ctypes.Structure):
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


class _CWexStats(ctypes.Structure):
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


# ---- Pythonic dataclasses --------------------------------------------------


@dataclass(frozen=True)
class WexInfo:
    """Container metadata, active backend, and verification state."""

    abi_version: int
    layers: int
    experts: int
    num_records: int
    num_tensors: int
    codec_kind: int
    value_exact: bool
    verified_root: bool
    verify_on: bool
    num_replicas: int
    file_bytes: int
    total_decoded_bytes: int
    max_record_decoded_bytes: int
    max_record_stored_bytes: int
    l2_capacity_bytes: int
    root_hex: str
    backend_name: str
    io_engine: str

    @property
    def codec_name(self) -> str:
        return CODEC_NAMES.get(self.codec_kind, f"UNKNOWN({self.codec_kind})")

    @property
    def compression_ratio(self) -> float:
        if self.file_bytes <= 0:
            return 0.0
        return float(self.total_decoded_bytes) / float(self.file_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "layers": self.layers,
            "experts": self.experts,
            "num_records": self.num_records,
            "num_tensors": self.num_tensors,
            "codec_kind": self.codec_kind,
            "codec_name": self.codec_name,
            "value_exact": self.value_exact,
            "verified_root": self.verified_root,
            "verify_on": self.verify_on,
            "num_replicas": self.num_replicas,
            "file_bytes": self.file_bytes,
            "total_decoded_bytes": self.total_decoded_bytes,
            "max_record_decoded_bytes": self.max_record_decoded_bytes,
            "max_record_stored_bytes": self.max_record_stored_bytes,
            "l2_capacity_bytes": self.l2_capacity_bytes,
            "compression_ratio": round(self.compression_ratio, 6),
            "root_hex": self.root_hex,
            "backend_name": self.backend_name,
            "io_engine": self.io_engine,
        }


@dataclass(frozen=True)
class WexLayout:
    """Decoded slab and tensor layout for a single record or expert."""

    record_index: int
    codec_kind: int
    flags: int
    tensor_count: int
    file_offset: int
    stored_bytes: int
    decoded_bytes: int
    tensor_offsets: Tuple[int, ...]
    tensor_bytes: Tuple[int, ...]
    scale_offsets: Tuple[int, int, int]
    weight_offsets: Tuple[int, int, int]
    scale_shapes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    weight_shapes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]

    @property
    def codec_name(self) -> str:
        return CODEC_NAMES.get(self.codec_kind, f"UNKNOWN({self.codec_kind})")

    @property
    def is_expert(self) -> bool:
        return (
            bool(self.flags & WEX_REC_EXPERT)
            and self.tensor_count == 6
            and self.scale_shapes[0][0] > 0
            and self.scale_shapes[0][1] > 0
        )

    @property
    def value_exact(self) -> bool:
        return bool(self.flags & WEX_REC_VALUE_EXACT)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_index": self.record_index,
            "codec_kind": self.codec_kind,
            "codec_name": self.codec_name,
            "flags": self.flags,
            "is_expert": self.is_expert,
            "value_exact": self.value_exact,
            "tensor_count": self.tensor_count,
            "file_offset": self.file_offset,
            "stored_bytes": self.stored_bytes,
            "decoded_bytes": self.decoded_bytes,
            "tensor_offsets": list(self.tensor_offsets),
            "tensor_bytes": list(self.tensor_bytes),
            "scale_offsets": list(self.scale_offsets),
            "weight_offsets": list(self.weight_offsets),
            "scale_shapes": [list(s) for s in self.scale_shapes],
            "weight_shapes": [list(w) for w in self.weight_shapes],
        }


@dataclass(frozen=True)
class WexStats:
    """Live I/O, L2 cache, decode, and integrity verification telemetry."""

    fetches: int
    records_requested: int
    records_from_l2: int
    records_from_disk: int
    gpu_decodes: int
    cpu_decodes: int
    prefetch_hits: int
    refused: int
    disk_bytes: int
    decoded_bytes: int
    l2_used_bytes: int
    l2_entries: int
    io_seconds: float
    decode_seconds: float
    verify_seconds: float

    @property
    def l2_hit_rate(self) -> float:
        total = self.records_from_l2 + self.records_from_disk
        if total <= 0:
            return 0.0
        return float(self.records_from_l2) / float(total)

    @property
    def effective_bandwidth_gbps(self) -> float:
        elapsed = self.io_seconds + self.decode_seconds
        if elapsed <= 0.0:
            return 0.0
        return (float(self.decoded_bytes) / 1e9) / elapsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fetches": self.fetches,
            "records_requested": self.records_requested,
            "records_from_l2": self.records_from_l2,
            "records_from_disk": self.records_from_disk,
            "l2_hit_rate": round(self.l2_hit_rate, 6),
            "gpu_decodes": self.gpu_decodes,
            "cpu_decodes": self.cpu_decodes,
            "prefetch_hits": self.prefetch_hits,
            "refused": self.refused,
            "disk_bytes": self.disk_bytes,
            "decoded_bytes": self.decoded_bytes,
            "l2_used_bytes": self.l2_used_bytes,
            "l2_entries": self.l2_entries,
            "io_seconds": self.io_seconds,
            "decode_seconds": self.decode_seconds,
            "verify_seconds": self.verify_seconds,
            "effective_bandwidth_gbps": round(self.effective_bandwidth_gbps, 4),
        }


# ---- Shared library discovery & prototype binding --------------------------

_LIB_CACHE: Dict[str, ctypes.CDLL] = {}
_DUMMY_BYTE = (ctypes.c_uint8 * 1)(0)


def _platform_lib_names() -> List[str]:
    if sys.platform == "win32":
        return ["libwex.dll", "wex.dll"]
    if sys.platform == "darwin":
        return ["libwex.dylib"]
    return ["libwex.so", "libwex.so.1"]


def _candidate_library_paths(lib_path: Optional[Union[str, os.PathLike]]) -> List[str]:
    if lib_path is not None:
        return [os.fspath(lib_path)]

    candidates: List[str] = []
    env_path = os.environ.get("WEX_LIB_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    pkg_dir = Path(__file__).resolve().parent
    names = _platform_lib_names()

    # Package directory (e.g., bundled wheel)
    for name in names:
        p = pkg_dir / name
        if p.is_file():
            candidates.append(str(p))

    # Repository build directories relative to libwex/bindings/python/pywex
    libwex_root = pkg_dir.parent.parent.parent
    search_dirs = [
        libwex_root / "build",
        libwex_root / "build" / "Release",
        libwex_root / "build" / "Debug",
        libwex_root,
    ]
    for d in search_dirs:
        if d.is_dir():
            for name in names:
                p = d / name
                if p.is_file():
                    candidates.append(str(p))

    # OS default dynamic linker search names
    candidates.extend(names)

    # Deduplicate preserving order
    seen = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _bind_prototypes(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.wex_abi_version.argtypes = []
    lib.wex_abi_version.restype = ctypes.c_uint32

    lib.wex_version_string.argtypes = []
    lib.wex_version_string.restype = ctypes.c_char_p

    lib.wex_open.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_CWexOpenOpts),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.wex_open.restype = ctypes.c_void_p

    lib.wex_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CWexInfo)]
    lib.wex_info.restype = ctypes.c_int

    lib.wex_has_expert.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.wex_has_expert.restype = ctypes.c_int

    lib.wex_record_layout.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.POINTER(_CWexLayout),
    ]
    lib.wex_record_layout.restype = ctypes.c_int

    lib.wex_fetch_host.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.wex_fetch_host.restype = ctypes.c_int

    lib.wex_fetch_device.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.wex_fetch_device.restype = ctypes.c_int

    lib.wex_prefetch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    lib.wex_prefetch.restype = ctypes.c_int

    lib.wex_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CWexStats)]
    lib.wex_stats.restype = None

    lib.wex_close.argtypes = [ctypes.c_void_p]
    lib.wex_close.restype = None

    return lib


def load_library(lib_path: Optional[Union[str, os.PathLike]] = None) -> ctypes.CDLL:
    """Locate, load, and verify ABI v1 of the libwex shared library."""
    candidates = _candidate_library_paths(lib_path)
    errors: List[str] = []

    for cand in candidates:
        cache_key = os.path.abspath(cand) if os.path.exists(cand) else cand
        if cache_key in _LIB_CACHE:
            return _LIB_CACHE[cache_key]
        try:
            lib = ctypes.CDLL(cand)
        except OSError as exc:
            errors.append(f"{cand}: {exc}")
            continue

        try:
            _bind_prototypes(lib)
            abi = int(lib.wex_abi_version())
        except (AttributeError, OSError) as exc:
            errors.append(f"{cand}: missing ABI symbols ({exc})")
            continue

        if abi != WEX_STORE_ABI_VERSION:
            raise WexError(
                WEX_E_UNSUPPORTED,
                f"libwex ABI mismatch in {cand}: got {abi}, expected {WEX_STORE_ABI_VERSION}",
            )

        _LIB_CACHE[cache_key] = lib
        return lib

    detail = "; ".join(errors) if errors else "no candidates found"
    raise WexError(
        WEX_E_IO,
        f"Failed to load libwex shared library (set WEX_LIB_PATH or pass lib_path): {detail}",
    )


def abi_version(lib_path: Optional[Union[str, os.PathLike]] = None) -> int:
    """Return the C ABI version reported by libwex."""
    return int(load_library(lib_path).wex_abi_version())


def version_string(lib_path: Optional[Union[str, os.PathLike]] = None) -> str:
    """Return the human-readable libwex version string."""
    raw = load_library(lib_path).wex_version_string()
    return raw.decode("utf-8", errors="replace") if raw else ""


# ---- Internal option parsing & buffer helpers ------------------------------


def _parse_verify(verify: Union[str, int, bool]) -> int:
    if isinstance(verify, bool):
        return WEX_VERIFY_ON if verify else WEX_VERIFY_OFF
    if isinstance(verify, int):
        if verify in (WEX_VERIFY_AUTO, WEX_VERIFY_OFF, WEX_VERIFY_ON):
            return verify
        raise ValueError(f"Invalid verify mode int: {verify}")
    if isinstance(verify, str):
        low = verify.strip().lower()
        mapping = {
            "auto": WEX_VERIFY_AUTO,
            "off": WEX_VERIFY_OFF,
            "false": WEX_VERIFY_OFF,
            "0": WEX_VERIFY_OFF,
            "on": WEX_VERIFY_ON,
            "true": WEX_VERIFY_ON,
            "1": WEX_VERIFY_ON,
        }
        if low in mapping:
            return mapping[low]
        raise ValueError(f"Invalid verify mode: {verify!r} (expected 'auto', 'off', or 'on')")
    raise TypeError(f"verify must be str, int, or bool, got {type(verify).__name__}")


def _parse_l2_bytes(l2_cache_bytes: Union[int, str]) -> int:
    if isinstance(l2_cache_bytes, str):
        if l2_cache_bytes.strip().lower() == "auto":
            return UINT64_MAX
        raise ValueError(f"Invalid l2_cache_bytes string: {l2_cache_bytes!r} (expected 'auto' or int)")
    if isinstance(l2_cache_bytes, bool) or not isinstance(l2_cache_bytes, int):
        raise TypeError("l2_cache_bytes must be a non-negative int or 'auto'")
    if l2_cache_bytes < 0 or l2_cache_bytes > UINT64_MAX:
        raise ValueError(f"l2_cache_bytes out of uint64 range: {l2_cache_bytes}")
    return l2_cache_bytes


def _classify_open_error(msg: str) -> int:
    low = msg.lower()
    if "root sha-256 mismatch" in low:
        return WEX_E_ROOT
    if "sha-256 mismatch" in low:
        return WEX_E_HASH
    if "invalid wex4" in low or "invalid wnm" in low or "non-zero padding" in low:
        return WEX_E_FORMAT
    if "out of memory" in low:
        return WEX_E_OOM
    if "struct_size" in low or "null or empty" in low:
        return WEX_E_ARGS
    return WEX_E_IO


def _pin_writable_buffer(buf: Any) -> Tuple[int, int, Any]:
    """Return (raw_address, nbytes, keepalive_obj) for a writable buffer with zero copy."""
    if isinstance(buf, bytearray):
        n = len(buf)
        if n == 0:
            return ctypes.addressof(_DUMMY_BYTE), 0, _DUMMY_BYTE
        arr = (ctypes.c_char * n).from_buffer(buf)
        return ctypes.addressof(arr), n, arr

    try:
        mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    except TypeError as exc:
        raise TypeError(
            f"dst_buffers elements must support the writable buffer protocol, got {type(buf).__name__}"
        ) from exc

    if mv.readonly:
        raise TypeError("dst_buffers element must be writable (not bytes or read-only memoryview)")
    if not mv.c_contiguous:
        raise ValueError("dst_buffers element must be C-contiguous")

    n = mv.nbytes
    if n == 0:
        return ctypes.addressof(_DUMMY_BYTE), 0, (_DUMMY_BYTE, mv)

    mv_b = mv.cast("B")
    arr = (ctypes.c_char * n).from_buffer(mv_b)
    return ctypes.addressof(arr), n, (arr, mv_b, mv)


# ---- Main WexStore handle wrapper ------------------------------------------


class WexStore:
    """Pythonic context manager wrapping an open libwex wex_store_t handle."""

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *,
        root_hex: Optional[str] = None,
        verify: Union[str, int, bool] = "auto",
        mirrors: Optional[Sequence[Union[str, os.PathLike]]] = None,
        l2_cache_bytes: Union[int, str] = 0,
        l2_pin_gpu: bool = False,
        io_direct: int = -1,
        io_queue_depth: int = 0,
        cpu_threads: int = 0,
        gpu_device: int = -1,
        gpu_helper_path: Optional[Union[str, os.PathLike]] = None,
        disk_weights: Optional[Sequence[int]] = None,
        lib_path: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        self._handle: Optional[int] = None
        self._lib: Optional[ctypes.CDLL] = None
        self.path = os.fspath(path)

        lib = load_library(lib_path)
        self._lib = lib

        opts = _CWexOpenOpts()
        opts.struct_size = ctypes.sizeof(_CWexOpenOpts)

        root_b = root_hex.encode("utf-8") if root_hex is not None else None
        opts.root_hex = root_b
        opts.verify = _parse_verify(verify)

        mirror_bytes: List[bytes] = []
        mirrors_arr = None
        if mirrors:
            mirror_bytes = [os.fspath(m).encode("utf-8") for m in mirrors]
            mirrors_arr = (ctypes.c_char_p * len(mirror_bytes))(*mirror_bytes)
            opts.mirrors = mirrors_arr
            opts.num_mirrors = len(mirror_bytes)
        else:
            opts.mirrors = None
            opts.num_mirrors = 0

        opts.l2_cache_bytes = _parse_l2_bytes(l2_cache_bytes)
        opts.l2_pin_gpu = 1 if l2_pin_gpu else 0
        opts.io_direct = int(io_direct)
        opts.io_queue_depth = int(io_queue_depth)
        opts.cpu_threads = int(cpu_threads)
        opts.gpu_device = int(gpu_device)

        helper_b = (
            os.fspath(gpu_helper_path).encode("utf-8")
            if gpu_helper_path is not None
            else None
        )
        opts.gpu_helper_path = helper_b

        weights_arr = None
        if disk_weights is not None:
            expected_n = 1 + opts.num_mirrors
            if len(disk_weights) != expected_n:
                raise ValueError(
                    f"disk_weights length ({len(disk_weights)}) must equal 1 + len(mirrors) ({expected_n})"
                )
            if any(int(w) <= 0 for w in disk_weights):
                raise ValueError("disk_weights entries must be positive integers")
            weights_arr = (ctypes.c_uint32 * expected_n)(*(int(w) for w in disk_weights))
            opts.disk_weights = weights_arr
        else:
            opts.disk_weights = None

        err_buf = ctypes.create_string_buffer(512)
        path_b = self.path.encode("utf-8")
        handle = lib.wex_open(path_b, ctypes.byref(opts), err_buf, len(err_buf))
        if not handle:
            msg = err_buf.value.decode("utf-8", errors="replace").strip()
            code = _classify_open_error(msg)
            raise WexError(code, msg or f"wex_open failed for {self.path}")

        self._handle = int(handle)

    @property
    def is_closed(self) -> bool:
        return self._handle is None

    def _require_handle(self) -> Tuple[ctypes.CDLL, int]:
        if self._handle is None or self._lib is None:
            raise WexError(WEX_E_ARGS, "WexStore handle is closed")
        return self._lib, self._handle

    def close(self) -> None:
        """Close the store and release L2 cache, I/O handles, and prefetch threads."""
        handle = self._handle
        lib = self._lib
        self._handle = None
        if handle and lib is not None:
            lib.wex_close(ctypes.c_void_p(handle))

    def __enter__(self) -> "WexStore":
        self._require_handle()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def info(self) -> WexInfo:
        """Return static container metadata and active backend/IO details."""
        lib, handle = self._require_handle()
        c_info = _CWexInfo()
        rc = int(lib.wex_info(ctypes.c_void_p(handle), ctypes.byref(c_info)))
        if rc != WEX_OK:
            raise WexError(rc, "wex_info failed")
        return WexInfo(
            abi_version=int(c_info.abi_version),
            layers=int(c_info.layers),
            experts=int(c_info.experts),
            num_records=int(c_info.num_records),
            num_tensors=int(c_info.num_tensors),
            codec_kind=int(c_info.codec_kind),
            value_exact=bool(c_info.value_exact),
            verified_root=bool(c_info.verified_root),
            verify_on=bool(c_info.verify_on),
            num_replicas=int(c_info.num_replicas),
            file_bytes=int(c_info.file_bytes),
            total_decoded_bytes=int(c_info.total_decoded_bytes),
            max_record_decoded_bytes=int(c_info.max_record_decoded_bytes),
            max_record_stored_bytes=int(c_info.max_record_stored_bytes),
            l2_capacity_bytes=int(c_info.l2_capacity_bytes),
            root_hex=c_info.root_hex.decode("ascii", errors="replace"),
            backend_name=c_info.backend_name.decode("utf-8", errors="replace"),
            io_engine=c_info.io_engine.decode("utf-8", errors="replace"),
        )

    def has_expert(self, layer: int, eid: int) -> bool:
        """Return True if the container holds expert (layer, eid)."""
        lib, handle = self._require_handle()
        return bool(lib.wex_has_expert(ctypes.c_void_p(handle), int(layer), int(eid)))

    def record_layout(
        self, layer: int, eid: int, *, by_record_id: bool = False
    ) -> WexLayout:
        """Inspect decoded slab size, tensor offsets, and expert matrix shapes."""
        lib, handle = self._require_handle()
        flags = WEX_FETCH_BY_RECORD_ID if by_record_id else WEX_FETCH_DEFAULT
        c_lay = _CWexLayout()
        rc = int(
            lib.wex_record_layout(
                ctypes.c_void_p(handle),
                int(layer),
                int(eid),
                flags,
                ctypes.byref(c_lay),
            )
        )
        if rc != WEX_OK:
            target = f"record {eid}" if by_record_id else f"expert (layer={layer}, eid={eid})"
            raise WexError(rc, f"wex_record_layout failed for {target}")

        tcount = int(c_lay.tensor_count)
        lim = min(tcount, WEX_MAX_RECORD_TENSORS)
        return WexLayout(
            record_index=int(c_lay.record_index),
            codec_kind=int(c_lay.codec_kind),
            flags=int(c_lay.flags),
            tensor_count=tcount,
            file_offset=int(c_lay.file_offset),
            stored_bytes=int(c_lay.stored_bytes),
            decoded_bytes=int(c_lay.decoded_bytes),
            tensor_offsets=tuple(int(c_lay.tensor_offsets[i]) for i in range(lim)),
            tensor_bytes=tuple(int(c_lay.tensor_bytes[i]) for i in range(lim)),
            scale_offsets=(
                int(c_lay.scale_offsets[0]),
                int(c_lay.scale_offsets[1]),
                int(c_lay.scale_offsets[2]),
            ),
            weight_offsets=(
                int(c_lay.weight_offsets[0]),
                int(c_lay.weight_offsets[1]),
                int(c_lay.weight_offsets[2]),
            ),
            scale_shapes=(
                (int(c_lay.scale_shapes[0][0]), int(c_lay.scale_shapes[0][1])),
                (int(c_lay.scale_shapes[1][0]), int(c_lay.scale_shapes[1][1])),
                (int(c_lay.scale_shapes[2][0]), int(c_lay.scale_shapes[2][1])),
            ),
            weight_shapes=(
                (int(c_lay.weight_shapes[0][0]), int(c_lay.weight_shapes[0][1])),
                (int(c_lay.weight_shapes[1][0]), int(c_lay.weight_shapes[1][1])),
                (int(c_lay.weight_shapes[2][0]), int(c_lay.weight_shapes[2][1])),
            ),
        )

    def fetch_host_ptrs(
        self,
        layer: int,
        eids: Sequence[int],
        dst_ptrs: Sequence[int],
        dst_caps: Sequence[int],
        *,
        scratch: bool = True,
        bypass_l2: bool = False,
        by_record_id: bool = False,
    ) -> None:
        """Low-level host fetch directly into raw host pointer addresses (zero copy)."""
        lib, handle = self._require_handle()
        eid_list = [int(e) for e in eids]
        n = len(eid_list)
        if len(dst_ptrs) != n or len(dst_caps) != n:
            raise ValueError(
                f"dst_ptrs ({len(dst_ptrs)}) and dst_caps ({len(dst_caps)}) must match len(eids) ({n})"
            )
        if n == 0:
            return

        c_eids = (ctypes.c_int * n)(*eid_list)
        c_ptrs = (ctypes.c_void_p * n)(*(int(p) for p in dst_ptrs))
        c_caps = (ctypes.c_size_t * n)(*(int(c) for c in dst_caps))
        c_status = (ctypes.c_int * n)()

        flags = WEX_FETCH_DEFAULT
        if scratch:
            flags |= WEX_FETCH_SCRATCH
        if bypass_l2:
            flags |= WEX_FETCH_BYPASS_L2
        if by_record_id:
            flags |= WEX_FETCH_BY_RECORD_ID

        failed = int(
            lib.wex_fetch_host(
                ctypes.c_void_p(handle),
                int(layer),
                c_eids,
                n,
                c_ptrs,
                c_caps,
                flags,
                c_status,
            )
        )
        if failed != 0:
            if failed < 0:
                raise WexError(failed, "wex_fetch_host rejected arguments")
            first_code = WEX_E_IO
            details: List[str] = []
            for i in range(n):
                st = int(c_status[i])
                if st != WEX_OK:
                    if first_code == WEX_E_IO and len(details) == 0:
                        first_code = st
                    name = ERROR_NAMES.get(st, str(st))
                    details.append(f"item[{i}] (eid={eid_list[i]}): {name}")
            raise WexError(
                first_code,
                f"wex_fetch_host failed for {failed}/{n} items ({'; '.join(details)})",
            )

    def fetch_host(
        self,
        layer: int,
        eids: Iterable[int],
        *,
        dst_buffers: Optional[Sequence[Any]] = None,
        scratch: bool = True,
        bypass_l2: bool = False,
        by_record_id: bool = False,
    ) -> List[Any]:
        """Fetch experts/records into newly allocated bytearrays or caller buffers (zero copy)."""
        self._require_handle()
        eid_list = [int(e) for e in eids]
        n = len(eid_list)

        if dst_buffers is None:
            if n == 0:
                return []
            out_bufs: List[Any] = [
                bytearray(self.record_layout(layer, eid, by_record_id=by_record_id).decoded_bytes)
                for eid in eid_list
            ]
        else:
            out_bufs = list(dst_buffers)
            if len(out_bufs) != n:
                raise ValueError(
                    f"dst_buffers length ({len(out_bufs)}) must match len(eids) ({n})"
                )
            if n == 0:
                return out_bufs

        ptrs: List[int] = []
        caps: List[int] = []
        keepalive: List[Any] = []
        for buf in out_bufs:
            ptr, cap, pin = _pin_writable_buffer(buf)
            ptrs.append(ptr)
            caps.append(cap)
            keepalive.append(pin)

        self.fetch_host_ptrs(
            layer,
            eid_list,
            ptrs,
            caps,
            scratch=scratch,
            bypass_l2=bypass_l2,
            by_record_id=by_record_id,
        )
        # Keep buffer exports alive until C call finishes
        del keepalive
        return out_bufs

    def fetch_device_ptrs(
        self,
        layer: int,
        eids: Sequence[int],
        *,
        dst_scales: Optional[Sequence[int]] = None,
        dst_weights: Optional[Sequence[int]] = None,
        dst_slabs: Optional[Sequence[int]] = None,
        stream_ptr: int = 0,
        scratch: bool = False,
        bypass_l2: bool = False,
        by_record_id: bool = False,
        expand_e8m0_f32: bool = False,
    ) -> None:
        """Low-level zero-copy GPU fetch into caller device pointer addresses."""
        lib, handle = self._require_handle()
        eid_list = [int(e) for e in eids]
        n = len(eid_list)
        if n == 0:
            return

        if dst_scales is not None and len(dst_scales) != 3 * n:
            raise ValueError(f"dst_scales length ({len(dst_scales)}) must equal 3 * len(eids) ({3 * n})")
        if dst_weights is not None and len(dst_weights) != 3 * n:
            raise ValueError(f"dst_weights length ({len(dst_weights)}) must equal 3 * len(eids) ({3 * n})")
        if dst_slabs is not None and len(dst_slabs) != n:
            raise ValueError(f"dst_slabs length ({len(dst_slabs)}) must equal len(eids) ({n})")
        if dst_slabs is None and (dst_scales is None or dst_weights is None):
            raise ValueError("Provide either (dst_scales and dst_weights) or dst_slabs")

        c_eids = (ctypes.c_int * n)(*eid_list)
        c_scales = (
            (ctypes.c_void_p * (3 * n))(*(int(p) for p in dst_scales))
            if dst_scales is not None
            else None
        )
        c_weights = (
            (ctypes.c_void_p * (3 * n))(*(int(p) for p in dst_weights))
            if dst_weights is not None
            else None
        )
        c_slabs = (
            (ctypes.c_void_p * n)(*(int(p) for p in dst_slabs))
            if dst_slabs is not None
            else None
        )
        c_status = (ctypes.c_int * n)()

        flags = WEX_FETCH_DEFAULT
        if scratch:
            flags |= WEX_FETCH_SCRATCH
        if bypass_l2:
            flags |= WEX_FETCH_BYPASS_L2
        if by_record_id:
            flags |= WEX_FETCH_BY_RECORD_ID
        if expand_e8m0_f32:
            flags |= WEX_FETCH_EXPAND_E8M0_F32

        failed = int(
            lib.wex_fetch_device(
                ctypes.c_void_p(handle),
                int(layer),
                c_eids,
                n,
                c_scales,
                c_weights,
                c_slabs,
                ctypes.c_void_p(int(stream_ptr)) if stream_ptr else None,
                flags,
                c_status,
            )
        )
        if failed != 0:
            if failed < 0:
                raise WexError(failed, "wex_fetch_device rejected arguments")
            first_code = WEX_E_DEVICE
            details: List[str] = []
            for i in range(n):
                st = int(c_status[i])
                if st != WEX_OK:
                    if len(details) == 0:
                        first_code = st
                    name = ERROR_NAMES.get(st, str(st))
                    details.append(f"item[{i}] (eid={eid_list[i]}): {name}")
            raise WexError(
                first_code,
                f"wex_fetch_device failed for {failed}/{n} items ({'; '.join(details)})",
            )

    def prefetch(
        self, layer: int, eids: Iterable[int], *, by_record_id: bool = False
    ) -> int:
        """Non-blocking hint to warm the L2 compressed-RAM cache."""
        lib, handle = self._require_handle()
        eid_list = [int(e) for e in eids]
        n = len(eid_list)
        if n == 0:
            return 0
        c_eids = (ctypes.c_int * n)(*eid_list)
        flags = WEX_FETCH_BY_RECORD_ID if by_record_id else WEX_FETCH_DEFAULT
        return int(
            lib.wex_prefetch(ctypes.c_void_p(handle), int(layer), c_eids, n, flags)
        )

    def stats(self) -> WexStats:
        """Return live telemetry snapshot for this store."""
        lib, handle = self._require_handle()
        c_st = _CWexStats()
        lib.wex_stats(ctypes.c_void_p(handle), ctypes.byref(c_st))
        return WexStats(
            fetches=int(c_st.fetches),
            records_requested=int(c_st.records_requested),
            records_from_l2=int(c_st.records_from_l2),
            records_from_disk=int(c_st.records_from_disk),
            gpu_decodes=int(c_st.gpu_decodes),
            cpu_decodes=int(c_st.cpu_decodes),
            prefetch_hits=int(c_st.prefetch_hits),
            refused=int(c_st.refused),
            disk_bytes=int(c_st.disk_bytes),
            decoded_bytes=int(c_st.decoded_bytes),
            l2_used_bytes=int(c_st.l2_used_bytes),
            l2_entries=int(c_st.l2_entries),
            io_seconds=float(c_st.io_seconds),
            decode_seconds=float(c_st.decode_seconds),
            verify_seconds=float(c_st.verify_seconds),
        )
