#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Comprehensive automated test suite for pywex, PyTorch CUDA extension, and adapters.

Run with:
    python -B -I libwex/tests/test_pywex.py
Set WEX_LIB_PATH (and optionally WEX_CUDA_LIB, WEX_REAL_CONTAINER) if libwex is outside
the default build directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import http.server
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PYWEX_ROOT = REPO_ROOT / "libwex" / "bindings" / "python"
STRATA_DIR = REPO_ROOT / "libwex" / "adapters" / "strata"
HERMES_DIR = REPO_ROOT / "libwex" / "adapters" / "hermes"
VECTORS_DIR = REPO_ROOT / "wex4" / "vectors" / "valid"

for _p in (PYWEX_ROOT, STRATA_DIR, HERMES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pywex
from pywex import _lib
from wex_provider import WexExpertProvider, dispatch_tool, get_hermes_tools
from wex_strata_bridge import StrataWexBridge


# ---- Pure-stdlib synthetic WEX4 RAW expert container writer ----------------

_WEX4_MAGIC = 0x57455834
_WEX4_PAGE = 4096
_WEX4_HEADER_HASHED = 4064
_WEX4_CODEC_RAW = 0
_WEX4_REC_EXPERT = 2
_WEX4_DT_I8 = 3
_WEX4_DT_F8_E8M0 = 6

_HEADER_STRUCT = struct.Struct("<IHHQQQQQIIIIIIQQQ32s")
_RECORD_STRUCT = struct.Struct("<QIIHHIIII32sI")
_TENSOR_STRUCT = struct.Struct("<IHBB8QIIQQ")


def _align_page(n: int) -> int:
    return (n + _WEX4_PAGE - 1) & ~(_WEX4_PAGE - 1)


def build_raw_expert_wex4(
    dst_path: Path,
    *,
    layers: int = 2,
    experts: int = 4,
    rows: int = 64,
    scols: int = 8,
) -> Tuple[Path, str, Dict[Tuple[int, int], bytes]]:
    """Write a valid WEX4 RAW MXFP4 expert container and return (path, root_hex, slabs)."""
    wcols = scols * 16
    sc_mat_bytes = rows * scols
    wt_mat_bytes = rows * wcols
    sc_total = 3 * sc_mat_bytes
    wt_total = 3 * wt_mat_bytes
    decoded_bytes = sc_total + wt_total
    stored_bytes = _align_page(decoded_bytes)

    slabs: Dict[Tuple[int, int], bytes] = {}
    records_meta: List[Tuple[int, int, bytes, List[Tuple[str, int, Tuple[int, int], int, int]]]] = []

    for layer in range(layers):
        for eid in range(experts):
            # Deterministic compressible MXFP4 payload
            seed = (layer * 131 + eid * 17 + 7) & 0xFF
            sc_bytes = bytes((118 + ((seed + i) & 7)) for i in range(sc_total))
            wt_bytes = bytes((((seed + i * 3) & 0x07) | (((seed + i * 5) & 0x07) << 4)) for i in range(wt_total))
            slab = sc_bytes + wt_bytes
            assert len(slab) == decoded_bytes
            slabs[(layer, eid)] = slab

            tensors: List[Tuple[str, int, Tuple[int, int], int, int]] = []
            for m in range(3):
                tensors.append((
                    f"layers.{layer}.ffn.experts.{eid}.w{m + 1}.scale",
                    _WEX4_DT_F8_E8M0,
                    (rows, scols),
                    m * sc_mat_bytes,
                    sc_mat_bytes,
                ))
            for m in range(3):
                tensors.append((
                    f"layers.{layer}.ffn.experts.{eid}.w{m + 1}.weight",
                    _WEX4_DT_I8,
                    (rows, wcols),
                    sc_total + m * wt_mat_bytes,
                    wt_mat_bytes,
                ))
            records_meta.append((layer, eid, slab, tensors))

    nrecs = len(records_meta)
    ntensors = nrecs * 6

    name_pool = bytearray()
    name_entries: List[Tuple[int, int, str]] = []
    for _, _, _, tensors in records_meta:
        for tname, _, _, _, _ in tensors:
            nb = tname.encode("utf-8")
            name_entries.append((len(name_pool), len(nb), tname))
            name_pool.extend(nb)

    order = sorted(range(ntensors), key=lambda idx: name_entries[idx][2].encode("utf-8"))

    rec_offsets: List[int] = []
    pos = _WEX4_PAGE
    for _ in range(nrecs):
        rec_offsets.append(pos)
        pos += stored_bytes

    index_offset = pos
    index_buf = bytearray()

    first_t = 0
    for r_idx, (layer, eid, slab, tensors) in enumerate(records_meta):
        padded = slab + b"\x00" * (stored_bytes - decoded_bytes)
        rec_sha = hashlib.sha256(padded).digest()
        owner = (layer << 16) | eid
        index_buf.extend(
            _RECORD_STRUCT.pack(
                rec_offsets[r_idx],
                stored_bytes,
                decoded_bytes,
                _WEX4_CODEC_RAW,
                _WEX4_REC_EXPERT,
                0,
                owner,
                first_t,
                len(tensors),
                rec_sha,
                0,
            )
        )
        first_t += len(tensors)

    t_idx = 0
    for r_idx, (_, _, _, tensors) in enumerate(records_meta):
        for _, dtype, (r0, r1), off, nbytes in tensors:
            name_at, name_len, _ = name_entries[t_idx]
            index_buf.extend(
                _TENSOR_STRUCT.pack(
                    name_at,
                    name_len,
                    dtype,
                    2,
                    r0,
                    r1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    r_idx,
                    1,
                    off,
                    nbytes,
                )
            )
            t_idx += 1

    for idx in order:
        index_buf.extend(struct.pack("<I", idx))
    index_buf.extend(name_pool)

    index_bytes = len(index_buf)
    index_len = _align_page(index_bytes)
    file_bytes = index_offset + index_len
    index_sha = hashlib.sha256(index_buf).digest()

    header = bytearray(_WEX4_PAGE)
    header[: _HEADER_STRUCT.size] = _HEADER_STRUCT.pack(
        _WEX4_MAGIC,
        4,
        0,
        0,
        0,
        file_bytes,
        index_offset,
        index_bytes,
        0,
        nrecs,
        ntensors,
        layers,
        experts,
        decoded_bytes,
        len(name_pool),
        0,
        decoded_bytes * nrecs,
        index_sha,
    )
    root_digest = hashlib.sha256(header[:_WEX4_HEADER_HASHED]).digest()
    header[_WEX4_HEADER_HASHED:] = root_digest

    with open(dst_path, "wb") as fp:
        fp.write(header)
        for _, _, slab, _ in records_meta:
            fp.write(slab)
            fp.write(b"\x00" * (stored_bytes - decoded_bytes))
        fp.write(index_buf)
        fp.write(b"\x00" * (index_len - index_bytes))

    return dst_path, root_digest.hex(), slabs


# ---- Local mock HTTP server for Strata & wex serve adapter tests -----------


class _MockLocalHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, code: int, payload: Dict[str, object]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "engine": "strata-mock"})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": "mock-moe-7b"}]})
        elif self.path == "/v1/telemetry":
            self._send_json(
                200,
                {
                    "tokens_per_sec": 128.5,
                    "active_requests": 1,
                    "store": {"abi_version": 1, "codec_name": "WNR4-MX", "layers": 2, "experts": 4},
                    "counters": {"fetches": 12, "l2_hit_ratio": 0.75},
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        parsed = json.loads(body) if body.strip() else {}
        if self.path == "/v1/prefetch":
            eids = parsed.get("eids", [])
            self._send_json(
                200,
                {
                    "ok": True,
                    "layer": parsed.get("layer", 0),
                    "requested": len(eids),
                    "queued": len(eids),
                },
            )
        else:
            self._send_json(404, {"error": "not found"})


def _start_mock_server() -> Tuple[http.server.ThreadingHTTPServer, threading.Thread, str]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockLocalHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th, f"http://127.0.0.1:{port}"


# ---- Test Suites -----------------------------------------------------------


class TestAbiAndStructLayout(unittest.TestCase):
    def test_abi_and_version(self) -> None:
        self.assertEqual(pywex.abi_version(), pywex.WEX_STORE_ABI_VERSION)
        ver = pywex.version_string()
        self.assertIsInstance(ver, str)
        self.assertTrue(len(ver) > 0)

    def test_struct_sizes_and_offsets_64bit(self) -> None:
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("64-bit struct offset check")
        self.assertEqual(ctypes.sizeof(_lib._CWexOpenOpts), 88)
        self.assertEqual(_lib._CWexOpenOpts.struct_size.offset, 0)
        self.assertEqual(_lib._CWexOpenOpts.root_hex.offset, 8)
        self.assertEqual(_lib._CWexOpenOpts.verify.offset, 16)
        self.assertEqual(_lib._CWexOpenOpts.mirrors.offset, 24)
        self.assertEqual(_lib._CWexOpenOpts.num_mirrors.offset, 32)
        self.assertEqual(_lib._CWexOpenOpts.l2_cache_bytes.offset, 40)
        self.assertEqual(_lib._CWexOpenOpts.l2_pin_gpu.offset, 48)
        self.assertEqual(_lib._CWexOpenOpts.gpu_device.offset, 64)
        self.assertEqual(_lib._CWexOpenOpts.gpu_helper_path.offset, 72)
        self.assertEqual(_lib._CWexOpenOpts.disk_weights.offset, 80)

        self.assertEqual(ctypes.sizeof(_lib._CWexInfo), 216)
        self.assertEqual(_lib._CWexInfo.file_bytes.offset, 40)
        self.assertEqual(_lib._CWexInfo.root_hex.offset, 80)
        self.assertEqual(_lib._CWexInfo.backend_name.offset, 145)
        self.assertEqual(_lib._CWexInfo.io_engine.offset, 177)

        self.assertEqual(ctypes.sizeof(_lib._CWexLayout), 368)
        self.assertEqual(_lib._CWexLayout.file_offset.offset, 16)
        self.assertEqual(_lib._CWexLayout.tensor_offsets.offset, 40)
        self.assertEqual(_lib._CWexLayout.tensor_bytes.offset, 168)
        self.assertEqual(_lib._CWexLayout.scale_offsets.offset, 296)
        self.assertEqual(_lib._CWexLayout.weight_offsets.offset, 308)
        self.assertEqual(_lib._CWexLayout.scale_shapes.offset, 320)
        self.assertEqual(_lib._CWexLayout.weight_shapes.offset, 344)

        self.assertEqual(ctypes.sizeof(_lib._CWexStats), 120)
        self.assertEqual(_lib._CWexStats.io_seconds.offset, 96)
        self.assertEqual(_lib._CWexStats.verify_seconds.offset, 112)


class TestWexStoreCore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wex_path, self.root_hex, self.slabs = build_raw_expert_wex4(
            Path(self._tmp.name) / "raw_experts.wex4"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_open_info_layout_fetch_and_errors(self) -> None:
        with pywex.WexStore(
            self.wex_path,
            root_hex=self.root_hex,
            verify="auto",
            gpu_device=-2,
        ) as store:
            info = store.info()
            self.assertEqual(info.abi_version, 1)
            self.assertEqual(info.layers, 2)
            self.assertEqual(info.experts, 4)
            self.assertEqual(info.num_records, 8)
            self.assertEqual(info.num_tensors, 48)
            self.assertEqual(info.codec_kind, pywex.WEX_CODEC_RAW)
            self.assertEqual(info.codec_name, "RAW")
            self.assertTrue(info.verified_root)
            self.assertTrue(info.verify_on)
            self.assertEqual(info.root_hex, self.root_hex)
            self.assertGreater(info.compression_ratio, 0.0)
            self.assertEqual(info.to_dict()["codec_name"], "RAW")

            self.assertTrue(store.has_expert(0, 0))
            self.assertTrue(store.has_expert(1, 3))
            self.assertFalse(store.has_expert(2, 0))
            self.assertFalse(store.has_expert(0, 4))
            self.assertFalse(store.has_expert(-1, 0))

            lay = store.record_layout(1, 2)
            self.assertEqual(lay.record_index, 6)
            self.assertTrue(lay.is_expert)
            self.assertEqual(lay.tensor_count, 6)
            self.assertEqual(lay.decoded_bytes, len(self.slabs[(1, 2)]))
            self.assertEqual(lay.scale_shapes, ((64, 8), (64, 8), (64, 8)))
            self.assertEqual(lay.weight_shapes, ((64, 128), (64, 128), (64, 128)))
            self.assertEqual(len(lay.tensor_offsets), 6)
            self.assertEqual(len(lay.tensor_bytes), 6)

            lay_by_id = store.record_layout(0, 6, by_record_id=True)
            self.assertEqual(lay_by_id, lay)

            # Allocated fetch_host
            got = store.fetch_host(1, [0, 3])
            self.assertEqual(len(got), 2)
            self.assertIsInstance(got[0], bytearray)
            self.assertEqual(bytes(got[0]), self.slabs[(1, 0)])
            self.assertEqual(bytes(got[1]), self.slabs[(1, 3)])

            # Zero-copy pre-allocated bytearray & memoryview
            pre_ba = bytearray(lay.decoded_bytes)
            pre_mv_backing = bytearray(lay.decoded_bytes)
            pre_mv = memoryview(pre_mv_backing)
            out = store.fetch_host(0, [1, 2], dst_buffers=[pre_ba, pre_mv], scratch=False)
            self.assertIs(out[0], pre_ba)
            self.assertIs(out[1], pre_mv)
            self.assertEqual(bytes(pre_ba), self.slabs[(0, 1)])
            self.assertEqual(bytes(pre_mv_backing), self.slabs[(0, 2)])

            # Flat record ID fetch
            rid_got = store.fetch_host(0, [7], by_record_id=True)
            self.assertEqual(bytes(rid_got[0]), self.slabs[(1, 3)])

            # Error: too small destination buffer (including 0-byte buffer) -> WEX_E_CAP
            with self.assertRaises(pywex.WexError) as cm_cap:
                store.fetch_host(0, [0], dst_buffers=[bytearray(64)])
            self.assertEqual(cm_cap.exception.code, pywex.WEX_E_CAP)

            with self.assertRaises(pywex.WexError) as cm_zero:
                store.fetch_host(0, [0], dst_buffers=[bytearray(0)])
            self.assertEqual(cm_zero.exception.code, pywex.WEX_E_CAP)

            # Error: missing expert -> WEX_E_NOT_FOUND
            with self.assertRaises(pywex.WexError) as cm_nf:
                store.fetch_host(0, [99])
            self.assertEqual(cm_nf.exception.code, pywex.WEX_E_NOT_FOUND)

            # Error: fetch_device on CPU-only store -> WEX_E_DEVICE
            with self.assertRaises(pywex.WexError) as cm_dev:
                store.fetch_device_ptrs(0, [0], dst_slabs=[0x1000])
            self.assertEqual(cm_dev.exception.code, pywex.WEX_E_DEVICE)

        # Idempotent close & closed-handle refusal
        self.assertTrue(store.is_closed)
        store.close()
        with self.assertRaises(pywex.WexError) as cm_closed:
            store.info()
        self.assertEqual(cm_closed.exception.code, pywex.WEX_E_ARGS)

    def test_root_mismatch_refused(self) -> None:
        bad_root = ("b" if self.root_hex[0] == "a" else "a") + self.root_hex[1:]
        with self.assertRaises(pywex.WexError) as cm:
            pywex.WexStore(self.wex_path, root_hex=bad_root, gpu_device=-2)
        self.assertEqual(cm.exception.code, pywex.WEX_E_ROOT)


class TestL2PrefetchAndCorruption(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wex_path, self.root_hex, self.slabs = build_raw_expert_wex4(
            Path(self._tmp.name) / "l2_test.wex4"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_l2_auto_hits_bypass_and_prefetch(self) -> None:
        # 1. l2_cache_bytes="auto" maps to UINT64_MAX -> positive capacity
        with pywex.WexStore(
            self.wex_path,
            l2_cache_bytes="auto",
            verify="on",
            gpu_device=-2,
        ) as store:
            info = store.info()
            self.assertGreater(info.l2_capacity_bytes, 0)

            # 1st fetch: 4 disk misses -> populates L2
            out0 = store.fetch_host(0, [0, 1, 2, 3])
            for e in range(4):
                self.assertEqual(bytes(out0[e]), self.slabs[(0, e)])
            st0 = store.stats()
            self.assertEqual(st0.records_from_disk, 4)
            self.assertEqual(st0.records_from_l2, 0)
            self.assertEqual(st0.l2_entries, 4)
            self.assertAlmostEqual(st0.l2_hit_rate, 0.0)

            # 2nd fetch: 4 L2 hits
            out1 = store.fetch_host(0, [0, 1, 2, 3])
            for e in range(4):
                self.assertEqual(bytes(out1[e]), self.slabs[(0, e)])
            st1 = store.stats()
            self.assertEqual(st1.records_from_disk, 4)
            self.assertEqual(st1.records_from_l2, 4)
            self.assertAlmostEqual(st1.l2_hit_rate, 0.5)

            # bypass_l2 does not add layer 1 entries into L2
            out_b = store.fetch_host(1, [0, 1], bypass_l2=True)
            self.assertEqual(bytes(out_b[0]), self.slabs[(1, 0)])
            st2 = store.stats()
            self.assertEqual(st2.l2_entries, 4)
            self.assertEqual(st2.records_from_disk, 6)

        # 2. Background prefetch warming L2
        with pywex.WexStore(
            self.wex_path,
            l2_cache_bytes=2 * 1024 * 1024,
            verify="on",
            gpu_device=-2,
        ) as store:
            queued = store.prefetch(0, [0, 1, 2, 3])
            self.assertEqual(queued, 4)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if store.stats().l2_entries >= 4:
                    break
                time.sleep(0.01)
            self.assertGreaterEqual(store.stats().l2_entries, 4)

            got = store.fetch_host(0, [0, 1, 2, 3])
            for e in range(4):
                self.assertEqual(bytes(got[e]), self.slabs[(0, e)])
            st = store.stats()
            self.assertEqual(st.records_from_l2, 4)
            self.assertEqual(st.records_from_disk, 0)
            self.assertEqual(st.prefetch_hits, 4)
            self.assertAlmostEqual(st.l2_hit_rate, 1.0)

    def test_corruption_hash_then_refused(self) -> None:
        corrupt_path = Path(self._tmp.name) / "corrupt_raw.wex4"
        shutil.copyfile(self.wex_path, corrupt_path)

        with pywex.WexStore(corrupt_path, gpu_device=-2) as probe:
            lay = probe.record_layout(0, 0)
            rec_off = lay.file_offset
            dec_len = lay.decoded_bytes

        # Flip 1 bit inside record 0 payload on disk
        with open(corrupt_path, "r+b") as fp:
            fp.seek(rec_off + 256)
            b = fp.read(1)
            fp.seek(rec_off + 256)
            fp.write(bytes([b[0] ^ 0x01]))

        with pywex.WexStore(
            corrupt_path,
            verify="on",
            l2_cache_bytes=1024 * 1024,
            gpu_device=-2,
        ) as store:
            sentinel = bytearray(b"\xA5" * dec_len)
            with self.assertRaises(pywex.WexError) as cm1:
                store.fetch_host(0, [0], dst_buffers=[sentinel], scratch=False)
            self.assertEqual(cm1.exception.code, pywex.WEX_E_HASH)
            self.assertEqual(sentinel, bytearray(b"\xA5" * dec_len))

            with self.assertRaises(pywex.WexError) as cm2:
                store.fetch_host(0, [0], dst_buffers=[sentinel], scratch=False)
            self.assertEqual(cm2.exception.code, pywex.WEX_E_REFUSED)

            # Sibling record 1 still succeeds
            ok_bufs = store.fetch_host(0, [1])
            self.assertEqual(bytes(ok_bufs[0]), self.slabs[(0, 1)])
            self.assertGreaterEqual(store.stats().refused, 2)


class TestWnr3AndTorchExt(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raw_path, self.raw_root, self.raw_slabs = build_raw_expert_wex4(
            Path(self._tmp.name) / "torch_raw.wex4"
        )
        self.wnr3_vec = VECTORS_DIR / "wnr3-one-expert.wex4"
        self.wnr3_cz_vec = VECTORS_DIR / "tiny2-cz-upgraded.wex4"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_wnr3_vector_cpu_and_corrupt_verify_off(self) -> None:
        if not self.wnr3_vec.is_file():
            self.skipTest("wex4/vectors/valid/wnr3-one-expert.wex4 not present")

        with pywex.WexStore(self.wnr3_vec, verify="on", gpu_device=-2) as store:
            info = store.info()
            self.assertEqual(info.codec_kind, pywex.WEX_CODEC_WNR3)
            self.assertEqual(info.codec_name, "WNR3")
            lay = store.record_layout(0, 0)
            self.assertTrue(lay.is_expert)
            cpu_buf = bytes(store.fetch_host(0, [0])[0])
            self.assertEqual(len(cpu_buf), lay.decoded_bytes)

        # Corrupt 1 bit in a copy and verify WEX_VERIFY_OFF catches it via Level-1 stream hash
        bad_copy = Path(self._tmp.name) / "bad_wnr3.wex4"
        shutil.copyfile(self.wnr3_vec, bad_copy)
        with open(bad_copy, "r+b") as fp:
            fp.seek(lay.file_offset + lay.stored_bytes // 2)
            orig = fp.read(1)
            fp.seek(lay.file_offset + lay.stored_bytes // 2)
            fp.write(bytes([orig[0] ^ 0x04]))

        with pywex.WexStore(bad_copy, verify="off", gpu_device=-2) as store:
            sentinel = bytearray(b"\x5A" * lay.decoded_bytes)
            with self.assertRaises(pywex.WexError) as cm:
                store.fetch_host(0, [0], dst_buffers=[sentinel], scratch=False)
            self.assertEqual(cm.exception.code, pywex.WEX_E_DECODE)
            self.assertEqual(sentinel, bytearray(b"\x5A" * lay.decoded_bytes))

    def test_wnr3_gpu_host_decode_if_cuda_helper(self) -> None:
        if not self.wnr3_vec.is_file():
            self.skipTest("wex4/vectors/valid/wnr3-one-expert.wex4 not present")
        cuda_helper = os.environ.get("WEX_CUDA_LIB")
        with pywex.WexStore(self.wnr3_vec, verify="on", gpu_device=-2) as cpu_store:
            cpu_bytes = bytes(cpu_store.fetch_host(0, [0])[0])
        with pywex.WexStore(
            self.wnr3_vec,
            verify="on",
            l2_cache_bytes=8 * 1024 * 1024,
            l2_pin_gpu=True,
            gpu_device=0,
            gpu_helper_path=cuda_helper,
        ) as gpu_store:
            info = gpu_store.info()
            if info.backend_name != "cuda":
                self.skipTest(f"libwex CUDA helper not active (backend={info.backend_name})")
            gpu_bytes_miss = bytes(gpu_store.fetch_host(0, [0], scratch=False)[0])
            gpu_bytes_l2 = bytes(gpu_store.fetch_host(0, [0], scratch=True)[0])
            self.assertEqual(gpu_bytes_miss, cpu_bytes)
            self.assertEqual(gpu_bytes_l2, cpu_bytes)
            st = gpu_store.stats()
            self.assertEqual(st.gpu_decodes, 2)
            self.assertEqual(st.records_from_disk, 1)
            self.assertEqual(st.records_from_l2, 1)

    def test_torch_lazy_import_behavior(self) -> None:
        from pywex import torch_ext

        try:
            import torch  # noqa: F401
        except ImportError:
            with pywex.WexStore(self.raw_path, gpu_device=-2) as store:
                with self.assertRaises(ImportError):
                    torch_ext.fetch_host_tensors(store, 0, [0])

    def test_torch_host_tensors_and_validation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        from pywex import torch_ext

        with pywex.WexStore(self.raw_path, gpu_device=-2) as store:
            tensors = torch_ext.fetch_host_tensors(store, 0, [0, 3])
            self.assertEqual(len(tensors), 2)
            self.assertEqual(tensors[0].dtype, torch.uint8)
            self.assertEqual(tensors[0].device.type, "cpu")
            self.assertEqual(tensors[0].numpy().tobytes(), self.raw_slabs[(0, 0)])
            self.assertEqual(tensors[1].numpy().tobytes(), self.raw_slabs[(0, 3)])

            # Zero-copy dtype view (int16)
            i16 = torch_ext.fetch_host_tensors(store, 1, [1], dtype=torch.int16)
            self.assertEqual(i16[0].dtype, torch.int16)
            self.assertEqual(i16[0].view(torch.uint8).numpy().tobytes(), self.raw_slabs[(1, 1)])

            # Validation: CPU tensor rejected by fetch_to_cuda
            cpu_t = torch.empty((16,), dtype=torch.uint8)
            with self.assertRaises(ValueError):
                torch_ext.fetch_to_cuda(store, 0, [0], dst_slabs=[cpu_t])

    def test_torch_cuda_direct_to_vram_live(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA not available in this interpreter")

        vectors = [p for p in (self.wnr3_vec, self.wnr3_cz_vec) if p.is_file()]
        if not vectors:
            self.skipTest("No WNR3 expert vectors present")

        from pywex import torch_ext

        cuda_helper = os.environ.get("WEX_CUDA_LIB")
        for vec_path in vectors:
            with pywex.WexStore(
                vec_path,
                verify="on",
                l2_cache_bytes=16 * 1024 * 1024,
                l2_pin_gpu=True,
                gpu_device=0,
                gpu_helper_path=cuda_helper,
            ) as store:
                info = store.info()
                if info.backend_name != "cuda":
                    self.skipTest(f"libwex CUDA backend not loaded (active={info.backend_name})")

                lay = store.record_layout(0, 0)
                self.assertTrue(lay.is_expert)

                # Reference decode via host path
                ref_slab = bytes(store.fetch_host(0, [0])[0])
                self.assertEqual(len(ref_slab), lay.decoded_bytes)

                # 1. Direct-to-VRAM raw uint8 E8M0 scales + packed FP4 weights (default stream)
                scales_u8, weights_u8 = torch_ext.allocate_expert_cuda_buffers(
                    store, 0, 0, device="cuda:0", expand_e8m0_f32=False
                )
                torch_ext.fetch_to_cuda(
                    store,
                    0,
                    [0],
                    dst_scales=scales_u8,
                    dst_weights=weights_u8,
                    expand_e8m0_f32=False,
                )
                torch.cuda.synchronize()

                for m in range(3):
                    s_nbytes = lay.scale_shapes[m][0] * lay.scale_shapes[m][1]
                    w_nbytes = lay.weight_shapes[m][0] * lay.weight_shapes[m][1]
                    exp_s = ref_slab[lay.scale_offsets[m] : lay.scale_offsets[m] + s_nbytes]
                    exp_w = ref_slab[lay.weight_offsets[m] : lay.weight_offsets[m] + w_nbytes]
                    self.assertEqual(scales_u8[m].cpu().numpy().tobytes(), exp_s)
                    self.assertEqual(weights_u8[m].cpu().numpy().tobytes(), exp_w)

                # 2. Direct-to-VRAM with in-VRAM E8M0 -> IEEE-754 float32 expansion on custom stream
                user_stream = torch.cuda.Stream(device="cuda:0")
                scales_f32, weights_u8_b = torch_ext.allocate_expert_cuda_buffers(
                    store, 0, 0, device="cuda:0", expand_e8m0_f32=True
                )
                self.assertEqual(scales_f32[0].dtype, torch.float32)

                with torch.cuda.stream(user_stream):
                    torch_ext.fetch_to_cuda(
                        store,
                        0,
                        [0],
                        dst_scales=[scales_f32],
                        dst_weights=[weights_u8_b],
                        stream=user_stream,
                        expand_e8m0_f32=True,
                    )
                user_stream.synchronize()

                for m in range(3):
                    s_nbytes = lay.scale_shapes[m][0] * lay.scale_shapes[m][1]
                    w_nbytes = lay.weight_shapes[m][0] * lay.weight_shapes[m][1]
                    exp_s = ref_slab[lay.scale_offsets[m] : lay.scale_offsets[m] + s_nbytes]
                    exp_w = ref_slab[lay.weight_offsets[m] : lay.weight_offsets[m] + w_nbytes]
                    got_f32_bytes = scales_f32[m].cpu().numpy().tobytes()
                    exp_f32_bytes = b"".join(
                        struct.pack(
                            "<f",
                            float("inf") if e == 255 else math.ldexp(1.0, e - 127),
                        )
                        for e in exp_s
                    )
                    self.assertEqual(got_f32_bytes, exp_f32_bytes)
                    self.assertEqual(weights_u8_b[m].cpu().numpy().tobytes(), exp_w)

                st = store.stats()
                self.assertGreaterEqual(st.gpu_decodes, 2)
                self.assertGreater(st.l2_hit_rate, 0.0)

    def test_optional_real_container_env(self) -> None:
        real_path = os.environ.get("WEX_REAL_CONTAINER", "").strip()
        if not real_path:
            self.skipTest("WEX_REAL_CONTAINER not set")
        with pywex.WexStore(
            real_path,
            l2_cache_bytes=256 * 1024 * 1024,
            gpu_helper_path=os.environ.get("WEX_CUDA_LIB"),
        ) as store:
            info = store.info()
            self.assertGreater(info.num_records, 0)
            lay = store.record_layout(0, 0, by_record_id=True)
            buf = store.fetch_host(0, [0], by_record_id=True)[0]
            self.assertEqual(len(buf), lay.decoded_bytes)


class TestAdaptersStrataAndHermes(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wex_path, self.root_hex, self.slabs = build_raw_expert_wex4(
            Path(self._tmp.name) / "adapter_test.wex4"
        )
        self.server, self.thread, self.base_url = _start_mock_server()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def test_strata_bridge_http_and_ipc(self) -> None:
        with StrataWexBridge(
            self.wex_path,
            strata_url=self.base_url,
            l2_cache_bytes=2 * 1024 * 1024,
            gpu_device=-2,
        ) as bridge:
            health = bridge.strata_health()
            self.assertTrue(health["ok"])
            self.assertEqual(health["data"]["engine"], "strata-mock")

            models = bridge.strata_models()
            self.assertTrue(models["ok"])
            self.assertEqual(models["data"]["data"][0]["id"], "mock-moe-7b")

            tele = bridge.telemetry()
            self.assertTrue(tele["ok"])
            self.assertTrue(tele["wex"]["attached"])
            self.assertEqual(tele["wex"]["info"]["layers"], 2)
            self.assertTrue(tele["strata"]["health"]["ok"])
            self.assertEqual(tele["strata"]["telemetry"]["data"]["tokens_per_sec"], 128.5)

            ipc_in = io.StringIO(
                "\n".join(
                    [
                        json.dumps({"op": "info"}),
                        json.dumps({"op": "has_expert", "layer": 0, "eid": 2}),
                        json.dumps({"op": "prefetch", "layer": 0, "eids": [0, 1]}),
                        json.dumps({"op": "stats"}),
                        json.dumps({"op": "strata_health"}),
                        "not valid json",
                        json.dumps({"op": "unknown_op"}),
                        json.dumps({"op": "quit"}),
                        json.dumps({"op": "info"}),  # must not run after quit
                    ]
                )
                + "\n"
            )
            ipc_out = io.StringIO()
            count = bridge.serve_ipc(stdin=ipc_in, stdout=ipc_out)
            self.assertEqual(count, 8)
            responses = [json.loads(line) for line in ipc_out.getvalue().strip().splitlines()]
            self.assertEqual(len(responses), 8)
            self.assertTrue(responses[0]["ok"])
            self.assertEqual(responses[0]["info"]["num_records"], 8)
            self.assertTrue(responses[1]["present"])
            self.assertEqual(responses[2]["queued"], 2)
            self.assertTrue(responses[3]["ok"])
            self.assertTrue(responses[4]["ok"])
            self.assertFalse(responses[5]["ok"])
            self.assertFalse(responses[6]["ok"])
            self.assertTrue(responses[7]["bye"])

    def test_hermes_provider_and_tool_dispatch(self) -> None:
        tools = get_hermes_tools()
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(
            names,
            ["wex_info", "wex_stats", "wex_prefetch", "wex_verify_expert", "wex_layout"],
        )

        # Bound local WexStore provider
        with WexExpertProvider(
            self.wex_path,
            verify="on",
            l2_cache_bytes=2 * 1024 * 1024,
            gpu_device=-2,
        ) as provider:
            r_info = dispatch_tool("wex_info", provider=provider)
            self.assertTrue(r_info["ok"])
            self.assertEqual(r_info["result"]["experts"], 4)

            r_lay = dispatch_tool("wex_layout", {"layer": 1, "eid": 2}, provider=provider)
            self.assertTrue(r_lay["ok"])
            self.assertEqual(r_lay["result"]["record_index"], 6)

            r_pf = dispatch_tool("wex_prefetch", {"layer": 0, "eids": [0, 1]}, provider=provider)
            self.assertTrue(r_pf["ok"])
            self.assertEqual(r_pf["result"]["queued"], 2)

            r_ver = dispatch_tool("wex_verify_expert", {"layer": 1, "eid": 0}, provider=provider)
            self.assertTrue(r_ver["ok"])
            expected_sha = hashlib.sha256(self.slabs[(1, 0)]).hexdigest()
            self.assertTrue(r_ver["result"]["verified"])
            self.assertEqual(r_ver["result"]["decoded_sha256"], expected_sha)

            r_st = dispatch_tool("wex_stats", provider=provider)
            self.assertTrue(r_st["ok"])
            self.assertGreaterEqual(r_st["result"]["fetches"], 1)

            r_bad = dispatch_tool("nonexistent_tool", provider=provider)
            self.assertFalse(r_bad["ok"])

        # Stateless container_path dispatch (auto open/close)
        r_stateless = dispatch_tool("wex_info", {"container_path": str(self.wex_path)})
        self.assertTrue(r_stateless["ok"])
        self.assertEqual(r_stateless["result"]["root_hex"], self.root_hex)

        # HTTP wex serve provider mode
        http_prov = WexExpertProvider(base_url=self.base_url)
        self.assertEqual(dispatch_tool("wex_info", provider=http_prov)["result"]["codec_name"], "WNR4-MX")
        self.assertEqual(dispatch_tool("wex_stats", provider=http_prov)["result"]["fetches"], 12)
        pf_http = dispatch_tool("wex_prefetch", {"layer": 0, "eids": [0, 2, 5]}, provider=http_prov)
        self.assertTrue(pf_http["ok"])
        self.assertEqual(pf_http["result"]["queued"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
