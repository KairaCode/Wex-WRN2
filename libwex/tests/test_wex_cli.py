#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""End-to-end stdlib unittest suite for wex_cli.py and wex_mcp.py."""

import contextlib
import hashlib
import http.client
import http.server
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LIBWEX_DIR = os.path.dirname(TESTS_DIR)
CLI_DIR = os.path.join(LIBWEX_DIR, "cli")
CLI_SCRIPT = os.path.join(CLI_DIR, "wex_cli.py")
MCP_SCRIPT = os.path.join(CLI_DIR, "wex_mcp.py")

if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import wex_cli  # noqa: E402
import wex_mcp  # noqa: E402

WEX4_MAGIC = 0x57455834
WEX4_PAGE = 4096
WEX4_HEADER_HASHED = 4064
WEX4_RECORD_BYTES = 72
WEX4_TENSOR_BYTES = 96
WEX4_CODEC_RAW = 0
WEX4_REC_EXPERT = 2
WEX4_DT_I8 = 3
WEX4_DT_F8_E8M0 = 6

EXP_ROWS = 64
EXP_SCOLS = 8
EXP_WCOLS = EXP_SCOLS * 16
EXP_SC_MAT = EXP_ROWS * EXP_SCOLS
EXP_WT_MAT = EXP_ROWS * EXP_WCOLS
EXP_SC_BYTES = 3 * EXP_SC_MAT
EXP_WT_BYTES = 3 * EXP_WT_MAT
EXP_DEC_BYTES = EXP_SC_BYTES + EXP_WT_BYTES
EXP_STORED_BYTES = (EXP_DEC_BYTES + WEX4_PAGE - 1) & ~(WEX4_PAGE - 1)


def _align4k(n):
    return (n + WEX4_PAGE - 1) & ~(WEX4_PAGE - 1)


def build_synthetic_raw_wex4(path, layers=2, experts=4):
    """Write a valid WEX4 RAW container with layers * experts records.

    Returns (root_hex, first_record_offset).
    """
    records = []
    tensors = []
    name_pool = bytearray()
    name_spans = []

    for lyr in range(layers):
        for eid in range(experts):
            rec_idx = len(records)
            slab = bytearray(EXP_STORED_BYTES)
            # Deterministic non-zero payload within decoded region; trailing pad is zero
            seed = (lyr * 131 + eid * 17 + 7) & 0xFF
            for i in range(EXP_DEC_BYTES):
                slab[i] = (seed + i * 31) & 0xFF

            first_t = len(tensors)
            for m in range(6):
                is_scale = m < 3
                mat = (m % 3) + 1
                suffix = "scale" if is_scale else "weight"
                tname = f"layers.{lyr}.ffn.experts.{eid}.w{mat}.{suffix}".encode("utf-8")
                nat = len(name_pool)
                name_pool.extend(tname)
                name_spans.append((nat, len(tname), tname))

                shape0 = EXP_ROWS
                shape1 = EXP_SCOLS if is_scale else EXP_WCOLS
                toff = (m * EXP_SC_MAT) if is_scale else (EXP_SC_BYTES + (m - 3) * EXP_WT_MAT)
                nbytes = shape0 * shape1
                tensors.append({
                    "name_at": nat,
                    "name_len": len(tname),
                    "dtype": WEX4_DT_F8_E8M0 if is_scale else WEX4_DT_I8,
                    "rank": 2,
                    "shape": [shape0, shape1, 0, 0, 0, 0, 0, 0],
                    "record": rec_idx,
                    "record_count": 1,
                    "offset": toff,
                    "nbytes": nbytes,
                })

            records.append({
                "stored_data": bytes(slab),
                "stored_bytes": EXP_STORED_BYTES,
                "decoded_bytes": EXP_DEC_BYTES,
                "codec": WEX4_CODEC_RAW,
                "flags": WEX4_REC_EXPERT,
                "blob": 0,
                "owner": (lyr << 16) | eid,
                "first_tensor": first_t,
                "tensor_count": 6,
            })

    nrecs = len(records)
    ntensors = len(tensors)
    names_bytes = len(name_pool)
    total_decoded = sum(r["decoded_bytes"] for r in records)
    max_decoded = max((r["decoded_bytes"] for r in records), default=0)

    order = sorted(range(ntensors), key=lambda idx: name_spans[idx][2])

    pos = WEX4_PAGE
    rec_offsets = []
    for r in records:
        rec_offsets.append(pos)
        pos += r["stored_bytes"]

    index_offset = pos
    index_bytes = (
        nrecs * WEX4_RECORD_BYTES
        + ntensors * (WEX4_TENSOR_BYTES + 4)
        + names_bytes
    )
    index_len = _align4k(index_bytes)
    file_bytes = index_offset + index_len

    buf = bytearray(file_bytes)

    for r, off in zip(records, rec_offsets):
        buf[off : off + r["stored_bytes"]] = r["stored_data"]

    idx = bytearray()
    for r, off in zip(records, rec_offsets):
        digest = hashlib.sha256(r["stored_data"]).digest()
        idx.extend(
            struct.pack(
                "<QIIHHIIII",
                off,
                r["stored_bytes"],
                r["decoded_bytes"],
                r["codec"],
                r["flags"],
                r["blob"],
                r["owner"],
                r["first_tensor"],
                r["tensor_count"],
            )
        )
        idx.extend(digest)
        idx.extend(b"\x00\x00\x00\x00")  # 68..72 reserved zero padding

    for ten in tensors:
        idx.extend(struct.pack("<IHBB", ten["name_at"], ten["name_len"], ten["dtype"], ten["rank"]))
        idx.extend(struct.pack("<8Q", *ten["shape"]))
        idx.extend(struct.pack("<IIQQ", ten["record"], ten["record_count"], ten["offset"], ten["nbytes"]))

    for tidx in order:
        idx.extend(struct.pack("<I", tidx))

    idx.extend(name_pool)
    assert len(idx) == index_bytes
    buf[index_offset : index_offset + index_bytes] = idx

    index_sha = hashlib.sha256(idx).digest()
    hdr = struct.pack(
        "<IHHQQQQQIIIIIIQQQ",
        WEX4_MAGIC,
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
        max_decoded,
        names_bytes,
        0,
        total_decoded,
    )
    assert len(hdr) == 96
    buf[0:96] = hdr
    buf[96:128] = index_sha

    root_digest = hashlib.sha256(buf[:WEX4_HEADER_HASHED]).digest()
    buf[WEX4_HEADER_HASHED : WEX4_HEADER_HASHED + 32] = root_digest

    with open(path, "wb") as fh:
        fh.write(buf)

    return root_digest.hex(), rec_offsets[0]


def _http_request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        hdrs = dict(headers or {})
        payload = None
        if body is not None:
            if isinstance(body, (dict, list)):
                payload = json.dumps(body).encode("utf-8")
                hdrs.setdefault("Content-Type", "application/json")
            elif isinstance(body, str):
                payload = body.encode("utf-8")
            else:
                payload = body
            hdrs["Content-Length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


class MockUpstreamHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    get_paths = []

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        # Every GET redirects, so a client that follows redirects shows up as "/landed"
        MockUpstreamHandler.get_paths.append(self.path)
        self.send_response(302)
        self.send_header("Location", "/landed")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self):
        clen = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(clen) if clen > 0 else b"{}"
        req = json.loads(raw.decode("utf-8"))

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                b'data: {"id":"cmpl-1","choices":[{"delta":{"content":"hello "}}]}\n\n',
                b'data: {"id":"cmpl-1","choices":[{"delta":{"content":"libwex"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            for ch in chunks:
                self.wfile.write(ch)
                self.wfile.flush()
            return

        payload = json.dumps({
            "id": "mock-resp-1",
            "path": self.path,
            "echo_model": req.get("model", ""),
            "content": "proxied-ok",
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


class TestWexCliAndMcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.wex_path = os.path.join(cls.tmpdir.name, "synth_raw.wex4")
        cls.root_hex, cls.first_rec_off = build_synthetic_raw_wex4(cls.wex_path, layers=2, experts=4)
        cls.env = os.environ.copy()
        cls.env["PYTHONDONTWRITEBYTECODE"] = "1"

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _run_cli(self, *args):
        cmd = [sys.executable, "-B", "-I", CLI_SCRIPT, *args]
        return subprocess.run(
            cmd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

    def test_inspect_json_and_text(self):
        p = self._run_cli("inspect", self.wex_path, "--cpu-only", "--output", "json")
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertEqual(data["abi_version"], 1)
        self.assertEqual(data["layers"], 2)
        self.assertEqual(data["experts"], 4)
        self.assertEqual(data["num_records"], 8)
        self.assertEqual(data["num_tensors"], 48)
        self.assertEqual(data["codec_kind"], 0)
        self.assertEqual(data["codec_name"], "RAW")
        self.assertEqual(data["root_hex"], self.root_hex)
        self.assertGreater(data["file_bytes"], 0)
        self.assertEqual(data["total_decoded_bytes"], 8 * EXP_DEC_BYTES)
        self.assertTrue(data["io_engine"])
        self.assertTrue(data["backend_name"].startswith("cpu-"))

        p_txt = self._run_cli("--output", "text", "inspect", self.wex_path, "--cpu-only")
        self.assertEqual(p_txt.returncode, 0, p_txt.stderr)
        self.assertIn(self.root_hex, p_txt.stdout)
        self.assertIn("RAW", p_txt.stdout)

    def test_verify_ok_root_pin_and_corrupt(self):
        p = self._run_cli("verify", self.wex_path, "--root", self.root_hex, "--cpu-only", "--output", "json")
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertTrue(data["ok"])
        self.assertTrue(data["verified_root"])
        self.assertEqual(data["records_total"], 8)
        self.assertEqual(data["records_verified"], 8)
        self.assertEqual(data["hash_errors"], 0)
        self.assertGreater(data["effective_gib_s"], 0.0)

        # Wrong root hex must fail non-zero
        bad_root = ("b" if self.root_hex[0] == "a" else "a") + self.root_hex[1:]
        p_bad = self._run_cli("verify", self.wex_path, "--root", bad_root, "--cpu-only", "--output", "json")
        self.assertNotEqual(p_bad.returncode, 0)

        # Corrupt 1 bit inside record 0 payload -> must fail with hash error
        corrupt_path = os.path.join(self.tmpdir.name, "corrupt_raw.wex4")
        with open(self.wex_path, "rb") as src:
            raw = bytearray(src.read())
        raw[self.first_rec_off + 128] ^= 0x01
        with open(corrupt_path, "wb") as dst:
            dst.write(raw)

        p_cor = self._run_cli("verify", corrupt_path, "--cpu-only", "--output", "json")
        self.assertNotEqual(p_cor.returncode, 0)
        cor_data = json.loads(p_cor.stdout)
        self.assertFalse(cor_data["ok"])
        self.assertGreaterEqual(cor_data["hash_errors"], 1)
        self.assertEqual(cor_data["records_verified"], 7)

    def test_bench_l2_sweep(self):
        p = self._run_cli(
            "bench",
            self.wex_path,
            "--l2-mb",
            "0,16",
            "--waves",
            "16",
            "--batch",
            "2",
            "--pattern",
            "zipf",
            "--verify",
            "1",
            "--cpu-only",
            "--seed",
            "42",
            "--output",
            "json",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertEqual(len(data["results"]), 2)
        r0, r16 = data["results"]
        self.assertEqual(r0["l2_mb"], 0)
        self.assertEqual(r0["records_from_l2"], 0)
        self.assertEqual(r0["failures"], 0)
        self.assertEqual(r16["l2_mb"], 16)
        self.assertGreater(r16["records_from_l2"], 0)
        self.assertGreater(r16["l2_hit_pct"], 0.0)
        self.assertEqual(r16["failures"], 0)
        self.assertGreater(r16["cpu_decodes"], 0)

        p_txt = self._run_cli(
            "bench",
            self.wex_path,
            "--l2-mb",
            "0,16",
            "--waves",
            "8",
            "--batch",
            "2",
            "--pattern",
            "seq",
            "--cpu-only",
        )
        self.assertEqual(p_txt.returncode, 0, p_txt.stderr)
        self.assertIn("Eff(GB/s)", p_txt.stdout)

    def test_ipc_protocol(self):
        proc = subprocess.Popen(
            [sys.executable, "-B", "-I", CLI_SCRIPT, "ipc", self.wex_path, "--l2-mb", "16", "--cpu-only"],
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cmds = [
            {"cmd": "info"},
            {"cmd": "has_expert", "layer": 0, "eid": 1},
            {"cmd": "has_expert", "layer": 9, "eid": 0},
            {"cmd": "prefetch", "layer": 0, "eids": [0, 1]},
            {"cmd": "stats"},
            {"cmd": "layout", "layer": 0, "eid": 0},
            {"cmd": "no_such_cmd"},
            {"cmd": "quit"},
        ]
        payload = "\n".join(json.dumps(c) for c in cmds) + "\n"
        out, err = proc.communicate(payload, timeout=20)
        self.assertEqual(proc.returncode, 0, err)

        lines = [json.loads(line) for line in out.strip().splitlines() if line.strip()]
        self.assertEqual(len(lines), len(cmds))
        self.assertTrue(lines[0]["ok"])
        self.assertEqual(lines[0]["info"]["num_records"], 8)
        self.assertTrue(lines[1]["ok"] and lines[1]["present"] is True)
        self.assertTrue(lines[2]["ok"] and lines[2]["present"] is False)
        self.assertTrue(lines[3]["ok"] and lines[3]["queued"] == 2)
        self.assertTrue(lines[4]["ok"] and "fetches" in lines[4]["stats"])
        self.assertTrue(lines[5]["ok"] and lines[5]["layout"]["tensor_count"] == 6)
        self.assertFalse(lines[6]["ok"])
        self.assertTrue(lines[7]["ok"] and lines[7]["bye"] is True)

    def test_serve_standalone_and_upstream_proxy(self):
        # 1. Standalone server on ephemeral port 0 (no --upstream)
        srv, store = wex_cli.create_wex_http_server(
            self.wex_path,
            host="127.0.0.1",
            port=0,
            l2_mb="16",
            verify="on",
            root_hex=self.root_hex,
            cpu_only=True,
        )
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        th.start()
        try:
            st, _, body = _http_request(port, "GET", "/health")
            self.assertEqual(st, 200)
            health = json.loads(body)
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["abi_version"], 1)

            st, _, body = _http_request(port, "GET", "/status")
            self.assertEqual(st, 200)
            status_obj = json.loads(body)
            self.assertEqual(status_obj["info"]["num_records"], 8)
            self.assertIn("l2_hit_ratio", status_obj)

            st, _, body = _http_request(port, "GET", "/v1/models")
            self.assertEqual(st, 200)
            models = json.loads(body)
            self.assertEqual(models["object"], "list")
            self.assertEqual(models["data"][0]["owned_by"], "libwex")
            self.assertEqual(models["data"][0]["wex"]["root_hex"], self.root_hex)

            st, hdrs, body = _http_request(port, "POST", "/v1/prefetch", {"layer": 0, "eids": [0, 1, 2]})
            self.assertEqual(st, 200)
            pf = json.loads(body)
            self.assertEqual(pf["queued"], 3)
            self.assertIn("x-wex-l2-hit-rate", hdrs)

            # Give background prefetch worker a brief moment to populate L2
            for _ in range(40):
                _, _, tbody = _http_request(port, "GET", "/v1/telemetry")
                tel = json.loads(tbody)
                if tel["l2"]["entries"] >= 3:
                    break
                time.sleep(0.02)

            st, _, body = _http_request(port, "GET", "/metrics")
            self.assertEqual(st, 200)
            metrics_text = body.decode("utf-8")
            for metric in (
                "wex_fetches_total",
                "wex_l2_hits_total",
                "wex_disk_reads_total",
                "wex_l2_hit_ratio",
                "wex_disk_bytes_total",
                "wex_decoded_bytes_total",
                "wex_gpu_decodes_total",
                "wex_cpu_decodes_total",
                "wex_io_seconds_total",
                "wex_decode_seconds_total",
                "wex_verify_seconds_total",
                "wex_l2_used_bytes",
                "wex_l2_capacity_bytes",
            ):
                self.assertIn(metric, metrics_text)

            st, _, body = _http_request(port, "GET", "/v1/telemetry")
            self.assertEqual(st, 200)
            tel = json.loads(body)
            self.assertEqual(tel["store"]["num_records"], 8)
            self.assertGreaterEqual(tel["l2"]["entries"], 3)

            # A request sent by a web page of another site (foreign Origin) is refused
            foreign = {"Origin": "https://evil.example"}
            st, _, _ = _http_request(port, "POST", "/v1/prefetch", {"layer": 0, "eids": [0]}, headers=foreign)
            self.assertEqual(st, 403)
            st, _, _ = _http_request(port, "GET", "/status", headers=foreign)
            self.assertEqual(st, 403)
            st, _, _ = _http_request(
                port, "POST", "/v1/prefetch", {"layer": 0, "eids": [0]},
                headers={"Origin": f"http://127.0.0.1:{port}"},
            )
            self.assertEqual(st, 200)

            # Without --upstream, inference endpoints return 503 with instructions
            for endpoint in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
                st, hdrs, body = _http_request(port, "POST", endpoint, {"model": "synth_raw"})
                self.assertEqual(st, 503)
                self.assertIn("--upstream", body.decode("utf-8"))
                self.assertIn("x-wex-l2-hit-rate", hdrs)
        finally:
            srv.shutdown()
            srv.server_close()
            store.close()

        # 2. Proxy mode with mock upstream server (JSON + SSE streaming)
        mock_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
        mock_srv.daemon_threads = True
        mock_port = mock_srv.server_address[1]
        mock_th = threading.Thread(target=mock_srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        mock_th.start()

        proxy_srv, proxy_store = wex_cli.create_wex_http_server(
            self.wex_path,
            host="127.0.0.1",
            port=0,
            l2_mb="16",
            cpu_only=True,
            upstream=f"http://127.0.0.1:{mock_port}",
        )
        proxy_port = proxy_srv.server_address[1]
        proxy_th = threading.Thread(target=proxy_srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        proxy_th.start()
        try:
            client_creds = {"Authorization": "Bearer client-key", "X-Api-Key": "client-key", "Cookie": "sid=1"}
            for ep in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
                st, hdrs, body = _http_request(
                    proxy_port, "POST", ep, {"model": "synth_raw", "stream": False},
                    headers={**client_creds, "X-Trace": "keep"},
                )
                self.assertEqual(st, 200)
                self.assertIn("x-wex-l2-hit-rate", hdrs)
                resp_obj = json.loads(body)
                self.assertEqual(resp_obj["path"], ep)
                self.assertEqual(resp_obj["content"], "proxied-ok")
                # The client's credentials stay at the proxy; other headers pass through
                for cred in client_creds:
                    self.assertNotIn(cred.lower(), resp_obj["headers"])
                self.assertEqual(resp_obj["headers"].get("x-trace"), "keep")

            # SSE streaming pass-through
            st, hdrs, body = _http_request(
                proxy_port, "POST", "/v1/chat/completions", {"model": "synth_raw", "stream": True}
            )
            self.assertEqual(st, 200)
            self.assertIn("x-wex-l2-hit-rate", hdrs)
            self.assertIn("text/event-stream", hdrs.get("content-type", ""))
            text = body.decode("utf-8")
            self.assertIn("hello ", text)
            self.assertIn("libwex", text)
            self.assertIn("[DONE]", text)
        finally:
            proxy_srv.shutdown()
            proxy_srv.server_close()
            proxy_store.close()
            mock_srv.shutdown()
            mock_srv.server_close()

    def test_mcp_stdio_and_telemetry_tool(self):
        # Spin up a live wex serve instance to exercise wex_server_telemetry via MCP
        srv, store = wex_cli.create_wex_http_server(
            self.wex_path,
            host="127.0.0.1",
            port=0,
            l2_mb="16",
            cpu_only=True,
        )
        srv_port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        th.start()
        try:
            # 1. Newline-delimited JSON-RPC transport (MCP stdio default)
            msgs = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "wex_inspect", "arguments": {"path": self.wex_path, "cpu_only": True}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "wex_verify", "arguments": {"path": self.wex_path, "root_hex": self.root_hex, "cpu_only": True}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "wex_bench", "arguments": {"path": self.wex_path, "l2_mb": "0,16", "waves": 8, "batch": 2, "cpu_only": True}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "wex_server_telemetry", "arguments": {"port": srv_port, "endpoint": "telemetry"}},
                },
            ]
            proc = subprocess.Popen(
                [sys.executable, "-B", "-I", MCP_SCRIPT, "--root-dir", self.tmpdir.name],
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            raw_in = ("\n".join(json.dumps(m) for m in msgs) + "\n").encode("utf-8")
            raw_out, raw_err = proc.communicate(raw_in, timeout=25)
            self.assertEqual(proc.returncode, 0, raw_err.decode("utf-8", errors="replace"))

            replies = [json.loads(line) for line in raw_out.decode("utf-8").splitlines() if line.strip()]
            # 7 requests have ids (notification produces no reply)
            self.assertEqual(len(replies), 7)
            by_id = {r["id"]: r for r in replies}

            self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "libwex-mcp")
            self.assertEqual(by_id[2]["result"], {})

            tool_names = [t["name"] for t in by_id[3]["result"]["tools"]]
            self.assertEqual(tool_names, ["wex_inspect", "wex_verify", "wex_bench", "wex_server_telemetry"])

            ins_res = by_id[4]["result"]
            self.assertFalse(ins_res["isError"])
            ins_json = json.loads(ins_res["content"][0]["text"])
            self.assertEqual(ins_json["root_hex"], self.root_hex)
            self.assertEqual(ins_json["num_records"], 8)

            ver_res = by_id[5]["result"]
            self.assertFalse(ver_res["isError"])
            ver_json = json.loads(ver_res["content"][0]["text"])
            self.assertTrue(ver_json["ok"])
            self.assertEqual(ver_json["records_verified"], 8)

            bnc_res = by_id[6]["result"]
            self.assertFalse(bnc_res["isError"])
            bnc_json = json.loads(bnc_res["content"][0]["text"])
            self.assertEqual(len(bnc_json["results"]), 2)

            tel_res = by_id[7]["result"]
            self.assertFalse(tel_res["isError"])
            tel_json = json.loads(tel_res["content"][0]["text"])
            self.assertEqual(tel_json["store"]["root_hex"], self.root_hex)

            # 2. Content-Length framed JSON-RPC round-trip
            proc2 = subprocess.Popen(
                [sys.executable, "-B", "-I", MCP_SCRIPT],
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            framed_req = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}).encode("utf-8")
            framed_wire = f"Content-Length: {len(framed_req)}\r\n\r\n".encode("ascii") + framed_req
            out2, err2 = proc2.communicate(framed_wire, timeout=15)
            self.assertEqual(proc2.returncode, 0, err2.decode("utf-8", errors="replace"))
            self.assertTrue(out2.startswith(b"Content-Length:"))
            _, body2 = out2.split(b"\r\n\r\n", 1)
            self.assertEqual(json.loads(body2.decode("utf-8"))["id"], 99)
        finally:
            srv.shutdown()
            srv.server_close()
            store.close()

    @staticmethod
    def _mcp_call(server, name, arguments):
        reply = server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        return reply["result"]

    def test_mcp_rejects_library_path_argument(self):
        for tool in wex_mcp.TOOLS:
            for prop in tool["inputSchema"]["properties"]:
                self.assertNotIn("lib", prop, tool["name"])
                self.assertNotIn("cuda", prop, tool["name"])

        server = wex_mcp.WexMcpServer(root_dirs=[self.tmpdir.name])
        evil = os.path.join(self.tmpdir.name, "evil.dll")
        for name in ("wex_inspect", "wex_verify", "wex_bench"):
            for extra in ("lib_path", "cuda_path"):
                with mock.patch.object(wex_cli.ctypes, "CDLL") as cdll:
                    res = self._mcp_call(server, name, {"path": self.wex_path, "cpu_only": True, extra: evil})
                self.assertTrue(res["isError"], (name, extra))
                self.assertIn(extra, res["structuredContent"]["error"])
                cdll.assert_not_called()

    def test_mcp_path_outside_roots_refused(self):
        allowed = os.path.join(self.tmpdir.name, "mcp_allowed")
        outside = os.path.join(self.tmpdir.name, "mcp_outside")
        os.makedirs(allowed)
        os.makedirs(outside)
        inside_path = os.path.join(allowed, "in.wex4")
        outside_path = os.path.join(outside, "out.wex4")
        shutil.copyfile(self.wex_path, inside_path)
        shutil.copyfile(self.wex_path, outside_path)
        link = os.path.join(allowed, "link")
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows without the symlink privilege: a directory junction needs none
            import _winapi

            _winapi.CreateJunction(outside, link)

        server = wex_mcp.WexMcpServer(root_dirs=[allowed])
        res = self._mcp_call(server, "wex_inspect", {"path": inside_path, "cpu_only": True})
        self.assertFalse(res["isError"], res)
        self.assertEqual(res["structuredContent"]["root_hex"], self.root_hex)

        escapes = (
            outside_path,
            os.path.join(allowed, "..", "mcp_outside", "out.wex4"),
            os.path.join(link, "out.wex4"),
        )
        for path in escapes:
            for name in ("wex_inspect", "wex_verify", "wex_bench"):
                with mock.patch.object(wex_cli, "WexLib") as lib:
                    res = self._mcp_call(server, name, {"path": path, "cpu_only": True})
                self.assertTrue(res["isError"], (name, path))
                self.assertIn("outside the allowed root", res["structuredContent"]["error"])
                lib.assert_not_called()

        # With no --root-dir the only root is the server's working directory
        self.assertEqual(
            wex_mcp.WexMcpServer().root_dirs,
            [os.path.normcase(os.path.realpath(os.getcwd()))],
        )

    def test_serve_requires_token_for_non_loopback_bind(self):
        # The server class is stubbed so that a regression cannot open a public socket here
        with mock.patch.object(wex_cli, "WexHTTPServer") as server_cls:
            for host in ("0.0.0.0", "", "::", "192.0.2.1", "example.org"):
                with self.assertRaisesRegex(wex_cli.WexError, "--token"):
                    wex_cli.create_wex_http_server(self.wex_path, host=host, port=0, cpu_only=True)
            server_cls.assert_not_called()

            err = io.StringIO()
            with mock.patch.dict(os.environ), contextlib.redirect_stderr(err):
                os.environ.pop("WEX_SERVE_TOKEN", None)
                rc = wex_cli.main(["serve", self.wex_path, "--host", "0.0.0.0", "--port", "0", "--cpu-only"])
            self.assertEqual(rc, 1)
            self.assertIn("--token", err.getvalue())
            server_cls.assert_not_called()

            # With a token the same bind is allowed
            _, store = wex_cli.create_wex_http_server(
                self.wex_path, host="0.0.0.0", port=0, cpu_only=True, token="s3cret"
            )
            store.close()
            self.assertEqual(server_cls.call_args.kwargs["token"], "s3cret")

        with mock.patch.dict(os.environ, {"WEX_SERVE_TOKEN": "from-env"}):
            args = wex_cli.build_parser().parse_args(["serve", self.wex_path])
        self.assertEqual(args.token, "from-env")

    def test_serve_token_on_loopback(self):
        srv, store = wex_cli.create_wex_http_server(
            self.wex_path, host="127.0.0.1", port=0, l2_mb="16", cpu_only=True, token="s3cret"
        )
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        th.start()
        try:
            routes = (
                ("GET", "/health"),
                ("GET", "/v1/telemetry"),
                ("POST", "/v1/prefetch"),
                ("POST", "/v1/chat/completions"),
            )
            for method, route in routes:
                body = {"layer": 0, "eids": [0]} if method == "POST" else None
                for bad in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
                    st, hdrs, _ = _http_request(port, method, route, body, headers=bad)
                    self.assertEqual(st, 401, (method, route, bad))
                    self.assertEqual(hdrs.get("www-authenticate"), "Bearer")

            auth = {"Authorization": "Bearer s3cret"}
            st, _, body = _http_request(port, "GET", "/health", headers=auth)
            self.assertEqual(st, 200)
            self.assertEqual(json.loads(body)["status"], "ok")
            # A caller that knows the token is trusted whatever page it runs in
            st, _, body = _http_request(
                port, "POST", "/v1/prefetch", {"layer": 0, "eids": [0]},
                headers={**auth, "Origin": "https://dashboard.example"},
            )
            self.assertEqual(st, 200)

            # The MCP telemetry tool takes the token from the server's environment only,
            # and never follows a redirect that could carry it elsewhere
            redirector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockUpstreamHandler)
            threading.Thread(target=redirector.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
            try:
                with mock.patch.dict(os.environ):
                    os.environ.pop("WEX_SERVE_TOKEN", None)
                    with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                        wex_mcp.query_server_telemetry(port=port)
                    os.environ["WEX_SERVE_TOKEN"] = "s3cret"
                    tel = wex_mcp.query_server_telemetry(port=port)
                    with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                        wex_mcp.query_server_telemetry(port=redirector.server_address[1])
                self.assertEqual(tel["store"]["root_hex"], self.root_hex)
                self.assertEqual(MockUpstreamHandler.get_paths, ["/v1/telemetry"])
            finally:
                redirector.shutdown()
                redirector.server_close()
        finally:
            srv.shutdown()
            srv.server_close()
            store.close()

    def test_pull_resumable_and_verify(self):
        with open(self.wex_path, "rb") as fh:
            full_bytes = fh.read()
        split_at = len(full_bytes) // 2

        class FakeResp:
            def __init__(self, data, status):
                self._data = data
                self._pos = 0
                self.status = status

            def read(self, n=-1):
                if n < 0:
                    n = len(self._data) - self._pos
                out = self._data[self._pos : self._pos + n]
                self._pos += len(out)
                return out

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        seen_ranges = []

        def fake_urlopen(req, timeout=60):
            rng = req.get_header("Range")
            seen_ranges.append(rng)
            if rng and rng.startswith("bytes="):
                start = int(rng.split("=", 1)[1].split("-", 1)[0])
                return FakeResp(full_bytes[start:], 206)
            return FakeResp(full_bytes, 200)

        out_dir = os.path.join(self.tmpdir.name, "pull_out")
        os.makedirs(out_dir, exist_ok=True)
        part_path = os.path.join(out_dir, "model.wex4.part")
        with open(part_path, "wb") as fh:
            fh.write(full_bytes[:split_at])

        orig_urlopen = wex_cli.urllib.request.urlopen
        wex_cli.urllib.request.urlopen = fake_urlopen
        try:
            res = wex_cli.run_pull(
                "example-org/example-model",
                "weights/model.wex4",
                out_dir=out_dir,
                root_hex=self.root_hex,
                verify=True,
                cpu_only=True,
                quiet=True,
            )
        finally:
            wex_cli.urllib.request.urlopen = orig_urlopen

        self.assertTrue(res["resumed"])
        self.assertEqual(res["total_bytes"], len(full_bytes))
        self.assertEqual(res["downloaded_bytes"], len(full_bytes) - split_at)
        self.assertEqual(seen_ranges, [f"bytes={split_at}-"])
        self.assertTrue(res["verify"]["ok"])
        self.assertFalse(os.path.exists(part_path))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "model.wex4")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
