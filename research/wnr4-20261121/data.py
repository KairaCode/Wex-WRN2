# SPDX-License-Identifier: Apache-2.0
"""Real expert records for the WNR4 research, read-only on the containers.

Datasets:
  q38mx  Qwen3.8-Flash-Next MXFP4, RAW WEX4 container (24,576 records of 2,611,200 B)
  v41    DeepSeek V4.1 REAP-192 MXFP4, WNR3-CZ WEX4 container (7,680 records of 18,800,640 B);
         the source here is the WNR3 decode = Z(safetensors bytes), what the engine gets today
  q38nv  Qwen3.8-Flash-Next NVFP4, RAW WEX4 container (24,576 records of 2,764,800 B)

coded(): the coded domain every codec of this study starts from.
  MXFP4: wex_nr3_prepare (canonical zero when cz, then WNR3 ceil form), rule-g anchors of the coded scales
  NVFP4: canonical zero when cz (no ceil form), rule-g anchors of the scale bytes
"""
import ctypes as C
import os
import random
import struct
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
# Local paths: environment WNR4_Q38MX, WNR4_V41, WNR4_Q38NV (containers) and WNR4_NV_TREE (a checkout of the
# nvfp4-wnr3 branch, whose wex4.py knows NVFP4 slabs), else out/paths.json with the same keys (not tracked).
_LOCAL = {}
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'paths.json')):
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'paths.json')) as _f:
        _LOCAL = json.load(_f)
NVTREE = os.environ.get('WNR4_NV_TREE', _LOCAL.get('WNR4_NV_TREE', ''))
PATHS = {k: os.environ.get(e, _LOCAL.get(e, '')) for k, e in (('q38mx', 'WNR4_Q38MX'), ('v41', 'WNR4_V41'), ('q38nv', 'WNR4_Q38NV'))}
ENTRY = struct.Struct('<QIII6H6H3I3II')   # legacy WEX3 entry, 72 B


def _wex4(nv):
    sys.path.insert(0, os.path.join(NVTREE if nv else ROOT, 'wex4'))
    try:
        import importlib
        if 'wex4' in sys.modules:
            del sys.modules['wex4']
        return importlib.import_module('wex4')
    finally:
        sys.path.pop(0)


def rule_g(scales):
    h = [0] * 256
    for b in scales:
        h[b] += 1
    best, top = 0, 0
    for i in range(255):
        if h[i] + h[i + 1] > top:
            top, best = h[i] + h[i + 1], i
    return best


ZTAB = bytes(((b & 0xF0) if b & 15 == 8 else b) & (0x0F if b >> 4 == 8 else 0xFF) for b in range(256))


class Dataset:
    def __init__(self, name, path=None):
        self.name, self.nv = name, name.endswith('nv')
        self.path = path or PATHS[name]
        w = _wex4(self.nv)
        self.wex4 = w
        f = open(self.path, 'rb')   # kept open: the reader reads blobs through it
        self.rd = w.Reader(f)
        self.experts = [r for r, rec in enumerate(self.rd.records) if rec[4] & w.REC_EXPERT]
        e = self.rd.expert_entry(self.experts[0])
        self.scale_bytes, self.weight_bytes = e['scale_bytes'], e['weight_bytes']
        self.raw = e['scale_bytes'] + e['weight_bytes']
        self.geo = []
        for m in range(3):
            rows, cols = e['scale_shapes'][m]
            self.geo += [e['scale_offsets'][m], rows, cols, e['weight_offsets'][m]]
        self.geo_c = (C.c_uint32 * 12)(*self.geo)
        self.model = None
        codec = self.rd.records[self.experts[0]][3]
        self.codec = codec
        if codec == 5:
            self.model = self.rd.blob(self.rd.records[self.experts[0]][5] - 1)
        self.local = threading.local()
        self.tools = C.CDLL(os.path.join(HERE, 'out', 'wex_nr3_tools.dll'))
        P = C.c_void_p
        self.tools.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]
        self.tools.wex_nr3_decode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]

    def entry(self, r, codec=0, total=0):
        e = self.rd.expert_entry(r)
        return ENTRY.pack(0, total, e['scale_bytes'], e['weight_bytes'], *[d for s in e['scale_shapes'] for d in s],
                          *[d for s in e['weight_shapes'] for d in s], *e['scale_offsets'], *e['weight_offsets'], codec)

    def owner(self, r):
        return self.rd.records[r][6]

    def stored(self, r):
        return self.rd.records[r][1]

    def read(self, r):
        f = getattr(self.local, 'f', None)
        if f is None:
            f = self.local.f = open(self.path, 'rb')
        off, n = self.rd.records[r][:2]
        f.seek(off)
        data = f.read(n)
        assert len(data) == n
        return data

    def source(self, r):
        """Output-domain slab: RAW bytes, or the WNR3 decode (= Z(source)) for v41."""
        data = self.read(r)
        if self.codec == 0:
            return data[:self.raw]
        own = self.owner(r)
        out = C.create_string_buffer(self.raw)
        st = self.tools.wex_nr3_decode(self.model, self.entry(r, 5, len(data)), data, own >> 16, own & 0xffff, out)
        assert st == 0, (r, st)
        return out.raw

    def coded(self, src, cz=True):
        """(coded slab, anchors[3], flags). Flags: MX = WNR3 ceil mask | CZ 8; NV = CZ 8."""
        if not self.nv:
            out = C.create_string_buffer(self.raw)
            stats = (C.c_uint32 * 16)()
            ent = self.entry(self.experts[0])   # every expert of a container has the same geometry
            flags = self.tools.wex_nr3_prepare(ent, src, int(cz), out, stats)
            assert flags >= 0
            return out.raw, bytes(stats[9:12]), flags
        s = bytearray(src)
        if cz:
            s[self.scale_bytes:] = bytes(s[self.scale_bytes:]).translate(ZTAB)
        anchors = bytes(rule_g(s[self.geo[4 * m]:self.geo[4 * m] + self.geo[4 * m + 1] * self.geo[4 * m + 2]])
                        for m in range(3))
        return bytes(s), anchors, 8 if cz else 0

    def split(self, ntrain=256, ntest=2048, seed=20261121):
        """train: evenly spaced records; test: other records, same count per layer, seeded."""
        ex = self.experts
        train = ex[::max(1, len(ex) // ntrain)][:ntrain]
        chosen, by_layer = set(train), {}
        for r in ex:
            if r not in chosen:
                by_layer.setdefault(self.owner(r) >> 16, []).append(r)
        rng, test = random.Random(seed), []
        layers = sorted(by_layer)
        for i, L in enumerate(layers):
            test += rng.sample(by_layer[L], ntest // len(layers) + (i < ntest % len(layers)))
        return train, test
