#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Entropy survey of the NON-expert tensors (F8_E4M3, BF16, F32) of an MXFP4 model.

Question: how far does the draft "WBP1" codec (byte planes, static order-0 rANS,
one 14-bit CDF per plane per record) get, and which richer *static* models pay
for their tables.  CPU only, numpy only, reads a bounded sample (READ_CAP).

    python -I measure.py <model_dir>            # writes results.json next to this file
    python -I measure.py --selftest

Cost accounting (the same for every model):
  payload = sum over streams of ceil(coded_bits / 8) + 6 B per 4096 symbols
            (u16 length + 4-byte rANS flush), coded_bits measured against the
            CDF quantised to 14 bits (every present symbol gets frequency >= 1),
            plus any side information the model needs (class ids, raw sign bits);
  table   = 2 B per CDF entry, (K + 1) entries per context that occurs;
  a stream that does not shrink (payload + table >= raw) is stored raw, no table.
  "bits" is the ideal (unquantised, no overhead, no table) entropy per element.
"""
import json
import re
import struct
import sys
import time
from collections import defaultdict
from math import ceil, log2
from pathlib import Path

import numpy as np

S = 1 << 14                      # CDF resolution
CHUNK, CHUNK_OVH = 4096, 6       # symbols per independent stream, bytes of framing
READ_CAP = 1536 << 20            # hard cap on bytes read from the model
DT_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}
DT_NUMPY = {"F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "BF16": np.uint16, "F32": np.uint32}
read_total = 0


# ---------------------------------------------------------------- entropy core
def ent_bits(c):
    """Ideal total bits; each row of c is the histogram of one context."""
    c = c.reshape(-1, c.shape[-1]).astype(np.float64)
    n = c.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(c > 0, c * np.log2(n / c), 0.0).sum())


def q14(c):
    """Quantise each histogram row to frequencies summing to 2^14, present symbols >= 1."""
    c = c.reshape(-1, c.shape[-1]).astype(np.int64)
    n = np.maximum(c.sum(1, keepdims=True), 1)
    f = np.where(c > 0, np.maximum(1, c * S // n), 0)
    d = S - f.sum(1)
    neg = d < 0                                   # too many "bumped to 1" symbols
    if neg.any():
        g = f[neg] - (f[neg] > 0)                 # mass that may be removed
        f[neg] -= -(-g * (-d[neg])[:, None] // np.maximum(g.sum(1, keepdims=True), 1))
        d = S - f.sum(1)
    f[np.arange(len(f)), c.argmax(1)] += d
    assert (f[c > 0] >= 1).all() and (f.sum(1) == S).all()
    return f


def xent_bits(c, f):
    """Bits to code histogram(s) c with quantised frequencies f."""
    c = c.reshape(-1, c.shape[-1])
    with np.errstate(divide="ignore"):
        return float(np.where(c > 0, c * (14 - np.log2(np.maximum(f, 1))), 0.0).sum())


def cost(c, K, raw_bits=None):
    """(ideal_bits, payload_bytes, table_bytes, 1 if entropy coded / 0 if stored raw) of one plane."""
    n = int(c.sum())
    used = int((c.reshape(-1, K).sum(1) > 0).sum())
    payload = ceil(xent_bits(c, q14(c)) / 8) + ceil(n / CHUNK) * CHUNK_OVH
    table = 2 * (K + 1) * used
    raw = ceil(n * (log2(K) if raw_bits is None else raw_bits) / 8)
    coded = 1
    if payload + table >= raw:
        payload, table, coded = raw, 0, 0
    return ent_bits(c), payload, table, coded


def stream(sym, K, ctx=None, nctx=1):
    idx = sym if ctx is None else ctx * K + sym
    return cost(np.bincount(idx.ravel(), minlength=nctx * K).reshape(nctx, K), K)


class M:
    """Accumulates the cost of one model on one tensor."""

    def __init__(self, cls):
        self.bits, self.payload, self.table, self.coded, self.cls = 0.0, 0, 0, 0, cls

    def add(self, r):
        self.bits += r[0]; self.payload += r[1]; self.table += r[2]; self.coded += r[3]
        return self

    def side(self, nbytes):                       # raw side information
        self.bits += 8 * nbytes; self.payload += ceil(nbytes)
        return self

    def out(self, n):
        # coded = rANS symbols decoded per element (planes stored raw do not count)
        return {"bits": round(self.bits / n, 5), "payload": self.payload, "table": self.table,
                "coded": self.coded, "cls": self.cls}


O0 = "order-0 table lookup"
SIDE = "context = side info or an already decoded plane (table select per symbol, streams stay independent)"
PREV = "context = previously decoded symbol (table select per symbol, serial dependency inside a stream)"


def qclass(a, k):
    """Quantile class 0..k-1 of each value (the class id is sent as side info)."""
    return np.searchsorted(np.quantile(a, np.linspace(0, 1, k + 1)[1:-1]), a).astype(np.intp)


def shifted(v, axis):
    p = np.zeros_like(v)
    if axis == 1:
        p[:, 1:] = v[:, :-1]
    else:
        p[1:] = v[:-1]
    return p


# ---------------------------------------------------------------- F8_E4M3
def f8_models(x, sc):
    """x: uint8 [R, C] (bit7 sign, bits6..3 exponent, bits2..0 mantissa); sc: E8M0 scale per 32x32 block."""
    R, C = x.shape
    n = x.size
    v = x.astype(np.intp)
    mag, e, m = v & 127, (v >> 3) & 15, v & 7
    o = {}
    o["o0_byte (WBP1)"] = M(O0).add(stream(v, 256))
    o["sign_raw + mag7_o0"] = M(O0).side(n / 8).add(stream(mag, 128))
    o["exp4_o0 + sign,mant3 raw"] = M(O0).side(n / 2).add(stream(e, 16))
    o["sign_raw + exp4_o0 + mant3|exp"] = M(SIDE).side(n / 8).add(stream(e, 16)).add(stream(m, 8, e, 16))
    prev, up = shifted(v, 1), shifted(v, 0)
    o["o1 prev byte in row"] = M(PREV).add(stream(v, 256, prev, 256))
    o["ctx prev exp in row (16)"] = M(PREV).add(stream(v, 256, (prev >> 3) & 15, 16))
    o["ctx exp above (16)"] = M(PREV).add(stream(v, 256, (up >> 3) & 15, 16))
    o["ctx prev exp x exp above (256)"] = M(PREV).add(stream(v, 256, ((prev >> 3) & 15) * 16 + ((up >> 3) & 15), 256))
    del prev, up
    o["per-row CDF"] = M(SIDE).add(stream(v, 256, np.broadcast_to(np.arange(R)[:, None], (R, C)), R))
    o["per-column CDF"] = M(SIDE).add(stream(v, 256, np.broadcast_to(np.arange(C)[None, :], (R, C)), C))
    rc = qclass(mag.mean(1), 8)[:, None] * 8 + qclass(mag.mean(0), 8)[None, :]
    o["ctx row class x column class (8x8)"] = M(SIDE).side((R + C) * 3 / 8).add(stream(v, 256, rc, 64))
    facts = {"neg_zero": int((x == 0x80).sum()), "pos_zero": int((x == 0).sum()),
             "nan_codes": int(((x & 127) == 127).sum()), "unused_symbols": int(256 - len(np.unique(x)))}
    if sc is not None:
        assert sc.shape == (R // 32, C // 32), (x.shape, sc.shape)
        nb = sc.size
        blocks = mag.reshape(R // 32, 32, C // 32, 32)
        up32 = lambda b: np.repeat(np.repeat(b, 32, 0), 32, 1)
        su, si = np.unique(sc, return_inverse=True)
        si = si.reshape(sc.shape).astype(np.intp)
        o["ctx block scale (E8M0 value)"] = M(SIDE).add(stream(v, 256, up32(si), len(su)))
        bmax = blocks.max((1, 3))
        o["ctx block max exponent (16)"] = M(SIDE).side(nb / 2).add(stream(v, 256, up32(bmax >> 3), 16))
        bcls = qclass(blocks.mean((1, 3)).ravel(), 16).reshape(sc.shape)
        o["ctx block class (16, by mean magnitude)"] = M(SIDE).side(nb / 2).add(stream(v, 256, up32(bcls), 16))
        b64 = qclass(blocks.mean((1, 3)).ravel(), 64).reshape(sc.shape)
        o["ctx block class (64, by mean magnitude)"] = M(SIDE).side(nb * 6 / 8).add(stream(v, 256, up32(b64), 64))
        o["ctx block class x block scale"] = M(SIDE).side(nb / 2).add(stream(v, 256, up32(bcls * len(su) + si), 16 * len(su)))
        sv = sc.astype(np.intp)
        facts.update(scale_distinct=len(su), scale_min=int(su.min()), scale_max=int(su.max()),
                     scale_o0_bits=round(stream(sv, 256)[0] / nb, 4),
                     scale_bits_given_left=round(stream(sv, 256, shifted(sv, 1), 256)[0] / nb, 4),
                     block_max_mag_hist_top=np.bincount(bmax.ravel(), minlength=128)[-16:].tolist(),
                     block_max_exp_bits=round(stream(bmax >> 3, 16)[0] / nb, 4))
    return o, [np.bincount(v.ravel(), minlength=256)], facts


# ---------------------------------------------------------------- BF16
def bf16_models(u):
    """u: uint16 [R, C]; bit15 sign, bits14..7 exponent, bits6..0 mantissa."""
    R, C = u.shape
    n = u.size
    v = u.astype(np.intp)
    hi, lo, se, mant, ex = v >> 8, v & 255, v >> 7, v & 127, (v >> 7) & 255
    o = {}
    r_lo, r_hi = stream(lo, 256), stream(hi, 256)
    o["o0 byte planes (WBP1)"] = M(O0).add(r_lo).add(r_hi)
    c16 = np.bincount(v.ravel(), minlength=65536)
    b = M("bound only (65536 symbols do not fit a 14-bit CDF)")
    b.bits = ent_bits(c16); b.payload = ceil(b.bits / 8); b.table = 4 * int((c16 > 0).sum())
    o["bound: joint 16-bit order-0"] = b
    o["hi|prev hi in row + lo o0"] = M(PREV).add(stream(hi, 256, shifted(hi, 1), 256)).add(r_lo)
    for bits in (2, 4):
        o[f"hi o0 + lo|hi low {bits} bits"] = M(SIDE).add(r_hi).add(stream(lo, 256, hi & ((1 << bits) - 1), 1 << bits))
    o["hi o0 + lo|hi (256 ctx)"] = M(SIDE).add(r_hi).add(stream(lo, 256, hi, 256))
    r_se, r_mant = stream(se, 512), stream(mant, 128)
    o["signexp9 o0 + mant7 o0"] = M(O0).add(r_se).add(r_mant)
    o["signexp9 o0 + mant7|exp low 3 bits"] = M(SIDE).add(r_se).add(stream(mant, 128, ex & 7, 8))
    o["sign raw + exp8 o0 + mant7 o0"] = M(O0).side(n / 8).add(stream(ex, 256)).add(r_mant)
    o["exp8 o0 + sign,mant7 raw"] = M(O0).side(n).add(stream(ex, 256))
    o["signexp9|prev signexp9 + mant7 o0"] = M(PREV).add(stream(se, 512, shifted(se, 1), 512)).add(r_mant)
    if R > 1:
        rows = np.broadcast_to(np.arange(R)[:, None], (R, C))
        cols = np.broadcast_to(np.arange(C)[None, :], (R, C))
        o["per-row CDF (both planes)"] = M(SIDE).add(stream(hi, 256, rows, R)).add(stream(lo, 256, rows, R))
        o["per-column CDF (both planes)"] = M(SIDE).add(stream(hi, 256, cols, C)).add(stream(lo, 256, cols, C))
        rc = qclass(ex.mean(1), 8)[:, None] * 8 + qclass(ex.mean(0), 8)[None, :]
        o["signexp9|row class x column class (8x8) + mant7 o0"] = (
            M(SIDE).side((R + C) * 3 / 8).add(stream(se, 512, rc, 64)).add(r_mant))
    facts = {"distinct_u16": int((c16 > 0).sum()), "sign_bits": round(stream(v >> 15, 2)[0] / n, 4),
             "exp_bits": round(stream(ex, 256)[0] / n, 4), "mant_bits": round(r_mant[0] / n, 4),
             "zeros": int(((v & 0x7FFF) == 0).sum())}
    return o, [np.bincount(lo.ravel(), minlength=256), np.bincount(hi.ravel(), minlength=256)], facts


# ---------------------------------------------------------------- F32
def f32_models(u):
    """u: uint32 [R, C]; bit31 sign, bits30..23 exponent, bits22..0 mantissa."""
    R, C = u.shape
    n = u.size
    v = u.astype(np.intp)
    b = [(v >> (8 * p)) & 255 for p in range(4)]
    r = [stream(p, 256) for p in b]
    o = {}
    w = M(O0)
    for x in r:
        w.add(x)
    o["o0 byte planes (WBP1)"] = w
    se, m7 = v >> 23, (v >> 16) & 127
    r_se = stream(se, 512)
    o["signexp9 o0 + mant 7/8/8 o0"] = M(O0).add(r_se).add(stream(m7, 128)).add(r[1]).add(r[0])
    o["signexp9 o0 + mant23 raw"] = M(O0).side(n * 23 / 8).add(r_se)
    o["b3 o0 + b2|b3 low 4 bits + b1 o0 + b0 o0"] = M(SIDE).add(r[3]).add(stream(b[2], 256, b[3] & 15, 16)).add(r[1]).add(r[0])
    o["b3|prev b3 in row + b2,b1,b0 o0"] = M(PREV).add(stream(b[3], 256, shifted(b[3], 1), 256)).add(r[2]).add(r[1]).add(r[0])
    o["b3,b2 o0 + b1|b2 high 4 bits + b0 o0"] = M(SIDE).add(r[3]).add(r[2]).add(stream(b[1], 256, b[2] >> 4, 16)).add(r[0])
    if R > 1:
        rows = np.broadcast_to(np.arange(R)[:, None], (R, C))
        pr = M(SIDE)
        for p in b:
            pr.add(stream(p, 256, rows, R))
        o["per-row CDF (4 planes)"] = pr
    f = u.view(np.float32)
    with np.errstate(over="ignore"):
        f16_exact = float((f.astype(np.float16).astype(np.float32) == f).mean())
    facts = {"distinct_values": int(len(np.unique(u))), "low16_zero_frac": round(float(((u & 0xFFFF) == 0).mean()), 6),
             "low8_zero_frac": round(float(((u & 0xFF) == 0).mean()), 6),
             "f16_exact_frac": round(f16_exact, 6),
             "plane_bits_b0_b3": [round(x[0] / n, 4) for x in r], "zeros": int(((u & 0x7FFFFFFF) == 0).sum())}
    return o, [np.bincount(p.ravel(), minlength=256) for p in b], facts


# ---------------------------------------------------------------- model access
def index_model(model_dir):
    """name -> (dtype, shape, shard path, absolute offset, nbytes) for the non-expert tensors."""
    idx = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with path.open("rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        for name, info in header.items():
            if name != "__metadata__" and ".ffn.experts." not in name:
                a, z = info["data_offsets"]
                idx[name] = (info["dtype"], info["shape"], path, 8 + hlen + a, z - a)
    return idx


def read(idx, name, rows=None):
    """Whole tensor, or a row range [a, b) of a 2-D tensor, as a 2-D unsigned array."""
    global read_total
    dtype, shape, path, off, nbytes = idx[name]
    shape = list(shape) if len(shape) == 2 else [1, int(np.prod(shape))]
    if rows:
        rb = shape[1] * DT_BYTES[dtype]
        off, nbytes, shape = off + rows[0] * rb, (rows[1] - rows[0]) * rb, [rows[1] - rows[0], shape[1]]
    read_total += nbytes
    assert read_total <= READ_CAP, "read cap exceeded"
    with path.open("rb") as fh:
        fh.seek(off)
        return np.frombuffer(fh.read(nbytes), DT_NUMPY[dtype]).reshape(shape)


def nat(name):
    return [int(t) for t in re.findall(r"\d+", name)]


def kind_of(name):      # same role in another layer/block: only the first index is a wildcard
    return re.sub(r"\d+", "#", name, count=1)


def coarse(name):       # population bucket used for the whole-dtype extrapolation
    return re.sub(r"\d+", "#", name)


def pick(names, k):
    """k evenly spaced names (first, ..., last) in natural order; all of them if k is None."""
    names = sorted(names, key=nat)
    if k is None or k >= len(names):
        return names
    return [names[round(i * (len(names) - 1) / (k - 1))] for i in range(k)] if k > 1 else names[:1]


SMALL = 1 << 20  # tensors below this are "small": every tensor of the kind is read

# (regex on kind, how many per kind, optional row slices) for the large tensors
PLAN = [
    (r"^layers\.#\.attn\.(wo_a|wo_b|wq_a|wq_b|wkv)\.weight$", 3, None),
    (r"^layers\.#\.ffn\.shared_experts\.w\d\.weight$", 3, None),
    (r"^layers\.#\.attn\.indexer\.wq_b\.weight$", 2, None),
    (r"^mtp\.#\.attn\.(wq_a|wkv)\.weight$", 1, None),
    (r"^mtp\.#\.ffn\.shared_experts\.w1\.weight$", 1, None),
    (r"^mtp\.#\.main_proj\.weight$", 1, [(0, 1024)]),
    (r"^(embed|head)\.weight$", 1, [(0, 3125), (31500, 34625), (63000, 66125), (94500, 97625), (126155, 129280)]),
    (r"^vision\.blocks\.#\.(mlp\.w\d|attn\.wqkv|attn\.wo)\.weight$", 3, None),
    (r"^aligner\.w#\.weight$", 2, None),
    (r"^layers\.#\.ffn\.gate\.weight$", 5, None),
    (r"^mtp\.#\.markov_head\.embed\.weight$", 1, None),
    (r"^mtp\.#\.markov_head\.head\.weight$", 1, [(0, 32768)]),
    (r"^layers\.#\.attn\.compressor\.(wkv|wgate)\.weight$", 2, None),
    (r"^layers\.#\.attn\.indexer\.(weights_proj|wk)\.weight$", 2, None),
    (r"^vision\.patch_embed\.proj\.weight$", 1, None),
    (r"^mtp\.#\.ffn\.gate\.weight$", 1, None),
    (r"^(layers|mtp)\.#\.hc_(attn|ffn)_fn$", 11, None),
]
MODELS = {"F8_E4M3": f8_models, "BF16": bf16_models, "F32": f32_models}


def sample_plan(idx):
    kinds = defaultdict(list)
    for name, (dtype, *_rest) in idx.items():
        if dtype in MODELS:
            kinds[kind_of(name)].append(name)
    jobs = []
    for kind, names in sorted(kinds.items()):
        if idx[names[0]][4] < SMALL:
            jobs += [(n, None) for n in pick(names, None)]
            continue
        for pattern, k, slices in PLAN:
            if re.match(pattern, kind):
                for n in pick(names, k):
                    jobs += [(n, s) for s in (slices or [None])]
    return jobs


def scale_for(idx, name, rows):
    sname = name[: -len("weight")] + "scale" if name.endswith(".weight") else None
    if sname not in idx:
        return None
    return read(idx, sname, (rows[0] // 32, rows[1] // 32) if rows else None)


def run(model_dir):
    idx = index_model(model_dir)
    t0 = time.time()
    tensors, planes = [], defaultdict(list)
    for name, rows in sample_plan(idx):
        dtype, shape, _, _, nbytes = idx[name]
        x = read(idx, name, rows)
        if dtype == "F8_E4M3":
            sc = scale_for(idx, name, rows)
            o, hists, facts = f8_models(x, sc)
            facts["has_scale"] = sc is not None
        else:
            o, hists, facts = MODELS[dtype](x)
        rec = {"name": name, "dtype": dtype, "shape": shape, "rows": rows, "elements": int(x.size),
               "raw_bytes": int(x.nbytes), "tensor_bytes": nbytes,
               "models": {k: m.out(x.size) for k, m in o.items()}, "facts": facts}
        tensors.append(rec)
        if rows is None:
            planes[(dtype, kind_of(name))].append((name, hists))
        w = rec["models"][next(iter(o))]
        print(f"{time.time() - t0:6.0f}s {dtype:8} {name}{rows or ''} WBP1 ratio "
              f"{(w['payload'] + w['table']) / x.nbytes:.4f}", flush=True)
    result = {"model_dir": str(model_dir), "read_bytes": read_total, "tensors": tensors,
              "shared": shared_models(planes, idx), "population": population(idx),
              "equality_checks": equality_checks(idx)}
    result["summary"] = summarise(result)
    for t in tensors:                                 # keep the file small: details only for big tensors
        if t["tensor_bytes"] < SMALL:
            t["models"] = dict(list(t["models"].items())[:1])
        for m in t["models"].values():
            del m["cls"]
    (Path(__file__).parent / "results.json").write_text(json.dumps(result, indent=1))
    print(f"read {read_total} B in total; wrote results.json")


def shared_models(planes, idx):
    """Per kind with >= 2 whole tensors: per-record tables vs one record for the whole kind vs a
    shared table trained on the OTHER tensors of the kind (leave-one-out = held-out, no table cost)."""
    out = []
    for (dtype, kind), items in sorted(planes.items()):
        if len(items) < 2:
            continue
        raw = per_record = held_out = 0
        total = [sum(h[p] for _, h in items) for p in range(len(items[0][1]))]
        for name, hists in items:
            raw += idx[name][4]
            for p, c in enumerate(hists):
                n = int(c.sum())
                _, pay, tab, _ = cost(c, 256)
                per_record += pay + tab
                f = q14(total[p] - c + 1)             # +1: every symbol stays codable
                held_out += min(n, ceil(xent_bits(c, f) / 8) + ceil(n / CHUNK) * CHUNK_OVH)
        one = sum(sum(cost(c, 256)[1:3]) for c in total)
        out.append({"dtype": dtype, "kind": kind, "tensors": len(items), "raw_bytes": raw,
                    "wbp1_per_record": per_record, "wbp1_one_record_for_kind": one,
                    "wbp1_shared_table_held_out": held_out})
    return out


def population(idx):
    pop = defaultdict(lambda: defaultdict(int))
    for name, (dtype, _s, _p, _o, nbytes) in idx.items():
        pop[dtype][coarse(name)] += nbytes
    return {d: dict(k) for d, k in pop.items()}


def equality_checks(idx):
    """Cheap exact-duplicate probes (first 4 MiB of each pair)."""
    pairs = [("embed.weight", "head.weight"),
             ("mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight"),
             ("layers.0.ffn.gate.bias", "layers.0.ffn.gate.bias_vl"),
             ("layers.0.hc_attn_fn", "layers.0.hc_ffn_fn"), ("layers.0.hc_attn_fn", "layers.1.hc_attn_fn"),
             ("layers.0.attn_norm.weight", "layers.0.ffn_norm.weight")]
    out = {}
    for a, b in pairs:
        if a in idx and b in idx and idx[a][4] == idx[b][4]:
            rows = (0, max(1, min(idx[a][1][0], (4 << 20) // (idx[a][4] // idx[a][1][0])))) if len(idx[a][1]) == 2 else None
            out[f"{a} == {b}"] = bool(np.array_equal(read(idx, a, rows), read(idx, b, rows)))
    return out


def summarise(result):
    """Per dtype and model: sample-weighted cost, and an extrapolation to the whole dtype that
    weights each coarse kind by its real size (unsampled kinds get the sample mean)."""
    out = {}
    for dtype in MODELS:
        ts = [t for t in result["tensors"] if t["dtype"] == dtype]
        pop = result["population"][dtype]
        total = sum(pop.values())
        names = list(dict.fromkeys(m for t in ts for m in t["models"]))
        for big in (True, False):
            sel = [t for t in ts if (t["tensor_bytes"] >= SMALL) == big]
            if not sel:
                continue
            rows = {}
            for m in names:
                use = [t for t in sel if m in t["models"]]
                if not use:
                    continue
                raw = sum(t["raw_bytes"] for t in use)
                el = sum(t["elements"] for t in use)
                pay = sum(t["models"][m]["payload"] for t in use)
                tab = sum(t["models"][m]["table"] for t in use)
                bits = sum(t["models"][m]["bits"] * t["elements"] for t in use) / el
                by_kind = defaultdict(lambda: [0, 0])
                for t in use:
                    k = by_kind[coarse(t["name"])]
                    k[0] += t["models"][m]["payload"] + t["models"][m]["table"]; k[1] += t["raw_bytes"]
                covered = sum(pop[k] for k in by_kind)
                est = sum(pop[k] * c / r for k, (c, r) in by_kind.items()) + (total - covered) * (pay + tab) / raw
                row = {"tensors": len(use), "sample_raw_bytes": raw, "ideal_bits_per_element": round(bits, 4),
                       "coded_bits_per_element": round(8 * pay / el, 4), "payload_bytes": pay, "table_bytes": tab,
                       "ratio_sample": round((pay + tab) / raw, 5), "cls": use[0]["models"][m]["cls"],
                       "coded_symbols_per_element": round(sum(t["models"][m]["coded"] * t["elements"] for t in use) / el, 3)}
                if big:
                    row["ratio_dtype_extrapolated"] = round(est / total, 5)
                    row["dtype_bytes_covered_by_sampled_kinds"] = round(covered / total, 4)
                if m != names[0]:
                    # elements one record needs before the extra table is repaid by the payload gain
                    # (WBP1 = first model, on the same records)
                    gain = (sum(t["models"][names[0]]["payload"] for t in use) - pay) / el      # bytes per element
                    extra = (tab - sum(t["models"][names[0]]["table"] for t in use)) / len(use)  # bytes per record
                    row["break_even_elements_vs_wbp1"] = (0 if extra <= 0 and gain >= 0 else
                                                          None if gain <= 0 else round(extra / gain))
                rows[m] = row
            out[f"{dtype} {'>=1MiB tensors' if big else '<1MiB tensors'}"] = rows
    return out


def selftest():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 256, 1 << 20).astype(np.intp)
    bits, pay, tab, _ = stream(u, 256)
    assert 7.99 < bits / u.size <= 8 and tab == 0 and pay == u.size           # incompressible -> raw
    z = np.zeros(1 << 16, np.intp)
    assert stream(z, 256)[0] == 0
    g = np.clip(rng.normal(128, 6, 1 << 20), 0, 255).astype(np.intp)
    bits, pay, tab, _ = stream(g, 256)
    assert 8 * (pay - ceil(g.size / CHUNK) * CHUNK_OVH) >= bits and pay < g.size * 0.62 and tab == 514
    assert stream(g, 256, g, 256)[0] == 0                                     # H(X | X) = 0
    c = np.zeros((1, 512), np.int64); c[0, :100] = 10_000; c[0, 100:] = 1     # forces the negative-deficit path
    assert q14(c).sum() == S
    x = rng.integers(0, 256, (64, 64)).astype(np.uint8)
    f8_models(x, rng.integers(120, 124, (2, 2)).astype(np.uint8))
    bf16_models(rng.integers(0, 65536, (8, 64)).astype(np.uint16))
    f32_models(rng.integers(0, 1 << 32, (8, 64), dtype=np.uint64).astype(np.uint32))
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        sys.exit(__doc__)
