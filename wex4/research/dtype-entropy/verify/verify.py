#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Adversarial re-check of ../measure.py with independent code (CPU, numpy, read-only).

    python -I verify.py selftest
    python -I verify.py repro|f8|bf16|f32 <model_dir>     # each writes out/verify_<stage>.json

repro : same tensors as the experiment -> must match results.json; plus a real rANS byte count
f8    : layers 10 and 30 (NOT in the experiment's sample: 0, 20, 39) + two unsampled MTP kinds
bf16  : embed/head by 32 stratified 500-row slices each (the experiment used 5 x 3125 rows,
        one of them the atypical tail), pooled-table variant, two other vision blocks
f32   : every hc_*_fn tensor (the whole dtype >= 1 MiB, no extrapolation)
"""
import json
import sys
import time
from math import ceil, log2
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True            # never drop __pycache__ outside verify/
sys.path.insert(0, str(Path(__file__).parent))  # python -I does not add the script folder
from vidx import index  # noqa: E402

S, CH, OVH, L = 1 << 14, 4096, 6, 1 << 23
HERE = Path(__file__).parent
READ = 0
WIDTH = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}


def read(idx, name, rows=None):
    global READ
    dt, shape, path, off, nb = idx[name]
    w = WIDTH[dt]
    if len(shape) != 2:
        shape = (1, nb // w)
    if rows:
        rb = shape[1] * w
        off, nb, shape = off + rows[0] * rb, (rows[1] - rows[0]) * rb, (rows[1] - rows[0], shape[1])
    READ += nb
    with open(path, "rb") as fh:
        fh.seek(off)
        buf = fh.read(nb)
    assert len(buf) == nb
    return np.frombuffer(buf, {1: np.uint8, 2: "<u2", 4: "<u4"}[w]).reshape(shape)


def quant(c):
    """Histogram -> 14-bit frequencies, present symbols >= 1 (largest remainder; not the experiment's rule)."""
    c = c.astype(np.int64)
    p = c * S / c.sum()
    f = np.where(c > 0, np.maximum(1, np.floor(p)), 0).astype(np.int64)
    d = S - int(f.sum())
    if d > 0:
        f[np.argsort(-np.where(c > 0, p - np.floor(p), -1.0))[:d]] += 1
    while d < 0:
        f[f.argmax()] -= 1
        d += 1
    assert f.sum() == S and (f[c > 0] >= 1).all()
    return f


def hist_cost(h, K, n):
    """h: [contexts, K] histograms -> (ideal bits, payload B, table B) with the draft's overheads."""
    ideal = xb = 0.0
    used = 0
    for row in h:
        m = int(row.sum())
        if m == 0:
            continue
        used += 1
        nz = row > 0
        ideal += float((row[nz] * np.log2(m / row[nz])).sum())
        xb += float((row[nz] * (14 - np.log2(quant(row)[nz]))).sum())
    pay, tab, raw = ceil(xb / 8) + ceil(n / CH) * OVH, 2 * (K + 1) * used, ceil(n * log2(K) / 8)
    return (ideal, raw, 0) if pay + tab >= raw else (ideal, pay, tab)


def plane(sym, K, ctx=None, nctx=1):
    sym = sym.ravel().astype(np.intp)
    key = sym if ctx is None else ctx.ravel().astype(np.intp) * K + sym
    return hist_cost(np.bincount(key, minlength=nctx * K).reshape(nctx, K), K, sym.size)


def tot(*parts, side=0.0):
    """Sum planes (+ raw side bytes) -> dict(bits, bytes)."""
    return {"bits": sum(p[0] for p in parts) + 8 * side, "bytes": sum(p[1] + p[2] for p in parts) + ceil(side)}


# ------------------------------------------------------------------ real rANS (size only + small round trip)
def rans_bytes(sym, f):
    """Exact coded size of 32-bit rANS (L = 2^23, 14-bit CDF, byte renormalisation), independent streams of
    4096 symbols, each with a 4-byte flush and a u16 length. Vectorised over streams; full chunks only."""
    s = sym.ravel()
    n = s.size // CH * CH
    s = s[:n].reshape(-1, CH)
    F = f.astype(np.uint64)
    C = (np.cumsum(f) - f).astype(np.uint64)
    x = np.full(s.shape[0], L, np.uint64)
    nb = np.zeros(s.shape[0], np.int64)
    for t in range(CH - 1, -1, -1):
        fs, cs = F[s[:, t]], C[s[:, t]]
        xmax = fs << np.uint64(17)                 # ((L >> 14) << 8) * freq
        for _ in range(2):                         # x < 2^31 and xmax >= 2^17: at most 2 bytes
            m = x >= xmax
            nb += m
            x = np.where(m, x >> np.uint64(8), x)
        x = ((x // fs) << np.uint64(14)) + (x % fs) + cs
    return int(nb.sum()) + s.shape[0] * OVH, n


def rans_roundtrip(sym, f):
    """Scalar encoder + decoder of one stream; returns the byte count incl. flush and u16 length."""
    f = [int(v) for v in f]
    cdf = [0]
    for v in f:
        cdf.append(cdf[-1] + v)
    out, x = bytearray(), L
    for s in reversed(sym):
        while x >= f[s] << 17:
            out.append(x & 255)
            x >>= 8
        x = (x // f[s] << 14) + x % f[s] + cdf[s]
    out += x.to_bytes(4, "little")
    data = bytes(reversed(out))
    lut = [s for s in range(len(f)) for _ in range(f[s])]
    x, pos, res = int.from_bytes(data[:4], "big"), 4, []
    for _ in sym:
        slot = x & (S - 1)
        s = lut[slot]
        res.append(s)
        x = f[s] * (x >> 14) + slot - cdf[s]
        while x < L:
            x, pos = (x << 8) | data[pos], pos + 1
    assert res == list(sym) and pos == len(data) and x == L, "rANS round trip failed"
    return len(data) + 2


def rans_check(sym, K, label):
    """Model payload (cross-entropy + 6 B / 4096) vs real rANS bytes on the same full chunks."""
    s = sym.ravel().astype(np.intp)
    s = s[: s.size // CH * CH]
    c = np.bincount(s, minlength=K)
    f = quant(c)
    nz = c > 0
    model = ceil(float((c[nz] * (14 - np.log2(f[nz]))).sum()) / 8) + s.size // CH * OVH
    real, n = rans_bytes(s, f)
    k = min(3, s.size // CH)
    scalar = sum(rans_roundtrip(s[i * CH:(i + 1) * CH].tolist(), f) for i in range(k))
    assert scalar == rans_bytes(s[: k * CH], f)[0], "vectorised count != scalar encoder"
    r = {"plane": label, "symbols": n, "model_bytes": model, "rans_bytes": real, "rel_diff": real / model - 1}
    print(f"  rANS {label}: model {model:,} B, real {real:,} B, diff {100 * r['rel_diff']:+.4f} %  (round trip ok on {k} streams)")
    return r


# ------------------------------------------------------------------ per-dtype models (subset of the experiment's)
def f8(x, sc, full=False):
    v = x.astype(np.intp)
    n = v.size
    o = {"wbp1": tot(plane(v, 256))}
    if sc is not None:
        assert sc.shape == (x.shape[0] // 32, x.shape[1] // 32), (x.shape, sc.shape)
        su, si = np.unique(sc, return_inverse=True)
        ctx = np.repeat(np.repeat(si.reshape(sc.shape), 32, 0), 32, 1)
        o["ctx_block_scale"] = tot(plane(v, 256, ctx, len(su)))
        o["scale_distinct"] = int(len(su))
        o["scale_o0_bits"] = plane(sc.astype(np.intp), 256)[0] / sc.size
    if full:
        o["exp4_coded_rest_raw"] = tot(plane((v >> 3) & 15, 16), side=n / 2)
        prev = np.zeros_like(v)
        prev[:, 1:] = v[:, :-1]
        o["o1_prev_byte"] = tot(plane(v, 256, prev, 256))
    return o, [np.bincount(v.ravel(), minlength=256)]


def bf16(u):
    v = u.astype(np.intp)
    n = v.size
    hi, lo, ex = v >> 8, v & 255, (v >> 7) & 255
    p_hi, p_lo, p_ex = plane(hi, 256), plane(lo, 256), plane(ex, 256)
    c16 = np.bincount(v.ravel(), minlength=65536)
    nz = c16 > 0
    o = {"wbp1": tot(p_lo, p_hi),
         "exp8_coded_rest_raw": tot(p_ex, side=n),
         "hi_o0_lo_given_hi4": tot(p_hi, plane(lo, 256, hi & 15, 16)),
         "joint16_ideal_bits": float((c16[nz] * np.log2(n / c16[nz])).sum()),
         "exp_bits": p_ex[0] / n}
    return o, [np.bincount(lo.ravel(), minlength=256), np.bincount(hi.ravel(), minlength=256)]


def f32(u):
    v = u.astype(np.intp)
    n = v.size
    b = [(v >> (8 * p)) & 255 for p in range(4)]
    ps = [plane(p, 256) for p in b]
    o = {"wbp1": tot(*ps), "signexp9_coded_mant23_raw": tot(plane(v >> 23, 512), side=n * 23 / 8),
         "plane_bits": [p[0] / n for p in ps], "planes_coded": sum(p[2] > 0 for p in ps)}
    return o, [np.bincount(p.ravel(), minlength=256) for p in b]


def scale_of(idx, name, rows=None):
    return read(idx, name[:-6] + "scale", (rows[0] // 32, rows[1] // 32) if rows else None)


def held_out(hists_train, hists_test):
    """WBP1 bytes of the test planes coded with tables built from the train planes only (no table charged)."""
    b = 0
    for tr, te in zip(hists_train, hists_test):
        f = quant(tr + 1)
        n = int(te.sum())
        b += min(n, ceil(float((te * (14 - np.log2(f))).sum()) / 8) + ceil(n / CH) * OVH)
    return b


def show(name, raw, o):
    keys = [k for k, v in o.items() if isinstance(v, dict)]
    print(f"{name:52} " + "  ".join(f"{k} {o[k]['bytes'] / raw:.5f}" for k in keys), flush=True)


def rec(name, x, o, rows=None):
    return {"name": name, "rows": rows, "elements": int(x.size), "raw_bytes": int(x.nbytes), "models": o}


# ------------------------------------------------------------------ stages
def stage_repro(idx):
    theirs = {(t["name"], tuple(t["rows"]) if t["rows"] else None): t
              for t in json.loads((HERE.parent / "results.json").read_text())["tensors"]}
    pairs = {"wbp1": ["o0_byte (WBP1)", "o0 byte planes (WBP1)"], "ctx_block_scale": ["ctx block scale (E8M0 value)"],
             "exp4_coded_rest_raw": ["exp4_o0 + sign,mant3 raw"], "o1_prev_byte": ["o1 prev byte in row"],
             "exp8_coded_rest_raw": ["exp8 o0 + sign,mant7 raw"], "hi_o0_lo_given_hi4": ["hi o0 + lo|hi low 4 bits"],
             "signexp9_coded_mant23_raw": ["signexp9 o0 + mant23 raw"]}
    out, rans = [], []
    jobs = [("layers.20.attn.wo_a.weight", None), ("layers.39.ffn.shared_experts.w2.weight", None),
            ("embed.weight", (126155, 129280)), ("layers.20.ffn.gate.weight", None), ("layers.20.hc_attn_fn", None)]
    for name, rows in jobs:
        dt = idx[name][0]
        x = read(idx, name, rows)
        if dt == "F8_E4M3":
            o, _ = f8(x, scale_of(idx, name, rows), full=True)
        else:
            o, _ = (bf16 if dt == "BF16" else f32)(x)
        t = theirs[(name, rows)]
        print(f"{name}{rows or ''}  raw {x.nbytes:,}")
        for k, v in o.items():
            if not isinstance(v, dict):
                continue
            tm = next(t["models"][a] for a in pairs[k] if a in t["models"])
            tb = tm["payload"] + tm["table"]
            v["theirs_bytes"], v["theirs_bits"] = tb, tm["bits"] * x.size
            print(f"  {k:28} mine {v['bytes']:>11,} B ({v['bytes'] / x.nbytes:.5f})  theirs {tb:>11,} B ({tb / x.nbytes:.5f})"
                  f"  diff {v['bytes'] - tb:+,} B | ideal bits/el mine {v['bits'] / x.size:.5f} theirs {tm['bits']:.5f}")
        out.append(rec(name, x, o, rows))
        if dt == "F8_E4M3" and x.size > 20_000_000:
            rans.append(rans_check(x, 256, name))
        if dt == "BF16" and rows:
            rans.append(rans_check(x >> 8, 256, name + " hi"))
            rans.append(rans_check(x & 255, 256, name + " lo"))
            rans.append(rans_check((x >> 7) & 255, 256, name + " exp8"))
        if dt == "F32":
            rans.append(rans_check(x >> 24, 256, name + " b3"))
            rans.append(rans_check((x >> 16) & 255, 256, name + " b2"))
    return {"tensors": out, "rans": rans}


def stage_f8(idx):
    out, hists = [], {}
    kinds = ["attn.wkv", "attn.wo_a", "attn.wo_b", "attn.wq_a", "attn.wq_b",
             "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"]
    names = [f"layers.{l}.{k}.weight" for l in (10, 30) for k in kinds] + ["mtp.1.attn.wo_a.weight", "mtp.1.attn.wq_b.weight"]
    for name in names:
        x = read(idx, name)
        o, h = f8(x, scale_of(idx, name))
        hists[name] = h
        show(name, x.nbytes, o)
        out.append(rec(name, x, o))
    ho = []                                           # table from layer 10 used on layer 30 and vice versa
    for k in kinds:
        a, b = f"layers.10.{k}.weight", f"layers.30.{k}.weight"
        raw = idx[a][4] + idx[b][4]
        ho.append({"kind": k, "raw_bytes": raw, "held_out_bytes": held_out(hists[a], hists[b]) + held_out(hists[b], hists[a])})
    return {"tensors": out, "held_out_10_vs_30": ho}


def stage_bf16(idx):
    out, pooled = [], {}
    for name in ("embed.weight", "head.weight"):
        R = idx[name][1][0]
        acc, raw = None, 0
        for i in range(32):
            a = i * (R // 32) + (R // 32 - 500) // 2
            x = read(idx, name, (a, a + 500))
            o, h = bf16(x)
            show(f"{name}[{a}:{a + 500}]", x.nbytes, o)
            out.append(rec(name, x, o, (a, a + 500)))
            acc = h if acc is None else [p + q for p, q in zip(acc, h)]
            raw += x.nbytes
        one = sum(sum(hist_cost(h[None], 256, int(h.sum()))[1:]) for h in acc)
        pooled[name] = {"raw_bytes": raw, "wbp1_one_table_for_all_slices": one}
        print(f"{name}: one table for the 32 slices {one / raw:.5f}")
    for name in [f"vision.blocks.{b}.{k}.weight" for b in (8, 24) for k in ("mlp.w1", "mlp.w2", "attn.wqkv", "attn.wo")]:
        x = read(idx, name)
        o, _ = bf16(x)
        show(name, x.nbytes, o)
        out.append(rec(name, x, o))
    return {"tensors": out, "pooled": pooled}


def stage_f32(idx):
    out, hists = [], {}
    names = sorted(n for n, t in idx.items() if t[0] == "F32" and t[4] >= 1 << 20)
    for name in names:
        x = read(idx, name)
        o, h = f32(x)
        hists[name] = h
        out.append(rec(name, x, o))
    raw = sum(t["raw_bytes"] for t in out)
    for k in ("wbp1", "signexp9_coded_mant23_raw"):
        print(f"F32 all {len(out)} tensors >= 1 MiB ({raw:,} B): {k} {sum(t['models'][k]['bytes'] for t in out) / raw:.5f}")
    print("  plane bits b0..b3 (mean):", np.mean([t["models"]["plane_bits"] for t in out], 0).round(4).tolist(),
          "| coded planes per record:", sorted(set(t["models"]["planes_coded"] for t in out)))
    total = [sum(h[p] for h in hists.values()) for p in range(4)]
    ho = sum(held_out([tp - hp for tp, hp in zip(total, h)], h) for h in hists.values())
    print(f"  shared table, leave-one-out over all {len(out)}: {ho / raw:.5f}")
    return {"tensors": out, "raw_bytes": raw, "held_out_leave_one_out_bytes": ho}


def stage_tail(idx):
    """Where is the atypical tail of embed/head that the experiment's last slice [126155:129280] hit?"""
    out = []
    for name in ("embed.weight", "head.weight"):
        for a in range(126155, 129280, 250):
            b = min(a + 250, 129280)
            x = read(idx, name, (a, b))
            o, _ = bf16(x)
            o["zero_frac"] = float(((x & 0x7FFF) == 0).mean())
            o["rows_all_equal_to_first"] = int((x == x[0]).all(1).sum())
            show(f"{name}[{a}:{b}] zeros {o['zero_frac']:.4f} dup-rows {o['rows_all_equal_to_first']}", x.nbytes, o)
            out.append(rec(name, x, o, (a, b)))
    return {"tensors": out}


def selftest():
    rng = np.random.default_rng(7)
    sys.path.insert(0, str(HERE.parent))
    import measure                                   # the experiment's code, for a cross-check on synthetic data
    g = np.clip(rng.normal(128, 9, 1 << 18), 0, 255).astype(np.intp)
    mine, theirs = plane(g, 256), measure.stream(g, 256)
    assert abs(mine[0] - theirs[0]) < 1e-6 * mine[0] and abs(mine[1] - theirs[1]) <= 0.001 * mine[1] and mine[2] == theirs[2] == 514
    ctx = rng.integers(0, 4, g.size)
    mine, theirs = plane((g + 20 * ctx) % 256, 256, ctx, 4), measure.stream((g + 20 * ctx) % 256, 256, ctx, 4)
    assert abs(mine[0] - theirs[0]) < 1e-6 * mine[0] and mine[2] == theirs[2] == 4 * 514
    u = rng.integers(0, 256, 1 << 16)
    assert plane(u, 256)[1:] == (u.size, 0)                                    # incompressible -> raw, no table
    r = rans_check(g, 256, "synthetic gaussian")
    assert abs(r["rel_diff"]) < 0.01
    skew = (rng.random(1 << 16) < 0.001).astype(np.intp) * 7                   # rare symbol -> 2-byte renormalisations
    rans_check(skew, 256, "synthetic skewed")
    print("selftest ok")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "selftest":
        selftest()
    else:
        t0 = time.time()
        res = {"repro": stage_repro, "f8": stage_f8, "bf16": stage_bf16, "f32": stage_f32, "tail": stage_tail}[stage](index(sys.argv[2]))
        res["read_bytes"] = READ
        (HERE / "out" / f"verify_{stage}.json").write_text(json.dumps(res, indent=1))
        print(f"stage {stage}: read {READ:,} B in {time.time() - t0:.0f} s")
