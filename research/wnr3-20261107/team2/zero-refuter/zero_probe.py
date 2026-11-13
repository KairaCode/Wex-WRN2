"""zero_probe.py - adversarial probe of "FP4 -0 -> +0 is bit-identical for inference".

Reads real V4.1 REAP-192 MXFP4 experts (<RESEARCH-PC>\\v41-sample, read-only) and
1. counts structural zeros: all-zero 32-blocks, all-zero rows, all-zero aligned quads/octets
   (the GLSL vec4 dot() and CUDA 8-per-lane partial granularity) that contain a -0 code;
2. computes float32 matvecs with ORIGINAL vs CANONICAL (-0 -> +0) codes under several exact
   IEEE float32 accumulation orders (numpy elementwise ops are RN binary32):
     A  numpy matmul (BLAS, SIMD FMA, +0-initialised accumulators)
     B  sequential, acc = +0, acc += (x*w)*scale          (colibri rows16 / CUDA mv order)
     C  sequential, acc = first product, then +=           (no +0 init: what a compiler that
                                                            drops "0.0 +" (NSZ) would produce)
     D  vec4 partial ((p0+p1)+p2)+p3, then acc = first partial, += (GLSL dot() w/o +0 init)
   and reports how many outputs differ bitwise and how many are exactly zero;
3. an adversarial activation that zeroes every column where a chosen row has a nonzero
   weight, to show the only observable difference is the SIGN of an exactly-zero output.
Run: python -I zero_probe.py <sample_dir> <out_json> [n_arith_experts]
"""
import json, os, sys
import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def load_expert(path, tensors):
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in tensors:
        n = t["bytes"]
        out[t["name"]] = raw[off:off + n].reshape(t["shape"])
        off += n
    assert off == raw.size
    return out


def unpack(wb):
    rows, half = wb.shape
    codes = np.empty((rows, half * 2), dtype=np.uint8)
    codes[:, 0::2] = wb & 15
    codes[:, 1::2] = wb >> 4
    return codes


def e8(s):
    # UE8M0 -> float32 via bit pattern s<<23 (colibri mx4_scale); s=0 -> +0, s=255 -> inf
    return (s.astype(np.uint32) << 23).view(np.float32)


def struct_stats(codes):
    mag = codes & 7
    rows, cols = codes.shape
    blk = mag.reshape(rows, cols // 32, 32)
    zero_blocks = int((blk.max(axis=2) == 0).sum())
    zero_rows = int((mag.max(axis=1) == 0).sum())
    res = {"rows": rows, "cols": cols, "n": rows * cols,
           "code0": int((codes == 0).sum()), "code8": int((codes == 8).sum()),
           "zero_blocks32": zero_blocks, "blocks32": rows * cols // 32, "zero_rows": zero_rows}
    for g in (4, 8):
        m = mag.reshape(rows, cols // g, g).max(axis=2) == 0
        has8 = (codes.reshape(rows, cols // g, g) == 8).any(axis=2)
        res[f"zero_groups{g}"] = int(m.sum())
        res[f"zero_groups{g}_with_neg0"] = int((m & has8).sum())
        res[f"groups{g}"] = int(m.size)
    # CUDA mv_fp4_grouped: thread j of 256 sums i = j, j+256, ... -> strided partial
    if cols % 256 == 0:
        st = mag.reshape(rows, cols // 256, 256).max(axis=1) == 0
        res["zero_strided256_partials"] = int(st.sum())
        res["strided256_partials"] = int(st.size)
    return res


def matvec_orders(codes, scale, x):
    """Return dict order -> float32 output vector (exact binary32 RN emulation)."""
    w = E2M1[codes]                                   # rows x cols float32 (code 8 -> -0.0)
    sc = np.repeat(e8(scale), 32, axis=1)             # rows x cols
    rows, cols = w.shape
    out = {}
    # A: BLAS on the dequantised matrix (w*scale exact: power-of-two scale, no under/overflow here)
    out["A_blas"] = (w * sc) @ x
    # products (x*w)*scale, each rounded to binary32 exactly as the kernels do; column-major copy
    pT = np.ascontiguousarray(((x[None, :] * w) * sc).T)   # cols x rows
    # B: +0 init, acc += p, columns ascending
    acc = np.zeros(rows, dtype=np.float32)
    for c in range(cols):
        acc = acc + pT[c]
    out["B_seq_pos0"] = acc
    # C: first product init
    acc = pT[0].copy()
    for c in range(1, cols):
        acc = acc + pT[c]
    out["C_seq_first"] = acc
    # D: vec4 partials ((p0+p1)+p2)+p3, first-partial init
    p4 = pT.reshape(cols // 4, 4, rows)
    part = ((p4[:, 0] + p4[:, 1]) + p4[:, 2]) + p4[:, 3]   # (cols/4) x rows
    acc = part[0].copy()
    for k in range(1, part.shape[0]):
        acc = acc + part[k]
    out["D_vec4_first"] = acc
    return out


def bits(v):
    return np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)


def compare(codes, scale, x):
    canon = codes.copy()
    canon[canon == 8] = 0
    a = matvec_orders(codes, scale, x)
    b = matvec_orders(canon, scale, x)
    r = {}
    for k in a:
        diff = bits(a[k]) != bits(b[k])
        r[k] = {"bit_diffs": int(diff.sum()), "exact_zero_outputs": int((a[k] == 0).sum()),
                "neg_zero_outputs_orig": int(((a[k] == 0) & np.signbit(a[k])).sum()),
                "neg_zero_outputs_canon": int(((b[k] == 0) & np.signbit(b[k])).sum())}
    return r


def main():
    sample, out_json = sys.argv[1], sys.argv[2]
    n_arith = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    man = json.load(open(os.path.join(sample, "manifest.json")))
    rng = np.random.default_rng(20261107)
    tot = {}
    arith = []
    used = 0
    for idx, ex in enumerate(man["experts"]):
        if idx % stride:
            continue
        used += 1
        t = load_expert(os.path.join(sample, ex["file"]), ex["tensors"])
        for m in ("w1", "w2", "w3"):
            codes = unpack(t[m + ".weight"])
            s = struct_stats(codes)
            for k, v in s.items():
                if k in ("rows", "cols"):
                    continue
                tot[f"{m}.{k}"] = tot.get(f"{m}.{k}", 0) + v
            if used <= n_arith:
                scale = t[m + ".scale"]
                cols = codes.shape[1]
                rec = {"expert": ex["file"], "matrix": m}
                # (i) dense gaussian activation
                x = rng.standard_normal(cols).astype(np.float32)
                rec["gauss"] = compare(codes, scale, x)
                # (ii) sparse activation: 90% exact zeros
                xs = x.copy(); xs[rng.random(cols) < 0.9] = 0
                rec["sparse90"] = compare(codes, scale, xs)
                # (iii) adversarial: zero every column where row r has a nonzero weight
                r0 = int(np.argmax(((codes & 7) == 0).sum(axis=1)))
                xa = x.copy(); xa[(codes[r0] & 7) != 0] = 0
                ca = compare(codes, scale, xa)
                rec["adversarial_row"] = r0
                rec["adversarial"] = ca
                arith.append(rec)
                print(ex["file"], m, {k: v["bit_diffs"] for k, v in rec["gauss"].items()},
                      {k: v["bit_diffs"] for k, v in rec["sparse90"].items()},
                      {k: (v["bit_diffs"], v["exact_zero_outputs"]) for k, v in ca.items()}, flush=True)
        print("done expert", idx, ex["file"], "used", used, flush=True)
    res = {"experts_used": used, "stride": stride, "struct": tot, "arith": arith}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
