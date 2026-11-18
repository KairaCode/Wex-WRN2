# Why L39 E120's scale payload is 1.08 MB: ideal cost of its coded scales under the production model's absolute
# order-0 scale tables, per value; and the anchors of the 7,680 records (headers.bin).
# usage: python3 explain_outlier.py > out/explain_outlier.txt
import numpy as np, json, struct, collections
b = open('/tmp/wnr3-data/recs/model.bin', 'rb').read()
meta = {(r['layer'], r['expert']): r for r in json.load(open('meta.json'))}
for key in [(39, 120), (39, 154), (0, 7)]:
    r = meta[key]; s = np.fromfile('scales/%02d_%03d.bin' % key, np.uint8); off = 0; tot = 0
    print(f"L{key[0]} E{key[1]} anchors {r['anchors']} stored SP {r['SP']}")
    for m, (R, C) in enumerate(r['scale_shapes']):
        a = s[off:off + R * C]; off += R * C
        f = np.diff(np.frombuffer(b[57866 + m * 514:57866 + (m + 1) * 514], np.uint16).astype(int))
        h = np.bincount(a, minlength=256); bits = -(h * np.log2(np.maximum(f, 1) / 16384)).sum(); tot += bits / 8
        print(f"  m{m}: ideal {bits/8:9.0f} B  " + ", ".join(f"{v}: n={h[v]} f={f[v]}/16384 ({-np.log2(f[v]/16384):.1f} bit)" for v in np.nonzero(h)[0]))
    print(f"  total ideal {tot:.0f} B")
h = open('headers.bin', 'rb').read()
A = [tuple(h[32 * i + 24:32 * i + 27]) for i in range(7680)]
SP = np.array([struct.unpack_from('<I', h, 32 * i + 8)[0] for i in range(7680)])
print("anchor triples over 7,680 records:", collections.Counter(A).most_common())
sh = np.array([a != (120, 120, 120) for a in A])
print(f"records with an anchor != 120: {sh.sum()}, their SP: mean {SP[sh].mean():.0f}, sum {SP[sh].sum()}; others: mean {SP[~sh].mean():.0f}")
print("10 largest SP:", [(i // 192, i % 192, int(SP[i]), A[i]) for i in np.argsort(-SP)[:10]])
