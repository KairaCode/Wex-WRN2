# SPDX-License-Identifier: Apache-2.0
"""WNR4 on small samples of other MoE models (64 routed experts each), the method of the WNR3 sample study:
models trained on the sample itself, K folds (expert i of the (layer, expert)-sorted list in fold i % K, each fold
coded with the model of the other folds), byte-exact and canonical zero, every record decoded back and compared,
ratio = stored / raw with both padded to 4096 B.

    python -I other_models.py DATA_DIR OUT.json [--folds 2] [--G 32,128] [--pb 5] [--threads 8]

DATA_DIR/manifest.json: {"model", "format": "mxfp4" | "nvfp4", "experts": [{"layer", "expert",
  "gate"/"down"/"up": {"weight": file, "weight_shape": [R, CB], "scale": file, "scale_shape": [R, CS]}}]}
(weight file: packed E2M1 bytes row-major, two nibbles a byte, element 2i low; scale file: U8 row-major).
Slab = scales gate, down, up, then weights gate, down, up (WEX4-SPEC 5.5)."""
import ctypes as C
import json
import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure import load_wnr4  # noqa: E402

ENTRY = struct.Struct('<QIII6H6H3I3II')
ORDER = ('gate', 'down', 'up')
P = C.c_void_p
ZTAB = bytes(((b & 0xF0) if b & 15 == 8 else b) & (0x0F if b >> 4 == 8 else 0xFF) for b in range(256))


def align(n):
    return (n + 4095) // 4096 * 4096


def main(argv):
    root, out = argv[1], argv[2]
    opt = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1, 2)}
    K, pb, threads = int(opt.get('--folds', 2)), int(opt.get('--pb', 5)), int(opt.get('--threads', 8))
    Gs = [int(x) for x in opt.get('--G', '32,128').split(',')]
    man = json.load(open(os.path.join(root, 'manifest.json')))
    fmt = man['format']
    kind = 1 if fmt == 'nvfp4' else 0
    experts = sorted(man['experts'], key=lambda e: (int(e['layer']), int(e['expert'])))
    e0 = experts[0]
    ss = [tuple(e0[m]['scale_shape']) for m in ORDER]
    ws = [tuple(e0[m]['weight_shape']) for m in ORDER]
    soff, woff, at = [], [], 0
    for r, c in ss:
        soff.append(at); at += r * c
    sbytes = at
    for r, c in ws:
        woff.append(at); at += r * c
    raw = at
    geo = (C.c_uint32 * 13)(kind, *[r * c for r, c in ss], *[c for r, c in ss], *soff, *woff)
    entry = ENTRY.pack(0, 0, sbytes, raw - sbytes, *[d for s in ss for d in s], *[d for s in ws for d in s], *soff, *woff, 0)
    lib = load_wnr4()
    t3 = C.CDLL(os.path.join(HERE, 'out', 'wex_nr3_tools.dll'))
    t3.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]

    def slab(e):
        parts = []
        for kindname, sh in (('scale', ss), ('weight', ws)):
            for m, shape in zip(ORDER, sh):
                assert tuple(e[m][kindname + '_shape']) == shape, 'all experts of a sample share one geometry'
                with open(os.path.join(root, e[m][kindname]), 'rb') as f:
                    b = f.read()
                assert len(b) == shape[0] * shape[1]
                parts.append(b)
        s = b''.join(parts)
        assert len(s) == raw
        return s

    def coded(src, cz):
        if kind == 0:
            outb = C.create_string_buffer(raw)
            st = (C.c_uint32 * 16)()
            flags = t3.wex_nr3_prepare(entry, src, int(cz), outb, st)
            assert flags >= 0
            return outb.raw, flags
        s = bytearray(src)
        if cz:
            s[sbytes:] = bytes(s[sbytes:]).translate(ZTAB)
        return bytes(s), 8 if cz else 0

    srcs = [slab(e) for e in experts]
    res = {'model': man.get('model'), 'format': fmt, 'experts': len(experts), 'raw_bytes': raw, 'folds': K, 'pb': pb,
           'configs': {}}
    for cz in (False, True):
        cod = [coded(s, cz) for s in srcs]
        want = [s if not cz else s[:sbytes] + s[sbytes:].translate(ZTAB) for s in srcs]
        for G in Gs:
            stored, fails = 0, 0
            for f in range(K):
                counts = (C.c_uint64 * lib.wnr4t_counts_words(kind, pb))()
                for i, (c, fl) in enumerate(cod):
                    if i % K != f:
                        assert lib.wnr4t_count(geo, c, pb, counts) == 0
                blob = C.create_string_buffer(lib.wnr4t_model_bytes(kind, pb))
                assert lib.wnr4t_build(kind, pb, counts, blob)
                model = lib.wnr4t_model_open(blob, len(blob.raw))
                lock = threading.Lock()

                def job(i):
                    c, fl = cod[i]
                    own = (int(experts[i]['layer']) << 16) | int(experts[i]['expert'])
                    cap = raw * 2 + (1 << 20)
                    buf = C.create_string_buffer(cap)
                    n = lib.wnr4t_encode(model, geo, c, fl, own, G, buf, cap)
                    assert n, ('encode', i)
                    back = C.create_string_buffer(raw)
                    err = lib.wnr4t_decode(model, geo, buf, n, own, back, 1)
                    return min(n, align(raw)), (err != 0 or back.raw != want[i])
                with ThreadPoolExecutor(threads) as pool:
                    for n, bad in pool.map(job, [i for i in range(len(cod)) if i % K == f]):
                        stored += n; fails += bad
            name = f"{'CZ' if cz else 'byte-exact'} G={G}"
            res['configs'][name] = {'ratio': stored / (align(raw) * len(experts)), 'fails': fails}
            print(f"{man.get('model')} {fmt}: {name}: {stored / (align(raw) * len(experts)):.4f} (fails {fails})", flush=True)
    with open(out, 'w') as g:
        json.dump(res, g, indent=1)


if __name__ == '__main__':
    main(sys.argv)
