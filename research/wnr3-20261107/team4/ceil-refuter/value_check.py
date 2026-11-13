"""Is T value-preserving? Dequantize original and T-transformed blocks of one real expert (all 3 matrices) to float64
(E2M1 value x 2^(s-127)) and compare. Usage: python -I -B value_check.py SAMPLE_DIR OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refute_count as rc

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def deq(blocks, sc):
    sym = rc.unpack(blocks)
    return E2M1[sym] * np.exp2(sc.astype(np.float64) - 127)[:, None], sym


def main():
    d, outp = sys.argv[1:3]
    man = json.load(open(os.path.join(d, 'manifest.json')))
    res = []
    for e in man['experts'][:2]:
        raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
        off, parts = 0, {}
        for t in e['tensors']:
            parts[t['name']] = raw[off:off + t['bytes']]; off += t['bytes']
        for w in ['w1', 'w2', 'w3']:
            b = parts[f'{w}.weight'].reshape(-1, 16); s = parts[f'{w}.scale']
            tw, ts, even, _ = rc.T_fwd(b, s)
            v0, s0 = deq(b, s); v1, s1 = deq(tw, ts)
            res.append({'file': e['file'], 'm': w, 'values_equal': bool(np.array_equal(v0, v1)),
                        'signbit_equal': bool(np.array_equal(np.signbit(v0), np.signbit(v1))),
                        'bytes_changed_blocks': int(even.sum())})
    json.dump(res, open(outp, 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
