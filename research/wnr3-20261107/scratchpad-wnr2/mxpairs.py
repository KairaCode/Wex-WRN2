"""Cross-expert redundancy probes on the sample.

Usage: python -I mxpairs.py SAMPLE_DIR OUT_JSON
1. Mutual information I(x_a; x_b) between FP4 codes at the same position of two experts of the same layer
   (w1, w2, w3), in bits/weight: an upper bound on what any delta/XOR or 'other expert as context' coding saves.
2. Exact duplicate rows (per matrix) across all sampled experts, and duplicate whole matrices.
3. Same for scales: I(s_a; s_b) at the same position.
"""
import json, os, sys, hashlib, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert


def mi(a, b, n):
    j = np.bincount(a.astype(np.int64) * n + b.astype(np.int64), minlength=n * n).reshape(n, n).astype(np.float64)
    j /= j.sum()
    pa = j.sum(1, keepdims=True); pb = j.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(j > 0, j * np.log2(j / (pa * pb)), 0)
    return float(t.sum())


def syms(wb):
    s = np.empty((wb.shape[0], wb.shape[1] * 2), np.uint8)
    s[:, 0::2] = wb & 15
    s[:, 1::2] = wb >> 4
    return s


def main():
    sample_dir, out_json = sys.argv[1:3]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    by_layer = {}
    for e in man['experts']:
        by_layer.setdefault(e['layer'], []).append(e)
    mis = {'w1': [], 'w2': [], 'w3': []}
    smis = {'w1': [], 'w2': [], 'w3': []}
    row_hashes = {}
    mat_hashes = {}
    for L in sorted(by_layer)[::4]:  # every 4th layer keeps it quick (10 layers x 6 pairs)
        exs = [load_expert(os.path.join(sample_dir, e['file']), e['tensors']) for e in by_layer[L]]
        for a, b in itertools.combinations(range(len(exs)), 2):
            for w in ['w1', 'w2', 'w3']:
                wa, shp, _ = exs[a][f'{w}.weight']; wbb, _, _ = exs[b][f'{w}.weight']
                sa = syms(wa.reshape(shp[0], -1)); sb = syms(wbb.reshape(shp[0], -1))
                mis[w].append(mi(sa.ravel(), sb.ravel(), 16))
                ca, _, _ = exs[a][f'{w}.scale']; cb, _, _ = exs[b][f'{w}.scale']
                # clamp to median +-7 (15 symbols) so the plug-in MI estimate is not dominated by bias
                q = lambda c: (np.clip(c.astype(np.int16) - int(np.sort(c)[(c.size - 1) // 2]), -7, 7) + 7).astype(np.uint8)
                smis[w].append(mi(q(ca), q(cb), 15))
    for e in man['experts']:
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        for w in ['w1', 'w2', 'w3']:
            raw, shp, _ = t[f'{w}.weight']
            sc, _, _ = t[f'{w}.scale']
            m = raw.reshape(shp[0], -1)
            s = sc.reshape(shp[0], -1)
            hm = hashlib.sha256(m.tobytes() + s.tobytes()).hexdigest()
            mat_hashes.setdefault((w, hm), []).append((e['layer'], e['expert']))
            for r in range(m.shape[0]):
                h = hashlib.blake2b(m[r].tobytes() + s[r].tobytes(), digest_size=16).digest()
                row_hashes.setdefault((w, h), 0)
                row_hashes[(w, h)] += 1
    dup_rows = {w: sum(c - 1 for (ww, _), c in row_hashes.items() if ww == w and c > 1) for w in ['w1', 'w2', 'w3']}
    tot_rows = {w: sum(c for (ww, _), c in row_hashes.items() if ww == w) for w in ['w1', 'w2', 'w3']}
    dup_mats = [v for v in mat_hashes.values() if len(v) > 1]
    out = {
        'mi_same_position_bits_per_weight': {w: {'mean': float(np.mean(v)), 'max': float(np.max(v)), 'pairs': len(v)} for w, v in mis.items()},
        'mi_same_position_bits_per_scale': {w: {'mean': float(np.mean(v)), 'max': float(np.max(v)), 'pairs': len(v)} for w, v in smis.items()},
        'duplicate_rows (row incl. its scales, across all sampled experts)': {w: [dup_rows[w], tot_rows[w]] for w in dup_rows},
        'duplicate_matrices': dup_mats[:20],
    }
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
