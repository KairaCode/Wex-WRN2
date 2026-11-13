"""Plain-loop check of hfeat fields on random positions of one real expert."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hfeat
from mxcount import load_expert

SD = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
man = json.load(open(os.path.join(SD, 'manifest.json')))
e = man['experts'][int(sys.argv[1]) if len(sys.argv) > 1 else 7]
t = load_expert(os.path.join(SD, e['file']), e['tensors'])
mats = hfeat.ceil_mats(t)
stride, idx = 16, 7
res = hfeat.expert_words(t, lambda m, R: np.arange((-idx) % stride, R, stride))
rng = np.random.default_rng(1)
bad = 0
for m in range(3):
    sym, S = mats[m]
    R, C = sym.shape; nb = C // 32
    rows = np.arange((-idx) % stride, R, stride)
    W, B = res[m]
    assert W.size == rows.size * C and B.size == rows.size * nb
    for _ in range(3000):
        k = int(rng.integers(rows.size)); col = int(rng.integers(C)); r = int(rows[k])
        b = col // 32; j = col % 32
        wi = (k * C + col); bi = k * nb + b
        g = {f[1]: int(hfeat.get(W[wi:wi + 1], f[1])[0]) if f[0] == 'W' else int(hfeat.get(B[bi:bi + 1], f[1])[0])
             for f in hfeat.FIELDS}
        blk = [int(x) for x in sym[r, 32 * b:32 * b + 32]]
        mag = [x & 7 for x in blk]
        h = mag[:j]
        exp = {'sym': blk[j], 'pm': h[-1] if j else 8, 'mx': max(h) if j else 8, 'pm2': h[-2] if j >= 2 else 8,
               'pos': j, 'psign': (1 + (blk[j - 1] >> 3)) if j else 0, 'nz': min(h.count(0), 15)}
        if j == 0:
            run = 0
        else:
            run = 1
            while run < j and h[-1 - run] == h[-1]:
                run += 1
            run = min(run, 7)
        exp['run'] = run
        exp['seen13'] = int(any(x in (1, 3) for x in h))
        exp['msum'] = 7 if j == 0 else min(max((2 * sum(h)) // j, 2), 8) - 2
        exp['nmax'] = 0 if j == 0 else min(h.count(max(h)), 3)
        exp['pmag'] = (int(mats[0][0][r, col]) & 7) if m == 2 else 8
        s = int(S[r, b])
        exp['srel15'] = min(max(s - 120, -7), 7) + 7
        exp['bpar'] = b % 2; exp['bcolq'] = b * 4 // nb
        pv = b > 0 and (r * nb + b) % 64 != 0
        pmg = [int(x) & 7 for x in sym[r, 32 * (b - 1):32 * b]] if b > 0 else []
        exp['pbmax'] = max(pmg) - 5 if pv else 3
        exp['pbsum'] = int(np.searchsorted(hfeat.PBSUM_T, sum(pmg), side='right')) if pv else 7
        exp['pbzero'] = min(pmg.count(0), 6) if pv else 7
        dc = lambda a: min(max(a - s, -2), 2) + 2
        exp['dl'] = dc(int(S[r, b - 1])) if b > 0 else 5
        exp['dr'] = dc(int(S[r, b + 1])) if b < nb - 1 else 5
        exp['du'] = dc(int(S[r - 1, b])) if r > 0 else 5
        exp['dd'] = dc(int(S[r + 1, b])) if r < R - 1 else 5
        if m == 1:
            exp['pscale'] = 5; exp['pbmx'] = 3
        else:
            ps = mats[2 - m][1]
            exp['pscale'] = dc(int(ps[r, b]))
            exp['pbmx'] = (max(int(x) & 7 for x in mats[0][0][r, 32 * b:32 * b + 32]) - 5) if m == 2 else 3
        for kk, v in exp.items():
            if g[kk] != v:
                bad += 1
                if bad < 20:
                    print('MISMATCH', m, r, col, kk, g[kk], v)
print('mismatches', bad)
