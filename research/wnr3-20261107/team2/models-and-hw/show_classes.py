"""Print fitted latent classes (theta_k over magnitudes 0..7, pi_k) for one matrix type and each srel.
Usage: python -I show_classes.py SAMPLE_DIR HIST_DIR M NTRAIN K"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parametric import pool, decode_keys
from lclass import fit

sample_dir, hist_dir, m, ntr, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ex = json.load(open(os.path.join(sample_dir, 'manifest.json')))['experts']
s0 = [e['file'] for i, e in enumerate(ex) if i % 2 == 0]
files = [s0[int(j)] for j in np.linspace(0, len(s0) - 1, ntr).round()]
k, w = pool(files, hist_dir, m)
st, N = decode_keys(k)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
for s in range(7):
    sel = st == s
    if w[sel].sum() < 1000:
        continue
    (th, pi), ll, its = fit(N[sel], w[sel], K)
    o = np.argsort(-pi)
    print(f'srel {s - 3:+d}: blocks {w[sel].sum():.0f} iters {its}')
    for j in o:
        mean = (th[j] * np.arange(8)).sum()
        print(f'   pi {pi[j]:.4f}  theta {th[j]}  mean mag {mean:.3f}  P(6or7) {th[j][6] + th[j][7]:.4f}')
    emp = (N[sel] * w[sel][:, None]).sum(0); emp /= emp.sum()
    print(f'   empirical     {emp}')
