"""Per-expert record estimate of MTP experts under (a) the main-trained model, (b) an MTP model trained on the other half.
Usage: python -I mtp_perexpert.py "DIR1;DIR2" MAIN_COUNTS_DIR"""
import json, os, sys, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_an import matrix_counts, quantize, cost_bits, W_STREAMS, S_STREAMS, RAW_RECORD

by = {}
for dd in sys.argv[1].split(';'):
    for it in json.load(open(os.path.join(dd, 'manifest.json'), encoding='utf-8'))['items']:
        by[it['name']] = {**it, 'dir': dd}
keys = sorted({tuple(map(int, re.match(r'mtp\.(\d+)\.ffn\.experts\.(\d+)\.', n).groups())) for n in by})
ex = []
for (s, e) in keys:
    W = np.zeros((3, 567, 16), np.int64); S = np.zeros((3, 256), np.int64); spread = []
    for m, w in enumerate(('w1', 'w2', 'w3')):
        wi = by[f'mtp.{s}.ffn.experts.{e}.{w}.weight']; si = by[f'mtp.{s}.ffn.experts.{e}.{w}.scale']
        wb = np.fromfile(os.path.join(wi['dir'], wi['tag'] + '.bin'), np.uint8).reshape(wi['shape'][0], -1)
        sc = np.fromfile(os.path.join(si['dir'], si['tag'] + '.bin'), np.uint8).reshape(si['shape'][0], -1)
        W[m], S[m], _ = matrix_counts(wb, sc)
        spread.append(int(sc.max()) - int(sc.min()))
    ex.append((s, e, W, S, max(spread)))
mains = [np.load(f) for f in sorted(glob.glob(os.path.join(sys.argv[2], '*.npz')))]
wq = quantize(sum(z['w_base'].astype(np.int64) for z in mains)); sq = quantize(sum(z['s_o0'].astype(np.int64) for z in mains))
A = [x for i, x in enumerate(ex) if i % 2 == 0]; B = [x for i, x in enumerate(ex) if i % 2 == 1]
qa = (quantize(sum(x[2] for x in A)), quantize(sum(x[3] for x in A)))
qb = (quantize(sum(x[2] for x in B)), quantize(sum(x[3] for x in B)))


def rec(wbits, sbits):
    return 32 + 2 * (W_STREAMS + S_STREAMS) + (wbits + sbits) / 8 + 4 * (W_STREAMS + S_STREAMS) + 2048

rows = []
for i, (s, e, W, S, sp) in enumerate(ex):
    qm = qb if i % 2 == 0 else qa
    r_main = rec(cost_bits(W, wq), cost_bits(S, sq)) / RAW_RECORD
    r_mtp = rec(cost_bits(W, qm[0]), cost_bits(S, qm[1])) / RAW_RECORD
    rows.append((s, e, sp, round(r_main, 4), round(r_mtp, 4)))
    print(f'stage {s} expert {e:3d} max scale spread {sp:2d}  ratio main-model {r_main:.4f}  mtp-model(held-out) {r_mtp:.4f}')
rm = np.array([r[3] for r in rows]); rt = np.array([r[4] for r in rows])
print('mean ratio main-model', rm.mean().round(4), 'expanding (>1):', int((rm > 1).sum()), 'of', len(rm),
      '| mean ratio mtp-model', rt.mean().round(4), '| with fallback to raw (min(main,1))', np.minimum(rm, 1).mean().round(4))
print('wide-spread experts (spread >= 6):', sum(1 for r in rows if r[2] >= 6))
