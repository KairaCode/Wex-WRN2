"""How many 32-blocks use no odd half-step magnitudes (0.5 = mag 1, 1.5 = mag 3)?  All 160 experts, from the
block histograms. Also: block-max distribution inside those blocks, per srel / matrix type / layer, and the most
common 'magnitude-presence masks'.
Usage: python -I evenblocks.py SAMPLE_DIR HIST_DIR OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parametric import decode_keys

sample_dir, hist_dir, out_json = sys.argv[1:4]
ex = json.load(open(os.path.join(sample_dir, 'manifest.json')))['experts']
by_srel = np.zeros((3, 7, 2))  # [type, srel, (all, even)]
by_layer = np.zeros((40, 2))
max_even = np.zeros(8); max_all = np.zeros(8)
masks = np.zeros(256)
masks_srel0 = np.zeros(256)
nonzero_even = np.zeros(33)  # number of nonzero codes in even blocks (to see how 'informative' they are)
for e in ex:
    z = np.load(os.path.join(hist_dir, e['file'].replace('.bin', '.npz')), allow_pickle=False)
    for m in range(3):
        k = z[f'key_m{m}']; w = z[f'cnt_m{m}'].astype(np.float64)
        s, N = decode_keys(k)
        even = (N[:, 1] == 0) & (N[:, 3] == 0)
        np.add.at(by_srel[m, :, 0], s, w)
        np.add.at(by_srel[m, :, 1], s[even], w[even])
        by_layer[e['layer'], 0] += w.sum(); by_layer[e['layer'], 1] += w[even].sum()
        bmax = 7 - np.argmax(N[:, ::-1] > 0, axis=1)
        np.add.at(max_all, bmax, w); np.add.at(max_even, bmax[even], w[even])
        mask = ((N > 0) * (1 << np.arange(8))).sum(1).astype(np.int64)
        np.add.at(masks, mask, w)
        np.add.at(masks_srel0, mask[s == 3], w[s == 3])
        nz = 32 - N[:, 0]
        np.add.at(nonzero_even, nz[even].astype(np.int64), w[even])
tot = by_srel[..., 0].sum()
out = {
    'blocks_total': tot,
    'even_fraction_all': by_srel[..., 1].sum() / tot,
    'even_fraction_per_type': (by_srel[:, :, 1].sum(1) / by_srel[:, :, 0].sum(1)).tolist(),
    'even_fraction_per_srel(-3..3)': (by_srel[:, :, 1].sum(0) / np.maximum(by_srel[:, :, 0].sum(0), 1)).tolist(),
    'block_share_per_srel': (by_srel[:, :, 0].sum(0) / tot).tolist(),
    'even_fraction_per_layer': (by_layer[:, 1] / by_layer[:, 0]).round(4).tolist(),
    'blockmax_hist_all (mag 0..7)': (max_all / max_all.sum()).round(5).tolist(),
    'blockmax_hist_even': (max_even / max_even.sum()).round(5).tolist(),
    'nonzero_codes_in_even_blocks_mean': float((nonzero_even * np.arange(33)).sum() / nonzero_even.sum()),
}
top = np.argsort(-masks)[:12]
out['top_presence_masks (bit m = magnitude m present)'] = [[int(t), format(int(t), '08b')[::-1], float(masks[t] / masks.sum())] for t in top]
top0 = np.argsort(-masks_srel0)[:8]
out['top_masks_srel0'] = [[format(int(t), '08b')[::-1], float(masks_srel0[t] / masks_srel0.sum())] for t in top0]
print(json.dumps(out, indent=1))
json.dump(out, open(out_json, 'w'), indent=1)
