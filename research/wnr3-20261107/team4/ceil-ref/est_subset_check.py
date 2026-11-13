import json, os, sys, numpy as np
sys.path.insert(0, r'<RESEARCH-PC>\team4\ceil-ref')
from wnr_common import quantize, P
man = json.load(open(r'<RESEARCH-PC>\v41-sample\manifest.json'))
Z = [np.load(os.path.join(r'<RESEARCH-PC>\v41-ceil', e['file'].replace('.bin', '.npz'))) for e in man['experts']]
test = [4*l+1 if l % 2 == 0 else 4*l+3 for l in range(40)]
even = [i for i in range(160) if i % 2 == 0]
def costs(train):
    wo = quantize(sum(Z[i]['w_orig'] for i in train)); wc = quantize(sum(Z[i]['w_ceil'] for i in train))
    so = quantize(sum(Z[i]['s_orig'] for i in train)); sc = quantize(sum(Z[i]['s_ceil'] for i in train))
    out = []
    for z in Z:
        bo = (z['w_orig'] * -np.log2(wo / P)).sum() + (z['s_orig'] * -np.log2(so / P)).sum()
        bc = (z['w_ceil'] * -np.log2(wc / P)).sum() + (z['s_ceil'] * -np.log2(sc / P)).sum()
        out.append((bc - bo) / 8)
    return np.array(out)
d_even = costs(even)
odd = [i for i in range(160) if i % 2]
RAW = 18800640
print('even-trained models: mean delta B  test40 %.1f (%.3f%% raw)  all odd80 %.1f (%.3f%%)  first14(layers0-13) %.1f' % (
    d_even[test].mean(), 100*d_even[test].mean()/RAW, d_even[odd].mean(), 100*d_even[odd].mean()/RAW, d_even[test[:14]].mean()))
lay = np.array([e['layer'] for e in man['experts']])
evf = np.array([z['stats'][:, 8].sum() / z['stats'][:, :8].sum() for z in Z])
for g in range(0, 40, 8):
    m = (lay >= g) & (lay < g + 8)
    print('layers %d-%d mean delta %.0f B (%.3f%% raw) even frac %.3f' % (g, g+7, d_even[m].mean(), 100*d_even[m].mean()/RAW, evf[m].mean()))
print('corr(even frac, delta)', np.corrcoef(evf, d_even)[0, 1])
print('all160 in-sample-from-even mean %.1f (%.3f%%)' % (d_even.mean(), 100*d_even.mean()/RAW))
