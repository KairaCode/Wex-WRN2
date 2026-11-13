# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import json, sys
import numpy as np
mine = json.load(open('out/proto_verify.json')); theirs = json.load(open('../out/proto.json'))
names = [c['name'] for c in mine['configs']]
assert names == [c['name'] for c in theirs['configs']]
T = {(e['layer'], e['expert']): e for e in theirs['experts']}
M = {(e['layer'], e['expert']): e for e in mine['experts']}
print('all ok flags mine:', all(all(c[0] for c in e['cfg']) and e['harness_ok'] for e in mine['experts']))
# overlap: exact equality of every size column
for k in M:
    if k in T:
        a = [c[:7] for c in M[k]['cfg']]; b = [c[:7] for c in T[k]['cfg']]
        print('overlap', k, 'identical sizes' if a == b and M[k]['used_bytes'] == T[k]['used_bytes'] else 'DIFFERENT')
fresh = [e for e in mine['experts'] if (e['layer'], e['expert']) not in T]
ext = {(37, 150), (39, 43), (39, 105)}
f16 = [e for e in fresh if (e['layer'], e['expert']) not in ext]
print('fresh records:', len(fresh), 'of which regular', len(f16))
def saved(es, i): return np.array([e['cfg'][0][1] - e['cfg'][i][1] for e in es], float)
print(f"{'cfg':28s} {'theirs41':>9s} {'fresh16':>9s} {'diff%':>7s} {'f16 min':>8s} {'f16 max':>8s}   extremes(37:150 largest, 39:43 smallest, 39:105 worst-lenc)")
out = {}
for i, n in enumerate(names):
    t = saved(theirs['experts'], i).mean(); m = saved(f16, i)
    ex = [int(M[k]['cfg'][0][1] - M[k]['cfg'][i][1]) for k in sorted(ext)]
    out[n] = {'theirs_mean_41': round(t, 1), 'fresh16_mean': round(m.mean(), 1), 'fresh16_min': int(m.min()), 'fresh16_max': int(m.max()), 'extremes': ex}
    if i: print(f"{n:28s} {t:9.0f} {m.mean():9.0f} {100*(m.mean()-t)/t:7.2f} {m.min():8.0f} {m.max():8.0f}   {ex}")
# overhead
def ovh(es):
    r = []
    for e in es:
        ideal = (e['ideal_weight_bits'] + e['ideal_scale_bits']) / 8
        c0 = e['cfg'][0]
        r.append((c0[3] + c0[4] - ideal) + 2 * c0[5])
    return np.array(r)
print('skeleton overhead B/record: theirs', ovh(theirs['experts']).mean(), 'fresh16', ovh(f16).mean(), 'all mine', ovh(mine['experts']).mean())
print('bits/weight model: theirs', np.mean([e['ideal_weight_bits']/e['symbols'] for e in theirs['experts']]), 'fresh16', np.mean([e['ideal_weight_bits']/e['symbols'] for e in f16]))
print('3-byte share fresh16', np.mean([e['cfg'][2][6]/e['cfg'][2][5] for e in f16]))
print('zero share fresh16', np.mean([e['zero_magnitudes']/e['symbols'] for e in f16]))
print('ref decode s (8 thr busy) mine', np.mean([e['ref_decode_s'] for e in mine['experts']]))
print('offsets us u16/lenc mine', np.mean([e['offsets_u16_us'] for e in mine['experts']]), np.mean([e['offsets_lenc_us'] for e in mine['experts']]))
print('flip rejected mine', {n: sum(e['cfg'][i][7] for e in mine['experts']) for i, n in enumerate(names) if n.startswith('lenc_sd') and '4096' not in n and '8192' not in n})
print('used mean fresh16', np.mean([e['used_bytes'] for e in f16]), 'theirs', np.mean([e['used_bytes'] for e in theirs['experts']]))
json.dump(out, open('out/compare.json', 'w'), indent=1)
