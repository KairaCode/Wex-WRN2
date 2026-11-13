"""Summarise b2_final.json: fixed-config comparison and best config per variant (by m3 and by m24)."""
import json, sys
r = json.load(open(sys.argv[1]))
enc = json.load(open(sys.argv[2]))
sz = enc['sizes']
res = r['results']
print('nvsmi start/end:', r.get('nvsmi_start'), '|', r.get('nvsmi_end'))
print('all byte-exact:', all(v['byte_exact'] for v in res.values()), len(res), 'configs;',
      'multistream re-verify bad streams total:', sum(v.get('multistream_bad_streams', 0) for v in res.values()))
SZ = {'A': 'A', 'A259': 'A', 'B1': 'B1', 'B1c': 'B1', 'B2': 'B2', 'B2c': 'B2', 'C': 'C', 'CB2': 'CB2', 'D': 'D', 'DB2': 'DB2'}


def row(k, v):
    m1, m3, m24, s8 = v['m1'], v['m3'], v['m24'], v['8streams']
    return (f"{k:24s} regs {v['regs']:3d} smem {v['smem']:6d} occW {v['occ_warps_per_sm']:2d} | size {sz[SZ[v['variant']]]['delta_vs_A_pct']:+.3f}% | "
            f"m1 {m1['x_us_median']:6.1f}us x{m1['ratio_median_of_pairs']:.3f} | rec(m3) {m3['x_us_median']:6.1f}us "
            f"x{m3['ratio_median_of_pairs']:.3f} {m3['x_decoded_GiB_s']:5.1f}GiB/s | 8rec(m24) {m24['x_us_median']:7.1f}us "
            f"({m24['x_us_median']/8:6.1f}/rec) x{m24['ratio_median_of_pairs']:.3f} {m24['x_decoded_GiB_s']:5.1f}GiB/s | "
            f"8str x{s8['ratio_median_of_pairs']:.3f} {s8['x_decoded_GiB_s']:5.1f}GiB/s")


print('\n== same launch config as kernel A (32 threads/block, plain stores) ==')
for k, v in res.items():
    if '/nt32/lb' in k:
        print(row(k, v))
print('\n== best config per variant for 1 record in flight (m3) ==')
vs = sorted({v['variant'] for v in res.values()}, key=list(SZ).index)
for var in vs:
    k, v = min(((k, v) for k, v in res.items() if v['variant'] == var), key=lambda kv: kv[1]['m3']['ratio_median_of_pairs'])
    print(row(k, v))
print('\n== best config per variant for 8 records in flight (m24 single launch) ==')
for var in vs:
    k, v = min(((k, v) for k, v in res.items() if v['variant'] == var), key=lambda kv: kv[1]['m24']['ratio_median_of_pairs'])
    print(row(k, v))
print('\n== best config per variant for one matrix (m1, latency-bound) ==')
for var in vs:
    k, v = min(((k, v) for k, v in res.items() if v['variant'] == var), key=lambda kv: kv[1]['m1']['ratio_median_of_pairs'])
    print(row(k, v))
