"""Critic arithmetic cross-checks (team 4). Reads only other members' result files; writes nothing outside stdout."""
import json, math

RAW = 18800640
NW = 35389440
NS = 1105920
NSTREAM = 17280 + 2160
REC_START = 4096 + 61440 + 552960  # header page + align4096(model<=61440) + entry table
PROD_WNR2 = 16914680.0

CR = r'<RESEARCH-PC>\team4\ceil-ref\results\heldout_summary.json'
CRG = r'<RESEARCH-PC>\team4\ceil-ref\results\heldout_summary_g0.json'
out = {}

def chk(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'OK ' if ok else 'BAD'} {name}: computed {got:.6g} vs stated {want:.6g} (tol {tol})")
    return ok

print('== 1. ceil-ref configs: ratio, bit/w, bit/scale, parts')
bad = 0
for fn in (CR, CRG):
    d = json.load(open(fn))
    for k, c in d['configs'].items():
        mp = c['mean_parts']
        tot = c['mean_total_bytes']
        if abs(sum(mp.values()) - tot) > 0.5: bad += 1; print('BAD parts', fn, k)
        if abs(tot / RAW - c['ratio_of_raw']) > 1e-9: bad += 1; print('BAD ratio', fn, k)
        bw = mp['weight_payload'] * 8 / NW
        if abs(bw - c['bit_per_w (weight payload incl. flush and side symbols)']) > 1e-9: bad += 1; print('BAD bw', fn, k)
        bs = mp['scale_payload'] * 8 / NS
        if abs(bs - c['bit_per_scale (scale payload incl. flush)']) > 1e-9: bad += 1; print('BAD bs', fn, k)
        allin = c['mean_used_bytes_unpadded'] * 8 / NW
        if abs(allin - c['all_in_bit_per_w (used bytes)']) > 1e-9: bad += 1; print('BAD allin', fn, k)
print('ceil-ref internal mismatches:', bad)

g = json.load(open(CRG))['configs']
c0 = json.load(open(CR))['configs']
w2 = c0['wnr2']['mean_total_bytes']
print('\n== 2. spec section 10 table (capacity = 1/ratio - 1; container = 7680 x mean + 618,496)')
spec10 = {  # name: (cfg dict, key, stated ratio, stated pp, stated capacity %, stated GB)
    'WNR2': (c0, 'wnr2', 0.89943, 0.0, 11.18, 129.9),
    'f': (c0, 'f', 0.88182, -1.761, 13.40, 127.3),
    'f g0': (g, 'fg', 0.88152, -1.791, 13.44, 127.3),
    'f g0 d': (g, 'fgd', 0.88011, -1.932, 13.62, 127.1),
    'f g0 d e': (g, 'fgde', 0.88021, -1.922, 13.61, 127.1),
    'f g0 d t': (g, 'fgdt', 0.87800, -2.144, 13.90, 126.8),
    'f g0 c': (g, 'fgc', 0.85411, -4.532, 17.08, 123.3),
    'f g0 c d': (g, 'fgcd', 0.85275, -4.669, 17.27, 123.1),
    'f g0 c d e': (g, 'fgcde', 0.85282, -4.661, 17.26, 123.1),
}
for n, (cfg, k, r, pp, cap, gb) in spec10.items():
    t = cfg[k]['mean_total_bytes']
    chk(f'{n} ratio', t / RAW, r, 6e-6)
    chk(f'{n} pp', (t - w2) / RAW * 100, pp, 6e-4)
    chk(f'{n} capacity%', (RAW / t - 1) * 100, cap, 6e-3)
    chk(f'{n} GB(held-out mean)', (t * 7680 + REC_START) / 1e9, gb, 0.051)
    print(f'     GB anchored on production WNR2: {((PROD_WNR2 + t - w2) * 7680 + REC_START)/1e9:.3f}')

print('\n== 3. spec v1/v1z estimates')
v1 = g['fgdt']['mean_total_bytes'] + 2436 + 16
chk('v1 spec layout ratio', v1 / RAW, 0.8781, 6e-5)
v1z = g['fgcd']['mean_total_bytes'] - 40199 + 2452
chk('v1z bytes', v1z, 15994400, 50)
chk('v1z ratio', v1z / RAW, 0.8507, 6e-5)
print('  v1z basis note: includes +2,452 B spec layout; the v1 row in the same table (0.87800) does not')
print('  tail saving in CZ domain unknown; fgdt-fgd measured only without CZ')

print('\n== 4. spec tok/s claim (section 10: v1 +1.4% U / +2.3% E; v1z +3.2% / +5.3%) vs elasticity definition')
eU, eE = 0.6473301620858796, 1.0853256217749432  # team 3 median elasticities (capacity_update_output.txt)
for name, r in (('v1 (spec layout)', v1 / RAW), ('v1 (ref layout fgdt g0)', g['fgdt']['mean_total_bytes'] / RAW), ('v1z est', v1z / RAW)):
    rw2 = w2 / RAW
    rel = (rw2 / r - 1) * 100       # % fewer bytes relative to WNR2 (the elasticity's definition)
    pp = (rw2 - r) * 100            # pp of raw (what the spec multiplied)
    print(f'  {name}: ratio {r:.5f}; %fewer bytes vs WNR2 {rel:.3f} -> U {eU*rel:.2f} E {eE*rel:.2f}; '
          f'pp-of-raw {pp:.3f} -> U {eU*pp:.2f} E {eE*pp:.2f}')
print('  capacity-update simulated: WNR3t_g0 U 1.61 E 2.63; WNR3zt_est U 3.65 E 6.23 (capacity_update_output.txt:36,55,38,57)')

print('\n== 5. capacity-update ratios and containers')
deltas = {'WNR3': (c0, 'fgde'), 'WNR3t': (c0, 'fgdt'), 'WNR3z': (c0, 'fgcde'),
          'WNR3_g0': (g, 'fgde'), 'WNR3t_g0': (g, 'fgdt'), 'WNR3z_g0': (g, 'fgcde'), 'F': (c0, 'f')}
stated = {'WNR3': (0.88074, 127.169363968), 'WNR3t': (0.87853, 126.850072576), 'WNR3z': (0.85335, 123.21439744),
          'WNR3_g0': (0.88047, 127.130042368), 'WNR3t_g0': (0.87825, 126.809964544), 'WNR3z_g0': (0.85307, 123.17507584),
          'F': (0.88207, 127.362039808)}
for n, (cfg, k) in deltas.items():
    rec = PROD_WNR2 + cfg[k]['mean_total_bytes'] - w2
    chk(f'{n} ratio', rec / RAW, stated[n][0], 6e-6)
    chk(f'{n} container GB', (rec * 7680 + REC_START) / 1e9, stated[n][1], 1e-6)
    print(f'     bytes fewer vs prod WNR2 {(PROD_WNR2/rec-1)*100:.3f}%  vs WEX1 {(RAW/rec-1)*100:.3f}%')
chk('WNR2 prod container GB', (PROD_WNR2 * 7680 + REC_START) / 1e9, 129.905360896, 1e-9)
chk('WEX1 container GB', (RAW * 7680 + REC_START) / 1e9, 144.389472256, 1e-9)
print('  U: margins at 122.6 GB:', {n: round(122.6 - stated[n][1], 2) for n in ('WNR3', 'WNR3t', 'WNR3z')})

print('\n== 6. ceil-gpu record bytes (encode_report*.json)')
er = json.load(open(r'<RESEARCH-PC>\team4\ceil-gpu\enc\encode_report.json'))['sizes']
er2 = json.load(open(r'<RESEARCH-PC>\team4\ceil-gpu\enc\encode_report_DB2_CFDB2.json'))['sizes']
for n, src in (('A259', er), ('D', er), ('CF', er), ('CFD', er), ('DB2', er2), ('CFDB2', er2)):
    s = src[n]
    pr = s['record_total_bytes_all8'] / 8
    print(f"  {n}: per record {pr:,.1f} ratio {pr/RAW:.5f} rANS bit/w {s['weight_payload_bytes']*8/(8*NW):.5f} "
          f"(no record header, no 4096 padding, u16 lengths, 4 B flush)")
print(f"  CF vs A259 % of record {(er['CF']['record_total_bytes_all8']/er['A259']['record_total_bytes_all8']-1)*100:.3f}, "
      f"% of raw {(er['CF']['record_total_bytes_all8']-er['A259']['record_total_bytes_all8'])/8/RAW*100:.3f}")
print(f"  CFDB2 vs CF % of record {(er2['CFDB2']['record_total_bytes_all8']/er['CF']['record_total_bytes_all8']-1)*100:.3f} (spec A.3 says +0.36%)")
print(f"  CFD - CF B/record {(er['CFD']['record_total_bytes_all8']-er['CF']['record_total_bytes_all8'])/8:.1f}")

print('\n== 7. start-state arithmetic')
e_bits = (2 * math.log(2) - 1) / math.log(2)
print(f'  HASH expected cost 19,440 x E[log2(1+U)] / 8 = {NSTREAM*e_bits/8:.1f} B (measured +1,351.6 used)')
print(f'  TAIL start-state cost log2(1+2^20/2^23) = {math.log2(1+2**20/2**23):.4f} bit/stream; window acceptance 2^20/(2^31-2^23) = 2^{math.log2(2**20/(2**31-2**23)):.2f}; scales 2^{math.log2(2**16/(2**31-2**23)):.2f}')
print(f'  measured tail saving per stream {40199.325/NSTREAM:.3f} B')
print(f'  g0 vs f in bit/w: {-6028.4*8/NW:.5f}')

print('\n== 8. MV and brief figures')
print(f'  MV -339.5 KB -> {-339507.8/RAW*100:.4f}% of raw; 0.8997-0.01806 = {0.8997-0.018058:.4f}; capacity {1/0.88164-1:.4f}')
print(f'  ceil-gpu in-sample delta -2.07% vs CR held-out -1.761% on different experts; gpu set includes idx 159 (orig medians 121,121,121)')

print('\n== 9. statistical check on refuter flip rates (Poisson z for orig vs ceil)')
for m, a, b in (('w1', 63, 82), ('w2', 69, 139)):
    z = (b - a) / math.sqrt(a + b)
    print(f'  {m}: orig {a}/24000 ceil {b}/24000  z = {z:.2f}')
print('  team 2 audit w2 62/12,000 vs refuter orig w2 69/24,000: rates 0.517% vs 0.288%, z = '
      f'{(62/12000-69/24000)/math.sqrt(62/12000**2+69/24000**2):.2f}')

print('\n== 10. fallback penalty: CR measured vs RF estimate')
print(f'  CR test_ceil fg fallback 17,285,120 vs WNR2 16,887,808: {(17285120-16887808)/RAW*100:.3f}% of raw; '
      f'vs same expert in ceil form 16,572,416: {(17285120-16572416)/RAW*100:.3f}% of raw')
print('  RF estimate +12.65% (P2) / +11.73% (P4) of raw with median anchors (ceil model trained at anchor 121, plain record at 120)')
