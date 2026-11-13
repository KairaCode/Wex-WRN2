"""team5/gpu-v1: summarise bench_v1.json (paired same-config ratios, min/median/max over launch configs, best configs).
Usage: python -I v1summ.py BENCH_JSON [BENCH_JSON2 ...] > table.txt ; also writes <first>_summary.json"""
import json, sys
import numpy as np

PAIRS = [('G5b', 'G5', 'A/A control (same kernel, same streams)'),
         ('G5', 'CF5g', 'G5 kernel vs team-4 CF5 kernel, same streams'), ('CF5g', 'CF5', 'team-4 CF5 kernel: v1 vs team-4 streams'),
         ('G5H', 'CF5g', 'HASH vs team-4 CF5 kernel'), ('G5HL', 'CF5g', 'v1 vs team-4 CF5 kernel, same model'),
         ('G5H', 'G5', 'HASH cost'), ('G5L', 'G5', 'LENC cost'), ('G5HL', 'G5', 'HASH+LENC cost'),
         ('G5HL', 'G5H', 'LENC on top of HASH'), ('G5HL', 'G5L', 'HASH on top of LENC'),
         ('G5HL', 'CF5', 'v1 vs team-4 CF5'), ('G5HL', 'A259', 'v1 vs A259'), ('G5', 'CF5', 'G5 vs CF5'),
         ('CF5', 'A259', 'CF5 vs A259'), ('G5', 'A259', 'G5 vs A259')]
SPAIRS = [('SH', 'SG', 'scale HASH'), ('SL', 'SG', 'scale LENC'), ('SHL', 'SG', 'scale HASH+LENC'),
          ('SHLd', 'SHL', 'DUAL write'), ('SG', 'S_t4c', 'SG vs team-4 sdec ceil'), ('SHL', 'S_t4c', 'SHL vs team-4 sdec ceil'),
          ('S_t4c', 'S_t4o', 'ceil vs orig scale streams')]
PPAIRS = [('P_G5HLc', 'P_G5c', 'pipeline HASH+LENC (copy form)'), ('P_G5HLd', 'P_G5d', 'pipeline HASH+LENC (dual form)'),
          ('P_G5HLc', 'P_CF5', 'pipeline v1(copy) vs CF5(copy)'), ('P_G5HLc', 'P_A259', 'pipeline v1(copy) vs A259'),
          ('P_G5HLd', 'P_A259', 'pipeline v1(dual) vs A259'), ('P_G5d', 'P_G5c', 'dual vs copy (G5)'),
          ('P_G5HLd', 'P_G5HLc', 'dual vs copy (v1)'), ('P_G5c', 'P_CF5', 'pipeline G5(copy) vs CF5(copy)'),
          ('P_CF5', 'P_A259', 'pipeline CF5 vs A259')]
SPAIRS_EXTRA = [('SG2', 'SG', 'DUAL 2 (end copy)'), ('SG3', 'SG', '16-B stores single'), ('SG4', 'SG', 'DUAL 4 (16-B both)'),
                ('SGd', 'SG', 'DUAL 1 (4-B both)'), ('SHL4', 'SHL', 'DUAL 4 v1'), ('SHL3', 'SHL', '16-B single v1'),
                ('SHL3', 'SG3', 'scale HASH+LENC (16-B stores)')]


def mmm(a):
    a = np.asarray(a)
    return [round(float(a.min()), 4), round(float(np.median(a)), 4), round(float(a.max()), 4)]


def main():
    out = {}
    for path in sys.argv[1:]:
        r = json.load(open(path))
        S = {'file': path, 'info': r['info'], 'nvsmi': [r.get('nvsmi_start'), r.get('nvsmi_end')]}
        print('=' * 100)
        print(path, r['info'], r.get('nvsmi_start'), '->', r.get('nvsmi_end'))
        W = r.get('weight', {})
        cfgs = sorted(W)
        sizes = None
        # resources
        print('\nweight kernels: regs / smem / occupancy (warps per SM)')
        res_t = {}
        for c in cfgs:
            row = {v: (W[c][v]['regs'], W[c][v]['smem_per_block'], W[c][v]['occ_warps_per_sm'], W[c][v]['local_bytes'])
                   for v in W[c] if 'regs' in W[c][v]}
            res_t[c] = row
            print(f'{c:11s}', '  '.join(f'{v} {a}r/{b}B/{o}w/l{l}' for v, (a, b, o, l) in row.items()))
        S['resources'] = res_t
        # exactness
        S['all_exact'] = {c: {v: W[c][v].get('all_exact') for v in W[c] if 'all_exact' in W[c][v]} for c in cfgs}
        S['negative_controls'] = {c: {v: {k: W[c][v][k] for k in ('weight_bad_streams', 'scale_bad_streams', 'errors')}
                                      for v in W[c] if v.startswith('NC_')} for c in cfgs}
        # timing
        if cfgs and 'timing' in W[cfgs[0]]['G5HL']:
            sizes = list(W[cfgs[0]]['G5HL']['timing'])
            print('\nmedian us per config')
            tt = {}
            for sz in sizes:
                print(f'-- {sz}')
                for c in cfgs:
                    t = {v: W[c][v]['timing'][sz]['us_median'] for v in W[c] if 'timing' in W[c][v]}
                    tt.setdefault(sz, {})[c] = t
                    print(f'{c:11s}', '  '.join(f'{v} {x:8.1f}' for v, x in t.items()))
            S['median_us'] = tt
            print('\nsame-config paired ratios: min / median / max over configs (per-round ratio medians)')
            rat = {}
            for a, b, lab in PAIRS:
                if 'timing' not in W[cfgs[0]].get(a, {}) or 'timing' not in W[cfgs[0]].get(b, {}):
                    continue
                for sz in sizes:
                    xs = [W[c][a]['timing'][sz][f'ratio_vs_{b}'] for c in cfgs]
                    rat.setdefault(lab, {})[sz] = {'min_med_max': mmm(xs), 'per_config': {c: round(x, 4) for c, x in zip(cfgs, xs)}}
                print(f'{lab:24s}', '  '.join(f"{sz} {rat[lab][sz]['min_med_max']}" for sz in sizes))
            S['ratios'] = rat
            print('\nbest config per variant (min over configs of median us) and best-vs-best ratios')
            best = {}
            for sz in sizes:
                for v in W[cfgs[0]]:
                    if 'timing' not in W[cfgs[0]][v]:
                        continue
                    xs = [(W[c][v]['timing'][sz]['us_median'], c) for c in cfgs]
                    m = min(xs)
                    best.setdefault(sz, {})[v] = {'us': round(m[0], 1), 'cfg': m[1]}
                print(f'-- {sz}: ' + '  '.join(f"{v} {d['us']} @{d['cfg']}" for v, d in best[sz].items()))
                bb = {lab: round(best[sz][a]['us'] / best[sz][b]['us'], 4) for a, b, lab in PAIRS if a in best[sz] and b in best[sz]}
                best[sz]['best_vs_best'] = bb
                print('   best-vs-best', bb)
            S['best'] = best
            # at the best config of G5HL: ratios
        # scale kernels
        if 'scale_timing' in r:
            print('\nscale kernels: median us and paired ratios per nt')
            sres = {}
            for nt, tm in sorted(r['scale_timing'].items()):
                sres[nt] = {}
                sp_ = SPAIRS + [p for p in SPAIRS_EXTRA if p[0] in tm and p[1] in tm]
                for sz in tm['SG']:
                    sres[nt][sz] = {'us': {v: round(tm[v][sz]['us_median'], 1) for v in tm},
                                    'ratios': {lab: round(tm[a][sz][f'ratio_vs_{b}'], 4) for a, b, lab in sp_}}
                    print(f'{nt:6s} {sz:9s}', sres[nt][sz]['us'], sres[nt][sz]['ratios'])
            S['scale'] = sres
            regs = {k: (v['regs'], v['static_smem'], v['occ_blocks_per_sm']) for k, v in r['scale_decode'].items()}
            S['scale_resources'] = regs
            print('scale regs/smem/occ', regs)
            S['scale_all_exact'] = all(v['byte_exact'] for v in r['scale_decode'].values())
        if 'pipeline' in r:
            print('\npipelines (scale kernel nt32 + weight kernel), median us; same-config ratios min/med/max')
            pres = {}
            pc = sorted(r['pipeline'])
            S['pipeline_checks'] = {c: r['pipeline'][c]['checks'] for c in pc}
            if r['pipeline'][pc[0]]['timing']:
                tp0 = r['pipeline'][pc[0]]['timing']
                szs = list(tp0[next(iter(tp0))])
                for c in pc:
                    tp = r['pipeline'][c]['timing']
                    print(f'{c:11s}', '  '.join(f"{v} " + '/'.join(f"{tp[v][sz]['us_median']:.1f}" for sz in szs) for v in tp))
                for a, b, lab in PPAIRS:
                    if a not in tp0 or b not in tp0:
                        continue
                    for sz in szs:
                        xs = [r['pipeline'][c]['timing'][a][sz][f'ratio_vs_{b}'] for c in pc]
                        pres.setdefault(lab, {})[sz] = mmm(xs)
                    print(f'{lab:30s}', pres[lab])
                bestp = {}
                for sz in szs:
                    for v in r['pipeline'][pc[0]]['timing']:
                        xs = [(r['pipeline'][c]['timing'][v][sz]['us_median'], c) for c in pc]
                        m = min(xs)
                        bestp.setdefault(sz, {})[v] = {'us': round(m[0], 1), 'cfg': m[1]}
                    print(f'best {sz}', bestp[sz])
                S['pipeline_ratios'] = pres
                S['pipeline_best'] = bestp
        out[path] = S
    json.dump(out, open(sys.argv[1].replace('.json', '_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
