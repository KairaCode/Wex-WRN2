"""W3 pilots: C (WNR2 + compressed RAM cache at equal bytes, ab/run_w3c.sh) and the first F1 gating grid
(f1/run_f1.sh). Per run and prompt: decode tok/s, TTFT, moeprof wait/gpu, disk counters; paired deltas;
greedy token identity (against the main ABBA's tokens.json prefix too) and logits sha256.

  python -I analyze_pilots.py DUMPDIR   -> pilots.json next to this script, text on stdout
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # -I leaves the script's folder off sys.path
import analyze_w3 as w   # noqa: E402

PILOT_C = ['w3p-pA1-wex1', 'w3p-pC1-wnr2crc', 'w3p-pC2-wnr2crc', 'w3p-pA2-wex1']
C_PAIRS = [('w3p-pA1-wex1', 'w3p-pC1-wnr2crc'), ('w3p-pA2-wex1', 'w3p-pC2-wnr2crc')]
F1 = ['f1-base', 'f1-t10', 'f1-t15', 'f1-m287', 'f1-m144', 'f1-base2']


def done(rundir, name):
    log = rundir / f'{name}.stderr.log'
    return log.exists() and 'wexprof prefetch' in log.read_text(encoding='utf-8', errors='replace')


def summarize(rundir, names, dump):
    w.RUNS = rundir
    names = [n for n in names if done(rundir, n)]
    runs = {n: w.run_summary(n) for n in names}
    toks, shas = {}, {}
    for n in names:
        toks[n], shas[n] = w.token_ids(dump, n, w.load(n)[0])
    for n in names:
        log = (rundir / f'{n}.stderr.log').read_text(encoding='utf-8', errors='replace').splitlines()
        runs[n]['crc'] = w.last_line(log, 'RAM cache of compressed records')
        runs[n]['npprof_total'] = w.last_line(log, 'npprof total')
    return runs, toks, shas


def show(runs, shas):
    for n, r in runs.items():
        print(f"{n}: READY {r['ready_s']} s, RAM_GB {r['ram_gb']}, logits {shas[n][:16]}, disk reads/fwd "
              f"{r['disk_reads_per_fwd']:.1f} ({r['host_reads']:.0f} host + {r['direct_reads']:.0f} direct), "
              f"{r['disk_gb_per_fwd']:.2f} GB/fwd | {r['ram_tiers'][:110]}")
        for extra in ('crc', 'npprof_total'):
            if r[extra]:
                print('   ' + r[extra])
        for p in r['prompts']:
            print(f"   {p['id']:9s} {p['tokens']:4d} tok {p['tok_s']:6.3f} tok/s TTFT {p['ttft_s']:6.2f} "
                  f"wait {p['wait_ms_per_tok']:6.1f} gpu {p['moe_gpu_ms_per_tok']:5.1f} attn {p['decprof_attn_ms']:5.1f} "
                  f"p90 {p['lat_p90_ms']:6.1f} host-hit {p['host_hit_pct']:5.1f}%")


def paired(runs, pairs):
    ds, per = [], {}
    for a, b in pairs:
        if a in runs and b in runs:
            pa = {p['id']: p['tok_s'] for p in runs[a]['prompts']}
            for p in runs[b]['prompts']:
                if p['id'] != 'w0' and p['id'] in pa:
                    d = 100 * (p['tok_s'] / pa[p['id']] - 1)
                    ds.append(d); per.setdefault(p['id'], []).append(round(d, 2))
    m, ci = w.mean_ci(ds) if ds else (float('nan'), float('nan'))
    return {'mean_pct': m, 'ci95_pct': ci, 'n': len(ds), 'per_prompt_pct': per}


def same_prefix(toks, ref):
    """every prompt's ids equal across runs, and a prefix of the main ABBA's (256-token) ids"""
    out = {}
    names = list(toks)
    for pid in (toks[names[0]] if names else {}):
        seqs = [toks[n][pid] for n in names]
        ok = all(s == seqs[0] for s in seqs)
        if ref and pid in ref:
            ok = ok and ref[pid][:len(seqs[0])] == seqs[0]
        out[pid] = ok
    return out


def main():
    dump = sys.argv[1]
    main_toks = json.loads((HERE / 'tokens.json').read_text(encoding='utf-8')) if (HERE / 'tokens.json').exists() else {}
    ref = main_toks.get('w3-A1-wex1')
    out = {}
    runs, toks, shas = summarize(HERE / 'runs', PILOT_C, dump)
    if runs:
        print('== pilot C: WEX1 16 GB vs WNR2 12 GB + 4 GB compressed RAM cache')
        show(runs, shas)
        out['pilot_c'] = {'runs': runs, 'delta_c_vs_a': paired(runs, C_PAIRS), 'tokens_ok': same_prefix(toks, ref),
                          'logits_sha256': shas}
        print('delta C vs A', json.dumps(out['pilot_c']['delta_c_vs_a']))
        print('tokens (identical, prefix of main ABBA)', out['pilot_c']['tokens_ok'])
    runs, toks, shas = summarize(HERE.parent / 'f1' / 'runs', F1, dump)
    if runs:
        print('== F1 first gating grid (WEX1)')
        show(runs, shas)
        bases = [n for n in ('f1-base', 'f1-base2') if n in runs]
        grid = {}
        for n in runs:
            if n in bases:
                continue
            grid[n] = paired(runs, [(b, n) for b in bases])
        grid['base2_vs_base'] = paired(runs, [('f1-base', 'f1-base2')])
        out['f1'] = {'runs': runs, 'delta_vs_bases': grid, 'tokens_ok': same_prefix(toks, ref), 'logits_sha256': shas}
        for k, v in grid.items():
            print(f"   {k}: {v['mean_pct']:+.2f}% +- {v['ci95_pct']:.2f} (n={v['n']}) {v['per_prompt_pct']}")
        print('tokens (identical, prefix of main ABBA)', out['f1']['tokens_ok'])
    (HERE / 'pilots.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
