"""A/B same-drive WEX1 vs WNR2 (run_ab.sh): per-prompt numbers, per-arm mean and 95% CI of decode
tok/s, paired delta of WNR2 vs WEX1 (ABBA pairs A1-B1, B2-A2), and greedy token identity from the
logits dumps (argmax per decode step).

  python -I analyze_ab.py DUMPDIR   -> summary.json + tokens.json next to this script, text on stdout
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

VOCAB = 129280
ROUTED = 40 * 6   # routed expert uses per decode token (40 layers x top-6)
HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
ABBA = ['ab-A1-wex1', 'ab-B1-wnr2', 'ab-B2-wnr2', 'ab-A2-wex1']
SMOKE = ['ab-smoke-A-wex1', 'ab-smoke-B-wnr2']
PAIRS = [('ab-A1-wex1', 'ab-B1-wnr2'), ('ab-A2-wex1', 'ab-B2-wnr2')]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 15: 2.131}


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float('nan')
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n - 1, 2.0) * sd / math.sqrt(n)


def load(name):
    rows = [json.loads(l) for l in (RUNS / f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if l]
    recs = [r for r in rows if 'id' in r]
    ready = next(r['secs'] for r in rows if r.get('event') == 'ready')
    meta = json.loads((RUNS / f'{name}.meta.json').read_text(encoding='utf-8'))
    log = (RUNS / f'{name}.stderr.log').read_text(encoding='utf-8', errors='replace').splitlines()
    return recs, ready, meta, log


def parse_done(rec):   # DONE id STAT tokens tok/s hit% rss prompt_tokens length_limited
    f = rec['done'].split()
    return {'tokens': int(f[3]), 'tok_s': float(f[4]), 'hit_pct': float(f[5]), 'prompt_tokens': int(f[7])}


def parse_prof(rec):   # PROF wall prompt completion disk_s wait_s matmul_s attn_s head_s forwards
    f = rec['prof'].split()
    keys = ['wall_s', 'prompt', 'completion', 'disk_s', 'wait_s', 'matmul_s', 'attn_s', 'head_s', 'forwards']
    return {k: float(v) for k, v in zip(keys, f[1:])}


def stamped(log, key):
    out = []
    for line in log:
        m = re.match(r'\[\s*([\d.]+)\] (.*)', line)
        if m and m.group(2).startswith(key):
            kv = dict(re.findall(r'(\w+)=([\d.]+)', m.group(2)))
            out.append((float(m.group(1)), {k: float(v) for k, v in kv.items()}))
    return out


def last_kv(log, key):
    for line in reversed(log):
        if key in line:
            return {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line.split(key, 1)[1])}
    return {}


def run_summary(name):
    recs, ready, meta, log = load(name)
    moe, dec = stamped(log, 'moeprof '), stamped(log, 'decprof ')
    t, prompts = ready, []
    for r in recs:
        t0, t1 = t, t + r['wall']
        t = t1
        d, p = parse_done(r), parse_prof(r)
        mo = [kv for ts, kv in moe if t0 <= ts < t1]
        de = [kv for ts, kv in dec if t0 <= ts < t1]
        avg = lambda rows, k: sum(x[k] for x in rows) / len(rows) if rows else float('nan')
        prompts.append({
            'id': r['id'], 'tokens': d['tokens'], 'tok_s': d['tok_s'], 'ttft_s': r['ttft'],
            'host_hit_pct': d['hit_pct'], 'prompt_tokens': d['prompt_tokens'],
            'decode_tokens_profiled': len(mo),
            'nonvram_per_tok': ROUTED - avg(mo, 'vram_hits'),   # routed uses served from RAM or disk
            'wait_ms_per_tok': avg(mo, 'wait'),                  # decode MoE wait for expert weights
            'moe_gpu_ms_per_tok': avg(mo, 'gpu'), 'attach_ms_per_tok': avg(mo, 'attach'),
            'decprof_moe_ms': avg(de, 'moe'), 'decprof_attn_ms': avg(de, 'attn'), 'decprof_hc_ms': avg(de, 'hc'),
            'prof_wait_ms_per_fwd': 1000 * p['wait_s'] / p['forwards'] if p['forwards'] else float('nan'),
            'prof_disk_s': p['disk_s'], 'prof': r['prof'], 'tiers': r['tiers'],
        })
    direct = last_kv(log, 'v4_direct ')
    gd = last_kv(log, 'v4_wex_gpu_direct ')
    fwd = sum(parse_prof(r)['forwards'] for r in recs)
    reads = direct.get('reads', 0) + gd.get('reads', 0)
    gbytes = (direct.get('payload_bytes', 0) + gd.get('payload_bytes', 0)) / 1e9
    return {'name': name, 'ready_s': ready, 'avail_ram_gb': meta['avail_ram_gb'], 'ram_gb': meta['ram_gb'],
            'start': meta['start'], 'prompts': prompts, 'host_reads': direct.get('reads', 0),
            'direct_reads': gd.get('reads', 0), 'device_decodes': gd.get('device_decodes', 0),
            'disk_reads_per_fwd': reads / fwd if fwd else float('nan'),
            'disk_gb': gbytes, 'disk_gb_per_fwd': gbytes / fwd if fwd else float('nan'), 'forwards': fwd}


def token_ids(dump, name, recs):
    path = Path(dump) / f'{name}.logits'
    rows = np.memmap(path, dtype=np.float32, mode='r').reshape(-1, VOCAB)
    out, at = {}, 0
    for r in recs:
        n = parse_done(r)['tokens']
        n += n < r.get('max_tokens', n)   # stopped on EOS: its step dumped one more row
        out[r['id']] = [int(x) for x in rows[at:at + n].argmax(1)]
        at += n
    if at != len(rows):
        print(f'  note: {name} has {len(rows)} logits rows, the DONE lines count {at}')
    return out


def main():
    dump = sys.argv[1]
    names = [n for n in SMOKE + ABBA if (RUNS / f'{n}.jsonl').exists()]
    runs = {n: run_summary(n) for n in names}
    toks = {n: token_ids(dump, n, load(n)[0]) for n in names}
    (HERE / 'tokens.json').write_text(json.dumps(toks), encoding='utf-8')
    ident = {}
    for label, group in (('smoke', SMOKE), ('abba', ABBA)):
        got = [n for n in group if n in toks]
        for pid in (toks[got[0]] if got else {}):
            seqs = [toks[n].get(pid) for n in got]
            ident[f'{label}:{pid}'] = all(s == seqs[0] for s in seqs)
    arms = {'WEX1': [n for n in ABBA if 'wex1' in n and n in runs], 'WNR2': [n for n in ABBA if 'wnr2' in n and n in runs]}
    stats = {}
    for arm, ns in arms.items():
        xs = [p['tok_s'] for n in ns for p in runs[n]['prompts'] if p['id'] != 'w0']
        if xs:
            m, ci = mean_ci(xs)
            stats[arm] = {'n': len(xs), 'mean_tok_s': m, 'ci95': ci}
    diffs = []
    for a, b in PAIRS:
        if a in runs and b in runs:
            pa = {p['id']: p['tok_s'] for p in runs[a]['prompts']}
            for p in runs[b]['prompts']:
                if p['id'] != 'w0' and p['id'] in pa:
                    diffs.append(100 * (p['tok_s'] / pa[p['id']] - 1))
    delta = dict(zip(['mean_pct', 'ci95_pct'], mean_ci(diffs))) if diffs else {}
    delta['n_pairs'] = len(diffs)
    delta['paired_pct'] = diffs
    if 'WEX1' in stats and 'WNR2' in stats:
        delta['arm_means_pct'] = 100 * (stats['WNR2']['mean_tok_s'] / stats['WEX1']['mean_tok_s'] - 1)
    same = all(ident.values()) if ident else False
    lo = delta.get('mean_pct', 0) - delta.get('ci95_pct', 0)
    hi = delta.get('mean_pct', 0) + delta.get('ci95_pct', 0)
    speed = same and delta.get('mean_pct', 0) >= 3 and (lo > 0)
    out = {'runs': runs, 'arm_stats': stats, 'delta_wnr2_vs_wex1': delta, 'token_identity': ident,
           'tokens_identical_all': same, 'verdict': 'speed format' if speed else 'capacity format',
           'ci_excludes_zero': lo > 0 or hi < 0}
    (HERE / 'summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for n, r in runs.items():
        print(f"{n}: READY {r['ready_s']} s, avail RAM {r['avail_ram_gb']} GB, RAM_GB {r['ram_gb']}, "
              f"disk reads/fwd {r['disk_reads_per_fwd']:.1f} ({r['host_reads']:.0f} host + {r['direct_reads']:.0f} direct), "
              f"{r['disk_gb_per_fwd']:.2f} GB/fwd")
        for p in r['prompts']:
            print(f"   {p['id']:9s} {p['tokens']:4d} tok {p['tok_s']:6.3f} tok/s TTFT {p['ttft_s']:7.2f} "
                  f"nonVRAM/tok {p['nonvram_per_tok']:5.1f} wait {p['wait_ms_per_tok']:6.1f} ms/tok "
                  f"gpu {p['moe_gpu_ms_per_tok']:5.1f} host-hit {p['host_hit_pct']:5.1f}%")
    print('arm stats', json.dumps(stats))
    print('delta', json.dumps({k: v for k, v in delta.items() if k != 'paired_pct'}),
          [round(x, 2) for x in delta.get('paired_pct', [])])
    print('token identity', ident)
    print('verdict', out['verdict'])


if __name__ == '__main__':
    main()
