"""W3 A/B WEX1 vs WNR2 (run_w3.sh): per-run and per-prompt numbers, paired delta of WNR2 vs WEX1 over the
ABBA pairs (A1-B1, A2-B2, A3-B3, A4-B4), TTFT deltas, token-latency p50/p90, where the time went, the
wexprof/v4_direct/GPU-direct counters, greedy token identity and logits sha256 from the dumps, and the
F1 offline pick (prefetch precision by confidence bucket, V4_PREDICT_STATS).

  python -I analyze_w3.py DUMPDIR   -> summary.json + tokens.json next to this script, text on stdout
"""
import hashlib
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
SMOKE = ['w3-smoke-A-wex1', 'w3-smoke-B-wnr2']
ABBA = ['w3-A1-wex1', 'w3-B1-wnr2', 'w3-B2-wnr2', 'w3-A2-wex1', 'w3-A3-wex1', 'w3-B3-wnr2', 'w3-B4-wnr2', 'w3-A4-wex1']
PAIRS = [(f'w3-A{i}-wex1', f'w3-B{i}-wnr2') for i in range(1, 5)]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 23: 2.069,
        31: 2.040}


def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)]


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float('nan')
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, t975(n - 1) * sd / math.sqrt(n)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float('nan')


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


def last_line(log, key):
    for line in reversed(log):
        if key in line:
            return line.split('] ', 1)[-1]
    return ''


def kv(text):
    return {k: float(v) for k, v in re.findall(r'([\w-]+)=([\d.]+)', text)}


def wexprof(log):   # exit totals: demand / prefetch, batched part and per-record part
    out = {}
    for line in log:
        m = re.search(r'wexprof (demand|prefetch) (.*)', line)
        if m:
            batched, _, record = m.group(2).partition('|')
            out[m.group(1)] = {'batched': kv(batched), 'per_record': kv(record)}
    return out


def run_summary(name):
    recs, ready, meta, log = load(name)
    moe, dec = stamped(log, 'moeprof '), stamped(log, 'decprof ')
    t, prompts = ready, []
    for r in recs:
        t0, t1 = t, t + r['wall']
        t = t1
        d, p = parse_done(r), parse_prof(r)
        mo = [x for ts, x in moe if t0 <= ts < t1]
        de = [(ts, x) for ts, x in dec if t0 <= ts < t1]
        lat = [1000 * (b[0] - a[0]) for a, b in zip(de, de[1:])]   # decode token latency, ms
        avg = lambda rows, k: sum(x[k] for x in rows) / len(rows) if rows else float('nan')
        prompts.append({
            'id': r['id'], 'tokens': d['tokens'], 'tok_s': d['tok_s'], 'ttft_s': r['ttft'],
            'host_hit_pct': d['hit_pct'], 'prompt_tokens': d['prompt_tokens'],
            'decode_tokens_profiled': len(mo),
            'nonvram_per_tok': ROUTED - avg(mo, 'vram_hits'),
            'wait_ms_per_tok': avg(mo, 'wait'), 'moe_gpu_ms_per_tok': avg(mo, 'gpu'),
            'attach_ms_per_tok': avg(mo, 'attach'),
            'decprof_moe_ms': avg([x for _, x in de], 'moe'), 'decprof_attn_ms': avg([x for _, x in de], 'attn'),
            'decprof_hc_ms': avg([x for _, x in de], 'hc'),
            'lat_p50_ms': pct(lat, 50), 'lat_p90_ms': pct(lat, 90),
            'head_ms_per_tok': 1000 * p['head_s'] / d['tokens'] if d['tokens'] else float('nan'),
            'prof': r['prof'], 'tiers': r['tiers'],
        })
    direct = kv(last_line(log, 'v4_direct reads'))
    gd = kv(last_line(log, 'v4_wex_gpu_direct reads'))
    fwd = sum(parse_prof(r)['forwards'] for r in recs)
    reads = direct.get('reads', 0) + gd.get('reads', 0)
    gbytes = (direct.get('payload_bytes', 0) + gd.get('payload_bytes', 0)) / 1e9
    return {'name': name, 'ready_s': ready, 'avail_ram_gb': meta['avail_ram_gb'], 'ram_gb': meta['ram_gb'],
            'start': meta['start'], 'prompts': prompts, 'host_reads': direct.get('reads', 0),
            'direct_reads': gd.get('reads', 0), 'device_decodes': gd.get('device_decodes', 0),
            'gd_failures': gd.get('failures', 0), 'ram_skips': gd.get('ram_skips', 0),
            'disk_reads_per_fwd': reads / fwd if fwd else float('nan'),
            'disk_gb': gbytes, 'disk_gb_per_fwd': gbytes / fwd if fwd else float('nan'), 'forwards': fwd,
            'wexprof': wexprof(log), 'ram_tiers': last_line(log, 'ram_tiers '),
            'mirror_epochs': last_line(log, 'v4_gpu_mirror_epochs'),
            'predict': last_line(log, 'v4_next_predict recall'),
            'predict_conf': last_line(log, 'v4_next_predict_conf')}


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
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return out, h.hexdigest()


def conf_precision(line):   # "used/guessed per 0.05 d1 a/b ... d2 a/b ..." -> per depth [(used, guessed)]
    out = {}
    for depth, body in re.findall(r'(d[12]) ((?:\d+/\d+\s*)+)', line):
        out[depth] = [tuple(map(int, x.split('/'))) for x in body.split()]
    return out


def main():
    dump = sys.argv[1]
    # finished runs only: the engine prints the wexprof totals at exit
    names = [n for n in SMOKE + ABBA if (RUNS / f'{n}.stderr.log').exists() and
             'wexprof prefetch' in (RUNS / f'{n}.stderr.log').read_text(encoding='utf-8', errors='replace')]
    runs = {n: run_summary(n) for n in names}
    toks, shas = {}, {}
    for n in names:
        toks[n], shas[n] = token_ids(dump, n, load(n)[0])
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
            stats[arm] = {'n': len(xs), 'mean_tok_s': m, 'ci95': ci,
                          'lat_p50_ms': float(np.median([p['lat_p50_ms'] for n in ns for p in runs[n]['prompts'] if p['id'] != 'w0'])),
                          'lat_p90_ms': float(np.median([p['lat_p90_ms'] for n in ns for p in runs[n]['prompts'] if p['id'] != 'w0']))}
    diffs, pair_means, ttft = [], [], {}
    per_prompt = {}
    for a, b in PAIRS:
        if a in runs and b in runs:
            pa = {p['id']: p for p in runs[a]['prompts']}
            ds = []
            for p in runs[b]['prompts']:
                if p['id'] != 'w0' and p['id'] in pa:
                    d = 100 * (p['tok_s'] / pa[p['id']]['tok_s'] - 1)
                    ds.append(d)
                    per_prompt.setdefault(p['id'], []).append(round(d, 2))
                    ttft.setdefault(p['id'], []).append(100 * (p['ttft_s'] / pa[p['id']]['ttft_s'] - 1))
            diffs += ds
            pair_means.append(sum(ds) / len(ds))
    delta = dict(zip(['mean_pct', 'ci95_pct'], mean_ci(diffs))) if diffs else {}
    delta['n'] = len(diffs)
    delta['per_prompt_pct'] = per_prompt
    if pair_means:
        delta['pairs_mean_pct'], delta['pairs_ci95_pct'] = mean_ci(pair_means)
        delta['pair_means_pct'] = pair_means
    if 'WEX1' in stats and 'WNR2' in stats:
        delta['arm_means_pct'] = 100 * (stats['WNR2']['mean_tok_s'] / stats['WEX1']['mean_tok_s'] - 1)
    ttft_delta = {k: dict(zip(['mean_pct', 'ci95_pct'], mean_ci(v))) for k, v in ttft.items()}
    same = all(ident.values()) if ident else False
    abba_shas = {shas[n] for n in ABBA if n in shas}
    m = delta.get('pairs_mean_pct', delta.get('mean_pct', 0))
    ci = delta.get('pairs_ci95_pct', delta.get('ci95_pct', 0))
    have = delta['n'] > 0
    old_rule = have and same and m >= 3 and m - ci > 0
    new_rule = have and same and m >= 0   # user 2026-11-08: keep WNR2 when tokens identical and it is not slower
    conf = {n: conf_precision(runs[n]['predict_conf']) for n in names if runs[n]['predict_conf']}
    out = {'runs': runs, 'arm_stats': stats, 'delta_wnr2_vs_wex1': delta, 'ttft_delta_pct': ttft_delta,
           'token_identity': ident, 'tokens_identical_all': same, 'logits_sha256': shas,
           'abba_logits_bit_identical': len(abba_shas) == 1 if abba_shas else None,
           'verdict_user_rule': 'WNR2 kept as the V4.1 container' if new_rule else 'WNR2 not adopted',
           'verdict_plan_3_2': 'speed format' if old_rule else 'capacity format',
           'prefetch_conf_precision': conf}
    (HERE / 'summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for n, r in runs.items():
        print(f"{n}: READY {r['ready_s']} s, avail RAM {r['avail_ram_gb']} GB, RAM_GB {r['ram_gb']}, logits {shas[n][:16]}, "
              f"disk reads/fwd {r['disk_reads_per_fwd']:.1f} ({r['host_reads']:.0f} host + {r['direct_reads']:.0f} direct), "
              f"{r['disk_gb_per_fwd']:.2f} GB/fwd, {r['disk_gb']:.0f} GB")
        for p in r['prompts']:
            print(f"   {p['id']:9s} {p['tokens']:4d} tok {p['tok_s']:6.3f} tok/s TTFT {p['ttft_s']:7.2f} "
                  f"nonVRAM {p['nonvram_per_tok']:5.1f} wait {p['wait_ms_per_tok']:6.1f} gpu {p['moe_gpu_ms_per_tok']:5.1f} "
                  f"attn {p['decprof_attn_ms']:5.1f} hc {p['decprof_hc_ms']:4.1f} head {p['head_ms_per_tok']:5.1f} "
                  f"p50 {p['lat_p50_ms']:6.1f} p90 {p['lat_p90_ms']:6.1f} host-hit {p['host_hit_pct']:5.1f}%")
        for k, v in r['wexprof'].items():
            print(f"   wexprof {k}: {v}")
    print('arm stats', json.dumps(stats))
    print('delta', json.dumps({k: v for k, v in delta.items() if k != 'per_prompt_pct'}))
    print('per prompt', json.dumps(per_prompt))
    print('ttft delta', json.dumps(ttft_delta))
    print('token identity', ident, 'ABBA logits bit-identical', len(abba_shas) == 1)
    print('verdict (user rule):', out['verdict_user_rule'], '| PLAN 3.2 rule:', out['verdict_plan_3_2'])


if __name__ == '__main__':
    main()
