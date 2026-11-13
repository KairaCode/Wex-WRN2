"""Short ABBA analysis (short_abba.sh; also reads the W3 ab/runs and ab-samedrive runs dirs).

Per run and prompt: decode tok/s, TTFT and the decode MoE wait from the moeprof lines (ms per token the MoE
pipeline waited for expert weights: NVMe/RAM -> VRAM). Paired delta of B vs A per (pair i, prompt): A<i> with
B<i>, same prompt. The 95% interval treats the prompts as fixed effects: residual = delta - that prompt's mean
over the pairs, df = n_pairs*n_prompts - n_prompts. The deltas of one prompt are not independent draws (the
prompt sets their level, SD ~3.8 points between prompts in W3, ~10x the pair noise), so this interval is the
pair-to-pair noise FOR THIS PROMPT SET; how much the result depends on the set is printed apart (validity).
The pair-means interval (one mean per pair, df = n_pairs - 1, W3 results.md's) is printed as a cross-check: it
also absorbs a run-level effect, which the fixed-prompt residuals assume away (W3 at 64 tokens: +-0.70 against
+-0.31), but with 2 pairs its df=1.
Greedy token identity: argmax of the V4_LOGITS_DUMP rows when --dump is given (plus each dump's sha256; with
--json the ids and shas also go to tokens.json beside it), else RUNDIR/tokens.json when it exists (short_abba.sh
deletes the dumps after its analysis), else the reply text.

Guards (a guarded set never prints KEEP B or NOT ADOPTED):
  INVALID set   RUNDIR/INVALID exists (short_abba.sh: quiet-machine, RAM or VRAM hit, a start gate that timed
                out, or a failed run check), an ABBA run left out (no READY, ERROR, incomplete), RAM budget
                (ram_tiers available, target_slots) differs between runs, the VRAM expert tier (v4_tier_prewarm
                vram_experts) differs by more than 5 % between runs or is missing in some, or no A/B pair;
  stall         a prompt is invalid in a run when one of its decode tokens has decprof hc+attn+moe
                > max(2 s, 3x the run's median token) or moeprof gpu > 1 s; it is dropped from every pair
                (the pairs stay balanced) and the set is INCONCLUSIVE (a pair lost a prompt);
  spread        an arm's run means (mean tok/s over the used prompts, one per run) differ by > 2%
                ((max-min)/mean): INCONCLUSIVE. A single prompt's same-arm spread > 3.5% is only a printed
                warning: at 64 tokens it trips 2% in 3 of W3's 6 two-pair subsets (largest 2.93%, A code1)
                while their run means stay within 1.4% (validate/cut_tokens.py --pairs).
Verdict (user rule 2026-11-08: keep B only with identical tokens and B not slower):
  KEEP B        mean delta >= 0, every pair mean >= 0 and token ids (logits argmax) identical in every run;
  NOT ADOPTED   token ids differ, or the whole CI is below 0 and every pair mean is below 0;
  INCONCLUSIVE  anything else (CI spans 0 with mean < 0, a pair mean on the other side of 0, reply-text
                basis only, a guard tripped).

Read path per run: the PROF edisk column (expert_load wall on the reading threads) summed over the turns,
divided by the engine's exit read counter (v4_direct reads + v4_wex_gpu_direct reads) = ms per read, and by the
summed turn wall = mean outstanding reads (Little's law). Both include prefill. Per-drive read share from the
engine's v4_mirror line when it logs one, and from the OS read-byte counters short_abba.sh saves per run
(<name>.disk.txt: "start C:=bytes U:=bytes ..." and "end ...").

A run is RUNDIR/<name>.jsonl + <name>.stderr.log (+ <name>.meta.json) from serve_bench.py; a name holding
A<i> or B<i> (A1, w3-B2-wnr2, ...) is an ABBA run, smoke runs are ignored.

  python -I analyze_short.py RUNDIR [--dump DIR] [--ids a,b] [--warmup w0] [--vocab N] [--json OUT]
"""
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]   # two-sided 95% t quantile, df 1..30; 2.0 above
NAME = re.compile(r'(?<![A-Za-z0-9])([AB])(\d+)(?![0-9])')
DECPROF = re.compile(r'\[\s*([\d.]+)\] decprof position=\d+ layers=\d+ hc=([\d.]+) attn=([\d.]+) moe=([\d.]+)')
MOEPROF = re.compile(r'\[\s*([\d.]+)\] moeprof wait=([\d.]+)(?: attach=[\d.]+)?(?: gpu=([\d.]+))?')
MIRROR = re.compile(r'v4_mirror(?: drives=\d+)?((?: (?:primary|mirror)=[\d.]+GiB/\d+)+)')
NAN = float('nan')
STALL_MS, STALL_X, GPU_MS, SPREAD_PCT = 2000.0, 3.0, 1000.0, 2.0
SPREAD_WARN_PCT = 3.5   # per prompt, warning only: W3 at 64 tokens, largest same-arm spread 2.93% (4 runs)
VRAM_TOL_PCT = 5.0      # vram_experts spread between runs: W3 934-966 (3.3%), 05:40 963-965; 1 GB less VRAM ~ 60 experts


def t975(df):
    return T975[df - 1] if 1 <= df <= len(T975) else 2.0 if df > len(T975) else NAN


def fixed_ci(cells):
    """cells {prompt: [delta of each pair]}, the same pairs for every prompt. Mean and 95% half-width with
    the prompts as fixed effects (pair-to-pair residuals), df = n_pairs*n_prompts - n_prompts."""
    xs = [x for v in cells.values() for x in v]
    n, df = len(xs), len(xs) - len(cells)
    if not n:
        return NAN, NAN, 0
    m = sum(xs) / n
    if df < 1:
        return m, NAN, df
    ss = sum((x - sum(v) / len(v)) ** 2 for v in cells.values() for x in v)
    return m, t975(df) * math.sqrt(ss / df / n), df


def load(rundir, name):
    rows = [json.loads(l) for l in (rundir / f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    meta = rundir / f'{name}.meta.json'
    disk = rundir / f'{name}.disk.txt'
    log = (rundir / f'{name}.stderr.log').read_text(encoding='utf-8', errors='replace').splitlines()
    return {'name': name, 'recs': [r for r in rows if 'id' in r],
            'ready': next((r['secs'] for r in rows if r.get('event') == 'ready'), None),
            'meta': json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}, 'log': log,
            'disk': disk.read_text(encoding='utf-8').splitlines() if disk.exists() else []}


def disk_share(lines):
    """OS read bytes per drive over the run, from short_abba.sh's <name>.disk.txt; drives under 1 GiB left out."""
    at = {l.split()[0]: dict(kv.split('=') for kv in l.split()[1:]) for l in lines if l.split()}
    if 'start' not in at or 'end' not in at:
        return {}
    d = {k: (int(at['end'][k]) - int(v)) / 2 ** 30 for k, v in at['start'].items() if k in at['end']}
    return {k: round(v, 1) for k, v in d.items() if v >= 1}


def summarize(run):
    dec = [(float(m[1]), float(m[2]) + float(m[3]) + float(m[4])) for m in map(DECPROF.match, run['log']) if m]
    moe = [(float(m[1]), float(m[2]), float(m[3] or 0)) for m in map(MOEPROF.match, run['log']) if m]
    med = statistics.median(x for _, x in dec) if dec else NAN
    limit = max(STALL_MS, STALL_X * med) if dec else NAN
    t, prompts, edisk, pwall = run['ready'], {}, 0.0, 0.0
    for r in run['recs']:   # serve_bench runs the prompts back to back from READY: stderr stamps -> prompt
        span = lambda ts: t <= ts < t + r['wall']
        w = [x for ts, x, _ in moe if span(ts)]
        slow = [f'{x / 1000:.1f} s token at {ts:.0f} s' for ts, x in dec if span(ts) and x > limit]
        slow += [f'moeprof gpu {g / 1000:.1f} s (wait {x / 1000:.1f} s) at {ts:.0f} s' for ts, x, g in moe
                 if span(ts) and g > GPU_MS]
        t += r['wall']
        f = r['done'].split()   # DONE id STAT tokens tok/s hit% rss prompt_tokens length_limited
        pf = r.get('prof', '').split()   # PROF wall prompt completion edisk ewait emm attn head n_fw
        if len(pf) > 4:
            pwall, edisk = pwall + float(pf[1]), edisk + float(pf[4])
        prompts[r['id']] = {'tokens': int(f[3]), 'tok_s': float(f[4]), 'ttft_s': r['ttft'], 'wall_s': r['wall'],
                            'wait_ms_per_tok': sum(w) / len(w) if w else NAN, 'text': r.get('text', ''),
                            'stall': slow}
    text = '\n'.join(run['log'])
    cont = re.search(r'Container v(\d+) active: (.*?) \(', text)
    ram = re.search(r'ram_tiers available=([\d.]+)GiB', text)
    slots = re.search(r'ram_tiers available=.*? target_slots=(\d+)', text)
    vram = re.search(r'v4_tier_prewarm vram_experts=(\d+)', text)
    host = re.findall(r'\] v4_direct reads=(\d+)', text)
    direct = re.findall(r'v4_wex_gpu_direct reads=(\d+)', text)
    reads = (int(host[-1]) if host else 0) + (int(direct[-1]) if direct else 0)
    mir = MIRROR.findall(text)
    split = re.search(r'\[v4_wex\] read split over (\d+) drives:([ \d%]+)', text)
    m = NAME.search(run['name'])
    return {'name': run['name'], 'arm': m[1], 'pair': int(m[2]), 'ready_s': run['ready'], 'wall_s': t,
            'container': f'WEX v{cont[1]} {cont[2]}' if cont else 'no container (safetensors)',
            'mirrors': re.findall(r'Matching mirror container active: (.*)', text),
            'read_split': f'{split[1]} drives:{split[2]}'.strip() if split else None,
            'engine_drive_reads': [(k, float(g), int(n)) for k, g, n in re.findall(r'(primary|mirror)=([\d.]+)GiB/(\d+)', mir[-1])] if mir else [],
            'os_drive_read_gib': disk_share(run['disk']),
            'ram_gb': run['meta'].get('ram_gb'), 'ram_tiers_gib': float(ram[1]) if ram else None,
            'target_slots': int(slots[1]) if slots else None, 'vram_experts': int(vram[1]) if vram else None,
            'vram_used_mib_at_start': run['meta'].get('vram_used_mib'),
            'avail_ram_gb': run['meta'].get('avail_ram_gb'), 'start': run['meta'].get('start'),
            'token_median_ms': med, 'stall_limit_ms': limit, 'edisk_s': round(edisk, 1), 'reads': reads,
            'ms_per_read': 1000 * edisk / reads if reads else NAN, 'outstanding_reads': edisk / pwall if pwall else NAN,
            'prompts': prompts}


def model_vocab(meta, default):
    cfg = Path(meta.get('model') or '-') / 'config.json'
    if not cfg.is_file():
        return default
    c = json.loads(cfg.read_text(encoding='utf-8'))
    return c.get('vocab_size') or c.get('text_config', {}).get('vocab_size') or default


def dump_tokens(path, run, vocab):
    flat = np.memmap(path, dtype=np.float32, mode='r')
    if flat.size % vocab:
        raise SystemExit(f'{path}: {flat.size} floats is not a whole number of vocab-{vocab} rows (--vocab?)')
    rows, out, at = flat.reshape(-1, vocab), {}, 0
    for r in run['recs']:
        n = int(r['done'].split()[3])
        n += n < r['max_tokens']   # stopped on EOS: its step dumped one more row
        out[r['id']] = rows[at:at + n].argmax(1).tolist()
        at += n
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return out, h.hexdigest(), '' if at == len(rows) else f'{len(rows)} logits rows, the DONE lines count {at}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('rundir', type=Path)
    ap.add_argument('--dump', type=Path, help='V4_LOGITS_DUMP dir (<name>.logits): token ids + sha256')
    ap.add_argument('--ids', help='only these prompts (default: every prompt but the warm-up)')
    ap.add_argument('--warmup', default='w0', help='warm-up prompt id, left out of the timing (default w0)')
    ap.add_argument('--vocab', type=int, default=129280, help="logits row length when meta's model has no config")
    ap.add_argument('--json', type=Path, help='write the summary here')
    a = ap.parse_args()

    invalid, guards, notes = [], [], []
    marker = a.rundir / 'INVALID'
    if marker.exists():
        invalid.append('short_abba.sh marked the set INVALID: ' + ' | '.join(marker.read_text(encoding='utf-8').split('\n')).strip(' |'))
    names = sorted({p.name[:-6] for p in a.rundir.glob('*.jsonl')} & {p.name[:-11] for p in a.rundir.glob('*.stderr.log')})
    raw = [load(a.rundir, n) for n in names if NAME.search(n)]
    longest = max((len(r['recs']) for r in raw), default=0)
    runs = []
    for r in raw:
        if r['ready'] is None or len(r['recs']) < longest or any('done' not in x for x in r['recs']):
            invalid.append(f"{r['name']} left out: {'no READY' if r['ready'] is None else 'incomplete or ERROR'}")
        else:
            runs.append(r)
    if not runs:
        raise SystemExit(f'no complete A<i>/B<i> runs in {a.rundir}')
    order = [x['id'] for x in runs[0]['recs']]
    ids = a.ids.split(',') if a.ids else [i for i in order if i != a.warmup]
    check_ids = [i for i in order if i == a.warmup or i in ids]
    S = {r['name']: summarize(r) for r in runs}

    # token identity over every ABBA run, warm-up included
    toks, shas = {}, {}
    for r in runs:
        path = a.dump / f"{r['name']}.logits" if a.dump else None
        if path and path.is_file() and path.stat().st_size:
            toks[r['name']], shas[r['name']], note = dump_tokens(path, r, model_vocab(r['meta'], a.vocab))
            if note:
                notes.append(f"{r['name']}: {note}")
        elif path:
            notes.append(f"{r['name']}: no dump at {path}, comparing reply text")
    saved = a.rundir / 'tokens.json'   # short_abba.sh deletes the dumps after the first analysis
    if not a.dump and saved.exists():
        t = json.loads(saved.read_text(encoding='utf-8'))
        if all(r['name'] in t['tokens'] for r in runs):
            toks = {r['name']: t['tokens'][r['name']] for r in runs}
            shas = {r['name']: t['logits_sha256'][r['name']] for r in runs}
            notes.append(f'token ids from {saved} (argmax of the deleted dumps)')
    elif a.json and len(toks) == len(runs):
        a.json.with_name('tokens.json').write_text(json.dumps({'logits_sha256': shas, 'tokens': toks}), encoding='utf-8')
    basis ='token ids (logits argmax)' if len(toks) == len(runs) else 'reply text'
    seqs = toks if len(toks) == len(runs) else {n: {i: p['text'] for i, p in s['prompts'].items()} for n, s in S.items()}
    first = runs[0]['name']
    ident = {i: [n for n in seqs if seqs[n].get(i) != seqs[first].get(i)] for i in check_ids}
    same = not any(ident.values())

    by = {(s['arm'], s['pair']): s for s in S.values()}
    pairs = sorted(i for (k, i) in by if k == 'A' and ('B', i) in by)
    if not pairs:
        invalid.append('no A<i>/B<i> pair')
    ram_levels = {s['ram_tiers_gib'] for s in S.values()}
    if len(ram_levels) > 1:
        invalid.append(f'RAM budget differs between runs (ram_tiers available={sorted(ram_levels, key=str)}): '
                       'not an equal-RAM A/B')
    slot_levels = {s['target_slots'] for s in S.values()}
    if len(slot_levels) > 1:
        invalid.append(f'RAM expert slots differ between runs (target_slots={sorted(slot_levels, key=str)})')
    vx = {s['name']: s['vram_experts'] for s in S.values()}
    have = [v for v in vx.values() if v is not None]
    if have and (len(have) < len(vx) or max(have) - min(have) > VRAM_TOL_PCT / 100 * max(have)):
        invalid.append(f'VRAM expert tier differs between runs (v4_tier_prewarm vram_experts {vx}, tolerance '
                       f'{VRAM_TOL_PCT:.0f}%): not an equal-VRAM A/B')

    # stall guard: a prompt stalled in any paired run leaves every pair (balanced cells), and the set is guarded
    paired = [by[k, i] for i in pairs for k in 'AB']
    stalled = {pid: [f"{s['name']}: {x}" for s in paired for x in s['prompts'].get(pid, {}).get('stall', [])[:2]]
               for pid in ids}
    stalled = {pid: v for pid, v in stalled.items() if v}
    lost = [pid for pid in ids if pid not in stalled and any(pid not in s['prompts'] for s in paired)]
    used = [pid for pid in ids if pid not in stalled and pid not in lost] if pairs else []
    for pid, v in stalled.items():
        guards.append(f'stall in {pid} (limit max({STALL_MS / 1000:.0f} s, {STALL_X:.0f}x run median) per token, '
                      f'or moeprof gpu > {GPU_MS / 1000:.0f} s): {"; ".join(v)}; {pid} dropped from every pair')
    for pid in lost:
        guards.append(f'{pid} missing from a paired run: dropped from every pair')

    # spread guard on each arm's run means over the used prompts; a single prompt's spread is only a warning
    sp = lambda v: round(100 * (max(v) - min(v)) / (sum(v) / len(v)), 2)
    spread, run_spread = {}, {}
    for k in 'AB':
        if len(pairs) > 1 and used:
            run_spread[k] = sp([sum(by[k, i]['prompts'][pid]['tok_s'] for pid in used) / len(used) for i in pairs])
        for pid in used:
            if len(pairs) > 1:
                spread[f'{k} {pid}'] = sp([by[k, i]['prompts'][pid]['tok_s'] for i in pairs])
    wide = {k: v for k, v in run_spread.items() if v > SPREAD_PCT}
    if wide:
        guards.append(f'same-arm spread of the run means > {SPREAD_PCT:.0f}% between runs: {wide}')
    warn = {k: v for k, v in spread.items() if v > SPREAD_WARN_PCT}
    if warn:
        notes.append(f'warning: per-prompt same-arm tok/s spread > {SPREAD_WARN_PCT}% (W3 at 64 tokens: '
                     f'at most 2.93%): {warn}')

    cells = {key: {} for key in ('tok_s_pct', 'ttft_pct', 'wait_ms')}
    for pid in used:
        pa = [by['A', i]['prompts'][pid] for i in pairs]
        pb = [by['B', i]['prompts'][pid] for i in pairs]
        cells['tok_s_pct'][pid] = [100 * (b['tok_s'] / x['tok_s'] - 1) for x, b in zip(pa, pb)]
        cells['ttft_pct'][pid] = [100 * (b['ttft_s'] / x['ttft_s'] - 1) for x, b in zip(pa, pb)]
        cells['wait_ms'][pid] = [b['wait_ms_per_tok'] - x['wait_ms_per_tok'] for x, b in zip(pa, pb)]
    delta = {k: dict(zip(('mean', 'ci95', 'df'), fixed_ci(v)), n=sum(map(len, v.values()))) for k, v in cells.items()}
    m, ci = delta['tok_s_pct']['mean'], delta['tok_s_pct']['ci95']
    per_prompt = {pid: [round(x, 2) for x in v] for pid, v in cells['tok_s_pct'].items()}
    pmeans = [sum(v) / len(v) for v in cells['tok_s_pct'].values()]
    prompt_sd = statistics.stdev(pmeans) if len(pmeans) > 1 else NAN
    # cross-check: one mean per pair over the prompts (W3 results.md's interval); it also absorbs a run-level
    # effect the fixed-prompt residuals assume away, but with 2 pairs df=1 makes it ~4x wider
    pairm = [sum(v[j] for v in cells['tok_s_pct'].values()) / len(used) for j in range(len(pairs))] if used else []
    pair_ci = t975(len(pairm) - 1) * statistics.stdev(pairm) / math.sqrt(len(pairm)) if len(pairm) > 1 else NAN

    stats = {}
    for k in 'AB':
        ss = [s for s in S.values() if s['arm'] == k]
        ps = [s['prompts'][i] for s in ss for i in used if i in s['prompts']]
        stats[k] = {'runs': [s['name'] for s in ss], 'n': len(ps),
                    **{f'{key}_mean': (sum(p[key] for p in ps) / len(ps) if ps else NAN)
                       for key in ('tok_s', 'ttft_s', 'wait_ms_per_tok')},
                    **{f'{key}_mean': (sum(s[key] for s in ss) / len(ss) if ss else NAN)
                       for key in ('edisk_s', 'reads', 'ms_per_read', 'outstanding_reads')}}

    for s in S.values():
        print(f"{s['name']}: {s['container']}, READY {s['ready_s']} s, run {s['wall_s']:.0f} s, "
              f"RAM_GB {s['ram_gb']} (budget {s['ram_tiers_gib']} GiB, {s['target_slots']} slots, OS available "
              f"{s['avail_ram_gb']}), VRAM experts {s['vram_experts']} (GPU memory used before start "
              f"{s['vram_used_mib_at_start']} MiB), logits {shas.get(s['name'], '-')[:16]}")
        print(f"   read path: edisk {s['edisk_s']:.0f} s / {s['reads']} reads = {s['ms_per_read']:.2f} ms per read, "
              f"{s['outstanding_reads']:.2f} outstanding; token median {s['token_median_ms']:.0f} ms, "
              f"stall limit {s['stall_limit_ms']:.0f} ms")
        if s['mirrors'] or s['read_split']:
            print(f"   mirrors {s['mirrors']}, engine split: {s['read_split'] or 'no read-split line (single drive)'}")
        if s['engine_drive_reads']:
            tot = sum(n for _, _, n in s['engine_drive_reads']) or 1
            print('   engine per-drive reads: ' + ', '.join(f'{k} {100 * n / tot:.0f}% ({g:.1f} GiB, {n})'
                                                         for k, g, n in s['engine_drive_reads']))
        if s['os_drive_read_gib']:
            tot = sum(s['os_drive_read_gib'].values())
            print('   OS read bytes per drive: ' + ', '.join(f'{k} {100 * v / tot:.0f}% ({v:.0f} GiB)'
                                                          for k, v in sorted(s['os_drive_read_gib'].items())))
        for i in check_ids:
            p = s['prompts'][i]
            print(f"   {i:9s} {p['tokens']:4d} tok {p['tok_s']:6.3f} tok/s TTFT {p['ttft_s']:7.2f} s "
                  f"wait {p['wait_ms_per_tok']:6.1f} ms/tok wall {p['wall_s']:6.1f} s"
                  + (f"  STALL: {p['stall'][0]}" + (f' (+{len(p["stall"]) - 1})' if len(p['stall']) > 1 else '')
                     if p['stall'] else ''))
    for k, st in stats.items():
        print(f"arm {k} ({len(st['runs'])} runs, {st['n']} run x prompt): mean {st['tok_s_mean']:.3f} tok/s, "
              f"TTFT {st['ttft_s_mean']:.2f} s, wait {st['wait_ms_per_tok_mean']:.1f} ms/tok (means only: the "
              f"prompt mix dominates their spread); read path {st['ms_per_read_mean']:.2f} ms per read, "
              f"{st['outstanding_reads_mean']:.2f} outstanding, edisk {st['edisk_s_mean']:.0f} s, "
              f"{st['reads_mean']:.0f} reads")
    df = delta['tok_s_pct']['df']
    print(f"paired B vs A over pairs {pairs} x {used} (prompts fixed, df={df}):")
    print(f"   tok/s {m:+.2f} +- {ci:.2f} % (n={delta['tok_s_pct']['n']}); per prompt {per_prompt}")
    print(f"   cross-check, pair means {[round(x, 2) for x in pairm]}: +- {pair_ci:.2f} % (df={len(pairm) - 1}; "
          f"not the verdict's interval)")
    print(f"   TTFT  {delta['ttft_pct']['mean']:+.2f} +- {delta['ttft_pct']['ci95']:.2f} %")
    print(f"   wait  {delta['wait_ms']['mean']:+.1f} +- {delta['wait_ms']['ci95']:.1f} ms/tok")
    print(f"   validity: per-prompt means {', '.join(f'{p} {x:+.2f}' for p, x in zip(cells['tok_s_pct'], pmeans))}; "
          f"SD between prompts {prompt_sd:.2f} points. The interval is pair noise for this prompt set only; "
          f"another set can move the mean by about SD/sqrt(n_prompts) = "
          f"{prompt_sd / math.sqrt(len(pmeans)) if pmeans else NAN:.2f} points.")
    print(f"   same-arm spread between runs, %: run means {run_spread} (guard > {SPREAD_PCT:.0f}%); "
          f"per prompt {spread} (warning > {SPREAD_WARN_PCT}%)")
    print(f"identity ({basis}) over {check_ids}: "
          + ('identical in every run' if same else f'DIFFERS: {json.dumps({i: v for i, v in ident.items() if v})}')
          + (f"; logits bit-identical: {len(set(shas.values())) == 1}" if shas else ''))
    if basis == 'reply text' and not same:
        notes.append('m0/serve_bench.py (W3, ab-samedrive) can drop reply bytes on short pipe reads: '
                     'a text difference is not proof of different tokens, check with --dump')

    rule = '(user rule: token ids identical and B not slower)'
    if invalid:
        verdict = 'INVALID: ' + '; '.join(invalid)
    elif basis.startswith('token ids') and not same:
        verdict = f'NOT ADOPTED {rule}: token ids differ'
    elif guards:
        verdict = 'INCONCLUSIVE: ' + '; '.join(guards)
    elif not used or math.isnan(ci):
        verdict = f'INCONCLUSIVE: no interval (prompts {used}, df={df})'
    # KEEP B / NOT ADOPTED also need every pair mean on their side: the fixed-prompt CI assumes away the
    # run-level effect the pair means carry (W3 at 64 tokens: +-0.31 against +-0.70)
    elif m >= 0 and same and basis.startswith('token ids') and all(x >= 0 for x in pairm):
        verdict = f'KEEP B {rule}: {m:+.2f} +- {ci:.2f} %, pair means {[round(x, 2) for x in pairm]}'
    elif m + ci < 0 and all(x < 0 for x in pairm):
        verdict = (f'NOT ADOPTED {rule}: B slower, whole CI below 0 ({m:+.2f} +- {ci:.2f} %) and every pair '
                   f'mean below 0 {[round(x, 2) for x in pairm]}')
    else:
        verdict = (f'INCONCLUSIVE: {m:+.2f} +- {ci:.2f} %, pair means {[round(x, 2) for x in pairm]}'
                   + (', a pair mean below 0' if m >= 0 and any(x < 0 for x in pairm) else
                      ', a pair mean at or above 0' if m + ci < 0 else ', CI spans 0' if m < 0 else
                      ', token identity only from reply text (run with --dump)'))
    for n in notes:
        print('note:', n)
    print(f'verdict: {verdict}')

    if a.json:
        for s in S.values():
            for p in s['prompts'].values():
                p.pop('text')
        a.json.write_text(json.dumps({
            'runs': S, 'ids': ids, 'used_ids': used, 'pairs': pairs, 'arm_stats': stats, 'delta_b_vs_a': delta,
            'per_prompt_tok_s_pct': per_prompt, 'prompt_sd_points': prompt_sd, 'same_arm_spread_pct': spread,
            'same_arm_run_mean_spread_pct': run_spread,
            'pair_means_tok_s_pct': pairm, 'pair_means_ci95': pair_ci,
            'identity_basis': basis, 'identical': same,
            'differs': {i: v for i, v in ident.items() if v}, 'logits_sha256': shas,
            'logits_bit_identical': len(set(shas.values())) == 1 if shas else None,
            'invalid': invalid, 'guards': guards, 'verdict': verdict, 'notes': notes}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    # fixed effects: residuals +-1 around each prompt's mean, ss=4, df=2, half-width t(2)*sqrt(4/2/4)
    assert fixed_ci({'a': [1.0, 3.0], 'b': [5.0, 7.0]})[:2] == (4.0, 4.303 * math.sqrt(0.5))
    assert t975(30) == 2.042 and t975(31) == 2.0
    main()
