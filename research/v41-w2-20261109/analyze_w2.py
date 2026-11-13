"""W2 runs (run_w2.sh): per arm and prompt decode tok/s, TTFT, moeprof wait; logits sha256 and greedy
token identity against arm A (WEX1); the wexprof lines (per-record vs batched decode ms).

  python -I analyze_w2.py DUMPDIR   -> text on stdout
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

VOCAB = 129280
RUNS = Path(__file__).resolve().parent / 'runs'
ARMS = ['w2-A-wex1', 'w2-R-wnr2-record', 'w2-B-wnr2-batch', 'w2-B2-wnr2-batch', 'w2-R2-wnr2-record']


def load(name):
    rows = [json.loads(l) for l in (RUNS / f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if l]
    log = (RUNS / f'{name}.stderr.log').read_text(encoding='utf-8', errors='replace').splitlines()
    return [r for r in rows if 'id' in r], next(r['secs'] for r in rows if r.get('event') == 'ready'), log


def done(rec):   # DONE id STAT tokens tok/s hit% rss prompt_tokens length_limited
    f = rec['done'].split()
    return int(f[3]), float(f[4])


def tokens(dump, name, recs):
    rows = np.memmap(Path(dump) / f'{name}.logits', dtype=np.float32, mode='r').reshape(-1, VOCAB)
    out, at = {}, 0
    for r in recs:
        n = done(r)[0]
        n += n < r.get('max_tokens', n)   # stopped on EOS: its step dumped one more row
        out[r['id']] = [int(x) for x in rows[at:at + n].argmax(1)]
        at += n
    return out, len(rows), at


def moe_wait(log, t0, t1):   # mean moeprof wait (ms/token) of the decode lines in [t0, t1)
    w = []
    for line in log:
        m = re.match(r'\[\s*([\d.]+)\] moeprof .*?wait=([\d.]+)', line)
        if m and t0 <= float(m.group(1)) < t1:
            w.append(float(m.group(2)))
    return sum(w) / len(w) if w else float('nan')


def main():
    dump = sys.argv[1]
    arms = ['w2-A-wex1'] + sys.argv[2:] if len(sys.argv) > 2 else ARMS   # A stays the token reference
    names = [n for n in arms if (RUNS / f'{n}.jsonl').exists()]
    toks, summary = {}, {}
    for n in names:
        recs, ready, log = load(n)
        sha = hashlib.sha256((Path(dump) / f'{n}.logits').read_bytes()).hexdigest()[:16]
        toks[n], rows, counted = tokens(dump, n, recs)
        print(f'{n}: READY {ready:.1f} s, logits {sha} ({rows} rows, DONE counts {counted})')
        t = ready
        for r in recs:
            t0, t = t, t + r['wall']
            k, tps = done(r)
            summary.setdefault(r['id'], {})[n] = tps
            print(f'   {r["id"]:9s} tok/s {tps:6.3f} ttft {r["ttft"]:7.2f} s  moeprof wait {moe_wait(log, t0, t):6.1f} ms/tok')
        for line in log:
            if 'wexprof' in line or 'v4_wex_gpu_direct reads' in line or 'v4_gpu_mirror_epochs' in line:
                print('   ' + line.split('] ', 1)[-1])
    ref = toks.get('w2-A-wex1')
    print('\ntoken identity against w2-A-wex1 (greedy argmax per decode step):')
    for n in names:
        same = {p: toks[n].get(p) == ref.get(p) for p in ref} if ref else {}
        print(f'   {n}: ' + ', '.join(f'{p} {"same" if s else "DIFF"}' for p, s in same.items()))
    print('\ndecode tok/s per prompt (w0 excluded from the means):')
    ids = [p for p in summary if p != 'w0']
    for n in names:
        xs = [summary[p][n] for p in ids if n in summary[p]]
        print(f'   {n:20s} ' + ' '.join(f'{summary[p].get(n, float("nan")):6.3f}' for p in ids) +
              f'   mean {sum(xs) / len(xs):.3f}' if xs else '')


if __name__ == '__main__':
    main()
