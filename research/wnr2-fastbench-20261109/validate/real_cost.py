"""Per-token decode cost of real engine runs (serve_bench .jsonl + .stderr.log), for the replay comparison.

  python -I real_cost.py [--ids code1,chat_it1,reason1] RUN_PREFIX [RUN_PREFIX ...]

RUN_PREFIX = path without extension (runs/A1, traces/wex1-x). Per run: means over the decode tokens of
--ids (READY + cumulative wall maps stderr stamps to prompts, as analyze_short.py) of moeprof wait /
attach / gpu and decprof hc / attn / moe, plus the exit read counters (prefill included).
"""
import json
import re
import sys

MOE = re.compile(r'\[\s*([\d.]+)\] moeprof wait=([\d.]+) attach=([\d.]+) gpu=([\d.]+) ms vram_hits=(\d+)')
DEC = re.compile(r'\[\s*([\d.]+)\] decprof position=\d+ layers=\d+ hc=(\d+) attn=(\d+) moe=(\d+) ms')


def run_cost(prefix, ids):
    recs = [json.loads(l) for l in open(prefix + '.jsonl', encoding='utf-8') if l.strip()]
    log = open(prefix + '.stderr.log', encoding='utf-8', errors='replace').read()
    t = next(r['secs'] for r in recs if r.get('event') == 'ready')
    spans = []
    for r in recs:
        if 'id' in r:
            if r['id'] in ids:
                spans.append((t, t + r['wall']))
            t += r['wall']
    inside = lambda ts: any(a <= ts < b for a, b in spans)
    moe = [tuple(map(float, m.groups()[1:])) for m in MOE.finditer(log) if inside(float(m[1]))]
    dec = [tuple(map(float, m.groups()[1:])) for m in DEC.finditer(log) if inside(float(m[1]))]
    avg = lambda rows, i: sum(r[i] for r in rows) / len(rows) if rows else float('nan')
    out = {'run': prefix, 'tokens': len(moe), 'wait': avg(moe, 0), 'attach': avg(moe, 1), 'gpu': avg(moe, 2),
           'vram_miss': 240 - avg(moe, 3), 'hc': avg(dec, 0), 'attn': avg(dec, 1), 'decprof_moe': avg(dec, 2)}
    out['wait_attach_gpu'] = out['wait'] + out['attach'] + out['gpu']
    reads = gb = 0
    for key in ('v4_direct', 'v4_wex_gpu_direct'):
        m = re.findall(key + r' reads=(\d+).*?payload_bytes=(\d+)', log)
        if m:
            reads += int(m[-1][0])
            gb += int(m[-1][1]) / 1e9
    out['exit_reads'], out['exit_gb'] = reads, round(gb, 1)
    return out


def main(argv):
    ids = {'code1', 'chat_it1', 'reason1'}
    if argv[:1] == ['--ids']:
        ids, argv = set(argv[1].split(',')), argv[2:]
    if not argv:
        raise SystemExit(__doc__)
    cols = ('tokens', 'wait', 'attach', 'gpu', 'wait_attach_gpu', 'vram_miss', 'hc', 'attn', 'decprof_moe',
            'exit_reads', 'exit_gb')
    print(f"{'run':40}" + ''.join(f'{c:>16}' for c in cols))
    for p in argv:
        o = run_cost(p, ids)
        print(f"{p[-40:]:40}" + ''.join(f'{o[c]:16.1f}' if isinstance(o[c], float) else f'{o[c]:16}' for c in cols))


if __name__ == '__main__':
    main(sys.argv[1:])
