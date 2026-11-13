"""Synthetic-run checks of analyze_short.py's guards and verdict (no engine): writes tiny A1 B1 B2 A2 runs
(7 prompts, tokens.json with identical ids) under OUTDIR/<case>, runs the analyzer on each, checks the verdict.

  python -I check_analyze.py [OUTDIR]     (default: runs/check-analyze next to this file)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDS = ['code1', 'code2', 'chat_it1', 'chat_it2', 'reason1', 'reason2', 'short1']


def write_run(d, name, rate, slots=9, vram=965):
    rates = rate if isinstance(rate, dict) else dict.fromkeys(IDS, rate)
    recs = [{'event': 'ready', 'secs': 1.0}] + [
        {'id': p, 'max_tokens': 64, 'done': f'DONE {p} OK 64 {r:.6f} 50 1 10 1', 'ttft': 1.0, 'wall': 64 / r,
         'text': 'x'} for p, r in rates.items()]
    (d / f'{name}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in recs), encoding='utf-8')
    (d / f'{name}.stderr.log').write_text(
        f'[    0.5] ram_tiers available=16.00GiB dense=resident(6.65GiB) target_slots={slots} target_cache=6.30GiB\n'
        f'[   15.0] v4_tier_prewarm vram_experts={vram} frequency_high=4 frequency_low=1\n', encoding='utf-8')


def case(root, name, runs, expect, also=''):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        f.unlink()
    for run, args in runs.items():
        write_run(d, run, *args) if isinstance(args, tuple) else write_run(d, run, args)
    (d / 'tokens.json').write_text(json.dumps({'logits_sha256': {r: '0' * 64 for r in runs},
                                               'tokens': {r: {p: [1, 2, 3] for p in IDS} for r in runs}}))
    out = subprocess.run([sys.executable, '-I', str(HERE / 'analyze_short.py'), str(d)],
                         capture_output=True, text=True, encoding='utf-8')
    verdict = next((l for l in out.stdout.splitlines() if l.startswith('verdict:')), out.stderr.strip()[-300:])
    ok = verdict.startswith('verdict: ' + expect) and also in out.stdout
    print(f"{'ok  ' if ok else 'FAIL'} {name:20s} {verdict[:170]}")
    return ok


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'runs' / 'check-analyze'
    alt = {p: (1.05 if p == 'code1' else 1.0) for p in IDS}, {p: (1.05 if p == 'code2' else 1.0) for p in IDS}
    results = [
        case(root, 'keep', {'A1': 1.0, 'B1': 1.01, 'B2': 1.012, 'A2': 1.0}, 'KEEP B'),
        case(root, 'keep-pair-below', {'A1': 1.0, 'B1': 1.01, 'B2': 0.999, 'A2': 1.0}, 'INCONCLUSIVE',
             'a pair mean below 0'),
        case(root, 'not-adopted', {'A1': 1.0, 'B1': 0.97, 'B2': 0.975, 'A2': 1.0}, 'NOT ADOPTED'),
        # pair means -1.95 / +0.05: the fixed-prompt CI (+-0.89, df=7) is below 0, a pair mean is not
        case(root, 'na-pair-above', {'A1': 1.0, 'B1': 0.9805, 'B2': 0.99 * 1.0005, 'A2': 0.99}, 'INCONCLUSIVE',
             'a pair mean at or above 0'),
        case(root, 'run-mean-spread', {'A1': 1.0, 'B1': 0.99, 'B2': 1.0197, 'A2': 1.03}, 'INCONCLUSIVE',
             'run means > 2%'),
        case(root, 'prompt-spread-warn', {'A1': alt[0], 'B1': {p: 1.01 * r for p, r in alt[0].items()},
                                          'B2': {p: 1.01 * r for p, r in alt[1].items()}, 'A2': alt[1]},
             'KEEP B', 'warning: per-prompt same-arm'),
        case(root, 'vram-differs', {'A1': 1.0, 'B1': 1.01, 'B2': (1.01, 9, 900), 'A2': 1.0}, 'INVALID',
             'VRAM expert tier'),
        case(root, 'vram-within-5pct', {'A1': 1.0, 'B1': 1.01, 'B2': (1.01, 9, 930), 'A2': 1.0}, 'KEEP B'),
        case(root, 'slots-differ', {'A1': 1.0, 'B1': 1.01, 'B2': 1.01, 'A2': (1.0, 7)}, 'INVALID',
             'RAM expert slots'),
        case(root, 'no-pair', {'B1': 1.0}, 'INVALID', 'no A<i>/B<i> pair'),
    ]
    print(f'{sum(results)}/{len(results)} cases ok')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
