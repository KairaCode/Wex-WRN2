"""Compact table of team 1 merged candidates with both verdicts."""
import json, sys

d = json.load(open(sys.argv[1], encoding='utf-8'))
res = d['result']
mode = sys.argv[2] if len(sys.argv) > 2 else 'table'
sv = {v['name']: v for v in res['srcCheck']['verdicts']}
tv = {v['name']: v for v in res['techCheck']['verdicts']}
if mode == 'table':
    for i, c in enumerate(res['merged']):
        n = c['name']
        s = sv.get(n, {}); t = tv.get(n, {})
        print(f"[{i}] {n} | {c['category']} | {c['lossless']} | src={s.get('status')} | tech={t.get('status')} | conf={c['confidence']}")
        print('    TECH-CORR:', (t.get('corrected_expected_effect') or '')[:300])
elif mode == 'overall':
    print('SRC OVERALL:\n', res['srcCheck']['overall'])
    print('\nTECH OVERALL:\n', res['techCheck']['overall'])
    print('\nMISSING (tech):')
    for m in res['techCheck']['missing']:
        print('-', m['name'], '\n  ', m['summary'][:900], '\n   EFFECT:', m['expected_effect'][:500])
elif mode == 'one':
    i = int(sys.argv[3])
    c = res['merged'][i]
    print(json.dumps(c, indent=1, ensure_ascii=False))
    print('SRC:', json.dumps(sv.get(c['name']), indent=1, ensure_ascii=False))
    print('TECH:', json.dumps(tv.get(c['name']), indent=1, ensure_ascii=False))
elif mode == 'angles':
    for sw in res['sweeps']:
        print('=== ', sw['angle'], '\n', sw['angle_verdict'][:2500], '\n')
