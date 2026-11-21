"""Summarise team 2 workflow output: per member verdict + findings (claim, kind, confidence, impact)."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(sys.argv[1], encoding='utf-8'))
res = d['result']
mode = sys.argv[2] if len(sys.argv) > 2 else 'verdicts'
print('TOP KEYS:', list(res.keys()))
members = res.get('team2') or res.get('team3', [])
if mode == 'verdicts':
    for m in members:
        if not m:
            continue
        print('=====', m.get('member'))
        print(m.get('verdict', '')[:6000])
        print()
elif mode == 'findings':
    who = sys.argv[3]
    for m in members:
        if m and m.get('member') == who:
            for f in m.get('findings', []):
                print('-', f.get('claim')[:900])
                print('   kind=', f.get('kind'), 'conf=', f.get('confidence'))
                print('   EVID:', (f.get('evidence') or '')[:700])
                print('   IMPACT:', (f.get('impact') or '')[:400])
            for k, v in m.items():
                if k not in ('member', 'verdict', 'findings'):
                    print('EXTRA', k, json.dumps(v, ensure_ascii=False)[:3000])
elif mode == 'other':
    for k, v in res.items():
        if k != 'team2':
            print(k, json.dumps(v, ensure_ascii=False)[:5000])

