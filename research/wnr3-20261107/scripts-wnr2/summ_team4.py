"""Summarise a team 4 workflow output: validators / integration / critic, per member verdict, findings, numbers."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(sys.argv[1], encoding='utf-8'))
res = d['result']
who = sys.argv[2] if len(sys.argv) > 2 else None
members = list(res.get('validators') or []) + list(res.get('integration') or [])
if res.get('critic'):
    members.append(res['critic'])
for m in members:
    if not m or (who and m.get('member') != who):
        continue
    print('=====', m.get('member'))
    print(m.get('verdict', ''))
    if who:
        for f in m.get('findings', []):
            print('-', f.get('claim'))
            print('   kind=', f.get('kind'), 'conf=', f.get('confidence'))
            print('   EVID:', (f.get('evidence') or '')[:900])
            print('   IMPACT:', (f.get('impact') or '')[:500])
        for n in m.get('numbers', []):
            print('  N:', n.get('what'), '=', n.get('value'), '|', n.get('conditions')[:200])
        print('  FILES:', m.get('files_written'))
        print('  OPEN:', m.get('open_questions'))
    print()
