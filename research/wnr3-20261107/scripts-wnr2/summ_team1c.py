"""Print sources (url, title, year) and tech findings for chosen team 1 candidate indices."""
import json, sys

d = json.load(open(sys.argv[1], encoding='utf-8'))
res = d['result']
tv = {v['name']: v for v in res['techCheck']['verdicts']}
sv = {v['name']: v for v in res['srcCheck']['verdicts']}
for i in [int(x) for x in sys.argv[2].split(',')]:
    c = res['merged'][i]
    print(f"[{i}] {c['name']}")
    for s in c['sources']:
        print('   -', s.get('url'), '|', s.get('title'), '|', s.get('year'))
    print('   TECH:', tv.get(c['name'], {}).get('findings', '')[:700])
    print('   SRC :', sv.get(c['name'], {}).get('findings', '')[:400])
