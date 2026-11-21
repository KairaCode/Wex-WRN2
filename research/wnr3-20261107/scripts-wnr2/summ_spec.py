import json, sys
sys.stdout.reconfigure(encoding='utf-8')
d = json.load(open(sys.argv[1], encoding='utf-8'))['result']
print('problems_count', d['problems_count'])
for v in d['verdicts']:
    pr = [i for i in v['items'] if i['status'] == 'problem']
    print('==', v['member'], 'problems', len(pr), '| sev', {s: sum(1 for i in pr if i['severity'] == s) for s in ('high','medium','low')})
    print('   ', v['summary'][:700])
f = d.get('fix')
if f:
    print('== fixer:', f['summary'][:1500])
    from collections import Counter
    print('   ', Counter(i['status'] for i in f['items']))
    for i in f['items']:
        if i['status'] != 'applied':
            print('   NOT APPLIED', i['id'], i['severity'], i['detail'][:500])
    print('   files', f['files_written'])
