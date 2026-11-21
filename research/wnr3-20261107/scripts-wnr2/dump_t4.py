import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
r = d['result']
json.dump(r['critic'], open(sys.argv[2], 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump({'validators': r['validators'], 'integration': r['integration']}, open(sys.argv[3], 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(len(r['critic']['findings']), 'critic findings;', [m['member'] for m in r['validators'] + r['integration']])
