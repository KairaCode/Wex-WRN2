"""40 held-out experts of the 5090 study + the N largest WNR2 production records (t1_record_sizes.json).
usage: python -I select_experts.py N > pairs.txt  (prints L:E,... and writes select.json)"""
import json
import sys
R = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'wnr3-5090-20261107')
n = int(sys.argv[1])
lab = json.load(open(R + r'\ceil-kaira\ceil_kaira_records.json'))['rows']
test = [(r['layer'], r['expert']) for r in lab if r['set'] == 'test']
big12 = [(r['layer'], r['expert']) for r in lab if r['set'] == 'big']
t1 = json.load(open(R + r'\t1_record_sizes.json'))['rows']
by_size = sorted(t1, key=lambda r: -r['total'])
tail = [(r['i'] // 192, r['i'] % 192) for r in by_size[:n]]
assert len(test) == 40 and set(big12) <= set(tail), 'the 12 lab tail records must be in the top N'
pairs = test + [p for p in tail if p not in test]
wnr2 = {(r['i'] // 192, r['i'] % 192): {'total': r['total'], 'used': r['used'], 'median': r['median']} for r in t1}
json.dump({'test': test, 'tail': tail, 'big12': big12, 'pairs': pairs,
           'wnr2': {f'{L}:{e}': wnr2[(L, e)] for L, e in pairs},
           'wnr2_tail_cut_total': by_size[n - 1]['total']}, open('select.json', 'w'), indent=1)
print(','.join(f'{L}:{e}' for L, e in pairs))
