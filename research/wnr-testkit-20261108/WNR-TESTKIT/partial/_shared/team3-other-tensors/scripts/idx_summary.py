import json, re, sys, collections
idx = json.load(open(sys.argv[1], encoding='utf-8'))
print('metadata:', json.dumps(idx.get('metadata', {}))[:500])
wm = idx['weight_map']
print('n tensors', len(wm))
files = collections.Counter(wm.values())
print('n files', len(files))
# generalize names: digits -> N
pat = collections.Counter()
pat_files = collections.defaultdict(set)
for k, f in wm.items():
    g = re.sub(r'\.\d+\.', '.N.', k)
    g = re.sub(r'\.\d+\.', '.N.', g)
    pat[g] += 1
    pat_files[g].add(f)
for g, c in sorted(pat.items()):
    fs = sorted(pat_files[g])
    print(f'{c:7d}  {g}   files={len(fs)} [{fs[0]} .. {fs[-1]}]')
# files summary
print()
for f, c in sorted(files.items())[:5]:
    print(f, c)
print('...')
for f, c in sorted(files.items())[-8:]:
    print(f, c)
