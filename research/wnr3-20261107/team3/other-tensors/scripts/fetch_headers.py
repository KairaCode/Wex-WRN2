"""Fetch all 48 safetensors headers (Range) and summarize sizes by tensor category and dtype.
Usage: python -I fetch_headers.py INDEX_JSON CACHE_DIR OUT_JSON"""
import json, re, sys, os, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf

idx, cache, outp = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(cache, exist_ok=True)
wm = json.load(open(idx, encoding='utf-8'))['weight_map']
files = sorted(set(wm.values()))
with ThreadPoolExecutor(4) as ex:
    hdrs = dict(zip(files, ex.map(lambda f: hf.header(f, cache), files)))

rows = []
for f, (base, h) in hdrs.items():
    for name, m in h.items():
        if name == '__metadata__':
            continue
        a, b = m['data_offsets']
        rows.append((name, f, m['dtype'], m['shape'], b - a, base + a))


def cat(name):
    g = re.sub(r'\.\d+\.', '.N.', name)
    g = re.sub(r'\.\d+\.', '.N.', g)
    return g

agg = collections.defaultdict(lambda: [0, 0, set(), set()])
for name, f, dt, sh, nb, off in rows:
    a = agg[cat(name)]
    a[0] += 1; a[1] += nb; a[2].add(dt); a[3].add(tuple(sh))
tot = sum(r[4] for r in rows)
print('tensors', len(rows), 'total bytes', tot)
for g, (c, nb, dts, shs) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    shs = sorted(shs)
    print(f'{nb:16,d} {100*nb/tot:6.2f}% {c:6d} {g:50s} {"/".join(sorted(dts)):10s} {shs[:3]}{"..." if len(shs)>3 else ""}')
# coarse groups
def group(n):
    if n.startswith('layers.') and '.ffn.experts.' in n: return 'main routed experts'
    if n.startswith('mtp.') and '.ffn.experts.' in n: return 'mtp routed experts'
    if '.engram.embed.' in n: return 'engram embed tables'
    if '.engram.' in n: return 'engram other'
    if 'shared_experts' in n: return ('mtp ' if n.startswith('mtp') else 'main ') + 'shared expert'
    if n.startswith('vision') or n.startswith('aligner') or n.startswith('image'): return 'vision'
    if n in ('embed.weight', 'head.weight'): return n
    if n.startswith('mtp.'): return 'mtp dense (attn/gate/norm/heads)'
    if '.attn.' in n: return 'main attention'
    return 'main other (gate/norm/hc)'
g2 = collections.defaultdict(lambda: [0, 0, collections.Counter()])
for name, f, dt, sh, nb, off in rows:
    a = g2[group(name)]; a[0] += 1; a[1] += nb; a[2][dt] += nb
print()
for g, (c, nb, dts) in sorted(g2.items(), key=lambda kv: -kv[1][1]):
    print(f'{nb:16,d} {nb/2**30:8.2f} GiB {100*nb/tot:6.2f}% {c:6d} {g:35s} {dict(dts)}')
json.dump([{'name': r[0], 'file': r[1], 'dtype': r[2], 'shape': r[3], 'bytes': r[4], 'offset': r[5]} for r in rows],
          open(outp, 'w', encoding='utf-8'))
