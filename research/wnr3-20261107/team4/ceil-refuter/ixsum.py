"""Summarise the safetensors index: groups of tensor names (numbers replaced by #), shards of mtp experts.
Usage: python -I ixsum.py INDEX_JSON"""
import json, re, sys, collections

wm = json.load(open(sys.argv[1]))['weight_map']
g = collections.Counter(re.sub(r'\.\d+\.', '.#.', re.sub(r'\.\d+\.', '.#.', k)) for k in wm)
for k, v in sorted(g.items()):
    if 'vision' in k:
        continue
    print(v, k)
mtp = sorted({int(m.group(1)) for k in wm for m in [re.match(r'mtp\.(\d+)\.ffn\.experts\.(\d+)\.', k)] if m})
print('mtp layers with experts', mtp)
for L in mtp:
    ex = sorted({int(re.match(r'mtp\.\d+\.ffn\.experts\.(\d+)\.', k).group(1)) for k in wm if k.startswith(f'mtp.{L}.ffn.experts.')})
    print('mtp', L, 'experts', len(ex), ex[:3], ex[-3:], 'shard e0 w1', wm.get(f'mtp.{L}.ffn.experts.0.w1.weight'))
print('shards', len(set(wm.values())))
