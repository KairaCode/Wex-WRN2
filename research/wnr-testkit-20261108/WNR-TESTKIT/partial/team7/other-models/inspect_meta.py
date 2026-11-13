"""Print config essentials and expert-tensor naming for the indexed checkpoints (meta files only, nothing downloaded).
Usage: python -I inspect_meta.py META_DIR"""
import json, os, re, sys
from collections import Counter

d = sys.argv[1]
KEYS = ['architectures', 'model_type', 'hidden_size', 'moe_intermediate_size', 'intermediate_size', 'num_hidden_layers',
        'n_routed_experts', 'num_local_experts', 'num_experts', 'num_experts_per_tok', 'first_k_dense_replace',
        'moe_layer_freq', 'n_shared_experts', 'shared_intermediate_size', 'quantization_config', 'torch_dtype', 'dtype']


def pick(cfg, depth=0):
    out = {}
    for k, v in cfg.items():
        if k in KEYS:
            if k == 'quantization_config':
                s = json.dumps(v)
                out[k] = s[:1500]
            else:
                out[k] = v
        elif isinstance(v, dict) and depth < 2 and k in ('text_config', 'language_config', 'llm_config'):
            out[k] = pick(v, depth + 1)
    return out


for f in sorted(os.listdir(d)):
    if f.endswith('__config.json'):
        cfg = json.load(open(os.path.join(d, f), encoding='utf-8'))
        print('=' * 20, f)
        print(json.dumps(pick(cfg), indent=1)[:4000])
for f in sorted(os.listdir(d)):
    if f.endswith('index.json'):
        wm = json.load(open(os.path.join(d, f), encoding='utf-8'))['weight_map']
        print('=' * 20, f, len(wm), 'tensors', len(set(wm.values())), 'shards')
        pat = Counter(re.sub(r'\d+', 'N', k) for k in wm)
        for k, c in sorted(pat.items()):
            if 'expert' in k or 'mlp' in k:
                print('  ', c, k)
