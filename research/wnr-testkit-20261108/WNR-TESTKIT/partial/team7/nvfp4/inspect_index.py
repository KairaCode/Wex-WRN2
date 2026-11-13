"""Print expert tensor naming patterns of safetensors index files. Usage: python -I inspect_index.py META_DIR tag1 [tag2 ...]"""
import json, collections, re, sys

m = sys.argv[1]
for tag in sys.argv[2:]:
    wm = json.load(open(f"{m}\\{tag}__model.safetensors.index.json"))["weight_map"]
    print(tag, len(wm), 'shards', len(set(wm.values())))
    sfx = collections.Counter(re.sub(r"\d+", "#", k) for k in wm)
    for k, v in sfx.most_common(40):
        if 'expert' in k or 'mlp' in k or 'ffn' in k:
            print("      ", v, k)
