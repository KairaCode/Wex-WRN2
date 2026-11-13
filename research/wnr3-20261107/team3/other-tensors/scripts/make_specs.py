"""Write the download specs (which tensors / row slices) to SPEC_DIR."""
import json, os, random, sys
spec_dir = sys.argv[1]
os.makedirs(spec_dir, exist_ok=True)

# 1) MTP routed experts: 6 of 128 per MTP layer, same seed convention as the main sample
rng = random.Random(20261107)
mtp = []
for s in range(3):
    for e in sorted(rng.sample(range(128), 6)):
        for w in ('w1', 'w2', 'w3'):
            for k in ('scale', 'weight'):
                mtp.append({'name': f'mtp.{s}.ffn.experts.{e}.{w}.{k}'})
json.dump(mtp, open(os.path.join(spec_dir, 'mtp.json'), 'w'), indent=0)

# 2) FP8 dense: shared experts of 4 layers, attention of 2 layers, one engram wkv
fp8 = []
for L in (0, 13, 26, 39):
    for w in ('w1', 'w2', 'w3'):
        for k in ('weight', 'scale'):
            fp8.append({'name': f'layers.{L}.ffn.shared_experts.{w}.{k}'})
for L in (0, 26):
    for w in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'):
        for k in ('weight', 'scale'):
            fp8.append({'name': f'layers.{L}.attn.{w}.{k}'})
for k in ('weight', 'scale'):
    fp8.append({'name': f'layers.1.engram.wkv.{k}'})
json.dump(fp8, open(os.path.join(spec_dir, 'fp8.json'), 'w'), indent=0)

# 3) BF16: embed/head row slices, one router
rng = random.Random(5)
bf = []
for nm in ('embed.weight', 'head.weight'):
    for r0 in sorted(rng.sample(range(0, 129280 - 800), 4)):
        bf.append({'name': nm, 'rows': [r0, r0 + 800]})
bf.append({'name': 'layers.20.ffn.gate.weight'})
json.dump(bf, open(os.path.join(spec_dir, 'bf16.json'), 'w'), indent=0)

# 4) Engram tables: 96 random 128-row chunks per table (+ matching scale rows), one 16384-row run
rng = random.Random(14)
eg = []
for L, N in ((1, 384006168), (14, 384016682)):
    for r0 in sorted(rng.sample(range(0, N - 128), 96)):
        for k in ('weight', 'scale'):
            eg.append({'name': f'layers.{L}.engram.embed.{k}', 'rows': [r0, r0 + 128]})
r0 = rng.randrange(0, 384006168 - 16384)
for k in ('weight', 'scale'):
    eg.append({'name': f'layers.1.engram.embed.{k}', 'rows': [r0, r0 + 16384], 'tag': f'run16k.layers.1.engram.embed.{k}.r{r0}'})
json.dump(eg, open(os.path.join(spec_dir, 'engram.json'), 'w'), indent=0)
print(len(mtp), len(fp8), len(bf), len(eg))
