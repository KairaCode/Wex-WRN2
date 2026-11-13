"""Second MTP sample: 6 more experts per stage, disjoint from the first sample (seed 20261108)."""
import json, random, sys
rng0 = random.Random(20261107)
first = {s: set(rng0.sample(range(128), 6)) for s in range(3)}
rng = random.Random(20261108)
spec = []
for s in range(3):
    pool = [e for e in range(128) if e not in first[s]]
    for e in sorted(rng.sample(pool, 6)):
        for w in ('w1', 'w2', 'w3'):
            for k in ('scale', 'weight'):
                spec.append({'name': f'mtp.{s}.ffn.experts.{e}.{w}.{k}'})
json.dump(spec, open(sys.argv[1], 'w'), indent=0)
print(len(spec))
