"""Range-download a sample of V4.1 REAP-192 MXFP4 expert tensors from Hugging Face.

Usage: python -I fetch_experts.py OUT_DIR PER_LAYER [SEED]
Writes OUT_DIR/L{layer:02d}_E{expert:03d}.bin (6 tensors back to back, in the order
w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight) and OUT_DIR/manifest.json
(per expert: tensor name, dtype, shape, byte length, source shard, offsets).
Only the bytes of the sampled tensors are fetched (HTTP Range); nothing is executed.
"""
import json, os, random, struct, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = 'patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
REV = '4ab649b2254afdcf6c2f7c1fbea1c05eda11828c'
BASE = f'https://huggingface.co/{REPO}/resolve/{REV}/'
UA = {'User-Agent': 'wnr2-research-fetch/1.0'}


def get_range(fname, start, end_incl, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(BASE + fname, headers={**UA, 'Range': f'bytes={start}-{end_incl}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != end_incl - start + 1:
                raise IOError(f'short read {len(data)} != {end_incl - start + 1}')
            return data
        except Exception as e:  # retry transient CDN errors
            if t == tries - 1:
                raise
            time.sleep(2 + 3 * t)


_hdr_cache = {}


def shard_header(fname):
    if fname not in _hdr_cache:
        n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
        hdr = json.loads(get_range(fname, 8, 8 + n - 1))
        _hdr_cache[fname] = (8 + n, hdr)
    return _hdr_cache[fname]


def main():
    out = sys.argv[1]
    per_layer = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261107
    os.makedirs(out, exist_ok=True)
    ix_path = os.path.join(os.path.dirname(out.rstrip('\\/')), 'v41-mxfp4-meta', 'model.safetensors.index.json')
    wm = json.load(open(ix_path))['weight_map']
    rng = random.Random(seed)
    picks = [(L, e) for L in range(40) for e in sorted(rng.sample(range(192), per_layer))]
    order = ['w1.scale', 'w2.scale', 'w3.scale', 'w1.weight', 'w2.weight', 'w3.weight']
    for f in sorted({wm[f'layers.{L}.ffn.experts.{e}.w1.weight'] for L, e in picks}):
        shard_header(f)  # warm the header cache serially

    def fetch(pe):
        L, e = pe
        path = os.path.join(out, f'L{L:02d}_E{e:03d}.bin')
        rows = []
        blobs = []
        for t in order:
            name = f'layers.{L}.ffn.experts.{e}.{t}'
            f = wm[name]
            base, hdr = shard_header(f)
            m = hdr[name]
            a, b = m['data_offsets']
            rows.append({'name': t, 'dtype': m['dtype'], 'shape': m['shape'], 'bytes': b - a, 'shard': f, 'offset': base + a})
            if not (os.path.exists(path)):
                blobs.append(get_range(f, base + a, base + b - 1))
        if blobs:
            data = b''.join(blobs)
            with open(path + '.part', 'wb') as fh:
                fh.write(data)
            os.replace(path + '.part', path)
        h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        return {'layer': L, 'expert': e, 'file': os.path.basename(path), 'sha256': h, 'tensors': rows}

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(fetch, picks))
    man = {'repo': REPO, 'revision': REV, 'seed': seed, 'per_layer': per_layer, 'experts': res}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    tot = sum(sum(t['bytes'] for t in r['tensors']) for r in res)
    print(f'{len(res)} experts, {tot} bytes, {time.time() - t0:.1f} s')
    print(json.dumps(res[0]['tensors'], indent=1))


if __name__ == '__main__':
    main()
