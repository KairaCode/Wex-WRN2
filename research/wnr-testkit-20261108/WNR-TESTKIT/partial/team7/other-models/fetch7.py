"""Range-download MoE expert slices (MXFP4 weights + E8M0 scales) of public ungated checkpoints (team7/other-models).
Adapted from team4 ceil-origin/fetch_other.py (same manifest layout).

Usage: python -I fetch7.py FAMILY INDEX_JSON OUT_DIR MAX_MB [--plan]
Each item OUT_DIR/<tag>.bin = listed tensors back to back; OUT_DIR/manifest.json lists tensors (name, role, part, dtype,
shape, bytes, shard, offset) and sha256 per item. Aborts before downloading if the planned bytes exceed MAX_MB.
Nothing downloaded is executed; only HTTP Range GETs on public resolve URLs (no token).
"""
import json, os, random, struct, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {'User-Agent': 'team7-other-models/1.0'}
DSROLES = (('w1', 'gate_proj'), ('w2', 'down_proj'), ('w3', 'up_proj'))


def items(prefix_fmt, layers, n_exp, seed, roles=DSROLES, wsuf='weight', ssuf='weight_scale', avoid=()):
    rng = random.Random(seed)
    out = []
    for L in layers:
        while True:
            e = rng.randrange(n_exp)
            if (L, e) not in avoid:
                break
        p = prefix_fmt.format(L=L, E=e)
        ts = []
        for role, nm in roles:
            ts.append({'name': f'{p}{nm}.{wsuf}', 'role': role, 'part': 'weight', 'slice0': None})
            ts.append({'name': f'{p}{nm}.{ssuf}', 'role': role, 'part': 'scale', 'slice0': None})
        out.append({'tag': f'L{L:02d}_E{e:03d}', 'layer': L, 'expert': e, 'tensors': ts})
    return out


def spread(lo, hi, n):
    return sorted({lo + round(k * (hi - lo) / (n - 1)) for k in range(n)})


FAMILIES = {
    # Kimi-K2.6 (DeepseekV3 text model): 61 layers, first_k_dense_replace 1 -> MoE layers 1..60, 384 routed experts
    'kimi26': ('amd/Kimi-K2.6-MXFP4', '18ecc30a67047bbf65bf55ad7643985288c7b8e0',
               lambda: items('language_model.model.layers.{L}.mlp.experts.{E}.', spread(1, 60, 20), 384, 71)),
    # MiniMax-M3: 60 layers, moe_layer_freq 0 for layers 0..2 -> MoE layers 3..59, 128 experts, w1/w2/w3 naming
    'minimax3': ('amd/MiniMax-M3-MXFP4', 'b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a',
                 lambda: items('language_model.model.layers.{L}.block_sparse_moe.experts.{E}.', spread(3, 59, 16), 128, 72,
                               roles=(('w1', 'w1'), ('w2', 'w2'), ('w3', 'w3')))),
    # extra items on layers not used by team 4 (team 4: R1 [3,14,25,36,47,58], GLM [3,18,33,48,63,77])
    'r1x': ('amd/DeepSeek-R1-0528-MXFP4', '913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a',
            lambda: items('model.layers.{L}.mlp.experts.{E}.', [9, 20, 31, 42, 53, 60], 256, 73)),
    'glm53x': ('RedHatAI/GLM-5.3-MXFP4', '881184221de44698ee1334451e3d355eee657b45',
               lambda: items('model.layers.{L}.mlp.experts.{E}.', [10, 26, 41, 56, 70, 75], 256, 74, wsuf='weight_packed')),
}


def main():
    fam, index_json, out, max_mb = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    plan_only = '--plan' in sys.argv
    repo, rev, mk = FAMILIES[fam]
    base_url = f'https://huggingface.co/{repo}/resolve/{rev}/'

    def get_range(fname, start, end_incl, tries=6):
        for t in range(tries):
            try:
                req = urllib.request.Request(base_url + fname, headers={**UA, 'Range': f'bytes={start}-{end_incl}'})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) != end_incl - start + 1:
                    raise IOError(f'short read {len(data)} != {end_incl - start + 1}')
                return data
            except Exception:
                if t == tries - 1:
                    raise
                time.sleep(2 + 4 * t)

    hdrs = {}

    def shard_header(fname):
        if fname not in hdrs:
            n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
            hdrs[fname] = (8 + n, json.loads(get_range(fname, 8, 8 + n - 1)))
        return hdrs[fname]

    os.makedirs(out, exist_ok=True)
    wm = json.load(open(index_json, encoding='utf-8'))['weight_map']
    its = mk()
    total = 0
    for it in its:
        for t in it['tensors']:
            f = wm[t['name']]
            b0, hdr = shard_header(f)
            m = hdr[t['name']]
            a, b = m['data_offsets']
            t.update({'dtype': m['dtype'], 'shape': list(m['shape']), 'bytes': b - a, 'shard': f, 'offset': b0 + a,
                      'full_shape': m['shape']})
            total += b - a
    print(fam, repo, rev, 'items', len(its), 'planned bytes', total, flush=True)
    for it in its:
        print(' ', it['tag'], [(t['role'], t['part'], t['dtype'], t['shape']) for t in it['tensors']][:2] if it is its[0] else '')
    if total > max_mb * 1e6:
        print('ABORT: over budget'); sys.exit(1)
    if plan_only:
        json.dump({'family': fam, 'repo': repo, 'revision': rev, 'planned_bytes': total, 'experts': its},
                  open(os.path.join(out, 'plan.json'), 'w'), indent=1)
        return

    def fetch(it):
        path = os.path.join(out, it['tag'] + '.bin')
        if not os.path.exists(path):
            data = b''.join(get_range(t['shard'], t['offset'], t['offset'] + t['bytes'] - 1) for t in it['tensors'])
            with open(path + '.part', 'wb') as fh:
                fh.write(data)
            os.replace(path + '.part', path)
        it['file'] = os.path.basename(path)
        it['sha256'] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        print('  got', it['tag'], flush=True)
        return it

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(fetch, its))
    json.dump({'family': fam, 'repo': repo, 'revision': rev, 'experts': res}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(fam, len(res), 'items', total, 'bytes', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
