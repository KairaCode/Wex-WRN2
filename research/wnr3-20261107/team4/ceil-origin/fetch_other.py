"""Range-download a few MoE expert matrices (MXFP4 weights + E8M0 scales) of other public checkpoints.

Usage: python -I fetch_other.py FAMILY INDEX_JSON OUT_DIR MAX_MB
Each item file OUT_DIR/<tag>.bin holds the listed tensors back to back (dim-0 slices for stacked-expert tensors);
OUT_DIR/manifest.json lists per item: tensors (name, role, part, dtype, sliced shape, bytes, shard, offset).
Aborts before downloading if the planned bytes exceed MAX_MB. Nothing downloaded is executed.
"""
import json, os, random, struct, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {'User-Agent': 'team4-ceil-origin/1.0'}


def ds_items(prefix_fmt, layers, n_exp, seed, roles=(('w1', 'w1'), ('w2', 'w2'), ('w3', 'w3')), wsuf='weight', ssuf='scale'):
    rng = random.Random(seed)
    out = []
    for L in layers:
        e = rng.randrange(n_exp)
        p = prefix_fmt.format(L=L, E=e)
        ts = []
        for role, nm in roles:
            ts.append({'name': f'{p}{nm}.{wsuf}', 'role': role, 'part': 'weight', 'slice0': None})
            ts.append({'name': f'{p}{nm}.{ssuf}', 'role': role, 'part': 'scale', 'slice0': None})
        out.append({'tag': f'L{L:02d}_E{e:03d}', 'layer': L, 'expert': e, 'tensors': ts})
    return out


def gptoss_items(layers, n_exp, seed):
    rng = random.Random(seed)
    out = []
    for L in layers:
        e = rng.randrange(n_exp)
        p = f'model.layers.{L}.mlp.experts.'
        ts = []
        for role in ['gate_up_proj', 'down_proj']:
            ts.append({'name': f'{p}{role}_blocks', 'role': role, 'part': 'weight', 'slice0': e})
            ts.append({'name': f'{p}{role}_scales', 'role': role, 'part': 'scale', 'slice0': e})
        out.append({'tag': f'L{L:02d}_E{e:03d}', 'layer': L, 'expert': e, 'tensors': ts})
    return out


def amd_gptoss_items(pairs):
    out = []
    for L, e in pairs:
        p = f'model.layers.{L}.mlp.experts.{e}.'
        ts = []
        for role in ['gate_up_proj', 'down_proj']:
            ts.append({'name': f'{p}{role}.weight', 'role': role, 'part': 'weight', 'slice0': None})
            ts.append({'name': f'{p}{role}.weight_scale', 'role': role, 'part': 'scale', 'slice0': None})
        out.append({'tag': f'L{L:02d}_E{e:03d}', 'layer': L, 'expert': e, 'tensors': ts})
    return out


FAMILIES = {
    # same (layer, expert) pairs as the first 6 items of gptoss120, for a byte comparison with OpenAI's release
    'amd_gptoss120': ('amd/gpt-oss-120b-w-mxfp4-a-fp8', '67f173edc7899c7cc8322de699df6b849d8448e9',
                      lambda: amd_gptoss_items([(0, 14), (3, 23), (6, 21), (9, 92), (12, 43), (15, 78)])),
    'gptoss20': ('openai/gpt-oss-20b', '6cee5e81ee83917806bbde320786a8fb61efebee',
                 lambda: gptoss_items(range(0, 24, 2), 32, 1)),
    'gptoss120': ('openai/gpt-oss-120b', 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a',
                  lambda: gptoss_items(range(0, 36, 3), 128, 2)),
    'ds_v4flash': ('deepseek-ai/DeepSeek-V4-Flash', '60d8d70770c6776ff598c94bb586a859a38244f1',
                   lambda: ds_items('layers.{L}.ffn.experts.{E}.', [0, 6, 12, 18, 24, 30, 36, 42], 256, 3)),
    'ds_v41flash': ('deepseek-ai/DeepSeek-V4.1-Flash', '314113f8a7dadb388215396ba5fbee2c610a04d6',
                    lambda: ds_items('layers.{L}.ffn.experts.{E}.', [0, 13, 26, 39], 384, 4)),
    'ds_v4pro': ('deepseek-ai/DeepSeek-V4-Pro', 'b5968e9190ef611bbf34a7229255be88a0e937c1',
                 lambda: ds_items('layers.{L}.ffn.experts.{E}.', [1, 20, 40, 60], 384, 5)),
    'amd_r1': ('amd/DeepSeek-R1-0528-MXFP4', '913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a',
               lambda: ds_items('model.layers.{L}.mlp.experts.{E}.', [3, 14, 25, 36, 47, 58], 256, 6,
                                roles=(('w1', 'gate_proj'), ('w2', 'down_proj'), ('w3', 'up_proj')), ssuf='weight_scale')),
    'rh_glm53': ('RedHatAI/GLM-5.3-MXFP4', '881184221de44698ee1334451e3d355eee657b45',
                 lambda: ds_items('model.layers.{L}.mlp.experts.{E}.', [3, 18, 33, 48, 63, 77], 256, 7,
                                  roles=(('w1', 'gate_proj'), ('w2', 'down_proj'), ('w3', 'up_proj')),
                                  wsuf='weight_packed', ssuf='weight_scale')),
    'amd_minimax3': ('amd/MiniMax-M3-MXFP4', 'b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a',
                     lambda: ds_items('language_model.model.layers.{L}.block_sparse_moe.experts.{E}.', [0, 15, 30, 45, 59], 128, 8,
                                      ssuf='weight_scale')),
}


def main():
    fam, index_json, out, max_mb = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    repo, rev, mk = FAMILIES[fam]
    base_url = f'https://huggingface.co/{repo}/resolve/{rev}/'

    def get_range(fname, start, end_incl, tries=5):
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
                time.sleep(2 + 3 * t)

    hdrs = {}

    def shard_header(fname):
        if fname not in hdrs:
            n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
            hdrs[fname] = (8 + n, json.loads(get_range(fname, 8, 8 + n - 1)))
        return hdrs[fname]

    os.makedirs(out, exist_ok=True)
    wm = json.load(open(index_json))['weight_map']
    items = mk()
    total = 0
    for it in items:
        for t in it['tensors']:
            f = wm[t['name']]
            b0, hdr = shard_header(f)
            m = hdr[t['name']]
            a, b = m['data_offsets']
            shape = list(m['shape'])
            nbytes = b - a
            off = b0 + a
            if t['slice0'] is not None:
                per = nbytes // shape[0]
                assert per * shape[0] == nbytes
                off += per * t['slice0']
                nbytes = per
                shape = shape[1:]
            t.update({'dtype': m['dtype'], 'shape': shape, 'bytes': nbytes, 'shard': f, 'offset': off,
                      'full_shape': m['shape']})
            total += nbytes
    print(fam, 'planned bytes', total, flush=True)
    for t in items[0]['tensors']:
        print(' ', t['name'], t['dtype'], t['full_shape'], '->', t['shape'], t['bytes'])
    if total > max_mb * 1e6:
        print('ABORT: over budget'); sys.exit(1)

    def fetch(it):
        path = os.path.join(out, it['tag'] + '.bin')
        if not os.path.exists(path):
            data = b''.join(get_range(t['shard'], t['offset'], t['offset'] + t['bytes'] - 1) for t in it['tensors'])
            with open(path + '.part', 'wb') as fh:
                fh.write(data)
            os.replace(path + '.part', path)
        it['file'] = os.path.basename(path)
        it['sha256'] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        return it

    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, items))
    json.dump({'family': fam, 'repo': repo, 'revision': rev, 'experts': res}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(fam, len(res), 'items', total, 'bytes', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
