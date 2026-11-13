"""HTTP-Range download of NVFP4 MoE expert tensors (packed E2M1 + E4M3 block scales + FP32 tensor scales).

Usage: python -I fetch_nv.py FAMILY INDEX_JSON OUT_DIR MAX_MB
Each item file OUT_DIR/<tag>.bin holds the item's tensors back to back, in manifest order;
OUT_DIR/manifest.json lists per item: tensors (name, role, part, dtype, shape, bytes, shard, offset) and sha256.
Aborts before downloading if the planned bytes exceed MAX_MB. Nothing downloaded is executed.
Public, ungated repos only; no token is sent.
"""
import json, os, random, struct, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {'User-Agent': 'team7-nvfp4/1.0'}


def mk_items(pairs, prefix_fmt, roles, parts, tagfmt='L{L:02d}_E{E:03d}', extra=None):
    out = []
    for L, E in pairs:
        p = prefix_fmt.format(L=L, E=E)
        ts = [{'name': f'{p}{nm}.{suf}', 'role': role, 'part': part}
              for role, nm in roles for part, suf in parts]
        it = {'tag': tagfmt.format(L=L, E=E), 'layer': L, 'expert': E, 'tensors': ts}
        if extra:
            it.update(extra(L, E))
        out.append(it)
    return out


DS_ROLES = (('w1', 'w1'), ('w2', 'w2'), ('w3', 'w3'))
QW_ROLES = (('w1', 'gate_proj'), ('w2', 'down_proj'), ('w3', 'up_proj'))
MO_PARTS = (('weight', 'weight'), ('scale', 'weight_scale'), ('tscale', 'weight_scale_2'), ('iscale', 'input_scale'))
CT_PARTS = (('weight', 'weight_packed'), ('scale', 'weight_scale'), ('tscale', 'weight_global_scale'),
            ('iscale', 'input_global_scale'))


# kit: paths of the original PC replaced by environment variables / kit-relative defaults
SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
T4MAN = os.environ.get('WNR_T4MAN') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '_shared', 'team4-data-manifests')
T4UP = os.environ.get('WNR_T4_ORIGIN_UP', 'origin-up')  # dir with UP_L*_U*.bin from team4-ceil-origin/fetch_upstream.py


def ds41_pairs():
    """V4.1-1017 upstream expert ids paired with MXFP4 bytes already on disk.
    (a) first sampled REAP expert of every even layer of WNR_SAMPLE (byte-identical to upstream),
    (b) 4 REAP-pruned upstream experts in WNR_T4_ORIGIN_UP."""
    keep = json.load(open(os.path.join(T4MAN, 'origin', 'src', 'reap__reap_keepset.json')))['keep']
    man = json.load(open(os.path.join(SAMPLE, 'manifest.json')))
    first = {}
    for e in man['experts']:
        first.setdefault(e['layer'], e)
    pairs, src = [], {}
    for L in range(0, 40, 2):
        e = first[L]
        U = keep[str(L)][e['expert']]
        pairs.append((L, U))
        src[(L, U)] = {'mx_file': os.path.join(SAMPLE, e['file']), 'reap_index': e['expert']}
    for L, U, f in [(0, 301, 'UP_L00_U301.bin'), (10, 199, 'UP_L10_U199.bin'), (25, 76, 'UP_L25_U076.bin'),
                    (39, 304, 'UP_L39_U304.bin')]:
        pairs.append((L, U))
        src[(L, U)] = {'mx_file': os.path.join(T4UP, f), 'reap_index': None}
    return pairs, src


def qwen_pairs(seed=36):
    rng = random.Random(seed)
    return [(L, rng.randrange(256)) for L in range(40) for _ in range(2)]


def fam(name):
    if name == 'ds41_1017':
        pairs, src = ds41_pairs()
        return ('nvidia/DeepSeek-V4.1-Flash-1017-NVFP4', '3ae9c0d3fbbea5370bd06e679d22044db15883f1',
                mk_items(pairs, 'layers.{L}.ffn.experts.{E}.', DS_ROLES, MO_PARTS, tagfmt='L{L:02d}_U{E:03d}',
                         extra=lambda L, E: src[(L, E)]))
    if name == 'qwen36_nv':
        return ('nvidia/Qwen3.6-35B-A3B-NVFP4', '1355db6a052410cfd62085d94b58866fd0f2c3c5',
                mk_items(qwen_pairs(), 'model.language_model.layers.{L}.mlp.experts.{E}.', QW_ROLES, MO_PARTS))
    if name == 'qwen36_rh':
        return ('RedHatAI/Qwen3.6-35B-A3B-NVFP4', 'e94a930c8e9b8a7d3763e449289c261c9c7d3e5e',
                mk_items(qwen_pairs(), 'model.language_model.layers.{L}.mlp.experts.{E}.', QW_ROLES, CT_PARTS))
    if name == 'glm53_nv':
        # same (layer, expert) pairs as team 4's RedHatAI/GLM-5.3-MXFP4 sample
        man = json.load(open(os.path.join(T4MAN, 'other', 'rh_glm53', 'manifest.json')))
        pairs = [(e['layer'], e['expert']) for e in man['experts']]
        return ('nvidia/GLM-5.3-NVFP4', 'e3b8f9f2fad29727dfa95e766c5cac18ea1414bf',
                mk_items(pairs, 'model.layers.{L}.mlp.experts.{E}.', QW_ROLES, MO_PARTS))
    if name == 'nemo3s_nv':
        # NVFP4-pretrained Nemotron-3-Super (LatentMoE, relu2: up_proj [2688,1024] + down_proj [1024,2688] per expert)
        pat = ('MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME')
        moe = [i for i, ch in enumerate(pat) if ch == 'E']
        rng = random.Random(88)
        pairs = [(L, rng.randrange(512)) for L in moe[::1]][:40]
        return ('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4', 'ff433f5493e25d631c9f12b5d55c674229923d02',
                mk_items(pairs, 'backbone.layers.{L}.mixer.experts.{E}.', (('w1', 'up_proj'), ('w2', 'down_proj')), MO_PARTS))
    raise SystemExit('unknown family ' + name)


def main():
    name, index_json, out, max_mb = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    repo, rev, items = fam(name)
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
                time.sleep(2 + 3 * t)

    hdrs = {}

    def shard_header(fname):
        if fname not in hdrs:
            n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
            hdrs[fname] = (8 + n, json.loads(get_range(fname, 8, 8 + n - 1)))
        return hdrs[fname]

    os.makedirs(out, exist_ok=True)
    wm = json.load(open(index_json))['weight_map']
    total = 0
    for it in items:
        for t in it['tensors']:
            f = wm[t['name']]
            b0, hdr = shard_header(f)
            m = hdr[t['name']]
            a, b = m['data_offsets']
            t.update({'dtype': m['dtype'], 'shape': list(m['shape']), 'bytes': b - a, 'shard': f, 'offset': b0 + a})
            total += b - a
    print(name, repo, 'items', len(items), 'planned bytes', total, flush=True)
    for t in items[0]['tensors']:
        print(' ', t['name'], t['dtype'], t['shape'], t['bytes'])
    if total > max_mb * 1e6:
        print('ABORT: over budget')
        sys.exit(1)

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
    json.dump({'family': name, 'repo': repo, 'revision': rev, 'downloaded_bytes': total, 'experts': res},
              open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(name, len(res), 'items', total, 'bytes', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
