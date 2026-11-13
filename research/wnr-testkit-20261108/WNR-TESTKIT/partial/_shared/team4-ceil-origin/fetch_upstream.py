"""Range-download a few routed experts of the UPSTREAM checkpoint deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1.

Usage: python -I fetch_upstream.py SRC_DIR OUT_DIR MAX_MB
SRC_DIR holds model.safetensors.index.json (upstream) and reap__reap_keepset.json (REAP repo), already downloaded.
Picks: 4 experts that REAP kept and that are in the team's v41 sample (to check the 'bit-exact copy' claim) and
8 experts that REAP pruned (never in the REAP repo, so any fingerprint there predates REAP), plus 1 MTP expert.
Writes OUT_DIR/U{tag}.bin with tensors w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight back to back
and OUT_DIR/manifest.json. Aborts before fetching if the total would exceed MAX_MB. Nothing downloaded is executed.
"""
import json, os, random, struct, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REV = '1bae0e1088993b9b82de08f59de5be5a34e42170'
BASE = f'https://huggingface.co/{REPO}/resolve/{REV}/'
UA = {'User-Agent': 'team4-ceil-origin/1.0'}
ORDER = ['w1.scale', 'w2.scale', 'w3.scale', 'w1.weight', 'w2.weight', 'w3.weight']


def get_range(fname, start, end_incl, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(BASE + fname, headers={**UA, 'Range': f'bytes={start}-{end_incl}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != end_incl - start + 1:
                raise IOError(f'short read {len(data)} != {end_incl - start + 1}')
            return data
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 + 3 * t)


_hdr = {}


def shard_header(fname):
    if fname not in _hdr:
        n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
        _hdr[fname] = (8 + n, json.loads(get_range(fname, 8, 8 + n - 1)))
    return _hdr[fname]


def main():
    src, out, max_mb = sys.argv[1], sys.argv[2], float(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    wm = json.load(open(os.path.join(src, 'model.safetensors.index.json')))['weight_map']
    keep = json.load(open(os.path.join(src, 'reap__reap_keepset.json')))['keep']
    picks = []  # (tag, prefix, info)
    for L, new in [(0, 97), (13, 18), (26, 2), (39, 32)]:  # in the v41 sample
        up = keep[str(L)][new]
        picks.append((f'K_L{L:02d}_N{new:03d}_U{up:03d}', f'layers.{L}.ffn.experts.{up}.',
                      {'kind': 'kept', 'layer': L, 'reap_index': new, 'upstream_index': up,
                       'sample_file': f'L{L:02d}_E{new:03d}.bin'}))
    rng = random.Random(20261107)
    for L in [0, 5, 10, 15, 20, 25, 30, 39]:
        pruned = sorted(set(range(384)) - set(keep[str(L)]))
        up = rng.choice(pruned)
        picks.append((f'P_L{L:02d}_U{up:03d}', f'layers.{L}.ffn.experts.{up}.',
                      {'kind': 'pruned', 'layer': L, 'upstream_index': up}))
    picks.append(('M_mtp0_U000', 'mtp.0.ffn.experts.0.', {'kind': 'mtp', 'layer': 'mtp.0', 'upstream_index': 0}))
    for f in sorted({wm[p + 'w1.weight'] for _, p, _ in picks}):
        shard_header(f)
    plan = []
    total = 0
    for tag, p, info in picks:
        rows = []
        for t in ORDER:
            f = wm[p + t]
            base, hdr = _hdr[f] if f in _hdr else shard_header(f)
            m = hdr[p + t]
            a, b = m['data_offsets']
            rows.append({'name': t, 'dtype': m['dtype'], 'shape': m['shape'], 'bytes': b - a, 'shard': f, 'offset': base + a})
            total += b - a
        plan.append((tag, info, rows))
    print('planned bytes', total, flush=True)
    if total > max_mb * 1e6:
        print('ABORT: over budget'); sys.exit(1)

    def fetch(item):
        tag, info, rows = item
        path = os.path.join(out, f'U{tag}.bin')
        if not os.path.exists(path):
            data = b''.join(get_range(r['shard'], r['offset'], r['offset'] + r['bytes'] - 1) for r in rows)
            with open(path + '.part', 'wb') as fh:
                fh.write(data)
            os.replace(path + '.part', path)
        return {**info, 'file': os.path.basename(path), 'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
                'tensors': rows}

    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, plan))
    json.dump({'repo': REPO, 'revision': REV, 'experts': res}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(len(res), 'experts', total, 'bytes', round(time.time() - t0, 1), 's')
    for r in res:
        print(r['file'], [(t['name'], t['dtype'], t['shape']) for t in r['tensors']][:4])


if __name__ == '__main__':
    main()
