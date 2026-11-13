"""Range-download extra V4.1 REAP-192 MXFP4 experts NOT in the 160-expert sample (adapted from fetch_experts.py,
same pinned repo/revision). Also a few MTP routed experts (mtp.{0,1,2}.ffn.experts.N).

Usage: python -I extra_fetch.py OUT_DIR SAMPLE_MANIFEST INDEX_JSON [--dry]
Writes OUT_DIR/L{LL}_E{EEE}.bin or M{L}_E{EEE}.bin (w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight) and
OUT_DIR/manifest.json. Aborts before downloading tensor bytes if the planned total exceeds BUDGET.
Only tensor bytes are fetched (HTTP Range); nothing is executed.
"""
import json, os, random, struct, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = 'patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
REV = '4ab649b2254afdcf6c2f7c1fbea1c05eda11828c'
BASE = f'https://huggingface.co/{REPO}/resolve/{REV}/'
UA = {'User-Agent': 'wnr2-research-fetch/1.0'}
BUDGET = 1_450_000_000  # tensor bytes; headers come on top (small)
SEED = 20261108


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
header_bytes = [0]


def shard_header(fname):
    if fname not in _hdr:
        n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
        hdr = json.loads(get_range(fname, 8, 8 + n - 1))
        header_bytes[0] += 8 + n
        _hdr[fname] = (8 + n, hdr)
    return _hdr[fname]


def main():
    out, man_path, ix_path = sys.argv[1:4]
    dry = '--dry' in sys.argv
    os.makedirs(out, exist_ok=True)
    wm = json.load(open(ix_path))['weight_map']
    sample = json.load(open(man_path))
    taken = {(e['layer'], e['expert']) for e in sample['experts']}
    rng = random.Random(SEED)
    picks = []
    for L in range(40):
        k = 1 if L % 4 == 3 else 2  # 30 layers x 2 + 10 layers x 1 = 70 routed
        pool = [e for e in range(192) if (L, e) not in taken]
        picks += [('layers', L, e) for e in sorted(rng.sample(pool, k))]
    for L in range(3):
        picks += [('mtp', L, e) for e in sorted(rng.sample(range(128), 2))]
    assert not any((L, e) in taken for kind, L, e in picks if kind == 'layers')
    order = ['w1.scale', 'w2.scale', 'w3.scale', 'w1.weight', 'w2.weight', 'w3.weight']
    for f in sorted({wm[f'{k}.{L}.ffn.experts.{e}.w1.weight'] for k, L, e in picks}):
        shard_header(f)
    plan = []
    total = 0
    for kind, L, e in picks:
        rows = []
        for t in order:
            name = f'{kind}.{L}.ffn.experts.{e}.{t}'
            f = wm[name]
            base, hdr = shard_header(f)
            m = hdr[name]
            a, b = m['data_offsets']
            rows.append({'name': t, 'dtype': m['dtype'], 'shape': m['shape'], 'bytes': b - a, 'shard': f, 'offset': base + a})
            total += b - a
        tag = 'L' if kind == 'layers' else 'M'
        plan.append((kind, L, e, f'{tag}{L:02d}_E{e:03d}.bin', rows))
    print('picks', len(plan), 'tensor bytes', total, 'header bytes', header_bytes[0], flush=True)
    print('first', json.dumps(plan[0][4]), flush=True)
    print('mtp', json.dumps(plan[-1][4]), flush=True)
    if total > BUDGET:
        raise SystemExit(f'over budget {total} > {BUDGET}')
    if dry:
        return

    def fetch(p):
        kind, L, e, fn, rows = p
        path = os.path.join(out, fn)
        if not os.path.exists(path):
            blobs = [get_range(r['shard'], r['offset'], r['offset'] + r['bytes'] - 1) for r in rows]
            with open(path + '.part', 'wb') as fh:
                fh.write(b''.join(blobs))
            os.replace(path + '.part', path)
        h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        return {'kind': kind, 'layer': L, 'expert': e, 'file': fn, 'sha256': h, 'tensors': rows}

    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, plan))
    man = {'repo': REPO, 'revision': REV, 'seed': SEED, 'excluded_sample': man_path, 'experts': res,
           'tensor_bytes': total, 'header_bytes': header_bytes[0]}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(f'{len(res)} experts, {total} bytes, {time.time() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main()
