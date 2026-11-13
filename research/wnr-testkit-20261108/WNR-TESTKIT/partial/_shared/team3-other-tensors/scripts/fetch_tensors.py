"""Range-download whole tensors or row slices into OUT_DIR (one .bin per item) + manifest.json.

Usage: python -I fetch_tensors.py TENSORS_JSON OUT_DIR SPEC_JSON
SPEC_JSON: list of {"name": str, "rows": [r0, r1] (optional, half-open), "tag": str (optional)}
Only bytes are fetched; nothing from the repository is executed.
"""
import json, os, sys, hashlib, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf

ITEM = {'F8_E4M3': 1, 'F8_E8M0': 1, 'I8': 1, 'U8': 1, 'BF16': 2, 'F16': 2, 'F32': 4}


def main():
    tens = {t['name']: t for t in json.load(open(sys.argv[1], encoding='utf-8'))}
    out = sys.argv[2]
    spec = json.load(open(sys.argv[3], encoding='utf-8'))
    os.makedirs(out, exist_ok=True)

    def one(s):
        t = tens[s['name']]
        shape = t['shape']
        row_bytes = t['bytes'] // shape[0]
        r0, r1 = s.get('rows', [0, shape[0]])
        start = t['offset'] + r0 * row_bytes
        n = (r1 - r0) * row_bytes
        tag = s.get('tag') or (s['name'] + ('' if 'rows' not in s else f'.r{r0}-{r1}'))
        path = os.path.join(out, tag + '.bin')
        if not os.path.exists(path):
            CH = 64 << 20
            parts = []
            for a in range(0, n, CH):
                b = min(n, a + CH)
                parts.append(hf.get_range(t['file'], start + a, start + b - 1))
            data = b''.join(parts)
            open(path + '.part', 'wb').write(data)
            os.replace(path + '.part', path)
        data = open(path, 'rb').read()
        assert len(data) == n
        return {'tag': tag, 'name': s['name'], 'file': t['file'], 'dtype': t['dtype'],
                'shape': [r1 - r0] + shape[1:], 'full_shape': shape, 'rows': [r0, r1],
                'bytes': n, 'abs_offset': start, 'sha256': hashlib.sha256(data).hexdigest()}

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, spec))
    man = {'repo': hf.REPO, 'revision': hf.REV, 'engram_repo': hf.BASE_REPO, 'engram_revision': hf.BASE_REV,
           'items': res}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8'), indent=1)
    print(len(res), 'items', sum(r['bytes'] for r in res), 'bytes', f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
