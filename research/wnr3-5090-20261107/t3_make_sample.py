"""Rebuild the research-PC V4.1 sample (160 experts, fetch_experts.py layout) from the LOCAL REAP safetensors and
verify every file against the archived manifest sha256 (<RESEARCH-PC>/v41-sample/manifest.json).
Also writes COUNTS_DIR/<name>.npz {w_base} with the KAIRA C trainer (wex_nr2_count, lower-median anchor)
through g0codec.dll, an independent cross-check of the team-1 mxceil.py w_orig counts used by ceilenc.py.

    python -I t3_make_sample.py SAMPLE_DIR COUNTS_DIR
"""
import hashlib, json, os, shutil, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'g0'))
import t2_g0 as G   # noqa: E402  (Shards, g0codec.dll bindings)

MAN = r'<RESEARCH-PC>\v41-sample\manifest.json'
ORDER = ['w1.scale', 'w2.scale', 'w3.scale', 'w1.weight', 'w2.weight', 'w3.weight']


def main(sample_dir, counts_dir):
    os.makedirs(sample_dir, exist_ok=True); os.makedirs(counts_dir, exist_ok=True)
    man = json.load(open(MAN))
    shutil.copy(MAN, os.path.join(sample_dir, 'manifest.json'))
    sh = G.Shards()
    ok = 0
    for e in man['experts']:
        L, x = e['layer'], e['expert']
        parts = []
        for t, row in zip(ORDER, e['tensors']):
            assert row['name'] == t
            a, shape, dtype = sh.tensor(f'layers.{L}.ffn.experts.{x}.{t}')
            assert list(shape) == row['shape'] and dtype == row['dtype'] and a.size == row['bytes'], (L, x, t)
            parts.append(a)
        data = b''.join(p.tobytes() for p in parts)
        h = hashlib.sha256(data).hexdigest()
        assert h == e['sha256'], (e['file'], h)
        ok += 1
        path = os.path.join(sample_dir, e['file'])
        if not os.path.exists(path):
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.replace(path + '.part', path)
        sc, wt = parts[:3], parts[3:]
        a, b = G.layout(sc, wt)
        wc = np.zeros(3 * 567 * 16, np.uint64); scn = np.zeros(3 * 256, np.uint64)
        G.lib.g0_set_anchor(-1)
        G.lib.g0_count(G.ptr(a), G.ptr(b), G.ptrs(sc), G.ptrs(wt), G.ptr(wc), G.ptr(scn))
        np.savez(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')),
                 w_base=wc.reshape(3, 567, 16).astype(np.int64), s_o0=scn.reshape(3, 256).astype(np.int64))
    print(json.dumps({'experts': len(man['experts']), 'sha256_ok': ok, 'revision': man['revision']}))


if __name__ == '__main__':
    main(*sys.argv[1:])
