"""team6 / cpu-simd: produce WNR2 and f+g0+e (ceil-form, anchor 120, 23-bit FNV/lowbias32 hash in x0) test records
with a COPY of team4 ceil-ref's coder (src/wnr_common.py, copied 2026-11-07, unmodified) and model even.npz
(trained on the 80 even-indexed experts). Experts: odd-indexed (held-out).
Usage: python -I gen_records.py SAMPLE_DIR OUT_DIR IDX,IDX,...
Writes OUT_DIR/<file>.<cfg>.rec, OUT_DIR/models/<key>.wf.u16 / .scf.u16, OUT_DIR/records.json"""
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
from wnr_common import Config

A0 = 120
W.anchor_pair = lambda sc: A0            # g0: constant anchor (as team4 run_fixed_anchor.py)

CFGS = [Config('wnr2', wnr2=True), Config('fge', f=True, g=True, e=True)]


def main():
    sample_dir, out_dir, idxs = sys.argv[1:4]
    idxs = [int(x) for x in idxs.split(',')]
    here = os.path.dirname(os.path.abspath(__file__))
    Z = np.load(os.path.join(here, 'even.npz'))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    tabs = {}
    for cfg in CFGS:
        k = cfg.model_key()
        if cfg.wnr2:
            tabs[cfg.name] = W.Tables(Z[k + '_wf'], Z[k + '_scf'])
        else:
            tabs[cfg.name] = W.Tables(Z[k + '_wf'], Z[k + '_scf'], None, Z[k + '_etab'], K=0, nctx=W.NCTX)
        t = tabs[cfg.name]
        t.wf.astype('<u2').tofile(os.path.join(out_dir, 'models', cfg.name + '.wf.u16'))
        t.scf.astype('<u2').tofile(os.path.join(out_dir, 'models', cfg.name + '.scf.u16'))
    man = W.load_manifest(sample_dir)
    meta_path = os.path.join(out_dir, 'records.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {'records': []}
    for i in idxs:
        e = man['experts'][i]
        raw, mats = W.load_expert(sample_dir, e)
        sha_ok = hashlib.sha256(raw.tobytes()).hexdigest() == e['sha256']
        shapes = [(mt['R'], mt['C']) for mt in mats]
        for cfg in CFGS:
            tab = tabs[cfg.name]
            t0 = time.perf_counter()
            rec, info, st, bi = W.encode_record(mats, tab, cfg, e['layer'], e['expert'])
            t1 = time.perf_counter()
            slab, _ = W.decode_record(rec, tab, e['layer'], e['expert'], shapes)
            t2 = time.perf_counter()
            ok = bool(slab.size == raw.size and (slab == raw).all())
            fn = f"{e['file'][:-4]}.{cfg.name}.rec"
            open(os.path.join(out_dir, fn), 'wb').write(rec)
            row = {'idx': i, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'config': cfg.name,
                   'rec': fn, 'rec_bytes': len(rec), 'raw_bytes': int(raw.size), 'ratio': len(rec) / raw.size,
                   'model_id': tab.model_id, 'rec_flags': info['rec_flags'], 'anchors': info['meds'],
                   'shapes': shapes, 'ceil_applied': None if info['ceil'] is None else info['ceil']['applied'],
                   'source_sha256_ok': sha_ok, 'py_decode_byte_exact': ok,
                   'py_enc_s': round(t1 - t0, 3), 'py_dec_s': round(t2 - t1, 3)}
            meta['records'] = [r for r in meta['records'] if r['rec'] != fn] + [row]
            print(json.dumps(row), flush=True)
    meta['models'] = {c.name: {'model_id': tabs[c.name].model_id, 'wf_rows': int(tabs[c.name].wf.shape[0])} for c in CFGS}
    json.dump(meta, open(meta_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
