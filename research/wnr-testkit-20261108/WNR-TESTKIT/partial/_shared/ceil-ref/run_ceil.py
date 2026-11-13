"""Encode real V4.1 experts into full records (WNR2 and ceil-form combinations), decode them back, verify, time.
team4 / ceil-ref, adapted from team 3 run_records.py.

Usage: python -I run_ceil.py SAMPLE_DIR MODELS_NPZ OUT_JSONL EXPERT_INDICES CONFIGS
One JSON line per (expert, config). Verification: decoded slab byte-equal to the source .bin
(canonical-zero configs: equal after mapping every FP4 nibble 8 -> 0, i.e. value-exact).
Every source file is first checked against the manifest sha256 of its tensors.
"""
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
from wnr_common import Config

FG = dict(f=True, g=True)
CONFIGS = {c.name: c for c in [
    Config('wnr2', wnr2=True),
    Config('f', f=True),
    Config('fg', **FG),
    Config('fgd', d=True, **FG),
    Config('fge', e=True, **FG),
    Config('fgde', d=True, e=True, **FG),
    Config('fgt', t=True, **FG),
    Config('fgdt', d=True, t=True, **FG),
    Config('fga', a=True, **FG),
    Config('fgad', a=True, d=True, **FG),
    Config('fgaK2', a=True, K=2, **FG),
    Config('fgaK2d', a=True, K=2, d=True, **FG),
    Config('fgc', c=True, **FG),
    Config('fgcd', c=True, d=True, **FG),
    Config('fgce', c=True, e=True, **FG),
    Config('fgcde', c=True, d=True, e=True, **FG),
    Config('fgaK2ce', a=True, K=2, c=True, e=True, **FG),
    Config('fgaK2cde', a=True, K=2, c=True, d=True, e=True, **FG),
    Config('fgct', c=True, t=True, **FG),
    Config('fgcdt', c=True, d=True, t=True, **FG),
]}
# D changes only the length table: these records are built from the payload of their sibling (same model, same x0)
D_SIBLING = {'fg': 'fgd', 'fge': 'fgde', 'fgt': 'fgdt', 'fga': 'fgad', 'fgaK2': 'fgaK2d', 'fgc': 'fgcd',
             'fgce': 'fgcde', 'fgaK2ce': 'fgaK2cde', 'fgct': 'fgcdt'}


def canon_bytes(raw, nscale_bytes):
    out = raw.copy()
    w = out[nscale_bytes:]
    lo = w & 15
    hi = w >> 4
    lo = np.where(lo == 8, 0, lo)
    hi = np.where(hi == 8, 0, hi)
    out[nscale_bytes:] = (lo | (hi << 4)).astype(np.uint8)
    return out


def main():
    sample_dir, models_npz, out_jsonl, idxs, cfgs = sys.argv[1:6]
    idxs = [int(x) for x in idxs.split(',')]
    cfgs = [CONFIGS[c] for c in cfgs.split(',')]
    man = W.load_manifest(sample_dir)
    Z = np.load(models_npz)
    cache = {}

    def tables(cfg):
        key = (cfg.model_key(), cfg.wnr2)
        if key not in cache:
            k = cfg.model_key()
            sf = Z[k + '_sf'] if (k + '_sf') in Z.files else None
            if cfg.wnr2:
                cache[key] = W.Tables(Z[k + '_wf'], Z[k + '_scf'])
            else:
                cache[key] = W.Tables(Z[k + '_wf'], Z[k + '_scf'], sf, Z[k + '_etab'], K=cfg.K,
                                      nctx=W.NACTX if cfg.a else W.NCTX)
        return cache[key]

    fo = open(out_jsonl, 'a')
    for i in idxs:
        e = man['experts'][i]
        t0 = time.perf_counter()
        raw, mats = W.load_expert(sample_dir, e)
        tload = time.perf_counter() - t0
        off, sha_ok = 0, True
        for d in e['tensors']:
            if 'sha256' in d:
                sha_ok &= hashlib.sha256(raw[off:off + d['bytes']].tobytes()).hexdigest() == d['sha256']
            off += d['bytes']
        shapes = [(mt['R'], mt['C']) for mt in mats]
        nsb = sum(mt['sc'].size for mt in mats)
        craw = canon_bytes(raw, nsb)
        work = []
        for cfg in cfgs:
            work.append((cfg, None))
            if cfg.name in D_SIBLING and D_SIBLING[cfg.name] in CONFIGS:
                work.append((CONFIGS[D_SIBLING[cfg.name]], cfg))
        st = None
        for cfg, parent in work:
            tab = tables(cfg)
            tm = {}
            if parent is None:
                rec, info, st, base_info = W.encode_record(mats, tab, cfg, e['layer'], e['expert'], tm)
                enc_tm = dict(tm)
            else:
                assert parent.model_key() == cfg.model_key() and parent.e == cfg.e and parent.t == cfg.t
                rec, info = W.build_record(st, tab, cfg, base_info)
                tm.update(enc_tm)
                tm['enc_total_s'] = enc_tm['enc_total_s'] + info['build_s']
                tm['enc_note'] = f'payload shared with {parent.name}; enc_total = its encode + this length-table build'
            try:
                slab, dinfo = W.decode_record(rec, tab, e['layer'], e['expert'], shapes, tm)
                derr = None
            except W.DecodeError as ex:
                slab, derr = np.zeros(0, np.uint8), str(ex)
            ref = craw if cfg.c else raw
            ci = info.get('ceil')
            row = {'idx': i, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'models': os.path.basename(models_npz),
                   'config': cfg.name, 'flags': cfg.flags, 'rec_flags': info['rec_flags'], 'K': cfg.K, 'ks': info['ks'],
                   'anchors': info['meds'], 'model_id': tab.model_id, 'model_blob_bytes': len(tab.blob),
                   'ceil': None if ci is None else {'applied': ci['applied'], 'fallback': not ci['applied'],
                                                    'even_blocks': ci['even'], 'blocks': ci['blocks'],
                                                    'below6': ci['below6'], 'overflow': ci['overflow']},
                   'source_sha256_ok': bool(sha_ok),
                   'sizes': {'header': info['header'], 'length_table': info['length_table'], 'scale_payload': info['scale_payload'],
                             'weight_payload': info['weight_payload'], 'used': info['used'], 'padding': info['padding'],
                             'total': info['total'], 'record_bytes_check': len(rec)},
                   'est': {'weight_bits': info['est_weight_bits'], 'side_bits': info['est_side_bits'], 'scale_bits': info['est_scale_bits'],
                           'est_weight_payload': info['est_weight_bits'] / 8 + info['est_side_bits'] / 8 + 4 * info['n_weight_streams'],
                           'est_scale_payload': info['est_scale_bits'] / 8 + 4 * info['n_scale_streams']},
                   'verify': {'decode_error': derr,
                              'equal_to_reference': bool(slab.size == ref.size and (slab == ref).all()),
                              'byte_exact_vs_source': bool(slab.size == raw.size and (slab == raw).all()),
                              'bytes_differing_from_source': int((slab != raw).sum()) if slab.size == raw.size else -1,
                              'mode': 'value-exact (8->0)' if cfg.c else 'byte-exact'},
                   'timing': {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in tm.items()}, 'load_s': round(tload, 3)},
                   'payload_parent': None if parent is None else parent.name,
                   'len_resid_std': [info.get('len_resid_std_s'), info.get('len_resid_std_w')], 'len_bits': info.get('len_bits'),
                   'max_weight_stream': int(info['wlens'].max()), 'max_scale_stream': int(info['slens'].max()),
                   'mean_weight_stream': float(info['wlens'].mean()), 'mean_scale_stream': float(info['slens'].mean())}
            fo.write(json.dumps(row) + '\n')
            fo.flush()
            print(e['file'], cfg.name, row['sizes']['total'], row['verify']['equal_to_reference'], derr,
                  f"enc {tm['enc_total_s']:.2f}s dec {tm.get('dec_total_s', -1):.2f}s", flush=True)
            del rec, slab
        st = base_info = None
    fo.close()


if __name__ == '__main__':
    main()
