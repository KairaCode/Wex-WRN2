"""Encode real V4.1 experts into full records (WNR2 and WNR3 feature combinations), decode them back, verify, time.

Usage: python -I run_records.py SAMPLE_DIR MODELS_NPZ OUT_JSONL EXPERT_INDICES CONFIGS [CORRUPT_EXPERTS]
  EXPERT_INDICES: comma list of manifest indices; CONFIGS: comma list of names below;
  CORRUPT_EXPERTS: comma list of manifest indices on which corruption tests are run (optional).
One JSON line per (expert, config). Verification: decoded slab byte-equal to the source .bin
(canonical-zero configs: equal after mapping every FP4 nibble 8 -> 0, i.e. value-exact).
"""
import json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
from wnr_common import Config

CONFIGS = {c.name: c for c in [
    Config('wnr2', wnr2=True),
    Config('w3', ),
    Config('d', d=True),
    Config('e', e=True),
    Config('a', a=True),
    Config('aK2', a=True, K=2),
    Config('aK4', a=True, K=4),
    Config('c', c=True),
    Config('aK4c', a=True, K=4, c=True),
    Config('aK2c', a=True, K=2, c=True),
    Config('aK4cd', a=True, K=4, c=True, d=True),
    Config('aK4ce', a=True, K=4, c=True, e=True),
    Config('aK4cde', a=True, K=4, c=True, d=True, e=True),
    Config('aK4d', a=True, K=4, d=True),
    Config('aK4de', a=True, K=4, d=True, e=True),
    Config('de', d=True, e=True),
    Config('ad', a=True, d=True),
    Config('cd', c=True, d=True),
]}
# D changes only the length table: these records are built from the payload of their sibling (same model)
D_SIBLING = {'w3': 'd', 'e': 'de', 'a': 'ad', 'aK4': 'aK4d', 'c': 'cd', 'aK4c': 'aK4cd', 'aK4ce': 'aK4cde', 'aK4e': 'aK4de'}


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
    corrupt = set(int(x) for x in sys.argv[6].split(',')) if len(sys.argv) > 6 and sys.argv[6] else set()
    idxs = [int(x) for x in idxs.split(',')]
    cfgs = [CONFIGS[c] for c in cfgs.split(',')]
    man = W.load_manifest(sample_dir)
    Z = np.load(models_npz)
    cache = {}

    def tables(cfg):
        key = (cfg.model_key(), cfg.wnr2)
        if key not in cache:
            while len(cache) >= 2:          # keep RAM low (decoder LUTs are 28-149 MB each)
                cache.pop(next(iter(cache)))
            k = cfg.model_key()
            sf = Z[k + '_sf'] if (k + '_sf') in Z.files else None
            if cfg.wnr2:
                cache[key] = W.Tables(Z[k + '_wf'], Z[k + '_scf'])
            else:
                cache[key] = W.Tables(Z[k + '_wf'], Z[k + '_scf'], sf, Z[k + '_etab'], K=cfg.K,
                                      nctx=W.NACTX if cfg.a else W.NCTX)
        return cache[key]

    rng = random.Random(20261107)
    fo = open(out_jsonl, 'a')
    for i in idxs:
        e = man['experts'][i]
        t0 = time.perf_counter()
        raw, mats = W.load_expert(sample_dir, e)
        tload = time.perf_counter() - t0
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
                assert parent.model_key() == cfg.model_key() and parent.e == cfg.e
                rec, info = W.build_record(st, tab, cfg, base_info)
                tm.update(enc_tm)
                tm['enc_total_s'] = enc_tm['enc_total_s'] + info['build_s']
                tm['enc_note'] = f'payload shared with {parent.name}; enc_total = its encode + this length-table build'
            slab, dinfo = W.decode_record(rec, tab, e['layer'], e['expert'], shapes, tm)
            ref = craw if cfg.c else raw
            row = {'idx': i, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'models': os.path.basename(models_npz),
                   'config': cfg.name, 'flags': cfg.flags, 'K': cfg.K, 'ks': info['ks'], 'model_id': tab.model_id,
                   'model_blob_bytes': len(tab.blob),
                   'sizes': {'header': info['header'], 'length_table': info['length_table'], 'scale_payload': info['scale_payload'],
                             'weight_payload': info['weight_payload'], 'used': info['used'], 'padding': info['padding'],
                             'total': info['total'], 'record_bytes_check': len(rec)},
                   'est': {'weight_bits': info['est_weight_bits'], 'side_bits': info['est_side_bits'], 'scale_bits': info['est_scale_bits'],
                           'est_weight_payload': info['est_weight_bits'] / 8 + info['est_side_bits'] / 8 + 4 * info['n_weight_streams'],
                           'est_scale_payload': info['est_scale_bits'] / 8 + 4 * info['n_scale_streams']},
                   'verify': {'equal_to_reference': bool(slab.size == ref.size and (slab == ref).all()),
                              'byte_exact_vs_source': bool(slab.size == raw.size and (slab == raw).all()),
                              'bytes_differing_from_source': int((slab != raw).sum()) if slab.size == raw.size else -1,
                              'mode': 'value-exact (8->0)' if cfg.c else 'byte-exact'},
                   'timing': {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in tm.items()}, 'load_s': round(tload, 3)},
                   'payload_parent': None if parent is None else parent.name,
                   'len_resid_std': [info.get('len_resid_std_s'), info.get('len_resid_std_w')], 'len_bits': info.get('len_bits'),
                   'max_weight_stream': int(info['wlens'].max()), 'max_scale_stream': int(info['slens'].max())}
            if i in corrupt and cfg.name in ('wnr2', 'd', 'e', 'aK4cde'):
                tests = {}
                hdr = row['sizes']['header'] + row['sizes']['length_table']
                for t in range(3):
                    for region, lo, hi in (('scale_payload', hdr, hdr + info['scale_payload']),
                                           ('weight_payload', hdr + info['scale_payload'], info['used']),
                                           ('length_table', row['sizes']['header'], hdr)):
                        b = bytearray(rec)
                        j = rng.randrange(lo, hi)
                        b[j] ^= 1 << rng.randrange(8)
                        try:
                            W.decode_record(bytes(b), tab, e['layer'], e['expert'], shapes)
                            res = 'ACCEPTED'
                        except W.DecodeError as ex:
                            res = 'rejected: ' + str(ex)
                        except Exception as ex:      # e.g. index errors on absurd lengths = also a rejection
                            res = 'rejected (exception): ' + type(ex).__name__
                        tests[f'{region}#{t}'] = res
                try:
                    W.decode_record(rec, tab, e['layer'], e['expert'] + 1, shapes)
                    tests['valid record, wrong expert id'] = 'ACCEPTED'
                except W.DecodeError as ex:
                    tests['valid record, wrong expert id'] = 'rejected: ' + str(ex)
                row['corruption'] = tests
            fo.write(json.dumps(row) + '\n')
            fo.flush()
            print(e['file'], cfg.name, row['sizes']['total'], row['verify']['equal_to_reference'],
                  f"enc {tm['enc_total_s']:.2f}s dec {tm['dec_total_s']:.2f}s", flush=True)
            del rec, slab
        st = base_info = None
    fo.close()


if __name__ == '__main__':
    main()
