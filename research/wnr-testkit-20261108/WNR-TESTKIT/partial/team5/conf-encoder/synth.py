"""team5 / conf-encoder: T13(b) encoder half - extra records with ceil_mask in {0, 1, 2, 4} (the 40 real records all have
ceil_mask 7). Synthetic sources derived from one real expert (default L00_E113): a matrix is forced to fail the gate by
rewriting ONE block to max magnitude 5 (magnitudes clamped to <= 5, nibble 0 magnitude set to 5, signs kept); every
gated matrix also gets the edge blocks of S:313-321 / T1: one block with max 3 (pass-through), one even block with
scale 254 (coded 255) and one even block with scale 255 (skipped). The modified sources are written next to the
records (they are the byte-exact reference for these records).
Usage: python -I synth.py SAMPLE_DIR RECORDS_DIR [IDX]
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr3c as W


def to_raw(mats, e):
    parts = {}
    for w, mt in zip(('w1', 'w2', 'w3'), mats):
        s = mt['sym'].reshape(mt['R'], mt['C'])
        parts[w + '.weight'] = (s[:, 0::2] | (s[:, 1::2] << 4)).astype(np.uint8).ravel()
        parts[w + '.scale'] = mt['sc']
    return np.concatenate([parts[d['name']] for d in e['tensors']]).tobytes()


def main():
    sample_dir, rdir = sys.argv[1:3]
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    blob = open(os.path.join(rdir, 'model.wnm3'), 'rb').read()
    model = W.parse_wnm3(blob)
    e = man['experts'][idx]
    sdir = os.path.join(rdir, 'synthetic')
    os.makedirs(sdir, exist_ok=True)
    rows = []
    for mask in (0, 1, 2, 4):
        raw, mats = W.load_expert(os.path.join(sample_dir, e['file']), e)
        edits = []
        for m, mt in enumerate(mats):
            blk = mt['sym'].reshape(-1, 32)
            sc = mt['sc']
            M = (blk & 7).max(1)
            even = (M == 7) & ~(((blk & 7) == 1) | ((blk & 7) == 3)).any(1)
            if not (mask >> m) & 1:                       # force gate failure: one max-5 block
                b = 1000
                mag = np.minimum(blk[b] & 7, 5)
                mag[0] = 5
                blk[b] = (blk[b] & 8) | mag
                edits.append({'m': m, 'block': b, 'edit': 'max-5 block (gate fails)'})
            else:                                         # gated matrix: edge blocks
                b3 = 2000
                mag = np.minimum(blk[b3] & 7, 3)
                mag[0] = 3
                blk[b3] = (blk[b3] & 8) | mag
                ev = np.flatnonzero(even)
                ev = ev[ev > 3000]
                sc[ev[0]] = 254
                sc[ev[1]] = 255
                edits.append({'m': m, 'block_max3': b3, 'even_block_scale254': int(ev[0]), 'even_block_scale255': int(ev[1])})
            mt['sym'] = blk.ravel()
        src = to_raw(mats, e)
        assert len(src) == raw.size
        sname = e['file'].replace('.bin', f'_mask{mask}.src.bin')
        open(os.path.join(sdir, sname), 'wb').write(src)
        rec, rep = W.encode_record(mats, model, model['model_id'], e['layer'], e['expert'])
        shapes = [(mt['R'], mt['C']) for mt in mats]
        slab, dinfo = W.decode_record(rec, model, rep['entry_flags'], (e['layer'], e['expert']), shapes)
        ok = slab == src
        rname = e['file'].replace('.bin', f'_mask{mask}.wnr3')
        open(os.path.join(sdir, rname), 'wb').write(rec)
        row = {'base_source': e['file'], 'ceil_mask_wanted': mask, 'ceil_mask': rep['ceil_mask'], 'record_flags': rep['flags'],
               'entry_flags_hex': f"0x{rep['entry_flags']:04X}", 'entry_flags': rep['entry_flags'], 'set_id': rep['set_id'],
               'anchor': rep['anchor'], 'bind': [e['layer'], e['expert']], 'shapes_symbols': shapes, 'edits': edits,
               'source_file': sname, 'source_sha256': hashlib.sha256(src).hexdigest(),
               'record_file': rname, 'record_bytes': len(rec), 'record_sha256': hashlib.sha256(rec).hexdigest(),
               'expected_decoded_sha256': hashlib.sha256(slab).hexdigest(), 'decoded_equals_source': bool(ok),
               'matrix_stats': rep['matrix_stats'], 'sizes': rep['sizes']}
        rows.append(row)
        print(mask, rep['ceil_mask'], hex(rep['flags']), rep['set_id'], len(rec), ok, flush=True)
        assert ok and rep['ceil_mask'] == mask
    json.dump({'model_file': '../model.wnm3', 'records': rows}, open(os.path.join(sdir, 'synthetic_index.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
