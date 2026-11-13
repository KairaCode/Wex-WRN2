"""team5 / conf-encoder: H32 function vectors (D1:285, S:303) and regenerated record-level vectors (T5) from the first
real record, recomputed with the scalar H32 from the decoded coded bytes.
Usage: python -I vectors.py RECORDS_DIR OUT_JSON"""
import json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr3c as W


def main():
    rdir, out = sys.argv[1:3]
    idx = json.load(open(os.path.join(rdir, 'index.json')))
    row = idx['records'][0]
    blob = open(os.path.join(rdir, idx['model_file']), 'rb').read()
    model = W.parse_wnm3(blob)
    rec = open(os.path.join(rdir, row['record_file']), 'rb').read()
    slab, info = W.decode_record(rec, model, row['entry_flags'], tuple(row['bind']), [tuple(s) for s in row['shapes_symbols']])
    H = W.unpack_header(rec[:64])
    v = {'smhasher_verification': f'0x{W.smhasher_verification():08X}', 'h32("",0)': f'0x{W.h32(b"", 0):08X}',
         'h32("",seed)': f'0x{W.h32(b"", W.SEED):08X}', 'h32(00..0F,seed)': f'0x{W.h32(bytes(range(16)), W.SEED):08X}'}
    d = bytes(((37 * i + 11) & 0x77) for i in range(1024))
    idt = struct.pack('<IIII', 0xAB4261BD, (7 << 16) | 191, 2165, 0x0F | (1 << 8) | (3 << 16) | (127 << 24))
    v['D1 weight-stream vector h23'] = f'0x{W.h32(idt + d, W.SEED) & W.HMASK:06X}'
    ids = struct.pack('<IIII', 0xAB4261BD, (7 << 16) | 191, 719, 0x0F | (127 << 24))
    v['D1 scale-stream vector h23'] = f'0x{W.h32(ids + bytes([0x7F, 0x7E, 0x80, 0x7F, 0x7D]), W.SEED) & W.HMASK:06X}'
    # D1 weight vector with the 0.2.1 ceil bit set in ident[2]
    idt_c = struct.pack('<IIII', 0xAB4261BD, (7 << 16) | 191, 2165 | (1 << 31), 0x1C | (1 << 8) | (3 << 16) | (127 << 24))
    v['0.2.1 weight-stream vector, same data, g=2165|ceil<<31, flags 0x1C, m1, set 3, anchor 127: h23'] = \
        f'0x{W.h32(idt_c + d, W.SEED) & W.HMASK:06X}'
    # record-level
    SS = info['SS']
    cm = H['ceil_mask']
    cw = info['coded_weights']
    rl = {'record': row['record_file'], 'model_id': f"0x{H['model_id']:08X}", 'flags': f"0x{H['flags']:02X}",
          'owner': [H['owner_layer'], H['owner_expert']], 'header_bytes_0_59_hex': rec[:60].hex(),
          'header_check': f"0x{H['header_check']:08X}",
          'header_check_recomputed': f"0x{W.h32(rec[:60], W.SEED):08X}"}
    # scale stream g = 0 (matrix 0): coded scales = first 512 decoded coded scales; recompute from the slab:
    # matrix 0 is gated, coded scale = output scale + (block transformed); get coded scales by re-applying T.
    raw_sc0 = np.frombuffer(slab, np.uint8, W.CS, 0)
    w0 = np.frombuffer(slab, np.uint8, 5898240, 3 * 368640)
    sym = np.empty(w0.size * 2, np.uint8)
    sym[0::2], sym[1::2] = w0 & 15, w0 >> 4
    ok, tsym, tsc, _ = W.gate_and_T(sym, np.frombuffer(slab, np.uint8, 368640, 0).copy())
    coded_s0 = bytes(tsc[:W.CS])
    for g, m, coded in [(0, 0, coded_s0),
                        (SS, 0, bytes((cw[0][0::2] | (cw[0][1::2] << 4)).astype(np.uint8)))]:
        ident = struct.pack('<IIII', H['model_id'], (H['owner_layer'] << 16) | H['owner_expert'],
                            g | (((cm >> m) & 1) << 31),
                            (H['flags'] & 0xFF) | (m << 8) | (H['s0'] << 16) | (H['a0'] << 24))
        h = W.h32(ident + coded, W.SEED)
        rl[f'stream g={g}'] = {'ident_hex': ident.hex(), 'coded_bytes': len(coded), 'coded_sha256':
                               __import__('hashlib').sha256(coded).hexdigest(), 'coded_first16_hex': coded[:16].hex(),
                               'H32': f'0x{h:08X}', 'h23': f'0x{h & W.HMASK:06X}', 'x0': f'0x{W.L + (h & W.HMASK):08X}',
                               'x0_be_bytes_in_record': None}
    rl['stream g=0']['x0_be_bytes_in_record'] = rec[64 + H['LS']:64 + H['LS'] + 4].hex()
    rl[f'stream g={SS}']['x0_be_bytes_in_record'] = rec[64 + H['LS'] + H['SP']:64 + H['LS'] + H['SP'] + 4].hex()
    rl['note'] = ('x0_be_bytes_in_record is the stream start = the encoder FINAL state (big-endian), not x0; '
                  'the decoder ends at x0 = L + h23.')
    rl['coded_scales_match_slab_retransform'] = bool(ok)
    json.dump({'function_vectors': v, 'record_level_vectors': rl}, open(out, 'w'), indent=1)
    print(json.dumps({'function_vectors': v, 'record_level_vectors': {k: rl[k] for k in rl if k != 'header_bytes_0_59_hex'}}, indent=1))


if __name__ == '__main__':
    main()
