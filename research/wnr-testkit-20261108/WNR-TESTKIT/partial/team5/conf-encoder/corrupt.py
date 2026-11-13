"""team5 / conf-encoder: 3 deliberately corrupted v1 records (+1 extra) with expected outcomes, and the decoder's result.
Usage: python -I corrupt.py SAMPLE_DIR RECORDS_DIR [RECORD_NAME]
A  sign flip: one payload bit flip in a weight stream found by exhaustive search over the stream's bits such that the
   rANS structure is intact (exact length consumption, final state == the encoder's x0) and the decoded symbols differ
   from the original in exactly one nibble by XOR 8 (one weight's sign). Today's WNR2-style check (structure only)
   would accept it; HASH must reject it.
B  one bit of one byte of the LENC weight-part words (first group with width > 0) -> group continuity.
C  one header byte (ceil_mask, byte 38) XOR 1, header_check left stale -> header_check mismatch.
D  (extra, T2 case) as C but header_check recomputed -> set form vs ceil_mask check.
"""
import hashlib, json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr3c as W


def try_decode(rec, model, row):
    try:
        slab, _ = W.decode_record(rec, model, row['entry_flags'], tuple(row['bind']), [tuple(s) for s in row['shapes_symbols']])
        return {'status': 0, 'reason': None, 'decoded_sha256': hashlib.sha256(slab).hexdigest(),
                'equals_source': hashlib.sha256(slab).hexdigest() == row['source_sha256']}
    except W.Invalid as ex:
        return {'status': -1, 'reason': str(ex)}


def main():
    sample_dir, rdir = sys.argv[1:3]
    index = json.load(open(os.path.join(rdir, 'index.json')))
    name = sys.argv[3] if len(sys.argv) > 3 else index['records'][0]['record_file']
    row = [r for r in index['records'] if r['record_file'] == name][0]
    blob = open(os.path.join(rdir, index['model_file']), 'rb').read()
    model = W.parse_wnm3(blob)
    rec = open(os.path.join(rdir, name), 'rb').read()
    slab, info = W.decode_record(rec, model, row['entry_flags'], tuple(row['bind']), [tuple(s) for s in row['shapes_symbols']])
    assert hashlib.sha256(slab).hexdigest() == row['expected_decoded_sha256']
    odir = os.path.join(rdir, 'corrupted')
    os.makedirs(odir, exist_ok=True)
    out = []
    # ---------------- A: sign flip search
    SS, wbeg = info['SS'], info['wbeg']
    tab = W.RecTables(model, info['set_id'])
    found = None
    search = []
    for j in range(0, 64):                                    # first weight streams of matrix 0 until a hit
        o, n = int(info['woffs'][j]), int(info['wlens'][j])
        sb = np.frombuffer(rec, np.uint8, n, wbeg + o)
        nb = 8 * n
        lane = n + 2 * W.CW + 64
        pl = np.zeros(nb * lane, np.uint8)
        pl.reshape(nb, lane)[:, :n] = sb
        bit = np.arange(nb)
        pl.reshape(nb, lane)[bit, bit // 8] ^= (1 << (bit % 8)).astype(np.uint8)
        offs = np.arange(nb, dtype=np.int64) * lane
        srelT = np.repeat(info['srelT'][:, j:j + 1], nb, 1)
        wT, x, ptr, ok = W.decode_weights(pl, offs, srelT, np.zeros(nb, np.int64), tab)
        orig = info['coded_weights'][j]
        want = int(info['wantW'][j])
        struct_ok = ok & (ptr == offs + n) & (x == want)
        diff = wT.T != orig[None, :]
        ndiff = diff.sum(1)
        xor8 = ((wT.T ^ orig[None, :]) == 8) | ~diff
        sign1 = struct_ok & (ndiff == 1) & xor8.all(1)
        search.append({'stream_j': j, 'g': SS + j, 'stream_bytes': n, 'bit_flips_tried': nb,
                       'structure_ok_final_state_eq_x0': int(struct_ok.sum()), 'of_which_exactly_one_sign_flip': int(sign1.sum()),
                       'structure_ok_other': int((struct_ok & ~sign1).sum())})
        if sign1.any():
            b = int(np.flatnonzero(sign1)[0])
            pos = int(np.flatnonzero(diff[b])[0])
            found = (j, b, pos, int(orig[pos]), int(wT[pos, b]))
            break
    j, b, pos, s0, s1 = found
    recA = bytearray(rec)
    recA[wbeg + int(info['woffs'][j]) + b // 8] ^= 1 << (b % 8)
    fa = name.replace('.wnr3', '.A_signflip.wnr3')
    open(os.path.join(odir, fa), 'wb').write(recA)
    out.append({'case': 'A', 'file': fa, 'what': 'single payload bit flip in weight stream; rANS structure intact, '
                'decoded = original with exactly one weight sign flipped',
                'byte_offset_in_record': wbeg + int(info['woffs'][j]) + b // 8, 'bit': b % 8,
                'weight_stream_j': j, 'global_stream_g': SS + j, 'symbol_position_in_stream': pos,
                'symbol_before_after': [s0, s1],
                'search': search,
                'expected_without_HASH (x == x0 and exact length only)': 'accepted silently (one sign flipped)',
                'expected_v1': f'status -1: weight stream final-state/hash check fails at g={SS + j}',
                'got': try_decode(bytes(recA), model, row)})
    # ---------------- B: LENC weight-part byte
    H = W.unpack_header(rec[:64])
    hs = rec[64:64 + 12]
    nss = [W.CS * 0 + 720] * 3
    ngS = sum(-(-c // 32) for c in [s[0] * s[1] // 32 // W.CS for s in row['shapes_symbols']])
    wS = np.frombuffer(rec, np.uint8, ngS, 64 + 12 + 4 * ngS)
    spart = 4 * ngS + W.pad4(ngS) + 4 * int(wS.astype(np.int64).sum())
    pw = 64 + 12 + spart
    ngW = sum(-(-(s[0] * s[1] // W.CW) // 32) for s in row['shapes_symbols'])
    widths = np.frombuffer(rec, np.uint8, ngW, pw + 4 * ngW).astype(np.int64)
    gi = int(np.flatnonzero(widths > 0)[0])
    words_at = pw + 4 * ngW + W.pad4(ngW) + 4 * int(widths[:gi].sum())
    recB = bytearray(rec)
    recB[words_at] ^= 0x01
    fb = name.replace('.wnr3', '.B_lenbyte.wnr3')
    open(os.path.join(odir, fb), 'wb').write(recB)
    z_old = rec[words_at] & ((1 << int(widths[gi])) - 1) if widths[gi] < 8 else None
    out.append({'case': 'B', 'file': fb, 'what': f'LENC weight part: byte at record offset {words_at} (first word byte of '
                f'weight group {gi}, width {int(widths[gi])}) XOR 0x01 = bit 0 of residual 0 of that group',
                'byte_offset_in_record': words_at, 'byte_before_after': [rec[words_at], recB[words_at]],
                'expected_v1': f'status -1: LENC weight group continuity fails at group {gi + 1} '
                               '(stream 0 of the group changes length, base_off of the next group no longer matches)',
                'got': try_decode(bytes(recB), model, row)})
    # ---------------- C: header byte, stale header_check
    recC = bytearray(rec)
    recC[38] ^= 0x01
    fc = name.replace('.wnr3', '.C_hdrbyte.wnr3')
    open(os.path.join(odir, fc), 'wb').write(recC)
    out.append({'case': 'C', 'file': fc, 'what': 'header byte 38 (ceil_mask) XOR 0x01 (7 -> 6), header_check not updated',
                'byte_offset_in_record': 38, 'byte_before_after': [rec[38], recC[38]],
                'expected_v1': 'status -1: header_check mismatch', 'got': try_decode(bytes(recC), model, row)})
    # ---------------- D: extra, header_check recomputed
    recD = bytearray(recC)
    struct.pack_into('<I', recD, 60, W.h32(bytes(recD[:60]), W.SEED))
    fd = name.replace('.wnr3', '.D_ceilmask_rehashed.wnr3')
    open(os.path.join(odir, fd), 'wb').write(recD)
    out.append({'case': 'D (extra)', 'file': fd, 'what': 'as C, header_check recomputed (T2 case: ceil_mask bit flipped '
                'with header_check recomputed)', 'expected_v1': 'status -1: ceil_mask bit 0 != form of set 0',
                'got': try_decode(bytes(recD), model, row)})
    for c in out:
        c['outcome_matches_expected'] = (c['got']['status'] == -1)
        print(c['case'], c['got'])
    res = {'source_record': name, 'source_record_sha256': row['record_sha256'], 'bind': row['bind'],
           'entry_flags': row['entry_flags'], 'cases': out}
    json.dump(res, open(os.path.join(odir, 'corrupted_index.json'), 'w'), indent=1, default=int)
    print(json.dumps(search, indent=1))


if __name__ == '__main__':
    main()
