"""Write weight-stream decode jobs for the C decoder (zdec.c), all from ONE expert, all in the same coded domain
(ceil-form, anchor 120, canonical zero), every variant with the per-stream hash (E):

  CF  : 16-symbol rows (team 4's fgcde streams, copied ceil-ref coder, model fg_a0_K0_c1)
  D   : 8-symbol magnitude rows + FIXED sign plane (one u32 per 32-block; zeros' bits are 0): Z1's rANS bytes, no tail plane
  Z1  : 8-symbol magnitude rows + compacted tail sign plane (this member's layout)
  Z2  : as Z1, last <= 7 signs in x0 above the hash

Job file (little endian): 'ZJOB' u32 version=1, variant (0 CF, 1 D, 2 Z1, 3 Z2), S, R (rows), alpha, rowstride (u16),
paylen; tables R*rowstride u16 (cum[0..alpha], zero padded); off[S] u32, len[S] u32, seed[S] u32 (FNV state after the
identity bytes), mat[S] u8; srel[S*64] u8; payload paylen bytes; D: signs[S*64] u32; expected nibbles S*1024 bytes
(byte i of a stream = sym[2i] | sym[2i+1] << 4). Streams are ordered w1 streams, w2, w3 (as in the record).
Usage: python -I make_jobs.py SAMPLE_DIR ZMODEL_NPZ CEILREF_MODELS_NPZ OUT_DIR EXPERT_IDX
"""
import json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W
import signplane as Z

VAR = {'CF': 0, 'D': 1, 'Z1': 2, 'Z2': 3}


def tables_u16(freq, stride):
    R, a = freq.shape
    t = np.zeros((R, stride), np.uint16)
    t[:, 1:a + 1] = np.cumsum(freq, 1)
    return t


def write_job(path, var, tab, offs, lens, seeds, mat, srel, pay, nib, signs=None):
    S = offs.size
    R, stride = tab.shape
    alpha = 16 if var == 'CF' else 8
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sIIIIIIQ', b'ZJOB', 1, VAR[var], S, R, alpha, stride, pay.size))
        f.write(tab.astype('<u2').tobytes())
        f.write(offs.astype('<u4').tobytes()); f.write(lens.astype('<u4').tobytes())
        f.write(seeds.astype('<u4').tobytes()); f.write(mat.astype(np.uint8).tobytes())
        f.write(srel.astype(np.uint8).tobytes())
        f.write(pay.astype(np.uint8).tobytes())
        if var == 'D':
            f.write(signs.astype('<u4').tobytes())
        f.write(nib.astype(np.uint8).tobytes())


def main():
    sample_dir, zmodel, refmodels, out_dir, idx = sys.argv[1:6]
    idx = int(idx)
    Z.use_g0(120)
    man = W.load_manifest(sample_dir)
    e = man['experts'][idx]
    raw, mats = W.load_expert(sample_dir, e)
    tz = Z.load_tables(zmodel)
    info_all = {'expert': e['file'], 'idx': idx}
    # ---- Z1 / Z2 (keep the stream internals)
    for var, z2 in (('Z1', False), ('Z2', True)):
        rec, info = Z.encode_record_z(mats, tz, e['layer'], e['expert'], d=True, e=True, z2=z2, keep=True)
        k = info['_keep']
        symT, nws, feats = k['symT'], k['nws'], k['feats']
        WS = symT.shape[1]
        wmat = np.repeat(np.arange(3), nws)
        widx = np.concatenate([np.arange(n) for n in nws])
        seeds = W.hash_seeds(e['layer'], e['expert'], tz.model_id, 1, wmat, widx, np.zeros(WS, np.int64))
        srel = np.concatenate([feats[m]['srel'].reshape(nws[m], 64) for m in range(3)])
        q = np.ascontiguousarray(symT.T)                      # [WS, 2048]
        nib = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        wl = info['wlens']
        offs = np.zeros(WS, np.int64); offs[1:] = np.cumsum(wl)[:-1]
        # payload = the record's weight payload (header + length table + scale payload precede it)
        p0 = info['header'] + info['length_table'] + info['scale_payload']
        pay = np.frombuffer(rec, np.uint8)[p0:p0 + int(wl.sum())]
        tab = tables_u16(tz.wf8, 16)
        write_job(os.path.join(out_dir, f'{var}.job'), var, tab, offs, wl, seeds, wmat, srel, pay, nib)
        info_all[var] = {'streams': WS, 'payload': int(pay.size), 'record_used': info['used'], 'model_id': tz.model_id}
        if var == 'Z1':
            # D: Z1's rANS bytes (same x0 = L + h23), fixed canonical sign plane
            rl = k['rlens']
            roffs = np.zeros(WS, np.int64); roffs[1:] = np.cumsum(rl)[:-1]
            sg = (q >> 3).astype(np.uint32).reshape(WS, 64, 32)
            signs = (sg << np.arange(32, dtype=np.uint32)).sum(2, dtype=np.uint64).astype(np.uint32)   # [WS, 64]
            write_job(os.path.join(out_dir, 'D.job'), 'D', tab, roffs, rl, seeds, wmat, srel, k['rpay'], nib, signs)
            info_all['D'] = {'streams': WS, 'payload': int(k['rpay'].size), 'sign_plane_bytes': int(signs.size * 4)}
            nib_ref, srel_ref = nib, srel
    # ---- CF: team 4's fgcde streams via the copied coder
    key = 'fg_a0_K0_c1'
    R = np.load(refmodels)
    t16 = W.Tables(R[key + '_wf'], R[key + '_scf'], None, R[key + '_etab'], K=0, nctx=W.NCTX)
    cfg = W.Config('fgcde', c=True, d=True, e=True, f=True, g=True)
    rec, info, st, bi = W.encode_record(mats, t16, cfg, e['layer'], e['expert'])
    nws = st['nws']
    WS = sum(nws)
    wmat = np.repeat(np.arange(3), nws)
    widx = np.concatenate([np.arange(n) for n in nws])
    seeds = W.hash_seeds(e['layer'], e['expert'], t16.model_id, 1, wmat, widx, np.zeros(WS, np.int64))
    wl = st['wlens']
    offs = np.zeros(WS, np.int64); offs[1:] = np.cumsum(wl)[:-1]
    write_job(os.path.join(out_dir, 'CF.job'), 'CF', tables_u16(t16.wf, 32), offs, wl, seeds, wmat, srel_ref,
              st['wpay'], nib_ref)
    info_all['CF'] = {'streams': WS, 'payload': int(st['wpay'].size), 'record_used': info['used'], 'model_id': t16.model_id}
    json.dump(info_all, open(os.path.join(out_dir, 'jobs.json'), 'w'), indent=1)
    print(json.dumps(info_all, indent=1))


if __name__ == '__main__':
    main()
