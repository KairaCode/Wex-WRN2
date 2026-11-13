"""Pair NVIDIA's DeepSeek-V4.1-Flash-1017-NVFP4 experts with the MXFP4 source bytes of the same upstream experts.

Usage: python -I nvpair_ds.py NV_DIR OUT_DIR
MXFP4 item files (from the NV manifest's 'mx_file'): w1.scale w2.scale w3.scale (E8M0 [R, C/32]) then w1/w2/w3.weight (packed E2M1).
Checks per matrix:
  codes   NV code == MX code with -0 (8) -> +0 (0)
  scales  both 16-halves of a 32-group carry the same E4M3 byte; value(s16) * tscale == 2^(e - 127)
  tscale  is a power of two; k = log2(tscale) vs max(e) - 127
  inverse NV -> MX(canonical zero) -> NV reproduces the NV bytes exactly (weights, scales; tscale stored raw)
Also writes, per item, WNR2-context counts (srel7 x pm9 x mx9 x 16, 32-blocks) on the reconstructed MXFP4 in three forms:
  'cz'   canonical-zero MXFP4 (= what the NVFP4 file holds), plain form
  'czc'  canonical-zero + ceil-form (WNR3 transform f), anchor = lower median of the coded scales
  'czcg' same with a fixed anchor (120 for w1/w3 = REF g0; median for w2 as coded)  [anchor rule: g0 = 120 all matrices]
and the scale order-0 counts of each form, to estimate WNR3z-on-NVFP4 record sizes.
"""
import importlib.util, json, os, sys, time
import numpy as np

_s = importlib.util.spec_from_file_location('nvcommon', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvcommon.py'))
nc = importlib.util.module_from_spec(_s); _s.loader.exec_module(nc)

HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.int16)


def load_mx(path):
    raw = np.fromfile(path, dtype=np.uint8)
    shapes = [('w1', 'scale', (2304, 160)), ('w2', 'scale', (5120, 72)), ('w3', 'scale', (2304, 160)),
              ('w1', 'weight', (2304, 2560)), ('w2', 'weight', (5120, 1152)), ('w3', 'weight', (2304, 2560))]
    out, off = {}, 0
    for r, p, sh in shapes:
        n = sh[0] * sh[1]
        out[(r, p)] = raw[off:off + n].reshape(sh)
        off += n
    assert off == raw.size
    return out


def ctx_counts(sym, sc, anchor):
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    srel = (np.clip(sc.astype(np.int32) - anchor, -3, 3) + 3)
    ctx = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    return np.bincount((ctx * 16 + blk.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)


def ceil_forward(sym, sc):
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = (blk & 7).astype(np.int16)
    bmax = mag.max(2)
    if (bmax == 5).any():
        return None, None, None  # gate: stored block with max 5 -> no ceil-form for this matrix
    even = (bmax == 7) & ~(((mag == 1) | (mag == 3)).any(2)) & (sc < 254)
    tb = np.where(even[..., None], (blk & 8).astype(np.int16) | HALF[mag], blk.astype(np.int16)).astype(np.uint8)
    return tb.reshape(R, C), np.where(even, sc.astype(np.int16) + 1, sc.astype(np.int16)), even


def e4m3_pow2_byte(k):
    """E4M3 byte of 2^k (k int array), -9 <= k <= 8."""
    k = np.asarray(k, np.int64)
    return np.where(k >= -6, (k + 7) << 3, 1 << (k + 9)).astype(np.uint8)


def main():
    nvd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(nvd, 'manifest.json')))
    rep = []
    t0 = time.time()
    for it in man['experts']:
        mx = load_mx(it['mx_file'])
        res = {'tag': it['tag'], 'mx_file': it['mx_file'], 'mats': []}
        cnt = {k: np.zeros((3, 567, 16), np.int64) for k in ('cz', 'czc', 'czcg')}
        sct = {k: np.zeros((3, 256), np.int64) for k in ('cz', 'czc', 'czcg')}
        for m, (r, codes, sb, ts, isc, nb) in enumerate(nc.matrices(nvd, it)):
            mw = mx[(r, 'weight')]
            ms = mx[(r, 'scale')]
            R = mw.shape[0]
            msym = nc.unpack(mw)
            neg0 = int((msym == 8).sum())
            czs = msym.copy(); czs[czs == 8] = 0
            code_mis = int((codes != czs).sum())
            # scales
            pair_eq = bool(np.array_equal(sb[:, 0::2], sb[:, 1::2]))
            k = np.log2(ts)
            tpow2 = float(k) == round(float(k))
            nv_e = np.log2(nc.e4m3_value(sb[:, 0::2]) * ts) + 127
            sc_mis = int((np.abs(nv_e - ms.astype(np.float64)) > 1e-9).sum())
            emax = int(ms.max()); emin = int(ms.min())
            # inverse NV -> MX(cz): e from the even-half byte and tscale; forward MX(cz) -> NV with the stored k
            e_rec = np.rint(nv_e).astype(np.int64)
            kk = int(round(float(k)))
            fwd_b = e4m3_pow2_byte(e_rec - 127 - kk)
            fwd_sb = np.repeat(fwd_b, 2, axis=1)
            inv_ok = bool(np.array_equal(fwd_sb, sb)) and code_mis == 0
            # WNR3z-style counts on the reconstructed MXFP4 (= canonical-zero source)
            sc8 = e_rec.astype(np.int16)
            fl = np.sort(sc8.ravel()); med = int(fl[(fl.size - 1) // 2])
            cnt['cz'][m] = ctx_counts(codes, sc8, med)
            sct['cz'][m] = np.bincount(sc8.ravel(), minlength=256)[:256]
            tsym, tsc, even = ceil_forward(codes, sc8)
            if tsym is None:
                tsym, tsc, even = codes, sc8, np.zeros(sc8.shape, bool)
            fl = np.sort(tsc.ravel()); medc = int(fl[(fl.size - 1) // 2])
            cnt['czc'][m] = ctx_counts(tsym, tsc, medc)
            cnt['czcg'][m] = ctx_counts(tsym, tsc, 120)
            sct['czc'][m] = sct['czcg'][m] = np.bincount(tsc.ravel().astype(np.int64), minlength=256)[:256]
            res['mats'].append({'role': r, 'shape': list(codes.shape), 'mx_minus0': neg0, 'code_mismatch_vs_cz': code_mis,
                                'halves_equal': pair_eq, 'tscale': ts, 'tscale_log2': float(k), 'tscale_pow2': tpow2,
                                'mx_e_min': emin, 'mx_e_max': emax, 'emax_minus_127_minus_log2t': emax - 127 - float(k),
                                'nv_byte_max': int(sb.max()), 'nv_byte_min': int(sb.min()),
                                'scale_value_mismatch_blocks': sc_mis, 'inverse_roundtrip_exact': inv_ok,
                                'ceil_even_share': float(even.mean()), 'median_cz': med, 'median_czc': medc,
                                'input_scale': isc})
        np.savez_compressed(os.path.join(out, it['tag'] + '.npz'), **{f'w_{k}': v for k, v in cnt.items()},
                            **{f's_{k}': v for k, v in sct.items()})
        rep.append(res)
        print(it['tag'], [(x['role'], x['code_mismatch_vs_cz'], x['halves_equal'], x['scale_value_mismatch_blocks'],
                           x['inverse_roundtrip_exact'], x['emax_minus_127_minus_log2t'], round(x['ceil_even_share'], 4)) for x in res['mats']],
              round(time.time() - t0, 1), flush=True)
    json.dump(rep, open(os.path.join(out, 'pair_report.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
