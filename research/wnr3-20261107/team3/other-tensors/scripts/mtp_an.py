"""MTP (DSpark) routed experts vs main routed experts under the WNR2 context rule.

Usage: python -I mtp_an.py MTP_DATA_DIR MAIN_COUNTS_DIR OUT_JSON
MTP_DATA_DIR: fetch_tensors.py output (one .bin per tensor, names mtp.S.ffn.experts.E.wK.{weight,scale}).
MAIN_COUNTS_DIR: the existing per-expert pass-1 counts of 160 main experts (read-only), keys w_base [3,567,16], s_o0 [3,256].
Context rule and 14-bit quantizer re-implemented from the WNR2 description (same as mxcount/mxanalyze).
"""
import json, os, sys, glob, re
import numpy as np

P = 16384
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
W_CHUNK, S_CHUNK = 2048, 512
RAW_RECORD = 18_800_640
W_STREAMS = sum(-(-r * c // W_CHUNK) for r, c in [(2304, 5120), (5120, 2304), (2304, 5120)])
S_STREAMS = sum(-(-r * c // 32 // S_CHUNK) for r, c in [(2304, 5120), (5120, 2304), (2304, 5120)])


def quantize(c):
    c = np.asarray(c, np.int64); n = c.shape[-1]; flat = c.reshape(-1, n)
    tot = flat.sum(1, keepdims=True); safe = np.where(tot == 0, 1, tot)
    num = flat * (P - n); fl = num // safe; rem = num % safe
    freq = 1 + fl; left = P - freq.sum(1)
    order = np.argsort(-rem, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(flat.shape[0], 0), axis=1)
    freq = freq + (rank < left[:, None])
    freq[tot[:, 0] == 0] = P // n
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * -np.log2(np.asarray(freq, np.float64) / P)).sum())


def ideal_bits(c):
    c = np.asarray(c, np.float64); t = c.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / np.where(t == 0, 1, t), 1)
    return float(-(c * np.log2(p)).sum())


def matrix_counts(wb, sc):
    R = wb.shape[0]; C = wb.shape[1] * 2; nb = C // 32
    assert sc.shape == (R, nb)
    sym = np.empty((R, C), np.uint8); sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
    blk = sym.reshape(R, nb, 32); mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    s16 = sc.astype(np.int16)
    med = int(np.sort(s16.ravel())[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -3, 3) + 3).astype(np.int32)[..., None]
    base = (srel * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    w = np.bincount((base * 16 + blk.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)
    s = np.bincount(sc.ravel(), minlength=256)
    bm = cm[..., -1]
    return w.astype(np.int64), s.astype(np.int64), {
        'neg0': int((sym == 8).sum()), 'pos0': int((sym == 0).sum()), 'n': int(sym.size),
        'bm_hist': np.bincount(bm.ravel(), minlength=8).tolist(), 'median_scale': med}


def main():
    dirs, cdir, outp = sys.argv[1].split(';'), sys.argv[2], sys.argv[3]
    by = {}
    for dd in dirs:
        man = json.load(open(os.path.join(dd, 'manifest.json'), encoding='utf-8'))
        for it in man['items']:
            by[it['name']] = {**it, 'dir': dd}
    keys = sorted({tuple(map(int, re.match(r'mtp\.(\d+)\.ffn\.experts\.(\d+)\.', n).groups())) for n in by})
    mtp = []
    for (s, e) in keys:
        W = np.zeros((3, 567, 16), np.int64); S = np.zeros((3, 256), np.int64); info = []
        for m, w in enumerate(('w1', 'w2', 'w3')):
            wi = by[f'mtp.{s}.ffn.experts.{e}.{w}.weight']; si = by[f'mtp.{s}.ffn.experts.{e}.{w}.scale']
            assert wi['dtype'] == 'I8' and si['dtype'] == 'F8_E8M0', (wi['dtype'], si['dtype'])
            wb = np.fromfile(os.path.join(wi['dir'], wi['tag'] + '.bin'), np.uint8).reshape(wi['shape'][0], -1)
            sc = np.fromfile(os.path.join(si['dir'], si['tag'] + '.bin'), np.uint8).reshape(si['shape'][0], -1)
            wc, scn, inf = matrix_counts(wb, sc)
            W[m] = wc; S[m] = scn; info.append(inf)
        mtp.append({'stage': s, 'expert': e, 'w': W, 's': S, 'info': info})
    main_files = sorted(glob.glob(os.path.join(cdir, '*.npz')))
    mains = [np.load(f) for f in main_files]
    MW = sum(z['w_base'].astype(np.int64) for z in mains); MS = sum(z['s_o0'].astype(np.int64) for z in mains)
    n_main = len(mains)
    n = len(mtp)
    TW = sum(x['w'] for x in mtp); TS = sum(x['s'] for x in mtp)
    Wn, Sn = n * W_PER_REC, n * S_PER_REC
    fw, fs = 32 * W_STREAMS * n, 32 * S_STREAMS * n

    def rec(wbits, sbits):  # same record estimate as mxanalyze: header 32 + u16 lengths + payload + flush + ~2 KB padding
        return 32 + 2 * (W_STREAMS + S_STREAMS) + (wbits + fw + sbits + fs) / 8 / n + 2048

    res = {'mtp_experts': n, 'mtp_keys': keys, 'main_experts_in_model': n_main}
    # (1) production-like: shared model trained on the 160 main experts, applied to MTP experts
    wq, sq = quantize(MW), quantize(MS)
    wb_main = cost_bits(TW, wq); sb_main = cost_bits(TS, sq)
    res['main_model_on_mtp'] = {'weights_bit_per_w': (wb_main + fw) / Wn, 'scales_bit_per_scale': (sb_main + fs) / Sn,
                                'record_est_bytes': rec(wb_main, sb_main), 'ratio': rec(wb_main, sb_main) / RAW_RECORD}
    # main model on main experts (in-sample, reference 3.7821)
    res['main_model_on_main_insample'] = {'weights_bit_per_w': (cost_bits(MW, wq) + 32 * W_STREAMS * n_main) / (n_main * W_PER_REC),
                                          'scales_bit_per_scale': (cost_bits(MS, sq) + 32 * S_STREAMS * n_main) / (n_main * S_PER_REC)}
    # (2) MTP-trained model, held-out (even/odd experts)
    A = [x for i, x in enumerate(mtp) if i % 2 == 0]; B = [x for i, x in enumerate(mtp) if i % 2 == 1]
    AW = sum(x['w'] for x in A); BW = sum(x['w'] for x in B); AS = sum(x['s'] for x in A); BS = sum(x['s'] for x in B)
    wb_ho = cost_bits(BW, quantize(AW)) + cost_bits(AW, quantize(BW))
    sb_ho = cost_bits(BS, quantize(AS)) + cost_bits(AS, quantize(BS))
    res['mtp_model_heldout'] = {'weights_bit_per_w': (wb_ho + fw) / Wn, 'scales_bit_per_scale': (sb_ho + fs) / Sn,
                                'record_est_bytes': rec(wb_ho, sb_ho), 'ratio': rec(wb_ho, sb_ho) / RAW_RECORD}
    wb_in = cost_bits(TW, quantize(TW)); sb_in = cost_bits(TS, quantize(TS))
    res['mtp_model_insample'] = {'weights_bit_per_w': (wb_in + fw) / Wn, 'scales_bit_per_scale': (sb_in + fs) / Sn,
                                 'ideal_entropy_bit_per_w': ideal_bits(TW) / Wn}
    # (2b) same, signed zero canonicalised (-0 -> +0; contexts use magnitudes only, so merging symbol 8 into 0 is exact)
    def cz(c):
        c = c.copy(); c[..., 0] += c[..., 8]; c[..., 8] = 0; return c
    wz = cost_bits(cz(BW), quantize(cz(AW))) + cost_bits(cz(AW), quantize(cz(BW)))
    res['mtp_model_heldout_canonical_zero_weights_bit_per_w'] = (wz + fw) / Wn
    # (2c) two models in one container, chosen per matrix: main model or MTP model (other half), held-out
    qA, qB = quantize(AW), quantize(BW)
    k2 = 0.0; picks = {'main': 0, 'mtp': 0}
    for i, x in enumerate(mtp):
        qm = qB if i % 2 == 0 else qA
        for m in range(3):
            a, b = cost_bits(x['w'][m], wq[m]), cost_bits(x['w'][m], qm[m])
            k2 += min(a, b); picks['main' if a < b else 'mtp'] += 1
    res['K2_main_or_mtp_model_per_matrix_heldout_weights_bit_per_w'] = (k2 + fw) / Wn
    res['K2_picks'] = picks
    # main experts under the MTP model (is the MTP model also usable for main experts?)
    res['mtp_model_on_main_weights_bit_per_w'] = (cost_bits(MW, quantize(TW)) + 32 * W_STREAMS * n_main) / (n_main * W_PER_REC)
    # (3) joint model main+MTP-train-half applied to the other MTP half (one model for both)
    jb = cost_bits(BW, quantize(MW + AW)) + cost_bits(AW, quantize(MW + BW))
    res['joint_main_plus_mtp_half_heldout_weights_bit_per_w'] = (jb + fw) / Wn
    # per stage, per matrix, order-0
    o0 = TW.sum(1)
    res['mtp_order0_bit_per_w'] = cost_bits(o0, quantize(o0)) / Wn
    res['main_order0_bit_per_w'] = cost_bits(MW.sum(1), quantize(MW.sum(1))) / (n_main * W_PER_REC)
    res['per_stage_main_model_bit_per_w'] = {}
    for s in sorted({x['stage'] for x in mtp}):
        xs = [x for x in mtp if x['stage'] == s]
        ww = sum(x['w'] for x in xs)
        res['per_stage_main_model_bit_per_w'][s] = (cost_bits(ww, wq) + 32 * W_STREAMS * len(xs)) / (len(xs) * W_PER_REC)
    res['per_matrix_main_model_bit_per_w'] = {nm: cost_bits(TW[m], wq[m]) / (n * 2304 * 5120) for m, nm in enumerate(('w1', 'w2', 'w3'))}
    # symbol distribution comparison: order-0 nibble histograms and KL
    pm_ = MW.sum((0, 1)) / MW.sum(); pt_ = TW.sum((0, 1)) / TW.sum()
    res['nibble_hist_main'] = pm_.round(5).tolist(); res['nibble_hist_mtp'] = pt_.round(5).tolist()
    res['KL_mtp_vs_main_order0_bits'] = float((pt_ * np.log2(pt_ / pm_)).sum())
    neg0 = sum(i['neg0'] for x in mtp for i in x['info']); pos0 = sum(i['pos0'] for x in mtp for i in x['info'])
    tot = sum(i['n'] for x in mtp for i in x['info'])
    res['mtp_neg0_frac'] = neg0 / tot; res['mtp_pos0_frac'] = pos0 / tot
    bm = np.sum([i['bm_hist'] for x in mtp for i in x['info']], axis=0)
    res['mtp_block_max_hist'] = bm.tolist()
    res['mtp_scale_hist_top'] = sorted(((int(v), int(k)) for k, v in enumerate(TS.sum(0)) if v), reverse=True)[:8]
    res['main_scale_hist_top'] = sorted(((int(v), int(k)) for k, v in enumerate(MS.sum(0)) if v), reverse=True)[:8]
    res['mtp_median_scales'] = sorted({i['median_scale'] for x in mtp for i in x['info']})
    json.dump(res, open(outp, 'w'), indent=1, default=int)
    print(json.dumps(res, indent=1, default=int))


if __name__ == '__main__':
    main()
