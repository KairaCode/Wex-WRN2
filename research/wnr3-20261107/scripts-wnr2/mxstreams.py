"""Pass 3: per-stream sizes under the in-sample WNR2 model, and how cheaply the u16 length table could be coded.

Usage: python -I mxstreams.py SAMPLE_DIR COUNTS_DIR OUT_JSON [max_experts]
Stream bytes are estimated as round(bits/8) + 4 (rANS 4-B flushed state). The decoder knows every block's scale
before it decodes weights, so a length can be predicted from the scales: pred = 4 + sum over the stream's
blocks of E[bits | srel] / 8. We report the entropy of (actual - pred) per matrix type: the bits a coded length
table would need instead of 16.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxanalyze import quantize, P, load_all, tot, ideal_bits
from mxcount import load_expert

W_CHUNK, S_CHUNK = 2048, 512


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    maxe = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    ex = load_all(sample_dir, counts_dir)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    cw = tot(ex, 'w_base')  # [3, 567, 16]
    Lw = -np.log2(quantize(cw) / P)
    cs = tot(ex, 's_o0')
    Ls = -np.log2(quantize(cs) / P)
    # expected bits of a 32-block given srel: sum over contexts with that srel / number of blocks
    side = tot(ex, 'bm_side')  # [3, 7, 8] blocks per srel
    blocks_per_srel = side.sum(2)  # [3, 7]
    eb = np.zeros((3, 7))
    for m in range(3):
        cost_ctx = (cw[m] * Lw[m]).sum(1)  # [567]
        for s in range(7):
            eb[m, s] = cost_ctx[s * 81:(s + 1) * 81].sum() / max(1, blocks_per_srel[m, s])
    es = (cs * Ls).sum(1) / cs.sum(1)  # mean bits per scale per matrix
    step = max(1, len(man['experts']) // maxe)
    res_w = [[] for _ in range(3)]
    res_s = [[] for _ in range(3)]
    act_w = [[] for _ in range(3)]
    for e in man['experts'][::step][:maxe]:
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, _ = t[f'{w}.weight']
            sraw, _, _ = t[f'{w}.scale']
            R = wshape[0]
            wb = wraw.reshape(R, -1)
            sc = sraw.reshape(R, -1).astype(np.int16)
            C = wb.shape[1] * 2
            nb = C // 32
            sym = np.empty((R, C), np.uint8)
            sym[:, 0::2] = wb & 15
            sym[:, 1::2] = wb >> 4
            blk = sym.reshape(R, nb, 32)
            mag = blk & 7
            pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
            cm = np.maximum.accumulate(mag, axis=2)
            mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
            med = int(np.sort(sc.ravel())[(sc.size - 1) // 2])
            srel = (np.clip(sc - med, -3, 3) + 3).astype(np.int32)
            ctx = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
            bits = Lw[m][ctx, blk.astype(np.int32)].ravel()
            ns = -(-bits.size // W_CHUNK)
            pad = ns * W_CHUNK - bits.size
            sb = np.concatenate([bits, np.zeros(pad)]).reshape(ns, W_CHUNK).sum(1)
            actual = np.round(sb / 8) + 4
            bl = eb[m][srel.ravel()]  # expected bits per block in row-major block order
            nbs = W_CHUNK // 32
            nsb = -(-bl.size // nbs)
            bl = np.concatenate([bl, np.zeros(nsb * nbs - bl.size)]).reshape(nsb, nbs).sum(1)
            pred = np.round(bl / 8) + 4
            res_w[m].append((actual - pred).astype(np.int64))
            act_w[m].append(actual.astype(np.int64))
            sbits = Ls[m][sc.astype(np.int32).ravel()]
            nss = -(-sbits.size // S_CHUNK)
            sbits = np.concatenate([sbits, np.zeros(nss * S_CHUNK - sbits.size)]).reshape(nss, S_CHUNK).sum(1)
            sact = np.round(sbits / 8) + 4
            spred = np.round(es[m] * S_CHUNK / 8) + 4
            res_s[m].append((sact - spred).astype(np.int64))

    def ent(v):
        v = np.concatenate(v)
        _, c = np.unique(v, return_counts=True)
        return ideal_bits(c[None, :]) / v.size, float(v.mean()), float(v.std()), int(v.min()), int(v.max())

    out = {'experts_used': len(res_w[0])}
    for m in range(3):
        h, mu, sd, lo, hi = ent(res_w[m])
        ha, mua, sda, loa, hia = ent(act_w[m])
        hs, mus, sds, los, his = ent(res_s[m])
        out[f'm{m}'] = {'weight_len_residual_bits': h, 'residual_mean': mu, 'residual_std': sd, 'residual_range': [lo, hi],
                        'weight_len_raw_entropy_bits': ha, 'weight_len_mean_bytes': mua, 'weight_len_std': sda, 'weight_len_range': [loa, hia],
                        'scale_len_residual_bits': hs, 'scale_len_std': sds, 'scale_len_range': [los, his]}
    # whole-record saving if the length table were entropy coded with these residual entropies
    w_streams = [5760, 5760, 5760]  # 2304*5120/2048 = 5120*2304/2048 = 5760
    s_streams = [720, 720, 720]     # 368,640 scales per matrix / 512
    coded = sum(w_streams[m] * out[f'm{m}']['weight_len_residual_bits'] + s_streams[m] * out[f'm{m}']['scale_len_residual_bits'] for m in range(3)) / 8
    out['length_table_bytes_now'] = 2 * (sum(w_streams) + sum(s_streams))
    out['length_table_bytes_coded_est'] = coded
    out['saving_pct_of_raw_record'] = 100 * (out['length_table_bytes_now'] - coded) / 18_800_640
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
