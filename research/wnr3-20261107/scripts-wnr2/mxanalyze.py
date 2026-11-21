"""Pass 2: code lengths of WNR2 and candidate models from the pass-1 counts.

Usage: python -I mxanalyze.py SAMPLE_DIR COUNTS_DIR OUT_JSON
Costs use 14-bit CDFs quantized exactly like wex_nr2_quantize (1 + floor(c*(P-n)/total), remaining
slots to the largest remainders, ties to the lower symbol; empty rows near-uniform), so they are real
rANS payload sizes up to the coder's ~0.01% inefficiency. Flush: 4 B per stream.
"held-out" = model trained on experts of split A, cost measured on split B, and vice versa (summed).
"""
import json, os, sys
import numpy as np

P = 16384
RAW_RECORD = 18_800_640
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
W_CHUNK, S_CHUNK = 2048, 512


def quantize(c):
    """c: int64 [..., n] counts -> int64 freq [..., n], each row sums to P, every freq >= 1."""
    c = np.asarray(c, np.int64)
    n = c.shape[-1]
    flat = c.reshape(-1, n)
    tot = flat.sum(1, keepdims=True)
    safe = np.where(tot == 0, 1, tot)
    num = flat * (P - n)
    fl = num // safe
    rem = num % safe
    freq = 1 + fl
    left = P - freq.sum(1)
    order = np.argsort(-rem, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(flat.shape[0], 0), axis=1)
    freq = freq + (rank < left[:, None])
    # empty rows: near-uniform (P/n each; P divisible by 16 and 256)
    freq[tot[:, 0] == 0] = P // n
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * -np.log2(np.asarray(freq, np.float64) / P)).sum())


def ideal_bits(counts):
    c = np.asarray(counts, np.float64)
    t = c.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / t, 1)
    return float(-(c * np.log2(p)).sum())


def load_all(sample_dir, counts_dir):
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        ex.append({'layer': e['layer'], 'expert': e['expert'], 'split': i % 2, **{k: z[k] for k in z.files}})
    return ex


def tot(ex, key, sel=lambda e: True):
    return sum(e[key].astype(np.int64) for e in ex if sel(e))


def eval_static(ex, key, sel_train=None, per_matrix=True):
    """Shared static model over key; returns (in-sample bits, held-out bits)."""
    allc = tot(ex, key)
    ins = cost_bits(allc, quantize(allc))
    a = tot(ex, key, lambda e: e['split'] == 0)
    b = tot(ex, key, lambda e: e['split'] == 1)
    ho = cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))
    return ins, ho


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    ex = load_all(sample_dir, counts_dir)
    n = len(ex)
    W = n * W_PER_REC
    S = n * S_PER_REC
    out = {'experts': n, 'layers': sorted({e['layer'] for e in ex})[:3] + ['...'], 'weights': W, 'scales': S}
    w_streams = n * sum(-(-r * c // W_CHUNK) for r, c in [(2304, 5120), (5120, 2304), (2304, 5120)])
    s_streams = n * sum(-(-r * c // 32 // S_CHUNK) for r, c in [(2304, 5120), (5120, 2304), (2304, 5120)])
    flush_w = 32 * w_streams
    flush_s = 32 * s_streams

    def bpw(bits):
        return (bits + flush_w) / W

    def bps(bits):
        return (bits + flush_s) / S

    # ---------- baseline ----------
    bw_in, bw_ho = eval_static(ex, 'w_base')
    bs_in, bs_ho = eval_static(ex, 's_o0')
    rec_bytes = lambda wb, sb: 32 + 2 * (w_streams + s_streams) / n + (wb + flush_w + sb + flush_s) / 8 / n + 2048
    base = {
        'weights_bit_per_w_insample': bpw(bw_in), 'weights_bit_per_w_heldout': bpw(bw_ho),
        'weights_ideal_entropy_insample': ideal_bits(tot(ex, 'w_base')) / W,
        'scales_bit_per_scale_insample': bps(bs_in), 'scales_bit_per_scale_heldout': bps(bs_ho),
        'record_bytes_est_insample': rec_bytes(bw_in, bs_in), 'ratio_insample': rec_bytes(bw_in, bs_in) / RAW_RECORD,
        'order0_weights_bit_per_w': cost_bits(tot(ex, 'w_base').sum(1), quantize(tot(ex, 'w_base').sum(1))) / W,
        'reference_measured': {'weights': 3.781, 'scales': 1.066, 'ratio': 0.8997, 'record_bytes': 16914680},
    }
    out['baseline'] = base
    B_in, B_ho = bw_in, bw_ho

    # per-matrix breakdown
    mt = {}
    for m, name in enumerate(['w1 gate', 'w2 down', 'w3 up']):
        c = tot(ex, 'w_base')[m]
        mt[name] = cost_bits(c, quantize(c)) / (n * 2304 * 5120)
    out['baseline_per_matrix_bit_per_w'] = mt

    # ---------- richer contexts (weights) ----------
    rich = {}
    for key in ['w_srel15', 'w_pos', 'w_pm2', 'w_psign', 'w_up', 'w_upsort', 'w_bm', 'w_bmpos']:
        i, h = eval_static(ex, key)
        side_i = side_h = 0.0
        if key in ('w_bm', 'w_bmpos'):
            side_i, side_h = eval_static(ex, 'bm_side')
        rich[key] = {'insample_bit_per_w': bpw(i + side_i), 'heldout_bit_per_w': bpw(h + side_h),
                     'delta_vs_base_heldout_pct_of_weight_bits': 100 * ((h + side_h) - B_ho) / (B_ho + flush_w),
                     'side_bits_per_w': side_i / W}
    # partner context on w3 only (w1 at same position)
    if all('w_part' in e for e in ex):
        allp = tot(ex, 'w_part'); a = tot(ex, 'w_part', lambda e: e['split'] == 0); b = tot(ex, 'w_part', lambda e: e['split'] == 1)
        w3 = tot(ex, 'w_base')[2]; w3a = tot(ex, 'w_base', lambda e: e['split'] == 0)[2]; w3b = tot(ex, 'w_base', lambda e: e['split'] == 1)[2]
        rich['w_part (w3 | w1 same position)'] = {
            'w3_insample_bit_per_w': cost_bits(allp, quantize(allp)) / (n * 2304 * 5120),
            'w3_base_insample_bit_per_w': cost_bits(w3, quantize(w3)) / (n * 2304 * 5120),
            'w3_heldout_bit_per_w': (cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))) / (n * 2304 * 5120),
            'w3_base_heldout_bit_per_w': (cost_bits(w3b, quantize(w3a)) + cost_bits(w3a, quantize(w3b))) / (n * 2304 * 5120)}
    out['weight_contexts'] = rich

    # ---------- block max statistics ----------
    side = tot(ex, 'bm_side')  # [3, 7, 8]
    out['block_max_hist'] = {f'm{m}': (side[m].sum(0) / side[m].sum()).round(5).tolist() for m in range(3)}
    out['block_max_entropy_bits_per_block'] = ideal_bits(side.sum(0).sum(0)) / 1
    neg0 = sum(int(e['meta'][:, 3].sum()) for e in ex)
    out['neg_zero_fraction'] = neg0 / W

    # ---------- WNR2z (value-exact) ----------
    c = tot(ex, 'w_base').copy()
    c[..., 0] += c[..., 8]; c[..., 8] = 0
    out['wnr2z_insample_bit_per_w'] = bpw(cost_bits(c, quantize(c)))

    # ---------- per-layer and grouped models ----------
    layers = sorted({e['layer'] for e in ex})
    per_layer_in = per_layer_loo = 0.0
    for L in layers:
        exl = [e for e in ex if e['layer'] == L]
        cl = tot(exl, 'w_base')
        per_layer_in += cost_bits(cl, quantize(cl))
        for e in exl:
            rest = cl - e['w_base']
            per_layer_loo += cost_bits(e['w_base'], quantize(rest))
    # shared model leave-one-expert-out for a fair comparison with per-layer LOO
    call = tot(ex, 'w_base')
    shared_loo = sum(cost_bits(e['w_base'], quantize(call - e['w_base'])) for e in ex)
    groups = {}
    for G in [2, 4, 5, 8, 10, 13, 20]:
        gi = gl = 0.0
        for g in range(G):
            sel = [e for e in ex if e['layer'] * G // 40 == g]
            if not sel:
                continue
            cg = tot(sel, 'w_base')
            gi += cost_bits(cg, quantize(cg))
            gl += sum(cost_bits(e['w_base'], quantize(cg - e['w_base'])) for e in sel)
        groups[G] = {'insample_bit_per_w': bpw(gi), 'loo_bit_per_w': bpw(gl)}
    # per-expert own model (semi-static) with the model cost: weight CDFs 3*567*17 u16 = 57,834 B
    own = sum(cost_bits(e['w_base'], quantize(e['w_base'])) for e in ex)
    out['model_granularity'] = {
        'shared_insample': bpw(B_in), 'shared_loo': bpw(shared_loo),
        'per_layer_insample': bpw(per_layer_in), 'per_layer_loo': bpw(per_layer_loo),
        'layer_groups (G models, layer*G//40)': groups,
        'per_expert_own_model_payload_only': bpw(own),
        'per_expert_own_model_incl_57834B_table': bpw(own + n * 57834 * 8),
    }
    # per-layer curve: shared vs per-layer by layer
    curve = []
    for L in layers:
        exl = [e for e in ex if e['layer'] == L]
        cl = tot(exl, 'w_base')
        nW = len(exl) * W_PER_REC
        curve.append({'layer': L, 'shared_loo': sum(cost_bits(e['w_base'], quantize(call - e['w_base'])) for e in exl) / nW,
                      'layer_loo': sum(cost_bits(e['w_base'], quantize(cl - e['w_base'])) for e in exl) / nW})
    out['per_layer_curve'] = curve

    # ---------- K clustered models, assignment per (expert, matrix) ----------
    units = [(i, m) for i in range(n) for m in range(3)]
    U = np.stack([ex[i]['w_base'][m] for i, m in units]).astype(np.int64)  # [u, 567, 16]
    mtype = np.array([m for _, m in units])
    clus = {}
    for K in [2, 4, 8, 16]:
        # init: per matrix type, split units by layer into K/… groups -> start from layer bands
        lay = np.array([ex[i]['layer'] for i, _ in units])
        assign = (lay * K // 40).astype(int)
        for it in range(12):
            models = []
            for k in range(K):
                c = U[assign == k].sum(0) if (assign == k).any() else np.zeros((567, 16), np.int64)
                models.append(-np.log2(quantize(c) / P))
            Lm = np.stack(models)  # [K, 567, 16]
            costs = np.einsum('ucs,kcs->uk', U.astype(np.float64), Lm)
            new = costs.argmin(1)
            if (new == assign).all():
                break
            assign = new
        bits = costs[np.arange(len(units)), assign].sum()
        clus[K] = {'insample_bit_per_w': bpw(bits), 'iterations': it + 1,
                   'note': 'one model id per matrix (2-4 bits side info per record, negligible); models shared across matrix types'}
    out['clustered_models'] = clus

    # ---------- scales ----------
    sc = {}
    for key in ['s_o0', 's_left', 's_up', 's_lu', 's_lusort']:
        i, h = eval_static(ex, key)
        sc[key] = {'insample_bit_per_scale': bps(i), 'heldout_bit_per_scale': bps(h)}
    out['scale_contexts'] = sc
    out['bytes_note'] = {
        'weights_bits_share_of_record': (B_in + flush_w) / 8 / n / RAW_RECORD,
        'scales_bits_share_of_record': (bs_in + flush_s) / 8 / n / RAW_RECORD,
        'lengths_table_share': 2 * (w_streams + s_streams) / n / RAW_RECORD,
        'flush_share': (flush_w + flush_s) / 8 / n / RAW_RECORD,
        'padding_share_avg': 2048 / RAW_RECORD,
    }
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
