"""team2/audit pass 4: analysis of exact rANS lengths (audit_rans.py output).

Usage: python -I audit_streams.py SAMPLE_DIR RANS_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import ideal_bits, PB

RAW = 18_800_640
NW = [5760, 5760, 5760]
NS = [720, 720, 720]
WPR = 3 * 2304 * 5120
SPR = WPR // 32


def ent(v):
    _, c = np.unique(v, return_counts=True)
    return ideal_bits(c[None, :]) / v.size


def main():
    sample_dir, rans_dir, out_json = sys.argv[1:4]
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    md = np.load(os.path.join(rans_dir, 'model.npz'))
    wf, base = md['wf'], md['base']
    recs = []
    for i, e in enumerate(man['experts']):
        p = os.path.join(rans_dir, e['file'].replace('.bin', '.rans.npz'))
        if os.path.exists(p):
            z = np.load(p)
            recs.append({'i': i, **{k: z[k] for k in z.files}})
    n = len(recs)
    wlen = np.stack([r['wlen'] for r in recs]).astype(np.int64)    # [n, 17280]
    wbits = np.stack([r['wbits'] for r in recs])
    slen = np.stack([r['slen'] for r in recs]).astype(np.int64)    # [n, 2160]
    sbits = np.stack([r['sbits'] for r in recs])
    wfin = np.stack([r['wfinal'] for r in recs]).astype(np.float64)
    whist = np.stack([r['whist'] for r in recs]).astype(np.int64)  # [n, 17280, 7]
    out = {'experts': n}
    over_w = wlen - wbits / 8
    over_s = slen - sbits / 8
    u = np.log2(wfin) - 23
    out['weight_stream'] = {
        'exact_bytes_mean': float(wlen.mean()),
        'exact_minus_idealbits_over_8_mean_B': float(over_w.mean()), 'std_B': float(over_w.std()),
        'final_state_log2_minus_23_mean': float(u.mean()),
        'exact_minus_est_round_bits8_plus4': {'mean': float((wlen - (np.round(wbits / 8) + 4)).mean()),
                                              'frac_equal': float((wlen == np.round(wbits / 8) + 4).mean())},
        'bit_per_w_exact_incl_state_bytes': float(wlen.sum() * 8 / (n * WPR)),
        'bit_per_w_est_bits_plus_4B': float((wbits.sum() + 32 * wlen.size) / (n * WPR)),
        'bit_per_w_ideal_payload_only': float(wbits.sum() / (n * WPR)),
    }
    out['scale_stream'] = {
        'exact_minus_idealbits_over_8_mean_B': float(over_s.mean()),
        'bit_per_scale_exact': float(slen.sum() * 8 / (n * SPR)),
        'bit_per_scale_est_bits_plus_4B': float((sbits.sum() + 32 * slen.size) / (n * SPR)),
    }
    used = 32 + 2 * (wlen.shape[1] + slen.shape[1]) + wlen.sum(1) + slen.sum(1)
    rec = -(-used // 4096) * 4096
    out['record'] = {'exact_bytes_mean': float(rec.mean()), 'exact_ratio': float(rec.mean() / RAW),
                     'padding_mean': float((rec - used).mean()),
                     'est_like_mxanalyze_bytes_mean': float((32 + 2 * 19440 + (wbits.sum(1) + sbits.sum(1)) / 8 + 4 * 19440 + 2048).mean())}
    # hash rule cost
    hs = [r for r in recs if 'wlen_h' in r]
    if hs:
        d = np.concatenate([r['wlen_h'].astype(np.int64) - r['wlen'] for r in hs])
        out['hash_rule_extra_bytes_per_stream'] = {'streams': int(d.size), 'mean': float(d.mean()),
                                                   'theory_E_log2(1+U)/8': float((2 * np.log(2) - 1) / np.log(2) / 8)}
    # length-table residual entropy, predictor as in mxstreams.py: pred = round(sum_blocks E[bits|srel]/8) + 4
    cost_ctx = (base * (PB - np.log2(wf.astype(np.float64)))).sum(2)  # [3, 567]
    blocks = whist.reshape(n, 3, 5760, 7).sum((0, 2))                  # [3, 7]
    eb = np.stack([np.array([cost_ctx[m, s * 81:(s + 1) * 81].sum() / max(1, blocks[m, s]) for s in range(7)]) for m in range(3)])
    s_mean_bits = (sbits.reshape(n, 3, 720).sum((0, 2)) / (n * 720 * 512))  # bits per scale, per matrix (in-sample, avg)
    lt = {}
    for label, sel in [('all_160', np.arange(n)), ('their_24 (manifest[::6][:24])', np.array([k for k, r in enumerate(recs) if r['i'] in set(range(0, 160, 6)[:24])]))]:
        rr = {}
        tot_exact = tot_est = 0.0
        for m in range(3):
            sl = slice(m * 5760, (m + 1) * 5760)
            h = whist[sel][:, sl]                     # [k, 5760, 7]
            pred = np.round((h * eb[m]).sum(-1) / 8) + 4
            ex = wlen[sel][:, sl]
            es = np.round(wbits[sel][:, sl] / 8) + 4
            hw_ex = ent((ex - pred).ravel()); hw_es = ent((es - pred).ravel())
            ssl = slice(m * 720, (m + 1) * 720)
            spred = np.round(s_mean_bits[m] * 512 / 8) + 4
            hs_ex = ent((slen[sel][:, ssl] - spred).ravel())
            hs_es = ent((np.round(sbits[sel][:, ssl] / 8) + 4 - spred).ravel())
            rr[f'm{m}'] = {'w_resid_bits_exact': hw_ex, 'w_resid_bits_est': hw_es,
                           'w_resid_std_exact': float((ex - pred).std()), 's_resid_bits_exact': hs_ex, 's_resid_bits_est': hs_es,
                           'w_raw_len_entropy_exact': ent(ex.ravel())}
            tot_exact += 5760 * hw_ex + 720 * hs_ex
            tot_est += 5760 * hw_es + 720 * hs_es
        rr['coded_table_bytes_exact'] = tot_exact / 8
        rr['coded_table_bytes_est'] = tot_est / 8
        rr['saving_pct_of_raw_exact'] = 100 * (38880 - tot_exact / 8) / RAW
        # simple parallel-friendly alternative: fixed-width residual field per stream (bits = ceil(log2(range)))
        lt[label] = rr
    out['length_table'] = lt
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
