"""ceil-refuter: does ceil-form AMPLIFY silent corruption? Random single-bit flips in real weight streams, decoded with
today's structural rule (team 2 audit decoder, copy audit_core_t2copy.py), in the original form (today's model) and in
the ceil form (ceil-trained model) followed by the inverse transform T^-1.

Usage: python -I -B flip_ceil.py SAMPLE_DIR COUNTS_SAMPLE_NPZ OUT_JSON [streams_per_expert] [experts]
Models: 567-context tables trained on sample groups {0, 2} (refute_count.py counts o_base / c_base); test experts are
odd-index sample experts (held-out). For every accepted (= silently wrong) flip it reports, in the ORIGINAL byte domain
after T^-1: number of changed weights, changed scales, and whether the damage is one sign flip.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core_t2copy import cum_of, encode_vec, decode_vec, inv_table, quantize
import refute_count as rc

DBL = np.array([0, 2, 4, 5, 6, 7, 6, 7], np.uint8)


def inv_nibbles(sym, ts):
    """sym uint8 [N, 2048] decoded coded-form nibbles, ts int [N, 64] coded scales -> original-form nibbles, scales."""
    b = sym.reshape(-1, 32)
    mag = b & 7
    dec = mag.max(1) == 5
    out = b.copy()
    out[dec] = (b[dec] & 8) | DBL[mag[dec]]
    s = ts.reshape(-1) - dec
    return out.reshape(sym.shape), s.reshape(ts.shape)


def ctx_of(sym32, srel):
    mag = (sym32 & 7).astype(np.int64)
    pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
    cm = np.maximum.accumulate(mag, axis=1)
    mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
    return (srel[:, None].astype(np.int64) * 9 + pm) * 9 + mx


def main():
    sdir, cnp, outp = sys.argv[1:4]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    nex = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    z = np.load(cnp)
    tabs = {}
    for form, key in (('orig', 'o_base'), ('ceil', 'c_base')):
        wf = quantize(z[key][[0, 2]].sum(0))
        tabs[form] = (wf, cum_of(wf), [inv_table(wf[m]) for m in range(3)])
    man = json.load(open(os.path.join(sdir, 'manifest.json')))
    odd = [i for i in range(len(man['experts'])) if i % 2 == 1]
    picks = [odd[k] for k in np.linspace(0, len(odd) - 1, nex).astype(int)]
    rng = np.random.default_rng(20261107)
    out = {'experts': [man['experts'][i]['file'] for i in picks], 'streams_per_expert': per}
    for m in (0, 1):
        w = ('w1', 'w2', 'w3')[m]
        S = {'orig': [], 'ceil': []}; SR = {'orig': [], 'ceil': []}; SC = {'orig': [], 'ceil': []}
        for i in picks:
            e = man['experts'][i]
            raw = np.fromfile(os.path.join(sdir, e['file']), dtype=np.uint8)
            off, parts = 0, {}
            for t in e['tensors']:
                parts[t['name']] = raw[off:off + t['bytes']]; off += t['bytes']
            blocks = parts[f'{w}.weight'].reshape(-1, 16); sc = parts[f'{w}.scale'].reshape(-1)
            tw, ts, even, _ = rc.T_fwd(blocks, sc)
            ns = blocks.shape[0] // 64
            k = rng.choice(ns, per, replace=False)
            for form, bb, ss in (('orig', blocks, sc.astype(np.int64)), ('ceil', tw, ts.astype(np.int64))):
                med = rc.lower_median(ss)
                sym = rc.unpack(bb).reshape(ns, 2048)
                srel = (np.clip(ss - med, -3, 3) + 3).reshape(ns, 64)
                S[form].append(sym[k]); SR[form].append(srel[k]); SC[form].append(ss.reshape(ns, 64)[k])
        res = {}
        for form in ('orig', 'ceil'):
            wf, wc, invs = tabs[form]
            sym = np.concatenate(S[form]).astype(np.int64); srel = np.concatenate(SR[form]); ss = np.concatenate(SC[form])
            N = sym.shape[0]
            ctx = ctx_of(sym.reshape(-1, 32), np.repeat(srel.reshape(-1), 1)).reshape(N, 2048)
            F = wf[m][ctx, sym]; C = wc[m][ctx, sym]
            lens, _, buf = encode_vec(F, C, want_bytes=True)
            ar = np.arange(N)
            t0 = time.time()
            v0, d0, _ = decode_vec(buf, lens, 2048, srel, wf[m], wc[m], invs[m])
            b = buf.copy(); j = (rng.random(N) * lens).astype(np.int64)
            b[ar, j] ^= (1 << rng.integers(0, 8, N)).astype(np.uint8)
            v, ds, _ = decode_vec(b, lens, 2048, srel, wf[m], wc[m], invs[m])
            acc = np.where(v)[0]
            r = {'trials': int(N), 'clean_accepted': int(v0.sum()), 'clean_symbols_ok': bool((d0 == sym).all()),
                 'flip_accepted': int(v.sum()), 'accepted_rate': float(v.mean())}
            dmg = []
            for a in acc:
                dsym = ds[a].astype(np.uint8)[None, :]; osym = sym[a].astype(np.uint8)[None, :]
                if form == 'ceil':
                    o_n, o_s = inv_nibbles(osym, ss[a][None, :]); d_n, d_s = inv_nibbles(dsym, ss[a][None, :])
                else:
                    o_n, o_s, d_n, d_s = osym, ss[a][None, :], dsym, ss[a][None, :]
                diff = np.where(o_n[0] != d_n[0])[0]
                sign_only = bool(diff.size == 1 and (int(o_n[0][diff[0]]) ^ int(d_n[0][diff[0]])) == 8)
                dmg.append({'coded_symbols_changed': int((dsym != osym).sum()), 'orig_domain_weights_changed': int(diff.size),
                            'orig_domain_scales_changed': int((o_s != d_s).sum()), 'single_sign_flip': sign_only})
            r['accepted_damage'] = dmg
            r['all_accepted_are_single_sign_flips_in_orig_domain'] = all(d['single_sign_flip'] and d['orig_domain_scales_changed'] == 0 for d in dmg)
            r['max_weights_changed'] = max([d['orig_domain_weights_changed'] for d in dmg], default=0)
            r['seconds'] = time.time() - t0
            res[form] = {k2: v2 for k2, v2 in r.items()}
            print(w, form, json.dumps({k2: v2 for k2, v2 in r.items() if k2 != 'accepted_damage'}), flush=True)
        out[w] = res
    json.dump(out, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main()
