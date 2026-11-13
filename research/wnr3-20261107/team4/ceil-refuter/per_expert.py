"""Per-expert held-out estimate of the ceil-form record gain (base 567 model + order-0 scales), all routed experts.
Sample experts of split s (index%2) are costed with tables trained on the other split; extra experts with tables
trained on the even split. Estimates match the real rANS coder to <0.002% of raw (e2e_est.json).
Usage: python -I -B per_expert.py COUNTS_SAMPLE_NPZ DATA_DIR SETNAME OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refute_count as rc
from refute_an import quantize, cost_bits


def main():
    cnp, d, setname, outp = sys.argv[1:5]
    z = np.load(cnp)
    tabs = {s: {k: -np.log2(quantize(z[k][g].sum(0)) / 16384) for k in ['o_base', 'c_base', 's_o', 's_c']}
            for s, g in [('even', [0, 2]), ('odd', [1, 3])]}
    man = json.load(open(os.path.join(d, 'manifest.json')))
    out = []
    for i, e in enumerate(man['experts']):
        if e.get('kind', 'layers') != 'layers':
            continue
        t = tabs['odd'] if (setname == 'sample' and i % 2 == 0) else tabs['even']
        raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
        off, parts = 0, {}
        for x in e['tensors']:
            parts[x['name']] = raw[off:off + x['bytes']]; off += x['bytes']
        bo = bc = so = scl = 0.0
        even_n = 0
        for m, w in enumerate(['w1', 'w2', 'w3']):
            blocks = parts[f'{w}.weight'].reshape(-1, 16); sc = parts[f'{w}.scale']
            tw, ts, even, _ = rc.T_fwd(blocks, sc)
            even_n += int(even.sum())
            for form, bb, ss in [('o', blocks, sc.astype(np.int32)), ('c', tw, ts.astype(np.int32))]:
                sym = rc.unpack(bb)
                base, _, _, _ = rc.ctx_arrays(sym, ss, rc.lower_median(ss))
                cnt = np.bincount((base * 16 + sym.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)
                sh = np.bincount(ss, minlength=256)[:256]
                if form == 'o':
                    bo += float((cnt * t['o_base'][m]).sum()); so += float((sh * t['s_o'][m]).sum())
                else:
                    bc += float((cnt * t['c_base'][m]).sum()); scl += float((sh * t['s_c'][m]).sum())
        r = {'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'even_pct': 100 * even_n / 552960,
             'orig_bpw': bo / 35389440, 'ceil_bpw': bc / 35389440,
             'd_record_pct_of_raw': 100 * ((bc + scl) - (bo + so)) / 8 / 18800640}
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main()
