"""ceil-refuter, resumed run: counts that let the analysis choose ANY srel anchor rule offline.

Usage: python -I -B anchor_count.py DATA_DIR OUT_DIR GROUPING     (GROUPING 'sample' | 'extra', as refute_count.py)

Per expert (OUT_DIR/pe/<file>.npz, resumable, one file per expert):
  ao, ac  int32 [3, 24, 81, 16]   weight counts by ABSOLUTE scale bin (s-104 clipped 0..23), (pm*9+mx), symbol;
                                  ao = original form, ac = ceil form (T applied). srel counts for any anchor a are
                                  sums of these scale slices (clip(s-a,-3,3)), so median / pair / mode / per-matrix
                                  search rules can all be evaluated without re-reading the data.
  so, sc  int32 [3, 256]          scale histograms (original, ceil form)
  soc     int32 [3, 3, 256]       original scales by block class (0 max6, 1 max7 with 1/3, 2 even)
  scb     int32 [3, 3, 256]       ceil scales by ceil block max (5, 6, 7)
  clip_o, clip_c                  number of scales outside 104..127 (must be 0, else the abs bins are not exact)
Group sums (OUT_DIR/agg_<GROUPING>.npz): o_clsh_pm2b_<rule> int64 [6, 3, 20412, 16] for rule in (med, pair) on the
  original form = (cls3, half, srel(rule), pm, mx, B6[pm2]); marginalise pm2b / half / cls offline.
  side_cls_<rule> [6, 3, 7, 3] (class given srel).
Anchor rules: med = lower median (WNR2 today, wex_nr2_median); pair = lower value v of the densest adjacent pair
  (count[v] + count[v+1], lowest v on ties; team 3's fix); mode = most frequent value (lowest on ties).
"""
import hashlib, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refute_count as rc  # T_fwd / T_inv / LUTs / unpack / B6 (packed-byte implementation, checked in the first run)

S0 = 104
NS = 24
NG = 6


def anchor(hist, rule):
    h = np.asarray(hist, np.int64)
    if rule == 'med':
        c = np.cumsum(h)
        n = int(c[-1])
        return int(np.searchsorted(c, (n - 1) // 2 + 1))  # lower median: index of element (n-1)//2 in sorted order
    if rule == 'pair':
        p = h[:-1] + h[1:]
        return int(np.argmax(p))
    if rule == 'mode':
        return int(np.argmax(h))
    raise ValueError(rule)


def pm_mx(sym):
    mag = (sym & 7).astype(np.int32)
    pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
    cm = np.maximum.accumulate(mag, axis=1)
    mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
    pm2 = np.empty_like(mag); pm2[:, :2] = 8; pm2[:, 2:] = mag[:, :-2]
    return pm * 9 + mx, rc.B6[pm2], cm[:, -1]


def main():
    data_dir, out_dir, grouping = sys.argv[1:4]
    pe_dir = os.path.join(out_dir, 'pe')
    os.makedirs(pe_dir, exist_ok=True)
    man = json.load(open(os.path.join(data_dir, 'manifest.json')))
    ap = os.path.join(out_dir, f'agg_{grouping}.npz')
    jp = os.path.join(out_dir, f'agg_{grouping}.json')
    done = json.load(open(jp)) if os.path.exists(jp) and os.path.exists(ap) else {}
    acc = {f'o_clsh_pm2b_{r}': np.zeros((NG, 3, 20412, 16), np.int64) for r in ('med', 'pair')}
    acc.update({f'side_cls_{r}': np.zeros((NG, 3, 7, 3), np.int64) for r in ('med', 'pair')})
    if done:
        z = np.load(ap)
        for k in acc:
            acc[k] = z[k].astype(np.int64)
    half = (np.arange(32, dtype=np.int32) >= 16)[None, :].astype(np.int32)
    t0 = time.time()
    todo = [(i, e) for i, e in enumerate(man['experts']) if e['file'] not in done]
    for n, (i, e) in enumerate(todo):
        g = (i % 4) if grouping == 'sample' else (4 if e.get('kind', 'layers') == 'layers' else 5)
        raw = np.fromfile(os.path.join(data_dir, e['file']), dtype=np.uint8)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == e['sha256'], e['file']
        parts, off = {}, 0
        for t in e['tensors']:
            parts[t['name']] = raw[off:off + t['bytes']]; off += t['bytes']
        pe = {'ao': np.zeros((3, NS, 81, 16), np.int32), 'ac': np.zeros((3, NS, 81, 16), np.int32),
              'so': np.zeros((3, 256), np.int32), 'sc': np.zeros((3, 256), np.int32),
              'soc': np.zeros((3, 3, 256), np.int32), 'scb': np.zeros((3, 3, 256), np.int32)}
        info = {'index': i, 'group': g, 'kind': e.get('kind', 'layers'), 'layer': e['layer'], 'expert': e['expert'], 'm': []}
        clip_o = clip_c = 0
        for m, w in enumerate(['w1', 'w2', 'w3']):
            blocks = parts[f'{w}.weight'].reshape(-1, 16)
            sc = parts[f'{w}.scale'].reshape(-1)
            tw, ts, even, bmax = rc.T_fwd(blocks, sc)
            assert int((bmax == 5).sum()) == 0 and int(ts.max()) <= 255
            sym = rc.unpack(blocks); s32 = sym.astype(np.int32)
            ctx, b6, cmax = pm_mx(sym)
            ab = sc.astype(np.int32) - S0
            clip_o += int(((ab < 0) | (ab >= NS)).sum())
            ab = np.clip(ab, 0, NS - 1)
            pe['ao'][m] = np.bincount(((ab[:, None] * 81 + ctx) * 16 + s32).ravel(), minlength=NS * 81 * 16).reshape(NS, 81, 16)
            cls = np.where(cmax == 7, np.where(even, 2, 1), 0).astype(np.int32)
            pe['so'][m] = np.bincount(sc, minlength=256)
            pe['soc'][m] = np.bincount(cls * 256 + sc, minlength=768).reshape(3, 256)
            mi = {'anchors_o': {r: anchor(pe['so'][m], r) for r in ('med', 'pair', 'mode')}}
            for r in ('med', 'pair'):
                a = mi['anchors_o'][r]
                srel = np.clip(sc.astype(np.int32) - a, -3, 3) + 3
                base = (srel[:, None] * 81 + ctx)
                cc = (((cls[:, None] * 2 + half) * 567 + base) * 6 + b6)
                acc[f'o_clsh_pm2b_{r}'][g, m] += np.bincount((cc * 16 + s32).ravel(), minlength=20412 * 16).reshape(20412, 16)
                acc[f'side_cls_{r}'][g, m] += np.bincount(srel * 3 + cls, minlength=21).reshape(7, 3)
                del base, cc
            del sym, s32, ctx, b6
            tsym = rc.unpack(tw); t32 = tsym.astype(np.int32)
            ctx, _, cmax_c = pm_mx(tsym)
            abc = ts.astype(np.int32) - S0
            clip_c += int(((abc < 0) | (abc >= NS)).sum())
            abc = np.clip(abc, 0, NS - 1)
            pe['ac'][m] = np.bincount(((abc[:, None] * 81 + ctx) * 16 + t32).ravel(), minlength=NS * 81 * 16).reshape(NS, 81, 16)
            pe['sc'][m] = np.bincount(ts.astype(np.int32), minlength=256)[:256]
            pe['scb'][m] = np.bincount((cmax_c - 5).astype(np.int32) * 256 + ts.astype(np.int32), minlength=768)[:768].reshape(3, 256)
            mi['anchors_c'] = {r: anchor(pe['sc'][m], r) for r in ('med', 'pair', 'mode')}
            mi['even'] = int(even.sum())
            info['m'].append(mi)
            del tsym, t32, ctx
        info['clip_o'] = clip_o; info['clip_c'] = clip_c
        np.savez_compressed(os.path.join(pe_dir, e['file'].replace('.bin', '.npz')), **pe)
        done[e['file']] = info
        print(f"{n + 1}/{len(todo)} {e['file']} g{g} anchors_o {[x['anchors_o'] for x in info['m']]} "
              f"anchors_c {[x['anchors_c'] for x in info['m']]} clip {clip_o},{clip_c} {time.time() - t0:.0f}s", flush=True)
        if (n + 1) % 8 == 0 or n + 1 == len(todo):
            np.savez_compressed(ap + '.tmp.npz', **acc)
            json.dump(done, open(jp + '.tmp', 'w'))
            os.replace(ap + '.tmp.npz', ap)
            os.replace(jp + '.tmp', jp)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
