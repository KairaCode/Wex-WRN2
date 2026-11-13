"""WNR2 / WNR3 expected ratios on other MXFP4 MoE checkpoints, own models (held-out) and V4.1 model transfer.

Usage: python -I ana7.py STATS_DIR OUT_JSON
Input: STATS_DIR/<label>/*.npz + index.json from stats7.py. All costs are code-length estimates with the exact WNR2
14-bit quantiser (wex_nr2_quantize: 1 + floor(c*(16384-n)/tot), largest remainders, ties to the lower symbol).

Record byte model (per item, Cw = 2048, Cs = 512 like V4.1):
  WNR2  : 32 B header + pad4(2 B x streams) u16 lengths + bits/8 + 3.5 B flush x streams, align 4096.
  WNR3  : 64 B header + LENC (12 + 0.6748 B x streams + 4 B x groups; V4.1 ref-impl 13,118.6 B / 19,440 streams plus
          the spec's base offsets) + bits/8 + 3.5 B flush x streams + HASH 0.0696 B x streams (V4.1 +1,352 B), align 4096.
Model: weights 567 contexts x 16 per matrix type t (w1, w2, w3), scales order-0 256 per type (WNR2 / WNM3 layout 0).
Anchor rules: 'median' = lower median of the matrix's (coded) scales (WNR2); 'g0' = anchor0[t] of the set =
  mode over the set's training matrices of type t of the densest-pair lower value argmax_v h[v]+h[v+1] (spec 5.3).
WNR3 v1 sets (spec 3.4 / 5.10): K = 2 (set 0 ceil, trained on T(train matrices passing the gate); set 1 plain, same
  matrices untransformed) iff the container even-block share >= 1%, else K = 1 plain. Per matrix: ceil set iff K = 2,
  gate passes (no stored block with max 5) and the matrix even share >= 1%.
Held-out: items sorted by layer, split by parity; each half is costed with the model trained on the other half.
"""
import json, math, os, sys
import numpy as np

P = 16384
CW, CS = 2048, 512
LENC_PER_STREAM = 13118.6 / 19440
HASH_PER_STREAM = 1352.0 / 19440
FLUSH = 3.5

CHECKPOINTS = [
    # name, labels, origin, repo@rev, MoE layers, experts per layer, top-k (config)
    ('DeepSeek-V4.1-Flash REAP-192 (KAIRA)', ['v41sample'], 'QAT FP4 (DeepSeek), REAP repack', 40, 192, 6),
    ('DeepSeek-V4-Flash', ['ds_v4flash'], 'QAT FP4 (DeepSeek)', 43, 256, 6),
    ('DeepSeek-V4-Pro', ['ds_v4pro'], 'QAT FP4 (DeepSeek)', 61, 384, 6),
    ('openai/gpt-oss-120b', ['gptoss120'], 'QAT MXFP4 (OpenAI)', 36, 128, 4),
    ('openai/gpt-oss-20b', ['gptoss20'], 'QAT MXFP4 (OpenAI)', 24, 32, 4),
    ('amd/gpt-oss-120b-w-mxfp4-a-fp8', ['amd_gptoss120'], 'Quark re-export of gpt-oss-120b (-0 -> +0)', 36, 128, 4),
    ('amd/Kimi-K2.6-MXFP4', ['kimi26'], 'PTQ (Quark) of INT4-QAT Kimi-K2.6 via BF16', 60, 384, 8),
    ('amd/MiniMax-M3-MXFP4', ['minimax3'], 'PTQ (Quark) from BF16', 57, 128, 4),
    ('RedHatAI/GLM-5.3-MXFP4', ['rh_glm53', 'glm53x'], 'PTQ (llm-compressor) from FP8', 75, 256, 8),
    ('amd/DeepSeek-R1-0528-MXFP4', ['amd_r1', 'r1x'], 'PTQ (Quark) from FP8 via BF16', 58, 256, 8),
]
QAT_POOL = ['DeepSeek-V4.1-Flash REAP-192 (KAIRA)', 'DeepSeek-V4-Flash', 'DeepSeek-V4-Pro', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b']
PTQ_POOL = ['amd/Kimi-K2.6-MXFP4', 'amd/MiniMax-M3-MXFP4', 'RedHatAI/GLM-5.3-MXFP4', 'amd/DeepSeek-R1-0528-MXFP4']


def quantize(c):
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
    freq[tot[:, 0] == 0] = P // n
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


LOG = None


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * (14.0 - np.log2(np.asarray(freq, np.float64)))).sum())


def lower_median(h):
    cs = np.cumsum(h)
    return int(np.searchsorted(cs, (cs[-1] - 1) // 2, side='right'))


def densest_pair(h):
    h = np.asarray(h, np.int64)
    return int(np.argmax(h[:-1] + h[1:]))


def ctx567(C, smin, anchor):
    ns = C.shape[0]
    srel = np.clip(smin + np.arange(ns) - anchor, -3, 3) + 3
    out = np.zeros((7, 81, 16), np.int64)
    np.add.at(out, srel, C)
    return out.reshape(567, 16)


def zmerge(c):
    c = np.array(c, np.int64, copy=True)
    c[..., 0] += c[..., 8]
    c[..., 8] = 0
    return c


def mode(vals):
    v, c = np.unique(np.asarray(vals), return_counts=True)
    return int(v[np.argmax(c)])


# ---------------------------------------------------------------- data
def load_label(sd, label):
    idx = json.load(open(os.path.join(sd, label, 'index.json')))
    items = []
    for it in idx['items']:
        if label == 'v41extra' and it.get('kind') not in (None, 'layers'):
            continue
        z = np.load(os.path.join(sd, label, it['npz']))
        mats = []
        for m in range(3):
            g = lambda k: z[f'm{m}_{k}']
            R, C = g('shape').tolist()
            scal = g('scal').tolist()
            mat = {'t': m, 'R': R, 'C': C, 'n': R * C, 'nsc': R * C // 32, 'bmax': g('bmax'), 'mask': g('mask'), 'sym': g('sym'),
                   'even': scal[0], 'blocks': scal[1], 'max5': scal[2], 'max_lt6': scal[3], 'even_s255': scal[4],
                   'max6_no13': scal[5], 'gate_ok': bool(scal[6]),
                   'p': (int(g('smin_p')), g('Cp'), g('sc_p'))}
            mat['c'] = (int(g('smin_c')), g('Cc'), g('sc_c')) if mat['gate_ok'] else None
            mat['even_share'] = mat['even'] / mat['blocks']
            mats.append(mat)
        raw = sum(mm['n'] // 2 + mm['nsc'] for mm in mats)
        nst = sum(math.ceil(mm['n'] / CW) + math.ceil(mm['nsc'] / CS) for mm in mats)
        ngr = sum(math.ceil(math.ceil(mm['n'] / CW) / 32) + math.ceil(math.ceil(mm['nsc'] / CS) / 32) for mm in mats)
        items.append({'tag': f"{label}:{it['tag']}", 'layer': it.get('layer'), 'mats': mats, 'raw': raw, 'nst': nst, 'ngr': ngr})
    return items


def record_bytes(fmt, bits, item):
    nst = item['nst']
    if fmt == 'wnr2':
        used = 32 + 4 * math.ceil(2 * nst / 4) + bits / 8 + FLUSH * nst
    else:
        used = 64 + (12 + LENC_PER_STREAM * nst + 4 * item['ngr']) + bits / 8 + FLUSH * nst + HASH_PER_STREAM * nst
    return used, 4096 * math.ceil(used / 4096)


# ---------------------------------------------------------------- models
def mat_anchor_median(mat, form):
    return lower_median(mat[form][2])


def train_set(items, form, amode, cz, anchor0=None, weight_norm=None):
    """Returns set dict: form, amode, anchor0[3], W[3] freq 567x16, S[3] freq 256, counts kept for pooling."""
    mats = [mm for it in items for mm in it['mats'] if mm[form] is not None]
    if amode == 'g0' and anchor0 is None:
        anchor0 = [mode([densest_pair(mm[form][2]) for mm in mats if mm['t'] == t]) for t in range(3)]
    Wc = [np.zeros((567, 16), np.int64) for _ in range(3)]
    Sc = [np.zeros(256, np.int64) for _ in range(3)]
    for mm in mats:
        t = mm['t']
        smin, C, sh = mm[form]
        a = anchor0[t] if amode == 'g0' else lower_median(sh)
        Wc[t] += ctx567(C, smin, a)
        Sc[t] += sh
    if cz:
        Wc = [zmerge(w) for w in Wc]
    return {'form': form, 'amode': amode, 'cz': cz, 'anchor0': anchor0, 'Wc': Wc, 'Sc': Sc,
            'W': [quantize(w) for w in Wc], 'S': [quantize(s) for s in Sc]}


def with_tables(base, W=None, S=None, anchor0=None):
    s = dict(base)
    if W is not None:
        s['W'] = W
    if S is not None:
        s['S'] = S
    if anchor0 is not None:
        s['anchor0'] = list(anchor0)
    return s


def choose_set(mat, sets, K):
    """sets: {'c': ceil set or None, 'p': plain set}. Returns key."""
    if K == 2 and sets.get('c') is not None and mat['gate_ok'] and mat['even_share'] >= 0.01:
        return 'c'
    return 'p'


def cost_item(item, sets, K, fmt):
    bw = bs = 0.0
    info = []
    for mm in item['mats']:
        k = choose_set(mm, sets, K)
        st = sets[k]
        form = st['form']
        smin, C, sh = mm[form]
        a = st['anchor0'][mm['t']] if st['amode'] == 'g0' else lower_median(sh)
        cc = ctx567(C, smin, a)
        if st['cz']:
            cc = zmerge(cc)
        w = cost_bits(cc, st['W'][mm['t']])
        s = cost_bits(sh, st['S'][mm['t']])
        bw += w; bs += s
        info.append({'set': k, 'anchor': a, 'dp': densest_pair(sh), 'med': lower_median(sh), 'w_bits': w, 's_bits': s})
    used, rec = record_bytes(fmt, bw + bs, item)
    return {'bw': bw, 'bs': bs, 'used': used, 'rec': rec, 'raw': item['raw'], 'info': info}


def summarize(rows, items):
    nW = sum(mm['n'] for it in items for mm in it['mats'])
    nS = sum(mm['nsc'] for it in items for mm in it['mats'])
    nst = sum(it['nst'] for it in items)
    raw = sum(r['raw'] for r in rows)
    per = np.array([r['rec'] / r['raw'] for r in rows])
    peru = np.array([r['used'] / r['raw'] for r in rows])
    return {'ratio_rec': sum(r['rec'] for r in rows) / raw, 'ratio_used': sum(r['used'] for r in rows) / raw,
            'w_bpw_no_flush': sum(r['bw'] for r in rows) / nW, 's_bps_no_flush': sum(r['bs'] for r in rows) / nS,
            'per_item_used_min_max': [float(peru.min()), float(peru.max())], 'per_item_used_se': float(peru.std(ddof=1) / math.sqrt(len(peru))) if len(peru) > 1 else None,
            'n_items': len(rows)}


def boot_ci(x, n=4000, seed=1):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ---------------------------------------------------------------- configs
def build_sets(train, cfg, K, anchor0s=None):
    """cfg in wnr2, wnr2g0, fmed, fg0, fg0c. anchor0s: optional {'p': [...], 'c': [...]} fixed anchors."""
    a = anchor0s or {}
    if cfg == 'wnr2':
        return {'p': train_set(train, 'p', 'median', False)}, 1, 'wnr2'
    if cfg == 'wnr2g0':
        return {'p': train_set(train, 'p', 'g0', False, a.get('p'))}, 1, 'wnr2'
    cz = cfg == 'fg0c'
    amode = 'median' if cfg == 'fmed' else 'g0'
    sets = {'p': train_set(train, 'p', amode, cz, a.get('p') if amode == 'g0' else None)}
    if K == 2:
        sets['c'] = train_set(train, 'c', amode, cz, a.get('c') if amode == 'g0' else None)
    return sets, K, 'wnr3'


CFGS = ['wnr2', 'wnr2g0', 'fmed', 'fg0', 'fg0c']


def heldout(items, cfg, K):
    A = [it for i, it in enumerate(items) if i % 2 == 0]
    B = [it for i, it in enumerate(items) if i % 2 == 1]
    rows = [None] * len(items)
    anchors = {}
    for test_idx, train in [(1, A), (0, B)]:
        sets, KK, fmt = build_sets(train, cfg, K)
        anchors['trained_on_' + ('even' if test_idx == 1 else 'odd')] = {k: v['anchor0'] for k, v in sets.items()}
        for i, it in enumerate(items):
            if i % 2 == test_idx:
                rows[i] = cost_item(it, sets, KK, fmt)
    return rows, anchors


def main():
    sd, out_json = sys.argv[1], sys.argv[2]
    data = {}
    for name, labels, origin, nl, ne, topk in CHECKPOINTS:
        its = []
        for lb in labels:
            if os.path.exists(os.path.join(sd, lb, 'index.json')):
                its += load_label(sd, lb)
        its.sort(key=lambda it: (it['layer'] if it['layer'] is not None else 0, it['tag']))
        data[name] = its
        print(name, len(its), 'items', flush=True)
    v41x = load_label(sd, 'v41extra') if os.path.exists(os.path.join(sd, 'v41extra', 'index.json')) else []
    res = {'notes': __doc__, 'checkpoints': {}}

    # ---------------- V4.1 transfer model: all 160 sample experts
    v41 = data['DeepSeek-V4.1-Flash REAP-192 (KAIRA)']
    v41_sets = {cfg: build_sets(v41, cfg, 1 if cfg in ('wnr2', 'wnr2g0') else 2) for cfg in CFGS}
    res['v41_model_anchor0'] = {cfg: {k: v['anchor0'] for k, v in s[0].items()} for cfg, s in v41_sets.items()}
    print('V4.1 model anchors', res['v41_model_anchor0'], flush=True)

    for name, labels, origin, nl, ne, topk in CHECKPOINTS:
        items = data[name]
        if not items:
            continue
        R = {'origin': origin, 'labels': labels, 'n_items': len(items), 'moe_layers': nl, 'experts_per_layer': ne, 'top_k': topk}
        it0 = items[0]
        R['geometry'] = {'matrices_RxC': [[mm['R'], mm['C']] for mm in it0['mats']], 'record_raw_bytes': it0['raw'],
                         'record_raw_aligned4096': 4096 * math.ceil(it0['raw'] / 4096), 'streams_per_record': it0['nst'],
                         'records': nl * ne, 'container_raw_GB': nl * ne * it0['raw'] / 1e9,
                         'expert_bytes_per_token_all_miss_GB': topk * nl * it0['raw'] / 1e9}
        assert all(it['raw'] == it0['raw'] for it in items), name
        # ---------------- fingerprint
        mats = [mm for it in items for mm in it['mats']]
        blocks = sum(mm['blocks'] for mm in mats)
        even = sum(mm['even'] for mm in mats)
        bmax = sum(mm['bmax'] for mm in mats)
        sym = sum(mm['sym'] for mm in mats).astype(np.float64)
        mask = sum(mm['mask'] for mm in mats)
        mag = np.array([sym[a] + sym[a + 8] for a in range(8)])
        p13 = (mag[1] + mag[3]) / mag.sum()
        top = np.argsort(-mask)[:12]
        fpr = {'blocks': int(blocks), 'even_pct': 100 * even / blocks,
               'even_pct_by_type': [100 * sum(mm['even'] for mm in mats if mm['t'] == t) / sum(mm['blocks'] for mm in mats if mm['t'] == t) for t in range(3)],
               'even_pct_per_item_min_max': [min(100 * sum(m['even'] for m in it['mats']) / sum(m['blocks'] for m in it['mats']) for it in items),
                                             max(100 * sum(m['even'] for m in it['mats']) / sum(m['blocks'] for m in it['mats']) for it in items)],
               'even_pct_per_matrix_min_max': [100 * min(mm['even_share'] for mm in mats), 100 * max(mm['even_share'] for mm in mats)],
               'even_pct_iid_rough': 100 * (bmax[7] / blocks) * (1 - p13) ** 31,
               'bmax_share_pct': (100 * bmax / blocks).tolist(),
               'bmax_share_pct_by_type': [(100 * sum(mm['bmax'] for mm in mats if mm['t'] == t) / sum(mm['blocks'] for mm in mats if mm['t'] == t)).tolist() for t in range(3)],
               'mag_share_pct': (100 * mag / mag.sum()).tolist(),
               'zero_pos_pct': 100 * sym[0] / sym.sum(), 'zero_neg_pct': 100 * sym[8] / sym.sum(),
               'top_block_masks': [{'mags_used': [a for a in range(8) if (int(k) >> a) & 1], 'pct': 100 * float(mask[k]) / blocks} for k in top],
               'max6_no13_pct': 100 * sum(mm['max6_no13'] for mm in mats) / blocks}
        R['fingerprint'] = fpr
        # ---------------- gate per matrix
        K = 2 if even / blocks >= 0.01 else 1
        gate = {'matrices': len(mats), 'gate_pass': sum(mm['gate_ok'] for mm in mats),
                'max5_blocks_total': int(sum(mm['max5'] for mm in mats)), 'max_lt6_blocks_total': int(sum(mm['max_lt6'] for mm in mats)),
                'even_s255_blocks_total': int(sum(mm['even_s255'] for mm in mats)),
                'even_ge_1pct_matrices': sum(mm['even_share'] >= 0.01 for mm in mats),
                'ceil_matrices_v1_rule': sum(1 for mm in mats if K == 2 and mm['gate_ok'] and mm['even_share'] >= 0.01),
                'container_K': K, 'v1_flags': '0x1C (CEIL|LENC|HASH)' if K == 2 else '0x0C (LENC|HASH), K=1 plain',
                'scale_range_stored': [int(min(np.nonzero(mm['p'][2])[0].min() for mm in mats)), int(max(np.nonzero(mm['p'][2])[0].max() for mm in mats))]}
        R['gate'] = gate
        # ---------------- anchors (all items = builder pre-pass)
        anc = {}
        for form in (['p', 'c'] if K == 2 else ['p']):
            fm = [mm for mm in mats if mm[form] is not None and (form == 'p' or mm['even_share'] >= 0.01)]
            per_t = {}
            for t in range(3):
                dps = [densest_pair(mm[form][2]) for mm in fm if mm['t'] == t]
                meds = [lower_median(mm[form][2]) for mm in fm if mm['t'] == t]
                a0 = mode(dps)
                vals, cnts = np.unique(dps, return_counts=True)
                mv, mc = np.unique(meds, return_counts=True)
                per_t[['w1', 'w2', 'w3'][t]] = {'anchor0_mode_all': a0, 'densest_pair_hist': {int(v): int(c) for v, c in zip(vals, cnts)},
                                                'median_hist': {int(v): int(c) for v, c in zip(mv, mc)},
                                                'n': len(dps), 'differ_from_anchor0': int(sum(d != a0 for d in dps)),
                                                'differ_pct': 100 * sum(d != a0 for d in dps) / max(1, len(dps))}
            anc[form] = per_t
        R['anchors_prepass'] = anc
        # ---------------- own model, held-out
        own = {}
        own_rows = {}
        for cfg in CFGS:
            rows, anchors = heldout(items, cfg, K)
            own_rows[cfg] = rows
            s = summarize(rows, items)
            s['anchor0_by_half'] = anchors
            own[cfg] = s
        base = own_rows['wnr2']
        for cfg in CFGS:
            d = [(r['used'] - b['used']) / r['raw'] * 100 for r, b in zip(own_rows[cfg], base)]
            own[cfg]['delta_vs_wnr2_pct_raw_used'] = float(np.mean(d))
            own[cfg]['delta_ci95'] = boot_ci(d)
        # anchor rule check: g0 vs densest pair vs median on the WNR3 sets (estimate, held-out tables trained with g0)
        own['_anchor_note'] = 'fmed = ceil/plain sets with per-matrix median anchors (tables trained with medians); fg0 = spec v1 g0'
        R['own_heldout'] = own
        # per-matrix anchor misalignment in the g0 config
        mis = {'p': [0, 0], 'c': [0, 0]}
        for r in own_rows['fg0']:
            for inf in r['info']:
                mis[inf['set']][0] += 1
                mis[inf['set']][1] += int(inf['dp'] != inf['anchor'])
        R['own_fg0_matrices_by_set_and_dp_ne_anchor'] = mis
        # ---------------- learning curve (default WNR3 config: fg0; test = odd half, train subsets of even half)
        A = [it for i, it in enumerate(items) if i % 2 == 0]
        B = [it for i, it in enumerate(items) if i % 2 == 1]
        lc = {}
        for n in sorted({1, 2, 3, 4, 6, 8, 12, 20, 40, 80, len(A)}):
            if n > len(A):
                continue
            reps = []
            for rot in range(3 if n < len(A) else 1):
                sel = [A[(rot * max(1, len(A) // 3) + round(k * len(A) / n)) % len(A)] for k in range(n)]
                sets, KK, fmt = build_sets(sel, 'fg0', K)
                rows = [cost_item(it, sets, KK, fmt) for it in B]
                reps.append(sum(r['used'] for r in rows) / sum(r['raw'] for r in rows))
            lc[n] = {'ratio_used_mean': float(np.mean(reps)), 'reps': reps}
        R['learning_curve_fg0_used'] = lc
        # ---------------- transfer: V4.1 model unchanged / re-anchored / re-anchored + own scales
        tr = {}
        own_anchor = {'p': [anc['p'][w]['anchor0_mode_all'] for w in ('w1', 'w2', 'w3')]}
        if 'c' in anc:
            own_anchor['c'] = [anc['c'][w]['anchor0_mode_all'] for w in ('w1', 'w2', 'w3')]
        for cfg in CFGS:
            sets, KK, fmt = v41_sets[cfg]
            variants = {'v41_unchanged': sets}
            if cfg != 'wnr2' and cfg != 'fmed':
                variants['v41_reanchored'] = {k: with_tables(v, anchor0=own_anchor.get(k, own_anchor['p'])) for k, v in sets.items()}
            out = {}
            for vn, vs in variants.items():
                rows = [cost_item(it, vs, KK, fmt) for it in items]
                s = summarize(rows, items)
                d = [(r['used'] - o['used']) / r['raw'] * 100 for r, o in zip(rows, own_rows[cfg])]
                s['delta_vs_own_heldout_pct_raw'] = float(np.mean(d)); s['delta_ci95'] = boot_ci(d)
                s['sets_used'] = {k: sum(1 for r in rows for inf in r['info'] if inf['set'] == k) for k in ('p', 'c')}
                out[vn] = s
            # own scales (held-out) with V4.1 weight tables (re-anchored when g0)
            rows = [None] * len(items)
            for test_idx, train in [(1, A), (0, B)]:
                own_sets, _, _ = build_sets(train, cfg, K)
                mix = {}
                for k, v in sets.items():
                    src = own_sets.get(k, own_sets['p'])
                    mix[k] = with_tables(v, S=src['S'], anchor0=(own_anchor.get(k, own_anchor['p']) if v['amode'] == 'g0' else None))
                for i, it in enumerate(items):
                    if i % 2 == test_idx:
                        rows[i] = cost_item(it, mix, KK, fmt)
            s = summarize(rows, items)
            d = [(r['used'] - o['used']) / r['raw'] * 100 for r, o in zip(rows, own_rows[cfg])]
            s['delta_vs_own_heldout_pct_raw'] = float(np.mean(d)); s['delta_ci95'] = boot_ci(d)
            out['v41_weights_reanchored_own_scales'] = s
            tr[cfg] = out
        R['transfer_v41'] = tr
        res['checkpoints'][name] = R
        print(name, 'K', K, {c: round(own[c]['ratio_used'], 5) for c in CFGS},
              {c: {v: round(x['ratio_used'], 5) for v, x in tr[c].items()} for c in ('wnr2', 'fg0', 'fg0c')}, flush=True)

    # ---------------- V4.1 model on 70 extra routed V4.1 experts (independent held-out check of the harness)
    if v41x:
        ext = {}
        for cfg in CFGS:
            sets, KK, fmt = v41_sets[cfg]
            rows = [cost_item(it, sets, KK, fmt) for it in v41x]
            ext[cfg] = summarize(rows, v41x)
        res['v41_model_on_v41extra'] = ext
        print('v41extra', {c: round(ext[c]['ratio_used'], 5) for c in CFGS}, flush=True)

    # ---------------- universal models, leave one checkpoint out (weight tables pooled with equal weight per checkpoint,
    # each checkpoint's counts at its own g0 anchors; target uses own anchors and own held-out scale tables)
    def pooled_sets(names, cfg):
        acc = {}
        for nm in names:
            its = data[nm]
            K = 2 if sum(mm['even'] for it in its for mm in it['mats']) / sum(mm['blocks'] for it in its for mm in it['mats']) >= 0.01 else 1
            sets, _, _ = build_sets(its, cfg, 2)
            for k, v in sets.items():
                if k == 'c' and K == 1:
                    continue
                for t in range(3):
                    w = v['Wc'][t].astype(np.float64)
                    acc.setdefault(k, [np.zeros((567, 16)) for _ in range(3)])[t] += w / w.sum() * 1e9
        return {k: [quantize(np.rint(w).astype(np.int64)) for w in v] for k, v in acc.items()}

    uni = {}
    for pool_name, pool in [('QAT', QAT_POOL), ('PTQ', PTQ_POOL), ('ALL', QAT_POOL + PTQ_POOL)]:
        for target in pool:
            items = data[target]
            if not items:
                continue
            K = res['checkpoints'][target]['gate']['container_K']
            others = [n for n in pool if n != target and data[n]]
            if target == 'openai/gpt-oss-120b' or target == 'amd/gpt-oss-120b-w-mxfp4-a-fp8':
                others = [n for n in others if n not in ('openai/gpt-oss-120b', 'amd/gpt-oss-120b-w-mxfp4-a-fp8')]
            out = {}
            for cfg in ['fg0', 'fg0c']:
                PW = pooled_sets(others, cfg)
                A = [it for i, it in enumerate(items) if i % 2 == 0]
                B = [it for i, it in enumerate(items) if i % 2 == 1]
                rows = [None] * len(items)
                for test_idx, train in [(1, A), (0, B)]:
                    own_sets, KK, fmt = build_sets(train, cfg, K)
                    mix = {}
                    for k, v in own_sets.items():
                        mix[k] = with_tables(v, W=PW.get(k, PW['p']))
                    for i, it in enumerate(items):
                        if i % 2 == test_idx:
                            rows[i] = cost_item(it, mix, KK, fmt)
                s = summarize(rows, items)
                ownr = res['checkpoints'][target]['own_heldout'][cfg]['ratio_used']
                s['delta_vs_own_pct_raw'] = 100 * (s['ratio_used'] - ownr)
                s['pool'] = others
                out[cfg] = s
            uni[f'{pool_name}:{target}'] = out
            print('universal', pool_name, target, {c: round(out[c]['ratio_used'], 5) for c in out}, flush=True)
    res['universal_loco'] = uni

    json.dump(res, open(out_json, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('wrote', out_json)


if __name__ == '__main__':
    main()
