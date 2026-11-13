"""team2/audit pass 3: EXACT rANS stream lengths (vectorized encoder, audit_core) for every weight and scale stream
of every sampled expert, under the in-sample static model built from audit_count.py counts (like production:
one model trained on all records of the container). Also re-encodes some experts with x0 = L + h (hash rule cost).

Usage: python -I audit_rans.py SAMPLE_DIR COUNTS_DIR OUT_DIR [workers] [stride]
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import quantize, cum_of, encode_vec, L, PB
from audit_count import read_expert, unpack, lower_median, block_contexts

W_CHUNK, S_CHUNK = 2048, 512


def build_model(sample_dir, counts_dir):
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    base = np.zeros((3, 567, 16), np.int64)
    s0 = np.zeros((3, 256), np.int64)
    for e in man['experts']:
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        base += z['J1'].reshape(3, 567, 2, 32, 16).sum((2, 3))
        s0 += z['s_o0']
    return man, base, s0


def do_expert(args):
    sample_dir, out_dir, ex, wf, sf, hash_too, seed = args
    outp = os.path.join(out_dir, ex['file'].replace('.bin', '.rans.npz'))
    if os.path.exists(outp):
        return ex['file'], -1.0
    t0 = time.time()
    tens, sha_ok = read_expert(sample_dir, ex)
    wc = cum_of(wf); sc_cum = cum_of(sf)
    Fs, Cs, hist, Fsc, Csc = [], [], [], [], []
    for m, w in enumerate(('w1', 'w2', 'w3')):
        wb, wshape = tens[f'{w}.weight']
        sc, _ = tens[f'{w}.scale']
        med, _ = lower_median(sc)
        srel = (np.clip(sc.astype(np.int32) - med, -3, 3) + 3).astype(np.int32)
        sym = unpack(wb).reshape(-1, 32)
        ctx, _, _ = block_contexts(sym, srel)
        s32 = sym.astype(np.int64).ravel(); c32 = ctx.astype(np.int64).ravel()
        ns = s32.size // W_CHUNK
        assert ns * W_CHUNK == s32.size
        Fs.append(wf[m][c32, s32].reshape(ns, W_CHUNK)); Cs.append(wc[m][c32, s32].reshape(ns, W_CHUNK))
        sr = srel.reshape(ns, W_CHUNK // 32)
        hist.append(np.stack([(sr == k).sum(1) for k in range(7)], 1).astype(np.int16))
        scv = sc.astype(np.int64)
        nss = scv.size // S_CHUNK
        assert nss * S_CHUNK == scv.size
        Fsc.append(sf[m][scv].reshape(nss, S_CHUNK)); Csc.append(sc_cum[m][scv].reshape(nss, S_CHUNK))
    F = np.concatenate(Fs); C = np.concatenate(Cs)
    wlen, wx, _ = encode_vec(F, C)
    wbits = (PB - np.log2(F.astype(np.float64))).sum(1)
    Fq = np.concatenate(Fsc); Cq = np.concatenate(Csc)
    slen, sx, _ = encode_vec(Fq, Cq)
    sbits = (PB - np.log2(Fq.astype(np.float64))).sum(1)
    res = dict(wlen=wlen.astype(np.int32), wbits=wbits, wfinal=wx, whist=np.concatenate(hist),
               slen=slen.astype(np.int32), sbits=sbits, sfinal=sx)
    if hash_too:
        rng = np.random.default_rng(seed)
        h = rng.integers(0, 1 << 23, F.shape[0])
        wlen_h, _, _ = encode_vec(F, C, x0=L + h)
        res['wlen_h'] = wlen_h.astype(np.int32)
    np.savez_compressed(outp + '.tmp.npz', **res)
    os.replace(outp + '.tmp.npz', outp)
    return ex['file'], time.time() - t0


def main():
    sample_dir, counts_dir, out_dir = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(out_dir, exist_ok=True)
    man, base, s0 = build_model(sample_dir, counts_dir)
    wf = quantize(base); sf = quantize(s0)
    np.savez(os.path.join(out_dir, 'model.npz'), wf=wf, sf=sf, base=base, s0=s0)
    exs = man['experts'][::stride]
    jobs = [(sample_dir, out_dir, e, wf, sf, (i % 8 == 0), 1000 + i) for i, e in enumerate(exs)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (fn, dt) in enumerate(pool.map(do_expert, jobs)):
            if i % 10 == 0:
                print(f'{i + 1}/{len(jobs)} {fn} {dt:.1f}s total {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
