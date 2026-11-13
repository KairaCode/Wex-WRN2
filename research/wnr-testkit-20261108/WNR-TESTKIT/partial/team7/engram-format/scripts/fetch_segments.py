"""Engram segment layout (primes as in inference/engram.py EngramLayout.from_args) + per-segment row-sample fetch.

Usage: python -I fetch_segments.py OUT_DIR [CHUNKS_PER_SEG] [ROWS_PER_CHUNK]
HTTP Range reads of the public base repo deepseek-ai/DeepSeek-V4.1-Flash-1017 at the pinned revision
(the same source team 3 used); only bytes are fetched, nothing is executed.
"""
import json, os, sys, time, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BASE_REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
BASE_REV = '1bae0e1088993b9b82de08f59de5be5a34e42170'
URL = f'https://huggingface.co/{BASE_REPO}/resolve/{BASE_REV}/'
UA = {'User-Agent': 'wnr2-research-team7/1.0'}
# from team3 tensors.json (headers of shards 47/48)
T = {1: dict(file='model-00047-of-00048.safetensors', rows=384006168, w_off=664, s_off=98305579672),
     14: dict(file='model-00048-of-00048.safetensors', rows=384016682, w_off=672, s_off=98308271264)}


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def layout(vocab=16000000, layers=(1, 14), max_ngram=4, n_heads=8):
    seen, out = set(), {}
    for L in layers:
        segs = []
        for ng in range(2, max_ngram + 1):
            cur = vocab - 1
            for h in range(n_heads):
                c = cur + 1
                while not is_prime(c) or c in seen:
                    c += 1
                seen.add(c); cur = c
                segs.append((ng, h, c))
        out[L] = segs
    return out


def get_range(fname, a, b, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(URL + fname, headers={**UA, 'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = r.read()
            if len(d) != b - a + 1:
                raise IOError('short read')
            return d
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 + 3 * t)


def main():
    out = sys.argv[1]
    cps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rpc = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    os.makedirs(out, exist_ok=True)
    lay = layout()
    segtab = {}
    for L, segs in lay.items():
        tot = sum(p for _, _, p in segs)
        assert tot == T[L]['rows'], (L, tot, T[L]['rows'])
        off = 0
        segtab[L] = []
        for ng, h, p in segs:
            segtab[L].append({'ngram': ng, 'head': h, 'prime': p, 'row0': off})
            off += p
    json.dump(segtab, open(os.path.join(out, 'segments.json'), 'w'), indent=1)
    rng = np.random.default_rng(7)
    jobs = []
    for L, segs in segtab.items():
        for si, s in enumerate(segs):
            for c in range(cps):
                r0 = s['row0'] + int(rng.integers(0, s['prime'] - rpc))
                jobs.append((L, si, r0))

    def one(j):
        L, si, r0 = j
        t = T[L]
        tag = f'L{L}_s{si:02d}_r{r0}_n{rpc}'
        pw, ps = os.path.join(out, tag + '.w.bin'), os.path.join(out, tag + '.s.bin')
        if not (os.path.exists(pw) and os.path.exists(ps)):
            w = get_range(t['file'], t['w_off'] + r0 * 256, t['w_off'] + (r0 + rpc) * 256 - 1)
            s = get_range(t['file'], t['s_off'] + r0 * 8, t['s_off'] + (r0 + rpc) * 8 - 1)
            open(pw + '.part', 'wb').write(w); os.replace(pw + '.part', pw)
            open(ps + '.part', 'wb').write(s); os.replace(ps + '.part', ps)
        w = open(pw, 'rb').read(); s = open(ps, 'rb').read()
        return {'tag': tag, 'layer': L, 'seg': si, 'row0': r0, 'rows': rpc,
                'sha256_w': hashlib.sha256(w).hexdigest(), 'sha256_s': hashlib.sha256(s).hexdigest()}

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    json.dump({'repo': BASE_REPO, 'revision': BASE_REV, 'items': res}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(len(res), 'chunks', len(res) * rpc * 264, 'bytes', f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
