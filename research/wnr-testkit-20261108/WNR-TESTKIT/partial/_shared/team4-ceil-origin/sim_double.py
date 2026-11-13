"""Synthetic check of the 'quantized twice' explanation of the doubled (even) blocks.

Usage: python -I sim_double.py OUT_JSON
Draw blocks of 32 Laplace/Gaussian weights with a log-normal per-row spread, then:
  A  ceil rule once           s = 2^ceil(log2(amax/6))       (DeepSeek kernel.py fast_round_scale, triton ROUND_UP)
  B  OCP floor rule once      s = 2^(floor(log2(amax)) - 2)  (OCP MX v1.0 6.3, clamp to 6)
  C  ceil rule, dequantize, ceil rule again on the dequantized values
  D  ceil rule, dequantize, OCP floor rule on the dequantized values
E2M1 rounding: nearest, ties to even mantissa (as a cast does). Reports block-max code histogram and the share of
'even' blocks (max code 7, no code 1 or 3) and whether the second pass changed any value.
"""
import json, sys
import numpy as np

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def e2m1_code(x):
    """x >= 0 array (already divided by the scale), clamp to 6, round to nearest, ties to even code."""
    x = np.minimum(x, 6.0)
    d = np.abs(x[..., None] - GRID)
    best = d.min(-1, keepdims=True)
    cand = np.isclose(d, best, rtol=0, atol=1e-12)
    # ties: prefer the even code (even mantissa bit)
    pref = cand & (np.arange(8) % 2 == 0)
    use = np.where(pref.any(-1, keepdims=True), pref, cand)
    return use.argmax(-1)


def quant(w, rule):
    a = np.abs(w).max(-1, keepdims=True)
    a = np.maximum(a, 1e-30)
    if rule == 'ceil':
        s = 2.0 ** np.ceil(np.log2(a / 6))
    else:
        s = 2.0 ** (np.floor(np.log2(a)) - 2)
    c = e2m1_code(np.abs(w) / s)
    return c, s


def stats(c):
    bmax = c.max(-1)
    has13 = ((c == 1) | (c == 3)).any(-1)
    even = (bmax == 7) & ~has13
    return {'bmax_hist_pct': (100 * np.bincount(bmax.ravel(), minlength=8) / bmax.size).round(3).tolist(),
            'even_pct': round(100 * even.mean(), 3)}


def main():
    rng = np.random.default_rng(20261107)
    out = {}
    for dist in ['gauss', 'laplace']:
        n = 400_000
        sig = np.exp(rng.normal(0, 0.35, size=(n, 1)))
        w = (rng.standard_normal((n, 32)) if dist == 'gauss' else rng.laplace(size=(n, 32))) * sig
        cA, sA = quant(w, 'ceil')
        cB, sB = quant(w, 'floor')
        deq = GRID[cA] * sA * np.sign(w)
        cC, sC = quant(deq, 'ceil')
        cD, sD = quant(deq, 'floor')
        r = {'A_ceil_once': stats(cA), 'B_floor_once': stats(cB), 'C_ceil_twice': stats(cC), 'D_ceil_then_floor': stats(cD)}
        r['C_value_exact'] = bool(np.array_equal(GRID[cC] * sC, GRID[cA] * sA))
        r['D_value_exact'] = bool(np.array_equal(GRID[cD] * sD, GRID[cA] * sA))
        r['C_equals_D_codes'] = bool(np.array_equal(cC, cD) and np.array_equal(sC, sD))
        moved = (sC != sA)[:, 0]
        r['C_blocks_with_changed_scale_pct'] = round(100 * moved.mean(), 3)
        r['A_share_max3_pct'] = round(100 * (cA.max(-1) == 5).mean(), 3)
        out[dist] = r
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
