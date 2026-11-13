"""ceil-refuter, resumed run: exhaustive bijection checks of T on small blocks, plus the 'skip T when scale == 255' rule.

Usage: python -I -B edge_tests2.py OUT_JSON
T is size-agnostic (block max over the block's nibbles), so exhaustive checks on 4-nibble blocks (all 65,536) with
scales {0, 1, 120, 253, 254, 255} exercise every case the 32-nibble rule has: even/non-even, every max, sign of zero.
Independent nibble-level implementation (no LUTs shared with refute_count.py).
"""
import json, sys
import itertools
import numpy as np

H = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
D = {v: k for k, v in H.items()}


def fwd(nibs, s, skip255):
    mags = [n & 7 for n in nibs]
    even = max(mags) == 7 and not any(m in (1, 3) for m in mags)
    if even and not (skip255 and s == 255):
        return tuple((n & 8) | H[n & 7] for n in nibs), s + 1, True
    return tuple(nibs), s, False


def inv(nibs, s):
    if max(n & 7 for n in nibs) == 5:
        return tuple((n & 8) | D[n & 7] for n in nibs), s - 1
    return tuple(nibs), s


def zc(nibs):
    return tuple(0 if n == 8 else n for n in nibs)


def run(skip255, allow_max5):
    scales = [0, 1, 120, 253, 254, 255]
    seen = {}
    n = rt_fail = coll = over = n_even = 0
    zfail = 0
    for nibs in itertools.product(range(16), repeat=4):
        if not allow_max5 and max(x & 7 for x in nibs) == 5:
            continue
        for s in scales:
            n += 1
            t, ts, ev = fwd(nibs, s, skip255)
            n_even += ev
            if ts > 255:
                over += 1
                continue
            key = (t, ts)
            if key in seen and seen[key] != (nibs, s):
                coll += 1
            seen[key] = (nibs, s)
            if inv(t, ts) != (nibs, s):
                rt_fail += 1
            # canonical zero applied in the coded domain, then inverse == canonical zero of the original
            if inv(zc(t), ts) != (zc(nibs), s):
                zfail += 1
    return {'domain_pairs': n, 'even': n_even, 'scale_overflow_256': over, 'collisions': coll,
            'roundtrip_fail': rt_fail, 'canonical_zero_commute_fail': zfail}


def main():
    out = {}
    out['rule_T_always__domain_no_max5'] = run(skip255=False, allow_max5=False)
    out['rule_T_skip_if_scale255__domain_no_max5'] = run(skip255=True, allow_max5=False)
    out['rule_T_skip_if_scale255__domain_all_blocks(incl_max5)'] = run(skip255=True, allow_max5=True)
    # how many 4-nibble blocks have max 5 (the only blocks that break T): share of the full space
    allb = list(itertools.product(range(16), repeat=4))
    out['share_of_4nibble_space_with_max5'] = sum(max(x & 7 for x in b) == 5 for b in allb) / len(allb)
    # E2M1 value check of T on every nibble, every scale 1..254: value(code) * 2^(s-127) preserved
    E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6]
    vfail = 0
    for m in (0, 2, 4, 5, 6, 7):
        for s in range(0, 255):
            if E2M1[m] * 2.0 ** (s - 127) != E2M1[H[m]] * 2.0 ** (s + 1 - 127):
                vfail += 1
    out['value_preservation_fail_per_magnitude_scale'] = vfail
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
