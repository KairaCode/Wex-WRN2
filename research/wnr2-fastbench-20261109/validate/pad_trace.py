"""Add EXTRA wasted (never waited for) reads to a replay trace, spread evenly over its (token, layer) lines.

  python -I pad_trace.py IN.trace OUT.trace EXTRA

Why: the tap sees residency only at token boundaries, so a trace misses records that were read and evicted
inside one token. For WNR2 (GPU-direct records skip the RAM tier and churn in VRAM) that gap is much larger
than for WEX1. EXTRA calibrates the trace's read count to the engine's exit counter (see VALIDATION.md).
shortcut: the missing reads are modelled as wasted guesses (optimistic for the padded arm: a missing re-read
that the layer waited for would cost more); the experts are any of the layer's not already in the line.
"""
import hashlib
import sys


def pad(lines, experts, extra):
    body = [l for l in lines if not l.startswith('#')]
    n, out, added = len(body), [l for l in lines if l.startswith('#')], 0
    for i, line in enumerate(body):
        t, layer, nd, np_, nw, *ids = map(int, line.split())
        k = (i + 1) * extra // n - i * extra // n      # even spread, sums to extra
        free = sorted((e for e in range(experts) if e not in ids),
                      key=lambda e: hashlib.blake2s(f'{t},{layer},{e}'.encode(), digest_size=4).digest())
        ids += free[:k]
        added += min(k, len(free))
        out.append(' '.join(map(str, [t, layer, nd, np_, nw + min(k, len(free))] + ids)))
    return out, added


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    lines = open(argv[0], encoding='ascii').read().splitlines()
    experts = int(lines[0].split('experts=')[1].split()[0])
    out, added = pad(lines, experts, int(argv[2]))
    assert added == int(argv[2]), (added, argv[2])
    out[0] += f' padded_wasted={added}'
    open(argv[1], 'w', encoding='ascii', newline='\n').write('\n'.join(out) + '\n')
    print(f'{argv[1]}: +{added} wasted reads over {len(out) - 2} lines')


if __name__ == '__main__':
    t, a = pad(['# h experts=4', '0 0 1 0 0 2', '0 1 0 1 0 3', '1 0 0 0 0'], 4, 5)
    assert a == 5 and t[1].split()[:5] == ['0', '0', '1', '0', '1'] and t[3].split()[4] == '2', t
    main(sys.argv[1:])
