# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Input of wnr4/tests/test_wnr4_gg_gpu.cu. Copies the routed-expert slices of a few (layer, expert) pairs out of
a GGUF part, one file per expert in the record order of gguf2wex4 (gate, gate_up, up, down), with a manifest of
their ggml types and byte sizes.

    python -I wnr4/tests/sample_gg_experts.py PART.gguf OUT_DIR LAYERS EXPERTS   (e.g. 0,6,12,24,36,47 0-15)

PART is the GGUF part that holds the expert tensors. Standard library only."""
import json
import os
import re
import struct
import sys

GGML = {2: (32, 18), 8: (32, 34), 10: (256, 84), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66),
        17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
        29: (256, 56), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18), 0: (1, 4), 1: (1, 2), 30: (1, 2)}
EXPERT = re.compile(r'blk\.(\d+)\.ffn_(gate|gate_up|up|down)_exps\.weight\Z')
ORDER = ('gate', 'gate_up', 'up', 'down')


def rd(f, fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))


def rstr(f):
    (n,) = rd(f, 'Q')
    return f.read(n).decode()


def skip_value(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.read(sizes[t])
    elif t == 8:
        rstr(f)
    elif t == 9:
        et, n = rd(f, 'IQ')
        for _ in range(n):
            skip_value(f, et)
    else:
        raise ValueError(t)


def tensors(path):
    with open(path, 'rb') as f:
        magic, version, nt, nkv = rd(f, '4sIQQ')
        assert magic == b'GGUF', magic
        align = 32
        for _ in range(nkv):
            key = rstr(f)
            (t,) = rd(f, 'I')
            if key == 'general.alignment' and t == 4:
                (align,) = rd(f, 'I')
            else:
                skip_value(f, t)
        infos = []
        for _ in range(nt):
            name = rstr(f)
            (nd,) = rd(f, 'I')
            ne = rd(f, f'{nd}Q')
            ttype, off = rd(f, 'IQ')
            infos.append((name, ne, ttype, off))
        data = (f.tell() + align - 1) // align * align
    return data, infos


def main(part, out, layers, experts):
    layers = [int(x) for x in layers.split(',')]
    lo, hi = (int(x) for x in experts.split('-'))
    data, infos = tensors(part)
    by_layer = {}
    for name, ne, t, off in infos:
        m = EXPERT.match(name)
        if m:
            nel = 1
            for x in ne:
                nel *= x
            per, nb = GGML[t]
            total = nel // per * nb
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = dict(name=name, ne=ne, type=t, offset=data + off,
                                                                        bytes=total, per_expert=total // ne[-1])
    os.makedirs(out, exist_ok=True)
    manifest = []
    with open(part, 'rb') as f:
        for L in layers:
            kinds = [k for k in ORDER if k in by_layer[L]]
            for E in range(lo, hi + 1):
                chunks, segs = [], []
                for k in kinds:
                    t = by_layer[L][k]
                    f.seek(t['offset'] + E * t['per_expert'])
                    chunks.append(f.read(t['per_expert']))
                    segs.append(dict(kind=k, type=t['type'], bytes=t['per_expert'], ne=t['ne'][:-1]))
                name = f'L{L:02d}_E{E:03d}.bin'
                with open(os.path.join(out, name), 'wb') as g:
                    g.write(b''.join(chunks))
                manifest.append(dict(file=name, layer=L, expert=E, segments=segs))
    json.dump(dict(source=os.path.basename(part), experts=manifest,
                   layers={L: {k: dict(type=v['type'], ne=v['ne'], per_expert=v['per_expert']) for k, v in d.items()}
                           for L, d in sorted(by_layer.items())}),
              open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(len(manifest), 'experts written')


if __name__ == '__main__':
    main(*sys.argv[1:5])
