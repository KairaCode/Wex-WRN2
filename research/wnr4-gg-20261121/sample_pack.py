# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Copies a seeded sample of expert records out of an expert pack into OUT_DIR/sample.bin + OUT_DIR/sample.json.

    python -I sample_pack.py PACK_DIR OUT_DIR

PACK_DIR holds experts.bin (every layer's experts back to back, one record per expert: the raw GGUF slices
gate | up | down) and native_experts.txt (one line per layer: layer, gate/up ggml type, down ggml type, offset of
expert 0 in experts.bin, record bytes; lines starting with # are comments). Per layer 10 distinct experts out of
256 (random.Random(20261121)): the first 4 are the training set, the other 6 the held-out set. Standard library only."""
import json
import os
import random
import sys

BPB = {18: (256, 98), 21: (256, 110), 22: (256, 82), 23: (256, 136), 20: (32, 18), 42: (64, 18)}  # ggml type -> (block elems, bytes)
H, F = 2560, 640  # hidden size, expert FFN size of Qwen3.8-Flash-Next


def nbytes(t, elems):
    be, bb = BPB[t]
    if elems % be:
        raise SystemExit(f'type {t}: {elems} elements are not whole blocks')
    return elems // be * bb


def main(pack, out):
    layers = []
    for line in open(os.path.join(pack, 'native_experts.txt')):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        L, gu, d, off, blob = (int(x) for x in f[:5])
        gs, ds = nbytes(gu, H * F), nbytes(d, H * F)
        if 2 * gs + ds != blob:
            raise SystemExit(f'layer {L}: 2 * {gs} + {ds} != record bytes {blob}')
        layers.append(dict(L=L, gu=gu, d=d, off=off, blob=blob, gs=gs, ds=ds))
    if len(layers) != 48:
        raise SystemExit(f'{len(layers)} layers, expected 48')
    rng = random.Random(20261121)
    recs, at = [], 0
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(pack, 'experts.bin'), 'rb', buffering=0) as fi, \
            open(os.path.join(out, 'sample.bin'), 'wb') as fo:
        for ly in layers:
            for i, e in enumerate(rng.sample(range(256), 10)):
                fi.seek(ly['off'] + e * ly['blob'])
                b = fi.read(ly['blob'])
                if len(b) != ly['blob']:
                    raise SystemExit(f'layer {ly["L"]} expert {e}: short read')
                fo.write(b)
                recs.append(dict(L=ly['L'], e=e, train=i < 4, at=at, types=[ly['gu'], ly['gu'], ly['d']],
                                 bytes=[ly['gs'], ly['gs'], ly['ds']]))
                at += len(b)
    with open(os.path.join(out, 'sample.json'), 'w') as f:
        json.dump(recs, f)
    print('records', len(recs), 'bytes', at)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
