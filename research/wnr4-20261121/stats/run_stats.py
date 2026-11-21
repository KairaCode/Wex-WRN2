# SPDX-License-Identifier: Apache-2.0
"""Context-model study: held-out cross-entropy of weight and scale models on real records.
    python -I stats/run_stats.py DATASET OUT.json [--train 256] [--test 2048] [--threads 8] [--cz 1]"""
import ctypes as C
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data  # noqa: E402

SNAMES = ['order0 even', 'order0 odd', 'left even', 'left odd', 'up']


def main(argv):
    name, out = argv[1], argv[2]
    opt = {argv[i]: int(argv[i + 1]) for i in range(3, len(argv) - 1, 2)}
    ds = data.Dataset(name)
    train, test = ds.split(opt.get('--train', 256), opt.get('--test', 2048))
    cz = bool(opt.get('--cz', 1))
    lib = C.CDLL(os.path.join(os.path.dirname(HERE), 'out', 'wnr4_stats.dll'))
    lib.st_new.restype = C.c_void_p
    lib.st_add.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_char_p, C.c_void_p, C.c_char_p]
    lib.st_report.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    lib.st_wname.restype = C.c_char_p
    st = lib.st_new()
    lock = threading.Lock()
    t0 = time.time()
    flags_seen = {}

    def job(item):
        s, r = item
        coded, anchors, flags = ds.coded(ds.source(r), cz)
        with lock:
            flags_seen[flags] = flags_seen.get(flags, 0) + 1
            lib.st_add(st, s, int(ds.nv), coded, ds.geo_c, anchors)
    items = [(0, r) for r in train] + [(1, r) for r in test]
    with ThreadPoolExecutor(opt.get('--threads', 8)) as pool:
        for k, _ in enumerate(pool.map(job, items)):
            if (k + 1) % 256 == 0:
                print(f'  {k + 1}/{len(items)} {time.time() - t0:.0f} s', flush=True)
    wn, sn = lib.st_wn(), lib.st_sn()
    stride = wn + sn + 4
    res = (C.c_double * (3 * stride))()
    lib.st_report(st, 12, 14, res)
    res14 = (C.c_double * (3 * stride))()
    lib.st_report(st, 14, 14, res14)
    raw_bits = len(test) * ds.raw * 8
    rep = {'dataset': name, 'cz': cz, 'train': len(train), 'test': len(test), 'raw_bytes_test': len(test) * ds.raw,
           'flags_seen': flags_seen, 'weights': {}, 'scales': {}}
    nib = sum(res[m * stride + wn + sn + 2] for m in range(3))
    nsc = sum(res[m * stride + wn + sn + 3] for m in range(3))
    signs = sum(res[m * stride + wn + sn + 1] for m in range(3))
    mcost = sum(res[m * stride + wn + sn] for m in range(3))
    for v in range(wn):
        vname = lib.st_wname(v).decode()
        for bits, rr in ((12, res), (14, res14)):
            b = sum(rr[m * stride + v] for m in range(3))
            if 'mag8' in vname:
                b += signs if cz else nib   # raw sign bits: nonzero only under CZ, every nibble otherwise
            if 'M' in vname.split('|')[-1] or 'blockmax' in vname:
                b += mcost
            rep['weights'][f'{vname} @{bits}'] = {'bits_per_weight': b / nib, 'of_raw': b / raw_bits}
    for v in range(sn):
        b = sum(res[m * stride + wn + v] for m in range(3))
        rep['scales'][SNAMES[v]] = {'bits': b, 'of_raw': b / raw_bits}
    sc = rep['scales']
    for k, (a, b) in {'order0': ('order0 even', 'order0 odd'), 'left (WNR3-NV)': ('left even', 'left odd'),
                      'pair (even order0, odd left)': ('order0 even', 'left odd')}.items():
        bits = sc[a]['bits'] + sc[b]['bits']
        rep['scales'][k] = {'bits_per_scale': bits / nsc, 'of_raw': bits / raw_bits}
    rep['scales']['up'] = {'bits_per_scale': sc['up']['bits'] / nsc, 'of_raw': sc['up']['of_raw']}
    rep['blockmax_bits_per_block'] = mcost / nsc
    rep['seconds'] = time.time() - t0
    with open(out, 'w') as g:
        json.dump(rep, g, indent=1)
    for k, v in rep['weights'].items():
        print(f'W {k:40s} {v["bits_per_weight"]:.4f} b/w  {v["of_raw"]:.5f} of raw')
    for k, v in rep['scales'].items():
        if 'bits_per_scale' in v:
            print(f'S {k:40s} {v["bits_per_scale"]:.4f} b/s  {v["of_raw"]:.5f} of raw')
    print('flags', flags_seen, f'{rep["seconds"]:.0f} s')


if __name__ == '__main__':
    main(sys.argv)
