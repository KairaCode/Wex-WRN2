"""Qwen3.8 routed experts (FP8 E4M3): how they are read today and how much a WEX
container could gain. Read-only on the model; writes nothing.

  1. Read cost, cold, of the engine's current path (buffered ReadFile: gate+up in
     one range, down in another) against unbuffered aligned reads of the same
     ranges (FILE_FLAG_NO_BUFFERING, what a 4096-aligned WEX record allows).
     Two disjoint random expert sets, so neither side reads what the other
     already pulled into the page cache.
  2. Compressibility of the FP8 bytes: order-0 byte entropy, and the entropy
     with the previous byte's high nibble (sign+3 exponent bits) as context,
     per projection.

usage: python -I probe_q38_experts.py MODEL_DIR [experts_per_side=96] [threads=8] OUT.json
"""
import ctypes
import json
import math
import random
import struct
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

MODEL = Path(sys.argv[1])
PER_SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 96
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else None
PREFIX = 'model.language_model.layers.'
ALIGN = 4096

k32 = ctypes.WinDLL('kernel32', use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


class OVERLAPPED(ctypes.Structure):
    _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD), ('hEvent', wintypes.HANDLE)]


def open_file(path, unbuffered):
    flags = 0x20000000 if unbuffered else 0x10000000   # NO_BUFFERING : RANDOM_ACCESS
    h = k32.CreateFileW(str(path), 0x80000000, 1, None, 3, flags, None)
    if h in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error(), str(path))
    return h


def read_at(h, buf, offset, nbytes, need=None):
    """need: bytes that must arrive (an unbuffered span may run past EOF)"""
    ov = OVERLAPPED(0, 0, offset & 0xFFFFFFFF, offset >> 32, None)
    got = wintypes.DWORD()
    if not k32.ReadFile(h, buf, nbytes, ctypes.byref(got), ctypes.byref(ov)) or got.value < (nbytes if need is None else need):
        raise OSError(ctypes.get_last_error(), f'read {offset}+{nbytes} got {got.value}')


def shard_tensors():
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    shards = {}
    for name in sorted(set(index.values())):
        with open(MODEL / name, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for key, meta in header.items():
            if key != '__metadata__':
                shards[key] = (name, base + meta['data_offsets'][0], meta['data_offsets'][1] - meta['data_offsets'][0], meta['dtype'])
    return shards


def expert_ranges(tensors, layer, expert):
    t = [tensors[f'{PREFIX}{layer}.mlp.experts.{expert}.{p}_proj.weight'] for p in ('gate', 'up', 'down')]
    assert all(x[3] == 'F8_E4M3' for x in t), t
    if t[0][0] == t[1][0] and t[1][1] == t[0][1] + t[0][2]:
        return [(t[0][0], t[0][1], t[0][2] + t[1][2]), (t[2][0], t[2][1], t[2][2])], t
    return [(x[0], x[1], x[2]) for x in t], t


def run_side(ranges_per_expert, unbuffered):
    """THREADS workers, each with its own handles and buffer; returns seconds, bytes."""
    jobs = list(ranges_per_expert)
    lock = threading.Lock()
    total = [0]
    data = {}

    def worker():
        handles = {}
        buf = k32.VirtualAlloc(None, 8 << 20, 0x3000, 0x04)
        while True:
            with lock:
                if not jobs:
                    break
                key, ranges = jobs.pop()
            pieces = []
            for shard, off, nbytes in ranges:
                h = handles.get(shard) or handles.setdefault(shard, open_file(MODEL / shard, unbuffered))
                if unbuffered:
                    start = off & ~(ALIGN - 1)
                    span = ((off + nbytes + ALIGN - 1) & ~(ALIGN - 1)) - start
                    read_at(h, buf, start, span, off + nbytes - start)
                    pieces.append(ctypes.string_at(buf + (off - start), nbytes))
                    moved = span
                else:
                    read_at(h, buf, off, nbytes)
                    pieces.append(ctypes.string_at(buf, nbytes))
                    moved = nbytes
                with lock:
                    total[0] += moved
            data[key] = b''.join(pieces)
        for h in handles.values():
            k32.CloseHandle(h)

    started = time.perf_counter()
    workers = [threading.Thread(target=worker) for _ in range(THREADS)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return time.perf_counter() - started, total[0], data


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    cfg = json.loads((MODEL / 'config.json').read_text())
    text = cfg.get('text_config', cfg)
    layers, experts = text['num_hidden_layers'], text['num_experts']
    tensors = shard_tensors()
    rng = random.Random(int(__import__("os").environ.get("PROBE_SEED", "20261105")))
    picks = rng.sample([(l, e) for l in range(layers) for e in range(experts)], 2 * PER_SIDE)
    sides = {'buffered': picks[:PER_SIDE], 'unbuffered': picks[PER_SIDE:]}
    report = {'model': str(MODEL), 'threads': THREADS, 'experts_per_side': PER_SIDE}
    blobs = {}
    split = {}
    for name, keys in sides.items():
        plan = []
        for key in keys:
            ranges, t = expert_ranges(tensors, *key)
            plan.append((key, ranges))
            split[key] = (t[0][2], t[1][2], t[2][2])
        seconds, moved, data = run_side(plan, name == 'unbuffered')
        payload = sum(sum(r[2] for r in rg) for _, rg in plan)
        report[name] = {'seconds': round(seconds, 3), 'payload_gib_s': round(payload / seconds / 2**30, 3),
                        'ms_per_expert_wall': round(1000 * seconds / len(plan), 3),
                        'reads_per_expert': round(sum(len(rg) for _, rg in plan) / len(plan), 2),
                        'bytes_moved_over_payload': round(moved / payload, 4)}
        blobs.update(data)
    # entropy, per projection
    stats = {}
    for proj in range(3):
        c0 = np.zeros(256, np.float64)
        c1 = np.zeros((16, 256), np.float64)
        for key, blob in blobs.items():
            g, u, d = split[key]
            lo = (0, g, g + u)[proj]
            hi = lo + (g, u, d)[proj]
            a = np.frombuffer(blob[lo:hi], np.uint8)
            c0 += np.bincount(a, minlength=256)
            ctx = a[:-1] >> 4
            np.add.at(c1, (ctx, a[1:]), 1)
        h0 = entropy(c0)
        h1 = sum(c1[k].sum() * entropy(c1[k]) for k in range(16) if c1[k].sum()) / c1.sum()
        sign_exp = np.bincount(np.arange(256) >> 3, weights=c0, minlength=32)
        stats[('gate', 'up', 'down')[proj]] = {
            'order0_bits_per_byte': round(h0, 4), 'ctx_prev_high_nibble_bits': round(h1, 4),
            'sign_exponent_bits': round(entropy(sign_exp), 4),
            'nan_codes': int(c0[0x7F] + c0[0xFF]), 'neg_zero': int(c0[0x80])}
    report['entropy'] = stats
    w = [split[k] for k in blobs]
    tot = sum(sum(x) for x in w)
    best = sum(stats[p]['ctx_prev_high_nibble_bits'] * sum(x[i] for x in w) for i, p in enumerate(('gate', 'up', 'down'))) / tot
    report['estimated_saving_vs_fp8'] = round(1 - best / 8, 4)
    text_out = json.dumps(report, indent=1)
    print(text_out)
    if OUT:
        OUT.write_text(text_out, encoding='utf-8')


if __name__ == '__main__':
    main()
