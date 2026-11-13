"""Micro-benchmark of the per-step NumPy ops of the lock-step decoder (S = 17280 lanes)."""
import time
import numpy as np

S = 17280
rng = np.random.default_rng(0)
x = rng.integers(1 << 23, 1 << 31, S).astype(np.int32)
ctx = rng.integers(0, 2268 * 3, S).astype(np.int32)
LUT = rng.integers(0, 16, 2268 * 3 * 16384).astype(np.uint8)
LUTs = rng.integers(0, 16, 567 * 3 * 16384).astype(np.uint8)
WF = rng.integers(1, 1000, 2268 * 3 * 16).astype(np.int32)
pl = rng.integers(0, 256, 17_000_000).astype(np.uint8)
ptr = np.sort(rng.integers(0, 16_900_000, S)).astype(np.int32)
N = 2000


def t(name, f):
    t0 = time.perf_counter()
    for _ in range(N):
        f()
    dt = (time.perf_counter() - t0) / N
    print(f'{name:28s} {dt * 1e6:8.1f} us/step  -> {dt * 2112:6.2f} s/record')


t('and', lambda: x & 16383)
t('shift', lambda: x >> 14)
t('mul+add', lambda: WF[:S] * (x >> 14) + x)
t('LUT gather 111MB', lambda: LUT[(ctx << 14) | (x & 16383)])
t('LUT gather 28MB', lambda: LUTs[((ctx % 1701) << 14) | (x & 16383)])
t('WF gather', lambda: WF[(ctx << 4) | 5])
t('pl gather', lambda: pl[ptr])
t('compare+any', lambda: (x < (1 << 23)).any())
t('where', lambda: np.where(x < (1 << 27), (x << 8) | 3, x))
t('maximum', lambda: np.maximum(x, ctx))
