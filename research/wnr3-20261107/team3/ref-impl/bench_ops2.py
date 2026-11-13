"""Symbol lookup alternatives for the lock-step decoder, realistic context concentration (S = 17280)."""
import time
import numpy as np

S = 17280
rng = np.random.default_rng(0)
x = rng.integers(1 << 23, 1 << 31, S).astype(np.int32)
R = 2268 * 4
# realistic: ~300 hot rows
hot = rng.choice(R, 300, replace=False)
ctx = hot[rng.zipf(1.5, S) % 300].astype(np.int32)
freq = rng.integers(1, 3000, (R, 16)); freq = (freq * 16384 // freq.sum(1, keepdims=True)); freq[:, 0] += 16384 - freq.sum(1)
cum = np.zeros((R, 17), np.int64); cum[:, 1:] = np.cumsum(freq, 1)
LUT = np.repeat(np.tile(np.arange(16, dtype=np.uint8), R), freq.ravel())
C15 = np.ascontiguousarray(cum[:, 1:16]).astype(np.int32)        # [R, 15]
C16 = cum[:, :16].astype(np.int32).ravel()
CN = cum[:, 1:17].astype(np.int32).ravel()                       # cum of s+1
B6 = np.zeros((R, 64), np.uint8)                                  # symbol at slot (b << 8)
for r in range(R):
    B6[r] = np.searchsorted(cum[r], np.arange(64) * 256, side='right') - 1
B6 = B6.ravel()
N = 1000
slot = x & 16383
ref = LUT[(ctx << 14) | slot]


def t(name, f):
    out = f()
    assert (out == ref).all(), name
    t0 = time.perf_counter()
    for _ in range(N):
        f()
    dt = (time.perf_counter() - t0) / N
    print(f'{name:28s} {dt * 1e6:8.1f} us/step  -> {dt * 2112:6.2f} s/record')


t('LUT 16K/ctx', lambda: LUT[(ctx << 14) | slot])
t('15-compare', lambda: (C15[ctx] <= slot[:, None]).sum(1))


def bucket():
    s = B6[(ctx << 6) | (slot >> 8)].astype(np.int32)
    base = ctx << 4
    for _ in range(16):
        adv = slot >= CN[base | s]
        if not adv.any():
            break
        s += adv
    return s


t('64-bucket + refine', bucket)
