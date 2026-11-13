import time
import numpy as np
N = 1 << 23
rng = np.random.default_rng(0)
cur = rng.integers(0, 140, N).astype(np.int32)
f = rng.integers(0, 9, N).astype(np.uint8)
sym = rng.integers(0, 16, N).astype(np.uint8)
lut = np.arange(16, dtype=np.int64)
lut32 = np.arange(16, dtype=np.int32)
T = {}
t = time.perf_counter(); c = cur.astype(np.int64); T['astype'] = time.perf_counter() - t
t = time.perf_counter(); v = lut[f]; T['lut_take'] = time.perf_counter() - t
t = time.perf_counter(); v32 = lut32.take(f); T['lut_take32'] = time.perf_counter() - t
t = time.perf_counter(); k = (c * 9 + v) * 16 + sym; T['arith64'] = time.perf_counter() - t
t = time.perf_counter(); k32 = (cur * 9 + v32) * 16 + sym; T['arith32'] = time.perf_counter() - t
t = time.perf_counter(); b = np.bincount(k, minlength=140 * 9 * 16); T['bincount64_small'] = time.perf_counter() - t
t = time.perf_counter(); b = np.bincount(k32, minlength=140 * 9 * 16); T['bincount32_small'] = time.perf_counter() - t
k2 = rng.integers(0, 5000 * 9 * 16, N)
t = time.perf_counter(); b = np.bincount(k2, minlength=5000 * 9 * 16); T['bincount64_720k'] = time.perf_counter() - t
t = time.perf_counter(); r = np.repeat(lut[f[:N // 32]], 32); T['repeat'] = time.perf_counter() - t
for kk, vv in T.items():
    print(f'{kk:20s} {vv * 1e3:8.1f} ms  {vv / N * 1e9:6.2f} ns/elem')
