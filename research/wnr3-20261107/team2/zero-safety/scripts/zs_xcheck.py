"""Cross-check: GPU outputs of colibri's CUDA kernels (dumped by zs_gpu.py) vs the CPU kernels and the CPU
IEEE emulations in zs_kern.dll, bit for bit, on the same weights (w1 of the first file) and activations.
Usage: python -I zs_xcheck.py <zs_kern.dll> <sample_dir> <file.bin> <xcheck.npz>"""
import ctypes as C, sys, json
import numpy as np
dll, sample_dir, fn, npz = sys.argv[1:5]
L = C.CDLL(dll); P = C.c_void_p
L.zs_run.argtypes = [C.c_int, P, P, P, P, P, P, C.c_int, C.c_int, C.c_int, P, P]; L.zs_set_tables.argtypes = [P, P]
v = np.arange(256)
e8lut = np.where(v == 255, np.float32(np.nan), np.ldexp(np.float32(1), v - 127)).astype(np.float32)
e8cuda = np.ldexp(np.float32(1), np.where(v == 0, -127, v - 127)).astype(np.float32)
L.zs_set_tables(e8lut.ctypes.data, e8cuda.ctypes.data)
SC, WB = 368640, 5898240
rec = np.fromfile(sample_dir + "\\" + fn, dtype=np.uint8)
w = rec[3*SC:3*SC+WB].copy(); s = rec[0:SC].copy()
d = np.load(npz); x = d["x1"].astype(np.float32); I, O = int(d["I"]), int(d["O"])
xq = np.zeros(5120, np.int8); xsc = np.zeros(160, np.float32); dummy = np.zeros(16, np.uint8)
def cpu(k):
    y = np.empty(O, np.float32)
    L.zs_run(k, y.ctypes.data, w.ctypes.data, s.ctypes.data, dummy.ctypes.data, dummy.ctypes.data, x.ctypes.data, 1, I, O, xq.ctypes.data, xsc.ctypes.data)
    return y
def bits(a, b): return int((a.view(np.uint32) != b.view(np.uint32)).sum())
g1 = d["mv_fp4_grouped_1_"]; gv = d["v4f_dot_fp4_warp_row_"]; gm = d["mv_4_"]
out = {"O": O, "I": I,
       "gpu mv_fp4_grouped<1> vs cpu K9a (fma emulation): differing outputs": bits(g1, cpu(9)),
       "gpu mv_fp4_grouped<1> vs cpu K9b (no-fma emulation): differing outputs": bits(g1, cpu(10)),
       "gpu mv<4> vs gpu mv_fp4_grouped<1>: differing outputs": bits(gm, g1),
       "gpu v4f_dot_fp4 vs cpu K10 emulation: differing outputs": bits(gv, cpu(11)),
       "gpu mv_fp4_grouped<1> vs cpu K1 (AVX2 production): differing outputs": bits(g1, cpu(0)),
       "max |gpu mv_fp4_grouped<1> - cpu K1| / max|K1|": float(np.max(np.abs(g1 - cpu(0))) / np.max(np.abs(cpu(0))))}
print(json.dumps(out, indent=1))
