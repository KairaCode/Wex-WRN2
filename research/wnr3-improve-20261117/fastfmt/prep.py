# Decode every sample record with the CPU reference (libwex_nr3_tools.so: wex_nr3_decode) and store
# Z(source) = the exact bytes WNR3-CZ decodes to, as data/Z_LL_EEE.bin (+ entry as data/E_LL_EEE.bin).
import ctypes, os, sys, glob
from multiprocessing import Pool
D = "/tmp/wnr3-data"; OUT = os.path.dirname(os.path.abspath(__file__)) + "/data"
model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()
def one(p):
    lib = ctypes.CDLL(D + "/libwex_nr3_tools.so")
    l, e = map(int, os.path.basename(p)[4:-4].split("_"))
    ent = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; rec = open(p, "rb").read()
    out = ctypes.create_string_buffer(18800640)
    r = lib.wex_nr3_decode(model, ent, rec, l, e, out)
    if r: return (l, e, r)
    open(f"{OUT}/Z_{l:02d}_{e:03d}.bin", "wb").write(out.raw)
    open(f"{OUT}/E_{l:02d}_{e:03d}.bin", "wb").write(ent)
    return (l, e, 0)
if __name__ == "__main__":
    recs = sorted(glob.glob(D + "/recs/rec_*.bin"))
    with Pool(2) as pool: res = pool.map(one, recs)
    print(len(res), "records, failures:", [r for r in res if r[2]])
