"""Quick look at downloaded NVFP4 items: tensor scales, scale-byte histogram, code histogram, block max, -0.

Usage: python -I nvinspect.py FAM_DIR [max_items]
"""
import importlib.util, json, os, sys
import numpy as np

_s = importlib.util.spec_from_file_location('nvcommon', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvcommon.py'))
nc = importlib.util.module_from_spec(_s); _s.loader.exec_module(nc)


def main():
    d = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    man = json.load(open(os.path.join(d, 'manifest.json')))
    print(man['repo'], man['revision'], len(man['experts']))
    ts = []
    for it in man['experts'][:k]:
        for r, codes, sc, tsc, isc, nb in nc.matrices(d, it):
            mag = codes & 7
            R, C = codes.shape
            bm = mag.reshape(R, C // 16, 16).max(2)
            sb = np.bincount(sc.ravel(), minlength=256)
            nz = np.nonzero(sb)[0]
            top = np.argsort(-sb)[:8]
            sv = nc.e4m3_value(sc.ravel())
            print(f" {it['tag']} {r} shape {codes.shape} tscale {tsc!r} iscale {isc!r} scale bytes {nz.min()}..{nz.max()} distinct {nz.size}"
                  f" top {[(int(t), int(sb[t])) for t in top]}")
            print(f"    pow2 scales {np.mean(np.isin(sc.ravel() & 7, [0])):.4f}  max scale value {np.nanmax(sv):.4g}  min {np.nanmin(sv):.4g}"
                  f"  code hist {np.round(np.bincount(codes.ravel(), minlength=16) / codes.size, 4).tolist()}")
            print(f"    blockmax16 hist {np.round(np.bincount(bm.ravel(), minlength=8) / bm.size, 5).tolist()}  -0 frac {np.mean(codes == 8):.5f}")
            ts.append((it['tag'], r, tsc, isc))
    print('tscales', ts[:12])


if __name__ == '__main__':
    main()
