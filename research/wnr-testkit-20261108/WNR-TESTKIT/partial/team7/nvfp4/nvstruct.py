"""Spatial structure of zero-scale blocks and zero codes. Usage: python -I nvstruct.py FAM_DIR [items]"""
import importlib.util, json, os, sys
import numpy as np

_s = importlib.util.spec_from_file_location('nvcommon', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvcommon.py'))
nc = importlib.util.module_from_spec(_s); _s.loader.exec_module(nc)

d = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
man = json.load(open(os.path.join(d, 'manifest.json')))
for it in man['experts'][:k]:
    for r, codes, sc, tsc, isc, nb in nc.matrices(d, it):
        R, C = codes.shape
        z = sc == 0
        rowf = z.mean(1)
        colf = z.mean(0)
        zc = (codes & 7) == 0
        colz = zc.mean(0)
        rowz = zc.mean(1)
        blk = codes.reshape(R, C // 16, 16)
        # codes inside scale-0 blocks
        inz = blk[z]
        print(it['tag'], r, 'zero-scale blocks', round(float(z.mean()), 4),
              'rows all-zero-scale', int((rowf == 1).sum()), 'rows none', int((rowf == 0).sum()),
              'blockcols all-zero', int((colf == 1).sum()), 'blockcols none', int((colf == 0).sum()), 'of', C // 16)
        if inz.size:
            print('   codes in zero-scale blocks hist', np.round(np.bincount(inz.ravel(), minlength=16) / inz.size, 4).tolist())
        print('   zero-code frac per column: quantiles', np.round(np.quantile(colz, [0, .1, .5, .9, 1]), 3).tolist(),
              ' per row', np.round(np.quantile(rowz, [0, .1, .5, .9, 1]), 3).tolist())
        nzs = sc[~z]
        if nzs.size:
            print('   nonzero scale bytes quantiles', np.quantile(nzs, [0, .01, .1, .5, .9, .99, 1]).tolist())
