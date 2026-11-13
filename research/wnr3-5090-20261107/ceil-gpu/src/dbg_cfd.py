import os, sys, numpy as np, cupy as cp
HERE = os.path.dirname(os.path.abspath(__file__))
enc = sys.argv[1]
com = np.load(os.path.join(enc, 'common.npz'))
spm = int(com['spm']); nmat = 3
src = open(os.path.join(HERE, 'wnr2dec.cu')).read()
names = ['wdec<8,1,0,64,1>', 'wdec<16,1,0,64,1>']
mod = cp.RawModule(code=src, options=('-arch=sm_86', '-std=c++17'), name_expressions=names)
sc_c = cp.asarray(com['scales_ceil']); sc_o = com['scales']
med = cp.asarray(com['med_ceil']); mt = cp.asarray(com['mtype'])
for vf, kn, alpha in (('CFD', names[0], 8), ('CF', names[1], 16)):
    z = np.load(os.path.join(enc, f'var_{vf}.npz'))
    pay = z['payload']; payp = np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])
    offs = np.zeros(z['lengths'].size + 1, np.uint32); offs[1:] = np.cumsum(z['lengths'].astype(np.int64))
    signs = cp.asarray(z['signs']) if 'signs' in z.files else cp.zeros(1, cp.uint32)
    out = cp.zeros(nmat * spm * 1024, cp.uint8); err = cp.zeros(1, cp.int32)
    osc = sc_c.copy()
    fn = mod.get_function(kn)
    bpm = -(-spm // 64)
    smem = 16 + 259 * alpha * 2
    fn((nmat * bpm,), (64,), (cp.asarray(payp.view(np.uint32)), cp.asarray(offs), sc_c, signs, cp.asarray(z['tables'].reshape(-1)),
                              cp.zeros(24, cp.uint16), mt, med, np.int32(0), np.int32(spm), np.int32(bpm), out, err, osc), shared_mem=smem)
    cp.cuda.runtime.deviceSynchronize()
    o = osc.get()[:nmat * spm * 64].astype(np.int16); ref = sc_o[:nmat * spm * 64].astype(np.int16); t = com['scales_ceil'][:nmat * spm * 64].astype(np.int16)
    d = np.flatnonzero(o != ref)
    print(vf, 'diff', d.size, 'first', d[:10], 'out', o[d[:10]], 'ref', ref[d[:10]], 'trans', t[d[:10]], 'err', int(err.get()[0]))
    print(' out-ref hist', np.unique(o[d] - ref[d], return_counts=True) if d.size else None)
    w = out.get(); print(' weights equal', np.array_equal(w, com['raw'][:w.size]))
