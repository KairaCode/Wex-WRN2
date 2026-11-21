# quick look at coded scale statistics (in-sample empirical entropies, exploration only)
import numpy as np, json
meta=json.load(open('meta.json'))
def H(c):
    c=c[c>0].astype(float); p=c/c.sum(); return -(c*np.log2(p)).sum()
def condH(a,b):  # H(a|b) total bits
    k=a.astype(np.int64)*256+b.astype(np.int64)
    joint=np.bincount(k,minlength=65536).reshape(256,256)
    return H(joint.ravel())-H(joint.sum(0))
for r in meta:
    if not ((r['layer'] in (0,20,39)) ): continue
    s=np.fromfile(f"scales/{r['layer']:02d}_{r['expert']:03d}.bin",np.uint8)
    off=0; out=[]
    for m,(R,C) in enumerate(r['scale_shapes']):
        a=s[off:off+R*C].reshape(R,C); off+=R*C
        n=a.size
        h0=H(np.bincount(a.ravel(),minlength=256))/n
        left=condH(a[:,1:].ravel(),a[:,:-1].ravel())/(n-R)
        up=condH(a[1:,:].ravel(),a[:-1,:].ravel())/(n-C)
        rowmed=np.median(a,axis=1).astype(np.uint8)
        rowc=condH(a.ravel(),np.repeat(rowmed,C))/n
        colmed=np.median(a,axis=0).astype(np.uint8)
        colc=condH(a.ravel(),np.tile(colmed,R))/n
        out.append(f"m{m} {R}x{C} uniq={len(np.unique(a))} min={a.min()} max={a.max()} H0={h0:.3f} |left={left:.3f} |up={up:.3f} |rowmed={rowc:.3f} |colmed={colc:.3f}")
    print(r['layer'],r['expert'],r['SP'],r['scale_streams'], '\n  '+'\n  '.join(out))
