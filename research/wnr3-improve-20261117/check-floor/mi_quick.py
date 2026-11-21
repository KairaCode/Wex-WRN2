"""Quick independent in-sample MI check: I(a3 ; a1 | srel3, srel1) in bits per w3 weight, plug-in + Miller-Madow."""
import sys, json, numpy as np, struct
sys.path.insert(0,'.')
from sib_check import mats
def H(c):
    c=c[c>0]; n=c.sum(); return -(c*np.log2(c/n)).sum()/n
def mi(l,e):
    ((s1,q1),(s3,q3)),anc3=mats(l,e)
    rec=open("<TMP>/wnr3-data/recs/rec_%02d_%03d.bin"%(l,e),"rb").read(32); anc1=rec[24]
    blk=np.arange(q3.size)//32
    z=(np.clip(s3[blk]-anc3,-3,3)+3)*7+np.clip(s1[blk]-anc1,-3,3)+3
    a3=q3&7; a1=q1&7; n=q3.size
    j=np.bincount((z*8+a1)*8+a3,minlength=49*64).reshape(49,8,8).astype(float)
    tot=0; K=0
    for zz in range(49):
        t=j[zz]; m=t.sum()
        if m==0: continue
        tot+=m*(H(t.sum(0))+H(t.sum(1))-H(t.ravel()))
        K+=((t.sum(0)>0).sum()-1)*((t.sum(1)>0).sum()-1)  # MM correction dof for MI (approx)
    return tot/n - K/(2*n*np.log(2))
for k in [(0,7),(0,11),(0,101),(5,80),(11,14),(22,21),(33,28)]:
    print(k, round(mi(*k),5))
