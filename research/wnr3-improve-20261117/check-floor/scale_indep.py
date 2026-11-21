"""Independent scale-table check: per record, order-0 empirical entropy of each matrix's coded scales (exact counts,
no quantisation = lower bound for a per-record table) + table bytes (1 + 3 per used symbol) vs the production ideal;
and the real stored scale payload SP vs rt_scale own_bytes (real)."""
import json, struct, numpy as np
D="<TMP>/wnr3-data"; RAW=18800640; ENTRY=struct.Struct('<QIII6H6H3I3II'); T=open(D+"/recs/entries.bin","rb").read()
blob=np.fromfile(D+"/recs/model.bin",np.uint8); S=np.frombuffer(blob[57866:57866+3*514].tobytes(),"<u2").reshape(3,257).astype(float); SF=np.diff(S,axis=1)
rt={(d["layer"],d["expert"]):d for d in map(json.loads,open("out/rt_scale.jsonl"))}
rows=[]
for (l,e),d in sorted(rt.items()):
    ent=ENTRY.unpack(T[(l*192+e)*72:(l*192+e+1)*72]); b=np.fromfile("out/coded/rec_%02d_%03d.coded"%(l,e),np.uint8)
    H0=prod=tb=0.0
    for m in range(3):
        R,NB=ent[4+2*m],ent[5+2*m]; c=np.bincount(b[ent[16+m]:ent[16+m]+R*NB],minlength=256).astype(float)
        p=c[c>0]/c.sum(); H0+=-(c[c>0]*np.log2(p)).sum(); prod+=(c*-np.log2(np.where(SF[m]>0,SF[m],1)/16384)).sum(); tb+=1+3*(c>0).sum()
    rows.append(dict(k=(l,e),bound=100*(H0/8+tb-prod/8)/RAW,real=100*(d["own_bytes"]+d["own_table_bytes"]-d["stored_bytes"])/RAW))
x=[r for r in rows if r["k"]!=(39,120)]
print("n",len(x),"entropy-bound mean %.5f  real mean %.5f  real max %.5f"%(np.mean([r["bound"] for r in x]),np.mean([r["real"] for r in x]),max(r["real"] for r in x)))
print("E120",[ (round(r["bound"],3),round(r["real"],3)) for r in rows if r["k"]==(39,120)])
