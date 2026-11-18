import ctypes, struct, os, sys, glob, hashlib
D="/tmp/wnr3-data"; lib=ctypes.CDLL(D+"/libwex_nr3_tools.so")
model=open(D+"/recs/model.bin","rb").read(); table=open(D+"/recs/entries.bin","rb").read()
ES=72; n=0; bad=[]
for p in sorted(glob.glob(D+"/recs/rec_*.bin")):
    l,e=map(int,os.path.basename(p)[4:-4].split("_"))
    ent=table[(l*192+e)*ES:(l*192+e+1)*ES]; rec=open(p,"rb").read()
    raw=18800640; out=ctypes.create_string_buffer(raw)
    r=lib.wex_nr3_decode(model,ent,rec,l,e,out)
    n+=1
    if r: bad.append((l,e,r))
print(n,"decoded, failures:",bad)
