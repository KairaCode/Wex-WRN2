"""Driver for ./cm over records; appends JSON lines (with the stored record's real sizes) to an output file.
  python3 run_cm.py OUT.jsonl JOBS FMASK L:E,... [--prime L:E]   (prime = TRAIN record coded first)"""
import sys, json, os, subprocess
from multiprocessing.pool import ThreadPool
from load import MAN, H

def args_for(l, e):
    r = MAN[(l, e)]; ent = r["entry"]
    geo = [str(x) for x in ent[16:19] + ent[19:22]]
    for m in range(3): geo += [str(ent[4 + 2 * m]), str(ent[5 + 2 * m])]
    return os.path.join(H, "out", "coded", "rec_%02d_%03d.coded" % (l, e)), geo, [str(a) for a in r["anchors"]]

def run(k, fmask, prime):
    f, geo, anc = args_for(*k)
    cmd = [os.path.join(H, "cm"), "-f", str(fmask)]
    if prime:
        pf, _, panc = args_for(*prime); cmd += ["-p", pf, "-P", ",".join(panc)]
    out = subprocess.run(cmd + [f] + geo + anc, capture_output=True, text=True, check=True).stdout
    d = json.loads(out); r = MAN[k]
    d.update(layer=k[0], expert=k[1], prime=list(prime) if prime else None, total_bytes=r["total_bytes"], SP=r["SP"], WP=r["WP"])
    return d

if __name__ == "__main__":
    outp, jobs, fmask, recs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3], 0), sys.argv[4]
    prime = tuple(map(int, sys.argv[6].split(":"))) if len(sys.argv) > 6 and sys.argv[5] == "--prime" else None
    ks = [tuple(map(int, t.split(":"))) for t in recs.split(",")]
    with ThreadPool(jobs) as p:
        for d in p.imap_unordered(lambda k: run(k, fmask, prime), ks):
            with open(outp, "a") as f: f.write(json.dumps(d) + "\n")
            print(json.dumps(d), flush=True)
