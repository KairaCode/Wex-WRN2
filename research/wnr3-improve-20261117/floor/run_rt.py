"""Real rANS round trips (rt.c) of the w3 streams on TEST records: variant 0 with the production model (must be
byte-identical to the stored streams) and the sibling variants with TRAIN-trained tables (split at layer 8).
  python3 run_rt.py L:E,... -> appends out/rt.jsonl"""
import sys, json, os, subprocess
from run_cm import args_for
from load import MAN, H
RAW = 18800640
VAR = {"V1": 1, "V3": 3}
if __name__ == "__main__":
    ks = [tuple(map(int, t.split(":"))) for t in sys.argv[1].split(",")]
    for k in ks:
        f, geo, anc = args_for(*k)
        rec = "/tmp/wnr3-data/recs/rec_%02d_%03d.bin" % k
        base = None
        runs = [("prod", "-", 0)] + [(v, os.path.join(H, "out", "tables", "m2_%s_%s.u16" % (v, "lo" if k[0] < 8 else "hi")), VAR[v]) for v in VAR]
        for name, tab, vid in runs:
            p = subprocess.run([os.path.join(H, "rt"), f, rec, "/tmp/wnr3-data/recs/model.bin", tab, str(vid)] + geo,
                               capture_output=True, text=True)
            d = json.loads(p.stdout); d.update(layer=k[0], expert=k[1], name=name, exit=p.returncode)
            if name == "prod": base = d["w3_bytes"]
            d["delta_bytes_vs_prod"] = d["w3_bytes"] - base
            d["delta_pct_raw"] = 100 * (d["w3_bytes"] - base) / RAW
            print(json.dumps(d), flush=True)
            with open(os.path.join(H, "out", "rt.jsonl"), "a") as fo: fo.write(json.dumps(d) + "\n")
