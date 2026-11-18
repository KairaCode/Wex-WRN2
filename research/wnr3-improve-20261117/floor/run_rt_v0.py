"""rt.c with variant 0 contexts but TRAIN-trained tables (layer split at 8): separates 'retrained tables' from
the sibling context. -> out/rt_v0train.jsonl"""
import json, os, subprocess
from run_cm import args_for
from load import test_set, H
RAW = 18800640
base = {(d["layer"], d["expert"]): d["w3_bytes"] for d in map(json.loads, open(os.path.join(H, "out", "rt.jsonl"))) if d["name"] == "prod"}
with open(os.path.join(H, "out", "rt_v0train.jsonl"), "w") as fo:
    for k in test_set():
        f, geo, anc = args_for(*k)
        tab = os.path.join(H, "out", "tables", "m2_V0_%s.u16" % ("lo" if k[0] < 8 else "hi"))
        p = subprocess.run([os.path.join(H, "rt"), f, "/tmp/wnr3-data/recs/rec_%02d_%03d.bin" % k, "/tmp/wnr3-data/recs/model.bin", tab, "0"] + geo, capture_output=True, text=True)
        d = json.loads(p.stdout); d.update(layer=k[0], expert=k[1], name="V0train", exit=p.returncode, delta_pct_raw=100 * (d["w3_bytes"] - base[k]) / RAW)
        fo.write(json.dumps(d) + "\n")
