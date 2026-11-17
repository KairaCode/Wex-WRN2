"""rt_scale.c on all 121 records -> out/rt_scale.jsonl (per-record scale tables, real rANS bytes)."""
import json, os, subprocess
from run_cm import args_for
from load import MAN, H
RAW = 18800640
with open(os.path.join(H, "out", "rt_scale.jsonl"), "w") as fo:
    for k in sorted(MAN):
        f, geo, anc = args_for(*k)
        p = subprocess.run([os.path.join(H, "rt_scale"), f, "/tmp/wnr3-data/recs/rec_%02d_%03d.bin" % k, "/tmp/wnr3-data/recs/model.bin"] + geo, capture_output=True, text=True)
        d = json.loads(p.stdout); d.update(layer=k[0], expert=k[1], exit=p.returncode,
                                           delta_pct_raw=100 * (d["own_bytes"] + d["own_table_bytes"] - d["prod_bytes"]) / RAW)
        fo.write(json.dumps(d) + "\n")
