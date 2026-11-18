"""Combine the measurements into the floor table -> out/floor.json and stdout.
Per TEST record and per part (scales of all three matrices; nibbles of w1, w2, w3) the best of
  prod  : ideal cost under the production WNM2 model (decomp.py; no framing)
  cmf0  : adaptive CM with srel/pm/mx contexts only (cm -f 0)
  cmall : adaptive CM, all features (cm -f 0xffffffff)
  cmpr  : cmall pre-trained on the TRAIN record L20 E175 (subset of 14 TEST records)
plus 1 B per record for the choice, 32 B header + 8 B coder flush per record. These are ideal code lengths
(-log2 p of an adaptive model = what an arithmetic coder achieves within a few bytes): an upper bound on the
floor, held-out (adaptive models learn only from the record's own past symbols; static/prime tables never saw it)."""
import json, os
import numpy as np
from load import H
RAW = 18800640
J = lambda p: {(d["layer"], d["expert"]): d for d in map(json.loads, open(os.path.join(H, "out", p)))}
dec, F, Z, P = J("decomp.jsonl"), J("cm_full_test.jsonl"), J("cm_f0_test.jsonl"), J("cm_primed_test.jsonl")
parts = lambda d: {"scales": sum(d["scale_bits_m%d" % m] for m in range(3)) / 8,
                   **{"w%d" % (m + 1): (d["mag_bits_m%d" % m] + d["sign_bits_m%d" % m]) / 8 for m in range(3)}}

def prod_parts(k):
    # production ideal split per matrix is not stored per matrix in decomp; recompute cheaply
    from decomp import one
    return None

rows = []
for k in sorted(F):
    D = dec[k]
    cands = {"cmall": parts(F[k]), "cmf0": parts(Z[k])}
    if k in P: cands["cmpr"] = parts(P[k])
    best = {p: min(c[p] for c in cands.values()) for p in ("scales", "w1", "w2", "w3")}
    which = {p: min(cands, key=lambda c: cands[c][p]) for p in best}
    cm_w = sum(best[p] for p in ("w1", "w2", "w3"))
    # the production model is also a candidate for each whole part group (weights / scales)
    w_best = min(cm_w, D["ideal_w_bytes"]); s_best = min(best["scales"], D["ideal_s_bytes"])
    floor = w_best + s_best + 1 + 32 + 8
    used = D["total_bytes"] - D["pad"]
    rows.append(dict(layer=k[0], expert=k[1], stored=D["total_bytes"], stored_unpadded=used,
                     framing=D["total_bytes"] - D["ideal_w_bytes"] - D["ideal_s_bytes"],
                     prod_ideal=D["ideal_w_bytes"] + D["ideal_s_bytes"], floor=floor,
                     floor_w=w_best, floor_s=s_best, which=which, primed=k in P,
                     stored_ratio=D["total_bytes"] / RAW, floor_ratio=floor / RAW,
                     gap_total_pct=100 * (D["total_bytes"] - floor) / RAW,
                     gap_framing_pct=100 * (D["total_bytes"] - D["ideal_w_bytes"] - D["ideal_s_bytes"]) / RAW,
                     gap_model_w_pct=100 * (D["ideal_w_bytes"] - w_best) / RAW,
                     gap_model_s_pct=100 * (D["ideal_s_bytes"] - s_best) / RAW,
                     floor_bpw=8 * w_best / D["n"], prod_ideal_bpw=D["ideal_bpw"], stored_bpw=D["real_bpw"]))
out = {"records": rows}
def agg(sel, name):
    r = [x for x in rows if sel(x)]
    a = {f: float(np.mean([x[f] for x in r])) for f in ("stored_ratio", "floor_ratio", "gap_total_pct", "gap_framing_pct",
                                                         "gap_model_w_pct", "gap_model_s_pct", "floor_bpw", "prod_ideal_bpw", "stored_bpw")}
    a.update(n=len(r), gap_total_min=float(min(x["gap_total_pct"] for x in r)), gap_total_max=float(max(x["gap_total_pct"] for x in r)),
             gap_model_w_min=float(min(x["gap_model_w_pct"] for x in r)), gap_model_w_max=float(max(x["gap_model_w_pct"] for x in r)))
    out[name] = a
    print(name, {kk: round(v, 5) if isinstance(v, float) else v for kk, v in a.items()})
agg(lambda x: (x["layer"], x["expert"]) != (39, 120), "TEST40_one_per_layer")
agg(lambda x: x["layer"] < 8, "layers_0_7")
agg(lambda x: 8 <= x["layer"] < 39, "layers_8_38")
agg(lambda x: (x["layer"], x["expert"]) == (39, 154), "L39_E154")
agg(lambda x: (x["layer"], x["expert"]) == (39, 120), "L39_E120")
agg(lambda x: x["primed"] and (x["layer"], x["expert"]) != (39, 120), "primed_subset_13")
json.dump(out, open(os.path.join(H, "out", "floor.json"), "w"), indent=0)
for x in rows:
    print(x["layer"], x["expert"], "stored %.5f floor %.5f gap %.3f%% (framing %.3f, model w %.3f s %.3f)" % (
        x["stored_ratio"], x["floor_ratio"], x["gap_total_pct"], x["gap_framing_pct"], x["gap_model_w_pct"], x["gap_model_s_pct"]), x["which"])
