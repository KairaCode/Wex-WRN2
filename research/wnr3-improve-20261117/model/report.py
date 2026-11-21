#!/usr/bin/env python3
"""Reduce out/real_*.json to the results table (results.json + results_table.md).
Delta = record bytes (unpadded) - stored production record bytes (unpadded), per record, held-out.
normal = the 40 TEST records with anchors (120,120,120); shifted = the 38 records of the container whose
rule-g anchors differ (37 fetched for this + L39 E120 of TEST). Container extrapolation (stratified):
7,642 x mean(normal) + 38 x mean(shifted) (+ the extra table bytes once)."""
import json, glob, os, numpy as np
W = "<TMP>/wnr3-research/model"; RAW = 18800640; N_NORMAL, N_SHIFT = 7642, 38
CONT = 123312910336
rows = {}
for p in sorted(glob.glob(W + "/out/real_*.json")):
    name = os.path.basename(p)[5:-5]
    d = json.load(open(p)); rows[name] = d
out = []
def paired_vs(name, ref, d):
    if ref not in rows: return None
    a = {x["tag"]: x["used"] for x in d["rows"]}; b = {x["tag"]: x["used"] for x in rows[ref]["rows"]}
    v = [a[t] - b[t] for t in a if t != "39_120" and t in b]
    return float(np.mean(v))
for name, d in rows.items():
    if name.endswith("_extra") or "_split" in name: continue
    r = d["rows"]; dl = np.array([x["used"] - x["base_used"] for x in r]); tags = [x["tag"] for x in r]
    no = np.array([t != "39_120" for t in tags])
    normal = dl[no]; outl = dl[~no]
    ex = rows.get(name + "_extra")
    shifted = None
    if ex:
        de = np.array([x["used"] - x["base_used"] for x in ex["rows"]])
        shifted = np.concatenate([de, outl])
    tb = d["summary"].get("table_bytes") or 0
    base_tb = 52560 if d["summary"].get("table_bytes") is not None else 0   # WNM2 tables as u16 frequencies
    row = dict(variant=name, verified=d["summary"]["all_ok"] and (ex["summary"]["all_ok"] if ex else True),
               n_test=len(r), normal_mean_B=float(normal.mean()), normal_pct_raw=float(normal.mean() / RAW * 100),
               normal_median_B=float(np.median(normal)), normal_min_B=float(normal.min()), normal_max_B=float(normal.max()),
               normal_sem_B=float(normal.std(ddof=1) / np.sqrt(len(normal))),
               L39E120_B=float(outl.sum()) if len(outl) else None,
               shifted_mean_B=float(shifted.mean()) if shifted is not None else None, n_shifted=int(len(shifted)) if shifted is not None else 0,
               sample41_mean_B=float(dl.mean()), extra_table_bytes=int(tb - base_tb) if tb else 0,
               normal_vs_P14_B=paired_vs(name, "P14", d))
    if shifted is not None:
        c = N_NORMAL * normal.mean() + N_SHIFT * shifted.mean() + row["extra_table_bytes"]
        row["container_B"] = float(c); row["container_pct"] = float(c / CONT * 100)
        row["container_shifted_part_B"] = float(N_SHIFT * shifted.mean())
    else:
        c = 7680 * normal.mean() + row["extra_table_bytes"]
        row["container_B_normal_only"] = float(c); row["container_pct_normal_only"] = float(c / CONT * 100)
    out.append(row)
out.sort(key=lambda x: x["normal_mean_B"])
json.dump(out, open(W + "/results.json", "w"), indent=1)
lines = ["| variant | verified | normal records: mean B/rec (% raw) | median | min..max | sem | vs P14 (T80 retrain) B | L39E120 B | shifted (n) mean B | container MB (stratified) | extra table B |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in out:
    cont = ("%.1f" % (r["container_B"] / 1e6)) if "container_B" in r else ("%.1f (normal only)" % (r["container_B_normal_only"] / 1e6))
    lines.append("| %s | %s | %+.0f (%+.4f%%) | %+.0f | %+.0f..%+.0f | %.0f | %s | %s | %s | %s | %d |" % (
        r["variant"], "41/41" + (" + %d" % (r["n_shifted"] - 1) if r["n_shifted"] else "") if r["verified"] else "NO",
        r["normal_mean_B"], r["normal_pct_raw"], r["normal_median_B"], r["normal_min_B"], r["normal_max_B"], r["normal_sem_B"],
        "%+.0f" % r["normal_vs_P14_B"] if r["normal_vs_P14_B"] is not None else "-",
        "%+.0f" % r["L39E120_B"] if r["L39E120_B"] is not None else "-",
        ("%+.0f (%d)" % (r["shifted_mean_B"], r["n_shifted"])) if r["shifted_mean_B"] is not None else "-", cont, r["extra_table_bytes"]))
open(W + "/results_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# split B (robustness): TRAIN = (37l+11),(53l+7); TEST = (37l+101) + L39 E120
lb = ["", "Split B (TRAIN = experts (37l+11),(53l+7); TEST = (37l+101) + L39 E120):", "",
      "| variant | verified | normal records mean B/rec (% raw) | median | sem | L39E120 B |", "|---|---|---|---|---|---|"]
sb = {}
for name, d in rows.items():
    if not name.endswith("_splitB"): continue
    r = d["rows"]; dl = np.array([x["used"] - x["base_used"] for x in r]); no = np.array([x["tag"] != "39_120" for x in r])
    sb[name] = dict(verified=d["summary"]["all_ok"], normal_mean_B=float(dl[no].mean()), median=float(np.median(dl[no])),
                    sem=float(dl[no].std(ddof=1) / np.sqrt(no.sum())), L39E120_B=float(dl[~no].sum()))
    lb.append("| %s | %s | %+.0f (%+.4f%%) | %+.0f | %.0f | %+.0f |" % (name, "41/41" if sb[name]["verified"] else "NO", sb[name]["normal_mean_B"],
              sb[name]["normal_mean_B"] / RAW * 100, sb[name]["median"], sb[name]["sem"], sb[name]["L39E120_B"]))
json.dump(dict(splitA=out, splitB=sb), open(W + "/results.json", "w"), indent=1)
open(W + "/results_table.md", "a").write("\n".join(lb) + "\n")
print("\n".join(lb))
