import json
for f in [r"<RESEARCH-PC>\team4\ceil-refuter\out\stats_sample.json", r"<RESEARCH-PC>\team4\ceil-refuter\out\stats_extra.json"]:
    st = json.load(open(f))
    for fn in ["L36_E066.bin", "L03_E142.bin", "L38_E167.bin", "L00_E097.bin"]:
        if fn in st:
            for j, m in enumerate(st[fn]["m"]):
                h = m["scale_hist_top"]
                print(fn, j, "med_o", m["med_o"], "med_c", m["med_c"], "even", round(m["even"]/m["blocks"]*100, 2), "hist", h)
pe = json.load(open(r"<RESEARCH-PC>\team4\ceil-refuter\out\per_expert_sample.json")) + json.load(open(r"<RESEARCH-PC>\team4\ceil-refuter\out\per_expert_extra.json"))
for r in pe:
    if r["file"] in ("L36_E066.bin", "L03_E142.bin", "L38_E167.bin"): print(r)
