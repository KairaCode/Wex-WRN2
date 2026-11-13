import json, hashlib, os, sys
d = r"<RESEARCH-PC>\team4-data\extra"
man = json.load(open(os.path.join(d, "manifest.json")))
sm = json.load(open(r"<RESEARCH-PC>\v41-sample\manifest.json"))
taken = {(e["layer"], e["expert"]) for e in sm["experts"]}
bad = []; n = 0; tot = 0
for e in man["experts"]:
    p = os.path.join(d, e["file"]); b = open(p, "rb").read(); n += 1; tot += len(b)
    if hashlib.sha256(b).hexdigest() != e["sha256"] or len(b) != sum(t["bytes"] for t in e["tensors"]): bad.append(e["file"])
    if e["kind"] == "layers" and (e["layer"], e["expert"]) in taken: bad.append("overlap " + e["file"])
lay = {}
for e in man["experts"]:
    if e["kind"] == "layers": lay[e["layer"]] = lay.get(e["layer"], 0) + 1
print("files", n, "bytes", tot, "bad", bad, "layers covered", len(lay), "routed", sum(lay.values()), "mtp", sum(1 for e in man["experts"] if e["kind"] == "mtp"))
print("repo", man["repo"], man["revision"], "tensor_bytes", man["tensor_bytes"], "header_bytes", man["header_bytes"])
