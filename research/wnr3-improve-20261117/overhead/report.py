# Reduce out/run_test.json (ovh_proto run on the TEST records, TRAIN model) to the savings table.
# usage: python3 report.py out/run_test.json > out/report.txt   (also writes out/report.json)
import json, sys, struct, numpy as np
RAW = 18800640
r = json.load(open(sys.argv[1]))
names = [c["name"] for c in r["configs"]]
E = r["experts"]
assert all(e["harness_ok"] for e in E), "harness failed"
assert all(c[0] == 1 for e in E for c in e["cfg"]), "round trip failed"
typ = [e for e in E if (e["layer"], e["expert"]) != (39, 120)]
out = [e for e in E if (e["layer"], e["expert"]) == (39, 120)]
# container facts (exact): every record's header (headers.bin) and total_bytes (entries.bin)
h = open("headers.bin", "rb").read()
SP = np.array([struct.unpack_from("<I", h, 32 * i + 8)[0] for i in range(7680)])
anch = np.array([list(h[32 * i + 24:32 * i + 27]) for i in range(7680)])
n_shift = int((anch != 120).any(1).sum())
print(f"TEST records: {len(E)} ({len(typ)} typical + L39 E120); all harness byte-exact, all {len(names)} profiles round-trip exact")
st = np.array([e["stored_total"] for e in E]); su = np.array([e["stored_used"] for e in E])
b0 = np.array([e["cfg"][0][1] for e in E])
print(f"stored records (production model): mean total {st.mean():.0f} B = {st.mean()/RAW:.5f} of raw; used {su.mean():.0f}")
print(f"WNR3 re-encoded with the TRAIN model: mean used {b0.mean():.0f} B (typical: TRAIN-model - stored = {np.mean([e['cfg'][0][1]-e['stored_used'] for e in typ]):+.0f} B/record)")
print(f"container (exact, 7,680 headers): mean SP {SP.mean():.0f} B; records with an anchor != 120: {n_shift}")
print()
print(f"{'profile':22s} {'saved B/rec (40 typ)':>20s} {'min..max':>18s} {'% raw':>7s} {'L39E120':>9s} {'container MB':>12s} {'+64B-align B':>12s} streams")
res = {}
for k, nm in enumerate(names):
    d = np.array([e["cfg"][0][1] - e["cfg"][k][1] for e in typ])
    dpad = np.array([e["cfg"][0][6] - e["cfg"][k][7] for e in typ])   # 4096-padded WNR3 -> 64-padded profile
    o = out[0]["cfg"][0][1] - out[0]["cfg"][k][1]
    cont = d.mean() * 7680 / 1e6
    res[nm] = dict(saved_mean=d.mean(), saved_min=int(d.min()), saved_max=int(d.max()), pct_raw=100 * d.mean() / RAW,
                   L39E120_saved=int(o), container_MB_extrapolated=cont, saved_with_64B_alignment_mean=dpad.mean(),
                   streams=E[0]["cfg"][k][8], config=r["configs"][k])
    print(f"{nm:22s} {d.mean():20.0f} {d.min():8d}..{d.max():<8d} {100*d.mean()/RAW:7.3f} {o:9d} {cont:12.1f} {dpad.mean():12.0f} {E[0]['cfg'][k][8]}")
json.dump(dict(per_profile=res, test_records=len(E), stored_mean_total=st.mean(), container_mean_SP=SP.mean(),
               container_records_anchor_not_120=n_shift), open(sys.argv[1].replace(".json", "_report.json"), "w"), indent=1)
