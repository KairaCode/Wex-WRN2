# Record sizes of the main profiles against the STORED production records (TEST records), as fraction of raw.
# usage: python3 final_table.py out/run_test.json
import json, sys, numpy as np
RAW = 18800640
r = json.load(open(sys.argv[1])); names = [c["name"] for c in r["configs"]]
E = r["experts"]; typ = [e for e in E if (e["layer"], e["expert"]) != (39, 120)]
st_all = np.mean([e["stored_total"] for e in E]); st_typ = np.mean([e["stored_total"] for e in typ])
print(f"stored (4096-padded) mean: all 41 {st_all:.0f} = {st_all/RAW:.5f}; 40 typical {st_typ:.0f} = {st_typ/RAW:.5f}")
print(f"{'profile':22s} {'typ used':>10s} {'typ pad64':>10s} {'pad64/raw':>9s} {'vs stored B':>11s} {'vs stored %raw':>14s} {'all41 pad64/raw':>15s}")
for n in ["wnr3_train", "A_lenc_sd", "B_A_rec16", "C_A_inl_rec16", "D_C_tail2_dig", "E_C_tail5_dig", "F_r16k16_inl_rec16", "G_C_w4096", "H_E_w4096", "I_C_w8192", "J_E_w8192"]:
    k = names.index(n)
    u = np.mean([e["cfg"][k][1] for e in typ]); p = np.mean([e["cfg"][k][7] for e in typ]); pa = np.mean([e["cfg"][k][7] for e in E])
    if n == "wnr3_train": p = np.mean([e["cfg"][k][6] for e in typ]); pa = np.mean([e["cfg"][k][6] for e in E])   # WNR3 keeps 4096
    print(f"{n:22s} {u:10.0f} {p:10.0f} {p/RAW:9.5f} {st_typ-p:11.0f} {100*(st_typ-p)/RAW:14.3f} {pa/RAW:15.5f}")
