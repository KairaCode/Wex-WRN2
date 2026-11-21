"""Summarise cm JSON lines: bit/weight of nibbles, bit/scale, and record bytes vs stored."""
import sys, json
RAW = 18800640
def summ(d, overhead=0):
    n = sum(d["n_m%d" % m] for m in range(3)); ns = sum(d["ns_m%d" % m] for m in range(3))
    wb = sum(d["mag_bits_m%d" % m] + d["sign_bits_m%d" % m] for m in range(3)); sb = sum(d["scale_bits_m%d" % m] for m in range(3))
    rec = (wb + sb) / 8 + overhead
    return dict(L=d["layer"], E=d["expert"], f=hex(d["fmask"]), primed=d["primed"], bpw=wb / n,
                bpw_m=[round((d["mag_bits_m%d" % m] + d["sign_bits_m%d" % m]) / d["n_m%d" % m], 5) for m in range(3)],
                bps=sb / ns, cm_bytes=rec, stored=d["total_bytes"], WP=d["WP"], SP=d["SP"],
                d_vs_stored_pct=100 * (rec - d["total_bytes"]) / RAW, d_w_vs_WP_pct=100 * (wb / 8 - d["WP"]) / RAW,
                d_s_vs_SP_pct=100 * (sb / 8 - d["SP"]) / RAW)
if __name__ == "__main__":
    for l in open(sys.argv[1]):
        s = summ(json.loads(l)); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()})
