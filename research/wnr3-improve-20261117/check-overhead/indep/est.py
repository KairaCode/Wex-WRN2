# Independent estimate of the lane's top numbers (lenc, sd, inline per-record scales, tail5) from the STORED
# production records, written without the lane's code. Scales are decoded here with the reference library.
import ctypes, struct, os, numpy as np, json, math
D = "/tmp/wnr3-data"; RAW = 18800640
lib = ctypes.CDLL(D + "/libwex_nr3_tools.so")
model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()
TEST = [(l, (53 * l + 7) % 192) for l in range(40)] + [(39, 120)]

def lenc_bytes(lens):
    tot = 0
    for g in range(0, len(lens), 32):
        grp = lens[g:g + 32]; lo, hi = min(grp), max(grp); w = (hi - lo).bit_length()
        tot += 3 + (len(grp) * w + 7) // 8
    return tot

def quant(counts):   # my own quantiser: each present symbol >= 1, total 16384, largest remainder
    c = np.array(counts, float); tot = c.sum(); f = np.floor(c / tot * 16384).astype(int); f[(c > 0) & (f == 0)] = 1
    while f.sum() > 16384: f[np.argmax(f)] -= 1
    while f.sum() < 16384: f[np.argmax(c / tot * 16384 - f)] += 1
    return f

rows = []
for (l, e) in TEST:
    rec = open(f"{D}/recs/rec_{l:02d}_{e:03d}.bin", "rb").read()
    ent = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; v = struct.unpack('<QIII6H6H3I3II', ent)
    magic, mid, SP, WP, Cw, Cs = struct.unpack('<6I', rec[:24]); anchors = list(rec[24:27])
    sc = [v[4 + 2 * m] * v[5 + 2 * m] for m in range(3)]
    nss = sum((c + Cs - 1) // Cs for c in sc); nws = sum((32 * c + Cw - 1) // Cw for c in sc); n = nss + nws
    lens = list(struct.unpack(f'<{n}H', rec[32:32 + 2 * n])); used = 32 + 2 * n + SP + WP
    assert sum(lens) == SP + WP
    out = ctypes.create_string_buffer(RAW); coded = ctypes.create_string_buffer(RAW); st = (ctypes.c_uint32 * 16)()
    assert lib.wex_nr3_decode(model, ent, rec, l, e, out) == 0
    assert lib.wex_nr3_prepare(ent, out, 1, coded, st) == rec[27]
    s = np.frombuffer(coded.raw[:sum(sc)], np.uint8).astype(int)
    # per-record 16-symbol table relative to the anchor (sym = s - a + 7 in 0..14, else escape 15 + 8 raw bits)
    bits = 0.0; tbytes = 0; off = 0
    for m in range(3):
        r = s[off:off + sc[m]] - anchors[m] + 7; off += sc[m]
        esc = (r < 0) | (r > 14); r = np.where(esc, 15, r)
        cnt = np.bincount(r, minlength=16); f = quant(cnt)
        bits += -(cnt[cnt > 0] * np.log2(f[cnt > 0] / 16384)).sum() + 8 * esc.sum()
        tbytes += 2 + 2 * (int((cnt > 0).sum()) - 1)
    wl = lens[nss:]
    sd_ws = sum(1 for g in range(nss, n) if rec[32 + 2 * n + sum(lens[:g])] == 0)   # first flush byte 0 -> 3-byte start
    sd_all = sd_ws + sum(1 for g in range(nss) if rec[32 + 2 * n + sum(lens[:g])] == 0)
    nib = 32 * sum(sc)
    bpn = (WP - nws * (4 - 23 / 8)) * 8 / nib   # coded bits per nibble (flush minus hash slot removed)
    rows.append(dict(l=l, e=e, used=used, total=len(rec), SP=SP, nss=nss, nws=nws,
                     lenc=2 * n - lenc_bytes(lens), sd=sd_all, inline_scale_bytes=bits / 8 + tbytes, tbytes=tbytes,
                     C_saving=(2 * n - lenc_bytes(wl)) + sd_ws + SP - (bits / 8 + tbytes) - 8,
                     tail5=nws * 5 * bpn / 8 - 0, bpn=bpn))
    print(l, e, rows[-1]["C_saving"], flush=True)
json.dump(rows, open(os.path.dirname(os.path.abspath(__file__)) + "/est.json", "w"), indent=0)
typ = [r for r in rows if (r["l"], r["e"]) != (39, 120)]; o = [r for r in rows if (r["l"], r["e"]) == (39, 120)][0]
for k in ["lenc", "sd", "inline_scale_bytes", "tbytes", "C_saving", "tail5", "bpn", "SP"]:
    a = np.mean([r[k] for r in typ]); print(f"{k:20s} typ mean {a:12.2f} ({100*a/RAW:.4f}% raw)  L39E120 {o[k]:.1f}")
