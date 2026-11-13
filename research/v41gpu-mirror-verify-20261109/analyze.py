"""Post-run checks for host_launch.py: log lines, W3 token-id match, per-drive read split from the 1 s samples."""
import csv, json, os, re

OUT = os.path.dirname(os.path.abspath(__file__))
W3 = r"<KAIRA>\lab\v41-gpu-20261107\ab\tokens.json"
log = open(os.path.join(OUT, "colibri.stderr.log"), encoding="utf-8", errors="replace").read()
res = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))

def lines(pat):
    return [l for l in log.splitlines() if re.search(pat, l)]

checks = {
    "container_v3_U": lines(r"Container v3 active: U:\\kaira_wex\\v41-reap\\v4_flash_experts\.nr2\.wex"),
    "mirror_C": lines(r"Matching mirror container active: C:\\kaira_wex\\v41-reap\\v4_flash_experts\.nr2\.wex"),
    "mirror_ignored": lines(r"Ignoring mismatched mirror|COLI_WEX_WEIGHTS .* ignored"),
    "read_split": lines(r"read split"),
    "gpu_direct_enabled": lines(r"GPU-direct Neural-rANS reconstruction enabled"),
    "gpu_direct_counters": lines(r"v4_wex_gpu_direct reads="),
    "wex_batch": lines(r"V4_WEX_BATCH=1"),
    "gpu_head": lines(r"v4_gpu head:"),
    "ram_tiers": lines(r"ram_tiers"),
    "v4_direct": lines(r"v4_direct reads="),
    "wexprof": lines(r"wexprof"),
    "v4_mirror": lines(r"v4_mirror "),
    "dsv4_backend": lines(r"\[DSV4 CUDA\] backend|kaira_wex_cuda"),
}
uploads = lines(r"v4_gpu layer=\d+ uploaded=")
for k, v in checks.items():
    print(f"{k}: {len(v)}")
    for l in v[:3]:
        print("   ", l[:300])
print("dense uploads:", len(uploads), "| dense_resident:", [l[:120] for l in lines(r"v4_dense_resident")])
gd = checks["gpu_direct_counters"]
if gd:
    m = dict(re.findall(r"(\w+)=(\d+)", gd[-1]))
    print("gpu-direct device_decodes", m.get("device_decodes"), "failures", m.get("failures"))
print("errors/warnings:", [l[:200] for l in lines(r"(?i)error|fail(?!ures=0)|fallback(?!s=0)") if "fallbacks=0" not in l][:10])

w3 = json.load(open(W3))
for p, n in (("reason1", 32), ("code1", 128), ("chat_it1", 128)):
    got = res.get(p + "_ids")
    refs = {k: v[p][:n] for k, v in w3.items() if p in v}
    same = [k for k, v in refs.items() if v[:len(got or [])] == got]
    print(f"{p}: {len(got or [])} ids, identical to {len(same)}/{len(refs)} W3 runs (first {n})")
    if got and not same:
        ref = next(iter(refs.values()))
        first = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
        print("   first divergence at", first, got[first:first + 5], ref[first:first + 5])

# decprof per request window: mean step ms
steps = [sum(int(x) for x in re.findall(r"(?:hc|attn|moe)=(\d+)", l)) for l in lines(r"^.{0,20}decprof position=")]
print("decprof steps:", len(steps))
for p in ("reason1", "code1", "chat_it1"):
    r = res.get(p, {})
    print(p, {k: r.get(k) for k in ("completion_tokens", "tok_s_engine_style", "tok_s_between_tokens", "ttft", "wall", "finish", "secs", "rows", "usage")})

rows = list(csv.DictReader(open(os.path.join(OUT, "disk_samples.csv"))))
def at(t_ms):  # last sample at or before t
    best = rows[0]
    for r in rows:
        if int(r["unix_ms"]) <= t_ms: best = r
    return best
snaps = res["disk_snaps"]
print("snap deltas GiB:", json.dumps(res["disk_deltas_gib"], indent=0))
tot_c = tot_u = 0
for p in ("reason1", "code1", "chat_it1"):
    pass
d = res["disk_deltas_gib"]
dec = [v for k, v in d.items() if k.startswith(("ready->", "after_reason1->", "after_code1->"))]
c = sum(v["C"] for v in dec); u = sum(v["U"] for v in dec)
print(f"prompt phase (ready -> after chat_it1): C: {c:.3f} GiB  U: {u:.3f} GiB  split C {100*c/(c+u):.1f}% / U {100*u/(c+u):.1f}%")
# peak 1 s rates
peak = {"C": 0, "U": 0}
for a, b in zip(rows, rows[1:]):
    dt = (int(b["unix_ms"]) - int(a["unix_ms"])) / 1000
    if dt <= 0: continue
    for k, col in (("C", "C_read_bytes"), ("U", "U_read_bytes")):
        peak[k] = max(peak[k], (int(b[col]) - int(a[col])) / dt / 2**20)
print("samples:", len(rows), "peak MiB/s:", {k: round(v) for k, v in peak.items()})
