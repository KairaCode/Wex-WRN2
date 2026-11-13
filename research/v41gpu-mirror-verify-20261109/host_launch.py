"""Launch the v41gpu profile the way Engine.StartColibriAsync does (src/host/Engine.cs:488-680), exactly as the
earlier verifier run did, with the NEW profile env (WNR2 on U: + C: mirror, COLI_WEX_WEIGHTS=38,62,
V4_WEX_BATCH=1). Then: W3 reason1 greedy 32 tok (raw /v1/completions, ids from the logits dump), code1 and
chat_it1 at 128 tok (raw, streamed, tok/s = completion / (t_done - t_first) like the engine's STAT), per-drive
read bytes (PhysicalDisk raw counters, disk 3 = C:, disk 5 = U:) sampled every second and at phase edges,
then a graceful stop through coli_proc.py (CTRL_BREAK, like Engine.Stop).
Differences from the host, on purpose: port 18766 (not 8766) and V4_LOGITS_DUMP (diagnostic)."""
import ctypes, json, os, subprocess, sys, time, urllib.request
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = sys.argv[1]           # logits dump outside OneDrive
KAIRA = r"<KAIRA>"
PORT = 18766
VOCAB = 129280
cfg = json.load(open(os.path.join(KAIRA, r"app\kaira.json"), encoding="utf-8"))
prof = [m for m in cfg["models"] if m["id"] == "v41gpu"][0]
c = prof["colibri"]
python, runtime, model = c["python"], c["runtime_dir"], c["model_dir"]
coli = os.path.join(runtime, "coli")
helper = os.path.join(KAIRA, r"app\tools\coli_proc.py")
prompts = {p["id"]: p for p in map(json.loads, open(os.path.join(KAIRA, r"lab\v41-gpu-20261107\ab\prompts.jsonl"), encoding="utf-8"))}
assert c["env"]["COLI_WEX_PATH"].endswith("v4_flash_experts.nr2.wex") and c["env"]["COLI_WEX_WEIGHTS"] == "38,62"

# FreePhysicalGiB (GlobalMemoryStatusEx ullAvailPhys) + EngineMemoryBudget.RamGB(ram_gib, free)
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint), ("dwMemoryLoad", ctypes.c_uint)] + [(n, ctypes.c_ulonglong) for n in
                ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
def avail_gib():
    ms = MS(); ms.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ap / 1073741824.0
free = avail_gib()
headroom = min(max(free * 0.08, 2), 6)
safe = int(max(0, (free - headroom) * (1073741824.0 / 1e9)))
ram = min(c["ram_gib"], safe) if c["ram_gib"] > 0 else safe
ctx = min(max(c["ctx"], 512), 1048576)  # ModelContexts.Launchable("v41", ctx)

env = dict(os.environ)
# ScrubSecretEnv
for k in list(env):
    n = k.upper()
    if n in ("TAVILY_API_KEY", "BRAVE_API_KEY") or n.endswith(("_API_KEY", "_APIKEY", "_SECRET")) or \
       (n.endswith("_TOKEN") and not n.startswith(("KAIRA_", "COLI_", "DSV4_", "V4_"))):
        env[k] = ""
for k in ("TAVILY_API_KEY", "BRAVE_API_KEY", "SEARXNG_API_KEY"):
    env[k] = ""
# ApplyProfileEnv: V4.1 on deepseek_v4.exe gets every profile variable (COLI_WEX_EXTRA_MIRRORS/WEIGHTS included)
for k, v in c["env"].items():
    env[k] = v
env["COLI_FAMILY_ENGINE"] = "deepseek_v4"
# no kaira-shards.json in the model dir -> COLI_MODEL_DIRS stays the profile's
assert not os.path.exists(os.path.join(model, "kaira-shards.json"))
# ApplyMtpFlags(mtp, v41OnV4=true) + V4_SKIP_COLD
for f in ("MTP", "V4_MTP", "V4_DRAFT", "Q38_MTP", "Q36_MTP", "Q36_MTP_TIER", "MIMO_MTP"):
    env[f] = "1" if cfg.get("mtp") else "0"
env["V4_MTP"] = "0"
env["V4_SKIP_COLD"] = "0"
# RouteHub.ApplyEngineEnv with the Mente hub listening (mind.enabled, KAIRA running)
m = cfg["mind"]
if m.get("enabled"):
    env["V4_ROUTE_UDP"] = f"127.0.0.1:{m['udp_port']}"
    env["V4_ROUTE_TEXT"] = "1" if m.get("text") else "0"
    env["V4_ROUTE_PREFILL"] = "1" if m.get("prefill") else "0"
    env["V4_ROUTE_RESID_EVERY"] = str(min(max(m.get("resid_every", 0), 0), 255))
# ApplyV4Reserve(pendingGemma = 0: subagents disabled)
env["DSV4_CUDA_VRAM_RESERVE_MB"] = str(max(int(env.get("DSV4_CUDA_VRAM_RESERVE_MB", "4096")), 4096))
# mirror_dir null: the host drops the shard mirror vars; COLI_WEX_EXTRA_MIRRORS/COLI_WEX_WEIGHTS pass through
env.pop("COLI_MODEL_MIRROR", None); env.pop("COLI_DISK_WEIGHTS", None)
front = [d for d in c["path_prepend"] if d.strip() and os.path.isdir(d)]
env["PATH"] = ";".join(front) + (";" + env["PATH"] if env.get("PATH") else "")
# diagnostic only
dump = os.path.join(SCRATCH, "mirror-verify.logits")
if os.path.exists(dump): os.remove(dump)
env["V4_LOGITS_DUMP"] = dump

serve = [python, "-u", coli, "serve", "--model", model, "--ram", str(ram), "--ctx", str(ctx), "--kv-slots", "1",
         "--policy", "quality", "--host", "127.0.0.1", "--port", str(PORT), "--model-id", "kaira"] + list(c.get("extra_args") or [])
outlog, errlog = os.path.join(OUT, "colibri.stdout.log"), os.path.join(OUT, "colibri.stderr.log")
print(f"free {free:.2f} GiB -> ram {ram} GB (automatic), ctx {ctx}")
print("cmd:", " ".join(serve))
print("env (profile + host):", {k: env[k] for k in sorted(set(c["env"]) | {"MTP", "V4_MTP", "V4_DRAFT", "V4_SKIP_COLD",
      "V4_ROUTE_UDP", "V4_ROUTE_TEXT", "V4_ROUTE_PREFILL", "V4_ROUTE_RESID_EVERY", "DSV4_CUDA_VRAM_RESERVE_MB"}) if k in env})
for k in ("COLI_DSV4_DLL", "COLI_WEX_CUDA_LIB", "COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS", "RAM_GB", "CTX", "SERVE", "SNAP"):
    print(f"  inherited {k}={env.get(k)!r}")
json.dump({"cmd": serve, "env": {k: env[k] for k in env if k.upper() not in ("PATH",)}, "path_front": front,
           "free_gib": free, "ram": ram}, open(os.path.join(OUT, "launch.json"), "w"), indent=1)

# PhysicalDisk raw cumulative read bytes: 1 s sampler for the whole run + snapshots at phase edges
PS_SNAP = ("Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -Filter \"Name='3 C:' OR Name='5 U:'\" | "
           "ForEach-Object { $_.Name + '=' + $_.DiskReadBytesPersec }")
def disk_snap():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_SNAP], capture_output=True, text=True, timeout=60)
    d = dict(l.strip().split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    return {"t": time.time(), "C": int(d["3 C:"]), "U": int(d["5 U:"])}
sampler = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(OUT, "disk_sampler.ps1"),
                            os.path.join(OUT, "disk_samples.csv")])
snaps = {"before_launch": disk_snap()}

t0 = time.time()
r = subprocess.run([python, "-I", helper, "start", outlog, errlog, runtime] + serve, env=env, cwd=runtime,
                   capture_output=True, text=True, timeout=30)
print("coli_proc start rc", r.returncode, r.stdout.strip(), r.stderr.strip())
pid = int(r.stdout.strip().splitlines()[-1])

def get(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")

def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post_stream(path, body, timeout=900):
    """SSE: returns text, usage, t_first (first non-empty text), t_done (last event), wall."""
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time(); t_first = t_last = None; text = []; usage = None; finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            ev = json.loads(data)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    now = time.time()
                    if t_first is None: t_first = now
                    t_last = now; text.append(ch["text"])
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    t_done = time.time()
    return {"text": "".join(text), "usage": usage, "finish": finish, "ttft": t_first - t, "t_first": t_first,
            "t_last": t_last, "t_done": t_done, "wall": t_done - t}

def rows():
    return os.path.getsize(dump) // (4 * VOCAB) if os.path.exists(dump) else 0

def raw_prompt(user):
    return "<｜begin▁of▁sentence｜><｜User｜>" + user + "<｜Assistant｜></think>"

res = {"pid": pid, "ram": ram, "free_gib": round(free, 2)}
try:
    ready = None
    while time.time() - t0 < 900:
        try:
            st, body = get("/health")
            if st == 200:
                ready = time.time() - t0; res["health"] = body[:400]; break
        except Exception:
            pass
        r2 = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        if str(pid) not in r2.stdout:
            raise SystemExit("gateway exited before /health")
        time.sleep(2)
    res["ready_s"] = ready
    print("health ready after", ready, res.get("health"))
    if ready is None:
        raise SystemExit("no /health in 900 s")
    snaps["ready"] = disk_snap()
    res["avail_gib_at_ready"] = round(avail_gib(), 2)

    r0 = rows(); t = time.time()
    a = post("/v1/completions", {"model": "kaira", "prompt": raw_prompt(prompts["reason1"]["user"]), "max_tokens": 32,
                                 "temperature": 0, "top_p": 1})
    r1 = rows()
    res["reason1"] = {"secs": round(time.time() - t, 2), "usage": a.get("usage"), "text": a["choices"][0]["text"],
                      "finish": a["choices"][0].get("finish_reason"), "rows": [r0, r1]}
    print("reason1", json.dumps(res["reason1"], ensure_ascii=False))
    snaps["after_reason1"] = disk_snap()
    L = np.memmap(dump, dtype=np.float32, mode="r").reshape(-1, VOCAB)
    res["reason1_ids"] = [int(x) for x in L[r0:r1].argmax(1)]
    del L

    for pid_ in ("code1", "chat_it1"):
        ra = rows()
        s = post_stream("/v1/completions", {"model": "kaira", "prompt": raw_prompt(prompts[pid_]["user"]), "max_tokens": 128,
                                            "temperature": 0, "top_p": 1, "stream": True,
                                            "stream_options": {"include_usage": True}})
        rb = rows()
        n = (s["usage"] or {}).get("completion_tokens") or (rb - ra)
        s["rows"] = [ra, rb]
        s["completion_tokens"] = n
        s["tok_s_engine_style"] = round(n / (s["t_done"] - s["t_first"]), 3)        # STAT: completion / decode_sec from first token
        s["tok_s_between_tokens"] = round((n - 1) / (s["t_last"] - s["t_first"]), 3) if n > 1 else None
        res[pid_] = s
        print(pid_, json.dumps({k: v for k, v in s.items() if k != "text"}), repr(s["text"][:120]))
        snaps["after_" + pid_] = disk_snap()
        L = np.memmap(dump, dtype=np.float32, mode="r").reshape(-1, VOCAB)
        res[pid_ + "_ids"] = [int(x) for x in L[ra:rb].argmax(1)]
        del L
    res["avail_gib_after_prompts"] = round(avail_gib(), 2)
    try:
        res["tiers_health"] = get("/health")[1][:800]
    except Exception as e:
        res["tiers_health"] = repr(e)
finally:
    t = time.time()
    s = subprocess.run([python, "-I", helper, "stop", str(pid), "60"], capture_output=True, text=True)
    res["stop"] = {"rc": s.returncode, "secs": round(time.time() - t, 1), "err": s.stderr.strip()}
    print("coli_proc stop", res["stop"])
    time.sleep(2)
    snaps["after_stop"] = disk_snap()
    sampler.terminate()
    res["disk_snaps"] = snaps
    keys = list(snaps)
    res["disk_deltas_gib"] = {f"{keys[i - 1]}->{keys[i]}": {
        "C": round((snaps[keys[i]]["C"] - snaps[keys[i - 1]]["C"]) / 2**30, 3),
        "U": round((snaps[keys[i]]["U"] - snaps[keys[i - 1]]["U"]) / 2**30, 3),
        "secs": round(snaps[keys[i]]["t"] - snaps[keys[i - 1]]["t"], 1)} for i in range(1, len(keys))}
    print("disk deltas GiB:", json.dumps(res["disk_deltas_gib"]))
    json.dump(res, open(os.path.join(OUT, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
