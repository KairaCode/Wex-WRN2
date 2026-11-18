"""Fetch (HTTP range, read-only) the records of the published container whose rule-g anchors are not
(120,120,120) (38 of 7,680, from headers.bin), except those already in /tmp/wnr3-data/recs.
Output: extra/rec_LL_EEE.bin (stored record bytes)."""
import struct, os, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
URL = "https://huggingface.co/Kaira666/Kaira-WEX4-Experts/resolve/main/v41-reap/v4_flash_experts.nr3.wex"
t = open("/tmp/wnr3-data/recs/entries.bin", "rb").read(); h = open("headers.bin", "rb").read()
os.makedirs("extra", exist_ok=True)
todo = []
for i in range(7680):
    if h[i * 32 + 24:i * 32 + 27] != bytes([120, 120, 120]):
        l, e = divmod(i, 192); p = "/tmp/wnr3-data/recs/rec_%02d_%03d.bin" % (l, e)
        if not os.path.exists(p): todo.append(i)
def get(i):
    off, n = struct.unpack_from("<QI", t, i * 72)
    for a in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off+n-1}"})
            with urllib.request.urlopen(req, timeout=300) as r: d = r.read()
            assert len(d) == n; break
        except Exception:
            time.sleep(2 ** a)
    else: raise SystemExit("fail %d" % i)
    open("extra/rec_%02d_%03d.bin" % divmod(i, 192), "wb").write(d); return i
with ThreadPoolExecutor(4) as ex: done = list(ex.map(get, todo))
print("fetched", len(done))
