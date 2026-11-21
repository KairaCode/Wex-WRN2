"""Fetch the 32-byte WNR3 header of all 7,680 records of the published container by HTTP range requests
(read only; nothing is written to <TMP>/wnr3-data). Output: headers.bin (7680 x 32 B, index = layer*192+expert)."""
import struct, urllib.request, time, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
URL = "https://huggingface.co/Kaira666/Kaira-WEX4-Experts/resolve/main/v41-reap/v4_flash_experts.nr3.wex"
t = open("<TMP>/wnr3-data/recs/entries.bin", "rb").read()
offs = [struct.unpack_from("<QI", t, i * 72) for i in range(7680)]
def get(i):
    off = offs[i][0]
    for a in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off+31}"})
            with urllib.request.urlopen(req, timeout=60) as r: d = r.read()
            assert len(d) == 32; return d
        except Exception as e:
            time.sleep(2 ** a)
    raise SystemExit("fail %d" % i)
with ThreadPoolExecutor(8) as ex: hs = list(ex.map(get, range(7680)))
open("headers.bin", "wb").write(b"".join(hs))
print("ok", sum(h[:4] == b"WNR3" for h in hs))
