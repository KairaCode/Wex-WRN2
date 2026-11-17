# Fetch the 32-B WNR3 header of every one of the 7,680 records of the published container by HTTP range
# requests (32 B each, ~250 KB in total) and save them to headers.bin (7680 x 32 B, index = layer*192+expert),
# so container-level figures for the scale payload (SP) are exact sums, not extrapolations.
# usage: python3 fetch_headers.py      (writes ./headers.bin; does not touch /tmp/wnr3-data)
import struct, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "https://huggingface.co/Kaira666/Kaira-WEX4-Experts/resolve/main/v41-reap/v4_flash_experts.nr3.wex"
HERE = os.path.dirname(os.path.abspath(__file__))
table = open("/tmp/wnr3-data/recs/entries.bin", "rb").read()
ENTRY = struct.Struct('<QIII6H6H3I3II')

def get(i):
    off = ENTRY.unpack_from(table, i * 72)[0]
    for a in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off+31}"})
            with urllib.request.urlopen(req, timeout=60) as r: d = r.read()
            assert len(d) == 32
            return d
        except Exception as e:
            print("retry", i, e, file=sys.stderr); time.sleep(2 ** a)
    raise SystemExit("failed %d" % i)

out = HERE + "/headers.bin"
with ThreadPoolExecutor(8) as ex: hs = list(ex.map(get, range(7680)))
open(out, "wb").write(b"".join(hs))
bad = [i for i, h in enumerate(hs) if h[:4] != b"WNR3" or struct.unpack_from('<I', h, 28)[0] != ((i // 192) << 16 | i % 192)]
print("headers", len(hs), "bad", bad)
