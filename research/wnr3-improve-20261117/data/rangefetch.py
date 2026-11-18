"""Sparse local copy of the published V4.1 WEX4 container (HF Kaira666/Kaira-WEX4-Experts), filled by HTTP
range requests. The local file has the real size; only fetched ranges hold data (the rest reads as zeros).
  python3 rangefetch.py meta                 header page, WEX4 index, saved WEX3 page, model, WEX3 entry table
  python3 rangefetch.py records L:E,L:E,...  those records (by the WEX3 entry table)
fetched.json records every range written."""
import json, os, struct, sys, urllib.request, time
URL = "https://huggingface.co/Kaira666/Kaira-WEX4-Experts/resolve/main/v41-reap/v4_flash_experts.nr3.wex"
SIZE = 123312910336
D = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(D, "v4_flash_experts.nr3.wex")
LOG = os.path.join(D, "fetched.json")
ENTRY = struct.Struct('<QIII6H6H3I3II')

def get(off, n):
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{off+n-1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            assert len(data) == n, (len(data), n)
            return data
        except Exception as e:
            print("retry", off, n, e, file=sys.stderr); time.sleep(2 ** attempt)
    raise SystemExit("fetch failed")

def fetched():
    return json.load(open(LOG)) if os.path.exists(LOG) else []

def put(off, n):
    done = fetched()
    if [off, n] in done: return
    data = get(off, n)
    with open(PATH, "r+b") as f:
        f.seek(off); f.write(data)
    done.append([off, n]); json.dump(done, open(LOG, "w"))

def read(off, n):
    with open(PATH, "rb") as f:
        f.seek(off); return f.read(n)

def main():
    if not os.path.exists(PATH):
        with open(PATH, "wb") as f: f.truncate(SIZE)
    mode = sys.argv[1]
    if mode == "meta":
        put(0, 4096)
        p = read(0, 4096)
        index_offset, index_bytes, blob_count = struct.unpack_from('<QQI', p, 32)
        put(index_offset, max(64 * blob_count, 4096) if index_offset + 4096 <= SIZE else 64 * blob_count)
        blobs = read(index_offset, 64 * blob_count)
        for i in range(blob_count):
            kind, _, off, size, _, _ = struct.unpack_from('<IIQQ32sQ', blobs, 64 * i)
            if size <= (64 << 20): put(off, size)
        # WEX3 page -> model + table
        sys.path.insert(0, "/workspace/Wex-WRN2/wex4/research/wnr4-record")
        import fetch
        f = open(PATH, "rb"); page = fetch.wex3_page(f)
        magic, version, layers, experts, aligned, table_offset, flags, _ = fetch.HDR.unpack_from(page, 0)
        _, model_bytes, model_offset, model_id, _ = fetch.WNMD.unpack_from(page, 64)
        put(model_offset, model_bytes); put(table_offset, layers * experts * ENTRY.size)
        print("layers", layers, "experts", experts, "flags", flags, "model", hex(model_id), "blobs", blob_count, "index", index_offset, index_bytes)
    else:
        sys.path.insert(0, "/workspace/Wex-WRN2/wex4/research/wnr4-record")
        import fetch
        f, page, model, table, layers, experts, _ = fetch.open_container(PATH)
        for item in sys.argv[2].split(","):
            l, e = map(int, item.split(":"))
            ent = ENTRY.unpack_from(table, (l * experts + e) * ENTRY.size)
            put(ent[0], ent[1]); print(l, e, ent[1], flush=True)
main()
