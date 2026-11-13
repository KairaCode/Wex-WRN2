# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Independent safetensors header index (headers only, no tensor data read).
import json, struct
from pathlib import Path

def index(model_dir):
    idx = {}
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        with p.open("rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(hlen))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            a, z = info["data_offsets"]
            idx[name] = (info["dtype"], tuple(info["shape"]), str(p), 8 + hlen + a, z - a)
    return idx

if __name__ == "__main__":
    import sys, re
    from collections import defaultdict
    idx = index(sys.argv[1])
    tot = defaultdict(lambda: [0, 0]); ex = defaultdict(lambda: [0, 0])
    for n, (dt, sh, _, _, nb) in idx.items():
        t = ex if ".ffn.experts." in n else tot
        t[dt][0] += nb; t[dt][1] += 1
    print("expert:", dict(ex)); print("non-expert:", dict(tot))
    for pat in sys.argv[2:]:
        for n in sorted(idx, key=lambda s: [int(t) for t in re.findall(r"\d+", s)]):
            if re.search(pat, n):
                print(n, idx[n][0], idx[n][1], idx[n][4])
