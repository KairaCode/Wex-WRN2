# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Byte check of a WEX1/WEX2/WEX3 container against the safetensors checkpoint it was built from.

usage: python -I verify_wex1.py --model-dir DIR --wex FILE [--stride N]

Metadata first (repack_v4_wex.validate_container: header, table, layout). Then every Nth
present expert (default: all) is read from the container and its decoded scales
(w1|w2|w3) and weights (w1|w2|w3) are compared byte for byte with the source tensors.
Raw and WEX2 scale-tile records are checked; palette3 and neural-rANS (WNR1) records
are counted as skipped. A WNR2 container is rejected by validate_container: check it
with archive/nr2/check_container.exe or archive/nr2/check_all.py against its WEX1 source.
Exit code 0 = every checked record identical, 1 = mismatch or invalid container.
The engine itself checks only shapes and offsets (v4_wex_matches_record), not bytes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / 'c' / 'tools')]   # repack_v4_wex.py: beside this file (colibri c/tools) or in this repo
import repack_v4_wex as W   # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--wex", required=True)
    p.add_argument("--stride", type=int, default=1, help="check every Nth present expert (1 = all)")
    a = p.parse_args(argv)
    if a.stride < 1:
        p.error("--stride must be >= 1")
    began = time.time()
    model = Path(a.model_dir)
    try:
        info = W.validate_container(a.wex)
    except ValueError as error:
        print(f"FAIL invalid container: {error}")
        return 1
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    layers, experts, dims = W._model_dims(model, config)
    if (layers, experts) != (info["layers"], info["experts"]):
        print(f"FAIL geometry: checkpoint {layers}x{experts}, container {info['layers']}x{info['experts']}")
        return 1
    plan = W._expert_plan(W.index_safetensors(model), layers, experts, dims)
    checked = skipped = present = 0
    handles = {}
    try:
        with open(a.wex, "rb") as wex:
            for index, (entry, matrices) in enumerate(zip(info["entries"], plan)):
                offset, total, scale_bytes, weight_bytes = entry[:4]
                codec, pages = entry[22] & 255, entry[22] >> 8
                if matrices is None or not total:
                    if bool(matrices) != bool(total):
                        print(f"FAIL presence: layer {index // experts} expert {index % experts}")
                        return 1
                    continue
                present += 1
                if (present - 1) % a.stride:
                    continue
                if codec not in (0, W.WEX_SCALE_TILES):
                    skipped += 1
                    continue
                for scale, weight in matrices:
                    for ref in (scale, weight):
                        if ref.path not in handles:
                            handles[ref.path] = ref.path.open("rb")
                scales = b"".join(W._read_tensor(s, handles) for s, _ in matrices)
                weights = b"".join(W._read_tensor(w, handles) for _, w in matrices)
                wex.seek(offset)
                record = wex.read(total)
                if codec:
                    magic, count, size = W.SCALE_PREFIX.unpack_from(record)
                    got_scales = W.unpack_scale_tiles(record[W.SCALE_PREFIX.size:W.SCALE_PREFIX.size + size], count)
                    start = pages * W.WEX_ALIGN
                else:
                    got_scales = record[:scale_bytes]
                    start = scale_bytes
                if got_scales != scales or record[start:start + weight_bytes] != weights:
                    print(f"FAIL bytes differ: layer {index // experts} expert {index % experts}")
                    return 1
                checked += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps({"ok": True, "version": info["version"], "layers": layers, "experts": experts,
                      "present": present, "checked": checked, "skipped_codec": skipped,
                      "stride": a.stride, "seconds": round(time.time() - began, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
