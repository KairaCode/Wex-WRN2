# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Bytes of the routed-expert tensors of a published safetensors checkpoint, read from its headers on Hugging Face.

    python -I scripts/expert_bytes.py REPO PROFILE [REVISION]

PROFILE is a naming profile of wex4/wex4_convert.py (PROFILES). The script lists the .safetensors files at
REVISION (default: the current commit, which it prints), then the header of each shard with two HTTP range requests,
and sums data_offsets over the tensors whose names match the profile's expert pattern (weights and scales of the
three matrices). It prints the commit, the number of experts and layers found and the byte total per layer count.
No tensor data is downloaded. Standard library only. scripts/make_charts.py copies the totals it prints."""
import collections
import json
import os
import struct
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wex4'))
import wex4_convert  # noqa: E402

HOST = 'https://huggingface.co'


def get(url, first=None, count=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'wex4-expert-bytes/1'})
    if first is not None:
        req.add_header('Range', f'bytes={first}-{first + count - 1}')
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if first is not None and len(data) != count:
        raise SystemExit(f'{url}: asked {count} bytes, got {len(data)}')
    return data


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit(__doc__)
    repo, profile = argv[1], argv[2]
    info = json.loads(get(f'{HOST}/api/models/{repo}' + (f'/revision/{argv[3]}' if len(argv) == 4 else '')))
    rev = info['sha']
    base = f'{HOST}/{repo}/resolve/{rev}/'
    # every top-level .safetensors file of the commit (some checkpoints ship no index)
    shards = sorted(s['rfilename'] for s in info['siblings']
                    if s['rfilename'].endswith('.safetensors') and '/' not in s['rfilename'])
    pattern = wex4_convert.expert_pattern(profile)
    total, experts, per_layer = 0, set(), collections.Counter()
    for shard in shards:
        (n,) = struct.unpack('<Q', get(base + shard, 0, 8))
        header = json.loads(get(base + shard, 8, n))
        for name, spec in header.items():
            m = pattern.match(name.encode()) if name != '__metadata__' else None
            if m:
                size = spec['data_offsets'][1] - spec['data_offsets'][0]
                total += size
                per_layer[int(m.group(1))] += size
                experts.add((int(m.group(1)), int(m.group(2))))
    print(f'{repo} @ {rev}: {len(shards)} shards, {len(experts)} experts in {len(per_layer)} layers, '
          f'{total:,} B of expert tensors')
    for layer in sorted(per_layer):
        print(f'  layer {layer}: {sum(1 for e in experts if e[0] == layer)} experts, {per_layer[layer]:,} B')


if __name__ == '__main__':
    main(sys.argv)
