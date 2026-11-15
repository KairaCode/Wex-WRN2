# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Fetch a few routed experts of public MXFP4 checkpoints on Hugging Face, with HTTP range requests only.

  python -I fetch_experts.py [OUT_DIR] [--only NAME[,NAME]] [--budget-mb 1500] [--plan]

OUT_DIR defaults to data/ next to this script (git-ignored). For each checkpoint of PLAN, at its pinned
revision: model.safetensors.index.json (to find the shard of each tensor), then for each shard that holds a
chosen expert two range requests (the 8-byte header length, then the JSON header), then one range request
per tensor: all six tensors (scale and weight of the three matrices) of every chosen expert. The tensor
names are those of the converter's naming profile (wex4/wex4_convert.py, PROFILES). Every range response must
be 206 with the Content-Range that was asked for and exactly the asked number of bytes; every tensor must
have the size its dtype and shape give.

Output: OUT_DIR/NAME/<shard>.experts.safetensors for each source shard, holding exactly the fetched tensors
(name, dtype, shape and bytes unchanged, data in the order of the source shard, the source's __metadata__),
OUT_DIR/NAME/config.json, and OUT_DIR/fetch-log.json: every request with its repository, revision, file,
byte range, byte count and the SHA-256 of the bytes received, and every file written with its SHA-256.
--plan prints the requests and the planned byte count after reading the indexes and headers, and fetches
no tensor. The script refuses to fetch tensors when the plan exceeds --budget-mb (MB = 10^6 bytes, index
and headers included) and stops when the bytes received would exceed it.

The weights stay under the licence of their checkpoint (research use here). Nothing of OUT_DIR is
redistributed or committed. Standard library only.
"""
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'wex4'))
import wex4_convert  # noqa: E402

HOST = 'https://huggingface.co'
AGENT = 'wex4-research-fetch/1'
# name -> (repository, full commit, naming profile, layers, experts). Two layers x four experts each;
# layers N//4 and 3N//4 of the N layers in config.json, experts 0 to 3 (not a random pick).
PLAN = {
    'deepseek-v41': ('deepseek-ai/DeepSeek-V4.1-Flash', '314113f8a7dadb388215396ba5fbee2c610a04d6', 'deepseek',
                     (10, 30), (0, 1, 2, 3)),
    'qwen3-235b': ('amd/Qwen3-235B-A22B-Instruct-2507-MXFP4', '381342aab13be9f91265d0f6b1760470bfd2704d', 'qwen3-moe',
                   (23, 70), (0, 1, 2, 3)),
    'minimax-m21': ('amd/MiniMax-M2.1-MXFP4', 'bbf15c45e91cfa2b4970f42619750415a214e24b', 'mixtral',
                    (15, 46), (0, 1, 2, 3)),
}
DTYPE_BYTES = {'U8': 1, 'I8': 1, 'F8_E8M0': 1, 'F8_E4M3': 1, 'BF16': 2, 'F16': 2, 'F32': 4}
INDEX_CAP = 64 << 20   # an index larger than this is refused, so a wrong file cannot eat the budget


class Budget:
    """Bytes counted against the limit: every range asked for (again on a retry) and every whole file read."""

    def __init__(self, limit):
        self.limit, self.used, self.retries, self.log = limit, 0, 0, []

    def take(self, n):
        if self.used + n > self.limit:
            raise SystemExit(f'budget: {self.used} + {n} bytes would exceed {self.limit}')
        self.used += n


def url_of(repo, revision, name):
    return f'{HOST}/{repo}/resolve/{revision}/{name}'


def get(budget, repo, revision, name, what, first=None, last=None):
    """One request: the whole file (first None) or bytes first..last inclusive. Returns the bytes; logs it."""
    headers = {'User-Agent': AGENT}
    want = None
    if first is not None:
        want = last - first + 1
        headers['Range'] = f'bytes={first}-{last}'
    for attempt in range(4):
        if want is not None:
            budget.take(want)   # before every attempt: a retry counts its bytes again
        try:
            with urllib.request.urlopen(urllib.request.Request(url_of(repo, revision, name), headers=headers),
                                        timeout=120) as response:
                if want is None:
                    if response.status != 200:
                        raise SystemExit(f'{repo} {name}: HTTP {response.status}')
                    data = response.read(INDEX_CAP + 1)
                    if len(data) > INDEX_CAP:
                        raise SystemExit(f'{repo} {name}: larger than {INDEX_CAP} bytes')
                    budget.take(len(data))
                else:
                    got = response.headers.get('Content-Range', '')
                    if response.status != 206 or not got.startswith(f'bytes {first}-{last}/'):
                        raise SystemExit(f'{repo} {name}: HTTP {response.status}, Content-Range {got!r}, '
                                         f'asked bytes {first}-{last}')
                    chunks, have = [], 0
                    while have < want:
                        chunk = response.read(min(want - have, 1 << 20))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        have += len(chunk)
                    data = b''.join(chunks)
                    if len(data) != want:
                        raise OSError(f'short read: {len(data)} of {want} bytes')
            break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            if attempt == 3:
                raise SystemExit(f'{repo} {name}: {error}')
            budget.retries += 1
            time.sleep(2 + 4 * attempt)
    row = {'repo': repo, 'revision': revision, 'file': name, 'what': what,
           'range': None if first is None else [first, last], 'bytes': len(data),
           'sha256': hashlib.sha256(data).hexdigest()}
    budget.log.append(row)
    print(f"{repo}@{revision[:8]} {name} {what} "
          f"{'whole file' if first is None else f'bytes {first}-{last}'} {len(data)} B sha256 {row['sha256']}",
          flush=True)
    return data


def expert_tensors(profile, layers, experts):
    return [wex4_convert.expert_name(profile, layer, expert, m, weight).encode()
            for layer in layers for expert in experts for m in range(3) for weight in (False, True)]


def checkpoint(budget, name, out_dir, plan_only):
    repo, revision, profile, layers, experts = PLAN[name]
    config = get(budget, repo, revision, 'config.json', 'config')
    weight_map = json.loads(get(budget, repo, revision, 'model.safetensors.index.json', 'index'))['weight_map']
    wanted = expert_tensors(profile, layers, experts)
    shards = {}
    for tensor in wanted:
        shard = weight_map.get(tensor.decode())
        if shard is None:
            raise SystemExit(f'{repo}: {tensor.decode()} is not in the index')
        shards.setdefault(shard, []).append(tensor)
    plan = []
    for shard, tensors in sorted(shards.items()):
        length = struct.unpack('<Q', get(budget, repo, revision, shard, 'header length', 0, 7))[0]
        if not 2 <= length <= 100_000_000:
            raise SystemExit(f'{repo} {shard}: header length {length}')
        header = json.loads(get(budget, repo, revision, shard, 'header', 8, 8 + length - 1))
        rows = []
        for tensor in tensors:
            info = header[tensor.decode()]
            begin, end = info['data_offsets']
            count = 1
            for d in info['shape']:
                count *= d
            if info['dtype'] not in DTYPE_BYTES or count * DTYPE_BYTES[info['dtype']] != end - begin:
                raise SystemExit(f'{repo} {shard}: {tensor.decode()} size does not match dtype and shape')
            rows.append((begin, end, tensor, info))
        rows.sort()
        plan.append((shard, 8 + length, header.get('__metadata__'), rows))
    planned = sum(end - begin for _, _, _, rows in plan for begin, end, _, _ in rows)
    print(f'{name}: {len(wanted)} tensors in {len(plan)} shards, {planned} B of tensor data planned', flush=True)
    if plan_only:
        return {'name': name, 'planned_tensor_bytes': planned}
    if budget.used + planned > budget.limit:
        raise SystemExit(f'budget: {budget.used} B used + {planned} B planned exceeds {budget.limit} B')
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'config.json'), 'wb') as f:
        f.write(config)
    outputs = []
    for shard, data_start, meta, rows in plan:
        doc, payload, sums = {}, [], {}
        if meta is not None:
            doc['__metadata__'] = meta
        at = 0
        for begin, end, tensor, info in rows:
            data = get(budget, repo, revision, shard, tensor.decode(), data_start + begin, data_start + end - 1)
            sums[tensor] = hashlib.sha256(data).hexdigest()
            doc[tensor.decode()] = {'dtype': info['dtype'], 'shape': info['shape'], 'data_offsets': [at, at + len(data)]}
            payload.append(data)
            at += len(data)
        raw = json.dumps(doc, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        path = os.path.join(folder, shard.rsplit('.', 1)[0] + '.experts.safetensors')
        with open(path + '.partial', 'wb') as f:
            f.write(struct.pack('<Q', len(raw)) + raw)
            for data in payload:
                f.write(data)
        os.replace(path + '.partial', path)
        _, start, tensors = wex4_convert.st_index(path)   # the file reads back, and each tensor has its bytes
        with open(path, 'rb') as f:
            for tensor, _, _, begin, end in tensors:
                f.seek(start + begin)
                if hashlib.sha256(f.read(end - begin)).hexdigest() != sums[tensor]:
                    raise SystemExit(f'{path}: {tensor.decode()} reads back with another SHA-256')
        outputs.append({'file': f'{name}/{os.path.basename(path)}', 'bytes': os.path.getsize(path),
                        'sha256': wex4_convert.file_sha256(path), 'tensors': len(rows), 'source': shard})
        print(f"wrote {outputs[-1]['file']} {outputs[-1]['bytes']} B sha256 {outputs[-1]['sha256']}", flush=True)
    return {'name': name, 'repo': repo, 'revision': revision, 'profile': profile, 'layers': list(layers),
            'experts': list(experts), 'tensor_bytes': planned, 'outputs': outputs}


def main(argv):
    def option(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default
    values = {argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ('--only', '--budget-mb')}
    positional = [a for a in argv[1:] if not a.startswith('--') and a not in values]
    out_dir = positional[0] if positional else os.path.join(HERE, 'data')
    names = option('--only', ','.join(PLAN)).split(',')
    if any(n not in PLAN for n in names):
        raise SystemExit(f'--only takes names of PLAN: {", ".join(PLAN)}')
    budget = Budget(int(float(option('--budget-mb', '1500')) * 1_000_000))
    plan_only = '--plan' in argv
    os.makedirs(out_dir, exist_ok=True)
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    done = [checkpoint(budget, name, out_dir, plan_only) for name in names]
    received = sum(row['bytes'] for row in budget.log)
    print(f'received {received} B in {len(budget.log)} requests ({budget.retries} retries); counted against the '
          f'budget {budget.used} B of {budget.limit} B', flush=True)
    if plan_only:
        return 0
    log = {'fetch_experts': 1, 'started_utc': started, 'budget_bytes': budget.limit, 'counted_bytes': budget.used,
           'received_bytes': received, 'retries': budget.retries, 'checkpoints': done, 'requests': budget.log}
    with open(os.path.join(out_dir, 'fetch-log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=1)
        f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
