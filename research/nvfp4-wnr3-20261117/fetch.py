# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Fetch the same routed experts of NVFP4 and MXFP4 exports of one base model, with HTTP range requests only.

  python -I fetch.py OUT_DIR [--only NAME[,NAME]] [--budget-mb 2500] [--plan]

For each pair of PLAN: config.json and model.safetensors.index.json of every checkpoint of the pair at its
pinned commit; the MoE layers are those whose expert 0 is in every index of the pair. Selection, with
random.Random(SEED) per pair: the MoE layers are cut into 8 runs of consecutive layers of (nearly) equal
length and one layer is drawn from each run (layers at or past num_hidden_layers, the MTP layers, are left out); two experts are drawn from range(n_routed_experts) per chosen
layer. Every checkpoint of a pair gets the same (layer, expert) list. Then for each shard that holds a chosen
tensor: the 8-byte header length and the JSON header (two range requests), and one range request per run of
adjacent tensors (never more than 64 MiB). Every response must be 206 with the asked Content-Range and
exactly the asked byte count; every tensor must have the size its dtype and shape give.

Output: OUT_DIR/<checkpoint>/experts.safetensors (the fetched tensors, name, dtype, shape and bytes unchanged),
OUT_DIR/<checkpoint>/config.json, OUT_DIR/fetch-log.json (every request with its byte range and SHA-256) and
OUT_DIR/samples.tsv (one row per tensor: checkpoint, repository, commit, shard, tensor, layer, expert,
absolute first and last byte in the shard, byte count, SHA-256). --plan stops before any tensor is fetched.
The weights stay under the licence of their checkpoint; nothing of OUT_DIR is committed. Standard library only.
"""
import hashlib
import json
import os
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.request

HOST = 'https://huggingface.co'
AGENT = 'wex4-research-fetch/1'
SEED = 20261117
NV = ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')
DS_W, PROJ = ('w1', 'w2', 'w3'), ('gate_proj', 'down_proj', 'up_proj')   # WEX order: gate, down, up
QWEN_PREFIX = 'model.language_model.layers.{L}.mlp.experts.{e}.'
# checkpoint -> (repository, full commit, format, tensor prefix, matrices, suffixes); commits from the HF API
CHECKPOINTS = {
    'ds-nvfp4': ('nvidia/DeepSeek-V4.1-Flash-1017-NVFP4', '3ae9c0d3fbbea5370bd06e679d22044db15883f1', 'nvfp4',
                 'layers.{L}.ffn.experts.{e}.', DS_W, NV),
    'ds-mxfp4': ('deepseek-ai/DeepSeek-V4.1-Flash', '314113f8a7dadb388215396ba5fbee2c610a04d6', 'mxfp4',
                 'layers.{L}.ffn.experts.{e}.', DS_W, ('weight', 'scale')),
    'ds-quark': ('amd/DeepSeek-V4.1-Flash-Quark-MXFP4', '8eea1b23c275a839a1cf3a4cfffcbda6ad49ef14', 'mxfp4',
                 'layers.{L}.ffn.experts.{e}.', DS_W, ('weight', 'weight_scale')),
    'qwen-nvfp4': ('nvidia/Qwen3.8-Flash-Next-NVFP4', 'fc694b54fb0174e0913e6adf86691ef85a4ead47', 'nvfp4',
                   QWEN_PREFIX, PROJ, NV),
    'qwen-mxfp4': ('amd/Qwen3.8-Flash-Next-Quark-MXFP4', 'b71e28afcd4b43add751159862235035613a58ab', 'mxfp4',
                   QWEN_PREFIX, PROJ, ('weight', 'weight_scale')),
    'glm-nvfp4': ('nvidia/GLM-5.3-Flash-NVFP4', '1d00738646cbb0c6d0a9eb403dac5f366f1473a7', 'nvfp4',
                  QWEN_PREFIX, PROJ, NV),
    'glm-mxfp4': ('amd/GLM-5.3-Flash-Quark-MXFP4', 'b5688f25491202978c19c4d036eef579f61bbe07', 'mxfp4',
                  QWEN_PREFIX, PROJ, ('weight', 'weight_scale')),
    # NVFP4_NZ5 (TensorFold's code map: E2M1 code 0b1000 read as +5), csf0_v1 scales (a base per row, a code per 32)
    'glm-nz5': ('Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4-NZ5', '451ca95509e14f7d1a23d81bbe4591cba5a1ade7', 'nz5',
                QWEN_PREFIX, PROJ, ('weight', 'weight_scale.csf0_codes', 'weight_scale.csf0_bases', 'weight_scale_2',
                                    'input_scale')),
}
PAIRS = {'deepseek': ('ds-nvfp4', 'ds-mxfp4', 'ds-quark'), 'qwen': ('qwen-nvfp4', 'qwen-mxfp4'),
         'glm': ('glm-nvfp4', 'glm-mxfp4'), 'glm-nz5': ('glm-nz5',)}
DTYPE_BYTES = {'U8': 1, 'I8': 1, 'F8_E8M0': 1, 'F8_E4M3': 1, 'BF16': 2, 'F16': 2, 'F32': 4}
MERGE_CAP = 64 << 20


class Budget:
    def __init__(self, limit):
        self.limit, self.used, self.retries, self.log = limit, 0, 0, []

    def take(self, n):
        if self.used + n > self.limit:
            raise SystemExit(f'budget: {self.used} + {n} bytes would exceed {self.limit}')
        self.used += n


def get(budget, repo, rev, name, what, first=None, last=None):
    headers, want = {'User-Agent': AGENT}, None
    if first is not None:
        want = last - first + 1
        headers['Range'] = f'bytes={first}-{last}'
    for attempt in range(5):
        if want is not None:
            budget.take(want)
        try:
            with urllib.request.urlopen(urllib.request.Request(f'{HOST}/{repo}/resolve/{rev}/{name}', headers=headers),
                                        timeout=180) as r:
                if want is None:
                    data = r.read(64 << 20)
                    budget.take(len(data))
                else:
                    got = r.headers.get('Content-Range', '')
                    if r.status != 206 or not got.startswith(f'bytes {first}-{last}/'):
                        raise SystemExit(f'{repo} {name}: HTTP {r.status}, Content-Range {got!r}')
                    parts, have = [], 0
                    while have < want:
                        chunk = r.read(min(want - have, 1 << 20))
                        if not chunk:
                            break
                        parts.append(chunk)
                        have += len(chunk)
                    data = b''.join(parts)
                    if len(data) != want:
                        raise OSError(f'short read {len(data)} of {want}')
            break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            if attempt == 4:
                raise SystemExit(f'{repo} {name}: {error}')
            budget.retries += 1
            time.sleep(3 + 5 * attempt)
    budget.log.append({'repo': repo, 'revision': rev, 'file': name, 'what': what,
                       'range': None if first is None else [first, last], 'bytes': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()})
    return data


def moe_layers(weight_map, prefix, matrix):
    found = set()
    pattern = re.compile(re.escape(prefix.split('{L}')[0]) + r'(\d+)\.')
    for name in weight_map:
        if name.endswith(f'experts.0.{matrix}.weight'):
            m = pattern.match(name)
            if m and name == prefix.format(L=m[1], e=0) + f'{matrix}.weight':
                found.add(int(m[1]))
    return found


def select(layers, n_experts):
    rng = random.Random(SEED)
    layers = sorted(layers)
    cuts = [round(i * len(layers) / 8) for i in range(9)]
    chosen = []
    for i in range(8):
        layer = rng.choice(layers[cuts[i]:cuts[i + 1]])
        chosen += [(layer, e) for e in sorted(rng.sample(range(n_experts), 2))]
    return chosen


def main(argv):
    out = argv[1]
    names = argv[argv.index('--only') + 1].split(',') if '--only' in argv else list(PAIRS)
    budget = Budget(int(float(argv[argv.index('--budget-mb') + 1] if '--budget-mb' in argv else 2500) * 1e6))
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, 'fetch-log.json')
    old = json.load(open(log_path)) if os.path.exists(log_path) else {'pairs': {}, 'requests': [], 'samples': []}
    rows = [r for r in old['samples'] if r[0] not in {c for p in names for c in PAIRS[p]}]
    for pair in names:
        ckpts = PAIRS[pair]
        meta = {}
        for c in ckpts:
            repo, rev, fmt, prefix, mats, suff = CHECKPOINTS[c]
            config = get(budget, repo, rev, 'config.json', 'config')
            wm = json.loads(get(budget, repo, rev, 'model.safetensors.index.json', 'index'))['weight_map']
            cfg = json.loads(config)
            cfg = cfg.get('text_config', cfg)
            meta[c] = (config, wm, cfg)
        n_exp = {meta[c][2].get('n_routed_experts', meta[c][2].get('num_experts')) for c in ckpts}
        assert len(n_exp) == 1, n_exp
        layers = set.intersection(*(moe_layers(meta[c][1], CHECKPOINTS[c][3], CHECKPOINTS[c][4][0]) for c in ckpts))
        depth = min(meta[c][2]['num_hidden_layers'] for c in ckpts)
        layers = {L for L in layers if L < depth}   # a layer past num_hidden_layers is an MTP layer
        chosen = select(layers, n_exp.pop())
        print(f'{pair}: {len(layers)} MoE layers {min(layers)}..{max(layers)}; chosen {chosen}', flush=True)
        old['pairs'][pair] = {'seed': SEED, 'moe_layers': sorted(layers), 'chosen': chosen, 'checkpoints': list(ckpts)}
        for c in ckpts:
            repo, rev, fmt, prefix, mats, suff = CHECKPOINTS[c]
            config, wm, _ = meta[c]
            wanted = [prefix.format(L=L, e=e) + f'{m}.{s}' for L, e in chosen for m in mats for s in suff]
            shards = {}
            for t in wanted:
                if t not in wm:
                    raise SystemExit(f'{c}: {t} not in the index')
                shards.setdefault(wm[t], []).append(t)
            plan = []
            for shard, ts in sorted(shards.items()):
                hl = struct.unpack('<Q', get(budget, repo, rev, shard, 'header length', 0, 7))[0]
                assert 2 <= hl <= 100_000_000
                hdr = json.loads(get(budget, repo, rev, shard, 'header', 8, 8 + hl - 1))
                for t in ts:
                    info = hdr[t]
                    b, e = info['data_offsets']
                    n = 1
                    for d in info['shape']:
                        n *= d
                    assert n * DTYPE_BYTES[info['dtype']] == e - b, t
                    plan.append((shard, 8 + hl + b, 8 + hl + e, t, info))
            planned = sum(p[2] - p[1] for p in plan)
            print(f'  {c}: {len(plan)} tensors in {len(shards)} shards, {planned} B planned', flush=True)
            if '--plan' in argv:
                continue
            plan.sort()
            data = {}
            i = 0
            while i < len(plan):   # runs of adjacent tensors of one shard, one request each
                j = i + 1
                while j < len(plan) and plan[j][0] == plan[i][0] and plan[j][1] == plan[j - 1][2] and \
                        plan[j][2] - plan[i][1] <= MERGE_CAP:
                    j += 1
                blob = get(budget, repo, rev, plan[i][0], f'{j - i} tensors from {plan[i][3]}', plan[i][1], plan[j - 1][2] - 1)
                for shard, b, e, t, info in plan[i:j]:
                    data[t] = blob[b - plan[i][1]:e - plan[i][1]]
                i = j
            folder = os.path.join(out, c)
            os.makedirs(folder, exist_ok=True)
            open(os.path.join(folder, 'config.json'), 'wb').write(config)
            doc, at = {}, 0
            order = sorted(plan, key=lambda p: wanted.index(p[3]))
            for shard, b, e, t, info in order:
                doc[t] = {'dtype': info['dtype'], 'shape': info['shape'], 'data_offsets': [at, at + e - b]}
                at += e - b
            raw = json.dumps(doc, separators=(',', ':')).encode()
            raw += b' ' * (-len(raw) % 8)
            path = os.path.join(folder, 'experts.safetensors')
            with open(path + '.partial', 'wb') as f:
                f.write(struct.pack('<Q', len(raw)) + raw)
                for p in order:
                    f.write(data[p[3]])
            os.replace(path + '.partial', path)
            for shard, b, e, t, info in order:
                m = re.search(r'layers\.(\d+)\..*experts\.(\d+)\.', t)
                rows.append([c, repo, rev, shard, t, int(m[1]), int(m[2]), b, e - 1, e - b,
                             hashlib.sha256(data[t]).hexdigest()])
            h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            print(f'  wrote {c}/experts.safetensors {os.path.getsize(path)} B sha256 {h}', flush=True)
            old.setdefault('files', {})[c] = {'bytes': os.path.getsize(path), 'sha256': h}
    if '--plan' in argv:
        print(f'counted {budget.used} B')
        return 0
    old['requests'] = [r for r in old['requests'] if r['repo'] not in {CHECKPOINTS[c][0] for p in names for c in PAIRS[p]}]
    old['requests'] += budget.log
    old['samples'] = rows
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(old, f, indent=1)
    with open(os.path.join(out, 'samples.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('checkpoint\trepository\tcommit\tshard\ttensor\tlayer\texpert\tfirst_byte\tlast_byte\tbytes\tsha256\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    print(f'received {sum(r["bytes"] for r in budget.log)} B in {len(budget.log)} requests, {budget.retries} retries')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
