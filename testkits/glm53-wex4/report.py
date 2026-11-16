# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The kit's results as one report: RESULTS/report.md (to paste into a GitHub issue) and RESULTS/report.json.

    python -I report.py RESULTS_DIR

It reads what the stages wrote (*.status, env.json, sizes.json, decode.json, roundtrip.json, speed-cpu-*.json,
engine*.json, times.tsv, the CUDA and Vulkan logs) and keeps the machine's description: OS, CPU, RAM, GPUs, tool
versions. It leaves out user names, host names and paths: every string is scrubbed of the login name, the host
name and the home folder, and anything that looks like an absolute path becomes <path>. Read the report before
you post it all the same. Standard library only.
"""
import getpass
import json
import os
import platform
import re
import socket
import sys

STAGES = ('env', 'fetch', 'build', 'verify', 'roundtrip', 'decode', 'speed', 'engine')
PATH = re.compile(r'(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|c|d|e|f|g|h|mnt|tmp|var|opt|root|media|Volumes)/)[^\s"\',;)]*')


def secrets():
    words = set()
    for get in (getpass.getuser, socket.gethostname, platform.node):
        try:
            words.add(get())
        except Exception:   # no login name in a container, for example
            pass
    for name in ('USER', 'USERNAME', 'LOGNAME', 'COMPUTERNAME', 'HOSTNAME', 'HOME', 'USERPROFILE'):
        if os.environ.get(name):
            words.add(os.environ[name])
    words.add(os.path.expanduser('~'))
    return sorted((w for w in words if w and len(w) >= 3), key=len, reverse=True)


SECRETS = secrets()


def scrub(value):
    if isinstance(value, str):
        value = PATH.sub('<path>', value)
        for word in SECRETS:
            value = re.sub(re.escape(word), '<redacted>', value, flags=re.I)
        return value
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    return value


def load(results, name):
    path = os.path.join(results, name)
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f) if name.endswith('.json') else f.read()
    except (OSError, ValueError):
        return None


def gb(n):
    return f'{n / 1e9:.2f} GB' if isinstance(n, (int, float)) else 'n/a'


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    results = argv[1]
    status = {s: (load(results, f'{s}.status') or 'not run').strip() for s in STAGES}
    for extra in ('decode-cuda', 'decode-vulkan'):
        status[extra] = (load(results, f'{extra}.status') or 'not run').strip()
    times = {}
    for line in (load(results, 'times.tsv') or '').splitlines():
        name, _, seconds = line.partition('\t')
        if seconds.strip().isdigit():
            times[name] = int(seconds)   # the last run of a step wins
    env, sizes = load(results, 'env.json') or {}, load(results, 'sizes.json') or {}
    decode, roundtrip = load(results, 'decode.json') or {}, load(results, 'roundtrip.json') or {}
    engine = {k: load(results, f'{k}.json') for k in ('engine', 'engine-cuda')}
    speed = [s for s in (load(results, f'speed-cpu-{p}.json') for p in (1, 2, 3)) if s]
    cuda_log = load(results, 'decode-cuda.txt') or ''
    vulkan_log = load(results, 'decode-vulkan.txt') or ''
    doc = {'kit': 'glm53-wex4', 'checkpoint': 'amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25', 'status': status, 'env': env,
           'seconds': times, 'sizes': sizes,
           'decode': {k: decode.get(k) for k in ('records', 'wnr3', 'raw', 'sample', 'ms_per_wnr3_record', 'mb_per_s',
                                                  'decoders')} | {'mismatches': len(decode.get('mismatch') or [])},
           'roundtrip': {k: roundtrip.get(k) for k in ('files', 'identical', 'different')},
           'speed_cpu': [{k: s.get(k) for k in ('records', 'ms_per_wnr3_record', 'mb_per_s')} for s in speed],
           'cuda': [line for line in cuda_log.splitlines() if re.search(r'ALL OK|compared|records|FAIL|device', line)][-12:],
           'vulkan': [line for line in vulkan_log.splitlines()
                      if re.search(r'all checks|FAIL|^device', line) or
                      ('WNR3 records' in line and 'vectors/' not in line and not line.startswith(('binary', 'batch')))][-6:],
           'engine': {k: ({x: v.get(x) for x in ('tokens_identical', 'first_difference')} |
                          {'tok_per_s_shards': (v.get('shards') or {}).get('tok_per_s'),
                           'tok_per_s_container': (v.get('container') or {}).get('tok_per_s'),
                           'wnr3_decoded': (v.get('container') or {}).get('wnr3_decoded'),
                           'helper_decodes': (v.get('container') or {}).get('helper_decodes'),
                           'cpu_decodes': (v.get('container') or {}).get('cpu_decodes'),
                           'greedy_tokens': len((v.get('shards') or {}).get('greedy') or [])}) if v else None
                      for k, v in engine.items()}}
    doc = scrub(doc)
    lines = ['## GLM-5.3-Flash WEX4 kit report', '',
             f'Checkpoint `{doc["checkpoint"]}`, naming profile `qwen38`, WNR3 expert records.', '',
             '### Machine', '',
             f'- OS: {env.get("os")} ({env.get("machine")}); CPU: {env.get("cpu")}, {env.get("cpu_threads")} threads, '
             f'flags {", ".join(env.get("cpu_flags") or []) or "none found"}; RAM: {env.get("ram_gb")} GB',
             f'- NVIDIA: {"; ".join((env.get("gpus") or {}).get("nvidia") or []) or "none"}; Vulkan: '
             f'{"; ".join((env.get("gpus") or {}).get("vulkan") or []) or "none"}',
             f'- tools: ' + '; '.join(f'{k} {v}' for k, v in (env.get('tools') or {}).items() if v), '',
             '### Stages', '', '| Stage | Status |', '|---|---|']
    lines += [f'| {s} | {doc["status"][s]} |' for s in doc['status']]
    lines += ['', '### Sizes', '']
    if sizes:
        lines += [f'- checkpoint {gb(sizes.get("checkpoint_bytes"))}; container {gb(sizes.get("container_bytes"))} '
                  f'(ratio to the checkpoint {sizes.get("ratio_container_to_checkpoint")}); the same container with RAW '
                  f'expert records {gb(sizes.get("raw_container_bytes"))} (computed from the index, ratio '
                  f'{sizes.get("ratio_raw_to_checkpoint")}); WNR3 over RAW {sizes.get("ratio_container_to_raw")}',
                  f'- {sizes.get("expert_records")} expert records ({sizes.get("expert_records_wnr3")} WNR3): '
                  f'{gb(sizes.get("expert_decoded_bytes"))} decoded, {gb(sizes.get("expert_stored_bytes"))} stored '
                  f'(ratio {sizes.get("ratio_expert_stored_to_decoded")}); root `{sizes.get("root")}`']
    else:
        lines.append('- no sizes.json (the build or verify stage did not run)')
    lines += ['', '### Checks', '',
              f'- round trip: {doc["roundtrip"].get("files")} files, {doc["roundtrip"].get("identical")} with the SHA-256 '
              f'of the checkpoint, different: {doc["roundtrip"].get("different") or "none"}',
              f'- decode: {doc["decode"].get("records")} expert records ({doc["decode"].get("wnr3")} WNR3, '
              f'{doc["decode"].get("raw")} RAW){" of a seeded sample" if doc["decode"].get("sample") else ""}, '
              f'{doc["decode"]["mismatches"]} not equal to the source; ms per WNR3 record '
              f'{doc["decode"].get("ms_per_wnr3_record")}, MB/s {doc["decode"].get("mb_per_s")}']
    for s in doc['speed_cpu']:
        lines.append(f'- speed, {s["records"]} records: ms per record {s["ms_per_wnr3_record"]}, MB/s {s["mb_per_s"]}')
    lines += [f'- CUDA helper: {doc["status"]["decode-cuda"]}'] + [f'  - `{line}`' for line in doc['cuda']]
    lines += [f'- Vulkan prototype: {doc["status"]["decode-vulkan"]}'] + [f'  - `{line}`' for line in doc['vulkan']]
    for k, v in doc['engine'].items():
        lines.append(f'- engine A/B ({k}): ' + (f'{v["greedy_tokens"]} greedy tokens, identical: {v["tokens_identical"]}, '
                                                 f'tok/s shards {v["tok_per_s_shards"]}, container {v["tok_per_s_container"]}; '
                                                 f'WNR3 records decoded {v["wnr3_decoded"]}: by the CUDA helper '
                                                 f'{v["helper_decodes"]}, by the CPU {v["cpu_decodes"]}'
                                                 if v else 'not run'))
    lines += ['', '### Times (s)', '', '| Step | Seconds |', '|---|---:|'] + [f'| {k} | {v} |' for k, v in times.items()]
    lines += ['', 'Each number is one run on this machine; the kit does not repeat the stages.', '']
    text = scrub('\n'.join(lines))
    with open(os.path.join(results, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(results, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    leaked = [w for w in SECRETS if w.lower() in (text + json.dumps(doc)).lower()]
    if leaked:
        print('the report still holds a login name, host name or home path: not written as final', file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
