# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Helpers of the GLM-5.3-Flash WEX4 test kit (run.sh calls them; README.md says what each stage proves).

    python -I kit.py env RESULTS_DIR WORK_DIR [CHECKPOINT_DIR]
    python -I kit.py check-checkpoint DIR [--sha256] [--manifest FILE]
    python -I kit.py sizes CONTAINER CHECKPOINT_DIR OUT.json
    python -I kit.py decode CONTAINER CHECKPOINT_DIR OUT.json --ref DLL [--fast DLL] [--sample N] [--seed S]
    python -I kit.py make-bin CONTAINER CHECKPOINT_DIR OUT.bin [--sample N] [--seed S]
    python -I kit.py compare-sha CHECKPOINT_DIR UNPACK.log OUT.json
    python -I kit.py engine-ab A.log B.log OUT.json

Standard library only. The container is opened read-only; the checkpoint is never written.
"""
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'wex4'))
import wex4  # noqa: E402
import wex4_convert as conv  # noqa: E402

PROFILE = 'qwen38'
MANIFEST = os.path.join(HERE, 'glm53-files.tsv')
ENTRY = conv.ENTRY


def fail(message):
    print(f'FAIL: {message}', file=sys.stderr)
    raise SystemExit(1)


def option(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def shards(directory):
    """The checkpoint's safetensors files, sorted by name."""
    names = sorted(n for n in os.listdir(directory) if n.endswith('.safetensors'))
    if not names:
        fail(f'no .safetensors file in {directory}')
    return [os.path.join(directory, n) for n in names]


def write_json(path, doc):
    with open(path + '.partial', 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    os.replace(path + '.partial', path)


# ---- env -------------------------------------------------------------------------------------------------
def run_text(cmd):
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return (done.stdout + done.stderr).strip() if done.returncode == 0 else None


def cpu_info():
    """(model name, sorted list of the flags this kit cares about)."""
    wanted = ('avx', 'avx2', 'fma', 'bmi2', 'f16c', 'avx512f', 'neon', 'asimd', 'sve')
    name, flags = platform.processor() or platform.machine(), set()
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as f:
            text = f.read()
        m = re.search(r'^model name\s*:\s*(.+)$', text, re.M)
        name = m.group(1).strip() if m else name
        for m in re.finditer(r'^(?:flags|Features)\s*:\s*(.+)$', text, re.M):
            flags |= set(m.group(1).split())
    except OSError:
        if platform.system() == 'Windows':
            out = run_text(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'])
            name = out.splitlines()[0].strip() if out else os.environ.get('PROCESSOR_IDENTIFIER', name)
        else:
            out = run_text(['sysctl', '-n', 'machdep.cpu.brand_string'])
            name = out or name
            feats = run_text(['sysctl', '-n', 'machdep.cpu.features', 'machdep.cpu.leaf7_features']) or ''
            flags |= {w.lower().replace('.', '_') for w in feats.split()}
        if platform.machine() in ('arm64', 'aarch64'):
            flags.add('neon')
    if not flags:   # what the compiler sees on this CPU (-march=native)
        try:
            done = subprocess.run([os.environ.get('CC', 'gcc'), '-march=native', '-dM', '-E', '-'], input='',
                                  capture_output=True, text=True, timeout=60)
            macros = {'__AVX__': 'avx', '__AVX2__': 'avx2', '__FMA__': 'fma', '__BMI2__': 'bmi2', '__F16C__': 'f16c',
                      '__AVX512F__': 'avx512f', '__ARM_NEON': 'neon', '__ARM_FEATURE_SVE': 'sve'}
            flags |= {v for k, v in macros.items() if f'#define {k} ' in done.stdout}
        except (OSError, subprocess.SubprocessError):
            pass
    return name, sorted(f for f in wanted if f in flags)


def ram_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    if platform.system() == 'Windows':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + \
                       [(n, ctypes.c_uint64) for n in ('total', 'avail', 'pt', 'pa', 'vt', 'va', 've')]
        s = Status()
        s.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return round(s.total / 2 ** 30, 1)
    out = run_text(['sysctl', '-n', 'hw.memsize'])
    return round(int(out) / 2 ** 30, 1) if out and out.isdigit() else None


def gpus():
    found = {'nvidia': [], 'vulkan': []}
    out = run_text(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])
    if out:
        found['nvidia'] = [line.strip() for line in out.splitlines() if line.strip()]
    out = run_text(['vulkaninfo', '--summary'])
    if out:
        found['vulkan'] = sorted({m.group(1).strip() for m in re.finditer(r'deviceName\s*=\s*(.+)', out)})
    return found


def env(argv):
    results, work = argv[2], argv[3]
    checkpoint = argv[4] if len(argv) > 4 else None
    name, flags = cpu_info()
    tools = {}
    cc = os.environ.get('CC', 'gcc')
    for tool in (cc, 'python3', 'python', 'nvcc', 'glslangValidator', 'hf', 'curl', 'git', 'make'):
        path = shutil.which(tool)
        version = run_text([tool, '--version']) if path else None
        tools[tool] = version.splitlines()[0][:120] if version else None
    disk = {}
    for label, path in (('results', results), ('work', work), ('checkpoint', checkpoint)):
        if path and os.path.isdir(path):
            disk[label] = round(shutil.disk_usage(path).free / 1e9, 1)
    doc = {'os': f'{platform.system()} {platform.release()}', 'machine': platform.machine(), 'cpu': name,
           'cpu_flags': flags, 'cpu_threads': os.cpu_count(), 'ram_gb': ram_gb(),
           'python': platform.python_version(), 'tools': tools, 'gpus': gpus(), 'free_disk_gb': disk}
    write_json(os.path.join(results, 'env.json'), doc)
    print(json.dumps(doc, indent=1))
    if not tools[cc]:
        fail(f'{cc} not found (README: prerequisites)')
    if platform.machine() in ('x86_64', 'AMD64', 'amd64') and 'avx2' not in flags:
        print('warning: no AVX2 flag found: the fast CPU decoder builds its scalar kernel', file=sys.stderr)


# ---- checkpoint ------------------------------------------------------------------------------------------
def manifest(path=MANIFEST):
    """{file name: (bytes, sha256 or '')} of the checkpoint at the kit's pinned revision."""
    rows = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            name, size, sha = line.rstrip('\n').split('\t')
            rows[name] = (int(size), sha)
    return rows


def check_checkpoint(argv):
    directory = argv[2]
    rows = manifest(option(argv, '--manifest', MANIFEST))
    bad, total, hashed = [], 0, 0
    for name, (size, sha) in sorted(rows.items()):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            bad.append(f'{name}: missing')
            continue
        got = os.path.getsize(path)
        total += got
        if got != size:
            bad.append(f'{name}: {got} bytes, the manifest says {size}')
        elif '--sha256' in argv and sha:
            if conv.file_sha256(path) != sha:
                bad.append(f'{name}: SHA-256 differs from the manifest')
            hashed += 1
    print(f'checkpoint: {len(rows)} files in the manifest, {total} bytes here, {hashed} SHA-256 checked, '
          f'{len(bad)} problems')
    for b in bad:
        print('  ' + b)
    if bad:
        fail('the checkpoint is incomplete or differs from the pinned revision (stage fetch resumes a download)')


# ---- sizes -----------------------------------------------------------------------------------------------
def open_container(path, clib=None):
    f = open(path, 'rb')
    return f, wex4.Reader(f, clib)


def raw_equivalent(rd):
    """Bytes of the container that pack --experts raw writes from the same sources: every WNR3 expert record
    stored as its slab padded to 4096 bytes, no WNM2 blob, and an index one blob row (64 bytes) shorter. The
    other records are the same bytes in both (pack encodes them alike)."""
    gain = sum(wex4.align(rec[2]) - rec[1] for rec in rd.records if rec[3] == wex4.CODEC_WNR3)
    blobs = [b for b in rd.blobs if b[0] == wex4.BLOB_WNM2]
    blob_bytes = sum(wex4.align(b[3]) for b in blobs)
    index_now, index_raw = wex4.align(rd.index_bytes), wex4.align(rd.index_bytes - wex4.BLOB.size * len(blobs))
    return rd.file_bytes + gain - blob_bytes - (index_now - index_raw)


def sizes(argv):
    container, checkpoint, out = argv[2], argv[3], argv[4]
    f, rd = open_container(container)
    with f:
        experts = [rec for rec in rd.records if rec[4] & wex4.REC_EXPERT]
        doc = {'container_bytes': rd.file_bytes, 'records': len(rd.records), 'expert_records': len(experts),
               'expert_records_wnr3': sum(rec[3] == wex4.CODEC_WNR3 for rec in experts),
               'expert_decoded_bytes': sum(rec[2] for rec in experts),
               'expert_stored_bytes': sum(rec[1] for rec in experts),
               'other_decoded_bytes': sum(rec[2] for rec in rd.records if not rec[4] & wex4.REC_EXPERT),
               'other_stored_bytes': sum(rec[1] for rec in rd.records if not rec[4] & wex4.REC_EXPERT),
               'raw_container_bytes': raw_equivalent(rd), 'root': rd.root.hex(), 'grid': [rd.layers, rd.experts]}
    doc['checkpoint_bytes'] = sum(os.path.getsize(p) for p in shards(checkpoint))
    doc['ratio_container_to_checkpoint'] = round(doc['container_bytes'] / doc['checkpoint_bytes'], 4)
    doc['ratio_raw_to_checkpoint'] = round(doc['raw_container_bytes'] / doc['checkpoint_bytes'], 4)
    doc['ratio_container_to_raw'] = round(doc['container_bytes'] / doc['raw_container_bytes'], 4)
    doc['ratio_expert_stored_to_decoded'] = round(doc['expert_stored_bytes'] / max(1, doc['expert_decoded_bytes']), 4)
    write_json(out, doc)
    print(json.dumps(doc, indent=1))


# ---- decode ----------------------------------------------------------------------------------------------
def source_experts(checkpoint):
    indexed = []
    for path in shards(checkpoint):
        _, start, tensors = conv.st_index(path)
        indexed.append((path, start, tensors))
    groups, _ = conv.find_experts(indexed, PROFILE)
    return groups


def pick_records(rd, argv):
    """Expert record numbers to check: all, or a seeded sample (--sample N, --seed S) of them."""
    owners = {rec[6]: r for r, rec in enumerate(rd.records) if rec[4] & wex4.REC_EXPERT}
    keys = [(o >> 16, o & 0xffff) for o in owners]
    count = int(option(argv, '--sample', '0'))
    keys = conv.sample_experts(keys, count, int(option(argv, '--seed', '20261116'))) if count else sorted(keys)
    return [(key, owners[key[0] << 16 | key[1]]) for key in keys]


def decode_one(clib, rd, r, stored, model):
    out = ctypes.create_string_buffer(rd.records[r][2])
    t0 = time.perf_counter()
    status = clib.lib.wex4_decode_record_buf(rd.page, rd.file_bytes, rd.index, len(rd.index), r, stored, model, 0, out)
    return status, out.raw, time.perf_counter() - t0


def decode(argv):
    container, checkpoint, out = argv[2], argv[3], argv[4]
    paths = {'reference': option(argv, '--ref'), 'fast': option(argv, '--fast')}
    libs = {k: wex4.CLib(v) for k, v in paths.items() if v}
    if 'reference' not in libs:
        fail('decode needs --ref wex4_tools_ref.dll')
    groups = source_experts(checkpoint)
    f, rd = open_container(container, libs['reference'])
    doc = {'records': 0, 'wnr3': 0, 'raw': 0, 'mismatch': [], 'seconds': {k: 0.0 for k in libs}, 'read_seconds': 0.0,
           'decoded_bytes': 0, 'value_exact': bool(rd.required & wex4.REQ_VALUE_EXACT)}
    if doc['value_exact']:
        fail('a value-exact (--cz) container: this kit compares decoded records with the source bytes, build without --cz')
    with f:
        picks = pick_records(rd, argv)
        model = None
        for n, ((layer, expert), r) in enumerate(picks):
            rec = rd.records[r]
            t0 = time.perf_counter()
            stored = rd.stored(r)
            doc['read_seconds'] += time.perf_counter() - t0
            if hashlib.sha256(stored).digest() != rec[9]:
                doc['mismatch'].append(f'L{layer} E{expert}: stored bytes differ from the index SHA-256')
                continue
            if (layer, expert) not in groups:
                fail(f'L{layer} E{expert}: in the container but not an MXFP4 expert of the checkpoint')
            source = expert_bytes(groups[(layer, expert)])
            doc['records'] += 1
            doc['decoded_bytes'] += rec[2]
            if rec[3] == wex4.CODEC_RAW:
                doc['raw'] += 1
                if stored[:rec[2]] != source:
                    doc['mismatch'].append(f'L{layer} E{expert}: RAW record differs from the source slab')
                continue
            doc['wnr3'] += 1
            doc['wnr3_decoded_bytes'] = doc.get('wnr3_decoded_bytes', 0) + rec[2]
            if model is None:
                model = rd.blob(rec[5] - 1)
            for name, lib in libs.items():
                status, data, seconds = decode_one(lib, rd, r, stored, model)
                doc['seconds'][name] += seconds
                if status or data != source:
                    doc['mismatch'].append(f'L{layer} E{expert}: {name} decoder status {status}, '
                                           f'{"bytes differ" if data != source else "bytes equal"}')
            if (n + 1) % 200 == 0:
                print(f'  {n + 1} of {len(picks)} records', flush=True)
    doc['ms_per_wnr3_record'] = {k: round(1000 * v / doc['wnr3'], 2) for k, v in doc['seconds'].items()} if doc['wnr3'] else {}
    doc['mb_per_s'] = {k: round(doc['wnr3_decoded_bytes'] / v / 1e6, 1) for k, v in doc['seconds'].items() if v}
    doc['sample'] = int(option(argv, '--sample', '0')) or None
    doc['decoders'] = {k: os.path.basename(v) for k, v in paths.items() if v}
    write_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != 'mismatch'} | {'mismatches': len(doc['mismatch'])}, indent=1))
    for line in doc['mismatch'][:20]:
        print('  ' + line)
    if doc['mismatch']:
        fail(f'{len(doc["mismatch"])} records do not decode to their source bytes')


def expert_bytes(parts):
    return conv.expert_slab(parts)[2]()


# ---- bins for c/tests/test_wex_nr3_gpu.cu ----------------------------------------------------------------
def make_bin(argv):
    """The layout of c/tools/build_nr3.py bins: u32 model bytes, the model, u32 count, then per record its 72-byte
    entry (file offset 0, total bytes = stored, codec 5), u32 raw bytes, the expected slab (the source) and the
    record as stored. Only WNR3 records go in."""
    container, checkpoint, out = argv[2], argv[3], argv[4]
    groups = source_experts(checkpoint)
    f, rd = open_container(container)
    rows = []
    with f, open(out + '.partial', 'wb') as o:
        model = None
        picks = [(k, r) for k, r in pick_records(rd, argv) if rd.records[r][3] == wex4.CODEC_WNR3]
        for (layer, expert), r in picks:
            rec = rd.records[r]
            if model is None:
                model = rd.blob(rec[5] - 1)
                o.write(struct.pack('<I', len(model)) + model + struct.pack('<I', len(picks)))
            e = rd.expert_entry(r)
            entry = ENTRY.pack(0, rec[1], e['scale_bytes'], e['weight_bytes'], *[d for s in e['scale_shapes'] for d in s],
                               *[d for s in e['weight_shapes'] for d in s], *e['scale_offsets'], *e['weight_offsets'], 5)
            source = expert_bytes(groups[(layer, expert)])
            o.write(entry + struct.pack('<I', len(source)) + source + rd.stored(r))
            rows.append(f'{layer}:{expert}')
    if not rows:
        os.remove(out + '.partial')
        fail('no WNR3 record to put in a bin')
    os.replace(out + '.partial', out)
    print(f'{len(rows)} WNR3 records: {",".join(rows)}')


# ---- round trip ------------------------------------------------------------------------------------------
def compare_sha(argv):
    """The SHA-256 that `unpack --sha-only` printed for every rebuilt file, against the checkpoint's files
    hashed here."""
    checkpoint, log, out = argv[2], argv[3], argv[4]
    rebuilt = {}
    with open(log, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'^(\S+\.safetensors) ([0-9a-f]{64}) ', line)
            if m:
                rebuilt[m.group(1)] = m.group(2)
    rows, bad = [], []
    for path in shards(checkpoint):
        name = os.path.basename(path)
        mine = conv.file_sha256(path)
        rows.append({'file': name, 'checkpoint': mine, 'rebuilt': rebuilt.get(name)})
        if rebuilt.get(name) != mine:
            bad.append(name)
    write_json(out, {'files': len(rows), 'identical': len(rows) - len(bad), 'different': bad, 'rows': rows})
    print(f'round trip: {len(rows)} files, {len(rows) - len(bad)} with the SHA-256 of the checkpoint, {len(bad)} different')
    if bad:
        fail(f'rebuilt files differ from the checkpoint: {", ".join(bad[:5])}')


# ---- engine A/B ------------------------------------------------------------------------------------------
def engine_log(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    tf = re.search(r'^teacher_forcing((?: \d+)*)\s*$', text, re.M)
    gr = re.search(r'^greedy((?: \d+)*)\s*$', text, re.M)
    sp = re.search(r'decode (\d+) token in ([\d.]+)s = ([\d.]+) tok/s', text)
    # the engine's last WEX4 line: records read, and the WNR3 ones decoded by the CUDA helper and by the CPU
    # (a record the helper refuses with status 1 falls back to the CPU and is counted there)
    wx = re.findall(r'WEX4 records: (\d+) read, (\d+) WNR3 decoded \(CUDA helper (\d+), CPU (\d+)\)', text)
    return {'wex_records_read': int(wx[-1][0]) if wx else None, 'wnr3_decoded': int(wx[-1][1]) if wx else None,
            'helper_decodes': int(wx[-1][2]) if wx else None, 'cpu_decodes': int(wx[-1][3]) if wx else None,'teacher_forcing': [int(x) for x in tf.group(1).split()] if tf else None,
            'greedy': [int(x) for x in gr.group(1).split()] if gr else None,
            'decode_tokens': int(sp.group(1)) if sp else None, 'decode_seconds': float(sp.group(2)) if sp else None,
            'tok_per_s': float(sp.group(3)) if sp else None}


def engine_ab(argv):
    a, b, out = engine_log(argv[2]), engine_log(argv[3]), argv[4]
    if not a['greedy'] or not b['greedy']:
        fail('an engine log has no "greedy" line: the run stopped (see its log)')
    same = a['greedy'] == b['greedy'] and a['teacher_forcing'] == b['teacher_forcing']
    first = next((i for i, (x, y) in enumerate(zip(a['greedy'], b['greedy'])) if x != y), None)
    doc = {'shards': a, 'container': b, 'tokens_identical': same, 'first_difference': first}
    write_json(out, doc)
    print(f'engine A/B: {len(a["greedy"])} and {len(b["greedy"])} greedy tokens, '
          f'{"identical" if same else "DIFFERENT"}; tok/s shards {a["tok_per_s"]}, container {b["tok_per_s"]}')
    if not same:
        fail(f'the container run gave other tokens (first difference at greedy token {first})')


def main(argv):
    commands = {'env': env, 'check-checkpoint': check_checkpoint, 'sizes': sizes, 'decode': decode,
                'make-bin': make_bin, 'compare-sha': compare_sha, 'engine-ab': engine_ab}
    if len(argv) < 3 or argv[1] not in commands:
        print(__doc__)
        return 2
    try:
        commands[argv[1]](argv)
    except wex4.FormatError as error:
        fail(f'refused: {error}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
