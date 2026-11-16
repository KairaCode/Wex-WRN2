# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WEX4 converters (docs/WEX4-SPEC.md, sections 9 and 10). Standard library only.

  pack      safetensors -> WEX4. The source headers are kept verbatim, so unpack rebuilds every
            source file with the same SHA-256 (unless the container is value-exact).
      python -I wex4_convert.py pack OUT.wex4 SRC.safetensors [SRC2 ...] [--dll wex4_tools.dll]
             [--experts raw|nr3 [--names PROFILE] --nr3-dll wex_nr3_tools.dll --model FILE.wnm2 [--cz] [--threads N]]
             [--part-mib 64]
            --threads N: WNR3 records encoded and checked on N threads ahead of the writer (the same file for any N).
  train     safetensors -> a WNM2 model for pack --experts nr3, from a seeded sample of the routed experts (the
            statistics c/tools/build_nr3.py --train collects from a WEX1 container; no container needed).
      python -I wex4_convert.py train OUT.wnm2 SRC.safetensors [SRC2 ...] --nr3-dll wex_nr3_tools.dll
             [--names PROFILE] [--sample 80] [--seed 20261116] [--cz] [--threads N]
  unpack    WEX4 -> the source safetensors files, each checked against its recorded SHA-256; prints each file's
            SHA-256. --sha-only rebuilds every file into its SHA-256 only and writes no file.
      python -I wex4_convert.py unpack IN.wex4 OUT_DIR [--dll wex4_tools.dll] [--sha-only]
  upgrade   WEX1 (raw) or WEX3/WNR3 expert container -> WEX4, in place and non-destructive: the records
            stay where they are, the old header page and table are kept as hashed blobs, the index is appended.
      python -I wex4_convert.py upgrade FILE.wex [--names PROFILE]
  downgrade undo an upgrade (restores the old header page, truncates the appended part).
      python -I wex4_convert.py downgrade FILE.wex
PROFILE is a naming profile (PROFILES below): how a model family names the six tensors of a routed expert.
Default deepseek.
"""
import ctypes
import hashlib
import io
import json
import os
import random
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
from wex4 import need, align, sha256, PAGE  # noqa: E402

ST_MAX_HEADER = 100_000_000
SMALL, GROUP = 64 << 10, 4 << 20      # tensors under SMALL share a record of up to GROUP bytes
ENTRY = struct.Struct('<QIII6H6H3I3II')   # legacy WEX3 entry, 72 B
WEX_HEADER = struct.Struct('<IIIIQQII')
# Naming profiles: how a model family names the six tensors of a routed expert in an MXFP4 checkpoint.
# name -> (prefix with {L} = layer and {e} = expert, names of the matrices w1, w2, w3 of the slab, suffix of the
# weight tensor, suffix of the scale tensor, scale dtype, weight dtype). The two dtypes are what upgrade writes
# (a WEX1/WEX3 container does not record them); pack takes the dtypes of the source. The names were read from
# the safetensors headers of published checkpoints: the table with the sources is in wex4/README.md.
# Roles: in every profile the three matrices are, in slab order, the gate, the down and the up matrix; by name for
# gate_proj, down_proj, up_proj, and from the model code for w1, w2, w3 (wex4/README.md, "Matrix roles").
_W, _PROJ = ('w1', 'w2', 'w3'), ('gate_proj', 'down_proj', 'up_proj')
PROFILES = {
    'deepseek': ('layers.{L}.ffn.experts.{e}.', _W, 'weight', 'scale', 'F8_E8M0', 'I8'),
    'deepseek-quark': ('layers.{L}.ffn.experts.{e}.', _W, 'weight', 'weight_scale', 'U8', 'U8'),
    'qwen38': ('model.language_model.layers.{L}.mlp.experts.{e}.', _PROJ, 'weight', 'weight_scale', 'U8', 'U8'),
    'qwen3-moe': ('model.layers.{L}.mlp.experts.{e}.', _PROJ, 'weight', 'weight_scale', 'U8', 'U8'),
    'kimi-k26': ('language_model.model.layers.{L}.mlp.experts.{e}.', _PROJ, 'weight', 'weight_scale', 'U8', 'U8'),
    'mixtral': ('model.layers.{L}.block_sparse_moe.experts.{e}.', _W, 'weight', 'weight_scale', 'U8', 'U8'),
    'minimax-m3': ('language_model.model.layers.{L}.block_sparse_moe.experts.{e}.', _W, 'weight', 'weight_scale', 'U8', 'U8'),
    'mistral-large4': ('layers.{L}.experts.{e}.', _W, 'weight', 'weight_scale', 'U8', 'U8'),
    'step3p5': ('model.layers.{L}.moe.{e}.', _PROJ, 'weight', 'weight_scale', 'U8', 'U8'),
    # exports in the compressed-tensors format (quant_method compressed-tensors): the weight is 'weight_packed'
    'deepseek-ct': ('layers.{L}.ffn.experts.{e}.', _W, 'weight_packed', 'weight_scale', 'U8', 'U8'),
    'qwen38-ct': ('model.language_model.layers.{L}.mlp.experts.{e}.', _PROJ, 'weight_packed', 'weight_scale', 'U8', 'U8'),
    'qwen3-moe-ct': ('model.layers.{L}.mlp.experts.{e}.', _PROJ, 'weight_packed', 'weight_scale', 'U8', 'U8'),
    'kimi-k26-ct': ('language_model.model.layers.{L}.mlp.experts.{e}.', _PROJ, 'weight_packed', 'weight_scale', 'U8', 'U8'),
    'mixtral-ct': ('model.layers.{L}.block_sparse_moe.experts.{e}.', _W, 'weight_packed', 'weight_scale', 'U8', 'U8'),
    'minimax-m3-ct': ('language_model.model.layers.{L}.block_sparse_moe.experts.{e}.', _W, 'weight_packed', 'weight_scale',
                      'U8', 'U8'),
    'step3p7-ct': ('model.layers.{L}.moe.experts.{e}.', _PROJ, 'weight_packed', 'weight_scale', 'U8', 'U8'),
}


def expert_name(family, layer, expert, matrix, weight):
    """Tensor name of matrix 0..2 (w1, w2, w3) of one expert: its weight tensor, or its scale tensor."""
    prefix, matrices, weight_suffix, scale_suffix = PROFILES[family][:4]
    return f'{prefix.format(L=layer, e=expert)}{matrices[matrix]}.{weight_suffix if weight else scale_suffix}'


def expert_pattern(family):
    """Compiled pattern of a profile's expert tensor names: groups layer, expert, matrix name, suffix."""
    prefix, matrices, weight_suffix, scale_suffix = PROFILES[family][:4]
    before, rest = prefix.split('{L}')
    between, after = rest.split('{e}')
    number = '(0|[1-9][0-9]*)'
    text = re.escape(before) + number + re.escape(between) + number + re.escape(after)
    text += f'({"|".join(map(re.escape, matrices))})\\.({re.escape(weight_suffix)}|{re.escape(scale_suffix)})\\Z'
    return re.compile(text.encode())


def mxfp4_dtypes(parts):
    """True when every tensor found for one expert, {(matrix, is weight): (Tensor, read)}, has a dtype of an
    MXFP4 slab: scale U8 or F8_E8M0, weight U8 or I8. False for a float or FP8 expert."""
    allowed = ((wex4.DTYPE_CODE['U8'], wex4.DTYPE_CODE['F8_E8M0']), (wex4.DTYPE_CODE['U8'], wex4.DTYPE_CODE['I8']))
    return all(spec.dtype in allowed[is_weight] for (_, is_weight), (spec, _) in parts.items())


def mxfp4_slab(parts):
    """Why the six MXFP4 tensors of one expert are not a slab as an EXPERT record needs it (WEX4-SPEC section
    5.5), or None when they are."""
    if len(parts) != 6:
        return f'incomplete expert ({len(parts)} of 6 tensors)'
    for m in range(3):
        scale, weight = parts[(m, False)][0], parts[(m, True)][0]
        if not (len(scale.shape) == 2 and all(0 < d <= 65535 for d in scale.shape + weight.shape) and
                weight.shape == (scale.shape[0], 16 * scale.shape[1])):
            return 'expert tensor shape'
    if sum(spec.nbytes for spec, _ in parts.values()) > wex4.MAX_DECODED:
        return 'expert larger than a record'
    return None


EXPERT_NAME = expert_pattern('deepseek')
BLOB_LEGACY_PAGE, BLOB_LEGACY_TABLE, BLOB_LEGACY_GAP = 16, 17, 18
FLOATS = ('F16', 'BF16', 'F32', 'F64')


def _unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, f'duplicate JSON key {key!r}')
        out[key] = value
    return out


def st_index(path, data_check=True):
    """Strict safetensors header: (header bytes, data start, [(name, dtype, shape, begin, end)] in data order)."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(8)
        need(len(head) == 8, 'safetensors: no header length')
        n = struct.unpack('<Q', head)[0]
        need(n <= ST_MAX_HEADER and n <= size - 8, 'safetensors: header length')
        raw = f.read(n)
    need(len(raw) == n, 'safetensors: short header')
    try:
        doc = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)
    except (ValueError, RecursionError) as error:
        raise wex4.FormatError(f'safetensors: header is not UTF-8 JSON ({error})') from None
    need(isinstance(doc, dict), 'safetensors: header is not an object')
    meta = doc.pop('__metadata__', None)
    need(meta is None or (isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values())), 'safetensors: metadata')
    tensors = []
    for name, info in doc.items():
        need(isinstance(info, dict) and set(info) == {'dtype', 'shape', 'data_offsets'}, f'safetensors: {name}')
        dtype, shape, span = info['dtype'], info['shape'], info['data_offsets']
        need(isinstance(dtype, str) and isinstance(shape, list) and len(shape) <= 8 and
             all(type(d) is int and 0 <= d <= wex4.U64_MAX for d in shape), f'safetensors: {name} shape')
        need(isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span) and
             0 <= span[0] <= span[1], f'safetensors: {name} offsets')
        elements = 1
        for d in shape:
            elements *= d
            need(elements <= wex4.U64_MAX, f'safetensors: {name} shape overflow')
        try:
            name = name.encode('utf-8')
        except UnicodeEncodeError:
            raise wex4.FormatError('safetensors: tensor name is not UTF-8') from None
        need(len(name) <= 65535 and b'\0' not in name, 'safetensors: tensor name too long or with NUL')
        bits = wex4.BITS[wex4.DTYPE_CODE[dtype]] if dtype in wex4.DTYPE_CODE else 0
        need(not bits or elements * bits == (span[1] - span[0]) * 8, f'safetensors: {name} size')
        tensors.append((name, dtype, tuple(shape), span[0], span[1]))
    tensors.sort(key=lambda t: (t[3], t[4]))
    at = 0
    for name, _, _, begin, end in tensors:   # contiguous from 0, no holes, no overlaps
        need(begin == at, f'safetensors: hole or overlap before {name!r}')
        at = end
    need(not data_check or 8 + n + at == size, 'safetensors: data does not end at the end of the file')
    return raw, 8 + n, tensors


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Nr3:
    """c/tools/wex_nr3_tools.c: the WNR3 encoder used for expert records."""

    def __init__(self, dll, model_path, cz):
        lib = ctypes.CDLL(os.fspath(dll))
        P, U32, SZ = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
        lib.wex_nr3_prepare.argtypes = [P, P, ctypes.c_int, P, P]
        lib.wex_nr3_encode_record.argtypes = [P, P, P, U32, U32, U32, U32, P, SZ, P, P]
        lib.wex_nr3_verify.argtypes = [P, P, P, U32, U32, P]
        lib.wex_nr3_canonicalize.argtypes = [P, P]
        self.lib, self.cz = lib, cz
        with open(model_path, 'rb') as f:
            self.model = f.read()
        self.model_id = wex4.wnm2_id(self.model)

    def encode(self, entry, slab, layer, expert):
        """The padded WNR3 record of a raw slab, verified against the slab (Z(slab) with cz); None = keep raw."""
        coded = ctypes.create_string_buffer(len(slab))
        flags = self.lib.wex_nr3_prepare(entry, slab, int(self.cz), coded, None)
        need(flags >= 0, 'wex_nr3_prepare failed')
        cap = len(slab) * 4
        out, used, total = ctypes.create_string_buffer(cap), ctypes.c_size_t(), ctypes.c_size_t()
        status = self.lib.wex_nr3_encode_record(self.model, entry, coded, flags, layer << 16 | expert, 2048, 512, out, cap,
                                                ctypes.byref(used), ctypes.byref(total))
        need(status >= 0, 'wex_nr3_encode_record failed')
        if status == 1:
            return None
        record = out.raw[:total.value]
        expected = ctypes.create_string_buffer(slab, len(slab))
        if self.cz:
            self.lib.wex_nr3_canonicalize(entry, expected)
        fields = list(ENTRY.unpack(entry))
        fields[1], fields[-1] = len(record), 5
        need(self.lib.wex_nr3_verify(self.model, ENTRY.pack(*fields), record, layer, expert, expected) == 0,
             f'L{layer} E{expert}: encoded record does not decode to the source')
        return record


def expert_entry(scales, weights):
    """Legacy entry bytes (offset/total/flags 0) of 3 (rows, cols) scale and 3 weight shapes."""
    sizes = [r * c for r, c in scales] + [r * c for r, c in weights]
    offsets = [sum(sizes[:i]) for i in range(6)]
    return ENTRY.pack(0, 0, sum(sizes[:3]), sum(sizes[3:]), *[d for s in scales for d in s], *[d for s in weights for d in s],
                      *offsets, 0)


def _reader(path, start, begin, end):
    def read():
        with open(path, 'rb') as f:
            f.seek(start + begin)
            return f.read(end - begin)
    return read


def find_experts(indexed, family):
    """The routed experts of indexed source files [(path, data start, st_index tensors)] under a naming profile:
    ({(layer, expert): {(matrix, is weight): (Tensor, read)}} of the experts that become expert records,
    [(layer, expert)] of the experts that stay ordinary tensors). The rules are pack's (its docstring); an
    expert whose tensors are all MXFP4 but do not form a slab is refused here."""
    need(family in PROFILES, f'unknown naming profile {family!r}: one of {", ".join(PROFILES)}')
    pattern, (_, matrices, weight_suffix) = expert_pattern(family), PROFILES[family][:3]
    expert_groups = {}
    for path, start, tensors in indexed:
        for name, dtype, shape, begin, end in tensors:
            match = pattern.match(name)
            if match:
                key = (int(match[1]), int(match[2]))
                part = (matrices.index(match[3].decode()), match[4].decode() == weight_suffix)
                expert_groups.setdefault(key, {})[part] = (wex4.Tensor(name, dtype, shape, end - begin),
                                                           _reader(path, start, begin, end))

    def other_form(layer, expert, parts):
        """True when a matrix has no weight tensor under the profile's suffix but has a plain '.weight' that is
        not MXFP4 (an FP8 or float expert of a compressed-tensors export, with its scale as 'weight_scale')."""
        if weight_suffix == 'weight':
            return False
        for m in range(3):
            if (m, True) not in parts:
                name = expert_name(family, layer, expert, m, True).rsplit('.', 1)[0] + '.weight'
                if dtypes.get(name.encode(), 'U8') not in ('U8', 'I8'):
                    return True
        return False
    dtypes = {name: dtype for _, _, tensors in indexed for name, dtype, *_ in tensors}
    as_tensors = sorted(key for key, parts in expert_groups.items() if not mxfp4_dtypes(parts) or other_form(*key, parts))
    for key in as_tensors:
        del expert_groups[key]
    for (layer, expert), parts in sorted(expert_groups.items()):
        problem = mxfp4_slab(parts)
        need(not problem, f'L{layer} E{expert}: {problem} under naming profile {family!r}')
    return expert_groups, as_tensors


def expert_slab(parts):
    """(tensors in slab order: the three scales, then the three weights; legacy entry bytes; read() of the slab)
    of one expert of find_experts."""
    order = [parts[(m, False)] for m in range(3)] + [parts[(m, True)] for m in range(3)]
    specs, reads = [s for s, _ in order], [r for _, r in order]
    entry = expert_entry([s.shape for s in specs[:3]], [s.shape for s in specs[3:]])
    return specs, entry, lambda: b''.join(r() for r in reads)


class _Ahead:
    """WNR3 records encoded ahead on a thread pool (the encoder and its check run in C, outside the GIL):
    get(k) returns the record of job k, with jobs k+1 .. k+depth already running. Jobs are taken in order."""

    def __init__(self, jobs, threads, depth):
        import concurrent.futures
        self.jobs, self.depth, self.next, self.running = jobs, depth, 0, {}
        self.pool = concurrent.futures.ThreadPoolExecutor(threads)

    def get(self, k):
        while self.next < len(self.jobs) and self.next <= k + self.depth:
            self.running[self.next] = self.pool.submit(self.jobs[self.next])
            self.next += 1
        result = self.running.pop(k).result()
        if not self.running and self.next == len(self.jobs):
            self.pool.shutdown()
        return result


def pack(out_path, sources, clib=None, experts=None, nr3=None, part_bytes=64 << 20, family='deepseek', report=None,
         threads=1):
    """safetensors files -> one WEX4 file. experts: None (tensors as they come), 'raw' or 'nr3' (expert records).
    family: the naming profile that says which tensors are routed experts. An expert with a tensor that is not
    MXFP4 (a float or FP8 layer of a mixed checkpoint) stays ordinary tensors; so does, under a profile whose
    weight is not '.weight', an expert with a plain '.weight' that is not U8 or I8. An expert whose tensors are all
    MXFP4 must be a whole slab: a missing tensor (a missing shard, the wrong profile) or a wrong shape is refused.
    threads: WNR3 records encoded (and checked) on that many threads ahead of the writer; the file is the same
    for any value. report: a dict that receives 'expert_records' and 'experts_as_tensors'."""
    records, empty, meta_sources, indexed = [], [], [], []
    seen = set()

    need(sources and part_bytes > 0 and threads >= 1, 'pack: no source file, bad part size or thread count')
    need(len({os.path.normcase(os.path.basename(p)) for p in sources}) == len(sources), 'two sources have the same file name')
    need(family in PROFILES, f'unknown naming profile {family!r}: one of {", ".join(PROFILES)}')
    for path in sources:   # every header first: the six tensors of one expert may sit in two files
        raw, start, tensors = st_index(path)
        meta_sources.append({'name': os.path.basename(path), 'bytes': os.path.getsize(path),
                             'sha256': file_sha256(path), 'header': raw.decode('utf-8')})
        indexed.append((path, start, tensors))
        for name, *_ in tensors:
            need(name not in seen, f'tensor {name!r} appears twice')
            seen.add(name)
    expert_groups, as_tensors = find_experts(indexed, family) if experts else ({}, [])
    reader = _reader
    in_slab = {spec.name for parts in expert_groups.values() for spec, _ in parts.values()}
    if report is not None:
        report.update(expert_records=len(expert_groups), experts_as_tensors=len(as_tensors))
    for path, start, tensors in indexed:
        pending = {}   # element width -> (tensors, readers, bytes) of the open small-tensor group

        def flush(width):
            specs, reads, size = pending.pop(width)
            records.append(wex4.Record(specs, size, lambda reads=reads: b''.join(r() for r in reads), ('bp', width, 0)))
        for name, dtype, shape, begin, end in tensors:
            if name in in_slab:
                continue
            spec, read, size = wex4.Tensor(name, dtype, shape, end - begin), reader(path, start, begin, end), end - begin
            if not size:
                empty.append(spec)
                continue
            code = wex4.DTYPE_CODE.get(dtype, 0)
            width = max(1, wex4.BITS[code] // 8) if code else 1
            codec = ('bp', width, int(dtype in FLOATS))   # floats: exponent into the top plane
            if size < SMALL:
                if pending.get(width, (0, 0, 0))[2] + size > GROUP:
                    flush(width)
                specs, reads, total = pending.get(width, ([], [], 0))
                pending[width] = (specs + [spec], reads + [read], total + size)
                continue
            parts = (size + part_bytes - 1) // part_bytes
            for k in range(parts):
                lo, hi = k * part_bytes, min(size, (k + 1) * part_bytes)
                records.append(wex4.Record([spec] if not k else [], hi - lo, reader(path, start, begin + lo, begin + hi),
                                           codec, parts=parts))
        for width in list(pending):
            flush(width)

    blobs, layers, n_experts, required = [], 0, 0, 0
    if expert_groups:
        if nr3:
            blobs.append((wex4.BLOB_WNM2, nr3.model_id, nr3.model))
            required = wex4.REQ_VALUE_EXACT if nr3.cz else 0
        layers, n_experts = max(k[0] for k in expert_groups) + 1, max(k[1] for k in expert_groups) + 1
        need(layers <= 65536 and n_experts <= 65536, 'expert grid too large')
        jobs = []
        ahead = _Ahead(jobs, threads, 2 * threads) if nr3 and threads > 1 else None
        for k, ((layer, expert), parts) in enumerate(sorted(expert_groups.items())):
            specs, entry, slab = expert_slab(parts)
            jobs.append(lambda entry=entry, slab=slab, layer=layer, expert=expert: nr3.encode(entry, slab(), layer, expert))
            codec = 'raw'
            if nr3:
                codec = ('pre', wex4.CODEC_WNR3, (lambda k=k: ahead.get(k)) if ahead else jobs[-1])
            flags = wex4.REC_EXPERT | (wex4.REC_VALUE_EXACT if nr3 and nr3.cz else 0)
            records.append(wex4.Record(specs, sum(s.nbytes for s in specs), slab, codec, flags, layer << 16 | expert,
                                       1 if nr3 else 0))
    meta = json.dumps({'wex4_meta': 1, 'sources': meta_sources}, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return wex4.write(out_path, records, blobs, empty, meta, layers, n_experts, required, wex4.OPT_ST_SOURCES, encoder=clib)


def sample_experts(keys, count, seed):
    """`count` of the (layer, expert) keys, drawn with a seeded generator and returned sorted; all of them when
    there are no more than `count`. The same keys, count and seed give the same sample on any machine."""
    keys = sorted(keys)
    if count >= len(keys):
        return keys
    return sorted(random.Random(seed).sample(keys, count))


def train(out_path, sources, nr3_dll, family='deepseek', count=80, seed=20261116, cz=False, threads=1):
    """A WNM2 model for `pack --experts nr3`, trained from safetensors files: the coding statistics of a seeded
    sample of `count` routed experts (sample_experts over the experts pack would turn into expert records), as
    c/tools/build_nr3.py trains from a WEX1 container. Writes the 59,408-byte model to out_path (through
    out_path.partial) and returns (model id, [(layer, expert)] of the sample)."""
    import concurrent.futures
    indexed = []
    for path in sources:
        _, start, tensors = st_index(path)
        indexed.append((path, start, tensors))
    groups, _ = find_experts(indexed, family)
    need(groups, f'train: no MXFP4 routed expert under naming profile {family!r}')
    picks = sample_experts(groups, count, seed)
    lib = ctypes.CDLL(os.path.abspath(nr3_dll))
    P = ctypes.c_void_p
    lib.wex_nr3_prepare.argtypes = [P, P, ctypes.c_int, P, P]
    lib.wex_nr3_count_expert.argtypes = [P, P, P, P]
    lib.wex_nr3_build_model.argtypes = [P, P, P, P]
    W, S = 3 * 567 * 16, 3 * 256

    def counts(key):
        _, entry, slab = expert_slab(groups[key])
        data = slab()
        coded = ctypes.create_string_buffer(len(data))
        wc, sc = (ctypes.c_uint64 * W)(), (ctypes.c_uint64 * S)()
        need(lib.wex_nr3_prepare(entry, data, int(cz), coded, None) >= 0, f'L{key[0]} E{key[1]}: wex_nr3_prepare failed')
        need(lib.wex_nr3_count_expert(entry, coded, wc, sc) == 0, f'L{key[0]} E{key[1]}: wex_nr3_count_expert failed')
        return list(wc), list(sc)
    total_w, total_s = [0] * W, [0] * S
    with concurrent.futures.ThreadPoolExecutor(max(1, threads)) as pool:
        for wc, sc in pool.map(counts, picks):   # summed in sample order: the same model for any thread count
            total_w = [a + b for a, b in zip(total_w, wc)]
            total_s = [a + b for a, b in zip(total_s, sc)]
    blob, ident = ctypes.create_string_buffer(wex4.WNM2_BYTES), ctypes.c_uint32()
    need(lib.wex_nr3_build_model((ctypes.c_uint64 * W)(*total_w), (ctypes.c_uint64 * S)(*total_s), blob,
                                 ctypes.byref(ident)) == 0, 'wex_nr3_build_model failed')
    model = blob.raw
    need(wex4.wnm2_id(model) == ident.value, 'the trained model does not pass the WNM2 check')
    with open(os.fspath(out_path) + '.partial', 'wb') as f:
        f.write(model)
    os.replace(os.fspath(out_path) + '.partial', out_path)
    return ident.value, picks


def _header_tensors(path, rd):
    """Names in data order of a file that so far holds only its header; every entry must equal the container's."""
    raw, start, tensors = st_index(path, data_check=False)
    for name, dtype, shape, begin, end in tensors:
        need(name in rd.by_name, f'tensor {name!r} missing from the container')
        t = rd.tensors[rd.by_name[name]]
        need((t[1], tuple(t[3]), t[7]) == (wex4.DTYPE_CODE.get(dtype, 0), shape, end - begin), f'header and container disagree on {name!r}')
    return [t[0] for t in tensors]


def unpack(in_path, out_dir, clib=None, sha_only=False, digests=None):
    """Rebuild every source safetensors file; returns [(name, sha256 matches)]. sha_only: rebuild each file as a
    stream into its SHA-256 and write nothing but a header-sized temporary file (removed), for a check on a disk
    that cannot hold the rebuilt files. digests: a dict that receives {name: SHA-256 hex of the rebuilt file}."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    with open(in_path, 'rb') as f:
        rd = wex4.Reader(f, clib)
        need(rd.optional & wex4.OPT_ST_SOURCES, 'container holds no safetensors sources')
        try:
            doc = json.loads(rd.meta.decode('utf-8'), object_pairs_hook=_unique)
            sources = [(s['name'], s['header'], s['sha256']) for s in doc['sources']]
            need(all(type(x) is str for s in sources for x in s), 'metadata: source fields')
        except (ValueError, RecursionError, KeyError, TypeError) as error:
            raise wex4.FormatError(f'metadata: {error!r}') from None
        names = [os.path.normcase(n) for n, _, _ in sources]
        need(len(set(names)) == len(names) and all(n and n == os.path.basename(n) and n not in ('.', '..') and
                                                   os.path.normcase(os.path.abspath(os.path.join(out_dir, n))) != os.path.normcase(os.path.abspath(in_path)) for n, _, _ in sources),
             'metadata: source file name (unsafe, repeated or the container itself)')
        cache = {}

        def record(r):
            if r not in cache:
                cache.clear()
                cache[r] = rd.record(r)
            return cache[r]
        for name, text, recorded in sources:
            try:
                raw = text.encode('utf-8')
            except UnicodeEncodeError:
                raise wex4.FormatError('metadata: header text') from None
            digest = hashlib.sha256()
            tmp = os.path.join(out_dir, name + '.partial')
            try:
              with open(tmp, 'wb') as out:
                for piece in [struct.pack('<Q', len(raw)), raw]:
                    out.write(piece)
                    digest.update(piece)
                out.flush()
                for key in _header_tensors(tmp, rd):
                    need(key in rd.by_name, f'tensor {key!r} missing from the container')
                    _, _, _, _, r, count, offset, nbytes = rd.tensors[rd.by_name[key]]
                    for k in range(count):
                        data = record(r + k)
                        piece = data[offset:offset + nbytes] if count == 1 else data
                        if not sha_only:
                            out.write(piece)
                        digest.update(piece)
              ok = digest.hexdigest() == recorded
              need(ok or rd.required & wex4.REQ_VALUE_EXACT, f'{name}: the rebuilt file does not match its SHA-256')
            except BaseException:
                os.remove(tmp)
                raise
            if sha_only:
                os.remove(tmp)
            else:
                os.replace(tmp, os.path.join(out_dir, name))
            if digests is not None:
                digests[name] = digest.hexdigest()
            results.append((name, ok))
    return results


def _sync(f):
    f.flush()
    try:
        os.fsync(f.fileno())
    except (AttributeError, io.UnsupportedOperation):   # an in-memory file (tests, dry runs); real I/O errors propagate
        pass


class _Probe:
    """What wex4.Reader reads to validate: the file size, the header page and the index."""

    def __init__(self, size, parts):
        self.size, self.parts, self.at = size, parts, 0

    def seek(self, offset, whence=0):
        self.at = self.size if whence else offset

    def tell(self):
        return self.at

    def read(self, n):
        return self.parts[self.at][:n]


# tensor names and dtypes of an upgraded expert container, per naming profile (matrices w1, w2, w3)
NAMES = {family: (lambda L, e, m, w, family=family: expert_name(family, L, e, m, w), profile[4], profile[5])
         for family, profile in PROFILES.items()}


def upgrade(path, hash_records=True, family='deepseek'):
    """WEX1 / WEX3-WNR3 container -> WEX4 in place: appends the old header page and the index, then rewrites
    the header page. Nothing else is touched; downgrade() restores the file byte for byte.
    path may be an open binary file object (r+b) instead of a path."""
    need(family in NAMES, f'unknown naming profile {family!r}: one of {", ".join(NAMES)}')
    with (open(path, 'r+b') if isinstance(path, (str, os.PathLike)) else path) as f:
        f.seek(0, os.SEEK_END)
        old_bytes = f.tell()
        f.seek(0)
        page = f.read(PAGE)
        need(len(page) == PAGE, 'file shorter than a header page')
        magic, version, layers, n_experts, _, table_offset, hflags, _ = WEX_HEADER.unpack_from(page)
        need((magic, version) in ((0x57455831, 1), (0x57455833, 3)), 'not a WEX1 or WEX3 container')
        nr3, cz = bool(hflags & 32), bool(hflags & 64)
        need(version == 1 or (nr3 and not hflags & ~(1 | 32 | 64)), 'only raw (WEX1) and WNR3 containers can be upgraded')
        table_bytes = layers * n_experts * ENTRY.size
        need(layers <= 65536 and n_experts <= 65536 and table_bytes <= old_bytes - min(table_offset, old_bytes), 'expert grid or short entry table')
        f.seek(table_offset)
        table = f.read(table_bytes)
        need(len(table) == table_bytes, 'short entry table')
        entries = [ENTRY.unpack_from(table, i * ENTRY.size) for i in range(layers * n_experts)]
        present = sorted((e[0], i) for i, e in enumerate(entries) if e[1])
        need(present and [i for _, i in present] == sorted(i for _, i in present), 'records are not in table order')
        first_record = present[0][0]
        need(first_record >= table_offset + align(table_bytes), 'records before the end of the table')

        # the area before the records: model blob, table, and whatever else as opaque hashed blobs
        front = [(table_offset, align(table_bytes), BLOB_LEGACY_TABLE, 0, table_bytes)]
        if nr3:
            dmagic, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
            need(dmagic == 0x444d4e57 and model_bytes == wex4.WNM2_BYTES, 'WNMD descriptor')
            front.append((model_offset, align(model_bytes), wex4.BLOB_WNM2, model_id, model_bytes))
        front.sort()
        blobs, at = [], PAGE
        for offset, size, kind, ident, used in front + [(first_record, 0, None, 0, 0)]:
            need(offset >= at and offset % PAGE == 0, 'overlapping front sections')
            if offset > at:
                blobs.append((BLOB_LEGACY_GAP, 0, at, offset - at))
            if kind is not None:
                blobs.append((kind, ident, offset, used))
            at = offset + size
        blob_rows = []
        for kind, ident, offset, size in blobs:
            f.seek(offset)
            data = f.read(size)
            need(not any(f.read(align(size) - size)), 'nonzero padding after a front section')
            need(kind != wex4.BLOB_WNM2 or wex4.wnm2_id(data) == ident, 'model id')
            blob_rows.append(wex4.BLOB.pack(kind, ident, offset, size, sha256(data), 0))
        model_blob = 1 + [b[0] for b in blobs].index(wex4.BLOB_WNM2) if nr3 else 0
        legacy_page_offset = old_bytes
        blob_rows.append(wex4.BLOB.pack(BLOB_LEGACY_PAGE, 0, legacy_page_offset, PAGE, sha256(page), 0))

        record_rows, tensor_rows, names, name_at, at, total, maximum = [], [], [], 0, first_record, 0, 0
        for r, (offset, i) in enumerate(present):
            _, stored, scale_bytes, weight_bytes = entries[i][:4]
            shapes, offsets, codec = entries[i][4:16], entries[i][16:22], entries[i][22]
            need(offset == at and codec in ((0, 5) if nr3 else (0,)), f'entry {i}: unexpected offset or codec')
            layer, expert = divmod(i, n_experts)
            decoded = scale_bytes + weight_bytes
            for m in range(6):
                name = NAMES[family][0](layer, expert, m % 3, m >= 3).encode()
                rows, cols = shapes[2 * m:2 * m + 2]
                tensor_rows.append(wex4.TENSOR.pack(name_at, len(name), wex4.DTYPE_CODE[NAMES[family][2 if m >= 3 else 1]], 2,
                                                    rows, cols, 0, 0, 0, 0, 0, 0, r, 1, offsets[m], rows * cols))
                names.append(name)
                name_at += len(name)
            digest = bytes(32)
            if hash_records:
                f.seek(offset)
                digest = sha256(f.read(stored))
            flags = wex4.REC_EXPERT | (wex4.REC_VALUE_EXACT if cz and codec == 5 else 0)
            record_rows.append(wex4.RECORD.pack(offset, stored, decoded, codec, flags, model_blob if codec == 5 else 0,
                                                layer << 16 | expert, 6 * r, 6, digest, 0))
            at += stored
            total += decoded
            maximum = max(maximum, decoded)
        need(at <= old_bytes, 'a record runs past the end of the file')
        need(at == old_bytes, f'records end at byte {at} but the file has {old_bytes}: an interrupted upgrade leaves '
                              f'appended bytes, truncate the file to {at} bytes and run again')
        order = sorted(range(len(names)), key=lambda k: names[k])
        index = b''.join(blob_rows) + b''.join(record_rows) + b''.join(tensor_rows) + \
            struct.pack(f'<{len(order)}I', *order) + b''.join(names)
        index_offset = legacy_page_offset + PAGE
        file_bytes = index_offset + align(len(index))
        new = wex4.HEADER.pack(wex4.MAGIC, 4, 0, wex4.REQ_VALUE_EXACT if cz else 0, 0, file_bytes, index_offset, len(index),
                               len(blob_rows), len(record_rows), len(names), layers, n_experts, maximum, name_at, 0, total,
                               sha256(index))
        new += bytes(wex4.HEADER_HASHED - len(new))
        new += sha256(new)
        wex4.Reader(_Probe(file_bytes, {0: new, index_offset: index + bytes(align(len(index)) - len(index))}))   # refuse before writing
        f.seek(legacy_page_offset)   # 1. append: the old container is still intact if this is interrupted
        f.write(page + index + bytes(align(len(index)) - len(index)))
        _sync(f)
        f.seek(0)                    # 2. one 4096-byte write switches the file to WEX4
        f.write(new)
        _sync(f)
    return dict(index_bytes=len(index), file_bytes=file_bytes, added=file_bytes - old_bytes, records=len(record_rows))


def _saved_page(f):
    """Offset of the old header page that upgrade appended, found without the WEX4 header: the first page from
    the end that is a WEX1/WEX3 header whose records end exactly there. For a file whose page 0 is damaged."""
    f.seek(0, os.SEEK_END)
    size = f.tell()
    for at in range(size - 2 * PAGE, max(PAGE, size - wex4.MAX_INDEX - 2 * PAGE) - 1, -PAGE):
        f.seek(at)
        page = f.read(PAGE)
        if page[:4] not in (b'1XEW', b'3XEW'):
            continue
        _, _, layers, n_experts, _, table_offset, _, _ = WEX_HEADER.unpack_from(page)
        if not 0 < layers * n_experts <= wex4.MAX_RECORDS or table_offset + layers * n_experts * ENTRY.size > at:
            continue
        f.seek(table_offset)
        table = f.read(layers * n_experts * ENTRY.size)
        ends = [e[0] + e[1] for e in (ENTRY.unpack_from(table, i * ENTRY.size) for i in range(layers * n_experts)) if e[1]]
        if ends and max(ends) == at:
            return at, page
    raise wex4.FormatError('no saved header page found')


def downgrade(path):
    with open(path, 'r+b') as f:
        try:
            rd = wex4.Reader(f)
        except wex4.FormatError:   # header page or index damaged (a torn switch): recover from the records' end
            at, page = _saved_page(f)
            f.seek(0)
            f.write(page)
            _sync(f)
            f.truncate(at)
            return
        legacy = [b for b in rd.blobs if b[0] == BLOB_LEGACY_PAGE]
        need(len(legacy) == 1, 'not an upgraded container')
        f.seek(legacy[0][2])
        page = f.read(PAGE)
        need(sha256(page) == legacy[0][4], 'saved header page is damaged')
        need(legacy[0][2] + PAGE == rd.index_offset and rd.index_offset + align(rd.index_bytes) == rd.file_bytes and
             page[:4] in (b'1XEW', b'3XEW'), 'not the layout that upgrade writes')
        f.seek(0)
        f.write(page)
        f.flush()
        os.fsync(f.fileno())
        f.truncate(legacy[0][2])


def main(argv):
    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    if len(argv) < 3:
        print(__doc__)
        return 2
    clib = wex4.CLib(option('--dll')) if '--dll' in argv else None
    values = {argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith('--') and a not in ('--cz', '--sha-only')}
    positional = [a for a in argv[2:] if not a.startswith('--') and a not in values]
    threads = int(option('--threads', '1'))
    if argv[1] == 'pack':
        mode = option('--experts')
        need(mode in (None, 'raw', 'nr3'), '--experts takes raw or nr3')
        nr3 = Nr3(option('--nr3-dll'), option('--model'), '--cz' in argv) if mode == 'nr3' else None
        need(mode or '--names' not in argv, 'pack: --names needs --experts raw or nr3')
        report, family = {}, option('--names', 'deepseek')
        root = pack(positional[0], positional[1:], clib, mode, nr3, int(option('--part-mib', '64')) << 20, family, report,
                    threads)
        if mode:
            print(f'{report["expert_records"]} expert records; {report["experts_as_tensors"]} experts kept as ordinary '
                  f'tensors (float or FP8, not MXFP4)')
            if not report['expert_records']:
                print(f'warning: --experts made no expert record: no MXFP4 expert has the names of profile {family!r} '
                      f'(--names; one of {", ".join(PROFILES)})', file=sys.stderr)
        print('root', root.hex())
    elif argv[1] == 'unpack':
        digests = {}
        for name, ok in unpack(positional[0], positional[1], clib, '--sha-only' in argv, digests):
            print(name, digests[name], 'sha256 identical' if ok else 'DIFFERS (value-exact container)')
    elif argv[1] == 'train':
        ident, picks = train(positional[0], positional[1:], option('--nr3-dll'), option('--names', 'deepseek'),
                             int(option('--sample', '80')), int(option('--seed', '20261116')), '--cz' in argv, threads)
        print(f'model {ident:08x} trained on {len(picks)} experts: {",".join(f"{L}:{e}" for L, e in picks)}')
    elif argv[1] == 'upgrade':
        print(upgrade(positional[0], family=option('--names', 'deepseek')))
    elif argv[1] == 'downgrade':
        downgrade(positional[0])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
