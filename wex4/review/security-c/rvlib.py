# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Shared helpers of the security-c review: a field-level WEX4 builder (every index/header field can be
overridden, hashes always valid), a WBP1 record builder with hostile knobs, and the three-reader verdict
(fast DLL, reference DLL, Python). Loads wex4/out/*.dll read-only; writes only under this folder."""
import ctypes
import hashlib
import io
import os
import struct
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
NEXT = os.path.join(ROOT, 'wex4')
NOUT = os.path.join(NEXT, 'out')
OUT = os.path.join(HERE, 'out')
FIND = os.path.join(HERE, 'findings')
sys.path.insert(0, NEXT)
import wex4  # noqa: E402

C = wex4.CLib(os.path.join(NOUT, 'wex4_tools.dll'))
CREF = wex4.CLib(os.path.join(NOUT, 'wex4_tools_ref.dll'))
HEADER, BLOB, RECORD, TENSOR = wex4.HEADER, wex4.BLOB, wex4.RECORD, wex4.TENSOR


def sha(b):
    return hashlib.sha256(bytes(b)).digest()


def al(n):
    return (n + 4095) & ~4095


def pad(b):
    return bytes(b) + bytes(al(len(b)) - len(b))


def reseal(blob):
    blob = bytearray(blob)
    io_, ib = struct.unpack_from('<QQ', blob, 32)
    if io_ + ib <= len(blob):
        blob[96:128] = sha(blob[io_:io_ + ib])
    blob[4064:4096] = sha(blob[:4064])
    return bytes(blob)


def build(blobs=(), records=(), tensors=(), meta=b'', layers=0, experts=0, required=0, layout=None, hdr=None,
          name_order=None, info=None):
    """blobs: dict(kind,id,data[,offset,bytes,sha,reserved]); records: dict(stored,decoded,codec,flags,blob,owner,
    first,count[,offset,stored_bytes,sha,reserved]); tensors: dict(name,dtype,shape,record,rcount,offset,nbytes
    [,rank,name_off,name_len]). layout: extents in file order, 'I', ('B', i), ('R', i). hdr overrides header fields."""
    blobs, records, tensors = list(blobs), list(records), list(tensors)
    names = b''.join(t['name'] for t in tensors)
    index_bytes = 64 * len(blobs) + 72 * len(records) + 100 * len(tensors) + len(names) + len(meta)
    if layout is None:
        layout = (['I'] if index_bytes else []) + [('B', i) for i in range(len(blobs))] + [('R', i) for i in range(len(records))]
    pos, offs = 4096, {}
    for e in layout:
        offs[e] = pos
        pos += al(index_bytes) if e == 'I' else al(len(blobs[e[1]]['data'])) if e[0] == 'B' else len(records[e[1]]['stored'])
    rows = []
    for i, b in enumerate(blobs):
        rows.append(BLOB.pack(b['kind'], b['id'], b.get('offset', offs.get(('B', i), 0)), b.get('bytes', len(b['data'])),
                              b.get('sha', sha(b['data'])), b.get('reserved', 0)))
    for i, r in enumerate(records):
        rows.append(RECORD.pack(r.get('offset', offs.get(('R', i), 0)), r.get('stored_bytes', len(r['stored'])), r['decoded'],
                                r['codec'], r['flags'], r['blob'], r['owner'], r['first'], r['count'],
                                r.get('sha', sha(r['stored'])), r.get('reserved', 0)))
    name_at = 0
    for t in tensors:
        shape = list(t['shape'])
        rows.append(TENSOR.pack(t.get('name_off', name_at), t.get('name_len', len(t['name'])), t['dtype'],
                                t.get('rank', len(shape)), *shape, *([0] * (8 - len(shape))), t['record'], t['rcount'],
                                t['offset'], t['nbytes']))
        name_at += len(t['name'])
    order = name_order if name_order is not None else sorted(range(len(tensors)), key=lambda i: tensors[i]['name'])
    index = b''.join(rows) + struct.pack(f'<{len(order)}I', *order) + names + meta
    assert len(index) == index_bytes
    h = dict(major=4, minor=0, required=required, optional=0, file_bytes=pos, index_offset=offs.get('I', 4096),
             index_bytes=index_bytes, nb=len(blobs), nr=len(records), nt=len(tensors), layers=layers, experts=experts,
             max_decoded=max([r['decoded'] for r in records], default=0), names_bytes=len(names), meta_bytes=len(meta),
             total_decoded=sum(r['decoded'] for r in records), index_sha=sha(index))
    h.update(hdr or {})
    page = HEADER.pack(wex4.MAGIC, h['major'], h['minor'], h['required'], h['optional'], h['file_bytes'], h['index_offset'],
                       h['index_bytes'], h['nb'], h['nr'], h['nt'], h['layers'], h['experts'], h['max_decoded'],
                       h['names_bytes'], h['meta_bytes'], h['total_decoded'], h['index_sha'])
    page += bytes(4064 - len(page))
    out = bytearray(page + sha(page))
    for e in layout:
        out += pad(index) if e == 'I' else pad(blobs[e[1]]['data']) if e[0] == 'B' else records[e[1]]['stored']
    if info is not None:
        io_ = offs.get('I', 4096)
        info.update(index_offset=io_, blob0=io_, rec0=io_ + 64 * len(blobs), ten0=io_ + 64 * len(blobs) + 72 * len(records),
                    order0=io_ + 64 * len(blobs) + 72 * len(records) + 96 * len(tensors), nb=len(blobs), nr=len(records),
                    nt=len(tensors), names0=io_ + 64 * len(blobs) + 72 * len(records) + 100 * len(tensors),
                    index_bytes=index_bytes, offs=offs)
    return bytes(out)


def T(name, dtype, shape, record, offset, nbytes, rcount=1, **kw):
    d = dict(name=name if isinstance(name, bytes) else name.encode(), dtype=dtype, shape=shape, record=record,
             rcount=rcount, offset=offset, nbytes=nbytes)
    d.update(kw)
    return d


def R(stored, decoded, codec, owner, first, count, flags=0, blob=0, **kw):
    d = dict(stored=stored, decoded=decoded, codec=codec, flags=flags, blob=blob, owner=owner, first=first, count=count)
    d.update(kw)
    return d


def raw_rec(data, owner, first, count, **kw):
    return R(pad(data), len(data), 0, owner, first, count, **kw)


# ---- verdicts ----
def cv(lib, blob, flags=3):
    try:
        return lib.check(blob, flags, True)
    except OSError as e:   # SEH fault inside the DLL
        return -999, repr(e)


def py(blob, verify=True, clib=C):
    try:
        return 0, wex4.check(blob, verify, clib)
    except wex4.FormatError as e:
        return -1, str(e)
    except Exception as e:   # the Python reference must only raise FormatError on hostile input
        return -99, repr(e)


class Log:
    def __init__(self, name):
        self.name, self.n, self.accepted, self.bad, self.pyexc = name, 0, 0, [], {}
        os.makedirs(FIND, exist_ok=True)

    def tri(self, label, blob, flags=3, expect=None, pyclib=C):
        """All three readers on the same bytes. expect: True (must be accepted), False (must be refused), None."""
        a, b, p = cv(C, blob, flags), cv(CREF, blob, flags), py(blob, bool(flags & 1), pyclib)
        self.n += 1
        acc = [a[0] == 0, b[0] == 0, p[0] == 0]
        problem = None
        if a[0] == -999 or b[0] == -999:
            problem = 'CRASH'
        elif len(set(acc)) != 1:
            problem = 'DISAGREE'
        elif acc[0] and not (a[1] == b[1] == p[1]):
            problem = 'DIGEST'
        elif expect is not None and acc[0] != expect:
            problem = 'EXPECTED ' + ('accept' if expect else 'refuse')
        if p[0] == -99:
            self.pyexc.setdefault(p[1][:60], label)
        self.accepted += acc[0] and not problem
        if problem:
            path = os.path.join(FIND, f'{self.name}_{len(self.bad):03d}.wex4')
            if len(self.bad) < 20:
                with open(path, 'wb') as f:
                    f.write(blob)
            self.bad.append((problem, label, a[0], b[0], p[0], p[1] if p[0] else '', os.path.basename(path)))
            if len(self.bad) <= 20:
                print(f'  !! {problem}: {label}: fast {a[0]} ref {b[0]} py {p[0]} {p[1] if p[0] else ""}', flush=True)
        return a[0], b[0], p[0]

    def done(self):
        print(f'[{self.name}] {self.n} files, {self.accepted} accepted by all, {len(self.bad)} problems, '
              f'python non-FormatError exceptions: {self.pyexc}', flush=True)
        return dict(name=self.name, files=self.n, accepted=self.accepted, problems=self.bad, py_exceptions=self.pyexc)


# ---- WBP1 builder with knobs ----
def bp_parts(data, planes, chunk_log2, owner, transform=0, force=None, cdf_fn=None, word=None):
    """Like wex4.bp_encode but: force = {plane: 'rans' | 'raw'}, cdf_fn(p, counts) -> 257 cumulative values (any valid
    CDF, not the canonical one). Returns dict(head, cdfs, tables, payloads) of lists, to be edited and joined."""
    decoded, chunk = len(data), 1 << chunk_log2
    coded = wex4.rotate(data, planes, True) if transform else data
    word = planes | chunk_log2 << 8 | transform << 24 if word is None else word
    cdfs, tables, payloads, mask, g = [], [], [], 0, 0
    for p in range(planes):
        plane = bytes(coded[p::planes])
        streams = (len(plane) + chunk - 1) // chunk
        if not plane:
            tables.append(b'')
            payloads.append(b'')
            continue
        counts = [0] * 256
        for b in plane:
            counts[b] += 1
        cdf = cdf_fn(p, counts) if cdf_fn else wex4.quantize(counts)
        parts = [wex4.rans_encode(cdf, plane[s * chunk:(s + 1) * chunk], wex4.RANS_L + (wex4.stream_hash(
            (wex4.BP_MAGIC, owner, g + s, word), plane[s * chunk:(s + 1) * chunk]) & 0x7fffff)) for s in range(streams)]
        use = (force or {}).get(p) or ('rans' if 514 + 2 * streams + sum(map(len, parts)) < 4 * streams + len(plane) else 'raw')
        if use == 'rans':
            mask |= 1 << p
            cdfs.append(struct.pack('<257H', *cdf))
            tables.append(struct.pack(f'<{streams}H', *[min(len(x), 65535) for x in parts]))
            payloads.append(b''.join(parts))
        else:
            tables.append(b''.join(struct.pack('<I', wex4.stream_hash((wex4.BP_MAGIC, owner, g + s, word),
                                                                      plane[s * chunk:(s + 1) * chunk])) for s in range(streams)))
            payloads.append(plane)
        g += streams
    return dict(head=[wex4.BP_MAGIC, owner, decoded, planes, chunk_log2, mask, transform], cdfs=cdfs, tables=tables,
                payloads=payloads)


def bp_join(parts, extra=b''):
    return pad(struct.pack('<IIIBBBB', *[x & (0xffffffff if i < 3 else 0xff) for i, x in enumerate(parts['head'])]) +
               b''.join(parts['cdfs']) + b''.join(parts['tables']) + b''.join(parts['payloads']) + extra)


def bp_file(record, decoded, codec=6, owner=0, front=b'front-record', **kw):
    """A container whose LAST extent is `record` and whose max_decoded is its decoded size: with wex4_fuzz_check
    both the stored record and the decode buffer end at a guard page. One raw record in front keeps owner = 1."""
    recs, tens = [], []
    if owner:
        for i in range(owner):
            recs.append(raw_rec(front, i, i, 1))
            tens.append(T(f'f{i}', 0, [len(front)], i, 0, len(front)))
    recs.append(R(record, decoded, codec, owner, owner, 1, **kw))
    tens.append(T('x', 0, [], owner, 0, decoded))
    return build(records=recs, tensors=tens)


def c_bp(lib, rec, decoded, owner):
    out = ctypes.create_string_buffer(max(decoded, 1))
    try:
        st = lib.lib.wex4_bp_decode_record(bytes(rec), len(rec), decoded, owner, out)
    except OSError as e:
        return -999, repr(e)
    return st, out.raw[:decoded] if st == 0 else None


def py_bp(rec, decoded, owner):
    try:
        return 0, wex4.bp_decode(rec, decoded, owner)
    except wex4.FormatError as e:
        return -1, str(e)
    except Exception as e:
        return -99, repr(e)
