# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Build the WNR2 sample container from the WEX1 pilot (lab only).

  pilot-v1.wex (688 raw canonical slabs)  ->  pilot-v3-nr2.wex
  --canonical-zero                        ->  pilot-v3-nr2z.wex (FP4 -0 stored as +0: value-exact)

Layout: header page (WexHeader WEX3 v3, uuid at 40..55, WNMD model descriptor
at 64), the WNM2 model at 4096 zero-padded to 65536 (= table_offset), the
WexEntry table 4096-aligned, then the records contiguous up to EOF.
Pass 1 counts every present expert on the GPU and builds the model; pass 2
encodes each expert on the GPU (weight_chunk 4096, scale_chunk 2048), pads it
to 4096 and decodes it with the CPU reference (wex_nr2_verify) before it is
written; a record that would not be smaller than the raw slab stores the raw
slab (flags 0). The finished file goes through wex_validate_header/_table +
wex_nr2_model_location + wex_nr2_model_open, a sample of records is decoded
again from disk through the engine's slab path, every on-disk record is
re-hashed against the bytes that were verified, and the mirror under
U:/kaira_lab is a hard link of the same file. Only U: is written (plus the
report and model copy in this folder).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import shutil
import struct
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import nr2lib as L

BUILD = Path(os.environ.get('WEX_PILOT_DIR', 'U:/kaira_build/wexgd-20261019'))   # same variable as run_test_nr2.cmd
MIRROR = Path(os.environ.get('WEX_PILOT_MIRROR', 'U:/kaira_lab/wexgd-20261019'))
SOURCE = BUILD / 'pilot-v1.wex'
NEURAL = BUILD / 'pilot-v3-neural512.wex'
BUILD_JSON = BUILD / 'build.json'
MODEL_OFFSET = L.WEX_ALIGN
TABLE_OFFSET = L.WEX_ALIGN + L.align(L.MODEL_BYTES)          # 4096 + 61440 = 65536
FLAGS = L.WEX_HEADER_SCALE_TILES | L.WEX_HEADER_DEDUP | L.WEX_HEADER_NEURAL_RANS2


def log(**fields):
    print(json.dumps(fields), flush=True)


def file_sha256(path, block=16 << 20):
    h = hashlib.sha256()
    buf = bytearray(block)
    view = memoryview(buf)
    with open(path, 'rb', buffering=0) as f:
        while n := f.readinto(buf):
            h.update(view[:n])
    return h.hexdigest()


def read_experts(path, items, depth=4):
    """Background sequential reads of the raw slabs (in table order)."""
    q = queue.Queue(depth)

    def run():
        try:
            with open(path, 'rb', buffering=0) as f:
                for index, e in items:
                    buf = bytearray(L.raw_size(e))
                    f.seek(e[L.F_OFFSET])
                    if f.readinto(buf) != len(buf):
                        raise IOError(f'short read of expert {index}')
                    q.put((index, e, buf))
            q.put(None)
        except BaseException as error:   # surfaced in the consumer
            q.put(error)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--canonical-zero', action='store_true', help='map FP4 -0 to +0 (value-exact) -> pilot-v3-nr2z.wex')
    ap.add_argument('--weight-chunk', type=int, default=4096)
    ap.add_argument('--scale-chunk', type=int, default=2048)
    ap.add_argument('--verify-workers', type=int, default=4, help='CPU threads for the reference decode')
    ap.add_argument('--crosscheck-every', type=int, default=43, help='also CPU-encode every Nth expert and compare bytes')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing output (and its mirror link)')
    args = ap.parse_args()
    cz = 1 if args.canonical_zero else 0
    suffix = 'nr2z' if cz else 'nr2'
    name = f'pilot-v3-{suffix}.wex'
    final, partial, mirror = BUILD / name, BUILD / (name + '.partial'), MIRROR / name
    report_path = L.HERE / f'build-{suffix}.json'
    model_path = L.HERE / f'model-{suffix}.wnm2'
    began = time.perf_counter()

    tools = L.Tools()
    header, src_page, _, entries = L.read_container(SOURCE)
    magic, version, layers, experts, expert_bytes, src_table_offset, src_flags, _ = header
    assert (magic, version, src_flags) == (L.WEX_MAGIC_V1, 1, 0), 'source must be a raw WEX1 container'
    count = layers * experts
    present = [(i, e) for i, e in enumerate(entries) if e[L.F_TOTAL]]
    assert all(e == (0,) * 23 for e in entries if not e[L.F_TOTAL]), 'absent entries must be all zero'
    offsets = [e[L.F_OFFSET] for _, e in present]
    assert offsets == sorted(offsets)
    for _, e in present:
        assert e[L.F_FLAGS] == 0 and e[L.F_TOTAL] == L.align(L.raw_size(e)) == L.raw_size(e) <= expert_bytes
    categories = {}
    if BUILD_JSON.exists():
        categories = {s['index']: s['category'] for s in json.loads(BUILD_JSON.read_text())['selected']}

    if final.exists() or mirror.exists():
        if not args.overwrite:
            sys.exit(f'{final} or its mirror exists; pass --overwrite')
        for p in (mirror, final):
            if p.exists():
                p.unlink()
    if partial.exists():
        partial.unlink()
    need = SOURCE.stat().st_size + (4 << 30)
    free = shutil.disk_usage(BUILD).free
    assert free > need, f'not enough space on U: ({free} free, {need} wanted)'
    log(stage='start', output=str(final), canonical_zero=bool(cz), experts=len(present), free_gib=round(free / 2**30, 1))

    # ---- pass 1: training counts on the GPU, model ------------------------
    t1 = time.perf_counter()
    wc, sc = np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64)
    gpu_count_s = 0.0
    for n, (index, e, raw) in enumerate(read_experts(SOURCE, present), 1):
        t = time.perf_counter()
        rc = tools.count(raw, L.ENTRY.pack(*e), cz, wc, sc, gpu=True)
        gpu_count_s += time.perf_counter() - t
        if rc:
            raise RuntimeError(f'wex_nr2_count_gpu({index}) = {rc}')
        if n % 172 == 0:
            log(stage='count', experts=n, seconds=round(time.perf_counter() - t1, 1))
    assert int(wc.sum()) == sum(2 * e[L.F_WEIGHT] for _, e in present)
    assert int(sc.sum()) == sum(e[L.F_SCALE] for _, e in present)
    blob = tools.build_model(wc, sc)
    model_id = tools.model_id(blob)
    assert model_id is not None and model_id == L.fnv1a(blob)
    model_path.write_bytes(blob)
    np.savez_compressed(L.HERE / f'counts-{suffix}.npz', weight=wc.reshape(3, 567, 16), scale=sc.reshape(3, 256))
    training_s = time.perf_counter() - t1
    log(stage='model', model_id=f'0x{model_id:08x}', seconds=round(training_s, 1))

    # ---- pass 2: GPU encode, CPU reference verify, write -------------------
    t2 = time.perf_counter()
    table_bytes = count * L.ENTRY.size
    records_start = TABLE_OFFSET + L.align(table_bytes)
    new_entries = [bytes(L.ENTRY.size)] * count
    cursor = records_start
    stats = {'header': 0, 'lengths': 0, 'scale_payload': 0, 'weight_payload': 0, 'padding': 0, 'total': 0}
    symbols = scales = 0
    encoded = fallback = crosschecked = verified = 0
    mapped_total = 0
    gpu_encode_s = 0.0
    per_record, record_digests = [], []
    by_category = {}
    pool = concurrent.futures.ThreadPoolExecutor(max(1, args.verify_workers))
    pending = deque()
    out_file = open(partial, 'xb')
    try:
        out_file.write(bytes(records_start))     # header page, model, table: written last

        def finish(item):
            nonlocal cursor, verified, crosschecked
            index, e, record, new, verify_future, cross_future = item
            if verify_future is not None:
                rc = verify_future.result()
                if rc != 0:
                    raise RuntimeError(f'wex_nr2_verify({index}) = {rc}')
                verified += 1
            if cross_future is not None:
                rc, cpu_record, _ = cross_future.result()
                if rc != 0 or cpu_record != bytes(memoryview(record)[:len(cpu_record)]) or \
                        any(memoryview(record)[len(cpu_record):]):
                    raise RuntimeError(f'GPU record of {index} differs from wex_nr2_encode')
                crosschecked += 1
            assert new[L.F_OFFSET] == cursor and out_file.tell() == cursor
            out_file.write(record)
            record_digests.append(hashlib.sha256(record).digest())
            cursor += len(record)

        for n, (index, e, raw) in enumerate(read_experts(SOURCE, present), 1):
            eb = L.ENTRY.pack(*e)
            raw_bytes = L.raw_size(e)
            capacity = raw_bytes
            out = bytearray(capacity)
            t = time.perf_counter()
            rc, _, written = tools.encode(blob, raw, eb, cz, args.weight_chunk, args.scale_chunk, capacity,
                                          gpu=True, out=out)
            gpu_encode_s += time.perf_counter() - t
            expected = raw
            if cz:
                expected, mapped = tools.canonical(eb, raw)
                mapped_total += mapped
            padded = L.align(written)
            new = list(e)
            new[L.F_OFFSET] = cursor + sum(len(p[2]) for p in pending)
            verify_future = cross_future = None
            if rc == 0 and padded < raw_bytes:
                out[written:padded] = bytes(padded - written)
                record = out[:padded]
                new[L.F_TOTAL], new[L.F_FLAGS] = padded, L.WEX_ENTRY_NEURAL_RANS2
                new_eb = L.ENTRY.pack(*new)
                verify_future = pool.submit(tools.verify, blob, new_eb, record, expected)
                if args.crosscheck_every and (n - 1) % args.crosscheck_every == 0:
                    cross_future = pool.submit(tools.encode, blob, raw, eb, cz, args.weight_chunk,
                                               args.scale_chunk, capacity, False)
                sp, wp = struct.unpack_from('<II', record, 8)
                lengths = written - 32 - sp - wp
                for k, v in (('header', 32), ('lengths', lengths), ('scale_payload', sp),
                             ('weight_payload', wp), ('padding', padded - written), ('total', padded)):
                    stats[k] += v
                symbols += 2 * e[L.F_WEIGHT]
                scales += e[L.F_SCALE]
                encoded += 1
            elif rc == 0 or (rc == -1 and written > capacity):
                record = raw                              # not smaller: the raw slab, flags 0
                new[L.F_TOTAL], new[L.F_FLAGS] = raw_bytes, 0
                new_eb = L.ENTRY.pack(*new)
                sp = wp = None
                fallback += 1
            else:
                raise RuntimeError(f'wex_nr2_encode_gpu({index}) = {rc}')
            new_entries[index] = new_eb
            category = categories.get(index, 'unknown')
            c = by_category.setdefault(category, {'experts': 0, 'stored_bytes': 0})
            c['experts'] += 1; c['stored_bytes'] += len(record)
            per_record.append([index, category, len(record), new[L.F_FLAGS], sp, wp])
            pending.append((index, e, record, new, verify_future, cross_future))
            while pending and (len(pending) > 2 * args.verify_workers or
                               (pending[0][4] is None or pending[0][4].done()) and
                               (pending[0][5] is None or pending[0][5].done())):
                finish(pending.popleft())
            if n % 86 == 0:
                log(stage='encode', experts=n, encoded=encoded, fallback=fallback,
                    avg_record=round(stats['total'] / max(encoded, 1)), seconds=round(time.perf_counter() - t2, 1))
        while pending:
            finish(pending.popleft())
        assert out_file.tell() == cursor
        # header page + model + table
        page = bytearray(L.WEX_ALIGN)
        L.HEADER.pack_into(page, 0, L.WEX_MAGIC_V3, 3, layers, experts, expert_bytes, TABLE_OFFSET, FLAGS, 0)
        page[40:56] = uuid.uuid4().bytes
        struct.pack_into('<IIQII', page, 64, L.WNMD_MAGIC, L.MODEL_BYTES, MODEL_OFFSET, model_id, 0)
        table = b''.join(new_entries)
        out_file.seek(0)
        out_file.write(page)
        out_file.write(blob + bytes(TABLE_OFFSET - MODEL_OFFSET - len(blob)))
        out_file.write(table + bytes(L.align(table_bytes) - table_bytes))
        out_file.flush()
        os.fsync(out_file.fileno())
        out_file.close()
    except BaseException:
        out_file.close()
        pool.shutdown(cancel_futures=True)
        partial.unlink(missing_ok=True)
        raise
    pool.shutdown()
    encoding_s = time.perf_counter() - t2
    log(stage='written', bytes=cursor, encoded=encoded, fallback=fallback, seconds=round(encoding_s, 1))

    # ---- validation of the finished file ------------------------------------
    t3 = time.perf_counter()
    file_bytes = partial.stat().st_size
    assert file_bytes == cursor
    with open(partial, 'rb') as f:
        disk_page = f.read(L.WEX_ALIGN)
        pre_table = f.read(TABLE_OFFSET - L.WEX_ALIGN)
        disk_table = f.read(L.align(table_bytes))
    assert disk_page == bytes(page) and disk_table[:table_bytes] == table
    rc, checked_id = tools.check_container(disk_page, pre_table, disk_table, file_bytes)
    if rc != 0 or checked_id != model_id:
        raise RuntimeError(f'wex_nr2_check_container = {rc} (id 0x{checked_id:08x})')
    # Engine slab path on records read back from disk: every 43rd, the last, all fallbacks.
    sample = set(range(0, len(present), 43)) | {len(present) - 1} | \
        {k for k, r in enumerate(per_record) if r[3] == 0}
    readback = 0
    with open(partial, 'rb') as f, open(SOURCE, 'rb') as src:
        for k in sorted(sample):
            index, e = present[k]
            new = L.ENTRY.unpack(new_entries[index])
            f.seek(new[L.F_OFFSET]); record = f.read(new[L.F_TOTAL])
            src.seek(e[L.F_OFFSET]); raw = src.read(L.raw_size(e))
            expected = tools.canonical(new_entries[index], raw)[0] if cz else raw
            if new[L.F_FLAGS] == L.WEX_ENTRY_NEURAL_RANS2:
                rc = tools.verify_slab(blob, new_entries[index], record, expected)
                if rc:
                    raise RuntimeError(f'slab-path readback of {index} = {rc}')
            elif record != raw:
                raise RuntimeError(f'raw readback of {index} differs')
            readback += 1
    os.replace(partial, final)
    # Whole-file hash; every on-disk record re-hashed against the verified bytes.
    h = hashlib.sha256()
    with open(final, 'rb', buffering=0) as f:
        head = f.read(records_start)
        h.update(head)
        for k, (index, _) in enumerate(present):
            data = f.read(L.ENTRY.unpack(new_entries[index])[L.F_TOTAL])
            h.update(data)
            if hashlib.sha256(data).digest() != record_digests[k]:
                raise RuntimeError(f'on-disk record {index} differs from the verified record')
        assert f.read(1) == b''
    digest = h.hexdigest()
    # Mirror: a hard link on the same volume is the identical file.
    MIRROR.mkdir(parents=True, exist_ok=True)
    os.link(final, mirror)
    assert os.path.samefile(final, mirror) and os.stat(final).st_nlink >= 2
    with open(mirror, 'rb') as f:
        assert f.read(L.WEX_ALIGN) == bytes(page)
        f.seek(TABLE_OFFSET)
        assert f.read(L.align(table_bytes)) == disk_table
    validation_s = time.perf_counter() - t3
    tools.release()

    v1_bytes, neural_bytes = SOURCE.stat().st_size, NEURAL.stat().st_size
    records_bytes = file_bytes - records_start
    v1_records = v1_bytes - (src_table_offset + L.align(table_bytes))
    neural_records = None
    if NEURAL.exists():
        nh, _, _, nentries = L.read_container(NEURAL)
        neural_records = sum(x[L.F_TOTAL] for x in nentries)
    avg = {k: v / encoded for k, v in stats.items()} if encoded else {}
    report = {
        'container': str(final), 'mirror': str(mirror), 'mirror_hardlink': True,
        'source': str(SOURCE), 'canonical_zero': bool(cz),
        'canonical_zero_note': 'FP4 code 8 (-0) stored as 0 (+0): value-exact, not byte-exact; '
                               'verified against the canonicalized raw slab' if cz else None,
        'file_bytes': file_bytes, 'sha256': digest,
        'header': {'magic': 'WEX3', 'version': 3, 'num_layers': layers, 'num_experts': experts,
                   'aligned_expert_bytes': expert_bytes, 'table_offset': TABLE_OFFSET, 'flags': FLAGS,
                   'flags_names': 'SCALE_TILES|DEDUP|NEURAL_RANS2', 'uuid': bytes(page[40:56]).hex(),
                   'model_descriptor': {'at': 64, 'magic': 'WNMD', 'bytes': L.MODEL_BYTES, 'offset': MODEL_OFFSET,
                                        'model_id': f'0x{model_id:08x}'},
                   'records_start': records_start},
        'model': {'id': f'0x{model_id:08x}', 'id_decimal': model_id, 'bytes': L.MODEL_BYTES,
                  'sha256': hashlib.sha256(blob).hexdigest(), 'copy': str(model_path),
                  'trained_on_experts': len(present)},
        'weight_chunk': args.weight_chunk, 'scale_chunk': args.scale_chunk,
        'experts': {'present': len(present), 'encoded_wnr2': encoded, 'fallback_raw': fallback},
        'average_record_bytes_wnr2': avg,
        'raw_record_bytes': L.raw_size(present[0][1]),
        'bits_per_weight': 8 * stats['weight_payload'] / symbols if symbols else None,
        'bits_per_scale': 8 * stats['scale_payload'] / scales if scales else None,
        'bits_per_weight_all_in': 8 * stats['total'] / symbols if symbols else None,
        'negative_zero_codes_mapped': mapped_total if cz else 0,
        'comparison': {
            'pilot-v1.wex': {'file_bytes': v1_bytes, 'records_bytes': v1_records,
                             'file_reduction_percent': 100 * (1 - file_bytes / v1_bytes),
                             'records_reduction_percent': 100 * (1 - records_bytes / v1_records)},
            'pilot-v3-neural512.wex': {'file_bytes': neural_bytes, 'records_bytes': neural_records,
                                       'file_reduction_percent': 100 * (1 - file_bytes / neural_bytes),
                                       'records_reduction_percent': 100 * (1 - records_bytes / neural_records)
                                       if neural_records else None},
            'this_records_bytes': records_bytes,
        },
        'by_category_average_stored_bytes': {k: v['stored_bytes'] / v['experts'] for k, v in by_category.items()},
        'seconds': {'training': training_s, 'training_gpu_count_calls': gpu_count_s,
                    'encoding': encoding_s, 'encoding_gpu_encode_calls': gpu_encode_s,
                    'validation_readback_hash_mirror': validation_s, 'total': time.perf_counter() - began},
        'verification': {
            'records_decoded_by_cpu_reference': verified,
            'all_records_exact': verified == encoded,
            'gpu_records_equal_to_cpu_encoder': crosschecked,
            'container_check': 'wex_validate_header + wex_validate_table + wex_nr2_model_location + '
                               'wex_nr2_model_open: ok, id matches descriptor',
            'slab_path_readback_from_disk': readback,
            'on_disk_records_rehashed_equal_to_verified': len(record_digests),
        },
        'verify_workers': args.verify_workers,
        'records': {'columns': ['index', 'category', 'stored_bytes', 'flags', 'scale_payload', 'weight_payload'],
                    'rows': per_record},
    }
    report_path.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != 'records'}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
