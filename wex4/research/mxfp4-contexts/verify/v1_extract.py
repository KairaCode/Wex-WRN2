# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Verifier step 1 (read-only on the container): population check of the sample, slab re-extraction check,
real reference-decoder timing, and 8 NEW experts from layers the experiment never touched.

  python -I v1_extract.py CONTAINER.wex ..\\out out
"""
import ctypes as C
import json
import os
import random
import struct
import sys
import time

HEADER = struct.Struct('<IIIIQQII')
ENTRY = struct.Struct('<QIII6H6H3I3II')
NEW_LAYERS = [2, 8, 13, 19, 25, 30, 36, 38]      # none of 0,5,11,16,22,28,33,39
SEED = 777


def main():
    container, theirs, out = sys.argv[1], sys.argv[2], sys.argv[3]
    lib = C.CDLL(os.path.abspath(os.path.join(out, 'wex_nr3_tools.dll')))
    P = C.c_void_p
    lib.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]
    lib.wex_nr3_decode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
    man = json.load(open(os.path.join(theirs, 'manifest.json')))
    os.makedirs(os.path.join(out, 'newslabs'), exist_ok=True)
    res = {}
    with open(container, 'rb') as f:
        page = f.read(4096)
        magic, version, layers, experts, _, table_offset, flags, _ = HEADER.unpack_from(page)
        wnmd, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
        f.seek(model_offset)
        blob = f.read(model_bytes)
        assert blob == open(os.path.join(theirs, 'prod_model.wnm2'), 'rb').read(), 'their model copy differs'
        f.seek(table_offset)
        table = f.read(layers * experts * ENTRY.size)
        # --- population: every record header (32 B) ---
        t = time.time()
        tot = wp = sp = 0
        codecs = {}
        per_layer = [0] * layers
        for i in range(layers * experts):
            ent = ENTRY.unpack_from(table, i * ENTRY.size)
            codecs[ent[-1]] = codecs.get(ent[-1], 0) + 1
            f.seek(ent[0])
            h = struct.unpack('<8I', f.read(32))
            assert h[0] == 0x33524e57 and h[4] == 2048 and h[5] == 512
            tot += ent[1]; sp += h[2]; wp += h[3]
            per_layer[i // experts] += h[3]
        n = layers * experts
        raw = 18800640
        W = 35389440
        res['population'] = {'records': n, 'codecs': codecs, 'mean_record_total_bytes': tot / n,
                             'mean_record_ratio_of_raw': tot / n / raw,
                             'mean_weight_payload_bytes': wp / n, 'real_bpw_weights_payload': wp * 8 / (n * W),
                             'mean_scale_payload_bytes': sp / n,
                             'per_layer_real_bpw': [x * 8 / (experts * W) for x in per_layer],
                             'seconds': time.time() - t}
        srows = man['rows']
        res['sample'] = {'mean_record_total_bytes': sum(r['record_total_bytes'] for r in srows) / 24,
                         'real_bpw_weights_payload': sum(r['weight_payload_bytes'] for r in srows) * 8 / (24 * W)}
        sl = sorted({r['layer'] for r in srows})
        res['sample']['population_bpw_of_sampled_layers'] = sum(per_layer[L] for L in sl) * 8 / (len(sl) * experts * W)
        print(json.dumps(res, indent=1), flush=True)

        def get(L, e):
            eb = table[(L * experts + e) * ENTRY.size:][:ENTRY.size]
            ent = ENTRY.unpack(eb)
            f.seek(ent[0])
            record = f.read(ent[1])
            raw = ent[2] + ent[3]
            slab = C.create_string_buffer(raw)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                assert lib.wex_nr3_decode(blob, eb, record, L, e, slab) == 0
                best = min(best, time.perf_counter() - t)
            coded = C.create_string_buffer(raw)
            stats = (C.c_uint32 * 16)()
            fl = lib.wex_nr3_prepare(eb, slab, 1, coded, stats)
            assert fl == record[27] and bytes(stats[9:12]) == record[24:27]
            h = struct.unpack_from('<8I', record)
            return eb, record, coded.raw[:raw], h, best

        # --- their slabs vs an independent re-extraction; real reference decode time ---
        res['recheck'] = []
        for idx in (2, 10, 21):
            r = srows[idx]
            eb, record, coded, h, best = get(r['layer'], r['expert'])
            same = coded == open(os.path.join(theirs, 'slabs', r['file']), 'rb').read()
            res['recheck'].append({'file': r['file'], 'slab_identical': same, 'entry_identical': eb.hex() == r['entry_hex'],
                                   'weight_payload_bytes': h[3], 'manifest_weight_payload': r['weight_payload_bytes'],
                                   'wex_nr3_decode_ms_best_of_3': best * 1e3})
            print(res['recheck'][-1], flush=True)
        # --- new experts ---
        rows = []
        rng = random.Random(SEED)
        for L in NEW_LAYERS:
            e = rng.randrange(experts)
            eb, record, coded, h, best = get(L, e)
            name = f'L{L:02d}_E{e:03d}.coded'
            open(os.path.join(out, 'newslabs', name), 'wb').write(coded)
            rows.append({'layer': L, 'expert': e, 'file': name, 'entry_hex': eb.hex(), 'raw_bytes': len(coded),
                         'record_total_bytes': len(record), 'scale_payload_bytes': h[2], 'weight_payload_bytes': h[3],
                         'anchors': list(record[24:27]), 'flags': record[27], 'wex_nr3_decode_ms_best_of_3': best * 1e3})
            print(name, h[3], f'{best * 1e3:.0f} ms', flush=True)
        res['new_rows'] = rows
    json.dump(res, open(os.path.join(out, 'v1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
