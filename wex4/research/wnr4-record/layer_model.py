# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Per-layer WNM2 model instead of one model per container: real record bytes, held-out experts.

  python -I layer_model.py CONTAINER wex_nr3_tools.dll OUT.json CACHE_DIR [layers]

For each layer: 4 training experts, 2 test experts (never trained on). Each test expert is encoded as a
WNR3-CZ record (the unchanged production codec, wex_nr3_encode_record) with
  prod    the container's own model (must reproduce the stored record size),
  layer   a model trained on that layer's 4 training experts,
  pooled  one model trained on the training experts of all tested layers (control: retraining alone),
and every record is decoded back with the CPU reference and compared (wex_nr3_verify).
The container is only read: 6 records per layer, kept in CACHE_DIR so that a rerun reads nothing.
"""
import ctypes as C, json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
from fetch import ENTRY, open_container

WCOUNTS, SCOUNTS, MODEL_BYTES = 3 * 567 * 16, 3 * 256, 59408


def main():
    path, dll_path, out_path, cache = sys.argv[1:5]
    layers = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else [0, 8, 16, 24, 32, 38, 39]
    f, page, prod, table, _, experts, read = open_container(path)
    dll = C.CDLL(dll_path)
    dll.wex_nr3_encode_record.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_uint32, C.c_uint32, C.c_uint32,
                                          C.c_uint32, C.c_char_p, C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    pooled_w, pooled_s = (C.c_uint64 * WCOUNTS)(), (C.c_uint64 * SCOUNTS)()
    tests, models = [], {}

    def load(layer, expert):
        nonlocal read
        entry = table[(layer * experts + expert) * ENTRY.size:][:ENTRY.size]
        e = ENTRY.unpack(entry)
        cached = os.path.join(cache, f'rec_{layer:02d}_{expert:03d}.bin')
        if os.path.exists(cached):
            record = open(cached, 'rb').read()
        else:
            f.seek(e[0]); record = f.read(e[1]); read += len(record)
            open(cached, 'wb').write(record)
        raw = e[2] + e[3]
        slab, coded, stats = C.create_string_buffer(raw), C.create_string_buffer(raw), (C.c_uint32 * 16)()
        assert dll.wex_nr3_decode(prod, entry, record, layer, expert, slab) == 0
        flags = dll.wex_nr3_prepare(entry, slab, 1, coded, stats)
        assert flags == record[27]
        wchunk, schunk = struct.unpack_from('<2I', record, 16)
        return entry, record, slab, coded, flags, wchunk, schunk

    def encode(blob, t):
        layer, expert, entry, record, slab, coded, flags, wchunk, schunk = t
        cap = len(slab) * 2
        out, used, total = C.create_string_buffer(cap), C.c_size_t(), C.c_size_t()
        assert dll.wex_nr3_encode_record(blob, entry, coded, flags, layer << 16 | expert, wchunk, schunk,
                                         out, cap, C.byref(used), C.byref(total)) == 0
        sized = entry[:8] + struct.pack('<I', total.value) + entry[12:]   # the entry of the new record
        assert dll.wex_nr3_verify(blob, sized, out, layer, expert, slab) == 0   # decodes to Z(source)
        return used.value, out.raw[:total.value]

    for layer in layers:
        base = (layer * 37 + 11) % experts
        wc, sc = (C.c_uint64 * WCOUNTS)(), (C.c_uint64 * SCOUNTS)()
        for k in range(1, 5):
            entry, _, _, coded, _, _, _ = load(layer, (base + 40 * k) % experts)
            assert dll.wex_nr3_count_expert(entry, coded, wc, sc) == 0
            assert dll.wex_nr3_count_expert(entry, coded, pooled_w, pooled_s) == 0
        blob, ident = C.create_string_buffer(MODEL_BYTES), C.c_uint32()
        assert dll.wex_nr3_build_model(wc, sc, blob, C.byref(ident)) == 0
        models[layer] = blob
        for expert in (base, (base + 200) % experts):
            tests.append((layer, expert) + load(layer, expert))
        print('layer', layer, 'loaded', flush=True)
    pooled, ident = C.create_string_buffer(MODEL_BYTES), C.c_uint32()
    assert dll.wex_nr3_build_model(pooled_w, pooled_s, pooled, C.byref(ident)) == 0
    rows = []
    for t in tests:
        layer, expert, record = t[0], t[1], t[3]
        used_prod, bytes_prod = encode(prod, t)
        assert bytes_prod == record, 'production model does not reproduce the stored record'
        used_layer, _ = encode(models[layer], t)
        used_pooled, _ = encode(pooled, t)
        sp_prod = struct.unpack_from('<I', record, 8)[0]
        rows.append({'layer': layer, 'expert': expert, 'prod_used': used_prod, 'layer_used': used_layer,
                     'pooled_used': used_pooled, 'prod_scale_payload': sp_prod})
        print(rows[-1], flush=True)
    json.dump({'train_experts_per_layer': 4, 'layers': layers, 'rows': rows, 'container_bytes_read': read},
              open(out_path, 'w'), indent=1)
    print('bytes read from the container:', read)


if __name__ == '__main__':
    main()
