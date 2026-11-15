# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Seed corpora for the fuzzing harnesses, from the conformance vectors.

    python -I wex4/fuzz/make_seeds.py OUT_DIR [LEGACY_SEEDS]

OUT_DIR/file gets every file of wex4/vectors/MANIFEST.tsv (valid, invalid, legacy and the review findings): the
seeds of fuzz_wex4_file and fuzz_wex4_verify. OUT_DIR/wnr3 gets the stored bytes of every WNR3 record of the
accepted vectors: the seeds of fuzz_wnr3_record. OUT_DIR/engine gets small expert containers for the two engine
harnesses (fuzz_store_wex reads it, fuzz_q38_wex reads it and OUT_DIR/file):

  - a WEX1 container made here, 1 layer x 2 experts, raw records of 4096 bytes (hidden 64, intermediate 32);
  - a WEX2 container made here, 1 x 1, its scales stored as tiles of 0 to 3 bits (the smallest such record:
    8192 scale bytes, 131072 weight bytes);
  - the legacy vector tiny2-cz.wex3 (WNR3 records, canonical zero; 6 x 8 with two experts present), and the
    same file with its grid cut to 1 x 2, where every expert is present, as the Qwen3.8 reader requires;
  - the WEX4 form of the WEX1 and of the 1 x 2 WNR3 container (wex4_convert.upgrade).

OUT_DIR/legacy gets WEX3 containers with the older record codecs, for fuzz_store_wex alone (the Qwen3.8 reader
takes none of them). All three hold the same four experts of one layer, made here (hidden 128, intermediate
128: 1536 scale bytes and 24576 weight bytes, a slab of 28672): two different ones, a third with a scale of
255, and a copy of the first.

  - palette (codec 2): c/tools/repack_v4_wex.py writes it from a safetensors checkpoint of those experts
    (format 3, palette3 weights, any error allowed). Two palette records, the expert with the scale of 255
    stored raw (the writer does not quantize it), the copy an alias of the first record. The random build id
    the writer puts at bytes 40..55 is set to zero, so the seed is the same file on every run.
  - WNR (codec 3) and WNR2 (codec 4): written by LEGACY_SEEDS, the program built from legacy_seeds.c, from the
    WEX1 container repack_v4_wex.py writes of the same checkpoint. Three coded records and an alias each.
    Without the second argument these two are not made. The WNR container is also read back here with the
    Python decoder of c/tools/wex_neural_rans.py, which shares no code with the encoder: every record must
    decode to the weights of its expert.

A seed is named by the first 16 hex digits of the SHA-256 of its bytes, so the same bytes are never stored
twice and the paths stay short (Windows opens no path longer than 260 characters by default). Standard library
only."""
import hashlib
import importlib.util
import io
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import wex4  # noqa: E402
import wex4_convert  # noqa: E402

VECTORS = os.path.join(HERE, '..', 'vectors')
PAGE = 4096
HEADER = struct.Struct('<IIIIQQII')        # legacy header: magic, version, layers, experts, slab, table offset, flags, 0
ENTRY = struct.Struct('<QIII6H6H3I3II')    # legacy entry, 72 B


def put(folder, data):
    with open(os.path.join(folder, hashlib.sha256(data).hexdigest()[:16]), 'wb') as f:
        f.write(data)


def pad(data):
    return data + bytes(-len(data) % PAGE)


def legacy(version, experts, rows_cols, tiles):
    """A WEX1 (raw records) or WEX2 (scales as tiles) container of one layer. rows_cols: for w1, w2, w3 the
    rows and the weight bytes per row; a row has one scale per 16 weight bytes."""
    rng = random.Random(20261114 + version)
    scale_shapes = [(r, c // 16) for r, c in rows_cols]
    scale_bytes = sum(r * c for r, c in scale_shapes)
    weight_bytes = sum(r * c for r, c in rows_cols)
    slab = len(pad(bytes(scale_bytes + weight_bytes)))
    scale_offsets, weight_offsets, at = [], [], 0
    for r, c in scale_shapes:
        scale_offsets.append(at)
        at += r * c
    for r, c in rows_cols:
        weight_offsets.append(at)
        at += r * c
    shapes = [x for shape in scale_shapes for x in shape] + [x for shape in rows_cols for x in shape]
    table, records, offset = b'', b'', 2 * PAGE
    for _ in range(experts):
        weights = bytes(rng.randrange(256) for _ in range(weight_bytes))
        if tiles:   # one page: "WCS1", the decoded size, the stream size, then per 256 scales a base, a bit width and the bits
            stream = b''
            for t in range(scale_bytes // 256):
                bits = t % 4
                stream += bytes((118 + t % 5, bits)) + bytes(rng.randrange(256) for _ in range(32 * bits))
            record = pad(struct.pack('<III', 0x31534357, scale_bytes, len(stream)) + stream) + pad(weights)
            flags = 1 << 8 | 1   # one page of scale stream, codec 1
        else:
            record = pad(bytes(118 + rng.randrange(8) for _ in range(scale_bytes)) + weights)
            flags = 0
        table += ENTRY.pack(offset, len(record), scale_bytes, weight_bytes, *shapes, *scale_offsets, *weight_offsets, flags)
        records += record
        offset += len(record)
    magic = 0x57455830 + version
    return pad(HEADER.pack(magic, version, 1, experts, slab, PAGE, 1 if tiles else 0, 0)) + pad(table) + records


def upgraded(folder, data):
    """The WEX4 form of a WEX1 or WNR3 container (in place, on a temporary copy)."""
    path = os.path.join(folder, 'upgrade.tmp')
    with open(path, 'wb') as f:
        f.write(data)
    wex4_convert.upgrade(path, family='qwen38')
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


def checkpoint(folder):
    """A safetensors checkpoint of one layer of four MXFP4 experts in `folder`, under the DeepSeek names. FP4
    codes and scales are skewed, as in fixtures/make_synth_mxfp4.py, so that the entropy coders gain something.
    The result is the slab of each expert: the scales of w1, w2, w3, then their weights."""
    rng = random.Random(20261115)
    hidden = inter = 128
    codes = (40, 22, 12, 6, 3, 2, 1, 1, 30, 18, 10, 5, 3, 2, 1, 1)
    slabs = []
    for expert in range(3):
        scales, weights = [], []
        for rows, cols in ((inter, hidden // 2), (hidden, inter // 2), (inter, hidden // 2)):   # weight bytes
            s = bytearray(rng.choices(range(114, 128), (1, 1, 2, 3, 5, 8, 12, 16, 20, 16, 10, 5, 2, 1), k=rows * cols // 16))
            if expert == 2:
                s[rng.randrange(len(s))] = 255   # a non-finite scale: the palette writer keeps this expert raw
            low = rng.choices(range(16), codes, k=rows * cols)
            high = rng.choices(range(16), codes, k=rows * cols)
            scales.append(bytes(s))
            weights.append(bytes(a | b << 4 for a, b in zip(low, high)))
        slabs.append((scales, weights))
    slabs.append(slabs[0])
    header, payload = {}, bytearray()
    for expert, (scales, weights) in enumerate(slabs):
        for m, (rows, cols) in enumerate(((inter, hidden // 2), (hidden, inter // 2), (inter, hidden // 2))):
            for suffix, dtype, shape, data in (('scale', 'F8_E8M0', (rows, cols // 16), scales[m]),
                                               ('weight', 'I8', (rows, cols), weights[m])):
                header[f'layers.0.ffn.experts.{expert}.w{m + 1}.{suffix}'] = {
                    'dtype': dtype, 'shape': list(shape), 'data_offsets': [len(payload), len(payload) + len(data)]}
                payload += data
    raw = json.dumps(header, separators=(',', ':')).encode()
    with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'num_hidden_layers': 1, 'n_routed_experts': 4, 'hidden_size': hidden,
                   'moe_intermediate_size': inter}, f)
    with open(os.path.join(folder, 'model.safetensors'), 'wb') as f:
        f.write(struct.pack('<Q', len(raw)) + raw + payload)
    return [b''.join(scales) + b''.join(weights) for scales, weights in slabs]


def legacy_codecs(folder, tool):
    """The containers of OUT_DIR/legacy (see the module text); the result is how many were written."""
    spec = importlib.util.spec_from_file_location('repack_v4_wex', os.path.join(HERE, '..', '..', 'c', 'tools', 'repack_v4_wex.py'))
    repack = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = repack   # its dataclass looks its module up
    spec.loader.exec_module(repack)
    made = 0
    with tempfile.TemporaryDirectory(dir=folder) as work:
        slabs = checkpoint(work)
        path = os.path.join(work, 'p.wex')
        result = repack.repack(work, path, format_version=3, weight_codec='palette3', max_weight_relative_error=1.0)
        assert result['palette3_experts'] == 3 and result['deduplicated_experts'] == 1, result
        repack.validate_container(path, verify_payload=True)
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        data[40:56] = bytes(16)   # the random build id
        put(folder, bytes(data))
        made += 1
        if tool:
            source = os.path.join(work, 's.wex')
            repack.repack(work, source, format_version=1)
            for kind in ('wnr', 'wnr2'):
                path = os.path.join(work, kind + '.wex')
                subprocess.run([tool, source, kind, path], check=True, stdout=subprocess.DEVNULL)
                if kind == 'wnr':   # the independent decoder
                    entries = repack.validate_container(path, verify_payload=True)['entries']
                    assert sum(entry[22] == repack.WEX_NEURAL_RANS for entry in entries) == 4, entries
                    with open(path, 'rb') as f:
                        data = f.read()
                    for slab, entry in zip(slabs, entries):
                        offset, total, scale_bytes, weight_bytes = entry[:4]
                        model, offsets, scale_stream, payload, chunk = repack.neural_rans.parse_record(
                            data[offset:offset + total], weight_bytes)
                        scales = repack.unpack_scale_tiles(scale_stream, scale_bytes)
                        weights = repack.neural_rans.decode_weights(model, offsets, payload, scales, weight_bytes, chunk)
                        assert scales + weights == slab, 'a WNR record does not decode to its expert'
                with open(path, 'rb') as f:
                    put(folder, f.read())
                made += 1
    return made


def main():
    out = sys.argv[1]
    tool = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    files, records, engine = os.path.join(out, 'file'), os.path.join(out, 'wnr3'), os.path.join(out, 'engine')
    legacy_folder = os.path.join(out, 'legacy')
    os.makedirs(files, exist_ok=True)
    os.makedirs(records, exist_ok=True)
    os.makedirs(engine, exist_ok=True)
    os.makedirs(legacy_folder, exist_ok=True)
    made = legacy_codecs(legacy_folder, tool)
    wex1 = legacy(1, 2, ((32, 32), (64, 16), (32, 32)), False)
    tiny2 = open(os.path.join(VECTORS, 'legacy', 'tiny2-cz.wex3'), 'rb').read()
    # its two experts are (0, 0) and (0, 1): a 1 x 2 grid has the same table page and the same record owners
    tiny2_1x2 = tiny2[:8] + struct.pack('<II', 1, 2) + tiny2[16:]
    for data in (wex1, legacy(2, 1, ((64, 512), (64, 512), (64, 1024)), True), tiny2, tiny2_1x2,
                 upgraded(engine, wex1), upgraded(engine, tiny2_1x2)):
        put(engine, data)
    rows = [line.split('\t') for line in open(os.path.join(VECTORS, 'MANIFEST.tsv'), encoding='utf-8').read().splitlines()
            if line and not line.startswith('#')]
    for path, verdict in ((row[0], row[1]) for row in rows):
        data = open(os.path.join(VECTORS, path), 'rb').read()
        put(files, data)
        if verdict != 'accept':
            continue
        reader = wex4.Reader(io.BytesIO(data))
        for record in reader.records:
            if record[3] == 5:   # codec 5: WNR3
                put(records, data[record[0]:record[0] + record[1]])
    print(f'seeds: {len(os.listdir(files))} files in {files}, {len(os.listdir(records))} WNR3 records in {records}, '
          f'{len(os.listdir(engine))} expert containers in {engine}, {made} containers of the older codecs in '
          f'{legacy_folder}' + ('' if tool else ' (palette only: no legacy_seeds program was given)'))


if __name__ == '__main__':
    main()
