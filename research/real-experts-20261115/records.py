# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Helpers of this run around the repository's tools. Standard library only.

  python -I records.py model NR3_DLL OUT.wnm2 PROFILE SRC.safetensors... [--layers L,L] [--cz]
      trains a WNM2 model on the expert slabs found in the safetensors files under the naming profile
      (all of them, or those of the layers named), with the training of c/tools/build_nr3.py (its train(),
      the counts and the model build of c/tools/wex_nr3_tools.c); prints the model id.
  python -I records.py extract CONTAINER.wex4 OUT_DIR [--dll wex4_tools_ref.dll]
      for every expert record of a WEX4 file: OUT_DIR/L<layer>_E<expert>.slab (the decoded slab, every hash
      checked; a WNR3 record is decoded by the DLL) and, for a WNR3 record, OUT_DIR/L<layer>_E<expert>.rec
      (the 72-byte legacy entry with codec 5, then the record bytes: the input of
      wex4/research/fast-cpu-decode and of bench_real.c) and OUT_DIR/model.wnm2. Prints one line per record and
      writes OUT_DIR/records.json (sizes, ratio, SHA-256 of the slab and of the stored record).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'wex4'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'c', 'tools'))
import build_nr3  # noqa: E402
import wex4  # noqa: E402
import wex4_convert  # noqa: E402


class Slabs:
    """The expert slabs of safetensors files, in the form build_nr3.train() reads (its Wex1.slab)."""

    def __init__(self, profile, sources):
        pattern, (_, matrices, weight_suffix) = wex4_convert.expert_pattern(profile), wex4_convert.PROFILES[profile][:3]
        self.parts = {}
        for path in sources:
            _, start, tensors = wex4_convert.st_index(path)
            for name, dtype, shape, begin, end in tensors:
                match = pattern.match(name)
                if match:
                    key = (int(match[1]), int(match[2]))
                    part = (matrices.index(match[3].decode()), match[4].decode() == weight_suffix)
                    self.parts.setdefault(key, {})[part] = (path, start + begin, end - begin, shape)
        for key, parts in self.parts.items():
            wex4.need(len(parts) == 6, f'L{key[0]} E{key[1]}: {len(parts)} of 6 tensors')

    def slab(self, layer, expert):
        parts = self.parts[(layer, expert)]
        order = [parts[(m, False)] for m in range(3)] + [parts[(m, True)] for m in range(3)]
        entry = wex4_convert.expert_entry([p[3] for p in order[:3]], [p[3] for p in order[3:]])
        data = b''
        for path, offset, size, _ in order:
            with open(path, 'rb') as f:
                f.seek(offset)
                data += f.read(size)
        return list(wex4_convert.ENTRY.unpack(entry)), data


def cmd_model(argv):
    dll, out, profile = argv[0], argv[1], argv[2]
    layers = None
    if '--layers' in argv:
        layers = {int(x) for x in argv[argv.index('--layers') + 1].split(',')}
    sources = [a for a in argv[3:] if a.endswith('.safetensors')]
    src = Slabs(profile, sources)
    pairs = sorted(k for k in src.parts if layers is None or k[0] in layers)
    blob, mid = build_nr3.train(build_nr3.Tools(os.path.abspath(dll)), src, pairs, '--cz' in argv)
    with open(out, 'wb') as f:
        f.write(blob)
    print(json.dumps({'model': os.path.basename(out), 'model_id': f'0x{mid:08x}', 'cz': '--cz' in argv,
                      'trained_on': [f'L{L}:E{e}' for L, e in pairs]}))


def cmd_extract(argv):
    container, out = argv[0], argv[1]
    clib = wex4.CLib(argv[argv.index('--dll') + 1]) if '--dll' in argv else None
    os.makedirs(out, exist_ok=True)
    rows = []
    with open(container, 'rb') as f:
        rd = wex4.Reader(f, clib)
        for r, (offset, stored, decoded, codec, flags, blob, owner, first, count, digest, _) in enumerate(rd.records):
            if not flags & wex4.REC_EXPERT:
                continue
            layer, expert = owner >> 16, owner & 0xffff
            name = f'L{layer}_E{expert}'
            slab = rd.record(r)
            with open(os.path.join(out, name + '.slab'), 'wb') as g:
                g.write(slab)
            shapes = [tuple(rd.tensors[first + k][3][:2]) for k in range(6)]
            row = {'record': name, 'codec': codec, 'decoded': decoded, 'stored': stored, 'ratio': round(stored / decoded, 6),
                   'sha256_slab': hashlib.sha256(slab).hexdigest(), 'sha256_stored': digest.hex(),
                   'scale_shapes': shapes[:3], 'weight_shapes': shapes[3:]}
            if codec == wex4.CODEC_WNR3:
                fields = list(wex4_convert.ENTRY.unpack(wex4_convert.expert_entry(shapes[:3], shapes[3:])))
                fields[1], fields[-1] = stored, wex4.CODEC_WNR3
                record = rd.stored(r)
                row['wnr3_flags'] = f'0x{record[27]:02x}'   # the flag byte of the record header: ceil mask, 8 = CZ
                with open(os.path.join(out, name + '.rec'), 'wb') as g:
                    g.write(wex4_convert.ENTRY.pack(*fields) + record)
                with open(os.path.join(out, 'model.wnm2'), 'wb') as g:
                    g.write(rd.blob(blob - 1))
            rows.append(row)
            print(f"{name} codec {codec} decoded {decoded} stored {stored} ratio {row['ratio']:.4f}"
                  f"{' flags ' + row['wnr3_flags'] if 'wnr3_flags' in row else ''}", flush=True)
    total = {'records': len(rows), 'decoded': sum(r['decoded'] for r in rows), 'stored': sum(r['stored'] for r in rows)}
    total['ratio'] = round(total['stored'] / total['decoded'], 6) if rows else None
    with open(os.path.join(out, 'records.json'), 'w', encoding='utf-8') as g:
        json.dump({'container_root': rd.root.hex(), 'total': total, 'rows': rows}, g, indent=1)
        g.write('\n')
    print(json.dumps(total))


if __name__ == '__main__':
    if len(sys.argv) < 4 or sys.argv[1] not in ('model', 'extract'):
        raise SystemExit(__doc__)
    (cmd_model if sys.argv[1] == 'model' else cmd_extract)(sys.argv[2:])
