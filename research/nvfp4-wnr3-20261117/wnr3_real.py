# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The repository's WNR3 encoder on the MXFP4 samples, and the GLM records of an existing container.

  python -I wnr3_real.py pack DATA_DIR WORK_DIR NR3_DLL REF_DLL CHECKPOINT PROFILE
      For one MXFP4 checkpoint of fetch.py: three WNM2 models with the training of c/tools/build_nr3.py
      (all 16 experts; fold A = the experts of layer bins 0, 2, 4, 6; fold B = bins 1, 3, 5, 7), each with and
      without canonical zero; wex4_convert.py pack --experts nr3 of the 16 experts with each model; every
      record decoded by the reference DLL and compared with the source slab (or Z(source) with CZ). The
      cross-validated size of an expert is its record under the model of the other fold. Writes
      WORK_DIR/<checkpoint>/wnr3.json and the .rec/.slab pairs of the in-sample models (input of bench_real.c).
  python -I wnr3_real.py container CONTAINER.wex4 REF_DLL DATA_DIR WORK_DIR
      The 16 GLM experts of fetch.py's glm pair in an existing WEX4 container: stored size, decoded slab
      compared with the slab of the glm-mxfp4 sample and with Z(that slab). Writes WORK_DIR/glm-container.json.
Standard library only (plus the repository's wex4 package).
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'wex4'))
sys.path.insert(0, os.path.join(ROOT, 'c', 'tools'))
sys.path.insert(0, HERE)
import build_nr3  # noqa: E402
import fetch  # noqa: E402
import wex4  # noqa: E402
import wex4_convert  # noqa: E402


def slabs(profile, path):
    pattern, (_, matrices, weight_suffix) = wex4_convert.expert_pattern(profile), wex4_convert.PROFILES[profile][:3]
    parts = {}
    _, start, tensors = wex4_convert.st_index(path)
    for name, dtype, shape, begin, end in tensors:
        m = pattern.match(name)
        if m:
            parts.setdefault((int(m[1]), int(m[2])), {})[(matrices.index(m[3].decode()), m[4].decode() == weight_suffix)] = \
                (start + begin, end - begin, shape)

    class Src:
        def slab(self, layer, expert):
            p = parts[(layer, expert)]
            order = [p[(m, False)] for m in range(3)] + [p[(m, True)] for m in range(3)]
            entry = wex4_convert.expert_entry([o[2] for o in order[:3]], [o[2] for o in order[3:]])
            data = b''
            with open(path, 'rb') as f:
                for off, size, _ in order:
                    f.seek(off)
                    data += f.read(size)
            return list(wex4_convert.ENTRY.unpack(entry)), data
    return Src(), sorted(parts)


def z_fast(slab, scale_bytes):
    """Z of WNR3-SPEC section 1: every weight nibble 8 (-0) replaced by 0, scales untouched."""
    t = bytes(((b & 0xF0) if b & 15 == 8 else b) for b in range(256))
    t = bytes(((x & 0x0F) if x >> 4 == 8 else x) for x in t)
    return slab[:scale_bytes] + slab[scale_bytes:].translate(t)


def cmd_pack(argv):
    data_dir, work, nr3, ref, ckpt, profile = argv
    out = os.path.join(work, ckpt)
    os.makedirs(out, exist_ok=True)
    src_path = os.path.join(data_dir, ckpt, 'experts.safetensors')
    src, keys = slabs(profile, src_path)
    pair = next(p for p in fetch.PAIRS.values() if ckpt in p)
    chosen = json.load(open(os.path.join(data_dir, 'fetch-log.json')))['pairs'][
        next(k for k, v in fetch.PAIRS.items() if v == pair)]['chosen']
    layers = sorted({L for L, _ in chosen})
    fold = {L: 'A' if i % 2 == 0 else 'B' for i, L in enumerate(layers)}
    tools = build_nr3.Tools(os.path.abspath(nr3))
    clib = wex4.CLib(ref)
    result = {'checkpoint': ckpt, 'profile': profile, 'folds': fold, 'models': {}, 'records': {}}
    for cz in (False, True):
        for model in ('all', 'A', 'B'):
            pairs = [k for k in keys if model == 'all' or fold[k[0]] == model]
            blob, mid = build_nr3.train(tools, src, pairs, cz)
            tag = f'{model}{"-cz" if cz else ""}'
            mpath = os.path.join(out, f'{tag}.wnm2')
            open(mpath, 'wb').write(blob)
            cpath = os.path.join(out, f'nr3-{tag}.wex4')
            cmd = [sys.executable, '-I', os.path.join(ROOT, 'wex4', 'wex4_convert.py'), 'pack', cpath, src_path,
                   '--experts', 'nr3', '--names', profile, '--nr3-dll', nr3, '--model', mpath, '--dll', ref]
            if cz:
                cmd.append('--cz')
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            result['models'][tag] = {'model_id': f'0x{mid:08x}', 'trained_on': [f'L{L}:E{e}' for L, e in pairs]}
            rows = {}
            with open(cpath, 'rb') as f:
                rd = wex4.Reader(f, clib)
                for r, rec in enumerate(rd.records):
                    offset, stored, decoded, codec, flags, bl, owner, first, count, digest, _ = rec
                    if not flags & wex4.REC_EXPERT:
                        continue
                    L, e = owner >> 16, owner & 0xffff
                    got = rd.record(r)
                    entry, slab = src.slab(L, e)
                    want = z_fast(slab, entry[2]) if cz else slab
                    assert got == want, f'{ckpt} {tag} L{L} E{e}: decoded slab differs'
                    rows[f'L{L}_E{e}'] = {'codec': codec, 'stored': stored, 'decoded': decoded,
                                          'sha256_slab': hashlib.sha256(slab).hexdigest()}
                    if model == 'all':   # .rec/.slab for bench_real.c
                        d = os.path.join(out, f'rec-{tag}')
                        os.makedirs(d, exist_ok=True)
                        shapes = [tuple(rd.tensors[first + k][3][:2]) for k in range(6)]
                        fields = list(wex4_convert.ENTRY.unpack(wex4_convert.expert_entry(shapes[:3], shapes[3:])))
                        fields[1], fields[-1] = stored, wex4.CODEC_WNR3
                        if codec == wex4.CODEC_WNR3:
                            open(os.path.join(d, f'L{L}_E{e}.rec'), 'wb').write(wex4_convert.ENTRY.pack(*fields) + rd.stored(r))
                            open(os.path.join(d, f'L{L}_E{e}.slab'), 'wb').write(want)
                            open(os.path.join(d, 'model.wnm2'), 'wb').write(blob)
            result['records'][tag] = rows
            os.remove(cpath)
            print(f'{ckpt} {tag}: {len(rows)} records, stored {sum(r["stored"] for r in rows.values())} of '
                  f'{sum(r["decoded"] for r in rows.values())} B, all decoded equal to the source', flush=True)
    summary = {}
    for cz in ('', '-cz'):
        recs = result['records']
        dec = sum(r['decoded'] for r in recs['all' + cz].values())
        insample = sum(r['stored'] for r in recs['all' + cz].values())
        cross = sum(recs[('B' if fold[int(k[1:].split('_')[0])] == 'A' else 'A') + cz][k]['stored'] for k in recs['all' + cz])
        summary['cz' if cz else 'byte-exact'] = {'decoded': dec, 'in_sample': insample, 'cross_fold': cross,
                                                 'ratio_in_sample': insample / dec, 'ratio_cross_fold': cross / dec}
    result['summary'] = summary
    print(json.dumps(summary))
    with open(os.path.join(out, 'wnr3.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)


def cmd_container(argv):
    container, ref, data_dir, work = argv
    chosen = json.load(open(os.path.join(data_dir, 'fetch-log.json')))['pairs']['glm']['chosen']
    want = {(L << 16) | e for L, e in chosen}
    src, _ = slabs('qwen38', os.path.join(data_dir, 'glm-mxfp4', 'experts.safetensors'))
    rows = {}
    with open(container, 'rb') as f:
        rd = wex4.Reader(f, wex4.CLib(ref))
        for r, rec in enumerate(rd.records):
            offset, stored, decoded, codec, flags, bl, owner, first, count, digest, _ = rec
            if not flags & wex4.REC_EXPERT or owner not in want:
                continue
            L, e = owner >> 16, owner & 0xffff
            got = rd.record(r)
            entry, slab = src.slab(L, e)
            stored_rec = rd.stored(r)
            rows[f'L{L}_E{e}'] = {'codec': codec, 'stored': stored, 'decoded': decoded,
                                  'wnr3_flags': f'0x{stored_rec[27]:02x}' if codec == wex4.CODEC_WNR3 else None,
                                  'equal_to_source': got == slab, 'equal_to_z_source': got == z_fast(slab, entry[2]),
                                  'sha256_stored': digest.hex()}
            print(f'L{L} E{e}: {rows[f"L{L}_E{e}"]}', flush=True)
    dec = sum(r['decoded'] for r in rows.values())
    st = sum(r['stored'] for r in rows.values())
    out = {'container_root': rd.root.hex(), 'records': rows, 'decoded': dec, 'stored': st, 'ratio': st / dec}
    print(json.dumps({k: v for k, v in out.items() if k != 'records'}))
    with open(os.path.join(work, 'glm-container.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    {'pack': cmd_pack, 'container': cmd_container}[sys.argv[1]](sys.argv[2:])
