# SPDX-License-Identifier: Apache-2.0
"""WNR4 vs WNR3 on real records: stored/raw, byte-exact round trip of every record.

    python -I measure.py DATASET OUT.json [--train 256] [--test 2048] [--threads 8] [--cz 1] [--G 32,64,128,256] [--pb 1|5]
        [--wnr3 1] [--pack DIR --pack-n N --pack-G G]

Models: WNR4 (WNM4) and, for MXFP4, WNR3 (WNM2) counted on the same --train records (evenly spaced).
Test records: seeded, the same number per layer (data.Dataset.split). Every test record is coded with
every G, decoded by the reference decoder and compared with the expected output domain: Z(source) under
CZ, else the source (v41: the WNR3 decode, which is Z(source)). WNR3 baseline: MX = WNR3 encoder with
the WNM2 model of the same train records at 2048:512 (v41 also reports its stored production records);
NV = no WNR3-NV encoder on this branch, see the report. --pack writes N test records (WNR4 at --pack-G,
plus WNR3 for MX) for the GPU bench."""
import ctypes as C
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

P = C.c_void_p


def load_wnr4():
    lib = C.CDLL(os.path.join(HERE, 'out', 'wnr4_tools.dll'))
    lib.wnr4t_counts_words.restype = C.c_size_t
    lib.wnr4t_model_bytes.restype = C.c_size_t
    lib.wnr4t_counts_words.argtypes = [C.c_uint32, C.c_uint32]
    lib.wnr4t_model_bytes.argtypes = [C.c_uint32, C.c_uint32]
    lib.wnr4t_count.argtypes = [P, P, C.c_uint32, P]
    lib.wnr4t_build.argtypes = [C.c_uint32, C.c_uint32, P, P]
    lib.wnr4t_build.restype = C.c_uint32
    lib.wnr4t_model_open.argtypes = [P, C.c_size_t]
    lib.wnr4t_model_open.restype = P
    lib.wnr4t_encode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, C.c_uint32, P, C.c_size_t]
    lib.wnr4t_encode.restype = C.c_size_t
    lib.wnr4t_decode.argtypes = [P, P, P, C.c_size_t, C.c_uint32, P, C.c_int]
    lib.wnr4t_decode.restype = C.c_uint32
    return lib


def geo_of(ds):
    g = ds.geo   # per m: soff, rows, cols, woff
    sc = [g[4 * m + 1] * g[4 * m + 2] for m in range(3)]
    return (C.c_uint32 * 13)(int(ds.nv), *sc, *[g[4 * m + 2] for m in range(3)], *[g[4 * m] for m in range(3)],
                             *[g[4 * m + 3] for m in range(3)])


def expected_output(ds, src, cz):
    if not cz:
        return src
    s = bytearray(src)
    s[ds.scale_bytes:] = bytes(s[ds.scale_bytes:]).translate(data.ZTAB)
    return bytes(s)


def main(argv):
    name, out = argv[1], argv[2]
    opt = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1, 2)}
    ntrain, ntest, threads = int(opt.get('--train', 256)), int(opt.get('--test', 2048)), int(opt.get('--threads', 8))
    cz = bool(int(opt.get('--cz', 1)))
    Gs = [int(x) for x in opt.get('--G', '32,64,128,256').split(',')]
    ds = data.Dataset(name)
    train, test = ds.split(ntrain, ntest)
    lib = load_wnr4()
    geo = geo_of(ds)
    kind = int(ds.nv)
    pb = int(opt.get('--pb', 1))
    t0 = time.time()
    # ---- train
    counts = (C.c_uint64 * lib.wnr4t_counts_words(kind, pb))()
    do3 = not ds.nv and bool(int(opt.get('--wnr3', 1)))
    if do3:
        t3 = ds.tools
        t3.wex_nr3_count_expert.argtypes = [P, P, P, P]
        t3.wex_nr3_build_model.argtypes = [P, P, P, P]
        t3.wex_nr3_encode_record.argtypes = [P, P, P, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, P, C.c_size_t, P, P]
        t3.wex_nr3_verify.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
        wc3 = (C.c_uint64 * (3 * 567 * 16))()
        sc3 = (C.c_uint64 * (3 * 256))()
    do_nv3 = ds.nv and bool(int(opt.get('--wnr3', 1))) and os.path.exists(os.path.join(HERE, 'out', 'wex_nv_tools.dll'))
    if do_nv3:   # WNR3-NV (codec 7 of branch nvfp4-wnr3), built from that tree, at its default chunks 8192:2048
        nvl = C.CDLL(os.path.join(HERE, 'out', 'wex_nv_tools.dll'))
        nvl.wex_nv_count_expert.argtypes = [P, P, C.c_int, C.c_uint32, P, P]
        nvl.wex_nv_build_model.argtypes = [P, P, P]
        nvl.wex_nv_model_bytes.restype = C.c_uint32
        nvl.wex_nv_encode_record.argtypes = [P, P, P, C.c_int, C.c_uint32, C.c_uint32, C.c_uint32, P, C.c_size_t, P]
        wcn = (C.c_uint64 * (3 * 1053 * 16))()
        scn = (C.c_uint64 * (3 * 257 * 256))()
    lock = threading.Lock()
    ent0 = ds.entry(ds.experts[0])

    def train_job(r):
        src = ds.source(r)
        coded, anchors, flags = ds.coded(src, cz)
        with lock:
            assert lib.wnr4t_count(geo, coded, pb, counts) == 0
            if do_nv3:
                assert nvl.wex_nv_count_expert(ent0, src, int(cz), 2048, wcn, scn) == 0
            if do3:
                assert t3.wex_nr3_count_expert(ent0, coded, wc3, sc3) == 0
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(train_job, train))
    blob = C.create_string_buffer(lib.wnr4t_model_bytes(kind, pb))
    mid = lib.wnr4t_build(kind, pb, counts, blob)
    assert mid
    model = lib.wnr4t_model_open(blob, len(blob.raw))
    assert model
    if do3:
        blob3 = C.create_string_buffer(59408)
        id3 = C.c_uint32()
        assert t3.wex_nr3_build_model(wc3, sc3, blob3, C.byref(id3)) == 0
    if do_nv3:
        blobn = C.create_string_buffer(nvl.wex_nv_model_bytes())
        nvl.wex_nv_build_model(wcn, scn, blobn)
    t1 = time.time()
    print(f'{name}: {len(ds.experts)} experts, train {len(train)}, test {len(test)}, WNM4 {mid:08x}'
          + (f', WNM2 {id3.value:08x}' if do3 else '') + f' ({t1 - t0:.0f} s)', flush=True)
    pack_dir, pack_n, pack_G = opt.get('--pack'), int(opt.get('--pack-n', 0)), int(opt.get('--pack-G', 128))
    packs = {}

    def job(r):
        src = ds.source(r)
        coded, anchors, flags = ds.coded(src, cz)
        want = expected_output(ds, src, cz)
        own = ds.owner(r)
        cap = ds.raw * 2 + (1 << 20)
        res = {'r': r, 'layer': own >> 16, 'raw': ds.raw, 'sizes': {}, 'fails': []}
        back = C.create_string_buffer(ds.raw)
        for G in Gs:
            buf = C.create_string_buffer(cap)
            n = lib.wnr4t_encode(model, geo, coded, flags, own, G, buf, cap)
            assert n, (r, G)
            err = lib.wnr4t_decode(model, geo, buf, n, own, back, 1)
            if err or back.raw != want:
                res['fails'].append((G, err))
            res['sizes'][G] = n
            if pack_dir and r in pack_set and G == pack_G:
                packs.setdefault(r, {})['wnr4'] = buf.raw[:n]
        if do3:
            ent = ds.entry(r)
            buf = C.create_string_buffer(cap)
            used, total = C.c_size_t(), C.c_size_t()
            st = t3.wex_nr3_encode_record(blob3, ent, coded, flags, own, 2048, 512, buf, cap, C.byref(used), C.byref(total))
            assert st in (0, 1)
            n3 = total.value
            ent5 = ds.entry(r, 5, n3)
            if st == 0 and t3.wex_nr3_verify(blob3, ent5, buf, own >> 16, own & 0xffff, want) != 0:
                res['fails'].append(('wnr3', -1))
            res['wnr3'] = n3 if st == 0 else data._wex4(ds.nv).align(ds.raw)
            if pack_dir and r in pack_set:
                packs.setdefault(r, {})['wnr3'] = buf.raw[:n3]
        if do_nv3:   # the encoder decodes and compares with the source (Z(source) under CZ) itself
            buf = C.create_string_buffer(cap)
            total = C.c_size_t()
            st = nvl.wex_nv_encode_record(blobn, ds.entry(r), src, int(cz), own, 8192, 2048, buf, cap, C.byref(total))
            if st < 0:
                res['fails'].append(('wnr3nv', st))
            res['wnr3nv'] = total.value if st == 0 else data._wex4(ds.nv).align(ds.raw)
        if ds.codec == 5:
            res['stored'] = ds.stored(r)
        if pack_dir and r in pack_set:
            packs.setdefault(r, {})['want'] = want
        return res
    pack_set = set(test[:pack_n]) if pack_dir else set()
    results = []
    with ThreadPoolExecutor(threads) as pool:
        for k, res in enumerate(pool.map(job, test)):
            results.append(res)
            if (k + 1) % 128 == 0:
                print(f'  {k + 1}/{len(test)} {time.time() - t1:.0f} s', flush=True)
    raw = sum(x['raw'] for x in results)
    rep = {'dataset': name, 'cz': cz, 'pb': pb, 'train': len(train), 'test': len(test), 'raw_bytes': raw, 'wnm4_id': f'{mid:08x}',
           'fails': [x['r'] for x in results if x['fails']], 'wnr4': {}}
    for G in Gs:
        tot = sum(x['sizes'][G] for x in results)
        rep['wnr4'][G] = {'stored': tot, 'ratio': tot / raw, 'max_record': max(x['sizes'][G] for x in results)}
    if do3:
        tot = sum(x['wnr3'] for x in results)
        rep['wnr3_same_train'] = {'stored': tot, 'ratio': tot / raw}
    if do_nv3:
        tot = sum(x['wnr3nv'] for x in results)
        rep['wnr3nv_same_train_8192_2048'] = {'stored': tot, 'ratio': tot / raw}
    if ds.codec == 5:
        tot = sum(x['stored'] for x in results)
        rep['wnr3_production_records'] = {'stored': tot, 'ratio': tot / raw}
    rep['seconds'] = time.time() - t0
    with open(out, 'w') as g:
        json.dump(rep, g, indent=1)
    for G in Gs:
        print(f'WNR4 G={G}: {rep["wnr4"][G]["ratio"]:.5f}')
    for k in ('wnr3_same_train', 'wnr3_production_records', 'wnr3nv_same_train_8192_2048'):
        if k in rep:
            print(f'{k}: {rep[k]["ratio"]:.5f}')
    print(f'round-trip failures: {len(rep["fails"])} of {len(results)} records; {rep["seconds"]:.0f} s')
    if pack_dir:
        os.makedirs(pack_dir, exist_ok=True)
        write_pack(os.path.join(pack_dir, f'{name}.pack'), ds, geo, blob.raw, blob3.raw if do3 else b'', packs, pack_G)


def write_pack(path, ds, geo, wnm4, wnm2, packs, G):
    """pack: 'WP4K' u32 version 1, u32 n, u32 kind, u32 G, geo u32[13], u32 len(wnm4), u32 len(wnm2), entry72,
    blobs; then per record: u32 owner, u32 len4, u32 len3, u32 rawlen, bytes."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sIIII', b'WP4K', 1, len(packs), int(ds.nv), G))
        f.write(bytes(geo))
        f.write(struct.pack('<II', len(wnm4), len(wnm2)))
        f.write(ds.entry(ds.experts[0]))
        f.write(wnm4)
        f.write(wnm2)
        for r, p in packs.items():
            w4, w3, want = p['wnr4'], p.get('wnr3', b''), p['want']
            f.write(struct.pack('<IIII', ds.owner(r), len(w4), len(w3), len(want)))
            f.write(w4)
            f.write(w3)
            f.write(want)
    print('pack', path, len(packs), 'records')


if __name__ == '__main__':
    main(sys.argv)
