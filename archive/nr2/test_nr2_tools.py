# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Equivalence tests of wex_nr2_tools.dll against the CPU reference on real experts.

GPU counts == wex_nr2_count, GPU records == wex_nr2_encode byte for byte (several
experts x chunk sizes, canonical zero on/off, a synthetic layout with partial
streams and unaligned matrices, uniform and adversarial models), every record
decoded back by wex_nr2_parse + wex_nr2_decode, and the engine slab path.
Reads pilot-v1.wex only, from WEX_PILOT_DIR (default U:/kaira_build/wexgd-20261019); writes test-nr2.json.
"""
import json
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import nr2lib as L

SOURCE = Path(os.environ.get('WEX_PILOT_DIR', 'U:/kaira_build/wexgd-20261019')) / 'pilot-v1.wex'
OUT = L.HERE / 'test-nr2.json'
CHUNKS = [(4096, 2048), (512, 256), (16384, 16384), (1024, 512), (8192, 4096), (2048, 16384)]


def main():
    tools = L.Tools()
    header, page, table, entries = L.read_container(SOURCE)
    present = [(i, e) for i, e in enumerate(entries) if e[L.F_TOTAL]]
    picks = [present[k] for k in (0, 137, 275, 412, 550, 687)]
    results = {'source': str(SOURCE), 'experts': [], 'checks': {}, 'timing_ms': {}}
    raws = {}
    with SOURCE.open('rb') as f:
        for index, e in picks:
            f.seek(e[L.F_OFFSET])
            raws[index] = f.read(L.raw_size(e))
            assert len(raws[index]) == L.raw_size(e)
    checks = results['checks']

    def check(name, ok):
        checks.setdefault(name, [0, 0])
        checks[name][0 if ok else 1] += 1
        if not ok:
            print('FAIL', name, flush=True)

    timing = {'count_gpu': [], 'count_cpu': [], 'encode_gpu': [], 'encode_cpu': [], 'verify_cpu': []}
    # 1. counts
    totals = {0: (np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64)),
              1: (np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64))}
    for index, e in picks:
        eb = L.ENTRY.pack(*e)
        raw = raws[index]
        _, mapped = tools.canonical(eb, raw)
        results['experts'].append({'index': index, 'layer': index // header[3], 'expert': index % header[3],
                                   'negative_zero_codes': mapped})
        for cz in (0, 1):
            wg, sg = np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64)
            wc, sc = np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64)
            t = time.perf_counter(); assert tools.count(raw, eb, cz, wg, sg, gpu=True) == 0
            timing['count_gpu'].append(time.perf_counter() - t)
            t = time.perf_counter(); assert tools.count(raw, eb, cz, wc, sc, gpu=False) == 0
            timing['count_cpu'].append(time.perf_counter() - t)
            check(f'count_gpu_eq_cpu_cz{cz}', np.array_equal(wg, wc) and np.array_equal(sg, sc))
            check('count_totals', int(wg.sum()) == e[L.F_WEIGHT] * 2 and int(sg.sum()) == e[L.F_SCALE])
            # accumulation (+=), like the CPU reference
            assert tools.count(raw, eb, cz, wg, sg, gpu=True) == 0
            check('count_gpu_accumulates', np.array_equal(wg, 2 * wc) and np.array_equal(sg, 2 * sc))
            totals[cz][0][:] += wc; totals[cz][1][:] += sc
    blob = tools.build_model(*totals[0])
    blob_z = tools.build_model(*totals[1])
    check('model_id_is_fnv1a', tools.model_id(blob) == L.fnv1a(blob) and tools.model_id(blob_z) == L.fnv1a(blob_z))
    check('models_differ', blob != blob_z or all(x['negative_zero_codes'] == 0 for x in results['experts']))

    def compare(tag, model, raw, eb, cz, wch, sch, expected, capacity=None, slab=False):
        capacity = capacity or 2 * len(raw) + (1 << 20)
        t = time.perf_counter(); rg, g, ng = tools.encode(model, raw, eb, cz, wch, sch, capacity, gpu=True)
        timing['encode_gpu'].append(time.perf_counter() - t)
        t = time.perf_counter(); rc, c, nc = tools.encode(model, raw, eb, cz, wch, sch, capacity, gpu=False)
        timing['encode_cpu'].append(time.perf_counter() - t)
        check(f'{tag}_rc_ok', rg == 0 and rc == 0)
        check(f'{tag}_gpu_eq_cpu', rg == 0 and g == c and ng == nc)
        if rg:
            return None
        padded = g + bytes(L.align(len(g)) - len(g))
        t = time.perf_counter(); v = tools.verify(model, eb, padded, expected)
        timing['verify_cpu'].append(time.perf_counter() - t)
        check(f'{tag}_verify', v == 0)
        if slab:
            e = list(L.ENTRY.unpack(eb)); e[L.F_TOTAL], e[L.F_FLAGS] = len(padded), L.WEX_ENTRY_NEURAL_RANS2
            check(f'{tag}_slab_path', tools.verify_slab(model, L.ENTRY.pack(*e), padded, expected) == 0)
        return padded

    # 2. encoder on real experts
    sizes = {}
    for n, (index, e) in enumerate(picks):
        eb, raw = L.ENTRY.pack(*e), raws[index]
        canon, mapped = tools.canonical(eb, raw)
        for k, (wch, sch) in enumerate(CHUNKS):
            if k and n % 2 and k > 2:
                continue   # every chunk size on half the experts, the main ones on all
            rec = compare('real', blob, raw, eb, 0, wch, sch, raw, slab=(k == 0))
            if rec is not None:
                sizes.setdefault(f'{wch}/{sch}', []).append(len(rec))
        for wch, sch in CHUNKS[:3]:
            rec = compare('real_cz', blob_z, raw, eb, 1, wch, sch, canon, slab=(wch == 4096))
            if rec is not None and mapped:
                check('cz_record_not_raw_exact', tools.verify(blob_z, eb, rec, raw) == -4)
        rec = compare('real_main', blob, raw, eb, 0, 4096, 2048, raw)
        check('other_model_refused', tools.verify(blob_z, eb, rec, raw) == -2 or blob == blob_z)
        broken = bytearray(rec); broken[len(rec) // 2] ^= 0x40
        check('corruption_detected', tools.verify(blob, eb, bytes(broken), raw) != 0)
    results['record_bytes_by_chunk'] = {k: sum(v) / len(v) for k, v in sizes.items()}

    # 3. capacity and argument failures agree
    index, e = picks[0]
    eb, raw = L.ENTRY.pack(*e), raws[index]
    rg, rec, need = tools.encode(blob, raw, eb, 0, 4096, 2048, len(raw), gpu=True)
    rg2, _, hint = tools.encode(blob, raw, eb, 0, 4096, 2048, need - 1, gpu=True)
    rc2, _, _ = tools.encode(blob, raw, eb, 0, 4096, 2048, need - 1, gpu=False)
    check('capacity_short_refused_both', rg == 0 and rg2 == -1 and rc2 == -1 and hint == need)
    rg3, _, _ = tools.encode(blob, raw, eb, 0, 4096, 2048, need, gpu=True)
    check('capacity_exact_ok', rg3 == 0)
    for wch, sch in ((300, 2048), (4096, 128), (256, 256), (32768, 2048), (4096, 3000)):
        a = tools.encode(blob, raw, eb, 0, wch, sch, 2 * len(raw), gpu=True)[0]
        b = tools.encode(blob, raw, eb, 0, wch, sch, 2 * len(raw), gpu=False)[0]
        check('bad_chunk_refused_both', a == -1 and b == -1)
    bad_blob = bytearray(blob); bad_blob[8] ^= 1   # precision 14 -> 15
    check('bad_model_refused', tools.encode(bytes(bad_blob), raw, eb, 0, 4096, 2048, 2 * len(raw))[0] == -1
          and tools.model_id(bytes(bad_blob)) is None)

    # 4. synthetic layout: partial streams, matrices not 16-aligned, extreme scales
    shapes = [((100, 7), (100, 112)), ((37, 9), (37, 144)), ((64, 3), (64, 48))]
    scale_off, weight_off, s_total = [], [], 0
    for (sr, sc_), _ in shapes:
        scale_off.append(s_total); s_total += sr * sc_
    w_total = s_total
    for _, (wr, wc_) in shapes:
        weight_off.append(w_total); w_total += wr * wc_
    syn = bytearray(w_total)
    rng = np.random.default_rng(7)
    for m, ((sr, sc_), (wr, wc_)) in enumerate(shapes):
        src_s = e[16 + m]; src_w = e[19 + m]
        n_s, n_w = sr * sc_, wr * wc_
        syn[scale_off[m]:scale_off[m] + n_s] = raw[src_s:src_s + n_s]
        syn[weight_off[m]:weight_off[m] + n_w] = raw[src_w + 4096:src_w + 4096 + n_w]
        for p in rng.choice(n_s, 12, replace=False):   # far from the median: srel 0 and 6
            syn[scale_off[m] + int(p)] = int(rng.choice([0, 3, 250, 255]))
    syn_entry = [0, 0, s_total, w_total - s_total]
    for (sr, sc_), _ in shapes:
        syn_entry += [sr, sc_]
    for _, (wr, wc_) in shapes:
        syn_entry += [wr, wc_]
    syn_entry += scale_off + weight_off + [0]
    seb, syn = L.ENTRY.pack(*syn_entry), bytes(syn)
    syn_canon, _ = tools.canonical(seb, syn)
    wg, sg = np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64)
    wc, sc = np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64)
    for cz in (0, 1):
        assert tools.count(syn, seb, cz, wg, sg, gpu=True) == 0 and tools.count(syn, seb, cz, wc, sc, gpu=False) == 0
        check('synthetic_count_gpu_eq_cpu', np.array_equal(wg, wc) and np.array_equal(sg, sc))
    for wch, sch in CHUNKS:
        compare('synthetic', blob, syn, seb, 0, wch, sch, syn, slab=True)
        compare('synthetic_cz', blob_z, syn, seb, 1, wch, sch, syn_canon, slab=True)
    syn_model = tools.build_model(wc, sc)   # model of the synthetic data itself
    compare('synthetic_own_model', syn_model, syn, seb, 0, 512, 256, syn, slab=True)

    # 5. uniform (no counts) and adversarial (real symbols get frequency 1) models
    uniform = tools.build_model(np.zeros(L.WEIGHT_BINS, np.uint64), np.zeros(L.SCALE_BINS, np.uint64))
    rec = compare('uniform_model', uniform, raw, eb, 0, 4096, 2048, raw)
    results['uniform_model_record_bytes'] = len(rec) if rec else None
    adv_w = np.zeros((3, 567, 16), np.uint64); adv_w[:, :, 15] = 1 << 40
    adv_s = np.zeros((3, 256), np.uint64); adv_s[:, 255] = 1 << 40
    adversarial = tools.build_model(adv_w.reshape(-1), adv_s.reshape(-1))
    rec = compare('adversarial_model', adversarial, raw, eb, 0, 16384, 16384, raw, capacity=64 << 20)
    if rec:
        sp, wp = struct.unpack_from('<II', rec, 8)
        _, _, ss, ws = L.layout_streams(e, 16384, 16384)
        lengths = struct.unpack_from(f'<{ss + ws}H', rec, 32)
        results['adversarial'] = {'record_bytes': len(rec), 'max_stream_bytes': max(lengths),
                                  'stream_bound_2n_plus_8': 2 * 16384 + 8}
        check('adversarial_streams_within_bound', max(lengths) <= 2 * 16384 + 8)
    # an adversarial model at the smallest chunks too
    compare('adversarial_small_chunks', adversarial, syn, seb, 0, 512, 256, syn, slab=False)

    # 6. GPU speed on the main setting (warm)
    t = time.perf_counter()
    for index, e in picks:
        eb = L.ENTRY.pack(*e)
        assert tools.count(raws[index], eb, 0, wg, sg) == 0
    results['timing_ms']['gpu_count_per_expert_warm'] = (time.perf_counter() - t) / len(picks) * 1e3
    out = bytearray(len(raw))
    t = time.perf_counter()
    for index, e in picks:
        assert tools.encode(blob, raws[index], L.ENTRY.pack(*e), 0, 4096, 2048, len(out), out=out)[0] == 0
    results['timing_ms']['gpu_encode_per_expert_warm'] = (time.perf_counter() - t) / len(picks) * 1e3
    for k, v in timing.items():
        if v:
            results['timing_ms'][k + '_median'] = float(np.median(v)) * 1e3
    tools.release()
    results['all_passed'] = all(bad == 0 for ok, bad in checks.values())
    results['checks'] = {k: {'passed': ok, 'failed': bad} for k, (ok, bad) in checks.items()}
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if results['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
