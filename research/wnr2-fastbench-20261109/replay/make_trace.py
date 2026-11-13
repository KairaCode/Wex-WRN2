"""Engine recording -> replay trace for replay.exe.

  python -I make_trace.py --tap RUN.v4rt --moe RUN.moe --out RUN.trace [--log RUN.stderr.log]
                          [--hidden 5120] [--pf-share F]
  python -I make_trace.py --synth LAYERS EXPERTS TOKENS --out SYN.trace [--seed 1] [--demand 3] [--needed 1] [--wasted 2]
  python -I make_trace.py --self-test

Inputs, both from one engine run (record_trace.sh):
  RUN.moe   V4_DUMP_MOE: every decode MoE call, {i32 layer, i32 token_in, i32 count,
            i32 ids[count], f32 weights[count], f32 input[hidden]}. On V4.1 only the decode
            goes through moe_token_pipeline, so these are the decode forwards, 40 records each.
  RUN.v4rt  the routing tap (tap_capture.py), wire v1. On V4.1 its FRAMEs carry no LayerBlocks
            (the tap's per-layer path is V4's block_token_pipeline), but every FRAME is followed
            by a RESID: the experts in VRAM and in RAM at that token boundary.
Alignment: the decode forward that takes token X follows the FRAME whose token_out is X, inside
one turn (a turn of N frames has N-1 decode forwards); the ids are checked one by one.

Per decode forward and layer, with A = VRAM|RAM before the forward (the RESID after the
previous frame) and A' = after it:
  reads  = routed experts not in A                  (measured: they came from disk)
  W      = experts of the layer in A' \\ A, not routed (wrong prefetch guesses that were read)
  reads are split into D (demand, read when the layer routes) and P (prefetched during the
  previous layer, waited for at this layer): P = a deterministic --pf-share of them (default:
  the run's own 'v4_next_predict_fresh recall', 0 without a log).
shortcut: the D/P split is a model (the tap carries no tiers on V4.1); reads of experts evicted
and re-read inside one token, and wrong guesses evicted before the token ended, are not seen.
Compare 'vram_miss_per_token' with the engine's moeprof (240 - vram_hits) and 'reads_total'
with its v4_direct/v4_wex_gpu_direct counters (which include the prefill) in the json.

Trace text, one line per (token, layer); ids are expert indices of that layer:
  # WEXREPLAY 1 layers=L experts=E ...
  <token> <layer> <nD> <nP> <nW> D... P... W...
"""
import hashlib
import json
import os
import random
import re
import struct
import sys

MAGIC = 0x54523456
HELLO, TURN, FRAME, PREFILL, END, RESID = 1, 2, 3, 4, 5, 7
HDR = struct.Struct('<IBBHIIQ')                       # 24
HELLO_S = struct.Struct('<HHBBBBBBBBIII')             # 24
FRAME_S = struct.Struct('<iiiifIBBBB')                # 28
assert (HDR.size, HELLO_S.size, FRAME_S.size) == (24, 24, 28)


def read_capture(path):
    data = open(path, 'rb').read()
    out, at = [], 0
    while at + 4 <= len(data):
        (n,) = struct.unpack_from('<I', data, at)
        at += 4
        if at + n > len(data):
            break   # truncated tail (capture killed mid-write)
        out.append(data[at:at + n])
        at += n
    return out


def parse_tap(datagrams):
    """-> hello or None, seq-ordered events: ('turn', id) ('frame', token_out) ('resid', (cols, vram, ram)) ('end', d)"""
    hello, events, bad = None, [], 0
    for d in datagrams:
        if len(d) < HDR.size:
            bad += 1
            continue
        magic, version, kind, length, seq, turn, t_us = HDR.unpack_from(d)
        if magic != MAGIC or version != 1 or length != len(d):
            bad += 1
            continue
        if kind == HELLO:
            hello = dict(zip(('layers', 'experts', 'topk', 'hash_layers', 'shared', 'flags', 'pf_k',
                              'pf_k2', 'lanes', 'resid_every', 'pid', 'vram_cap', 'record'),
                             HELLO_S.unpack_from(d, 24)))
        elif kind == TURN:
            events.append((seq, 'turn', d[40:40 + d[36]].decode('utf-8', 'replace')))
        elif kind == FRAME:
            events.append((seq, 'frame', FRAME_S.unpack_from(d, 24)[2]))
        elif kind == RESID:
            rows, cols = struct.unpack_from('<2H', d, 24)
            b = (rows * cols + 7) // 8
            bits = [int.from_bytes(d[32 + k * b:32 + (k + 1) * b], 'little') for k in (0, 1)]
            events.append((seq, 'resid', (cols, bits[0], bits[1])))
        elif kind == END:
            gen, prompt, tok_s, dropped = struct.unpack_from('<iifI', d, 24)
            events.append((seq, 'end', {'generated': gen, 'prompt': prompt, 'dropped': dropped}))
    events.sort(key=lambda e: e[0])
    return hello, [(k, v) for _, k, v in events], bad


def read_moe(path, hidden):
    """V4_DUMP_MOE -> [(token_in, [ids of layer 0], [ids of layer 1], ...)] per decode forward"""
    data = open(path, 'rb').read()
    forwards, at, last = [], 0, None
    while at + 12 <= len(data):
        layer, token, count = struct.unpack_from('<3i', data, at)
        if not 0 <= layer < 256 or not 0 < count <= 64 or at + 12 + 8 * count + 4 * hidden > len(data):
            raise SystemExit(f'{path}: bad record at byte {at} (wrong --hidden?)')
        ids = list(struct.unpack_from(f'<{count}i', data, at + 12))
        at += 12 + 8 * count + 4 * hidden
        if last is None or layer <= last:
            if layer != 0:
                raise SystemExit(f'{path}: a forward starts at layer {layer}')
            forwards.append([token])
        elif layer != last + 1 or token != forwards[-1][0]:
            raise SystemExit(f'{path}: layer {layer} after {last}, or a token change inside a forward')
        forwards[-1].append(ids)
        last = layer
    return forwards


def align(events, forwards):
    """-> [(resid_before, resid_after)] per forward; checks token ids"""
    turns, cur = [], None
    for kind, value in events:
        if kind == 'turn':
            cur = []
            turns.append(cur)
        elif kind == 'frame' and cur is not None:
            cur.append([value, None])
        elif kind == 'resid' and cur:
            if cur[-1][1] is None:
                cur[-1][1] = value
    pairs, k = [], 0
    for frames in turns:
        for i in range(len(frames) - 1):
            if k >= len(forwards):
                raise SystemExit(f'the tap has more decode forwards than the MoE dump ({len(forwards)})')
            if forwards[k][0] != frames[i][0]:
                raise SystemExit(f'forward {k}: MoE dump token {forwards[k][0]} != FRAME token_out {frames[i][0]}')
            if frames[i][1] is None or frames[i + 1][1] is None:
                raise SystemExit(f'forward {k}: no RESID after its frames (V4_ROUTE_RESID_EVERY=1?)')
            pairs.append((frames[i][1], frames[i + 1][1]))
            k += 1
    if k != len(forwards):
        raise SystemExit(f'the MoE dump has {len(forwards)} decode forwards, the tap {k}')
    return pairs


def layer_set(bits, layer, cols):
    v = (bits >> (layer * cols)) & ((1 << cols) - 1)
    return {e for e in range(cols) if v >> e & 1}


def derive(forwards, pairs, pf_share):
    rows, vmiss, ram_hits = [], 0, 0
    for t, (fw, (before, after)) in enumerate(zip(forwards, pairs)):
        cols = before[0]
        for layer, routed in enumerate(fw[1:]):
            vram = layer_set(before[1], layer, cols)
            have = vram | layer_set(before[2], layer, cols)
            now = layer_set(after[1], layer, cols) | layer_set(after[2], layer, cols)
            reads = [e for e in routed if e not in have]
            vmiss += sum(e not in vram for e in routed)
            ram_hits += sum(e not in vram and e in have for e in routed)
            # deterministic choice of which reads were prefetched: a hash of (token, layer, expert)
            order = sorted(reads, key=lambda e: hashlib.blake2s(f'{t},{layer},{e}'.encode(), digest_size=4).digest())
            n_pf = int(round(pf_share * len(reads)))
            P = sorted(order[:n_pf])
            D = [e for e in reads if e not in P]
            W = sorted(now - have - set(routed))
            rows.append((t, layer, D, P, W))
    n = max(len(forwards), 1)
    stats = {'tokens': len(forwards), 'pf_share': pf_share,
             'vram_miss_per_token': round(vmiss / n, 2), 'ram_hits_per_token': round(ram_hits / n, 2)}
    for name, i in (('demand', 2), ('needed', 3), ('wasted', 4)):
        stats[name + '_per_token'] = round(sum(len(r[i]) for r in rows) / n, 2)
    stats['reads_per_token'] = round(stats['demand_per_token'] + stats['needed_per_token'] + stats['wasted_per_token'], 2)
    stats['reads_total'] = sum(len(r[2]) + len(r[3]) + len(r[4]) for r in rows)
    return rows, stats


def engine_log_stats(path):
    """per-token moeprof, exit read counters, tiers, predictor recall and hidden size from a serve_bench stderr log"""
    waits, hits, out = [], [], {}
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.search(r'moeprof wait=([\d.]+) attach=[\d.]+ gpu=[\d.]+ ms vram_hits=(\d+)', line)
        if m:
            waits.append(float(m.group(1)))
            hits.append(int(m.group(2)))
        for key in ('v4_direct reads', 'v4_wex_gpu_direct reads'):
            m = re.search(key + r'=(\d+)', line)
            if m:
                out[key.replace(' ', '_')] = int(m.group(1))
        m = re.search(r'ram_tiers available=([\d.]+)GiB .*target_slots=(\d+)', line)
        if m:
            out['ram_tiers'] = {'available_gib': float(m.group(1)), 'slots_per_layer': int(m.group(2))}
        m = re.search(r'v4_tier_prewarm vram_experts=(\d+)', line)
        if m:
            out['vram_experts'] = int(m.group(1))
        m = re.search(r'v4_next_predict_fresh recall=([\d.]+)%', line)
        if m:
            out['predict_fresh_recall'] = float(m.group(1)) / 100
        m = re.search(r'v41_moe path=\S+ experts=(\d+) hidden=(\d+) moe=\d+ topk=(\d+)', line)
        if m:
            out['experts'], out['hidden'], out['topk'] = map(int, m.groups())
    if waits:
        s = sorted(waits)
        out['engine_moe_wait_ms_per_token'] = {'n': len(s), 'mean': round(sum(s) / len(s), 1),
                                               'p50': s[len(s) // 2], 'p95': s[int(len(s) * 0.95)]}
        out['engine_vram_hits_per_token'] = round(sum(hits) / len(hits), 1)
    return out


def write_trace(path, rows, header):
    with open(path, 'w', encoding='ascii', newline='\n') as f:
        f.write('# WEXREPLAY 1 ' + ' '.join(f'{k}={v}' for k, v in header.items()) + '\n')
        f.write('# token layer nD nP nW D... P... W...\n')
        for t, l, D, P, W in rows:
            f.write(' '.join(map(str, [t, l, len(D), len(P), len(W)] + D + P + W)) + '\n')


def synth(layers, experts, tokens, seed, nd, npd, nw):
    rng = random.Random(seed)
    rows = []
    for t in range(tokens):
        for l in range(layers):
            ids = rng.sample(range(experts), min(experts, nd + npd + nw))
            rows.append((t, l, ids[:nd], ids[nd:nd + npd], ids[nd + npd:]))
    return rows


def self_test():
    import tempfile

    def dgram(kind, seq, body):
        return HDR.pack(MAGIC, 1, kind, HDR.size + len(body), seq, 1, 0) + body

    def resid(seq, vram, ram, rows=2, cols=8):
        b = (rows * cols + 7) // 8
        pack = lambda cells: sum(1 << c for c in cells).to_bytes(b, 'little')
        return dgram(RESID, seq, struct.pack('<4H', rows, cols, len(vram), len(ram)) + pack(vram) + pack(ram) + bytes(b))

    def frame(seq, token):
        return dgram(FRAME, seq, FRAME_S.pack(9, 0, token, -1, 0.0, 0, 0, 0, 2, 0))

    def turn(seq):
        return dgram(TURN, seq, struct.pack('<3i', 5, 0, 3) + bytes([1, 0, 0, 0]) + b'x')

    def moe(layer, token, ids, hidden=4):
        return struct.pack(f'<3i{len(ids)}i{len(ids)}f{hidden}f', layer, token, len(ids), *ids,
                           *([0.5] * len(ids)), *([0.0] * hidden))

    hello = dgram(HELLO, 1, HELLO_S.pack(2, 8, 2, 0, 1, 7, 2, 0, 8, 1, 1, 3, 4096))
    # turn 1: frames 7 -> 8 -> 9 (2 decode forwards), turn 2: frames 4 -> 5 (1 forward)
    # cells: layer 0 = 0..7, layer 1 = 8..15
    tap = [hello, turn(2), frame(3, 7), resid(4, {0}, {1, 9}),
           frame(5, 8), resid(6, {0, 5}, {1, 9, 6, 12}),
           frame(7, 9), resid(8, {0, 5}, {1, 9, 6, 12}),
           turn(9), frame(10, 4), resid(11, {0}, {}), frame(12, 5), resid(13, {0}, {3})]
    dump = (moe(0, 7, [0, 5]) + moe(1, 7, [1, 4]) +          # L0: 5 read; L1: 1 (cell 9) RAM, 4 read
            moe(0, 8, [5, 6]) + moe(1, 8, [4, 2]) +          # all resident before except L1 4, 2
            moe(0, 4, [0, 3]) + moe(1, 4, [0, 1]))
    with tempfile.TemporaryDirectory() as d:
        p, q = os.path.join(d, 'x.v4rt'), os.path.join(d, 'x.moe')
        open(p, 'wb').write(b''.join(struct.pack('<I', len(c)) + c for c in tap) + b'\x05\x00')
        open(q, 'wb').write(dump)
        h, ev, bad = parse_tap(read_capture(p))
        fw = read_moe(q, 4)
    assert h['layers'] == 2 and bad == 0 and len(fw) == 3, (h, bad, fw)
    rows, st = derive(fw, align(ev, fw), 0.0)
    want = [(0, 0, [5], [], [6]),        # 5 read; 6 became resident unrouted -> wasted
            (0, 1, [4], [], []),         # cell 12 = L1 e4 read (routed), nothing else new
            (1, 0, [], [], []),          # 5, 6 resident
            (1, 1, [2], [], []),         # 4 resident now, 2 read
            (2, 0, [3], [], []),         # turn 2: 3 read (it shows up in RAM after)
            (2, 1, [0, 1], [], [])]      # this turn's RESID holds nothing of layer 1: both read
    assert rows == want, rows
    assert st['tokens'] == 3 and st['reads_total'] == 7 and st['vram_miss_per_token'] == 3.0, st
    rows, _ = derive(fw, align(ev, fw), 1.0)
    assert all(not D and P for _, _, D, P, _ in rows if D or P), rows
    try:   # a dump that does not match the tap must be refused
        align(ev, fw[:2])
        raise AssertionError('short dump accepted')
    except SystemExit:
        pass
    print('make_trace self-test OK')


def main(argv):
    if argv[:1] == ['--self-test']:
        return self_test()
    args, i = {}, 0
    while i < len(argv):
        if argv[i] == '--synth':
            args['--synth'] = argv[i + 1:i + 4]
            i += 4
        elif argv[i].startswith('--') and i + 1 < len(argv):
            args[argv[i]] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(__doc__)
    if '--out' not in args:
        raise SystemExit(__doc__)
    if '--synth' in args:
        L, E, T = map(int, args['--synth'])
        rows = synth(L, E, T, int(args.get('--seed', 1)), int(args.get('--demand', 3)),
                     int(args.get('--needed', 1)), int(args.get('--wasted', 2)))
        write_trace(args['--out'], rows, {'layers': L, 'experts': E, 'tokens': T, 'source': 'synthetic'})
        print(json.dumps({'trace': args['--out'], 'tokens': T, 'lines': len(rows)}))
        return
    if '--tap' not in args or '--moe' not in args:
        raise SystemExit(__doc__)
    engine = engine_log_stats(args['--log']) if '--log' in args else {}
    hidden = int(args.get('--hidden', engine.get('hidden', 5120)))
    pf_share = float(args.get('--pf-share', engine.get('predict_fresh_recall', 0.0)))
    hello, events, bad = parse_tap(read_capture(args['--tap']))
    if hello is None:
        raise SystemExit('no HELLO datagram: was V4_ROUTE_UDP set and the engine in serve mode?')
    if hello['resid_every'] != 1:
        raise SystemExit(f"resid_every={hello['resid_every']}: the trace needs a RESID after every token")
    forwards = read_moe(args['--moe'], hidden)
    if not forwards:
        raise SystemExit('the MoE dump holds no decode forward (V4_DUMP_MOE set? decode on the GPU pipeline?)')
    rows, stats = derive(forwards, align(events, forwards), pf_share)
    stats.update(hello=hello, bad_datagrams=bad, hidden=hidden,
                 turns=[v for k, v in events if k == 'turn'],
                 tap_dropped=max([v['dropped'] for k, v in events if k == 'end'], default=None))
    if engine:
        stats['engine'] = engine
        total = engine.get('v4_direct_reads', 0) + engine.get('v4_wex_gpu_direct_reads', 0)
        if total:   # the engine's counters also hold the prefill's reads: the trace stays below them
            stats['reads_vs_engine'] = round(stats['reads_total'] / total, 3)
        if 'engine_vram_hits_per_token' in engine:
            stats['engine_vram_miss_per_token'] = round(hello['layers'] * hello['topk'] - engine['engine_vram_hits_per_token'], 1)
    header = {'layers': hello['layers'], 'experts': hello['experts'], 'topk': hello['topk'],
              'record': hello['record'], 'vram_cap': hello['vram_cap'], 'lanes': hello['lanes'],
              'pf_k': hello['pf_k'], 'pf_k2': hello['pf_k2'], 'pf_share': pf_share, 'tokens': stats['tokens'],
              'source': os.path.basename(args['--tap'])}
    write_trace(args['--out'], rows, header)
    json.dump(stats, open(args.get('--json', args['--out'] + '.json'), 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: stats.get(k) for k in ('tokens', 'reads_per_token', 'demand_per_token', 'needed_per_token',
                                                'wasted_per_token', 'vram_miss_per_token', 'engine_vram_miss_per_token',
                                                'reads_vs_engine')}))


if __name__ == '__main__':
    main(sys.argv[1:])
