"""WNR3 draft: executable reference for the small normative pieces of WNR3-DRAFT.md (team3 / wnr3-spec).

Usage: python -I -B wnr3_ref.py            (runs all self-checks, prints test vectors as JSON)

Covers: H32 (= MurmurHash3_x86_32) + the SMHasher verification value, the per-stream hash / initial-state rule,
the canonical context->row index for layouts 0 and 1 and the closed form the GPU uses, the grouped residual
bit-packing of the coded length table (round trip), and the integer length predictor.
Pure Python, no data files.
"""
import json, struct

M32 = 0xFFFFFFFF
L = 1 << 23


def rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def fmix32(h):
    h ^= h >> 16; h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13; h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


def h32(data: bytes, seed: int) -> int:
    """MurmurHash3_x86_32 (Appleby, public domain), little-endian blocks."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & M32
    n = len(data)
    nb = n // 4
    for i in range(nb):
        k = struct.unpack_from('<I', data, 4 * i)[0]
        k = (k * c1) & M32; k = rotl(k, 15); k = (k * c2) & M32
        h ^= k; h = rotl(h, 13); h = (h * 5 + 0xE6546B64) & M32
    t = data[4 * nb:]
    k = 0
    if len(t) >= 3: k ^= t[2] << 16
    if len(t) >= 2: k ^= t[1] << 8
    if len(t) >= 1:
        k ^= t[0]
        k = (k * c1) & M32; k = rotl(k, 15); k = (k * c2) & M32
        h ^= k
    h ^= n & M32
    return fmix32(h)


def smhasher_verification():
    """SMHasher VerificationTest: keys {}, {0}, {0,1}, ... (len i, seed 256-i), then hash of the 1024-B result, seed 0."""
    key = bytearray(256)
    out = bytearray()
    for i in range(256):
        key[i] = i
        out += struct.pack('<I', h32(bytes(key[:i]), 256 - i))
    return h32(bytes(out), 0)


WNR3_SEED = 0x33524E57  # the record magic 'WNR3' read as a little-endian u32


def stream_h23(model_id, owner_layer, owner_expert, stream_index, flags, matrix, set_id, median, decoded: bytes):
    ident = struct.pack('<IIII', model_id, (owner_layer << 16) | owner_expert, stream_index,
                        (flags & 0xFF) | (matrix << 8) | (set_id << 16) | (median << 24))
    return h32(ident + decoded, WNR3_SEED) & 0x7FFFFF


# ---------------- context -> row (canonical structural index)
def tri(pm, mx):
    return mx * (mx + 1) // 2 + pm


def full_ctx(layout, srel, pm, mx, bm7=0, half=0):
    base = (srel * 9 + pm) * 9 + mx
    return base if layout == 0 else (base * 2 + bm7) * 2 + half


GOFF = {(0, 0): 0, (0, 1): 29, (1, 0): 57, (1, 1): 94}  # layout 1 group offsets inside one srel (130 rows)
NTRI = {0: 28, 1: 36}


def row_closed_form(layout, srel, pm, mx, bm7=0, half=0):
    """What the GPU computes (no table). pm == mx == 8 is the block-start row."""
    start = (pm == 8)
    if layout == 0:
        return srel * 37 + (36 if start else tri(pm, mx))
    return srel * 130 + GOFF[(bm7, half)] + (NTRI[bm7] if start else tri(pm, mx))


def canonical_index(layout):
    nfull = 567 if layout == 0 else 2268
    idx = [0xFFFF] * nfull
    for srel in range(7):
        for pm in range(9):
            for mx in range(9):
                for bm7 in (range(1) if layout == 0 else range(2)):
                    for half in (range(1) if layout == 0 else range(2)):
                        if pm == 8 or mx == 8:
                            ok = pm == 8 and mx == 8 and half == 0
                        else:
                            ok = pm <= mx and (layout == 0 or bm7 == 1 or mx <= 6)
                        if ok:
                            idx[full_ctx(layout, srel, pm, mx, bm7, half)] = row_closed_form(layout, srel, pm, mx, bm7, half)
    return idx


def check_index(layout):
    idx = canonical_index(layout)
    rows = [r for r in idx if r != 0xFFFF]
    nrows = 259 if layout == 0 else 910
    assert sorted(rows) == list(range(nrows)), 'index is not a bijection onto 0..rows-1'
    # closed form stays inside the table for EVERY pm, mx in 0..7 (also invalid ones such as bm7=0 with mx=7)
    worst = 0
    for srel in range(7):
        for bm7 in (0, 1):
            for half in (0, 1):
                for pm in range(8):
                    for mx in range(8):
                        r = row_closed_form(layout, srel, pm, mx, bm7, half) if layout == 1 else row_closed_form(0, srel, pm, mx)
                        worst = max(worst, r)
    assert worst < nrows, (layout, worst)
    import hashlib
    blob = struct.pack(f'<{len(idx)}H', *idx)
    return {'full_contexts': len(idx), 'rows': nrows, 'max_closed_form_row_any_pm_mx_0_7': worst,
            'index_sha256': hashlib.sha256(blob).hexdigest()}


# ---------------- coded length table: grouped bit-packing (G = 32)
def zz(r):
    return 2 * r if r >= 0 else -2 * r - 1


def unzz(z):
    return z >> 1 if (z & 1) == 0 else -((z + 1) >> 1)


def pack_part(lengths, preds, groups_of):
    """lengths/preds per stream of one part, groups_of = list of stream counts per matrix.
    Returns bytes of: u32 base_off[ng] | u8 width[ng] pad4 | words."""
    base, widths, words = [], [], []
    off = 0
    s = 0
    for nm in groups_of:
        for g0 in range(0, nm, 32):
            n = min(32, nm - g0)
            z = [zz(lengths[s + i] - preds[s + i]) for i in range(n)]
            w = max(z).bit_length() if max(z) > 0 else 0
            assert w <= 17
            acc = 0
            for i, v in enumerate(z):
                acc |= v << (i * w)
            words += [(acc >> (32 * q)) & M32 for q in range(w)]
            base.append(off); widths.append(w)
            off += sum(lengths[s:s + n]); s += n
    wb = bytes(widths) + bytes((-len(widths)) % 4)
    return struct.pack(f'<{len(base)}I', *base) + wb + struct.pack(f'<{len(words)}I', *words), off


def unpack_stream(part, ng, g, lane, pred):
    """What one GPU lane does: its group's base offset, width, word offset (sum of earlier widths), its residual."""
    base = struct.unpack_from('<I', part, 4 * g)[0]
    widths = part[4 * ng:4 * ng + ng]
    w = widths[g]
    wo = sum(widths[:g])
    words_at = 4 * ng + ng + ((-ng) % 4)
    if w == 0:
        return base, pred
    bit = lane * w
    q, b = divmod(bit, 32)
    lo = struct.unpack_from('<I', part, words_at + 4 * (wo + q))[0]
    hi = struct.unpack_from('<I', part, words_at + 4 * (wo + q + 1))[0] if b + w > 32 else 0
    z = ((lo | (hi << 32)) >> b) & ((1 << w) - 1)
    return base, pred + unzz(z)


def pred_weight(srels, lp_w_k, bias_k):
    s = sum(lp_w_k[x] for x in srels) + bias_k
    return 4 + (max(0, s) + 1024 >> 11)


def pred_scale(n, lp_s_m):
    return 4 + ((n * lp_s_m + 1024) >> 11)


def main():
    out = {}
    v = smhasher_verification()
    assert v == 0xB0F57EE3, hex(v)
    out['smhasher_verification_Murmur3A'] = f'0x{v:08X} (expected 0xB0F57EE3)'
    out['h32_vectors'] = {
        'h32(b"", 0)': f'0x{h32(b"", 0):08X}',
        'h32(b"", WNR3_SEED)': f'0x{h32(b"", WNR3_SEED):08X}',
        'h32(bytes(range(16)), WNR3_SEED)': f'0x{h32(bytes(range(16)), WNR3_SEED):08X}',
    }
    # stream hash vectors: a weight stream of 2048 symbols = 1024 bytes, a short scale stream of 5 bytes
    dec_w = bytes((i * 37 + 11) & 0x77 for i in range(1024))  # no code 8 nibble... just a pattern
    dec_s = bytes([127, 126, 128, 127, 125])
    hw = stream_h23(0xAB4261BD, 7, 191, 2160 + 5, 0x0F, 1, 3, 127, dec_w)
    hs = stream_h23(0xAB4261BD, 7, 191, 719, 0x0F, 0, 0, 127, dec_s)
    out['stream_vectors'] = {
        'weight: model 0xAB4261BD, layer 7, expert 191, stream 2165, flags 0x0F, matrix 1, set 3, median 127, data[i]=(37i+11)&0x77 x1024':
            {'h23': f'0x{hw:06X}', 'x0': f'0x{L + hw:08X}'},
        'scale: same record, stream 719, matrix 0, set 0, data 7F 7E 80 7F 7D':
            {'h23': f'0x{hs:06X}', 'x0': f'0x{L + hs:08X}'},
    }
    # one-bit change in the identity or data changes h23 (sanity)
    assert stream_h23(0xAB4261BD, 7, 191, 2166, 0x0F, 1, 3, 127, dec_w) != hw
    assert stream_h23(0xAB4261BD, 7, 190, 2165, 0x0F, 1, 3, 127, dec_w) != hw
    out['index_layout0'] = check_index(0)
    out['index_layout1'] = check_index(1)
    # length table round trip on a synthetic part: 3 matrices of 70, 33, 32 streams
    import random
    rnd = random.Random(20261107)
    counts = [70, 33, 32]
    n = sum(counts)
    preds = [rnd.randrange(900, 1000) for _ in range(n)]
    lens = [p + int(rnd.gauss(0, 7)) + (400 if i == 50 else 0) for i, p in enumerate(preds)]
    part, total = pack_part(lens, preds, counts)
    ng = sum(-(-c // 32) for c in counts)
    s = 0; g = 0
    for c in counts:
        for g0 in range(0, c, 32):
            nn = min(32, c - g0)
            base, _ = unpack_stream(part, ng, g, 0, preds[s])
            # lane-level: each lane gets its own length, warp exclusive scan gives the offset
            got = [unpack_stream(part, ng, g, i, preds[s + i])[1] for i in range(nn)]
            assert got == lens[s:s + nn]
            assert base == sum(lens[:s])
            s += nn; g += 1
    out['length_part_roundtrip'] = {'streams': n, 'groups': ng, 'bytes': len(part), 'payload_total': total}
    out['pred_examples'] = {
        'scale n=512 lp_s=256 (1.0 bit/scale)': pred_scale(512, 256),
        'weight 64 blocks srel=3 lp=31472 bias=0': pred_weight([3] * 64, [0, 0, 0, 31472, 0, 0, 0], 0),
    }
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
