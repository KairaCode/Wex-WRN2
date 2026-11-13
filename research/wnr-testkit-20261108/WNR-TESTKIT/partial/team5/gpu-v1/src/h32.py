"""team5/gpu-v1: H32 = MurmurHash3_x86_32 (WNR3 DRAFT 0.2.1 section 5.1 / DRAFT 0.1 section 5.6), scalar and
vectorised over streams, plus the spec's 16-B stream identity. Written from the spec text (constants c1 0xcc9e2d51,
c2 0x1b873593, block mix, tail, h ^= len, fmix32); checked against the SMHasher verification value 0xB0F57EE3 and
DRAFT 0.1's test vectors (run this file).
"""
import numpy as np

M32 = 0xFFFFFFFF
C1, C2 = 0xCC9E2D51, 0x1B873593
SEED = 0x33524E57          # WEX_NR3_HASH_SEED
MASK23 = 0x7FFFFF
L = 1 << 23


def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def fmix32(h):
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


def h32(data: bytes, seed: int) -> int:
    h = seed & M32
    n = len(data)
    nb = n // 4
    for i in range(nb):
        k = int.from_bytes(data[4 * i:4 * i + 4], 'little')
        k = (k * C1) & M32
        k = _rotl(k, 15)
        k = (k * C2) & M32
        h ^= k
        h = _rotl(h, 13)
        h = (h * 5 + 0xE6546B64) & M32
    t = data[4 * nb:]
    if t:
        k = 0
        for j, b in enumerate(t):
            k |= b << (8 * j)
        k = (k * C1) & M32
        k = _rotl(k, 15)
        k = (k * C2) & M32
        h ^= k
    h ^= n
    return fmix32(h)


def ident_words(model_id, layer, expert, g, ceil_bit, flags, m, set_id, anchor):
    """DRAFT 0.2.1 section 5.1 identity (4 x u32 LE)."""
    return [model_id & M32, ((layer << 16) | expert) & M32, (g | (ceil_bit << 31)) & M32,
            ((flags & 0xFF) | (m << 8) | (set_id << 16) | (anchor << 24)) & M32]


def ident_bytes(*a):
    return b''.join(w.to_bytes(4, 'little') for w in ident_words(*a))


# ---------------------------------------------------------------- vectorised (all inputs a multiple of 4 bytes)
def _vrotl(x, r):
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def h32_words_vec(words, seed):
    """words uint32 [S, nw] (each row = one message of 4*nw bytes, LE words) -> uint32 [S]."""
    words = np.ascontiguousarray(words, np.uint32)
    S, nw = words.shape
    h = np.full(S, seed, np.uint32)
    c1, c2, c3 = np.uint32(C1), np.uint32(C2), np.uint32(0xE6546B64)
    with np.errstate(over='ignore'):
        for i in range(nw):
            k = words[:, i] * c1
            k = _vrotl(k, 15)
            k = k * c2
            h ^= k
            h = _vrotl(h, 13)
            h = h * np.uint32(5) + c3
        h ^= np.uint32(4 * nw)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
    return h


def smhasher_verification():
    key = bytearray(256)
    hashes = bytearray(1024)
    for i in range(256):
        key[i] = i
        hashes[4 * i:4 * i + 4] = h32(bytes(key[:i]), 256 - i).to_bytes(4, 'little')
    return h32(bytes(hashes), 0)


def self_test():
    out = {}
    v = smhasher_verification()
    assert v == 0xB0F57EE3, hex(v)
    out['smhasher_verification'] = f'0x{v:08X}'
    assert h32(b'', 0) == 0
    assert h32(b'', SEED) == 0xE7F741F1
    assert h32(bytes(range(16)), SEED) == 0x31CE0919
    dec_w = bytes(((37 * i + 11) & 0x77) for i in range(1024))
    hw = h32(ident_bytes(0xAB4261BD, 7, 191, 2165, 0, 0x0F, 1, 3, 127) + dec_w, SEED) & MASK23
    assert hw == 0x0C48D3, hex(hw)
    assert L + hw == 0x008C48D3
    hs = h32(ident_bytes(0xAB4261BD, 7, 191, 719, 0, 0x0F, 0, 0, 127) + bytes([0x7F, 0x7E, 0x80, 0x7F, 0x7D]), SEED) & MASK23
    assert hs == 0x3C0999, hex(hs)
    out['draft01_vectors'] = {'H32("",0)': '0x00000000', 'H32("",seed)': '0xE7F741F1', 'H32(00..0F,seed)': '0x31CE0919',
                              'weight h23': f'0x{hw:06X}', 'scale h23': f'0x{hs:06X}'}
    # vectorised == scalar on random multiples of 4
    rng = np.random.default_rng(1)
    W = rng.integers(0, 2 ** 32, (64, 37), dtype=np.uint64).astype(np.uint32)
    hv = h32_words_vec(W, SEED)
    for i in range(64):
        assert int(hv[i]) == h32(W[i].astype('<u4').tobytes(), SEED)
    out['vectorised_eq_scalar'] = '64/64'
    # the ceil bit (ident[2] bit 31) changes the hash
    hc = h32(ident_bytes(0xAB4261BD, 7, 191, 2165, 1, 0x0F, 1, 3, 127) + dec_w, SEED) & MASK23
    assert hc != hw
    out['ceil_bit_vector_weight_h23'] = f'0x{hc:06X}'
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), indent=1))
