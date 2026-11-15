// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! SHA-256 written from FIPS 180-4: the constants of section 4.2.2, the initial hash value of
//! section 5.3.3, the padding of section 5.1.1 and the computation of section 6.2.2.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

const H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

/// An incremental SHA-256: `update` any number of times, then `finish`.
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    block: [u8; 64],
    used: usize,
    bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Sha256::new()
    }
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 { h: H0, block: [0; 64], used: 0, bytes: 0 }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.bytes = self.bytes.wrapping_add(data.len() as u64);
        if self.used > 0 {
            let take = (64 - self.used).min(data.len());
            self.block[self.used..self.used + take].copy_from_slice(&data[..take]);
            self.used += take;
            data = &data[take..];
            if self.used < 64 {
                return;
            }
            let block = self.block;
            compress(&mut self.h, &block);
            self.used = 0;
        }
        while data.len() >= 64 {
            compress(&mut self.h, &data[..64]);
            data = &data[64..];
        }
        self.block[..data.len()].copy_from_slice(data);
        self.used = data.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        // 5.1.1: a 1 bit, zeros up to 448 mod 512 bits, then the message length in bits as 64 bits
        let bits = self.bytes.wrapping_mul(8);
        let zeros = if self.used < 56 { 55 - self.used } else { 119 - self.used };
        let mut pad = [0u8; 72];
        pad[0] = 0x80;
        pad[1 + zeros..9 + zeros].copy_from_slice(&bits.to_be_bytes());
        self.update(&pad[..9 + zeros]);
        debug_assert_eq!(self.used, 0);
        let mut out = [0u8; 32];
        for (i, w) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

/// SHA-256 of one byte string.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finish()
}

fn compress(h: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for t in 0..16 {
        w[t] = u32::from_be_bytes([block[4 * t], block[4 * t + 1], block[4 * t + 2], block[4 * t + 3]]);
    }
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = s1.wrapping_add(w[t - 7]).wrapping_add(s0).wrapping_add(w[t - 16]);
    }
    let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
    let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);
    for t in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
        *x = x.wrapping_add(v);
    }
}

/// Lower-case hexadecimal.
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 * bytes.len());
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn primes(n: usize) -> Vec<u128> {
        let mut out = Vec::new();
        let mut p = 2u128;
        while out.len() < n {
            if out.iter().all(|q| p % q != 0) {
                out.push(p);
            }
            p += 1;
        }
        out
    }

    /// Largest x with x^k <= n, by bisection.
    fn iroot(n: u128, k: u32) -> u128 {
        let (mut lo, mut hi) = (0u128, 1u128 << 40);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if mid.pow(k) <= n {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// 4.2.2 and 5.3.3: the first 32 bits of the fractional parts of the cube roots of the first 64
    /// primes and of the square roots of the first 8, computed exactly in integers.
    #[test]
    fn constants_follow_their_definition() {
        let p = primes(64);
        for i in 0..64 {
            assert_eq!(iroot(p[i] << 96, 3) as u32, K[i], "K[{}]", i);
        }
        for i in 0..8 {
            assert_eq!(iroot(p[i] << 64, 2) as u32, H0[i], "H0[{}]", i);
        }
    }

    /// The published test messages: empty, "abc" and the 448-bit message (FIPS 180-2 appendix B), the
    /// 896-bit message and one million times "a".
    #[test]
    fn known_answers() {
        let cases: [(&[u8], &str); 4] = [
            (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ];
        for (msg, want) in cases {
            assert_eq!(hex(&sha256(msg)), want);
        }
        let million = vec![b'a'; 1_000_000];
        assert_eq!(hex(&sha256(&million)), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    /// Splitting the input anywhere gives the same digest (the padding boundaries 55, 56, 63, 64 included).
    #[test]
    fn incremental_equals_one_shot() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 31 + 7) as u8).collect();
        for n in 0..data.len() {
            let whole = sha256(&data[..n]);
            for cut in [0, 1, n / 3, n / 2, n.saturating_sub(1), n] {
                let cut = cut.min(n);
                let mut s = Sha256::new();
                s.update(&data[..cut]);
                s.update(&data[cut..n]);
                assert_eq!(s.finish(), whole, "n {} cut {}", n, cut);
            }
        }
    }
}
