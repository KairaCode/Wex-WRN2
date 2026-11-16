// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! The two small hashes of the record codecs: MurmurHash3_x86_32 (WNR3-SPEC section 4.5), used by the
//! stream checks of WNR3 and WBP1, and the 32-bit FNV-1a of a WNM2 model (WNR3-SPEC erratum 12).

/// The seed of every stream hash: the WNR3 magic.
pub const SEED: u32 = 0x3352_4E57;

const C1: u32 = 0xcc9e_2d51;
const C2: u32 = 0x1b87_3593;

/// MurmurHash3_x86_32 over any number of `update` calls.
pub struct Murmur {
    h: u32,
    tail: [u8; 4],
    ntail: usize,
    len: u32,
}

impl Murmur {
    pub fn new(seed: u32) -> Murmur {
        Murmur { h: seed, tail: [0; 4], ntail: 0, len: 0 }
    }

    fn block(&mut self, k: u32) {
        let k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        self.h ^= k;
        self.h = self.h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    /// One 4-byte little-endian block (an identity word); only valid on a block boundary.
    pub fn word(&mut self, w: u32) {
        debug_assert_eq!(self.ntail, 0);
        self.block(w);
        self.len = self.len.wrapping_add(4);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u32);
        while self.ntail > 0 && !data.is_empty() {
            self.tail[self.ntail] = data[0];
            self.ntail += 1;
            data = &data[1..];
            if self.ntail == 4 {
                self.ntail = 0;
                let k = u32::from_le_bytes(self.tail);
                self.block(k);
            }
        }
        let whole = data.len() / 4 * 4;
        for i in (0..whole).step_by(4) {
            self.block(u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]));
        }
        for &b in &data[whole..] {
            self.tail[self.ntail] = b;
            self.ntail += 1;
        }
    }

    pub fn finish(mut self) -> u32 {
        if self.ntail > 0 {
            let mut k = 0u32;
            for i in 0..self.ntail {
                k |= (self.tail[i] as u32) << (8 * i);
            }
            let k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
            self.h ^= k;
        }
        let mut h = self.h ^ self.len;
        h ^= h >> 16;
        h = h.wrapping_mul(0x85eb_ca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2_ae35);
        h ^= h >> 16;
        h
    }
}

#[cfg(test)]
pub fn murmur3(data: &[u8], seed: u32) -> u32 {
    let mut m = Murmur::new(seed);
    m.update(data);
    m.finish()
}

/// H32(ident || data) with the stream seed: the identity words are hashed as four blocks.
pub fn stream_hash(ident: [u32; 4], data: &[u8]) -> u32 {
    let mut m = Murmur::new(SEED);
    for w in ident {
        m.word(w);
    }
    m.update(data);
    m.finish()
}

/// The model id: 32-bit FNV-1a of the blob, 0 replaced by 1.
pub fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16_777_619);
    }
    if h == 0 {
        1
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WNR3-SPEC section 4.5, "Test vectors".
    #[test]
    fn wnr3_spec_vectors() {
        // SMHasher's verification value: keys 0..255 hashed with seed 256 - i, then the table with seed 0
        let mut table = Vec::new();
        let key: Vec<u8> = (0..=255u8).collect();
        for i in 0..256usize {
            table.extend_from_slice(&murmur3(&key[..i], 256 - i as u32).to_le_bytes());
        }
        assert_eq!(murmur3(&table, 0), 0xB0F5_7EE3);
        assert_eq!(murmur3(b"", 0), 0);
        assert_eq!(murmur3(b"", SEED), 0xE7F7_41F1);
        let sixteen: Vec<u8> = (0..16u8).collect();
        assert_eq!(murmur3(&sixteen, SEED), 0x31CE_0919);
        // the weight-stream vector: 1024 bytes data[i] = (37i + 11) & 0x77
        let data: Vec<u8> = (0..1024u32).map(|i| ((37 * i + 11) & 0x77) as u8).collect();
        let h = stream_hash([0x299D_C586, 0x0027_0078, 2165, 0x0F78_7978], &data);
        assert_eq!(h, 0x1264_85B2);
        assert_eq!((1u32 << 23) + (h & 0x7F_FFFF), 0x00E4_85B2);
        // the scale-stream vector: 5 bytes, which exercise the tail
        let h = stream_hash([0x299D_C586, 0x0027_0078, 719, 0x0F78_7978], &[0x7F, 0x7E, 0x80, 0x7F, 0x7D]);
        assert_eq!(h, 0x8A49_76D8);
        assert_eq!((1u32 << 23) + (h & 0x7F_FFFF), 0x00C9_76D8);
    }

    #[test]
    fn incremental_equals_one_shot() {
        let data: Vec<u8> = (0..64u8).collect();
        for n in 0..data.len() {
            for cut in 0..=n {
                let mut m = Murmur::new(SEED);
                m.update(&data[..cut]);
                m.update(&data[cut..n]);
                assert_eq!(m.finish(), murmur3(&data[..n], SEED));
            }
        }
    }

    #[test]
    fn fnv1a_known() {
        // the published FNV-1a 32-bit values of "" and "a"
        assert_eq!(fnv1a(b""), 0x811c_9dc5);
        assert_eq!(fnv1a(b"a"), 0xe40c_292c);
    }
}
