// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! Test-only WNR4 encoder pieces, written from docs/WNR4-SPEC.md like the decoder: the tile encoder of
//! section 4 (every coded symbol in exact reverse decode order, 16-bit words, L = 2^16), the end states
//! the encoder starts from, a record assembler for section 3, random tables and a small PRNG. A test that
//! passes with them shows that this encoder and this decoder agree, not that they agree with the reference.

use super::core::{HEADER_BYTES, L, PREC, T};

/// splitmix64, so the tests need no crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in 0..n (n > 0).
    pub fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

/// A valid 14-bit cumulative table of n symbols with random, skewed frequencies.
pub fn random_cdf(rng: &mut Rng, n: usize) -> Vec<u16> {
    let w: Vec<u64> = (0..n).map(|_| 1 + (rng.next() % 64).pow(2)).collect();
    cdf_from(&w)
}

/// A table of n symbols from counts: every symbol at least 1, the rest in proportion, the remainder to the
/// most frequent symbol.
pub fn cdf_from(counts: &[u64]) -> Vec<u16> {
    let n = counts.len() as u64;
    let total: u64 = counts.iter().sum::<u64>().max(1);
    let spare = T as u64 - n;
    let mut f: Vec<u64> = counts.iter().map(|&c| 1 + c * spare / total).collect();
    let used: u64 = f.iter().sum();
    let top = (0..counts.len()).max_by_key(|&i| counts[i]).unwrap_or(0);
    f[top] += T as u64 - used;
    let mut cdf = vec![0u16; counts.len() + 1];
    for i in 0..counts.len() {
        cdf[i + 1] = cdf[i] + f[i] as u16;
    }
    cdf
}

/// The table's bytes, little endian.
pub fn table_bytes(cdf: &[u16]) -> Vec<u8> {
    cdf.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The states a tile's decoder must end at (section 4): lane 0 at L + (H & 0xFFFF), lane 1 at L + (H >> 16),
/// lanes 2..31 at L + their 16 tail bits, or L for a lane without items (None).
pub fn end_states(h: u32, tails: &[Option<u16>; 32]) -> [u32; 32] {
    let mut s = [L; 32];
    s[0] = L + (h & 0xFFFF);
    s[1] = L + (h >> 16);
    for l in 2..32 {
        s[l] = L + tails[l].unwrap_or(0) as u32;
    }
    s
}

/// One tile: `events` are (lane, table, symbol) in decode order; each lane starts at `ends[lane]` and the
/// symbols are encoded last first. Returns the 32 final states (the decoder's initial states), then the
/// words in the order the decoder reads them.
pub fn encode_tile(events: &[(usize, &[u16], usize)], ends: &[u32; 32]) -> Vec<u8> {
    let mut x: Vec<u64> = ends.iter().map(|&s| s as u64).collect();
    let mut rev: Vec<u16> = Vec::new();
    for &(l, cdf, s) in events.iter().rev() {
        let start = cdf[s] as u64;
        let freq = cdf[s + 1] as u64 - start;
        // the state stays in [L, 2^32): renormalise when encoding would leave it
        let x_max = ((L as u64) >> PREC << 16) * freq;
        if x[l] >= x_max {
            rev.push(x[l] as u16);
            x[l] >>= 16;
        }
        x[l] = ((x[l] / freq) << PREC) + x[l] % freq + start;
    }
    let mut out = Vec::with_capacity(128 + 2 * rev.len());
    for &s in &x {
        out.extend_from_slice(&(s as u32).to_le_bytes());
    }
    for w in rev.iter().rev() {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// A record: the 64-byte header `hdr` (bytes 20..27 are set here: ntiles, P), the offset table, the kind's
/// area `extra`, the tiles, and zero padding to a multiple of 4096.
pub fn record(hdr: &[u8; HEADER_BYTES], extra: &[u8], tiles: &[Vec<u8>]) -> Vec<u8> {
    let mut r = hdr.to_vec();
    let p: usize = tiles.iter().map(|t| t.len()).sum();
    r[20..24].copy_from_slice(&(tiles.len() as u32).to_le_bytes());
    r[24..28].copy_from_slice(&(p as u32).to_le_bytes());
    let mut off = 0u32;
    r.extend_from_slice(&off.to_le_bytes());
    for t in tiles {
        off += t.len() as u32;
        r.extend_from_slice(&off.to_le_bytes());
    }
    r.extend_from_slice(extra);
    for t in tiles {
        r.extend_from_slice(t);
    }
    while r.len() % 4096 != 0 {
        r.push(0);
    }
    r
}
