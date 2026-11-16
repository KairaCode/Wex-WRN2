// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! Test-only encoders for WBP1 and WNR3 records, written from the same pages as the decoders. They make
//! the records of the round-trip and corruption tests below; a test that passes shows that encoder and
//! decoder agree with each other, not with the reference (the vectors are for that). Not canonical: the
//! frequency tables are simpler than `wex_nr2_quantize`.
#![allow(clippy::needless_range_loop)]

use crate::hash::stream_hash;
use crate::rans::{L, TOTAL};
use crate::wbp1::{self, WBP1_MAGIC};
use crate::wex4::align;
use crate::wnr3::{self, WNM2_BYTES, WNR3_MAGIC};

/// rANS encoding of `symbols`, symbol i with table `cdfs[i]`, last symbol first, from state `x0`.
pub fn rans_encode(symbols: &[usize], cdfs: &[&[u16]], x0: u32) -> Vec<u8> {
    let mut x = x0;
    let mut rev = Vec::new();
    for (i, &s) in symbols.iter().enumerate().rev() {
        let start = cdfs[i][s] as u32;
        let freq = cdfs[i][s + 1] as u32 - start;
        let x_max = ((L >> 14) << 8) * freq;
        while x >= x_max {
            rev.push(x as u8);
            x >>= 8;
        }
        x = ((x / freq) << 14) + (x % freq) + start;
    }
    let mut out = x.to_be_bytes().to_vec();
    out.extend(rev.iter().rev());
    out
}

/// A CDF of `n` symbols from counts: every symbol at least 1, the rest in proportion, the remainder
/// to the most frequent symbol.
pub fn cdf_from(counts: &[u64]) -> Vec<u16> {
    let n = counts.len() as u64;
    let total: u64 = counts.iter().sum::<u64>().max(1);
    let spare = TOTAL as u64 - n;
    let mut f: Vec<u64> = counts.iter().map(|&c| 1 + c * spare / total).collect();
    let used: u64 = f.iter().sum();
    let top = (0..counts.len()).max_by_key(|&i| counts[i]).unwrap_or(0);
    f[top] += TOTAL as u64 - used;
    let mut cdf = vec![0u16; counts.len() + 1];
    for i in 0..counts.len() {
        cdf[i + 1] = cdf[i] + f[i] as u16;
    }
    cdf
}

/// A WBP1 record of `data` with the given parameters; `mask` bit p = 1 codes plane p with rANS.
pub fn wbp1_record(data: &[u8], owner: u32, planes: usize, chunk_log2: u8, mask: u8, transform: u8) -> Vec<u8> {
    let mut coded = data.to_vec();
    if transform == 1 {
        for e in coded.chunks_exact_mut(planes) {
            let top = e[planes - 1] >> 7;
            for i in (0..planes).rev() {
                let carry = if i > 0 { e[i - 1] >> 7 } else { top };
                e[i] = (e[i] << 1) | carry;
            }
        }
    }
    let c = 1usize << chunk_log2;
    let id3 = planes as u32 | (chunk_log2 as u32) << 8 | (transform as u32) << 24;
    let (mut cdfs, mut tables, mut payloads) = (Vec::new(), Vec::new(), Vec::new());
    let mut g = 0u32;
    for p in 0..planes {
        let plane: Vec<u8> = coded.iter().skip(p).step_by(planes).copied().collect();
        if mask & (1 << p) != 0 {
            let mut counts = vec![0u64; 256];
            for &b in &plane {
                counts[b as usize] += 1;
            }
            let cdf = cdf_from(&counts);
            for v in &cdf {
                cdfs.extend_from_slice(&v.to_le_bytes());
            }
            for chunk in plane.chunks(c) {
                let h = stream_hash([WBP1_MAGIC, owner, g, id3], chunk);
                let syms: Vec<usize> = chunk.iter().map(|&b| b as usize).collect();
                let tabs: Vec<&[u16]> = vec![&cdf[..]; syms.len()];
                let s = rans_encode(&syms, &tabs, L + (h & 0x7F_FFFF));
                tables.extend_from_slice(&(s.len() as u16).to_le_bytes());
                payloads.extend_from_slice(&s);
                g += 1;
            }
        } else {
            for chunk in plane.chunks(c) {
                tables.extend_from_slice(&stream_hash([WBP1_MAGIC, owner, g, id3], chunk).to_le_bytes());
                payloads.extend_from_slice(chunk);
                g += 1;
            }
        }
    }
    let mut rec = Vec::new();
    rec.extend_from_slice(&WBP1_MAGIC.to_le_bytes());
    rec.extend_from_slice(&owner.to_le_bytes());
    rec.extend_from_slice(&(data.len() as u32).to_le_bytes());
    rec.extend_from_slice(&[planes as u8, chunk_log2, mask, transform]);
    rec.extend(cdfs);
    rec.extend(tables);
    rec.extend(payloads);
    rec.resize(align(rec.len() as u64) as usize, 0);
    rec
}

/// A valid WNM2 blob: weight tables skewed towards small magnitudes, scale tables uniform.
pub fn wnm2_blob() -> Vec<u8> {
    let mut blob = vec![0u8; WNM2_BYTES];
    for (i, w) in [0x324D_4E57u32, 1, 14, 567, 3, 0, 0, 0].iter().enumerate() {
        blob[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    for t in 0..3 * 567 {
        let counts: Vec<u64> = (0..16u64).map(|s| 1 + (16 - (s & 7)) * (1 + (t as u64 % 5))).collect();
        for (k, v) in cdf_from(&counts).iter().enumerate() {
            blob[32 + 34 * t + 2 * k..32 + 34 * t + 2 * k + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    for m in 0..3 {
        let counts: Vec<u64> = (0..256u64).map(|s| 1 + (s % 7)).collect();
        for (k, v) in cdf_from(&counts).iter().enumerate() {
            let at = 57_866 + 514 * m + 2 * k;
            blob[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    blob
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// A WNR3 record of already coded scales and nibbles (canonical zero and ceil form applied by the
/// caller), with the anchors and flags given: WNR3-SPEC section 4.6, steps 4 and 5.
#[allow(clippy::too_many_arguments)]
pub fn wnr3_coded(
    blob: &[u8],
    scales: &[Vec<u8>; 3],
    weights: &[Vec<u8>; 3],
    anchor: [u8; 3],
    flags: u8,
    owner: u32,
    cw: usize,
    cs: usize,
) -> Vec<u8> {
    let model_id = crate::hash::fnv1a(blob);
    let word24 = u32::from_le_bytes([anchor[0], anchor[1], anchor[2], flags]);
    let wtab = |m: usize, ctx: usize| -> Vec<u16> {
        (0..17).map(|k| u16le(blob, 32 + 34 * (m * 567 + ctx) + 2 * k)).collect()
    };
    let stab = |m: usize| -> Vec<u16> { (0..257).map(|k| u16le(blob, 57_866 + 514 * m + 2 * k)).collect() };
    let (mut lengths, mut sp, mut wp) = (Vec::new(), Vec::new(), Vec::new());
    let mut g = 0u32;
    for m in 0..3 {
        let t = stab(m);
        for chunk in scales[m].chunks(cs) {
            let h = stream_hash([model_id, owner, g, word24], chunk);
            let syms: Vec<usize> = chunk.iter().map(|&b| b as usize).collect();
            let tabs: Vec<&[u16]> = vec![&t[..]; syms.len()];
            let s = rans_encode(&syms, &tabs, L + (h & 0x7F_FFFF));
            lengths.push(s.len() as u16);
            sp.extend(s);
            g += 1;
        }
    }
    for m in 0..3 {
        let nib: Vec<usize> =
            (0..2 * weights[m].len()).map(|k| ((weights[m][k / 2] >> (4 * (k & 1))) & 15) as usize).collect();
        for (j, chunk) in nib.chunks(cw).enumerate() {
            let first = j * cw;
            let mut ctxs = Vec::with_capacity(chunk.len());
            let (mut pm, mut mx, mut srel) = (8usize, 8usize, 0usize);
            for (i, &x) in chunk.iter().enumerate() {
                let k = first + i;
                if k % 32 == 0 {
                    let cs_b = scales[m][k / 32] as i32;
                    srel = ((cs_b - anchor[m] as i32).clamp(-3, 3) + 3) as usize;
                    pm = 8;
                    mx = 8;
                }
                ctxs.push(wtab(m, (srel * 9 + pm) * 9 + mx));
                let a = x & 7;
                pm = a;
                if mx == 8 || a > mx {
                    mx = a;
                }
            }
            let bytes = &weights[m][first / 2..(first + chunk.len()) / 2];
            let h = stream_hash([model_id, owner, g, word24], bytes);
            let tabs: Vec<&[u16]> = ctxs.iter().map(|c| &c[..]).collect();
            let s = rans_encode(chunk, &tabs, L + (h & 0x7F_FFFF));
            lengths.push(s.len() as u16);
            wp.extend(s);
            g += 1;
        }
    }
    let mut rec = Vec::new();
    for w in [WNR3_MAGIC, model_id, sp.len() as u32, wp.len() as u32, cw as u32, cs as u32] {
        rec.extend_from_slice(&w.to_le_bytes());
    }
    rec.extend_from_slice(&[anchor[0], anchor[1], anchor[2], flags]);
    rec.extend_from_slice(&owner.to_le_bytes());
    for l in lengths {
        rec.extend_from_slice(&l.to_le_bytes());
    }
    rec.extend(sp);
    rec.extend(wp);
    rec.resize(align(rec.len() as u64) as usize, 0);
    rec
}

/// Section 4.3, rule g: the v in 0..254 with the most scales equal to v or v + 1, first maximum.
pub fn rule_g(scales: &[u8]) -> u8 {
    let mut h = [0u64; 256];
    for &s in scales {
        h[s as usize] += 1;
    }
    let mut best = 0usize;
    for v in 1..255 {
        if h[v] + h[v + 1] > h[best] + h[best + 1] {
            best = v;
        }
    }
    best as u8
}

/// Sections 4.4 and 4.2 forward on a copy of a source slab: Z when `cz`, then ceil form for every
/// matrix of `want_ceil` that passes the gate. Returns the coded matrices and the flags.
pub fn wnr3_forward(
    scales: &[Vec<u8>; 3],
    weights: &[Vec<u8>; 3],
    cz: bool,
    want_ceil: u8,
) -> ([Vec<u8>; 3], [Vec<u8>; 3], u8) {
    let mut s = scales.clone();
    let mut w = weights.clone();
    let mut flags = if cz { 8 } else { 0 };
    for m in 0..3 {
        if cz {
            for b in w[m].iter_mut() {
                let lo = if *b & 15 == 8 { 0 } else { *b & 15 };
                let hi = if *b >> 4 == 8 { 0 } else { *b >> 4 };
                *b = lo | hi << 4;
            }
        }
        let mag = |v: u8, half: usize| (v >> (4 * half)) & 7;
        let max5 = w[m].chunks(16).any(|blk| blk.iter().map(|&v| mag(v, 0).max(mag(v, 1))).max() == Some(5));
        if want_ceil & (1 << m) == 0 || max5 {
            continue;
        }
        flags |= 1 << m;
        for (b, blk) in w[m].chunks_mut(16).enumerate() {
            let mags: Vec<u8> = blk.iter().flat_map(|&v| [mag(v, 0), mag(v, 1)]).collect();
            let even = mags.iter().max() == Some(&7) && !mags.iter().any(|&a| a == 1 || a == 3);
            if !even || s[m][b] > 254 {
                continue;
            }
            s[m][b] += 1;
            let half = |a: u8| [0, 0, 1, 0, 2, 3, 4, 5][a as usize];
            for v in blk.iter_mut() {
                let lo = (*v & 8) | half(*v & 7);
                let hi = ((*v >> 4) & 8) | half((*v >> 4) & 7);
                *v = lo | hi << 4;
            }
        }
    }
    (s, w, flags)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 56) as u8
        }
    }

    /// Bytes with a skewed distribution, so that rANS planes are worth coding.
    fn data(n: usize, seed: u64) -> Vec<u8> {
        let mut r = Rng(seed | 1);
        (0..n).map(|_| if r.byte() < 200 { r.byte() & 7 } else { r.byte() }).collect()
    }

    #[test]
    fn wbp1_round_trips() {
        let mut cases = 0;
        for &planes in &[1usize, 2, 4, 8] {
            for &n in &[1usize, 7, 1000, 4096, 9001, 70000] {
                for &chunk_log2 in &[10u8, 12, 15] {
                    for transform in 0..2u8 {
                        if transform == 1 && (planes == 1 || n % planes != 0) {
                            continue;
                        }
                        let d = data(n, (n * 31 + planes) as u64);
                        let full = if planes == 8 { 0xff } else { (1u8 << planes) - 1 };
                        for mask in [0u8, full, 0x55 & full, 0xAA & full] {
                            // a coded plane must not be empty
                            if (0..planes).any(|p| mask & (1 << p) != 0 && n <= p) {
                                continue;
                            }
                            let rec = wbp1_record(&d, 9, planes, chunk_log2, mask, transform);
                            let mut out = vec![0u8; n];
                            wbp1::decode(&rec, 9, n as u32, &mut out).unwrap();
                            assert_eq!(
                                out, d,
                                "P {} n {} chunk {} transform {} mask {}",
                                planes, n, chunk_log2, transform, mask
                            );
                            assert!(wbp1::decode(&rec, 8, n as u32, &mut out).is_err(), "another owner");
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 200);
    }

    #[test]
    fn wbp1_refuses_damage() {
        let d = data(20000, 5);
        let rec = wbp1_record(&d, 3, 2, 10, 0x3, 1);
        let used = rec.iter().rposition(|&b| b != 0).unwrap() + 1;
        let mut out = vec![0u8; d.len()];
        let mut r = Rng(77);
        for _ in 0..300 {
            let mut bad = rec.clone();
            let at = 16 + (r.next() as usize) % (used - 16);
            bad[at] ^= 1 << (r.next() % 8);
            match wbp1::decode(&bad, 3, d.len() as u32, &mut out) {
                Err(_) => {}
                Ok(()) => assert_eq!(out, d, "a flip at {} decoded to other bytes", at),
            }
        }
        // a whole page of zeros too many
        let mut long = rec.clone();
        long.extend(vec![0u8; 4096]);
        assert!(wbp1::decode(&long, 3, d.len() as u32, &mut out).is_err());
        // a non-zero byte in the padding
        let mut pad = rec.clone();
        *pad.last_mut().unwrap() = 1;
        assert!(wbp1::decode(&pad, 3, d.len() as u32, &mut out).is_err());
    }

    /// A synthetic expert: matrix m has rows x cols scales. Blocks are chosen to hit every case of
    /// ceil form: even blocks (max 7, no 1 or 3), at scale 254 and 255, blocks of max 0 to 4, -0 nibbles.
    fn expert(rows: usize, cols: [usize; 3], seed: u64, max5_in: Option<usize>) -> ([Vec<u8>; 3], [Vec<u8>; 3]) {
        let mut r = Rng(seed | 1);
        let mut s: [Vec<u8>; 3] = Default::default();
        let mut w: [Vec<u8>; 3] = Default::default();
        for m in 0..3 {
            let n = rows * cols[m];
            for b in 0..n {
                let kind = r.next() % 6;
                s[m].push(match b % 9 {
                    0 => 254,
                    1 => 255,
                    2 => 0,
                    _ => 118 + (r.next() % 8) as u8,
                });
                let mut nib = [0u8; 32];
                for q in nib.iter_mut() {
                    let a = match kind {
                        0 | 1 => [0u8, 2, 4, 5, 6, 7][(r.next() % 6) as usize], // even when a 7 is present
                        2 => (r.next() % 5) as u8,                              // max 0..4
                        3 => 0,
                        _ => (r.next() % 8) as u8, // anything, often with 1 or 3
                    };
                    let sign = if r.next() % 4 == 0 { 8 } else { 0 };
                    *q = a | sign;
                }
                if kind == 0 {
                    nib[(r.next() % 32) as usize] = 7;
                }
                if kind == 1 || kind >= 4 {
                    // never a block maximum of 5 by chance: that would turn ceil form off for the matrix
                    nib[1] = 6;
                }
                if max5_in == Some(m) && b == n / 2 {
                    nib = [1; 32];
                    nib[3] = 5;
                }
                for k in 0..16 {
                    w[m].push(nib[2 * k] | nib[2 * k + 1] << 4);
                }
            }
        }
        (s, w)
    }

    fn slab(s: &[Vec<u8>; 3], w: &[Vec<u8>; 3]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in s {
            out.extend_from_slice(v);
        }
        for v in w {
            out.extend_from_slice(v);
        }
        out
    }

    fn z(w: &[Vec<u8>; 3]) -> [Vec<u8>; 3] {
        let mut w = w.clone();
        for v in w.iter_mut().flat_map(|m| m.iter_mut()) {
            let lo = if *v & 15 == 8 { 0 } else { *v & 15 };
            let hi = if *v >> 4 == 8 { 0 } else { *v >> 4 };
            *v = lo | hi << 4;
        }
        w
    }

    #[test]
    fn wnr3_round_trips() {
        let blob = wnm2_blob();
        let model = wnr3::model_open(&blob).unwrap();
        let mut ceil_blocks = 0;
        for (i, &(rows, cols, cw, cs)) in
            [(3usize, [5usize, 2, 7], 512usize, 256usize), (40, [33, 9, 17], 2048, 512), (17, [64, 64, 3], 4096, 16384)]
                .iter()
                .enumerate()
        {
            for cz in [false, true] {
                for max5 in [None, Some(1)] {
                    let (s, w) = expert(rows, cols, 100 + i as u64, max5);
                    let (cs_, cw_, flags) = wnr3_forward(&s, &w, cz, 7);
                    let want_flags = (if cz { 8 } else { 0 }) | if max5 == Some(1) { 5 } else { 7 };
                    assert_eq!(flags, want_flags);
                    ceil_blocks +=
                        (0..3).map(|m| s[m].iter().zip(&cs_[m]).filter(|(a, b)| a != b).count()).sum::<usize>();
                    let anchor = [rule_g(&cs_[0]), rule_g(&cs_[1]), rule_g(&cs_[2])];
                    let owner = 0x0003_0007;
                    let rec = wnr3_coded(&blob, &cs_, &cw_, anchor, flags, owner, cw, cs);
                    let scount = [(rows * cols[0]) as u64, (rows * cols[1]) as u64, (rows * cols[2]) as u64];
                    let mut out = vec![0u8; 17 * scount.iter().sum::<u64>() as usize];
                    wnr3::decode(&rec, &model, scount, &mut out).unwrap();
                    let want = if cz { slab(&s, &z(&w)) } else { slab(&s, &w) };
                    assert_eq!(out, want, "case {} cz {} max5 {:?}", i, cz, max5);
                    // the same record under another owner, anchor or flag fails its stream hashes
                    let mut other = rec.clone();
                    other[28] ^= 1;
                    assert!(wnr3::decode(&other, &model, scount, &mut out).is_err());
                    let mut other = rec.clone();
                    other[24] = other[24].wrapping_add(1);
                    assert!(wnr3::decode(&other, &model, scount, &mut out).is_err());
                    let mut other = rec.clone();
                    other[27] ^= 2;
                    assert!(wnr3::decode(&other, &model, scount, &mut out).is_err());
                    let mut other = rec.clone();
                    other[27] |= 0x10;
                    assert!(wnr3::decode(&other, &model, scount, &mut out).is_err());
                }
            }
        }
        assert!(ceil_blocks > 100, "only {} blocks in ceil form", ceil_blocks);
    }

    #[test]
    fn wnr3_refuses_what_section_4_forbids() {
        let blob = wnm2_blob();
        let model = wnr3::model_open(&blob).unwrap();
        let (rows, cols) = (4usize, [3usize, 3, 3]);
        let scount = [12u64, 12, 12];
        let (s, w) = expert(rows, cols, 9, None);
        let mut out = vec![0u8; 17 * 36];
        let anchor = [120, 120, 120];
        // a CZ record whose coded data still holds -0 (nibble 8)
        let mut w8 = w.clone();
        w8[2][5] = 0x08;
        let rec = wnr3_coded(&blob, &s, &w8, anchor, 0x08, 1, 512, 256);
        assert!(wnr3::decode(&rec, &model, scount, &mut out).is_err());
        let rec = wnr3_coded(&blob, &s, &w8, anchor, 0x00, 1, 512, 256);
        assert!(wnr3::decode(&rec, &model, scount, &mut out).is_ok());
        // a ceil-form matrix with a block of coded maximum 5 at coded scale 0
        let mut s0 = s.clone();
        let mut w5 = w.clone();
        s0[0][0] = 0;
        for v in w5[0][..16].iter_mut() {
            *v = 0x51;
        }
        let rec = wnr3_coded(&blob, &s0, &w5, anchor, 0x01, 1, 512, 256);
        assert!(wnr3::decode(&rec, &model, scount, &mut out).is_err());
        s0[0][0] = 1;
        let rec = wnr3_coded(&blob, &s0, &w5, anchor, 0x01, 1, 512, 256);
        wnr3::decode(&rec, &model, scount, &mut out).unwrap();
        assert_eq!(out[0], 0, "scale 1 - 1");
        assert_eq!(out[36], 0x72, "nibbles 1 and 5 doubled to 2 and 7");
        // damage anywhere in the streams: refused, or the same bytes
        let rec = wnr3_coded(&blob, &s, &w, anchor, 0, 1, 512, 256);
        wnr3::decode(&rec, &model, scount, &mut out).unwrap();
        let good = out.clone();
        let used = rec.iter().rposition(|&b| b != 0).unwrap() + 1;
        let mut r = Rng(3);
        for _ in 0..300 {
            let mut bad = rec.clone();
            let at = (r.next() as usize) % used;
            bad[at] ^= 1 << (r.next() % 8);
            if wnr3::decode(&bad, &model, scount, &mut out).is_ok() {
                assert_eq!(out, good, "a flip at {} decoded to other bytes", at);
            }
        }
    }
}
