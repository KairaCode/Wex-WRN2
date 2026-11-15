// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! The WNM2 model blob (WNR3-SPEC erratum 12, WNR2-REFERENCE section 7) and the WNR3 record decoder
//! (WNR3-SPEC sections 1, 3 and 4, errata 9 to 11).

use crate::hash::{fnv1a, stream_hash};
use crate::rans::{cdf_ok, Stream, L};
use crate::wex4::align;

pub const WNM2_BYTES: usize = 59_408;
pub const WNR3_MAGIC: u32 = 0x3352_4E57;
const CONTEXTS: usize = 567;
const WEIGHT_TABLES: usize = 32;
const SCALE_TABLES: usize = 57_866;

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// A checked WNM2 model: 3 x 567 weight CDFs of 17 values, 3 scale CDFs of 257 values.
pub struct Model {
    pub id: u32,
    weight: Vec<u16>,
    scale: Vec<u16>,
}

pub fn model_open(blob: &[u8]) -> Result<Model, String> {
    if blob.len() != WNM2_BYTES {
        return Err(format!("a WNM2 model has {} bytes, this one {}", WNM2_BYTES, blob.len()));
    }
    let want = [0x324D_4E57, 1, 14, CONTEXTS as u32, 3, 0, 0, 0];
    for (i, w) in want.iter().enumerate() {
        if u32le(blob, 4 * i) != *w {
            return Err(format!("WNM2 header word {} is {:#x}, not {:#x}", i, u32le(blob, 4 * i), w));
        }
    }
    let weight: Vec<u16> = (0..3 * CONTEXTS * 17).map(|i| u16le(blob, WEIGHT_TABLES + 2 * i)).collect();
    let scale: Vec<u16> = (0..3 * 257).map(|i| u16le(blob, SCALE_TABLES + 2 * i)).collect();
    for (t, cdf) in weight.chunks(17).enumerate() {
        if !cdf_ok(cdf) {
            return Err(format!(
                "WNM2 weight table {} (matrix {}, context {}) is not a CDF",
                t,
                t / CONTEXTS,
                t % CONTEXTS
            ));
        }
    }
    for (m, cdf) in scale.chunks(257).enumerate() {
        if !cdf_ok(cdf) {
            return Err(format!("WNM2 scale table of matrix {} is not a CDF", m));
        }
    }
    Ok(Model { id: fnv1a(blob), weight, scale })
}

fn div_ceil(a: u64, b: u64) -> u64 {
    a / b + u64::from(a % b != 0)
}

fn chunk_ok(c: u32, lo: u32) -> bool {
    c.is_power_of_two() && (lo..=16384).contains(&c)
}

/// D(a) = a + min(a, 2), the inverse of the halving of section 4.2.
fn double(a: u8) -> u8 {
    a + a.min(2)
}

/// Decodes one WNR3 record (`rec`: its stored bytes) into `out` (the slab: scales of w1, w2, w3, then
/// their packed FP4 weights). `scales[m]` is rows x cols of the scale tensor of matrix m. On an error
/// `out` holds garbage and must not be used; the caller decodes into scratch.
pub fn decode(rec: &[u8], model: &Model, scales: [u64; 3], out: &mut [u8]) -> Result<(), String> {
    if rec.len() < 32 {
        return Err("a WNR3 record is shorter than its 32-byte header".into());
    }
    if u32le(rec, 0) != WNR3_MAGIC {
        return Err(format!("record magic {:#010x} is not WNR3", u32le(rec, 0)));
    }
    let model_id = u32le(rec, 4);
    if model_id == 0 || model_id != model.id {
        return Err(format!("record names model {:#010x}, the blob is {:#010x}", model_id, model.id));
    }
    let sp = u32le(rec, 8) as u64;
    let wp = u32le(rec, 12) as u64;
    let cw = u32le(rec, 16);
    let cs = u32le(rec, 20);
    if !chunk_ok(cw, 512) || !chunk_ok(cs, 256) {
        return Err(format!("chunks {}:{} are not powers of 2 in 512..16384 and 256..16384", cw, cs));
    }
    let anchor = [rec[24], rec[25], rec[26]];
    let flags = rec[27];
    if flags & 0xF0 != 0 {
        return Err(format!("record flags {:#04x} set bits 4 to 7", flags));
    }
    let owner = u32le(rec, 28);
    let word24 = u32le(rec, 24);
    let cz = flags & 0x08 != 0;
    // erratum 11: S and W streams, the last stream of a matrix holds what is left
    let counts: Vec<u64> = scales.iter().map(|&s| 32 * s).collect();
    let ns: u64 = scales.iter().map(|&s| div_ceil(s, cs as u64)).sum();
    let nw: u64 = counts.iter().map(|&c| div_ceil(c, cw as u64)).sum();
    let table_end = 32 + 2 * (ns + nw);
    if table_end > rec.len() as u64 {
        return Err("the stream-length table runs past the record".into());
    }
    let lengths: Vec<u64> = (0..(ns + nw) as usize).map(|g| u16le(rec, 32 + 2 * g) as u64).collect();
    if let Some(g) = lengths.iter().position(|&n| n < 4) {
        return Err(format!("stream {} has {} bytes, less than 4", g, lengths[g]));
    }
    let sum_s: u64 = lengths[..ns as usize].iter().sum();
    let sum_w: u64 = lengths[ns as usize..].iter().sum();
    if sum_s != sp || sum_w != wp {
        return Err(format!("stream lengths add up to {}:{}, the header says {}:{}", sum_s, sum_w, sp, wp));
    }
    let used = table_end + sp + wp;
    let total = rec.len() as u64;
    if total != align(used) {
        return Err(format!("the record has {} bytes, align({}) is {}", total, used, align(used)));
    }
    if rec[used as usize..].iter().any(|&b| b != 0) {
        return Err("a non-zero byte in the padding of the record".into());
    }
    let s_total: u64 = scales.iter().sum();
    if out.len() as u64 != 17 * s_total {
        return Err("the slab size does not match the expert tensors".into());
    }
    for b in out.iter_mut() {
        *b = 0;
    }
    let mut s_base = [0usize; 3];
    let mut w_base = [0usize; 3];
    let mut sb = 0u64;
    let mut wb = s_total;
    for m in 0..3 {
        s_base[m] = sb as usize;
        w_base[m] = wb as usize;
        sb += scales[m];
        wb += 16 * scales[m];
    }
    let mut pos = table_end as usize;
    let mut g = 0usize;
    // scale streams of m0, m1, m2: order-0 with the scale CDF of the matrix
    for m in 0..3 {
        let cdf = &model.scale[257 * m..257 * (m + 1)];
        let n = scales[m];
        let mut first = 0u64;
        while first < n {
            let count = (n - first).min(cs as u64) as usize;
            let len = lengths[g] as usize;
            let mut st = Stream::open(&rec[pos..pos + len]).ok_or(format!("scale stream {}: initial state", g))?;
            let at = s_base[m] + first as usize;
            for k in 0..count {
                let s = st.symbol(cdf).ok_or(format!("scale stream {} runs out of bytes", g))?;
                out[at + k] = s as u8;
            }
            let h = stream_hash([model_id, owner, g as u32, word24], &out[at..at + count]);
            if !st.done(L + (h & 0x7F_FFFF)) {
                return Err(format!("scale stream {}: length or final state (hash) wrong", g));
            }
            pos += len;
            g += 1;
            first += count as u64;
        }
    }
    // weight streams of m0, m1, m2: 567 contexts per matrix
    for m in 0..3 {
        let count_m = counts[m];
        let mut first = 0u64;
        while first < count_m {
            let count = (count_m - first).min(cw as u64);
            let len = lengths[g] as usize;
            let mut st = Stream::open(&rec[pos..pos + len]).ok_or(format!("weight stream {}: initial state", g))?;
            let (mut pm, mut mx, mut srel) = (8usize, 8usize, 0usize);
            for k in first..first + count {
                if k % 32 == 0 {
                    let cs_b = out[s_base[m] + (k / 32) as usize] as i32;
                    srel = ((cs_b - anchor[m] as i32).clamp(-3, 3) + 3) as usize;
                    pm = 8;
                    mx = 8;
                }
                let ctx = (srel * 9 + pm) * 9 + mx;
                let t = (m * CONTEXTS + ctx) * 17;
                let x = st.symbol(&model.weight[t..t + 17]).ok_or(format!("weight stream {} runs out of bytes", g))?;
                if cz && x == 8 {
                    return Err(format!("weight stream {}: nibble 8 (-0) in a canonical-zero record", g));
                }
                let a = x & 7;
                pm = a;
                if mx == 8 || a > mx {
                    mx = a;
                }
                out[w_base[m] + (k / 2) as usize] |= (x as u8) << (4 * (k & 1));
            }
            let at = w_base[m] + (first / 2) as usize;
            let h = stream_hash([model_id, owner, g as u32, word24], &out[at..at + (count / 2) as usize]);
            if !st.done(L + (h & 0x7F_FFFF)) {
                return Err(format!("weight stream {}: length or final state (hash) wrong", g));
            }
            pos += len;
            g += 1;
            first += count;
        }
    }
    // section 4.2, the inverse of ceil form, after every stream: blocks whose coded maximum is 5
    for m in 0..3 {
        if flags & (1 << m) == 0 {
            continue;
        }
        for b in 0..scales[m] as usize {
            let w = &mut out[w_base[m] + 16 * b..w_base[m] + 16 * b + 16];
            let max = w.iter().map(|&v| (v & 7).max((v >> 4) & 7)).max().unwrap_or(0);
            if max != 5 {
                continue;
            }
            let s = &mut out[s_base[m] + b];
            if *s == 0 {
                return Err(format!("matrix {} block {}: ceil form with coded scale 0", m, b));
            }
            *s -= 1;
            let w = &mut out[w_base[m] + 16 * b..w_base[m] + 16 * b + 16];
            for v in w.iter_mut() {
                let lo = (*v & 8) | double(*v & 7);
                let hi = ((*v >> 4) & 8) | double((*v >> 4) & 7);
                *v = lo | (hi << 4);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_inverts_halving() {
        // H: 0->0, 2->1, 4->2, 5->3, 6->4, 7->5 (section 4.2)
        for (a, h) in [(0u8, 0u8), (2, 1), (4, 2), (5, 3), (6, 4), (7, 5)] {
            assert_eq!(double(h), a);
        }
    }

    #[test]
    fn model_checks() {
        let mut blob = vec![0u8; WNM2_BYTES];
        for (i, w) in [0x324D_4E57u32, 1, 14, 567, 3, 0, 0, 0].iter().enumerate() {
            blob[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let put = |blob: &mut Vec<u8>, at: usize, n: usize| {
            for s in 0..n {
                let v = if s == n - 1 { 16384 } else { (s * 16384 / (n - 1)) as u16 };
                blob[at + 2 * s..at + 2 * s + 2].copy_from_slice(&v.to_le_bytes());
            }
        };
        for t in 0..3 * 567 {
            put(&mut blob, 32 + 34 * t, 17);
        }
        for m in 0..3 {
            put(&mut blob, 57_866 + 514 * m, 257);
        }
        let model = model_open(&blob).unwrap();
        assert_eq!(model.id, fnv1a(&blob));
        let mut bad = blob.clone();
        bad[32 + 34 * 100 + 2] = 0; // second value of a weight table: 0, not increasing
        bad[32 + 34 * 100 + 3] = 0;
        assert!(model_open(&bad).is_err());
        assert!(model_open(&blob[..WNM2_BYTES - 1]).is_err());
    }
}
