// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! The WBP1 record decoder (WEX4-SPEC section 7.2): byte planes, each order-0 rANS or raw, with a
//! MurmurHash3 check per stream and per raw chunk.

use crate::hash::stream_hash;
use crate::rans::{cdf_ok, Stream, L};
use crate::wex4::align;

pub const WBP1_MAGIC: u32 = 0x3150_4257;

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn div_ceil(a: u64, b: u64) -> u64 {
    a / b + u64::from(a % b != 0)
}

/// Decodes the stored bytes `rec` of a WBP1 record whose index entry says `owner` and `decoded`
/// into `out` (`decoded` bytes). On an error `out` must not be used.
pub fn decode(rec: &[u8], owner: u32, decoded: u32, out: &mut [u8]) -> Result<(), String> {
    if rec.len() < 16 || out.len() != decoded as usize {
        return Err("record shorter than the WBP1 header".into());
    }
    if u32le(rec, 0) != WBP1_MAGIC {
        return Err(format!("record magic {:#010x} is not WBP1", u32le(rec, 0)));
    }
    if u32le(rec, 4) != owner {
        return Err(format!("record owner {} is not the index entry's {}", u32le(rec, 4), owner));
    }
    if u32le(rec, 8) != decoded {
        return Err(format!("record says {} decoded bytes, the index {}", u32le(rec, 8), decoded));
    }
    let planes = rec[12] as u64;
    let chunk_log2 = rec[13];
    let mask = rec[14];
    let transform = rec[15];
    if ![1, 2, 4, 8].contains(&planes) {
        return Err(format!("{} planes", planes));
    }
    if !(10..=15).contains(&chunk_log2) {
        return Err(format!("chunk_log2 {}", chunk_log2));
    }
    if planes < 8 && mask >> planes != 0 {
        return Err(format!("mask {:#04x} has a bit at or above plane {}", mask, planes));
    }
    let d = decoded as u64;
    match transform {
        0 => {}
        1 if planes > 1 && d % planes == 0 => {}
        _ => return Err(format!("transform {} with {} planes and {} bytes", transform, planes, d)),
    }
    let c = 1u64 << chunk_log2;
    let n: Vec<u64> = (0..planes).map(|p| if d > p { div_ceil(d - p, planes) } else { 0 }).collect();
    let s: Vec<u64> = n.iter().map(|&np| div_ceil(np, c)).collect();
    let coded = |p: u64| mask & (1 << p) != 0;
    for p in 0..planes {
        if coded(p) && n[p as usize] == 0 {
            return Err(format!("plane {} is rANS coded and empty", p));
        }
    }
    let len = rec.len() as u64;
    // the three groups: CDFs of the rANS planes, stream tables of all planes, payloads of all planes
    let mut pos = 16u64;
    let mut cdf_at = vec![0usize; planes as usize];
    for p in 0..planes {
        if coded(p) {
            if pos + 514 > len {
                return Err("the CDF tables run past the record".into());
            }
            cdf_at[p as usize] = pos as usize;
            pos += 514;
        }
    }
    let mut table_at = vec![0usize; planes as usize];
    for p in 0..planes {
        let width = if coded(p) { 2 } else { 4 };
        let bytes = width * s[p as usize];
        if bytes > len - pos {
            return Err("the stream tables run past the record".into());
        }
        table_at[p as usize] = pos as usize;
        pos += bytes;
    }
    let mut payload_at = vec![0usize; planes as usize];
    for p in 0..planes {
        let pi = p as usize;
        let bytes = if coded(p) {
            let mut sum = 0u64;
            for j in 0..s[pi] as usize {
                let l = u16le(rec, table_at[pi] + 2 * j) as u64;
                if l < 4 {
                    return Err(format!("plane {} stream {} has {} bytes, less than 4", p, j, l));
                }
                sum += l;
            }
            sum
        } else {
            n[pi]
        };
        if bytes > len - pos {
            return Err("the payloads run past the record".into());
        }
        payload_at[pi] = pos as usize;
        pos += bytes;
    }
    if len != align(pos) {
        return Err(format!("stored bytes {} are not align({}) = {}", len, pos, align(pos)));
    }
    if rec[pos as usize..].iter().any(|&b| b != 0) {
        return Err("a non-zero byte in the padding of the record".into());
    }
    let id3 = planes as u32 | (chunk_log2 as u32) << 8 | (transform as u32) << 24;
    let mut g = 0u32;
    let mut plane = vec![0u8; c as usize];
    for p in 0..planes as usize {
        let mut at = payload_at[p];
        let cdf: Vec<u16> = if coded(p as u64) {
            let t: Vec<u16> = (0..257).map(|i| u16le(rec, cdf_at[p] + 2 * i)).collect();
            if !cdf_ok(&t) {
                return Err(format!("the CDF of plane {} is not a CDF", p));
            }
            t
        } else {
            Vec::new()
        };
        for j in 0..s[p] as usize {
            let first = j as u64 * c;
            let count = (n[p] - first).min(c) as usize;
            let ident = [WBP1_MAGIC, owner, g, id3];
            let bytes = &mut plane[..count];
            if coded(p as u64) {
                let l = u16le(rec, table_at[p] + 2 * j) as usize;
                let mut st =
                    Stream::open(&rec[at..at + l]).ok_or(format!("plane {} stream {}: initial state", p, j))?;
                for b in bytes.iter_mut() {
                    *b = st.symbol(&cdf).ok_or(format!("plane {} stream {} runs out of bytes", p, j))? as u8;
                }
                if !st.done(L + (stream_hash(ident, bytes) & 0x7F_FFFF)) {
                    return Err(format!("plane {} stream {}: length or final state (hash) wrong", p, j));
                }
                at += l;
            } else {
                bytes.copy_from_slice(&rec[at..at + count]);
                if stream_hash(ident, bytes) != u32le(rec, table_at[p] + 4 * j) {
                    return Err(format!("plane {} raw chunk {}: hash wrong", p, j));
                }
                at += count;
            }
            for (i, &b) in bytes.iter().enumerate() {
                out[((first + i as u64) * planes) as usize + p] = b;
            }
            g += 1;
        }
    }
    if transform == 1 {
        rotate_right(out, planes as usize);
    }
    Ok(())
}

/// Undoes transform 1: every little-endian element of `w` bytes is rotated right by one bit.
fn rotate_right(data: &mut [u8], w: usize) {
    for e in data.chunks_exact_mut(w) {
        let low = e[0] & 1;
        for i in 0..w {
            let next = if i + 1 < w { e[i + 1] & 1 } else { low };
            e[i] = (e[i] >> 1) | (next << 7);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_right_undoes_rotate_left() {
        for v in [0x8001u16, 0x0001, 0x8000, 0x1234, 0xffff] {
            let mut e = v.rotate_left(1).to_le_bytes();
            rotate_right(&mut e, 2);
            assert_eq!(u16::from_le_bytes(e), v);
        }
        for v in [0x8000_0001u32, 0x3f80_0000, 0xdead_beef] {
            let mut e = v.rotate_left(1).to_le_bytes();
            rotate_right(&mut e, 4);
            assert_eq!(u32::from_le_bytes(e), v);
        }
        let v = 0x8123_4567_89ab_cdefu64;
        let mut e = v.rotate_left(1).to_le_bytes();
        rotate_right(&mut e, 8);
        assert_eq!(u64::from_le_bytes(e), v);
    }
}
