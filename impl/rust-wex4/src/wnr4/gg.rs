// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! WNR4 kind 2 (GG), docs/WNR4-SPEC.md section 5: the 18-byte blocks of the Q2_0 (ggml type 42) and
//! IQ4_NL (ggml type 20) slices of a GGUF routed-expert record in WNR4 tiles, every other slice stored
//! verbatim. `model_open` checks a WNM4 kind-2 blob (5.2); `decode` checks a record against its geometry,
//! model and owner (5.3 to 5.5) and decodes it. A failure leaves `out` partly written: the caller must not
//! use it.

use super::core::{self, is_tail_item, lane_hash, lane_items, last_item, le32, Tile, HEADER_BYTES, LANES, WNM4_MAGIC};

/// The kind byte of a GG record.
pub const KIND: u8 = 2;
/// The size of a WNM4 kind-2 blob.
pub const MODEL_BYTES: usize = 36_644;
/// ggml types of the coded slices: Q2_0 (subtype 0) and IQ4_NL (subtype 1).
pub const TYPE_Q2_0: u32 = 42;
pub const TYPE_IQ4_NL: u32 = 20;
/// Bytes of one block: the f16 scale d (low byte first), then 16 code bytes.
pub const BLOCK: u64 = 18;
const CLASSES: usize = 32;
/// One subtype's part of the model: class map, high-byte table, 32 low-byte tables, 32 nibble tables.
const SUB_BYTES: usize = 256 + 514 + CLASSES * 514 + CLASSES * 34;
const W256: usize = 257;
const W16: usize = 17;

/// A checked WNM4 kind-2 model.
pub struct Model {
    pub id: u32,
    map: [[u8; 256]; 2],
    high: Vec<u16>,
    low: Vec<u16>,
    nib: Vec<u16>,
}

impl Model {
    fn high(&self, u: usize) -> &[u16] {
        &self.high[u * W256..(u + 1) * W256]
    }
    fn low(&self, u: usize, c: usize) -> &[u16] {
        let r = u * CLASSES + c;
        &self.low[r * W256..(r + 1) * W256]
    }
    fn nib(&self, u: usize, c: usize) -> &[u16] {
        let r = u * CLASSES + c;
        &self.nib[r * W16..(r + 1) * W16]
    }
}

/// Section 5.2: header WNM4, 1, 2, 14, 32 classes, 2 subtypes, 0, 0; then per subtype the class map (every
/// value below 32), the high-byte table, the 32 low-byte tables and the 32 nibble tables, each a 14-bit
/// cumulative table; 36,644 bytes in all. The id is FNV-1a of the blob, 0 replaced by 1.
pub fn model_open(blob: &[u8]) -> Result<Model, String> {
    let id = core::wnm4_open(blob, KIND as u32, [CLASSES as u32, 2, 0, 0], MODEL_BYTES)?;
    debug_assert_eq!(le32(blob, 0), WNM4_MAGIC);
    let mut map = [[0u8; 256]; 2];
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut nib = Vec::new();
    for (u, m) in map.iter_mut().enumerate() {
        let at = 32 + u * SUB_BYTES;
        m.copy_from_slice(&blob[at..at + 256]);
        if let Some(v) = m.iter().position(|&c| c as usize >= CLASSES) {
            return Err(format!("WNM4 kind 2: subtype {} maps high byte {} to class {}", u, v, m[v]));
        }
        high.extend(core::read_tables(blob, at + 256, 1, 256, "high-byte")?);
        low.extend(core::read_tables(blob, at + 256 + 514, CLASSES, 256, "low-byte")?);
        nib.extend(core::read_tables(blob, at + 256 + 514 * (1 + CLASSES), CLASSES, 16, "nibble")?);
    }
    Ok(Model { id, map, high, low, nib })
}

/// One slice of the geometry (5.1): its ggml type and its size in bytes.
#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub ggml_type: u32,
    pub bytes: u64,
}

impl Slice {
    /// The subtype of a coded slice, None for a verbatim one.
    pub fn subtype(&self) -> Option<usize> {
        match self.ggml_type {
            TYPE_Q2_0 => Some(0),
            TYPE_IQ4_NL => Some(1),
            _ => None,
        }
    }
}

/// The checks of 5.1 on a geometry; returns the decoded size, the verbatim total and the blocks of each
/// coded slice (0 for a verbatim one).
pub fn geometry(slices: &[Slice]) -> Result<(u64, u64, Vec<u64>), String> {
    if slices.is_empty() || slices.len() > 4 {
        return Err(format!("GG: {} slices, 1 to 4 allowed", slices.len()));
    }
    let (mut total, mut verbatim) = (0u64, 0u64);
    let mut blocks = Vec::with_capacity(slices.len());
    for (s, sl) in slices.iter().enumerate() {
        if sl.bytes == 0 || sl.bytes > u32::MAX as u64 {
            return Err(format!("GG: slice {} of {} bytes", s, sl.bytes));
        }
        total += sl.bytes;
        match sl.subtype() {
            Some(_) => {
                if sl.bytes % BLOCK != 0 || sl.bytes / BLOCK >= 1 << 27 {
                    return Err(format!(
                        "GG: coded slice {} (ggml type {}) of {} bytes: not a multiple of 18 below 2^27 blocks",
                        s, sl.ggml_type, sl.bytes
                    ));
                }
                blocks.push(sl.bytes / BLOCK);
            }
            None => {
                verbatim += sl.bytes;
                blocks.push(0);
            }
        }
    }
    if total > u32::MAX as u64 || verbatim > i32::MAX as u64 {
        return Err(format!("GG: {} bytes of slices, {} verbatim: above 2^32 - 1 or 2^31 - 1", total, verbatim));
    }
    Ok((total, verbatim, blocks))
}

/// Decodes the GG record `rec` (its stored bytes) of the geometry `slices` with `model`, for the expected
/// `owner`, into `out` (exactly the sum of the slices).
pub fn decode(rec: &[u8], model: &Model, owner: u32, slices: &[Slice], out: &mut [u8]) -> Result<(), String> {
    let (total, verbatim, blocks) = geometry(slices)?;
    if out.len() as u64 != total {
        return Err(format!("GG: an output of {} bytes for {} bytes of slices", out.len(), total));
    }
    let h = core::header(rec)?;
    if h.kind != KIND {
        return Err(format!("GG: record kind {}", h.kind));
    }
    if h.flags != 0 {
        return Err(format!("GG: flags {:#04x}, must be 0", h.flags));
    }
    if !core::zero(rec, 16, 20) || !core::zero(rec, 28, HEADER_BYTES) {
        return Err("GG: a non-zero byte in record bytes 16..19 or 28..63".to_string());
    }
    if h.model_id != model.id {
        return Err(format!("GG: model id {:#010x}, the model's is {:#010x}", h.model_id, model.id));
    }
    if h.owner != owner {
        return Err(format!("GG: owner {:#010x}, expected {:#010x}", h.owner, owner));
    }
    let coded: Vec<u64> = blocks.iter().copied().filter(|&b| b > 0).collect();
    let v_area = (verbatim + 3) & !3;
    let lay = core::layout(rec, &h, core::tile_count(&coded, h.g), v_area)?;

    // the verbatim slices, in slice order, back to back, then zero padding to V
    let mut offs = Vec::with_capacity(slices.len());
    let (mut at, mut v) = (0usize, lay.extra.start);
    for sl in slices {
        offs.push(at);
        let n = sl.bytes as usize;
        if sl.subtype().is_none() {
            out[at..at + n].copy_from_slice(&rec[v..v + n]);
            v += n;
        }
        at += n;
    }
    if !core::zero(rec, v, lay.extra.end) {
        return Err("GG: a non-zero byte in the padding of the verbatim area".to_string());
    }

    // the coded slices in slice order, their tiles in record order
    let coded_slices: Vec<usize> = (0..slices.len()).filter(|&s| blocks[s] > 0).collect();
    let spans = core::tile_spans(&coded, h.g);
    for (t, (&(cs, c0, n), range)) in spans.iter().zip(lay.tiles.iter()).enumerate() {
        let s = coded_slices[cs];
        let base = offs[s] + (BLOCK * c0) as usize;
        let dst = &mut out[base..base + BLOCK as usize * n as usize];
        decode_tile(&rec[range.clone()], model, &h, t as u32, slices[s].ggml_type, n, dst)?;
    }
    Ok(())
}

/// One tile of `n` blocks of a slice of `ggml_type` into `dst` (18 n bytes): phases 1 to 3 of 5.4, the tail,
/// the hash and the end states of 5.5.
fn decode_tile(
    bytes: &[u8],
    model: &Model,
    h: &core::Header,
    t: u32,
    ggml_type: u32,
    n: u32,
    dst: &mut [u8],
) -> Result<(), String> {
    let u = if ggml_type == TYPE_Q2_0 { 0 } else { 1 };
    let n = n as usize;
    let steps = (n + 31) / 32;
    let mut tile = Tile::open(bytes)?;
    let mut hi = vec![0u8; n];
    let mut lo = vec![0u8; n];
    let mut code = vec![0u8; 16 * n];
    // phase 1, high bytes; phase 2, low bytes with the class of the high byte
    for k in 0..steps {
        for l in 0..LANES.min(n - 32 * k) {
            hi[32 * k + l] = tile.decode(l, model.high(u))? as u8;
        }
    }
    for k in 0..steps {
        for l in 0..LANES.min(n - 32 * k) {
            let i = 32 * k + l;
            let c = model.map[u][hi[i] as usize] as usize;
            lo[i] = tile.decode(l, model.low(u, c))? as u8;
        }
    }
    // phase 3, the 32 nibbles of each block; a tail lane skips nibbles 28..31 of its last block
    for j in 0..steps {
        let lanes = LANES.min(n - 32 * j);
        for k in 0..32 {
            for l in 0..lanes {
                let i = 32 * j + l;
                if k >= 28 && is_tail_item(n, i) {
                    continue;
                }
                let c = model.map[u][hi[i] as usize] as usize;
                let y = tile.decode(l, model.nib(u, c))? as u8;
                code[16 * i + k / 2] |= y << (4 * (k & 1));
            }
        }
    }
    // the tails: code bytes 14 and 15 of the lane's last block
    let mut owns = 0u32;
    for l in 0..LANES.min(n) {
        owns |= 1 << l;
        if l >= 2 {
            let v = tile.tail(l)?;
            let i = last_item(n, l);
            code[16 * i + 14] = v as u8;
            code[16 * i + 15] = (v >> 8) as u8;
        }
    }
    // the lane hashes: ident words, the high bytes, the low bytes, then 4 words per block
    let mut th = 0u32;
    for l in 0..LANES {
        let mut m = lane_hash(h.model_id, h.owner, t, l as u32, KIND as u32 | h.g << 16, ggml_type);
        let lc = lane_items(n as u32, l) as usize;
        for j in 0..lc {
            m.word(hi[32 * j + l] as u32);
        }
        for j in 0..lc {
            m.word(lo[32 * j + l] as u32);
        }
        for j in 0..lc {
            let i = 32 * j + l;
            for w in 0..4 {
                m.word(le32(&code, 16 * i + 4 * w));
            }
        }
        th ^= m.finish();
    }
    tile.finish(th, owns)?;
    for i in 0..n {
        let b = &mut dst[18 * i..18 * i + 18];
        b[0] = lo[i];
        b[1] = hi[i];
        b[2..].copy_from_slice(&code[16 * i..16 * i + 16]);
    }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::wnr4::core::L;
    use crate::wnr4::testenc::{self, Rng};

    /// A random valid model blob; with `skew`, the tables favour small symbols so the records shrink.
    pub fn random_model(rng: &mut Rng) -> Vec<u8> {
        let mut b = Vec::with_capacity(MODEL_BYTES);
        for w in [WNM4_MAGIC, 1, 2, 14, 32, 2, 0, 0] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for _ in 0..2 {
            for _ in 0..256 {
                b.push(rng.below(32) as u8);
            }
            for _ in 0..1 + CLASSES {
                b.extend(testenc::table_bytes(&testenc::random_cdf(rng, 256)));
            }
            for _ in 0..CLASSES {
                b.extend(testenc::table_bytes(&testenc::random_cdf(rng, 16)));
            }
        }
        assert_eq!(b.len(), MODEL_BYTES);
        b
    }

    /// A GG record of `data` (the decoded slices) with model `m`, written from 5.3 to 5.5.
    pub fn encode(m: &Model, owner: u32, g: u32, slices: &[Slice], data: &[u8]) -> Vec<u8> {
        let mut hdr = [0u8; HEADER_BYTES];
        hdr[0..4].copy_from_slice(&core::WNR4_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&m.id.to_le_bytes());
        hdr[8..12].copy_from_slice(&owner.to_le_bytes());
        hdr[12] = KIND;
        hdr[14..16].copy_from_slice(&(g as u16).to_le_bytes());
        let mut extra = Vec::new();
        let mut tiles = Vec::new();
        let mut at = 0usize;
        let mut t = 0u32;
        for sl in slices {
            let bytes = &data[at..at + sl.bytes as usize];
            at += sl.bytes as usize;
            let u = match sl.subtype() {
                None => {
                    extra.extend_from_slice(bytes);
                    continue;
                }
                Some(u) => u,
            };
            for chunk in bytes.chunks(18 * 32 * g as usize) {
                tiles.push(encode_tile(m, owner, g, t, u, sl.ggml_type, chunk));
                t += 1;
            }
        }
        while extra.len() % 4 != 0 {
            extra.push(0);
        }
        testenc::record(&hdr, &extra, &tiles)
    }

    fn encode_tile(m: &Model, owner: u32, g: u32, t: u32, u: usize, ty: u32, blocks: &[u8]) -> Vec<u8> {
        let n = blocks.len() / 18;
        let blk = |i: usize| &blocks[18 * i..18 * i + 18];
        let class = |i: usize| m.map[u][blk(i)[1] as usize] as usize;
        let mut ev: Vec<(usize, &[u16], usize)> = Vec::new();
        for i in 0..n {
            ev.push((i % 32, m.high(u), blk(i)[1] as usize));
        }
        for i in 0..n {
            ev.push((i % 32, m.low(u, class(i)), blk(i)[0] as usize));
        }
        let rounds = (n + 31) / 32;
        for j in 0..rounds {
            for k in 0..32 {
                for l in 0..32 {
                    let i = 32 * j + l;
                    if i >= n || (l >= 2 && i + 32 >= n && k >= 28) {
                        continue;
                    }
                    let y = (blk(i)[2 + k / 2] >> (4 * (k & 1))) & 15;
                    ev.push((l, m.nib(u, class(i)), y as usize));
                }
            }
        }
        let mut tails = [None; 32];
        let mut h = 0u32;
        for (l, tail) in tails.iter_mut().enumerate() {
            let mut hs = lane_hash(m.id, owner, t, l as u32, 2 | g << 16, ty);
            let own: Vec<usize> = (l..n).step_by(32).collect();
            for &i in &own {
                hs.word(blk(i)[1] as u32);
            }
            for &i in &own {
                hs.word(blk(i)[0] as u32);
            }
            for &i in &own {
                for w in 0..4 {
                    hs.word(le32(blk(i), 2 + 4 * w));
                }
            }
            h ^= hs.finish();
            if l >= 2 {
                if let Some(&i) = own.last() {
                    *tail = Some(blk(i)[16] as u16 | (blk(i)[17] as u16) << 8);
                }
            }
        }
        testenc::encode_tile(&ev, &testenc::end_states(h, &tails))
    }

    /// Random blocks whose code bytes lean on small nibbles, as real codes do, and whose high bytes take few
    /// values.
    fn random_data(rng: &mut Rng, slices: &[Slice]) -> Vec<u8> {
        let total: u64 = slices.iter().map(|s| s.bytes).sum();
        (0..total).map(|i| if i % 18 == 1 { 0x30 + rng.below(4) as u8 } else { rng.next() as u8 }).collect()
    }

    fn sl(ggml_type: u32, bytes: u64) -> Slice {
        Slice { ggml_type, bytes }
    }

    fn setup(seed: u64) -> (Rng, Model) {
        let mut rng = Rng::new(seed);
        let blob = random_model(&mut rng);
        let m = model_open(&blob).unwrap();
        (rng, m)
    }

    #[test]
    fn model_checks() {
        let mut rng = Rng::new(1);
        let blob = random_model(&mut rng);
        let m = model_open(&blob).unwrap();
        assert_eq!(m.id, crate::hash::fnv1a(&blob));
        assert!(model_open(&blob[..MODEL_BYTES - 1]).is_err());
        let mut b = blob.clone();
        b.push(0);
        assert!(model_open(&b).is_err());
        // header words 4..7: classes 32, subtypes 2, 0, 0
        for (w, v) in [(4usize, 31u32), (5, 1), (6, 1), (7, 1), (1, 2), (2, 3), (3, 15)] {
            let mut b = blob.clone();
            b[4 * w..4 * w + 4].copy_from_slice(&v.to_le_bytes());
            assert!(model_open(&b).is_err(), "word {}", w);
        }
        // a class map value of 32
        let mut b = blob.clone();
        b[32 + SUB_BYTES + 7] = 32;
        assert!(model_open(&b).is_err());
        // a table not strictly increasing, or not ending at 16384
        for at in [32 + 256 + 2, 32 + 256 + 514 * 5 + 512, MODEL_BYTES - 2, 32 + SUB_BYTES + 256 + 514 * 33 + 6] {
            let mut b = blob.clone();
            b[at] = 0;
            b[at + 1] = 0;
            assert!(model_open(&b).is_err(), "table byte {}", at);
        }
    }

    #[test]
    fn geometry_checks() {
        assert!(geometry(&[]).is_err());
        assert!(geometry(&[sl(42, 18); 5]).is_err());
        assert!(geometry(&[sl(42, 17)]).is_err());
        assert!(geometry(&[sl(20, 0)]).is_err());
        assert!(geometry(&[sl(16, 0)]).is_err());
        assert!(geometry(&[sl(42, 18 << 27)]).is_err());
        assert!(geometry(&[sl(42, 18 * ((1 << 27) - 1))]).is_ok());
        assert!(geometry(&[sl(16, 1 << 31)]).is_err());
        assert!(geometry(&[sl(16, (1 << 31) - 1)]).is_ok());
        // two slices of 2^27 - 1 blocks are 4.8 GB: above the 2^32 - 1 bytes of a record
        assert!(geometry(&[sl(42, 18 * ((1 << 27) - 1)), sl(20, 18 * ((1 << 27) - 1))]).is_err());
        assert!(geometry(&[sl(42, 18 * ((1 << 27) - 1)), sl(16, 4_294_967_295 - 18 * ((1 << 27) - 1))]).is_ok());
        assert!(geometry(&[sl(42, 18 * ((1 << 27) - 1)), sl(16, 4_294_967_296 - 18 * ((1 << 27) - 1))]).is_err());
        assert_eq!(geometry(&[sl(16, 7), sl(42, 36)]).unwrap(), (43, 7, vec![0, 2]));
    }

    fn round_trip(m: &Model, rng: &mut Rng, g: u32, slices: &[Slice]) -> Vec<u8> {
        let data = random_data(rng, slices);
        let rec = encode(m, 0x0003_0011, g, slices, &data);
        let mut out = vec![0u8; data.len()];
        decode(&rec, m, 0x0003_0011, slices, &mut out).unwrap_or_else(|e| panic!("G {} {:?}: {}", g, slices, e));
        assert!(out == data, "G {} {:?}", g, slices);
        rec
    }

    #[test]
    fn round_trips() {
        let (mut rng, m) = setup(2);
        for g in [1u32, 2, 7, 32, 4096] {
            for slices in [
                vec![sl(42, 18 * 444)],
                vec![sl(20, 18 * 33)],
                vec![sl(42, 18)],
                vec![sl(42, 18 * 31)],
                vec![sl(16, 230), sl(8, 4), sl(42, 18 * 444)],
                vec![sl(42, 18 * 20), sl(16, 101), sl(20, 18 * 65), sl(42, 18 * 69)],
                vec![sl(16, 5), sl(22, 3)],
                vec![sl(16, 8)],
            ] {
                round_trip(&m, &mut rng, g, &slices);
            }
        }
        // nothing coded: no tiles, the verbatim area only
        let r = round_trip(&m, &mut rng, 32, &[sl(16, 5), sl(22, 3)]);
        assert_eq!(le32(&r, 20), 0);
        assert_eq!(le32(&r, 24), 0);
    }

    #[test]
    fn tails_and_lanes() {
        // n = 1, 2, 3, 33, 34: tiles where lanes 0 and 1 own blocks with no tail, and lanes without blocks
        let (mut rng, m) = setup(3);
        for n in [1u64, 2, 3, 31, 32, 33, 34, 63, 64, 65] {
            round_trip(&m, &mut rng, 1, &[sl(42, 18 * n)]);
            round_trip(&m, &mut rng, 3, &[sl(20, 18 * n)]);
        }
    }

    /// The record and the bytes it decodes to, one Q2_0 slice between two verbatim ones (V padded by 2).
    fn sample() -> (Model, Vec<Slice>, Vec<u8>, Vec<u8>) {
        let (mut rng, m) = setup(4);
        let slices = vec![sl(16, 150), sl(42, 18 * 40), sl(8, 4)];
        let data = random_data(&mut rng, &slices);
        let rec = encode(&m, 9, 1, &slices, &data);
        (m, slices, data, rec)
    }

    #[test]
    fn record_checks() {
        let (m, slices, data, rec) = sample();
        let mut out = vec![0u8; data.len()];
        decode(&rec, &m, 9, &slices, &mut out).unwrap();
        let refused = |r: &[u8], owner: u32, s: &[Slice]| {
            let mut o = vec![0u8; data.len()];
            decode(r, &m, owner, s, &mut o).is_err()
        };
        let with = |at: usize, v: u8| {
            let mut b = rec.clone();
            b[at] = v;
            b
        };
        assert!(refused(&rec, 10, &slices), "owner");
        assert!(refused(&with(0, b'X'), 9, &slices), "magic");
        assert!(refused(&with(4, rec[4] ^ 1), 9, &slices), "model id");
        assert!(refused(&with(12, 3), 9, &slices), "kind 3");
        assert!(refused(&with(12, 0), 9, &slices), "kind 0");
        assert!(refused(&with(13, 1), 9, &slices), "flags");
        for at in [16, 17, 18, 19, 28, 40, 63] {
            assert!(refused(&with(at, 1), 9, &slices), "byte {}", at);
        }
        assert!(refused(&with(14, 0), 9, &slices), "G 0");
        let mut b = rec.clone();
        b[14..16].copy_from_slice(&4097u16.to_le_bytes());
        assert!(refused(&b, 9, &slices), "G 4097");
        assert!(refused(&with(20, rec[20] + 1), 9, &slices), "ntiles");
        // padding of the verbatim area: 154 bytes of verbatim slices, V = 156
        let pad = 64 + 4 * (le32(&rec, 20) as usize + 1) + 154;
        assert_eq!(rec[pad], 0);
        assert!(refused(&with(pad, 1), 9, &slices), "verbatim padding");
        assert!(refused(&with(pad + 1, 1), 9, &slices), "verbatim padding");
        assert!(refused(&with(rec.len() - 1, 1), 9, &slices), "record padding");
        let mut long = rec.clone();
        long.extend_from_slice(&[0u8; 4096]);
        assert!(refused(&long, 9, &slices), "a page more");
        assert!(refused(&rec[..rec.len() - 4096], 9, &slices), "a page less");
        // another geometry: another type for the coded slice, another size, a verbatim slice turned coded
        assert!(refused(&rec, 9, &[sl(16, 150), sl(20, 18 * 40), sl(8, 4)]), "subtype");
        assert!(refused(&rec, 9, &[sl(16, 150), sl(42, 18 * 41), sl(8, 3)]), "sizes");
        assert!(refused(&rec, 9, &[sl(16, 150), sl(42, 18 * 40)]), "slices");
        // an output buffer of another size
        let mut o = vec![0u8; data.len() + 1];
        assert!(decode(&rec, &m, 9, &slices, &mut o).is_err());
        // the model: another blob
        let other = model_open(&random_model(&mut Rng::new(99))).unwrap();
        let mut o = vec![0u8; data.len()];
        assert!(decode(&rec, &other, 9, &slices, &mut o).is_err());
    }

    #[test]
    fn single_bit_flips() {
        let (m, slices, data, rec) = sample();
        let vstart = 64 + 4 * (le32(&rec, 20) as usize + 1);
        let verbatim = vstart..vstart + 154;
        let mut out = vec![0u8; data.len()];
        let (mut refused, mut accepted) = (0, 0);
        for bit in 0..8 * rec.len() {
            let mut b = rec.clone();
            b[bit / 8] ^= 1 << (bit % 8);
            let r = decode(&b, &m, 9, &slices, &mut out);
            if verbatim.contains(&(bit / 8)) {
                // 5.5: a verbatim byte is in no tile hash; it decodes, to other bytes
                assert!(r.is_ok() && out != data, "bit {}", bit);
                accepted += 1;
            } else {
                assert!(r.is_err(), "bit {} (byte {}) decoded", bit, bit / 8);
                refused += 1;
            }
        }
        assert_eq!(accepted, 8 * 154);
        assert!(refused > 30000);
    }

    #[test]
    fn mutations_never_panic() {
        // random multi-byte changes of a record with several tiles: refused or decoded, never a panic
        let (mut rng, m) = setup(5);
        let slices = vec![sl(16, 10), sl(20, 18 * 100), sl(42, 18 * 70)];
        let data = random_data(&mut rng, &slices);
        let rec = encode(&m, 1, 1, &slices, &data);
        let mut out = vec![0u8; data.len()];
        for _ in 0..3000 {
            let mut b = rec.clone();
            for _ in 0..1 + rng.below(4) {
                let at = rng.below(b.len() as u32) as usize;
                b[at] = rng.next() as u8;
            }
            let _ = decode(&b, &m, 1, &slices, &mut out);
        }
        // states at the limits
        let mut b = rec.clone();
        let p0 = 64 + 4 * (le32(&rec, 20) as usize + 1) + 12;
        b[p0..p0 + 4].copy_from_slice(&(L - 1).to_le_bytes());
        assert!(decode(&b, &m, 1, &slices, &mut out).is_err());
        b[p0..p0 + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&b, &m, 1, &slices, &mut out).is_err());
    }
}
