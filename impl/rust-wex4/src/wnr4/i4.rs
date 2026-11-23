// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! WNR4 kind 3 (I4), docs/WNR4-SPEC.md section 6: int4-g64 expert records (1 to 3 matrices of nibble codes
//! with one f32 scale per group of 64 inputs), the scales coded as indices into the lattice of an FP8
//! block-scaled source or escaped. `model_open` checks a WNM4 kind-3 blob (6.2); `decode` checks a record
//! against its geometry, model and owner (6.3 to 6.5) and decodes it. A failure leaves `out` partly
//! written: the caller must not use it. Rust's f32 multiply and divide are IEEE 754 binary32 operations
//! rounded to nearest even, never fused and never flushed to zero, which is what the lattice needs.

use super::core::{self, is_tail_item, lane_hash, lane_items, last_item, le16, le32, Tile, HEADER_BYTES, LANES};

/// The kind byte of an I4 record.
pub const KIND: u8 = 3;
/// The size of a WNM4 kind-3 blob.
pub const MODEL_BYTES: usize = 196_688;
const NIB_ROWS: usize = 900;
const SCALE_ROWS: usize = 68;
const W16: usize = 17;
const W256: usize = 257;
/// The escape index.
pub const ESCAPE: u8 = 127;

/// A checked WNM4 kind-3 model.
pub struct Model {
    pub id: u32,
    nib: Vec<u16>,
    scale: Vec<u16>,
}

impl Model {
    fn nib(&self, m: usize, row: usize) -> &[u16] {
        let r = m * NIB_ROWS + row;
        &self.nib[r * W16..(r + 1) * W16]
    }
    fn scale(&self, m: usize, row: usize) -> &[u16] {
        let r = m * SCALE_ROWS + row;
        &self.scale[r * W256..(r + 1) * W256]
    }
}

/// Section 6.2: header WNM4, 1, 3, 14, 900, 68, 0, 0; 3 x 900 nibble tables of 17 u16, then 3 x 68 scale
/// tables of 257 u16, each a 14-bit cumulative table; 196,688 bytes. The id is FNV-1a, 0 replaced by 1.
pub fn model_open(blob: &[u8]) -> Result<Model, String> {
    let id = core::wnm4_open(blob, KIND as u32, [NIB_ROWS as u32, SCALE_ROWS as u32, 0, 0], MODEL_BYTES)?;
    let nib = core::read_tables(blob, 32, 3 * NIB_ROWS, 16, "nibble")?;
    let scale = core::read_tables(blob, 32 + 3 * NIB_ROWS * 2 * W16, 3 * SCALE_ROWS, 256, "scale")?;
    Ok(Model { id, nib, scale })
}

/// One matrix of the geometry (6.1).
#[derive(Clone, Copy, Debug)]
pub struct Matrix {
    pub rows: u32,
    pub cols: u32,
}

/// What 6.1 derives from the matrices, per matrix m.
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub gc: Vec<u64>,
    pub cells: Vec<u64>,
    pub nbc: Vec<u64>,
    pub bbase: Vec<u64>,
    pub nbases: u64,
    pub code_off: Vec<u64>,
    pub scale_off: Vec<u64>,
    pub total: u64,
}

/// The checks of 6.1: 1 to 3 matrices, rows at least 1, cols a multiple of 64 (at least 64), the record at
/// most 2^32 - 1 bytes.
pub fn geometry(mats: &[Matrix]) -> Result<Geometry, String> {
    if mats.is_empty() || mats.len() > 3 {
        return Err(format!("I4: {} matrices, 1 to 3 allowed", mats.len()));
    }
    let mut g = Geometry::default();
    let mut cells_sum = 0u64;
    for (m, mt) in mats.iter().enumerate() {
        if mt.rows == 0 || mt.cols == 0 || mt.cols % 64 != 0 {
            return Err(format!(
                "I4: matrix {} is {} x {}: rows >= 1 and cols a positive multiple of 64",
                m, mt.rows, mt.cols
            ));
        }
        let gc = mt.cols as u64 / 64;
        // rows < 2^32 and gc < 2^26: cells < 2^58, and each matrix is checked against the 2^32 - 1 byte
        // limit before the next is added, so no sum below can overflow (release builds check overflow)
        let cells = mt.rows as u64 * gc;
        cells_sum += cells;
        if 36 * cells_sum > u32::MAX as u64 {
            return Err(format!("I4: a record of {} groups, above 2^32 - 1 bytes", cells_sum));
        }
        let nbc = core::ceil_div(gc, 2);
        g.gc.push(gc);
        g.cells.push(cells);
        g.nbc.push(nbc);
        g.bbase.push(g.nbases);
        g.nbases += core::ceil_div(mt.rows as u64, 128) * nbc;
    }
    g.total = 36 * cells_sum;
    let mut at = 0;
    for &c in &g.cells {
        g.code_off.push(at);
        at += 32 * c;
    }
    for &c in &g.cells {
        g.scale_off.push(at);
        at += 4 * c;
    }
    Ok(g)
}

/// E4M3(k) for k = 0..126 (6.1): k x 2^-9 below 8, else (1 + (k & 7) / 8) x 2^((k >> 3) - 7); every value is
/// exact in binary32.
pub fn e4m3(k: u32) -> f32 {
    debug_assert!(k <= 126);
    if k < 8 {
        k as f32 * f32::from_bits((127 - 9) << 23)
    } else {
        // (8 + (k & 7)) x 2^((k >> 3) - 10), exponent -9..5
        (8 + (k & 7)) as f32 * f32::from_bits((((k >> 3) as i32 - 10 + 127) as u32) << 23)
    }
}

/// lat(k, B) = fl(fl(E4M3(k) x B) / 7): a binary32 multiply, then a binary32 division by 7.
pub fn lat(k: u32, b: f32) -> f32 {
    let p = e4m3(k) * b;
    p / 7.0
}

/// A base of the record (6.3): 0, or a positive normal finite f32.
pub fn valid_base(b: u32) -> bool {
    b == 0 || (0x0080_0000..0x7F80_0000).contains(&b)
}

/// Decodes the I4 record `rec` (its stored bytes) of the matrices `mats` with `model`, for the expected
/// `owner`, into `out` (exactly 36 x the groups). `pairs` is the packing of the geometry when the caller
/// knows it (flags bit 0 must then equal it); None takes the packing from the record (in WEX4 the record
/// is the only place that holds it, WEX4-SPEC 7.3), and only bits 1-7 must then be 0.
pub fn decode(
    rec: &[u8],
    model: &Model,
    owner: u32,
    mats: &[Matrix],
    pairs: Option<bool>,
    out: &mut [u8],
) -> Result<(), String> {
    let geo = geometry(mats)?;
    if out.len() as u64 != geo.total {
        return Err(format!("I4: an output of {} bytes for a record of {}", out.len(), geo.total));
    }
    let h = core::header(rec)?;
    if h.kind != KIND {
        return Err(format!("I4: record kind {}", h.kind));
    }
    if h.flags & 0xFE != 0 {
        return Err(format!("I4: flags {:#04x}, bits 1-7 must be 0", h.flags));
    }
    let p = h.flags & 1 == 1;
    if let Some(want) = pairs {
        if p != want {
            return Err(format!("I4: packing {} in the record, the geometry says {}", p as u8, want as u8));
        }
    }
    if !core::zero(rec, 16, 20) || !core::zero(rec, 34, HEADER_BYTES) {
        return Err("I4: a non-zero byte in record bytes 16..19 or 34..63".to_string());
    }
    let anchors = [le16(rec, 28), le16(rec, 30), le16(rec, 32)];
    if let Some(m) = (mats.len()..3).find(|&m| anchors[m] != 0) {
        return Err(format!("I4: anchor[{}] = {} in a record of {} matrices", m, anchors[m], mats.len()));
    }
    if h.model_id != model.id {
        return Err(format!("I4: model id {:#010x}, the model's is {:#010x}", h.model_id, model.id));
    }
    if h.owner != owner {
        return Err(format!("I4: owner {:#010x}, expected {:#010x}", h.owner, owner));
    }
    let lay = core::layout(rec, &h, core::tile_count(&geo.cells, h.g), 4 * geo.nbases)?;
    let mut bases = Vec::with_capacity(geo.nbases as usize);
    for b in 0..geo.nbases as usize {
        let v = le32(rec, lay.extra.start + 4 * b);
        if !valid_base(v) {
            return Err(format!("I4: base {} is {:#010x}, not 0 nor a positive normal finite f32", b, v));
        }
        bases.push(f32::from_bits(v));
    }
    let ctx = Ctx { model, geo: &geo, bases: &bases, h: &h, anchors, pairs: p };
    for (t, (&(m, c0, n), range)) in core::tile_spans(&geo.cells, h.g).iter().zip(lay.tiles.iter()).enumerate() {
        ctx.tile(&rec[range.clone()], t as u32, m, c0, n as usize, out)?;
    }
    Ok(())
}

struct Ctx<'a> {
    model: &'a Model,
    geo: &'a Geometry,
    bases: &'a [f32],
    h: &'a core::Header,
    anchors: [u16; 3],
    pairs: bool,
}

/// Byte and shift of element e (0..63) of a group's 32 code bytes (6.1, packing).
fn place(pairs: bool, e: usize) -> (usize, u32) {
    if pairs {
        (e / 2, 4 * (e % 2) as u32)
    } else {
        (e % 32, 4 * (e / 32) as u32)
    }
}

impl<'a> Ctx<'a> {
    /// One tile of `n` groups of matrix m from group c0: phases 1 and 2 of 6.4, the tail, the hash and the
    /// end states of 6.5; the groups' code bytes and scales go to their places in `out`.
    fn tile(&self, bytes: &[u8], t: u32, m: usize, c0: u64, n: usize, out: &mut [u8]) -> Result<(), String> {
        let gc = self.geo.gc[m];
        let d = if gc >= 32 { gc } else { core::ceil_div(32, gc) * gc } as usize;
        let steps = (n + 31) / 32;
        let mut tile = Tile::open(bytes)?;
        let mut kidx = vec![0u8; n];
        let mut sbits = vec![0u32; n];
        // phase 1: per step, the index symbols, then the four escape bytes, high byte first
        for k in 0..steps {
            let lanes = LANES.min(n - 32 * k);
            for l in 0..lanes {
                let i = 32 * k + l;
                let (row, kref) = if i >= d {
                    let r = kidx[i - d];
                    (32, if r == ESCAPE { 126 } else { r as usize })
                } else {
                    (0, 120)
                };
                let y = tile.decode(l, self.model.scale(m, row + kref / 4))?;
                let kk = (y + kref) % 256;
                if kk > 127 {
                    return Err(format!("I4: tile {} lane {}: lattice index {}", t, l, kk));
                }
                kidx[i] = kk as u8;
            }
            for b in 0..4 {
                for l in 0..lanes {
                    let i = 32 * k + l;
                    if kidx[i] == ESCAPE {
                        let v = tile.decode(l, self.model.scale(m, 64 + b))? as u32;
                        sbits[i] |= v << (8 * (3 - b));
                    }
                }
            }
            for l in 0..lanes {
                let i = 32 * k + l;
                if kidx[i] != ESCAPE {
                    let g = c0 + i as u64;
                    let (row, cg) = (g / gc, g % gc);
                    let blk = self.geo.bbase[m] + (row / 128) * self.geo.nbc[m] + cg / 2;
                    sbits[i] = lat(kidx[i] as u32, self.bases[blk as usize]).to_bits();
                }
            }
        }
        // phase 2: 64 elements per group; a tail lane skips elements 60..63 of its last group
        let anchor = self.anchors[m] as i32;
        let cls: Vec<usize> = sbits.iter().map(|&s| (((s >> 21) as i32 - anchor).clamp(-4, 4) + 4) as usize).collect();
        let mut code = vec![0u8; 32 * n];
        for j in 0..steps {
            let lanes = LANES.min(n - 32 * j);
            let mut pm = [9usize; LANES];
            let mut mx = [9usize; LANES];
            for e in 0..64 {
                for l in 0..lanes {
                    let i = 32 * j + l;
                    if e >= 60 && is_tail_item(n, i) {
                        continue;
                    }
                    let row = (cls[i] * 10 + pm[l]) * 10 + mx[l];
                    let y = tile.decode(l, self.model.nib(m, row))?;
                    let a = y.abs_diff(8);
                    pm[l] = a;
                    mx[l] = if mx[l] == 9 { a } else { mx[l].max(a) };
                    let (byte, sh) = place(self.pairs, e);
                    code[32 * i + byte] |= (y as u8) << sh;
                }
            }
        }
        let mut owns = 0u32;
        for l in 0..LANES.min(n) {
            owns |= 1 << l;
            if l >= 2 {
                let v = tile.tail(l)?;
                let i = last_item(n, l);
                for e in 60..64 {
                    let (byte, sh) = place(self.pairs, e);
                    code[32 * i + byte] |= (((v >> (4 * (e - 60))) & 15) as u8) << sh;
                }
            }
        }
        // the lane hashes: ident words, one block per scale, then 8 words per group
        let w4 = KIND as u32 | (self.pairs as u32) << 8 | self.h.g << 16;
        let w5 = m as u32 | (self.anchors[m] as u32) << 16;
        let mut th = 0u32;
        for l in 0..LANES {
            let mut hs = lane_hash(self.h.model_id, self.h.owner, t, l as u32, w4, w5);
            let lc = lane_items(n as u32, l) as usize;
            for j in 0..lc {
                hs.word(sbits[32 * j + l]);
            }
            for j in 0..lc {
                let i = 32 * j + l;
                for w in 0..8 {
                    hs.word(le32(&code, 32 * i + 4 * w));
                }
            }
            th ^= hs.finish();
        }
        tile.finish(th, owns)?;
        let co = (self.geo.code_off[m] + 32 * c0) as usize;
        out[co..co + 32 * n].copy_from_slice(&code);
        let so = (self.geo.scale_off[m] + 4 * c0) as usize;
        for (i, s) in sbits.iter().enumerate() {
            out[so + 4 * i..so + 4 * i + 4].copy_from_slice(&s.to_le_bytes());
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::wnr4::core::WNM4_MAGIC;
    use crate::wnr4::testenc::{self, Rng};

    pub fn random_model(rng: &mut Rng) -> Vec<u8> {
        let mut b = Vec::with_capacity(MODEL_BYTES);
        for w in [WNM4_MAGIC, 1, 3, 14, 900, 68, 0, 0] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for _ in 0..3 * NIB_ROWS {
            b.extend(testenc::table_bytes(&testenc::random_cdf(rng, 16)));
        }
        for _ in 0..3 * SCALE_ROWS {
            b.extend(testenc::table_bytes(&testenc::random_cdf(rng, 256)));
        }
        assert_eq!(b.len(), MODEL_BYTES);
        b
    }

    /// The input of an I4 record: codes (in the record's packing) and scale bits per matrix, bases, anchors.
    pub struct Input {
        pub mats: Vec<Matrix>,
        pub codes: Vec<Vec<u8>>,
        pub scales: Vec<Vec<u32>>,
        pub bases: Vec<u32>,
        pub anchors: [u16; 3],
    }

    impl Input {
        pub fn bytes(&self) -> Vec<u8> {
            let mut v = Vec::new();
            for c in &self.codes {
                v.extend_from_slice(c);
            }
            for s in &self.scales {
                for x in s {
                    v.extend_from_slice(&x.to_le_bytes());
                }
            }
            v
        }
    }

    fn nibble(code: &[u8], pairs: bool, e: usize) -> usize {
        let (byte, sh) = place(pairs, e);
        ((code[byte] >> sh) & 15) as usize
    }

    /// An I4 record of `inp`, written from 6.3 to 6.5: a scale is coded as the smallest lattice index whose
    /// value has its 32 bits, else escaped.
    pub fn encode(md: &Model, owner: u32, g: u32, pairs: bool, inp: &Input) -> Vec<u8> {
        let geo = geometry(&inp.mats).unwrap();
        let mut hdr = [0u8; HEADER_BYTES];
        hdr[0..4].copy_from_slice(&core::WNR4_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&md.id.to_le_bytes());
        hdr[8..12].copy_from_slice(&owner.to_le_bytes());
        hdr[12] = KIND;
        hdr[13] = pairs as u8;
        hdr[14..16].copy_from_slice(&(g as u16).to_le_bytes());
        for m in 0..3 {
            hdr[28 + 2 * m..30 + 2 * m].copy_from_slice(&inp.anchors[m].to_le_bytes());
        }
        let extra: Vec<u8> = inp.bases.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut tiles = Vec::new();
        for (t, &(m, c0, n)) in core::tile_spans(&geo.cells, g).iter().enumerate() {
            tiles.push(encode_tile(md, owner, g, pairs, inp, &geo, t as u32, m, c0 as usize, n as usize));
        }
        testenc::record(&hdr, &extra, &tiles)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_tile(
        md: &Model,
        owner: u32,
        g: u32,
        pairs: bool,
        inp: &Input,
        geo: &Geometry,
        t: u32,
        m: usize,
        c0: usize,
        n: usize,
    ) -> Vec<u8> {
        let gc = geo.gc[m] as usize;
        let d = if gc >= 32 { gc } else { (32 + gc - 1) / gc * gc };
        let scale = |i: usize| inp.scales[m][c0 + i];
        let code = |i: usize| &inp.codes[m][32 * (c0 + i)..32 * (c0 + i) + 32];
        let kidx: Vec<u8> = (0..n)
            .map(|i| {
                let grp = c0 + i;
                let blk = geo.bbase[m] as usize + (grp / gc) / 128 * geo.nbc[m] as usize + (grp % gc) / 2;
                let b = f32::from_bits(inp.bases[blk]);
                (0..127u32).find(|&k| lat(k, b).to_bits() == scale(i)).unwrap_or(127) as u8
            })
            .collect();
        let mut ev: Vec<(usize, &[u16], usize)> = Vec::new();
        for k in 0..(n + 31) / 32 {
            let lanes = 32.min(n - 32 * k);
            for l in 0..lanes {
                let i = 32 * k + l;
                let (row, kref) =
                    if i >= d { (32, if kidx[i - d] == 127 { 126 } else { kidx[i - d] as usize }) } else { (0, 120) };
                ev.push((l, md.scale(m, row + kref / 4), (kidx[i] as usize + 256 - kref) % 256));
            }
            for b in 0..4 {
                for l in 0..lanes {
                    let i = 32 * k + l;
                    if kidx[i] == 127 {
                        ev.push((l, md.scale(m, 64 + b), (scale(i) >> (8 * (3 - b)) & 255) as usize));
                    }
                }
            }
        }
        let anchor = inp.anchors[m] as i32;
        for j in 0..(n + 31) / 32 {
            let lanes = 32.min(n - 32 * j);
            let mut pm = [9usize; 32];
            let mut mx = [9usize; 32];
            for e in 0..64 {
                for l in 0..lanes {
                    let i = 32 * j + l;
                    if l >= 2 && i + 32 >= n && e >= 60 {
                        continue;
                    }
                    let cls = (((scale(i) >> 21) as i32 - anchor).clamp(-4, 4) + 4) as usize;
                    let y = nibble(code(i), pairs, e);
                    ev.push((l, md.nib(m, (cls * 10 + pm[l]) * 10 + mx[l]), y));
                    let a = (y as i32 - 8).unsigned_abs() as usize;
                    pm[l] = a;
                    mx[l] = if mx[l] == 9 { a } else { mx[l].max(a) };
                }
            }
        }
        let mut tails = [None; 32];
        let mut h = 0u32;
        for (l, tail) in tails.iter_mut().enumerate() {
            let mut hs = lane_hash(
                md.id,
                owner,
                t,
                l as u32,
                3 | (pairs as u32) << 8 | g << 16,
                m as u32 | (anchor as u32) << 16,
            );
            let own: Vec<usize> = (l..n).step_by(32).collect();
            for &i in &own {
                hs.word(scale(i));
            }
            for &i in &own {
                for w in 0..8 {
                    hs.word(le32(code(i), 4 * w));
                }
            }
            h ^= hs.finish();
            if l >= 2 {
                if let Some(&i) = own.last() {
                    let v = (60..64).map(|e| (nibble(code(i), pairs, e) as u16) << (4 * (e - 60))).sum::<u16>();
                    *tail = Some(v);
                }
            }
        }
        testenc::encode_tile(&ev, &testenc::end_states(h, &tails))
    }

    /// A random input: bases normal or 0, scales mostly on the lattice of their block, some escaped (NaN,
    /// infinities, -0, subnormals, random bits), nibbles leaning on 8.
    pub fn random_input(rng: &mut Rng, mats: &[Matrix], escapes: u32) -> Input {
        let geo = geometry(mats).unwrap();
        let bases: Vec<u32> = (0..geo.nbases)
            .map(|_| match rng.below(10) {
                0 => 0,
                1 => 0x0080_0000 + rng.below(0x0100_0000),
                _ => 0x3A00_0000 + rng.below(0x0200_0000),
            })
            .collect();
        let hostile = [0x7FC0_0000u32, 0x7FA0_0001, 0x7F80_0000, 0xFF80_0000, 0x8000_0000, 0x0000_0001, 0x007F_FFFF];
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for m in 0..mats.len() {
            let gc = geo.gc[m];
            let mut s = Vec::new();
            for grp in 0..geo.cells[m] {
                let blk = geo.bbase[m] + (grp / gc) / 128 * geo.nbc[m] + (grp % gc) / 2;
                let v = if rng.below(100) < escapes {
                    if rng.below(2) == 0 {
                        hostile[rng.below(hostile.len() as u32) as usize]
                    } else {
                        rng.next() as u32
                    }
                } else {
                    lat(80 + rng.below(47), f32::from_bits(bases[blk as usize])).to_bits()
                };
                s.push(v);
            }
            let c: Vec<u8> = (0..32 * geo.cells[m])
                .map(|_| {
                    let lo = (8 + rng.below(5)) as u8 - 2;
                    let hi = (8 + rng.below(5)) as u8 - 2;
                    if rng.below(20) == 0 {
                        rng.next() as u8
                    } else {
                        lo | hi << 4
                    }
                })
                .collect();
            codes.push(c);
            scales.push(s);
        }
        let mut anchors = [0u16; 3];
        for (m, a) in anchors.iter_mut().enumerate().take(mats.len()) {
            let mut v: Vec<u32> = scales[m].iter().map(|s| s >> 21).collect();
            v.sort_unstable();
            *a = v[v.len() / 2] as u16;
        }
        Input { mats: mats.to_vec(), codes, scales, bases, anchors }
    }

    fn mat(rows: u32, cols: u32) -> Matrix {
        Matrix { rows, cols }
    }

    fn setup(seed: u64) -> (Rng, Model) {
        let mut rng = Rng::new(seed);
        let blob = random_model(&mut rng);
        (rng, model_open(&blob).unwrap())
    }

    fn round_trip(md: &Model, rng: &mut Rng, g: u32, pairs: bool, mats: &[Matrix], esc: u32) -> (Vec<u8>, Vec<u8>) {
        let inp = random_input(rng, mats, esc);
        let data = inp.bytes();
        let rec = encode(md, 0x0002_0005, g, pairs, &inp);
        let mut out = vec![0u8; data.len()];
        decode(&rec, md, 0x0002_0005, mats, Some(pairs), &mut out)
            .unwrap_or_else(|e| panic!("G {} pairs {} {:?}: {}", g, pairs, mats, e));
        assert!(out == data, "G {} pairs {} {:?}", g, pairs, mats);
        (rec, data)
    }

    #[test]
    fn lattice() {
        assert_eq!(e4m3(0), 0.0);
        assert_eq!(e4m3(1), 1.0 / 512.0);
        assert_eq!(e4m3(7), 7.0 / 512.0);
        assert_eq!(e4m3(8), 1.0 / 64.0);
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(126), 448.0);
        assert_eq!(lat(126, 1.0), 64.0);
        assert_eq!(lat(0x38, 7.0), 1.0);
        // the two roundings, checked against binary64: the product of a 4-bit and a 24-bit significand is
        // exact in binary64, and a binary64 quotient rounded to binary32 is the binary32 quotient (53 >= 2 x
        // 24 + 2), subnormal results included
        let bases = [
            1.0f32,
            0.00123,
            f32::MIN_POSITIVE,
            f32::from_bits(0x0080_0001),
            f32::from_bits(0x0380_0000),
            f32::from_bits(0x03b0_0000),
            1.7e38,
            3.0e-30,
            f32::from_bits(0x3b9d_43c1),
        ];
        let mut subnormal = 0;
        for &b in &bases {
            for k in 0..=126u32 {
                let p = (e4m3(k) as f64 * b as f64) as f32;
                let want = (p as f64 / 7.0) as f32;
                let got = lat(k, b);
                assert_eq!(got.to_bits(), want.to_bits(), "k {} B {:#x}", k, b.to_bits());
                if got != 0.0 && !got.is_normal() {
                    subnormal += 1;
                }
            }
        }
        assert!(subnormal > 50, "{} subnormal lattice values", subnormal);
        assert!(valid_base(0) && valid_base(0x0080_0000) && valid_base(0x7F7F_FFFF));
        for b in [1, 0x007F_FFFF, 0x7F80_0000, 0x7FC0_0000, 0x8000_0000, 0xBF80_0000] {
            assert!(!valid_base(b), "{:#x}", b);
        }
    }

    #[test]
    fn model_checks() {
        let mut rng = Rng::new(1);
        let blob = random_model(&mut rng);
        let md = model_open(&blob).unwrap();
        assert_eq!(md.id, crate::hash::fnv1a(&blob));
        assert!(model_open(&blob[..MODEL_BYTES - 2]).is_err());
        for (w, v) in [(4usize, 899u32), (5, 67), (6, 1), (7, 1), (2, 2), (3, 15)] {
            let mut b = blob.clone();
            b[4 * w..4 * w + 4].copy_from_slice(&v.to_le_bytes());
            assert!(model_open(&b).is_err(), "word {}", w);
        }
        for at in [32 + 2, 32 + 91_800 - 2, 32 + 91_800 + 2, MODEL_BYTES - 2] {
            let mut b = blob.clone();
            b[at] = 0;
            b[at + 1] = 0;
            assert!(model_open(&b).is_err(), "table byte {}", at);
        }
    }

    #[test]
    fn geometry_checks() {
        assert!(geometry(&[]).is_err());
        assert!(geometry(&[mat(1, 64); 4]).is_err());
        assert!(geometry(&[mat(0, 64)]).is_err());
        assert!(geometry(&[mat(1, 0)]).is_err());
        assert!(geometry(&[mat(1, 96)]).is_err());
        let g = geometry(&[mat(129, 64), mat(128, 192), mat(1, 128)]).unwrap();
        assert_eq!(g.gc, vec![1, 3, 2]);
        assert_eq!(g.nbc, vec![1, 2, 1]);
        assert_eq!(g.bbase, vec![0, 2, 4]);
        assert_eq!(g.nbases, 5);
        assert_eq!(g.total, 36 * (129 + 384 + 2));
        assert_eq!(g.scale_off[0], 32 * (129 + 384 + 2));
        // 2^32 - 1 bytes at most: 119,304,647 groups of 36 bytes fit, one more does not
        assert!(geometry(&[mat(119_304_647, 64)]).is_ok());
        assert!(geometry(&[mat(119_304_648, 64)]).is_err());
        assert!(geometry(&[mat(u32::MAX, 64 * 1024)]).is_err());
        // the largest geometry the u32 fields allow is refused, not an arithmetic overflow
        let big = mat(u32::MAX, u32::MAX / 64 * 64);
        assert!(geometry(&[big, big, big]).is_err());
        assert!(geometry(&[mat(1, 64), big]).is_err());
    }

    #[test]
    fn round_trips() {
        let (mut rng, md) = setup(2);
        let shapes: Vec<Vec<Matrix>> = vec![
            vec![mat(1, 64)],
            vec![mat(129, 64), mat(128, 64), mat(127, 128)],
            vec![mat(8, 2048)],
            vec![mat(4, 2112), mat(129, 64)],
            vec![mat(3, 64 * 31)],
            vec![mat(2, 64 * 33), mat(5, 64 * 32), mat(130, 128)],
        ];
        for g in [1u32, 2, 32, 100, 4096] {
            for mats in &shapes {
                for pairs in [false, true] {
                    round_trip(&md, &mut rng, g, pairs, mats, 10);
                }
            }
        }
    }

    #[test]
    fn hostile_scales_and_subnormal_lattices() {
        let (mut rng, md) = setup(3);
        // every scale escaped, or none
        for esc in [0, 100] {
            round_trip(&md, &mut rng, 32, false, &[mat(256, 64)], esc);
        }
        // tiny bases: lattice values that are subnormal f32 values
        let mats = [mat(130, 128)];
        let mut inp = random_input(&mut rng, &mats, 0);
        // the two smallest normal exponents: lat(k, B) is subnormal for most k
        inp.bases = vec![0x0080_0000, 0x0100_0000];
        let geo = geometry(&mats).unwrap();
        let mut sub = 0;
        for grp in 0..geo.cells[0] as usize {
            let blk = (grp / 2) / 128;
            let v = lat(rng.below(127), f32::from_bits(inp.bases[blk]));
            sub += (v != 0.0 && !v.is_normal()) as u32;
            inp.scales[0][grp] = v.to_bits();
        }
        assert!(sub > 50);
        for pairs in [false, true] {
            let rec = encode(&md, 1, 32, pairs, &inp);
            let mut out = vec![0u8; geo.total as usize];
            decode(&rec, &md, 1, &mats, Some(pairs), &mut out).unwrap();
            assert!(out == inp.bytes());
        }
    }

    #[test]
    fn planar_and_pairs_decode_the_same_values() {
        // the same int4 values in both packings: the records differ, the elements are the same
        let (mut rng, md) = setup(4);
        let mats = [mat(40, 192)];
        let planar = random_input(&mut rng, &mats, 5);
        let mut paired = Input {
            mats: planar.mats.clone(),
            codes: planar.codes.clone(),
            scales: planar.scales.clone(),
            bases: planar.bases.clone(),
            anchors: planar.anchors,
        };
        for (gi, grp) in planar.codes[0].chunks(32).enumerate() {
            let mut c = [0u8; 32];
            for e in 0..64 {
                let (byte, sh) = place(true, e);
                c[byte] |= (nibble(grp, false, e) as u8) << sh;
            }
            paired.codes[0][32 * gi..32 * gi + 32].copy_from_slice(&c);
        }
        let mut a = vec![0u8; planar.bytes().len()];
        let mut b = a.clone();
        let ra = encode(&md, 1, 32, false, &planar);
        let rb = encode(&md, 1, 32, true, &paired);
        assert!(ra != rb);
        decode(&ra, &md, 1, &mats, None, &mut a).unwrap();
        decode(&rb, &md, 1, &mats, None, &mut b).unwrap();
        for gi in 0..40 * 3 {
            for e in 0..64 {
                assert_eq!(nibble(&a[32 * gi..], false, e), nibble(&b[32 * gi..], true, e));
            }
        }
        assert_eq!(a[32 * 120..], b[32 * 120..]);
        // and a record read with the other packing is refused (its hash was made with the record's)
        let mut flipped = ra.clone();
        flipped[13] = 1;
        assert!(decode(&flipped, &md, 1, &mats, None, &mut a).is_err());
        assert!(decode(&ra, &md, 1, &mats, Some(true), &mut a).is_err());
    }

    /// One record of 3 matrices with escapes, for the refusals and the bit flips.
    fn sample() -> (Model, Vec<Matrix>, Vec<u8>, Vec<u8>, Input) {
        let (mut rng, md) = setup(5);
        let mats = vec![mat(130, 64), mat(3, 128)];
        let inp = random_input(&mut rng, &mats, 15);
        let rec = encode(&md, 7, 8, false, &inp);
        (md, mats, inp.bytes(), rec, inp)
    }

    #[test]
    fn record_checks() {
        let (md, mats, data, rec, inp) = sample();
        let mut out = vec![0u8; data.len()];
        decode(&rec, &md, 7, &mats, Some(false), &mut out).unwrap();
        assert!(out == data);
        let refused = |r: &[u8], owner: u32, mats: &[Matrix], pairs: Option<bool>| {
            let mut o = vec![0u8; data.len()];
            decode(r, &md, owner, mats, pairs, &mut o).is_err()
        };
        let with = |at: usize, v: u8| {
            let mut b = rec.clone();
            b[at] = v;
            b
        };
        let s = Some(false);
        assert!(refused(&rec, 8, &mats, s), "owner");
        assert!(refused(&with(1, 0), 7, &mats, s), "magic");
        assert!(refused(&with(7, rec[7] ^ 0x80), 7, &mats, s), "model id");
        assert!(refused(&with(12, 2), 7, &mats, s), "kind");
        assert!(refused(&with(13, 2), 7, &mats, None), "flags bit 1");
        assert!(refused(&with(13, 0x80), 7, &mats, None), "flags bit 7");
        assert!(refused(&with(13, 1), 7, &mats, None), "packing");
        assert!(refused(&rec, 7, &mats, Some(true)), "packing of the geometry");
        for at in [16, 19, 34, 35, 50, 63] {
            assert!(refused(&with(at, 1), 7, &mats, s), "byte {}", at);
        }
        assert!(refused(&with(32, 1), 7, &mats, s), "anchor of a matrix it does not have");
        assert!(refused(&with(33, 1), 7, &mats, s), "anchor of a matrix it does not have");
        assert!(refused(&with(14, 0), 7, &mats, s), "G 0");
        assert!(refused(&with(20, rec[20] + 1), 7, &mats, s), "ntiles");
        let bases_at = 64 + 4 * (le32(&rec, 20) as usize + 1);
        for v in [0x7F80_0000u32, 0x7FC0_0000, 0x8000_0000, 1, 0x007F_FFFF] {
            let mut b = rec.clone();
            b[bases_at..bases_at + 4].copy_from_slice(&v.to_le_bytes());
            assert!(refused(&b, 7, &mats, s), "base {:#x}", v);
        }
        assert!(refused(&with(rec.len() - 1, 1), 7, &mats, s), "padding");
        assert!(refused(&rec, 7, &[mat(130, 64)], s), "geometry: one matrix");
        assert!(refused(&rec, 7, &[mat(130, 64), mat(3, 128), mat(1, 64)], s), "geometry: three matrices");
        assert!(refused(&rec, 7, &[mat(129, 64), mat(3, 128)], s), "geometry: rows");
        let other = model_open(&random_model(&mut Rng::new(77))).unwrap();
        assert!(decode(&rec, &other, 7, &mats, s, &mut out).is_err(), "another model");
        // an output buffer of another size
        let mut o = vec![0u8; data.len() - 1];
        assert!(decode(&rec, &md, 7, &mats, s, &mut o).is_err(), "output size");
        // the bases are those of the input, in block order
        assert_eq!(
            &rec[bases_at..bases_at + 4 * inp.bases.len()],
            &inp.bases.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>()[..]
        );
    }

    #[test]
    fn unread_base_changes_nothing() {
        let (mut rng, md) = setup(6);
        let mats = [mat(256, 64)];
        let mut inp = random_input(&mut rng, &mats, 0);
        // every scale of block 1 (rows 128..255) escapes: negative values
        for grp in 128..256 {
            inp.scales[0][grp] = 0xBF00_0000 | rng.below(1 << 20);
        }
        inp.bases[1] = 0;
        let rec = encode(&md, 1, 32, false, &inp);
        let mut a = vec![0u8; inp.bytes().len()];
        decode(&rec, &md, 1, &mats, None, &mut a).unwrap();
        let at = 64 + 4 * (le32(&rec, 20) as usize + 1) + 4;
        let mut b = rec.clone();
        b[at..at + 4].copy_from_slice(&0x3F80_0000u32.to_le_bytes());
        let mut o = vec![0u8; a.len()];
        decode(&b, &md, 1, &mats, None, &mut o).unwrap();
        assert!(o == a);
        // the base of block 0 is read: changing it is refused
        let mut c = rec.clone();
        c[at - 4..at].copy_from_slice(&0x3F80_0000u32.to_le_bytes());
        assert!(decode(&c, &md, 1, &mats, None, &mut o).is_err());
    }

    #[test]
    fn single_bit_flips() {
        let (md, mats, data, rec, _) = sample();
        let mut out = vec![0u8; data.len()];
        let mut refused = 0;
        for bit in 0..8 * rec.len() {
            let mut b = rec.clone();
            b[bit / 8] ^= 1 << (bit % 8);
            match decode(&b, &md, 7, &mats, None, &mut out) {
                Err(_) => refused += 1,
                // the only changes that may decode: an anchor bit that moves no class, a base that is never
                // read; both must give the same bytes
                Ok(()) => {
                    let byte = bit / 8;
                    let bases_at = 64 + 4 * (le32(&rec, 20) as usize + 1);
                    assert!((28..32).contains(&byte) || (bases_at..bases_at + 8).contains(&byte), "byte {}", byte);
                    assert!(out == data, "bit {}", bit);
                }
            }
        }
        assert!(refused > 8 * rec.len() - 200, "{} of {} refused", refused, 8 * rec.len());
    }

    #[test]
    fn mutations_never_panic() {
        let (md, mats, data, rec, _) = sample();
        let mut rng = Rng::new(8);
        let mut out = vec![0u8; data.len()];
        for _ in 0..3000 {
            let mut b = rec.clone();
            for _ in 0..1 + rng.below(4) {
                let at = rng.below(b.len() as u32) as usize;
                b[at] = rng.next() as u8;
            }
            let _ = decode(&b, &md, 7, &mats, None, &mut out);
        }
    }
}
