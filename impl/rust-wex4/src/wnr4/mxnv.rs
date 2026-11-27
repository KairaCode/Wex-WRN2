// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! WNR4 kinds 0 (MXFP4) and 1 (NVFP4), docs/WNR4-SPEC.md sections 1 to 4: EXPERT slabs of three matrices
//! (scales of the 3 matrices back to back, then packed FP4 weights). `model_open` validates a WNM4 kind-0
//! or kind-1 blob (`pb` = 1 or 5); `decode` validates and decodes a record into `out`.

use super::core::{self, is_tail_item, lane_hash, le32, Tile, LANES, WNM4_MAGIC};
use crate::hash::Murmur;

pub const KIND_MX: u8 = 0;
pub const KIND_NV: u8 = 1;
pub const MODEL_MX_PB1: usize = 59_408;
pub const MODEL_MX_PB5: usize = 290_744;
pub const MODEL_NV_PB1: usize = 6_522_158;
pub const MODEL_NV_PB5: usize = 6_951_782;
pub const FLAG_CEIL_MASK: u8 = 0x07;
pub const FLAG_CZ: u8 = 0x08;

const W16: usize = 17;
const W256: usize = 257;

/// True when `bytes` is one of the six exact WNM4 blob sizes (MX/NV pb=1/5, GG, I4).
pub fn is_valid_wnm4_size(bytes: u64) -> bool {
    matches!(
        bytes as usize,
        MODEL_MX_PB1 | MODEL_MX_PB5 | MODEL_NV_PB1 | MODEL_NV_PB5 | super::gg::MODEL_BYTES | super::i4::MODEL_BYTES
    )
}

/// True when `bytes` is an exact WNM4 blob size for kind 0 (MX) or kind 1 (NV).
pub fn is_kind_size(kind: u8, bytes: u64) -> bool {
    match kind {
        KIND_MX => bytes as usize == MODEL_MX_PB1 || bytes as usize == MODEL_MX_PB5,
        KIND_NV => bytes as usize == MODEL_NV_PB1 || bytes as usize == MODEL_NV_PB5,
        _ => false,
    }
}

/// A validated WNM4 kind-0 (MX) or kind-1 (NV) model.
pub struct Model {
    pub id: u32,
    pub kind: u8,
    pub pb: u32,
    pub wctx: usize,
    pub sctx: usize,
    wcdf: Vec<u16>,
    scdf: Vec<u16>,
}

impl Model {
    fn w(&self, m: usize, row: usize) -> &[u16] {
        let base = (m * self.wctx + row) * W16;
        &self.wcdf[base..base + W16]
    }
    fn s(&self, m: usize, ctx: usize) -> &[u16] {
        let base = (m * self.sctx + ctx) * W256;
        &self.scdf[base..base + W256]
    }
}

/// Validates and opens a WNM4 kind-0 (MX) or kind-1 (NV) blob (pb = 1 or 5).
pub fn model_open(blob: &[u8]) -> Result<Model, String> {
    if blob.len() < 32 {
        return Err(format!("WNM4 MX/NV: {} bytes, shorter than header", blob.len()));
    }
    let kind = le32(blob, 8);
    let pb = le32(blob, 24);
    if kind > 1 || (pb != 1 && pb != 5) {
        return Err(format!("WNM4 MX/NV: kind {} pb {}", kind, pb));
    }
    let wctx = (if kind == 1 { 1053 } else { 567 }) * pb as usize;
    let sctx = if kind == 1 { 4160 } else { 1 };
    let size = 32 + 3 * wctx * 34 + 3 * sctx * 514;
    let id = core::wnm4_open(blob, kind, [wctx as u32, sctx as u32, pb, 0], size)?;
    debug_assert_eq!(le32(blob, 0), WNM4_MAGIC);
    let wcdf = core::read_tables(blob, 32, 3 * wctx, 16, "weight")?;
    let scdf = core::read_tables(blob, 32 + 3 * wctx * 34, 3 * sctx, 256, "scale")?;
    Ok(Model { id, kind: kind as u8, pb, wctx, sctx, wcdf, scdf })
}

/// Geometry of an MXFP4 (kind 0) or NVFP4 (kind 1) 3-matrix EXPERT slab.
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub kind: u8,
    pub rows: [u32; 3],
    pub scols: [u32; 3],
    pub scount: [u32; 3],
    pub soff: [usize; 3],
    pub woff: [usize; 3],
    pub cells: [u32; 3],
    pub raw: usize,
}

fn scale_cls(s: u8, anchor: u8, r: i32) -> usize {
    let d = (s as i32 - anchor as i32).clamp(-r, r);
    (d + r) as usize
}

fn scale_ctx(kind: u8, s: &[u8], i: usize, lo: usize, cols: usize) -> (usize, u8) {
    if kind == KIND_MX {
        return (0, 0);
    }
    if cols < 33 || i < lo + cols {
        return (63, 0);
    }
    let up = s[i - cols];
    let col = i % cols;
    let dul = if col > 0 && i - cols - 1 >= lo {
        ((s[i - cols - 1] as i32 - up as i32).clamp(-3, 3) + 3) as usize
    } else {
        7
    };
    let dur = if col + 1 < cols { ((s[i - cols + 1] as i32 - up as i32).clamp(-3, 3) + 3) as usize } else { 7 };
    ((1 + (up as usize >> 2)) * 64 + dul * 8 + dur, up)
}

fn wrow(cls: usize, pm: usize, mx: usize, pb: u32, p: usize, b: usize) -> usize {
    let ctx = (cls * 9 + pm) * 9 + mx;
    if pb == 1 {
        ctx
    } else {
        ctx * 5 + if p + 5 <= b { 0 } else { p + 5 - b }
    }
}

fn ceil_dbl8(w: u32) -> u32 {
    let m = w & 0x7777_7777;
    let g = ((m >> 1) | (m >> 2)) & 0x1111_1111;
    let mn = (g << 1) | (m & 0x1111_1111 & !g);
    (w & 0x8888_8888) | (m + mn)
}

/// Decodes a WNR4 kind-0 (MX) or kind-1 (NV) EXPERT record into `out` (`geo.raw` bytes).
pub fn decode(
    rec: &[u8],
    model: &Model,
    owner: u32,
    value_exact: bool,
    geo: &Geo,
    out: &mut [u8],
) -> Result<(), String> {
    if out.len() != geo.raw {
        return Err(format!("WNR4 MX/NV: output buffer {} bytes, geometry gives {}", out.len(), geo.raw));
    }
    if model.kind != geo.kind {
        return Err(format!("WNR4 MX/NV: model kind {}, geometry kind {}", model.kind, geo.kind));
    }
    let h = core::header(rec)?;
    if h.model_id != model.id {
        return Err(format!("WNR4 MX/NV: record model id {:#010x}, blob id {:#010x}", h.model_id, model.id));
    }
    if h.owner != owner {
        return Err(format!("WNR4 MX/NV: record owner {:#x}, index owner {:#x}", h.owner, owner));
    }
    if h.kind != geo.kind {
        return Err(format!("WNR4 MX/NV: record kind {}, expected {}", h.kind, geo.kind));
    }
    if h.flags & !(FLAG_CEIL_MASK | FLAG_CZ) != 0 {
        return Err(format!("WNR4 MX/NV: unknown flags {:#04x}", h.flags));
    }
    if geo.kind == KIND_NV && (h.flags & FLAG_CEIL_MASK != 0) {
        return Err(format!("WNR4 NV: non-zero ceil flags {:#04x}", h.flags));
    }
    let cz = h.flags & FLAG_CZ != 0;
    if cz != value_exact {
        return Err(format!("WNR4 MX/NV: CZ flag {} does not match index VALUE_EXACT {}", cz as u8, value_exact as u8));
    }
    if rec[19] != 0 || !core::zero(rec, 28, 64) {
        return Err("WNR4 MX/NV: reserved header bytes not zero".to_string());
    }
    let anchors = [rec[16], rec[17], rec[18]];
    let anchors_word = anchors[0] as u32 | (anchors[1] as u32) << 8 | (anchors[2] as u32) << 16;
    let w4 = geo.kind as u32 | (h.flags as u32) << 8 | h.g << 16;
    let counts = [geo.cells[0] as u64, geo.cells[1] as u64, geo.cells[2] as u64];
    let ntiles = core::tile_count(&counts, h.g);
    let lay = core::layout(rec, &h, ntiles, 0)?;
    let spans = core::tile_spans(&counts, h.g);
    let spc = if geo.kind == KIND_NV { 2usize } else { 1usize };
    let b_nib = if geo.kind == KIND_NV { 16usize } else { 32usize };
    let r_lim = if geo.kind == KIND_NV { 6i32 } else { 3i32 };

    for (t, &(m, c0_u64, n_u32)) in spans.iter().enumerate() {
        let (c0, n) = (c0_u64 as usize, n_u32 as usize);
        let (ns, s0, cols) = (n * spc, c0 * spc, geo.scols[m] as usize);
        let ceil = (h.flags >> m) & 1 != 0;
        let mut tile = Tile::open(&rec[lay.tiles[t].clone()])?;
        let mut hash: Vec<Murmur> =
            (0..LANES).map(|l| lane_hash(model.id, owner, t as u32, l as u32, w4, anchors_word)).collect();

        // Phase 1: scales
        let scale_steps = (ns + LANES - 1) / LANES;
        let soff = geo.soff[m];
        let scount = geo.scount[m] as usize;
        for k in 0..scale_steps {
            for l in 0..LANES {
                let idx = LANES * k + l;
                if idx >= ns {
                    break;
                }
                let i = s0 + idx;
                let (sc, pred) = scale_ctx(geo.kind, &out[soff..soff + scount], i, s0, cols);
                let sym = tile.decode(l, model.s(m, sc))? as u8;
                let y = sym.wrapping_add(pred);
                out[soff + i] = y;
                hash[l].word(y as u32);
            }
        }

        // Phase 2: cells (32 nibbles per cell)
        let rounds = (n + LANES - 1) / LANES;
        let woff = geo.woff[m];
        for j in 0..rounds {
            let act = (n - LANES * j).min(LANES);
            let mut pm = [8usize; LANES];
            let mut mx = [8usize; LANES];
            let mut cls = [0usize; LANES];
            let mut wd = [[0u32; 4]; LANES];
            for k in 0..32usize {
                if k % b_nib == 0 {
                    for l in 0..act {
                        let scale = out[soff + (c0 + LANES * j + l) * spc + k / b_nib];
                        cls[l] = scale_cls(scale, anchors[m], r_lim);
                        pm[l] = 8;
                        mx[l] = 8;
                    }
                }
                for l in 0..act {
                    if k >= 28 && is_tail_item(n, LANES * j + l) {
                        continue;
                    }
                    let row = wrow(cls[l], pm[l], mx[l], model.pb, k % b_nib, b_nib);
                    let y = tile.decode(l, model.w(m, row))? as u32;
                    if cz && y == 8 {
                        return Err(format!("WNR4 MX/NV: -0 nibble under CZ in tile {}", t));
                    }
                    wd[l][k / 8] |= y << ((k % 8) * 4);
                    let a = (y & 7) as usize;
                    pm[l] = a;
                    mx[l] = if mx[l] == 8 { a } else { mx[l].max(a) };
                }
            }
            for l in 0..act {
                let cell = c0 + LANES * j + l;
                if is_tail_item(n, LANES * j + l) {
                    let tail = tile.tail(l)?;
                    for q in 0..4u32 {
                        let y = ((tail >> (4 * q)) & 15) as u32;
                        if cz && y == 8 {
                            return Err(format!("WNR4 MX/NV: -0 tail nibble under CZ in tile {}", t));
                        }
                        let a = (y & 7) as usize;
                        mx[l] = if mx[l] == 8 { a } else { mx[l].max(a) };
                    }
                    wd[l][3] |= (tail as u32) << 16;
                }
                for &word in &wd[l] {
                    hash[l].word(word);
                }
                if ceil && mx[l] == 5 {
                    let s_at = soff + cell;
                    if out[s_at] == 0 {
                        return Err(format!("WNR4 MX: ceil underflow at cell {} in tile {}", cell, t));
                    }
                    out[s_at] -= 1;
                    for word in &mut wd[l] {
                        *word = ceil_dbl8(*word);
                    }
                }
                let dst = woff + cell * 16;
                for k in 0..4 {
                    out[dst + 4 * k..dst + 4 * k + 4].copy_from_slice(&wd[l][k].to_le_bytes());
                }
            }
        }

        let mut h_xor = 0u32;
        for m_hash in hash {
            h_xor ^= m_hash.finish();
        }
        let owns = if n >= LANES { u32::MAX } else { (1u32 << n) - 1 };
        tile.finish(h_xor, owns)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wnr4::testenc::{self, Rng};

    fn make_geo(kind: u8, shapes: [(u32, u32); 3]) -> Geo {
        let spc = if kind == KIND_NV { 2 } else { 1 };
        let wmult = if kind == KIND_NV { 8 } else { 16 };
        let mut g = Geo {
            kind,
            rows: [0; 3],
            scols: [0; 3],
            scount: [0; 3],
            soff: [0; 3],
            woff: [0; 3],
            cells: [0; 3],
            raw: 0,
        };
        let mut at = 0usize;
        for m in 0..3 {
            g.rows[m] = shapes[m].0;
            g.scols[m] = shapes[m].1;
            g.scount[m] = shapes[m].0 * shapes[m].1;
            g.cells[m] = g.scount[m] / spc;
            g.soff[m] = at;
            at += g.scount[m] as usize;
        }
        for m in 0..3 {
            g.woff[m] = at;
            at += g.scount[m] as usize * wmult;
        }
        g.raw = at;
        g
    }

    fn make_model_blob(kind: u8, pb: u32, rng: &mut Rng) -> Vec<u8> {
        let wctx = (if kind == 1 { 1053 } else { 567 }) * pb as usize;
        let sctx = if kind == 1 { 4160 } else { 1 };
        let mut blob = Vec::with_capacity(32 + 3 * wctx * 34 + 3 * sctx * 514);
        for w in [WNM4_MAGIC, 1, kind as u32, core::PREC, wctx as u32, sctx as u32, pb, 0] {
            blob.extend_from_slice(&w.to_le_bytes());
        }
        // uniform + a few random tables keeps NV test fast
        let u16_cdf = testenc::cdf_from(&[1; 16]);
        let u256_cdf = testenc::cdf_from(&[1; 256]);
        let r16 = testenc::random_cdf(rng, 16);
        let r256 = testenc::random_cdf(rng, 256);
        let u16_b = testenc::table_bytes(&u16_cdf);
        let r16_b = testenc::table_bytes(&r16);
        let u256_b = testenc::table_bytes(&u256_cdf);
        let r256_b = testenc::table_bytes(&r256);
        for i in 0..3 * wctx {
            blob.extend_from_slice(if i % 7 == 0 { &r16_b } else { &u16_b });
        }
        for i in 0..3 * sctx {
            blob.extend_from_slice(if i % 13 == 0 { &r256_b } else { &u256_b });
        }
        blob
    }

    fn encode_mxnv(
        model: &Model,
        geo: &Geo,
        coded: &[u8],
        flags: u8,
        anchors: [u8; 3],
        owner: u32,
        g_val: u32,
    ) -> Vec<u8> {
        let spc = if geo.kind == KIND_NV { 2usize } else { 1usize };
        let b_nib = if geo.kind == KIND_NV { 16usize } else { 32usize };
        let r_lim = if geo.kind == KIND_NV { 6i32 } else { 3i32 };
        let anchors_word = anchors[0] as u32 | (anchors[1] as u32) << 8 | (anchors[2] as u32) << 16;
        let w4 = geo.kind as u32 | (flags as u32) << 8 | g_val << 16;
        let counts = [geo.cells[0] as u64, geo.cells[1] as u64, geo.cells[2] as u64];
        let spans = core::tile_spans(&counts, g_val);
        let mut tiles = Vec::with_capacity(spans.len());
        for (t, &(m, c0_u64, n_u32)) in spans.iter().enumerate() {
            let (c0, n) = (c0_u64 as usize, n_u32 as usize);
            let (ns, s0, cols) = (n * spc, c0 * spc, geo.scols[m] as usize);
            let soff = geo.soff[m];
            let woff = geo.woff[m];
            let s_slice = &coded[soff..soff + geo.scount[m] as usize];
            let mut events: Vec<(usize, &[u16], usize)> = Vec::new();
            let mut hash: Vec<Murmur> =
                (0..LANES).map(|l| lane_hash(model.id, owner, t as u32, l as u32, w4, anchors_word)).collect();
            for k in 0..(ns + LANES - 1) / LANES {
                for l in 0..LANES {
                    let idx = LANES * k + l;
                    if idx >= ns {
                        break;
                    }
                    let i = s0 + idx;
                    let (sc, pred) = scale_ctx(geo.kind, s_slice, i, s0, cols);
                    let y = s_slice[i];
                    let sym = y.wrapping_sub(pred) as usize;
                    events.push((l, model.s(m, sc), sym));
                    hash[l].word(y as u32);
                }
            }
            let mut tails = [None; LANES];
            for j in 0..(n + LANES - 1) / LANES {
                let act = (n - LANES * j).min(LANES);
                let mut pm = [8usize; LANES];
                let mut mx = [8usize; LANES];
                let mut cls = [0usize; LANES];
                for k in 0..32usize {
                    if k % b_nib == 0 {
                        for l in 0..act {
                            cls[l] = scale_cls(s_slice[(c0 + LANES * j + l) * spc + k / b_nib], anchors[m], r_lim);
                            pm[l] = 8;
                            mx[l] = 8;
                        }
                    }
                    for l in 0..act {
                        let cell = c0 + LANES * j + l;
                        let byte = coded[woff + cell * 16 + k / 2];
                        let y = ((byte >> ((k & 1) * 4)) & 15) as usize;
                        if k >= 28 && is_tail_item(n, LANES * j + l) {
                            continue;
                        }
                        let row = wrow(cls[l], pm[l], mx[l], model.pb, k % b_nib, b_nib);
                        events.push((l, model.w(m, row), y));
                        let a = y & 7;
                        pm[l] = a;
                        mx[l] = if mx[l] == 8 { a } else { mx[l].max(a) };
                    }
                }
                for l in 0..act {
                    let cell = c0 + LANES * j + l;
                    let base = woff + cell * 16;
                    if is_tail_item(n, LANES * j + l) {
                        let tail = u16::from_le_bytes([coded[base + 14], coded[base + 15]]);
                        tails[l] = Some(tail);
                    }
                    for k in 0..4 {
                        hash[l].word(le32(coded, base + 4 * k));
                    }
                }
            }
            let mut h_xor = 0u32;
            for mh in hash {
                h_xor ^= mh.finish();
            }
            tiles.push(testenc::encode_tile(&events, &testenc::end_states(h_xor, &tails)));
        }
        let mut hdr = [0u8; core::HEADER_BYTES];
        hdr[0..4].copy_from_slice(&core::WNR4_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&model.id.to_le_bytes());
        hdr[8..12].copy_from_slice(&owner.to_le_bytes());
        hdr[12] = geo.kind;
        hdr[13] = flags;
        hdr[14..16].copy_from_slice(&(g_val as u16).to_le_bytes());
        hdr[16..19].copy_from_slice(&anchors);
        testenc::record(&hdr, &[], &tiles)
    }

    #[test]
    fn round_trip_mx_and_nv() {
        let mut rng = Rng::new(20261127);
        for &(kind, pb, cz, ceil) in &[
            (KIND_MX, 1u32, false, false),
            (KIND_MX, 5u32, true, true),
            (KIND_NV, 1u32, false, false),
            (KIND_NV, 5u32, true, false),
        ] {
            let shapes = if kind == KIND_NV { [(6, 36), (4, 12), (6, 36)] } else { [(7, 11), (5, 9), (4, 13)] };
            let geo = make_geo(kind, shapes);
            let blob = make_model_blob(kind, pb, &mut rng);
            let model = model_open(&blob).unwrap();
            let mut coded = vec![0u8; geo.raw];
            for m in 0..3 {
                for i in 0..geo.scount[m] as usize {
                    coded[geo.soff[m] + i] = 100 + (rng.below(15) as u8);
                }
                for c in 0..geo.cells[m] as usize {
                    let max_a = if ceil && c % 2 == 0 { 5 } else { 7 };
                    for j in 0..32usize {
                        let a = if j == 31 && ceil && c % 2 == 0 { 5 } else { rng.below(max_a + 1) };
                        let mut x = a | ((rng.next() as u32 & 1) << 3);
                        if cz && x == 8 {
                            x = 0;
                        }
                        if j & 1 == 0 {
                            coded[geo.woff[m] + c * 16 + j / 2] = x as u8;
                        } else {
                            coded[geo.woff[m] + c * 16 + j / 2] |= (x as u8) << 4;
                        }
                    }
                }
            }
            let mut want = coded.clone();
            if ceil {
                for m in 0..3 {
                    for c in 0..geo.cells[m] as usize {
                        let base = geo.woff[m] + c * 16;
                        let mut mx = 0u32;
                        for b in &want[base..base + 16] {
                            mx = mx.max((b & 7) as u32).max(((b >> 4) & 7) as u32);
                        }
                        if mx == 5 {
                            want[geo.soff[m] + c] -= 1;
                            for k in 0..4 {
                                let w = ceil_dbl8(le32(&want, base + 4 * k));
                                want[base + 4 * k..base + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
                            }
                        }
                    }
                }
            }
            let flags = (if ceil { FLAG_CEIL_MASK } else { 0 }) | (if cz { FLAG_CZ } else { 0 });
            let rec = encode_mxnv(&model, &geo, &coded, flags, [105, 105, 105], 0x0001_0002, 2);
            let mut out = vec![0u8; geo.raw];
            decode(&rec, &model, 0x0001_0002, cz, &geo, &mut out).unwrap();
            assert_eq!(out, want, "kind {} pb {} cz {} ceil {}", kind, pb, cz, ceil);
            // refusals: wrong owner, wrong value_exact, flipped byte
            assert!(decode(&rec, &model, 0x0001_0003, cz, &geo, &mut out).is_err());
            assert!(decode(&rec, &model, 0x0001_0002, !cz, &geo, &mut out).is_err());
            let mut bad = rec.clone();
            bad[140] ^= 1;
            assert!(decode(&bad, &model, 0x0001_0002, cz, &geo, &mut out).is_err());
        }
    }

    #[test]
    fn cz_and_ceil_errors_refused() {
        let mut rng = Rng::new(99);
        let geo = make_geo(KIND_MX, [(2, 4), (2, 4), (2, 4)]);
        let blob = make_model_blob(KIND_MX, 1, &mut rng);
        let model = model_open(&blob).unwrap();
        // CZ with -0 nibble
        let mut coded = vec![0u8; geo.raw];
        coded[geo.woff[0]] = 0x08;
        let rec = encode_mxnv(&model, &geo, &coded, FLAG_CZ, [0, 0, 0], 1, 2);
        let mut out = vec![0u8; geo.raw];
        assert!(decode(&rec, &model, 1, true, &geo, &mut out).is_err());
        // Ceil underflow: scale 0 with max magnitude 5
        coded[geo.woff[0]] = 0x05;
        coded[geo.soff[0]] = 0;
        let rec = encode_mxnv(&model, &geo, &coded, 1, [0, 0, 0], 1, 2);
        assert!(decode(&rec, &model, 1, false, &geo, &mut out).is_err());
    }
}
