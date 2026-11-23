// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! What every WNR4 kind shares (docs/WNR4-SPEC.md sections 1, 3 and 4): the rANS of section 1 (L = 2^16,
//! 14-bit tables, 16-bit words), the record header and its tile offset table (section 3), the 32 lanes of a
//! tile over one shared word stream, the tail in the state, the lane hash and the end states (section 4),
//! and the header and table checks of a WNM4 model blob (section 2). The kinds (gg.rs, i4.rs) bring their
//! own geometry, model layout and phases. Every failure is an `Err` with a message; only valid / invalid is
//! normative, the message is not.

use crate::hash::{fnv1a, Murmur};

/// "WNR4", the record magic and the seed of every lane hash.
pub const WNR4_MAGIC: u32 = 0x3452_4E57;
/// "WNM4", the model blob magic.
pub const WNM4_MAGIC: u32 = 0x344D_4E57;
/// The lane hash seed (section 4): the record magic.
pub const SEED: u32 = WNR4_MAGIC;
/// The rANS lower bound: a state is in [L, 2^32).
pub const L: u32 = 1 << 16;
/// Table precision: every table sums to T = 2^14.
pub const PREC: u32 = 14;
pub const T: u32 = 1 << PREC;
/// Lanes per tile.
pub const LANES: usize = 32;
/// Bytes of the record header (section 3).
pub const HEADER_BYTES: usize = 64;
/// The largest G (cells, blocks or groups per lane per tile).
pub const G_MAX: u32 = 4096;
/// The 32 initial states that start every tile.
pub const STATE_BYTES: usize = 4 * LANES;

pub fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

pub fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// n rounded up to a multiple of 4096 (n at most 2^62).
pub fn align4096(n: u64) -> u64 {
    (n + 4095) & !4095
}

/// ceil(a / b), b > 0.
pub fn ceil_div(a: u64, b: u64) -> u64 {
    a / b + u64::from(a % b != 0)
}

/// The items a lane owns of a tile of `n` items, lane l owning items l, l + 32, ..: ceil((n - l) / 32), 0
/// when l >= n (section 4: `ls` and `lc`).
pub fn lane_items(n: u32, l: usize) -> u32 {
    let l = l as u32;
    if l >= n {
        0
    } else {
        (n - l + 31) / 32
    }
}

/// The index in the tile of the last item of lane l (l < n): the item whose tail rides in the lane's state
/// (round `(n - l - 1) / 32`, sections 4, 5.4 and 6.4).
pub fn last_item(n: usize, l: usize) -> usize {
    debug_assert!(l < n);
    l + (n - l - 1) / 32 * 32
}

/// True when item i of a tile of n items is the last item of a tail lane (lanes 2..31): its tail symbols
/// are not coded. Item i belongs to lane i % 32 and is its last when no item i + 32 exists.
pub fn is_tail_item(n: usize, i: usize) -> bool {
    i % 32 >= 2 && i + 32 >= n
}

/// A 14-bit cumulative table as section 2 says: first entry 0, last 16384, strictly increasing.
pub fn valid_cdf(t: &[u16]) -> bool {
    t.len() >= 2 && t[0] == 0 && t[t.len() - 1] as u32 == T && t.windows(2).all(|w| w[0] < w[1])
}

/// The 8 header words of a WNM4 blob, after checking the magic, version 1, the kind, precision 14 at word 3,
/// the four kind-specific words 4..7 and the exact size. Returns the model id: FNV-1a of the whole blob,
/// 0 replaced by 1.
pub fn wnm4_open(blob: &[u8], kind: u32, words4to7: [u32; 4], size: usize) -> Result<u32, String> {
    if blob.len() != size {
        return Err(format!("WNM4 kind {}: {} bytes, the layout has {}", kind, blob.len(), size));
    }
    let w: Vec<u32> = (0..8).map(|i| le32(blob, 4 * i)).collect();
    if w[0] != WNM4_MAGIC {
        return Err(format!("WNM4: magic {:#010x}", w[0]));
    }
    if w[1] != 1 {
        return Err(format!("WNM4: version {}, only 1 is defined", w[1]));
    }
    if w[2] != kind {
        return Err(format!("WNM4: kind {}, expected {}", w[2], kind));
    }
    if w[3] != PREC {
        return Err(format!("WNM4: precision word {}, expected 14", w[3]));
    }
    if w[4..8] != words4to7 {
        return Err(format!("WNM4 kind {}: header words 4..7 {:?}, expected {:?}", kind, &w[4..8], words4to7));
    }
    Ok(fnv1a(blob))
}

/// `count` cumulative tables of `n` symbols (n + 1 u16 each) from byte `off` of a blob, each checked.
pub fn read_tables(blob: &[u8], off: usize, count: usize, n: usize, what: &str) -> Result<Vec<u16>, String> {
    let w = n + 1;
    let mut out = Vec::with_capacity(count * w);
    for i in 0..count * w {
        out.push(le16(blob, off + 2 * i));
    }
    for (i, t) in out.chunks_exact(w).enumerate() {
        if !valid_cdf(t) {
            return Err(format!("WNM4: {} table {} is not a 14-bit cumulative table", what, i));
        }
    }
    Ok(out)
}

/// The fields of the 64-byte record header (section 3) that every kind reads the same way. Bytes 16..19
/// and 28..63 are kind-specific and left to the kind.
#[derive(Clone, Debug)]
pub struct Header {
    pub model_id: u32,
    pub owner: u32,
    pub kind: u8,
    pub flags: u8,
    pub g: u32,
    pub ntiles: u32,
    pub payload: u32,
}

/// Reads the record header: at least 64 bytes, magic, model id not 0, G in 1..4096.
pub fn header(rec: &[u8]) -> Result<Header, String> {
    if rec.len() < HEADER_BYTES {
        return Err(format!("WNR4 record of {} bytes, shorter than its header", rec.len()));
    }
    if le32(rec, 0) != WNR4_MAGIC {
        return Err(format!("WNR4: magic {:#010x}", le32(rec, 0)));
    }
    let h = Header {
        model_id: le32(rec, 4),
        owner: le32(rec, 8),
        kind: rec[12],
        flags: rec[13],
        g: le16(rec, 14) as u32,
        ntiles: le32(rec, 20),
        payload: le32(rec, 24),
    };
    if h.model_id == 0 {
        return Err("WNR4: model id 0".to_string());
    }
    if h.g == 0 || h.g > G_MAX {
        return Err(format!("WNR4: G = {}, not in 1..4096", h.g));
    }
    Ok(h)
}

/// True when every byte of rec[a..b] is 0.
pub fn zero(rec: &[u8], a: usize, b: usize) -> bool {
    rec[a..b].iter().all(|&v| v == 0)
}

/// Where the parts of a record are, as byte ranges of the record, after the checks of section 3.
#[derive(Clone, Debug)]
pub struct Layout {
    /// The kind's area between the offset table and the payload (GG: the verbatim area; I4: the bases).
    pub extra: std::ops::Range<usize>,
    /// Tile t: its 32 initial states and its words.
    pub tiles: Vec<std::ops::Range<usize>>,
}

/// The checks of section 3 that do not depend on the kind: ntiles equal to `ntiles_expected` (the count
/// from the geometry), the offset table (off[0] = 0, every tile at least 128 bytes and even, off[ntiles] =
/// P), the record size `align4096(64 + 4 (ntiles + 1) + extra + P)` and the zero padding after the payload.
/// `extra` is the size of the kind's area between the offset table and the payload.
pub fn layout(rec: &[u8], h: &Header, ntiles_expected: u64, extra: u64) -> Result<Layout, String> {
    if h.ntiles as u64 != ntiles_expected {
        return Err(format!("WNR4: ntiles {}, the geometry gives {}", h.ntiles, ntiles_expected));
    }
    let nt = h.ntiles as u64;
    let table = HEADER_BYTES as u64 + 4 * (nt + 1);
    let extra_end = table + extra;
    let end = extra_end + h.payload as u64;
    let total = align4096(end);
    if rec.len() as u64 != total {
        return Err(format!("WNR4: record of {} bytes, its header and geometry give {}", rec.len(), total));
    }
    if !zero(rec, end as usize, rec.len()) {
        return Err("WNR4: a non-zero byte in the padding after the payload".to_string());
    }
    let base = extra_end as usize;
    let mut tiles = Vec::with_capacity(nt as usize);
    let mut prev = le32(rec, HEADER_BYTES);
    if prev != 0 {
        return Err(format!("WNR4: off[0] = {}, not 0", prev));
    }
    for t in 0..nt as usize {
        let next = le32(rec, HEADER_BYTES + 4 * (t + 1));
        if next < prev || next - prev < STATE_BYTES as u32 || (next - prev) % 2 != 0 {
            return Err(format!("WNR4: tile {} spans off {} .. {}: not at least 128 bytes and even", t, prev, next));
        }
        tiles.push(base + prev as usize..base + next as usize);
        prev = next;
    }
    if prev != h.payload {
        return Err(format!("WNR4: off[ntiles] = {}, the payload has {} bytes", prev, h.payload));
    }
    Ok(Layout { extra: table as usize..base, tiles })
}

/// The tiles of a sequence of segments (matrices, coded slices) of `counts[s]` items each, at G items per
/// lane: (segment, first item c0, items n) per tile, in record order (section 3).
pub fn tile_spans(counts: &[u64], g: u32) -> Vec<(usize, u64, u32)> {
    let per = 32 * g as u64;
    let mut out = Vec::new();
    for (s, &c) in counts.iter().enumerate() {
        let mut c0 = 0;
        while c0 < c {
            out.push((s, c0, (c - c0).min(per) as u32));
            c0 += per;
        }
    }
    out
}

/// sum over the segments of ceil(counts[s] / (32 G)).
pub fn tile_count(counts: &[u64], g: u32) -> u64 {
    counts.iter().map(|&c| ceil_div(c, 32 * g as u64)).sum()
}

/// The 32 lanes of one tile over its shared word stream (section 4, "Words").
pub struct Tile<'a> {
    x: [u32; LANES],
    words: &'a [u8],
    next: usize,
}

impl<'a> Tile<'a> {
    /// A tile's bytes: 32 u32 initial states in lane order, each at least L, then its u16 words.
    pub fn open(bytes: &'a [u8]) -> Result<Tile<'a>, String> {
        if bytes.len() < STATE_BYTES || bytes.len() % 2 != 0 {
            return Err(format!("WNR4: a tile of {} bytes", bytes.len()));
        }
        let mut x = [0u32; LANES];
        for (l, s) in x.iter_mut().enumerate() {
            *s = le32(bytes, 4 * l);
            if *s < L {
                return Err(format!("WNR4: initial state of lane {} is {:#x}, below L", l, *s));
            }
        }
        Ok(Tile { x, words: &bytes[STATE_BYTES..], next: 0 })
    }

    /// One symbol of lane `lane` with table `cdf` (a checked cumulative table of n + 1 entries), then at most
    /// one renormalisation, which takes the next word of the tile. Reading past the tile's words is refused.
    pub fn decode(&mut self, lane: usize, cdf: &[u16]) -> Result<usize, String> {
        let x = self.x[lane];
        let slot = x & (T - 1);
        // the largest s with cdf[s] <= slot: cdf[0] = 0 <= slot < cdf[n] = T, so 0 <= s < n
        let s = cdf.partition_point(|&c| c as u32 <= slot) - 1;
        let start = cdf[s] as u32;
        let freq = cdf[s + 1] as u32 - start;
        // freq * (x >> 14) + (slot - start) < freq * ((x >> 14) + 1) <= 2^14 * 2^18: no overflow
        let mut y = freq * (x >> PREC) + slot - start;
        if y < L {
            if self.next + 2 > self.words.len() {
                return Err(format!("WNR4: lane {} reads past the tile's {} words", lane, self.words.len() / 2));
            }
            y = (y << 16) | le16(self.words, self.next) as u32;
            self.next += 2;
        }
        self.x[lane] = y;
        Ok(s)
    }

    /// The 16 tail bits of a tail lane: x_final - L, with L <= x_final < L + 2^16 required.
    pub fn tail(&self, lane: usize) -> Result<u16, String> {
        let x = self.x[lane];
        if !(L..L + (1 << 16)).contains(&x) {
            return Err(format!("WNR4: tail lane {} ends at {:#x}, not in [L, L + 2^16)", lane, x));
        }
        Ok((x - L) as u16)
    }

    /// The end checks of section 4, after every symbol of the tile: every word used; lanes 0 and 1 at
    /// L + (H & 0xFFFF) and L + (H >> 16) (whether or not they own items); lanes 2..31 that own no item at
    /// L, the others (tail lanes) in [L, L + 2^16). Bit l of `owns` is set when lane l owns an item.
    pub fn finish(&self, h: u32, owns: u32) -> Result<(), String> {
        if self.next != self.words.len() {
            return Err(format!("WNR4: {} of the tile's {} words used", self.next / 2, self.words.len() / 2));
        }
        if self.x[0] != L + (h & 0xFFFF) || self.x[1] != L + (h >> 16) {
            return Err(format!(
                "WNR4: tile hash {:#010x}, lanes 0 and 1 end at {:#x} and {:#x}",
                h, self.x[0], self.x[1]
            ));
        }
        for l in 2..LANES {
            if owns >> l & 1 == 1 {
                self.tail(l)?;
            } else if self.x[l] != L {
                return Err(format!("WNR4: lane {} owns nothing and ends at {:#x}, not L", l, self.x[l]));
            }
        }
        Ok(())
    }
}

/// A lane hash started with the six ident words of section 4: model_id, owner, t, l, then the kind word
/// (kind | flags << 8 | G << 16) and the kind's sixth word (MX/NV: the anchors; GG: the ggml type; I4:
/// m | anchor << 16). The kind adds its 4-byte blocks with `word` and ends with `finish`; every block is a
/// whole word, so MurmurHash3's length is 4 x the number of blocks, the `hlen` of each kind.
pub fn lane_hash(model_id: u32, owner: u32, t: u32, l: u32, w4: u32, w5: u32) -> Murmur {
    let mut m = Murmur::new(SEED);
    for w in [model_id, owner, t, l, w4, w5] {
        m.word(w);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wnr4::testenc::{self, Rng};

    fn cdf_of(rng: &mut Rng, n: usize) -> Vec<u16> {
        testenc::random_cdf(rng, n)
    }

    #[test]
    fn cdf_rules() {
        assert!(valid_cdf(&[0, 16384]));
        assert!(valid_cdf(&[0, 1, 16384]));
        assert!(!valid_cdf(&[0, 0, 16384]));
        assert!(!valid_cdf(&[1, 2, 16384]));
        assert!(!valid_cdf(&[0, 2, 16383]));
        assert!(!valid_cdf(&[0, 9000, 8000, 16384]));
        assert!(!valid_cdf(&[0]));
    }

    #[test]
    fn lanes_and_tiles() {
        assert_eq!(lane_items(0, 0), 0);
        assert_eq!(lane_items(1, 0), 1);
        assert_eq!(lane_items(1, 1), 0);
        assert_eq!(lane_items(33, 0), 2);
        assert_eq!(lane_items(33, 1), 1);
        assert_eq!(lane_items(64, 31), 2);
        assert_eq!(lane_items(65, 31), 2);
        assert_eq!(tile_count(&[0, 1, 64, 65], 1), 6);
        assert_eq!(tile_spans(&[65, 0, 3], 1), vec![(0, 0, 32), (0, 32, 32), (0, 64, 1), (2, 0, 3)]);
        assert_eq!(tile_spans(&[200], 4096), vec![(0, 0, 200)]);
        // the last item of each lane, and which items carry a tail: lanes 0 and 1 never do
        for n in 1..200usize {
            for l in 0..LANES.min(n) {
                let last = last_item(n, l);
                assert_eq!(last, *(l..n).step_by(32).collect::<Vec<usize>>().last().unwrap(), "n {} lane {}", n, l);
                for i in (l..n).step_by(32) {
                    assert_eq!(is_tail_item(n, i), l >= 2 && i == last, "n {} item {}", n, i);
                }
            }
        }
        assert_eq!(align4096(0), 0);
        assert_eq!(align4096(1), 4096);
        assert_eq!(align4096(4096), 4096);
    }

    /// A random schedule: positions with a random subset of lanes, each symbol with a random table of 16
    /// or 256 symbols; lanes 2..31 that own items keep 16 tail bits in the state.
    struct Schedule {
        tables: Vec<Vec<u16>>,
        events: Vec<(usize, usize, usize)>, // lane, table, symbol
        owns: u32,
        tails: [Option<u16>; 32],
        h: u32,
    }

    fn schedule(rng: &mut Rng, positions: usize, density: u32) -> Schedule {
        let tables: Vec<Vec<u16>> = (0..6).map(|i| cdf_of(rng, if i % 2 == 0 { 16 } else { 256 })).collect();
        let mut events = Vec::new();
        for _ in 0..positions {
            for l in 0..LANES {
                if rng.below(100) < density {
                    let t = rng.below(tables.len() as u32) as usize;
                    let n = tables[t].len() - 1;
                    events.push((l, t, rng.below(n as u32) as usize));
                }
            }
        }
        let owns = rng.next() as u32 | 3;
        let mut tails = [None; 32];
        for (l, t) in tails.iter_mut().enumerate().skip(2) {
            if owns >> l & 1 == 1 {
                *t = Some(rng.next() as u16);
            }
        }
        Schedule { tables, events, owns, tails, h: rng.next() as u32 }
    }

    fn encode(s: &Schedule) -> Vec<u8> {
        let ev: Vec<(usize, &[u16], usize)> = s.events.iter().map(|&(l, t, y)| (l, &s.tables[t][..], y)).collect();
        testenc::encode_tile(&ev, &testenc::end_states(s.h, &s.tails))
    }

    fn decode(s: &Schedule, bytes: &[u8]) -> Result<Vec<usize>, String> {
        let mut tile = Tile::open(bytes)?;
        let mut out = Vec::new();
        for &(l, t, _) in &s.events {
            out.push(tile.decode(l, &s.tables[t])?);
        }
        for l in 2..LANES {
            if let Some(v) = s.tails[l] {
                if tile.tail(l)? != v {
                    return Err("tail".to_string());
                }
            }
        }
        tile.finish(s.h, s.owns)?;
        Ok(out)
    }

    #[test]
    fn round_trip_random_schedules() {
        let mut rng = Rng::new(0x5745_4e52);
        for case in 0..300 {
            let s = schedule(&mut rng, case % 40, [0, 5, 50, 100][case % 4]);
            let bytes = encode(&s);
            let got = decode(&s, &bytes).unwrap_or_else(|e| panic!("case {}: {}", case, e));
            let want: Vec<usize> = s.events.iter().map(|e| e.2).collect();
            assert_eq!(got, want, "case {}", case);
        }
    }

    #[test]
    fn end_state_failures() {
        let mut rng = Rng::new(7);
        let mut checked = 0;
        for _ in 0..50 {
            let s = schedule(&mut rng, 20, 60);
            let bytes = encode(&s);
            decode(&s, &bytes).unwrap();
            // another hash: lanes 0 and 1 no longer end at it
            let mut bad = Schedule { h: s.h ^ 1, ..clone_schedule(&s) };
            assert!(decode(&bad, &bytes).is_err());
            bad.h = s.h ^ 0x1_0000;
            assert!(decode(&bad, &bytes).is_err());
            // a lane 2..31 without items must end at L; one that ends elsewhere is refused
            if let Some(l) = (2..LANES).find(|&l| matches!(s.tails[l], Some(v) if v != 0)) {
                let mut nb = clone_schedule(&s);
                nb.owns &= !(1 << l);
                nb.tails[l] = None;
                assert!(decode(&nb, &bytes).is_err());
                checked += 1;
            }
            // tail lanes must end in [L, L + 2^16): a lane told to end there by the encoder but given a state
            // above is refused
            let mut tails = s.tails;
            let l = 2 + rng.below(30) as usize;
            tails[l] = Some(0);
            let mut ends = testenc::end_states(s.h, &tails);
            ends[l] = L + (1 << 16) + rng.below(1000);
            let ev: Vec<(usize, &[u16], usize)> = s.events.iter().map(|&(l, t, y)| (l, &s.tables[t][..], y)).collect();
            let bytes = testenc::encode_tile(&ev, &ends);
            let mut tile = Tile::open(&bytes).unwrap();
            for &(l, t, _) in &s.events {
                tile.decode(l, &s.tables[t]).unwrap();
            }
            assert!(tile.tail(l).is_err());
            assert!(tile.finish(s.h, s.owns | 1 << l).is_err());
        }
        assert!(checked > 10);
    }

    fn clone_schedule(s: &Schedule) -> Schedule {
        Schedule { tables: s.tables.clone(), events: s.events.clone(), owns: s.owns, tails: s.tails, h: s.h }
    }

    #[test]
    fn word_count_exact() {
        let mut rng = Rng::new(11);
        let mut short = 0;
        for _ in 0..100 {
            let s = schedule(&mut rng, 30, 80);
            let bytes = encode(&s);
            // one word more: refused at the end (word count)
            let mut long = bytes.clone();
            long.extend_from_slice(&[0x12, 0x34]);
            assert!(decode(&s, &long).is_err());
            // one word less: some lane reads past the tile, or (when no word is left to read) the states
            // differ; either way refused, and a read past the end never panics
            if bytes.len() > STATE_BYTES {
                let cut = &bytes[..bytes.len() - 2];
                assert!(decode(&s, cut).is_err());
                short += 1;
            }
        }
        assert!(short > 50);
    }

    #[test]
    fn read_past_tile_refused() {
        // every lane at L: the first symbol of a lane renormalises at once, and there is no word
        let mut bytes = Vec::new();
        for _ in 0..LANES {
            bytes.extend_from_slice(&L.to_le_bytes());
        }
        let cdf: Vec<u16> = (0..=16).map(|i| (i * 1024) as u16).collect();
        let mut tile = Tile::open(&bytes).unwrap();
        let e = tile.decode(5, &cdf).unwrap_err();
        assert!(e.contains("past"), "{}", e);
    }

    #[test]
    fn initial_state_below_l_refused() {
        let mut bytes = vec![0u8; STATE_BYTES];
        for l in 0..LANES {
            bytes[4 * l..4 * l + 4].copy_from_slice(&L.to_le_bytes());
        }
        assert!(Tile::open(&bytes).is_ok());
        for l in [0, 1, 17, 31] {
            let mut b = bytes.clone();
            b[4 * l..4 * l + 4].copy_from_slice(&(L - 1).to_le_bytes());
            assert!(Tile::open(&b).is_err());
        }
        assert!(Tile::open(&bytes[..STATE_BYTES - 2]).is_err());
        let mut odd = bytes.clone();
        odd.push(0);
        assert!(Tile::open(&odd).is_err());
    }

    fn record(ntiles: u32, extra: usize, tiles: &[Vec<u8>]) -> Vec<u8> {
        let mut hdr = [0u8; HEADER_BYTES];
        hdr[0..4].copy_from_slice(&WNR4_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        hdr[12] = 2;
        hdr[14..16].copy_from_slice(&32u16.to_le_bytes());
        let mut r = testenc::record(&hdr, &vec![0u8; extra], tiles);
        r[20..24].copy_from_slice(&ntiles.to_le_bytes());
        r
    }

    #[test]
    fn header_and_offsets() {
        let tiles = vec![vec![0u8; 128], vec![0u8; 130], vec![0u8; 200]];
        let r = record(3, 8, &tiles);
        let h = header(&r).unwrap();
        assert_eq!((h.ntiles, h.payload, h.g, h.kind), (3, 458, 32, 2));
        let lay = layout(&r, &h, 3, 8).unwrap();
        assert_eq!(lay.extra, 80..88);
        assert_eq!(lay.tiles, vec![88..216, 216..346, 346..546]);
        assert_eq!(r.len(), 4096);
        // the count from the geometry
        assert!(layout(&r, &h, 2, 8).is_err());
        // a wrong extra size moves the end of the payload
        assert!(layout(&r, &h, 3, 4).is_err() || layout(&r, &h, 3, 4).unwrap().tiles != lay.tiles);
        // padding
        let mut b = r.clone();
        b[4095] = 1;
        assert!(layout(&b, &header(&b).unwrap(), 3, 8).is_err());
        // a page more of zeros
        let mut b = r.clone();
        b.extend_from_slice(&[0u8; 4096]);
        assert!(layout(&b, &header(&b).unwrap(), 3, 8).is_err());
        // off[0] != 0, a tile below 128 bytes, an odd tile, off[ntiles] != P
        let set = |r: &[u8], i: usize, v: u32| {
            let mut b = r.to_vec();
            b[64 + 4 * i..68 + 4 * i].copy_from_slice(&v.to_le_bytes());
            b
        };
        for (i, v) in [(0, 2), (1, 126), (1, 129), (2, 128), (3, 456), (3, 460), (2, 900)] {
            let b = set(&r, i, v);
            assert!(layout(&b, &header(&b).unwrap(), 3, 8).is_err(), "off[{}] = {}", i, v);
        }
        // magic, model id 0, G 0 and 4097
        for (at, v) in [(0usize, 0u8), (4, 0)] {
            let mut b = r.clone();
            b[at..at + 4].fill(v);
            assert!(header(&b).is_err());
        }
        for g in [0u16, 4097, 65535] {
            let mut b = r.clone();
            b[14..16].copy_from_slice(&g.to_le_bytes());
            assert!(header(&b).is_err());
        }
        assert!(header(&r[..63]).is_err());
        // no tiles at all: off[0] = 0 = P
        let r0 = record(0, 0, &[]);
        let h0 = header(&r0).unwrap();
        assert_eq!(layout(&r0, &h0, 0, 0).unwrap().tiles.len(), 0);
    }

    #[test]
    fn wnm4_header_checks() {
        let mut blob = vec![0u8; 64];
        for (i, w) in [WNM4_MAGIC, 1, 2, 14, 32, 2, 0, 0].iter().enumerate() {
            blob[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(wnm4_open(&blob, 2, [32, 2, 0, 0], 64).unwrap(), fnv1a(&blob));
        assert!(wnm4_open(&blob, 2, [32, 2, 0, 0], 65).is_err());
        assert!(wnm4_open(&blob, 3, [32, 2, 0, 0], 64).is_err());
        for i in 0..8 {
            let mut b = blob.clone();
            b[4 * i] ^= 1;
            assert!(wnm4_open(&b, 2, [32, 2, 0, 0], 64).is_err(), "word {}", i);
        }
    }

    #[test]
    fn lane_hash_is_murmur_of_the_words() {
        let words = [1u32, 2, 3, 4, 5, 6, 0xAB, 0xFFFF_FFFF];
        let mut m = lane_hash(1, 2, 3, 4, 5, 6);
        m.word(0xAB);
        m.word(0xFFFF_FFFF);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(m.finish(), crate::hash::murmur3(&bytes, SEED));
    }
}
