// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! The rANS decode step of WNR3-SPEC erratum 10, shared by WNR3 and WBP1: 32-bit state,
//! L = 2^23, 14-bit CDFs, byte renormalisation, a 4-byte big-endian initial state.

pub const L: u32 = 1 << 23;
pub const TOTAL: u16 = 1 << 14;

/// A CDF is usable when it starts at 0, ends at 16384 and is strictly increasing.
pub fn cdf_ok(cdf: &[u16]) -> bool {
    cdf.len() >= 2 && cdf[0] == 0 && cdf[cdf.len() - 1] == TOTAL && cdf.windows(2).all(|w| w[0] < w[1])
}

pub struct Stream<'a> {
    x: u32,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Stream<'a> {
    /// The first 4 bytes, big-endian, must be in [L, 256 L).
    pub fn open(buf: &'a [u8]) -> Option<Stream<'a>> {
        if buf.len() < 4 {
            return None;
        }
        let x = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if !(L..L << 8).contains(&x) {
            return None;
        }
        Some(Stream { x, buf, pos: 4 })
    }

    /// One symbol with `cdf` (already checked by `cdf_ok`); None when a byte is needed and none is left.
    pub fn symbol(&mut self, cdf: &[u16]) -> Option<usize> {
        let slot = self.x & (TOTAL as u32 - 1);
        // the s with cdf[s] <= slot < cdf[s + 1]
        let s = cdf[1..].partition_point(|&c| (c as u32) <= slot);
        let start = cdf[s] as u32;
        let freq = cdf[s + 1] as u32 - start;
        self.x = freq * (self.x >> 14) + slot - start;
        while self.x < L {
            let b = *self.buf.get(self.pos)?;
            self.pos += 1;
            self.x = (self.x << 8) | b as u32;
        }
        Some(s)
    }

    /// After the last symbol: every byte read and the state equal to `expect`.
    pub fn done(&self, expect: u32) -> bool {
        self.pos == self.buf.len() && self.x == expect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testenc::{cdf_from, rans_encode};

    #[test]
    fn round_trip() {
        // skewed frequencies, every one at least 1
        let counts: Vec<u64> = (0..256).map(|s| if s < 4 { 3000 } else { 16 }).collect();
        let cdf = cdf_from(&counts);
        assert!(cdf_ok(&cdf));
        let symbols: Vec<usize> = (0..5000).map(|i| (i * 7 + i / 13) % 256).collect();
        let x0 = L + 12345;
        let bytes = rans_encode(&symbols, &vec![&cdf[..]; symbols.len()], x0);
        let mut s = Stream::open(&bytes).unwrap();
        for &want in &symbols {
            assert_eq!(s.symbol(&cdf), Some(want));
        }
        assert!(s.done(x0));
        // one byte short: either a symbol cannot be read or the stream does not end in its state
        let short = &bytes[..bytes.len() - 1];
        let mut s = Stream::open(short).unwrap();
        let ok = symbols.iter().all(|&w| s.symbol(&cdf) == Some(w)) && s.done(x0);
        assert!(!ok);
    }

    #[test]
    fn initial_state_range() {
        assert!(Stream::open(&[0x00, 0x7F, 0xFF, 0xFF]).is_none());
        assert!(Stream::open(&[0x00, 0x80, 0x00, 0x00]).is_some());
        assert!(Stream::open(&[0x7F, 0xFF, 0xFF, 0xFF]).is_some());
        assert!(Stream::open(&[0x80, 0x00, 0x00, 0x00]).is_none());
        assert!(Stream::open(&[0x00, 0x80, 0x00]).is_none());
    }
}
