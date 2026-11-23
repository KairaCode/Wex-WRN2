// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// A WNR4 tile encoder for the tests only, derived from the decoder of WNR4-SPEC sections 1 and 4: every
// coded symbol in exact reverse decode order, from the start states the spec gives, the 32 final states
// stored. With the record builder below the kind tests make their own records.

package main

import "math/rand"

// wnr4Event is one coded symbol in decode order: the lane, its table and the symbol.
type wnr4Event struct {
	lane int
	cdf  []uint16
	sym  uint32
}

// wnr4EncodeTile encodes events (in decode order) into a tile: 32 u32 states, then the 16-bit words. start
// holds the states each lane must end at when decoded (lanes 0 and 1: L + the hash halves, tail lanes
// L + tail, lanes without cells L).
func wnr4EncodeTile(events []wnr4Event, start [wnr4Lanes]uint32) []byte {
	x := start
	var out []uint16 // words put out, the last one first
	for i := len(events) - 1; i >= 0; i-- {
		e := events[i]
		lo, f := uint32(e.cdf[e.sym]), uint32(e.cdf[e.sym+1])-uint32(e.cdf[e.sym])
		s := x[e.lane]
		// the state after this symbol's decode must lie in [L, 2^32): renormalise first
		if uint64(s) >= uint64(wnr4L>>cdfBits)<<16*uint64(f) {
			out = append(out, uint16(s))
			s >>= 16
		}
		s = (s/f)<<cdfBits + s%f + lo
		x[e.lane] = s
	}
	tile := make([]byte, wnr4States+2*len(out))
	for l := 0; l < wnr4Lanes; l++ {
		le.PutUint32(tile[4*l:], x[l])
	}
	for i, w := range out {
		le.PutUint16(tile[wnr4States+2*(len(out)-1-i):], w)
	}
	return tile
}

// wnr4RefLaneHash is the lane hash computed in one call by the murmur3 of rans.go, which the decoder's
// block-by-block hash must agree with: the ident words (word 3 = the lane), then the blocks.
func wnr4RefLaneHash(ident [wnr4Ident]uint32, l int, blocks []uint32) uint32 {
	ident[3] = uint32(l)
	return murmur3(magicWNR4, append(ident[:], blocks...), nil)
}

// wnr4Starts gives the start states of section 4 from the 32 lane hash inputs and the tails (ignored for
// lanes that do not own cells, and for lanes 0 and 1).
func wnr4Starts(ident [wnr4Ident]uint32, blocks *[wnr4Lanes][]uint32, owns uint32, tails *[wnr4Lanes]uint16) [wnr4Lanes]uint32 {
	var H uint32
	for l := 0; l < wnr4Lanes; l++ {
		H ^= wnr4RefLaneHash(ident, l, blocks[l])
	}
	var st [wnr4Lanes]uint32
	st[0], st[1] = wnr4L+(H&0xFFFF), wnr4L+(H>>16)
	for l := 2; l < wnr4Lanes; l++ {
		st[l] = wnr4L
		if owns&(1<<l) != 0 {
			st[l] += uint32(tails[l])
		}
	}
	return st
}

// wnr4BuildRecord lays out a record: the 64-byte header (bytes 20..27 are filled in here), the tile offsets,
// the kind's area, the payload, the zero padding to 4096.
func wnr4BuildRecord(head [wnr4Head]byte, area []byte, tiles [][]byte) []byte {
	var P uint64
	for _, t := range tiles {
		P += uint64(len(t))
	}
	n := uint64(len(tiles))
	end := wnr4Head + 4*(n+1) + uint64(len(area)) + P
	rec := make([]byte, align(end))
	copy(rec, head[:])
	le.PutUint32(rec[20:], uint32(n))
	le.PutUint32(rec[24:], uint32(P))
	p := uint64(wnr4Head)
	var off uint32
	for _, t := range tiles {
		le.PutUint32(rec[p:], off)
		p += 4
		off += uint32(len(t))
	}
	le.PutUint32(rec[p:], off)
	p += 4
	p += uint64(copy(rec[p:], area))
	for _, t := range tiles {
		p += uint64(copy(rec[p:], t))
	}
	return rec
}

// wnr4RandCDF is a random valid cumulative table of n symbols (n + 1 entries); with skew, one symbol takes
// most of the range, so that some lanes renormalise often and others seldom.
func wnr4RandCDF(r *rand.Rand, n int, skew bool) []uint16 {
	f := make([]int, n)
	left := cdfTotal - n
	if skew {
		big := r.Intn(n)
		f[big] = left - r.Intn(64)
		left -= f[big]
	}
	for left > 0 {
		k := r.Intn(n)
		add := 1 + r.Intn(1+left/4)
		if add > left {
			add = left
		}
		f[k] += add
		left -= add
	}
	cdf := make([]uint16, n+1)
	for i := 0; i < n; i++ {
		cdf[i+1] = cdf[i] + uint16(f[i]+1)
	}
	return cdf
}

// wnr4Uniform is the uniform table of n symbols.
func wnr4Uniform(n int) []uint16 {
	cdf := make([]uint16, n+1)
	for i := 0; i <= n; i++ {
		cdf[i] = uint16(i * cdfTotal / n)
	}
	return cdf
}
