// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// The rANS stream decoder and the stream hash shared by WNR3 and WBP1 records
// (docs/WNR3-SPEC.md sections 1 and 4.5, docs/WNR2-REFERENCE.md section 6).

package main

const (
	ransL     = 1 << 23 // lower bound of the state
	cdfBits   = 14
	cdfTotal  = 1 << cdfBits
	hashSeed  = 0x33524E57
	hashMask  = 0x007FFFFF
	minStream = 4 // a stream is at least its 4-byte initial state
)

// Seen counts what the decoders met: the tests print it, so that a page can say what the vectors exercise.
var Seen struct {
	WBP1Rans, WBP1Raw, WBP1Rotated, WBP1Ragged uint64 // rANS streams, raw chunks, records with transform 1, records whose size is not a multiple of P
	WNR3Scale, WNR3Weight                      uint64 // streams
	WNR3CeilMatrices, WNR3CeilBlocks           uint64 // matrices in ceil form, blocks the ceil inverse changed
}

func rotl32(x uint32, r uint) uint32 { return x<<r | x>>(32-r) }

func mmBlock(h, k uint32) uint32 {
	k *= 0xcc9e2d51
	k = rotl32(k, 15)
	k *= 0x1b873593
	h ^= k
	h = rotl32(h, 13)
	return h*5 + 0xe6546b64
}

// murmur3 is MurmurHash3_x86_32 over the little-endian words of ident followed by data.
func murmur3(seed uint32, ident []uint32, data []byte) uint32 {
	h := seed
	for _, k := range ident {
		h = mmBlock(h, k)
	}
	n := len(data)
	i := 0
	for ; i+4 <= n; i += 4 {
		h = mmBlock(h, le.Uint32(data[i:]))
	}
	if tail := n - i; tail > 0 {
		var k uint32
		for j := tail - 1; j >= 0; j-- {
			k = k<<8 | uint32(data[i+j])
		}
		k *= 0xcc9e2d51
		k = rotl32(k, 15)
		k *= 0x1b873593
		h ^= k
	}
	h ^= uint32(4*len(ident) + n)
	h ^= h >> 16
	h *= 0x85ebca6b
	h ^= h >> 13
	h *= 0xc2b2ae35
	h ^= h >> 16
	return h
}

// h32 is the stream hash: the 16-byte identity, then the stream's decoded bytes.
func h32(ident [4]uint32, data []byte) uint32 { return murmur3(hashSeed, ident[:], data) }

// validCDF: n+1 cumulative frequencies, the first 0, the last 2^14, strictly increasing.
func validCDF(cdf []uint16) bool {
	if cdf[0] != 0 || cdf[len(cdf)-1] != cdfTotal {
		return false
	}
	for i := 1; i < len(cdf); i++ {
		if cdf[i] <= cdf[i-1] {
			return false
		}
	}
	return true
}

// slotTable maps each of the 2^14 slots of a 256-symbol CDF to its symbol.
func slotTable(cdf []uint16) []byte {
	t := make([]byte, cdfTotal)
	for s := 0; s < 256; s++ {
		for k := cdf[s]; k < cdf[s+1]; k++ {
			t[k] = byte(s)
		}
	}
	return t
}

// rans reads one stream: a 4-byte big-endian initial state in [L, 256 L), then renormalisation bytes.
// It never reads outside b. A stream that asks for a byte it does not have sets short and goes on with
// the state it has, so the work per symbol stays fixed; the caller refuses the stream at its end.
type rans struct {
	b     []byte
	pos   int
	x     uint32
	short bool
}

func ransOpen(b []byte) (rans, bool) {
	if len(b) < minStream {
		return rans{}, false
	}
	x := uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
	return rans{b: b, pos: 4, x: x}, x >= ransL && x < 256*ransL
}

func (r *rans) slot() uint32 { return r.x & (cdfTotal - 1) }

// advance consumes the symbol whose CDF interval is [start, start + freq).
func (r *rans) advance(start, freq uint32) {
	r.x = freq*(r.x>>cdfBits) + (r.x & (cdfTotal - 1)) - start
	for r.x < ransL {
		if r.pos >= len(r.b) {
			r.short = true
			return
		}
		r.x = r.x<<8 | uint32(r.b[r.pos])
		r.pos++
	}
}

// done: the stream ended exactly at its length in the state L + h23 of what it decoded.
func (r *rans) done(ident [4]uint32, decoded []byte) bool {
	return !r.short && r.pos == len(r.b) && r.x == ransL+(h32(ident, decoded)&hashMask)
}
