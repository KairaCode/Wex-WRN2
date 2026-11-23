// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Codec 8, WNR4 records (docs/WEX4-SPEC.md section 7.3, docs/WNR4-SPEC.md DRAFT 0.1): the parts every kind
// shares. The record header and the tile offsets (WNR4-SPEC section 3), the 32 rANS lanes of a tile over one
// shared stream of 16-bit words, the lane hashes, the tail in the state and the end states (sections 1 and
// 4), and the common header and tables of a WNM4 model blob (section 2). The kinds (wnr4_gg.go, wnr4_i4.go)
// give the symbol schedule, the tables and what each lane hashes. Decoder only, CPU, one thread. Written
// from the specification pages; what they left open is part F of SPEC-NOTES.md.
//
// Not called by Verify yet: what a failed codec-8 record makes of the whole file is the owner's decision.

package main

const (
	codecWNR4    = 8
	blobKindWNM4 = 3
	magicWNR4    = 0x34524E57 // "WNR4", also the seed of the lane hash
	magicWNM4    = 0x344D4E57 // "WNM4"
	wnr4L        = 1 << 16    // lower bound of a lane's state
	wnr4Lanes    = 32
	wnr4Head     = 64     // record header bytes, before the tile offsets
	wnr4States   = 4 * 32 // the 32 initial states at the start of every tile
	wnr4MaxG     = 4096   // G is 1..4096
	wnr4CDF16    = 17     // entries of a 16-symbol table
	wnr4CDF256   = 257    // entries of a 256-symbol table
	wnr4Ident    = 6      // ident words of a lane hash
	wnr4ModelHdr = 8 * 4  // WNM4 header bytes
)

// SeenWNR4 counts what the WNR4 decoders met, like Seen for the other codecs.
var SeenWNR4 struct {
	Records, Tiles, TailLanes, EmptyLanes uint64
	GGVerbatim, GGCoded                   uint64 // slices of GG records
	I4Escapes, I4Pairs                    uint64 // escaped scales, records in pairs packing
}

// wnr4Header holds the fields of the 64-byte header that every kind reads the same way (WNR4-SPEC section 3).
// The bytes a kind gives its own meaning (13, 16..19, 28..63) are left to the kind.
type wnr4Header struct {
	Model  uint32 // bytes 4..7
	Owner  uint32 // bytes 8..11
	Kind   byte   // byte 12
	Flags  byte   // byte 13
	G      uint32 // bytes 14..15
	Ntiles uint32 // bytes 20..23
	P      uint32 // bytes 24..27, payload bytes
}

// wnr4ParseHeader checks what every kind requires of the header: the size, the magic, a model id that is not 0
// and G in 1..4096. The caller compares Model, Owner and Kind with what it expects.
func wnr4ParseHeader(rec []byte) (wnr4Header, *Error) {
	var h wnr4Header
	if len(rec) < wnr4Head {
		return h, bad("WNR4: record of %d bytes", len(rec))
	}
	if m := le.Uint32(rec[0:]); m != magicWNR4 {
		return h, bad("WNR4: magic 0x%08x", m)
	}
	h.Model = le.Uint32(rec[4:])
	h.Owner = le.Uint32(rec[8:])
	h.Kind = rec[12]
	h.Flags = rec[13]
	h.G = uint32(le.Uint16(rec[14:]))
	h.Ntiles = le.Uint32(rec[20:])
	h.P = le.Uint32(rec[24:])
	if h.Model == 0 {
		return h, bad("WNR4: model id 0")
	}
	if h.G < 1 || h.G > wnr4MaxG {
		return h, bad("WNR4: G = %d", h.G)
	}
	return h, nil
}

// wnr4Tiles is the tile count of a run of n cells (blocks, groups) at G per lane: ceil(n / (32 G)).
func wnr4Tiles(n uint64, G uint32) uint64 { return ceilDiv(n, uint64(wnr4Lanes)*uint64(G)) }

// wnr4Layout is a record cut into its parts once the header and the geometry agree.
type wnr4Layout struct {
	off     []uint32 // ntiles + 1 tile offsets into the payload
	area    []byte   // the kind's area between the offsets and the payload (GG: verbatim, I4: bases)
	payload []byte
}

// wnr4Cut checks the layout of section 3 for a kind whose geometry gives ntiles tiles and an area of
// areaBytes between the offset table and the payload: the tile count, total = align4096(64 + 4 (ntiles + 1)
// + areaBytes + P) equal to the record's size, the zero padding after the payload, off[0] = 0, every tile at
// least 128 bytes and even, off[ntiles] = P. Every size is checked against the record before anything is
// allocated, so a hostile tile count costs nothing.
func wnr4Cut(rec []byte, h wnr4Header, ntiles, areaBytes uint64) (*wnr4Layout, *Error) {
	if uint64(h.Ntiles) != ntiles {
		return nil, bad("WNR4: %d tiles in the record, the geometry gives %d", h.Ntiles, ntiles)
	}
	offAt := uint64(wnr4Head)
	areaAt := offAt + 4*(ntiles+1)
	payAt := areaAt + areaBytes
	end := payAt + uint64(h.P)
	if align(end) != uint64(len(rec)) {
		return nil, bad("WNR4: record of %d bytes, the header and the geometry give %d", len(rec), align(end))
	}
	if !allZero(rec[end:]) {
		return nil, bad("WNR4: bytes after the payload are not zero")
	}
	lay := &wnr4Layout{
		off:     make([]uint32, ntiles+1),
		area:    rec[areaAt:payAt:payAt],
		payload: rec[payAt:end:end],
	}
	for t := range lay.off {
		lay.off[t] = le.Uint32(rec[offAt+4*uint64(t):])
	}
	if lay.off[0] != 0 {
		return nil, bad("WNR4: off[0] = %d", lay.off[0])
	}
	for t := uint64(0); t < ntiles; t++ {
		a, b := lay.off[t], lay.off[t+1]
		if b < a || b-a < wnr4States || (b-a)&1 != 0 {
			return nil, bad("WNR4: tile %d from %d to %d", t, a, b)
		}
	}
	if lay.off[ntiles] != h.P {
		return nil, bad("WNR4: off[%d] = %d, P = %d", ntiles, lay.off[ntiles], h.P)
	}
	return lay, nil
}

// tile is the bytes of tile t: its 32 initial states, then its words. The slice has no capacity past the
// tile, so nothing past it can be reached.
func (lay *wnr4Layout) tile(t uint64) []byte {
	a, b := lay.off[t], lay.off[t+1]
	return lay.payload[a:b:b]
}

// wnr4Lanes32 decodes one tile: 32 rANS lanes over one shared stream of 16-bit little-endian words, and one
// MurmurHash3_x86_32 per lane (seed "WNR4") fed block by block.
//
// The kinds call sym for every symbol position in the order the spec gives, lanes in ascending order, so
// a lane that renormalises takes the next word of the tile at once: within one position that is the
// ascending lane order of WNR4-SPEC section 4, "Words". A lane that asks for a word the tile does not have
// sets short and keeps its state (the work per symbol stays fixed); finish refuses the tile.
type wnr4Lanes32 struct {
	words  []byte
	pos    int
	short  bool
	x      [wnr4Lanes]uint32
	h      [wnr4Lanes]uint32 // running hash of each lane, before finalisation
	blocks [wnr4Lanes]uint32 // 4-byte blocks hashed so far, the ident words included
}

// wnr4Open reads the 32 initial states of a tile (u32 little endian, lane order) and starts the lane hashes
// with the ident words, word 3 being the lane. Every initial state must be at least L. The tile is at least
// 128 bytes and even (wnr4Cut).
func wnr4Open(tile []byte, ident [wnr4Ident]uint32) (*wnr4Lanes32, *Error) {
	if len(tile) < wnr4States || len(tile)&1 != 0 {
		return nil, bad("WNR4: tile of %d bytes", len(tile))
	}
	d := &wnr4Lanes32{words: tile[wnr4States:]}
	for l := 0; l < wnr4Lanes; l++ {
		d.x[l] = le.Uint32(tile[4*l:])
		if d.x[l] < wnr4L {
			return nil, bad("WNR4: initial state of lane %d is 0x%x, below L", l, d.x[l])
		}
		d.h[l] = magicWNR4
		ident[3] = uint32(l)
		for _, w := range ident {
			d.word(l, w)
		}
	}
	return d, nil
}

// word hashes one 4-byte block into lane l.
func (d *wnr4Lanes32) word(l int, w uint32) {
	d.h[l] = mmBlock(d.h[l], w)
	d.blocks[l]++
}

// bytes hashes b, a multiple of 4 bytes long, into lane l as little-endian words.
func (d *wnr4Lanes32) bytes(l int, b []byte) {
	for i := 0; i+4 <= len(b); i += 4 {
		d.word(l, le.Uint32(b[i:]))
	}
}

// sym decodes one symbol of lane l with the cumulative table cdf (17 or 257 entries, checked when the model
// was opened): s is the largest symbol with cdf[s] <= slot. Then at most one word.
func (d *wnr4Lanes32) sym(l int, cdf []uint16) uint32 {
	x := d.x[l]
	slot := x & (cdfTotal - 1)
	lo, hi := 0, len(cdf)-1 // cdf[lo] <= slot < cdf[hi]
	for hi-lo > 1 {
		mid := (lo + hi) >> 1
		if uint32(cdf[mid]) <= slot {
			lo = mid
		} else {
			hi = mid
		}
	}
	start, freq := uint32(cdf[lo]), uint32(cdf[lo+1])-uint32(cdf[lo])
	x = freq*(x>>cdfBits) + slot - start
	if x < wnr4L {
		if d.pos+2 <= len(d.words) {
			x = x<<16 | uint32(le.Uint16(d.words[d.pos:]))
			d.pos += 2
		} else {
			d.short = true
		}
	}
	d.x[l] = x
	return uint32(lo)
}

// tail is the 16 bits a lane l >= 2 that owns cells carries in its final state ("Tail in the state"), and
// whether the state is in [L, L + 2^16). Call it after the last symbol of the tile.
func (d *wnr4Lanes32) tail(l int) (uint16, bool) {
	x := d.x[l]
	return uint16(x - wnr4L), x >= wnr4L && x-wnr4L < 1<<16
}

// finish runs the end-state checks of WNR4-SPEC section 4 once every symbol was decoded and every block
// hashed (the tail lanes' last cells included): no word asked for that the tile did not have and exactly all
// words used, lanes 0 and 1 at L + the low and high halves of the tile hash (the XOR of the 32 finalised lane
// hashes), lanes 2..31 without cells at L, tail lanes in [L, L + 2^16). owns has bit l set when lane l owns at
// least one cell.
func (d *wnr4Lanes32) finish(owns uint32) *Error {
	if d.short {
		return bad("WNR4: a lane asked for a word past the end of the tile")
	}
	if d.pos != len(d.words) {
		return bad("WNR4: %d of the tile's %d words used", d.pos/2, len(d.words)/2)
	}
	var empty, tails uint64
	for l := 2; l < wnr4Lanes; l++ {
		if owns&(1<<l) == 0 {
			if d.x[l] != wnr4L {
				return bad("WNR4: lane %d owns no cell and ends at 0x%x", l, d.x[l])
			}
			empty++
		} else {
			if _, ok := d.tail(l); !ok {
				return bad("WNR4: lane %d ends at 0x%x, outside [L, L + 2^16)", l, d.x[l])
			}
			tails++
		}
	}
	var H uint32
	for l := 0; l < wnr4Lanes; l++ {
		H ^= wnr4Final(d.h[l], 4*d.blocks[l])
	}
	if d.x[0] != wnr4L+(H&0xFFFF) || d.x[1] != wnr4L+(H>>16) {
		return bad("WNR4: tile hash 0x%08x, lanes 0 and 1 end at 0x%x and 0x%x", H, d.x[0], d.x[1])
	}
	SeenWNR4.Tiles++
	SeenWNR4.EmptyLanes += empty
	SeenWNR4.TailLanes += tails
	return nil
}

// wnr4Final is the finalisation of MurmurHash3_x86_32 with the length in bytes.
func wnr4Final(h, n uint32) uint32 {
	h ^= n
	h ^= h >> 16
	h *= 0x85ebca6b
	h ^= h >> 13
	h *= 0xc2b2ae35
	h ^= h >> 16
	return h
}

// wnr4Owns is the lane mask of a tile of n cells: lane l owns cells when l < n.
func wnr4Owns(n uint64) uint32 {
	if n >= wnr4Lanes {
		return 0xFFFFFFFF
	}
	return uint32(1)<<n - 1
}

// wnr4ModelHeader checks the 8-word header of a WNM4 blob against want, and the blob's size.
func wnr4ModelHeader(b []byte, size int, want [8]uint32) *Error {
	if len(b) != size {
		return fail(EBlob, "WNM4 kind %d: %d bytes, expected %d", want[2], len(b), size)
	}
	for i, w := range want {
		if v := le.Uint32(b[4*i:]); v != w {
			return fail(EBlob, "WNM4 kind %d: header word %d is %d, expected %d", want[2], i, v, w)
		}
	}
	return nil
}

// wnr4Tables reads count cumulative tables of entries u16 each from b at *p and checks each one (cdf[0] = 0,
// the last entry 16384, strictly increasing). It returns them back to back.
func wnr4Tables(b []byte, p *int, count, entries int, what string) ([]uint16, *Error) {
	t := make([]uint16, count*entries)
	for i := range t {
		t[i] = le.Uint16(b[*p:])
		*p += 2
	}
	for i := 0; i < count; i++ {
		if !validCDF(t[i*entries : (i+1)*entries]) {
			return nil, fail(EBlob, "WNM4: %s table %d", what, i)
		}
	}
	return t, nil
}
