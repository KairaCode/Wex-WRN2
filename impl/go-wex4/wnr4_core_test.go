// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Tests of the WNR4 core (wnr4.go) on hand-made tiles and records: random lane schedules through the test
// encoder, every end-state check, reads past the tile, word counts, the record layout.

package main

import (
	"math/rand"
	"strings"
	"testing"
)

// a schedule: per symbol position, which lanes decode with which table; the lanes hash every symbol they
// decode as one block, and a tail lane its tail as one more block at the end
type coreSched struct {
	ident  [wnr4Ident]uint32
	pos    [][]wnr4Event
	owns   uint32
	tails  [wnr4Lanes]uint16
	blocks [wnr4Lanes][]uint32
}

func randSched(r *rand.Rand) *coreSched {
	s := &coreSched{ident: [wnr4Ident]uint32{r.Uint32(), r.Uint32(), uint32(r.Intn(100)), 0, r.Uint32(), r.Uint32()}}
	n := r.Intn(48) // cells of the tile: lanes l < n own cells
	s.owns = wnr4Owns(uint64(n))
	tables := [][]uint16{wnr4RandCDF(r, 16, false), wnr4RandCDF(r, 16, true), wnr4RandCDF(r, 256, false),
		wnr4RandCDF(r, 256, true), wnr4Uniform(16), wnr4Uniform(256)}
	npos := r.Intn(120)
	density := r.Float64()
	for p := 0; p < npos; p++ {
		cdf := tables[r.Intn(len(tables))]
		var evs []wnr4Event
		for l := 0; l < wnr4Lanes; l++ {
			if r.Float64() < density {
				sym := uint32(r.Intn(len(cdf) - 1))
				evs = append(evs, wnr4Event{l, cdf, sym})
				s.blocks[l] = append(s.blocks[l], sym)
			}
		}
		s.pos = append(s.pos, evs)
	}
	for l := 2; l < wnr4Lanes; l++ {
		if s.owns&(1<<l) != 0 {
			s.tails[l] = uint16(r.Uint32())
			s.blocks[l] = append(s.blocks[l], uint32(s.tails[l]))
		}
	}
	return s
}

func (s *coreSched) encode() []byte {
	var evs []wnr4Event
	for _, p := range s.pos {
		evs = append(evs, p...)
	}
	return wnr4EncodeTile(evs, wnr4Starts(s.ident, &s.blocks, s.owns, &s.tails))
}

// decode runs the decoder over the schedule; it reports the first symbol that differs as an error too.
func (s *coreSched) decode(tile []byte) *Error {
	d, e := wnr4Open(tile, s.ident)
	if e != nil {
		return e
	}
	for _, p := range s.pos {
		for _, ev := range p {
			y := d.sym(ev.lane, ev.cdf)
			d.word(ev.lane, y)
		}
	}
	for l := 2; l < wnr4Lanes; l++ {
		if s.owns&(1<<l) != 0 {
			t, _ := d.tail(l)
			d.word(l, uint32(t))
		}
	}
	return d.finish(s.owns)
}

func (s *coreSched) symbolsMatch(t *testing.T, tile []byte) {
	d, e := wnr4Open(tile, s.ident)
	if e != nil {
		t.Fatal(e)
	}
	for i, p := range s.pos {
		for _, ev := range p {
			if y := d.sym(ev.lane, ev.cdf); y != ev.sym {
				t.Fatalf("position %d lane %d: decoded %d, encoded %d", i, ev.lane, y, ev.sym)
			}
		}
	}
	for l := 2; l < wnr4Lanes; l++ {
		if s.owns&(1<<l) != 0 {
			if v, ok := d.tail(l); !ok || v != s.tails[l] {
				t.Fatalf("lane %d: tail 0x%x (%v), encoded 0x%x", l, v, ok, s.tails[l])
			}
		}
	}
}

func TestWNR4HashBlocks(t *testing.T) {
	// the block-by-block lane hash is murmur3 of rans.go over the same words
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 200; i++ {
		ident := [wnr4Ident]uint32{r.Uint32(), r.Uint32(), r.Uint32(), 0, r.Uint32(), r.Uint32()}
		blocks := make([]uint32, r.Intn(40))
		for k := range blocks {
			blocks[k] = r.Uint32()
		}
		tile := make([]byte, wnr4States)
		for l := 0; l < wnr4Lanes; l++ {
			le.PutUint32(tile[4*l:], wnr4L)
		}
		d, e := wnr4Open(tile, ident)
		if e != nil {
			t.Fatal(e)
		}
		l := r.Intn(wnr4Lanes)
		for _, b := range blocks {
			d.word(l, b)
		}
		if got, want := wnr4Final(d.h[l], 4*d.blocks[l]), wnr4RefLaneHash(ident, l, blocks); got != want {
			t.Fatalf("lane hash 0x%08x, murmur3 0x%08x", got, want)
		}
	}
}

func TestWNR4RoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	words := 0
	for i := 0; i < 400; i++ {
		s := randSched(r)
		tile := s.encode()
		words += (len(tile) - wnr4States) / 2
		s.symbolsMatch(t, tile)
		if e := s.decode(tile); e != nil {
			t.Fatalf("schedule %d: %v", i, e)
		}
	}
	if words == 0 {
		t.Fatal("no schedule used a word")
	}
}

func TestWNR4EndStates(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	expect := func(name string, e *Error, want string) {
		t.Helper()
		if e == nil || !strings.Contains(e.Msg, want) {
			t.Fatalf("%s: got %v, want an error with %q", name, e, want)
		}
	}
	for i := 0; i < 60; i++ {
		s := randSched(r)
		for s.owns&0xFFFFFFFC == 0 || s.owns == 0xFFFFFFFF { // need a tail lane and an empty lane
			s = randSched(r)
		}
		tile := s.encode()

		// an initial state below L
		b := append([]byte(nil), tile...)
		le.PutUint32(b[4*r.Intn(wnr4Lanes):], wnr4L-1-uint32(r.Intn(wnr4L)))
		expect("state below L", s.decode(b), "below L")

		// lanes 0 and 1 not at their hash: change one hashed block, keep the stream
		s2 := *s
		s2.ident[5] ^= 1 << uint(r.Intn(32))
		expect("hash", s2.decode(tile), "tile hash")

		// a lane without cells that does not end at L: encode it from L + 1
		empty := 2
		for s.owns&(1<<empty) != 0 {
			empty++
		}
		var evs []wnr4Event
		for _, p := range s.pos {
			evs = append(evs, p...)
		}
		st := wnr4Starts(s.ident, &s.blocks, s.owns, &s.tails)
		st[empty]++
		expect("empty lane", s.decode(wnr4EncodeTile(evs, st)), "owns no cell")

		// a tail lane ending at L + 2^16 or above: its tail cannot be in the state
		st = wnr4Starts(s.ident, &s.blocks, s.owns, &s.tails)
		st[2] = wnr4L + 1<<16 + uint32(r.Intn(1000))
		expect("tail lane", s.decode(wnr4EncodeTile(evs, st)), "outside [L, L + 2^16)")

		// one word too many
		b = append(append([]byte(nil), tile...), byte(r.Intn(256)), byte(r.Intn(256)))
		expect("extra word", s.decode(b), "words used")

		// one word too few: the decoder must stop at the tile's end, not read past it
		if len(tile) > wnr4States {
			b = tile[: len(tile)-2 : len(tile)-2]
			expect("missing word", s.decode(b), "past the end")
		}
	}
}

func TestWNR4BitFlips(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	flips, refused := 0, 0
	for i := 0; i < 40; i++ {
		s := randSched(r)
		tile := s.encode()
		for k := 0; k < 8*len(tile); k += 1 + r.Intn(7) {
			b := append([]byte(nil), tile...)
			b[k/8] ^= 1 << uint(k%8)
			flips++
			if s.decode(b) != nil {
				refused++
			}
		}
	}
	if refused != flips {
		t.Fatalf("%d of %d single-bit flips decoded without error", flips-refused, flips)
	}
	t.Logf("%d single-bit flips, all refused", flips)
}

func TestWNR4Layout(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	var head [wnr4Head]byte
	le.PutUint32(head[0:], magicWNR4)
	le.PutUint32(head[4:], 0x1234)
	le.PutUint16(head[14:], 32)
	var tiles [][]byte
	for len(tiles) < 3 {
		if tile := randSched(r).encode(); len(tile) > wnr4States+2 {
			tiles = append(tiles, tile)
		}
	}
	area := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	rec := wnr4BuildRecord(head, area, tiles)
	h, e := wnr4ParseHeader(rec)
	if e != nil {
		t.Fatal(e)
	}
	lay, e := wnr4Cut(rec, h, 3, 8)
	if e != nil {
		t.Fatal(e)
	}
	if string(lay.area) != string(area) || string(lay.tile(1)) != string(tiles[1]) {
		t.Fatal("area or tile 1 not where it was written")
	}
	if cap(lay.tile(1)) != len(tiles[1]) {
		t.Fatal("a tile slice reaches past the tile")
	}

	type mut struct {
		name string
		f    func(b []byte) []byte
		want string
	}
	off := func(i int) int { return wnr4Head + 4*i }
	muts := []mut{
		{"short record", func(b []byte) []byte { return b[:63] }, "record of 63 bytes"},
		{"magic", func(b []byte) []byte { b[0] ^= 1; return b }, "magic"},
		{"model 0", func(b []byte) []byte { le.PutUint32(b[4:], 0); return b }, "model id 0"},
		{"G 0", func(b []byte) []byte { le.PutUint16(b[14:], 0); return b }, "G = 0"},
		{"G 4097", func(b []byte) []byte { le.PutUint16(b[14:], 4097); return b }, "G = 4097"},
		{"ntiles", func(b []byte) []byte { le.PutUint32(b[20:], 4); return b }, "geometry gives 3"},
		{"ntiles huge", func(b []byte) []byte { le.PutUint32(b[20:], 0xFFFFFFFF); return b }, "tiles in the record"},
		{"P", func(b []byte) []byte { le.PutUint32(b[24:], le.Uint32(b[24:])+4096); return b }, "bytes, the header"},
		{"page more", func(b []byte) []byte { return append(b, make([]byte, 4096)...) }, "bytes, the header"},
		{"padding", func(b []byte) []byte { b[len(b)-1] = 1; return b }, "not zero"},
		{"off[0]", func(b []byte) []byte { le.PutUint32(b[off(0):], 2); return b }, "off[0]"},
		{"odd tile", func(b []byte) []byte { le.PutUint32(b[off(1):], le.Uint32(b[off(1):])+1); return b }, "tile 0 from"},
		{"short tile", func(b []byte) []byte { le.PutUint32(b[off(2):], le.Uint32(b[off(1):])+126); return b }, "tile 1 from"},
		{"backwards", func(b []byte) []byte { le.PutUint32(b[off(2):], 0); return b }, "tile 1 from"},
		{"off[n]", func(b []byte) []byte { le.PutUint32(b[off(3):], le.Uint32(b[off(3):])-2); return b }, "off[3]"},
	}
	for _, m := range muts {
		b := m.f(append([]byte(nil), rec...))
		h, e := wnr4ParseHeader(b)
		if e == nil {
			_, e = wnr4Cut(b, h, 3, 8)
		}
		if e == nil || !strings.Contains(e.Msg, m.want) {
			t.Errorf("%s: got %v, want %q", m.name, e, m.want)
		}
	}

	// a tile count of 2^32 - 1 that the geometry also gives is refused by size, before any allocation
	b := append([]byte(nil), rec...)
	le.PutUint32(b[20:], 0xFFFFFFFF)
	h, _ = wnr4ParseHeader(b)
	if _, e := wnr4Cut(b, h, 0xFFFFFFFF, 0); e == nil || !strings.Contains(e.Msg, "bytes, the header") {
		t.Errorf("ntiles 2^32 - 1: %v", e)
	}

	// no tile at all: one offset, P = 0
	rec = wnr4BuildRecord(head, nil, nil)
	h, _ = wnr4ParseHeader(rec)
	if lay, e = wnr4Cut(rec, h, 0, 0); e != nil || len(lay.off) != 1 {
		t.Fatalf("ntiles 0: %v", e)
	}
}

func TestWNR4ModelTables(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	b := make([]byte, 2*(17+257))
	p := 0
	for _, c := range [][]uint16{wnr4RandCDF(r, 16, true), wnr4RandCDF(r, 256, false)} {
		for _, v := range c {
			le.PutUint16(b[p:], v)
			p += 2
		}
	}
	p = 0
	if _, e := wnr4Tables(b, &p, 1, 17, "x"); e != nil {
		t.Fatal(e)
	}
	if _, e := wnr4Tables(b, &p, 1, 257, "x"); e != nil || p != len(b) {
		t.Fatal(e)
	}
	for _, at := range []int{0, 2 * 16, 2 * 5, 2*17 + 2*256} {
		c := append([]byte(nil), b...)
		switch at {
		case 2 * 5: // equal neighbours
			copy(c[at:], c[at-2:at])
		default:
			c[at] ^= 1
		}
		p = 0
		_, e1 := wnr4Tables(c, &p, 1, 17, "x")
		_, e2 := wnr4Tables(c, &p, 1, 257, "x")
		if e1 == nil && e2 == nil {
			t.Fatalf("table damaged at byte %d accepted", at)
		}
	}
}
