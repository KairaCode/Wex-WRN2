// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Tests of WNR4 kind 2 (wnr4_gg.go) on hand-made models and records: a test encoder written from
// WNR4-SPEC 5.3 to 5.5, round trips, each record check, single-bit flips.

package main

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"
)

// randGGModel is a WNM4 kind-2 blob with a random class map and random tables.
func randGGModel(r *rand.Rand) []byte {
	b := make([]byte, ggModelBytes)
	for i, w := range []uint32{magicWNM4, 1, ggKind, cdfBits, ggClasses, 2, 0, 0} {
		le.PutUint32(b[4*i:], w)
	}
	p := wnr4ModelHdr
	put := func(c []uint16) {
		for _, v := range c {
			le.PutUint16(b[p:], v)
			p += 2
		}
	}
	for u := 0; u < 2; u++ {
		for v := 0; v < 256; v++ {
			b[p+v] = byte(r.Intn(ggClasses))
		}
		p += 256
		put(wnr4RandCDF(r, 256, r.Intn(2) == 0))
		for c := 0; c < ggClasses; c++ {
			put(wnr4RandCDF(r, 256, r.Intn(2) == 0))
		}
		for c := 0; c < ggClasses; c++ {
			put(wnr4RandCDF(r, 16, r.Intn(2) == 0))
		}
	}
	return b
}

// encodeGG is the test encoder: the record of data (the slices back to back) under geometry sl.
func encodeGG(m *GGModel, sl []GGSlice, owner uint32, G uint32, data []byte) []byte {
	var head [wnr4Head]byte
	le.PutUint32(head[0:], magicWNR4)
	le.PutUint32(head[4:], m.ID)
	le.PutUint32(head[8:], owner)
	head[12] = ggKind
	le.PutUint16(head[14:], uint16(G))
	var area []byte
	var tiles [][]byte
	var at uint64
	per := uint64(wnr4Lanes) * uint64(G)
	for _, s := range sl {
		u := ggSubtype(s.Type)
		if u < 0 {
			area = append(area, data[at:at+s.Bytes]...)
		} else {
			blocks := s.Bytes / ggBlock
			for c0 := uint64(0); c0 < blocks; c0 += per {
				n := min(per, blocks-c0)
				ident := [wnr4Ident]uint32{m.ID, owner, uint32(len(tiles)), 0, ggKind | G<<16, s.Type}
				tiles = append(tiles, encodeGGTile(m, u, ident, data[at+c0*ggBlock:at+(c0+n)*ggBlock]))
			}
		}
		at += s.Bytes
	}
	for len(area)%4 != 0 {
		area = append(area, 0)
	}
	return wnr4BuildRecord(head, area, tiles)
}

func encodeGGTile(m *GGModel, u int, ident [wnr4Ident]uint32, blk []byte) []byte {
	n := uint64(len(blk) / ggBlock)
	steps := ceilDiv(n, wnr4Lanes)
	var evs []wnr4Event
	var hb [wnr4Lanes][]uint32
	cls := func(i uint64) int { return int(m.cmap[u][blk[i*ggBlock+1]]) }
	for k := uint64(0); k < steps; k++ {
		for l := uint64(0); l < wnr4Lanes && 32*k+l < n; l++ {
			y := uint32(blk[(32*k+l)*ggBlock+1])
			evs = append(evs, wnr4Event{int(l), m.high[u], y})
			hb[l] = append(hb[l], y)
		}
	}
	for k := uint64(0); k < steps; k++ {
		for l := uint64(0); l < wnr4Lanes && 32*k+l < n; l++ {
			i := 32*k + l
			c := cls(i)
			y := uint32(blk[i*ggBlock])
			evs = append(evs, wnr4Event{int(l), m.low[u][c*wnr4CDF256 : (c+1)*wnr4CDF256], y})
			hb[l] = append(hb[l], y)
		}
	}
	var tails [wnr4Lanes]uint16
	for j := uint64(0); j < steps; j++ {
		for k := 0; k < 32; k++ {
			for l := uint64(0); l < wnr4Lanes && 32*j+l < n; l++ {
				i := 32*j + l
				if k >= 28 && l >= 2 && j == (n-l-1)/32 {
					continue
				}
				c := cls(i)
				y := uint32(blk[i*ggBlock+2+uint64(k/2)]>>(4*(k&1))) & 15
				evs = append(evs, wnr4Event{int(l), m.nib[u][c*wnr4CDF16 : (c+1)*wnr4CDF16], y})
			}
		}
		for l := uint64(0); l < wnr4Lanes && 32*j+l < n; l++ {
			b := blk[(32*j+l)*ggBlock:]
			for w := 0; w < 4; w++ {
				hb[l] = append(hb[l], le.Uint32(b[2+4*w:]))
			}
			if l >= 2 && j == (n-l-1)/32 {
				tails[l] = le.Uint16(b[16:])
			}
		}
	}
	return wnr4EncodeTile(evs, wnr4Starts(ident, &hb, wnr4Owns(n), &tails))
}

// ggData is random slice content: coded blocks with a few high bytes (so that classes repeat), verbatim
// bytes at random.
func ggData(r *rand.Rand, sl []GGSlice) []byte {
	var out []byte
	for _, s := range sl {
		b := make([]byte, s.Bytes)
		r.Read(b)
		if ggSubtype(s.Type) >= 0 {
			for i := uint64(0); i < s.Bytes; i += ggBlock {
				b[i+1] = byte(0x30 + r.Intn(4))
			}
		}
		out = append(out, b...)
	}
	return out
}

type ggCase struct {
	name string
	G    uint32
	sl   []GGSlice
}

func ggCases() []ggCase {
	q, nl := uint32(ggTypeQ2_0), uint32(ggTypeIQ4NL)
	return []ggCase{
		{"one Q2_0 slice, one block", 1, []GGSlice{{q, 18}}},
		{"one IQ4_NL slice, G 1, 5 tiles", 1, []GGSlice{{nl, 18 * 150}}},
		{"G 2, partial last tile", 2, []GGSlice{{q, 18 * 100}}},
		{"G 32, 4 slices", 32, []GGSlice{{16, 1001}, {nl, 18 * 40}, {22, 77}, {q, 18 * 1100}}},
		{"G 4096, n < 32", 4096, []GGSlice{{q, 18 * 7}, {nl, 18 * 33}}},
		{"nothing coded", 32, []GGSlice{{16, 5}, {22, 3}}},
		{"verbatim a multiple of 4", 3, []GGSlice{{0, 64}, {q, 18 * 70}, {1, 4}}},
		{"exactly 32 G blocks", 2, []GGSlice{{nl, 18 * 64}, {q, 18 * 65}}},
	}
}

func TestWNR4GGRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	blob := randGGModel(r)
	m, e := OpenGGModel(blob)
	if e != nil {
		t.Fatal(e)
	}
	before := SeenWNR4
	for _, c := range ggCases() {
		data := ggData(r, c.sl)
		owner := r.Uint32()
		rec := encodeGG(m, c.sl, owner, c.G, data)
		out := make([]byte, len(data))
		if e := decodeGG(rec, m, c.sl, owner, out); e != nil {
			t.Fatalf("%s: %v", c.name, e)
		}
		if !bytes.Equal(out, data) {
			t.Fatalf("%s: decoded bytes differ", c.name)
		}
	}
	if SeenWNR4.TailLanes == before.TailLanes || SeenWNR4.EmptyLanes == before.EmptyLanes {
		t.Fatal("no tail lane or no empty lane met")
	}
}

func TestWNR4GGModel(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	blob := randGGModel(r)
	if _, e := OpenGGModel(blob); e != nil {
		t.Fatal(e)
	}
	muts := map[string]func(b []byte) []byte{
		"short":           func(b []byte) []byte { return b[:len(b)-1] },
		"magic":           func(b []byte) []byte { b[0] ^= 1; return b },
		"version":         func(b []byte) []byte { b[4] = 2; return b },
		"kind":            func(b []byte) []byte { b[8] = 3; return b },
		"precision":       func(b []byte) []byte { b[12] = 15; return b },
		"classes":         func(b []byte) []byte { b[16] = 31; return b },
		"subtypes":        func(b []byte) []byte { b[20] = 3; return b },
		"word 6":          func(b []byte) []byte { b[24] = 1; return b },
		"word 7":          func(b []byte) []byte { b[31] = 1; return b },
		"class 32":        func(b []byte) []byte { b[wnr4ModelHdr+ggSubBytes+7] = 32; return b },
		"high table":      func(b []byte) []byte { b[wnr4ModelHdr+256] = 1; return b },
		"last nib table":  func(b []byte) []byte { b[len(b)-2] ^= 1; return b },
		"low table order": func(b []byte) []byte { p := wnr4ModelHdr + 256 + 514 + 10; copy(b[p:p+2], b[p-2:p]); return b },
	}
	for name, f := range muts {
		if _, e := OpenGGModel(f(append([]byte(nil), blob...))); e == nil || e.Class != EBlob {
			t.Errorf("%s: got %v", name, e)
		}
	}
}

func TestWNR4GGGeometry(t *testing.T) {
	q := uint32(ggTypeQ2_0)
	bad := map[string][]GGSlice{
		"no slice":          nil,
		"5 slices":          {{q, 18}, {q, 18}, {q, 18}, {q, 18}, {q, 18}},
		"empty slice":       {{16, 0}},
		"coded not 18 x n":  {{q, 19}},
		"2^27 blocks":       {{ggTypeIQ4NL, 18 << 27}},
		"verbatim 2^31":     {{16, 1 << 31}},
		"total 2^32":        {{16, 1<<31 - 1}, {q, 18 * (1<<27 - 1)}, {q, 18 * (1<<27 - 1)}},
		"total 2^32, coded": {{q, 18 * (1<<27 - 1)}, {q, 18 * (1<<27 - 1)}},
	}
	for name, sl := range bad {
		if _, _, e := ggGeometry(sl); e == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if total, v, e := ggGeometry([]GGSlice{{16, 1<<31 - 1}, {q, 18 * 1000}}); e != nil || v != 1<<31-1 || total != 1<<31-1+18*1000 {
		t.Errorf("largest verbatim slice: %v", e)
	}
}

func TestWNR4GGRecordChecks(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	m, _ := OpenGGModel(randGGModel(r))
	c := ggCases()[3] // 4 slices, 2 verbatim of 1001 and 77 bytes: V = 1080, 2 bytes of padding
	data := ggData(r, c.sl)
	owner := uint32(0x00070005)
	rec := encodeGG(m, c.sl, owner, c.G, data)
	out := make([]byte, len(data))
	if e := decodeGG(rec, m, c.sl, owner, out); e != nil {
		t.Fatal(e)
	}
	ntiles := int(le.Uint32(rec[20:]))
	areaAt := wnr4Head + 4*(ntiles+1)
	muts := []struct {
		name string
		f    func(b []byte) []byte
		want string
	}{
		{"magic", func(b []byte) []byte { b[3] = 0; return b }, "magic"},
		{"model id", func(b []byte) []byte { b[4] ^= 1; return b }, "model id"},
		{"model id 0", func(b []byte) []byte { le.PutUint32(b[4:], 0); return b }, "model id 0"},
		{"owner", func(b []byte) []byte { b[8] ^= 1; return b }, "owner"},
		{"kind", func(b []byte) []byte { b[12] = 3; return b }, "kind"},
		{"flags", func(b []byte) []byte { b[13] = 8; return b }, "flags"},
		{"byte 16", func(b []byte) []byte { b[16] = 1; return b }, "16..19"},
		{"byte 19", func(b []byte) []byte { b[19] = 1; return b }, "16..19"},
		{"byte 28", func(b []byte) []byte { b[28] = 1; return b }, "28..63"},
		{"byte 63", func(b []byte) []byte { b[63] = 1; return b }, "28..63"},
		{"G 0", func(b []byte) []byte { le.PutUint16(b[14:], 0); return b }, "G = 0"},
		{"G other", func(b []byte) []byte { le.PutUint16(b[14:], 1); return b }, "tiles in the record"},
		{"ntiles", func(b []byte) []byte { b[20]++; return b }, "tiles in the record"},
		{"P", func(b []byte) []byte { le.PutUint32(b[24:], le.Uint32(b[24:])+2); return b }, "off["},
		{"total", func(b []byte) []byte { return append(b, make([]byte, 4096)...) }, "record of"},
		{"total short", func(b []byte) []byte { return b[:len(b)-4096] }, "record of"},
		{"verbatim padding", func(b []byte) []byte { b[areaAt+1079] = 1; return b }, "padding of the verbatim"},
		{"payload padding", func(b []byte) []byte { b[len(b)-1] = 1; return b }, "after the payload"},
		{"off[0]", func(b []byte) []byte { le.PutUint32(b[64:], 2); return b }, "off[0]"},
		{"tile step odd", func(b []byte) []byte { b[68]++; return b }, "tile 0 from"},
	}
	for _, mu := range muts {
		b := mu.f(append([]byte(nil), rec...))
		e := decodeGG(b, m, c.sl, owner, out)
		if e == nil || !strings.Contains(e.Msg, mu.want) {
			t.Errorf("%s: got %v, want %q", mu.name, e, mu.want)
		}
	}
	// the geometry must be the encoder's
	other := append([]GGSlice(nil), c.sl...)
	other[1].Type = ggTypeQ2_0 // same size, other subtype: the hash ident's type word differs
	if decodeGG(rec, m, other, owner, out) == nil {
		t.Error("another subtype for slice 1 accepted")
	}
	other = append([]GGSlice(nil), c.sl...)
	other[0].Type = 22 // another verbatim type: not in the record, decodes the same
	if e := decodeGG(rec, m, other, owner, out); e != nil || !bytes.Equal(out, data) {
		t.Errorf("another verbatim type: %v", e)
	}
	if decodeGG(rec, m, c.sl[:3], owner, out[:len(out)-int(c.sl[3].Bytes)]) == nil {
		t.Error("a geometry without slice 3 accepted")
	}
	if decodeGG(rec, m, c.sl, owner, out[:len(out)-1]) == nil {
		t.Error("an output of the wrong size accepted")
	}
	m2, _ := OpenGGModel(randGGModel(r))
	if decodeGG(rec, m2, c.sl, owner, out) == nil {
		t.Error("another model accepted")
	}
}

// Every single-bit flip of a record is refused, except in the bytes of a verbatim slice: those decode, to
// the flipped bytes, and only the container's SHA-256 can catch them (WNR4-SPEC 5.5).
func TestWNR4GGBitFlips(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	m, _ := OpenGGModel(randGGModel(r))
	sl := []GGSlice{{16, 21}, {ggTypeQ2_0, 18 * 70}, {ggTypeIQ4NL, 18 * 3}}
	data := ggData(r, sl)
	rec := encodeGG(m, sl, 9, 1, data)
	ntiles := int(le.Uint32(rec[20:]))
	vAt := wnr4Head + 4*(ntiles+1)
	end := vAt + 24 + int(le.Uint32(rec[24:]))
	out := make([]byte, len(data))
	refused, verbatim := 0, 0
	for k := 0; k < 8*end; k++ {
		b := append([]byte(nil), rec...)
		b[k/8] ^= 1 << uint(k%8)
		e := decodeGG(b, m, sl, 9, out)
		if p := k / 8; p >= vAt && p < vAt+21 {
			want := append([]byte(nil), data...)
			want[p-vAt] ^= 1 << uint(k%8)
			if e != nil || !bytes.Equal(out, want) {
				t.Fatalf("flip in verbatim byte %d: %v", p-vAt, e)
			}
			verbatim++
			continue
		}
		if e == nil {
			t.Fatalf("flip of bit %d (byte %d) decoded without error", k%8, k/8)
		}
		refused++
	}
	t.Logf("%d flips refused, %d in verbatim bytes decoded to the flipped bytes", refused, verbatim)
}
