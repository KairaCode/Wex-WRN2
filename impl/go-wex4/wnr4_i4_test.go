// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Tests of WNR4 kind 3 (wnr4_i4.go) on hand-made models and records: the lattice against exact rational
// arithmetic, a test encoder written from WNR4-SPEC 6.3 to 6.5, round trips, each record check, single-bit
// flips.

package main

import (
	"bytes"
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

// i4RefLat is lat(k, B) computed exactly with rationals and rounded once per step by big.Rat.Float32
// (nearest even, subnormals kept).
func i4RefLat(k uint32, B float32) float32 {
	var v big.Rat
	if k < 8 {
		v.SetFrac64(int64(k), 512)
	} else {
		v.SetFrac64(int64(8+k&7), 8)
		e := int(k>>3) - 7
		if e >= 0 {
			v.Mul(&v, new(big.Rat).SetInt64(1<<e))
		} else {
			v.Mul(&v, new(big.Rat).SetFrac64(1, 1<<-e))
		}
	}
	if B == 0 {
		return 0
	}
	var b big.Rat
	b.SetFloat64(float64(B))
	v.Mul(&v, &b)
	p, _ := v.Float32()
	if math.IsInf(float64(p), 0) {
		return p
	}
	v.SetFloat64(float64(p))
	v.Quo(&v, big.NewRat(7, 1))
	q, _ := v.Float32()
	return q
}

func TestWNR4I4Lattice(t *testing.T) {
	if e4m3(126) != 448 || e4m3(1) != 1.0/512 || e4m3(8) != 1.0/64 || e4m3(0) != 0 {
		t.Fatal("E4M3 values")
	}
	if i4Lat(126, 1) != 64 {
		t.Fatalf("lat(126, 1) = %v", i4Lat(126, 1))
	}
	bases := []uint32{0x3F800000, 0x00800000, 0x00800001, 0x0BADBEEF, 0x2F000000, 0x3E4CCCCD, 0x7F7FFFFF, 0x7E000000, 0x01234567}
	r := rand.New(rand.NewSource(20))
	for len(bases) < 400 {
		if b := r.Uint32() & 0x7FFFFFFF; i4BaseOK(b) && b != 0 {
			bases = append(bases, b)
		}
	}
	subnormal, inf := 0, 0
	for _, bb := range bases {
		B := math.Float32frombits(bb)
		for k := uint32(0); k <= 126; k++ {
			got, want := i4Lat(k, B), i4RefLat(k, B)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("lat(%d, 0x%08x) = 0x%08x, exact rounding gives 0x%08x", k, bb, math.Float32bits(got), math.Float32bits(want))
			}
			if g := math.Float32bits(got); g != 0 && g < 0x00800000 {
				subnormal++
			}
			if math.IsInf(float64(got), 0) {
				inf++
			}
		}
	}
	if subnormal == 0 || inf == 0 {
		t.Fatalf("%d subnormal and %d infinite results: the table does not reach them", subnormal, inf)
	}
	t.Logf("%d bases x 127 codes, %d subnormal results, %d infinite", len(bases), subnormal, inf)
}

// randI4Model is a WNM4 kind-3 blob of random tables.
func randI4Model(r *rand.Rand) []byte {
	b := make([]byte, i4ModelBytes)
	for i, w := range []uint32{magicWNM4, 1, i4Kind, cdfBits, i4NibCtx, i4ScaleCtx, 0, 0} {
		le.PutUint32(b[4*i:], w)
	}
	p := wnr4ModelHdr
	put := func(c []uint16) {
		for _, v := range c {
			le.PutUint16(b[p:], v)
			p += 2
		}
	}
	nibs := []([]uint16){wnr4RandCDF(r, 16, false), wnr4RandCDF(r, 16, true), wnr4Uniform(16)}
	for i := 0; i < 3*i4NibCtx; i++ {
		if r.Intn(8) == 0 {
			put(wnr4RandCDF(r, 16, r.Intn(2) == 0))
		} else {
			put(nibs[r.Intn(len(nibs))])
		}
	}
	for i := 0; i < 3*i4ScaleCtx; i++ {
		put(wnr4RandCDF(r, 256, r.Intn(2) == 0))
	}
	return b
}

// i4Nib reads element e of a group's code bytes in a packing.
func i4Nib(c []byte, e int, pairs bool) uint32 {
	if pairs {
		return uint32(c[e/2]>>(4*(e%2))) & 15
	}
	return uint32(c[e%32]>>(4*(e/32))) & 15
}

// i4BadIndex, when >= 0, makes the encoder write for group 0 of tile 0 the index symbol of lattice index
// i4BadIndex (above 127: invalid).
var i4BadIndex = -1

// encodeI4 is the test encoder: the record of data (codes, then scales, as 6.1 lays them out).
func encodeI4(m *I4Model, g I4Geo, owner, G uint32, anchors [3]uint16, bases []uint32, data []byte) []byte {
	L, e := i4Geometry(g)
	if e != nil {
		panic(e)
	}
	var head [wnr4Head]byte
	le.PutUint32(head[0:], magicWNR4)
	le.PutUint32(head[4:], m.ID)
	le.PutUint32(head[8:], owner)
	head[12] = i4Kind
	P := uint32(0)
	if g.Pairs {
		P = 1
	}
	head[13] = byte(P)
	le.PutUint16(head[14:], uint16(G))
	for i, a := range anchors {
		le.PutUint16(head[28+2*i:], a)
	}
	area := make([]byte, 4*len(bases))
	for i, b := range bases {
		le.PutUint32(area[4*i:], b)
	}
	var tiles [][]byte
	per := uint64(wnr4Lanes) * uint64(G)
	for mm := range g.Rows {
		for c0 := uint64(0); c0 < L.cells[mm]; c0 += per {
			n := min(per, L.cells[mm]-c0)
			ident := [wnr4Ident]uint32{m.ID, owner, uint32(len(tiles)), 0, i4Kind | P<<8 | G<<16, uint32(mm) | uint32(anchors[mm])<<16}
			tiles = append(tiles, encodeI4Tile(m, L, g, mm, c0, n, uint32(anchors[mm]), bases, data, ident, len(tiles) == 0))
		}
	}
	return wnr4BuildRecord(head, area, tiles)
}

func encodeI4Tile(m *I4Model, L *i4Layout, g I4Geo, mm int, c0, n uint64, anchor uint32, bases []uint32, data []byte, ident [wnr4Ident]uint32, first bool) []byte {
	gc := L.gc[mm]
	scale := func(j uint64) uint32 { return le.Uint32(data[L.scaleAt[mm]+4*(c0+j):]) }
	code := func(j uint64) []byte { a := L.codeAt[mm] + 32*(c0+j); return data[a : a+32] }
	base := func(j uint64) float32 {
		i := c0 + j
		return math.Float32frombits(bases[L.bbase[mm]+(i/gc/128)*L.nbc[mm]+(i%gc)/2])
	}
	idx := make([]uint32, n)
	for j := range idx {
		idx[j] = i4Escape
		s := scale(uint64(j))
		for k := uint32(0); k <= 126; k++ {
			if math.Float32bits(i4Lat(k, base(uint64(j)))) == s {
				idx[j] = k
				break
			}
		}
	}
	dist := gc
	if gc < 32 {
		dist = ceilDiv(32, gc) * gc
	}
	steps := ceilDiv(n, wnr4Lanes)
	var evs []wnr4Event
	var hb [wnr4Lanes][]uint32
	for k := uint64(0); k < steps; k++ {
		for l := uint64(0); l < wnr4Lanes && 32*k+l < n; l++ {
			j := 32*k + l
			kref, row := uint32(120), 0
			if j >= dist {
				kref, row = min(idx[j-dist], 126), 32
			}
			y := (idx[j] - kref) & 255
			if first && j == 0 && i4BadIndex >= 0 {
				y = (uint32(i4BadIndex) - kref) & 255
			}
			evs = append(evs, wnr4Event{int(l), m.scaleRow(mm, row+int(kref/4)), y})
		}
		for b := 0; b < 4; b++ {
			for l := uint64(0); l < wnr4Lanes && 32*k+l < n; l++ {
				if j := 32*k + l; idx[j] == i4Escape {
					evs = append(evs, wnr4Event{int(l), m.scaleRow(mm, 64+b), scale(j) >> (8 * (3 - b)) & 255})
				}
			}
		}
		for l := uint64(0); l < wnr4Lanes && 32*k+l < n; l++ {
			hb[l] = append(hb[l], scale(32*k+l))
		}
	}
	var tails [wnr4Lanes]uint16
	for r := uint64(0); r < steps; r++ {
		var pm, mx, cls [wnr4Lanes]int
		for l := uint64(0); l < wnr4Lanes && 32*r+l < n; l++ {
			cls[l] = min(max(int(scale(32*r+l)>>21)-int(anchor), -4), 4) + 4
			pm[l], mx[l] = 9, 9
		}
		for el := 0; el < 64; el++ {
			for l := uint64(0); l < wnr4Lanes && 32*r+l < n; l++ {
				if el >= 60 && l >= 2 && r == (n-l-1)/32 {
					continue
				}
				y := i4Nib(code(32*r+l), el, g.Pairs)
				evs = append(evs, wnr4Event{int(l), m.nibRow(mm, (cls[l]*10+pm[l])*10+mx[l]), y})
				a := int(y) - 8
				if a < 0 {
					a = -a
				}
				pm[l] = a
				if mx[l] == 9 {
					mx[l] = a
				} else {
					mx[l] = max(mx[l], a)
				}
			}
		}
		for l := uint64(0); l < wnr4Lanes && 32*r+l < n; l++ {
			c := code(32*r + l)
			for w := 0; w < 8; w++ {
				hb[l] = append(hb[l], le.Uint32(c[4*w:]))
			}
			if l >= 2 && r == (n-l-1)/32 {
				for el := 60; el < 64; el++ {
					tails[l] |= uint16(i4Nib(c, el, g.Pairs)) << (4 * (el - 60))
				}
			}
		}
	}
	return wnr4EncodeTile(evs, wnr4Starts(ident, &hb, wnr4Owns(n), &tails))
}

// i4Data makes random record content for g: per block a base (sometimes 0), scales on the lattice of their
// block's base with neighbours close to each other, a share of hostile scales that escape, codes near 8.
// allEscape lists the blocks all of whose scales escape.
func i4Data(r *rand.Rand, g I4Geo, hostile float64) (data []byte, bases []uint32, allEscape map[uint64]bool) {
	L, _ := i4Geometry(g)
	data = make([]byte, L.total)
	bases = make([]uint32, L.nbases)
	for i := range bases {
		switch r.Intn(10) {
		case 0:
			bases[i] = 0
		case 1:
			bases[i] = 0x00800000 + uint32(r.Intn(1000)) // tiny: subnormal lattice values
		default:
			bases[i] = math.Float32bits(float32(math.Ldexp(1+r.Float64(), -12+r.Intn(6))))
		}
	}
	hostiles := []uint32{0x7FC00000, 0x7F800000, 0xFF800000, 0x80000000, 0x00000001, 0x007FFFFF, 0xBF800000, 0x3F800001}
	escaped := map[uint64]int{}
	total := map[uint64]int{}
	for mm := range g.Rows {
		gc := L.gc[mm]
		k := uint32(100 + r.Intn(27))
		for i := uint64(0); i < L.cells[mm]; i++ {
			blk := L.bbase[mm] + (i/gc/128)*L.nbc[mm] + (i%gc)/2
			B := math.Float32frombits(bases[blk])
			k = uint32(min(max(int(k)+r.Intn(9)-4, 0), 126))
			s := math.Float32bits(i4Lat(k, B))
			if r.Float64() < hostile {
				s = hostiles[r.Intn(len(hostiles))]
				if r.Intn(3) == 0 {
					s = r.Uint32()
				}
			}
			le.PutUint32(data[L.scaleAt[mm]+4*i:], s)
			total[blk]++
			onLattice := false
			for kk := uint32(0); kk <= 126; kk++ {
				if math.Float32bits(i4Lat(kk, B)) == s {
					onLattice = true
					break
				}
			}
			if !onLattice {
				escaped[blk]++
			}
			c := data[L.codeAt[mm]+32*i : L.codeAt[mm]+32*i+32]
			for e := range c {
				lo, hi := 8+r.Intn(5)-2, 8+r.Intn(5)-2
				if r.Intn(16) == 0 {
					lo = r.Intn(16)
				}
				c[e] = byte(lo | hi<<4)
			}
		}
	}
	allEscape = map[uint64]bool{}
	for blk, n := range total {
		if escaped[blk] == n {
			allEscape[blk] = true
		}
	}
	return
}

type i4Case struct {
	name    string
	G       uint32
	g       I4Geo
	hostile float64
}

func i4Cases() []i4Case {
	geo := func(pairs bool, rc ...uint64) I4Geo {
		g := I4Geo{Pairs: pairs}
		for i := 0; i < len(rc); i += 2 {
			g.Rows = append(g.Rows, rc[i])
			g.Cols = append(g.Cols, rc[i+1])
		}
		return g
	}
	return []i4Case{
		{"gc 1, one group", 1, geo(false, 1, 64), 0},
		{"gc 1, rows 129, G 1", 1, geo(false, 129, 64), 0.05},
		{"gc 31, rows 127", 32, geo(false, 127, 1984), 0.02},
		{"gc 32, rows 128, pairs", 32, geo(true, 128, 2048), 0.02},
		{"gc 33, rows 129, G 64", 64, geo(false, 129, 2112), 0.02},
		{"3 matrices, G 100", 100, geo(true, 5, 128, 3, 192, 40, 64), 0.1},
		{"2 matrices, G 4096", 4096, geo(false, 64, 128, 7, 320), 0.3},
		{"all hostile", 2, geo(true, 9, 640), 1},
	}
}

func TestWNR4I4RoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	m, e := OpenI4Model(randI4Model(r))
	if e != nil {
		t.Fatal(e)
	}
	before := SeenWNR4
	for _, c := range i4Cases() {
		data, bases, _ := i4Data(r, c.g, c.hostile)
		var anchors [3]uint16
		for i := range c.g.Rows {
			anchors[i] = uint16(400 + r.Intn(100))
		}
		owner := r.Uint32()
		for pk := 0; pk < 2; pk++ { // both packings decode the same bytes
			g := c.g
			g.Pairs = pk == 1
			rec := encodeI4(m, g, owner, c.G, anchors, bases, data)
			out := make([]byte, len(data))
			if e := decodeI4(rec, m, g, owner, out); e != nil {
				t.Fatalf("%s, pairs %v: %v", c.name, g.Pairs, e)
			}
			if !bytes.Equal(out, data) {
				t.Fatalf("%s, pairs %v: decoded bytes differ", c.name, g.Pairs)
			}
		}
	}
	if SeenWNR4.I4Escapes == before.I4Escapes || SeenWNR4.TailLanes == before.TailLanes || SeenWNR4.EmptyLanes == before.EmptyLanes {
		t.Fatal("no escape, tail lane or empty lane met")
	}
}

func TestWNR4I4Model(t *testing.T) {
	r := rand.New(rand.NewSource(22))
	blob := randI4Model(r)
	if len(blob) != 196688 {
		t.Fatalf("model of %d bytes", len(blob))
	}
	if _, e := OpenI4Model(blob); e != nil {
		t.Fatal(e)
	}
	muts := map[string]func(b []byte) []byte{
		"short":       func(b []byte) []byte { return b[:len(b)-2] },
		"kind":        func(b []byte) []byte { b[8] = 2; return b },
		"contexts":    func(b []byte) []byte { b[16] = 0x85; return b },
		"scale ctx":   func(b []byte) []byte { b[20] = 69; return b },
		"word 6":      func(b []byte) []byte { b[24] = 1; return b },
		"nib table":   func(b []byte) []byte { b[wnr4ModelHdr+34*1000] = 1; return b },
		"scale table": func(b []byte) []byte { b[len(b)-1] ^= 0x40; return b },
	}
	for name, f := range muts {
		if _, e := OpenI4Model(f(append([]byte(nil), blob...))); e == nil || e.Class != EBlob {
			t.Errorf("%s: got %v", name, e)
		}
	}
}

func TestWNR4I4Geometry(t *testing.T) {
	bad := map[string]I4Geo{
		"no matrix":    {},
		"4 matrices":   {Rows: []uint64{1, 1, 1, 1}, Cols: []uint64{64, 64, 64, 64}},
		"rows 0":       {Rows: []uint64{0}, Cols: []uint64{64}},
		"cols 32":      {Rows: []uint64{1}, Cols: []uint64{32}},
		"cols 96":      {Rows: []uint64{1}, Cols: []uint64{96}},
		"cols missing": {Rows: []uint64{1, 1}, Cols: []uint64{64}},
		"2^32 bytes":   {Rows: []uint64{1 << 20}, Cols: []uint64{64 * 114}}, // 36 x 119,537,664 groups
	}
	for name, g := range bad {
		if _, e := i4Geometry(g); e == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	L, e := i4Geometry(I4Geo{Rows: []uint64{129, 1}, Cols: []uint64{192, 64}})
	if e != nil || L.nbases != 2*2+1 || L.bbase[1] != 4 || L.total != 36*(129*3+1) || L.scaleAt[0] != 32*(129*3+1) {
		t.Fatalf("layout %+v, %v", L, e)
	}
}

func TestWNR4I4RecordChecks(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	m, _ := OpenI4Model(randI4Model(r))
	g := I4Geo{Rows: []uint64{130, 3}, Cols: []uint64{192, 128}}
	data, bases, _ := i4Data(r, g, 0.05)
	anchors := [3]uint16{450, 470, 0}
	owner := uint32(0x00030011)
	rec := encodeI4(m, g, owner, 4, anchors, bases, data)
	out := make([]byte, len(data))
	if e := decodeI4(rec, m, g, owner, out); e != nil {
		t.Fatal(e)
	}
	ntiles := int(le.Uint32(rec[20:]))
	baseAt := wnr4Head + 4*(ntiles+1)
	setBase := func(i int, v uint32) func(b []byte) []byte {
		return func(b []byte) []byte { le.PutUint32(b[baseAt+4*i:], v); return b }
	}
	muts := []struct {
		name string
		f    func(b []byte) []byte
		want string
	}{
		{"magic", func(b []byte) []byte { b[1] ^= 4; return b }, "magic"},
		{"model id", func(b []byte) []byte { b[7] ^= 1; return b }, "model id"},
		{"owner", func(b []byte) []byte { b[11] ^= 1; return b }, "owner"},
		{"kind", func(b []byte) []byte { b[12] = 2; return b }, "kind"},
		{"packing", func(b []byte) []byte { b[13] = 1; return b }, "packing"},
		{"flags bit 1", func(b []byte) []byte { b[13] = 2; return b }, "packing"},
		{"byte 16", func(b []byte) []byte { b[16] = 1; return b }, "16..19"},
		{"byte 34", func(b []byte) []byte { b[34] = 1; return b }, "34..63"},
		{"byte 63", func(b []byte) []byte { b[63] = 0x80; return b }, "34..63"},
		{"anchor of no matrix", func(b []byte) []byte { b[32] = 1; return b }, "anchor of matrix 2"},
		{"G 4097", func(b []byte) []byte { le.PutUint16(b[14:], 4097); return b }, "G = 4097"},
		{"G 1", func(b []byte) []byte { le.PutUint16(b[14:], 1); return b }, "tiles in the record"},
		{"ntiles", func(b []byte) []byte { b[20]--; return b }, "tiles in the record"},
		{"total", func(b []byte) []byte { return append(b, make([]byte, 4096)...) }, "record of"},
		{"base NaN", setBase(0, 0x7FC00000), "base 0"},
		{"base inf", setBase(1, 0x7F800000), "base 1"},
		{"base -0", setBase(2, 0x80000000), "base 2"},
		{"base negative", setBase(3, 0xBF800000), "base 3"},
		{"base subnormal", setBase(4, 0x00000001), "base 4"},
		{"base subnormal max", setBase(4, 0x007FFFFF), "base 4"},
		{"payload padding", func(b []byte) []byte { b[len(b)-3] = 1; return b }, "after the payload"},
		{"off[1] odd", func(b []byte) []byte { b[68] ^= 1; return b }, "tile 0 from"},
	}
	for _, mu := range muts {
		b := mu.f(append([]byte(nil), rec...))
		e := decodeI4(b, m, g, owner, out)
		if e == nil || !strings.Contains(e.Msg, mu.want) {
			t.Errorf("%s: got %v, want %q", mu.name, e, mu.want)
		}
	}
	g2 := g
	g2.Pairs = true
	if e := decodeI4(rec, m, g2, owner, out); e == nil || !strings.Contains(e.Msg, "packing") {
		t.Errorf("geometry with the other packing: %v", e)
	}
	g2 = I4Geo{Rows: []uint64{130, 3}, Cols: []uint64{192, 192}}
	if decodeI4(rec, m, g2, owner, make([]byte, 36*(130*3+3*3))) == nil {
		t.Error("another geometry accepted")
	}

	// a lattice index above 127 is invalid
	i4BadIndex = 200
	b := encodeI4(m, g, owner, 4, anchors, bases, data)
	i4BadIndex = -1
	if e := decodeI4(b, m, g, owner, out); e == nil || !strings.Contains(e.Msg, "lattice index 200") {
		t.Errorf("index 200: %v", e)
	}
	// 127 + 1 too
	i4BadIndex = 128
	b = encodeI4(m, g, owner, 4, anchors, bases, data)
	i4BadIndex = -1
	if e := decodeI4(b, m, g, owner, out); e == nil || !strings.Contains(e.Msg, "lattice index 128") {
		t.Errorf("index 128: %v", e)
	}
}

// Every single-bit flip of a record is refused, except a flip of the base of a block whose scales all
// escape into another valid base: it is never read, and the record decodes to the same bytes (WNR4-SPEC
// 6.5, "What the tile hash covers"). An anchor is in the hash ident, so a changed anchor is refused.
func TestWNR4I4BitFlips(t *testing.T) {
	r := rand.New(rand.NewSource(24))
	m, _ := OpenI4Model(randI4Model(r))
	g := I4Geo{Rows: []uint64{40, 2}, Cols: []uint64{256, 64}, Pairs: true}
	data, bases, allEsc := i4Data(r, g, 0.1)
	bases[0] = 0 // block 0: base 0, every scale escaped
	L, _ := i4Geometry(g)
	for i := uint64(0); i < L.cells[0]; i++ {
		if (i/L.gc[0]/128)*L.nbc[0]+(i%L.gc[0])/2 == 0 {
			le.PutUint32(data[L.scaleAt[0]+4*i:], 0xFFC00000|uint32(i))
		}
	}
	allEsc[0] = true
	rec := encodeI4(m, g, 5, 1, [3]uint16{455, 460, 0}, bases, data)
	ntiles := int(le.Uint32(rec[20:]))
	baseAt := wnr4Head + 4*(ntiles+1)
	end := baseAt + 4*len(bases) + int(le.Uint32(rec[24:]))
	out := make([]byte, len(data))
	refused, unread := 0, 0
	for k := 0; k < 8*end; k++ {
		b := append([]byte(nil), rec...)
		b[k/8] ^= 1 << uint(k%8)
		e := decodeI4(b, m, g, 5, out)
		if e == nil {
			p := k/8 - baseAt
			blk := uint64(p / 4)
			if p < 0 || blk >= uint64(len(bases)) || !allEsc[blk] || !bytes.Equal(out, data) {
				t.Fatalf("flip of bit %d (byte %d) decoded without error", k%8, k/8)
			}
			unread++
			continue
		}
		refused++
	}
	if unread == 0 {
		t.Fatal("no flip of an unread base met")
	}
	t.Logf("%d flips refused, %d of unread bases decoded to the same bytes", refused, unread)
}
