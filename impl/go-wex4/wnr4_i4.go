// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// WNR4 kind 3, I4: int4-g64 expert records (1 to 3 matrices of int4 codes with one f32 scale per 64 inputs)
// in WNR4 tiles, the scales as indices into the lattice of an FP8 block-scaled source or escaped
// (docs/WNR4-SPEC.md section 6), and its WNM4 model (6.2). Decoder only, CPU, one thread, on the core of
// wnr4.go. Not called by Verify yet.

package main

import "math"

const (
	i4Kind       = 3
	i4NibCtx     = 900                                                               // nibble tables per matrix slot: 9 classes x 10 pm x 10 mx
	i4ScaleCtx   = 68                                                                // scale tables per matrix slot: 64 index contexts, 4 escape bytes
	i4ModelBytes = wnr4ModelHdr + 3*i4NibCtx*2*wnr4CDF16 + 3*i4ScaleCtx*2*wnr4CDF256 // 196,688
	i4Escape     = 127
	i4Group      = 64 // inputs per group
	i4CodeBytes  = 32 // code bytes per group
)

// I4Model is a WNM4 blob of kind 3: 3 x 900 nibble tables, 3 x 68 scale tables.
type I4Model struct {
	ID    uint32
	nib   []uint16 // 3 x 900 x 17
	scale []uint16 // 3 x 68 x 257
}

// OpenI4Model checks a WNM4 kind-3 blob: size, header, every table.
func OpenI4Model(b []byte) (*I4Model, *Error) {
	if e := wnr4ModelHeader(b, i4ModelBytes, [8]uint32{magicWNM4, 1, i4Kind, cdfBits, i4NibCtx, i4ScaleCtx, 0, 0}); e != nil {
		return nil, e
	}
	m := &I4Model{ID: fnv1a(b)}
	p := wnr4ModelHdr
	var e *Error
	if m.nib, e = wnr4Tables(b, &p, 3*i4NibCtx, wnr4CDF16, "I4 nibble"); e != nil {
		return nil, e
	}
	if m.scale, e = wnr4Tables(b, &p, 3*i4ScaleCtx, wnr4CDF256, "I4 scale"); e != nil {
		return nil, e
	}
	return m, nil
}

func (m *I4Model) nibRow(mat, row int) []uint16 {
	r := mat*i4NibCtx + row
	return m.nib[r*wnr4CDF16 : (r+1)*wnr4CDF16]
}

func (m *I4Model) scaleRow(mat, row int) []uint16 {
	r := mat*i4ScaleCtx + row
	return m.scale[r*wnr4CDF256 : (r+1)*wnr4CDF256]
}

// I4Geo is the geometry of an I4 record (WNR4-SPEC 6.1): nmat = len(Rows) = len(Cols), and the packing.
type I4Geo struct {
	Rows, Cols []uint64
	Pairs      bool
}

// i4Layout is what the geometry gives: per matrix the groups per row, the groups, the first base block and
// the base blocks per row of blocks, and where its codes and scales lie in the decoded record.
type i4Layout struct {
	gc, cells, bbase, nbc   [3]uint64
	codeAt, scaleAt         [3]uint64
	total, nbases, allCells uint64
}

func i4Geometry(g I4Geo) (*i4Layout, *Error) {
	n := len(g.Rows)
	if n < 1 || n > 3 || len(g.Cols) != n {
		return nil, bad("WNR4 I4: %d matrices (%d cols)", n, len(g.Cols))
	}
	L := &i4Layout{}
	for m := 0; m < n; m++ {
		r, c := g.Rows[m], g.Cols[m]
		if r < 1 || c < i4Group || c%i4Group != 0 || r > 1<<32 || c > 1<<32 {
			return nil, bad("WNR4 I4: matrix %d is %d x %d", m, r, c)
		}
		L.gc[m] = c / i4Group
		L.cells[m] = r * L.gc[m]
		if L.cells[m] > 1<<32 {
			return nil, bad("WNR4 I4: matrix %d is %d x %d", m, r, c)
		}
		L.nbc[m] = ceilDiv(L.gc[m], 2)
		L.bbase[m] = L.nbases
		L.nbases += ceilDiv(r, 128) * L.nbc[m]
		L.allCells += L.cells[m]
	}
	L.total = 36 * L.allCells
	if L.total > 1<<32-1 {
		return nil, bad("WNR4 I4: %d bytes", L.total)
	}
	var at uint64
	for m := 0; m < n; m++ {
		L.codeAt[m] = at
		at += i4CodeBytes * L.cells[m]
	}
	for m := 0; m < n; m++ {
		L.scaleAt[m] = at
		at += 4 * L.cells[m]
	}
	return L, nil
}

// e4m3 is the value of the FP8 E4M3 code k (0..126), exact in binary32.
func e4m3(k uint32) float32 {
	if k < 8 {
		return float32(k) / 512 // k x 2^-9, exact
	}
	m := float32(8+k&7) / 8 // 1 + (k & 7) / 8, exact
	e := int(k>>3) - 7
	for ; e > 0; e-- {
		m *= 2
	}
	for ; e < 0; e++ {
		m /= 2
	}
	return m
}

// i4Lat is lat(k, B) = fl(fl(E4M3(k) x B) / 7): a binary32 multiply, then a binary32 division by 7, each
// rounded to nearest even, subnormals kept. The explicit conversions make Go round each step to binary32
// (the Go spec allows a fused or wider computation of an expression otherwise).
func i4Lat(k uint32, B float32) float32 {
	p := float32(e4m3(k) * B)
	return float32(p / 7)
}

// i4BaseOK: a base is 0 or a positive normal finite f32.
func i4BaseOK(b uint32) bool { return b == 0 || (b >= 0x00800000 && b < 0x7F800000) }

// decodeI4 decodes a kind-3 record into out (36 x the groups: the codes, then the scales). On an error out
// holds no usable data.
func decodeI4(rec []byte, model *I4Model, g I4Geo, owner uint32, out []byte) *Error {
	L, e := i4Geometry(g)
	if e != nil {
		return e
	}
	if uint64(len(out)) != L.total {
		return bad("WNR4 I4: the geometry gives %d bytes, the record decodes to %d", L.total, len(out))
	}
	h, e := wnr4ParseHeader(rec)
	if e != nil {
		return e
	}
	if h.Model != model.ID {
		return bad("WNR4 I4: model id 0x%08x in the record, the blob is 0x%08x", h.Model, model.ID)
	}
	if h.Owner != owner {
		return bad("WNR4 I4: owner 0x%08x in the record, 0x%08x expected", h.Owner, owner)
	}
	P := byte(0)
	if g.Pairs {
		P = 1
	}
	if h.Kind != i4Kind || h.Flags != P {
		return bad("WNR4 I4: kind %d flags 0x%02x, the geometry's packing is %d", h.Kind, h.Flags, P)
	}
	if !allZero(rec[16:20]) || !allZero(rec[34:64]) {
		return bad("WNR4 I4: header bytes 16..19 or 34..63 not zero")
	}
	nmat := len(g.Rows)
	var anchor [3]uint32
	for m := 0; m < 3; m++ {
		anchor[m] = uint32(le.Uint16(rec[28+2*m:]))
		if m >= nmat && anchor[m] != 0 {
			return bad("WNR4 I4: anchor of matrix %d is %d, the geometry has %d matrices", m, anchor[m], nmat)
		}
	}
	var ntiles uint64
	for m := 0; m < nmat; m++ {
		ntiles += wnr4Tiles(L.cells[m], h.G)
	}
	lay, e := wnr4Cut(rec, h, ntiles, 4*L.nbases)
	if e != nil {
		return e
	}
	for i := uint64(0); i < L.nbases; i++ {
		if b := le.Uint32(lay.area[4*i:]); !i4BaseOK(b) {
			return bad("WNR4 I4: base %d is 0x%08x", i, b)
		}
	}
	t := uint64(0)
	per := uint64(wnr4Lanes) * uint64(h.G)
	var idx []byte // the lattice indices of the tile's groups, the contexts of later steps
	for m := 0; m < nmat; m++ {
		for c0 := uint64(0); c0 < L.cells[m]; c0 += per {
			n := min(per, L.cells[m]-c0)
			if uint64(cap(idx)) < n {
				idx = make([]byte, n)
			}
			ident := [wnr4Ident]uint32{model.ID, owner, uint32(t), 0, i4Kind | uint32(P)<<8 | h.G<<16, uint32(m) | anchor[m]<<16}
			tl := i4Tile{model: model, lay: L, g: g, m: m, c0: c0, n: n, anchor: anchor[m], bases: lay.area, out: out, idx: idx[:n]}
			if e := tl.decode(lay.tile(t), ident); e != nil {
				return fail(e.Class, "%s (tile %d)", e.Msg, t)
			}
			t++
		}
	}
	if g.Pairs {
		SeenWNR4.I4Pairs++
	}
	SeenWNR4.Records++
	return nil
}

type i4Tile struct {
	model  *I4Model
	lay    *i4Layout
	g      I4Geo
	m      int
	c0, n  uint64
	anchor uint32
	bases  []byte
	out    []byte
	idx    []byte
}

// base is the base of group i (an index into matrix m) as an f32.
func (tl *i4Tile) base(i uint64) float32 {
	L, m := tl.lay, tl.m
	row, col := i/L.gc[m], i%L.gc[m]
	blk := L.bbase[m] + (row/128)*L.nbc[m] + col/2
	return math.Float32frombits(le.Uint32(tl.bases[4*blk:]))
}

// code returns the 32 code bytes of group i.
func (tl *i4Tile) code(i uint64) []byte {
	a := tl.lay.codeAt[tl.m] + i4CodeBytes*i
	return tl.out[a : a+i4CodeBytes]
}

// setNib puts element e (0..63) of a group's code bytes in the record's packing.
func (tl *i4Tile) setNib(c []byte, e int, v byte) {
	var b, sh int
	if tl.g.Pairs {
		b, sh = e/2, 4*(e%2)
	} else {
		b, sh = e%32, 4*(e/32)
	}
	c[b] = c[b]&^(15<<sh) | v<<sh
}

func (tl *i4Tile) decode(tile []byte, ident [wnr4Ident]uint32) *Error {
	d, e := wnr4Open(tile, ident)
	if e != nil {
		return e
	}
	L, m, n, c0 := tl.lay, tl.m, tl.n, tl.c0
	steps := ceilDiv(n, wnr4Lanes)
	gc := L.gc[m]
	dist := gc // the reference: the group dist / gc rows above
	if gc < 32 {
		dist = ceilDiv(32, gc) * gc
	}
	var esc [wnr4Lanes]uint32 // phase 1: the escaped scales of the step, by lane
	for k := uint64(0); k < steps; k++ {
		for l := 0; l < wnr4Lanes; l++ {
			j := 32*k + uint64(l) // the group's index in the tile: group c0 + j of the matrix
			if j >= n {
				break
			}
			kref, row := uint32(120), 0
			if j >= dist {
				kref = uint32(tl.idx[j-dist])
				if kref == i4Escape {
					kref = 126
				}
				row = 32
			}
			y := d.sym(l, tl.model.scaleRow(m, row+int(kref/4)))
			kk := (y + kref) & 255
			if kk > i4Escape {
				return bad("WNR4 I4: lattice index %d", kk)
			}
			tl.idx[j] = byte(kk)
			esc[l] = 0
		}
		for b := 0; b < 4; b++ {
			for l := 0; l < wnr4Lanes; l++ {
				j := 32*k + uint64(l)
				if j >= n {
					break
				}
				if tl.idx[j] == i4Escape {
					y := d.sym(l, tl.model.scaleRow(m, 64+b))
					esc[l] |= y << (8 * (3 - b))
				}
			}
		}
		for l := 0; l < wnr4Lanes; l++ {
			j := 32*k + uint64(l)
			if j >= n {
				break
			}
			s := esc[l]
			if tl.idx[j] != i4Escape {
				s = math.Float32bits(i4Lat(uint32(tl.idx[j]), tl.base(c0+j)))
			} else {
				SeenWNR4.I4Escapes++
			}
			le.PutUint32(tl.out[L.scaleAt[m]+4*(c0+j):], s)
			d.word(l, s)
		}
	}
	for r := uint64(0); r < steps; r++ { // phase 2, rounds
		var pm, mx, cls [wnr4Lanes]int
		for l := 0; l < wnr4Lanes; l++ {
			j := 32*r + uint64(l)
			if j >= n {
				break
			}
			s := le.Uint32(tl.out[L.scaleAt[m]+4*(c0+j):])
			c := int(s>>21) - int(tl.anchor)
			cls[l] = min(max(c, -4), 4) + 4
			pm[l], mx[l] = 9, 9
		}
		for el := 0; el < i4Group; el++ {
			for l := 0; l < wnr4Lanes; l++ {
				j := 32*r + uint64(l)
				if j >= n {
					break
				}
				if el >= 60 && l >= 2 && r == (n-uint64(l)-1)/32 {
					continue // the tail: in the lane's final state
				}
				y := d.sym(l, tl.model.nibRow(m, (cls[l]*10+pm[l])*10+mx[l]))
				tl.setNib(tl.code(c0+j), el, byte(y))
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
		for l := 0; l < wnr4Lanes; l++ {
			j := 32*r + uint64(l)
			if j >= n {
				break
			}
			if l >= 2 && r == (n-uint64(l)-1)/32 {
				continue // hashed once its tail is known
			}
			d.bytes(l, tl.code(c0+j))
		}
	}
	for l := 2; l < wnr4Lanes && uint64(l) < n; l++ {
		v, _ := d.tail(l) // range checked by finish
		c := tl.code(c0 + uint64(l) + 32*((n-uint64(l)-1)/32))
		for el := 60; el < 64; el++ {
			tl.setNib(c, el, byte(v>>(4*(el-60))&15))
		}
		d.bytes(l, c)
	}
	return d.finish(wnr4Owns(n))
}
