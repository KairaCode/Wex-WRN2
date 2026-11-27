// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

package main

import (
	"bytes"
	"crypto/sha256"
	"math/rand"
	"testing"
)

func testMakeMXNVModel(r *rand.Rand, kind, pb uint32) []byte {
	wctx, sctx := 567*int(pb), 1
	if kind == mxnvKindNV {
		wctx, sctx = 1053*int(pb), 4160
	}
	size := wnr4ModelHdr + 3*wctx*2*wnr4CDF16 + 3*sctx*2*wnr4CDF256
	b := make([]byte, size)
	hdr := [8]uint32{magicWNM4, 1, kind, cdfBits, uint32(wctx), uint32(sctx), pb, 0}
	for i, w := range hdr {
		le.PutUint32(b[4*i:], w)
	}
	u16 := wnr4Uniform(16)
	r16 := wnr4RandCDF(r, 16, true)
	u256 := wnr4Uniform(256)
	r256 := wnr4RandCDF(r, 256, true)
	p := wnr4ModelHdr
	for i := 0; i < 3*wctx; i++ {
		c := u16
		if i%7 == 0 {
			c = r16
		}
		for _, v := range c {
			le.PutUint16(b[p:], v)
			p += 2
		}
	}
	for i := 0; i < 3*sctx; i++ {
		c := u256
		if i%13 == 0 {
			c = r256
		}
		for _, v := range c {
			le.PutUint16(b[p:], v)
			p += 2
		}
	}
	return b
}

func testEncodeMXNV(model *MXNVModel, g Geo, coded []byte, flags byte, anchors [3]byte, owner, G uint32) []byte {
	spc, wmult, bNib, rLim := uint64(1), uint64(16), 32, 3
	if g.Kind == mxnvKindNV {
		spc, wmult, bNib, rLim = 2, 8, 16, 6
	}
	var scount, cells, soff, woff [3]uint64
	var at uint64
	for m := 0; m < 3; m++ {
		scount[m] = g.scales(m)
		cells[m] = scount[m] / spc
		soff[m] = at
		at += scount[m]
	}
	for m := 0; m < 3; m++ {
		woff[m] = at
		at += scount[m] * wmult
	}
	anchorsWord := uint32(anchors[0]) | uint32(anchors[1])<<8 | uint32(anchors[2])<<16
	w4 := g.Kind | uint32(flags)<<8 | G<<16
	per := uint64(wnr4Lanes) * uint64(G)
	var tiles [][]byte
	var t uint64
	for m := 0; m < 3; m++ {
		so := coded[soff[m] : soff[m]+scount[m]]
		wo := coded[woff[m] : woff[m]+cells[m]*16]
		cols := g.Cols[m]
		for c0 := uint64(0); c0 < cells[m]; c0 += per {
			n := min(per, cells[m]-c0)
			ident := [wnr4Ident]uint32{model.ID, owner, uint32(t), 0, w4, anchorsWord}
			var events []wnr4Event
			var blocks [wnr4Lanes][]uint32
			ns, s0 := n*spc, c0*spc
			for k := uint64(0); k < ceilDiv(ns, wnr4Lanes); k++ {
				for l := 0; l < wnr4Lanes; l++ {
					idx := 32*k + uint64(l)
					if idx >= ns {
						break
					}
					i := s0 + idx
					sc, pred := mxnvScaleCtx(g.Kind, so, i, s0, cols)
					y := so[i]
					d := y - pred
					events = append(events, wnr4Event{l, model.sTable(m, sc), uint32(d)})
					blocks[l] = append(blocks[l], uint32(y))
				}
			}
			var tails [wnr4Lanes]uint16
			for j := uint64(0); j < ceilDiv(n, wnr4Lanes); j++ {
				act := int(min(uint64(wnr4Lanes), n-32*j))
				var pm, mx, cls [wnr4Lanes]int
				for k := 0; k < 32; k++ {
					if k%bNib == 0 {
						for l := 0; l < act; l++ {
							cls[l] = mxnvScaleCls(so[(c0+32*j+uint64(l))*spc+uint64(k/bNib)], anchors[m], rLim)
							pm[l], mx[l] = 8, 8
						}
					}
					for l := 0; l < act; l++ {
						cell := c0 + 32*j + uint64(l)
						b := wo[cell*16+uint64(k/2)]
						if k&1 != 0 {
							b >>= 4
						}
						y := uint32(b) & 15
						if k >= 28 && l >= 2 && j == (n-uint64(l)-1)/32 {
							continue
						}
						row := mxnvWRow(cls[l], pm[l], mx[l], model.PB, k%bNib, bNib)
						events = append(events, wnr4Event{l, model.wTable(m, row), y})
						a := int(y & 7)
						pm[l] = a
						if mx[l] == 8 || a > mx[l] {
							mx[l] = a
						}
					}
				}
				for l := 0; l < act; l++ {
					cell := c0 + 32*j + uint64(l)
					if l >= 2 && j == (n-uint64(l)-1)/32 {
						tails[l] = le.Uint16(wo[cell*16+14:])
					}
					for k := 0; k < 4; k++ {
						blocks[l] = append(blocks[l], le.Uint32(wo[cell*16+uint64(4*k):]))
					}
				}
			}
			tiles = append(tiles, wnr4EncodeTile(events, wnr4Starts(ident, &blocks, wnr4Owns(n), &tails)))
			t++
		}
	}
	var head [wnr4Head]byte
	le.PutUint32(head[0:], magicWNR4)
	le.PutUint32(head[4:], model.ID)
	le.PutUint32(head[8:], owner)
	head[12] = byte(g.Kind)
	head[13] = flags
	le.PutUint16(head[14:], uint16(G))
	head[16], head[17], head[18] = anchors[0], anchors[1], anchors[2]
	return wnr4BuildRecord(head, nil, tiles)
}

func TestWNR4MXNV(t *testing.T) {
	rng := rand.New(rand.NewSource(20261127))
	cases := []struct {
		name     string
		kind, pb uint32
		cz, ceil bool
		g        Geo
	}{
		{"mx-pb1-plain", mxnvKindMX, 1, false, false, Geo{Kind: mxnvKindMX, Rows: [3]uint64{7, 5, 4}, Cols: [3]uint64{11, 9, 13}}},
		{"mx-pb5-cz-ceil", mxnvKindMX, 5, true, true, Geo{Kind: mxnvKindMX, Rows: [3]uint64{7, 5, 4}, Cols: [3]uint64{11, 9, 13}}},
		{"nv-pb1-plain", mxnvKindNV, 1, false, false, Geo{Kind: mxnvKindNV, Rows: [3]uint64{6, 4, 6}, Cols: [3]uint64{36, 12, 36}}},
		{"nv-pb5-cz", mxnvKindNV, 5, true, false, Geo{Kind: mxnvKindNV, Rows: [3]uint64{6, 4, 6}, Cols: [3]uint64{36, 12, 36}}},
	}
	for _, tc := range cases {
		blob := testMakeMXNVModel(rng, tc.kind, tc.pb)
		model, e := OpenMXNVModel(blob)
		if e != nil {
			t.Fatalf("%s: OpenMXNVModel: %v", tc.name, e)
		}
		var raw uint64
		var soff, woff [3]uint64
		for m := 0; m < 3; m++ {
			soff[m] = raw
			raw += tc.g.scales(m)
		}
		for m := 0; m < 3; m++ {
			woff[m] = raw
			raw += tc.g.cells(m) * 16
		}
		coded := make([]byte, raw)
		for m := 0; m < 3; m++ {
			for i := uint64(0); i < tc.g.scales(m); i++ {
				s := 100 + rng.Intn(15)
				coded[soff[m]+i] = byte(s)
			}
			for c := uint64(0); c < tc.g.cells(m); c++ {
				maxA := 7
				if tc.ceil && c%2 == 0 {
					maxA = 5
				}
				for j := 0; j < 32; j++ {
					lim := maxA + 1
					a := rng.Intn(lim)
					if j == 31 && tc.ceil && c%2 == 0 {
						a = 5
					}
					x := a | rng.Intn(2)<<3
					if tc.cz && x == 8 {
						x = 0
					}
					if j&1 == 0 {
						coded[woff[m]+c*16+uint64(j/2)] = byte(x)
					} else {
						coded[woff[m]+c*16+uint64(j/2)] |= byte(x) << 4
					}
				}
			}
		}
		want := append([]byte(nil), coded...)
		if tc.ceil {
			for m := 0; m < 3; m++ {
				for c := uint64(0); c < tc.g.cells(m); c++ {
					base := woff[m] + c*16
					var mx byte
					for _, b := range want[base : base+16] {
						if b&7 > mx {
							mx = b & 7
						}
						hi := b >> 4
						if hi&7 > mx {
							mx = hi & 7
						}
					}
					if mx == 5 {
						want[soff[m]+c]--
						for k := uint64(0); k < 4; k++ {
							le.PutUint32(want[base+4*k:], mxnvCeilDbl8(le.Uint32(want[base+4*k:])))
						}
					}
				}
			}
		}
		var flags byte
		if tc.ceil {
			flags |= mxnvFlagCeil
		}
		if tc.cz {
			flags |= mxnvFlagCZ
		}
		rec := testEncodeMXNV(model, tc.g, coded, flags, [3]byte{105, 105, 105}, 0x00010002, 2)
		out := make([]byte, raw)
		if e := decodeMXNV(rec, model, tc.g, 0x00010002, tc.cz, out); e != nil {
			t.Fatalf("%s: decodeMXNV: %v", tc.name, e)
		}
		if !bytes.Equal(out, want) {
			t.Fatalf("%s: decoded bytes differ", tc.name)
		}
		if decodeMXNV(rec, model, tc.g, 0x00010003, tc.cz, out) == nil {
			t.Fatalf("%s: wrong owner accepted", tc.name)
		}
		if decodeMXNV(rec, model, tc.g, 0x00010002, !tc.cz, out) == nil {
			t.Fatalf("%s: wrong VALUE_EXACT accepted", tc.name)
		}
		badRec := append([]byte(nil), rec...)
		badRec[140] ^= 1
		if decodeMXNV(badRec, model, tc.g, 0x00010002, tc.cz, out) == nil {
			t.Fatalf("%s: corrupted record accepted", tc.name)
		}
	}
}

func TestWNR4NVContainer(t *testing.T) {
	g := Geo{Kind: mxnvKindNV, Rows: [3]uint64{6, 6, 4}, Cols: [3]uint64{36, 36, 12}}
	wctx, sctx := 1053, 4160
	size := wnr4ModelHdr + 3*wctx*2*wnr4CDF16 + 3*sctx*2*wnr4CDF256
	blob := make([]byte, size)
	hdr := [8]uint32{magicWNM4, 1, mxnvKindNV, cdfBits, uint32(wctx), uint32(sctx), 1, 0}
	for i, w := range hdr {
		le.PutUint32(blob[4*i:], w)
	}
	sk16 := make([]uint64, 16)
	sk256 := make([]uint64, 256)
	for i := range sk16 {
		sk16[i] = 1
	}
	sk16[0] = 4000
	for i := range sk256 {
		sk256[i] = 1
	}
	sk256[0] = 8000
	c16 := quantize(sk16, 4015)
	c256 := quantize(sk256, 8255)
	p := wnr4ModelHdr
	for i := 0; i < 3*wctx; i++ {
		for _, v := range c16 {
			le.PutUint16(blob[p:], v)
			p += 2
		}
	}
	for i := 0; i < 3*sctx; i++ {
		for _, v := range c256 {
			le.PutUint16(blob[p:], v)
			p += 2
		}
	}
	model, e := OpenMXNVModel(blob)
	if e != nil {
		t.Fatalf("OpenMXNVModel: %v", e)
	}
	var raw uint64
	var soff, woff [3]uint64
	for m := 0; m < 3; m++ {
		soff[m] = raw
		raw += g.scales(m)
	}
	for m := 0; m < 3; m++ {
		woff[m] = raw
		raw += g.cells(m) * 16
	}
	slab := make([]byte, raw)
	rec0 := testEncodeMXNV(model, g, slab, mxnvFlagCZ, [3]byte{0, 0, 0}, 0, 32)
	if uint64(len(rec0)) >= align(raw) {
		t.Fatalf("expected NV record (%d) smaller than raw (%d)", len(rec0), align(raw))
	}
	tnames := [6]string{"e0.s0", "e0.s1", "e0.s2", "e0.w0", "e0.w1", "e0.w2"}
	var nameBytes uint64
	for _, s := range tnames {
		nameBytes += uint64(len(s))
	}
	idxBytes := blobEntry + recordEntry + 6*(tensorEntry+4) + nameBytes
	idxSpan := align(idxBytes)
	blobOff := pageBytes + idxSpan
	blobSpan := align(uint64(len(blob)))
	recOff := blobOff + blobSpan
	fileBytes := recOff + uint64(len(rec0))
	f := make([]byte, fileBytes)
	copy(f[blobOff:], blob)
	copy(f[recOff:], rec0)
	at := uint64(pageBytes)
	le.PutUint32(f[at:], blobKindWNM4)
	le.PutUint32(f[at+4:], model.ID)
	le.PutUint64(f[at+8:], blobOff)
	le.PutUint64(f[at+16:], uint64(len(blob)))
	bsum := sha256.Sum256(blob)
	copy(f[at+24:], bsum[:])
	at += blobEntry
	le.PutUint64(f[at:], recOff)
	le.PutUint32(f[at+8:], uint32(len(rec0)))
	le.PutUint32(f[at+12:], uint32(raw))
	le.PutUint16(f[at+16:], codecWNR4)
	le.PutUint16(f[at+18:], flagExpert|flagValueExact)
	le.PutUint32(f[at+20:], 1)
	le.PutUint32(f[at+32:], 6)
	rsum := sha256.Sum256(rec0)
	copy(f[at+36:], rsum[:])
	at += recordEntry
	var noff uint32
	for m := 0; m < 6; m++ {
		dt := byte(dtypeF8E4M3)
		rows, cols, off := g.Rows[m%3], g.Cols[m%3], soff[m%3]
		if m >= 3 {
			dt = byte(dtypeU8)
			cols *= 8
			off = woff[m-3]
		}
		le.PutUint32(f[at:], noff)
		le.PutUint16(f[at+4:], uint16(len(tnames[m])))
		f[at+6] = dt
		f[at+7] = 2
		le.PutUint64(f[at+8:], rows)
		le.PutUint64(f[at+16:], cols)
		le.PutUint32(f[at+76:], 1)
		le.PutUint64(f[at+80:], off)
		le.PutUint64(f[at+88:], rows*cols)
		noff += uint32(len(tnames[m]))
		at += tensorEntry
	}
	for m := uint32(0); m < 6; m++ {
		le.PutUint32(f[at:], m)
		at += 4
	}
	for _, s := range tnames {
		at += uint64(copy(f[at:], s))
	}
	le.PutUint32(f[0:], magicWEX4)
	le.PutUint16(f[4:], 4)
	le.PutUint64(f[8:], reqValueExact|reqWNR4Expert)
	le.PutUint64(f[24:], fileBytes)
	le.PutUint64(f[32:], pageBytes)
	le.PutUint64(f[40:], idxBytes)
	le.PutUint32(f[48:], 1)
	le.PutUint32(f[52:], 1)
	le.PutUint32(f[56:], 6)
	le.PutUint32(f[60:], 1)
	le.PutUint32(f[64:], 1)
	le.PutUint32(f[68:], uint32(raw))
	le.PutUint64(f[72:], nameBytes)
	le.PutUint64(f[88:], raw)
	isum := sha256.Sum256(f[pageBytes : pageBytes+idxBytes])
	copy(f[96:], isum[:])
	hsum := sha256.Sum256(f[:4064])
	copy(f[4064:], hsum[:])

	res, e := verifyBytes(f, nil)
	if e != nil || res.Undecoded != 0 {
		t.Fatalf("valid NV container refused: %v (undecoded %v)", e, res)
	}
	bad := append([]byte(nil), f...)
	bad[recOff+200] ^= 1
	rsum2 := sha256.Sum256(bad[recOff : recOff+uint64(len(rec0))])
	copy(bad[pageBytes+blobEntry+36:], rsum2[:])
	isum2 := sha256.Sum256(bad[pageBytes : pageBytes+idxBytes])
	copy(bad[96:], isum2[:])
	hsum2 := sha256.Sum256(bad[:4064])
	copy(bad[4064:], hsum2[:])
	if _, e := verifyBytes(bad, nil); e == nil || e.Class != EData {
		t.Fatalf("corrupt NV codec-8 record: got %v, want %s", e, EData)
	}
}

