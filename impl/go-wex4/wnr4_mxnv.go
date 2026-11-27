// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// WNR4 kinds 0 (MXFP4) and 1 (NVFP4), docs/WNR4-SPEC.md sections 1 to 4: EXPERT slabs of three matrices
// (scales back to back, then packed FP4 weights) and their WNM4 model blobs (pb = 1 or 5).

package main

const (
	mxnvKindMX      = 0
	mxnvKindNV      = 1
	mxModelPB1Bytes = 59408
	mxModelPB5Bytes = 290744
	nvModelPB1Bytes = 6522158
	nvModelPB5Bytes = 6951782
	mxnvFlagCeil    = 0x07
	mxnvFlagCZ      = 0x08
)

func wnm4SizeOK(n uint64) bool {
	switch n {
	case mxModelPB1Bytes, mxModelPB5Bytes, nvModelPB1Bytes, nvModelPB5Bytes, ggModelBytes, i4ModelBytes:
		return true
	}
	return false
}

func wnm4KindSizeOK(kind uint32, n uint64) bool {
	if kind == mxnvKindMX {
		return n == mxModelPB1Bytes || n == mxModelPB5Bytes
	}
	if kind == mxnvKindNV {
		return n == nvModelPB1Bytes || n == nvModelPB5Bytes
	}
	return false
}

// MXNVModel is a validated WNM4 blob of kind 0 (MX) or kind 1 (NV).
type MXNVModel struct {
	ID         uint32
	Kind       byte
	PB         uint32
	wctx, sctx int
	wcdf, scdf []uint16
}

func (m *MXNVModel) wTable(mat, row int) []uint16 {
	base := (mat*m.wctx + row) * wnr4CDF16
	return m.wcdf[base : base+wnr4CDF16]
}

func (m *MXNVModel) sTable(mat, ctx int) []uint16 {
	base := (mat*m.sctx + ctx) * wnr4CDF256
	return m.scdf[base : base+wnr4CDF256]
}

// OpenMXNVModel validates a WNM4 kind-0 (MX) or kind-1 (NV) model blob (pb = 1 or 5).
func OpenMXNVModel(b []byte) (*MXNVModel, *Error) {
	if len(b) < wnr4ModelHdr {
		return nil, fail(EBlob, "WNM4 MX/NV: %d bytes", len(b))
	}
	kind := le.Uint32(b[8:])
	pb := le.Uint32(b[24:])
	if kind > 1 || (pb != 1 && pb != 5) {
		return nil, fail(EBlob, "WNM4 MX/NV: kind %d pb %d", kind, pb)
	}
	wctx, sctx := 567*int(pb), 1
	if kind == mxnvKindNV {
		wctx, sctx = 1053*int(pb), 4160
	}
	size := wnr4ModelHdr + 3*wctx*2*wnr4CDF16 + 3*sctx*2*wnr4CDF256
	if e := wnr4ModelHeader(b, size, [8]uint32{magicWNM4, 1, kind, cdfBits, uint32(wctx), uint32(sctx), pb, 0}); e != nil {
		return nil, e
	}
	p := wnr4ModelHdr
	wcdf, e := wnr4Tables(b, &p, 3*wctx, wnr4CDF16, "MX/NV weight")
	if e != nil {
		return nil, e
	}
	scdf, e := wnr4Tables(b, &p, 3*sctx, wnr4CDF256, "MX/NV scale")
	if e != nil {
		return nil, e
	}
	return &MXNVModel{ID: fnv1a(b), Kind: byte(kind), PB: pb, wctx: wctx, sctx: sctx, wcdf: wcdf, scdf: scdf}, nil
}

func mxnvScaleCls(s, anchor byte, r int) int {
	d := int(s) - int(anchor)
	if d < -r {
		d = -r
	} else if d > r {
		d = r
	}
	return d + r
}

func mxnvScaleCtx(kind uint32, s []byte, i, lo, cols uint64) (int, byte) {
	if kind == mxnvKindMX {
		return 0, 0
	}
	if cols < 33 || i < lo+cols {
		return 63, 0
	}
	up := s[i-cols]
	col := i % cols
	dul, dur := 7, 7
	if col > 0 && i-cols-1 >= lo {
		d := int(s[i-cols-1]) - int(up)
		if d < -3 {
			d = -3
		} else if d > 3 {
			d = 3
		}
		dul = d + 3
	}
	if col+1 < cols {
		d := int(s[i-cols+1]) - int(up)
		if d < -3 {
			d = -3
		} else if d > 3 {
			d = 3
		}
		dur = d + 3
	}
	return (1+int(up>>2))*64 + dul*8 + dur, up
}

func mxnvWRow(cls, pm, mx int, pb uint32, p, b int) int {
	ctx := (cls*9+pm)*9 + mx
	if pb == 1 {
		return ctx
	}
	bucket := 0
	if p+5 > b {
		bucket = p + 5 - b
	}
	return ctx*5 + bucket
}

func mxnvCeilDbl8(w uint32) uint32 {
	m := w & 0x77777777
	g := ((m >> 1) | (m >> 2)) & 0x11111111
	mn := (g << 1) | (m & 0x11111111 &^ g)
	return (w & 0x88888888) | (m + mn)
}

// decodeMXNV decodes a WNR4 kind-0 (MX) or kind-1 (NV) EXPERT record into out.
func decodeMXNV(rec []byte, model *MXNVModel, g Geo, owner uint32, valueExact bool, out []byte) *Error {
	if model == nil || uint32(model.Kind) != g.Kind {
		return bad("WNR4 MX/NV: model kind mismatch for geometry kind %d", g.Kind)
	}
	spc, wmult, bNib, rLim := uint64(1), uint64(16), 32, 3
	if g.Kind == mxnvKindNV {
		spc, wmult, bNib, rLim = 2, 8, 16, 6
	}
	var scount, cells, soff, woff [3]uint64
	var at uint64
	for m := 0; m < 3; m++ {
		sc := g.scales(m)
		if sc == 0 || sc%spc != 0 {
			return bad("WNR4 MX/NV: matrix %d has %d scales", m, sc)
		}
		scount[m] = sc
		cells[m] = sc / spc
		soff[m] = at
		at += sc
	}
	for m := 0; m < 3; m++ {
		woff[m] = at
		at += scount[m] * wmult
	}
	if uint64(len(out)) != at {
		return bad("WNR4 MX/NV: output %d bytes, geometry gives %d", len(out), at)
	}
	h, e := wnr4ParseHeader(rec)
	if e != nil {
		return e
	}
	if h.Model != model.ID {
		return bad("WNR4 MX/NV: model id 0x%08x in the record, the blob is 0x%08x", h.Model, model.ID)
	}
	if h.Owner != owner {
		return bad("WNR4 MX/NV: owner 0x%08x in the record, 0x%08x expected", h.Owner, owner)
	}
	if uint32(h.Kind) != g.Kind {
		return bad("WNR4 MX/NV: kind %d, expected %d", h.Kind, g.Kind)
	}
	if h.Flags&^(mxnvFlagCeil|mxnvFlagCZ) != 0 {
		return bad("WNR4 MX/NV: unknown flags 0x%02x", h.Flags)
	}
	if g.Kind == mxnvKindNV && h.Flags&mxnvFlagCeil != 0 {
		return bad("WNR4 NV: non-zero ceil flags 0x%02x", h.Flags)
	}
	cz := h.Flags&mxnvFlagCZ != 0
	if cz != valueExact {
		return bad("WNR4 MX/NV: CZ flag %v does not match VALUE_EXACT %v", cz, valueExact)
	}
	if rec[19] != 0 || !allZero(rec[28:64]) {
		return bad("WNR4 MX/NV: reserved header bytes not zero")
	}
	anchors := [3]byte{rec[16], rec[17], rec[18]}
	anchorsWord := uint32(anchors[0]) | uint32(anchors[1])<<8 | uint32(anchors[2])<<16
	w4 := g.Kind | uint32(h.Flags)<<8 | h.G<<16
	var ntiles uint64
	for m := 0; m < 3; m++ {
		ntiles += wnr4Tiles(cells[m], h.G)
	}
	lay, e := wnr4Cut(rec, h, ntiles, 0)
	if e != nil {
		return e
	}
	per := uint64(wnr4Lanes) * uint64(h.G)
	var t uint64
	for m := 0; m < 3; m++ {
		ceil := (h.Flags>>uint(m))&1 != 0
		so := out[soff[m] : soff[m]+scount[m]]
		wo := out[woff[m] : woff[m]+cells[m]*16]
		cols := g.Cols[m]
		for c0 := uint64(0); c0 < cells[m]; c0 += per {
			n := min(per, cells[m]-c0)
			ident := [wnr4Ident]uint32{model.ID, owner, uint32(t), 0, w4, anchorsWord}
			d, e := wnr4Open(lay.tile(t), ident)
			if e != nil {
				return fail(e.Class, "%s (tile %d)", e.Msg, t)
			}
			// Phase 1: scales
			ns, s0 := n*spc, c0*spc
			scaleSteps := ceilDiv(ns, wnr4Lanes)
			for k := uint64(0); k < scaleSteps; k++ {
				for l := 0; l < wnr4Lanes; l++ {
					idx := 32*k + uint64(l)
					if idx >= ns {
						break
					}
					i := s0 + idx
					sc, pred := mxnvScaleCtx(g.Kind, so, i, s0, cols)
					sym := byte(d.sym(l, model.sTable(m, sc)))
					y := sym + pred
					so[i] = y
					d.word(l, uint32(y))
				}
			}
			// Phase 2: cells
			rounds := ceilDiv(n, wnr4Lanes)
			for j := uint64(0); j < rounds; j++ {
				act := int(min(uint64(wnr4Lanes), n-32*j))
				var pm, mx, cls [wnr4Lanes]int
				var wd [wnr4Lanes][4]uint32
				for k := 0; k < 32; k++ {
					if k%bNib == 0 {
						for l := 0; l < act; l++ {
							scale := so[(c0+32*j+uint64(l))*spc+uint64(k/bNib)]
							cls[l] = mxnvScaleCls(scale, anchors[m], rLim)
							pm[l], mx[l] = 8, 8
						}
					}
					for l := 0; l < act; l++ {
						if k >= 28 && l >= 2 && j == (n-uint64(l)-1)/32 {
							continue
						}
						row := mxnvWRow(cls[l], pm[l], mx[l], model.PB, k%bNib, bNib)
						y := d.sym(l, model.wTable(m, row))
						if cz && y == 8 {
							return bad("WNR4 MX/NV: -0 nibble under CZ (tile %d)", t)
						}
						wd[l][k/8] |= y << uint((k%8)*4)
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
						tail, ok := d.tail(l)
						if !ok {
							return bad("WNR4 MX/NV: tail state out of range in lane %d (tile %d)", l, t)
						}
						for q := uint(0); q < 4; q++ {
							y := (tail >> (4 * q)) & 15
							if cz && y == 8 {
								return bad("WNR4 MX/NV: -0 tail nibble under CZ (tile %d)", t)
							}
							a := int(y & 7)
							if mx[l] == 8 || a > mx[l] {
								mx[l] = a
							}
						}
						wd[l][3] |= uint32(tail) << 16
					}
					for k := 0; k < 4; k++ {
						d.word(l, wd[l][k])
					}
					if ceil && mx[l] == 5 {
						if so[cell] == 0 {
							return bad("WNR4 MX: ceil underflow at cell %d (tile %d)", cell, t)
						}
						so[cell]--
						for k := 0; k < 4; k++ {
							wd[l][k] = mxnvCeilDbl8(wd[l][k])
						}
					}
					for k := 0; k < 4; k++ {
						le.PutUint32(wo[cell*16+uint64(4*k):], wd[l][k])
					}
				}
			}
			if e := d.finish(wnr4Owns(n)); e != nil {
				return fail(e.Class, "%s (tile %d)", e.Msg, t)
			}
			t++
		}
	}
	SeenWNR4.Records++
	return nil
}
