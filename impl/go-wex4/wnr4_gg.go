// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// WNR4 kind 2, GG: the 18-byte blocks of the Q2_0 and IQ4_NL slices of a GGUF routed-expert record in WNR4
// tiles, every other slice verbatim (docs/WNR4-SPEC.md section 5), and its WNM4 model (5.2). Decoder only,
// CPU, one thread, on the core of wnr4.go. Verify calls it through wnr4_wex4.go.

package main

const (
	ggKind       = 2
	ggTypeQ2_0   = 42 // subtype 0
	ggTypeIQ4NL  = 20 // subtype 1
	ggBlock      = 18 // bytes per block: f16 scale (low byte, high byte), then 16 code bytes
	ggClasses    = 32
	ggModelBytes = wnr4ModelHdr + 2*ggSubBytes
	ggSubBytes   = 256 + 2*wnr4CDF256 + ggClasses*2*wnr4CDF256 + ggClasses*2*wnr4CDF16 // 18,306
	ggMaxSlices  = 4
	ggMaxBlocks  = 1 << 27
)

// GGModel is a WNM4 blob of kind 2: per subtype a class map, a high-byte table, 32 low-byte tables and 32
// nibble tables.
type GGModel struct {
	ID   uint32
	cmap [2][256]byte
	high [2][]uint16 // 257 entries
	low  [2][]uint16 // 32 x 257
	nib  [2][]uint16 // 32 x 17
}

// OpenGGModel checks a WNM4 kind-2 blob: size, header, class maps below 32, every table.
func OpenGGModel(b []byte) (*GGModel, *Error) {
	if e := wnr4ModelHeader(b, ggModelBytes, [8]uint32{magicWNM4, 1, ggKind, cdfBits, ggClasses, 2, 0, 0}); e != nil {
		return nil, e
	}
	m := &GGModel{ID: fnv1a(b)}
	p := wnr4ModelHdr
	var e *Error
	for u := 0; u < 2; u++ {
		for v := 0; v < 256; v++ {
			if c := b[p+v]; c >= ggClasses {
				return nil, fail(EBlob, "WNM4 kind 2: subtype %d maps high byte %d to class %d", u, v, c)
			}
			m.cmap[u][v] = b[p+v]
		}
		p += 256
		if m.high[u], e = wnr4Tables(b, &p, 1, wnr4CDF256, "GG high-byte"); e != nil {
			return nil, e
		}
		if m.low[u], e = wnr4Tables(b, &p, ggClasses, wnr4CDF256, "GG low-byte"); e != nil {
			return nil, e
		}
		if m.nib[u], e = wnr4Tables(b, &p, ggClasses, wnr4CDF16, "GG nibble"); e != nil {
			return nil, e
		}
	}
	return m, nil
}

// GGSlice is one slice of a GG record's geometry: its ggml type and its size in bytes.
type GGSlice struct {
	Type  uint32
	Bytes uint64
}

func ggSubtype(typ uint32) int {
	switch typ {
	case ggTypeQ2_0:
		return 0
	case ggTypeIQ4NL:
		return 1
	}
	return -1
}

// ggGeometry checks a geometry (WNR4-SPEC 5.1) and returns the decoded size and the verbatim bytes.
func ggGeometry(sl []GGSlice) (total, verbatim uint64, e *Error) {
	if len(sl) < 1 || len(sl) > ggMaxSlices {
		return 0, 0, bad("WNR4 GG: %d slices", len(sl))
	}
	for i, s := range sl {
		if s.Bytes < 1 {
			return 0, 0, bad("WNR4 GG: slice %d is empty", i)
		}
		if ggSubtype(s.Type) >= 0 {
			if s.Bytes%ggBlock != 0 || s.Bytes/ggBlock >= ggMaxBlocks {
				return 0, 0, bad("WNR4 GG: coded slice %d of %d bytes", i, s.Bytes)
			}
		} else {
			verbatim += s.Bytes
		}
		total += s.Bytes
	}
	if total > 1<<32-1 || verbatim > 1<<31-1 {
		return 0, 0, bad("WNR4 GG: slices of %d bytes, %d of them verbatim", total, verbatim)
	}
	return total, verbatim, nil
}

// decodeGG decodes a kind-2 record into out (the sum of the slices). On an error out holds no usable data.
func decodeGG(rec []byte, model *GGModel, sl []GGSlice, owner uint32, out []byte) *Error {
	total, verbatim, e := ggGeometry(sl)
	if e != nil {
		return e
	}
	if uint64(len(out)) != total {
		return bad("WNR4 GG: the slices have %d bytes, the record decodes to %d", total, len(out))
	}
	h, e := wnr4ParseHeader(rec)
	if e != nil {
		return e
	}
	if h.Model != model.ID {
		return bad("WNR4 GG: model id 0x%08x in the record, the blob is 0x%08x", h.Model, model.ID)
	}
	if h.Owner != owner {
		return bad("WNR4 GG: owner 0x%08x in the record, 0x%08x expected", h.Owner, owner)
	}
	if h.Kind != ggKind || h.Flags != 0 {
		return bad("WNR4 GG: kind %d flags 0x%02x", h.Kind, h.Flags)
	}
	if !allZero(rec[16:20]) || !allZero(rec[28:64]) {
		return bad("WNR4 GG: header bytes 16..19 or 28..63 not zero")
	}
	var ntiles uint64
	for _, s := range sl {
		if ggSubtype(s.Type) >= 0 {
			ntiles += wnr4Tiles(s.Bytes/ggBlock, h.G)
		}
	}
	V := (verbatim + 3) &^ 3
	lay, e := wnr4Cut(rec, h, ntiles, V)
	if e != nil {
		return e
	}
	if !allZero(lay.area[verbatim:]) {
		return bad("WNR4 GG: padding of the verbatim area not zero")
	}
	// the verbatim slices, then the tiles of the coded ones
	var at, va uint64
	for _, s := range sl {
		if ggSubtype(s.Type) < 0 {
			copy(out[at:at+s.Bytes], lay.area[va:va+s.Bytes])
			va += s.Bytes
			SeenWNR4.GGVerbatim++
		}
		at += s.Bytes
	}
	at = 0
	t := uint64(0)
	per := uint64(wnr4Lanes) * uint64(h.G)
	for _, s := range sl {
		u := ggSubtype(s.Type)
		if u >= 0 {
			blocks := s.Bytes / ggBlock
			for c0 := uint64(0); c0 < blocks; c0 += per {
				n := min(per, blocks-c0)
				ident := [wnr4Ident]uint32{model.ID, owner, uint32(t), 0, ggKind | h.G<<16, s.Type}
				if e := ggTile(lay.tile(t), model, u, ident, n, out[at+c0*ggBlock:at+(c0+n)*ggBlock]); e != nil {
					return fail(e.Class, "%s (tile %d)", e.Msg, t)
				}
				t++
			}
			SeenWNR4.GGCoded++
		}
		at += s.Bytes
	}
	SeenWNR4.Records++
	return nil
}

// ggTile decodes one tile of n blocks of subtype u into dst (n x 18 bytes): phases 1 to 3 of WNR4-SPEC 5.4,
// the tail in the state, the hash and end states of 5.5.
func ggTile(tile []byte, model *GGModel, u int, ident [wnr4Ident]uint32, n uint64, dst []byte) *Error {
	d, e := wnr4Open(tile, ident)
	if e != nil {
		return e
	}
	steps := ceilDiv(n, wnr4Lanes) // steps of phases 1 and 2, rounds of phase 3
	high := model.high[u]
	for k := uint64(0); k < steps; k++ { // phase 1, high bytes
		for l := 0; l < wnr4Lanes; l++ {
			i := 32*k + uint64(l)
			if i >= n {
				break
			}
			y := d.sym(l, high)
			dst[i*ggBlock+1] = byte(y)
			d.word(l, y)
		}
	}
	for k := uint64(0); k < steps; k++ { // phase 2, low bytes
		for l := 0; l < wnr4Lanes; l++ {
			i := 32*k + uint64(l)
			if i >= n {
				break
			}
			c := int(model.cmap[u][dst[i*ggBlock+1]])
			y := d.sym(l, model.low[u][c*wnr4CDF256:(c+1)*wnr4CDF256])
			dst[i*ggBlock] = byte(y)
			d.word(l, y)
		}
	}
	for j := uint64(0); j < steps; j++ { // phase 3, code bytes
		for k := 0; k < 32; k++ {
			for l := 0; l < wnr4Lanes; l++ {
				i := 32*j + uint64(l)
				if i >= n {
					break
				}
				if k >= 28 && l >= 2 && j == (n-uint64(l)-1)/32 {
					continue // the tail: in the lane's final state
				}
				b := dst[i*ggBlock:]
				c := int(model.cmap[u][b[1]])
				y := d.sym(l, model.nib[u][c*wnr4CDF16:(c+1)*wnr4CDF16])
				if k&1 == 0 {
					b[2+k/2] = byte(y)
				} else {
					b[2+k/2] |= byte(y) << 4
				}
			}
		}
		for l := 0; l < wnr4Lanes; l++ {
			i := 32*j + uint64(l)
			if i >= n {
				break
			}
			if l >= 2 && j == (n-uint64(l)-1)/32 {
				continue // hashed once its tail is known
			}
			d.bytes(l, dst[i*ggBlock+2:(i+1)*ggBlock])
		}
	}
	for l := 2; l < wnr4Lanes && uint64(l) < n; l++ {
		v, _ := d.tail(l) // range checked by finish
		i := uint64(l) + 32*((n-uint64(l)-1)/32)
		b := dst[i*ggBlock:]
		b[16], b[17] = byte(v), byte(v>>8)
		d.bytes(l, b[2:ggBlock])
	}
	return d.finish(wnr4Owns(n))
}
