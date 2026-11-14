// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Codec 5, WNR3 (docs/WNR3-SPEC.md sections 3 and 4), and the WNM2 model it needs
// (docs/WNR2-REFERENCE.md section 7). Decoder only, CPU, one thread.

package main

const (
	magicWNR3    = 0x33524E57
	magicWNM2    = 0x324d4e57
	wnmContexts  = 567
	wnr3FlagCZ   = 0x08
	wnr3FlagCeil = 0x07
)

// Model is a WNM2 blob: 3 x 567 weight CDFs of 16 symbols and 3 scale CDFs of 256 symbols.
type Model struct {
	ID     uint32
	weight [3][wnmContexts][17]uint16
	scale  [3][257]uint16
	slots  [3][]byte
}

// fnv1a is the model id: FNV-1a 32 over the whole blob, 0 mapped to 1.
func fnv1a(b []byte) uint32 {
	h := uint32(2166136261)
	for _, c := range b {
		h = (h ^ uint32(c)) * 16777619
	}
	if h == 0 {
		h = 1
	}
	return h
}

// OpenModel checks a WNM2 blob: size, header, every CDF.
func OpenModel(b []byte) (*Model, *Error) {
	if len(b) != wnm2Bytes {
		return nil, fail(EBlob, "WNM2: %d bytes", len(b))
	}
	want := [8]uint32{magicWNM2, 1, cdfBits, wnmContexts, 3, 0, 0, 0}
	for i, w := range want {
		if v := le.Uint32(b[4*i:]); v != w {
			return nil, fail(EBlob, "WNM2: header word %d is %d, expected %d", i, v, w)
		}
	}
	m := &Model{ID: fnv1a(b)}
	p := 32
	for k := 0; k < 3; k++ {
		for c := 0; c < wnmContexts; c++ {
			cdf := m.weight[k][c][:]
			for i := range cdf {
				cdf[i] = le.Uint16(b[p:])
				p += 2
			}
			if !validCDF(cdf) {
				return nil, fail(EBlob, "WNM2: weight CDF of matrix %d context %d", k, c)
			}
		}
	}
	for k := 0; k < 3; k++ {
		cdf := m.scale[k][:]
		for i := range cdf {
			cdf[i] = le.Uint16(b[p:])
			p += 2
		}
		if !validCDF(cdf) {
			return nil, fail(EBlob, "WNM2: scale CDF of matrix %d", k)
		}
		m.slots[k] = slotTable(cdf)
	}
	return m, nil
}

func pow2In(v, lo, hi uint32) bool { return v >= lo && v <= hi && v&(v-1) == 0 }

// D of section 4.2: the inverse of the halving of an even block
var ceilD = [8]byte{0, 2, 4, 5, 6, 7, 6, 7} // only 0..5 are reached

// decodeWNR3 decodes the stored bytes of a WNR3 record into out: the slab of the geometry g (scales of
// w1, w2, w3, then packed FP4 weights of w1, w2, w3). owner and exact are the index entry's owner and
// VALUE_EXACT flag; model is the entry's blob. out is scratch: on an error it holds no usable data.
func decodeWNR3(rec []byte, model *Model, g Geo, owner uint32, exact bool, out []byte) *Error {
	if len(rec) < 32 {
		return bad("WNR3: record of %d bytes", len(rec))
	}
	if m := le.Uint32(rec[0:]); m != magicWNR3 {
		return bad("WNR3: magic 0x%08x", m)
	}
	modelID := le.Uint32(rec[4:])
	if modelID == 0 || modelID != model.ID {
		return bad("WNR3: model id 0x%08x in the record, the blob is 0x%08x", modelID, model.ID)
	}
	sp, wp := uint64(le.Uint32(rec[8:])), uint64(le.Uint32(rec[12:]))
	cw, cs := le.Uint32(rec[16:]), le.Uint32(rec[20:])
	if !pow2In(cw, 512, 16384) || !pow2In(cs, 256, 16384) {
		return bad("WNR3: chunks %d:%d", cw, cs)
	}
	anchor, flags := rec[24:27], rec[27]
	if flags&0xF0 != 0 {
		return bad("WNR3: flags 0x%02x", flags)
	}
	if o := le.Uint32(rec[28:]); o != owner {
		return bad("WNR3: owner 0x%08x in the record, 0x%08x in the index", o, owner)
	}
	cz := flags&wnr3FlagCZ != 0
	if cz != exact {
		return bad("WNR3: CZ bit %v in the record, VALUE_EXACT %v in the index", cz, exact)
	}

	// geometry: where each matrix sits in the slab and how many streams it has
	var scaleAt, weightAt [3]uint64
	var nScale, nWeight [3]uint64 // streams per matrix
	var slab, streams uint64
	for m := 0; m < 3; m++ {
		scaleAt[m] = slab
		slab += g.scales(m)
		nScale[m] = ceilDiv(g.scales(m), uint64(cs))
		nWeight[m] = ceilDiv(g.nibbles(m), uint64(cw))
		streams += nScale[m] + nWeight[m]
	}
	for m := 0; m < 3; m++ {
		weightAt[m] = slab
		slab += g.nibbles(m) / 2
	}
	if slab != uint64(len(out)) {
		return bad("WNR3: the slab has %d bytes, the record decodes to %d", slab, len(out))
	}
	total := uint64(len(rec))
	if 32+2*streams > total {
		return bad("WNR3: %d streams do not fit in %d bytes", streams, total)
	}
	lengths := rec[32 : 32+2*streams]
	var sum [2]uint64 // scale payload, weight payload
	var seen uint64
	for m := 0; m < 6; m++ {
		count := nScale[m%3]
		if m >= 3 {
			count = nWeight[m%3]
		}
		for j := uint64(0); j < count; j++ {
			l := uint64(le.Uint16(lengths[2*seen:]))
			if l < minStream {
				return bad("WNR3: stream %d has %d bytes", seen, l)
			}
			sum[m/3] += l
			seen++
		}
	}
	if sum[0] != sp || sum[1] != wp {
		return bad("WNR3: stream lengths %d + %d, payload sizes %d + %d", sum[0], sum[1], sp, wp)
	}
	used := 32 + 2*streams + sp + wp
	if align(used) != total || !allZero(rec[used:]) {
		return bad("WNR3: %d bytes used in a record of %d, or non-zero padding", used, total)
	}

	ident := [4]uint32{modelID, owner, 0, le.Uint32(rec[24:])}
	pos := 32 + 2*streams
	next := func() []byte { // the bytes of stream ident[2]
		l := uint64(le.Uint16(lengths[2*ident[2]:]))
		b := rec[pos : pos+l]
		pos += l
		return b
	}

	// scale streams: order 0, the model's scale CDF of the matrix
	for m := 0; m < 3; m++ {
		cdf, slot := &model.scale[m], model.slots[m]
		scales := out[scaleAt[m] : scaleAt[m]+g.scales(m)]
		for j := uint64(0); j < nScale[m]; j++ {
			part := scales[j*uint64(cs) : min((j+1)*uint64(cs), uint64(len(scales)))]
			r, ok := ransOpen(next())
			if !ok {
				return bad("WNR3: initial state of scale stream %d", ident[2])
			}
			for k := range part {
				s := int(slot[r.slot()]) // an int: s + 1 is 256 for symbol 255
				part[k] = byte(s)
				r.advance(uint32(cdf[s]), uint32(cdf[s+1]-cdf[s]))
			}
			if !r.done(ident, part) {
				return bad("WNR3: scale stream %d does not end at its length in the state of its hash", ident[2])
			}
			Seen.WNR3Scale++
			ident[2]++
		}
	}

	// weight streams: 32 nibbles per block, the context from the block's coded scale and the nibbles before
	for m := 0; m < 3; m++ {
		scales := out[scaleAt[m] : scaleAt[m]+g.scales(m)]
		weights := out[weightAt[m] : weightAt[m]+g.nibbles(m)/2]
		for j := uint64(0); j < nWeight[m]; j++ {
			first := j * uint64(cw) / 2 // in bytes; a chunk is a whole number of 16-byte blocks
			part := weights[first:min(first+uint64(cw)/2, uint64(len(weights)))]
			r, ok := ransOpen(next())
			if !ok {
				return bad("WNR3: initial state of weight stream %d", ident[2])
			}
			for b := 0; b < len(part); b += 16 {
				srel := int(scales[(first+uint64(b))/16]) - int(anchor[m])
				srel = min(max(srel, -3), 3) + 3
				pm, mx := 8, 8 // 8 = none yet
				for k := 0; k < 32; k++ {
					cdf := &model.weight[m][(srel*9+pm)*9+mx]
					slot := uint16(r.slot())
					s := 0
					for cdf[s+1] <= slot {
						s++
					}
					r.advance(uint32(cdf[s]), uint32(cdf[s+1]-cdf[s]))
					if cz && s == 8 {
						return bad("WNR3: a -0 nibble in a canonical-zero record, stream %d", ident[2])
					}
					if k&1 == 0 {
						part[b+k/2] = byte(s)
					} else {
						part[b+k/2] |= byte(s) << 4
					}
					pm = s & 7
					if mx == 8 || pm > mx {
						mx = pm
					}
				}
			}
			if !r.done(ident, part) {
				return bad("WNR3: weight stream %d does not end at its length in the state of its hash", ident[2])
			}
			Seen.WNR3Weight++
			ident[2]++
		}
	}

	// inverse of the ceil form, after every stream and hash: blocks with coded maximum 5
	for m := 0; m < 3; m++ {
		if flags>>m&1 == 0 {
			continue
		}
		Seen.WNR3CeilMatrices++
		scales := out[scaleAt[m] : scaleAt[m]+g.scales(m)]
		weights := out[weightAt[m] : weightAt[m]+g.nibbles(m)/2]
		for b := range scales {
			block := weights[16*b : 16*b+16]
			var mx byte
			for _, c := range block {
				mx = max(mx, c&7, c>>4&7)
			}
			if mx != 5 {
				continue
			}
			if scales[b] == 0 {
				return bad("WNR3: a ceil-form block with coded scale 0, matrix %d block %d", m, b)
			}
			scales[b]--
			Seen.WNR3CeilBlocks++
			for k, c := range block {
				block[k] = c&0x88 | ceilD[c&7] | ceilD[c>>4&7]<<4
			}
		}
	}
	return nil
}
