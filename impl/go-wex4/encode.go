// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// The encoders of the writer: the rANS stream writer, the count quantiser, codec 6 (WBP1, docs/WEX4-SPEC.md
// section 7.2, "Canonical encoder") and codec 5 (WNR3, docs/WNR3-SPEC.md sections 3 and 4 with errata 9 to
// 11). Written from those pages and docs/WNR2-REFERENCE.md section 7 only. Every encoder returns the record
// padded to a multiple of 4096; the caller stores the record RAW when that is not smaller than raw.

package main

import "sort"

// sym is one coded symbol: the start and the width of its interval in a 14-bit CDF.
type sym struct{ start, freq uint16 }

// ransEncode writes one stream: the symbols are coded last first from the state x0, then the 4 bytes of the
// final state go in front, big-endian (WNR3-SPEC erratum 10; the bound of the renormalisation is
// x_max = ((L >> 14) << 8) * freq of WNR2-REFERENCE section 7). A decoder reads the bytes in order and ends
// in x0.
func ransEncode(dst []byte, syms []sym, x0 uint32) []byte {
	var tail []byte // renormalisation bytes, last written first read
	x := x0
	for k := len(syms) - 1; k >= 0; k-- {
		f := uint32(syms[k].freq)
		for xmax := ((ransL >> cdfBits) << 8) * f; x >= xmax; x >>= 8 {
			tail = append(tail, byte(x))
		}
		x = (x/f)<<cdfBits + x%f + uint32(syms[k].start)
	}
	dst = append(dst, byte(x>>24), byte(x>>16), byte(x>>8), byte(x))
	for k := len(tail) - 1; k >= 0; k-- {
		dst = append(dst, tail[k])
	}
	return dst
}

// quantize turns the counts of n symbols (16 or 256) into a CDF of n + 1 values that ends at 2^14
// (WNR2-REFERENCE section 7, wex_nr2_quantize): every symbol gets 1 + floor(count * (16384 - n) / total),
// and the slots that are left go one each to the largest remainders, the lower symbol first on a tie.
// "Empty counts give a near-uniform CDF": with total 0 every symbol gets 16384 / n (SPEC-NOTES.md, E8).
func quantize(counts []uint64, total uint64) []uint16 {
	n := uint64(len(counts))
	freq := make([]uint64, n)
	rem := make([]uint64, n)
	left := uint64(cdfTotal)
	for s, c := range counts {
		if total == 0 {
			freq[s] = cdfTotal / n
		} else {
			freq[s] = 1 + c*(cdfTotal-n)/total
			rem[s] = c * (cdfTotal - n) % total
		}
		left -= freq[s]
	}
	order := make([]int, n)
	for s := range order {
		order[s] = s
	}
	sort.SliceStable(order, func(a, b int) bool { return rem[order[a]] > rem[order[b]] })
	for k := uint64(0); k < left; k++ { // left < n: the remainders add up to left * total
		freq[order[k]]++
	}
	cdf := make([]uint16, n+1)
	var acc uint64
	for s := range freq {
		cdf[s] = uint16(acc)
		acc += freq[s]
	}
	cdf[n] = cdfTotal
	return cdf
}

// encodeWBP1 is the canonical encoder of section 7.2: given the bytes and (planes, chunkLog2, transform) the
// record is fully determined. The caller has checked the parameters (transform 1 needs planes > 1 and a
// whole number of elements).
func encodeWBP1(data []byte, owner uint32, planes uint64, chunkLog2 uint, transform byte) []byte {
	n := uint64(len(data))
	if transform == 1 {
		// every little-endian element is rotated left by one bit
		rot := make([]byte, n)
		for o := uint64(0); o < n; o += planes {
			carry := data[o+planes-1] >> 7
			for k := uint64(0); k < planes; k++ {
				c := data[o+k]
				rot[o+k] = c<<1 | carry
				carry = c >> 7
			}
		}
		data = rot
	}
	chunk := uint64(1) << chunkLog2
	ident := [4]uint32{magicWBP1, owner, 0, uint32(planes) | uint32(chunkLog2)<<8 | uint32(transform)<<24}
	var mask byte
	var cdfs, tables, payloads []byte
	var syms []sym
	for p := uint64(0); p < planes; p++ {
		var np uint64
		if n > p {
			np = ceilDiv(n-p, planes)
		}
		plane := make([]byte, np)
		var counts [256]uint64
		for k := range plane {
			plane[k] = data[uint64(k)*planes+p]
			counts[plane[k]]++
		}
		streams := ceilDiv(np, chunk)
		// both forms of the plane; the smaller one is kept
		var cdf []uint16
		var lengths, coded, hashes []byte
		if np > 0 {
			cdf = quantize(counts[:], np)
		}
		for j := uint64(0); j < streams; j++ {
			part := plane[j*chunk : min((j+1)*chunk, np)]
			h := h32(ident, part)
			hashes = le.AppendUint32(hashes, h)
			syms = syms[:0]
			for _, c := range part {
				syms = append(syms, sym{cdf[c], cdf[int(c)+1] - cdf[c]})
			}
			before := len(coded)
			coded = ransEncode(coded, syms, ransL+h&hashMask)
			lengths = le.AppendUint16(lengths, uint16(len(coded)-before))
			ident[2]++
		}
		if np > 0 && 514+2*streams+uint64(len(coded)) < 4*streams+np {
			mask |= 1 << p
			for _, c := range cdf {
				cdfs = le.AppendUint16(cdfs, c)
			}
			tables = append(tables, lengths...)
			payloads = append(payloads, coded...)
		} else {
			tables = append(tables, hashes...)
			payloads = append(payloads, plane...)
		}
	}
	rec := make([]byte, 16, align(16+uint64(len(cdfs)+len(tables)+len(payloads))))
	le.PutUint32(rec[0:], magicWBP1)
	le.PutUint32(rec[4:], owner)
	le.PutUint32(rec[8:], uint32(n))
	rec[12], rec[13], rec[14], rec[15] = byte(planes), byte(chunkLog2), mask, transform
	rec = append(append(append(rec, cdfs...), tables...), payloads...)
	return rec[:cap(rec)]
}

// H of WNR3-SPEC section 4.2: the halving of the magnitudes of an even block (only 0, 2, 4, 5, 6, 7 occur)
var ceilH = [8]byte{0, 0, 1, 0, 2, 3, 4, 5}

// coded is an expert slab in the coded domain of WNR3-SPEC section 1: steps 1 to 3 of the encoder order of
// section 4.6.
type coded struct {
	decoded           []byte // what the record decodes to: the slab, or Z(slab) with canonical zero
	slab              []byte // the coded scales and nibbles, in the layout of the slab
	scaleAt, weightAt [3]uint64
	flags             byte
	anchor            [3]byte
}

func (c *coded) scales(g Geo, m int) []byte { return c.slab[c.scaleAt[m] : c.scaleAt[m]+g.scales(m)] }
func (c *coded) weights(g Geo, m int) []byte {
	return c.slab[c.weightAt[m] : c.weightAt[m]+g.nibbles(m)/2]
}

// weightSymbols calls f for every weight nibble of matrix m, in order, with its context (section 4.1).
func (c *coded) weightSymbols(g Geo, m int, f func(ctx, s int)) {
	scales, weights := c.scales(g, m), c.weights(g, m)
	for b := range scales {
		srel := min(max(int(scales[b])-int(c.anchor[m]), -3), 3) + 3
		pm, mx := 8, 8 // 8 = none yet
		for k := 0; k < 32; k++ {
			s := int(weights[16*b+k/2] >> (4 * (k & 1)) & 15)
			f((srel*9+pm)*9+mx, s)
			pm = s & 7
			if mx == 8 || pm > mx {
				mx = pm
			}
		}
	}
}

// codedForm: canonical zero (with cz), the gate and the transform T of each matrix, and the rule-g anchors
// on the coded scales.
func codedForm(slab []byte, g Geo, cz bool) *coded {
	c := &coded{decoded: append([]byte(nil), slab...)}
	var at uint64
	for m := 0; m < 3; m++ {
		c.scaleAt[m] = at
		at += g.scales(m)
	}
	for m := 0; m < 3; m++ {
		c.weightAt[m] = at
		at += g.nibbles(m) / 2
	}
	if cz {
		copy(c.decoded[c.weightAt[0]:], zeroNibbles(c.decoded[c.weightAt[0]:at]))
		c.flags |= wnr3FlagCZ
	}
	c.slab = append([]byte(nil), c.decoded...)
	for m := 0; m < 3; m++ {
		scales, weights := c.scales(g, m), c.weights(g, m)
		blockMax := func(b int) (mx byte, odd bool) {
			for _, q := range weights[16*b : 16*b+16] {
				lo, hi := q&7, q>>4&7
				mx = max(mx, lo, hi)
				odd = odd || lo == 1 || lo == 3 || hi == 1 || hi == 3
			}
			return
		}
		// the gate: ceil form iff no block of the matrix has maximum 5
		ceil := true
		for b := range scales {
			if mx, _ := blockMax(b); mx == 5 {
				ceil = false
				break
			}
		}
		if ceil {
			c.flags |= 1 << m
			for b := range scales {
				if mx, odd := blockMax(b); mx == 7 && !odd && scales[b] <= 254 {
					block := weights[16*b : 16*b+16]
					for k, q := range block {
						block[k] = q&0x88 | ceilH[q&7] | ceilH[q>>4&7]<<4
					}
					scales[b]++
				}
			}
		}
		// rule g on the coded scales: the first v in 0..254 with the largest h[v] + h[v+1]
		var h [256]uint64
		for _, s := range scales {
			h[s]++
		}
		best := h[0] + h[1]
		for v := 1; v < 255; v++ {
			if h[v]+h[v+1] > best {
				best, c.anchor[m] = h[v]+h[v+1], byte(v)
			}
		}
	}
	return c
}

// zeroNibbles is Z of WNR3-SPEC section 1 on packed FP4 bytes: every nibble 0x8 becomes 0x0.
func zeroNibbles(b []byte) []byte {
	out := make([]byte, len(b))
	for k, c := range b {
		if c&0x0f == 0x08 {
			c &= 0xf0
		}
		if c&0xf0 == 0x80 {
			c &= 0x0f
		}
		out[k] = c
	}
	return out
}

// encodeWNR3 encodes the slab of one expert (scales of w1, w2, w3, then packed FP4 weights of w1, w2, w3)
// as a WNR3 record with the chunks cw (weight nibbles per stream) and cs (scales per stream). With cz every
// -0 nibble is stored as +0 first. The second result is what the record decodes to: the slab, or Z(slab).
func encodeWNR3(slab []byte, model *Model, g Geo, owner uint32, cz bool, cw, cs uint32) (rec, decoded []byte) {
	c := codedForm(slab, g, cz)
	ident := [4]uint32{model.ID, owner, 0, uint32(c.anchor[0]) | uint32(c.anchor[1])<<8 | uint32(c.anchor[2])<<16 | uint32(c.flags)<<24}
	var lengths, scalePayload, weightPayload []byte
	var syms []sym
	for m := 0; m < 3; m++ {
		cdf := &model.scale[m]
		scales := c.scales(g, m)
		for first := uint64(0); first < uint64(len(scales)); first += uint64(cs) {
			part := scales[first:min(first+uint64(cs), uint64(len(scales)))]
			syms = syms[:0]
			for _, s := range part {
				syms = append(syms, sym{cdf[s], cdf[int(s)+1] - cdf[s]})
			}
			before := len(scalePayload)
			scalePayload = ransEncode(scalePayload, syms, ransL+h32(ident, part)&hashMask)
			lengths = le.AppendUint16(lengths, uint16(len(scalePayload)-before))
			ident[2]++
		}
	}
	for m := 0; m < 3; m++ {
		// the forward pass for the contexts, then one stream per cw nibbles
		var all []sym
		c.weightSymbols(g, m, func(ctx, s int) {
			cdf := &model.weight[m][ctx]
			all = append(all, sym{cdf[s], cdf[s+1] - cdf[s]})
		})
		weights := c.weights(g, m)
		for first := uint64(0); first < uint64(len(weights)); first += uint64(cw) / 2 {
			part := weights[first:min(first+uint64(cw)/2, uint64(len(weights)))]
			before := len(weightPayload)
			weightPayload = ransEncode(weightPayload, all[2*first:2*first+2*uint64(len(part))], ransL+h32(ident, part)&hashMask)
			lengths = le.AppendUint16(lengths, uint16(len(weightPayload)-before))
			ident[2]++
		}
	}
	rec = make([]byte, 32, align(32+uint64(len(lengths)+len(scalePayload)+len(weightPayload))))
	le.PutUint32(rec[0:], magicWNR3)
	le.PutUint32(rec[4:], model.ID)
	le.PutUint32(rec[8:], uint32(len(scalePayload)))
	le.PutUint32(rec[12:], uint32(len(weightPayload)))
	le.PutUint32(rec[16:], cw)
	le.PutUint32(rec[20:], cs)
	le.PutUint32(rec[24:], ident[3])
	le.PutUint32(rec[28:], owner)
	rec = append(append(append(rec, lengths...), scalePayload...), weightPayload...)
	return rec[:cap(rec)], c.decoded
}
