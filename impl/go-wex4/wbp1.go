// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Codec 6, WBP1: byte-plane rANS (docs/WEX4-SPEC.md section 7.2). Decoder only.

package main

const magicWBP1 = 0x31504257

func bad(format string, a ...any) *Error { return fail(EData, format, a...) }

// decodeWBP1 decodes the stored bytes of a WBP1 record into out (decoded_bytes long). On an error out
// holds no usable data.
func decodeWBP1(rec []byte, owner uint32, out []byte) *Error {
	n := uint64(len(out))
	if len(rec) < 16 {
		return bad("WBP1: record of %d bytes", len(rec))
	}
	if m := le.Uint32(rec[0:]); m != magicWBP1 {
		return bad("WBP1: magic 0x%08x", m)
	}
	if o := le.Uint32(rec[4:]); o != owner {
		return bad("WBP1: owner %d in the record, %d in the index", o, owner)
	}
	if d := le.Uint32(rec[8:]); uint64(d) != n {
		return bad("WBP1: decoded_bytes %d in the record, %d in the index", d, n)
	}
	planes, chunkLog2, mask, transform := uint64(rec[12]), uint(rec[13]), rec[14], rec[15]
	if planes != 1 && planes != 2 && planes != 4 && planes != 8 {
		return bad("WBP1: %d planes", planes)
	}
	if chunkLog2 < 10 || chunkLog2 > 15 {
		return bad("WBP1: chunk_log2 %d", chunkLog2)
	}
	if uint64(mask)>>planes != 0 {
		return bad("WBP1: mask 0x%02x with %d planes", mask, planes)
	}
	if transform > 1 || (transform == 1 && (planes == 1 || n%planes != 0)) {
		return bad("WBP1: transform %d with %d planes and %d bytes", transform, planes, n)
	}
	chunk := uint64(1) << chunkLog2
	total := uint64(len(rec))
	pos := uint64(16)
	// take returns the next size bytes of the record, or nil when the record is too short
	take := func(size uint64) []byte {
		if size > total-pos {
			return nil
		}
		b := rec[pos : pos+size]
		pos += size
		return b
	}

	var planeBytes, streams [8]uint64
	var slots [8][]byte
	var cdfs [8][]uint16
	for p := uint64(0); p < planes; p++ {
		if n > p {
			planeBytes[p] = ceilDiv(n-p, planes)
		}
		streams[p] = ceilDiv(planeBytes[p], chunk)
		if mask>>p&1 == 0 {
			continue
		}
		b := take(514)
		if b == nil {
			return bad("WBP1: the record ends inside the CDF of plane %d", p)
		}
		cdf := make([]uint16, 257)
		for i := range cdf {
			cdf[i] = le.Uint16(b[2*i:])
		}
		if !validCDF(cdf) {
			return bad("WBP1: the CDF of plane %d is not a strictly increasing 0..16384", p)
		}
		cdfs[p], slots[p] = cdf, slotTable(cdf)
	}
	var tables [8][]byte
	for p := uint64(0); p < planes; p++ {
		width := uint64(4) // raw plane: u32 chunk hashes
		if mask>>p&1 != 0 {
			width = 2 // rANS plane: u16 stream lengths
		}
		if tables[p] = take(width * streams[p]); tables[p] == nil {
			return bad("WBP1: the record ends inside the stream table of plane %d", p)
		}
	}

	ident := [4]uint32{magicWBP1, owner, 0, uint32(planes) | uint32(chunkLog2)<<8 | uint32(transform)<<24}
	plane := make([]byte, planeBytes[0]) // plane 0 is the longest
	for p := uint64(0); p < planes; p++ {
		pl := plane[:planeBytes[p]]
		for j := uint64(0); j < streams[p]; j++ {
			part := pl[j*chunk : min((j+1)*chunk, planeBytes[p])]
			if mask>>p&1 == 0 {
				src := take(uint64(len(part)))
				if src == nil {
					return bad("WBP1: the record ends inside raw chunk %d", ident[2])
				}
				if h32(ident, src) != le.Uint32(tables[p][4*j:]) {
					return bad("WBP1: hash of raw chunk %d", ident[2])
				}
				copy(part, src)
				Seen.WBP1Raw++
			} else {
				length := uint64(le.Uint16(tables[p][2*j:]))
				src := take(length)
				if length < minStream || src == nil {
					return bad("WBP1: stream %d has %d bytes", ident[2], length)
				}
				r, ok := ransOpen(src)
				if !ok {
					return bad("WBP1: initial state of stream %d", ident[2])
				}
				cdf, slot := cdfs[p], slots[p]
				for k := range part {
					s := int(slot[r.slot()]) // an int: s + 1 is 256 for symbol 255
					part[k] = byte(s)
					r.advance(uint32(cdf[s]), uint32(cdf[s+1]-cdf[s]))
				}
				if !r.done(ident, part) {
					return bad("WBP1: stream %d does not end at its length in the state of its hash", ident[2])
				}
				Seen.WBP1Rans++
			}
			ident[2]++
		}
		for k, c := range pl {
			out[uint64(k)*planes+p] = c
		}
	}
	if !allZero(rec[pos:]) {
		return bad("WBP1: non-zero byte in the padding of the record")
	}
	if n%planes != 0 {
		Seen.WBP1Ragged++
	}
	if transform == 1 {
		Seen.WBP1Rotated++
		// every little-endian element was rotated left by one bit: rotate right
		for o := uint64(0); o < n; o += planes {
			e := out[o : o+planes]
			carry := e[0] & 1
			for k := planes; k > 0; k-- {
				c := e[k-1]
				e[k-1] = c>>1 | carry<<7
				carry = c & 1
			}
		}
	}
	return nil
}
