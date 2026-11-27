// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Whole-file verification: steps 1 to 3 of docs/WEX4-SPEC.md section 6, every blob and every record.

package main

import (
	"bytes"
	"crypto/sha256"
	"io"
)

// Result is what a file that was not refused looks like.
type Result struct {
	H *Header
	X *Index
	// Undecoded counts the records whose codec this reader does not know (section 10, rule 6) and the
	// codec-8 records that did not decode (section 7.3: such a record is undecodable, the file is not
	// refused for it): their SHA-256 was checked, their content was not decoded, so the file is not verified
	// in full. Why says, for each of them, "record N: " and the reason.
	Undecoded uint32
	Why       []string
	// Digest is the content digest of section 5. It is defined only when Undecoded is 0.
	Digest [32]byte
}

func readAt(f io.ReaderAt, off, n uint64) ([]byte, *Error) {
	b := make([]byte, n)
	if got, err := f.ReadAt(b, int64(off)); uint64(got) != n {
		return nil, fail(EIO, "read of %d bytes at offset %d: %v", n, off, err)
	}
	return b, nil
}

// Verify checks a whole file. pin, when not nil, is the root the file must have. visit, when not nil, is
// called with every decoded record (the buffer is reused: copy what must outlive the call).
func Verify(f io.ReaderAt, size uint64, pin []byte, visit func(i uint32, r Record, x *Index, decoded []byte)) (*Result, *Error) {
	page := make([]byte, pageBytes)
	n, _ := f.ReadAt(page, 0)
	h, e := ParseHeader(page[:n], size)
	if e != nil {
		return nil, e
	}
	if pin != nil && !bytes.Equal(pin, h.Root[:]) {
		return nil, fail(ERoot, "the root is %x", h.Root)
	}
	raw, e := readAt(f, h.IndexOffset, align(h.IndexBytes))
	if e != nil {
		return nil, e
	}
	x, e := OpenIndex(h, raw)
	if e != nil {
		return nil, e
	}

	// step 3, blobs: SHA-256, zero padding; a WNM2 model is opened and its id compared; a WNM4 model (kind
	// 3) named by a codec-8 record is kept and opened when the first such record is decoded (section 7.3)
	models := map[uint32]*Model{}
	wnr4 := newWNR4Container(x)
	for i := uint32(0); i < h.Blobs; i++ {
		b := x.Blob(i)
		data, e := readAt(f, b.Offset, align(b.Bytes))
		if e != nil {
			return nil, e
		}
		if sum := sha256.Sum256(data[:b.Bytes]); !bytes.Equal(sum[:], b.SHA) {
			return nil, fail(EHash, "blob %d: SHA-256 mismatch", i)
		}
		if !allZero(data[b.Bytes:]) {
			return nil, fail(EBlob, "blob %d: non-zero byte in its padding", i)
		}
		if b.Kind == blobKindWNM2 {
			m, e := OpenModel(data[:b.Bytes])
			if e != nil {
				return nil, e
			}
			if m.ID != b.ID {
				return nil, fail(EBlob, "blob %d: id 0x%08x in the index, the model's FNV-1a is 0x%08x", i, b.ID, m.ID)
			}
			models[i] = m
		}
		if b.Kind == blobKindWNM4 {
			if h.Required&reqWNR4Expert != 0 {
				if e := wnr4.validateStrict(i, b, data[:b.Bytes]); e != nil {
					return nil, e
				}
			} else {
				wnr4.keep(i, data[:b.Bytes])
			}
		}
	}

	// step 3, records: SHA-256 of the stored bytes, then the decoder. The SHA-256 is checked for every
	// record before it is decoded: for a codec-8 record that is the only check of the verbatim slices of
	// kind GG, which no tile hash covers (section 7.3, "Integrity")
	res := &Result{H: h, X: x}
	out := make([]byte, h.MaxDecoded)
	dig := newDigest()
	for i := uint32(0); i < h.Records; i++ {
		r := x.Record(i)
		rec, e := readAt(f, r.Offset, uint64(r.Stored))
		if e != nil {
			return nil, e
		}
		if sum := sha256.Sum256(rec); !bytes.Equal(sum[:], r.SHA) {
			return nil, fail(EHash, "record %d: SHA-256 mismatch", i)
		}
		dec := out[:r.Decoded]
		switch r.Codec {
		case codecRAW:
			if !allZero(rec[r.Decoded:]) {
				return nil, fail(EData, "record %d: non-zero byte in the padding of a RAW record", i)
			}
			dec = rec[:r.Decoded]
		case codecWBP1:
			e = decodeWBP1(rec, r.Owner, dec)
		case codecWNR3:
			g, ge := x.ExpertGeo(i, r) // passed in step 2
			if ge != nil {
				return nil, ge
			}
			e = decodeWNR3(rec, models[r.Blob-1], g, r.Owner, r.Flags&flagValueExact != 0, dec)
		case codecWNR4:
			if r.Flags&flagExpert != 0 {
				e = wnr4.decodeExpert(i, r, rec, dec)
			} else if we := wnr4.decode(r, rec, dec); we != nil {
				// section 7.3: the record is undecodable (fail closed), the file is not invalid
				res.Undecoded++
				res.Why = append(res.Why, "record "+utoa(i)+": "+we.Msg)
				continue
			}
		default:
			res.Undecoded++
			res.Why = append(res.Why, "record "+utoa(i)+": codec "+utoa(uint32(r.Codec))+", which this reader does not know")
			continue
		}
		if e != nil {
			e.Msg = "record " + utoa(i) + ": " + e.Msg
			return nil, e
		}
		dig.record(x, r, dec)
		if visit != nil {
			visit(i, r, x, dec)
		}
	}
	dig.zeroSize(x)
	res.Digest = dig.sum()
	return res, nil
}

func utoa(v uint32) string {
	if v == 0 {
		return "0"
	}
	var b [10]byte
	i := len(b)
	for ; v > 0; v /= 10 {
		i--
		b[i] = byte('0' + v%10)
	}
	return string(b[i:])
}
