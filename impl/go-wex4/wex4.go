// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// WEX4 header page and index: the validation of docs/WEX4-SPEC.md section 6, steps 1 and 2.
// Written from the specification only. The index is validated in place: no table is copied.

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/bits"
	"unicode/utf8"
)

const (
	pageBytes = 4096
	magicWEX4 = 0x57455834

	capBlobs   = 1 << 16
	capRecords = 1 << 24
	capTensors = 1 << 24
	capIndex   = 1 << 28
	capDecoded = 1 << 30
	capGrid    = 1 << 24

	blobEntry   = 64
	recordEntry = 72
	tensorEntry = 96

	codecRAW  = 0
	codecWNR3 = 5
	codecWBP1 = 6

	flagValueExact = 1
	flagExpert     = 2

	blobKindWNM2 = 1
	wnm2Bytes    = 59408
)

// The reject classes. The names are the ones the manifest of wex4/vectors uses for the C reader; the
// specification does not define them (SPEC-NOTES.md, note 1).
const (
	EMagic   = "WEX4_E_MAGIC"
	EHash    = "WEX4_E_HASH"
	EVersion = "WEX4_E_VERSION"
	EFeature = "WEX4_E_FEATURE"
	EHeader  = "WEX4_E_HEADER"
	EIndex   = "WEX4_E_INDEX"
	EName    = "WEX4_E_NAME"
	ETensor  = "WEX4_E_TENSOR"
	EBlob    = "WEX4_E_BLOB"
	ERecord  = "WEX4_E_RECORD"
	EExpert  = "WEX4_E_EXPERT"
	ELayout  = "WEX4_E_LAYOUT"
	EData    = "WEX4_E_DATA"
	EIO      = "IO"          // a read failed: not a property of the file's content
	ERoot    = "PINNED_ROOT" // the root is not the one given with -root
)

var le = binary.LittleEndian

// Error is a refusal: the class and what was found.
type Error struct {
	Class string
	Msg   string
}

func (e *Error) Error() string { return e.Class + ": " + e.Msg }

func fail(class, format string, a ...any) *Error {
	return &Error{Class: class, Msg: fmt.Sprintf(format, a...)}
}

func align(n uint64) uint64 { return (n + pageBytes - 1) &^ (pageBytes - 1) }

func allZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// Header is the header page (section 4).
type Header struct {
	Minor        uint16
	Required     uint64
	Optional     uint64
	FileBytes    uint64
	IndexOffset  uint64
	IndexBytes   uint64
	Blobs        uint32
	Records      uint32
	Tensors      uint32
	Layers       uint32
	Experts      uint32
	MaxDecoded   uint32
	NamesBytes   uint64
	MetaBytes    uint64
	TotalDecoded uint64
	IndexSHA     [32]byte
	Root         [32]byte
}

// ParseHeader is step 1. page holds what could be read at offset 0 (it may be short), realSize is the size
// of the file.
func ParseHeader(page []byte, realSize uint64) (*Header, *Error) {
	if len(page) < pageBytes {
		return nil, fail(EHeader, "the file has %d bytes, less than a header page", len(page))
	}
	page = page[:pageBytes]
	if m := le.Uint32(page[0:]); m != magicWEX4 {
		return nil, fail(EMagic, "magic 0x%08x is not WEX4", m)
	}
	root := sha256.Sum256(page[:4064])
	if !bytes.Equal(root[:], page[4064:4096]) {
		return nil, fail(EHash, "header_sha256 does not match bytes [0, 4064)")
	}
	if v := le.Uint16(page[4:]); v != 4 {
		return nil, fail(EVersion, "version_major %d", v)
	}
	h := &Header{
		Minor:        le.Uint16(page[6:]),
		Required:     le.Uint64(page[8:]),
		Optional:     le.Uint64(page[16:]),
		FileBytes:    le.Uint64(page[24:]),
		IndexOffset:  le.Uint64(page[32:]),
		IndexBytes:   le.Uint64(page[40:]),
		Blobs:        le.Uint32(page[48:]),
		Records:      le.Uint32(page[52:]),
		Tensors:      le.Uint32(page[56:]),
		Layers:       le.Uint32(page[60:]),
		Experts:      le.Uint32(page[64:]),
		MaxDecoded:   le.Uint32(page[68:]),
		NamesBytes:   le.Uint64(page[72:]),
		MetaBytes:    le.Uint64(page[80:]),
		TotalDecoded: le.Uint64(page[88:]),
		Root:         root,
	}
	copy(h.IndexSHA[:], page[96:128])
	if h.Required&^1 != 0 {
		return nil, fail(EFeature, "unknown required feature bits 0x%x", h.Required&^1)
	}
	if h.FileBytes != realSize {
		return nil, fail(EHeader, "file_bytes %d, the file has %d bytes", h.FileBytes, realSize)
	}
	if h.FileBytes%pageBytes != 0 {
		return nil, fail(EHeader, "file_bytes %d is not a multiple of 4096", h.FileBytes)
	}
	if h.Blobs > capBlobs || h.Records > capRecords || h.Tensors > capTensors {
		return nil, fail(EHeader, "a count is above its cap: %d blobs, %d records, %d tensors", h.Blobs, h.Records, h.Tensors)
	}
	if h.NamesBytes > capIndex || h.MetaBytes > capIndex {
		return nil, fail(EHeader, "names_bytes %d or meta_bytes %d is above 2^28", h.NamesBytes, h.MetaBytes)
	}
	if h.Layers > 65536 || h.Experts > 65536 || (h.Layers == 0) != (h.Experts == 0) ||
		uint64(h.Layers)*uint64(h.Experts) > capGrid {
		return nil, fail(EHeader, "expert grid %d x %d", h.Layers, h.Experts)
	}
	if h.MaxDecoded > capDecoded {
		return nil, fail(EHeader, "max_decoded %d is above 2^30", h.MaxDecoded)
	}
	// the counts and sizes are capped, so the sum is below 2^34
	sum := blobEntry*uint64(h.Blobs) + recordEntry*uint64(h.Records) + (tensorEntry+4)*uint64(h.Tensors) +
		h.NamesBytes + h.MetaBytes
	if h.IndexBytes != sum || h.IndexBytes > capIndex {
		return nil, fail(EHeader, "index_bytes %d, the tables need %d (cap 2^28)", h.IndexBytes, sum)
	}
	if h.IndexOffset%pageBytes != 0 || h.IndexOffset < pageBytes || h.IndexOffset > h.FileBytes ||
		align(h.IndexBytes) > h.FileBytes-h.IndexOffset {
		return nil, fail(EHeader, "index_offset %d with %d index bytes in a file of %d", h.IndexOffset, h.IndexBytes, h.FileBytes)
	}
	return h, nil
}

// Blob is a blob entry (section 5.1).
type Blob struct {
	Kind, ID uint32
	Offset   uint64
	Bytes    uint64
	SHA      []byte
	Reserved uint64
}

// Record is a record entry (section 5.2).
type Record struct {
	Offset      uint64
	Stored      uint32
	Decoded     uint32
	Codec       uint16
	Flags       uint16
	Blob        uint32
	Owner       uint32
	FirstTensor uint32
	TensorCount uint32
	SHA         []byte
	Reserved    uint32
}

// Tensor is a tensor entry (section 5.3).
type Tensor struct {
	NameOffset  uint32
	NameBytes   uint16
	Dtype       uint8
	Rank        uint8
	Shape       [8]uint64
	Record      uint32
	RecordCount uint32
	Offset      uint64
	NBytes      uint64
}

// Index is a validated index: the bytes and where its parts start.
type Index struct {
	H *Header
	B []byte // the index_bytes bytes
	// offsets of the record table, tensor table, name order, name pool and metadata
	rt, tt, no, np, md uint64
	DataTensors        uint32 // tensors with record_count != 0; the zero-size tensors follow them
	ExpertRecords      uint32
	UnknownCodec       uint32 // records that open and cannot be decoded (section 10, rule 6)
}

func (x *Index) Blob(i uint32) Blob {
	b := x.B[uint64(i)*blobEntry:]
	return Blob{le.Uint32(b[0:]), le.Uint32(b[4:]), le.Uint64(b[8:]), le.Uint64(b[16:]), b[24:56], le.Uint64(b[56:])}
}

func (x *Index) Record(i uint32) Record {
	b := x.B[x.rt+uint64(i)*recordEntry:]
	return Record{le.Uint64(b[0:]), le.Uint32(b[8:]), le.Uint32(b[12:]), le.Uint16(b[16:]), le.Uint16(b[18:]),
		le.Uint32(b[20:]), le.Uint32(b[24:]), le.Uint32(b[28:]), le.Uint32(b[32:]), b[36:68], le.Uint32(b[68:])}
}

func (x *Index) Tensor(i uint32) Tensor {
	b := x.B[x.tt+uint64(i)*tensorEntry:]
	t := Tensor{NameOffset: le.Uint32(b[0:]), NameBytes: le.Uint16(b[4:]), Dtype: b[6], Rank: b[7],
		Record: le.Uint32(b[72:]), RecordCount: le.Uint32(b[76:]), Offset: le.Uint64(b[80:]), NBytes: le.Uint64(b[88:])}
	for k := 0; k < 8; k++ {
		t.Shape[k] = le.Uint64(b[8+8*k:])
	}
	return t
}

// Name is valid only after the tensor pass of OpenIndex has checked the name ranges.
func (x *Index) Name(t Tensor) []byte {
	o := x.np + uint64(t.NameOffset)
	return x.B[o : o+uint64(t.NameBytes)]
}

func (x *Index) Metadata() []byte { return x.B[x.md:] }

// bits per element of the dtype codes 1..22 (section 5.4); 0 = opaque
var dtypeBits = [23]uint64{0, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 4, 6, 6, 8, 8}

// Geo is the geometry of an expert record: rows and columns of the three scale matrices.
type Geo struct {
	Rows, Cols [3]uint64
}

func (g Geo) scales(m int) uint64  { return g.Rows[m] * g.Cols[m] }
func (g Geo) nibbles(m int) uint64 { return g.Rows[m] * g.Cols[m] * 32 }

func ceilDiv(a, b uint64) uint64 { return (a + b - 1) / b }

// OpenIndex is step 2. raw holds the align(index_bytes) bytes read at index_offset.
func OpenIndex(h *Header, raw []byte) (*Index, *Error) {
	if uint64(len(raw)) != align(h.IndexBytes) {
		return nil, fail(EIO, "index buffer of %d bytes", len(raw))
	}
	b := raw[:h.IndexBytes]
	if sha256.Sum256(b) != h.IndexSHA {
		return nil, fail(EHash, "index_sha256 does not match the index")
	}
	if !allZero(raw[h.IndexBytes:]) {
		return nil, fail(EIndex, "non-zero byte in the padding after the index")
	}
	x := &Index{H: h, B: b}
	x.rt = blobEntry * uint64(h.Blobs)
	x.tt = x.rt + recordEntry*uint64(h.Records)
	x.no = x.tt + tensorEntry*uint64(h.Tensors)
	x.np = x.no + 4*uint64(h.Tensors)
	x.md = x.np + h.NamesBytes

	if e := x.checkTensors(); e != nil {
		return nil, e
	}
	if e := x.checkNameOrder(); e != nil {
		return nil, e
	}
	if e := x.checkBlobs(); e != nil {
		return nil, e
	}
	if e := x.checkRecords(); e != nil {
		return nil, e
	}
	if e := x.checkTiling(); e != nil {
		return nil, e
	}
	return x, nil
}

// checkTensors: every tensor entry by itself (section 5.3: names, rank, shape, size) and the order of the
// table (data tensors, then zero-size tensors).
func (x *Index) checkTensors() *Error {
	h := x.H
	var nameOff uint64
	zeroSeen := false
	for i := uint32(0); i < h.Tensors; i++ {
		t := x.Tensor(i)
		if uint64(t.NameOffset) != nameOff {
			return fail(EName, "tensor %d: name_offset %d, the names before it have %d bytes", i, t.NameOffset, nameOff)
		}
		if uint64(t.NameBytes) > h.NamesBytes-nameOff {
			return fail(EName, "tensor %d: its name ends after the name pool", i)
		}
		name := x.B[x.np+nameOff : x.np+nameOff+uint64(t.NameBytes)]
		if !utf8.Valid(name) || bytes.IndexByte(name, 0) >= 0 {
			return fail(EName, "tensor %d: its name is not valid UTF-8 or holds a NUL byte", i)
		}
		nameOff += uint64(t.NameBytes)

		if t.Rank > 8 {
			return fail(ETensor, "tensor %d: rank %d", i, t.Rank)
		}
		prod, zeroDim := uint64(1), false
		for k := 0; k < 8; k++ {
			if k >= int(t.Rank) {
				if t.Shape[k] != 0 {
					return fail(ETensor, "tensor %d: shape[%d] is not 0 at rank %d", i, k, t.Rank)
				}
				continue
			}
			if t.Shape[k] == 0 {
				zeroDim = true
			}
			hi, lo := bits.Mul64(prod, t.Shape[k])
			if hi != 0 {
				// a later 0 dimension would bring the product back to 0: the rule is on every multiplication
				return fail(ETensor, "tensor %d: the product of its shape overflows 64 bits", i)
			}
			prod = lo
		}
		if int(t.Dtype) < len(dtypeBits) && dtypeBits[t.Dtype] != 0 {
			hi, elemBits := bits.Mul64(prod, dtypeBits[t.Dtype])
			if hi != 0 || t.NBytes > ^uint64(0)/8 || t.NBytes*8 != elemBits {
				return fail(ETensor, "tensor %d: %d bytes for %d elements of dtype %d", i, t.NBytes, prod, t.Dtype)
			}
		}
		if t.RecordCount == 0 {
			if t.NBytes != 0 || t.Record != 0 || t.Offset != 0 || !zeroDim {
				return fail(ETensor, "tensor %d: record_count 0 on a tensor that is not a zero-size tensor with record 0 and offset 0", i)
			}
			zeroSeen = true
			continue
		}
		if zeroSeen {
			return fail(ETensor, "tensor %d: a data tensor after a zero-size tensor", i)
		}
		if t.NBytes == 0 {
			return fail(ETensor, "tensor %d: 0 bytes inside a record", i)
		}
		if t.Record >= h.Records || t.RecordCount > h.Records-t.Record {
			return fail(ETensor, "tensor %d: records [%d, +%d) of %d", i, t.Record, t.RecordCount, h.Records)
		}
		if t.RecordCount != 1 && t.Offset != 0 {
			return fail(ETensor, "tensor %d: a split tensor with offset %d", i, t.Offset)
		}
		x.DataTensors++
	}
	if nameOff != h.NamesBytes {
		return fail(EName, "the names have %d bytes, names_bytes is %d", nameOff, h.NamesBytes)
	}
	return nil
}

// checkNameOrder: the name-order table lists tensors in strictly increasing name order (bytewise, a prefix
// before the longer name). Strictly increasing names of in-range indices are a permutation with unique names.
func (x *Index) checkNameOrder() *Error {
	var prev []byte
	for k := uint32(0); k < x.H.Tensors; k++ {
		i := le.Uint32(x.B[x.no+4*uint64(k):])
		if i >= x.H.Tensors {
			return fail(EName, "name order entry %d: tensor %d of %d", k, i, x.H.Tensors)
		}
		name := x.Name(x.Tensor(i))
		if k > 0 && bytes.Compare(prev, name) >= 0 {
			return fail(EName, "name order entry %d: not after the name before it", k)
		}
		prev = name
	}
	return nil
}

func (x *Index) checkBlobs() *Error {
	for i := uint32(0); i < x.H.Blobs; i++ {
		b := x.Blob(i)
		if b.Bytes < 1 || b.Bytes > capDecoded {
			return fail(EBlob, "blob %d: %d bytes", i, b.Bytes)
		}
		if b.Reserved != 0 {
			return fail(EBlob, "blob %d: reserved field 0x%x", i, b.Reserved)
		}
		if b.Kind == blobKindWNM2 && (b.Bytes != wnm2Bytes || b.ID == 0) {
			return fail(EBlob, "blob %d: a WNM2 model of %d bytes with id 0x%08x", i, b.Bytes, b.ID)
		}
	}
	return nil
}

// checkRecordEntry: the rules of section 5.2 that need the entry alone (and the blob table).
func (x *Index) checkRecordEntry(i uint32, r Record) *Error {
	if r.Stored == 0 || r.Stored%pageBytes != 0 {
		return fail(ERecord, "record %d: stored_bytes %d", i, r.Stored)
	}
	if r.Decoded < 1 || r.Decoded > capDecoded {
		return fail(ERecord, "record %d: decoded_bytes %d", i, r.Decoded)
	}
	raw := align(uint64(r.Decoded))
	if uint64(r.Stored) > raw {
		return fail(ERecord, "record %d: stored_bytes %d above the raw size %d", i, r.Stored, raw)
	}
	if r.Flags&^(flagValueExact|flagExpert) != 0 {
		return fail(ERecord, "record %d: unknown flag bits 0x%x", i, r.Flags)
	}
	if r.Reserved != 0 {
		return fail(ERecord, "record %d: reserved field 0x%x", i, r.Reserved)
	}
	if r.Blob > x.H.Blobs {
		return fail(ERecord, "record %d: blob %d of %d", i, r.Blob, x.H.Blobs)
	}
	expert, exact := r.Flags&flagExpert != 0, r.Flags&flagValueExact != 0
	if r.Codec == codecRAW {
		if uint64(r.Stored) != raw || r.Blob != 0 {
			return fail(ERecord, "record %d: a RAW record of %d stored bytes (raw %d) with blob %d", i, r.Stored, raw, r.Blob)
		}
	} else if uint64(r.Stored) >= raw {
		return fail(ERecord, "record %d: codec %d and not smaller than raw", i, r.Codec)
	}
	switch r.Codec {
	case codecRAW:
	case codecWNR3:
		if !expert {
			return fail(ERecord, "record %d: a WNR3 record without the EXPERT flag", i)
		}
		if r.Blob == 0 || x.Blob(r.Blob-1).Kind != blobKindWNM2 {
			return fail(ERecord, "record %d: a WNR3 record without a WNM2 model blob", i)
		}
	case codecWBP1:
		if r.Blob != 0 || r.Flags != 0 {
			return fail(ERecord, "record %d: a WBP1 record with blob %d and flags 0x%x", i, r.Blob, r.Flags)
		}
		if uint64(r.Stored) < align(16+6*ceilDiv(uint64(r.Decoded), 32768)) {
			return fail(ERecord, "record %d: %d stored bytes cannot hold a WBP1 record of %d decoded bytes", i, r.Stored, r.Decoded)
		}
	default:
		if expert || exact {
			return fail(ERecord, "record %d: codec %d on an EXPERT or VALUE_EXACT record", i, r.Codec)
		}
	}
	if exact && (r.Codec != codecWNR3 || x.H.Required&1 == 0) {
		return fail(ERecord, "record %d: VALUE_EXACT on codec %d, required features 0x%x", i, r.Codec, x.H.Required)
	}
	if !expert && r.Owner != i {
		return fail(ERecord, "record %d: owner %d", i, r.Owner)
	}
	return nil
}

// ExpertGeo checks the six tensors of an EXPERT record (section 5.5) and returns its geometry.
func (x *Index) ExpertGeo(i uint32, r Record) (Geo, *Error) {
	var g Geo
	if r.TensorCount != 6 {
		return g, fail(EExpert, "record %d: an EXPERT record with %d tensors", i, r.TensorCount)
	}
	for m := 0; m < 3; m++ {
		s := x.Tensor(r.FirstTensor + uint32(m))
		w := x.Tensor(r.FirstTensor + 3 + uint32(m))
		if s.Rank != 2 || w.Rank != 2 || s.RecordCount != 1 || w.RecordCount != 1 {
			return g, fail(EExpert, "record %d: matrix %d is not two rank-2 tensors inside the record", i, m)
		}
		if (s.Dtype != 2 && s.Dtype != 6) || (w.Dtype != 2 && w.Dtype != 3) {
			return g, fail(EExpert, "record %d: matrix %d has scale dtype %d and weight dtype %d", i, m, s.Dtype, w.Dtype)
		}
		rows, cols := s.Shape[0], s.Shape[1]
		if rows < 1 || rows > 65535 || cols < 1 || cols > 65535 || w.Shape[0] != rows || w.Shape[1] != 16*cols || 16*cols > 65535 {
			return g, fail(EExpert, "record %d: matrix %d has scales %d x %d and weights %d x %d", i, m, rows, cols, w.Shape[0], w.Shape[1])
		}
		g.Rows[m], g.Cols[m] = rows, cols
	}
	return g, nil
}

// checkRecords walks the record table and the tensor table together: every record entry, the tensors that
// tile it, split tensors and their continuation parts, expert records, and the totals of the header.
func (x *Index) checkRecords() *Error {
	h := x.H
	var cur uint32 // the next tensor
	var maxDec uint32
	var total uint64
	lastOwner := int64(-1)
	for i := uint32(0); i < h.Records; i++ {
		r := x.Record(i)
		if e := x.checkRecordEntry(i, r); e != nil {
			return e
		}
		maxDec, total = max(maxDec, r.Decoded), total+uint64(r.Decoded)
		if r.TensorCount == 0 {
			return fail(ERecord, "record %d: a continuation part that no split tensor claims", i)
		}
		if r.FirstTensor != cur {
			return fail(ERecord, "record %d: first_tensor %d, the next tensor is %d", i, r.FirstTensor, cur)
		}
		if r.TensorCount > x.DataTensors-cur {
			return fail(ERecord, "record %d: %d tensors, %d are left", i, r.TensorCount, x.DataTensors-cur)
		}
		first := x.Tensor(cur)
		parts := uint32(1)
		if first.RecordCount >= 2 {
			// a split tensor: the only tensor of this record, then its continuation parts
			if first.Record != i {
				return fail(ETensor, "tensor %d: in record %d, its record field says %d", cur, i, first.Record)
			}
			if r.TensorCount != 1 {
				return fail(ERecord, "record %d: a split tensor shares its first record", i)
			}
			parts = first.RecordCount // record + record_count <= records was checked with the tensor
			sum := uint64(r.Decoded)
			for p := uint32(1); p < parts; p++ {
				c := x.Record(i + p)
				if e := x.checkRecordEntry(i+p, c); e != nil {
					return e
				}
				if c.TensorCount != 0 || c.FirstTensor != cur {
					return fail(ERecord, "record %d: not a continuation part of tensor %d", i+p, cur)
				}
				if c.Flags&flagExpert != 0 {
					return fail(EExpert, "record %d: an EXPERT record that is a continuation part", i+p)
				}
				maxDec, total, sum = max(maxDec, c.Decoded), total+uint64(c.Decoded), sum+uint64(c.Decoded)
			}
			if first.NBytes != sum {
				return fail(ERecord, "tensor %d: %d bytes, its %d parts decode to %d", cur, first.NBytes, parts, sum)
			}
		} else {
			var off uint64
			for j := uint32(0); j < r.TensorCount; j++ {
				t := x.Tensor(cur + j)
				if t.Record != i || t.RecordCount != 1 {
					return fail(ETensor, "tensor %d: in record %d, its entry says record %d, record_count %d", cur+j, i, t.Record, t.RecordCount)
				}
				if t.Offset != off || t.NBytes > uint64(r.Decoded)-off {
					return fail(ETensor, "tensor %d: bytes [%d, +%d) of record %d, expected offset %d of %d", cur+j, t.Offset, t.NBytes, i, off, r.Decoded)
				}
				off += t.NBytes
			}
			if off != uint64(r.Decoded) {
				return fail(ETensor, "record %d: its tensors cover %d of %d bytes", i, off, r.Decoded)
			}
		}
		if r.Flags&flagExpert != 0 {
			layer, expert := r.Owner>>16, r.Owner&0xffff
			if layer >= h.Layers || expert >= h.Experts {
				return fail(EExpert, "record %d: owner layer %d expert %d outside the grid %d x %d", i, layer, expert, h.Layers, h.Experts)
			}
			if int64(r.Owner) <= lastOwner {
				return fail(EExpert, "record %d: owner 0x%08x is not above the expert record before it", i, r.Owner)
			}
			lastOwner = int64(r.Owner)
			g, e := x.ExpertGeo(i, r)
			if e != nil {
				return e
			}
			if r.Codec == codecWNR3 {
				// the smallest record: a 32-byte header and 6 bytes per stream at the longest chunk, 16384
				var streams uint64
				for m := 0; m < 3; m++ {
					streams += ceilDiv(g.scales(m), 16384) + ceilDiv(g.nibbles(m), 16384)
				}
				if uint64(r.Stored) < align(32+6*streams) {
					return fail(ERecord, "record %d: %d stored bytes cannot hold a WNR3 record of %d streams", i, r.Stored, streams)
				}
			}
			x.ExpertRecords++
		}
		if r.Codec != codecRAW && r.Codec != codecWNR3 && r.Codec != codecWBP1 {
			x.UnknownCodec++
		}
		cur += r.TensorCount
		i += parts - 1
	}
	if cur != x.DataTensors {
		return fail(ETensor, "%d data tensors, the records hold %d", x.DataTensors, cur)
	}
	if maxDec != h.MaxDecoded {
		return fail(ERecord, "max_decoded %d, the largest record decodes to %d", h.MaxDecoded, maxDec)
	}
	if total != h.TotalDecoded {
		return fail(ERecord, "total_decoded %d, the records decode to %d", h.TotalDecoded, total)
	}
	if h.Layers != 0 && x.ExpertRecords == 0 {
		return fail(EExpert, "an expert grid %d x %d and no expert record", h.Layers, h.Experts)
	}
	return nil
}

// checkTiling: the header page, the blobs in table order, the records in table order and the index cover
// [0, file_bytes) exactly once (section 3). Every extent has at least one byte, so at each position at most
// one of the three candidates can be the next extent.
func (x *Index) checkTiling() *Error {
	h := x.H
	pos := uint64(pageBytes)
	var bi, ri uint32
	indexDone := h.IndexBytes == 0
	for bi < h.Blobs || ri < h.Records || !indexDone {
		var size uint64
		switch {
		case bi < h.Blobs && x.Blob(bi).Offset == pos:
			size = align(x.Blob(bi).Bytes)
			bi++
		case ri < h.Records && x.Record(ri).Offset == pos:
			size = uint64(x.Record(ri).Stored)
			ri++
		case !indexDone && h.IndexOffset == pos:
			size = align(h.IndexBytes)
			indexDone = true
		default:
			return fail(ELayout, "no extent starts at offset %d (next blob %d, next record %d)", pos, bi, ri)
		}
		if size > h.FileBytes-pos {
			return fail(ELayout, "the extent at offset %d ends after the file", pos)
		}
		pos += size
	}
	if pos != h.FileBytes {
		return fail(ELayout, "the extents end at %d, file_bytes is %d", pos, h.FileBytes)
	}
	return nil
}
