// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Codec 8 inside a WEX4 file (docs/WEX4-SPEC.md section 7.3, "At decode"): the WNM4 blob of a record, its
// kind, its id and the record's model id and owner, and the geometry of a GG or I4 record taken from its
// tensors. A failure of any of these checks, or of the decoder, makes that record undecodable: the file is
// not refused for it (7.3, "a failure makes that record undecodable (fail closed), not the file invalid"),
// and Verify reports it as it reports a record of an unknown codec (exit code 3, no content digest).

package main

const (
	dtypeU8  = 2
	dtypeF32 = 13
	// the opaque WEX4 dtypes of the coded GG slices: 128 + the ggml type (7.3)
	dtypeGGQ2_0  = 128 + ggTypeQ2_0  // 170
	dtypeGGIQ4NL = 128 + ggTypeIQ4NL // 148
)

// ggmlType is the type a GG slice of WEX4 dtype d has in the record's geometry (SPEC-NOTES F6): 42 for
// dtype 170, 20 for dtype 148, and for every other dtype a type that is neither, so that a tensor of WEX4
// dtype 42 or 20 is a verbatim slice.
func ggmlType(d uint8) uint32 {
	switch d {
	case dtypeGGQ2_0:
		return ggTypeQ2_0
	case dtypeGGIQ4NL:
		return ggTypeIQ4NL
	}
	return 0x100 | uint32(d)
}

// wnr4GGSlices is the geometry of a GG record (7.3): tensor_count 1 to 4, each tensor inside this record
// only; slice s is the record's s-th tensor.
func (x *Index) wnr4GGSlices(r Record) ([]GGSlice, *Error) {
	if r.TensorCount < 1 || r.TensorCount > ggMaxSlices {
		return nil, bad("WNR4 GG: a record of %d tensors, 1 to 4 allowed", r.TensorCount)
	}
	sl := make([]GGSlice, r.TensorCount)
	for s := range sl {
		t := x.Tensor(r.FirstTensor + uint32(s))
		if t.RecordCount != 1 {
			return nil, bad("WNR4 GG: tensor %d is split over %d records", r.FirstTensor+uint32(s), t.RecordCount)
		}
		sl[s] = GGSlice{Type: ggmlType(t.Dtype), Bytes: t.NBytes}
	}
	return sl, nil
}

// wnr4I4Geo is the geometry of an I4 record (7.3): tensor_count 2, 4 or 6, each tensor inside this record
// only and of rank 2; with n = tensor_count / 2, tensor m < n is U8 [rows, cols / 2] and tensor n + m is F32
// [rows, cols / 64], cols a multiple of 64. The tiling of section 5.3 already puts each tensor where the one
// before it ends. The packing is record byte 13: 1 is pairs, anything else is read as planar and refused
// by the decoder unless it is 0 (WNR4-SPEC 6.5, the flags).
func (x *Index) wnr4I4Geo(r Record, packing byte) (I4Geo, *Error) {
	var g I4Geo
	if r.TensorCount != 2 && r.TensorCount != 4 && r.TensorCount != 6 {
		return g, bad("WNR4 I4: a record of %d tensors, 2, 4 or 6 allowed", r.TensorCount)
	}
	n := r.TensorCount / 2
	for m := uint32(0); m < n; m++ {
		ci, si := r.FirstTensor+m, r.FirstTensor+n+m
		c, s := x.Tensor(ci), x.Tensor(si)
		if c.RecordCount != 1 || s.RecordCount != 1 || c.Rank != 2 || s.Rank != 2 {
			return g, bad("WNR4 I4: tensors %d and %d are not two tensors of rank 2 inside the record", ci, si)
		}
		if c.Dtype != dtypeU8 || s.Dtype != dtypeF32 {
			return g, bad("WNR4 I4: matrix %d has codes of dtype %d and scales of dtype %d (U8 and F32 expected)", m, c.Dtype, s.Dtype)
		}
		rows, half := c.Shape[0], c.Shape[1]
		if half > 1<<32 || s.Shape[0] != rows || s.Shape[1] > 1<<32 || 2*half != 64*s.Shape[1] || half%32 != 0 {
			return g, bad("WNR4 I4: matrix %d has codes [%d, %d] and scales [%d, %d]", m, rows, half, s.Shape[0], s.Shape[1])
		}
		g.Rows = append(g.Rows, rows)
		g.Cols = append(g.Cols, 2*half)
	}
	g.Pairs = packing == 1
	return g, nil
}

// wnr4Container holds what the codec-8 records of one file share: the bytes of the kind-3 blobs they name
// (checked in step 3: SHA-256 and zero padding) and the models opened from them, each opened once, when a
// record first uses it, as the kind that record needs.
type wnr4Container struct {
	x     *Index
	named map[uint32]bool   // the blobs (0-based) a codec-8 record names
	blobs map[uint32][]byte // their bytes, kept by step 3 (0-based)
	mxnv  map[uint32]*MXNVModel
	gg    map[uint32]*GGModel
	i4    map[uint32]*I4Model
	fails map[[2]uint32]*Error // (blob, kind): the blob did not open as a model of that kind
}

func newWNR4Container(x *Index) *wnr4Container {
	c := &wnr4Container{x: x, named: map[uint32]bool{}, blobs: map[uint32][]byte{},
		mxnv: map[uint32]*MXNVModel{}, gg: map[uint32]*GGModel{}, i4: map[uint32]*I4Model{},
		fails: map[[2]uint32]*Error{}}
	for k := uint32(0); k < x.H.Records; k++ {
		if r := x.Record(k); r.Codec == codecWNR4 && r.Blob != 0 && x.Blob(r.Blob-1).Kind == blobKindWNM4 {
			c.named[r.Blob-1] = true
		}
	}
	return c
}

// validateStrict checks a kind-3 (WNM4) blob in step 3 under REQ_WNR4_EXPERT: every WNM4 blob (used or
// unused) must parse cleanly and match its entry id, else the container is refused with EBlob.
func (c *wnr4Container) validateStrict(i uint32, b Blob, data []byte) *Error {
	if len(data) < wnr4ModelHdr {
		return fail(EBlob, "blob %d: WNM4 shorter than header", i)
	}
	kind := le.Uint32(data[8:])
	var id uint32
	switch kind {
	case mxnvKindMX, mxnvKindNV:
		m, e := OpenMXNVModel(data)
		if e != nil {
			return fail(EBlob, "blob %d: %s", i, e.Msg)
		}
		id = m.ID
		c.mxnv[i+1] = m
	case ggKind:
		m, e := OpenGGModel(data)
		if e != nil {
			return fail(EBlob, "blob %d: %s", i, e.Msg)
		}
		id = m.ID
		c.gg[i+1] = m
	case i4Kind:
		m, e := OpenI4Model(data)
		if e != nil {
			return fail(EBlob, "blob %d: %s", i, e.Msg)
		}
		id = m.ID
		c.i4[i+1] = m
	default:
		return fail(EBlob, "blob %d: unknown WNM4 kind %d", i, kind)
	}
	if id != b.ID {
		return fail(EBlob, "blob %d: id 0x%08x in the index, the WNM4 model's FNV-1a is 0x%08x", i, b.ID, id)
	}
	return nil
}

// decodeExpert decodes an EXPERT codec-8 record (MX or NV) under REQ_WNR4_EXPERT; any error rejects the file.
func (c *wnr4Container) decodeExpert(i uint32, r Record, rec, out []byte) *Error {
	g, e := c.x.ExpertGeo(i, r)
	if e != nil {
		return e
	}
	m := c.mxnv[r.Blob]
	if m == nil {
		return bad("EXPERT codec 8 without an opened MX/NV WNM4 model")
	}
	return decodeMXNV(rec, m, g, r.Owner, r.Flags&flagValueExact != 0, out)
}

// keep holds on to the bytes of blob i (checked by step 3) when a codec-8 record names it. An unused
// kind-3 blob, or one that only records of other codecs name, is not opened (WEX4-SPEC 5.1: a WNM4 blob is
// checked when a record first uses it).
func (c *wnr4Container) keep(i uint32, data []byte) {
	if c.named[i] {
		c.blobs[i] = data
	}
}

// model opens blob b (1-based, as in the record entry) as a WNM4 model of the record's kind and checks its
// id against the blob entry.
func (c *wnr4Container) model(b uint32, kind byte) (gg *GGModel, i4 *I4Model, e *Error) {
	if b == 0 {
		return nil, nil, bad("codec 8 without a WNM4 blob")
	}
	be := c.x.Blob(b - 1)
	if be.Kind != blobKindWNM4 {
		return nil, nil, bad("codec 8 with blob %d of kind %d, not a WNM4 model (kind 3)", b-1, be.Kind)
	}
	key := [2]uint32{b, uint32(kind)}
	if e := c.fails[key]; e != nil {
		return nil, nil, e
	}
	var id uint32
	switch kind {
	case ggKind:
		if gg = c.gg[b]; gg == nil {
			if gg, e = OpenGGModel(c.blobs[b-1]); e == nil {
				c.gg[b] = gg
			}
		}
		if gg != nil {
			id = gg.ID
		}
	case i4Kind:
		if i4 = c.i4[b]; i4 == nil {
			if i4, e = OpenI4Model(c.blobs[b-1]); e == nil {
				c.i4[b] = i4
			}
		}
		if i4 != nil {
			id = i4.ID
		}
	}
	if e == nil && id != be.ID {
		e = bad("a WNM4 model of id 0x%08x in the index, the model's FNV-1a is 0x%08x", be.ID, id)
	}
	if e != nil {
		e = fail(EData, "blob %d: %s", b-1, e.Msg)
		c.fails[key] = e
		return nil, nil, e
	}
	return gg, i4, nil
}

// decode checks a codec-8 record as 7.3 says at decode and decodes it into out (decoded_bytes long). rec
// holds its stored bytes, whose SHA-256 the caller has checked. An error means the record is undecodable.
func (c *wnr4Container) decode(r Record, rec, out []byte) *Error {
	kind := rec[12] // stored_bytes is at least 4096: the 64-byte header is there
	if kind != ggKind && kind != i4Kind {
		return bad("WNR4 kind %d: a WEX4 record that is not EXPERT may be of kind 2 (GG) or 3 (I4) only", kind)
	}
	gg, i4, e := c.model(r.Blob, kind)
	if e != nil {
		return e
	}
	if kind == ggKind {
		sl, e := c.x.wnr4GGSlices(r)
		if e != nil {
			return e
		}
		return decodeGG(rec, gg, sl, r.Owner, out)
	}
	g, e := c.x.wnr4I4Geo(r, rec[13])
	if e != nil {
		return e
	}
	return decodeI4(rec, i4, g, r.Owner, out)
}
