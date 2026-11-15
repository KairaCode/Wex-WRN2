// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// The writer: a WEX4 file from a manifest and its source files. Written from docs/WEX4-SPEC.md sections 3
// to 7 only. One record is in memory at a time. The file is written under a temporary name, read back with
// the reader of this folder, and gets its name only when that reader accepts it and finds the content
// digest the writer computed from the sources.

package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"math/bits"
	"os"
	"sort"
	"unicode/utf8"
)

// Written is what the writer reports about a file.
type Written struct {
	Root, Digest                [32]byte
	FileBytes                   uint64
	Records, RAW, WBP1, WNR3    int
	Fallbacks                   int // records asked as WBP1 or WNR3 and stored RAW, because not smaller
	Blobs, Tensors, EmptyTensor int
}

// checkTensor: the rules of section 5.3 that one tensor must satisfy by itself.
func checkTensor(t *srcTensor) error {
	if len(t.name) > 65535 || !utf8.Valid(t.name) || bytes.IndexByte(t.name, 0) >= 0 {
		return fmt.Errorf("line %d: the name is longer than 65535 bytes, not valid UTF-8 or holds a NUL byte", t.line)
	}
	prod := uint64(1)
	for k := 0; k < int(t.rank); k++ {
		hi, lo := bits.Mul64(prod, t.shape[k])
		if hi != 0 {
			return fmt.Errorf("line %d: the product of the shape overflows 64 bits", t.line)
		}
		prod = lo
	}
	if int(t.dtype) < len(dtypeBits) && dtypeBits[t.dtype] != 0 {
		hi, elemBits := bits.Mul64(prod, dtypeBits[t.dtype])
		if hi != 0 || t.nbytes > ^uint64(0)/8 || t.nbytes*8 != elemBits {
			return fmt.Errorf("line %d: %d bytes for %d elements of %s", t.line, t.nbytes, prod, dtypeNames[t.dtype])
		}
	}
	return nil
}

// geoOf: the geometry of an expert record from its six tensors (section 5.5).
func geoOf(r *srcRecord) (Geo, error) {
	var g Geo
	if len(r.tensors) != 6 {
		return g, fmt.Errorf("line %d: an expert record has 6 tensors, this one has %d", r.line, len(r.tensors))
	}
	for m := 0; m < 3; m++ {
		s, w := &r.tensors[m], &r.tensors[3+m]
		rows, cols := s.shape[0], s.shape[1]
		if s.rank != 2 || w.rank != 2 || (s.dtype != 2 && s.dtype != 6) || (w.dtype != 2 && w.dtype != 3) ||
			rows < 1 || rows > 65535 || cols < 1 || 16*cols > 65535 || w.shape[0] != rows || w.shape[1] != 16*cols {
			return g, fmt.Errorf("line %d: matrix %d is not a scale tensor [rows, cols] (U8 or F8_E8M0) and a weight tensor [rows, 16 * cols] (U8 or I8)", r.line, m)
		}
		g.Rows[m], g.Cols[m] = rows, cols
	}
	return g, nil
}

func padded(b []byte) []byte {
	out := make([]byte, align(uint64(len(b))))
	copy(out, b)
	return out
}

// WriteFile writes the file the manifest describes to out.
func WriteFile(m *Manifest, out string) (*Written, error) {
	w := &Written{Records: len(m.records), Blobs: len(m.blobs), EmptyTensor: len(m.empties)}

	// the tensor table: the tensors of the records in order, then the tensors of 0 bytes
	var tensors []*srcTensor
	firstTensor := make([]int, len(m.records))
	for i := range m.records {
		r := &m.records[i]
		firstTensor[i] = len(tensors)
		if r.part {
			if len(r.tensors) != 0 || i == 0 {
				return nil, fmt.Errorf("line %d: a part record has no tensor of its own and follows the record of a split tensor", r.line)
			}
			firstTensor[i] = firstTensor[i-1]
			continue
		}
		if len(r.tensors) == 0 {
			return nil, fmt.Errorf("line %d: a record without a tensor", r.line)
		}
		for k := range r.tensors {
			t := &r.tensors[k]
			if err := checkTensor(t); err != nil {
				return nil, err
			}
			if t.nbytes == 0 {
				return nil, fmt.Errorf("line %d: a tensor inside a record has at least 1 byte; a tensor of 0 bytes is an empty line", t.line)
			}
			tensors = append(tensors, t)
		}
	}
	dataTensors := len(tensors)
	for k := range m.empties {
		if err := checkTensor(&m.empties[k]); err != nil {
			return nil, err
		}
		tensors = append(tensors, &m.empties[k])
	}
	w.Tensors = len(tensors)

	// names: the pool in table order, the order table sorted bytewise; equal names are refused
	var names []byte
	order := make([]int, len(tensors))
	for i, t := range tensors {
		names = append(names, t.name...)
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return bytes.Compare(tensors[order[a]].name, tensors[order[b]].name) < 0 })
	for k := 1; k < len(order); k++ {
		if bytes.Equal(tensors[order[k-1]].name, tensors[order[k]].name) {
			return nil, fmt.Errorf("line %d: the name of this tensor is also the name of the tensor of line %d",
				tensors[order[k]].line, tensors[order[k-1]].line)
		}
	}
	var meta []byte
	if m.meta != "" {
		var err error
		if meta, err = os.ReadFile(m.meta); err != nil {
			return nil, err
		}
	}
	if len(m.blobs) > capBlobs || len(m.records) > capRecords || len(tensors) > capTensors {
		return nil, fmt.Errorf("a count is above its cap: %d blobs, %d records, %d tensors", len(m.blobs), len(m.records), len(tensors))
	}
	indexBytes := blobEntry*uint64(len(m.blobs)) + recordEntry*uint64(len(m.records)) + (tensorEntry+4)*uint64(len(tensors)) +
		uint64(len(names)) + uint64(len(meta))
	if indexBytes > capIndex {
		return nil, fmt.Errorf("the index would have %d bytes, the cap is 2^28", indexBytes)
	}
	index := make([]byte, indexBytes)
	recordAt := blobEntry * len(m.blobs)
	tensorAt := recordAt + recordEntry*len(m.records)
	orderAt := tensorAt + tensorEntry*len(tensors)
	poolAt := orderAt + 4*len(tensors)
	copy(index[poolAt:], names)
	copy(index[poolAt+len(names):], meta)
	var nameOff uint32
	for i, t := range tensors {
		e := index[tensorAt+tensorEntry*i:]
		le.PutUint32(e[0:], nameOff)
		le.PutUint16(e[4:], uint16(len(t.name)))
		e[6], e[7] = t.dtype, t.rank
		for k := 0; k < 8; k++ {
			le.PutUint64(e[8+8*k:], t.shape[k])
		}
		le.PutUint64(e[88:], t.nbytes) // record, record_count and offset are set with the tensor's record
		nameOff += uint32(len(t.name))
		le.PutUint32(index[orderAt+4*i:], uint32(order[i]))
	}

	// extents in the order of the manifest; without an index line the index comes first (canonical layout)
	extents := m.order
	if !m.hasIndex {
		extents = append([]extent{{'i', 0}}, extents...)
	}
	tmp := out + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return nil, err
	}
	keep := false
	defer func() {
		f.Close()
		if !keep {
			os.Remove(tmp)
		}
	}()

	pos := uint64(pageBytes)
	put := func(b []byte) error { // b is a whole number of pages
		if _, err := f.WriteAt(b, int64(pos)); err != nil {
			return err
		}
		pos += uint64(len(b))
		return nil
	}
	indexOffset := uint64(pageBytes)
	models := map[int]*Model{}
	dig := newDigest()
	required := m.required
	var maxDecoded uint32
	var totalDecoded uint64
	// the split tensor whose parts are being written
	var split struct {
		file      *os.File
		left, off uint64
		head      int // its first record
		t         *srcTensor
	}
	defer func() {
		if split.file != nil {
			split.file.Close()
		}
	}()

	for _, x := range extents {
		switch x.kind {
		case 'i':
			if indexBytes != 0 {
				indexOffset = pos
				pos += align(indexBytes) // written at the end, when the record entries are known
			}
		case 'b':
			b := &m.blobs[x.n]
			data, err := os.ReadFile(b.file)
			if err != nil {
				return nil, err
			}
			if len(data) < 1 || len(data) > capDecoded {
				return nil, fmt.Errorf("line %d: a blob has 1 .. 2^30 bytes, this one has %d", b.line, len(data))
			}
			id := b.id
			if b.kind == blobKindWNM2 {
				model, e := OpenModel(data)
				if e != nil {
					return nil, fmt.Errorf("line %d: %v", b.line, e)
				}
				if b.hasID && b.id != model.ID {
					return nil, fmt.Errorf("line %d: id 0x%08x, the model's FNV-1a is 0x%08x", b.line, b.id, model.ID)
				}
				id, models[x.n] = model.ID, model
			}
			e := index[blobEntry*x.n:]
			le.PutUint32(e[0:], b.kind)
			le.PutUint32(e[4:], id)
			le.PutUint64(e[8:], pos)
			le.PutUint64(e[16:], uint64(len(data)))
			sum := sha256.Sum256(data)
			copy(e[24:], sum[:])
			if err := put(padded(data)); err != nil {
				return nil, err
			}
		case 'r':
			i := x.n
			r := &m.records[i]
			lastPart := i+1 == len(m.records) || !m.records[i+1].part

			// the bytes this record decodes to
			var data []byte
			var first *srcTensor // the tensor whose dtype decides the defaults of a split record
			switch {
			case r.part:
				if split.file == nil {
					return nil, fmt.Errorf("line %d: a part record with no split tensor before it", r.line)
				}
				first = split.t
			case r.bytes != 0 || !lastPart:
				if len(r.tensors) != 1 || r.expert {
					return nil, fmt.Errorf("line %d: a split tensor is the only tensor of its first record, and not an expert record", r.line)
				}
				if lastPart {
					return nil, fmt.Errorf("line %d: bytes= without a part record after it", r.line)
				}
				first = &r.tensors[0]
				if split.file, err = os.Open(first.file); err != nil {
					return nil, err
				}
				split.left, split.off, split.head, split.t = first.nbytes, 0, i, first
				dig.tensor(first.name, first.dtype, first.rank, first.shape, first.nbytes)
			default:
				first = &r.tensors[0]
				for k := range r.tensors {
					t := &r.tensors[k]
					b, err := os.ReadFile(t.file)
					if err != nil {
						return nil, err
					}
					if uint64(len(b)) != t.nbytes {
						return nil, fmt.Errorf("line %d: the file changed while it was read", t.line)
					}
					e := index[tensorAt+tensorEntry*(firstTensor[i]+k):]
					le.PutUint32(e[72:], uint32(i))
					le.PutUint32(e[76:], 1)
					le.PutUint64(e[80:], uint64(len(data)))
					data = append(data, b...)
					if len(data) > capDecoded {
						return nil, fmt.Errorf("line %d: the record decodes to more than 2^30 bytes", r.line)
					}
				}
			}
			if data == nil { // a part of the split tensor
				n := r.bytes
				if n == 0 {
					if !lastPart {
						return nil, fmt.Errorf("line %d: only the last part of a split tensor can leave bytes= out", r.line)
					}
					n = split.left
				}
				if n == 0 || n > split.left || n > capDecoded || (lastPart && n != split.left) {
					return nil, fmt.Errorf("line %d: a part of %d bytes, the tensor has %d left (a record holds 1 .. 2^30)", r.line, n, split.left)
				}
				data = make([]byte, n)
				if _, err := io.ReadFull(io.NewSectionReader(split.file, int64(split.off), int64(n)), data); err != nil {
					return nil, err
				}
				split.left, split.off = split.left-n, split.off+n
				if lastPart {
					e := index[tensorAt+tensorEntry*firstTensor[split.head]:]
					le.PutUint32(e[72:], uint32(split.head))
					le.PutUint32(e[76:], uint32(i-split.head+1))
					split.file.Close()
					split.file = nil
				}
			}
			n := uint64(len(data))

			// the codec and its parameters
			codec := r.codec
			if codec == "auto" {
				switch {
				case !r.expert:
					codec = "wbp1"
				case r.model != 0:
					codec = "wnr3"
				default:
					codec = "raw"
				}
			}
			var geo Geo
			if r.expert {
				if geo, err = geoOf(r); err != nil {
					return nil, err
				}
				if codec == "wbp1" {
					return nil, fmt.Errorf("line %d: an expert record is RAW or WNR3", r.line)
				}
			} else if codec == "wnr3" {
				return nil, fmt.Errorf("line %d: a WNR3 record is an expert record (expert=LAYER:EXPERT)", r.line)
			}
			owner := uint32(i)
			var flags uint16
			if r.expert {
				owner, flags = r.owner, flagExpert
			}
			stored, codecID, blob := padded(data), uint16(codecRAW), uint32(0)
			decoded := data
			switch codec {
			case "wbp1":
				planes, chunk, transform, err := wbp1Params(r, first, n)
				if err != nil {
					return nil, err
				}
				rec := encodeWBP1(data, owner, planes, chunk, transform)
				if len(rec) >= len(stored) {
					w.Fallbacks++ // a compressed record is strictly smaller than raw, or it is stored raw
					break
				}
				back := make([]byte, n)
				if e := decodeWBP1(rec, owner, back); e != nil || !bytes.Equal(back, data) {
					return nil, fmt.Errorf("line %d: the WBP1 record does not decode to its source: %v", r.line, e)
				}
				stored, codecID = rec, codecWBP1
			case "wnr3":
				model := models[int(r.model)-1]
				if r.model == 0 || model == nil {
					return nil, fmt.Errorf("line %d: a WNR3 record needs model=N, the number (from 1) of a blob of kind 1 that is before it", r.line)
				}
				if !pow2In(r.cw, 512, 16384) || !pow2In(r.cs, 256, 16384) {
					return nil, fmt.Errorf("line %d: chunks cw=%d cs=%d: powers of 2 in 512..16384 and 256..16384", r.line, r.cw, r.cs)
				}
				if r.cz {
					required |= 1 // set by a canonical-zero build even when the record falls back to raw
				}
				rec, dec := encodeWNR3(data, model, geo, r.owner, r.cz, r.cw, r.cs)
				if len(rec) >= len(stored) {
					w.Fallbacks++ // stored raw: the source bytes, no VALUE_EXACT flag
					break
				}
				back := make([]byte, n)
				if e := decodeWNR3(rec, model, geo, r.owner, r.cz, back); e != nil || !bytes.Equal(back, dec) {
					return nil, fmt.Errorf("line %d: the WNR3 record does not decode to its source: %v", r.line, e)
				}
				stored, codecID, blob, decoded = rec, codecWNR3, r.model, dec
				if r.cz {
					flags |= flagValueExact
				}
			}
			switch codecID {
			case codecRAW:
				w.RAW++
			case codecWBP1:
				w.WBP1++
			default:
				w.WNR3++
			}

			// the content digest: what the record decodes to
			if r.part || split.file != nil {
				dig.bytes(decoded)
			} else {
				var off uint64
				for k := range r.tensors {
					t := &r.tensors[k]
					dig.tensor(t.name, t.dtype, t.rank, t.shape, t.nbytes)
					dig.bytes(decoded[off : off+t.nbytes])
					off += t.nbytes
				}
			}

			e := index[recordAt+recordEntry*i:]
			le.PutUint64(e[0:], pos)
			le.PutUint32(e[8:], uint32(len(stored)))
			le.PutUint32(e[12:], uint32(n))
			le.PutUint16(e[16:], codecID)
			le.PutUint16(e[18:], flags)
			le.PutUint32(e[20:], blob)
			le.PutUint32(e[24:], owner)
			le.PutUint32(e[28:], uint32(firstTensor[i]))
			if !r.part {
				le.PutUint32(e[32:], uint32(len(r.tensors)))
			}
			sum := sha256.Sum256(stored)
			copy(e[36:], sum[:])
			maxDecoded, totalDecoded = max(maxDecoded, uint32(n)), totalDecoded+n
			if err := put(stored); err != nil {
				return nil, err
			}
		}
	}
	if split.file != nil {
		return nil, fmt.Errorf("the last split tensor has %d bytes that no part record takes", split.left)
	}
	for k := dataTensors; k < len(tensors); k++ {
		t := tensors[k]
		if t.nbytes != 0 {
			return nil, fmt.Errorf("line %d: an empty tensor has 0 bytes", t.line)
		}
		dig.tensor(t.name, t.dtype, t.rank, t.shape, 0)
	}

	// the index, then the header page
	if indexBytes != 0 {
		if _, err := f.WriteAt(padded(index), int64(indexOffset)); err != nil {
			return nil, err
		}
	}
	page := make([]byte, pageBytes)
	le.PutUint32(page[0:], magicWEX4)
	le.PutUint16(page[4:], 4)
	le.PutUint16(page[6:], m.minor)
	le.PutUint64(page[8:], required)
	le.PutUint64(page[16:], m.optional)
	le.PutUint64(page[24:], pos)
	le.PutUint64(page[32:], indexOffset)
	le.PutUint64(page[40:], indexBytes)
	le.PutUint32(page[48:], uint32(len(m.blobs)))
	le.PutUint32(page[52:], uint32(len(m.records)))
	le.PutUint32(page[56:], uint32(len(tensors)))
	le.PutUint32(page[60:], m.layers)
	le.PutUint32(page[64:], m.experts)
	le.PutUint32(page[68:], maxDecoded)
	le.PutUint64(page[72:], uint64(len(names)))
	le.PutUint64(page[80:], uint64(len(meta)))
	le.PutUint64(page[88:], totalDecoded)
	sum := sha256.Sum256(index)
	copy(page[96:], sum[:])
	w.Root = sha256.Sum256(page[:4064])
	copy(page[4064:], w.Root[:])
	if _, err := f.WriteAt(page, 0); err != nil {
		return nil, err
	}
	w.Digest, w.FileBytes = dig.sum(), pos

	// read the file back: it gets its name only when the reader accepts it
	res, e := Verify(f, pos, w.Root[:], nil)
	if e != nil {
		return nil, fmt.Errorf("the file was written and the reader of this folder refuses it (%v): it was removed", e)
	}
	if res.Undecoded != 0 || res.Digest != w.Digest {
		return nil, fmt.Errorf("the file was written and does not decode to its sources (content digest %x, from the sources %x): it was removed", res.Digest, w.Digest)
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, out); err != nil {
		return nil, err
	}
	keep = true
	return w, nil
}

// wbp1Params: the parameters of a WBP1 record. What the manifest does not give is chosen as section 7 says
// the converters choose: planes = the element size in bytes, transform 1 for floats (F16, BF16, F32, F64).
// The specification does not give the converters' chunk_log2 (SPEC-NOTES.md, E3): 12 here.
func wbp1Params(r *srcRecord, first *srcTensor, n uint64) (planes uint64, chunk uint, transform byte, err error) {
	elem, float := uint64(1), true
	ts := r.tensors
	if len(ts) == 0 {
		ts = []srcTensor{*first}
	}
	for k := range ts {
		d := ts[k].dtype
		e := uint64(1)
		if int(d) < len(dtypeBits) && dtypeBits[d] >= 16 {
			e = dtypeBits[d] / 8
		}
		if k > 0 && e != elem {
			e = 1 // tensors of different element sizes in one record: one plane
		}
		elem = e
		float = float && (d == 9 || d == 10 || d == 13 || d == 16)
	}
	planes = elem
	if r.planes >= 0 {
		planes = uint64(r.planes)
	}
	chunk = 12
	if r.chunk >= 0 {
		chunk = uint(r.chunk)
	}
	if float && planes > 1 && n%planes == 0 {
		transform = 1
	}
	if r.transform >= 0 {
		transform = byte(r.transform)
	}
	if planes != 1 && planes != 2 && planes != 4 && planes != 8 {
		return 0, 0, 0, fmt.Errorf("line %d: planes=%d: 1, 2, 4 or 8", r.line, planes)
	}
	if chunk < 10 || chunk > 15 {
		return 0, 0, 0, fmt.Errorf("line %d: chunk=%d: 10 .. 15", r.line, chunk)
	}
	if transform > 1 || (transform == 1 && (planes == 1 || n%planes != 0)) {
		return 0, 0, 0, fmt.Errorf("line %d: transform=%d with %d planes and %d bytes: 1 needs more than one plane and a whole number of elements", r.line, transform, planes, n)
	}
	return planes, chunk, transform, nil
}
