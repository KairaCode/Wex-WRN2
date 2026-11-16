// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// The content digest of docs/WEX4-SPEC.md section 5: a SHA-256 over the tensors in table order, which does
// not depend on the layout, the grouping into records or the codecs. Written from that paragraph only.

package main

import (
	"crypto/sha256"
	"hash"
)

// digest hashes tensors one after the other, in table order.
type digest struct{ h hash.Hash }

func newDigest() *digest { return &digest{sha256.New()} }

// tensor adds the fields of one tensor: name length u16, the name, dtype u8, rank u8, the 8 shape slots as
// u64, nbytes u64, all little-endian. The tensor's bytes follow through bytes().
func (d *digest) tensor(name []byte, dtype, rank uint8, shape [8]uint64, nbytes uint64) {
	var b [84]byte
	le.PutUint16(b[0:], uint16(len(name)))
	d.h.Write(b[:2])
	d.h.Write(name)
	b[0], b[1] = dtype, rank
	for k := 0; k < 8; k++ {
		le.PutUint64(b[2+8*k:], shape[k])
	}
	le.PutUint64(b[66:], nbytes)
	d.h.Write(b[:74])
}

func (d *digest) bytes(b []byte) { d.h.Write(b) }

func (d *digest) sum() (s [32]byte) {
	copy(s[:], d.h.Sum(nil))
	return s
}

// record adds what record i decoded to: its tensors with their fields, or, for a continuation part, the
// bytes alone (the fields went in with the first part).
func (d *digest) record(x *Index, r Record, dec []byte) {
	for j := uint32(0); j < r.TensorCount; j++ {
		t := x.Tensor(r.FirstTensor + j)
		d.tensor(x.Name(t), t.Dtype, t.Rank, t.Shape, t.NBytes)
		if t.RecordCount == 1 {
			d.bytes(dec[t.Offset : t.Offset+t.NBytes])
		}
	}
	if r.TensorCount == 0 || x.Tensor(r.FirstTensor).RecordCount != 1 {
		d.bytes(dec) // a part of a split tensor: the whole record
	}
}

// zeroSize adds the tensors that have no bytes: they come after all data tensors.
func (d *digest) zeroSize(x *Index) {
	for i := x.DataTensors; i < x.H.Tensors; i++ {
		t := x.Tensor(i)
		d.tensor(x.Name(t), t.Dtype, t.Rank, t.Shape, t.NBytes)
	}
}
