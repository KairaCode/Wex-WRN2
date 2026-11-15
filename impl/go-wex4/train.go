// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// A WNM2 model trained on expert slabs: docs/WNR2-REFERENCE.md section 7 ("Training", "WNM2 blob") and
// docs/WNR3-SPEC.md section 5 and erratum 12 (the counts are taken in the coded domain of each record, with
// its rule-g anchors). Written from those pages only.

package main

import (
	"fmt"
	"os"
)

// Trained is what TrainFile reports.
type Trained struct {
	ID            uint32
	Records, CZ   int // expert records counted, those with canonical zero among them
	EmptyContexts int // of the 3 x 567 weight contexts, those no symbol reached: their CDF is uniform
	EmptyScales   int // of the 3 scale tables
}

// TrainFile trains the model of blob line n (from 1) of the manifest on every expert record whose model= is
// n, each in its own coded domain (canonical zero when the record has cz), and writes it to out.
func TrainFile(m *Manifest, n uint32, out string) (*Trained, error) {
	if n == 0 || int(n) > len(m.blobs) || m.blobs[n-1].kind != blobKindWNM2 {
		return nil, fmt.Errorf("blob line %d of the manifest is not a blob of kind 1", n)
	}
	var c counts // one slab in memory at a time: its counts are added, then it is dropped
	tr := &Trained{}
	for i := range m.records {
		r := &m.records[i]
		if !r.expert || r.model != n {
			continue
		}
		g, err := geoOf(r)
		if err != nil {
			return nil, err
		}
		var slab []byte
		for k := range r.tensors {
			t := &r.tensors[k]
			if err := checkTensor(t); err != nil { // the size against the shape, as the writer checks it
				return nil, err
			}
			b, err := os.ReadFile(t.file)
			if err != nil {
				return nil, err
			}
			if uint64(len(b)) != t.nbytes {
				return nil, fmt.Errorf("line %d: the file changed while it was read", t.line)
			}
			slab = append(slab, b...)
		}
		c.add(slab, g, r.cz)
		tr.Records++
		if r.cz {
			tr.CZ++
		}
	}
	if tr.Records == 0 {
		return nil, fmt.Errorf("no expert record of the manifest names model=%d", n)
	}
	blob, empty, emptyScale := c.model()
	model, e := OpenModel(blob)
	if e != nil {
		return nil, fmt.Errorf("the trained model does not open: %v", e)
	}
	tr.ID, tr.EmptyContexts, tr.EmptyScales = model.ID, empty, emptyScale
	tmp := out + ".tmp"
	if err := os.WriteFile(tmp, blob, 0o666); err != nil {
		os.Remove(tmp)
		return nil, err
	}
	if err := os.Rename(tmp, out); err != nil {
		os.Remove(tmp)
		return nil, err
	}
	return tr, nil
}

// counts: the counts of the 16 symbols of every weight context and of the 256 scales of each matrix, taken
// as the decoder sees them (after canonical zero when cz is set, the ceil form and the anchors of each
// slab).
type counts struct {
	weights [3][wnmContexts][16]uint64
	scales  [3][256]uint64
}

// add counts one slab of geometry g; the caller checked that its size is the one g gives.
func (n *counts) add(slab []byte, g Geo, cz bool) {
	c := codedForm(slab, g, cz)
	for m := 0; m < 3; m++ {
		for _, s := range c.scales(g, m) {
			n.scales[m][s]++
		}
		c.weightSymbols(g, m, func(ctx, s int) { n.weights[m][ctx][s]++ })
	}
}

// model builds the WNM2 blob of the counts, quantised with quantize. It also returns the number of weight
// contexts and of scale tables that had no count.
func (n *counts) model() (blob []byte, empty, emptyScale int) {
	weights, scales := &n.weights, &n.scales
	sum := func(counts []uint64) (t uint64) {
		for _, c := range counts {
			t += c
		}
		return t
	}
	// the layout of WNR3-SPEC erratum 12: header, weight tables by matrix and context, scale tables
	b := make([]byte, 0, wnm2Bytes)
	for _, w := range [8]uint32{magicWNM2, 1, cdfBits, wnmContexts, 3, 0, 0, 0} {
		b = le.AppendUint32(b, w)
	}
	for m := 0; m < 3; m++ {
		for ctx := range weights[m] {
			t := sum(weights[m][ctx][:])
			if t == 0 {
				empty++
			}
			for _, v := range quantize(weights[m][ctx][:], t) {
				b = le.AppendUint16(b, v)
			}
		}
	}
	for m := 0; m < 3; m++ {
		t := sum(scales[m][:])
		if t == 0 {
			emptyScale++
		}
		for _, v := range quantize(scales[m][:], t) {
			b = le.AppendUint16(b, v)
		}
	}
	return b, empty, emptyScale
}
