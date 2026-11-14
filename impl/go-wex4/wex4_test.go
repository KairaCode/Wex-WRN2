// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

const vectors = "../../wex4/vectors"

var validFiles = []string{
	"generic.wex4", "generic-index-last.wex4", "tiny2-cz-upgraded.wex4", "tiny-raw-upgraded.wex4",
	"minor-version-9.wex4", "unknown-optional-feature-bits.wex4", "bytes-in-the-reserved-header-area.wex4",
}

func load(t testing.TB, name string) []byte {
	b, err := os.ReadFile(filepath.Join(vectors, "valid", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func verifyBytes(b []byte, visit func(uint32, Record, *Index, []byte)) (*Result, *Error) {
	return Verify(bytes.NewReader(b), uint64(len(b)), nil, visit)
}

// The vectors of docs/WNR3-SPEC.md section 4.5.
func TestStreamHash(t *testing.T) {
	seq := make([]byte, 16)
	for i := range seq {
		seq[i] = byte(i)
	}
	for _, c := range []struct {
		seed uint32
		data []byte
		want uint32
	}{{0, nil, 0}, {hashSeed, nil, 0xE7F741F1}, {hashSeed, seq, 0x31CE0919}} {
		if got := murmur3(c.seed, nil, c.data); got != c.want {
			t.Errorf("murmur3(seed 0x%08x, %d bytes) = 0x%08x, want 0x%08x", c.seed, len(c.data), got, c.want)
		}
	}
	// SMHasher verification value of MurmurHash3_x86_32
	var table [1024]byte
	key := make([]byte, 256)
	for i := 0; i < 256; i++ {
		key[i] = byte(i)
		le.PutUint32(table[4*i:], murmur3(uint32(256-i), nil, key[:i]))
	}
	if got := murmur3(0, nil, table[:]); got != 0xB0F57EE3 {
		t.Errorf("SMHasher verification value 0x%08x, want 0xB0F57EE3", got)
	}
	weight := make([]byte, 1024)
	for i := range weight {
		weight[i] = byte((37*i + 11) & 0x77)
	}
	if got := h32([4]uint32{0x299DC586, 0x00270078, 2165, 0x0F787978}, weight); got != 0x126485B2 {
		t.Errorf("weight stream vector 0x%08x, want 0x126485B2", got)
	}
	if got := h32([4]uint32{0x299DC586, 0x00270078, 719, 0x0F787978}, []byte{0x7F, 0x7E, 0x80, 0x7F, 0x7D}); got != 0x8A4976D8 {
		t.Errorf("scale stream vector 0x%08x, want 0x8A4976D8", got)
	}
}

// Every row of the manifest: file hash, verdict, reject class, root.
func TestVectors(t *testing.T) {
	rows, wrong, err := runManifest(filepath.Join(vectors, "MANIFEST.tsv"), false)
	if err != nil {
		t.Fatal(err)
	}
	if rows == 0 || wrong != 0 {
		t.Fatalf("%d rows, %d wrong", rows, wrong)
	}
	t.Logf("%d rows as the manifest says", rows)
	t.Logf("the decoders met: %+v", Seen)
}

// The decoded bytes of the WNR3 records, checked against data this reader did not decode. The two experts of
// tiny2-cz-upgraded.wex4 (WNR3 with canonical zero) are also RAW records of tiny-raw-upgraded.wex4, under
// the same owners: a canonical-zero record must decode to the RAW slab with every -0 weight nibble set to +0.
func TestWNR3AgainstRAW(t *testing.T) {
	type slab struct {
		data   []byte
		scales uint64
	}
	raw := map[uint32]slab{}
	if _, e := verifyBytes(load(t, "tiny-raw-upgraded.wex4"), func(i uint32, r Record, x *Index, dec []byte) {
		if r.Flags&flagExpert != 0 {
			g, _ := x.ExpertGeo(i, r)
			raw[r.Owner] = slab{append([]byte(nil), dec...), g.scales(0) + g.scales(1) + g.scales(2)}
		}
	}); e != nil {
		t.Fatal(e)
	}
	checked, mapped := 0, 0
	if _, e := verifyBytes(load(t, "tiny2-cz-upgraded.wex4"), func(i uint32, r Record, x *Index, dec []byte) {
		if r.Codec != codecWNR3 {
			return
		}
		src, ok := raw[r.Owner]
		if !ok || len(src.data) != len(dec) {
			t.Errorf("record %d: no RAW record of the same owner and size", i)
			return
		}
		want := append([]byte(nil), src.data...)
		for k := src.scales; k < uint64(len(want)); k++ {
			if want[k]&0x0F == 0x08 {
				want[k] &= 0xF0
				mapped++
			}
			if want[k]&0xF0 == 0x80 {
				want[k] &= 0x0F
				mapped++
			}
		}
		if !bytes.Equal(want, dec) {
			t.Errorf("record %d (owner 0x%08x): the decoded slab is not Z(RAW slab)", i, r.Owner)
		}
		checked++
	}); e != nil {
		t.Fatal(e)
	}
	if checked == 0 {
		t.Fatal("no WNR3 record was compared")
	}
	t.Logf("%d WNR3 records equal to Z(RAW slab), %d bytes each side, %d nibbles were -0 in the RAW slabs", checked, len(raw[0].data), mapped)
}

// generic.wex4 and generic-index-last.wex4 hold the same tensors in another layout: every decoded record
// must be the same.
func TestTwoLayouts(t *testing.T) {
	var sums [2][][32]byte
	for k, name := range []string{"generic.wex4", "generic-index-last.wex4"} {
		if _, e := verifyBytes(load(t, name), func(i uint32, r Record, x *Index, dec []byte) {
			sums[k] = append(sums[k], sha256.Sum256(dec))
		}); e != nil {
			t.Fatal(e)
		}
	}
	if len(sums[0]) == 0 || len(sums[0]) != len(sums[1]) {
		t.Fatalf("%d and %d records", len(sums[0]), len(sums[1]))
	}
	for i := range sums[0] {
		if sums[0][i] != sums[1][i] {
			t.Errorf("record %d decodes to other bytes in the two layouts", i)
		}
	}
}

// testdata/ceil-form.wex4: two WNR3 records without canonical zero, five of their six matrices in ceil form
// (the vectors have none). The slabs they must decode to are known from the source, not from a decoder:
// testdata/make_ceil_form.py prints these SHA-256, and the repository's builder recorded the same two.
func TestCeilForm(t *testing.T) {
	want := []string{
		"7155fea8ed2207ff93b595e7680c811df9de69c69dbd6cce7d51792ef2c51904",
		"1cc3b23906b40fa6e051f81e28fc9c9c9bd5f375bb5a9d793a8d41f94f403bd5",
	}
	b, err := os.ReadFile("testdata/ceil-form.wex4")
	if err != nil {
		t.Fatal(err)
	}
	before := Seen
	n := 0
	res, e := verifyBytes(b, func(i uint32, r Record, x *Index, dec []byte) {
		if got := fmt.Sprintf("%x", sha256.Sum256(dec)); int(i) >= len(want) || r.Codec != codecWNR3 || got != want[i] {
			t.Errorf("record %d: codec %d decodes to %s", i, r.Codec, got)
		}
		n++
	})
	if e != nil {
		t.Fatal(e)
	}
	matrices, blocks := Seen.WNR3CeilMatrices-before.WNR3CeilMatrices, Seen.WNR3CeilBlocks-before.WNR3CeilBlocks
	if n != len(want) || matrices != 5 || blocks == 0 {
		t.Fatalf("%d records, %d matrices in ceil form, %d blocks changed by the inverse", n, matrices, blocks)
	}
	t.Logf("root %x: %d records byte-identical to their source slabs, %d matrices in ceil form, %d blocks changed by the inverse",
		res.H.Root, n, matrices, blocks)
}

// testdata/planes.wex4: WBP1 records of 4 and 8 planes and rANS planes full of the byte 255 (the vectors
// have 1 and 2 planes and no 255 in a coded plane). testdata/make_planes.py prints the SHA-256 of every
// tensor of the source; the repository's converter packed the file.
func TestPlanes(t *testing.T) {
	want := map[string]string{
		"f64.walk":    "b951a7f605efbbbbf0a50a3324ed598f02c108471cc9154d9cf53ba5b464d072",
		"f32.walk":    "51280615b2cbc9d770ef16b3b23fd07ef3edafba01241aab0854b42f77505fd5",
		"f32.sine":    "16cd22aeb555df446a9f8f10b7d4770ee842f1592addb57c0c9767f05e8f4543",
		"f16.walk":    "ce1285fd70160565ae49ddd37fc70747710331917259686f1f6d5419ac5001e5",
		"i64.small":   "f42cb3fff797c51b3c26fa82b9d27ffdd1a1248aa911bda135c790f76a8fa482",
		"i32.small":   "94ef34f2ce862d069b315f546644f3dd101ab0d4c38014cd5cc905e2ca8e0ee2",
		"i16.small":   "9cdf330629679138be0237d9c53d84f760f925c90519da2c5c430e977ba6984c",
		"u64.count":   "e8c9ceaf5aacc63c25b4cdd8542592f9d58aff50e3e8fc6c55591d3d8f596562",
		"i8.small":    "127e8abb17e5e267ccc8eba9d3f86552decddec42c97ad28ef2503fe4f78225d",
		"u8.high":     "657614a9d091dd5609247c9dc881e62dc3af3dad3c7408003415da7971b4f1f3",
		"bool.sparse": "b9d1e3e53e4928daa6a2c401b43a169f4389b08098ef220346600594e8aa137c",
		"f32.big":     "5ec091a148db4abf484adac441898eba1731ea67e2ea249018048adfbe4b71e1",
		"f64.big":     "9bd7a2570cf491aaf770d6addcf9c4b50bc5efdbe02b900238812226bac54d07",
	}
	b, err := os.ReadFile("testdata/planes.wex4")
	if err != nil {
		t.Fatal(err)
	}
	seen, coded := 0, 0
	if _, e := verifyBytes(b, func(i uint32, r Record, x *Index, dec []byte) {
		if r.Codec == codecWBP1 {
			coded++
		}
		for k := r.FirstTensor; k < r.FirstTensor+r.TensorCount; k++ {
			tn := x.Tensor(k)
			name := string(x.Name(tn))
			if got := fmt.Sprintf("%x", sha256.Sum256(dec[tn.Offset:tn.Offset+tn.NBytes])); got != want[name] {
				t.Errorf("tensor %q decodes to %s", name, got)
			}
			seen++
		}
	}); e != nil {
		t.Fatal(e)
	}
	if seen != len(want) || coded == 0 {
		t.Fatalf("%d of %d tensors, %d WBP1 records", seen, len(want), coded)
	}
	t.Logf("%d tensors byte-identical to their source, %d WBP1 records", seen, coded)
}

// rehash recomputes index_sha256 and header_sha256, so that a change of the header page or of the index
// reaches the validators. It does nothing on a file whose index is not where the header says.
func rehash(b []byte) {
	if len(b) < pageBytes {
		return
	}
	off, n := le.Uint64(b[32:]), le.Uint64(b[40:])
	if off <= uint64(len(b)) && n <= uint64(len(b))-off {
		sum := sha256.Sum256(b[off : off+n])
		copy(b[96:128], sum[:])
	}
	root := sha256.Sum256(b[:4064])
	copy(b[4064:], root[:])
}

// decodedDigest verifies b and returns one SHA-256 over every decoded record.
func decodedDigest(b []byte) ([32]byte, *Error) {
	h := sha256.New()
	_, e := verifyBytes(b, func(i uint32, r Record, x *Index, dec []byte) { h.Write(dec) })
	var sum [32]byte
	h.Sum(sum[:0])
	return sum, e
}

// Mutations of the valid vectors and of the two files of testdata/, the policy of WEX4-SPEC section 9:
// (a) a bit flipped anywhere with the hashes left alone must be refused; (b) a byte changed in the header
// page or the index with the hashes recomputed must be refused or verify in full; (c) a byte changed in a
// WBP1 or WNR3 record with the record's SHA-256 recomputed must be refused by the decoder or decode to the
// same bytes. A panic (an out-of-bounds access in Go) fails the test.
func TestMutations(t *testing.T) {
	rng := rand.New(rand.NewSource(20261114))
	files := map[string][]byte{}
	names := append([]string{"testdata/ceil-form.wex4", "testdata/planes.wex4"}, validFiles...)
	for _, name := range validFiles {
		files[name] = load(t, name)
	}
	for _, name := range names[:2] {
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		files[name] = b
	}
	var flips, accepted, refused, coded, codedSame int
	for _, name := range names {
		orig := files[name]
		b := append([]byte(nil), orig...)
		for k := 0; k < 1000; k++ {
			pos := rng.Intn(len(b))
			b[pos] ^= 1 << rng.Intn(8)
			if _, e := verifyBytes(b, nil); e == nil {
				t.Errorf("%s: bit flipped at byte %d and the file is accepted", name, pos)
			}
			b[pos] = orig[pos]
			flips++
		}
		off, n := int(le.Uint64(b[32:])), int(le.Uint64(b[40:]))
		for k := 0; k < 2000; k++ {
			pos := 4 + rng.Intn(124) // the fields of the header page
			if k%4 != 0 {
				pos = off + rng.Intn(n)
			}
			b[pos] = byte(rng.Intn(256))
			rehash(b)
			if _, e := verifyBytes(b, nil); e == nil {
				accepted++
			} else {
				refused++
			}
			copy(b[:pageBytes], orig[:pageBytes])
			b[pos] = orig[pos]
		}
		for _, cut := range []int{0, 1, 2048, 4095, 4096, len(b) - 4096, len(b) - 1} {
			if _, e := verifyBytes(orig[:cut], nil); e == nil {
				t.Errorf("%s: accepted when cut to %d bytes", name, cut)
			}
		}

		// (c): the compressed records of this file
		h, e := ParseHeader(orig[:pageBytes], uint64(len(orig)))
		if e != nil {
			t.Fatal(e)
		}
		x, e := OpenIndex(h, orig[off:off+int(align(uint64(n)))])
		if e != nil {
			t.Fatal(e)
		}
		var records []uint32
		for i := uint32(0); i < h.Records; i++ {
			if c := x.Record(i).Codec; c == codecWBP1 || c == codecWNR3 {
				records = append(records, i)
			}
		}
		if len(records) == 0 {
			continue
		}
		want, e := decodedDigest(orig)
		if e != nil {
			t.Fatal(e)
		}
		for k := 0; k < 1000; k++ {
			i := records[rng.Intn(len(records))]
			r := x.Record(i)
			pos := int(r.Offset) + rng.Intn(int(r.Stored))
			if k%2 == 0 {
				b[pos] ^= 1 << rng.Intn(8)
			} else {
				b[pos] = byte(rng.Intn(256))
			}
			sum := sha256.Sum256(b[r.Offset : r.Offset+uint64(r.Stored)])
			entry := off + int(x.rt) + int(i)*recordEntry + 36
			copy(b[entry:], sum[:])
			rehash(b)
			if got, e := decodedDigest(b); e == nil {
				codedSame++
				if got != want {
					t.Errorf("%s: record %d changed at file offset %d decodes to other bytes and is accepted", name, i, pos)
				}
			}
			coded++
			b[pos] = orig[pos]
			copy(b[entry:entry+32], orig[entry:])
			copy(b[:pageBytes], orig[:pageBytes])
		}
	}
	t.Logf("%d files", len(names))
	t.Logf("(a) %d bit flips with the hashes left alone: all refused", flips)
	t.Logf("(b) %d header and index bytes changed with the hashes recomputed: %d refused, %d accepted", accepted+refused, refused, accepted)
	t.Logf("(c) %d bytes of WBP1 and WNR3 records changed with the record hash recomputed: %d refused, %d accepted with the same decoded bytes, 0 accepted with other bytes",
		coded, coded-codedSame, codedSame)
}
