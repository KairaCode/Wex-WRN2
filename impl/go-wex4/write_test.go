// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Tests of the content digest and of the writer. The reference readers are asked when run.sh found them
// (WEX4_REF_PYTHON, WEX4_REF_DLL, WEX4_REF_VERIFY); with WEX4_NEED_REFERENCE=1 a missing one is a failure.

package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The content digest the Python reference printed for each file (python -I wex4/wex4.py verify FILE, with
// --dll for the three files that hold WNR3 records; 2026-11-15).
var referenceDigest = map[string]string{
	"generic.wex4":                           "76beeab372b67fa0dd77e11d0261ce3eb2d894c5e6e456b1138c6d2bed66837a",
	"generic-index-last.wex4":                "76beeab372b67fa0dd77e11d0261ce3eb2d894c5e6e456b1138c6d2bed66837a",
	"minor-version-9.wex4":                   "76beeab372b67fa0dd77e11d0261ce3eb2d894c5e6e456b1138c6d2bed66837a",
	"unknown-optional-feature-bits.wex4":     "76beeab372b67fa0dd77e11d0261ce3eb2d894c5e6e456b1138c6d2bed66837a",
	"bytes-in-the-reserved-header-area.wex4": "76beeab372b67fa0dd77e11d0261ce3eb2d894c5e6e456b1138c6d2bed66837a",
	"tiny2-cz-upgraded.wex4":                 "6a98449f13cf26311aae2db503612c49304dd471dedc753dc79faf0ba56d824c",
	"tiny-raw-upgraded.wex4":                 "d8ce6ef867f6b0f955bae030938511caaf0be6baa923e1ade22ac223bd090ac0",
	"testdata/ceil-form.wex4":                "8807c0ab7ba3bc555107c9e4749d70fc97cd93dd83c316c6717c44b633678d82",
	"testdata/planes.wex4":                   "26ccf6f46b0ae36cef9889f6945e53e8b622f4913ca32da5f8692a071e879bd6",
	"testdata/ragged.wex4":                   "c30449ba23b441c38aed2b6805e39251b2b9531595ba06a6eb8cdb029c389346",
}

func pathOf(name string) string {
	if strings.HasPrefix(name, "testdata/") {
		return name
	}
	return filepath.Join(vectors, "valid", name)
}

func TestContentDigest(t *testing.T) {
	for name, want := range referenceDigest {
		res, e := verifyFile(pathOf(name), nil)
		if e != nil {
			t.Fatalf("%s: %v", name, e)
		}
		if got := fmt.Sprintf("%x", res.Digest); got != want {
			t.Errorf("%s: content digest %s, the reference prints %s", name, got, want)
		}
	}
}

// rewrite exports a file and writes it again from the exported manifest, changed by edit when not nil.
func rewrite(t *testing.T, name string, edit func(*Manifest)) (string, *Written) {
	t.Helper()
	dir := t.TempDir()
	if _, err := Export(pathOf(name), filepath.Join(dir, "src")); err != nil {
		t.Fatalf("%s: export: %v", name, err)
	}
	m, err := ParseManifest(filepath.Join(dir, "src", "manifest.txt"))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if edit != nil {
		edit(m)
	}
	out := filepath.Join(dir, "out.wex4")
	w, err := WriteFile(m, out)
	if err != nil {
		t.Fatalf("%s: write: %v", name, err)
	}
	return out, w
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The sentence of WEX4-SPEC section 5 that the encoding is canonical: a file taken apart and written again
// with the same choices is the same file, byte for byte. The files were written by the reference writers.
func TestCanonical(t *testing.T) {
	before := seenNow()
	for name := range referenceDigest {
		out, _ := rewrite(t, name, nil)
		got, want := mustRead(t, out), mustRead(t, pathOf(name))
		if name == "bytes-in-the-reserved-header-area.wex4" {
			// a writer writes the reserved area as zero (section 4), so this vector cannot come out of one:
			// everything but the reserved area and the hash of the header page is the same
			if len(got) != len(want) || !bytes.Equal(got[:128], want[:128]) || !bytes.Equal(got[4096:], want[4096:]) ||
				!bytes.Equal(got, mustRead(t, pathOf("generic.wex4"))) {
				t.Errorf("%s: differs outside the reserved area of the header page", name)
			}
			continue
		}
		if !bytes.Equal(got, want) {
			diff := 0
			for k := range want {
				if k >= len(got) || got[k] != want[k] {
					diff = k
					break
				}
			}
			t.Errorf("%s: written again, %d bytes against %d, first difference at byte %d", name, len(got), len(want), diff)
		}
	}
	t.Logf("the decoders met, on the files and on the copies: %+v", diffSeen(before))
}

// diffSeen: what the decoders met since before was copied from Seen.
func diffSeen(before [5]uint64) string {
	now := seenNow()
	return fmt.Sprintf("WBP1 %d rANS streams, %d raw chunks; WNR3 %d scale streams, %d weight streams, %d matrices in ceil form",
		now[0]-before[0], now[1]-before[1], now[2]-before[2], now[3]-before[3], now[4]-before[4])
}

func seenNow() [5]uint64 {
	return [5]uint64{Seen.WBP1Rans, Seen.WBP1Raw, Seen.WNR3Scale, Seen.WNR3Weight, Seen.WNR3CeilMatrices}
}

// TestMain leaves, for run.sh, the number of times a reference reader was asked.
func TestMain(m *testing.M) {
	code := m.Run()
	if path := os.Getenv("WEX4_REF_LOG"); path != "" {
		os.WriteFile(path, []byte(fmt.Sprintf("%d\n", referenceRuns)), 0o666)
	}
	os.Exit(code)
}

// reference asks the two reference readers about a file the writer made: the Python reader alone when alone
// is true (it cannot decode WNR3 records alone, and it is slow on large WBP1 records), the Python reader
// with the C codec, and the C checker c/tools/wex4_verify.c (the C reader's validation of header and index,
// and every SHA-256).
func reference(t *testing.T, file string, w *Written, alone bool) {
	t.Helper()
	py, dll, cverify := os.Getenv("WEX4_REF_PYTHON"), os.Getenv("WEX4_REF_DLL"), os.Getenv("WEX4_REF_VERIFY")
	if os.Getenv("WEX4_NEED_REFERENCE") == "1" && (py == "" || dll == "" || cverify == "") {
		t.Fatalf("WEX4_NEED_REFERENCE=1 and a reference reader is missing: python %q, C codec %q, wex4_verify %q", py, dll, cverify)
	}
	root, digest := fmt.Sprintf("%x", w.Root), fmt.Sprintf("%x", w.Digest)
	python := func(args ...string) {
		out, err := exec.Command(py, append([]string{"-I", filepath.Join("..", "..", "wex4", "wex4.py"), "verify", file}, args...)...).CombinedOutput()
		if err != nil {
			t.Fatalf("the Python reference refuses %s: %v\n%s", file, err, out)
		}
		r := regexp.MustCompile(`root ([0-9a-f]{64})`).FindSubmatch(out)
		d := regexp.MustCompile(`content digest ([0-9a-f]{64})`).FindSubmatch(out)
		if r == nil || d == nil || string(r[1]) != root || string(d[1]) != digest {
			t.Fatalf("the Python reference on %s: expected root %s and content digest %s, it printed\n%s", file, root, digest, out)
		}
		referenceRuns++
	}
	if py != "" && alone {
		python()
	}
	if py != "" && dll != "" {
		python("--dll", dll)
	}
	if cverify != "" {
		out, err := exec.Command(cverify, file).CombinedOutput()
		if err != nil || !bytes.Contains(out, []byte("COLI_WEX_ROOT="+root)) {
			t.Fatalf("wex4_verify on %s: %v\n%s", file, err, out)
		}
		referenceRuns++
	}
}

var referenceRuns int

// xorshift: the test data is the same on every run
type prng uint64

func (p *prng) next() uint64 {
	x := uint64(*p)
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*p = prng(x)
	return x
}

func genBytes(kind string, n int, seed uint64) []byte {
	p := prng(seed*2654435761 + 88172645463325252)
	b := make([]byte, n)
	switch kind {
	case "rand":
		for i := range b {
			b[i] = byte(p.next() >> 32)
		}
	case "skew": // few values, the byte 255 among them
		for i := range b {
			v := p.next() >> 40 % 16
			b[i] = []byte{0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 255, 255, 255, 7, 128, 254}[v]
		}
	case "f32":
		for i := 0; i+4 <= n; i += 4 {
			binary.LittleEndian.PutUint32(b[i:], math.Float32bits(float32(math.Sin(float64(i)/97)*3)+float32(p.next()>>60)/1024))
		}
	case "f64":
		for i := 0; i+8 <= n; i += 8 {
			binary.LittleEndian.PutUint64(b[i:], math.Float64bits(math.Cos(float64(i)/301)*1e-3-float64(p.next()>>61)))
		}
	case "bf16": // the top 16 bits of a float32
		for i := 0; i+2 <= n; i += 2 {
			binary.LittleEndian.PutUint16(b[i:], uint16(math.Float32bits(float32(math.Sin(float64(i)/53))*float32(1+p.next()>>63))>>16))
		}
	case "i64":
		for i := 0; i+8 <= n; i += 8 {
			binary.LittleEndian.PutUint64(b[i:], uint64(int64(p.next()>>56)-128))
		}
	}
	return b
}

// A file made from nothing but the writer: every WBP1 parameter, the choices left to the writer, a split
// tensor of three parts with mixed codecs, tensors of 0 bytes, an opaque dtype, names that need escapes,
// the index between the records. The Go reader must give every tensor back, and so must the reference.
func TestWriterFromScratch(t *testing.T) {
	dir := t.TempDir()
	type src struct {
		file, kind string
		n          int
	}
	sources := []src{
		{"skew.bin", "skew", 5000}, {"bf16.bin", "bf16", 80002}, {"f32.bin", "f32", 120000}, {"f64.bin", "f64", 160000},
		{"i64.bin", "i64", 24000}, {"ragged.bin", "skew", 150001}, {"f32b.bin", "f32", 40000}, {"one.bin", "f32", 4},
		{"mixa.bin", "skew", 30000}, {"mixb.bin", "bf16", 3000}, {"noise.bin", "rand", 9000}, {"big.bin", "bf16", 60000},
		{"opaque.bin", "rand", 7}, {"i8.bin", "skew", 600},
	}
	content := map[string][]byte{}
	for k, s := range sources {
		content[s.file] = genBytes(s.kind, s.n, uint64(k+1))
		if err := os.WriteFile(filepath.Join(dir, s.file), content[s.file], 0o666); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), []byte(`{"made by":"the writer's test"}`), 0o666); err != nil {
		t.Fatal(err)
	}
	manifest := `wex4-manifest 1
# every line of the syntax
minor 3
optional 0x10
meta meta.json
record codec=wbp1 planes=1 chunk=10 transform=0
tensor name=a%20name%25with%09escapes dtype=U8 shape=50,100 file=skew.bin
record codec=wbp1 planes=2 chunk=11 transform=1
tensor name=bf16 dtype=BF16 shape=40001 file=bf16.bin
record codec=wbp1 planes=4 chunk=12 transform=1
tensor name=f32 dtype=F32 shape=300,100 file=f32.bin
record codec=wbp1 planes=8 chunk=13 transform=1
tensor name=f64 dtype=F64 shape=2,100,100 file=f64.bin
record codec=wbp1 planes=8 chunk=14 transform=0
tensor name=i64 dtype=I64 shape=3000 file=i64.bin
record codec=wbp1 planes=4 chunk=15
tensor name=ragged dtype=U8 shape=150001 file=ragged.bin
index
record
tensor name=auto.f32 dtype=F32 shape=10000 file=f32b.bin
tensor name=auto.scalar dtype=F32 shape= file=one.bin
record codec=auto
tensor name=mixed.u8 dtype=U8 shape=30000 file=mixa.bin
tensor name=mixed.bf16 dtype=BF16 shape=1500 file=mixb.bin
record codec=wbp1
tensor name=noise dtype=U8 shape=9000 file=noise.bin
record codec=wbp1 planes=2 chunk=10 transform=1 bytes=20000
tensor name=été.split dtype=BF16 shape=3,10000 file=big.bin
record codec=raw bytes=20000 part
record part
record codec=raw
tensor name=opaque dtype=200 shape=3,3 file=opaque.bin
tensor name=i8 dtype=I8 shape=600 file=i8.bin
empty name=empty.f32 dtype=F32 shape=0,5
empty name=empty.opaque dtype=77 shape=2,2
empty name= dtype=U8 shape=0
`
	if err := os.WriteFile(filepath.Join(dir, "manifest.txt"), []byte(manifest), 0o666); err != nil {
		t.Fatal(err)
	}
	m, err := ParseManifest(filepath.Join(dir, "manifest.txt"))
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "scratch.wex4")
	w, err := WriteFile(m, out)
	if err != nil {
		t.Fatal(err)
	}
	if w.Records != 13 || w.WBP1 != 10 || w.RAW != 3 || w.Fallbacks != 1 || w.Tensors != 17 {
		t.Errorf("the writer reports %+v", *w)
	}

	// the reader, on its own: every data tensor is the source file, byte for byte
	file := mustRead(t, out)
	got := map[string][]byte{}
	res, e := verifyBytes(file, func(i uint32, r Record, x *Index, dec []byte) {
		if r.TensorCount == 0 {
			name := string(x.Name(x.Tensor(r.FirstTensor)))
			got[name] = append(got[name], dec...)
		}
		for j := uint32(0); j < r.TensorCount; j++ {
			tn := x.Tensor(r.FirstTensor + j)
			part := dec
			if tn.RecordCount == 1 {
				part = dec[tn.Offset : tn.Offset+tn.NBytes]
			}
			got[string(x.Name(tn))] = append([]byte(nil), part...)
		}
	})
	if e != nil {
		t.Fatal(e)
	}
	if res.H.Minor != 3 || res.H.Optional != 0x10 || res.H.IndexOffset == pageBytes || string(res.X.Metadata()) != `{"made by":"the writer's test"}` {
		t.Errorf("header or metadata not as the manifest says: minor %d, optional 0x%x, index at %d", res.H.Minor, res.H.Optional, res.H.IndexOffset)
	}
	want := map[string]string{"a name%with\tescapes": "skew.bin", "bf16": "bf16.bin", "f32": "f32.bin", "f64": "f64.bin", "i64": "i64.bin",
		"ragged": "ragged.bin", "auto.f32": "f32b.bin", "auto.scalar": "one.bin", "mixed.u8": "mixa.bin", "mixed.bf16": "mixb.bin",
		"noise": "noise.bin", "été.split": "big.bin", "opaque": "opaque.bin", "i8": "i8.bin"}
	for name, f := range want {
		if !bytes.Equal(got[name], content[f]) {
			t.Errorf("tensor %q: %d bytes read back, they are not the %d bytes of %s", name, len(got[name]), len(content[f]), f)
		}
	}
	if len(got) != len(want) || res.H.Tensors-res.X.DataTensors != 3 {
		t.Errorf("%d data tensors and %d tensors of 0 bytes read back", len(got), res.H.Tensors-res.X.DataTensors)
	}
	// the choices left to the writer: planes 4 with the rotation for F32, 1 plane for mixed element sizes
	for rec, wantParams := range map[uint32][4]byte{6: {4, 12, 0, 1}, 7: {1, 12, 0, 0}, 11: {2, 12, 0, 1}} {
		r := res.X.Record(rec)
		p := file[r.Offset+12 : r.Offset+16]
		if r.Codec != codecWBP1 || p[0] != wantParams[0] || p[1] != wantParams[1] || p[3] != wantParams[3] {
			t.Errorf("record %d: codec %d, planes %d, chunk_log2 %d, transform %d", rec, r.Codec, p[0], p[1], p[3])
		}
	}
	reference(t, out, w, true)
}

// Expert records from the writer: the slabs of tiny-raw-upgraded.wex4 with the model of
// tiny2-cz-upgraded.wex4, as WNR3 with canonical zero, as WNR3 without (which is not smaller on this data
// and is stored RAW), as RAW; the two experts of testdata/ceil-form.wex4 with other chunk sizes; and the
// same two with canonical zero, which puts canonical zero and the ceil form in the same record.
func TestWriterExperts(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"tiny-raw-upgraded.wex4", "tiny2-cz-upgraded.wex4", "testdata/ceil-form.wex4"} {
		if _, err := Export(pathOf(name), filepath.Join(dir, filepath.Base(name))); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := ParseManifest(filepath.Join(dir, "tiny-raw-upgraded.wex4", "manifest.txt"))
	if err != nil {
		t.Fatal(err)
	}
	model := srcBlob{kind: blobKindWNM2, file: filepath.Join(dir, "tiny2-cz-upgraded.wex4", "b0.bin")}
	other := srcBlob{kind: 18, id: 7, file: filepath.Join(dir, "tiny-raw-upgraded.wex4", "b0.bin")}
	m := &Manifest{layers: 6, experts: 8, blobs: []srcBlob{model, other}, hasIndex: true}
	m.order = []extent{{'b', 0}}
	for k, set := range []struct {
		codec string
		model uint32
		cz    bool
	}{{"wnr3", 1, true}, {"wnr3", 1, false}, {"raw", 0, false}, {"auto", 1, true}, {"auto", 0, false}, {"wnr3", 1, true}} {
		r := raw.records[5*k] // expert records of the source, in owner order
		r.codec, r.model, r.cz, r.cw, r.cs = set.codec, set.model, set.cz, 2048, 512
		m.order = append(m.order, extent{'r', len(m.records)})
		m.records = append(m.records, r)
		if k == 2 {
			m.order = append(m.order, extent{'i', 0}, extent{'b', 1})
		}
	}
	out := filepath.Join(dir, "experts.wex4")
	w, err := WriteFile(m, out)
	if err != nil {
		t.Fatal(err)
	}
	if w.WNR3 != 3 || w.RAW != 3 || w.Fallbacks != 1 {
		t.Errorf("the writer reports %+v", *w)
	}
	res, e := verifyFile(out, nil)
	if e != nil || res.H.Required != 1 || res.X.ExpertRecords != 6 {
		t.Fatalf("the file of expert records: %v", e)
	}
	reference(t, out, w, false)

	// the non-CZ experts of ceil-form.wex4 at the smallest and at the largest chunks
	ceil, err := ParseManifest(filepath.Join(dir, "ceil-form.wex4", "manifest.txt"))
	if err != nil {
		t.Fatal(err)
	}
	ceil.records[0].cw, ceil.records[0].cs = 512, 256
	ceil.records[1].cw, ceil.records[1].cs = 16384, 16384
	out2 := filepath.Join(dir, "chunks.wex4")
	before := seenNow()
	w2, err := WriteFile(ceil, out2)
	if err != nil {
		t.Fatal(err)
	}
	if w2.WNR3 != 2 || fmt.Sprintf("%x", w2.Digest) != referenceDigest["testdata/ceil-form.wex4"] {
		t.Errorf("the writer reports %+v", *w2)
	}
	t.Logf("chunks 512:256 and 16384:16384, the decoder met: %s", diffSeen(before))
	reference(t, out2, w2, false)

	// canonical zero and the ceil form together: the source slabs of ceil-form.wex4 hold -0 nibbles
	cz, err := ParseManifest(filepath.Join(dir, "ceil-form.wex4", "manifest.txt"))
	if err != nil {
		t.Fatal(err)
	}
	cz.records[0].cz, cz.records[1].cz = true, true
	out3 := filepath.Join(dir, "cz-ceil.wex4")
	w3, err := WriteFile(cz, out3)
	if err != nil {
		t.Fatal(err)
	}
	file := mustRead(t, out3)
	res3, e := verifyBytes(file, nil)
	if e != nil || w3.WNR3 != 2 || res3.H.Required != 1 {
		t.Fatalf("canonical zero with the ceil form: %v, %+v", e, *w3)
	}
	for i, flags := range []byte{0x0f, 0x0d} { // CZ, and the CEIL bits of the two records of ceil-form.wex4
		r := res3.X.Record(uint32(i))
		if got := file[r.Offset+27]; got != flags || r.Flags != flagExpert|flagValueExact {
			t.Errorf("record %d: record flags 0x%02x, index flags %d", i, got, r.Flags)
		}
	}
	if w3.Digest == w2.Digest {
		t.Errorf("the content digest with canonical zero is the one without it, although the slabs hold -0 nibbles")
	}
	reference(t, out3, w3, false)
}

// The tensors of generic.wex4 packed with the writer's own choices and the index at the end: another file
// (another root) with the same content digest, the one the reference prints for the vector.
func TestWriterOtherChoices(t *testing.T) {
	out, w := rewrite(t, "generic.wex4", func(m *Manifest) {
		for k := range m.records {
			r := &m.records[k]
			r.codec, r.planes, r.chunk, r.transform = "auto", -1, -1, -1
		}
		m.order = append(m.order[1:], m.order[0]) // the index line was first
	})
	if got := fmt.Sprintf("%x", w.Digest); got != referenceDigest["generic.wex4"] {
		t.Errorf("content digest %s after repacking, the vector has %s", got, referenceDigest["generic.wex4"])
	}
	res, e := verifyFile(pathOf("generic.wex4"), nil)
	if e != nil || res.H.Root == w.Root || res.H.IndexOffset != pageBytes {
		t.Errorf("the repacked file has the root of the vector, or the vector is not index first: %v", e)
	}
	reference(t, out, w, true)

	// testdata/planes.wex4 was written by wex4/wex4_convert.py pack (testdata/README.md). With the grouping,
	// the order and the metadata taken from the file but every record left to the writer's defaults
	// (codec=auto, no planes, chunk or transform), the writer gives that file byte for byte: the converter's
	// chunk_log2, plane counts, rotation and RAW fallback are the writer's on this file
	out, _ = rewrite(t, "testdata/planes.wex4", func(m *Manifest) {
		for k := range m.records {
			r := &m.records[k]
			r.codec, r.planes, r.chunk, r.transform = "auto", -1, -1, -1
		}
	})
	if !bytes.Equal(mustRead(t, out), mustRead(t, pathOf("testdata/planes.wex4"))) {
		t.Errorf("testdata/planes.wex4 with the writer's defaults is not the file of pack")
	}
}

// Manifests the writer must refuse, each with one thing wrong.
func TestWriterRefuses(t *testing.T) {
	dir := t.TempDir()
	for name, n := range map[string]int{"a.bin": 8, "b.bin": 5000} {
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, n), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	cases := map[string]string{
		"size against shape":          "record\ntensor name=a dtype=F32 shape=3 file=a.bin\n",
		"duplicate name":              "record\ntensor name=a dtype=U8 shape=8 file=a.bin\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"NUL in a name":               "record\ntensor name=a%00b dtype=U8 shape=8 file=a.bin\n",
		"invalid UTF-8 name":          "record\ntensor name=%FF dtype=U8 shape=8 file=a.bin\n",
		"rank 9":                      "record\ntensor name=a dtype=U8 shape=1,1,1,1,1,1,1,1,8 file=a.bin\n",
		"shape overflow":              "record\ntensor name=a dtype=99 shape=4294967296,4294967296 file=a.bin\n",
		"record without tensor":       "record\n",
		"tensor before record":        "tensor name=a dtype=U8 shape=8 file=a.bin\n",
		"part without split":          "record part\n",
		"bytes without part":          "record bytes=4\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"parts do not add up":         "record bytes=4\ntensor name=a dtype=U8 shape=8 file=a.bin\nrecord part bytes=3\n",
		"transform with one plane":    "record codec=wbp1 planes=1 transform=1\ntensor name=a dtype=U8 shape=5000 file=b.bin\n",
		"3 planes":                    "record codec=wbp1 planes=3\ntensor name=a dtype=U8 shape=5000 file=b.bin\n",
		"chunk 9":                     "record codec=wbp1 chunk=9\ntensor name=a dtype=U8 shape=5000 file=b.bin\n",
		"WNR3 without expert":         "record codec=wnr3\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"expert with one tensor":      "grid 1 1\nrecord codec=raw expert=0:0\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"grid without expert record":  "record codec=raw\ntensor name=a dtype=U8 shape=8 file=a.bin\ngrid 2 2\n",
		"empty tensor with bytes":     "empty name=a dtype=F32 shape=3\n",
		"unknown keyword":             "tensors\n",
		"unknown key":                 "record colour=red\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"missing file":                "record\ntensor name=a dtype=U8 shape=8 file=none.bin\n",
		"bad escape":                  "record\ntensor name=a%2 dtype=U8 shape=8 file=a.bin\n",
		"unknown dtype name":          "record\ntensor name=a dtype=F128 shape=8 file=a.bin\n",
		"zero-size tensor in record":  "record\ntensor name=a dtype=U8 shape=8 file=a.bin\ntensor name=b dtype=U8 shape=0 file=empty.bin\n",
		"required bit a reader lacks": "required 2\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		// words that do not apply to the record's codec, and lines given twice
		"planes on a RAW record":             "record codec=raw planes=4\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"transform on WNR3":                  "grid 1 1\nrecord codec=wnr3 expert=0:0 model=1 transform=1\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"chunk on an auto expert":            "grid 1 1\nrecord expert=0:0 chunk=12\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"model on WBP1":                      "record codec=wbp1 model=1\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"cz on a RAW record":                 "record codec=raw cz\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"cw on an auto non-expert":           "record cw=512\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"cz on an auto expert without model": "grid 1 1\nrecord codec=auto expert=0:0 cz\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"meta twice":                         "meta a.bin\nmeta a.bin\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"grid twice":                         "grid 1 1\ngrid 0 0\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"minor twice":                        "minor 1\nminor 2\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"required twice":                     "required 0\nrequired 0\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
		"optional twice":                     "optional 0\noptional 1\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n",
	}
	if err := os.WriteFile(filepath.Join(dir, "empty.bin"), nil, 0o666); err != nil {
		t.Fatal(err)
	}
	for what, body := range cases {
		path := filepath.Join(dir, "m.txt")
		if err := os.WriteFile(path, []byte("wex4-manifest 1\n"+body), 0o666); err != nil {
			t.Fatal(err)
		}
		out := filepath.Join(dir, "out.wex4")
		m, err := ParseManifest(path)
		if err == nil && (strings.HasSuffix(what, " twice") || strings.Contains(what, " on ")) {
			t.Errorf("%s: the manifest is not refused when it is read", what)
		}
		if err == nil {
			_, err = WriteFile(m, out)
		}
		if err == nil {
			t.Errorf("%s: the writer made a file", what)
		}
		for _, left := range []string{out, out + ".tmp"} {
			if _, serr := os.Stat(left); serr == nil {
				t.Errorf("%s: %s was left behind", what, filepath.Base(left))
				os.Remove(left)
			}
		}
	}
	// and the smallest good one, so that the cases above fail for their reason and not for the frame
	path := filepath.Join(dir, "m.txt")
	os.WriteFile(path, []byte("wex4-manifest 1\nrecord\ntensor name=a dtype=U8 shape=8 file=a.bin\n"), 0o666)
	m, err := ParseManifest(path)
	if err == nil {
		_, err = WriteFile(m, filepath.Join(dir, "good.wex4"))
	}
	if err != nil {
		t.Errorf("the smallest manifest: %v", err)
	}
	// a file with no tensor at all: a header page and nothing else
	os.WriteFile(path, []byte("wex4-manifest 1\n"), 0o666)
	if m, err = ParseManifest(path); err == nil {
		var w *Written
		if w, err = WriteFile(m, filepath.Join(dir, "nothing.wex4")); err == nil && w.FileBytes != pageBytes {
			t.Errorf("a file of no tensors has %d bytes", w.FileBytes)
		} else if err == nil {
			reference(t, filepath.Join(dir, "nothing.wex4"), w, true)
		}
	}
	if err != nil {
		t.Errorf("the empty manifest: %v", err)
	}
}

// The quantiser and the stream writer against the decoder, on counts that the files do not have: one
// symbol only, every symbol once, a count far above the others.
func TestQuantizeAndStreams(t *testing.T) {
	for k, plane := range [][]byte{bytes.Repeat([]byte{255}, 5000), genBytes("rand", 256, 9), append(bytes.Repeat([]byte{0}, 70000), genBytes("rand", 3000, 5)...)} {
		var counts [256]uint64
		for _, c := range plane {
			counts[c]++
		}
		cdf := quantize(counts[:], uint64(len(plane)))
		if !validCDF(cdf) {
			t.Fatalf("case %d: the quantiser gave a CDF that is not strictly increasing from 0 to 16384", k)
		}
		for _, planes := range []uint64{1, 2, 4, 8} {
			for chunk := uint(10); chunk <= 15; chunk++ {
				rec := encodeWBP1(plane, 77, planes, chunk, 0)
				back := make([]byte, len(plane))
				if e := decodeWBP1(rec, 77, back); e != nil || !bytes.Equal(back, plane) {
					t.Fatalf("case %d, %d planes, chunk_log2 %d: %v", k, planes, chunk, e)
				}
				if e := decodeWBP1(rec, 78, back); e == nil {
					t.Fatalf("case %d: the record decodes under another owner", k)
				}
			}
		}
	}
}

// The trainer against the two models of the reference that are in the files: trained on the 2 source slabs
// of testdata/ceil-form.wex4 (testdata/README.md: build_nr3.py --train 0:0,0:1, no canonical zero) and on
// the 48 slabs of tiny-raw-upgraded.wex4 with canonical zero, the models are the blobs of ceil-form.wex4
// and of tiny2-cz-upgraded.wex4, byte for byte. With the trained models in place of the exported ones the
// writer gives both files again; and 48 WNR3-CZ records written from nothing but the slabs and the trained
// model pass the reference.
func TestTrain(t *testing.T) {
	dir := t.TempDir()
	exported := map[string]*Manifest{}
	for sub, name := range map[string]string{"tiny": "tiny-raw-upgraded.wex4", "tiny2": "tiny2-cz-upgraded.wex4", "ceil": "testdata/ceil-form.wex4"} {
		if _, err := Export(pathOf(name), filepath.Join(dir, sub)); err != nil {
			t.Fatal(err)
		}
		m, err := ParseManifest(filepath.Join(dir, sub, "manifest.txt"))
		if err != nil {
			t.Fatal(err)
		}
		exported[sub] = m
	}

	// ceil-form.wex4: its records name their model; it is trained on them as they are
	ceil := exported["ceil"]
	n := ceil.records[0].model
	trained := filepath.Join(dir, "ceil.wnm2")
	tr, err := TrainFile(ceil, n, trained)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(mustRead(t, trained), mustRead(t, ceil.blobs[n-1].file)) || tr.Records != 2 || tr.CZ != 0 || tr.EmptyContexts == 0 {
		t.Fatalf("the model trained on the slabs of ceil-form.wex4 is not its blob: %+v", *tr)
	}
	t.Logf("ceil-form.wex4: %+v", *tr)
	ceil.blobs[n-1].file = trained
	out := filepath.Join(dir, "ceil-again.wex4")
	if _, err := WriteFile(ceil, out); err != nil || !bytes.Equal(mustRead(t, out), mustRead(t, pathOf("testdata/ceil-form.wex4"))) {
		t.Fatalf("ceil-form.wex4 with the trained model is not written again byte for byte: %v", err)
	}

	// the 48 slabs of tiny-raw-upgraded.wex4 as WNR3 records with canonical zero, in a file of their own
	m := &Manifest{layers: 6, experts: 8, blobs: []srcBlob{{kind: blobKindWNM2, file: filepath.Join(dir, "tiny.wnm2")}}}
	m.order = []extent{{'b', 0}}
	for _, r := range exported["tiny"].records {
		r.codec, r.model, r.cz, r.cw, r.cs = "wnr3", 1, true, 2048, 512
		m.order = append(m.order, extent{'r', len(m.records)})
		m.records = append(m.records, r)
	}
	if tr, err = TrainFile(m, 1, m.blobs[0].file); err != nil {
		t.Fatal(err)
	}
	tiny2 := exported["tiny2"]
	n = tiny2.records[0].model
	if !bytes.Equal(mustRead(t, m.blobs[0].file), mustRead(t, tiny2.blobs[n-1].file)) || tr.Records != 48 || tr.CZ != 48 {
		t.Fatalf("the model trained on the 48 slabs of tiny-raw-upgraded.wex4 is not the model of tiny2-cz-upgraded.wex4: %+v", *tr)
	}
	t.Logf("tiny2-cz-upgraded.wex4: %+v", *tr)
	tiny2.blobs[n-1].file = m.blobs[0].file
	out = filepath.Join(dir, "tiny2-again.wex4")
	if _, err := WriteFile(tiny2, out); err != nil || !bytes.Equal(mustRead(t, out), mustRead(t, pathOf("tiny2-cz-upgraded.wex4"))) {
		t.Fatalf("tiny2-cz-upgraded.wex4 with the trained model is not written again byte for byte: %v", err)
	}
	out = filepath.Join(dir, "tiny-cz.wex4")
	w, err := WriteFile(m, out)
	if err != nil {
		t.Fatal(err)
	}
	if w.WNR3+w.Fallbacks != 48 || w.WNR3 == 0 {
		t.Errorf("48 expert records asked as WNR3: %+v", *w)
	}
	t.Logf("48 slabs with canonical zero and the trained model: %d WNR3 records, %d stored RAW because not smaller, %d B", w.WNR3, w.Fallbacks, w.FileBytes)
	reference(t, out, w, false)

	// a manifest whose blob line is not a model, or whose experts do not name it, is refused
	if _, err := TrainFile(m, 2, filepath.Join(dir, "none.wnm2")); err == nil {
		t.Error("a model trained for a blob line that does not exist")
	}
	m.records = m.records[:0]
	if _, err := TrainFile(m, 1, filepath.Join(dir, "none.wnm2")); err == nil {
		t.Error("a model trained on no expert record")
	}

	// tensor files shorter than their shapes are refused, as the writer refuses them, and not trained on;
	// the expert has 2 x 2 scales (4 B a matrix) and 2 x 32 weight bytes (64 B a matrix)
	for name, n := range map[string]int{"s4.bin": 4, "s3.bin": 3, "w64.bin": 64, "w60.bin": 60, "w1.bin": 1} {
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, n), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct{ scale, weight string }{{"s4", "w64"}, {"s3", "w64"}, {"s4", "w60"}, {"s4", "w1"}} {
		body := "wex4-manifest 1\ngrid 1 1\nblob kind=1 file=none.bin\nrecord codec=wnr3 expert=0:0 model=1\n"
		for k := 0; k < 3; k++ {
			body += fmt.Sprintf("tensor name=s%d dtype=U8 shape=2,2 file=%s.bin\n", k, c.scale)
		}
		for k := 0; k < 3; k++ {
			body += fmt.Sprintf("tensor name=w%d dtype=U8 shape=2,32 file=%s.bin\n", k, c.weight)
		}
		path := filepath.Join(dir, "short.txt")
		if err := os.WriteFile(path, []byte(body), 0o666); err != nil {
			t.Fatal(err)
		}
		sm, err := ParseManifest(path)
		if err != nil {
			t.Fatal(err)
		}
		_, err = TrainFile(sm, 1, filepath.Join(dir, "short.wnm2"))
		if good := c.scale == "s4" && c.weight == "w64"; good != (err == nil) {
			t.Errorf("scales of %s, weights of %s: trained %v, error %v", c.scale, c.weight, err == nil, err)
		}
	}
}

// A file that fails verification is refused before export writes anything: the last record of
// generic.wex4 with one byte changed.
func TestExportRefused(t *testing.T) {
	b := mustRead(t, pathOf("generic.wex4"))
	res, e := verifyBytes(b, nil)
	if e != nil {
		t.Fatal(e)
	}
	b[res.X.Record(res.H.Records-1).Offset] ^= 1
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.wex4")
	if err := os.WriteFile(bad, b, 0o666); err != nil {
		t.Fatal(err)
	}
	if _, err := Export(bad, filepath.Join(dir, "out")); err == nil {
		t.Fatal("a file with a wrong record hash was exported")
	}
	if _, err := os.Stat(filepath.Join(dir, "out")); err == nil {
		t.Error("export of a refused file left its folder behind")
	}
}

func envUint(t *testing.T, name string, def uint64) uint64 {
	s := os.Getenv(name)
	if s == "" {
		return def
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		t.Fatalf("%s=%s is not a number", name, s)
	}
	return v
}

// Random manifests through the writer, then through the Go reader and the reference readers. The manifests
// mix every kind of line: records of the three codecs with random parameters or with the writer's own
// choices, split tensors whose parts have codecs of their own, typed and opaque dtypes, tensors of 0 bytes,
// blobs, metadata, version and feature words, the index at a random place, and expert records made of the
// slabs of tiny-raw-upgraded.wex4 and testdata/ceil-form.wex4 with random chunks, with and without
// canonical zero. WEX4_RANDOM_FILES is the number of files (default 24), WEX4_RANDOM_SEED the seed.
func TestWriterRandom(t *testing.T) {
	files, seed := envUint(t, "WEX4_RANDOM_FILES", 24), envUint(t, "WEX4_RANDOM_SEED", 20261115)
	dir := t.TempDir()
	exported := map[string]*Manifest{}
	for sub, name := range map[string]string{"tiny": "tiny-raw-upgraded.wex4", "tiny2": "tiny2-cz-upgraded.wex4", "ceil": "testdata/ceil-form.wex4"} {
		if _, err := Export(pathOf(name), filepath.Join(dir, sub)); err != nil {
			t.Fatal(err)
		}
		m, err := ParseManifest(filepath.Join(dir, sub, "manifest.txt"))
		if err != nil {
			t.Fatal(err)
		}
		exported[sub] = m
	}
	modelOf := func(sub string) string {
		for _, b := range exported[sub].blobs {
			if b.kind == blobKindWNM2 {
				return "../" + sub + "/" + filepath.Base(b.file)
			}
		}
		t.Fatalf("%s has no WNM2 blob", sub)
		return ""
	}
	// the expert slabs, with the blob line of their model in the manifests below (1 or 2)
	type slab struct {
		tensors []srcTensor
		sub     string
		model   int
	}
	var pools [2][]slab // the 48 slabs of tiny-raw-upgraded.wex4, the 2 of ceil-form.wex4
	for k, s := range []slab{{sub: "tiny", model: 1}, {sub: "ceil", model: 2}} {
		for _, r := range exported[s.sub].records {
			if r.expert {
				pools[k] = append(pools[k], slab{r.tensors, s.sub, s.model})
			}
		}
	}
	models := []string{modelOf("tiny2"), modelOf("ceil")}
	typed := []struct {
		dtype string
		bytes uint64 // per element; 0 for F4, two elements in a byte
	}{{"U8", 1}, {"I8", 1}, {"BOOL", 1}, {"F8_E4M3", 1}, {"BF16", 2}, {"F16", 2}, {"I16", 2}, {"F32", 4},
		{"I32", 4}, {"F64", 8}, {"I64", 8}, {"C64", 8}, {"F4", 0}}
	kinds := []string{"rand", "skew", "f32", "f64", "bf16", "i64", "const", "ramp"}

	p := prng(seed*2654435761 + 1)
	pick := func(n uint64) uint64 { return (p.next() >> 11) % n }
	chance := func(percent uint64) bool { return pick(100) < percent }
	data := func(n uint64) []byte {
		switch k := kinds[pick(uint64(len(kinds)))]; k {
		case "const":
			return bytes.Repeat([]byte{byte(p.next())}, int(n))
		case "ramp":
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(i*3 + i>>5)
			}
			return b
		default:
			return genBytes(k, int(n), p.next())
		}
	}
	size := func(unit uint64) uint64 { // a multiple of unit, mostly small
		top := []uint64{64, 5000, 40000, 150000}[pick(4)]
		return unit * (1 + pick(max(1, top/unit)))
	}
	shapeOf := func() string { // of an opaque tensor: anything of rank 0 to 3
		dims := make([]string, pick(4))
		for k := range dims {
			dims[k] = fmt.Sprint(pick(10))
		}
		return strings.Join(dims, ",")
	}
	wbp1 := func(unit *uint64) string { // WBP1 parameters; unit: what the sizes must be a multiple of
		planes := uint64(1) << pick(4)
		tr := 0
		if planes > 1 && chance(50) {
			tr, *unit = 1, planes
		}
		return fmt.Sprintf(" planes=%d chunk=%d transform=%d", planes, 10+pick(6), tr)
	}

	var stats struct {
		Files, Records, RAW, WBP1, WNR3, Fallbacks, Tensors, Empty, Split, CZ, CZCeil, Alone int
		MaxFile, MaxRecord                                                                   uint64
	}
	for it := uint64(0); it < files; it++ {
		fd := filepath.Join(dir, fmt.Sprintf("f%03d", it))
		if err := os.MkdirAll(fd, 0o777); err != nil {
			t.Fatal(err)
		}
		save := func(name string, b []byte) {
			if err := os.WriteFile(filepath.Join(fd, name), b, 0o666); err != nil {
				t.Fatal(err)
			}
		}
		type expect struct {
			src    []byte
			weight bool // a weight tensor of an expert record: Z(src) when its record is value-exact
		}
		want := map[string]expect{}
		head := []string{"wex4-manifest 1"}
		var body []string // blob, record, tensor and index lines, in the order of the extents
		if chance(20) {
			head = append(head, fmt.Sprintf("minor %d", pick(10)))
		}
		if chance(20) {
			head = append(head, fmt.Sprintf("optional 0x%x", p.next()>>pick(64)))
		}
		if chance(30) {
			save("meta.bin", genBytes("rand", int(1+pick(300)), p.next()))
			head = append(head, "meta meta.bin")
		}
		experts, perLayer := 0, uint64(0)
		var slots []uint64
		if chance(40) {
			experts = int(1 + pick(3))
			layers := 1 + pick(4)
			perLayer = 1 + pick(8)
			for uint64(experts) > layers*perLayer {
				perLayer++
			}
			head = append(head, fmt.Sprintf("grid %d %d", layers, perLayer))
			body = append(body, "blob kind=1 file="+models[0], "blob kind=1 file="+models[1])
			taken := map[uint64]bool{}
			for len(slots) < experts {
				if s := pick(layers * perLayer); !taken[s] {
					taken[s] = true
					slots = append(slots, s)
				}
			}
			sort.Slice(slots, func(a, b int) bool { return slots[a] < slots[b] })
		}
		if chance(30) {
			save("blob.bin", genBytes("rand", int(1+pick(9000)), p.next()))
			body = append(body, fmt.Sprintf("blob kind=%d id=%d file=blob.bin", 2+pick(30), p.next()>>32))
		}
		tno, ek := 0, 0
		nrec := int(1+pick(5)) + experts
		for r := 0; r < nrec; r++ {
			switch {
			case ek < experts && pick(uint64(nrec-r)) < uint64(experts-ek): // an expert record
				pool := pools[pick(2)]
				s := pool[pick(uint64(len(pool)))]
				slot := slots[ek]
				ek++
				codec := []string{"wnr3", "wnr3", "raw", "auto"}[pick(4)]
				line := fmt.Sprintf("record codec=%s expert=%d:%d", codec, slot/perLayer, slot%perLayer)
				if codec != "raw" {
					line += fmt.Sprintf(" model=%d cw=%d cs=%d", s.model, 512<<pick(6), 256<<pick(7))
					if chance(50) {
						line += " cz"
					}
				}
				body = append(body, line)
				for k, st := range s.tensors {
					name := fmt.Sprintf("x%d.%d", tno, k)
					body = append(body, fmt.Sprintf("tensor name=%s dtype=%s shape=%d,%d file=../%s/%s",
						name, dtypeNames[st.dtype], st.shape[0], st.shape[1], s.sub, filepath.Base(st.file)))
					want[name] = expect{mustRead(t, st.file), k >= 3}
				}
			case chance(20): // a split tensor of 2 to 4 parts, each with its own codec
				parts := 2 + pick(3)
				var lines []string
				var all []byte
				for k := uint64(0); k < parts; k++ {
					unit := uint64(1)
					line := "record codec=" + []string{"raw", "wbp1", "auto"}[pick(3)]
					if strings.HasSuffix(line, "wbp1") && chance(70) {
						line += wbp1(&unit)
					}
					n := size(unit)
					all = append(all, data(n)...)
					if k+1 < parts || chance(50) {
						line += fmt.Sprintf(" bytes=%d", n)
					}
					if k > 0 {
						line += " part"
					}
					lines = append(lines, line)
				}
				name := fmt.Sprintf("s%d", tno)
				save(name+".bin", all)
				dtype, shape := "U8", fmt.Sprint(len(all))
				if chance(50) {
					dtype, shape = fmt.Sprint(23+pick(233)), shapeOf()
				}
				body = append(body, lines[0], fmt.Sprintf("tensor name=%s dtype=%s shape=%s file=%s.bin", name, dtype, shape, name))
				body = append(body, lines[1:]...)
				want[name] = expect{all, false}
				stats.Split++
			default: // a record of 1 to 4 tensors
				unit := uint64(1)
				codec := []string{"raw", "wbp1", "wbp1", "wbp1", "auto"}[pick(5)]
				line := "record codec=" + codec
				own := codec == "auto" || (codec == "wbp1" && chance(40)) // the writer picks the parameters
				if codec == "wbp1" && !own {
					line += wbp1(&unit)
				}
				body = append(body, line)
				same := typed[pick(uint64(len(typed)))]
				for k := 1 + pick(4); k > 0; k-- {
					name := fmt.Sprintf("n%d", tno) + []string{"", " with spaces", ".été", "%", "\t"}[pick(5)]
					var dtype, shape string
					var n uint64
					switch {
					case own && chance(80): // typed, so that the writer's choice of parameters depends on it
						ty := same
						if chance(30) {
							ty = typed[pick(uint64(len(typed)))]
						}
						dtype = ty.dtype
						if ty.bytes == 0 {
							n = size(1)
							shape = fmt.Sprint(2 * n)
						} else if n = size(ty.bytes); n/ty.bytes%4 == 0 && chance(50) {
							shape = fmt.Sprintf("4,%d", n/ty.bytes/4)
						} else {
							shape = fmt.Sprint(n / ty.bytes)
						}
					case chance(20):
						dtype, n, shape = fmt.Sprint(23+pick(233)), size(unit), shapeOf()
					default:
						n = size(unit)
						dtype, shape = "U8", fmt.Sprint(n)
					}
					b := data(n)
					save(fmt.Sprintf("t%d.bin", tno), b)
					body = append(body, fmt.Sprintf("tensor name=%s dtype=%s shape=%s file=t%d.bin", escape([]byte(name)), dtype, shape, tno))
					want[name] = expect{b, false}
					tno++
				}
			}
			tno++
		}
		empties := int(pick(3))
		for k := 0; k < empties; k++ {
			if chance(50) {
				body = append(body, fmt.Sprintf("empty name=z%d dtype=F32 shape=0,%d", k, pick(5)))
			} else { // an opaque tensor of 0 bytes, whose shape need not have a 0 (SPEC-NOTES A6)
				body = append(body, fmt.Sprintf("empty name=z%d dtype=%d shape=%s", k, 23+pick(233), shapeOf()))
			}
		}
		if chance(50) { // the index at a random place; without the line it follows the header page
			var at []int
			for k, l := range body {
				if strings.HasPrefix(l, "record") || strings.HasPrefix(l, "blob") {
					at = append(at, k)
				}
			}
			k := append(at, len(body))[pick(uint64(len(at)+1))]
			body = append(body[:k], append([]string{"index"}, body[k:]...)...)
		}
		text := strings.Join(append(head, body...), "\n") + "\n"
		save("manifest.txt", []byte(text))
		m, err := ParseManifest(filepath.Join(fd, "manifest.txt"))
		if err != nil {
			t.Fatalf("file %d: %v\n%s", it, err, text)
		}
		out := filepath.Join(fd, "out.wex4")
		w, err := WriteFile(m, out)
		if err != nil {
			t.Fatalf("file %d: %v\n%s", it, err, text)
		}

		// the Go reader: every tensor is its source, or Z(source) for the weights of a value-exact record
		file := mustRead(t, out)
		got, exact := map[string][]byte{}, map[string]bool{}
		res, e := verifyBytes(file, func(i uint32, r Record, x *Index, dec []byte) {
			if r.TensorCount == 0 {
				name := string(x.Name(x.Tensor(r.FirstTensor)))
				got[name] = append(got[name], dec...)
				return
			}
			for j := uint32(0); j < r.TensorCount; j++ {
				tn := x.Tensor(r.FirstTensor + j)
				part := dec
				if tn.RecordCount == 1 {
					part = dec[tn.Offset : tn.Offset+tn.NBytes]
				}
				got[string(x.Name(tn))] = append([]byte(nil), part...)
				exact[string(x.Name(tn))] = r.Flags&flagValueExact != 0
			}
		})
		if e != nil || res.Digest != w.Digest {
			t.Fatalf("file %d: the reader: %v\n%s", it, e, text)
		}
		for name, x := range want {
			src := x.src
			if x.weight && exact[name] { // Z, written here apart from the writer's: nibble by nibble
				z := make([]byte, len(src))
				for k := 0; k < 2*len(src); k++ {
					if q := src[k/2] >> (4 * (k & 1)) & 15; q != 8 {
						z[k/2] |= q << (4 * (k & 1))
					}
				}
				src = z
			}
			if !bytes.Equal(got[name], src) {
				t.Fatalf("file %d, tensor %q: %d bytes read back, not the %d of the source\n%s", it, name, len(got[name]), len(src), text)
			}
		}
		if len(got) != len(want) || int(res.H.Tensors-res.X.DataTensors) != empties {
			t.Fatalf("file %d: %d data tensors and %d of 0 bytes, the manifest has %d and %d", it, len(got), res.H.Tensors-res.X.DataTensors, len(want), empties)
		}
		for i := uint32(0); i < res.H.Records; i++ {
			if r := res.X.Record(i); r.Codec == codecWNR3 && file[r.Offset+27]&wnr3FlagCZ != 0 {
				stats.CZ++
				if file[r.Offset+27]&7 != 0 {
					stats.CZCeil++
				}
			}
		}
		stats.Files++
		stats.Records += w.Records
		stats.RAW += w.RAW
		stats.WBP1 += w.WBP1
		stats.WNR3 += w.WNR3
		stats.Fallbacks += w.Fallbacks
		stats.Tensors += w.Tensors
		stats.Empty += empties
		stats.MaxFile, stats.MaxRecord = max(stats.MaxFile, w.FileBytes), max(stats.MaxRecord, uint64(res.H.MaxDecoded))
		alone := w.WNR3 == 0 && w.FileBytes <= 300000
		if alone {
			stats.Alone++
		}
		reference(t, out, w, alone)
		os.RemoveAll(fd)
	}
	t.Logf("%+v", stats)
}
