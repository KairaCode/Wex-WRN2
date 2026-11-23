// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Codec 8 in whole files: the codec-8 files of testdata/ (wnr4-MANIFEST.tsv, wnr4-records.tsv), made with the
// repository's codec-8 tools, through Verify, the command line's verdicts and Export; and changed copies of
// them with every hash recomputed, which must never be refused as files (WEX4-SPEC 7.3).

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func wnr4Rows(t *testing.T, name string) [][]string {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows [][]string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if l := sc.Text(); l != "" && !strings.HasPrefix(l, "#") {
			rows = append(rows, strings.Split(l, "\t"))
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return rows
}

// wnr4Why is the reason this reader must give for each changed copy that has an undecodable record: the
// check the change aims at (the message of the decoder or of wnr4_wex4.go).
var wnr4Why = map[string]string{
	"wnr4-gg-bad-word.wex4":     "(tile 0)",
	"wnr4-gg-bad-kind0.wex4":    "WNR4 kind 0",
	"wnr4-gg-bad-flags.wex4":    "kind 2 flags 0x01",
	"wnr4-gg-bad-reserved.wex4": "16..19 or 28..63",
	"wnr4-gg-bad-ntiles.wex4":   "2 tiles in the record, the geometry gives 1",
	"wnr4-gg-bad-owner.wex4":    "owner 0x00000001",
	"wnr4-gg-bad-model.wex4":    "model id",
	"wnr4-gg-bad-zero16.wex4":   "16..19 or 28..63",
	"wnr4-gg-bad-pad.wex4":      "padding of the verbatim area",
	"wnr4-gg-bad-modelid.wex4":  "in the index, the model's FNV-1a",
	"wnr4-gg-bad-blob0.wex4":    "without a WNM4 blob",
	"wnr4-i4-bad-word.wex4":     "(tile 0)",
	"wnr4-i4-bad-base.wex4":     "base 0 is 0x7f800000",
	"wnr4-i4-bad-flags.wex4":    "(tile 0)",
	"wnr4-i4-bad-anchor.wex4":   "anchor of matrix 1",
	"wnr4-i4-bad-owner.wex4":    "owner 0x00000001",
}

// TestWNR4ContainerManifest: every row of wnr4-MANIFEST.tsv. The manifest says accept or refuse, as the
// codec-8 tool said; for this reader a refused copy is a file with an undecodable record (exit code 3),
// except bad-sha, whose record SHA-256 does not match and which is refused (WEX4_E_HASH, exit code 1).
func TestWNR4ContainerManifest(t *testing.T) {
	rows := wnr4Rows(t, "wnr4-MANIFEST.tsv")
	if len(rows) != 23 {
		t.Fatalf("%d rows in wnr4-MANIFEST.tsv, 23 expected", len(rows))
	}
	for _, c := range rows {
		name, size, root, verdict, what := c[0], c[1], c[2], c[3], c[4]
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		if utoa(uint32(len(b))) != size {
			t.Errorf("%s: %d bytes, the manifest says %s", name, len(b), size)
		}
		if got := hex.EncodeToString(b[4064:4096]); got != root {
			t.Errorf("%s: root %s, the manifest says %s", name, got, root)
		}
		res, e := verifyBytes(b, nil)
		switch {
		case verdict == "accept":
			want := strings.Fields(what)[0]
			if e != nil || res.Undecoded != 0 {
				t.Errorf("%s: %v %v, the manifest says accept", name, e, res)
			} else if got := hex.EncodeToString(res.Digest[:]); got != want {
				t.Errorf("%s: content digest %s, the manifest says %s", name, got, want)
			}
		case what == "record hash mismatch":
			if e == nil || e.Class != EHash {
				t.Errorf("%s: %v, WEX4_E_HASH expected", name, e)
			}
		default:
			if e != nil {
				t.Errorf("%s: refused (%v): a codec-8 record that does not decode makes the record undecodable, not the file invalid", name, e)
			} else if res.Undecoded != 1 || len(res.Why) != 1 || !strings.Contains(res.Why[0], wnr4Why[name]) || wnr4Why[name] == "" {
				t.Errorf("%s: %d undecodable records %q, one expected for %q", name, res.Undecoded, res.Why, wnr4Why[name])
			}
		}
	}
}

// TestWNR4ContainerRecords: the SHA-256 of every decoded record of the four valid files, against
// wnr4-records.tsv (so that a wrong record is named, not only a wrong content digest).
func TestWNR4ContainerRecords(t *testing.T) {
	want := map[string][]string{}
	for _, c := range wnr4Rows(t, "wnr4-records.tsv") {
		want[c[0]] = append(want[c[0]], c[4])
	}
	if len(want) != 4 {
		t.Fatalf("%d files in wnr4-records.tsv, 4 expected", len(want))
	}
	before := SeenWNR4.Records
	coded := 0
	for name, sums := range want {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		res, e := verifyBytes(b, func(i uint32, r Record, x *Index, dec []byte) {
			s := sha256.Sum256(dec)
			got = append(got, hex.EncodeToString(s[:]))
			if r.Codec == codecWNR4 {
				coded++
			}
		})
		if e != nil || res.Undecoded != 0 {
			t.Fatalf("%s: %v %v", name, e, res)
		}
		if strings.Join(got, " ") != strings.Join(sums, " ") {
			t.Errorf("%s: decoded records\n%v\nwnr4-records.tsv says\n%v", name, got, sums)
		}
	}
	if coded != 13 || SeenWNR4.Records-before != 13 {
		t.Errorf("%d codec-8 records decoded (%d by the decoders), 13 expected", coded, SeenWNR4.Records-before)
	}
}

// wnr4Rehash recomputes the SHA-256 of record rec (when rec >= 0) and of blob blob (when blob >= 0), then the
// index hash and the header hash, so that a change reaches the decoders.
func wnr4Rehash(b []byte, rec, blob int) {
	io, ib, nb := le.Uint64(b[32:]), le.Uint64(b[40:]), uint64(le.Uint32(b[48:]))
	if blob >= 0 {
		e := b[io+64*uint64(blob):]
		off, n := le.Uint64(e[8:]), le.Uint64(e[16:])
		s := sha256.Sum256(b[off : off+n])
		copy(e[24:56], s[:])
	}
	if rec >= 0 {
		e := b[io+64*nb+72*uint64(rec):]
		off, n := le.Uint64(e[0:]), uint64(le.Uint32(e[8:]))
		s := sha256.Sum256(b[off : off+n])
		copy(e[36:68], s[:])
	}
	s := sha256.Sum256(b[io : io+ib])
	copy(b[96:128], s[:])
	s = sha256.Sum256(b[:4064])
	copy(b[4064:4096], s[:])
}

// wnr4Area is the kind's area of the record (GG: the verbatim slices, I4: the bases) as [from, to) in the
// stored record, from the file's own geometry.
func wnr4Area(t *testing.T, b []byte) (rec Record, from, to uint64) {
	t.Helper()
	page := b[:pageBytes]
	h, e := ParseHeader(page, uint64(len(b)))
	if e != nil {
		t.Fatal(e)
	}
	x, e := OpenIndex(h, b[h.IndexOffset:h.IndexOffset+align(h.IndexBytes)])
	if e != nil {
		t.Fatal(e)
	}
	rec = x.Record(0)
	stored := b[rec.Offset : rec.Offset+uint64(rec.Stored)]
	from = wnr4Head + 4*(uint64(le.Uint32(stored[20:]))+1)
	switch stored[12] {
	case ggKind:
		sl, e := x.wnr4GGSlices(rec)
		if e != nil {
			t.Fatal(e)
		}
		_, v, e := ggGeometry(sl)
		if e != nil {
			t.Fatal(e)
		}
		to = from + (v+3)&^3
	case i4Kind:
		g, e := x.wnr4I4Geo(rec, stored[13])
		if e != nil {
			t.Fatal(e)
		}
		L, e := i4Geometry(g)
		if e != nil {
			t.Fatal(e)
		}
		to = from + 4*L.nbases
	}
	return rec, from, to
}

// TestWNR4ContainerFlips: single-bit flips of the codec-8 record of wnr4-gg-one and wnr4-i4-one, the record
// SHA-256 and the index and header hashes recomputed: every bit of the first 512 bytes and 2000 random bits
// after them. No flip may make the file refused (WEX4-SPEC 7.3) or the reader panic. A flip is accepted only
// inside the kind's area: in the GG verbatim slices (another content digest; only the record SHA-256 covers
// them) or in an I4 base that no scale reads (the same content digest, WNR4-SPEC 6.5); anywhere else the
// record is undecodable.
func TestWNR4ContainerFlips(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	for _, name := range []string{"wnr4-gg-one.wex4", "wnr4-i4-one.wex4"} {
		orig, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		res0, e := verifyBytes(orig, nil)
		if e != nil || res0.Undecoded != 0 {
			t.Fatalf("%s: %v", name, e)
		}
		rec, from, to := wnr4Area(t, orig)
		if rec.Codec != codecWNR4 || to <= from {
			t.Fatalf("%s: record 0 is codec %d, area [%d, %d)", name, rec.Codec, from, to)
		}
		gg := orig[rec.Offset+12] == ggKind
		var bits []uint64
		for k := uint64(0); k < 512*8; k++ {
			bits = append(bits, k)
		}
		rng := rand.New(rand.NewSource(20261123))
		for k := 0; k < 2000; k++ {
			bits = append(bits, 512*8+uint64(rng.Int63n(int64(rec.Stored-512)*8)))
		}
		b := make([]byte, len(orig))
		accepted, undecodable := 0, 0
		for _, k := range bits {
			copy(b, orig)
			at := k / 8
			b[rec.Offset+at] ^= 1 << (k % 8)
			wnr4Rehash(b, 0, -1)
			res, e := verifyBytes(b, nil)
			switch {
			case e != nil:
				t.Fatalf("%s: byte %d bit %d: the file is refused: %v", name, at, k%8, e)
			case res.Undecoded == 0:
				accepted++
				same := res.Digest == res0.Digest
				if at < from || at >= to || same == gg {
					t.Errorf("%s: byte %d bit %d accepted (area [%d, %d), same content digest %v)", name, at, k%8, from, to, same)
				}
			default:
				undecodable++
			}
		}
		t.Logf("%s: %d flips: %d accepted (in the area [%d, %d)), %d undecodable", name, len(bits), accepted, from, to, undecodable)
		if accepted == 0 && gg {
			t.Errorf("%s: no verbatim flip was accepted", name)
		}
	}
}

// TestWNR4ContainerBindings: changes of the index and of the WNM4 blob, every hash recomputed. A record of
// kind 1 (NV), a record whose blob is of another kind, a model blob changed (its id then no longer the FNV-1a)
// and a G of 0 make the record undecodable; the file opens.
func TestWNR4ContainerBindings(t *testing.T) {
	orig, err := os.ReadFile(filepath.Join("testdata", "wnr4-gg-one.wex4"))
	if err != nil {
		t.Fatal(err)
	}
	rec, _, _ := wnr4Area(t, orig)
	io := le.Uint64(orig[32:])
	blob := int(rec.Blob) - 1
	cases := []struct {
		name, why string
		edit      func(b []byte) (rec, blob int)
	}{
		{"kind 1 (NV)", "WNR4 kind 1", func(b []byte) (int, int) { b[rec.Offset+12] = 1; return 0, -1 }},
		{"kind 3 on a GG record", "blob ", func(b []byte) (int, int) { b[rec.Offset+12] = 3; return 0, -1 }},
		{"blob of kind 4", "not a WNM4 model", func(b []byte) (int, int) { b[io+64*uint64(blob)] = 4; return -1, -1 }},
		{"model byte", "blob", func(b []byte) (int, int) {
			off := le.Uint64(b[io+64*uint64(blob)+8:])
			b[off+100] ^= 1
			return -1, blob
		}},
		{"G 0", "G = 0", func(b []byte) (int, int) { b[rec.Offset+14], b[rec.Offset+15] = 0, 0; return 0, -1 }},
	}
	for _, c := range cases {
		b := append([]byte(nil), orig...)
		r, bl := c.edit(b)
		wnr4Rehash(b, r, bl)
		res, e := verifyBytes(b, nil)
		if e != nil {
			t.Errorf("%s: refused: %v", c.name, e)
			continue
		}
		if res.Undecoded != 1 || !strings.Contains(res.Why[0], c.why) {
			t.Errorf("%s: %d undecodable %q, %q expected", c.name, res.Undecoded, res.Why, c.why)
		}
	}
}

// TestWNR4ContainerExport: the writer has no codec 8, so a codec-8 file is not exported, and nothing is
// written.
func TestWNR4ContainerExport(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "out")
	_, err := Export(filepath.Join("testdata", "wnr4-gg-mix.wex4"), dir)
	if err == nil || !strings.Contains(err.Error(), "codec-8") {
		t.Fatalf("export of a codec-8 file: %v", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("export of a codec-8 file created %s", dir)
	}
	_, err = Export(filepath.Join("testdata", "wnr4-gg-bad-word.wex4"), dir)
	if err == nil || !strings.Contains(err.Error(), "undecodable") {
		t.Fatalf("export of a file with an undecodable record: %v", err)
	}
}
