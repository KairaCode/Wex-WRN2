// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// The input of the writer: a text manifest that names the source files and fixes every choice that
// docs/WEX4-SPEC.md leaves to a writer (the layout, the grouping of tensors into records, the codecs and
// their parameters, the metadata). The syntax is in README.md, "The writer's manifest".

package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var dtypeNames = [23]string{"", "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "I16", "U16", "F16", "BF16",
	"I32", "U32", "F32", "I64", "U64", "F64", "C64", "F4", "F6_E2M3", "F6_E3M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}

type srcTensor struct {
	name   []byte
	dtype  uint8
	rank   uint8
	shape  [8]uint64
	file   string // empty for a tensor of 0 bytes
	nbytes uint64
	line   int
}

type srcRecord struct {
	codec     string // raw, wbp1, wnr3, auto
	planes    int    // -1: not given
	chunk     int    // chunk_log2, -1: not given
	transform int    // -1: not given
	expert    bool
	owner     uint32
	model     uint32 // 0: none, else 1 + the blob of the manifest
	cw, cs    uint32
	cz        bool
	part      bool   // a continuation part of the split tensor before it
	bytes     uint64 // of a split tensor: the bytes this record takes; 0: not given
	tensors   []srcTensor
	line      int
}

type srcBlob struct {
	kind, id uint32
	hasID    bool
	file     string
	line     int
}

type extent struct {
	kind byte // 'i' index, 'b' blob, 'r' record
	n    int
}

// Manifest is a parsed manifest. File names are already joined with the manifest's folder.
type Manifest struct {
	minor              uint16
	required, optional uint64
	layers, experts    uint32
	meta               string
	blobs              []srcBlob
	records            []srcRecord
	empties            []srcTensor
	order              []extent
	hasIndex           bool
}

// escape writes a name or a file name as one manifest word: bytes up to 0x20, '%' and 0x7f become %XX.
func escape(b []byte) string {
	var s strings.Builder
	for _, c := range b {
		if c <= 0x20 || c == '%' || c == 0x7f {
			fmt.Fprintf(&s, "%%%02X", c)
		} else {
			s.WriteByte(c)
		}
	}
	return s.String()
}

func unescape(s string) ([]byte, error) {
	var b []byte
	for i := 0; i < len(s); i++ {
		if s[i] != '%' {
			b = append(b, s[i])
			continue
		}
		var v uint64
		err := strconv.ErrSyntax
		if i+3 <= len(s) {
			v, err = strconv.ParseUint(s[i+1:i+3], 16, 8)
		}
		if err != nil {
			return nil, fmt.Errorf("%% without two hex digits in %q", s)
		}
		b = append(b, byte(v))
		i += 2
	}
	return b, nil
}

func parseDtype(s string) (uint8, error) {
	for code, name := range dtypeNames {
		if code != 0 && name == s {
			return uint8(code), nil
		}
	}
	v, err := strconv.ParseUint(s, 10, 8)
	if err != nil {
		return 0, fmt.Errorf("dtype %q is not a name of WEX4-SPEC 5.4 and not a code 0..255", s)
	}
	return uint8(v), nil
}

// ParseManifest reads a manifest. Sizes of the source files are taken from the files.
func ParseManifest(path string) (*Manifest, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dir := filepath.Dir(path)
	m := &Manifest{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<16), 1<<22)
	line, started := 0, false
	once := map[string]bool{} // the lines that are given at most once
	for sc.Scan() {
		line++
		words := strings.FieldsFunc(sc.Text(), func(r rune) bool { return r == ' ' || r == '\t' || r == '\r' })
		if len(words) == 0 || strings.HasPrefix(words[0], "#") {
			continue
		}
		bad := func(format string, a ...any) error {
			return fmt.Errorf("%s line %d: %s", path, line, fmt.Sprintf(format, a...))
		}
		if !started {
			if len(words) != 2 || words[0] != "wex4-manifest" || words[1] != "1" {
				return nil, bad("the first line must be: wex4-manifest 1")
			}
			started = true
			continue
		}
		// key=value words and bare flags after the keyword
		kv := map[string]string{}
		var args []string
		for _, w := range words[1:] {
			k, v, isKV := strings.Cut(w, "=")
			if !isKV {
				args = append(args, w)
				continue
			}
			if _, dup := kv[k]; dup {
				return nil, bad("%s is given twice", k)
			}
			b, err := unescape(v)
			if err != nil {
				return nil, bad("%v", err)
			}
			kv[k] = string(b)
		}
		take := func(k string) (string, bool) {
			v, ok := kv[k]
			delete(kv, k)
			return v, ok
		}
		flag := func(name string) bool { // a bare word, at most once
			for i, a := range args {
				if a == name {
					args = append(args[:i], args[i+1:]...)
					return true
				}
			}
			return false
		}
		num := func(k string, bits int, def uint64, need bool) (uint64, error) {
			s, ok := take(k)
			if !ok {
				if need {
					return 0, bad("%s needs %s=", words[0], k)
				}
				return def, nil
			}
			v, err := strconv.ParseUint(s, 0, bits)
			if err != nil {
				return 0, bad("%s=%s is not a number of %d bits", k, s, bits)
			}
			return v, nil
		}
		file := func(need bool) (string, error) {
			s, ok := take("file")
			if !ok || s == "" {
				if need {
					return "", bad("%s needs file=", words[0])
				}
				return "", nil
			}
			return filepath.Join(dir, filepath.FromSlash(s)), nil
		}
		plain := func(n int) ([]uint64, error) { // a keyword followed by n numbers
			if len(args) != n {
				return nil, bad("%s takes %d numbers", words[0], n)
			}
			out := make([]uint64, n)
			for i, a := range args {
				v, err := strconv.ParseUint(a, 0, 64)
				if err != nil {
					return nil, bad("%q is not a number", a)
				}
				out[i] = v
			}
			args = nil
			return out, nil
		}
		tensor := func(needFile bool) (srcTensor, error) {
			t := srcTensor{line: line}
			name, ok := take("name")
			if !ok {
				return t, bad("%s needs name=", words[0])
			}
			t.name = []byte(name)
			ds, _ := take("dtype")
			var err error
			if t.dtype, err = parseDtype(ds); err != nil {
				return t, bad("%v", err)
			}
			shape, ok := take("shape")
			if !ok {
				return t, bad("%s needs shape= (empty for a scalar)", words[0])
			}
			if shape != "" {
				dims := strings.Split(shape, ",")
				if len(dims) > 8 {
					return t, bad("rank %d is above 8", len(dims))
				}
				for k, d := range dims {
					if t.shape[k], err = strconv.ParseUint(d, 10, 64); err != nil {
						return t, bad("shape=%s", shape)
					}
				}
				t.rank = uint8(len(dims))
			}
			if t.file, err = file(needFile); err != nil {
				return t, err
			}
			if t.file != "" {
				st, err := os.Stat(t.file)
				if err != nil {
					return t, bad("%v", err)
				}
				t.nbytes = uint64(st.Size())
			}
			return t, nil
		}

		switch words[0] {
		case "minor", "required", "optional", "grid", "meta":
			if once[words[0]] {
				return nil, bad("%s is given once", words[0])
			}
			once[words[0]] = true
		}
		switch words[0] {
		case "minor", "required", "optional":
			v, err := plain(1)
			if err != nil {
				return nil, err
			}
			switch words[0] {
			case "minor":
				if v[0] > 65535 {
					return nil, bad("minor %d", v[0])
				}
				m.minor = uint16(v[0])
			case "required":
				m.required = v[0]
			default:
				m.optional = v[0]
			}
		case "grid":
			v, err := plain(2)
			if err != nil {
				return nil, err
			}
			if v[0] > 65536 || v[1] > 65536 {
				return nil, bad("grid %d x %d", v[0], v[1])
			}
			m.layers, m.experts = uint32(v[0]), uint32(v[1])
		case "meta":
			if len(args) != 1 {
				return nil, bad("meta takes one file name")
			}
			b, err := unescape(args[0])
			if err != nil {
				return nil, bad("%v", err)
			}
			m.meta = filepath.Join(dir, filepath.FromSlash(string(b)))
			args = nil
		case "index":
			if m.hasIndex {
				return nil, bad("index takes no word and is given once")
			}
			m.hasIndex = true
			m.order = append(m.order, extent{'i', 0})
		case "blob":
			b := srcBlob{line: line}
			v, err := num("kind", 32, 0, true)
			if err != nil {
				return nil, err
			}
			b.kind = uint32(v)
			_, b.hasID = kv["id"]
			if v, err = num("id", 32, 0, false); err != nil {
				return nil, err
			}
			b.id = uint32(v)
			if b.file, err = file(true); err != nil {
				return nil, err
			}
			m.order = append(m.order, extent{'b', len(m.blobs)})
			m.blobs = append(m.blobs, b)
		case "record":
			r := srcRecord{line: line, planes: -1, chunk: -1, transform: -1}
			var ok bool
			if r.codec, ok = take("codec"); !ok {
				r.codec = "auto"
			}
			if r.codec != "raw" && r.codec != "wbp1" && r.codec != "wnr3" && r.codec != "auto" {
				return nil, bad("codec=%s: raw, wbp1, wnr3 or auto", r.codec)
			}
			given := func(keys ...string) string { // the first of keys this line gives, or ""
				for _, k := range keys {
					if _, ok := kv[k]; ok {
						return k + "="
					}
				}
				return ""
			}
			wbp1Word, wnr3Word := given("planes", "chunk", "transform"), given("model", "cw", "cs")
			hasModel := given("model") != ""
			for _, p := range []struct {
				key string
				dst *int
			}{{"planes", &r.planes}, {"chunk", &r.chunk}, {"transform", &r.transform}} {
				if _, ok := kv[p.key]; ok {
					v, err := num(p.key, 8, 0, true)
					if err != nil {
						return nil, err
					}
					*p.dst = int(v)
				}
			}
			if s, ok := take("expert"); ok {
				l, e, _ := strings.Cut(s, ":")
				lv, err1 := strconv.ParseUint(l, 10, 16)
				ev, err2 := strconv.ParseUint(e, 10, 16)
				if err1 != nil || err2 != nil {
					return nil, bad("expert=%s: LAYER:EXPERT, each 0..65535", s)
				}
				r.expert, r.owner = true, uint32(lv)<<16|uint32(ev)
			}
			v, err := num("model", 32, 0, false)
			if err != nil {
				return nil, err
			}
			r.model = uint32(v)
			if v, err = num("cw", 32, 2048, false); err != nil {
				return nil, err
			}
			r.cw = uint32(v)
			if v, err = num("cs", 32, 512, false); err != nil {
				return nil, err
			}
			r.cs = uint32(v)
			if r.bytes, err = num("bytes", 64, 0, false); err != nil {
				return nil, err
			}
			r.cz, r.part = flag("cz"), flag("part")
			if r.cz && wnr3Word == "" {
				wnr3Word = "cz"
			}
			// a word that does not apply to the codec the record gets is refused, not dropped
			codec := r.codec
			if codec == "auto" {
				switch {
				case !r.expert:
					codec = "wbp1"
				case hasModel:
					codec = "wnr3"
				default:
					codec = "raw"
				}
			}
			if wbp1Word != "" && codec != "wbp1" {
				return nil, bad("%s is a word of a WBP1 record; this record is %s", wbp1Word, strings.ToUpper(codec))
			}
			if wnr3Word != "" && codec != "wnr3" {
				return nil, bad("%s is a word of a WNR3 record; this record is %s (codec=auto is WNR3 only for an expert record with model=)",
					wnr3Word, strings.ToUpper(codec))
			}
			m.order = append(m.order, extent{'r', len(m.records)})
			m.records = append(m.records, r)
		case "tensor":
			if len(m.records) == 0 {
				return nil, bad("a tensor line before the first record line")
			}
			t, err := tensor(true)
			if err != nil {
				return nil, err
			}
			r := &m.records[len(m.records)-1]
			r.tensors = append(r.tensors, t)
		case "empty":
			t, err := tensor(false)
			if err != nil {
				return nil, err
			}
			if t.file != "" {
				return nil, bad("an empty tensor has no file")
			}
			m.empties = append(m.empties, t)
		default:
			return nil, bad("unknown keyword %q", words[0])
		}
		for k := range kv {
			return nil, bad("%s does not take %s=", words[0], k)
		}
		if len(args) != 0 {
			return nil, bad("%s does not take %q", words[0], args[0])
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if !started {
		return nil, fmt.Errorf("%s: not a manifest (the first line must be: wex4-manifest 1)", path)
	}
	return m, nil
}
