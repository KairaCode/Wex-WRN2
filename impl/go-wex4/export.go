// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// Export: a verified WEX4 file taken apart into the writer's input. Every tensor's decoded bytes, every blob
// and the metadata go to files, and a manifest records the choices the file was written with (the order of
// the extents, the grouping of tensors into records, the codec and the parameters of every record). The
// writer on that manifest should give the file again: that is the test of the sentence of docs/WEX4-SPEC.md
// section 5 that the encoding is canonical.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Export writes dir/manifest.txt and the source files of the WEX4 file at path.
func Export(path, dir string) (*Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return nil, err
	}
	var failed error
	save := func(name string, b []byte, add bool) {
		if failed != nil {
			return
		}
		flag := os.O_WRONLY | os.O_CREATE | os.O_TRUNC
		if add {
			flag = os.O_WRONLY | os.O_APPEND
		}
		out, err := os.OpenFile(filepath.Join(dir, name), flag, 0o666)
		if err == nil {
			_, err = out.Write(b)
			if cerr := out.Close(); err == nil {
				err = cerr
			}
		}
		failed = err
	}
	tensorFile := func(i uint32) string { return fmt.Sprintf("t%05d.bin", i) }

	type line struct {
		offset uint64
		text   string
	}
	var lines []line
	res, e := Verify(f, uint64(st.Size()), nil, func(i uint32, r Record, x *Index, dec []byte) {
		var s strings.Builder
		s.WriteString("record")
		head := make([]byte, 32)
		if _, err := f.ReadAt(head, int64(r.Offset)); err != nil && failed == nil {
			failed = err
		}
		switch r.Codec {
		case codecRAW:
			s.WriteString(" codec=raw")
		case codecWBP1:
			fmt.Fprintf(&s, " codec=wbp1 planes=%d chunk=%d transform=%d", head[12], head[13], head[15])
		case codecWNR3:
			fmt.Fprintf(&s, " codec=wnr3 model=%d cw=%d cs=%d", r.Blob, le.Uint32(head[16:]), le.Uint32(head[20:]))
			if r.Flags&flagValueExact != 0 {
				s.WriteString(" cz")
			}
		}
		if r.Flags&flagExpert != 0 {
			fmt.Fprintf(&s, " expert=%d:%d", r.Owner>>16, r.Owner&0xffff)
		}
		if r.TensorCount == 0 {
			fmt.Fprintf(&s, " bytes=%d part\n", r.Decoded)
			save(tensorFile(r.FirstTensor), dec, true)
		} else if t := x.Tensor(r.FirstTensor); t.RecordCount != 1 {
			fmt.Fprintf(&s, " bytes=%d\n", r.Decoded)
			save(tensorFile(r.FirstTensor), dec, false)
			s.WriteString(tensorLine("tensor", x, r.FirstTensor, t))
		} else {
			s.WriteString("\n")
			for j := uint32(0); j < r.TensorCount; j++ {
				t := x.Tensor(r.FirstTensor + j)
				save(tensorFile(r.FirstTensor+j), dec[t.Offset:t.Offset+t.NBytes], false)
				s.WriteString(tensorLine("tensor", x, r.FirstTensor+j, t))
			}
		}
		lines = append(lines, line{r.Offset, s.String()})
	})
	if e != nil {
		return nil, e
	}
	if failed != nil {
		return nil, failed
	}
	if res.Undecoded != 0 {
		return nil, fmt.Errorf("%d records of a codec this reader does not know: the file cannot be exported", res.Undecoded)
	}
	x, h := res.X, res.H
	for i := uint32(0); i < h.Blobs; i++ {
		b := x.Blob(i)
		data, e := readAt(f, b.Offset, b.Bytes)
		if e != nil {
			return nil, e
		}
		name := fmt.Sprintf("b%d.bin", i)
		save(name, data, false)
		lines = append(lines, line{b.Offset, fmt.Sprintf("blob kind=%d id=0x%08x file=%s\n", b.Kind, b.ID, name)})
	}
	if h.IndexBytes != 0 {
		lines = append(lines, line{h.IndexOffset, "index\n"})
	}
	sort.Slice(lines, func(a, b int) bool { return lines[a].offset < lines[b].offset })

	var s strings.Builder
	fmt.Fprintf(&s, "wex4-manifest 1\n# exported from a file with the root %x\nminor %d\nrequired 0x%x\noptional 0x%x\ngrid %d %d\n",
		h.Root, h.Minor, h.Required, h.Optional, h.Layers, h.Experts)
	if h.MetaBytes != 0 {
		save("meta.bin", x.Metadata(), false)
		s.WriteString("meta meta.bin\n")
	}
	for _, l := range lines {
		s.WriteString(l.text)
	}
	for i := x.DataTensors; i < h.Tensors; i++ {
		s.WriteString(tensorLine("empty", x, i, x.Tensor(i)))
	}
	save("manifest.txt", []byte(s.String()), false)
	return res, failed
}

func tensorLine(keyword string, x *Index, i uint32, t Tensor) string {
	dtype := fmt.Sprint(t.Dtype)
	if int(t.Dtype) < len(dtypeNames) && t.Dtype != 0 {
		dtype = dtypeNames[t.Dtype]
	}
	dims := make([]string, t.Rank)
	for k := range dims {
		dims[k] = fmt.Sprint(t.Shape[k])
	}
	s := fmt.Sprintf("%s name=%s dtype=%s shape=%s", keyword, escape(x.Name(t)), dtype, strings.Join(dims, ","))
	if keyword == "tensor" {
		s += fmt.Sprintf(" file=t%05d.bin", i)
	}
	return s + "\n"
}
