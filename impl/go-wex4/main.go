// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli

// wex4check: a WEX4 reader, verifier and writer written from docs/WEX4-SPEC.md, docs/WNR3-SPEC.md and
// docs/WNR2-REFERENCE.md only. Standard library only.
//
//	wex4check [-v] [-root HEX] FILE... verify files: header page, index, every blob and record, decode
//	wex4check -manifest MANIFEST.tsv   check the conformance vectors against their manifest
//	wex4check write MANIFEST.txt OUT   write a WEX4 file from source files and a manifest (exit 0 or 1)
//	wex4check train MANIFEST.txt N OUT train the WNM2 model of blob line N on the manifest's experts
//	wex4check export FILE DIR          take a WEX4 file apart into source files and a manifest (exit 0 or 1)
//
// Exit code, files: 0 every file verified, 1 a file was refused, 3 a file opened and holds records of a
// codec this reader does not know (hashes checked, not decoded), 2 usage or a read error.
// Exit code, manifest: 0 every row as the manifest says, 1 otherwise.

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func verifyFile(path string, pin []byte) (*Result, *Error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fail(EIO, "%v", err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, fail(EIO, "%v", err)
	}
	return Verify(f, uint64(st.Size()), pin, nil)
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// runManifest checks every row of a MANIFEST.tsv: the file's SHA-256, the verdict, the reject class and
// the root. It returns the number of rows that differ.
func runManifest(path string, verbose bool) (rows, wrong int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	dir := filepath.Dir(path)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		col := strings.Split(line, "\t")
		if len(col) < 5 {
			return rows, wrong, fmt.Errorf("manifest row with %d columns: %q", len(col), line)
		}
		name, verdict, class, root, sum := col[0], col[1], col[2], col[3], col[4]
		rows++
		file := filepath.Join(dir, filepath.FromSlash(name))
		got, err := fileSHA256(file)
		if err != nil {
			return rows, wrong, err
		}
		problem := ""
		if got != sum {
			problem = "the vector is damaged: its SHA-256 is " + got
		} else {
			res, e := verifyFile(file, nil)
			switch {
			case e != nil && verdict == "accept":
				problem = "refused, the manifest says accept: " + e.Error()
			case e != nil && e.Class != class:
				problem = "refused as " + e.Error() + ", the manifest says " + class
			case e == nil && verdict != "accept":
				problem = "accepted, the manifest says reject " + class
			case e == nil && res.Undecoded != 0:
				problem = "accepted with records that were not decoded"
			case e == nil && hex.EncodeToString(res.H.Root[:]) != root:
				problem = "root " + hex.EncodeToString(res.H.Root[:]) + ", the manifest says " + root
			}
			if problem == "" && verbose {
				if e != nil {
					fmt.Printf("ok    %s: %s\n", name, e.Error())
				} else {
					fmt.Printf("ok    %s: accept\n", name)
				}
			}
		}
		if problem != "" {
			wrong++
			fmt.Printf("WRONG %s: %s\n", name, problem)
		}
	}
	return rows, wrong, sc.Err()
}

const usage = `usage: wex4check [-v] [-root HEX] FILE...
       wex4check [-v] -manifest MANIFEST.tsv
       wex4check write MANIFEST.txt OUT.wex4
       wex4check train MANIFEST.txt N OUT.wnm2
       wex4check export FILE.wex4 DIR`

// cmdWrite: a WEX4 file from a writer's manifest (README.md, "The writer's manifest").
func cmdWrite(args []string) int {
	if len(args) != 2 {
		fmt.Fprintln(os.Stderr, usage)
		return 2
	}
	m, err := ParseManifest(args[0])
	if err == nil {
		var w *Written
		if w, err = WriteFile(m, args[1]); err == nil {
			fmt.Printf("%s: root %x, %d bytes, %d blobs, %d records (%d RAW, %d WBP1, %d WNR3; %d stored RAW because not smaller), %d tensors, content digest %x: written\n",
				args[1], w.Root, w.FileBytes, w.Blobs, w.Records, w.RAW, w.WBP1, w.WNR3, w.Fallbacks, w.Tensors, w.Digest)
			return 0
		}
	}
	fmt.Fprintln(os.Stderr, "wex4check write:", err)
	return 1
}

// cmdTrain: the WNM2 model of blob line N of a manifest, trained on the expert records that name it.
func cmdTrain(args []string) int {
	if len(args) != 3 {
		fmt.Fprintln(os.Stderr, usage)
		return 2
	}
	n, err := strconv.ParseUint(args[1], 10, 32)
	if err != nil {
		fmt.Fprintln(os.Stderr, usage)
		return 2
	}
	m, err := ParseManifest(args[0])
	if err == nil {
		var tr *Trained
		if tr, err = TrainFile(m, uint32(n), args[2]); err == nil {
			fmt.Printf("%s: WNM2 model, id 0x%08x, trained on %d expert records (%d with canonical zero); %d of the %d weight contexts and %d of the 3 scale tables had no count: written\n",
				args[2], tr.ID, tr.Records, tr.CZ, tr.EmptyContexts, 3*wnmContexts, tr.EmptyScales)
			return 0
		}
	}
	fmt.Fprintln(os.Stderr, "wex4check train:", err)
	return 1
}

// cmdExport: the sources and the manifest of a WEX4 file, for the writer.
func cmdExport(args []string) int {
	if len(args) != 2 {
		fmt.Fprintln(os.Stderr, usage)
		return 2
	}
	res, err := Export(args[0], args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "wex4check export:", err)
		return 1
	}
	fmt.Printf("%s: root %x, %d tensors, content digest %x: exported to %s\n", args[0], res.H.Root, res.H.Tensors, res.Digest, args[1])
	return 0
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "write" {
		os.Exit(cmdWrite(os.Args[2:]))
	}
	if len(os.Args) > 1 && os.Args[1] == "export" {
		os.Exit(cmdExport(os.Args[2:]))
	}
	if len(os.Args) > 1 && os.Args[1] == "train" {
		os.Exit(cmdTrain(os.Args[2:]))
	}
	manifest := flag.String("manifest", "", "check the vectors listed in this MANIFEST.tsv")
	rootHex := flag.String("root", "", "the root (64 hex digits) every FILE must have")
	verbose := flag.Bool("v", false, "print every manifest row, and what the decoders met")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, usage)
	}
	flag.Parse()

	if *manifest != "" {
		rows, wrong, err := runManifest(*manifest, *verbose)
		if err != nil {
			fmt.Fprintln(os.Stderr, "wex4check:", err)
			os.Exit(2)
		}
		if *verbose {
			fmt.Printf("the decoders met: %+v\n", Seen)
		}
		fmt.Printf("%d rows, %d as the manifest says, %d wrong\n", rows, rows-wrong, wrong)
		if wrong != 0 || rows == 0 {
			os.Exit(1)
		}
		return
	}

	var pin []byte
	if *rootHex != "" {
		var err error
		if pin, err = hex.DecodeString(*rootHex); err != nil || len(pin) != 32 {
			fmt.Fprintln(os.Stderr, "wex4check: -root takes 64 hex digits")
			os.Exit(2)
		}
	}
	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}
	code := 0
	worse := func(c int) {
		// 2 (could not read) over 1 (refused) over 3 (not decoded in full) over 0
		rank := map[int]int{0: 0, 3: 1, 1: 2, 2: 3}
		if rank[c] > rank[code] {
			code = c
		}
	}
	for _, path := range flag.Args() {
		res, e := verifyFile(path, pin)
		switch {
		case e != nil && e.Class == EIO:
			fmt.Printf("%s: error %s\n", path, e.Msg)
			worse(2)
		case e != nil:
			fmt.Printf("%s: reject %s\n", path, e.Error())
			worse(1)
		default:
			h := res.H
			fmt.Printf("%s: root %x, %d blobs, %d records (%d expert), %d tensors, %d decoded bytes",
				path, h.Root, h.Blobs, h.Records, res.X.ExpertRecords, h.Tensors, h.TotalDecoded)
			if res.Undecoded != 0 {
				fmt.Printf(": hashes pass, %d records of an unknown codec were not decoded\n", res.Undecoded)
				worse(3)
			} else {
				fmt.Printf(", content digest %x: accept\n", res.Digest)
			}
		}
	}
	if *verbose {
		fmt.Printf("the decoders met: %+v\n", Seen)
	}
	os.Exit(code)
}
