# impl/go-wex4/

**Current.** The Go reader: a WEX4 reader and verifier in Go, standard library only, beside the two reference readers (`c/wex4.h` and `wex4/wex4.py`). It was written on 2026-11-14 from the specification pages and not from the reference code: [WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md), [WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md) sections 5 to 7 (the WNM2 model and the stream framing, which WNR3-SPEC does not repeat) and the [conformance vectors](../../wex4/vectors/README.md) with their manifest. `c/wex4.h`, `c/wex_format.h`, `c/wex_neural_rans.h`, `c/tools/wex4_verify.c`, `wex4/wex4.py`, `wex4/wex4_tools.c` and `wex4/test_wex4.py` were not opened while it was written; the reference readers and the reference writer were run as black boxes to make test files and to ask for verdicts. It is the answer to item 4 of [HELP-WANTED](../../docs/HELP-WANTED.md); what the specifications left open is in [SPEC-NOTES.md](SPEC-NOTES.md).

It reads; it does not write. It is not used by any engine.

## Build and run

Go and nothing else. The code uses the `min` and `max` builtins, which came with Go 1.21; `run.sh` was run with Go 1.21.13 (once, Windows), 1.24.13 (CI, Linux) and 1.27.4 (Windows), and with no other version. With the MSYS2 package `mingw-w64-ucrt-x86_64-go`, run `export GOROOT=/ucrt64/lib/go` first.

```sh
bash impl/go-wex4/run.sh                 # gofmt, go vet, go test, then every row of wex4/vectors/MANIFEST.tsv
bash impl/go-wex4/run.sh FILE.wex4 ...   # the same, then these files
```

The last line on success is `go reader: all checks passed`. The binary is `impl/go-wex4/out/wex4check` (git ignores `out/`):

```sh
wex4check [-v] [-root HEX] FILE...     # verify files; -root: the root every file must have
wex4check [-v] -manifest MANIFEST.tsv  # check vectors against their manifest
```

One line per file: `FILE: root <64 hex>, ... : accept`, or `FILE: reject WEX4_E_<CLASS>: <what was found>`. Exit code for files: 0 every file verified; 1 a file was refused; 3 a file opened and holds records of a codec this reader does not know (their SHA-256 was checked, they were not decoded: WEX4-SPEC section 10, rule 6); 2 usage or a read error. Exit code for a manifest: 0 when every row is as the manifest says, else 1. `scripts/check_all.sh` does not call this folder, so Go is not needed for the repository's checks; CI runs `run.sh` in its `go-reader` job.

## What it checks

Everything of WEX4-SPEC section 6, on the whole file, in the order of that section:

| Step | What |
|---|---|
| 1, header page | magic, `header_sha256` (the root), major version, required feature bits, exact file size, every cap, the expert grid, `index_bytes` against the counts, the place of the index |
| 2, index | `index_sha256`, zero padding, every tensor entry (name range, UTF-8, rank, shape, size against the dtype), the name-order table, every blob entry, every record entry with its codec rules, the tensors that tile each record, split tensors and their parts, expert records and their six tensors, `max_decoded` and `total_decoded`, the tiling of the file by its extents. In place: the tables are not copied |
| 3, blobs | SHA-256, zero padding; a WNM2 model is opened (header, every CDF) and its FNV-1a id compared with the entry |
| 3, records | SHA-256 of the stored bytes, then the decoder: RAW (zero padding), WBP1 (codec 6, WEX4-SPEC 7.2), WNR3 (codec 5, WNR3-SPEC sections 3 and 4: stream hashes, owner, canonical-zero rule, inverse of the ceil form). A record is given to the caller only after its decoder returned without an error. Since 2026-11-15 also the content digest of WEX4-SPEC section 5 over the decoded tensors, which `-manifest` compares with the manifest's `content` column |

The reject classes are the names the manifest uses for the C reader (`WEX4_E_HEADER`, `WEX4_E_RECORD`, ...). The specification says they are informative (WEX4-SPEC section 17, since 2026-11-15); here a class is the kind of rule that is broken.

## What was checked, and how

All on 2026-11-14 unless a row says otherwise. Local: Windows 10 desktop, Go 1.27.4 from MSYS2 UCRT64, one run each unless said. Linux: the `go-reader` job of CI (ubuntu-24.04, x86-64, the runner's Go, 1.24.13 on that day) runs `run.sh`: the tests and the 112 rows passed there too; the files of `wex4/out/`, the reference's verdicts and the differential run were done locally only. "The reference" is `wex4/wex4.py` and the C reader behind `wex4/out/wex4_tools.dll`.

| Check | Result |
|---|---|
| The 112 rows of `wex4/vectors/MANIFEST.tsv`: file SHA-256, verdict, reject class, root | 112 as the manifest says: 8 accepted with the manifest's root, 104 refused with the manifest's class. This was the result of the first run, before any change to the reader |
| 2026-11-15: the 134 rows of the manifest, with the 22 vectors of that day and the `content` column | 134 as the manifest says: 16 accepted with the manifest's root and content digest, 118 refused with the manifest's class. The only change to the reader for it was the content digest (`verify.go`, `main.go`); the code that gives verdicts and classes (`wex4.go`, `wbp1.go`, `wnr3.go`, `rans.go`) is the code of 2026-11-14 |
| 2026-11-15: a record of an unknown codec that names a blob (WEX4-SPEC 5.2), three files made outside the repository | opened, hashes checked, exit code 3, as both reference readers open them |
| WNR3 output against data this reader did not decode | the 2 WNR3-CZ records of `valid/tiny2-cz-upgraded.wex4` decode to the RAW records of the same owners in `valid/tiny-raw-upgraded.wex4` with every -0 weight nibble set to +0 (13,056 B each, 2,850 such nibbles) |
| The ceil form of WNR3, which no vector had before 2026-11-15 | the 2 records of `testdata/ceil-form.wex4` (5 matrices in ceil form, 808 blocks changed by the inverse) decode to the SHA-256 of their source slabs |
| WBP1 with 4 and 8 planes, with and without the rotation, and byte 255 in coded planes, which no vector had before 2026-11-15 | the 13 tensors of `testdata/planes.wex4` decode to the SHA-256 of their source bytes |
| WBP1 records whose size is not a multiple of the plane count, and `chunk_log2 = 11`, which no vector had before 2026-11-15 | the 4 records of `testdata/ragged.wex4` (2, 4, 8 and 8 planes; 5 raw chunks, 31 rANS streams) decode to the SHA-256 of their source bytes |
| A WBP1 record with one more page of zeros than it needs, every hash right ([SPEC-NOTES.md](SPEC-NOTES.md), A9) | `testdata/extra-page.wex4` is refused with `WEX4_E_DATA`, as both reference readers refuse it; with that page dropped and the sizes and hashes repaired the file verifies and decodes to its source |
| Tensors of 0 bytes outside every record: 8 changes of one tensor entry of `generic.wex4`, both hashes recomputed (SPEC-NOTES, A5 and A6) | 4 accepted and 4 refused, the verdicts both reference readers gave on the same 8 files |
| `generic.wex4` against `generic-index-last.wex4` | the 8 records decode to the same bytes in both layouts |
| The stream hash | the four function vectors and the two stream vectors of WNR3-SPEC 4.5, and the SMHasher verification value |
| Mutations, WEX4-SPEC section 9, on the 7 valid vectors and on `ceil-form.wex4` and `planes.wex4` of `testdata/`, fixed seed | (a) 9,000 single-bit flips with the hashes left alone: all refused. (b) 18,000 bytes of the header page or the index changed with both hashes recomputed: 15,158 refused, 2,842 accepted, no panic. This part has no oracle: the test does not ask the reference whether an accepted file is valid (many are changes of metadata, names or optional bits), so it shows the absence of a panic and nothing about the verdicts; the differential run below is the check of the verdicts. (c) 8,000 bytes of WBP1 and WNR3 records changed with the record's SHA-256 recomputed: 7,982 refused by the decoder, 18 accepted with the same decoded bytes, none accepted with other bytes |
| Verdicts against the reference on changed files: a review's differential run, with scripts that are not in the repository. 20,000 files (4,000 from each of `generic.wex4`, `generic-index-last.wex4`, `tiny2-cz-upgraded.wex4`, `testdata/planes.wex4`, `testdata/ceil-form.wex4`): one or two fields of the header or of an index entry set to a boundary or random value, or one byte of a compressed record with its hash recomputed; both file hashes recomputed; one seed, one run, after the two fixes below | the C and the Python reader agreed on all 20,000 (2,136 accepted, 166 with a record of an unknown codec). The Go reader gave the same verdict on all 20,000; on 184 refused files its reject class was another one. These mutations do not build the two cases below (the review reports no difference from its own run of the same scripts on the first version), so the run says nothing about them |
| The 8 WEX4 files that `bash wex4/build_and_test.sh` leaves in `wex4/out/` (not tracked) | all accepted after the fix below. Roots were compared with the reference only for the three that are copies of a vector |
| The reference on the four files of `testdata/` | `python -I wex4/wex4.py verify` accepts `ceil-form.wex4`, `planes.wex4` and `ragged.wex4` and prints the roots this reader prints; it refuses `extra-page.wex4` |

**One bug the vectors did not find.** The first version passed all 112 rows and refused three files of `wex4/out/`: symbol 255 of a rANS-coded plane read the wrong CDF entry. No rANS-coded plane and no scale stream of the vectors of 2026-11-14 holds that byte. `ceil-form.wex4` and `planes.wex4` of `testdata/` fail with the bug put back ([SPEC-NOTES.md](SPEC-NOTES.md), part D), and since 2026-11-15 so do the two WBP1 vectors of 4 and 8 planes and of ragged sizes; the two new WNR3 vectors fail when the same mistake is put in the scale streams (SPEC-NOTES, "After the sentences and vectors of 2026-11-15").

**Two verdicts that differed from the reference.** The first version passed all 112 rows and still disagreed with both reference readers on two kinds of file, found by a review that asked the reference. It accepted a WBP1 record with one more page of zeros than it needs, which the reference refuses; and it refused an opaque tensor of 0 bytes outside every record whose shape has no 0 dimension, which the reference accepts. The revision of the specification it was written from did not have the sentence for either case; the revision of 2026-11-14 has both ([SPEC-NOTES.md](SPEC-NOTES.md), "After the spec review"). The reader now gives the reference's verdict on both ([SPEC-NOTES.md](SPEC-NOTES.md), A6 and A9), and the tests above hold them.

## Limits

- **A reader and verifier only:** no writer, no upgrade or downgrade, no safetensors unpack. The content digest of WEX4-SPEC section 5 is computed since 2026-11-15, from the field widths of the revision of 2026-11-14 (the revision the reader was first written from did not give them). Unlike the rest of the reader, it was added after the reference code had been read, so it is not independent of it; the manifest's digests, which come from the C and the Python reader, are what it is checked against.
- **Not exercised by any file:** `chunk_log2` other than 10, 11 and 12; a record of an unknown codec in the vectors (exit code 3 was run on three files made outside the repository, 2026-11-15).
- **No coverage-guided fuzzing.** The mutation test above is the only fuzzing. One attempt with Go's native fuzzer on whole files ran 18 inputs in 60 seconds and was dropped.
- **Speed and size were not measured.** One thread, one record in memory at a time (`max_decoded` bytes plus the stored record); the largest file read is 2,977,792 B. It was not run on a real container.
- **Go checks every index**, so an out-of-bounds access is a panic, not a read of other memory; a panic on a hostile file would still be a bug. None was seen in the runs above.
- **Little-endian or big-endian host:** the code reads integers byte by byte; it was run on x86-64 only.
- **Class names** follow the C reader's order for the files of the manifest only. On another broken file the class can differ from the C reader's (184 of the 20,000 files of the differential run). The verdict should not differ, and that is known only for the files of the table above. Of the readings of SPEC-NOTES part B, B3 and B8 have vectors since 2026-11-15 and B5 is checked on three files; B2 and B4 are about the wording and no file decides them.

## Files

| File | What it is |
|---|---|
| `wex4.go` | header page and index: steps 1 and 2 |
| `verify.go` | step 3: blobs, records, the call of the decoders |
| `rans.go` | the rANS stream reader and the MurmurHash3 stream hash shared by WBP1 and WNR3 |
| `wbp1.go` | codec 6 decoder |
| `wnr3.go` | codec 5 decoder and the WNM2 model |
| `main.go` | the command line and the manifest check |
| `wex4_test.go` | the tests of the table above |
| `run.sh` | builds, tests, checks the vectors |
| `testdata/` | four WEX4 files for what the vectors did not cover before 2026-11-15, and the scripts that make them or their sources ([README](testdata/README.md)) |
| `SPEC-NOTES.md` | what the specifications left open and what the vectors did not exercise on 2026-11-14, and where each point went since |
