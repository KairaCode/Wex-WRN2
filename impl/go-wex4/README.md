# impl/go-wex4/

**Current.** A second WEX4 reader and verifier, in Go, standard library only. It was written on 2026-11-14 from the specification pages and not from the reference code: [WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md), [WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md) sections 5 to 7 (the WNM2 model and the stream framing, which WNR3-SPEC does not repeat) and the [conformance vectors](../../wex4/vectors/README.md) with their manifest. `c/wex4.h`, `c/wex_format.h`, `c/wex_neural_rans.h`, `c/tools/wex4_verify.c`, `wex4/wex4.py`, `wex4/wex4_tools.c` and `wex4/test_wex4.py` were not opened while it was written. It is the answer to item 4 of [HELP-WANTED](../../docs/HELP-WANTED.md); what the specifications left open is in [SPEC-NOTES.md](SPEC-NOTES.md).

It reads; it does not write. It is not used by any engine.

## Build and run

Go 1.21 or newer, nothing else. With the MSYS2 package `mingw-w64-ucrt-x86_64-go`, run `export GOROOT=/ucrt64/lib/go` first.

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
| 3, records | SHA-256 of the stored bytes, then the decoder: RAW (zero padding), WBP1 (codec 6, WEX4-SPEC 7.2), WNR3 (codec 5, WNR3-SPEC sections 3 and 4: stream hashes, owner, canonical-zero rule, inverse of the ceil form). A record is given to the caller only after its decoder returned without an error |

The reject classes are the names the manifest uses for the C reader (`WEX4_E_HEADER`, `WEX4_E_RECORD`, ...). The specification does not define them; here a class is the kind of rule that is broken.

## What was checked, and how

All on 2026-11-14. Local: Windows 10 desktop, Go 1.27.4 from MSYS2 UCRT64, one run each unless said. Linux: the `go-reader` job of CI (ubuntu-24.04, the runner's Go).

| Check | Result |
|---|---|
| The 112 rows of `wex4/vectors/MANIFEST.tsv`: file SHA-256, verdict, reject class, root | 112 as the manifest says: 8 accepted with the manifest's root, 104 refused with the manifest's class. This was the result of the first run, before any change to the reader |
| WNR3 output against data this reader did not decode | the 2 WNR3-CZ records of `valid/tiny2-cz-upgraded.wex4` decode to the RAW records of the same owners in `valid/tiny-raw-upgraded.wex4` with every -0 weight nibble set to +0 (13,056 B each, 2,850 such nibbles) |
| The ceil form of WNR3, which no vector has | the 2 records of `testdata/ceil-form.wex4` (5 matrices in ceil form, 808 blocks changed by the inverse) decode to the SHA-256 of their source slabs |
| WBP1 with 4 and 8 planes, with and without the rotation, and byte 255 in coded planes, which no vector has | the 13 tensors of `testdata/planes.wex4` decode to the SHA-256 of their source bytes |
| `generic.wex4` against `generic-index-last.wex4` | the 8 records decode to the same bytes in both layouts |
| The stream hash | the four function vectors and the two stream vectors of WNR3-SPEC 4.5, and the SMHasher verification value |
| Mutations, WEX4-SPEC section 9, on the 7 valid vectors and the 2 files of `testdata/`, fixed seed | (a) 9,000 single-bit flips with the hashes left alone: all refused. (b) 18,000 bytes of the header page or the index changed with both hashes recomputed: 15,158 refused, 2,842 accepted (changes that leave a valid file: metadata, names, optional bits), no panic. (c) 8,000 bytes of WBP1 and WNR3 records changed with the record's SHA-256 recomputed: 7,982 refused by the decoder, 18 accepted with the same decoded bytes, none accepted with other bytes |
| The 8 WEX4 files that `bash wex4/build_and_test.sh` leaves in `wex4/out/` (not tracked) | all accepted after the fix below. Roots were compared with the reference only for the three that are copies of a vector |
| The reference on the two files of `testdata/` | `python -I wex4/wex4.py verify` accepts both and prints the roots this reader prints |

**One bug the vectors did not find.** The first version passed all 112 rows and refused three files of `wex4/out/`: symbol 255 of a rANS-coded plane read the wrong CDF entry. No rANS-coded plane and no scale stream of the vectors holds that byte. The two files of `testdata/` fail with the bug put back ([SPEC-NOTES.md](SPEC-NOTES.md), part D).

## Limits

- **A reader and verifier only:** no writer, no upgrade or downgrade, no safetensors unpack. The content digest of WEX4-SPEC section 5 is not computed, because the specification does not give the encoding of its fields.
- **Not exercised by any file:** a WBP1 record whose size is not a multiple of its plane count; `chunk_log2` other than 10 and 12; WNR3 chunks other than 2048:512; canonical zero together with the ceil form; the error for a ceil-form block with coded scale 0; a record of an unknown codec (exit code 3 was not run on a file).
- **No coverage-guided fuzzing.** The mutation test above is the only fuzzing. One attempt with Go's native fuzzer on whole files ran 18 inputs in 60 seconds and was dropped.
- **Speed and size were not measured.** One thread, one record in memory at a time (`max_decoded` bytes plus the stored record); the largest file read is 2,977,792 B. It was not run on a real container.
- **Go checks every index**, so an out-of-bounds access is a panic, not a read of other memory; a panic on a hostile file would still be a bug. None was seen in the runs above.
- **Little-endian or big-endian host:** the code reads integers byte by byte; it was run on x86-64 only.
- **Class names** follow the C reader's order for the files of the manifest only. On another broken file the class can differ from the C reader's; the verdict must not.

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
| `testdata/` | two WEX4 files for what the vectors do not cover, and the scripts that make their sources ([README](testdata/README.md)) |
| `SPEC-NOTES.md` | what the specifications left open, and what the vectors do not exercise |
