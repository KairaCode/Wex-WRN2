# impl/go-wex4/

**Current.** The Go reader: a WEX4 reader and verifier in Go, standard library only, beside the two reference readers (`c/wex4.h` and `wex4/wex4.py`). It was written on 2026-11-14 from the specification pages and not from the reference code: [WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md), [WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md) sections 5 to 7 (the WNM2 model and the stream framing, which WNR3-SPEC does not repeat) and the [conformance vectors](../../wex4/vectors/README.md) with their manifest. `c/wex4.h`, `c/wex_format.h`, `c/wex_neural_rans.h`, `c/tools/wex4_verify.c`, `wex4/wex4.py`, `wex4/wex4_tools.c` and `wex4/test_wex4.py` were not opened while it was written; the reference readers and the reference writer were run as black boxes to make test files and to ask for verdicts. Its content digest is the one the writer of 2026-11-15 uses (`digest.go`), written the same way (see "Limits"). It is the answer to item 4 of [HELP-WANTED](../../docs/HELP-WANTED.md); what the specifications left open is in [SPEC-NOTES.md](SPEC-NOTES.md).

Since 2026-11-15 the same program also computes the content digest of WEX4-SPEC section 5, writes WEX4 files (`wex4check write`), with RAW, WBP1 and WNR3 records, and trains the WNM2 model that WNR3 records need (`wex4check train`). The writer was made the same way: from the three pages, with the reference code closed (the files named above, and `wex4/wex4_convert.py`), and with the reference readers run as black boxes, from the command line only (`verify`, `info` and `--help` of `wex4/wex4.py`; the C tools were built and run). The trainer too, with the reference builder and its encoder library (`c/tools/build_nr3.py`, `c/tools/wex_nr3_tools.c`) not opened either. What the pages left open for a writer is part E of [SPEC-NOTES.md](SPEC-NOTES.md).

Since 2026-11-23 it also decodes codec 8 (WNR4, DRAFT; [WEX4-SPEC](../../docs/WEX4-SPEC.md) 7.3), kinds GG (2) and I4 (3), the only kinds a WEX4 record may carry. The decoders (`wnr4.go`, `wnr4_gg.go`, `wnr4_i4.go`) were written from [WNR4-SPEC](../../docs/WNR4-SPEC.md) sections 1 to 6 and WEX4-SPEC 7.3 with the reference code of WNR4 closed (`wnr4/`, `wex4/wnr4gg.py`, `wex4/wnr4i4.py` and their tests, and the Rust reader); `wnr4_wex4.go` binds them to the container. A codec-8 record that fails a check, or is of kind 0 or 1, is undecodable (exit code 3): the file is not refused for it (WEX4-SPEC 7.3). What the pages left open is part F of [SPEC-NOTES.md](SPEC-NOTES.md).

It is not used by any engine.

## Build and run

Go and nothing else. The code uses the `min` and `max` builtins, which came with Go 1.21; `run.sh` was run with Go 1.24.13 (CI, Linux) and 1.27.4 (Windows), and with no other version. The reader of 2026-11-14 alone was also run once with Go 1.21.13 (Windows, 2026-11-14); the code of 2026-11-15 (the writer, the trainer, the content digest, one class in `wex4.go`) was not. With the MSYS2 package `mingw-w64-ucrt-x86_64-go`, run `export GOROOT=/ucrt64/lib/go` first.

```sh
bash impl/go-wex4/run.sh                 # gofmt, go vet, go test, every row of wex4/vectors/MANIFEST.tsv and of testdata/wnr4-MANIFEST.tsv
bash impl/go-wex4/run.sh FILE.wex4 ...   # the same, then these files
```

The last line on success is `go reader: all checks passed`. When Python 3 and a C compiler are on `PATH` (`PYTHON` and `CC` name them; the defaults are `python` or `python3`, and `gcc`), `run.sh` also builds the reference readers from the repository's sources into `out/` and the tests give them the files of the Go writer; a line of the output says whether that part ran. `WEX4_NEED_REFERENCE=1` makes it a failure when it cannot run (CI sets it). `WEX4_RANDOM_FILES=N` sets the number of random files the writer's tests make (24 by default; 400 took 89 seconds, the reference readers included, in one run on a Windows 10 desktop with an Intel Core i7-7700K). The binary is `impl/go-wex4/out/wex4check` (git ignores `out/`):

```sh
wex4check [-v] [-root HEX] FILE...       # verify files; -root: the root every file must have
wex4check [-v] -manifest MANIFEST.tsv    # check vectors against their manifest
wex4check write MANIFEST.txt OUT.wex4    # write a file from source files and a manifest (below)
wex4check train MANIFEST.txt N OUT.wnm2  # train the WNM2 model of blob line N on the manifest's experts
wex4check export FILE.wex4 DIR           # take a file apart: DIR/manifest.txt and the source files
```

One line per file: `FILE: root <64 hex>, ..., content digest <64 hex>: accept`, or `FILE: reject WEX4_E_<CLASS>: <what was found>`. Exit code for files: 0 every file verified; 1 a file was refused; 3 a file opened and holds undecodable records (their SHA-256 was checked, they were not decoded, and no content digest is printed; the line names the first one and why, `-v` the others): records of a codec this reader does not know (WEX4-SPEC section 10, rule 6), and since 2026-11-23 codec-8 records that fail a check of WEX4-SPEC 7.3 or of their decoder, or are of kind 0 or 1 (7.3: undecodable, the file not invalid); 2 usage or a read error. Exit code for a manifest: 0 when every row is as the manifest says, else 1. `write`, `train` and `export`: 0 done, 1 refused (the reason is on stderr), 2 usage. `scripts/check_all.sh` does not call this folder, so Go is not needed for the repository's checks; CI runs `run.sh` in its `go-reader` job.

## The writer

`wex4check write MANIFEST.txt OUT.wex4` reads a text manifest and the files it names, and writes one WEX4 file. The manifest fixes every choice the specification leaves to a writer, so the same manifest and the same source files give the same bytes. One record is in memory at a time. A word that does not apply to the codec a record gets (`planes=` on a RAW record, `cz` on an expert record with no `model=`, ...) is refused, not dropped, and so is a second `minor`, `required`, `optional`, `grid`, `meta` or `index` line.

The file is written as `OUT.wex4.tmp`. Every compressed record is decoded again before it is written, and the finished file is read back by the reader of this folder. It gets its name only when that reader accepts it and finds the content digest the writer computed from the source files; otherwise it is removed and the exit code is 1.

`wex4check export FILE.wex4 DIR` is the other direction: it verifies the whole file (a refused file leaves nothing in `DIR`), then reads it a second time and writes every tensor's decoded bytes to `DIR/tNNNNN.bin` (the tensor's index; names of the file are never used as paths), every blob to `DIR/bN.bin`, the metadata to `DIR/meta.bin`, and a `DIR/manifest.txt` that records the choices the file was made with. `write` on that manifest should give the file again; that is how the canonical encoding was tested (below).

### The writer's manifest

A text file, one line per item. The first line is `wex4-manifest 1`. A line that starts with `#` is a comment. Words are separated by spaces or tabs; a word is a keyword, a number, or `key=value`. In a value, a byte up to 0x20, `%` and 0x7f is written `%XX` (two hex digits); every other byte stands for itself, so a name is UTF-8 as it is. File names are relative to the manifest's folder, with `/`.

| Line | Meaning |
|---|---|
| `minor N`, `required N`, `optional N` | `version_minor` and the two feature words of the header page; 0 when absent. Bit 0 of `required` is also set by any record with `cz` |
| `grid LAYERS EXPERTS` | the expert grid; needed when the file has expert records |
| `meta FILE` | the metadata: the bytes of this file |
| `index` | the place of the index among the blobs and records, which are stored in the order of their lines. Without this line the index follows the header page (the canonical layout of WEX4-SPEC section 3) |
| `blob kind=N [id=N] file=F` | a blob. For kind 1 (a WNM2 model) the file is checked and the id is its FNV-1a; `id=` is then only compared. For another kind `id=` is stored (default 0) |
| `record [codec=raw\|wbp1\|wnr3\|auto] ...` | starts a record; the `tensor` lines after it are its tensors, in order. `auto` is the default: WBP1, or for an expert record WNR3 when `model=` is given, else RAW |
| `... planes=P chunk=K transform=T` | the parameters of a WBP1 record. A parameter left out is chosen as WEX4-SPEC sections 7 and 14 say the converters choose: `planes` = the element size in bytes when all tensors of the record have the same one, else 1; `chunk` (chunk_log2) 12; `transform` 1 for a record that holds one F16, BF16, F32 or F64 tensor of 64 KiB or more, or a part of a split one, when the size is a whole number of elements, and 0 for every other record (the converters group tensors under 64 KiB and code a group with 0, also a group of one). On `testdata/planes.wex4`, which `wex4/wex4_convert.py pack` wrote, these defaults give the file byte for byte (`TestWriterOtherChoices`; the grouping and the metadata were taken from the file) |
| `... expert=LAYER:EXPERT` | an EXPERT record: six tensors as WEX4-SPEC section 5.5 says. RAW or WNR3 |
| `... model=N cw=N cs=N [cz]` | a WNR3 record: `model` is the number, from 1, of its model among the `blob` lines (a blob of kind 1, before the record); `cw` and `cs` are the chunks (default 2048 and 512); `cz` is canonical zero |
| `... bytes=N` and `record ... [bytes=N] part` | a split tensor: the first record has one `tensor` line and takes the first N bytes of its file; each `part` record after it takes the next bytes, the last one what is left. Each part has its own codec |
| `tensor name=NAME dtype=D shape=A,B,... file=F` | a tensor of the record before it. `dtype` is a name of WEX4-SPEC section 5.4 or a code 0..255 (a code without a name is an opaque dtype); `shape=` with nothing after it is a scalar. The size of the file is the size of the tensor |
| `empty name=NAME dtype=D shape=...` | a tensor of 0 bytes; stored after all others |

A record asked as WBP1 or WNR3 whose coded form is not smaller than raw is stored RAW, as section 5.2 asks (for a WNR3 record with `cz`: the source bytes, without the VALUE_EXACT flag); the line `write` prints counts them.

```
wex4-manifest 1
meta notes.json
record
tensor name=embed.weight dtype=BF16 shape=1000,64 file=embed.bin
record codec=wbp1 planes=4 chunk=12 transform=1 bytes=4000000
tensor name=proj.weight dtype=F32 shape=2000,1000 file=proj.bin
record part
record codec=raw
tensor name=a%20name%20with%20spaces dtype=I64 shape= file=step.bin
empty name=unused dtype=F32 shape=0,7
```

`TestWriterFromScratch` in `write_test.go` holds a manifest with every kind of line, and `wex4check export` writes one for any file it accepts.

### A model for the WNR3 records

A WNR3 record needs a WNM2 model in a kind-1 blob. `wex4check train MANIFEST.txt N OUT.wnm2` makes one from the manifest: it counts the symbols of every expert record whose `model=` is N, as the decoder sees them (after canonical zero when the record has `cz`, after the ceil form, with the record's own anchors), and quantises the counts (WNR2-REFERENCE section 7, WNR3-SPEC section 5). Blob line N then names `OUT.wnm2`, and `write` uses it:

```
wex4-manifest 1
grid 1 2
blob kind=1 file=experts.wnm2
record codec=wnr3 expert=0:0 model=1 cz
tensor name=e0.w1.scale dtype=F8_E8M0 shape=64,4 file=e0/s1.bin
...
```

```sh
wex4check train manifest.txt 1 experts.wnm2   # the model, from the expert records that name blob 1
wex4check write manifest.txt experts.wex4     # the records, with that model
```

Which experts a model is trained on is the writer's choice. Of the two reference models in the files, one was trained on the 2 experts of its own file and one on 48 experts of which its file holds 2; both came out byte for byte when trained again on the same experts (below).

What the writer does not do: it does not read safetensors, it does not upgrade a WEX1 or WEX3 container in place, and it writes the reserved area of the header page as zero.

## What it checks

Everything of WEX4-SPEC section 6, on the whole file, in the order of that section:

| Step | What |
|---|---|
| 1, header page | magic, `header_sha256` (the root), major version, required feature bits, exact file size, every cap, the expert grid, `max_decoded` and `total_decoded` 0 when there is no record, `index_bytes` against the counts, the place of the index |
| 2, index | `index_sha256`, zero padding, every tensor entry (name range, UTF-8, rank, shape, size against the dtype), the name-order table, every blob entry, every record entry with its codec rules, the tensors that tile each record, split tensors and their parts, expert records and their six tensors, `max_decoded` and `total_decoded`, the tiling of the file by its extents. In place: the tables are not copied |
| 3, blobs | SHA-256, zero padding; a WNM2 model is opened (header, every CDF) and its FNV-1a id compared with the entry. A WNM4 model (kind 3) that a codec-8 record names is kept and opened when the first such record is decoded, as the model of that record's kind; an unused kind-3 blob gets the SHA-256 and padding checks only (WEX4-SPEC 5.1) |
| 3, records | SHA-256 of the stored bytes, then the decoder: RAW (zero padding), WBP1 (codec 6, WEX4-SPEC 7.2), WNR3 (codec 5, WNR3-SPEC sections 3 and 4: stream hashes, owner, canonical-zero rule, inverse of the ceil form), WNR4 kinds GG and I4 (codec 8, since 2026-11-23: the bindings of WEX4-SPEC 7.3, the geometry from the record's tensors, WNR4-SPEC sections 5 and 6; a failure makes the record undecodable). A record is given to the caller only after its decoder returned without an error. Since 2026-11-15 also the content digest of WEX4-SPEC section 5 over the decoded tensors, which `-manifest` compares with the manifest's `content` column |

The reject classes are the names the manifest uses for the C reader (`WEX4_E_HEADER`, `WEX4_E_RECORD`, ...). The specification says they are informative (WEX4-SPEC section 17, since 2026-11-15); here a class is the kind of rule that is broken.

## What was checked, and how

All on 2026-11-14 unless a row says otherwise. Local: Windows 10 desktop, Go 1.27.4 from MSYS2 UCRT64, one run each unless said. Linux: the `go-reader` job of CI (ubuntu-24.04, x86-64, the runner's Go, 1.24.13 on that day) runs `run.sh`: the tests and the 112 rows passed there too; the files of `wex4/out/`, the reference's verdicts and the differential run were done locally only. "The reference" is `wex4/wex4.py` and the C reader behind `wex4/out/wex4_tools.dll`.

| Check | Result |
|---|---|
| The 112 rows of `wex4/vectors/MANIFEST.tsv`: file SHA-256, verdict, reject class, root | 112 as the manifest says: 8 accepted with the manifest's root, 104 refused with the manifest's class. This was the result of the first run, before any change to the reader |
| 2026-11-15: the 136 rows of the manifest, with the 24 vectors of that day and the `content` column | 136 as the manifest says: 16 accepted with the manifest's root and content digest, 120 refused with the manifest's class. The changes to the reader for it: the content digest and its comparison with the column (`main.go`), and one class. The digest was first written in `verify.go`; since the two were merged it is the one of `digest.go`, which also gives the manifest's digest on all 16 accepted files (one local run, Windows, Go 1.27.4, at the merge). On `invalid/wnr3-record-smaller-than-its-streams-need.wex4`, the first vector of the minimum size of a WNR3 record, the reader gave the same verdict and reason as the C reader with the class `WEX4_E_RECORD`; the manifest says `WEX4_E_EXPERT`, and the reader now says that too (one word of `wex4.go`). The rest of the code that gives verdicts and classes (`wex4.go`, `wbp1.go`, `wnr3.go`, `rans.go`) is the code of 2026-11-14 |
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

### The content digest and the writer (2026-11-15)

Local: the same Windows 10 desktop, Go 1.27.4, gcc 16.2 and Python 3.14 from MSYS2 UCRT64, one run each unless said. "The reference" here is three programs built from the repository as they are: `python -I wex4/wex4.py verify FILE` (the Python reader with its own WBP1 decoder; it cannot decode WNR3 alone), the same with `--dll` and `wex4/wex4_tools.c` built with the reference WNR3 decoder (the Python reader's validation, the C codecs), and `c/tools/wex4_verify.c` (the C reader's validation of the header page and the index, and every SHA-256; it decodes nothing). The first two print the root and the content digest; a file "passes the reference" when all that apply accept it and print the root and the digest the Go writer printed. The tests of `write_test.go` do this in `run.sh`, and the `go-reader` job of CI runs them on Linux (ubuntu-24.04, x86-64, Go 1.24.13, the runner's gcc and python3) with `WEX4_NEED_REFERENCE=1`; the numbers below are from the local runs. They were made with the writer's default rotation of that day, which gave `transform` 1 also to a record of small tensors that are all floats; since 2026-11-21 such a record gets 0, as WEX4-SPEC section 7 says (the manifest table above), and the runs were not repeated with that default.

| Check | Result |
|---|---|
| The content digest, written from the paragraph of WEX4-SPEC section 5 (revision 2026-11-14), against the digest the Python reference prints | equal on the 10 files tried: the 7 valid vectors and `ceil-form.wex4`, `planes.wex4`, `ragged.wex4` of `testdata/` (6 different digests; the 5 files made from `generic.wex4` share one, and the two layouts of it give the same digest). The files hold a split tensor of 3 parts, tensors of 0 bytes, a scalar, an opaque dtype and the empty name. This was the first reading of the paragraph; no other reading was needed (`TestContentDigest`) |
| The canonical encoding (WEX4-SPEC section 5): each file is exported and written again from the exported manifest, and the bytes are compared with the file (`TestCanonical`) | **byte-identical for 9 of the 10 files**: 6 valid vectors (both layouts of `generic.wex4`, the two upgraded containers with their blobs, minor version 9, the optional feature bits) and the three files of `testdata/`. That covers the header page, the index, the WBP1 records (279 rANS streams with the CDF tables of their planes and 168 raw chunks in the 9 files; the four files with the content of `generic.wex4` hold the same 51 and 31) with the raw-or-rANS choice of every plane, and the 4 WNR3 records (2 with canonical zero, 2 without it and with 5 matrices in ceil form; 60 streams), whose anchors, flags and streams come out as the reference encoder wrote them; the 5 matrices in ceil form hold 11 even blocks at scale 255, which the transform leaves alone, and 18 at scale 254, which it moves to 255. The tenth, `bytes-in-the-reserved-header-area.wex4`, cannot come out of a writer that follows section 4 (reserved area written as zero): the copy is `generic.wex4`, and differs from the vector only in bytes 128 to 4095 |
| A file made from nothing but the writer (`TestWriterFromScratch`): 13 records, 17 tensors; WBP1 with 1, 2, 4 and 8 planes, `chunk_log2` 10 to 15, with and without the rotation, a size that is not a multiple of the plane count, the byte 255, the writer's own choice of parameters; a split tensor of 3 parts (WBP1, RAW, WBP1); a record of random bytes that is stored RAW; an opaque dtype; 3 tensors of 0 bytes; names with a space, `%`, a tab, non-ASCII, and the empty name; the index between the records | the Go reader gives every tensor back byte for byte; the file passes the reference |
| Expert records (`TestWriterExperts`): 6 slabs of `tiny-raw-upgraded.wex4` with the model of `tiny2-cz-upgraded.wex4`: 3 as WNR3 with canonical zero, 1 as WNR3 without it (not smaller on this data, so stored RAW), 2 RAW; two blobs, the index between the records. The 2 experts of `ceil-form.wex4` written again with the chunks 512:256 and 16384:16384. The same 2 with canonical zero (their slabs hold 4,059 and 3,979 nibbles 0x8) | the three files pass the reference (Python reader with the C codec, and `wex4_verify`); the second has the content digest of `ceil-form.wex4`. These are the first WNR3 records with chunks other than 2048:512 that this reader decoded. The third has record flags 0x0f and 0x0d: canonical zero and the ceil form in the same record, which no file of the reference writers has |
| The trainer (`TestTrain`): a WNM2 model trained on the 2 source slabs of `ceil-form.wex4` (the training set `testdata/README.md` names), and one trained on the 48 slabs of `tiny-raw-upgraded.wex4` with canonical zero | **both byte-identical to the models of the reference**: the blob of `ceil-form.wex4` (id 0xf0c9a065; 926 of its 1,701 weight contexts had no count) and the blob of `tiny2-cz-upgraded.wex4` (id 0x7d79f608; 1,395 without a count). With the trained models in place of the exported ones, the writer gives both files again byte for byte. The training set of the second is the one `wex4/build_and_test.sh` uses (`--train` on the 48 experts with `--cz`, then `--model` for the two-expert file): the 2 slabs of `tiny2-cz-upgraded.wex4` give another model, and so do the 48 slabs without canonical zero. Expert files shorter than their shapes are refused, as `write` refuses them (4 cases). The 48 slabs written as WNR3 with canonical zero and the trained model, in a file of their own: 48 WNR3 records, none stored RAW, 700,416 B; passes the reference |
| The tensors of `generic.wex4` packed with other choices (`TestWriterOtherChoices`): every record with the writer's default parameters, the index last. Then `testdata/planes.wex4` (written by `wex4/wex4_convert.py pack`) with the writer's default codec and parameters for every record | another root, the content digest of the vector; passes the reference. `planes.wex4` comes out byte for byte (221,184 B): chunk_log2 12, plane counts, rotation and the RAW fallback of its F16 and I16 record are the converter's on that file |
| A header page and nothing else (no tensor, no record: `TestWriterRefuses`) | 4,096 B; passes the reference |
| 37 manifests with one thing wrong each (a size that does not fit the shape, a name twice, a NUL or invalid UTF-8 in a name, rank 9, parts that do not add up, a rotation with one plane, an expert with one tensor, a required feature bit no reader knows, a word of another codec, a header line twice, ...) | all refused, no file left behind |
| A file whose last record has one byte changed, given to `export` (`TestExportRefused`) | refused; the output folder is not created |
| Random manifests (`TestWriterRandom`): records of the three codecs with random parameters or with the writer's own choice, 1 to 4 tensors each of typed and opaque dtypes, eight kinds of data, split tensors of 2 to 4 parts with a codec each, tensors of 0 bytes, blobs, metadata, version and feature words, the index at a random place, and 0 to 3 expert records per file made of the slabs of `tiny-raw-upgraded.wex4` and `ceil-form.wex4`, with random chunks (512 to 16384 : 256 to 16384) and canonical zero or not. 24 files in every run of `run.sh` (CI too); one local run of 400 (`WEX4_RANDOM_FILES=400`, seed 20261115): 1,947 records, 4,685 tensors (415 of 0 bytes), 240 split tensors, files of 761,856 B and records of 386,001 B at most | the writer wrote all 400: 1,196 RAW records, 582 WBP1, 169 WNR3 (103 with canonical zero, 54 of them also in ceil form); 722 records asked as WBP1 or WNR3 were not smaller and went RAW. The Go reader gave every tensor back (Z of the source for the weights of a value-exact record). The Python reader with the C codec and `wex4_verify` accepted all 400 with the writer's root and digest; the Python reader alone was asked on the 245 files without WNR3 records and under 300,000 B and agreed on all (1,045 runs of a reference reader in all) |

**What the canonical test does and does not show.** The source tensors of the vectors are not in the repository, so they were taken from the vectors with the Go reader, and the choices (grouping, codec and parameters of every record, order of the extents) were read from the same files. The test therefore shows that the specification, with WNR2-REFERENCE section 7 for the quantiser, determines every byte of these 9 files once those choices are known. It does not show that the pages say which choices the reference writers make. For the converter they do since the revision of 2026-11-15: WEX4-SPEC section 7 gives its plane count, `chunk_log2` and rotation (12 gives `planes.wex4` again; 10 and 11 are in the other files), and section 14 how it groups tensors and orders its records. The test took the grouping from the files, not from section 14. Until the revision of 2026-11-21 the sentence of section 5 listed only "the layout, the grouping of tensors into records, the codecs and their parameters, and the metadata" as the choices; the test needed more of them, and that revision names them ([SPEC-NOTES.md](SPEC-NOTES.md), E1).

### Codec 8 (2026-11-23)

Local: the same Windows 10 desktop, Go 1.27.4 from MSYS2 UCRT64, one run each. The files are the 23 codec-8 files of `testdata/` (its [README](testdata/README.md)), made with the repository's codec-8 tools (`wex4/wnr4gg.py`, `wex4/wnr4i4.py`) used as black boxes. Their manifest says accept or refuse, the verdicts of those tools; for this reader a refused copy has an undecodable record, except `wnr4-gg-bad-sha.wex4`, whose record SHA-256 does not match.

| Check | Result |
|---|---|
| The 23 rows of `testdata/wnr4-MANIFEST.tsv` (`TestWNR4ContainerManifest`, and `run.sh` through the command line) | the 6 valid files accepted with the manifest's root and content digest (the 4 RAW sources' digests, the other digest of `mut-verbatim`, the digest of `wnr4-i4-one` for `mut-base`); `bad-sha` refused as `WEX4_E_HASH` (exit code 1); the other 16 opened with the manifest's root and record 0 undecodable (exit code 3), each for the check its change aims at: a word, kind 0, flags, reserved bytes, ntiles, owner, model id, the verbatim padding, a base, the packing, an anchor, and the two index changes (blob 0, a blob id + 1) that only the bindings of 7.3 find. The first run gave all of these |
| The SHA-256 of every decoded record of the four valid files against `wnr4-records.tsv` (`TestWNR4ContainerRecords`) | 17 of 17 (13 codec-8, 4 RAW) |
| Single-bit flips of the codec-8 record of `wnr4-gg-one` and `wnr4-i4-one`, record SHA-256, index and header hashes recomputed: every bit of the first 512 bytes and 2,000 random bits after them (`TestWNR4ContainerFlips`) | no file refused and no panic. GG: accepted only inside the verbatim slices, with another content digest; I4: accepted only inside the bases, with the same content digest (the base of the block whose scales all escape); every other flip undecodable. A run with the first 1,024 bytes: 1,872 of 11,192 GG flips and 8 of 11,192 I4 flips accepted |
| Kind 1 (NV), kind 3 in a GG record, a blob entry of kind 4, a changed byte of the WNM4 model (its SHA-256 recomputed), G = 0 (`TestWNR4ContainerBindings`) | each makes record 0 undecodable; the file opens |
| `export` of a codec-8 file (`TestWNR4ContainerExport`, `run.sh`) | refused before anything is written: the writer has no codec 8 |
| `impl/rust-wex4/differential.py --go` ([impl/rust-wex4](../rust-wex4/README.md)), default seeds, on a second Windows PC (MSYS2 UCRT64, the C library built there, the two readers built here) | the 23 rows as above for both readers; 1,937 mutated copies of the four valid codec-8 files: 1,332 refused by the C reader with hashes only, and by both readers; of the 605 it opens, 111 accepted, 487 with an undecodable record and 7 refused as `WEX4_E_DATA` (a RAW record) by both readers. The Go and the Rust reader gave the same verdict on all 1,960 codec-8 files, and the same content digest on every one both accepted |

## Limits

- **No upgrade or downgrade of a WEX1 or WEX3 container, no safetensors pack or unpack.** The writer takes raw tensor files and a manifest.
- **The content digest** is the one of `digest.go`, written from WEX4-SPEC section 5 for the writer. The same day a second digest was written in `verify.go` on another branch, after the reference code had been read; when the two branches were merged, the reader was given the digest of `digest.go` and the second one was removed. The manifest's digests, which come from the C and the Python reader, are what it is checked against.
- **The writer was run on small files only:** no file above 843,776 B (`tiny-raw-upgraded.wex4` written again) and no record above 386,001 B (the random test). A record near 2^30 bytes, a file above 4 GB and an index near 2^28 bytes were not written. The source bytes of one record are held in memory twice or three times while it is encoded and checked.
- **WNR3 from the writer** was checked on synthetic experts of 13,056 B only (the tiny fixtures and `ceil-form.wex4`), with the two models of those files, which the trainer gives again. The trainer was run on these two training sets only (2 and 48 experts). It holds one training slab at a time and the counts (about 220 KB). No real expert was encoded or trained on, and no speed was measured: the encoder and the trainer are one thread and were not written for speed.
- **Not exercised in the writer:** a manifest line longer than 4 MiB (a name of 65,535 bytes fits); a scale table with no count in the trainer (every matrix of the training sets has scales; such a table would get 64 for each of its 256 symbols, SPEC-NOTES E8).
- **Not exercised by any file of the repository:** a record of an unknown codec (exit code 3 was run on three files made outside the repository, 2026-11-15). **Exercised only on files the Go writer made** (which the reference decoders accepted): `chunk_log2` 13, 14 and 15, and WNR3 chunk pairs other than the four of the vectors (2048:512, 512:256, 4096:1024 and 16384:16384, in `valid/wnr3-ceil-form-chunk-pairs-and-anchors.wex4`). Since 2026-11-15 the vectors also hold canonical zero together with the ceil form, and the error for a ceil-form block with coded scale 0.
- **No coverage-guided fuzzing.** The mutation test above is the only fuzzing. One attempt with Go's native fuzzer on whole files ran 18 inputs in 60 seconds and was dropped.
- **Speed and size were not measured.** One thread, one record in memory at a time (`max_decoded` bytes plus the stored record); the largest file read is 2,977,792 B. It was not run on a real container.
- **Go checks every index**, so an out-of-bounds access is a panic, not a read of other memory; a panic on a hostile file would still be a bug. None was seen in the runs above.
- **Little-endian or big-endian host:** the code reads integers byte by byte; it was run on x86-64 only.
- **Class names** follow the C reader's order for the files of the manifest only. On another broken file the class can differ from the C reader's (184 of the 20,000 files of the differential run). The verdict should not differ, and that is known only for the files of the table above. Of the readings of SPEC-NOTES part B, B3, B4 and B8 have vectors since 2026-11-15 and B5 is checked on three files; B2 is about the wording and no file decides it.

## Files

| File | What it is |
|---|---|
| `wex4.go` | header page and index: steps 1 and 2 |
| `verify.go` | step 3: blobs, records, the call of the decoders |
| `rans.go` | the rANS stream reader and the MurmurHash3 stream hash shared by WBP1 and WNR3 |
| `wbp1.go` | codec 6 decoder |
| `wnr3.go` | codec 5 decoder and the WNM2 model |
| `wnr4.go` | codec 8 (WNR4, DRAFT): the record header and tile offsets, the 32 rANS lanes of a tile, the lane hashes, the tail in the state and the end states, the WNM4 header and tables |
| `wnr4_gg.go`, `wnr4_i4.go` | the WNR4 decoders of kinds GG (2) and I4 (3) and their WNM4 models (SPEC-NOTES part F) |
| `wnr4_wex4.go` | codec 8 in the container (WEX4-SPEC 7.3): the WNM4 blob, its kind and id, record kinds 2 and 3, the GG and I4 geometry from the tensors (dtype 170 -> ggml type 42, 148 -> 20), the undecodable verdict |
| `digest.go` | the content digest of WEX4-SPEC section 5 |
| `encode.go` | the rANS stream writer, the count quantiser, the WBP1 and the WNR3 encoder |
| `train.go` | the WNM2 trainer |
| `manifest.go` | the writer's manifest |
| `write.go` | the writer: records, index, header page, and the read-back of the file |
| `export.go` | a file taken apart into the writer's input |
| `main.go` | the command line and the manifest check |
| `wex4_test.go` | the reader's tests (the first table above) |
| `write_test.go` | the tests of the digest, the writer and the trainer (the second table), with the calls of the reference readers |
| `wnr4_core_test.go`, `wnr4_gg_test.go`, `wnr4_i4_test.go`, `wnr4enc_test.go` | the WNR4 tests (`go test -run WNR4`): a test-only tile encoder and record builder, round trips, every end-state and record check, single-bit flips, the I4 lattice against exact rational rounding |
| `wnr4_container_test.go` | codec 8 in whole files (`go test -run WNR4Container`): the rows of `testdata/wnr4-MANIFEST.tsv` and `wnr4-records.tsv`, bit flips of the codec-8 records with the hashes recomputed, the bindings of 7.3, `export` |
| `run.sh` | builds, tests, checks the vectors and their content digests; builds the reference readers when it can |
| `testdata/` | four WEX4 files for what the vectors did not cover before 2026-11-15, the 23 codec-8 files of 2026-11-23 with their two tables, and the scripts that make them or their sources ([README](testdata/README.md)) |
| `SPEC-NOTES.md` | what the specifications left open and what the vectors did not exercise on 2026-11-14, and where each point went since; part E, the notes of the writer; part F, the notes of the WNR4 decoders |
