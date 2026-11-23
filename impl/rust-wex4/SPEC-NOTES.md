# Notes on the specifications, from writing the Rust reader

Parts A to D were written on 2026-11-15 while the reader was written from
[WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md),
[WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md), [GLOSSARY](../../docs/GLOSSARY.md),
[wex4/vectors/README.md](../../wex4/vectors/README.md) and `MANIFEST.tsv`, and before any of
`wex4/wex4.py`, `wex4/wex4_tools.c`, `wex4/test_wex4.py`, `c/` or the Go reader's code and notes was
opened. Part E was written after the reader passed all 112 rows, when those were compared. Parts A to D
are left as they were written; where part E changed what the reader does, the note says so in brackets.

Section numbers are those of WEX4-SPEC unless a note names WNR3-SPEC.

## A. What the text leaves open, and what this reader does

- **A1. A file shorter than one header page.** Step 1 says "Read 4096 bytes at offset 0 and the real
  file size. Refuse unless, in this order: magic; ...". The list does not say where a short file is
  refused. This reader refuses it before it looks at the magic, as `WEX4_E_HEADER`
  (`only-half-a-header-page.wex4` has that class), so a 4-byte file that holds the magic is a
  `HEADER` refusal, not a `MAGIC` one.
- **A2. "`align(index_bytes)` fits before `file_bytes`".** Read as
  `index_offset + align(index_bytes) <= file_bytes`, computed without overflow.
- **A3. The empty index.** Section 4 says "`index_offset` is 4096 when the index is empty". It is
  written as a fact, not as a rule a reader checks. This reader checks it (`WEX4_E_LAYOUT`). No vector
  has an empty index (a file of only the header page). [Since part E: `WEX4_E_HEADER`, at the end of
  step 1; the C reader refuses such a file too, with that class (E3).]
- **A4. Where a tensor's place is checked.** Step 2 lists "every tensor entry satisfies 5.3 (names,
  rank, shape, size)" before "every record entry satisfies 5.2 and its tensors tile it". The table of
  5.3 holds `record` and `offset` too. This reader checks name, rank, shape and size in the pass over
  the tensors, and the record and offset of a tensor in the pass over the records, after the record's
  own fields. Only the parenthesis of step 2 says so; part D tells what the other reading gave.
- **A5. Rules of 5.2 that need 5.5.** The minimum size of a WNR3 record needs the shapes of the expert
  tensors. This reader checks it after the rules of 5.5, as the last check of the record.
- **A6. An unknown codec.** 5.2 says what `blob` holds for no codec and for codec 5; RAW and WBP1 have
  no blob. For an unknown codec on a record that is neither EXPERT nor VALUE_EXACT nothing is said; this
  reader accepts any `blob <= blob_count` and asks only that the record be strictly smaller than raw.
  It does not check the padding of such a record (it cannot decode it).
- **A7. The verdict on such a file.** Rule 6 of section 10 says the file opens and that record cannot
  be decoded. It is neither accepted in full nor refused: this reader prints `undecodable` with the
  root and the records, and exits with status 3.
- **A8. UTF-8.** "Valid UTF-8 (shortest form, no surrogates)" does not mention code points above
  U+10FFFF. This reader uses the UTF-8 check of the Rust standard library, which refuses them, as RFC
  3629 does. No vector has one.
- **A9. Rank 0.** The product of an empty shape is taken as 1 (a scalar). Not written; the scalar of
  `generic.wex4` (F32, 4 bytes) decides it.
- **A10. The order of the tensor table.** "Tensors are stored in data order" and the tiling rule
  together mean that the `first_tensor` of each record with tensors is the sum of the `tensor_count`s
  before it. The text does not write that equation; this reader checks it (`WEX4_E_RECORD`).
- **A11. A record with `tensor_count = 0`.** It must be a continuation part of the split tensor of
  the records just before it; the text implies this ("the next `k - 1` records are its continuation
  parts") but does not say what an orphan part is. This reader refuses one (`WEX4_E_RECORD`).
- **A12. `max_decoded`.** Section 4 says "largest `decoded_bytes`", step 2 "the totals equal": equality,
  not a bound. This reader requires equality.
- **A13. Reserved fields.** The blob and record entries have reserved fields whose rule column says 0;
  the header's reserved area is "ignored by readers". This reader refuses a non-zero reserved field in
  an entry and ignores the header area. The vectors decide all three.
- **A14. Step 3, which blobs.** "Per blob, when it is first used" fits an engine. A verifier has no
  first use, so this reader checks every blob, known kind or not, before the records: SHA-256
  (`WEX4_E_HASH`), zero padding, and for kind 1 the model checks and its FNV-1a id (`WEX4_E_BLOB`).
- **A15. The cross-checks of 7.1.** Owner, CZ bit and model id of a WNR3 record are compared with the
  index when the record is decoded, so a failure is `WEX4_E_DATA` (as the vectors say). They could also
  be read as rules of the index; they cannot be checked without the record's bytes.
- **A16. Which scale a weight block uses (WNR3).** WNR3-SPEC 4.1 uses `cs_b`, "the block's scale".
  That block `b` of matrix `m` (nibbles `32b` to `32b + 31` of the matrix in row order) has scale `b`
  of the scale tensor follows from the shapes `[rows, cols]` and `[rows, 16 * cols]` (bytes) of 5.5 and
  from "Block = 32 consecutive FP4 nibbles of a row with one UE8M0 scale"; it is not written as an
  equation.
- **A17. Byte order of the tables inside records.** The u16 stream lengths of WNR3 and WBP1 and the
  u32 chunk hashes of WBP1 are read little-endian, by the convention that all integers are; only the
  rANS state is said to be big-endian.
- **A18. WNR3 inside WEX4.** WNR3-SPEC section 3 checks `total_bytes == align4096(used)`; in WEX4
  that size is the record's `stored_bytes`, and the record is read at its `offset` (the WEX3 rule that
  places a record at the end of the slab does not apply).
- **A19. Canonical encoding and the reader.** Section 5 says the index encoding is canonical, and 7.2
  that the WBP1 encoder is. A reader is not told to refuse a WBP1 record whose mask or tables a
  canonical encoder would not have chosen; this reader accepts any mask that decodes.
- **A20. The anchor byte.** WNR3-SPEC erratum 11: an encoder writes 0 to 254 and the parser does not
  check it. This reader does not check it.

## B. Error classes

The specifications do not define the classes; `MANIFEST.tsv` gives one per rejected file, and the
vectors README says another reader only has to refuse. This reader names its refusals with the same
`WEX4_E_*` names and gives the manifest's class on all 104 rejected rows. Where a rule falls (the
table as it is since part E; the six rules marked E3 had another class before):

| class | rules |
|---|---|
| `MAGIC` | magic |
| `HASH` | header SHA-256, index SHA-256, a blob's or a record's SHA-256 |
| `VERSION` | `version_major` |
| `FEATURE` | an unknown required feature bit |
| `HEADER` | a file shorter than a page, every other rule of step 1, and the empty index at 4096 (E3) |
| `INDEX` | a non-zero byte in the index padding |
| `NAME` | name offset and length, UTF-8, NUL, the names filling the pool, the name order |
| `TENSOR` | rank, shape, size, zero-size tensors; and in the record pass, a tensor's record, offset, end and part count, a zero-size tensor listed by a record (E3), a split tensor that is not alone in its record (E3) or whose next record is not a part (E3), and the tiling of a record (E3) |
| `BLOB` | the rules of 5.1; in step 3 a blob's padding, its WNM2 model and id |
| `RECORD` | the rules of 5.2 (codec, flags, sizes, blob, own owner, VALUE_EXACT), `first_tensor` and `tensor_count`, continuation parts (an orphan part, the part of another tensor, the sum of the parts), the two totals |
| `EXPERT` | the rules of 5.5, "EXPERT allows codec 0 or 5" (E3), the owner of an EXPERT record inside the grid and increasing, a grid without an expert record, the WNR3 minimum size (F) |
| `LAYOUT` | the tiling of the file by its extents (section 3) |
| `DATA` | a record that its decoder refuses, with the cross-checks of 7.1 |

Two of these placements come from the manifest, not from the text: the owner of an EXPERT record is a
rule of 5.2 but its class is `EXPERT`, and a file shorter than a page is `HEADER` (A1). [Before part E
this paragraph named a third, "EXPERT allows codec 0 or 5" as `RECORD` from
`expert-record-stored-as-wbp1.wex4`. That was a misreading: the file is refused by the WBP1 rule of 5.2
(no blob, no flags) before that sentence is reached, so the manifest does not decide its class.] The
order inside each pass is the order of the section's sentences where there is one, else the order of
the table's fields.

## C. What the vectors do not exercise, seen from this reader

Found by reading the record headers of the valid files (2026-11-15):

- **WNR3:** the two records (`tiny2-cz-upgraded.wex4`) are canonical-zero records with no CEIL bit
  and chunks 2048:512. So the inverse of ceil form (WNR3-SPEC 4.2), a record without canonical zero
  and other chunk sizes are not exercised by any vector.
- **WBP1:** the four records of `generic.wex4` have 1 or 2 planes, `chunk_log2` 10, and either one
  rANS plane alone or a raw plane 0 with a rANS plane 1. No record has 4 or 8 planes, another chunk
  size, every plane raw or every plane coded with more than one plane, or a size that is not a
  multiple of the plane count.
- **Index:** no empty index, no blob of kind 18, no file above 4 GiB (no offset above 2^32), no
  record of an unknown codec (left out on purpose, vectors README).
- **Names:** no code point above U+10FFFF.

The tests of `src/testenc.rs` build such records with a test encoder written from the same pages
(ceil form with every case of 4.2, canonical zero on and off, chunks 512:256 to 4096:16384, WBP1 with
1, 2, 4 and 8 planes, chunk sizes 2^10, 2^12, 2^15, masks all raw, all rANS and mixed, transform 0
and 1) and decode them. That shows that this decoder and this encoder agree, not that they agree with
the reference.

## D. Before all rows passed

The first run gave the manifest's verdict and root on all 112 rows and its class on 103 of the 104
rejected ones. `record-decoded-bytes-0.wex4` was `WEX4_E_TENSOR` instead of `WEX4_E_RECORD`: the
first version checked that a tensor fits in its record in the pass over the tensors, so the tensor
inside the record of 0 decoded bytes was refused first. Step 2 lists only "names, rank, shape, size"
for the tensor pass (A4); with the place of a tensor checked in the record pass, all 104 classes agree.

## E. After all rows passed

Written on 2026-11-15, after the reader gave the manifest's verdict, class and root on all 112 rows.
Only then were the Go reader's notes read and the other readers run, as black boxes, on files built
here. "The C reader" is `c/wex4.h` inside `wex4/out/wex4_tools.dll` (built by
`bash wex4/build_and_test.sh`, gcc 16.2), called as `wex4.CLib(...).check(data, 3, True)`; "the Python
reader" is `wex4.check(data, True, C)` of `wex4/wex4.py`; "the Go reader" is `impl/go-wex4` built with
Go 1.27. All runs on one Windows 10 desktop (Intel Core i7-7700K, MSYS2 UCRT64, Rust 1.99.1,
Python 3.14), one run each; the same checks on Linux are the `rust-reader` job of CI.

### E1. What was run

- **The files of [impl/go-wex4/testdata](../go-wex4/testdata/README.md).** `ceil-form.wex4`,
  `planes.wex4` and `ragged.wex4` are accepted with the roots of that README. With `check -v` both WNR3
  records of `ceil-form.wex4` and the 17 tensors of the other two have the SHA-256 that the Go reader's
  tests take from the sources (`wex4_test.go`). `extra-page.wex4` is refused (`WEX4_E_DATA`,
  `stored_bytes` 8192 is not `align(3356)`), as the C and the Go reader refuse it. That file is the
  question of the Go reader's note A9; the sentence that answers it was already in the revision of 7.2
  this reader was written from.
- **The content digest** (section 5) that this reader prints equals the one that
  `python -I wex4/wex4.py verify FILE --dll wex4/out/wex4_tools.dll` prints, on 18 files: the 7 valid
  vectors, those three files, and the 8 WEX4 files that `wex4/build_and_test.sh` leaves in `wex4/out/`
  (RAW, WBP1 and WNR3 records; 48 WNR3 records with canonical zero in `tiny_nr3_cz.wex4`, 4 without in
  `synth.wex4`). `run.sh` checks the digests of the 10 tracked files.
- **[differential.py](differential.py)**, seed 20261115, with the Go reader:
  - 35 hand-made files (its list `EDGES` and five files built whole: a header page alone, two empty
    indexes, a 4-byte and a 100-byte file). The C reader accepts 7, refuses 27, and cannot decode one
    (an unknown codec). This reader and the Go reader give the C reader's verdict on all 35, and the
    Python reader gives it on the 34 that the C reader can decode.
  - 21,000 mutated files, of which 20,596 differ from their base. The C reader accepts 2,111, refuses
    18,378 and cannot decode 107. This reader and the Go reader give its verdict on all 20,596; the C
    and the Python reader agree on every one that the C reader can decode; this reader never panicked.
  - On Linux, the `rust-reader` job of CI (ubuntu-24.04, Rust 1.99.1, the C library built with gcc
    13.3, GitHub Actions run 43740748837 on commit fda816d) ran the same script without the Go reader: the 35 hand-made
    files and 6,866 mutated ones (`--count 7000`), the C reader's verdict on all of them; the class
    differs on 3 hand-made and 3 mutated files.

### E2. The verdicts agree; some classes do not

No verdict differs. The class is the C reader's on 24 of the 27 refused hand-made files and on 18,363 of
the 18,378 refused mutated files (the Go reader's is on 21 of the 27). The hand-made files where it is
not:

| file (`differential.py`, `EDGES`) | broken sentence | C | this reader | Go |
|---|---|---|---|---|
| `split-tensor-k4-record-7-not-a-part` | 5.3, "the next `k - 1` records are its continuation parts" | `RECORD` | `TENSOR` | `RECORD` |
| `first-part-larger-than-the-tensor` | 5.3, "`nbytes` is the sum of the `k` decoded sizes" | `TENSOR` | `RECORD` | `RECORD` |
| `expert-flag-on-a-continuation-part` | 5.5, "exactly 6 tensors" | `RECORD` | `EXPERT` | `EXPERT` |

The first row is the same sentence as `tensor-2-k2-record-2-not-a-part`, which the C reader refuses
with `TENSOR`; so do 12 of the 13 mutated files that break it (the thirteenth is the first row's case).
No class can agree with the C reader on both, and this reader gives `TENSOR` to both. The 15 mutated
files are 4 files of these three kinds and 11 files with two changed fields, where each reader gives the
class of the rule it checks first.

### E3. What the comparison changed in the code

No verdict changed. Six rules moved to the class the C reader gives, on files where the text does
not decide it (the table of part B marks them):

| rule | before | now | C | Go |
|---|---|---|---|---|
| an empty index whose `index_offset` is not 4096 | `LAYOUT` | `HEADER` | `HEADER` | `LAYOUT` |
| "EXPERT allows codec 0 or 5 only" | `RECORD` | `EXPERT` | `EXPERT` | `RECORD` |
| a record whose `tensor_count` reaches the zero-size tensors | `RECORD` | `TENSOR` | `TENSOR` | `RECORD` |
| a split tensor that is not alone in its first record | `RECORD` | `TENSOR` | `TENSOR` | `TENSOR` |
| a record after the first part of a split tensor that is not a part | `RECORD` | `TENSOR` | `TENSOR` on 13 of 15 files (E2) | `RECORD` |
| the tensors of a record do not reach its `decoded_bytes` | `RECORD` | `TENSOR` | `TENSOR` | `TENSOR` |

The first version (the commit that added this folder), built apart and put through the same run
(same seed, same 35 + 20,596 files), gives the same verdicts on every file; its class is the C
reader's on 18 of the 27 hand-made files and on 18,003 of the 18,378 mutated ones (now 24 and 18,363).
`check -v` was added: the SHA-256 of every decoded record and tensor, for the comparison of E1.

### E4. The Go reader's notes, side by side

| Go note | Now in the text | This reader |
|---|---|---|
| A1, A3, A4, A8 (the weight context, the rANS step, the WNM2 layout and checks) | WNR3-SPEC errata 9, 10, 12 | written from the errata; no question left |
| A2 (the order of a WBP1 record) | WEX4-SPEC 7.2 | the same |
| A5 (the order of the overflow check of a shape) | no [since 2026-11-15: 5.3] | the same reading: left to right, refused at the first overflow even when a later dimension is 0. `[0, 2^32, 2^32]` is accepted and `[2^32, 2^32, 0]` refused, by all four readers (E1). Parts A to D did not list it as a question: the loop was the obvious one |
| A6 (a zero-size tensor is one of 0 bytes) | 5.3 | the same; three hand-made files with an opaque dtype, 0 bytes and no 0 dimension are accepted by all four |
| A7 (`blob <= blob_count`) | 5.2 | the same |
| A9 (the exact `stored_bytes` of a WBP1 record) | 7.2 | the same; `extra-page.wex4` is refused |
| B2 (`n_p` when `p >= decoded_bytes`) | 7.2 | the same |
| B3 (the weight shape of 5.5, and the bound on `16 * cols`) | no [since 2026-11-15: 5.5] | the same reading: `[rows_m, 16 * cols_m]` of scale tensor m, and `16 * cols_m <= 65535`. Not listed in parts A to D either |
| B4 (the stream count of the WNR3 minimum) | 5.2 | the same |
| B5 (an unknown codec that names a blob) | no [since 2026-11-15: 5.2] | the same (A6 here) |
| B6 (the fields of the content digest) | 5 | computed, and equal to the Python reader's on 18 files (E1) |
| B7 (the classes) | no [since 2026-11-15: 17, the class is informative] | part B and E2 |
| B8 (a blob's offset is aligned only through the tiling) | no [since 2026-11-15: 5.1] | the same |
| C (the pages a reader needs) | - | the same three pages; WNR2-REFERENCE for the stream framing and the WNM2 model (A16 to A18 here) |
| D (what the vectors do not exercise) | - | part C here lists the same gaps, and four of the index: no empty index, no blob of kind 18, no file above 4 GiB, no unknown codec |

Notes of this reader that the Go reader does not have, with what the comparison showed: A1 (a file
shorter than a page: `HEADER` in all three readers that give classes, on a 4-byte and a 100-byte
file); A3 (an empty index elsewhere than 4096: refused by all four, classes in E3); A7 (a record of an
unknown codec: the C reader returns its status for "unsupported", the Go reader and this one report the
file valid with one record not decoded, and `wex4.py verify` prints `REFUSED: unsupported codec 7` and
exits 1); A8 (U+10FFFF is accepted in a name and the next code point refused, by all four); A20 (the
anchor byte: set to 255 or to 0 in a valid record with its hash recomputed, the record is refused by all
four readers, `WEX4_E_DATA` where there is a class; the Go reader and this one name the first scale
stream, whose hash fails: record bytes 24 to 27 are part of every stream's hash ident (WNR3-SPEC 4.5).
So these files do not show whether a record encoded with the anchor 255 is valid; that needs an
encoder that writes it, and none was run).

### E5. Open for the specification

What the comparison left undecided, each a sentence the specification could add:

1. **The reject classes.** The C, the Go and this reader do not give one class on 7 of the 27 refused
   hand-made files, and the C reader gives two classes to one sentence (E2). Either the specification
   defines a class per rule, or it says that the class is informative and only the verdict is
   conformance (as the vectors README already says). [WEX4-SPEC section 17 says so since 2026-11-15:
   the class is informative, and the class a conforming reader reports is its own.]
2. **The order of the overflow check of a shape** (Go A5): all four readers refuse
   `[2^32, 2^32, 0]` and accept `[0, 2^32, 2^32]`; the text does not say so. [WEX4-SPEC 5.3 says so
   since 2026-11-15, with these two shapes and a third as examples.]
3. **The bound on `16 * cols`** of an expert weight tensor (Go B3). [WEX4-SPEC 5.5 since 2026-11-15:
   the bound holds for every dimension of the six tensors, `16 * cols` included.]
4. **The empty index at 4096** (A3): written as a fact of section 4, not as a rule of step 1.
5. **What a verifier reports for a record of an unknown codec** (A7): section 10, rule 6 says the file
   is valid; the two reference readers report it differently from each other.
6. **The anchor byte 255** of a WNR3 record (A20, WNR3-SPEC erratum 11): not tested by any reader here.
   [WNR3-SPEC erratum 14 (2026-11-15): a decoder accepts any anchor byte, 0 to 255. The last record of
   the vector `valid/wnr3-ceil-form-chunk-pairs-and-anchors.wex4` was encoded with the anchors 255, 0
   and 122 and decodes; this reader accepts the file with the manifest's content digest (part F).]

## F. After the merges (2026-11-16)

This reader was merged with the 24 vectors of 2026-11-15 (136 rows) and the 7th column of the manifest,
the content digest of every accepted row, which the manifest mode now checks too. In the first CI run of
the merged tree (GitHub Actions run 43811647711) it gave the manifest's verdict, root and content digest
on all 136 rows and its class on 119 of the 120 rejected ones. The one that differed is
`invalid/wnr3-record-smaller-than-its-streams-need.wex4`: a WNR3 record below its minimum size, which
the manifest gives as `WEX4_E_EXPERT` and this reader gave as `WEX4_E_RECORD`. The rule is a sentence of
5.2 whose stream count needs the shapes of 5.5 (A5); this reader checks it in its pass over the expert
records, and now gives it that pass's class, `EXPERT` (part B). No verdict changed. In the next run
(43812551602) the class was the manifest's on all 120. Parts C and E5 are unchanged by the new vectors
except as the vectors README and WNR3-SPEC section 12 now say: the ceil form, records without canonical
zero and other chunk pairs are vectors since 2026-11-15; the four gaps of the index of part C are not.
The sentences of WEX4-SPEC revision 2026-11-15 and WNR3-SPEC erratum 14 answer items 1, 2, 3 and 6 of
E5 and five rows of the table of E4 (brackets there).

## G. WNR4 kinds GG and I4 (2026-11-23)

Written with the WNR4 decoders of `src/wnr4/` (kinds 2 and 3, and what they share), from
[WNR4-SPEC](../../docs/WNR4-SPEC.md) sections 1 to 6 and WEX4-SPEC section 7.3 only: the reference
codec under `wnr4/`, its tools and tests, the Python packers of `wex4/` and the Go reader's WNR4 code
were not opened. The decoders are not yet called by `check_file` (codec 8 still makes a file
`undecodable`); they are tested by their own unit tests, with a test-only encoder written from the same
pages (`src/wnr4/testenc.rs` and the `tests` modules of `gg.rs` and `i4.rs`), so those tests show that
this encoder and this decoder agree, not that they agree with the reference.

### G1. What the text leaves open, and what these decoders do

1. **Validity of a WNM4 blob** is not listed as a set of checks. These decoders require: the exact size
   (36,644 or 196,688 bytes), magic, version 1, the kind, word 3 = 14, words 4 to 7 exactly as the
   kind's header gives them (32, 2, 0, 0 for kind 2; 900, 68, 0, 0 for kind 3), every table a 14-bit
   cumulative table (first entry 0, last 16384, strictly increasing) and, for kind 2, every class map
   value below 32. The id is FNV-1a of the blob, 0 replaced by 1.
2. **Record bytes 16..19 of kind 2 and 3** replace the anchors and byte 19 of section 3: both kinds
   require them zero (5.3 and 6.3 say "0"; the record checks of 5.5 and 6.5 say so too).
3. **"Flags equal to the packing of the geometry"** (6.5). Inside WEX4 the packing is given only by
   record byte 13 (WEX4-SPEC 7.3), so the check is circular there. `i4::decode` takes the packing as an
   option: when the caller knows it, flags must equal it; when it does not (WEX4), only bits 1-7 must be
   0 and bit 0 is the packing. A record read with the other packing fails its tile hashes anyway,
   because the code bytes are hashed in the record's packing.
4. **The ggml type of a verbatim slice** (5.1) does not enter the record, and a WEX4 tensor of dtype
   20 or 42 (not 148 or 170) is a verbatim slice (7.3). `gg::Slice` holds a ggml type and treats exactly
   42 and 20 as coded, so whoever maps a WEX4 record to slices must give a verbatim tensor a type other
   than 42 and 20, not its dtype.
5. **A dtype-170/148 tensor whose size is not a multiple of 18**, or of 2^27 blocks or more, and slices
   above 2^32 - 1 bytes (2^31 - 1 verbatim): refused when the record is decoded (5.1), since WEX4 has no
   open-time rule for them.
6. **Sizes.** Every sum of section 3 and 6.1 is computed in 64 bits (the release build stops on an
   overflow), so `64 + 4 (ntiles + 1) + V + P` cannot wrap; an I4 geometry is checked against the
   2^32 - 1 byte limit matrix by matrix, so no product of rows and column groups overflows either.
7. **Escapes and bases.** The text lets an encoder escape a scale that lies on the lattice, and under a
   base of 0 every index k gives +0; these decoders accept both (the scale's bits are what is hashed and
   written). A base whose block has only escaped scales is never read and is still checked (0 or a
   positive normal finite f32), as 6.3 says.
8. **Order of the checks** is not normative (only valid / invalid is). These decoders check the
   geometry, the header fields, the model id and owner, the offset table, the size and the padding
   before any tile, then per tile: initial states, every symbol (a read past the tile's words and an I4
   index above 127 fail at once), the tails, the hash, the end states and the word count.

### G2. Against the reference's files

On the codec-8 test files made from the reference tools for both readers (`impl/go-wex4/testdata`,
`wnr4-*.wex4`, on their own branch at the time), with a throwaway harness that maps each codec-8 record
to its geometry as WEX4-SPEC 7.3 says and calls these decoders: all 17 records of the table of
decoded SHA-256 values (13 codec-8 records and 4 RAW ones, in the four valid files; G 1, 7, 32, 64 and
100, planar and pairs) decode to the listed SHA-256; the record of `wnr4-i4-mut-base.wex4` decodes to the bytes of `wnr4-i4-one.wex4`; the
verbatim byte change of `wnr4-gg-mut-verbatim.wex4` decodes, to other bytes; every other changed copy
is refused (kind 0, flags, reserved bytes, ntiles, owner, model id, padding, a word, a base, an anchor,
the packing). The harness is not in the repository: wiring codec 8 into `check_file` comes next.
