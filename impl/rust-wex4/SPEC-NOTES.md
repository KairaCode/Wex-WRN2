# Notes on the specifications, from writing the Rust reader

Parts A to D were written on 2026-11-15 while the reader was written from
[WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md),
[WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md), [GLOSSARY](../../docs/GLOSSARY.md),
[wex4/vectors/README.md](../../wex4/vectors/README.md) and `MANIFEST.tsv`, and before any of
`wex4/wex4.py`, `wex4/wex4_tools.c`, `wex4/test_wex4.py`, `c/` or the Go reader's code and notes was
opened. Part E was written after the reader passed all 112 rows, when those were compared.

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
  has an empty index (a file of only the header page).
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
`WEX4_E_*` names and gives the manifest's class on all 104 rejected rows. Where a rule falls:

| class | rules |
|---|---|
| `MAGIC` | magic |
| `HASH` | header SHA-256, index SHA-256, a blob's or a record's SHA-256 |
| `VERSION` | `version_major` |
| `FEATURE` | an unknown required feature bit |
| `HEADER` | a file shorter than a page, and every other rule of step 1 |
| `INDEX` | a non-zero byte in the index padding |
| `NAME` | name offset and length, UTF-8, NUL, the names filling the pool, the name order |
| `TENSOR` | rank, shape, size, zero-size tensors; and in the record pass, a tensor's record, offset, end and part count |
| `BLOB` | the rules of 5.1; in step 3 a blob's padding, its WNM2 model and id |
| `RECORD` | the rules of 5.2 (codec, flags, sizes, blob, own owner, VALUE_EXACT), `first_tensor` and `tensor_count`, the tiling of a record, continuation parts, the two totals, the WNR3 minimum size |
| `EXPERT` | the rules of 5.5, the owner of an EXPERT record inside the grid and increasing, a grid without an expert record |
| `LAYOUT` | the tiling of the file by its extents (section 3), the empty index at 4096 |
| `DATA` | a record that its decoder refuses, with the cross-checks of 7.1 |

Three of these placements come from the manifest, not from the text: the owner of an EXPERT record is
a rule of 5.2 but its class is `EXPERT`; "EXPERT allows codec 0 or 5" is next to the expert rules but
its class is `RECORD` (`expert-record-stored-as-wbp1.wex4`); and a file shorter than a page is
`HEADER` (A1). The order inside each pass is the order of the section's sentences where there is one,
else the order of the table's fields.

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

To be written: the comparison with the Go reader's notes, its test files and mutated vectors.
