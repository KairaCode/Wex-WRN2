# WEX4 specification (format version 4.0)

**Format version:** 4.0, frozen on 2026-11-11. **Document revision:** 2026-11-13. **Status:** stable, in production. **Change log:** "Document history" at the end.

**In production** since 2026-11-11 for both models that use these containers: DeepSeek V4.1 REAP-192 (WNR3-CZ records) and Qwen3.8-Flash-Next MXFP4 (RAW records). What runs today, with files, roots and builds: [STATUS.md](../STATUS.md).

**Implementation:** reader and validator `c/wex4.h` (mirror of KAIRA's file) and the checker `c/tools/wex4_verify.c`; the engine loaders are in KAIRA master and are shown here as the patches `integration/kaira-wex4-loader-75932c1.patch` (DeepSeek store), `integration/kaira-wex4-first-read-4ba5568.patch` (first-read SHA-256 of section 8.2, `COLI_WEX_VERIFY`) and `integration/kaira-q38-wex4-loader-de17198.patch` (qwen38 store), and what followed them in the excerpt `integration/engine/deepseek_v4.wex4.wex-excerpt.c` (KAIRA 24f7e72); the Python reference, the converter, tests, reviews and research are in `wex4/`. Gates M1 to M4 of section 13.5 passed on 2026-11-11; the record of M3, M4 and of the install is `research/wex4-m3-20261111/RESULT.md`.

**One-paragraph summary.** WEX4 is a self-describing tensor container: any tensor, with its name, dtype and shape, like safetensors, but with lossless compression, hashes on everything, 4096-byte alignment and a binary index that a C reader validates in place without a JSON parser. It does **not** change the expert codec: a WNR3 record is stored byte for byte as in a WEX3 container, so the read path and the GPU decoder are the same, and the decode speed measured in the engine is the same within the noise (section 11.1). What changes is the container around the records (WEX3's 40-byte header and 72-byte entry table become a hashed header page and a hashed index), one new codec for the tensors WNR3 cannot code (FP8, BF16, F32: byte-plane rANS, "WBP1"), and a non-destructive in-place upgrade of a WEX1 or WEX3/WNR3 container: on the 123 GB V4.1 container it appended 6,717,440 B and rewrote one 4096-byte page (section 13.1).

Paths: `wex4/...`, `c/...`, `docs/...` are in this repository; `<KAIRA>` is the KAIRA tree; `<KAIRA>\lab\...` its lab folder.

**Conventions.**

- All integers are little-endian and are read byte by byte, never through a cast of an unaligned pointer. All offsets are from the start of the file unless a field says otherwise.
- "Must" and "must not" are requirements on a conforming reader or writer (section 17); "should" is advice. Sections and paragraphs marked *Informative* describe measurements, the engine and the history of the format: they are not needed to implement a reader or a writer.
- KAIRA is the (private) application this format was built for; its engines are builds of the colibri C engine. Where this document describes what KAIRA's engines do, it is informative. `<KAIRA>/lab` paths name files of KAIRA's lab that are not in this repository; copies of the runs that are quoted are under `research/`.

---

## 1. Goals, and what safetensors does and does not give

### 1.1 Goals

| # | goal | how (section) |
|---|---|---|
| G1 | **As safe as safetensors:** no code execution, bounded header, every length checked before use, no overlapping or dangling ranges, exact file size | 6, 9 |
| G2 | **Safer than safetensors:** corruption is detected (it is silent in safetensors), decoders fail closed, a canonical index encoding, no JSON parser in the reader | 5, 8, 9 |
| G3 | **As general as safetensors:** any tensor up to rank 8 (safetensors has no rank limit), the same dtype names, original names (up to 65,535 bytes, no NUL byte) and metadata kept, and the source files can be rebuilt byte for byte | 5, 14 |
| G4 | **Lossless compression per dtype:** MXFP4 experts (WNR3, canonical zero optional), FP8, BF16, F32, raw fallback; a record is never larger than the raw record padded to 4096 | 7, 12 |
| G5 | **Today's speed, or better:** one aligned read per expert, GPU-direct and batched decode unchanged, mirrors, bounded open time | 11 |
| G6 | **Cheap migration:** the production container is upgraded in place, reversibly, without rewriting 123 GB | 13 |
| G7 | **Versioned:** explicit version and feature flags with reader rules for unknown values | 10 |

Non-goals: a new expert codec (section 12 lists what a "WNR4" record could add and why it is not in this version), encryption, signatures (section 16), writing from the engine.

### 1.2 safetensors, verified facts

Source: the safetensors README and its reference parser `safetensors/src/tensor.rs` (main branch, read on 2026-11-11), and the published summary of the 2023 external security audit.

| property | safetensors | WEX4 |
|---|---|---|
| layout | 8-byte little-endian header length N, N bytes of UTF-8 JSON (may be padded with spaces), then one byte buffer | 4096-byte header page, then 4096-aligned extents: index, blobs, records |
| header cap | 100,000,000 bytes of JSON, all of it parsed | index at most 2^28 bytes, fixed-size binary rows, validated in place |
| tensor description | name -> dtype, shape, `data_offsets` [begin, end) into the byte buffer | name, dtype, shape, record, offset and size inside the decoded record |
| dtypes | BOOL, U8, I8, F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ, I16, U16, F16, BF16, I32, U32, F32, I64, U64, F64, C64, and the sub-byte F4, F6_E2M3, F6_E3M2 | the same 22, by code (section 5.4); other codes are opaque bytes |
| validation | header size, UTF-8, JSON, offsets contiguous from 0 with no hole and no overlap, size = shape x dtype, buffer ends at end of file | the same properties on records and tensors, plus alignment, exact tiling of the whole file, caps on every count, owner and codec rules |
| duplicate names | "disallowed" by the spec, "not all parsers may respect this" | impossible to accept: the name-order table must be strictly increasing |
| metadata | `__metadata__`: string -> string | opaque bytes in the hashed index; the converters keep the source header verbatim |
| integrity | **none** (no checksum anywhere: a flipped bit in a weight is loaded silently) | per stream / chunk (decode time), per record SHA-256, index and header SHA-256 (root) |
| compression | none | per record codec |
| version field | **none** | major, minor, required and optional feature flags |
| alignment | header padded to 8 bytes; tensors not aligned to pages | every extent starts on a 4096-byte boundary |
| code execution | none (its reason to exist) | none: tables and byte streams only |
| audit finding | "some missing validation allowed polyglot files, which was fixed"; "imprecisions in the spec" | exact file size is in the hashed header, every byte belongs to a hashed extent, trailing bytes are refused. Free-form areas remain (metadata, blobs of unknown kind, the reserved header area, tensor data), so a WEX4 file can still be made to parse as another format too |

What safetensors does better and WEX4 does not try to match: it is a public standard with many readers; a tensor can be memory-mapped and used in place (in WEX4 only RAW records can); its header is human-readable.

---

## 2. Terms

- **Extent:** a 4096-aligned byte range of the file. The header page is `[0, 4096)`; every other extent is the index, a blob or a record.
- **Record:** the unit of I/O and of decoding: `stored_bytes` on disk (a multiple of 4096) that decode to `decoded_bytes`. One record is read with one aligned read. Today's expert record (WNR3 or raw slab) is a WEX4 record unchanged.
- **Tensor:** a name, a dtype, a shape and a byte range of decoded data. Several tensors can share one record (an expert slab is 6 tensors; small tensors are grouped), and one large tensor can span several consecutive records (its **parts**).
- **Blob:** side data of a codec or of a tool, hashed like everything else: the WNM2 model of the WNR3 records; after an in-place upgrade also the old WEX3 header page and entry table.
- **Index:** the tables that describe blobs, records and tensors, plus the tensor names and the free-form metadata.
- **Root:** the SHA-256 of the header page: 32 bytes that identify the whole container (section 8).
- **Expert record:** a record flagged EXPERT: the canonical slab of one routed expert (scales w1, w2, w3, then packed FP4 weights w1, w2, w3), owner = `layer << 16 | expert`.
- **Value-exact:** a record that decodes to the source values but not the source bytes (WNR3 canonical zero: every FP4 -0 stored as +0). Flagged per record and in the header.

All integers are little-endian. "align(n)" is n rounded up to a multiple of 4096.

---

## 3. File layout

```
[0, 4096)   header page (section 4)
then, in any order, each starting on a 4096-byte boundary:
            index   (section 5)   align(index_bytes)
            blobs                 align(bytes) each
            records               stored_bytes each
```

**Tiling rule.** The header page, the blobs in table order, the records in table order and the index must cover `[0, file_bytes)` exactly once: no gap, no overlap, nothing after the last extent. It follows that blobs and records are stored in increasing offset order. Padding inside an extent (after a blob's bytes, after the index, after a record's payload) is zero.

Two layouts are in use:

| layout | order | written by |
|---|---|---|
| canonical | header, index, blobs, records | `wex4/wex4.py write()`, the safetensors converter: the index can be fetched with the second read |
| upgraded | header, old model/table blobs, records (untouched), old header page blob, index | the in-place upgrade (section 13.1): only appends |

Readers do not care which one they get: they read the header page, then the index where the header says it is.

---

## 4. Header page (4096 bytes)

| off | size | field | rule |
|---|---|---|---|
| 0 | 4 | magic | `0x57455834` (bytes `34 58 45 57`, "WEX4" in the convention of WEX1..3) |
| 4 | 2 | version_major | 4 |
| 6 | 2 | version_minor | 0; readers accept any value (section 10) |
| 8 | 8 | required_features | bit 0 VALUE_EXACT: the file may hold value-exact records (set by a canonical-zero build even if every record fell back to raw); a reader refuses the file if any bit it does not know is set |
| 16 | 8 | optional_features | bit 0 ST_SOURCES (the metadata holds safetensors sources, section 14); unknown bits are ignored |
| 24 | 8 | file_bytes | exact size of the file, a multiple of 4096 |
| 32 | 8 | index_offset | multiple of 4096, >= 4096 |
| 40 | 8 | index_bytes | exact (unpadded) size of the index, <= 2^28 |
| 48 | 4 | blob_count | <= 2^16 |
| 52 | 4 | record_count | <= 2^24 |
| 56 | 4 | tensor_count | <= 2^24 |
| 60 | 4 | num_layers | expert grid, <= 65536; non-zero only when the file has at least one expert record; `num_layers * num_experts <= 2^24` (this bounds the entry table a reader builds) |
| 64 | 4 | num_experts | <= 65536; zero iff num_layers is zero |
| 68 | 4 | max_decoded | largest `decoded_bytes` of any record (<= 2^30): the slab size a reader needs |
| 72 | 8 | names_bytes | size of the name pool, <= 2^28 |
| 80 | 8 | meta_bytes | size of the metadata, <= 2^28 |
| 88 | 8 | total_decoded | sum of `decoded_bytes` of all records: the memory a full load needs, known before anything is read |
| 96 | 32 | index_sha256 | SHA-256 of the `index_bytes` bytes at `index_offset` |
| 128 | 3936 | reserved | written as zero; **ignored by readers** (later minor versions may put optional fields here; the header hash covers them) |
| 4064 | 32 | header_sha256 | SHA-256 of bytes `[0, 4064)` = the **root** |

`index_bytes` must equal `64*blob_count + 72*record_count + 100*tensor_count + names_bytes + meta_bytes` exactly. The counts are capped first, so the sum cannot overflow 64 bits. The 2^28 cap on the index is the binding one: at 100 bytes per tensor plus its name it allows about 1.5 million tensors. `index_offset` is 4096 when the index is empty.

---

## 5. Index

Four tables and two byte areas, in this order, with no padding between them:

| part | size | content |
|---|---|---|
| blob table | 64 x blob_count | section 5.1 |
| record table | 72 x record_count | section 5.2 |
| tensor table | 96 x tensor_count | section 5.3 |
| name order | 4 x tensor_count | tensor indices sorted by name (bytewise, shorter prefix first) |
| name pool | names_bytes | the names, concatenated in tensor-table order |
| metadata | meta_bytes | opaque to readers; hashed |

The index encoding is **canonical**: once the writer has fixed the layout, the grouping of tensors into records, the codecs and their parameters, and the metadata, the index has exactly one byte representation (no optional fields, no free ordering, no padding choices). Two writers that make the same choices produce the same file and the same root (tested: the C and the Python codec, section 15). The root identifies a file, not a set of tensors: the same tensors packed with other choices give another root. A layout-independent identity is the **content digest** used by the tests: SHA-256 over, for each tensor in table order, its name length, name, dtype, rank, shape, size and bytes (it does not cover metadata, blobs or feature bits).

### 5.1 Blob entry (64 bytes)

| off | size | field | rule |
|---|---|---|---|
| 0 | 4 | kind | 1 = WNM2 model (must be 59,408 bytes, id != 0). Other kinds are opaque to readers. Used by the upgrade: 16 = old WEX header page, 17 = old WEX entry table, 18 = other old bytes |
| 4 | 4 | id | kind 1: the model's FNV-1a id (checked when the blob is loaded) |
| 8 | 8 | offset | start of the extent |
| 16 | 8 | bytes | 1 .. 2^30; the extent is `align(bytes)`, zero-padded |
| 24 | 32 | sha256 | of the `bytes` bytes |
| 56 | 8 | reserved | 0 |

### 5.2 Record entry (72 bytes)

| off | size | field | rule |
|---|---|---|---|
| 0 | 8 | offset | start of the extent |
| 8 | 4 | stored_bytes | > 0, multiple of 4096, `<= align(decoded_bytes)` |
| 12 | 4 | decoded_bytes | 1 .. 2^30 |
| 16 | 2 | codec | 0 RAW, 5 WNR3, 6 WBP1 (section 7). Unknown, on a record that is neither EXPERT nor VALUE_EXACT: the file is still valid, that record cannot be decoded |
| 18 | 2 | flags | bit 0 VALUE_EXACT, bit 1 EXPERT. Any other bit: the file is refused |
| 20 | 4 | blob | 0 = none, else 1 + index of the record's blob (codec 5: its WNM2 model) |
| 24 | 4 | owner | EXPERT: `layer << 16 | expert`, strictly increasing over the expert records. Else: the record's own index |
| 28 | 4 | first_tensor | index of the record's first tensor; for a continuation part, the index of the tensor it belongs to |
| 32 | 4 | tensor_count | tensors that tile this record (>= 1), or 0 for a continuation part |
| 36 | 32 | sha256 | of the `stored_bytes` bytes on disk |
| 68 | 4 | reserved | 0 |

Codec rules at open time: RAW has `stored_bytes == align(decoded_bytes)` and no blob. Every other codec has `stored_bytes < align(decoded_bytes)`: **a compressed record is strictly smaller than raw**, or the writer stores it raw. WNR3 needs EXPERT and a WNM2 blob, and at least `align(32 + 6 * streams)` bytes (streams counted at the longest chunk, as WEX3 does). WBP1 has no blob and no flags, and at least `align(16 + 6 * ceil(decoded_bytes / 32768))` bytes. VALUE_EXACT is legal only on a WNR3 record and only when the header's required bit 0 is set. EXPERT allows codec 0 or 5 only.

### 5.3 Tensor entry (96 bytes)

| off | size | field | rule |
|---|---|---|---|
| 0 | 4 | name_offset | into the name pool; must equal the sum of the previous names' lengths |
| 4 | 2 | name_bytes | 0 .. 65535 |
| 6 | 1 | dtype | section 5.4 |
| 7 | 1 | rank | 0 .. 8 |
| 8 | 64 | shape | 8 x u64; entries at and after `rank` are 0 |
| 72 | 4 | record | index of the record that holds the tensor (its first part) |
| 76 | 4 | record_count | 1 = inside one record; >= 2 = the tensor is exactly the concatenation of that many consecutive records; 0 = zero-size tensor |
| 80 | 8 | offset | byte offset inside the decoded record (0 when record_count != 1) |
| 88 | 8 | nbytes | size in bytes |

Tensors are stored in data order: record by record, and inside a record by increasing offset. Rules:

- **Tiling:** the tensors of a record cover `[0, decoded_bytes)` exactly once, in order: the first starts at 0, each next one starts where the previous ends, the last ends at `decoded_bytes`.
- **Split tensor:** a tensor with `record_count = k >= 2` is the only tensor of its first record; the next `k - 1` records are its continuation parts (`tensor_count = 0`, `first_tensor` = that tensor); `nbytes` is the sum of the `k` decoded sizes.
- **Zero-size tensors** (a dimension is 0) have `record_count = 0`, `record = 0`, `offset = 0`, and come after all data tensors.
- **Size:** for a known dtype, `nbytes * 8 == product(shape) * bits(dtype)` with every multiplication checked for overflow (as safetensors: the total must be a whole number of bytes). For every dtype, opaque ones included, the product of the shape must fit in 64 bits.
- **Names:** valid UTF-8 (shortest form, no surrogates), no NUL byte, unique. Uniqueness is not searched for: the name-order table lists every tensor once in strictly increasing name order, which a reader checks with one comparison per tensor, and then uses for binary search.

### 5.4 dtype codes

`1 BOOL, 2 U8, 3 I8, 4 F8_E5M2, 5 F8_E4M3, 6 F8_E8M0, 7 I16, 8 U16, 9 F16, 10 BF16, 11 I32, 12 U32, 13 F32, 14 I64, 15 U64, 16 F64, 17 C64, 18 F4, 19 F6_E2M3, 20 F6_E3M2, 21 F8_E4M3FNUZ, 22 F8_E5M2FNUZ` (the safetensors dtype names; bits per element 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 4, 6, 6, 8, 8). Any other code is an **opaque** tensor: its bytes and tiling are validated, its shape is not checked against its size, and a consumer that needs typed data must refuse it. MXFP4 experts need no new dtype: their weights are `I8` (two FP4 nibbles per byte) and their scales `F8_E8M0`, exactly as in the source safetensors; what makes them an expert slab is the record's EXPERT flag.

### 5.5 Expert records

An EXPERT record has exactly 6 tensors: scales of w1, w2, w3 (dtype U8 or F8_E8M0, shape `[rows, cols]`), then weights of w1, w2, w3 (dtype U8 or I8, shape `[rows, 16 * cols]`, in bytes), every dimension 1..65535, in the canonical slab order. This is exactly what a WEX3 entry describes, and `wex4_expert_entry()` rebuilds the legacy 72-byte `WexEntry` from it (file offset, stored size, shapes, offsets, codec). The engine store and the CUDA helper keep consuming `WexEntry` (section 13.2).

---

## 6. Validation (normative)

A reader works in three steps and uses nothing from a step that failed.

**Step 1: header page** (`wex4_parse_header`). Read 4096 bytes at offset 0 and the real file size. Refuse unless, in this order: magic; `header_sha256`; major = 4; no unknown required feature; `file_bytes` equals the real size and is a multiple of 4096; the three counts and the two sizes are within their caps; the expert grid is consistent and `num_layers * num_experts <= 2^24`; `max_decoded <= 2^30`; `index_bytes` equals the sum of section 4 and is `<= 2^28`; `index_offset` is aligned, `>= 4096`, and `align(index_bytes)` fits before `file_bytes`.

**Step 2: index** (`wex4_open`). Read `align(index_bytes)` bytes at `index_offset`. Refuse unless: `index_sha256` matches; the padding is zero; every tensor entry satisfies 5.3 (names, rank, shape, size); the name order is strictly increasing; every blob entry satisfies 5.1; every record entry satisfies 5.2 and its tensors tile it; expert records satisfy 5.5; continuation parts add up; the totals equal `max_decoded` and `total_decoded`; a non-zero grid has at least one expert record; the extents tile the file (section 3).

After step 2 every offset, size and name in the index is in bounds and consistent. Step 2 allocates nothing and takes time linear in the index size (measured in section 11).

**Step 3: data.** Per blob, when it is first used: read it, check its SHA-256 and zero padding; a WNM2 model also passes `wex_nr2_model_open` and its FNV-1a id must equal the entry's id. Per record: read `stored_bytes` at `offset` into a buffer of `align(decoded_bytes)`; optionally check its SHA-256 (section 8.2); decode with its codec into `decoded_bytes` bytes. Any failure returns an error and the output must not be used ("fail closed"); the CPU and the GPU decoder give the same verdict (WNR3: tested in `docs/reports/WNR3-VERIFY.md`).

**Memory budget.** Before step 3 the caller knows from the hashed header everything it will be asked to allocate: `max_decoded` (one decode buffer), `total_decoded` (a full load), `index_bytes`, and `num_layers * num_experts * 72` (the entry table). It compares them with its budget and refuses a file it cannot afford; the prototype leaves that comparison to the caller.

---

## 7. Codecs

| id | name | for | decode needs | checked at decode time |
|---|---|---|---|---|
| 0 | RAW | anything; the fallback of every other codec | nothing: the bytes are the tensor (can be read straight into place, or memory-mapped) | zero padding; the record SHA-256 when asked |
| 5 | WNR3 | MXFP4 routed experts | the container's WNM2 model (a blob) | everything of `docs/WNR3-SPEC.md`: structure, per-stream hash, owner, canonical-zero rule |
| 6 | WBP1 | FP8, BF16, F16, F32, F64, integers, opaque bytes | nothing (tables are in the record) | structure, per-stream hash (rANS planes), per-chunk hash (raw planes), owner |
| 1-4 | (WEX2/WEX3 scale tiles, palette3, WNR1, WNR2) | not written into WEX4 | | a reader treats them like any unknown id |
| 7+ | reserved | | | unknown to a reader: that record cannot be decoded, the rest of the file can (whole-file verification then reports the file as not verifiable). A new codec for expert or value-exact records must come with a required feature bit, because EXPERT allows codec 0 or 5 only |

Which codec a writer picks (the converters of this prototype): expert slab -> WNR3 when a model is given, else RAW; everything else -> WBP1 with `planes` = element size in bytes and `transform` = 1 for floats (F16, BF16, F32, F64); and in every case RAW when the coded record is not smaller.

### 7.1 Codec 5: WNR3 records, unchanged

The record bytes are exactly `docs/WNR3-SPEC.md` section 3. The index adds nothing to them and repeats two of their fields so that a record cannot be served for the wrong slot:

- record bytes 28..31 (owner) must equal the index entry's `owner`;
- record byte 27 bit 3 (CZ) must equal the index entry's VALUE_EXACT flag;
- record bytes 4..7 (model id) must equal the id of the entry's blob, which must equal the FNV-1a of the blob's bytes.

Canonical zero keeps its meaning and its limits (`docs/WNR3-SPEC.md` section 4.4): the record decodes to Z(source), value-exact and not byte-exact, and it is refused with the masked DeepGEMM path. A fallback RAW expert record stores the source bytes and carries no VALUE_EXACT flag.

### 7.2 Codec 6: WBP1, byte-plane rANS

For tensors with no trained model. The decoded bytes are seen as elements of `P` bytes; plane `p` is byte `p` of every element. Each plane is coded on its own, either with order-0 rANS or raw, whichever is smaller. The rANS is the WNR3 scale-stream coder (32-bit state, `L = 2^23`, 14-bit CDF, byte renormalisation, independent streams, u16 stream lengths, initial state `L + h23`), so the CPU code and the GPU kernel of WNR3 scale streams are reused; the only new thing is that the CDF travels in the record instead of in a model blob.

Record layout (`stored_bytes` in total, zero padding at the end):

| off | size | field | rule |
|---|---|---|---|
| 0 | 4 | magic | `0x31504257` "WBP1" |
| 4 | 4 | owner | the record's index (must equal the index entry's owner) |
| 8 | 4 | decoded_bytes | must equal the index entry |
| 12 | 1 | planes P | 1, 2, 4 or 8 |
| 13 | 1 | chunk_log2 | 10..15: a stream holds at most `C = 2^chunk_log2` bytes of one plane |
| 14 | 1 | mask | bit p = 1: plane p is rANS coded; 0: raw. Bits >= P are 0 |
| 15 | 1 | transform | 0, or 1 = rotate (needs P > 1 and `decoded_bytes % P == 0`) |
| 16 | 514 per rANS plane | CDFs | 257 x u16 cumulative frequencies (first 0, last 16384, strictly increasing), planes in order |
| | per plane | stream table | rANS plane: `S_p` x u16 stream lengths (each >= 4). Raw plane: `S_p` x u32 chunk hashes |
| | per plane | payload | rANS plane: its streams, concatenated. Raw plane: its `n_p` bytes |

`n_p` = number of bytes in plane p = `ceil((decoded_bytes - p) / P)`, `S_p = ceil(n_p / C)`. Streams and chunks are numbered globally: `g` counts through plane 0, then plane 1, and so on.

**Transform 1 (floats).** Before the split every little-endian element is rotated left by one bit. For BF16 and F32 this moves the whole 8-bit exponent into the top byte and the sign next to the mantissa bits, which are incompressible anyway. Estimated from an entropy survey of the production model's BF16 tensors (extrapolated): 0.6615 of raw with one rANS symbol per element, against 0.6687 with two for plain byte planes (section 12.1); one real shard through the codec gave 0.6625. The decoder rotates right after all planes are decoded; hashes are computed on the coded (rotated) bytes.

**Integrity inside the record.** Identity words `ident = {0x31504257, owner, g, P | chunk_log2 << 8 | transform << 24}`. A rANS stream starts at `x0 = L + (H32(ident || the stream's symbols, that is the plane bytes after transform 1) & 0x7FFFFF)` and the decoder requires exactly that final state after consuming exactly the stream's length (H32 = MurmurHash3_x86_32 with the WNR3 seed, `docs/WNR3-SPEC.md` section 4.5). A raw chunk stores the full 32-bit `H32(ident || chunk bytes)` in its table. So a stream decoded at another position, in another record, or with a damaged table fails with probability `1 - 2^-23` (rANS) or `1 - 2^-32` (raw), and the check runs in the same lane that decodes the bytes, on CPU and on GPU.

**Canonical encoder.** Given the bytes and the parameters `(P, chunk_log2, transform)` the record is fully determined: per plane, the 256 counts are quantised with `wex_nr2_quantize` (every symbol >= 1, largest remainders first, ties to the lower symbol); the plane is rANS coded iff `514 + 2*S_p + coded bytes < 4*S_p + n_p`. The C encoder (`wex4_bp_encode`) and the Python one (`wex4.bp_encode`) are byte-identical on 288 test cases.

**Bounds.** Every stream of at most 32,768 bytes costs at least 6 bytes, so a WBP1 record cannot expand more than about 5,461 times (about 330 times once the 4096-byte record granularity and a CDF table are counted for a 1 MiB record), and the index rule of 5.2 rejects a smaller record before any allocation. A decoder does a fixed amount of work per output byte and never loops on input it has not length-checked.

---

## 8. Integrity

### 8.1 Three levels

| level | covers | hash | where it is checked | cost |
|---|---|---|---|---|
| 1 | each rANS stream / raw chunk of a WNR3 or WBP1 record: decoded symbols and their identity (model, owner, stream index, anchors and flags) | MurmurHash3, 23 bits in the rANS state (32 bits for raw chunks) | always, inside the decoder, on CPU and GPU | WNR3: +1,352 B per record (+0.007 %), no measurable decode time (`docs/reports/WNR3-VERIFY.md`) |
| 2 | each record's stored bytes, each blob | SHA-256 in the index | tools always; the engine when asked (8.2) | 32 B per record; one hash pass over the bytes read |
| 3 | the index; the header page | SHA-256 in the header page | always, at open | one hash of the index (section 11) |

Together they form a hash tree of depth two: **root** (header) -> index -> records and blobs. Whoever knows the 32-byte root can verify any single record without reading the others, and the whole file in one pass.

### 8.2 What the engine checks, and when

- **Open:** level 3 always (the index is small). The root is compared with a pinned value if one is configured (section 13.3): a container replaced or damaged in its header or index does not open.
- **Every record read:** level 1 always (it is part of decoding). For RAW records level 1 does not exist; they are covered by level 2 only.
- **Level 2 on the read path:** a RAW record is SHA-256-checked the first time a process reads it (nothing else protects it). When a root is pinned (13.3), every record is checked at its first read, because only level 2 ties record bytes to the root; for that the store keeps the 32-byte record hashes next to its entry table (7,680 x 32 B). Without a pinned root, compressed records rely on level 1 on the hot path (hashing a 16 MB record costs milliseconds against 0.37 ms of GPU decode). Every tool checks level 2. `python -I wex4/wex4.py verify FILE --dll wex4/out/wex4_tools.dll` checks all three levels and decodes every record (WNR3 records need the C library). *(As implemented: the DeepSeek store does the first-read check only with a pinned root or `COLI_WEX_VERIFY=1`, since 2026-11-11. The qwen38 store, whose production records are all RAW, does it with `COLI_WEX_VERIFY=1` since KAIRA cb225b3 (2026-11-13; default 0, whole and striped reads tracked apart, a mismatch stops the engine); unlike the DeepSeek store, a pinned root does not turn it on there: section 13.2.)*

Why level 1 is not enough alone: it covers the decoded symbols, not the bytes on disk (in a WNR3 record the two chunk-size fields are outside it: a changed value that keeps the same stream layout decodes to the same bytes), it does not exist for RAW data, and it is an accident check: it has no key, so anyone who can run the encoder can recompute it. Why level 2 is not enough alone: it costs a pass over the data and tells nothing about a decoder that disagrees with the encoder. Measured on WNR3: 0 of 2,231,336 single-bit flips accepted wrongly with level 1 alone (WNR2, which had none, accepted 1.8-1.9 % of random flips silently).

### 8.3 Mirrors and split reads

A mirror is a byte copy of the file on another drive. Two copies are the same container iff their header pages are equal (the root covers everything else); the engine no longer needs to compare whole entry tables. Split reads (`COLI_WEX_WEIGHTS`) and partial reads work as today: every record is addressable alone, and a damaged record on one copy can be re-read from the other after a level 1 or level 2 failure.

### 8.4 What the hashes do not do

They detect corruption and substitution; they do not authenticate. An attacker who can rewrite the file can rewrite its hashes. Authentication needs the root pinned somewhere the attacker cannot write (configuration, or a signature over the root: section 16).

---

## 9. Threat model and safety

The file is untrusted input: it may be damaged (bit rot, a torn copy, a wrong file with the right name) or built by an attacker who knows the reader's code. The reader must never execute data, read or write out of bounds, allocate without limit, loop without bound, or return wrong tensor bytes as if they were right.

| threat | defence | tested by |
|---|---|---|
| code execution through the file | no code, no object graph, no format strings: fixed-size rows, byte streams and frequency tables | by construction |
| oversized header / index | header page fixed at 4096 B; index capped at 2^28 B; every count capped before it is multiplied | hostile cases: huge `index_bytes`, `record_count` 2^32-1, `names_bytes` 2^63 |
| integer overflow | all offsets 64-bit; counts capped; `offset + size` never computed without checking `size <= end - offset`; shape products checked per multiplication | hostile cases: offset 2^63, shape product overflow, `nbytes` 2^64-1 |
| out-of-bounds offsets | every extent is inside `[4096, file_bytes)` by the tiling rule; every tensor range inside its record; every name inside the pool | hostile cases; mutation fuzzing with guard pages |
| overlapping ranges, aliasing | exact tiling of the file by extents and of each record by its tensors; no two tensors share bytes; no deduplicated records | hostile cases: overlapping records, overlapping tensors |
| hidden or trailing data | `file_bytes` is hashed and must equal the real size; no byte of the file is outside a hashed extent; padding is zero. Not prevented: free-form content inside metadata, unknown blobs, the reserved header area and tensor data, so a polyglot file is still possible | hostile case: appended bytes; mode 0 fuzzing (every changed byte is rejected) |
| decompression bomb | decoded sizes are declared in the hashed index and capped at 2^30 per record; `total_decoded` is known before any data is read; a compressed record has a minimum size for its decoded size; decoders produce exactly the declared size | hostile case: a 4096-byte WBP1 record claiming 2^30 bytes |
| truncated or extended file | exact size check; a short read is an error | hostile cases |
| duplicate or ambiguous names | strictly increasing name order; UTF-8 shortest form; no NUL | hostile cases: duplicate order entry, NUL, invalid UTF-8 |
| parser differentials (two readers disagree on what the file means) | binary canonical index encoding; header and index validated by two independent implementations compared on mutated files (record decoding on the Python side uses the C codec for speed; the pure-Python WBP1 decoder is compared separately) | differential fuzzing: C and Python agree on every verdict and on the content digest |
| wrong record for a slot (table bug, cache bug, swapped records) | the record names its owner; the stream hashes include owner and stream index; the index repeats owner, model id and value-exact flag | WNR3 corruption tests (`c/tests/test_wex_format.c`), WBP1 owner test |
| corrupted compressed data | level 1 hashes inside the decoder; structural checks before any symbol is decoded; output is scratch until the decoder returns 0 | mode 1 fuzzing (record bytes mutated, SHA-256 off) |
| corrupted raw data | level 2 SHA-256 | mode 0 fuzzing |
| unknown future feature misread as a known one | major version, required feature bits and record flag bits are refused when unknown | hostile cases: major 5, unknown required bit, unknown record flag |
| GPU decoder disagrees with CPU decoder | same checks in both, error word per record, commit only when the error word is 0 (WNR3 today) | `docs/reports/WNR3-VERIFY.md` (CPU/GPU differential on mutated records) |
| malicious model blob (CDF tables) | `wex_nr2_model_open`: fixed size, every CDF starts at 0, ends at 2^14, strictly increasing; WBP1 CDFs the same | hostile tables are caught by the same check; mode 1 fuzzing |

Fuzzing policy for any implementation: (a) every byte of a file mutated with hashes left alone must be rejected; (b) header and index mutated **with the hashes recomputed**, so that the mutation reaches the validators, must be rejected or yield a file that fully verifies; (c) compressed record bytes mutated with the record SHA-256 switched off must be rejected by the codec or decode to the identical bytes. The file copy, each stored record and each decode buffer sit against inaccessible pages so that an out-of-bounds access is a crash, not a pass. Results of the prototype: section 15.

Limits that remain: a hash collision at level 1 has probability 2^-23 per stream (a corrupted record has thousands of streams, and a random corruption normally hits one); denial of service by a valid but huge file is bounded only by the reader's budget; side channels and authenticity are out of scope.

---

## 10. Compatibility and versioning

**Reader rules.**

1. Magic not `WEX4`: not this format (a WEX1/2/3 file goes to the old code path, which stays).
2. `version_major != 4`: refuse.
3. `version_minor`: informational. A newer minor may add optional header fields in the reserved area, blob kinds, dtype codes, codec ids and optional feature bits. None of these changes how existing fields are read.
4. Any unknown bit in `required_features`: refuse. This is the only way a newer writer may change the meaning of existing data, and it makes old readers fail closed.
5. Unknown bits in `optional_features`: ignore.
6. Unknown record flag bit: refuse the file. Unknown codec id on a record that is neither EXPERT nor VALUE_EXACT: the file opens; decoding that record returns "unsupported", and a consumer lists at open the records it cannot decode. A new codec for expert or value-exact records needs a required feature bit (rule 4). Unknown dtype code: the tensor is opaque bytes. Unknown blob kind: ignored (still hashed and inside the tiling).

**Old readers.** An engine built before the WEX4 loader refuses a WEX4 file. `deepseek_v4` (before KAIRA 75932c1) fails the magic check in `wex_validate_header`, logs "Invalid or incompatible container" and falls back to safetensors as it does for any unusable container. `qwen38` (before KAIRA de17198, merged with 485e80b) stops with "is not a WEX1 or WNR3 container ... refusing": it never falls back to the FP8 experts. `c/tools/repack_v4_wex.py` and the lab tools refuse it for the same reason. No old reader can misread a WEX4 file as something else.

**Record codecs keep their own magic and version** (WNR3 is "WNR3", WBP1 is "WBP1"): a change of a codec is a new codec id, never a silent change of an existing one.

---

## 11. Performance

*Informative.*

### 11.1 Why the expert path keeps today's speed

| step of a token | WEX3 + WNR3 (before the upgrade) | WEX4 | evidence |
|---|---|---|---|
| find the record | entry table in RAM, index `layer * experts + expert` | the same table, rebuilt once at open by `wex4_expert_entry` | rebuilt table == original table, byte for byte, on the tiny containers and on the production container's real header, model and table |
| read | one unbuffered read of `total_bytes` at a 4096-aligned offset; split over two drives | the same offsets and sizes: the upgrade does not move a byte of any record | bytes `[4096, old size)` identical after the upgrade (tested) |
| decode | helper feature 16, GPU-direct, batched; about 0.37 ms per record at batch 16 on the 5090 | the same `WexEntry` and the same record bytes go to the same helper | 4 experts packed from the safetensors with the production model are byte-identical to the production records |
| integrity | per-stream hash in the rANS state | the same, plus the owner / model / value-exact cross-checks against the index (three integer compares per record) | |

So decode tok/s was expected to be unchanged, and gate M3 (section 13.5) measured it on the production container: mean 2.070 tok/s in WEX3 form and 2.077 in WEX4 form (+0.32 %, per prompt -0.45 to +0.96 %, inside the run-to-run noise), with identical tokens and logits. Conditions: U: only, 7 prompts of 64 tokens, 2 runs per form, the two forms run one after the other and not interleaved (`research/wex4-m3-20261111/RESULT.md`).

### 11.2 Cost of opening

| quantity | WEX3 (before the upgrade) | WEX4 (production container, measured) |
|---|---|---|
| bytes read at open | 4096 + 59,408 (model) + 552,960 (table) | 4096 + 6,712,992 (index) + 59,408 (model) |
| validation | table checks | `wex4_open`: **26.9 ms** in C, about 20 ms of it the scalar SHA-256 (328 MB/s; a SHA-NI implementation hashes at 2.4 GB/s, about 3 ms) |
| kept in RAM | the 552,960-byte table | the same table; the 6.7 MB index can be dropped after open |
| file size | 123,306,192,896 B | +6,717,440 B (0.0054 %) |

### 11.3 CPU decode of WNR3 records: 6-7x on one thread, about 30x on eight, no format change

The reference decoder (`wex_nr2_decode`) is scalar and single-threaded: about 0.5 s per V4.1 record. `wex4/research/fast-cpu-decode/wex_fast_decode.h` decodes the same bytes to the same output with the same verdicts: OpenMP across streams (they are independent), an AVX2 16-way compare instead of the binary search, 4 streams interleaved in one loop, the hash and the ceil inverse fused into that loop. Measured on real production records (decoded size 18,800,640 B, best of 5, desktop Ryzen 9 7950X3D; an independent verification pass reproduced the numbers and extended the test to held-out records):

| decoder | 1 thread | 8 threads | 16 threads |
|---|---|---|---|
| reference | 505-590 ms (about 35 MB/s) | 76 ms per record when 8 records are decoded at once | |
| fast, decode only | 72-90 ms over a session on a busy machine (5.6-7.1x; best runs 71.5-72.7) | 11.3-13.8 ms (12-16 ms is the realistic range on a busy machine) | 8.8-9.5 ms |
| fast, as a drop-in in `wex_decode_nr2_record` (scratch buffer + copy included) | 92-94 ms | 17.3-18.4 ms (about 30x the shipped wrapper) | |
| laptop i7-7700HQ, 1 record (not independently verified) | 1077 -> 129 ms | 38 ms | |

Correctness: byte-identical to the reference on 14 production records (6 + 8 held-out, including the smallest and the largest record of the container) and to the safetensors source (with -0 set to +0) on 3; 36,000 + 20,000 mutated records with identical accept/reject and identical bytes; 20,000 more with the stream-hash comparison removed from both decoders so that structural differences cannot hide behind it (1,521 accepted by both, no mismatch); 10 of 11 deliberately broken decoders caught (the missed one removes the model-id check, which the mutation test cannot see because it always uses the right model). Built into the prototype with `-DWEX4_FAST_NR3`: `wex4/build_and_test.sh` builds the tools with the fast and with the reference decoder and requires identical counts in every fuzz mode.

What it is for: the CPU fallback (today about 1,400 times slower than the GPU path, with 8 threads about 40 times), the tools, and above all **full verification**: extrapolated from the per-record times (no full pass was run), decoding all 7,680 records on the CPU takes about 65 minutes with the reference on one thread and 1-2 minutes with the fast decoder on 8-16 threads, plus the read, so "every record CPU-decoded" becomes a routine check instead of a long run. What it is not: a replacement for the GPU decoder (0.37 ms per record), and it does not reach the "2 GB/s on 3 cores" that an NVMe-to-RAM CPU path would need (it gives about 0.26 GB/s per thread).

Limits: it takes `omp_get_max_threads()` threads unless told otherwise (the engine must set a count); concurrent calls from several threads were not measured; the scale kernel's read-ahead bound check could not be triggered by any test (the weight kernels' was).

### 11.4 Non-expert tensors (WBP1)

Scalar C, one thread: 161.5 MB/s of decoded BF16 (one rANS plane after the rotate; the decoder uses a 16384-entry slot table per plane instead of a search). Records and streams are independent, so this scales with threads like 11.3; no SIMD or threaded WBP1 decoder was written. These tensors (11.1 GB in the production model) are loaded once at start-up, not per token: compressing them saves 2.6 GB of disk (section 12.1) and, at one thread, costs more start-up time than the smaller read saves. Whether to move them into the container at all is an open question (section 16).

### 11.5 What a format cannot make faster here

Measured before this work (`docs/WNR2-REFERENCE.md`, `docs/reports/WNR3-VERIFY.md`, `research/wnr2-fastbench-20261109`): the expert wait on disk is the largest part of a token (58 % on one drive; 130 ms per token with the two-drive split); the GPU decode of a record at batch 16 (0.366-0.373 ms) is already next to the PCIe copy time of the same bytes (about 0.32 ms); the engine's real batches are 1.6-1.8 records. Nothing in a record's skeleton speeds up the GPU kernels (section 12.2). Speed therefore comes from bytes read per token (compression, section 12), from the drives, and from caching and prefetch policy, none of which is a container property.

---

## 12. Compression: what was measured, what is adopted

*Informative.*

Every number below was produced by an experiment in `wex4/research/` and then re-checked by an independent verification pass that tried to refute it (re-reading the scripts, re-running the decisive part, sampling other data). Corrections from that pass are already applied. "Extrapolated" means sample mean times population size.

### 12.1 Non-expert tensors: 11.1 GB -> 8.5 GB with WBP1

The production model has 11,130,045,256 B of tensors outside the routed experts, uncompressed in the safetensors today.

| dtype | bytes | WBP1 plain byte planes | WBP1 with transform 1 (rotate) | basis |
|---|---|---|---|---|
| F8_E4M3 | 7,086,800,896 | 0.8187 | not applicable (one byte) | entropy survey on a sample, extrapolated |
| BF16 | 3,874,076,672 | 0.6687 | **0.6615** (exponent coded, sign + mantissa raw; one rANS symbol per element instead of two) | entropy survey, extrapolated; a whole real shard through the real codec gave 0.6625 |
| F32 | 169,167,688 | 0.8439 | 0.8421 | all 86 large tensors measured |
| all three | 11,130,045,256 | **0.767, about 2.6 GB saved** | | |

Real files through the prototype codec (pack, verify, unpack, SHA-256 of the rebuilt file identical): one BF16 shard 970,533,624 -> 643,002,368 B; the non-expert tensors of one layer shard 168,090,456 -> 137,338,880 B (0.817).

Richer static models were measured and **not adopted**: together they are worth about 70-80 MB on 11.1 GB. Previous-symbol contexts gain at most 0.03 bit per element on BF16 and FP8 (less than their tables cost) and add a serial dependency; per-row and per-column tables lose to their own table cost; for F8_E4M3 the only cheap context is the block's existing E8M0 scale (0.8144 against 0.8187, about 30 MB); for BF16 the low byte conditioned on 4 bits of the high byte reaches 0.6588 at two symbols per element. Tables stay per record: a shared table is 0.1-0.3 point worse on tensors above 1 MiB.

### 12.2 MXFP4 experts: WNR3-CZ is at the floor of its model family

Population facts (all 7,680 production record headers read): weight payload 3.58969 bit per weight, mean record 16,055,413 B = **0.85398 of raw** (padded).

**Context models (`wex4/research/mxfp4-contexts`).** On 24 real experts, held-out, 14-bit quantised tables: no static context addition inside the limits (stream independence, table lookup per symbol) moves the weight cost by more than about 0.05 % of raw. The best combination (row level + position class + 4 table sets chosen per stream) gets -0.033 % on experts from layers the model never saw, about 6 KB per 18.8 MB record. In the coded domain the nibbles are almost independent given the block's scale class: dropping the previous magnitude from the context costs +0.0003 %, and the sign is an exact coin (1.00001 bit per non-zero weight). Earlier research found the same from the other side (adaptive, mixing and neural models: at most 0.001 bit per weight). There is no better model to find at this complexity.

**Record skeleton (`wex4/research/wnr4-record`).** A production record spends 108,266 B (0.576 % of raw) outside the model's own code length: 5.57 B per stream (2 B of u16 length, 3.57 B of rANS start state including the 23-bit hash) times 19,440 streams. Every skeleton change takes a slice of that and none can exceed it:

| change | saves, % of raw | on the 123.3 GB container | cost |
|---|---|---|---|
| coded stream-length table (groups of 32: minimum + width + packed residuals) | 0.130 | 188 MB (exact, all records scanned) | host offsets builder about 8 -> 35 us per record on a loaded machine (5 -> 25 us idle); GPU kernels untouched |
| start state stored in 3 bytes when it fits | 0.013 | 19 MB | three lines in the GPU stream open |
| records aligned to 64 B instead of 4096 B | 0.011 | 15.6 MB (exact) | the store reads the enclosing sectors; bytes read do not go down |
| **those three together ("WNR4 record")** | **0.154** | **about 0.22 GB (0.18 %)** | a new record codec with its own gates |
| weight:scale chunks 4096:1024 instead of 2048:512 | 0.288 | 416 MB | no new codec (chunks are per record already); half the streams: single-record GPU decode was 30-65 % slower on V4 records, mixed at batch 8-16, **not measured on V4.1** |
| chunks 8192:2048 | 0.432 | 624 MB | a quarter of the streams; unmeasured |
| last 5 nibbles of each stream in the start state | 0.205 | 297 MB | the per-stream check drops from 23 bits to 3 and needs a record digest verified on the GPU: unaudited |
| small initial state | up to 0.188 | 271 MB | a clamp on every symbol of the GPU hot loop |
| everything at once | 0.62 % of the container | 0.76 GB | all of the above |
| per-layer models | median +188 B per record | nothing | 40 model blobs |
| raw sign plane under canonical zero | **-2.76** (larger) | | gives back the whole canonical-zero gain |

**Decision for this version:** no new expert codec. The reachable gain without touching integrity or parallelism is 0.18 % of the container, less than the run-to-run noise of any speed test, for the price of new GPU code and a full gate cycle. If a new expert codec is ever made for another reason, it should include the first three rows (codec id 7, announced by a required feature bit).

### 12.3 Where the bytes still are

| lever | size | status |
|---|---|---|
| canonical zero | -2.74 % of raw | in production since WNR3 |
| non-expert tensors into the container with WBP1 | -2.6 GB of 11.1 GB (disk only: they are not read per token) | codec in this prototype; engine loader not written (section 16) |
| chunks 4096:1024 | -0.4 GB | a rebuild with other parameters; needs a V4.1 GPU measurement at the engine's real batch sizes first |
| WNR4 record | -0.2 GB | not worth a codec alone |
| richer context model | under 0.05 % of raw | measured, closed |
| lossless beyond that | none found | the nibbles are near-independent given the scale class |

Not in the container today and not measured here: the 384 MTP experts (the safetensors hold 151.6 GB of expert tensors, the container's 7,680 records decode to 144.4 GB; `docs/WNR3-SPEC.md` section 8). They have the routed-expert shape, so they could become WNR3 records if the engine wants them there.

### 12.4 A faster expert codec is possible, at a small size cost (not adopted)

The same study measured a deliberately simpler weight model: context = the block's scale class only (fixed for the whole block), 8 magnitudes, sign as a raw bit, 12-bit tables. On one CPU thread it decodes 1.75-2.0 times faster than a plain AVX2-search decoder of today's streams in the same harness (about 200-216 ms -> 101-120 ms per record). It was not compared with `wex_fast_decode` (72-90 ms on today's format), which is already faster; whether the simpler model gains anything with the same engineering is not measured. Also, its weight tables shrink from 57,834 B to 378 B; it costs +0.06 to +0.08 % of raw with the 23-bit stream hash kept. Its effect on the GPU decoder is unknown (GPU runs were out of scope), and the GPU is where production decodes. Recorded here as a candidate; it needs a GPU measurement before anything else.

---

## 13. Migration

### 13.1 In-place upgrade of a WEX1 or WEX3/WNR3 container

`python -I wex4/wex4_convert.py upgrade FILE.wex`. Steps:

1. Read the old header page, the WNMD descriptor, the model blob and the entry table. Refuse anything that is not a raw (WEX1) or a WNR3 container with records contiguous in table order.
2. Hash every record (one sequential read of the container) and the areas in front of the records. The hashes seal the bytes as they are found: a RAW record that is already damaged is sealed damaged, and a WNR3 record is proven good only by decoding it (the `verify` of gate M3). Before anything is written, the new header page and index are run through the WEX4 validator; a container it would refuse is left untouched.
3. **Append** a copy of the old header page and then the WEX4 index at the end of the file; flush. If this is interrupted, every byte of the WEX3 container is intact, but old readers refuse a file that does not end with its last record: the repair is truncating it to the old size, which the tool prints when it is run again.
4. **Overwrite the header page** (one 4096-byte write) with the WEX4 header; flush. This is the switch.

Nothing else is written. The model blob stays where it is and becomes blob kind 1; the old entry table becomes an opaque blob (kind 17); the old header page copy is blob kind 16. `downgrade` restores the old page from that blob and truncates the file to its old size: the result is byte-identical to the container before the upgrade (tested). If the header page itself is unreadable (a torn switch), `downgrade` finds the saved page without it, as the first page from the end that is a WEX1/WEX3 header whose records end exactly there, and restores the container the same way (tested with a destroyed header page).

Cost on the production container (measured on a sparse stub with the real header, model and table; section 11): the index adds a few MB to 123.3 GB, and step 2 is one pass of reading and hashing. On the real file at gate M3 the upgrade took 93.6 s and added 6,717,440 B (`research/wex4-m3-20261111/RESULT.md`); the two 64 GB Qwen3.8 containers took 42 s and 39 s and grew by 26,091,520 B each (`research/qwen38-speed-20261111/README.md` section 3).

The tensor names of an upgraded container are generated from the grid (`layers.{L}.ffn.experts.{E}.w{1,2,3}.{scale,weight}`, dtypes F8_E8M0 and I8), which are the names of the source model. `--names qwen38` generates the Qwen3.8 names instead (default `--names deepseek`).

### 13.2 Engine store (`deepseek_v4.c` `v4_wex_load_manifest`, `expert_store_wex.c`)

*Informative.*

The store's in-memory state stays what it is: a table of `WexEntry` and a registered model. Only the way it is filled changes:

- read the header page; if the magic is WEX4: `wex4_parse_header`, read the index, `wex4_open`, compare the root with the configured one (13.3);
- load the WNM2 blob named by the expert records (SHA-256, `wex_nr2_model_open`, id), register it as today; refuse a container whose WNR3 records name more than one model blob (the store registers one);
- fill the entry table with `wex4_expert_entry()` for every EXPERT record (absent experts stay all-zero and are served from safetensors, as today); header flags for the existing checks: NEURAL_RANS3 when any record is codec 5, CANONICAL_ZERO when any record is VALUE_EXACT (the canonical-zero refusal of `coli_wex_container_refused` applies unchanged); the per-record owner check gets that record's own VALUE_EXACT flag; the slab size is the largest `align(decoded_bytes)` over the expert records; the 32-byte record hashes are kept beside the table for the level-2 checks of 8.2;
- everything after that (RAM cache, read split, mirrors, owner check, GPU-direct fill, batch decode, CPU fallback) sees the same entries and the same record bytes as today. The prototype test rebuilds the legacy table from an upgraded container and compares it with the original table byte for byte.

Mirror pairing compares the 4096-byte header pages instead of header + table. The reference for this loader is `wex4/wex4_store.h` (`wex4_store_open`: header, pinned root, table budget, index, entry table, record hashes, legacy flags, slab size, one verified model; `wex4_store_check`: the first-read SHA-256 of 8.2), tested on three upgraded containers (raw, WNR3, WNR3-CZ). The engines do not include that file: they do the same steps with `c/wex4.h` directly, with no change in the hot path. In KAIRA's `deepseek_v4.c` it is `v4_wex4_load_manifest`, called from `v4_wex_load_manifest` (patch `integration/kaira-wex4-loader-75932c1.patch`; the first-read check is `integration/kaira-wex4-first-read-4ba5568.patch`); in the qwen38 engine it is `q38_wex4_manifest` in `c/qwen38_wex.h` (patch `integration/kaira-q38-wex4-loader-de17198.patch`), which checks header, index and root at open; since KAIRA cb225b3 it also does the per-record first-read SHA-256 when `COLI_WEX_VERIFY=1` (default 0).

### 13.3 `app\kaira.json`

*Informative.*

One optional setting next to the container path: the expected root (64 hex digits). As implemented it is the environment variable `COLI_WEX_ROOT`, which the host profile sets (in KAIRA, the `env` of the model profile in `app\kaira.json`). With it the engine refuses a container whose header or index is not the one that was verified at install time (log line `WEX4 root ... is not the pinned COLI_WEX_ROOT`), and the DeepSeek engine also turns the first-read SHA-256 on (`COLI_WEX_VERIFY`, section 8.2). Under a pinned root a container that is not WEX4 is refused too, because it has no root and no record hashes (KAIRA cb225b3): the DeepSeek engine then reads the safetensors, the qwen38 engine stops (log line `a container that is not WEX4 is not the pinned COLI_WEX_ROOT` in the DeepSeek engine). Without it any valid WEX4 container opens. No other key changes; the container path may keep its name.

### 13.4 CUDA helper (`kaira_wex_cuda.dll`)

*Informative.*

No change for experts: the helper takes a `WexEntry` and the record bytes, both identical. Feature 16 (WNR3) is all a WEX4 expert container needs. Only if non-expert tensors are to be decoded on the GPU does the helper need a WBP1 entry point; its kernel is the scale-stream kernel `wex_cuda_nr2_scales` with the CDF taken from the record and an output stride of `P` (plus the rotate for transform 1). That is a new feature bit and its own gates; nothing in this version depends on it, because the non-expert tensors are loaded once at start-up, where the CPU decoder is fast enough (section 11).

### 13.5 Order of work and gates

*Informative.*

| step | what | gate | needs |
|---|---|---|---|
| M1 | `wex4.h` + loader change in a KAIRA worktree | **passed 2026-11-11**: `test_wex_format`, `test_expert_store_wex`, `test_deepseek_v4` and the WEX3 profiles of `test_v4_wex_nr3` unchanged and green | nothing running |
| M2 | the engine test `c/tests/test_v4_wex_nr3.c` on the same container in WEX4 form (non-CZ and CZ) | **passed 2026-11-11** (CPU stand-in helper): same bytes served; a pinned root (`COLI_WEX_ROOT`) accepted and a foreign one refused; a damaged header page or index refuses the container; a corrupted record and swapped records fall back to safetensors; the masked path refuses the CZ container | nothing running |
| M3 | upgrade the production container on one drive (U:), keep the C: copy as WEX3 until M4 passes | `verify` passes (every hash, every record CPU-decoded: minutes with the fast decoder, announced first); engine opens it; a fixed-prompt run gives the same tokens and tok/s within noise as the same drive read as WEX3 before the upgrade. **Passed 2026-11-11**: upgrade 93.6 s, `verify` 8 min 7 s, same tokens and logits, +0.32 % tok/s; the container was then downgraded and its whole-file SHA-256 was the original one (`research/wex4-m3-20261111/RESULT.md`) | the owner's OK; KAIRA stopped during the upgrade |
| M4 | upgrade the C: mirror (or copy the header page and the appended tail: the records are identical), pin the root in `kaira.json` | a level-2 pass (`verify`, or at least every record hash) on the C: copy after its header page and tail are written, then equal roots: root equality alone does not read the second copy's records. **Passed 2026-11-11**: both copies upgraded and verified in full, equal header pages and the same content digest; the root was pinned in the `v41gpu` profile with the install of the first-read check (same file) | the owner's OK |
| rollback | `downgrade` on each copy (seconds), remove the key | byte-identical to today | |

---

## 14. safetensors mapping

**Pack** (`python -I wex4/wex4_convert.py pack OUT.wex4 SRC.safetensors ...`). Every source header is parsed strictly (length cap, UTF-8, JSON object, no duplicate keys, only `dtype`/`shape`/`data_offsets`, contiguous offsets with no hole or overlap, sizes equal shape x dtype, data ending at the end of the file) before anything is used. Then:

- tensor names, dtypes and shapes go to the tensor table unchanged; a dtype string the table does not know becomes an opaque tensor;
- tensors under 64 KiB of the same element size share a record (up to 4 MiB); a tensor above the part size (64 MiB by default) is split into parts; each record is WBP1 or RAW, whichever is smaller;
- with `--experts raw|nr3`, the six tensors of each routed expert (`layers.L.ffn.experts.E.w1|w2|w3.weight|scale`) become one EXPERT record in the canonical slab order, RAW or WNR3 (`--model FILE.wnm2`, `--cz` for canonical zero), each WNR3 record verified by the CPU decoder before it is written;
- the metadata is a JSON document `{"wex4_meta":1,"sources":[{"name","bytes","sha256","header"}]}` holding, per source file, its name, size, SHA-256 and its **header text verbatim** (which includes `__metadata__`). The header's optional bit 0 is set. The C reader never parses this; only the converter does.

**Unpack** (`... unpack IN.wex4 OUT_DIR`) rebuilds each source file: 8-byte length, the verbatim header, then every tensor in its original offset order, and compares the SHA-256 with the recorded one. Tested byte-identical on synthetic files (all corner cases: empty tensor, scalar, sub-byte dtype, unpadded header, two files) and on the tiny V4.1 fixture (634 tensors, 5 dtypes) with RAW expert records. On the tiny fixture every non-CZ expert falls back to RAW, so the byte-exact round trip through non-CZ WNR3 records is tested on a second synthetic model with compressible experts (4 WNR3 records). Unpack re-parses the stored header strictly and requires every entry to match the container's tensor before writing; a file that fails its SHA-256 is not left in place.

**Value-exact containers.** With `--cz` the rebuilt file differs from the source exactly in the FP4 -0 nibbles (tested: every differing nibble is 0x8 in the source and 0x0 in the rebuilt file). Such a container cannot replace the safetensors as a byte-exact archive; a non-CZ container can.

---

## 15. Prototype and tests

*Informative.*

### 15.1 Files

| file | what |
|---|---|
| `c/wex4.h` | C header: format constants, SHA-256, `wex4_parse_header`, `wex4_open` (in-place validation), `wex4_find`, `wex4_expert_entry` (legacy `WexEntry`), WBP1 encoder and decoder, `wex4_decode_record`. Includes `c/wex_format.h` for the WNR3 codec and the rANS primitives; it always calls the reference WNR3 decoder |
| `c/tools/wex4_verify.c` | the checker of a whole WEX4 file: header page, index, every blob and record against its SHA-256, zero padding, in one read-only unbuffered pass on N threads (OpenMP, 8 by default); prints `COLI_WEX_ROOT=<root>`; exit 0 pass, 1 mismatch, 2 not WEX4. It hashes and does not decode. Tested by `c/tests/test_wex4_verify.c` with the in-memory fixture `c/tests/wex4_fixture.h` |
| `wex4/wex4_store.h` | the engine-side loader of section 13.2 as a reference with its tests; the engines carry the same steps in their own stores (section 13.2) and do not include this file |
| `wex4/wex4_tools.c` | DLL for the tests and the converters: whole-file check with content digest, reader-style entry points, legacy table, guard-page fuzz entry, in-process mutation fuzzer. Built with `-DWEX4_FAST_NR3` it puts the fast CPU decoder (`wex4/research/fast-cpu-decode/wex_fast_decode.h`) in place of the reference inside `wex4_decode_record`; `wex4/build_and_test.sh` checks that the two DLLs differ there |
| `wex4/wex4.py` | Python reference, standard library only, written independently of the C code: reader, validator, writer, WBP1 codec, WNM2 check. `info` / `verify` command line |
| `wex4/wex4_convert.py` | safetensors -> WEX4 (`pack`), WEX4 -> safetensors (`unpack`), WEX1/WNR3 container -> WEX4 in place (`upgrade`) and back (`downgrade`) |
| `wex4/vectors/` | the conformance vectors of section 17 and their `MANIFEST.tsv` |
| `wex4/test_wex4.py`, `wex4/build_and_test.sh` | every test below, and the check of the conformance vectors. `bash wex4/build_and_test.sh [--fuzz N]` builds the tools twice (fast and reference WNR3 decoder) and the nr3 encoder DLL, makes the tiny WEX1 and WNR3-CZ containers from `fixtures/v41-tiny`, and runs the tests: no GPU, no real model, under 2 minutes with the default `--fuzz 1000` |
| `wex4/research/fast-cpu-decode/` | the fast WNR3 CPU decoder (`wex_fast_decode.h`), its own harness, mutation test and results |
| `wex4/research/measure-real/` | the measurements on the real model and on the production container (read-only), `results.json`, and the log of the last full test run |
| `wex4/research/dtype-entropy/`, `wnr4-record/`, `mxfp4-contexts/` | the compression experiments of section 12, each with scripts, `results.json`, `RESULTS.md` and the scripts of its verification pass |

### 15.2 Tests and results (2026-11-11, desktop, gcc 16.2, Python 3.14; `wex4/research/measure-real/test_run.log`)

| test | result |
|---|---|
| SHA-256, C against `hashlib` | equal on 203 lengths |
| WBP1 codec | 288 cases (data kinds and tiny sizes x planes 1/2/4/8 x chunk 2^10/2^12/2^15 x transform 0/1): C and Python encoders byte-identical, both decoders exact, a record under another owner rejected by both |
| container | 12 tensors: several per record, one split in 3 parts with mixed codecs, zero-size, scalar, non-ASCII name, empty name, opaque dtype. Files written with the C and with the Python codec are identical; index-first and index-last layouts give the same content digest |
| safetensors round trip | 2 synthetic files (8 dtypes incl. sub-byte F4, empty tensor, scalar, padded and unpadded header): rebuilt byte-identical with the Python codec and with the C codec at 16 KiB parts. 6 malformed safetensors headers refused (short, oversized length, not an object, duplicate key, trailing byte, hole) |
| tiny V4.1 model (`fixtures/v41-tiny`, 634 tensors, 5 dtypes, 48 experts, 3,118,984 B) | raw experts 2,916,352 B; with a model and no CZ 2,977,792 B (all 48 experts fell back to RAW on this synthetic data; the file carries the 61 KB model); WNR3-CZ 2,781,184 B (48 WNR3 records). The two non-CZ files rebuild the safetensors byte-identical; CZ differs only in nibbles that were 0x8 and are 0x0 |
| non-CZ WNR3 inside WEX4 (`fixtures/make_synth_mxfp4.py`, 4 compressible experts, model trained on them) | 4 WNR3 records, 837,992 B of safetensors -> 729,088 B; rebuilt byte-identical; fast and reference decoder agree |
| engine-side loader `wex4/wex4_store.h` | on upgraded raw, WNR3 and WNR3-CZ containers: entry table equal to the original, legacy flags 0 / 32 / 96, model verified, every record passes the first-read SHA-256; a wrong pinned root, a table over budget and a flipped record bit are refused |
| torn header page | page 0 of an upgraded file overwritten with random bytes: `downgrade` restores the original container byte for byte |
| in-place upgrade | WEX1 (48 records), WNR3-CZ (48 records), WNR3-CZ (2 records): bytes `[4096, old size)` untouched; the legacy entry table rebuilt from the index equals the original table; every expert decodes to the WEX1 slab (to Z(slab) for CZ); `downgrade` restores the original file byte for byte |
| hostile files | 46 hand-made attacks (section 9), header and index hashes recomputed so that they reach the validators: all refused by C and by Python. 3 forward-compatible variations (minor 9, unknown optional bits, bytes in the reserved header area): accepted by both |
| fuzz, mode 0 (any byte, hashes checked) | 51,091 mutations on 3 files: 0 accepted (287 were no-ops) |
| fuzz, mode 1 (compressed record bytes, SHA-256 off) | 10,217 mutations: 0 accepted (52 no-ops): the codecs' own checks rejected every one |
| fuzz, mode 2 (header and index, hashes recomputed) | 51,091 mutations: 48,976 refused; 732 accepted with identical content (changes in metadata, optional bits, reserved bytes); 617 accepted as a different valid file (for example a renamed tensor) that still verifies in full; 766 no-ops; no fault |
| fast against reference WNR3 decoder | all three modes run through both builds: identical counts in every mode |
| differential fuzz | 2,042 mutated files through the C reader (guard pages) and the Python reader: same verdict on all, same content digest on the 99 accepted (header and index are validated independently; records are decoded by the C codec on both sides). Separately, 4,200 mutated WBP1 records through the C and the pure-Python decoder during the review: no disagreement |

All C fuzzing runs with the file copy, each stored record and each decode buffer placed against an inaccessible page, so a read or write past the end is a crash. **112,399 mutations (1,105 of them no-ops), 0 silent corruptions, 0 faults.**

### 15.3 On real data (read-only; `wex4/research/measure-real/results.json`)

| what | result |
|---|---|
| production container index, built in memory from the real header page, model and entry table | 7,680 records, 46,080 tensors: index **6,712,992 B**, the upgrade adds 6,717,440 B (0.0054 % of 123,306,192,896 B). `wex4_open` in C: **26.9 ms**. The legacy table rebuilt from the index equals the production table byte for byte. Record hashes were not computed (that is the full read of step M3) |
| 4 experts of layer 18 packed from the safetensors with the production model and canonical zero | the 4 WNR3 records are **byte-identical to the production records** |
| one whole BF16 shard (970,533,624 B) | WEX4 643,002,368 B (**0.6625**), 133 WBP1 records; pack 9.2 s, verify + decode 6.7 s, unpack 7.4 s (Python driving the C codec, one thread); rebuilt file SHA-256 identical |
| the non-expert tensors of one layer shard (F8_E4M3 + its scales + F32 + BF16, 168,090,456 B) | 137,338,880 B (**0.8171**); rebuilt SHA-256 identical |
| WBP1 decode, scalar C, one thread, BF16 | 161.5 MB/s of decoded output |
| SHA-256 | `hashlib` (SHA-NI) 2,396 MB/s; the header's portable C 328 MB/s |
| read + SHA-256 of 64 production records (1.03 GB) | 1,030 MB/s on one thread: about 2 minutes for the whole container (extrapolated) |

**Tested after this section was written** (gates M3 and M4, `research/wex4-m3-20261111/RESULT.md`): the engine on the real WEX4 container through the GPU path, and the full-container upgrade with every record hash checked and all 7,680 records decoded on the CPU, on both copies.

**Not tested:** the first-read SHA-256 of section 8.2 inside the engine beyond one off/on pair (same file); `wex4.py verify` on the two Qwen3.8 containers, which were upgraded and opened by the engine and on 2026-11-13 checked with `wex4_verify`, which hashes every record but decodes none (`research/wex-rd-20261112/README.md`); the prototype on the laptop (only the fast decoder was timed there, on one record); files above 1 GB through the Python writer; big-endian hosts.

---

## 16. Open questions

1. **Experts only, or the whole model?** WEX4 can hold every tensor, and the non-expert tensors would shrink by 2.6 GB. The engine today reads them from the safetensors at start-up through its own loader; moving them means a second loader path (names -> tensors through `wex4_find`, WBP1 decode on CPU threads) and its own tests. Nothing in the expert path depends on the answer. The in-place upgrade produces an experts-only container either way.
2. **Where the root is pinned** (section 13.3), and whether a signature over the root is wanted. The format needs no change for either. Answered for V4.1: `COLI_WEX_ROOT` in the `v41gpu` profile. For Qwen3.8: both copies were verified with `wex4_verify` on 2026-11-13 (root `db5389f6...c0df`; `wex4.py verify` was not run on them), and the root is recorded and **not pinned**. The A/B of the pin measured only the check at open, because the qwen38 engine hashes records only with `COLI_WEX_VERIFY=1` (decode +0.5% to -9.1% per request, one pair: `research/wex-rd-20261112/README.md`). The signature is still open.
3. **SHA-256 or a faster tree hash.** SHA-256 is in every standard library and runs at 2.4 GB/s with SHA-NI; the header's portable C version runs at 0.33 GB/s. With SHA-NI a full-container pass takes about 2 minutes (extrapolated from 64 records); with the portable C version on one thread about 6. BLAKE3 would allow partial verification inside a record; no present use needs it.
4. **Level-2 cost at run time:** section 8.2 requires the first-read check for RAW records and, with a pinned root, for every record; measured on one off/on pair with the mirror and a pinned root: same logits, total generation time +0.1 %, the cost sits in start-up (+5.6 s) and the first prompt (+4.9 s). A full ABBA set is still to run (a background scrub remains the alternative for compressed records). Hashing on a pool of 4 threads while the records decode (`V4_WEX_SHA_ASYNC=1`) was measured neutral on 2026-11-13 (start to first token -0.2%, decode +0.5%, one pair each) and is not used. On Qwen3.8 the cost of `COLI_WEX_VERIFY=1` is not measured.
5. **Chunk size.** A rebuild at 4096:1024 saves 0.4 GB and halves the stream count; the engine's batches are 1.6-1.8 records, where larger chunks were slower on V4. Needs one V4.1 GPU measurement.
6. **The simpler, faster weight model** of section 12.4: needs a GPU measurement; would be a new codec.
7. **GPU kernel for WBP1:** only if non-expert tensors are to be decoded on the GPU.
8. **Other engines** (qwen38 FP8, qwen36, glm53, olmoe): the container and WBP1 are model-independent (F8_E4M3 tensors measured at 0.82 of raw here; Qwen3.8's FP8 experts were estimated at 0.817 earlier, from entropy, with no coder run), but each engine needs its own store wiring, and for per-token expert reads a GPU WBP1 kernel.
9. **Thread policy of the fast CPU decoder** inside the engine, and a second timing on the laptop.
10. **Limits accepted in this version:** one file per container (no shard sets); one expert grid per file with at most 2^24 slots, owner packed as `layer << 16 | expert`; index up to 2^28 bytes (about 1.5 million tensors); 2^30 decoded bytes per record; expert matrix dimensions up to 65,535; rank up to 8; names up to 65,535 bytes without NUL (safetensors accepts NUL in names and higher ranks: such files are refused by the converter); little-endian hosts only.
11. **The Python writer** holds one record in memory at a time and is single-threaded; a production build of a whole model would use the existing threaded builder pattern (`c/tools/build_nr3.py`) with the WEX4 index written at the end.

---

## 17. Conformance

A **conforming reader**:

- refuses a file whose magic is not `0x57455834`, whose `version_major` is not 4, or which sets a bit of `required_features` it does not know (section 10, rules 1, 2 and 4);
- refuses a file with a record flag bit it does not know (section 10, rule 6);
- validates the header page and the whole index by every rule of section 6 before it uses any byte of them, and never reads outside the extent a field allows;
- checks the root against a pinned value when one is configured, before it opens anything else;
- checks the SHA-256 of a record or blob before it returns its bytes as verified (section 8), and returns no decoded byte of a record whose decoder failed.

A **conforming writer** produces files that a conforming reader accepts: they pass every rule of section 6, every extent is 4096-aligned and the file is tiled exactly, every hash is correct, and the index is in the canonical encoding of section 5.

The reference outcomes are the files of `wex4/vectors/` and their `MANIFEST.tsv` ([wex4/vectors/README.md](../wex4/vectors/README.md)): 7 valid files with their roots, 46 invalid ones, a WEX3 file whose in-place upgrade must give one of the valid files byte for byte, and the trigger files of the 2026-11-11 security review. `c/wex4.h` and `wex4/wex4.py`, written independently of each other, give the verdicts of the manifest; `bash wex4/build_and_test.sh` checks this on every run.

---

## Document history

| Date | Change |
|---|---|
| 2026-11-11 | Format version 4.0 frozen after the adversarial review; in production for DeepSeek V4.1 and Qwen3.8 |
| 2026-11-12 | The document moved from `docs/NEXT-FORMAT-SPEC.md` (now a stub) to this file; no format change |
| 2026-11-13 | Revision: version header and conventions; informative sections marked; sections 8.2, 13.2, 13.3 and 16 brought to KAIRA 24f7e72 (qwen38 first-read SHA-256, refusal of a legacy file under a pinned root, `wex4_verify`, hash pool measured); section 17, conformance, added with the vectors. No format change |
