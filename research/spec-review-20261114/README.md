# Spec review of WEX4-SPEC and WNR3-SPEC (2026-11-14)

The review that [docs/HELP-WANTED.md](../../docs/HELP-WANTED.md) section 2 asks for: both specifications read as by someone who writes a reader from them, every candidate checked against the code or the vectors before it was kept.

## Question

Which sentences of [WEX4-SPEC](../../docs/WEX4-SPEC.md) and [WNR3-SPEC](../../docs/WNR3-SPEC.md) can be read two ways, which fields have no stated range, and where do the text and the code (`c/wex4.h`, `wex4/wex4.py`, `c/wex_neural_rans.h`, `c/tools/wex4_verify.c`, `wex4/vectors/`) disagree?

## Setup

- Tree: this repository at the commit that adds this folder. Line numbers below are those of that commit.
- Machine: Windows 10, MSYS2 UCRT64, gcc 16.2, Python 3.14; no GPU used.
- Command: `bash wex4/build_and_test.sh`, then `python -I research/spec-review-20261114/spec_check.py` from the repository root. One run; its output is `spec_check.log`.
- `spec_check.py` holds a WNR3 decoder and a MurmurHash3 written from WNR3-SPEC sections 1 to 4 (with errata 9 to 12, which this review added), not from the C code. It compares that decoder with `wex_nr2_decode` (through `wex4/out/wex4_tools_ref.dll`), and gives hand-made WEX4 files and WBP1 records to both WEX4 readers.

## Result

16 findings were clarifications that change no rule: both readers already agreed, and the text now says what they do. They are applied on the branch of this run, 12 in WEX4-SPEC (one document-history row) and 4 as errata 9 to 12 of WNR3-SPEC. 4 findings are left open: they need a decision, a new vector or a new section. No place was found where the C reader and the Python reader give different verdicts. No format byte and no verdict of a vector changed.

| Check (line of `spec_check.log`) | Result |
|---|---|
| R0: the five hash values of WNR3-SPEC 4.5 | all reproduce |
| R1: the decoder written from the spec against the C reference | 9 of 9 WNR3 records byte-identical (2 of the valid vector `tiny2-cz-upgraded.wex4`, 4 of `wex4/out/synth.wex4`, 3 of `wex4/out/tiny_nr3_cz.wex4`); with the `mx` sentence of 4.1 read literally, 9 of 9 fail |
| R2: ceil-form (4.2) on one synthetic expert, encoded by `c/tools/wex_nr3_tools.c` | record flags `0x03` and `0x0B`, 1,220 blocks inverted, output equal to the source and to Z(source) |
| W1, W2: WBP1 records | a mask bit on an empty plane and a trailing zero page are refused by both decoders |
| W5: content digest from the stated field widths | equals the C and the Python digest of `generic.wex4` |
| W6, W7, W9: hand-made containers | both readers give the same verdict in each case (details in the findings) |

### Findings applied to the text

Each row: where, the sentence, the problem, the evidence, and the wording now in the specification.

| # | Where | Sentence | Problem | Evidence | Wording applied |
|---|---|---|---|---|---|
| 1 | WEX4 Conventions; 7.2 | "All integers are little-endian" | the 4-byte state that opens a rANS stream is big-endian; 7.2 does not say so | `c/wex_neural_rans.h:377`, `wex4/wex4.py:139` | Conventions name the exception; 7.2: "A rANS stream starts with its 4-byte state, big-endian, in `[2^23, 2^31)`" |
| 2 | WEX4 7.2, record layout table | rows "per plane: stream table", "per plane: payload" | can be read as table and payload interleaved plane by plane | `c/wex4.h:497-521`, `wex4/wex4.py:209-210` | "The three parts after the header are grouped, not interleaved: first the CDFs of all rANS planes, then the stream tables of all planes, then the payloads of all planes" |
| 3 | WEX4 7.2 | "`n_p` = ... `ceil((decoded_bytes - p) / P)`"; "bit p = 1: plane p is rANS coded" | `n_p` is not defined for `decoded_bytes <= p`; both decoders refuse a mask bit on an empty plane, the text does not | `c/wex4.h:465-467` and `:502`, `wex4/wex4.py:159` and `:228`; log W1 | "and 0 when `decoded_bytes <= p`"; "A plane whose mask bit is set must not be empty" |
| 4 | WEX4 7.2 | "`stored_bytes` in total, zero padding at the end" | the length of the padding is not bounded; both decoders refuse a whole extra page of zeros | `c/wex4.h:522`, `wex4/wex4.py:250`; log W2 | "`stored_bytes` must equal `align(end of the last payload)`" |
| 5 | WEX4 5 | "SHA-256 over, for each tensor in table order, its name length, name, dtype, rank, shape, size and bytes" | no field widths, so the digest cannot be recomputed from the text | `wex4/wex4_tools.c:63-66`, `wex4/wex4.py:561-563`; log W5 | u16, name bytes, u8, u8, 8 x u64, u64, then the bytes; split and zero-size tensors stated |
| 6 | WEX4 5.3 | "Zero-size tensors (a dimension is 0) have `record_count = 0` ..." | for an opaque dtype both readers go by `nbytes`, not by the shape: shape (3,) with 0 bytes is accepted in the tail, shape (0, 5) with 5 bytes inside a record | `c/wex4.h:389` and `:419-422`, `wex4/wex4.py:437` and `:460-461`; log W6a to W6c | "(`nbytes = 0`; for a known dtype that is exactly when a dimension is 0)"; "A tensor inside a record has `nbytes > 0`" |
| 7 | WEX4 5.1 | "id: kind 1: the model's FNV-1a id" | value of `id` for the other kinds not stated | `c/wex4.h:350-354`, `wex4/wex4.py:407-409`, `wex4/wex4_convert.py:348` and `:358`; log W7 | "Other kinds: any value, not read by readers (the upgrade writes 0)" |
| 8 | WEX4 5.2 | "blob: 0 = none, else 1 + index"; "streams counted at the longest chunk, as WEX3 does" | range of `blob` not stated; the stream count needs the WEX3 code to be computed | `c/wex4.h:363` and `:407-413`, `wex4/wex4.py:415` and `:453-455`; vector `record-blob-index-past-the-blob-table.wex4` | "`<= blob_count`"; the sum over the three matrices of `ceil(scales / 16384) + ceil(32 * scales / 16384)` |
| 9 | WEX4 4 and 6, step 1 | step 1 lists "`max_decoded <= 2^30`" only | both readers refuse, at the header step, a file with no record and a non-zero `max_decoded` or `total_decoded` | `c/wex4.h:206`, `wex4/wex4.py:355`; log W9 (C status -6, `WEX4_E_HEADER`) | "`max_decoded` and `total_decoded` are 0 when `record_count` is 0", in section 4 and in step 1 |
| 10 | WEX4 5.3 | "name_offset: ... must equal the sum of the previous names' lengths" | that the names use the whole pool is checked and not stated | `c/wex4.h:334`, `wex4/wex4.py:401`; vector `tensor-names-shorter-than-the-name-pool.wex4` | "(and the names of all tensors fill the pool exactly)" |
| 11 | WEX4 15.1 | "every blob and record against its SHA-256, zero padding, in one read-only unbuffered pass" | the checker tests the padding of the index and of blobs, not the padding inside a record; its reads are unbuffered on Windows only | `c/tools/wex4_verify.c:26`, `:51`, `:63-79`, `:119`; `wex4/vectors/README.md` (the seven files it passes) | both stated in 15.1 |
| 12 | WEX4 header | "Document revision: 2026-11-13" | the history already had a row of 2026-11-14 | the document history | "2026-11-14" |
| 13 | WNR3 4.1 | "`pm = mx = 8`, after symbol x `pm = x & 7`, `mx = max(mx, x & 7)`" | read as written, `mx` stays 8 for the whole block; in the code 8 means "none" and the first symbol replaces it | `c/wex_neural_rans.h:512`, `:592`, `:684`; log R1 (9 of 9 records fail when read literally) | erratum 9; the same words added to WNR2-REFERENCE section 7 |
| 14 | WNR3 1 | "rANS exactly as WNR2: 32-bit state, `L = 2^23`, 14-bit CDFs, byte renormalisation ..." | the decode step itself is in no page, only in the code | `c/wex_neural_rans.h:356-380`; log R1, R2 | erratum 10 |
| 15 | WNR3 3 | "`[32, 32 + 2(S+W))` u16 stream lengths"; "anchor[m]: srel anchor of matrix m" | `S`, `W`, `scount_m`, `count_m` are used and not defined; range of the anchor byte and whether a decoder checks it not stated | `c/wex_neural_rans.h:308-316` (the parser copies the byte and requires 32 nibbles per scale), `:399-405` (rule g gives 0..254) | erratum 11 |
| 16 | WNR3 2.1 and 5 | "no new model format"; "Format unchanged (WNM2 ...)" | a reader with only this file cannot parse the model: the layout is in WNR2-REFERENCE, the order of the tables and the 0 -> 1 rule of the id only in the code | `c/wex_neural_rans.h:235-274`, `wex4/wex4.py:277-288`; log R1 | erratum 12; WEX4-SPEC section 6, step 3 points to it |

### Findings left open

| # | Where | Sentence | Problem | Evidence | Proposed |
|---|---|---|---|---|---|
| 17 | WEX4 17, `wex4/vectors/` | "7 valid files with their roots" | the only valid vector with WNR3 records has canonical zero and no CEIL bit: the ceil inverse of WNR3-SPEC 4.2 and a record without canonical zero are tested by no vector | log R1 (record flags `[8]` for `tiny2-cz-upgraded.wex4`); `wex4/vectors/MANIFEST.tsv` | one more valid vector with a CEIL matrix and one WNR3 record without canonical zero, for example from the expert that `spec_check.py` builds; it needs `wex4/test_wex4.py` to write it and new manifest rows |
| 18 | WNR3 (no section) | none | the codec has no conformance section: what a conforming decoder must accept and refuse is spread over sections 3, 4.4, 4.5, 4.7 and the errata, and the only record-level reference outcomes are the hash values of 4.5 | WNR3-SPEC as a whole; WEX4-SPEC 17 for the contrast | a short section "Conformance" in WNR3-SPEC that lists the checks of a decoder in order and names the vectors of finding 17 |
| 19 | WEX4 6; `wex4/vectors/README.md` | "Refuse unless, in this order: ..."; "`class`: ... the status the C reader returns" | the error classes (`WEX4_E_*`) are defined in `c/wex4.h` only; the manifest lists one per rejected file, and the specification gives no rule-to-class table, so a second reader can match the verdict and not the class | `c/wex4.h:45-47`; `wex4/vectors/README.md` ("Another reader only has to refuse the file") | decide: either the classes stay informative (then say so in section 17), or section 6 gets the table |
| 20 | WNR3 3, 4.3 | "decoders read it, never re-derive it" | a WNR3 record is not a canonical coding of a slab: the parser accepts any anchor byte and any legal chunk pair, so different records decode to the same slab. The specification does not say whether that is intended | `c/wex_neural_rans.h:306-309`; WEX4-SPEC 8.2 already says the chunk fields are outside the stream hashes | one sentence in 4.3 or in a conformance section: "A decoder accepts any anchor byte and any legal chunk pair; only an encoder is bound by rule g" (or the opposite, which would be a rule change with both decoders and a vector) |

## Limits

- One reviewer pass, one run of the script, on one machine. No GPU: `c/wex_cuda.cuh` was not compared with the text.
- Read against the code: WEX4-SPEC sections 2 to 10, 15.1 and 17; WNR3-SPEC sections 1 to 4 and the errata. Not reviewed: the numbers of the informative sections of WEX4-SPEC (11, 12, 13.5, 15.2, 15.3, 16), sections 13.1 and 14 against `wex4/wex4_convert.py`, sections 5 to 11 of WNR3-SPEC (model training, gates and the historical plan), and WNR2-REFERENCE apart from the one sentence of finding 13.
- The decoder written from the spec was run on 9 records of test containers and one synthetic expert, all small (the synthetic record has 54 streams and 53,248 raw bytes). No real production record was decoded with it. The error paths of a decoder (a max-5 block at coded scale 0, a nibble 8 in a canonical-zero record, a wrong owner) were not exercised by it; `c/tests/test_wex_format.c` covers them for the C decoder.
- Finding 1 rests on the code lines, not on a test: in the one record looked at, the first stream state is in `[2^23, 2^31)` in both byte orders (log W17), so the range check alone would not have shown the difference.
- Finding 11 rests on reading `wex4_verify.c` and on the note in `wex4/vectors/README.md`; the checker was not run again for this review.
- The earlier review ([wex4/review/REVIEW.md](../../wex4/review/REVIEW.md), 2026-11-11) had already compared the WEX4 tables with both readers field by field; that work was not repeated. Three of its spec-only proposals for section 7.2 had not reached the text; they are in findings 1, 3 and 4 here.

## Raw files

- `spec_check.py`: the script.
- `spec_check.log`: its output on 2026-11-14.
