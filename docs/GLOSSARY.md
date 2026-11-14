# Glossary

The words a first-time reader meets in this repository, grouped by subject. Every definition names the file it was checked against; paths are relative to the repository root. Numbers are quoted with the conditions their source gives.

Back to the [documentation index](README.md).

## The data

**Mixture-of-Experts (MoE), router.** A model whose layers each hold many feed-forward experts and a router that picks, for every token, the experts to compute. When the experts do not all fit in memory the engine keeps some resident and reads the others from NVMe when the router picks them; these formats are laid out for that read. *Source: `README.md` ("What WEX is"); `docs/ARCHITECTURE.md` ("Read path").*

**Routed expert.** One feed-forward expert of a Mixture-of-Experts layer, chosen per token by the layer's router. It is three matrices: w1 (gate), w2 (down) and w3 (up). DeepSeek V4.1 REAP-192 has 40 layers x 192 = 7,680 of them, Qwen3.8-Flash-Next 48 x 512 = 24,576. Only routed experts are in the containers: shared experts, attention and the 384 MTP experts of V4.1 stay in the safetensors. *Source: `docs/WNR3-SPEC.md` sections 1 and 8; `docs/reports/wnr2-history-20261110.md` section 2 (geometry); `docs/reports/QWEN38-MXFP4-20261111.md` ("Containers"); `docs/WEX4-SPEC.md` section 12.3.*

**MXFP4.** The weight format of these experts: weights are 4-bit FP4 codes in blocks of 32 along a row, and each block has one 8-bit E8M0 scale. *Source: `docs/WNR3-SPEC.md` section 1.*

**FP4 (E2M1).** One 4-bit weight code, a nibble: bit 3 is the sign and bits 0 to 2 pick the magnitude from 0, 0.5, 1, 1.5, 2, 3, 4, 6. Two nibbles share a byte, the even one in the low half. Code 8 is -0 and code 0 is +0. *Source: `docs/WNR3-SPEC.md` section 1.*

**Block, nibble.** A nibble is one 4-bit FP4 code. A block is 32 consecutive nibbles of a row, 16 bytes, with the one E8M0 scale they share. A matrix with `rows x cols` scales has `rows x 16 * cols` weight bytes, and a WNR3 weight stream always holds whole blocks. *Source: `docs/WNR3-SPEC.md` sections 1 and 4.1 and erratum 11; `docs/WEX4-SPEC.md` section 5.5.*

**E8M0 (also written UE8M0).** The scale of a block of 32 weights: one byte `s`, and a weight's value is its FP4 value times 2^(s - 127). The C comments say UE8M0 and the safetensors dtype name is `F8_E8M0`; it is the same byte. *Source: `c/tools/repack_q38_mxfp4_wex.py` (function that rebuilds the values); `c/wex_neural_rans.h`; `docs/WEX4-SPEC.md` section 5.4.*

**Slab.** The decoded form of one expert in memory, and the content of a raw record: the scales of w1, w2, w3, then the packed FP4 weights of w1, w2, w3. A V4.1 slab is 18,800,640 B (17,694,720 B of weights and 1,105,920 B of scales), a Qwen3.8 slab 2,611,200 B (stored as a 2,613,248 B record, padded to 4096). *Source: `c/wex_format.h` (`WexEntry`, `aligned_expert_bytes`); `docs/WNR2-REFERENCE.md` section 5 and `docs/reports/wnr2-history-20261110.md` section 2; `docs/reports/QWEN38-MXFP4-20261111.md`.*

**Record.** What the container stores for one expert and what the engine fetches with one unbuffered read: a multiple of 4096 bytes at a 4096-aligned offset. A record is either the slab itself (raw, codec 0) or a coded form that decodes to it. *Source: `docs/WEX4-SPEC.md` section 2; `docs/reports/colibri-integration-20261109.md` section 1.1.*

**REAP-192.** The DeepSeek V4.1 checkpoint this work uses (DeepSeek V4.1 Flash 1017 REAP-192 MXFP4): 192 routed experts in each of 40 layers, where DeepSeek V4 Flash 0731 had 256 in each of 43. One planning note describes REAP as having pruned the cold experts; no file in this repository expands the acronym. *Source: `docs/reports/wnr2-history-20261110.md` section 2 (geometry); `research/v41-gpu-20261109/PLAN.md` (expected impact of G2).*

**safetensors, shard.** safetensors is the checkpoint file format the experts come from: an 8-byte little-endian header length, a JSON header (name, dtype, shape and byte range of every tensor), then one byte buffer. It has no checksum, no version field and no compression. A shard is one of the safetensors files a checkpoint is split into. The WEX4 converter (`pack`) keeps each source file's header verbatim, so `unpack` rebuilds the files byte for byte (not from a container with canonical zero). *Source: `docs/WEX4-SPEC.md` sections 1.2 and 14.*

## Containers

**Container (WEX1 to WEX4).** The file layout around the records. WEX1, WEX2 and WEX3 share one layout: a 4096-byte header page, a table with one 72-byte entry per (layer, expert), then the records. WEX4 replaces the header and the table with a hashed header page and a hashed index and leaves the records as they are. The container version and the record codec are numbered separately; the full table of ids is in the root [README](../README.md) ("Format family"). The name is expanded in one place only, the October architecture page: "Windows Expert eXtension". *Source: `c/wex_format.h`; `c/wex4.h`; `docs/WEX4-SPEC.md` sections 3 to 5; `docs/reports/wex-architecture-20261018.it.md` (title).*

**Record codec (also "entry codec").** How one record is coded, a number in the record's table entry: 0 raw, 1 scale tiles (WEX2), 2 palette3 (lossy, never used in production), 3 WNR1, 4 WNR2, 5 WNR3, 6 WBP1 (WEX4 only). *Source: `c/wex_format.h` (`WEX_ENTRY_*`); `c/wex4.h` (`WEX4_CODEC_*`).*

**Entry table, `WexEntry` ("legacy entry").** The table of a WEX1, WEX2 or WEX3 file: one 72-byte `WexEntry` per (layer, expert) with the record's file offset and stored size, the shapes and slab offsets of its three scale and three weight matrices, and its codec. An all-zero entry means the expert is absent and is served from the safetensors. WEX4 has no such table on disk; `wex4_expert_entry()` rebuilds an entry from the index, because the engine store and the CUDA helper still consume it. In the WEX4 pages "legacy" means these WEX1 to WEX3 forms. *Source: `c/wex_format.h` (`WexEntry`, `wex_validate_entry`); `docs/WEX4-SPEC.md` sections 5.5 and 13.2.*

**Header page.** The first 4096 bytes of a container. In WEX4: magic, version, the two feature words, the exact file size, where the index is and how long, the counts, the expert grid, the SHA-256 of the index, a reserved area that is written as zero and ignored by readers, and at byte 4064 the SHA-256 of bytes 0 to 4063, the root. *Source: `docs/WEX4-SPEC.md` section 4.*

**Extent, tiling rule.** An extent is a 4096-aligned byte range of a WEX4 file: the header page, the index, a blob or a record. The tiling rule says that the header page, the blobs in table order, the records in table order and the index cover the file exactly once, with no gap, no overlap and nothing after the last extent; padding inside an extent is zero. *Source: `docs/WEX4-SPEC.md` sections 2 and 3.*

**Index.** The part of a WEX4 file that describes everything else: the blob table (64 bytes per blob), the record table (72 per record), the tensor table (96 per tensor), the name order (4 per tensor), the name pool and the metadata, in that order with no padding between them. It is at most 2^28 bytes, its SHA-256 is in the header page, and a reader validates it in place before it uses any of it. *Source: `docs/WEX4-SPEC.md` sections 4, 5 and 6.*

**Blob.** Side data of a codec or of a tool inside a WEX4 file, with its own extent and its SHA-256 in the index. Kind 1 is the WNM2 model of the WNR3 records; kinds 16, 17 and 18 are what an in-place upgrade keeps of the old file (header page, entry table, other bytes). A reader ignores a kind it does not know, but the blob is still hashed and inside the tiling. *Source: `docs/WEX4-SPEC.md` sections 2, 5.1 and 10.*

**Tensor, part, continuation part.** In WEX4 a tensor is a name, a dtype, a shape of at most 8 dimensions and a byte range of decoded data. Several tensors can share one record, and one large tensor can be split over consecutive records, its parts: the first holds only that tensor, the others are continuation parts with no tensor of their own. A tensor with `nbytes = 0` belongs to no record and comes last in the table. *Source: `docs/WEX4-SPEC.md` sections 2 and 5.3.*

**Opaque tensor.** A tensor whose dtype code is not one of the 22 known ones. Its bytes and its place in the record are validated, its shape is not checked against its size, and a consumer that needs typed data must refuse it. *Source: `docs/WEX4-SPEC.md` sections 5.3 and 5.4.*

**Expert record, expert grid.** An expert record is a WEX4 record with the EXPERT flag: exactly six tensors, the scales of w1, w2, w3 and then their weights, which is the slab of one routed expert; its codec is 0 or 5 and its owner is `layer << 16 | expert`. The expert grid is `num_layers x num_experts` in the header page (each at most 65,536, the product at most 2^24), non-zero only when the file has an expert record. A slot of the grid without a record is an expert the engine takes from the safetensors. *Source: `docs/WEX4-SPEC.md` sections 2, 4, 5.2, 5.5 and 13.2.*

**RAW record, raw fallback.** Codec 0: the stored bytes are the decoded bytes, zero-padded to a multiple of 4096. A writer stores a record raw whenever the coded form is not strictly smaller (the raw fallback); in a canonical-zero build such a record keeps the source bytes and carries no VALUE_EXACT flag. A RAW record has no check inside it and is covered by its SHA-256 only. *Source: `docs/WEX4-SPEC.md` sections 5.2, 7, 7.1 and 8.2.*

**Feature bits of WEX4 (required, optional).** Two 64-bit words in the header page. A reader refuses a file that sets a bit of `required_features` it does not know (today only bit 0, VALUE_EXACT: the file may hold value-exact records) and ignores unknown bits of `optional_features` (bit 0, ST_SOURCES: the metadata holds safetensors sources). They are not the feature bits of the helper DLL. *Source: `docs/WEX4-SPEC.md` sections 4 and 10.*

**Canonical and upgraded layout; in-place upgrade, downgrade.** A file written from scratch is in the canonical layout: header page, index, blobs, records. `wex4_convert.py upgrade` turns a WEX1 or WEX3/WNR3 file into WEX4 where it lies: it hashes every record, appends a copy of the old header page and the index, and overwrites the header page; no record moves. That is the upgraded layout, with the index last. `downgrade` restores the old page and truncates the file, which is then byte-identical to the file before the upgrade. On the V4.1 production file the upgrade took 93.6 s and added 6,717,440 B. *Source: `docs/WEX4-SPEC.md` sections 3 and 13.1; `research/wex4-m3-20261111/RESULT.md`.*

**Canonical encoding, content digest.** The index of a WEX4 file has exactly one byte representation once the writer has fixed the layout, the grouping of tensors into records, the codecs with their parameters and the metadata, so two writers that make the same choices produce the same file and the same root. The root identifies a file, not a set of tensors; the content digest is the identity that does not depend on the layout: a SHA-256 over every tensor's name, dtype, shape, size and bytes in table order. The tests use it to compare readers and layouts. *Source: `docs/WEX4-SPEC.md` section 5.*

**Integrity levels 1, 2, 3.** The three layers of hashes of a WEX4 file. Level 1 is the hash inside a coded record (23 bits per rANS stream, 32 bits per raw chunk of a WBP1 record), checked by the decoder every time. Level 2 is the SHA-256 of each record's stored bytes and of each blob, held in the index; tools always check it, the engine when asked. Level 3 is the SHA-256 of the index and of the header page, checked at every open. Together they are a tree: root, index, records and blobs. *Source: `docs/WEX4-SPEC.md` section 8.1.*

**Fail closed.** The rule that a failure never yields data: a decoder that finds an error returns it and its output must not be used (it decodes into scratch memory), and a reader refuses a file with a required feature bit or a record flag it does not know. *Source: `docs/WEX4-SPEC.md` sections 6 (step 3) and 10.*

**Conforming reader and writer; normative, informative.** A conforming reader refuses what section 17 of the WEX4 specification lists and validates the header page and the whole index before it uses them; a conforming writer produces files such a reader accepts. In that specification "must" marks a requirement on them (normative text), and sections marked *Informative* describe measurements, the engine and history and are not needed to implement the format. *Source: `docs/WEX4-SPEC.md` ("Conventions") and section 17.*

**Error class (`WEX4_E_*`).** The status with which the C reader refuses a file, for example `WEX4_E_HEADER`, `WEX4_E_HASH` or `WEX4_E_DATA`. `MANIFEST.tsv` lists one for every rejected vector; it says which rule the file breaks first in the C reader's order, and another reader only has to refuse the file. *Source: `c/wex4.h` (the `WEX4_E_*` enum); `wex4/vectors/README.md`.*

**Hostile file, decompression bomb, polyglot.** A hostile file is one built to break a reader: in the vectors, a valid file with one thing changed and the hashes recomputed so that the change reaches the validators. A decompression bomb is a small record that claims a huge decoded size (the vector is a 4096-byte WBP1 record claiming 2^30 bytes; the minimum-size rule of the index refuses it). A polyglot is a file that is also valid in another format: WEX4 refuses trailing bytes, but metadata, unknown blobs, the reserved header area and tensor data are free-form, so one is still possible. *Source: `docs/WEX4-SPEC.md` sections 1.2 and 9; `wex4/vectors/README.md`.*

**Root.** The SHA-256 of the first 4064 bytes of a WEX4 header page, stored at byte 4064. The header page holds the hash of the index and the index holds a SHA-256 of every record and blob, so these 32 bytes identify the whole file. Two copies are the same container when their roots are equal, and an engine can be told the expected root with `COLI_WEX_ROOT` ("pinning"). *Source: `docs/WEX4-SPEC.md` sections 4, 8 and 13.3.*

**Root pin (`COLI_WEX_ROOT`).** The expected root of a WEX4 container, 64 hex digits, given to the engine in its environment. A container with another root does not open, and under a pin a container that is not WEX4 is refused too (it has no root). The pin is what turns the hashes into protection against substitution: without it they detect damage only. *Source: `docs/WEX4-SPEC.md` sections 8 and 13.3.*

**`wex4_verify`.** The checker `c/tools/wex4_verify.c`: one read-only pass over a WEX4 file on several threads that checks the header page, the index and every blob and record against its SHA-256, then prints the line to pin, `COLI_WEX_ROOT=<root>`. It hashes and does not decode; `wex4/wex4.py verify` also decodes. *Source: `docs/WEX4-SPEC.md` section 15.1.*

**Conformance vectors.** The files of `wex4/vectors/`, each with the verdict, error class and root a reader must give (`MANIFEST.tsv`), for testing a reader written from the specification. *Source: `docs/WEX4-SPEC.md` section 17; `wex4/vectors/README.md`.*

**WBP1.** Record codec 6 of WEX4: order-0 rANS over byte planes, for tensors that are not MXFP4 experts (FP8, BF16, F32, integers). It is in `c/wex4.h` and tested; no production file holds such a record and no engine calls it. *Source: `docs/WEX4-SPEC.md` section 7.2; `STATUS.md` ("Not measured, not run or not verified").*

## Record codecs for MXFP4 experts

**rANS.** Range asymmetric numeral systems, the entropy coder of WNR1, WNR2, WNR3 and WBP1. Here: a 32-bit state, lower bound L = 2^23, 14-bit cumulative frequency tables (12-bit in the retired WNR1), byte-wise renormalisation. *Source: `docs/WNR3-SPEC.md` section 1; `docs/WNR2-REFERENCE.md` section 6.*

**CDF.** A cumulative frequency table, the form in which a probability model reaches the rANS coder: one u16 more than there are symbols, the first 0, the last 16384 (2^14), strictly increasing, so every symbol has a frequency of at least 1. A weight table has 17 values (16 nibble codes), a scale table or a WBP1 plane table 257. A table that breaks the rule is refused before any symbol is decoded. *Source: `docs/WNR3-SPEC.md` errata 10 and 12; `docs/WEX4-SPEC.md` sections 7.2 and 9; `c/wex_neural_rans.h` (`wex_nr2_cdf_ok`).*

**Context (weight context), `srel`, `pm`, `mx`.** Which of the 567 weight tables of a matrix codes the next nibble: `(srel * 9 + pm) * 9 + mx`. `srel` is the block's coded scale minus the matrix's anchor, clamped to -3..3, plus 3; `pm` is the previous nibble's magnitude and `mx` the largest magnitude of the block so far, both 8 ("none") at the start of every block. Scales have no context: one table per matrix. *Source: `docs/WNR3-SPEC.md` section 4.1 and erratum 9.*

**Stream, chunk (Cw, Cs).** A record's symbols are cut into independent rANS streams so that a GPU decodes them in parallel: at most Cw weight symbols or Cs scale symbols per stream, never across a matrix. The two chunk sizes are stored in every record. V4.1 uses 2048:512, which gives 19,440 streams per record. *Source: `docs/WNR2-REFERENCE.md` section 6 and `docs/reports/wnr2-history-20261110.md` section 2.*

**WNM2 (and WNMD).** The probability model of a WNR2 or WNR3 container: a 59,408-byte table of cumulative frequencies, one 16-symbol table for each of 567 contexts of each of the 3 matrices, plus 3 scale tables. It is built from symbol counts over the experts; no neural network is involved, although code names say "neural". Every record names its model by a 32-bit FNV-1a id. WNMD is the 24-byte descriptor in a WEX3 header page that says where the model is; in WEX4 the model is a blob. *Source: `docs/WNR2-REFERENCE.md` sections 4, 6 and 7; `c/wex_neural_rans.h`.*

**WNR1, WNR2, WNR3.** Three generations of lossless rANS record codecs (codecs 3, 4, 5). WNR1 carried a model in every record and is retired. WNR2 uses the container's WNM2 model, with 567 weight contexts and u16 stream lengths; it has no checksum, and 1.8-1.9% of random single-bit flips in a whole real record were accepted on the CPU. WNR3 keeps WNR2's record skeleton and model format and adds ceil-form, rule-g anchors, optional canonical zero, a hash in every stream and the record's owner. The repository name spells the second one "WRN2"; every constant and file says WNR2. *Source: `docs/WNR2-REFERENCE.md` sections 4 and 7; `docs/WNR3-SPEC.md` sections 0 and 3; `docs/reports/WNR3-VERIFY.md` (flip audit).*

**Ceil-form.** A reversible rewrite WNR3 applies per matrix before coding. A block whose largest magnitude code is 7 and that uses neither code 1 nor code 3 is stored with its magnitudes halved and its scale raised by one (unless the scale is already 255); the decoder undoes this for every block whose coded maximum is 5. The encoder uses it for a matrix only when no block of that matrix already has maximum 5, which is what makes it exactly reversible. *Source: `docs/WNR3-SPEC.md` section 4.2.*

**Rule g (anchor).** How WNR3 picks the anchor of a matrix, the scale value that the weight context measures a block's scale against: the value `v` with the most scales equal to `v` or `v + 1`. It replaces WNR2's median, is stored in record bytes 24 to 26, and decoders read it and never recompute it. Ceil-form and rule g together measured 0.88179 of raw against 0.89943 for WNR2 on 40 held-out V4.1 experts. *Source: `docs/WNR3-SPEC.md` sections 0 and 4.3.*

**Canonical zero (CZ, "WNR3-CZ"), value-exact.** An option of a WNR3 build: every FP4 -0 is stored as +0. The record is then **value-exact**: it decodes to the source values but not to the source bytes (the opposite is byte-exact, and canonical zero is the only value-exact case). The container carries header flag 64 (in WEX4 the VALUE_EXACT bits), and the original bytes cannot be rebuilt from it. It is worth 2.74% of raw on V4.1. The engine refuses such a container on the masked DeepGEMM path (`DSV4_CUDA_MOE_MASKED`), which is not proven value-exact. The Quark export of Qwen3.8 has no -0, so the option gives nothing there. *Source: `docs/WNR3-SPEC.md` section 4.4; `docs/WEX4-SPEC.md` sections 2 and 12.3; `docs/reports/QWEN38-MXFP4-20261111.md`.*

**Stream hash (h23).** WNR3's integrity check: 23 bits of a MurmurHash3 over a stream's decoded symbols and its identity (model id, owner, stream index, anchors and flags) are folded into the stream's start state, and the decoder requires that exact final state. A damaged or misplaced stream passes with probability about 2^-23. Measured: 0 of 2,231,336 single-bit flips accepted on the GPU, 0 of 3,000 random ones on the CPU. *Source: `docs/WNR3-SPEC.md` section 4.5; `docs/reports/WNR3-VERIFY.md` section 2.*

**Owner.** The slot a record belongs to, `layer << 16 | expert`, written in WNR3 record bytes 28 to 31 and repeated in a WEX4 index entry. The engine compares it before decoding, so a record served for the wrong expert is refused. *Source: `docs/WNR3-SPEC.md` sections 3 and 4.7; `docs/WEX4-SPEC.md` sections 5.2 and 7.1.*

**Coded domain, output domain, Z(x).** The coded domain is what the streams of a WNR3 record carry: the scales and nibbles after canonical zero and ceil-form. The output domain is the decoded slab. Contexts, stream hashes and the canonical-zero check all use the coded domain; the ceil inverse is applied last and feeds back into none of them. Z(x) is x with every weight nibble 0x8 replaced by 0x0: a canonical-zero record decodes to Z(source). *Source: `docs/WNR3-SPEC.md` sections 1 and 4.7.*

**Byte plane, transform 1.** How WBP1 cuts a tensor: the decoded bytes are seen as elements of P bytes (1, 2, 4 or 8), and plane p is byte p of every element. Each plane is coded alone, with order-0 rANS or raw, whichever is smaller. Transform 1 rotates every little-endian element left by one bit before the cut; for BF16 and F32 that puts the whole exponent into the top plane. *Source: `docs/WEX4-SPEC.md` section 7.2.*

**Reference decoder, fast decoder, scalar kernel.** The reference decoder of WNR3 records is `wex_nr2_decode` in `c/wex_neural_rans.h`: scalar, one thread, about 0.5 s per V4.1 record. The fast decoder, `wex4/research/fast-cpu-decode/wex_fast_decode.h`, decodes the same bytes to the same output with the same verdicts: threads across streams, and AVX2 kernels on x86 (72-90 ms per record on one thread, best of 5, desktop Ryzen 9 7950X3D). On any other CPU it builds its scalar kernel, the reference loop per stream. `c/wex4.h` always calls the reference; the tests build both and require the same verdicts. *Source: `docs/WEX4-SPEC.md` sections 11.3 and 15.1; `wex4/README.md`; `wex4/build_and_test.sh`.*

**Errata (of the WNR3 specification).** The numbered corrections at the top of `docs/WNR3-SPEC.md`. The sections below them are the text as frozen before the implementation; where the two differ, the errata and the code win. Errata 1 to 8 record where the implementation left the plan, 9 to 12 write down what a decoder needs and the sections leave open. *Source: `docs/WNR3-SPEC.md` ("Errata").*

## Reading and decoding

**Unbuffered read.** A read that goes to the drive without passing through the operating system's file cache (on Windows, a file opened with `FILE_FLAG_NO_BUFFERING`). It works on aligned offsets and sizes, which is why every record starts on a 4096-byte boundary and is a multiple of 4096 bytes long: one such read fetches one expert. The read paths were tuned on Windows; `O_DIRECT` against buffered reads on Linux is not measured. *Source: `README.md` ("What WEX is"); `c/tools/wex4_verify.c` (`w4v_open`, `w4v_read`); `docs/HELP-WANTED.md` section 7.*

**Helper DLL (`kaira_wex_cuda.dll`) and its feature bits.** The optional CUDA library that decodes compressed records, built from `c/tools/wex_cuda_bridge.cu` with the kernels of `c/wex_cuda.cuh`. The engine loads it the first time a record needs it and falls back to the CPU decoder, or refuses the container, without it. ABI 1 means it has `wex_cuda_decode_to_host`. The feature bits are what `wex_cuda_helper_features()` returns, one bit per optional ability; today's bridge returns 63.

| bit | ability |
|---|---|
| 1 | decode one record straight into caller-owned GPU buffers |
| 2 | page-lock the caller's host buffers |
| 4 | WNR2 records, with a registered WNM2 model |
| 8 | a batch of records decoded into GPU buffers in one call |
| 16 | WNR3 records in the single-record and batch-to-GPU entry points |
| 32 | a batch of WNR2 or WNR3 records decoded into host buffers in one call |

*Source: `c/tools/wex_cuda_bridge.cu` (header comment and `wex_cuda_helper_features`); `c/wex_decode.h`.*

**GPU-direct, mode 0, 1, 2.** `COLI_WEX_GPU_DIRECT` in the DeepSeek engine. Mode 0 reads a record into a host slab, decodes it there and uploads the slab. Mode 1 sends the compressed record to the GPU and decodes it straight into the expert's buffers in VRAM, so the decoded slab never passes through host RAM. Mode 2 also lets the next-layer prefetcher do that. Raw records never go direct. *Source: `docs/reports/colibri-integration-20261109.md` sections 1.2 and 5.5.*

**Level A, B, C.** The three steps in which the container hooks are ported into `deepseek_v4.c`: A is mode 0 and is enough to read any container; B is the store side of GPU-direct and the RAM cache of compressed records; C is the GPU side of GPU-direct. Only level A has a patch for stock colibri. *Source: `docs/reports/colibri-integration-20261109.md` section 5.*

**Mirror, replica.** Four different things carry the word "mirror". (1) A **container replica**: a byte copy of the container on another drive, in a folder named by `COLI_WEX_EXTRA_MIRRORS`, accepted only when it matches the primary (header page and entry table; for WEX4, the root). (2) A **shard mirror**: a folder with copies of the safetensors shards, named by `COLI_MODEL_MIRROR`. (3) A **VRAM mirror**: the copy of one expert's weights and scales that the DeepSeek engine keeps on the GPU. (4) The **source mirror**: the `c/` folder of this repository, which the maintainer keeps identical to `src/colibri/c` of the application tree (`sync-map.tsv`, the `mirror` line of `check_repo.py`). When a page says "U: with a full mirror on C:" it means (1). *Source: `docs/reports/colibri-integration-20261109.md` sections 1.2, 5.1 (hook A6), 5.3 and 5.5; `docs/WEX4-SPEC.md` section 8.3.*

**Read split, striped read.** With a replica, `COLI_WEX_WEIGHTS=38,62` sends 38% of the experts to the primary and 62% to the replica; a fixed hash of (layer, expert) decides, so one expert always comes from the same drive. A striped read (`V4_WEX_STRIPE`, `COLI_WEX_STRIPE`) instead cuts one record in two and reads the parts from both drives at once; it measured slower on V4.1 and is off. *Source: `docs/reports/colibri-integration-20261109.md` section 12 and the box at its top; `STATUS.md`.*

**Tier (RAM tier, VRAM tier).** Where an expert that is not read from disk at this token is kept: decoded in host RAM slots, or resident on the GPU. An expert in neither tier is a miss and costs a read. *Source: `docs/reports/colibri-integration-20261109.md` section 1.2; `docs/reports/QWEN38-MXFP4-20261111.md`.*

**Integrity lane, first-read SHA-256.** The engine's check of a record's stored bytes against the SHA-256 in the WEX4 index, the first time a process reads that record (`COLI_WEX_VERIFY`; on by default in the DeepSeek engine when the root is pinned, off by default in qwen38). A record that fails stays refused for the rest of the process. `V4_WEX_SHA_ASYNC=1` moves the hashing of the DeepSeek engine onto a pool of 4 threads (the hash pool). "W06" is the label, in the engine's code comments, of the rule that a pinned root refuses a file that is not WEX4. *Source: `docs/WEX4-SPEC.md` section 8.2; `docs/COLIBRI-INTEGRATION.md` sections 5 and 7.*

**Sub-reads (`V4_WEX_SUBREADS`).** The DeepSeek engine reads each drive's part of a record as N overlapped reads instead of one, and with a mirror cuts the record across both drives; N = 4 in the `v41gpu` profile. *Source: `docs/COLIBRI-INTEGRATION.md` section 5; `docs/BENCHMARKS.md`.*

**Tier prewarm (`COLI_V4_TIER_PREWARM_AHEAD`, `V4_PREWARM_BATCH`).** At start-up, with `COLI_V4_TIER_PREWARM=1`, the DeepSeek engine fills the VRAM tier with the experts it used most. By default it does so one record at a time; `COLI_V4_TIER_PREWARM_AHEAD=N` keeps N reads in flight, and `V4_PREWARM_BATCH=1` reads and decodes batches of 8 records through the GPU-direct batch path, two batches deep. *Source: `docs/COLIBRI-INTEGRATION.md` section 5; `docs/BENCHMARKS.md`.*

**`wexio_probe`.** `c/tools/wexio_probe.py`: a read-only drive probe (Windows) that reads a container at random 4096-aligned offsets with N reads in flight, per request size, from each drive alone and from both, and prints GB/s and latency. *Source: its header comment; `docs/BENCHMARKS.md` section "Drives".*

**Arena.** The qwen38 engine's VRAM tier with slots of each expert's exact size (`QT_ARENA=1`) instead of padded slots; by the research's count it holds 9,049 experts instead of 7,178 at the same charged budget. Engine code, not container code. *Source: `docs/reports/QWEN38-MXFP4-20261111.md` ("After the install").*

**Prefix store.** The qwen38 engine's cache of the model state after the system prompt, reused by later turns; in one control run it held up to 1089 MiB and reused 65-75% of later turns' prompts. Engine code, not container code. The "prefix-checkpoint slots" of the DeepSeek V4.1 RAM plan (210 MiB each) are a different mechanism of that engine, which this repository does not describe further. *Source: `docs/reports/QWEN38-MXFP4-20261111.md`; `STATUS.md` (Qwen3.8 and low-RAM sections).*

## Engine and host

**colibri.** The inference engine these formats plug into: C sources in one folder (`c/` upstream, `src/colibri/c` in the maintainer's tree), one executable per model family. `deepseek_v4` and `qwen38` are the two that read these containers. "Stock" or "upstream" colibri is the public engine without the maintainer's changes; "the fork" is the maintainer's tree with them. *Source: `docs/reports/colibri-integration-20261109.md` section 2.*

**U: and C:.** The two NVMe drives of the test machine, U: on PCIe 4 and C: on PCIe 5, each holding a full copy of the production containers; the engines split reads 38/62 between them. *Source: `docs/BENCHMARKS.md`.*

**Application and its profiles (`v41gpu`, `qwen38`).** The maintainer's application ships and runs colibri: a host program, the engine binaries under `app\`, and model profiles in its configuration file. Its source tree also carries everything under `c/`; in paths it is written `<TREE>`. A profile is one entry of that configuration file: a model, its engine and the environment variables passed to it, among them the container path, the replica folder and the read split. *Source: `docs/reports/colibri-integration-20261109.md` sections 5.8 and 8.2; `CONTRIBUTING.md` section 1.*

## Words in the tests

**Mutation fuzzing (modes 0, 1, 2), differential fuzzing.** The in-process fuzzer of `wex4/wex4_tools.c` changes random bytes of a valid file and counts what the reader does. Mode 0: any byte, hashes checked; every change must be rejected. Mode 1: bytes of compressed records with the record SHA-256 switched off; the codec must reject the change or decode to the identical bytes. Mode 2: header and index with the hashes recomputed, so that the change reaches the validators; the file must be rejected or still verify in full. Differential fuzzing gives the same mutated file to the C reader and to the Python reader, which must agree on the verdict and on the content digest. *Source: `docs/WEX4-SPEC.md` sections 9 (fuzzing policy) and 15.2.*

**Guard page.** An inaccessible page placed right after a buffer, so that a read or write past its end is a crash and not a pass. The fuzzer of `wex4/wex4_tools.c` puts the file copy, each stored record and each decode buffer against one, in its Windows build only. *Source: `docs/WEX4-SPEC.md` sections 9 and 15.2; `SECURITY.md` ("Testing, and its gaps").*

**libFuzzer harness, seed, corpus, oracle.** A harness is a small program that feeds one input to the code under test; libFuzzer mutates inputs and keeps those that reach new code. The three harnesses of `wex4/fuzz/` cover the C reader, the WNR3 decoders and `wex4_verify`. The seeds are the inputs a run starts from (the conformance vectors), the corpus is what libFuzzer kept (`wex4/fuzz/out/corpus/`), and an oracle is a property a harness checks besides the sanitizers, reported as `ORACLE FAILED`. *Source: `wex4/fuzz/README.md`.*

**Sanitizers (ASan, UBSan).** AddressSanitizer and UndefinedBehaviorSanitizer: compiler instrumentation that stops a program at a memory error or at undefined behaviour. CI runs `scripts/check_all.sh` under both on Linux (the `linux-sanitizers` job) and builds the libFuzzer harnesses with them. *Source: `CHANGELOG.md` (2026-11-14); `docs/HELP-WANTED.md` section 3; `wex4/fuzz/README.md`.*

## Words in the measurements

**Held-out.** Said of experts that were not used to train the model with which they are coded, so that a size measured on them is not flattered by the training: for example "40 held-out experts, model trained on 80 others". *Source: `docs/WNR3-SPEC.md` section 0 ("Measured basis").*

**TTFT and decode tok/s.** TTFT is the time to first token: from sending the prompt to the first generated token, on the client's clock. It grows with the prompt length, so it is always quoted with the prompt. Decode tok/s is the generated tokens per second counted from the first token, so the time spent on the prompt (prefill) is in TTFT and not in it. *Source: `docs/reports/v41-reap-baseline-m0-20261107.md` (setup, and section 2 for what a run records).*

**ABBA.** The run order A1 B1 B2 A2 of two arms A and B, each run in a fresh process, so that a slow drift of the machine does not favour one arm. "ABBA x2" is two such sets, four pairs. "One pair" is one A and one B and gives no interval. *Source: `docs/reports/colibri-integration-20261109.md` section 12; `docs/reports/wnr2-history-20261110.md` section 1 (measurement protocol) and "W3 final".*

**Bit-identical, identical tokens.** Two runs produced the same logits file byte for byte, or the same token ids. These are statements about correctness and are kept apart from speed. *Source: `CONTRIBUTING.md` section 5.*

**[M], [I], [R].** Tags in the 2026-11-07 to 2026-11-09 plans and reports: measured here, inferred, read in the code. The Italian reports of 2026-11-05 use [misurato], [simulazione], [pubblicato], [inferenza]. A run marked `*` in `docs/reports/wnr2-history-20261110.md` (section 12) is contaminated: other work was running on the machine. *Source: `research/v41-gpu-20261109/PLAN.md` (header); `docs/reports/wnr2-cache-decode-cpu-wddm-20261105.it.md`; `docs/WNR2-REFERENCE.md`.*

**Lane, gate and goal labels.** Short labels name work packages and acceptance gates. The same letter can mean different things in different files:

| label | where | meaning |
|---|---|---|
| M0 | V4.1 GPU plan | CPU-only baseline and drive probe |
| G1, G2, G3 | V4.1 GPU plan | V4.1 on the GPU tiers; G2 is the routed experts through the GPU pipeline with VRAM mirrors |
| P1 | V4.1 GPU plan | prefill changes |
| W1 | V4.1 GPU plan | the build of the V4.1 WNR2 container |
| W2 | V4.1 GPU plan | one batched decode per layer (`V4_WEX_BATCH`) |
| W3 | V4.1 GPU plan | the A/B of WEX1 against WNR2 on V4.1 |
| F1 | V4.1 GPU plan | confidence-gated prefetch |
| D1 | V4.1 GPU plan | a second drive |
| H1 | V4.1 GPU plan | the host passing the container variables for V4.1 |
| G1 to G8 | WNR3 specification, section 7.2 | acceptance gates of WNR3; G7 is the end-to-end proof of canonical zero, G8 the full container |
| M1 to M4 | WEX4 specification, section 13.5 | migration gates of WEX4 |
| G1 to G7 | WEX4 specification, section 1.1 | design goals of WEX4 |

*Source: `research/v41-gpu-20261109/PLAN.md`; `docs/WNR3-SPEC.md` section 7.2; `docs/WEX4-SPEC.md` sections 1.1 and 13.5.*
