# WNR2 reference: the WEX1-WEX3 containers, the WNR2 record and the WNM2 model

**What this page is.** The reference of the containers before WEX4 and of the WNR2 record codec: the WEX1, WEX2 and WEX3 header page and entry table (section 5), the WNR2 record and its model descriptor (6), the rANS codec and the WNM2 model (7), the encoder and the build procedure (8), the CPU decoder (9) and the API of the CUDA helper `kaira_wex_cuda.dll` (10). Reference code: `c/wex_format.h`, `c/wex_neural_rans.h`, `c/wex_decode.h`, `c/tools/wex_cuda_bridge.cu`.

**Status.** WEX1 to WEX3 are read by the current code and are the input of the WEX4 upgrade; no production file is in one of these forms any more. WNR2 (codec 4) was superseded by WNR3 (codec 5, [WNR3-SPEC.md](WNR3-SPEC.md)) on 2026-11-10 and every WNR2 container was deleted; its decoder is kept. The current container is WEX4 ([WEX4-SPEC.md](WEX4-SPEC.md)); what runs today is in [STATUS.md](../STATUS.md).

**History.** The record of the WNR2 period (the owner's decisions, the drives, the open work of the time, the engine integration of that time, the measurements, the timeline, the contradictions between sources, the source index and the dated updates of 2026-11-08 to 2026-11-10) moved on 2026-11-13 to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), with the same section numbers. Sections 1 to 3 and 11 to 15 below are one-line pointers to it.

## 1. Owner decisions and rules

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 2. Containers and files on disk

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 3. Open work / next steps

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 4. What WNR2 is

WNR2 is a lossless entropy coding of MXFP4 MoE expert records (FP4 E2M1 weights + UE8M0 scales). It is entry codec 4 inside a WEX3 container (header magic WEX3, version 3, header flag NEURAL_RANS2 = 16). One shared probability model (WNM2, 59,408 B) is stored once per container, and every record names it by its FNV-1a id. Weights use a 567-context rANS per matrix and scales an order-0 rANS. All CDFs are 14-bit.
- Naming: the repository slug spells it "WRN2" (KairaCode/Wex-WRN2), but every code constant, file and doc says WNR2: record magic 'WNR2', model 'WNM2', descriptor 'WNMD'. WNR2z is the canonical-zero variant. WNR1 is the older per-record-model codec (codec 3).
- The name says "neural", but the WNM2 model is a count-based context table. The trainer counts symbols and quantizes them to CDFs. No neural net is involved (see section 14).
- Container family in 3 lines:
  - **WEX1**: raw aligned MXFP4 records (one per expert, 1 UE8M0 byte per 32 weights) read unbuffered (FILE_FLAG_NO_BUFFERING). No decode step.
  - **WEX2**: UE8M0 scales packed in 256-value tiles (u8 base, 0..8 bits, residuals LSB-first, a constant tile is 2 B; prefix `<III>` = magic 0x31534357 + decoded scale count + stream bytes; entry flags = (scale_pages<<8)|1). FP4 weights are unchanged and read zero-copy to slab + scale_bytes_raw - scale_pages*4096.
  - **WEX3**: Palette3 (codec 2, 3-bit, 8 medoid levels instead of 15 FP4 values; LOSSY; only with `--format-version 3 --weight-codec palette3 --max-weight-relative-error`; never used in production), WNR1/WNR2 lossless rANS, and cross-layer SHA-256 DEDUP (entries share file_offset).

## 5. WEX container format (wex_format.h)

*This section is the WEX1 / WEX2 / WEX3 container as of WNR2. Added since: header flags 32 (WNR3) and 64 (canonical zero) and entry codec 5 ([WNR3-SPEC.md](WNR3-SPEC.md) section 2), and the WEX4 container, which has another header page and an index instead of the entry table ([WEX4-SPEC.md](WEX4-SPEC.md)).*

- All fields are little-endian. WEX_ALIGN = 4096: the header page, the entry table and every record offset/size are multiples of 4096.
- Magics: WEX1 0x57455831 (version 1, flags must be 0); WEX2 0x57455832 (version 2, only SCALE_TILES); WEX3 0x57455833 (version 3, any of flags 1|2|4|8|16). WNR2 exists only in WEX3.
- **WexHeader**, 40 B packed inside a 4096-B page: `{u32 magic, u32 version, u32 num_layers, u32 num_experts, u64 aligned_expert_bytes (max DECODED 4096-aligned slab), u64 table_offset, u32 flags, u32 reserved=0}`.
  - V2/V3 put a 16-B random build UUID at bytes 40..55. It matches primary and mirror copies and is not a checksum.
  - In WNR2 containers the WNMD descriptor sits at byte 64.
- **Header flags**: SCALE_TILES 1, PALETTE3 2, DEDUP 4, NEURAL_RANS 8 (WNR1), NEURAL_RANS2 16 ("WNR2 records + a WNM2 model before the table").
- **WexEntry**, 72 B: `{u64 file_offset, u32 total_bytes (aligned ON-DISK size), u32 scale_bytes, u32 weight_bytes (DECODED layout), u16 scale_shapes[3][2], u16 weight_shapes[3][2], u32 scale_offsets[3], u32 weight_offsets[3], u32 flags}`.
  - flags: codec in bits 0-7 (CODEC_MASK 0xff), scale_pages in bits 8-31.
  - An all-zero entry means absent/dense, and the engine uses safetensors for that expert.
  - Static asserts enforce 40/72 B.
- **Entry codecs**: 0 raw, 1 SCALE_TILES, 2 PALETTE3 (lossy), 3 NEURAL_RANS (WNR1), 4 NEURAL_RANS2 (WNR2).
- **Matrix order**: m=0 gate w1, m=1 down w2, m=2 up w3. The decoded slab holds the scales of w1,w2,w3, then the packed FP4 weights of w1,w2,w3. wex_entry_raw_sizes checks scale cols*16 == weight cols and that offsets are contiguous.
- **Record placement**: for codecs 2/3/4, `wex_record_read_offset = aligned_raw - total_bytes`, so the compressed record is read into the END of the decoded slab.
- **wex_validate_entry**, codec 4: requires header version 3 with flag 16, and total_bytes 4096-aligned, >= align(32 + 6*streams at max chunk 16384) and < aligned_raw. A record that is not smaller than raw must be stored raw (flags 0).
- **wex_validate_table** (v3): physical records contiguous; aliases only with DEDUP and byte-identical 72-B entries; the last record ends at EOF; max aligned raw == aligned_expert_bytes.
- **Layout written by the WNR2 builder**:
  - header page (WEX3 v3, uuid, WNMD at 64);
  - WNM2 model at offset 4096, zero-padded to 65536 (= table_offset = 4096 + align(59408) = 4096 + 61440);
  - 4096-aligned entry table;
  - records contiguous to EOF.
  - Header flags written: 21 (SCALE_TILES|DEDUP|NEURAL_RANS2).

## 6. WNR2 record and model descriptor

- **Descriptor 'WNMD'** (WEX_NR2_DESCRIPTOR_MAGIC 0x444d4e57) at header byte 64 (WEX_NR2_DESCRIPTOR_OFFSET): `{u32 magic, u32 bytes (must equal WEX_NR2_MODEL_BYTES 59408), u64 offset, u32 model id (non-zero), u32 0}`.
  - The model lies in [offset, offset+bytes); the offset is 4096-aligned, >= 4096, and sits between the header page and the table.
  - wex_nr2_model_location requires magic WEX3, version 3 and flag 16.
- **Record**: a 32-B header `{u32 magic 'WNR2' 0x32524e57, u32 model_id, u32 scale_payload_bytes, u32 weight_payload_bytes, u32 weight_chunk, u32 scale_chunk, u8 median[3] (bytes 24..26), u8 0 (27), u32 0 (28)}`. The header is followed by:
  - the u16 lengths of every scale stream, then of every weight stream;
  - the scale payload;
  - the weight payload;
  - zero padding to 4096 (canonical, checked).
- **Streams** never cross a matrix: matrix m has ceil(count_m/chunk) streams.
  - Each stream is a 4-B big-endian initial state in [L, 256*L) followed by byte-renormalization bytes (state = state<<8 | byte while state < L).
  - It must end exactly at its length with final state == WEX_NR_L = 1<<23.
  - Stream length is >= 4 and <= 65535 (u16).
- **Chunks**: both are powers of 2. Weight chunk 512..16384 (WEX_NR2_MIN_WEIGHT_CHUNK..WEX_NR2_MAX_CHUNK), scale chunk 256..16384 (WEX_NR2_MIN_SCALE_CHUNK..MAX).
  - Chunks are stored per record, so any legal value works without a format change: V4 used 4096:2048, V4.1 uses 2048:512.
  - The encoder validates chunks at encode_nr2.cu:60-62. The decoders read them per record at wex_neural_rans.h:229-232 and wex_cuda.cuh:404-405.
- Using u16 lengths instead of u32 offsets and long streams cut stream overhead from 2.94% (WNR1) to 0.1%.
- **Integrity model**: WNR2 records carry NO checksum or CRC. Corruption is caught only by structural rANS invariants (wex_nr2_parse, wex_neural_rans.h:213-262; wex_cuda.cuh:280-283):
  - initial state in [L, 256L); each stream consumes exactly its u16 length with final state == L;
  - length sums equal the payload sizes; record size == align4096(used bytes); zero padding.
  - That catches almost everything (300 fuzz cases, 0 accepted) but guarantees nothing. Whole-file integrity rests only on the sha256 values and the .verified markers; the only extra runtime check is COLI_WEX_RAM_CACHE_VERIFY=1 (64-bit checksum on RAM-cache slots).

## 7. Codec and WNM2 model (wex_neural_rans.h)

- **Weight context**:
  - Symbols are row-major 4-bit FP4 codes; symbol 2k is in the low nibble.
  - For each 32-weight block with UE8M0 scale s, at block start: srel = clamp(s - median_m, -3, 3) + 3 (7 values), previous magnitude pm = 8 (none), block max mx = 8 (none).
  - After symbol x: pm = x&7, mx = max(mx, x&7).
  - context = (srel*9 + pm)*9 + mx, giving WEX_NR2_CONTEXTS = 567 contexts per matrix (7x9x9), each a 16-symbol CDF.
  - median_m is the lower median of the matrix's scales (wex_nr2_median), stored per record.
  - Contexts depend only on FP4 nibble and E8M0 statistics; geometry comes from each WexEntry.
- **Scales**: order-0 rANS, one 256-symbol CDF per matrix.
- **CDFs**: integers, precision WEX_NR2_PRECISION = 14 (total 16384), every symbol frequency >= 1.
- **WNM2 blob** = WEX_NR2_MODEL_BYTES 59,408:
  - 32-B header `{u32 'WNM2' 0x324d4e57, version 1, precision 14, contexts 567, matrices 3, 0, 0, 0}`;
  - weight CDFs 3*567*17 u16 = 57,834 B (WEX_NR2_WCDF_BYTES);
  - scale CDFs 3*257 u16 = 1,542 B (WEX_NR2_SCDF_BYTES).
  - wex_nr2_model_open validates the header and every CDF.
- **Model id**: FNV-1a 32-bit over the whole blob (offset 2166136261, prime 16777619; 0 is mapped to 1). Every record repeats the id.
- **Training (deterministic)**:
  - wex_nr2_count accumulates counts[3][567][16] for weights and counts[3][256] for scales, using exactly the decode contexts.
  - wex_nr2_quantize gives each symbol 1 + floor(count*(16384-symbols)/total) and hands the remaining slots to the largest remainders (ties go to the lower symbol). Empty counts give a near-uniform CDF.
  - wex_nr2_model_build writes the blob.
- **Reference encoder wex_nr2_encode**:
  - A forward pass computes the contexts, then each stream is encoded backwards (wex_nr2_put, x_max = ((L>>14)<<8)*freq) and 4 state bytes are flushed.
  - Order: scales m0..m2, then weights m0..m2.
  - Scratch per stream is bounded by 2*WEX_NR2_MAX_CHUNK+8.
  - CPU reference functions: wex_nr2_encode/put/flush, wex_nr2_decode/decode_symbol/stream_open, wex_nr2_count, wex_nr2_quantize, wex_nr2_model_build, wex_nr2_median (wex_neural_rans.h:264-517; decode_symbol 273, stream_open 291, decode 302-350, median 355, encode 450-517).
- **Why WNR2 (2026-10-20 design)**:
  - WNR1 byte budget (687 experts, % of raw): weights 3.681 bit/weight = 86.6%; rANS flush 4 B per 512 symbols 1.47%; u32 offsets 1.47%; scale tiles (1.24 bit/scale) 0.91%; per-record CDF + header + padding 0.05%. So 2.94% was parallelism overhead.
  - Held-out cross-entropy (shared model): scale+previous (WNR1) 3.684 bit/w; + block max 3.665; + position 3.660. Rows, columns, the upper neighbour and gate<->up gave nothing. Entropy-coded scales reach 0.97 bit.
- **WNR2z** (`--canonical-zero`, wex_nr2_canonical_zero):
  - FP4 code 8 (-0) is mapped to 0 (+0) before counting and encoding. Contexts are unchanged (8&7 == 0&7).
  - Value-exact but NOT byte-exact. +/-0 is 12.8% of FP4 weights and carries 0.128 bit/weight of no information.
  - Needs a logprob comparison before production; tokens were identical in 4 runs only.
- **WNR1, for contrast** (codec 3, header flag 8, magic 'WNR1' 0x31524e57, L = 2^23):
  - Per-record model: a 256-entry u16 scale->class map plus up to 256 classes x 17 contexts x 17 u16 CDFs (12-bit total 4096; WEX_NR_CDF_BYTES 17*17*2).
  - Context = previous symbol, reset every 32 weights. Chunks of 256/512/1024 symbols with u32 stream offsets; scales as a tile stream.
  - Each WNR1 record stores its own chunk (256, 512 or 1024) at byte 20, read by wex_nr_parse (wex_neural_rans.h:34-35). The pilot used 512 (pilot-v3-neural512). WEX_NR_CHUNK_SYMBOLS 1024 (wex_neural_rans.h:12) is only the largest chunk, used by wex_validate_entry (wex_format.h:225) for the minimum-size bound; no contradiction.
  - Its GPU kernel wex_cuda_neural_rans_weights uses `__shared__ unsigned char input_cache[32768]`.
  - The pilot uniform fallback CDF is cdf-uniform-fallback.bin.
- **Generic codecs measured** (16 decoded MXFP4 slabs, 213.9 MB; ratio whole / FP4 weights / UE8M0 scales):

| codec | whole | weights | scales |
|---|---|---|---|
| zstd-3 | 0.9225 | 0.968 | 0.189 |
| zstd-19 | 0.917 | 0.965 | 0.135 |
| deflate-6 (GDeflate/DirectStorage family) | 0.920 | 0.968 | 0.158 |
| lzma-6 | 0.920 | 0.969 | 0.132 |
| WNR2 | 0.866 | 0.917 (3.67 bit) | 1.02 bit/scale |

  - Order-0 entropy: FP4 nibbles 3.866 bits (an order-0 coder saves at most -3.4% on weights); scales 0.98 bits.
  - The 567-context model gives 2.5x the weight saving of order-0 or generic codecs. GDeflate reaches only ~-8% vs -13.4% for WNR2, so DirectStorage is not worth it.

## 8. Encoder and build procedure

- **Lab home**: `lab\wex-gpu-direct-20261019\nr2\`: encode_nr2.cu, build_nr2.py (pilot), build_nr2_full.py, nr2lib.py, check_container.c, check_all.py, test_nr2_cuda.cu, test_nr2_tools.py, build_tools.cmd, build_test_nr2.cmd, run_test_nr2.cmd, dump_exports.cmd, temp_move.py (lab only), src-before-nr2\ (pre-WNR2 source snapshot, the rollback reference).
  - The V4.1 helpers sample_records.py and precheck_records.py are NOT in lab nr2: they are in `lab\v41-gpu-20261107\w1\` (build script build_helper.cmd).
  - `<Wex-WRN2>\nr2` has the nr2 sources and cmd scripts (no temp_move.py) plus the V4.1 helpers. The lab build script build_helper.cmd is not in the repo: `integration\build\build_wex_windows.cmd` builds the same helper DLL and test_decode_batch.exe.
- **wex_nr2_tools.dll**, built from encode_nr2.cu. It is lab only and not linked into the engine.
  - ABI: wex_nr2_tools_abi = 1.
  - Exports: wex_nr2_count_gpu, wex_nr2_build_model, wex_nr2_model_id, wex_nr2_encode_gpu, wex_nr2_verify, wex_nr2_verify_slab, wex_nr2_check_container, wex_nr2_count_cpu, wex_nr2_encode_cpu, wex_nr2_canonical_zero, wex_nr2_release.
  - Return codes: 0 ok, -1 invalid, 1 CUDA failed. One process-wide workspace, serialized by a mutex.
  - GPU output is byte-identical to the CPU reference. Geometry comes from each WexEntry (encode_nr2.cu:44-79).
- **Build commands**:
  - wex_nr2_tools.dll: `archive\nr2\build_tools.cmd` (nvcc, `-O3 -std=c++17 -arch=sm_120`, VS 18 BuildTools vcvars64, `-I c`; `VCVARS`, `NVCC` and `ARCH` come from the environment; the lab copy was pinned to CUDA v13.4 and `-I src/colibri/c`).
  - Helper: `lab\wex-gpu-direct-20261019\build_helper.cmd` (`nvcc -O3 -arch=sm_120 -Xcompiler /O2,/EHsc,/arch:AVX2 -shared tools/wex_cuda_bridge.cu`).
  - Test: `archive\nr2\build_test_nr2.cmd` builds test_nr2_cuda.exe.
- **nr2lib.py**: ctypes bindings (DLL_PATH = HERE/'wex_nr2_tools.dll') plus wire structs: HEADER `'<IIIIQQII'` (40 B), ENTRY `'<QIII6H6H3I3II'` (72 B), MODEL_BYTES 59408, WNR2_MAGIC 0x32524e57, WNMD_MAGIC 0x444d4e57, WEIGHT_BINS 3*567*16, SCALE_BINS 3*256, fnv1a().
- **check_container.c**: an independent MSYS2-gcc checker. Usage: `check_container <wnr2.wex> <source-v1.wex> <canonical_zero 0|1> [stride]`.
- **build_nr2_full.py procedure**:
  1. The source must be a raw WEX1 container (magic V1, version 1, flags 0, absent entries all-zero).
  2. Pass 1: GPU training counts over every present expert build the WNM2. It ALWAYS retrains; there is no load-model option.
  3. Pass 2: each expert is GPU-encoded, padded to 4096 and verified with the CPU reference (wex_nr2_verify) in a thread pool before it is written. A record that is not smaller than raw is stored raw (flags 0). Every 43rd expert is also CPU-encoded and byte-compared.
  4. Final checks: wex_nr2_check_container, a slab-path readback of every 43rd expert plus the last and all fallbacks, a whole-file sha256 and a per-record rehash.
  5. Output goes to `<out>.partial`, then os.replace.
- **build_nr2_full.py options**:
  - `--source`, `--out` (both required);
  - `--weight-chunk` (default 4096), `--scale-chunk` (default 2048);
  - `--canonical-zero`;
  - `--verify-workers` (4), `--crosscheck-every` (43);
  - `--overwrite`;
  - `--min-free-gib` (2.0; aborts and deletes the partial file below it);
  - `--report-dir` (default: the nr2 folder; added 2026-11-07 so V4.1 reports do not overwrite the V4 build-full-nr2.json / model-full-nr2.wnm2 / counts-full-nr2.npz).
  - Up-front free-space check: 0.90 x source size + margin. Outputs: build-<suffix>.json, model-<suffix>.wnm2, counts-<suffix>.npz.
- **c\tools\repack_v4_wex.py does NOT write WNR2.**
  - `--weight-codec` accepts native|palette3|neural-rans. neural-rans writes WNR1 (codec 3) and needs `--neural-cdf`, `--neural-encoder-dll` and `--neural-chunk-symbols` 256/512/1024 (default 512).
  - Its validate_container allows v3 header flags only up to 15, so it would reject a WNR2 container (flag 16). WNR2 only comes from the lab nr2 builder.
  - It writes a unique .tmp file and then os.replace.
- **Tool timings** (test_nr2_tools, 2026-10-20, V4 records):

| step | GPU | CPU |
|---|---|---|
| count | 1.51 ms/expert warm (median 1.55) | 48.0 ms |
| encode | 4.82 ms warm (median 8.84) | 148.1 ms |
| CPU verify (reference decode) | | median 317 ms |

  All checks passed: real, synthetic, canonical-zero, own-model, uniform-model, adversarial.
- **Record size vs chunk** (weight/scale, average of 3 real V4 experts): 512/256 11,914,581 B; 1024/512 11,777,365; 2048/16384 11,702,272; 4096/2048 11,664,043; 8192/4096 11,650,389; 16384/16384 11,636,053.
  - With a uniform model a record is 13,410,304 B, larger than raw, so it would fall back to raw.
  - Adversarial input: record 44,609,536 B, max stream 28,676 B, within the bound 2n+8 = 32,776.
- **Pilot build** (build_nr2.py, 688 experts): training 3.3 s, encode 57 s.
- **V4 full build** (2026-10-20): 779 s total for 147 GB (training 53.7 s, encode 453.7 s, validation 272 s; "13 minutes"). 11,008/11,008 experts verified on CPU during the build and again on GPU (check_full.json).
- **V4.1 W1 build** (2026-11-07), exact command:
  `python -I lab\wex-gpu-direct-20261019\nr2\build_nr2_full.py --source U:\kaira_wex\v41-reap\v4_flash_experts.wex --out E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex --weight-chunk 2048 --scale-chunk 512 --verify-workers 12 --min-free-gib 5 --report-dir lab\v41-gpu-20261107\w1`
  - Ran 02:34:57-02:47:01 = 723.2 s under heavy.lock (taken 02:34:33, released 02:47:55). Exit 0.
  - Phases: training 62.05 s (GPU count calls 22.39 s); encode with CPU verify on 12 workers 373.35 s (GPU encode calls 100.33 s); readback + whole-file hash 287.81 s.
  - Build verification:
    - all 7680 records exact under the CPU reference wex_nr2_verify;
    - 179 GPU records byte-equal to the CPU encoder (every 43rd);
    - wex_validate_header + wex_validate_table + wex_nr2_model_location + wex_nr2_model_open OK, and the id matches the descriptor;
    - 180 slab-path readbacks;
    - all 7680 on-disk records rehashed equal;
    - 7680/7680 records are WNR2, 0 raw fallbacks, 0 negative-zero codes mapped (canonical_zero false).
  - 64-record GPU check:
    - w1\sample_records.py (seed 20261107) pairs each record from E: with the same expert's slab from the original WEX1 on D:, asserting geometry and model id.
    - test_decode_batch.exe (source in lab\wex-prod-20261105) ran with a kaira_wex_cuda.dll built from the clean engine source of 2026-11-07 (wex_cuda_bridge.cu sha a346652f..., wex_cuda.cuh b6a6563f..., wex_neural_rans.h d875e1d1...).
    - Batch sizes 1,2,3,4,6,8,10,16 = 512 decodes, all byte-equal; ALL OK, rc=0.
    - Corruption statuses [0,0,-1,0,-1,0]: a damaged payload and an invalid entry return -1 with the destination untouched.
    - Scratch records64.bin is 2,284,530,456 B.
  - Retraining paid off: in the pre-check, 8 V4.1 experts encoded with the old V4 model came out at 0.941-0.950 of raw, vs 0.8997 with the V4.1 model. Any embedded model is lossless, so a V4 model would have been valid, just larger.
  - W1 files in lab\v41-gpu-20261107\w1:
    - build-v41-nr2.log (15,428 B);
    - build-full-nr2.json (923,467 B, per-record rows);
    - model-full-nr2.wnm2 (59,408);
    - counts-full-nr2.npz (37,633);
    - verify64.log, sample_records.py;
    - precheck_records.py (its single-call timing loop needs >= 16 records and failed on 8 by design);
    - build_helper.cmd and take_lock.ps1;
    - build\kaira_wex_cuda.dll (387,072 B) + .lib/.exp, and build\test_decode_batch.exe (208,384 B).
    - build_helper.log shows a harmless "vswhere.exe non riconosciuto" warning.
- **Re-run the GPU check**:
  1. `python -I lab\v41-gpu-20261107\w1\sample_records.py NR2.wex WEX1.wex OUT.bin [count=64] [seed=20261107]`
  2. `lab\v41-gpu-20261107\w1\build\test_decode_batch.exe HELPER.dll OUT.bin` (source lab\wex-prod-20261105\test_decode_batch.cu; helper and test built by w1\build_helper.cmd, nvcc 13.4 -arch=sm_120).
  - Bin layout: u32 model_bytes + WNM2 blob + u32 count + per record (72-B entry, u32 raw_bytes, raw slab, record of entry.total_bytes).
  - records64.bin (2.28 GB) was written to a temporary folder (`<TMP>`), so it is ephemeral and probably gone; regenerate it.
  - "failed:2" in verify64.log is the two deliberate corruptions, not a failure.

## 9. CPU decoder

- **wex_decode_nr2_record**(entry, slab, capacity, ..., model) in wex_format.h:480-507:
  - validates sizes, the read offset, and wex_nr2_parse with the model id;
  - decodes into malloc scratch and memcpy's to the slab only on success.
  - The generic wex_decode_record refuses codec 4 (returns -1, "needs its model").
- **wex_nr2_decode** decodes scales stream by stream (binary search in the 256-symbol CDF), then weights with the exact context rule.
- **Speed**: ~0.29-0.32 s per V4 expert (288-317 ms median), the same order as WNR1 (~0.3 s), about 44 MB/s.
  - Keeping up with a 6.6 GiB/s disk would need ~7.6 GiB/s of output (~11-12 of 16 cores at ~0.7 GB/s/core). Published SIMD rANS runs 0.5-1.4 GB/s/core; Recoil reaches 11+ GB/s on a 16-core Xeon W-3245.
  - With 13 inference threads, ~3 cores would be left and would need ~2.7 GB/s/core, which is not reachable. No rANS measurement exists on the 7950X3D.
  - CPU decode also loses the PCIe saving, because the raw 12.75 MiB slab goes to VRAM.
- **Table sizes**: a CDF-only table is 567 x 32 B = 18 KB (fits L2). A 16384-entry slot->symbol LUT per context would be 9.3 MB (does not fit L1/L2).
- **Gate**: CPU decode is worth doing only if the container stays on a single ~1.8 GiB/s drive AND a SIMD 567-context rANS microbench exceeds 2 GB/s on 3 cores; otherwise discard it.
- **Model registry** (wex_decode.c:97-154), coli_wex_register_model(blob, bytes, &id):
  - validates with wex_nr2_model_open and keeps a private copy; at most WEX_NR2_MAX_MODELS = 16 per process, never freed;
  - the same id with different bytes returns -1 (hash collision); re-registering an identical model is a no-op;
  - on Windows it also calls the helper's wex_cuda_register_model and records `in_helper`.
- **coli_wex_decode_record** (host-slab path, wex_decode.c:257-285):
  - For codec 4 it reads the model id at record bytes 4..7; an unknown id returns -1.
  - For codec 3, or codec 4 with the model registered in the helper, it tries the helper's wex_cuda_decode_to_host first. It prints once "[wex] CUDA reconstruction failed/unavailable (%d); using CPU fallback." and then falls back to the CPU decoder.
- coli_wex_decode_record was moved from expert_store_wex.c to wex_decode.c on 2026-10-18 23:39 so test_deepseek_v4 links.
  - Makefile.deepseek-v4 adds wex_decode.o to V4_WIN_EXTRA_OBJS; the Makefile adds it to V4_TEST_LINK_OBJS, test_expert_store_wex and build/segment.

## 10. GPU helper kaira_wex_cuda.dll

- **Source**: src\colibri\c\tools\wex_cuda_bridge.cu, kernels in wex_cuda.cuh. ABI 1 (wex_cuda_helper_abi returns 1; provides wex_cuda_decode_to_host).
- **Loading** (wex_decode.c:13-94, Windows only, lazy pthread_once):
  - COLI_WEX_CUDA_LIB gives an absolute helper path; "0" forces the CPU.
  - Default: kaira_wex_cuda.dll next to the exe, loaded with LoadLibraryExA (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|DEFAULT_DIRS).
  - Requires abi == 1 and decode_to_host. Optional symbols are resolved by feature bits.
  - Logs "[wex] Neural CUDA reconstruction enabled: <path>" or "[wex] Neural CUDA helper unavailable; using the CPU decoder." The DLL is kept until exit.
  - Non-Windows builds report every CUDA path as unavailable.
- **Feature bits** (wex_cuda_helper_features, bridge :290-296):

| bit | name | enables |
|---|---|---|
| 1 | DEVICE | wex_cuda_decode_to_device |
| 2 | REGISTER | wex_cuda_host_register / _unregister (cudaHostRegisterPortable) |
| 4 | NR2 | wex_cuda_register_model; decode_to_host and decode_to_device accept codec 4 |
| 8 | BATCH | wex_cuda_decode_batch_to_device |

  - On 2026-11-09 src returned 15. The 2026-10-20 helper (sha cac5348d...) reported 7, with no batch support. *(Later bits: 16 = WNR3, 32 = `wex_cuda_decode_batch_to_host`; the bridge in `c/tools/wex_cuda_bridge.cu` now returns 63.)*
- **Exported API**:
  - `unsigned wex_cuda_helper_abi(void)`; `unsigned wex_cuda_helper_features(void)`;
  - `int wex_cuda_decode_to_host(const WexEntry*, unsigned char *slab, size_t cap)`;
  - `int wex_cuda_decode_to_device(const WexEntry*, const unsigned char *record, int device, unsigned char *const *dst_scale, unsigned char *const *dst_weight)`;
  - `int wex_cuda_register_model(const unsigned char *blob, size_t bytes)`: 0 ok/already; -1 invalid, or another blob with the same id; 1 table of 16 full or no memory;
  - `int wex_cuda_decode_batch_to_device(int n, const WexEntry *const*, const unsigned char *const*, int device, unsigned char *const *dst_scale, unsigned char *const *dst_weight, int *status)` (bridge :409);
  - `int wex_cuda_host_register(void*, size_t)`; `int wex_cuda_host_unregister(void*)`.
- **Engine wrappers** (wex_decode.h:24-57):
  - coli_wex_decode_record; coli_wex_cuda_decode_count; coli_wex_cuda_device_available (feature 1);
  - coli_wex_cuda_decode_to_device: 0 ok, 1 unavailable, -1 invalid; codec 4 returns 1 unless feature 4 is present;
  - coli_wex_cuda_host_register/unregister (1 when unavailable); coli_wex_cuda_device_decode_count; coli_wex_register_model;
  - coli_wex_cuda_nr2_available (feature 4); coli_wex_cuda_batch_available (features 8 and 4);
  - coli_wex_cuda_decode_batch_to_device (wex_decode.c:198-219): returns the count of non-zero statuses; without feature 8 every status is 1 and the caller falls back per record.
- **decode_to_device semantics**:
  - It validates, uploads, decodes into private scratch, checks the error word, commits the 6 regions, then synchronizes.
  - An invalid record returns -1 and leaves the destination untouched. Scratch follows the current device.
  - Backend and helper runtimes share the CUDA primary context, so device pointers pass between them (tested).
  - A WNR2 record without a registered model returns 1.
  - Every decode entry point calls cudaSetDevice(device) (bridge :307, :346, :440) and never restores the caller's previous device, so the calling thread is left on `device`. Harmless on the single 5090 today; a trap for W2 integration.
- **Helper memory footprint** (wex_cuda_bridge.cu:23-90, 384-399, 409-500): workspaces are thread_local, allocated with cudaMalloc OUTSIDE the backend's VRAM accounting, and freed only when the thread exits.
  - Batch path: up to 8 lanes per calling thread, each holding device_record + device_raw at the largest aligned record. On V4.1: 8 x 2 x 17.93 MiB = ~287 MiB per thread (~16 expert mirrors).
  - Single-record path: another 2 x aligned of VRAM per thread, plus pinned host_raw on the host path.
  - Each ensure() requires free >= 2*bytes + tiles*4 + lut + 8 MiB.
  - The G2/W2 mirror budget (~1,030-1,110) or DSV4_CUDA_VRAM_RESERVE_MB must cover the helper lanes (WDDM over-commit risk).
- **Helper model registry**: process-global, WEX_CUDA_NR2_MAX_MODELS = 16, WEX_CUDA_NR2_MAX_DEVICES = 64. Device tables are built lazily per (model, device) and kept for the process lifetime.
  - Layout (one cudaMalloc): blob (align256 = 59,648 B) + weight rows [3*567] x 32 B + scale_direct [3][16384] u32 (slot -> (freq-1) | (slot-start)<<14) + scale_symbol [3][16384] B.
  - Weight buckets [3*567][1<<10] u32 (~6.97 MB) exist only for lookup 0. With lookup 1 the tables total ~360 KB (359,936 B).
- **CUDA decoder** (bit-identical to wex_nr2_decode):
  - Host side: wex_cuda_plan_nr2 validates with wex_nr2_parse; wex_cuda_nr2_offsets prefix-sums the u16 lengths into u32 offsets (streams+1).
  - **Scale kernel** wex_cuda_nr2_scales: one thread per stream, 32 threads/block, direct tables in dynamic shared memory. WEX_CUDA_NR2_SCALE_SMEM = 16384*5 + ring 8192 = 90,112 B ("88 KB", opt-in via cudaFuncSetAttribute).
  - **Weight kernel** wex_cuda_nr2_weights<Lookup>: one thread per stream, the matrix's 567 CDF rows in shared memory, branch-free 16-way compare (PRMT sign trick), one uint4 store per 32 symbols. Each block serves one matrix.
  - **Payload ring**: a 64-word (WEX_CUDA_NR2_RING_WORDS) shared ring per lane, refilled by cp.async at warp-uniform points, with WEX_CUDA_NR2_RING_AHEAD = 40 words ahead. Renormalization is branch-free (0/1/2 bytes per symbol). Compared with register prefetch, the scale kernel went 193 -> 75 us and the weight kernel 300 -> 253 us per expert.
  - **Errors**: any violation (bad initial state, read past end, early end, final state != L) sets *error.
  - **WEX_CUDA_NR2_LOOKUP** (default 1): 1 = shared CDF rows + 16-way compare (~253 us/expert at chunk 4096); 0 = global coarse buckets (WEX_CUDA_NR2_BUCKET_BITS 10) + exact walk (~925 us; the 7 MB table misses L1).
  - **Model table kernels**: wex_cuda_nr2_build_rows / _build_scales / _build_buckets.
  - **Commit** wex_cuda_commit_regions (grid 64x6, 256 threads): copies the 3 scale + 3 weight regions into the caller's buffers ONLY if *error == 0, on the same stream, so the destination never holds partial or corrupt data. In the raw buffer, weights start at align16(scale_bytes).
  - **Occupancy**: a single 11 MiB V4 record launches only 12 scale blocks + 192 weight blocks of 32 threads (about 1 warp/SM on 170 SMs). It is latency-bound at ~200 cycles/symbol/lane.
- **Single-record pipeline** (wex_cuda_nr2_decode): a thread-local workspace with two non-blocking streams.
  - `stream` uploads the offsets, then the record head (header + lengths + scale payload) and records event `head`, then the weight payload and event `data`.
  - `aux` waits on `head`, memsets the error word and runs the scale kernel; it waits on `data` and runs the weight kernel, then the commit (device path) or a D2H copy to pinned host_raw (host path), then the error D2H and a sync.
  - Code comment: on WDDM each copy-engine/kernel switch costs ~15-35 us.
  - ensure() needs free VRAM >= 2*bytes + 4*tiles + lut + 8 MiB.
  - Cost: ~14 CUDA calls and 2 syncs per decode.
- **October bridge optimization** (2026-10-19):
  - The record goes straight from the caller's slab to CUDA: no second pinned host buffer (saves 12.75 MiB per thread on V4) and no duplicate validation.
  - Output stays in private scratch until kernels, return copy and error check finish.
  - Scripts: lab\wex_neural\run_bridge_optimization.py, verify_bridge_optimization.py, validate_integrated_optimization.py.
  - Direct-path cost was estimated from code, not measured: ~2 x 12.75 MiB VRAM workspace per decoding thread plus a LUT, and two page-locked 12.75 MiB RAM buffers.
- **Batch API** (feature 8; committed in the engine tree):
  - Spreads n records round-robin over up to WEX_CUDA_BATCH_LANES = 8 thread-local lanes. Each lane has its own copy `stream` and kernel `aux` stream plus events head/data/consumed (`consumed` orders a lane's next record after its previous kernels).
  - Offsets and error words live in page-locked scratch (WexCudaBatchHost). Workspaces grow per call. ONE host wait per batch.
  - Destinations: record i, matrix m -> dst_scale[3*i+m] / dst_weight[3*i+m].
  - status[i]: 0 written; -1 invalid record or destination (untouched); 1 CUDA/resources unavailable or model not registered (data correct or untouched, never partial). If queueing fails at record i, i and every later record report 1.
- **Deployed helper** (since 2026-11-09 06:17): app\colibri-optimized\kaira_wex_cuda.dll = the W1 build, sha256 cceb4fb573b3b3e7b71843bf85a2b87cf9efdd94155bfc5deec717cce949e782, 387,072 B, copied from lab\v41-gpu-20261107\w1\build (built 02:04 on 2026-11-07 from the clean engine source). This binary passed the 64-record V4.1 check, was the helper in W3 and the 11-08 A/B, and ran the mirror verify (41,271 device decodes, 0 failures). Features 4 and 8 are in it.
  - The previous helper a79772004426c38fe2ed1196d27cb4e17a52512c0368d5bb71b86b2e3286b049 (2026-11-07 00:47:44, built in ws-int for the V4.1 install, same size) is kept in lab\app-before-mirror-20261109-061724.

## 11. Engine integration (deepseek_v4.c only)

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 12. Measurements

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 13. Timeline

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 14. Contradictions between sources

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.

## 15. Source index

Moved to [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md), same section number.
