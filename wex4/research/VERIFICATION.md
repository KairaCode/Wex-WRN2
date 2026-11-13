# Verification of the experiments (2026-11-11)
Each experiment in this folder was re-checked by an independent pass that tried to refute it. The corrected numbers below are the ones the spec uses (docs/WEX4-SPEC.md, sections 11 and 12).

## dtype-entropy

Confirmed: yes.

Re-run: Independent code (own safetensors index, own 14-bit quantiser, own cost function, a real rANS byte counter), CPU only, numpy, single thread, every command under 10 s. Read 1,284,088,064 B of model data in total (1.196 GiB), read-only. Wrote only inside wex4\research\dtype-entropy\verify.

1. Experiment's own `measure.py --selftest`: passes. Header index: non-expert totals match exactly (F8_E4M3 7,086,800,896 B / 353 tensors, BF16 3,874,076,672 / 516, F32 169,167,688 / 387, F8_E8M0 6,920,704 / 353 = exactly 1/1024 of F8_E4M3, so one scale per 32x32 block holds).

2. Stage `repro` (same 5 tensors as the experiment, 81 MB): ideal bits per element identical to 5 decimals for all 12 model/tensor pairs checked (F8 WBP1, block-scale context, exp4 + raw, order-1; BF16 WBP1, exp8 + raw, lo given hi low 4 bits; F32 WBP1, signexp9 + raw). Bytes agree within 0.03% (e.g. layers.20.attn.wo_a WBP1 27,550,178 B mine vs 27,550,690 B theirs; embed[126155:129280] 0.69118 vs 0.69143). Real rANS size counted on 6 planes, with a scalar encode + decode round trip on 3 streams per plane: real is 0.012-0.033% below the model.

3. Stage `f8` (layers 10 and 30, all 8 large kinds, plus mtp.1 wo_a and wq_b, 400 MB, none in the experiment's sample): WBP1 per tensor 0.8113 to 0.8310; kind-level numbers and the 0.81868 / 0.81440 estimates above. Scales: 2 to 6 distinct values per record, 0.56 bit per scale.

4. Stage `bf16` (379 MB): 32 stratified 500-row slices of embed and of head, vision blocks 8 and 24. embed slices 0.6689-0.6706, head 0.6661-0.6671; one table for all 32 slices 0.67055 (embed), 0.66631 (head).

5. Stage `tail` (64 MB): rows [126155:129280] of embed and head in 250-row steps - located the anomaly in about 375 rows. `one_record.py` (191 MB): embed as one record 0.67123 (WBP1), 0.66309 (exponent split).

6. Stage `f32` (169 MB): every F32 tensor >= 1 MiB; exactly 2 of 4 planes coded in every record; plane entropies b0..b3 = 7.9996, 7.9996, 7.9708, 2.993 bits.

7. `combine.py`: kind-by-kind comparison against the experiment's results.json and the re-extrapolated totals.

Files (all under wex4\research\dtype-entropy\verify): verify.py, vidx.py, combine.py, one_record.py; logs and JSON in out\ (repro.log, f8.log, bf16.log, tail.log, f32.log, one_record.log, combine.log, verify_*.json, combine.json).

Not checked: decode time (no decoder timed, no C build, no GPU); the richer models I did not re-implement (block classes, row/column classes, per-row and per-column CDFs, sign/exponent/mantissa splits other than the three above) - they share the cost core that I did verify; BF16 kinds other than embed, head and vision (10% of bytes) and F8 indexer/MTP remainder (5%) keep the experiment's numbers; head as a single record was not estimated (only embed); the -0/+0, NaN and duplicate-probe facts; whether these tensors are on the per-token read path.

Issues found:
- BF16 whole-dtype WBP1 (0.67165) is biased high by about 0.3 point (about 11 MB). embed and head (68% of BF16 bytes) were sampled with 5 slices of 3125 rows, and the last slice [126155:129280] (0.6914 embed, 0.6784 head) got 20% of each tensor's weight. A 250-row probe of those same rows shows the anomaly is confined to about 375 rows (0.3% of the tensor): [128655:128905] at 0.707/0.687 and [129155:129280] at 0.693/0.684; the other 11 sub-slices are 0.668 to 0.670. With 32 stratified 500-row slices per tensor plus the exact tail, embed is 0.66986 (reported kind value 0.67474) and head 0.66656 (reported 0.66926).
- Record granularity matters on that tail and RESULTS.md does not say so: the same 3125 embed rows cost 0.6914 as one 32 MB record but 0.6730 as 250-row records (one order-0 table over a mixture of row populations). embed as ONE record with one table per plane is estimated at 0.67123 for WBP1 and 0.66309 for the exponent split (body histogram scaled from 31 slices, tail exact) - still below the reported 0.67474.
- F8_E4M3 whole-dtype WBP1 (0.82001) is biased high by about 0.13 point (about 9 MB): layers 0/20/39 give the two extreme layers 2/3 of the weight. On layers 10 and 30 (not in the experiment's sample) the attention kinds are 0.3 to 0.4 point lower (wo_b 0.8115 vs 0.8143, wq_b 0.8176 vs 0.8213, wo_a 0.8187 vs 0.8226) and the shared experts 0.5 point higher (0.8280 vs 0.8230).
- The gain of the block-scale context on F8_E4M3 is somewhat understated: on layers 10/30 it is 0.81440 vs 0.81868 = -30 MB extrapolated (reported -25.6 MB), concentrated in the shared experts (0.828 -> 0.815, up to -1.6 point per tensor); on wo_b and wq_a it is zero.
- F32 needed no extrapolation: all 86 hc_*_fn tensors are 169 MB. Measured on all of them: WBP1 0.84389 (reported 0.84495), sign+exponent coded with 23 raw mantissa bits 0.84209 (reported 0.84304), top-byte plane 2.993 bits (reported 3.015).
- Minor: the experiment's 14-bit quantiser dumps the whole rounding remainder on the most frequent symbol; a largest-remainder rule saves up to 0.025 point on near-flat planes (8,024 B on the 32 MB embed tail slice), elsewhere under 0.003 point.
- Minor: results.json read_bytes = 1,253,688,648 but run.log and RESULTS.md say 1,278,339,912; the JSON field is taken before the equality-check reads.
- Not a defect but a limit for the owner's decode-speed question: no decoder was timed by the experiment or by me. 'Decode cost' is only a count of entropy-decoded symbols per element. What my byte counts add: under WBP1 the BF16 low-byte plane and the F32 byte-2 plane are rANS-coded for about 0.2% gain each (15,963,407 of 16,000,000 B; 490,440 of 491,520 B), so half of WBP1's entropy decoding on those dtypes buys almost nothing - consistent with the field-split recommendation.
- Shared-table 'held-out' column is genuinely leave-one-out and only covers order-0 byte planes, as disclosed. My own held-out checks agree in direction: F8 table from layer 10 used on layer 30 and vice versa 0.81881 vs per-record 0.81822 (+0.06 point); F32 leave-one-out over all 86 tensors 0.84698 vs 0.84389 (+0.31 point).

Corrected numbers:

Method is sound and per-tensor numbers reproduce exactly; only the whole-dtype extrapolations move, all slightly downward (the report was conservative). Corrected values use my samples for the kinds I measured and the experiment's numbers for the rest; F8 and BF16 are still extrapolations, F32 is measured in full.

| Quantity | Reported | Corrected | Basis |
|---|---|---|---|
| F8_E4M3 WBP1 | 0.82001 (5,811.2 MB) | 0.81868 (5,801.8 MB) | layers 10, 30 + 2 MTP kinds, 94.6% of bytes by kind |
| F8_E4M3 block-scale context | 0.81640 (-25.6 MB vs WBP1) | 0.81440 (-30.3 MB vs WBP1) | same |
| BF16 WBP1 | 0.67165 (2,602.0 MB) | 0.66870 (2,590.6 MB) with records of 250-500 rows; about 0.669 if embed/head are single records | 32 stratified slices + exact tail for embed/head, vision blocks 8 and 24 |
| BF16 exponent coded, sign + mantissa raw | 0.66489 | 0.66154 (-27.7 MB vs WBP1; reported gain -26.2 MB) | same |
| BF16 hi order-0 + lo given low 4 bits of hi | 0.66175 | 0.65878 (-38.4 MB vs WBP1; reported gain -38.4 MB) | same |
| F32 WBP1 | 0.84495 | 0.84389 | all 86 tensors >= 1 MiB (169,082,880 B) |
| F32 sign+exponent coded, 23 mantissa bits raw | 0.84304 | 0.84209 | same |
| Three dtypes with WBP1 (11,130,045,256 B) | 8,556.2 MB, 0.7687, about 2,574 MB saved | 8,535.2 MB, 0.7669, about 2,595 MB saved | sum of the above |

Unchanged conclusions: BF16 field split beats byte planes with one coded symbol per element; F8 stays order-0 on the byte, with the block scale as the only cheap context; order-1 does not pay (F8 layer 20 wo_a: 0.82414 vs WBP1 0.82106); per-record tables beat shared ones for tensors >= 1 MiB; total stake beyond WBP1 is about 70-80 MB.

Cost model vs real coder: real rANS bytes (32-bit state, L = 2^23, 14-bit CDF, byte renormalisation, 4096-symbol streams, 4-byte flush + u16 length) are 0.012% to 0.033% BELOW the experiment's "cross-entropy + 6 B per 4096 symbols" on 6 real planes (33.5 M-symbol F8 plane: 27,545,553 B real vs 27,549,664 B model).

## fast-cpu-decode

Confirmed: yes.

Re-run: Everything ran on the desktop, CPU only, from wex4\research\fast-cpu-decode\verify (run.sh, scan.py, srccmp.py, vh.c, tailcut.inc; logs in verify\out).

1. scan.py (read-only): the experiment's model.wnm2 and 6 .rec copies are byte-equal to the production container. Entry table: 7,680 records, all codec 5, flags 0x0f, chunks 2048/512, mean 0.8540 of raw. Extracted 8 held-out records.
2. Rebuilt test_fast_decode.c from the experiment's sources (AVX2, scalar, UBSan trap, and one build without OpenMP).
   - Equality on held-out records: AVX2 8 records x 19 variants x {1, 8} threads + default, 0 mismatches; scalar 3 records, UBSan 1 record, no-OpenMP 1 record, all 0 mismatches.
3. The experiment's bench, best of 5:
   - same 6 records: reference 508.61 ms, default 71.51 / 13.56 ms, per-call tables 73.48 / 14.03 ms;
   - 8 held-out records: reference 518.11 ms, default 72.65 / 11.26 ms.
4. Own harness vh.c, own timer:
   - held-out: reference 516.1 ms; fast 78.7 / 41.0 / 20.3 / 12.5 / 8.8 / 9.1 ms at 1 / 2 / 4 / 8 / 16 / 32 threads; default entry 9.5 ms (16 threads);
   - fresh output buffer: 93.2 / 15.1 ms; drop-in wrapper: 92.0 / 17.3 ms; shipped wrapper: 512.1 ms; reference at -O2: 513.3 ms;
   - later run, machine about 10% slower: 568.8 / 90.0 / 14.8 ms, drop-in 18.4 ms, shipped 587 ms.
5. Source check: fast-decoded slabs of L8 E100, L37 E150 and L39 E120 equal the safetensors tensors with canonical zero applied.
6. Mutation test with the experiment's harness, new seed, held-out data: 20,000 records at the decoders, 14,158 byte-mutated (all rejected), 5,842 constructed, 380,000 fast decodes, 0 mismatches, 3m11.
7. Hash-blind differential (hash check removed from both decoders): 20,000 records, 1,521 of 14,115 byte-mutated accepted by the reference, 0 mismatches, 3m35.
8. Full-size checks (vh reenc): 12 re-encoded full records (chunks 512-16384, CZ off/on with -0, ceil underflow, ceil flags cleared) and 24 full-size boundary mutations; 36 cases x (11 variants + default), 0 mismatches.
9. Over-read test (vh tailcut): 64 cases against guard pages, 0 mismatches, no fault. Controls: bound check removed in kernel 1 or kernel 2 gives a segfault; WEXF_SLACK = 0 gives a segfault in the experiment's mutate; a UBSan trap probe traps.
10. Negative control replicated: the experiment's 6 edits all caught; 4 of my 5 extras caught (the model-id check is not reachable by the harness).

Not checked: the laptop, thread counts in a busy engine, concurrent or nested calls, other compilers or operating systems, the scale-kernel bound check under a guard page, WNR2 records, the malloc-failure path.

Issues found:
- No refutation: the headline numbers reproduce and I found no semantic difference from the reference by reading wex_fast_decode.h or by testing. Rebuilt from the experiment's sources, same 6 records, best of 5: reference 508.6 ms (claimed 505.2), default configuration 71.5 ms on 1 thread (claimed 71.6) and 13.6 ms on 8 threads (claimed 13.8). The other 19 table rows are within about 3%, worst case 11% (k2 gather 16 streams, 8 threads: 20.1 vs 18.1 ms).
- The 13.8 ms figure is decode-only into warm, reused output buffers; it is not what a drop-in inside wex_decode_nr2_record costs. That wrapper adds an 18.8 MB scratch malloc and a memcpy per record. Measured with the fast call swapped in, 8 held-out records: 17.3 ms at 8 threads and 92.0 ms at 1 thread, against 12.5 and 78.7 ms decode-only in the same run (+39% at 8 threads). Against the shipped wrapper (512 ms) that is about 30x, not 36.6x. Decoding into a fresh, untouched buffer costs 15.1 ms against 12.5 warm.
- The default entry point wex_fast_decode does not run the 8-thread configuration: it passes threads = 0, which means omp_get_max_threads(). In this process on the desktop that was 16 threads, 9.5 ms per record. In the engine it would take every core unless OMP_NUM_THREADS or options.threads is set. Calling it from several threads at once or from inside an OpenMP region was not measured by either of us.
- Timing drifts by more than 20% within one session, although the speed-up ratios hold. Best-of-5 for the same code ranged 71.5 to 90.0 ms (1 thread) and 11.3 to 14.8 ms (8 threads), with the reference moving 505 to 569 ms alongside; ratios stayed at 6.3-7.1x and 37-46x. Single-decode medians in the slower state were 14.3-16.0 ms (8 threads) and 88-93 ms (1 thread), so 13.8 ms is the favourable end; 12-16 ms is the realistic range. A 30 ms or 200 ms idle gap before each decode made no measurable difference.
- The in-sample caveat is closed for the desktop. On 8 other records, including the smallest (0.8505 of raw) and largest (0.9266) in the container, all 19 variants x {1, 8} threads plus the default entry were byte-identical to the reference. The experiment's bench on those 8 gives reference 518.1 ms and default 72.65 / 11.26 ms. The container is homogeneous (all 7,680 records codec 5, flags 0x0f, chunks 2048/512, one shape set), so records without ceil flags, without CZ or with other chunk sizes are covered only by re-encoded test records.
- The experiment compared only with the reference decoder. I also compared wex_fast_decode output with the safetensors source (with -0 nibbles set to +0) for L8 E100, L37 E150 and L39 E120: equal over all 18,800,640 bytes each.
- The mutation test's blind spot (every byte-mutated record rejected through the hash) hides nothing in my sample. With the final-state hash comparison removed from both decoders (7 one-line edits on copies), 20,000 cases: 1,521 of 14,115 byte-mutated records were accepted by the reference, and all 19 variants plus the default agreed in status and bytes (380,000 fast decodes, 0 mismatches).
- The experiment's fuzz never exercises the 32-symbol bound check that keeps the read-ahead kernels inside the payload, because its mutations overrun a fast-path stream by a few bytes at most. In my targeted test (tail of a stream deleted, fewer bytes following than deleted, record flush against a PAGE_NOACCESS page), 64 cases gave 0 status mismatches and no fault. With the check removed from kernel 1 or kernel 2 the same test segfaults, so it does detect a missing check. For the scale kernel the overrun cannot be constructed with this model, so that kernel's bound check remains untested by anyone.
- The model-id check (wex_fast_decode.h line 517) is not exercised by any test, because the harness always parses with the model id. Removing it is not caught in 6,000 cases. By reading, it is the same condition as the reference's.
- The baseline is not inflated: the reference built with the production flags (-O2, no -march) takes 513 ms, against 516 ms at -O3 -march=haswell.
- Laptop numbers (129.2 / 38.0 ms, 1 record, 3,000 mutation cases) were not re-run, since I may write only inside verify\. I only checked that the report matches out/laptop.log. They remain single-record and unverified.
- negative_control.sh: when I ran a copy from another folder, the WEX_SRC override did not take effect and the build failed (cause not investigated; it may be my invocation). I replicated its 6 edits by hand with held-out data and another seed: 6 of 6 caught. Four more of my own (kernel 2 CZ check, unfused ceil underflow, kernel 0 CZ check, a wrong constant in the vectorised hash) were also caught.
- My own deviation from the limits: before I set OMP_NUM_THREADS=8, the default entry in the equality test and my thread-scaling run used 16 and 32 threads for a few seconds in total. No GPU, no commit, nothing written outside verify\; about 0.3 GiB of model data read.

Corrected numbers:

No headline number needs correcting; these qualify or extend them (desktop only).

- Decode-only, default configuration (kernel 1, 4 streams, fused), best of 5:
  - the experiment's 6 records: 71.5 ms (1 thread), 13.6 ms (8 threads), reference 508.6 ms;
  - 8 held-out records: 72.65 ms, 11.26 ms, reference 518.1 ms;
  - range over the session: 71.5-90.0 ms and 11.3-14.8 ms, reference 505-569 ms; speed-up 6.3-7.1x on 1 thread, 37-46x on 8 threads.
- As a drop-in inside wex_decode_nr2_record (parse + 18.8 MB malloc + decode + memcpy + free): 17.3-18.4 ms at 8 threads and 92-94 ms at 1 thread, against 512-587 ms for the shipped wrapper. That is about 30-32x, not 36.6x.
- wex_fast_decode default entry: uses omp_get_max_threads(), 16 threads here, 9.5 ms per record. Explicit thread counts gave 41.0 ms (2), 20.3 ms (4), 12.5 ms (8), 8.8 ms (16) and 9.1 ms (32).
- Realistic single-decode figure at 8 threads: 12-16 ms (medians 14.3-16.0 ms in the slower machine state), not a fixed 13.8 ms.
- Correctness statement, extended: byte-identical to the reference on 14 records (6 in-sample + 8 held-out) and to the safetensors source on 3. A further 20,000 mutation cases on held-out data with a new seed gave 0 mismatches, as did 20,000 hash-blind cases (1,521 mutated records accepted by both). Bound-check over-read safety is demonstrated for the weight kernels only.
- Laptop figures: unchanged and unverified.

## wnr4-record

Confirmed: yes.

Re-run: All in wex4\research\wnr4-record\verify (scripts kept, record copies deleted; run_verify.sh lists the commands but was not run as one script).

1. scan.py, my own code: header + u16 length table of all 7,680 records (299.4 MB read). 0 of 7,680 rows differ from the experiment's container_scan.json. Used sum 123,289,766,966 B (mean 16,053,355.07 = 0.853873 of raw); padding 15,807,434 B; alignment savings 13,840,384 (512 B) / 15,569,024 (64 B) / 15,807,434 (none); coded length table 14,396.29 B per record, saving 188,034,914 B. All records are 2048:512 with flags 15.

2. Prototype rebuilt from the experiment's unmodified wnr4_proto.c (staged codec sources identical to repo c/ and to the experiment's stage), run on 23 records, 8 threads, 73 s: 16 fresh records disjoint from the experiment's sample, 4 overlapping, 3 extremes (largest 37:150, smallest 39:43, worst length table 39:105). Harness byte-exact 23/23; all 26 candidate profiles round-trip 23/23; bit-flip smoke test rejected 23/23. The 4 overlapping records (0:11, 20:175, 39:110, 39:120) give identical byte counts in every profile. Fresh-16 means are within 0.4% of the experiment's for every candidate. Extremes for lenc_sd: 22,482 / 24,557 / 22,066 B.

3. st_check.py: the reference decode of the production record equals Z(safetensors source), with my own canonical zero, for L39 E120 and L16 E177 (43 MB read from the shards). Chunk variants through the unchanged WNR3 DLL: sizes equal the prototype's minus 8 B, reference decode OK for all 7 pairs.

4. extra.c (includes the experiment's source with main renamed): reference CPU decode 0.579 s per record single-thread; host offsets 5.40 us (u16) against 24.80 us (coded table); 1,220 single-bit flips per profile rejected 1,220/1,220 in lenc_sd, lenc_sd_tail5 and tail5_hash3, with 0 accepted.

5. The experiment's layer_model.py rerun for layers 38 and 39 (177.6 MB read): the 4 rows reproduce exactly (prod and layer sizes identical; saved 6,084 / 50,479 / 4,485 / 403 B).

6. The experiment's report.py rerun on its out/*.json reproduces results.json exactly, apart from the 76-record length-table alternatives, which were not rerun. Arithmetic checked: 26,825 x 7,680 = 206.0 MB; plus 15.57 MB = 0.22 GB = 0.180% of the container; maximum 759 MB = 0.616%.

7. Read, not run: c/wex_cuda.cuh (offsets on the host, open, renorm, both kernels, plan builder), c/wex_neural_rans.h (parse and decode), docs/WNR2-REFERENCE.md lines 550-551 (the quoted timings are there), the drafts' flushx.json and WNR3-DRAFT-v2.1 (small_k0 -0.1876% of raw against the experiment's 0.188%; small_k8 -0.184% against 0.186%; drafts' TAIL -41,548 B without CZ against the experiment's 39,207 B with CZ and a 3-bit hash kept, which is consistent).

Issues found:
- No headline number was refuted. All checks ran on the CPU, read-only, about 0.89 GB read in total. Not checked: any GPU build or run, the other 37 records of the experiment's 41-record sample, layer-model layers other than 38-39, the 76-record length-table alternatives, and the drafts' LENC figure.
- Coded length table is not exactly zero decode cost. Host offsets build measured single-thread, best of 7, on L16 E177: 5.4 us (u16 prefix sum) -> 24.8 us (naive unpack + prefix sum), about +19 us per record. Relative to the parent's 0.37 ms per record GPU batch figure that is about 5% (my derivation) if it runs serially on the submit thread. The experiment's 8.5 -> 36.5 us was taken with 8 threads busy. The GPU kernels really are untouched: wex_cuda_nr2_offsets is host code and uploads a u32 table (read in c/wex_cuda.cuh, not run).
- The summary's 'no loss at batch 8-16' for larger chunks overstates the quoted doc. docs/WNR2-REFERENCE.md line 551 (test_decode_batch, V4/WNR2, 2048:512 vs 4096:2048): b8 0.339 vs 0.399 ms (+18%), b10 0.403 vs 0.431, b16 0.382 vs 0.352; line 550 (bench_nr2_rechunk, 8 concurrent): 0.132 vs 0.137. Only b16 favours the larger chunks, the comparison changes weight and scale chunk together, and nothing exists for weight chunk 8192 or for V4.1/WNR3. The speed side of the chunk rows (0.29-0.45% of raw) is therefore unmeasured, as the experiment's caveats also say.
- Per-layer model 'mean 3,197 B saved' is carried by one record (L38 E81, +50,479 B). The experiment's own 14 deltas have median +188 B and mean -440 B without that record. The cost of 40 model blobs (40 x 59,408 B = 309 B per record) is not charged. The 'no gain' conclusion stands on that sample but is weak (14 records, 7 layers).
- By-product of my rerun, 4 records only, not a conclusion: one model trained on the 8 training experts of layers 38-39 gives +44,512, +17,591, +11,265 and -9,468 B against the production model on the 4 held-out records (mean +15,975 B, 0.085% of raw). Late layers may differ enough for a group model; untested beyond this.
- '188.0 MB exact' for the coded length table is gross of the 8-B larger prototype header: 188,034,914 B gross, 187,973,474 B net unpadded, 187,985,920 B if today's 4096-B padding is kept (exact, my scan). Immaterial.
- Groups of 64 beat the recommended groups of 32 on the exact scan: 14,123.3 vs 14,396.3 B per record (273 B per record, 2.1 MB over the container). Immaterial.
- The chunk-size rows were measured with the prototype's 40-B header. The unchanged WNR3 codec saves exactly 8 B more per record (checked through the DLL on L16 E177 for 7 chunk pairs up to 16384:512, reference decode OK each time). This supports 'no new codec needed' on the CPU reference; for the GPU it rests on reading (kernels take plan.chunk, WEX_NR2_MAX_CHUNK = 16384, spec line 58 'any legal pair works').
- The prototype decoder is weaker than the reference on structure, with no effect on sizes: the profile (lenc/sd/k/tail) is passed out of band, the scale and weight payload sums are checked only as a total, and reserved flag bits are not checked. The round trip itself is strict (output buffer pre-filled with 0xA5, memcmp over the whole raw slab against the reference decode).
- Corruption evidence is still thin, as the experiment said. My extra test on one record (L16 E177): 1,220 single-bit flips per profile (all 320 header bits, 300 length section, 300 payload, 300 stream-start bytes) were rejected 1,220/1,220 in lenc_sd, lenc_sd_tail5 and tail5_hash3. This is not an audit of the 3-bit-hash + digest profiles, and no GPU digest path exists.
- Alignment: wex_nrx_parse enforces total & 4095 == 0, so any alignment change is a parser change (the experiment says so). 4-B alignment would save 15,795,604 B against 15,569,024 B for 64-B; both are 0.013% of the container and do not reduce bytes read.
- The parent's '~0.8545 of raw' mean record is the spec estimate. The exact container mean is 0.85387 (used) or 0.85398 (padded), as the experiment reported.
- Nothing here improves decode speed, as the experiment states. CPU reference decode measured 0.579 s per record single-thread (best of 3, L16 E177, 32.5 MB/s of raw); 0.676 s mean with 8 threads busy.

Corrected numbers:

No headline number needs correcting. Refinements only:
- Coded length table, exact over 7,680 records: 188,034,914 B gross; 187,973,474 B net of the 8-B larger header; 187,985,920 B with 4096-B padding kept. Groups of 64 would give 14,123.3 B per record instead of 14,396.3 B.
- Host cost of the coded length table: 5.4 -> 24.8 us per record single-thread (experiment: 8.5 -> 36.5 us under load), so it is not strictly zero decode cost.
- Chunk-size rows through the real WNR3 codec save 8 B more per record than tabulated (e.g. L16 E177 8192:2048: 81,160 B against the prototype's 81,152 B).
- Fresh disjoint 16-record sample, mean saved B per record (the experiment's 41-record mean in brackets): lenc 24,384 (24,399); sd 2,410 (2,418); lenc_sd 26,800 (26,825); 4096:1024 54,110 (54,119); 8192:2048 81,190 (81,187); tail5 38,651 (38,628); lenc_sd_tail5 65,479 (65,478); lenc_sd 8192:2048 87,184 (87,182); everything 96,852 (96,845). Skeleton overhead 108,255 B (108,266 B).
- Per-layer model: say 'median +188 B, mean +3,197 B driven by one record (+50,479 B)' instead of the bare mean.
- Chunk-size speed: replace 'no loss at batch 8-16' with 'V4/WNR2 data are mixed (b8 +18% slower, b16 8% faster for 4096:2048); not measured for V4.1 or for weight chunk 8192'.

## mxfp4-contexts

Confirmed: yes.

Re-run: All work is in wex4\research\mxfp4-contexts\verify (scripts v1_extract.py, v2_counts.py, v3_eval.py, v4_kmix.py, v6_more.py, patch_decbench.py, decbench_v.c; results in verify\out\v1.json, v3.json, v4.json, v6.json). CPU only, no git operations, about 0.2 GiB read from the container plus the experiment's 0.45 GiB of slab copies.

1. Rebuilt wex_nr3_tools.dll, ctxlib.dll and decbench.exe from source with gcc 16.2.
2. v1_extract.py: read the table and all 7,680 record headers (population figures above). Re-extracted 3 of its records with my DLL: slabs byte-identical to its copies, payload sizes match the manifest. Timed wex_nr3_decode (best of 3) on 11 records: 581-625 ms. Extracted 8 new experts from 8 layers it never read.
3. v2_counts.py + v3_eval.py: independent numpy context walk and my own quantiser, no use of ctxlib. Reproduced to 5 decimals: baseline A/B/C; real 3.59204 vs ideal 3.57809 bit/w, flush 3.5704 B/stream (also 3.5707 on the 8 new experts); srel+rmax, srel only, srel+prev, no-srel, position class, row level, row level+position class, the 945-context model; sign-plane and 14/12/10-bit rows (7 contexts, 8 magnitudes, 12 bit: +0.0126 / +0.0147; new layers +0.0146); srel-only K=2/K=4 with my own Lloyd on per-stream histograms, 3 starts (A -0.0227, B -0.0174 vs today). On the new layers: baseline 3.57435, srel+rmax +0.0004, srel only +0.0158.
4. v6_more.py: independent numpy check of nibble above (+0.0019 / +0.0022 / +0.0076), magnitude two back (+0.0008 / +0.0010 / +0.0058), exact position (-0.0145 / -0.0142 / +0.0070), two-pass block max (+0.0996 / +0.0981 / +0.1067, side 0.0329 bit/w): all identical to the report. Per-layer models: -0.0196 / +0.0058, identical.
5. v4_kmix.py (its C walker, my driver): split A reproduces 945-context K=2 -0.0416, K=4 -0.0509, and 567-context K=2 -0.0335, K=4 -0.0443; then the new-layer split (numbers in corrected).
6. decbench.exe twice on both experts: sizes identical to the report (A 15,898,628 / F 15,910,121 / Fw 15,901,579 and 15,956,946 / 15,969,878 / 15,961,365 B); timings in the same range. decbench_v.c adds reference-style checks, the hash folded into x0, and ceil-inverse and hash timings; all checked decoders exact, status 0. -O3 -march=haswell changed nothing.

Issues found:
- Fw size claim depends on dropping the stream hash. decbench.c starts every stream at x0 = L with no hash; WNR3 folds a 23-bit hash into x0. With that hash folded in (my decbench_v.c, hash mode 1), Fw becomes larger than F: L16_E066 A 15,899,812 / F 15,911,345 / Fw 15,913,631 B; L39_E018 A 15,958,163 / F 15,971,118 / Fw 15,973,279 B. So Fw is +0.074% / +0.080% of raw vs A, not +0.016% / +0.024%. Fw keeps +0.016% only with no hash or a hash cut to 16 bits (measured: 15,902,801 B on L16_E066). F vs A is unchanged by the hash (+0.061% / +0.069%).
- K-table-set gains were measured only on splits A/B, where test experts share layers with training. On 8 new experts from 8 layers the experiment never read (2, 8, 13, 19, 25, 30, 36, 38; model trained on its 24), K=4 gives: today's 567 contexts -0.028% of raw (report: -0.044 / -0.039), combined best 945 contexts -0.033% (report: -0.051 / -0.048), srel-only -0.007% vs today (report: -0.023 / -0.017). Production trains on all 7,680 records, so the real figure lies between; the 'no useful gain' conclusion gets stronger, but recommendation (c) 'K=4 srel-only recovers the cost' is not shown for unseen layers (-0.007% against a +0.016% cost).
- Row level 'helps across layers (split C -0.040%)' is not representative. On the 8 new layers it is -0.0044% (24-expert model) and the 945-context single-set model is -0.0092% (report's C: -0.0494%). Split C's three test layers include layer 39, the population outlier (3.6059 bit/w real against a 3.5897 mean).
- Speed framing: '3.5-4.5x' is against the scalar binary-search reference. The part that needs the format change is F against the AVX2-search decoder on today's streams: 1.75-2.0x in my reruns (200-216 ms vs 105-120 ms per record). The AVX2 search alone, with no format change, gave 2.2-2.7x in my runs (report: 2.1-2.3x).
- Decode timings exclude work a full decoder does, but this does not change the picture. Adding the reference's per-stream checks (bounds on every renormalisation byte, nibble-8 reject, end of stream, final state equal to the hashed x0) moved no decoder by more than about 5%. Fixed extra per record: scalar ceil inverse 26 ms, stream hashes 3.4 ms. Streams plus that fixed work gives about 3.5x for F vs the reference search, instead of about 4.2x on streams alone. The real reference (wex_nr3_decode from the DLL, full record) took 581-625 ms per record on this machine, against 465-545 ms for decbench's 'reference search'.
- The sample is slightly heavier than the population. Reading all 7,680 record headers: population weight payload 3.58969 bit/w, mean record 16,055,413 B = 0.85398 of raw. The 24-expert sample has 3.59204 bit/w and 16,067,755 B = 0.85464 (+0.055% of raw on weights). The report's 'mean record 0.85464 of raw' is the sample, not the container. Deltas are unaffected.
- 'The rANS flush alone is 0.328% of raw' understates the per-stream overhead. weight_payload_bytes excludes the u16 length per stream; flush 3.57 B + length 2 B = 5.57 B per weight stream = 96.2 KB per record = 0.512% of raw. Checked on a real record: 32 + 2 x 19,440 + scale payload + weight payload, padded to 4 KiB, equals the table's total_bytes.
- Model F's byte count assumes the sign bits need no length field of their own. decbench.c keeps them in a separate buffer with a per-stream offset table that is not counted. If a format needed a separate u16 per stream that would be +34,560 B = +0.18% of raw; placing the sign bits inside the stream's existing length avoids it. The report says F is 'not a designed format' but gives no figure.
- Minor: K=3 is charged log2(3) = 1.585 bits per stream; a plain 2-bit selector costs +0.0048% of raw more (K=3 becomes about -0.037 / -0.031). A, F and Fw sizes are in-sample on the slab for all three (disclosed).
- My own slip: importing the experiment's ctxeval.py from my driver created __pycache__/ctxeval.cpython-314.pyc in its folder, outside verify/. I deleted that file and the empty directory at once and ran everything after with -B; the folder listing is back to its original 9 entries plus verify/.
- Not checked by me: GPU decode of any variant, the i7-7700HQ laptop, multi-thread CPU decode, scale streams. Rows checked only through the experiment's own C walker (not my independent numpy walk): neighbour scales other than row level, zero counts, the zero-count side symbols, previous block max, no-reset previous magnitude, and the K-set search on the 567 and 945 context models.

Corrected numbers:

Headline context results stand unchanged: baseline held-out A 3.57579 / B 3.57686 / C 3.58286 bit/w; every single-table variant I recomputed is within +-0.02% of raw on A/B; srel+running max (63 contexts) +0.0002 / +0.0005 / -0.0015; srel only +0.0146 / +0.0161 / -0.0012; sign 1.00001 bit per non-zero.

Corrections:
- Fw real bytes with WNR3's 23-bit stream hash kept: +0.074% (L16_E066) / +0.080% (L39_E018) of raw vs A, i.e. worse than F (+0.061% / +0.069%). The reported +0.016% / +0.024% holds only with no hash or a 16-bit hash.
- K=4 table sets on 8 experts from unseen layers: 567 contexts -0.028%; combined best (945 contexts) -0.033%, K=2 -0.027%; srel-only K=4 -0.007% vs today. Reported same-layer figures (-0.044 / -0.039, -0.051 / -0.048, -0.023 / -0.017) reproduce but are the optimistic end.
- Row level on unseen layers: -0.004% (not -0.040%); 945-context single set -0.009% (not -0.049%).
- Decode speed, one thread, Ryzen 9 7950X3D, weight streams of one record: reference search 449-545 ms, AVX2 search 201-216 ms, F 101-120 ms, Fw 110-127 ms. F vs reference 3.9-4.8x; F vs AVX2-search A 1.75-2.0x; AVX2 search vs reference 2.2-2.7x. Fixed extra per record: ceil inverse 26 ms, hashes 3.4 ms. Real reference DLL, full record: 581-625 ms.
- Population (all 7,680 records): 3.58969 bit/w weight payload, mean record 16,055,413 B = 0.85398 of raw (sample: 3.59204, 0.85464).
- Per-stream overhead in the record: 5.57 B (flush 3.57 + u16 length 2) = 0.512% of raw, not 0.328%.
