# WNR3 draft specification (DRAFT 0.2.1, 2026-11-07, team4 / editor)

**Revises** DRAFT 0.2 (`<RESEARCH-PC>\team4\spec-update\WNR3-DRAFT-v2.md`, team4 / spec-update, unmodified), which revised team 3's DRAFT 0.1 (`<RESEARCH-PC>\team3\wnr3-spec\WNR3-DRAFT.md`, unmodified; byte copy next to this file as `WNR3-DRAFT.md`, sha256 086C4B56...844BED0). DRAFT 0.2.1 applies the errata of team 4's critic (`<RESEARCH-PC>\team4\critic\critic_result.json`) and, in a second pass, the verifiers' corrections to the first 0.2.1 draft (listed at the end of section 0.3); every change from DRAFT 0.2 is listed in section 0.3, every change from DRAFT 0.1 in section 0.2. Everything the critic confirmed is kept, often verbatim.

**Status:** paper design. Nothing here is implemented in KAIRA C/CUDA. **All real-record numbers come from one code lineage**: team 3's NumPy reference `wnr_common.py` (team 3 `ref-impl`), of which team 4 `ceil-ref`'s `wnr_common.py` is a self-declared extended copy ("team4 / ceil-ref copy of team3/ref-impl/wnr_common.py" [C `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py:3`]; 740 of its 934 lines appear verbatim in team 3's 804-line file [I critic line-set comparison]); the refuter's real-coder runs used a further copy (`wnr_common_t3copy.py` [C `<RESEARCH-PC>\team4\ceil-refuter\e2e_real.py:1, 16`]). The ceil-ref decoder is handed the encoder's in-memory tables, not a re-parsed model blob [C `<RESEARCH-PC>\team4\ceil-ref\run_ceil.py:107`]. One CuPy/NVRTC GPU prototype (team 2 `gpu-proto`, extended by team 4 `ceil-gpu`) decodes its own WNR2-style streams (u16 lengths, x0 = L, 4-B flush), not ceil-ref records. So byte-exactness proves that this encoder/decoder pair is lossless; it does not prove conformance to WNR2/KAIRA and cannot catch symmetric encoder/decoder bugs (test T13). The KAIRA sources (`wex_format.h`, `wex_neural_rans.h`, `wex_cuda.cuh`, `wex_cuda_bridge.cu`, engine files) are NOT on this PC: every rule that depends on them is marked **OPEN-K** and collected in section 13. Section 0.4 lists which normative rules have actually run in any codec.

**Evidence tags.** [M] measured on real data (source file given); [R] read in `D:\Desktop\WNR2-REFERENCE.md` (line given as REF:n); [C] read in code on this PC (path:line); [S] external source; [I] inferred or arithmetic (never a measurement). **[I, untested]** marks a normative rule or size that no codec on this PC has implemented or run. **[Sim CU ...]** marks an output of capacity-update's Monte Carlo value model (a simulation with assumed parameters, never a measurement; file and line given). Measured CR byte deltas that the model takes as input keep their [M CR] tag.

**Source abbreviations** (all paths on this PC):

| tag | what | where |
|---|---|---|
| CR | real rANS records, held-out: models from the 80 even-indexed sample experts, test on 40 odd-indexed experts (one per layer, team 3's protocol: 4l+1 for even l, 4l+3 for odd l), 18 configs x 40 records, all decoded from bytes alone (by the same code lineage that encoded them) | `<RESEARCH-PC>\team4\ceil-ref\results\heldout_summary.json`, `heldout_summary_g0.json` (constant anchor), `test_ceil.json` (fallback tests), `heldout_A.jsonl`; code `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py` |
| CG | GPU prototype, RTX 3070 sm_86, NVRTC, 8 experts [17,0,40,80,120,159,60,100], in-sample model; WNR2-style streams; per-record bytes exclude the record header and the 4096 padding | `<RESEARCH-PC>\team4\ceil-gpu\out\bench_final.json`, `best_final.txt`, `ratios_final.json`; `...\ceil-gpu\enc\encode_report*.json`; kernel `...\ceil-gpu\src\wnr2dec.cu`; harness `...\ceil-gpu\src\ceilbench.py` |
| RF | refutation study, 236 records (160 sample + 70 extra routed + 6 MTP); estimates unless stated | `<RESEARCH-PC>\team4\ceil-refuter\out\` (`refute_an.json`, `anchor_an.json`, `per_expert_an.json`, `edge_tests2.json`, `flip_ceil_big.json`, `signsym.json`, `e2e_real.json`); code `<RESEARCH-PC>\team4\ceil-refuter\` |
| CO | origin / generality study, 9 checkpoints | `<RESEARCH-PC>\team4\ceil-origin\` (`summary.json`, `upstream_v41_1017.json`, `other_*.json`, `sim_double.json`, `transfer_ds.json`) |
| CU | team 4 capacity-update value model (team 3's Monte Carlo, re-run with CR's real record deltas applied to production WNR2 16,914,680 B) | `<RESEARCH-PC>\team4\capacity-update\capacity_update_output.txt`, `capacity_update_results.json`, `capacity_update.py` |
| CRIT | team 4 critic (errata, consistency checks) | `<RESEARCH-PC>\team4\critic\critic_result.json`, `consistency_out.txt`, `dump_cr.txt` |
| MV | earlier measurements on the 160-expert sample | `...\research\measured-v41.md` (same folder as `wnr2-brief.md`) |
| RIC | consolidated research report (team summaries; line numbers of the 2026-11-07 version that already includes team 4) | `...\research\WNR-RICERCA-2026-11-07.md` |
| T2A | team 2 corruption audit | `<RESEARCH-PC>\team2\audit\corrupt_audit.json`, `symperm_audit.json` |
| T2Z | team 2 canonical-zero kernel tests | `<RESEARCH-PC>\team2\zero-safety\out\run_full4_quick160.json`, `gpu_all160.json`, `mma_signedzero.json`; `<RESEARCH-PC>\team2\zero-refuter\out\zero_probe.json`, `gpu_probe.log`; kernel source map `<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c:4-19` |
| T2G | team 2 GPU prototype table | `<RESEARCH-PC>\team2\gpu-proto\out\final_table.txt`, `signent.json` |
| T3S | team 3 expert statistics (srel anchor) | `<RESEARCH-PC>\team3\expert-stats\out\anchor_clusters.json` |
| ANS | team 2 start-state experiments | `<RESEARCH-PC>\team2\ans-prior-art\flushx.json` |

All record sizes are for one V4.1 record: raw 18,800,640 B = 17,694,720 B FP4 (35,389,440 weights) + 1,105,920 B scales [R brief:7]; 17,280 weight streams (2048 weights) + 2,160 scale streams (512 scales) at the 2048:512 chunks [M CG bench_final.json: 138,240 weight and 17,280 scale streams per 8 records]. "% of raw" always means percent of 18,800,640 B; "pp of raw" is a difference of two such percentages.

---

## 0. Summary

### 0.1 Decisions

| id | feature | record flag | decision | measured effect per V4.1 record (CR, 40 held-out experts, real rANS bytes) | decoder cost |
|---|---|---|---|---|---|
| f | **ceil-form** (even blocks halved, scale + 1; inverse in the decoder) | CEIL (bit 4) + `ceil_mask` | **ADOPT** (new, first-class) | f alone vs WNR2: -331,162 B = **-1.761% of raw** (CI -1.948..-1.632), ratio 0.89943 -> 0.88182 [M CR]. About 0.24-0.29 pp of this is the srel anchor fix, which plain WNR2 records could carry without a decoder change (WNR2-g0, 5.3) [M RF anchor_an.json, estimates]; the format change's own share is therefore about -1.47..-1.52% of raw on CR's 40 experts [I: -1.761 + 0.24..0.29], vs -1.59..-1.60% in RF's direct estimate with fixed anchors on both forms on RF's own sets [M RF anchor_an.json ceil_best_vs_orig_best_anchor_base] (different expert sets and estimate vs real bytes; 5.2) | nibble LUT on blocks with coded max 5: x0.995-0.999 (free) [M CG]; deferred scale write: +0.3% (8 records) .. +1.7% (1 matrix) median own-cost [M CG]; 0 B shared memory, 0-7 registers [M CG]; sm_86 only, CEIL only (section 7) |
| g | **fixed srel anchor** (constant per set and matrix type, stored in the record's anchor byte) | none | **ADOPT** as rule g0 (constant), with a builder anchor check (5.3) | f+g0 vs f: -6,028 B (CI -16,967..-459) [M CR]; f+g0 = 0.88152 (-1.791%) [M CR g0]. The whole g0-vs-densest-pair difference is one near-tie matrix (n = 1, 5.3), and g0 was chosen after the test results were known. The densest-pair rule of DRAFT 0.1 is a wash on ceil-form (-853 B, CI -16,952..+15,062) [M CR] | none (decoder unchanged, reads the byte) |
| g' | **WNR2-g0 baseline** (plain WNR2 record, constant anchor 120 written into `median[m]`, retrained WNM2 model, no decoder change) | none (codec 4 = WNR2) | **MEASURE FIRST** (rollout step R0); the reference against which WNR3's value is stated | never encoded; estimated -0.24..-0.29 pp of raw vs WNR2 [I from M RF anchor_an.json; T3S -0.25% of bit/w], subject to OPEN-K item 10 | none, if no KAIRA code re-derives `median[m]` (OPEN-K) |
| d | coded stream-length table | LENC (bit 2) | **KEEP** | -25,791 B (-0.137%) with the ref-impl layout [M CR fgd vs fg, densest-pair summary `heldout_summary.json`; -25,764 B in the g0 summary, CRIT dump_cr.txt:53, 327]; about -23.4 KB / -23.3 KB (-0.124%) with this spec's GPU layout (+2,436 B group base offsets) on the two bases [I, untested] | as DRAFT 0.1 (warp scan, no extra kernel) [I, not run on any GPU] |
| e | per-stream 23-bit hash in x0 | HASH (bit 3) | **ADOPT as v1 default** (mutually exclusive with TAIL) | +1,352 B (+0.0072%) [M CR fge vs fg, both summaries]; sign-flip hole closed: 104/104 detected, 0/9,000 accepted [M team 2 audit, MV; RIC:182]. These ran with other 23-bit hashes (CR/RF: FNV-1a + lowbias32 over symbols; team 2: CRC32-based); the normative H32-over-ident-and-coded-bytes rule (5.1) never ran in a codec [I, untested; bytes transfer, detection assumed] | per-lane H32, as DRAFT 0.1 [I, GPU cost not measured] |
| t | raw **tail seeding** in the rANS start state + 32-bit **record digest** | TAIL (bit 5) + DIGEST (bit 7) | **OPTIONAL profile, gated** (section 5.8): allowed only after (i) TAIL and DIGEST are implemented in the CPU encoder/decoder, the GPU scale, weight and commit kernels and the tools, with T7 passing for the TAIL profile, and (ii) team 2's corruption audit (cases A-J) passes on ceil-form records | tail: -40,199 B (-0.214%) vs no start-state use [M CR fgt vs fg, densest-pair summary; g0: -40,188 B]; tail instead of hash: -41,548 B (**-0.221% of raw**) [M CR fgdt vs fgde, used bytes, densest-pair summary; g0: -41,537 B used, -41,677 B total, CRIT dump_cr.txt:128, 402], worth +0.16% (U:) / +0.25% (E:) tok/s [Sim CU capacity_update_output.txt:41, 60], below the +-0.85% A/B resolution [RIC:251]; digest 0 B (reserved header word). Integrity side: **not measured** (DIGEST implemented nowhere; TAIL-alone acceptance never measured) | tail: extract 5 nibbles / 2 bytes from the final state; digest: per-lane H32 + one atomicXor per warp + one compare in the commit kernel [I, not measured] |
| c | canonical zero (code 8 -> 0, value-exact) | CZ (bit 1) + header flag 64 | **KEEP, separate decision** (needs the 5090 bit test + per-step logits memcmp, section 5.6) | -514,848 B (**-2.738%**) on top of f+g [M CR fgc vs fg, densest-pair summary; g0: -514,845 B]; f+g0+c+d+e = 0.85282 (-4.661%) [M CR g0] | one nibble test per output word |
| s | sign bypass (8-symbol magnitude CDFs + raw sign plane) | SIGNPLANE (bit 6), model layout 2 | **OPTIONAL decode-speed layout** (not the v1 default) | bytes +1.03e-6 bit/w [M T2G signent.json]; CFD vs CF +120 B/record [M CG] | smaller table (4,160 vs 8,304 B smem); best-vs-best kernel time -6.5..-11.4% vs A259, -4.3..-9.1% vs CF5 [M CG]; over launch configs the D/A259 median is 0.856..1.120 [M CG ratios_final.json] (5.9) |
| a | block-max side symbol (3-valued on ceil-form) + half | (bit 0 reserved, must be 0) | **NOT ADOPTED** -> Appendix A.1 | +10,642 B (+0.057%), worse on 40/40 experts [M CR fga vs fg] | table 1,211-3,402 rows [I] |
| b | K table sets chosen by k-means | (none) | **NOT ADOPTED as a compression feature** -> Appendix A.2; the set mechanism is kept (K <= 16, v1 writes K = 2) only to hold the plain-form fallback set (and later MTP sets) | K=2 on top of a: -2,087 B; a+K2 on c+d+e: +8,585 B [M CR] | none per symbol |

**v1 default profile (byte-exact): CEIL | LENC | HASH (flags 0x1C) with anchor rule g0.** Measured equivalent fg0de = 16,548,556.8 B = **0.88021 of raw, -1.922% vs WNR2** (CI -2.114..-1.773), 40/40 byte-exact [M CR g0, `heldout_summary_g0.json`; CRIT dump_cr.txt]; with this spec's layout (64-B header, group base offsets) about 16,551,009 B = 0.88034 (-1.909%) [I, untested]. Capacity vs WEX1 +13.61% instead of +11.18% (WNR2 held-out 0.89943); container (7,680 records) 127.1 GB instead of 129.9 GB (held-out mean basis), 127.130 GB vs 129.905 GB anchored on production WNR2 [I; CU]. Against the not-yet-encoded WNR2-g0 baseline the v1 default is worth about -1.63..-1.69% of raw [I: -1.922 + 0.235..0.290].

**Optional profile TAIL + DIGEST: CEIL | LENC | TAIL | DIGEST (flags 0xB4), gated (5.8).** Measured equivalent fg0dt = 16,506,880.0 B = 0.87800 of raw, -2.144% vs WNR2 (CI -2.334..-1.992), 40/40 byte-exact [M CR g0]; spec layout about 16,509,332 B = 0.87813 (-2.131%) [I, untested]; 41,677 B (0.222% of raw) smaller than the default per record [I: 16,548,556.8 - 16,506,880.0, g0 mean totals incl. padding; the paired used-byte delta is -41,537 B, CI -41,588..-41,495, M CR g0, CRIT dump_cr.txt:402-410]; about -1.85..-1.91% of raw vs WNR2-g0 [I].

**v1 value-exact profile: v1 + CZ (flags 0x1E)**, only after section 5.6's tests. Measured: fg0cde 0.85282 (-4.661%, CI -4.855..-4.512), 40/40 value-exact [M CR g0]; spec layout about 0.85295 [I, untested]. With the optional tail (flags 0xB6): fg0cdt was not run; estimated 15,994,407 B = 0.85074 (-4.870%) in spec layout [I: fg0cd - 40,199 B tail + 2,452 B layout; tail saving measured only without CZ].

### 0.2 Changes from DRAFT 0.1 (as introduced by DRAFT 0.2; corrections of 0.2.1 noted)

1. New feature **f (ceil-form)**: exact encoder rule, per-matrix gate, per-matrix fallback to a plain-form table set, decoder inverse and its position in the GPU kernel (sections 5.2, 6, 7.2).
2. New feature **g (anchor)**: the record byte `median[m]` becomes `anchor[m]`; the v1 encoder writes a constant per (set, matrix type) stored in the model (rule g0); decoders are unchanged (section 5.3).
3. **a and b re-decided** on top of f+g: both cost bytes or gain nothing, a also needs a 4-13x larger table: moved to Appendix A. Model layout 1 (BM7H) is reserved and rejected; record flag bit 0 is reserved.
4. **Start-state bits**: TAIL + DIGEST defined as a mutually exclusive alternative to HASH (section 5.8). (DRAFT 0.2 made TAIL + DIGEST the default; 0.2.1 makes HASH the default and TAIL + DIGEST a gated optional profile, section 0.3 E1.)
5. **Hash domain**: every hash/digest covers the CODED domain (the symbols the streams carry, before the ceil inverse), so the scale kernel can hash without knowing which blocks were transformed (section 5.1).
6. **Sign bypass** added as optional model layout 2 + SIGNPLANE record flag (section 5.9).
7. WNM3 tables are per matrix type inside each set (as WNR2 and as measured), not pooled across types as DRAFT 0.1 assumed. Every set carries its own scale CDFs, a `form` byte (plain or ceil) and `anchor0[3]`.
8. LENC predictor replaced by the one measured on real ceil-form records (25 scale bins per type, per-record fitted slope), still with DRAFT 0.1's GPU-friendly per-group base offsets.
9. Record header re-laid (64 B kept): `ceil_mask`, `record_digest`; owner fields moved by 2 B.
10. Expected sizes replaced by real-record measurements (section 10); canonical-zero safety evidence, tooling breakages and the required 5090 tests written out (section 5.6); new test plan and ordered rollout (sections 11-12).

### 0.3 Changes from DRAFT 0.2

Each item is a critic finding (`critic_result.json`); E-numbers follow the errata list, C-numbers are further critic findings.

| id | change | applied in |
|---|---|---|
| E1 | **Integrity default.** HASH (feature e, measured: +1,352 B/record, 104/104 sign flips caught) is the v1 default profile CEIL\|LENC\|HASH (0x1C; 0x0C without a gated matrix; 0x1E / 0x0E with CZ). TAIL + DIGEST (0xB4 family) becomes an optional profile gated on (i) a DIGEST implementation in the CPU decoder, the GPU commit kernel and the tools and (ii) a passing team-2 corruption audit (cases A-J) on ceil-form records. Reasons written out: DIGEST is implemented nowhere, TAIL-alone acceptance was never measured, a corruption ending in the tail bits passes the 2^20 window, and with ceil-form one changed tail magnitude can flip the coded max==5 decision for a whole 32-weight block [I]. Cost of the change: tail saves ~41.5 KB/record = -0.221% of raw, worth about +0.16% (U:) / +0.25% (E:) tok/s, below the +-0.85% A/B resolution | 0.1 (rows e, t; profile lines), 0.2 item 4, 0.4, 2 (entry flags example), 4.2 (fields, flag constraints, builder flags), 5.8 (rewritten), 5.10, 6, 7.1, 7.2 items 6-7, 7.4, 8, 9, 10, 11 (T1, T2, T3, T9), 12 (R1, R2b, R4), 13.2, A.4 |
| E2 | **tok/s figures.** DRAFT 0.2 multiplied the elasticity by pp of raw instead of % fewer bytes vs WNR2 (+1.4/+2.3% for v1, +3.2/+5.3% for v1z) and garbled the assumption. Replaced by CU's simulated medians and 5-95% ranges for every profile defined here; the assumption now reads: the model assumes G2 and W2 are built; neither exists yet. Cited to CU, not to team 3 | 10 (new table 10.3), 12 R5 |
| E3 | **Fallback penalty.** Two numbers with their conditions: +397,312 B = +2.11% of raw (CR, anchor 120 in both forms) vs +11.7..+12.7% of raw (RF estimate, median anchors misaligned: ceil model at 121, plain record at 120). New normative builder/encoder check that `anchor[m]` equals the selected set's `anchor0`, plus a build-report line. With plain data on ceil tables the anchor misalignment added ~10% of raw [I: RF estimate minus CR measurement]; in the same form a one-step mismatch on one matrix cost 1.09-1.16% of that record's raw size [M CR] (second pass) | 3.4, 4.2, 5.2 (fallback), 5.3 (anchor check), 5.10 steps 3 and 7, 8 (rows g, b), 11 (T1, T10), 13.2 |
| E4 | **Codec independence.** "Two independent NumPy reference codecs" removed: all real-record numbers come from one code lineage. Added test T13 (cross-decode of ceil-ref/WNR3 records through an independently written decoder, ideally the GPU prototype or a C decoder) and the OPEN-K byte comparison of the Python WNR2 encoder with KAIRA's C encoder | Status line, 0.4, 11 (T13), 12 R1, 13.1 item 11 |
| E5 | **Decoder-compatible baseline.** Added WNR2-g0 (plain form, fixed anchor, retrained model, no decoder change) as baseline and rollout step R0 (DRAFT 0.2.1 first pass: R0b; merged into R0 in the second pass); WNR3's incremental value vs WNR2-g0 stated as [I] (about -1.6..-1.7% of raw for the HASH default, -1.85..-1.91% for TAIL; the brief's "~-1.9%" referred to the TAIL profile of DRAFT 0.2); the transfer result (+4.19% -> +0.30% of weight bits) re-credited to anchor alignment, not to ceil-form | 0.1 (rows f, g'; profile lines), 5.2 (attribution, transfer), 5.3, 10, 11 (T14), 12 (R0), 13.1 item 10 |
| E6 | **Never-run normative rules** tagged [I, untested]: per-matrix gate "no max-5 block", max 0..4 blocks allowed, skip-255 with coded 255 legal, per-matrix `ceil_mask`/`set_id`, WNM3 K=2 259-row blob, 64-B header, LENC base offsets, DIGEST; fg0cdt not run. ceil-ref's gate is record-level and stricter. T1, T3 and T9 (and T13) are blocking before R1 counts as done (second pass: refined to the [gates R1] parts T1 R1 sets, T3 HASH part, T9 v1 sizes, T13 (b), plus T2, T5, T6, T8) | 0.4 (new), 3.1-3.5, 4.1-4.3, 5.2, 5.8, 11, 12 R1 |
| E7 | **GPU evidence scope.** GPU evidence covers only CEIL (WNR2-style streams: u16 lengths, x0 = L, 4-B flush; no TAIL/LENC/HASH/DIGEST/CZ on the GPU; nothing on the 5090). "352 checks of 138,240 weight streams each" corrected (224 full checks; 128 are m1/m3 rechecks covering 5,760 / 17,280 streams). CF5 patch-mode timing excludes the output-scale pre-fill (CF6 full-write includes it and costs about the same). PIPE timings used the in-place variant that 7.2 item 4 forbids. "Same launch config D/A259 0.708" replaced by the over-config medians 0.856 (m1), 1.069 (m3), 1.120 (m24); sign-plane speed claims hold only best-vs-best | 0.1 (rows f, s), 0.4, 5.9, 7.1, 7.2, 7.5 |
| E8 | **g0 evidence is n = 1**: one near-tie test matrix (L39_E156 w2, pair sums 349,985 vs 350,076), and g0 was chosen after the test results; robustness of a stored constant anchor on other checkpoints and on MTP (densest-pair anchors 113-114 in w1/w3 of 3 of the 6 MTP experts, 122 in the other 3 [M RF agg_extra.json]) is unmeasured | 0.1 row g, 3.4, 5.3, 13.2 |
| E9 | ceil-gpu's in-sample -2.07% vs held-out -1.761% is confounded with the choice of the 8 experts and anchor flips; its per-record bytes exclude header and 4096 padding: GPU sizes compared only as deltas | Source table (CG), 5.2, 7.5 |
| E10 | The refuter's "P3 (team 3's 40)" relabelled: it is refuter group 1 (manifest index % 4 == 1), which shares 20 of 40 experts with CR's / team 3's test set | 5.2 (attribution), A.8 |
| E11 | Integrity rates stated per model: team 2's model w1 0.258% / w2 0.517%; refuter's model original-form w1 0.263% / w2 0.288%, ceil-form w1 0.342% / w2 0.579%; w1 63 vs 82 of 24,000 not significant (z = 1.58), w2 69 vs 139 significant (z = 4.85); team 2 vs refuter original w2 differ (z = 3.09): rates depend on the model. The mixed "0.26-0.58%" range removed | 5.8 |
| E12 | Minor: CFDB2 vs CF is +0.366% of the record (+0.32 pp of raw); section 10 uses one layout basis per table (ref-impl layout table; spec-layout estimates in a separate table); residual std labels (3.80/7.20 quoted by CR for fgd; fgdt 3.79/7.19); per-expert even share on the 160 sample 22.78-31.88%; "-1.81%" is the all-160 (-1.806%) / odd-80 (-1.846%) estimate, the measured test-40 figure is -1.761% | 4.3, 5.2, 5.9, 10, A.3 |
| C1 | CU prose not imported: CU's tables are cited, its sentences "the real records gain about 1.1 pp more tok/s" (E: is 1.76-1.88 pp, paired medians [Sim CU capacity_update_output.txt:65-66]) and "the worst measured ceil-form decode penalties cost at most 0.4 pp" are not; the decode-cost sensitivity dk = 0.157 is labelled as the CF/CF0 m3 median (the measured worst CF/CF0 is 1.213 at m24, 1.205 at m3), and the CU difference "real WNR3 vs team 3 estimate" is a format change (team 3's WNR3 had no ceil-form), not an estimate error | 10.3 |
| C2 | Configurations not measured and risks not discussed (critic finding 15) added to the open items: real records in the spec layout; fg0cdt; WNR2-g0; DIGEST; TAIL corruption audit; GPU TAIL/LENC/H32/digest/CZ; RTX 5090 and KAIRA's kernel; CPU-side T^-1 in C; maximum record over 7,680 (slot sizing); production-like training on all 7,680; MTP and shared experts; token/logprob check for CZ; TAIL changing which corruption classes pass; single-symbol-to-block damage amplification; the anchor check | 0.4, 13.2 |
| C3 | Section 10 container sizes given on both bases: held-out mean (as DRAFT 0.2) and anchored on production WNR2 16,914,680 B (CU / CRIT) | 10.1 |
| C4 | Numbers the critic re-derived and confirmed (CR configs 0 mismatches of 36 rows, section-10 ratios/capacities/GB, CU ratios and containers, CG bytes and best-config times, HASH cost 1,354 vs 1,351.6 B, tail 2.068 B/stream, WNM3 sizes, wire struct, T2Z/T2A/T3S/T2G figures) kept unchanged (except the layout-2 `table_offset`, second pass) | throughout |

**Second pass (verifier findings on the first 0.2.1 draft, each re-checked against the cited file).**

| change | applied in |
|---|---|
| **Tags**: new tag [Sim CU] for capacity-update Monte Carlo outputs (tok/s, P(>=3%), sensitivities), which were tagged [M CU]; [M] kept only for real-record quantities (e.g. max/mean 1.0113); disk margins re-tagged [I] (container size minus REF:139 free space) | evidence-tag key, 0.1 row t, 0.3 C1, 5.8, 10.3, 12 R5, 13.2 |
| **Anchor check made satisfiable**: the per-matrix equality `anchor[m] == anchor0` stays hard (encoder, CPU verify, `check_container`); the densest-pair difference is report-only per (set, type); the build stops only on a set-level condition (mode != anchor0, or > 1% of a (set, type)'s matrices differ) found by a defined pre-pass, with a recorded override. The old rule ("any matrix differs -> stop") stopped every full V4.1 build (~33 of 23,040 matrices at 1/690 [M RF anchor_an.json]) and flagged the case where 120 was measured better. The equality is a v1 builder rule while no WNM3 per-anchor feature bit is set | 4.2, 5.2, 5.3, 9, 11 T10, 12 R4 |
| **Anchor-mismatch cost split**: ~+10% of raw is inferred (RF estimate minus CR measurement) and holds for plain data on ceil tables; in the same form a one-step mismatch on one matrix cost 1.09-1.16% of that record's raw [M CR] | 0.3 E3, 4.2, 5.2, 5.3, 8, 13.2 |
| **Flag rules**: TAIL <=> DIGEST (DIGEST without TAIL was undefined); SIGNPLANE => (HASH or TAIL); fail-closed rejection also for unimplemented CZ and layout 2; CEIL set iff `ceil_mask != 0`; encoder may code a gated matrix plain only when its even share is < 1%, reported | 4.2, 5.8, 5.9, 5.10, 11 T2 |
| **Hash construction tagged**: the normative H32 over ident \|\| coded bytes never ran in a codec; CR/RF used FNV-1a + lowbias32, team 2 a CRC32-based h23; bytes transfer, detection assumed; T1/T3/T13 must exercise H32 | 0.1 row e, 0.4, 5.1, 5.8, 11 |
| **Decoder text**: `M'` 0..4 legal (porting hazard vs ceil-ref's `ceil_inverse`); final scale = `cs_b` for every non-transformed block; TAIL last-block completion in step 8; per-block order decode -> sign merge -> h_g -> D -> store; mask form defined (output scales pre-filled from `cs` before the weight kernel); untested tags on sections 2, 5.10 and 6 | 2, 5.2, 5.10, 6, 7.2 |
| **Rollout and test gating**: every test part names the step it gates; R0 and R0b merged (OPEN-K items moved to the steps that need them); TAIL + DIGEST step renamed R2b and placed after R2, with GPU scale/weight/commit kernels and T7 for the TAIL profile in its gate; T1 restricted to valid flag/ceil_mask combinations; T3, T9 split; T13 split into (a) CR-format cross-decode, (b) spec-format double decode, (c) KAIRA comparison (OPEN-K); one tool name `test_decode_batch` | 0.1 row t, 5.8, 11, 12 |
| **Numbers**: layout-2 K = 2 `table_offset` 36,864 -> 40,960 (4096 + align4096(32,872)); device tables ~0.55 -> ~0.54 MB; "-1.803%" all-160 -> -1.806% (all-160 two-way) / -1.846% (odd-80); layers 0-36 about -1.7% (means -1.61..-2.10), not -1.65%; MTP anchors 113-114 only in 3 of 6 MTP experts [M RF agg_extra.json]; CFDB2 413 us/record (8 in flight) vs 431.1 us (1 in flight); register delta +8-10 is vs DB2; 2.18% / 2.44% are WNR2/WNR3 - 1 (fewer bytes 2.14% / 2.38%); C1 E: gap 1.76-1.88 pp; densest-pair vs g0 bases labelled in 0.1 rows d/t/c and the TAIL line; 59,708-B blob includes the etab; raw record = 17,694,720 B FP4 + 1,105,920 B scales [R brief:7]; row f share -1.47..-1.52% [I] vs RF -1.59..-1.60% stated with bases; citations added (e2e_real.py:1, 16; team4_results.json:37, 44, 228; RIC:214; refute_count.py:222 as code; 384 MTP experts; flushx.json measured on streams) | 0, 1, 3.4, 3.5, 5.2, 5.3, 5.7, 5.9, 7.5, 9, 10.3, 13.2, A.3, A.5 |

### 0.4 What has run, and where

| item | CPU reference (ceil-ref lineage) | GPU prototype (ceil-gpu, RTX 3070) | status in this spec |
|---|---|---|---|
| ceil transform T / T^-1 | yes, record-level gate "no block with M < 6 and no even block with s >= 254" [C `wnr_common.py:217-265`]; its inverse rejects decoded blocks with max < 5 [C `:236-237`], which a conforming decoder must accept (5.2) | yes, CEIL only, 8 experts, 0 bad of 352 checks (224 full, 128 partial-launch rechecks, 7.2) | [M] for V4.1 data; the spec's per-matrix gate is [I, untested] |
| per-matrix gate "no max-5 block", max 0..4 blocks pass, skip-255 with coded scale 255 legal | no (only exhaustive transform test on 4-nibble blocks, RF edge_tests2.json) | no | [I, untested in a codec] |
| per-matrix `ceil_mask`, `set_id`, plain fallback set | no (fallback was coded with the ceil model) | no | [I, untested] |
| WNM3 blob (K = 2, 259-row index, form/anchor0/etab per set) | no (ceil-ref used a 59,708-B blob: WNM2-style 1,701-row tables, 59,408 B, + 300-B etab [M CR; CRIT dump_cr.txt blob 59708 vs wnr2 59408; team4_results.json:107]) | 259-row compacted tables, team-2 format, not WNM3 | [I, untested] |
| 64-B record header, wire struct `'<8I8B2H5I'` | no (48-B header) | no | [I, untested] |
| LENC predictor (25 bins, fitted alpha) | yes, without group base offsets | no | [M] bytes; base offsets [I, untested] |
| HASH (h23 in x0, coded domain) | yes, but with another hash: FNV-1a seed over 24 LE identity bytes, FNV-1a per symbol value, lowbias32 finaliser, 23-bit mask [C `wnr_common.py:292-323`, used at `:404`] (fge, fgde; RF e2e 17/17 with the same lineage); team 2's audit used CRC32-based h23 [C `team2\audit\audit_resync.py:14-15`, `team2\ans-prior-art\flushx.py:148`] | no | [M] bytes and audit (team 2); the normative H32 over ident \|\| coded bytes (5.1) exists only as function test vectors [C team 3 `wnr3_ref.py:27, 188`]: [I, untested in a codec; bytes transfer, detection assumed]; GPU cost not measured |
| TAIL | yes, with the 2^20 / 2^16 window check [C `wnr_common.py:398-405`] | no | [M] bytes; integrity not measured |
| DIGEST | **no** (implemented nowhere) | no | [I, untested] |
| CZ | yes (fgc...) | no | [M] bytes; kernel-level bit tests [M T2Z] |
| CZ + TAIL (fg0cdt) | not run | no | [I] estimate |
| SIGNPLANE layout 2 | no | yes (CFD5, CFDB2, team-2 variant D streams) | [M] GPU only |
| cross-decode by an independent decoder | no | no | missing (T13) |
| RTX 5090, KAIRA kernel, C/CUDA code | no | no | OPEN-K / not measured |

---

## 1. Scope, terms, constants

- Scope: one record = one expert = 3 matrices m = 0 gate w1, 1 down w2, 2 up w3 (unchanged matrix order, unchanged DECODED slab layout: scales of w1, w2, w3, then packed FP4 weights of w1, w2, w3) [R REF:184].
- rANS is exactly WNR2's [R REF:206-208, 228, 240]: 32-bit state, `L = 2^23`, precision 14 bits (total 16384), byte renormalisation, 4-B big-endian initial state in `[L, 256L)`, every symbol frequency >= 1, encoder `x_max = ((L >> 14) << 8) * freq`. Decoder step: `slot = x & 16383`; symbol s with `cum[s] <= slot < cum[s+1]`; `x = freq[s]*(x >> 14) + slot - cum[s]`; renormalise.
- `align4096(n)`, `pad4(n)`: n rounded up to a multiple of 4096 / 4. Integers little-endian unless stated (initial rANS state big-endian, as WNR2).
- Block = 32 consecutive FP4 nibbles of a row with one UE8M0 scale. `count_m % 32 == 0` is enforced by `wex_entry_raw_sizes` [R REF:184]; streams never cross a matrix [R REF:205]; with Cw a power of 2 >= 512 every weight stream holds whole blocks [R REF:209] (the refuter's counter asserts complete 32-blocks per row on every matrix it reads [C RF `refute_count.py:222`] and ran on all 236 records = 708 matrices [M RF anchor_an.json n_experts 236]).
- Magnitude of a nibble q: `q & 7`; sign: `q >> 3`. FP4 E2M1 magnitudes 0..7 = values 0, 0.5, 1, 1.5, 2, 3, 4, 6.
- **Coded domain**: the scale bytes `cs` and nibbles `cq` that the record's streams (and, with TAIL, the start states; with SIGNPLANE, the sign plane) carry. **Output domain**: the decoded slab. Output = `T_m^-1(coded)` per matrix (identity when ceil is off for m); with CZ the output equals the canonicalised source (section 5.6).
- Chunks `Cw` (512..16384) and `Cs` (256..16384), streams `Ws_m = ceil(count_m / Cw)`, `Ss_m = ceil(scount_m / Cs)`, global stream index g (scale streams of m0..m2, then weight streams of m0..m2), groups of 32 streams per matrix and kind: unchanged from DRAFT 0.1 section 1. V4.1: ngS = 69, ngW = 540.

Constants (DRAFT 0.1 values kept unless marked new):

| name | value | meaning |
|---|---|---|
| WEX_HEADER_NEURAL_RANS3 | 32 | header flag: WNR3 records + a WNM3 model before the table |
| WEX_HEADER_CANONICAL_ZERO | 64 | header flag: some records may be value-exact |
| WEX_CODEC_NEURAL_RANS3 | 5 | entry codec |
| WEX_NR3_RECORD_MAGIC / MODEL_MAGIC / DESCRIPTOR_MAGIC | 0x33524E57 / 0x334D4E57 / 0x33444E57 | 'WNR3' / 'WNM3' / 'WND3' |
| WEX_NR3_DESCRIPTOR_OFFSET | 64 | |
| WEX_NR3_HEADER_BYTES | 64 | |
| WEX_NR3_MAX_MODEL_BYTES | 1,048,576 | |
| WEX_NR3_MAX_SETS | 16 | |
| WEX_NR3_GROUP | 32 | streams per length group |
| WEX_NR3_HASH_SEED | 0x33524E57 | H32 seed |
| WEX_NR3_HASH_MASK | 0x7FFFFF | 23 bits (HASH) |
| WEX_NR3_TAIL_W (new) | 5 | weight-stream tail symbols (4 bits each in layout 0, 3 in layout 2); optional TAIL profile only |
| WEX_NR3_TAIL_S (new) | 2 | scale-stream tail bytes; optional TAIL profile only |
| WEX_NR3_LEN_BINS (new) | 25 | length-predictor scale bins, dbin = clamp(cs - anchor, -12, 12) + 12 |
| WEX_CUDA_FEATURE_NR3 | 16 | helper feature bit |

---

## 2. Container (WEX3 version 3) - unchanged rules

DRAFT 0.1 section 2 stays normative (header page with WND3 descriptor at byte 64, header flags 32 and 64, entry codec 5, record placement at the slab end, DEDUP owner/bind rule, why no new version or magic). Changes [I, untested: no container builder or validator on this PC implements codec 5, the entry-flag mirror or these validation rules]:

- **Header flags**: unchanged. Ceil-form, hash, tail seeding, digest and sign plane are byte-exact record features and need NO header flag. The WNR3 builder writes 37 (SCALE_TILES|DEDUP|NEURAL_RANS3), or 101 with canonical zero. 32 requires version 3 and excludes 8/16; 64 requires 32.
- **Entry flags for codec 5**: bits 0-7 = 5; **bits 8-15 mirror record flag bits 0-7** (DRAFT 0.1: bits 8-11); bits 16-31 = 0. v1 default record flags 0x1C give entry flags 0x1C05. Same caveat as DRAFT 0.1: no code path may read `flags >> 8` as scale_pages for codec 5 (**OPEN-K**, section 13 item 2).
- `wex_validate_entry` for codec 5: DRAFT 0.1 rules 1, 2, 5, 6, plus: bit 8 (reserved) = 0; bit 14 (SIGNPLANE) set <=> WND3 layout == 2; bit 10 (LENC) => WND3 features has LENPRED; the minimum size bound uses the record's minimum stream count as in DRAFT 0.1.
- WND3 `model_summary` = `layout | K << 8 | features << 16` (unchanged meaning; layout now 0 or 2).
- Old readers fail closed exactly as in DRAFT 0.1 section 2.5 (unknown header flag 32, unknown codec 5; `repack_v4_wex.py` accepts v3 flags only up to 15 [R REF:303]); the first point is **OPEN-K** (section 13 item 1).
- Model region: v1 builder K = 2, layout 0, LENPRED -> 57,736 B -> `table_offset` 65,536, the same as WNR2's [R REF:190] (section 3.5) [I, untested].

---

## 3. WNM3 model blob (revised) [I, untested: no codec on this PC writes or parses this blob, section 0.4]

### 3.1 Header (64 B)

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | 0x334D4E57 'WNM3' |
| 4 | u16 | version | 1 (DRAFT 0.1's WNM3 was never built; this layout supersedes it) |
| 6 | u16 | precision | 14 |
| 8 | u16 | layout | 0 BASE16 (16-symbol rows); 2 SIGN8 (8-symbol magnitude rows, records carry a sign plane); 1 reserved (DRAFT 0.1 BM7H, rejected) |
| 10 | u16 | K | number of sets, 1..16 |
| 12 | u16 | full_contexts | 567 |
| 14 | u16 | rows | 259 |
| 16 | u16 | matrices | 3 |
| 18 | u16 | alphabet | 16 (layout 0) or 8 (layout 2) |
| 20 | u32 | features | bit 0 LENPRED; other bits 0 |
| 24 | u32 | index_off | 64 |
| 28 | u32 | sets_off | index_off + pad4(2 * 567) = 1,200 |
| 32 | u32 | set_bytes | pad4(8 + 3 * rows * (alphabet + 1) * 2 + 3 * 257 * 2) |
| 36 | u32 | lenpred_off | LENPRED: sets_off + K * set_bytes; else 0 |
| 40 | u32 | total_bytes | size of the blob |
| 44 | u32 | index_kind | 1 (canonical structural index; only value accepted) |
| 48 | u32[4] | reserved | 0 |

### 3.2 Sections
1. **index** `u16[567]`, full context -> row (0xFFFF unreachable), must equal the canonical layout-0 index of DRAFT 0.1 section 3.3: `row = srel*37 + (start ? 36 : tri(pm, mx))`, 259 rows, SHA-256 of the u16-LE index `e0aebac7d1b9b257737b92d1065bde74243f7f8f617a1619b077f6e6639d0738` [M, team 3 `wnr3_ref.py`]. Ceil-form needs no new rows: coded blocks have max 0..7 like plain blocks, and pm <= mx always holds [I]. Layout 2 uses the same index (contexts use only `q & 7`). The measured CR records used WNM2-style per-context rows (1,701 rows, 572 used for fg [M CR]); the 259-row index is a compaction of the reachable contexts, so code lengths should not change [I, untested].
2. **sets**: K sets at `sets_off + k * set_bytes`, each:

| off in set | type | field | rule |
|---|---|---|---|
| 0 | u8 | form | 0 = plain (tables for untransformed data), 1 = ceil (tables for ceil-form data) |
| 1 | u8 | role | 0 routed, 1 MTP (informational; other values rejected) |
| 2 | u8[3] | anchor0[t] | training srel anchor per matrix type t (section 5.3) |
| 5 | u8[3] | reserved | 0 |
| 8 | u16 | weight CDFs `[3][rows][alphabet+1]` | per matrix type t, per row: `cum[0] = 0 < ... < cum[alphabet] = 16384` |
| 8 + 3*rows*(alphabet+1)*2 | u16 | scale CDFs `[3][257]` | as WNM2 [R REF:232] |

3. **lenpred** (LENPRED only): `u32 etab[K][3][25]`, expected coded bits of one 32-weight block of set k, matrix type t and scale bin dbin, in 1/256 bit, each <= 32 * 14 * 256 = 114,688.

Set selection: record field `set_id[m]` (< K) selects set k for matrix m, and the decoder uses table `sets[k].weights[m]`, `sets[k].scales[m]`, `etab[k][m]`. Only the selected table of one matrix is ever in shared memory (section 7.3).

### 3.3 Validation (`wex_nr3_model_open`)
Reject unless: magic, version 1, precision 14, layout in {0, 2} with alphabet 16 / 8, full_contexts 567, rows 259, 1 <= K <= 16, matrices 3, no unknown feature bits, index_kind 1, every offset and `set_bytes` equal to its formula, `total_bytes` equal to the formula and to the buffer size, padding/reserved zero, index == canonical, every CDF strictly increasing from 0 to 16384, every `form` in {0, 1}, every `role` in {0, 1}, every `etab` <= 114,688.

### 3.4 Training (reference trainer, deterministic)
- Counts use exactly the decode contexts in the coded domain of each set's form, computed with the set's own `anchor0` (the same anchor the encoder will write, 5.3).
- **v1 sets** (builder default when the container's even-block share is >= 1%): set 0 form 1 (ceil), trained on `T(training matrices)` (section 5.2) of every training matrix that passes the gate; set 1 form 0 (plain), trained on the same training matrices untransformed. Below 1% even share (PTQ checkpoints: 0.0034-0.0049% [M CO other_amd_r1.json, other_rh_glm53.json]) the builder writes K = 1, form 0 and never sets CEIL [I policy: the transform then gains ~0, measured -0.0001..-0.0004% of weight bits [M CO]]. The plain set 1 was never trained or used for a fallback record in any codec: CR's fallback test coded the plain record with the ceil model (5.2) [I, untested].
- Quantisation per row: `wex_nr2_quantize` (1 + floor(count*(16384-n)/total), largest remainders, ties to the lower symbol, empty rows near-uniform) [R REF:237].
- `anchor0[t]`: section 5.3. `etab`: round(256 x mean coded bits of a block of (set, type, dbin)) on the training data.
- MTP sets (role 1) only if the container stores MTP records (**open**, section 13.2); their anchor0 must come from MTP data: the densest-pair anchors of w1/w3 are 113-114 in 3 of the 6 MTP experts measured (M00_E001, M00_E094, M02_E126; medians 114-116) and 122 in the other 3 (M01_E024, M01_E039, M02_E079); every w2 is at 120-122 [M RF `ceil-refuter\out\agg_extra.json`]. A single constant per type may therefore not fit MTP (**open**).

### 3.5 Sizes [I, computed from 3.1-3.2]

| layout | K | LENPRED | bytes | table_offset |
|---|---|---|---|---|
| 0 | 1 | yes | 29,468 | 36,864 |
| 0 | 2 (v1) | yes | 57,736 | 65,536 |
| 2 | 2 | yes | 32,872 | 40,960 |

(`table_offset` = 4096 + align4096(bytes); WNM2: 59,408 B, table_offset 65,536 = 4096 + 61,440 [R REF:190]. DRAFT 0.2 gave 36,864 for layout 2, K = 2; corrected to 40,960.) The blob is stored once per container. Model id = FNV-1a 32 over the blob (0 -> 1), registry rules as DRAFT 0.1 section 3.7.

---

## 4. WNR3 record (revised)

### 4.1 Layout [I, untested: CR records used a 48-B header and a length table without base offsets]
`[0, 64)` header; `[64, 64 + LS)` length section; scale payload (SP B); sign plane (SG B, SIGNPLANE only: `SG = sum_m count_m / 8`, 4,423,680 B on V4.1); weight payload (WP B); zero padding. `total_bytes == align4096(64 + LS + SP + SG + WP)`, padding all zero.

### 4.2 Header (64 B) [I, untested]

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | 'WNR3' |
| 4 | u32 | model_id | registered WNM3 id == WND3 model_id |
| 8 | u32 | flags | bit 0 reserved (0); 1 CZ; 2 LENC; 3 HASH; 4 CEIL; 5 TAIL; 6 SIGNPLANE; 7 DIGEST; bits 8-31 = 0; == entry flags bits 8-15 |
| 12 | u32 | LS | length section bytes |
| 16 | u32 | SP | scale payload bytes |
| 20 | u32 | WP | weight payload bytes |
| 24 | u32 | Cw | power of 2, 512..16384 |
| 28 | u32 | Cs | power of 2, 256..16384 |
| 32 | u8[3] | anchor[m] | srel anchor of matrix m (was `median[m]`; section 5.3); v1 encoder: == `sets[set_id[m]].anchor0[m]` |
| 35 | u8[3] | set_id[m] | < K |
| 38 | u8 | ceil_mask | bit m set = matrix m is in ceil form; bits 3-7 = 0 |
| 39 | u8 | reserved | 0 |
| 40 | u16 | owner_layer | == bind layer |
| 42 | u16 | owner_expert | == bind expert |
| 44 | u32 | cz_count | CZ: number of code-8 nibbles mapped to 0 (>= 1); else 0 |
| 48 | u32 | record_digest | DIGEST (optional TAIL profile): section 5.8; else 0 |
| 52 | u32[2] | reserved | 0 |
| 60 | u32 | header_check | `H32(bytes 0..59, seed 0x33524E57)` (DRAFT 0.1) |

Wire struct: `'<8I8B2H5I'` = 64 B.

**Flag constraints** (all checked by every decoder):
- bit 0 == 0.
- CEIL <=> `ceil_mask != 0`; for every m: `((ceil_mask >> m) & 1) == sets[set_id[m]].form`.
- HASH and TAIL never both. TAIL <=> DIGEST (TAIL => DIGEST and DIGEST => TAIL: the digest is defined only in the TAIL profile, 5.8; CEIL|LENC|DIGEST 0x94, CEIL|LENC|HASH|DIGEST 0x9C and SIGNPLANE|DIGEST without TAIL are invalid).
- SIGNPLANE <=> model layout 2; SIGNPLANE => (HASH or TAIL) (TAIL implies DIGEST); SIGNPLANE => not CZ (section 5.9).
- CZ => header flag 64 and `cz_count >= 1`.
- LENC => model LENPRED.
- DIGEST = 0 => `record_digest == 0`.
- **Fail closed on unimplemented features**: a decoder that does not implement TAIL and DIGEST (allowed in v1, section 12 R1) rejects every record with bit 5 or bit 7 set; one that does not implement CZ rejects bit 1; one that does not implement layout 2 rejects a model with layout 2 (and so every SIGNPLANE record). Status -1, destination untouched.

**v1 builder flags (default): 0x1C (CEIL|LENC|HASH) iff `ceil_mask != 0`**, else 0x0C; with CZ 0x1E / 0x0E. **Optional TAIL profile** (only after the gate of 5.8): 0xB4 (CEIL|LENC|TAIL|DIGEST) / 0xA4; with CZ 0xB6 / 0xA6. Which matrices get a ceil-form set is the encoder rule of 5.10 step 2.

**Encoder anchor rule (normative for the v1 builder, new)**: for every m the encoder writes `anchor[m] == sets[set_id[m]].anchor0[m]` (rule g0) and refuses to write a record otherwise; `check_container` and the CPU verify step (5.10 step 7) check the same equality. Decoders do not check it (5.3). This equality is a rule of the v1 builder and of `check_container` that holds while no WNM3 feature bit for per-anchor tables is set (3.1: features bit 0 LENPRED, other bits 0). A later encoder that aligns the anchor per matrix needs such a new feature bit (and tables trained per anchor); with it `check_container` checks `anchor[m]` against the anchors that bit allows. Reason: the inferred penalty of coding data whose anchor is misaligned with the anchor the tables were trained with is large. Plain data on ceil tables with the anchor one step off: ~+10% of raw [I: RF estimate +11.7..+12.7% with misaligned anchors minus CR's measured +2.11% with aligned anchors, 5.2]. Same form, one matrix one step off: 1.09-1.16% of that record's raw size [M CR, 5.3].

### 4.3 Length section
- **LENC = 0**: u16 lengths of the S scale streams then of the W weight streams in g order, zero-padded to a multiple of 4 (WNR2 table).
- **LENC = 1**: `u16 sbase[3]`, `u16 alpha[3]` (12 B), then the scale part, then the weight part, each laid out as in DRAFT 0.1 section 4.3 (`u32 base_off[ng]`, `u8 width[ng]`, pad4, `u32 words[sum(width)]`, residual i of a group at bits `[i*w, i*w+w)` LSB-first, zigzag) [I, untested: the base offsets were never encoded]. Checks as DRAFT 0.1 (width <= 17, `base_off[0] == 0`, group continuity, part totals == SP / WP, unused slots 0, every length in `[4, min(65535, 2n + 8)]` with n = coded symbols of the stream).
- **Prediction** (integer, identical on CPU and GPU; the predictor measured on real records [C `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py:573-599, 771-783`]):
  - scale stream of matrix m: `pred = sbase[m]` (the encoder writes the median scale-stream length of m).
  - weight stream j of matrix m, set k = `set_id[m]`: `Q = sum over the stream's blocks b of etab[k][m][dbin_b]`, `dbin_b = clamp(cs_b - anchor[m], -12, 12) + 12` on the CODED scales; `pred = 4 + ((Q * alpha[m] + 2^25) >> 26)` (u64). The encoder fits `alpha[m]` in 0..65535 to minimise the sum of |residual| (coarse search step 4 within +-64 of the least-squares value, then +-4) [C `wnr_common.py:582-599`].
  - `len = pred + unzigzag(z)`.
- Measured (ref-impl layout: widths + packed residuals, no base offsets): 38,880 -> 13,118.6 B mean on fg0de (13,121.2 B on fg0dt) [M CR g0, CRIT dump_cr.txt]; residual std 3.80 B (scales) / 7.20 B (weights) as quoted by CR for fgd [M CR], 3.79 / 7.19 B recomputed for fgdt [I CRIT]. This spec adds `4 * (ngS + ngW)` = 2,436 B of base offsets (V4.1) so each warp computes its offsets alone (DRAFT 0.1 section 4.3 rationale): about 15,555 B for the v1 default [I, untested].

### 4.4 Size bounds
Unchanged from DRAFT 0.1 for layout 0 without side symbols: a stream of n coded symbols is <= ceil(14n/8) + 4 < 2n + 8 [I].

---

## 5. Codec

### 5.1 Coded domain, hash input, identity
- All integrity data (HASH h23, DIGEST, header_check) is computed on the CODED domain. Reason: the scale kernel decodes scale streams before any weight block is known, so it cannot know which scales will be decremented; hashing coded scales keeps the scale kernel independent of ceil-form [I]. Hashing the coded domain was run end to end on 17 real records (decode exact, inverse byte-exact) [M RF e2e_real.json] (same code lineage, Status).
- Per-stream hash input: 16-B identity then the stream's coded bytes (weight stream: n/2 bytes of coded nibbles, low nibble = even symbol, including tail symbols (TAIL profile) and, with SIGNPLANE, the sign bits as bit 3 of each nibble; scale stream: n coded bytes):
  ```
  ident[0] = model_id
  ident[1] = owner_layer << 16 | owner_expert
  ident[2] = g | ((ceil_mask >> m) & 1) << 31        (g = global stream index)
  ident[3] = (flags & 0xFF) | m << 8 | set_id[m] << 16 | anchor[m] << 24
  h_g = H32(ident || coded_bytes, seed 0x33524E57)   (MurmurHash3_x86_32, DRAFT 0.1 section 5.6)
  ```
  [I, untested in a codec: this whole construction (H32 = MurmurHash3_x86_32, the 16-B ident, the n/2 packed nibble bytes) never ran in any encoder or decoder. CR and RF (the same lineage) used FNV-1a over 24 LE identity bytes, then FNV-1a per symbol value, then the lowbias32 finaliser, masked to 23 bits [C `wnr_common.py:292-323`]; team 2's audit used a CRC32-based h23 [C `team2\audit\audit_resync.py:14-15`]. The byte cost transfers (any uniform 23-bit x0 offset costs the same [I]); the detection results transfer only by inference. T1, T3 and T13 must exercise this H32 rule, with T5's regenerated record-level vectors.]
- H32 and its verification value 0xB0F57EE3 are unchanged (DRAFT 0.1 section 5.6 [S, M]). DRAFT 0.1's stream test vectors remain valid as H32 function vectors; record-level vectors with the new ident/flags must be regenerated (**open**, test T5).

### 5.2 Ceil-form (feature f)

**Observation** [M MV, RF]: in all 260,997,120 blocks checked (160 sample + 70 extra routed experts over all 40 layers + 6 MTP experts) the block max magnitude is 6 or 7, never below; 30.65% of routed blocks (sample 30.647%, extra 30.646%; w1/w2/w3 30.31/31.12/30.51%; per expert 22.78-31.88% on the 160 sample, 26.02-31.66% on the 70 extra) and 22.15% of MTP blocks have max 7 and no magnitude 1 or 3 [M RF refute_an.json, MV]. Under i.i.d. magnitudes the share would be 1.9e-4 [M MV]. The artefact is already in DeepSeek's upstream checkpoint (8 REAP-pruned upstream experts 27.8-31.6%, all 13 upstream items 29.415%, kept experts byte-identical to the sample) and in every QAT-native MXFP4 checkpoint measured (V4-Flash 25.98%, V4-Pro 23.89%, gpt-oss-20b 21.94%, gpt-oss-120b 21.36%, AMD Quark gpt-oss-120b 21.50%), but not in PTQ checkpoints (0.0034-0.0049%) [M CO]. Mechanism (sources read, export path inferred): DeepSeek's `fast_round_scale` and triton_kernels' default ROUND_UP are ceil rules (s = 2^ceil(log2(amax/6))), so a first pass yields block max 3.0/4.0/6.0 x s; re-quantising with ceil or OCP floor moves exactly the max-3.0 blocks to s/2 with every code doubled, value-exact [C/S CO, sim_double.json]. OCP MX v1.0 section 6.3 specifies floor but allows other algorithms [S CO]. So T below rewrites each such block into the form the ceil rule produced: same values, cheaper to code.

**Definitions** for one block (nibbles q_0..q_31, scale s, magnitudes `a_j = q_j & 7`, `M = max a_j`):
- `even` := `M == 7` and no `a_j` in {1, 3} (so every `a_j` is in {0, 2, 4, 5, 6, 7}).
- `H` (halve): 0->0, 2->1, 4->2, 5->3, 6->4, 7->5.  `D` (double): `D(a) = a + min(a, 2)`: 0->0, 1->2, 2->4, 3->5, 4->6, 5->7. `D(H(a)) = a` on {0,2,4,5,6,7}.

**Gate (per matrix m, encoder)** [I, untested in a codec]: matrix m is eligible for ceil form iff **no block of m (after CZ, if CZ is on) has M == 5**; whether an eligible matrix is coded in ceil form is the encoder rule of 5.10 step 2. This is the exact bijection condition of the transform [M RF edge_tests2.json: exhaustive over all 65,536 4-nibble blocks x scales {0, 1, 120, 253, 254, 255}: 0 collisions and 0 round-trip failures over 328,800 pairs without max-5 blocks; 32,208 collisions when max-5 blocks are allowed]; that test exercised the transform alone, not a codec with `ceil_mask` and set selection. Blocks with M in 0..4 (including all-zero blocks) are legal and pass through unchanged [I, untested in a codec]. The only measured codec (ceil-ref lineage) used a stricter **record-level** gate "no block with M < 6 and no even block with s >= 254": a single block with max 0, or an even block with scale 254, cleared F for the whole record [C `wnr_common.py:217-265`; M CR test_ceil.json a/b]. On V4.1 both gates give identical records because 0 blocks have M < 6 and the largest scale of an even block is 124 routed / 125 MTP [M RF].

**Encoder transform T_m** (only if the gate passes for m):
```
for every block b of matrix m:
    if even(b) and s_b <= 254:  q_j' = (q_j & 8) | H(q_j & 7)   for j = 0..31;   s_b' = s_b + 1
    else:                       q_j' = q_j;                      s_b' = s_b
```
An even block with s_b = 255 is left unchanged (it keeps M = 7, so the decoder leaves it alone; this removes the only overflow without a fallback [M RF edge_tests2.json: 0 overflow, 0 collisions]). Coded scale 255 can occur (from s_b = 254) and is a legal symbol; the coded form is never dequantised. [I, untested in a codec: skip-255 and coded 255 were only exercised by the exhaustive transform test; ceil-ref instead falls back at s >= 254.]

**Decoder inverse T_m^-1** (only if bit m of `ceil_mask` is set):
```
for every decoded block b of matrix m (coded nibbles q', coded scale cs_b):
    M' = max(q_j' & 7)
    if M' == 5:  if cs_b == 0: INVALID
                 q_j = (q_j' & 8) | D(q_j' & 7);   s_b = cs_b - 1
    else:        q_j = q_j';  s_b = cs_b
```
`M'` in 0..4 is legal in a gated matrix and passes unchanged. **Porting hazard**: unlike ceil-ref's `ceil_inverse`, which raises "ceil: decoded block max < 5" [C `wnr_common.py:236-237`] (consistent with its record-level gate), a conforming decoder must not reject such blocks.
Why it is exact [I, with the exhaustive check above]: transformed blocks have coded max H(7) = 5; untransformed blocks of a gated matrix have max in {0..4, 6, 7}; so `M' == 5` selects exactly the transformed blocks, and D, `- 1` undo H, `+ 1`. The sign bit is never touched (sign plane unchanged on 708/708 matrices [M RF]). The `cs_b == 0` test can only fire on corrupted data (a transformed scale is >= 1) and closes the SWAR-borrow hazard noted in the prototype (`wnr2dec.cu:208-211` uses a plain u32 subtraction) [C CG]. Note: a single wrong coded magnitude in a block can move M' across 5 (5 -> 6/7 or the reverse), so T^-1 is then skipped or misapplied on the whole block (32 weights x2 or /2, scale +-1) [I]; the per-stream check must catch such a symbol error (5.8).

**Per-matrix fallback** [I, untested: no codec implements per-matrix fallback or a plain set]: a matrix that fails the gate is coded plain: `ceil_mask` bit m = 0 and `set_id[m]` = a set with form 0. Fallback must NOT use the ceil-trained tables. Two measurements, with different anchor setups:
- **aligned anchors** (CR): a plain record (L00_E113 with one block set to 0, in memory) coded with the ceil-trained fg model cost 17,285,120 B vs 16,887,808 B for the same expert in WNR2 = **+397,312 B = +2.11% of raw** (+3.79% of raw vs the same expert in ceil form, 16,572,416 B); the anchor was 120 in both forms [M CR test_ceil.json; heldout_A.jsonl anchors [120,120,120]; CRIT consistency_out.txt section 10].
- **misaligned anchors** (RF estimate): ceil model trained at the ceil-form median anchor 121, plain record at its median 120: **+12.65% (P2) / +11.73% (P4) of raw** [M RF refute_an.json fallback_record_coded_with_ceil_model; anchor_an.json anchor_hist c_med 121 x688].
The roughly 6x gap is the anchor misalignment on top of the form mismatch: plain data on ceil tables with the anchor byte one step off the anchor the tables were trained with cost about +10% of raw more than with the anchor aligned [I: RF estimate minus CR measurement, two different expert sets]. In the same form the cost is much smaller: a one-step mismatch on one matrix cost 204,800-217,350 B = 1.09-1.16% of that record's raw size (5.3) [M CR]. Either way it is silent. Hence the encoder anchor rule of 4.2 and the build-report line below. Expected fallbacks on V4.1: 0 of 7,680 records (0 of 230 routed records checked; 95% upper bound 100 records by the rule of three) [M/I RF]. A record with no gated matrix has CEIL = 0.

**Where the encoder checks**: the build report prints per record `ceil_mask`, `set_id[m]`, number of transformed blocks per matrix, number of max-5 blocks (gate), number of even blocks with scale 255 (skipped), and the **anchor line**: per matrix `anchor[m]`, `anchor0` of the selected set, and the densest-pair value of the matrix's coded scales (5.3); the container report prints the fallback count and the number of matrices whose densest-pair value differs from `anchor0`, by set and type (report-only; the set-level stop condition is in 5.3). The CPU verify of every record (section 11 T10) covers T^-1 and the anchor equality.

**Measured gain** (real records, CR, 40 held-out experts, 0 fallbacks, 40/40 byte-exact):
- f alone: 16,578,764.8 B = 0.88182 (WNR2 16,909,926.4 = 0.89943): **-1.761% of raw**, CI -1.948..-1.632, median expert -1.656%; weights 3.7801 -> 3.7080 bit/w, scales 1.0610 -> 0.9755 bit/scale (both incl. flush) [M CR].
- Real minus code-length estimate: 5.1 B (CI -16.6..+26.2) [M CR]. The often-quoted -1.81% (MV) is not a measurement on these 40 experts: it is the all-160 two-way / odd-80 figure of the same estimate (-1.806% all-160 two-way [CRIT dump_cr.txt:270 quoted_entropy_estimate; M RF refute_an.json P1 C0_ceil_base] / -1.846% odd-80 [M RF anchor_an.json P2 c_med]); the measured test-40 figure is -1.761% [M CR]. Real flush is ~3.5 B/stream, cancels in deltas [M CR, T2A].
- In-sample GPU-prototype records (8 experts [17,0,40,80,120,159,60,100]): 0.90227 -> 0.88155 (-2.07% of raw) [M CG encode_report.json]. This is **not** an in-sample bonus over CR's -1.761%: it is confounded with the expert choice (4 of the 24 original-form matrices have median 121 [M CG encode_report.json medians], vs 43/690 = 6.2% in the routed population [M RF anchor_an.json]; the set includes an L39 expert, and layers 37-39 gain -2.67/-3.56/-3.73% vs about -1.7% elsewhere (layer means of layers 0-36: -1.61..-2.10, mean -1.713, median -1.675) [M RF per_expert_an.json per_layer_mean]) and with anchor flips; CG's per-record bytes also exclude the record header and the 4096 padding. Compare GPU sizes only as deltas.
- Attribution [M RF anchor_an.json, estimates]: about 0.24-0.29 pp of the f-vs-WNR2 gain is the srel-anchor fix that f brings implicitly (the ceil-form median rarely flips: 2/690 matrices vs 43/690 in plain form): anchor fix alone -0.272 (P2, odd 80), -0.235 (P3), -0.290 (P4, 70 extra) pp of raw; team 3 -0.25% of bit/w [M T3S]. With fixed anchors on both forms the ceil gain is -1.592% of raw (P2), -1.601% (P3), -1.592% (P4) [M RF anchor_an.json ceil_best_vs_orig_best_anchor_base_rec_pct_raw, estimates on RF's sets]; 0.1 row f's -1.47..-1.52% is on a different basis (CR's measured -1.761% on its own 40 experts minus the RF anchor-fix range) [I], so the two differ by about 0.1 pp. **P3 is the refuter's group 1 (manifest index % 4 == 1, 40 experts), not CR's / team 3's test set: the two share 20 of 40 experts** [C RF anchor_an.py:7, 136; refute_count.py:5]; that is why refuter P3 with median anchors gives -1.803% while CR's same-model estimate on its own set gives -1.761%. The anchor-fix share on CR's exact set is unmeasured. A non-ceil model with a 3-class side stream reaches 97% of the clean ceil gain, but needs 3,402 rows and a class stream; ceil-form keeps the 259-row table and no extra stream [M RF].
- Transfer (estimate, weight bits): reusing a V4.1-trained model on V4-Flash costs +4.19% of weight bits in plain form and +0.30% in ceil form (3.9063 vs 3.7493 and 3.7214 vs 3.7104 bit/w, mean over 8 experts) [M CO transfer_ds.json]. **This is anchor alignment, not a property of ceil-form**: the plain-form matrix median is 120 on V4.1 and 121 on V4 / V4-Pro, and ceil-form happens to shift V4.1's to 121 (CO's own explanation). Under g0 (constant anchor0 from training) that alignment does not happen automatically; the anchor line of the build report is what detects a misaligned checkpoint.

### 5.3 srel anchor (feature g)
- Plain form: the lower median flips 120 <-> 121 in 27 of 480 sample matrices (5.6%), shifting every srel label of the matrix; fixing the anchor to the lower value of the densest adjacent pair gives 3.78243 -> 3.77290 bit/w held-out (-0.25%) with one model [M T3S anchor_clusters.json] (the consolidated report quotes -0.257% [RIC:214]). Over 690 routed matrices the original-form median is 121 in 43 (6.2%) [M RF anchor_an.json].
- On ceil-form both rules flip once among the 120 test matrices: the median gives 120 instead of 121 on L24_E058 w1 (costs 217,350 B on that record); the densest pair gives 121 on L39_E156 w2, a near tie (pair sums 349,985 vs 350,076, 0.026% apart; costs 204,800-212,992 B by config) [M CR, as reported in `<RESEARCH-PC>\team4\critic\team4_results.json:37, 44, 228`; records in `ceil-ref\results\fixed120_L39.jsonl`, `heldout_g0.jsonl`]. Net: densest pair vs median -853 B, CI -16,952..+15,062 [M CR heldout_summary.json].
- **Rule g0 (v1)**: `anchor[m] = sets[set_id[m]].anchor0[m]`, where the trainer sets `anchor0[t]` = the mode, over the set's training matrices of type t (coded domain of the set's form), of the densest-pair lower value `argmax_v (h[v] + h[v+1])` (first on ties). On V4.1 this is 120 for every type in both forms (ceil: 240/240 training matrices [M CR, output of `anchor_diag.py` as reported in team4_results.json:37]; plain: pair = 120 in 690/690 routed matrices, which include the 480 sample matrices [M RF anchor_an.json o_pair]; T3S anchor_census: pair = median or median - 1 in 480/480 [M T3S]). Measured with 120: fg0 0.88152 (-1.791%, CI -1.992..-1.640), g0 vs f -6,028 B (CI -16,967..-459) [M CR g0].
- **Strength of the g0 evidence: n = 1.** The whole g0-vs-densest-pair difference (~5.2 KB/record on the mean) is the one near-tie matrix L39_E156 w2 [M CR], and g0 was chosen after the test results were known. On V4.1 both rules are almost always 120 (refuter census: c_pair = 121 in 1/690, c_med = 120 in 2/690 routed matrices [M RF anchor_an.json]). Whether a stored constant anchor is robust on other checkpoints (V4 / V4-Pro medians 121 [CO]) and on MTP (densest-pair anchors 113-114 in w1/w3 of 3 of the 6 MTP experts, 122 in the other 3 [M RF agg_extra.json]) is **unmeasured**; since a misaligned anchor costs 1.1-1.2% of raw per matrix in the same form and ~10% with a form mismatch (5.2), that is the real question (section 13.2).
- **Anchor check (normative for the builder, new)** [I policy, untested]:
  1. **Hard, per matrix**: the encoder writes `anchor[m] == anchor0` of the selected set and refuses otherwise (4.2); the CPU verify (5.10 step 7) and `check_container` check the same equality. In the v1 encoder this holds by construction; the check guards against encoder bugs and wrong set selection.
  2. **Report-only, per matrix**: the builder computes the densest-pair value of each matrix's coded scales and counts the matrices where it differs from `anchor0`, per (set, type) (5.2 build-report line). A differing matrix is not an error: the one known case, L39_E156 w2 (densest pair 121), is where the constant 120 was measured cheaper by 204,800-212,992 B [M CR].
  3. **Set-level stop**: before encoding, the builder runs a pre-pass over all source records (contexts are not needed, only the gate and the coded-scale histogram per matrix) and stops, writing nothing, if for any (set, type) the mode of the densest-pair values differs from `anchor0`, or more than 1% of that (set, type)'s matrices differ. The operator may override with an explicit builder option; the override and the counts are recorded in the build report and next to the container's sha256.
  Expected on V4.1: c_pair = 121 in 1/690 routed matrices (0.14%) [M RF anchor_an.json], i.e. about 33 of the 23,040 matrices of a full build [I]: reported, below the 1% threshold, no stop. The threshold is sized on the same-form cost: at 1% of matrices one step off, about 1.1% x 3 x 1% = 0.03% of raw on the container [I].
- Decoders do not check `anchor[m]` against `anchor0` or against the decoded scales: for the decoder the byte stays an encoder choice. A later encoder that aligns per matrix (e.g. argmin of estimated cost over anchor0 +-1; not measured) needs tables trained per anchor and a new WNM3 feature bit, because the v1 builder and `check_container` enforce `anchor[m] == anchor0` while no such bit is set (4.2). WNR2 stores the byte per record and the documented decoders read it [R REF:200, 222-225]; whether any KAIRA decoder or checker re-derives it as the lower median of the decoded scales is **OPEN-K** (section 13.1 item 10): such a check would reject g0 records and WNR2-g0 records.
- **WNR2-g0 baseline (new)**: because WNR2 records already carry `median[m]` and the decoders read it [R REF:200, 225], a plain WNR2 build whose encoder writes the constant 120 and whose WNM2 model is retrained with that anchor needs no decoder change, subject to OPEN-K item 10. It would capture the anchor-fix share of the gain, estimated at -0.24..-0.29 pp of raw (ratio about 0.8965..0.8971 vs 0.89943) [I from M RF anchor_an.json; T3S]; it was never encoded (CU's WNR2 baseline uses median anchors). WNR3's incremental value must be stated against WNR2-g0: about -1.63..-1.69% of raw for the HASH default and -1.85..-1.91% for the TAIL profile [I]. Measuring WNR2-g0 on CR's 40 experts is rollout step R0 and test T14.
- Label offset does not matter on V4.1: the refuter's estimate gives the same 3.7081 bit/w for a fixed ceil anchor and for "pair(original) + 1" (= 121) [M RF]. It matters for MTP (median anchors make MTP layer 0 worse, +0.14% of raw; pair(original)+1 gains -1.25/-1.53/-1.40% on the 3 MTP layers) [M RF]: whether the mode rule is right for MTP is **open**.

### 5.4 Scale streams
Scale stream g of matrix m holds coded scales `cs` `[j*Cs, min((j+1)*Cs, scount_m))` in row-major order, order-0 with `sets[set_id[m]].scales[m]`. With TAIL (optional profile) the last `kt = min(2, n)` coded scales are not rANS-coded (section 5.8).

### 5.5 Weight streams and contexts
Weight stream j of matrix m holds coded nibbles `[j*Cw, min((j+1)*Cw, count_m))`, row-major, symbol 2i = low nibble of byte i [R REF:221]. Layout 0: the coded symbol is the 4-bit nibble; layout 2: the 3-bit magnitude (sign in the sign plane). Context: at block start `srel = clamp(cs_b - anchor[m], -3, 3) + 3`, `pm = mx = 8`; after symbol x: `pm = x & 7`, `mx = max(mx, x & 7)` (8 = none); `row = srel*37 + (start ? 36 : tri(pm, mx))` [R REF:222-224; DRAFT 0.1 3.3]. Contexts use the CODED scale and CODED nibbles. With TAIL (optional profile) the last `kw = min(5, n)` coded symbols of the stream are not rANS-coded and need no context. Per-record checks: CZ => no coded nibble (rANS-decoded or tail) equals 8; ceil checks as in 5.2.

### 5.6 Canonical zero (feature c, value-exact)

**Rule** (DRAFT 0.1 5.4, unchanged): with `--canonical-zero` the encoder maps every code 8 (-0) to 0 (+0) before counting, the gate, T and encoding; `cz_count` = mapped nibbles; CZ only if `cz_count >= 1`. T commutes with it (T keeps magnitude 0 and the sign bit): 0 failures on 708 matrices and on the exhaustive 4-nibble test [M RF]. Output = canonicalised source, equal in value, not in bytes.

**Size** [M CR]: c on top of f+g: -514,848 B (-2.738% of raw, CI -515,456..-514,249); fg0c 0.85411, fg0cd 0.85275, fg0cde 0.85282 (-4.661%, 40/40 value-exact). -0 is 5.8263% and +0 5.8255% of V4.1 weights [M MV] (the brief's "12.8%" is the V4 pilot's share of all zeros, RIC section 9); the sign of a zero is a fair coin worth 0.1165 bit/w.

**Kernel-level argument** (team 2; code read in the colibri dev tree, line ranges listed in `<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c:4-19`; KAIRA's fork NOT checked, **OPEN-K**):
- Inputs: most CPU arms decode code 8 to +0.0f or int8 0 (`deepseek_v4.c:20123-20129` coli_e2m1_decode; `quant.h:1284`, `:1370`, `:1392`), so for them canonicalisation does not even change the decoded operands. The scalar `matmul_mxfp4` arm (`quant.h:1266-1267` mx4_lut), the default CUDA FP32 kernels (`backend_cuda_dsv4.cu:99` f4(), `:2250-2313` v4f_dot_fp4) and the Vulkan shader (`shaders/chain_moe_gv.comp:97-114`) decode -0.0f [C T2Z, RIC section 2].
- Arithmetic [I, IEEE 754 round-to-nearest]: every accumulator starts at +0; adding a zero of either sign to a nonzero value returns it unchanged, and (+0) + (-0) = +0, so a sum never differs between -0 and +0 weight operands; products with finite activations are zeros either way. The only possible difference is the sign of an exactly-zero output when an accumulator is seeded with the first product instead of +0; the negative-control kernel K12 (Vulkan first-term emulation) shows exactly that (256 sign-only mismatches on synthetic zero rows) while every real kernel shows none [M T2Z].
- **Measured, original vs canonicalised records, 0 differing output bits**:
  - CPU kernels copied verbatim (AVX2, scalar, FMA, int8 IDOT, rows16, batch, CUDA and Vulkan emulations): 350,208 outputs per kernel (14 kernels incl. negative control, 4 experts, 9 activation sets incl. all -0, all +0, denormals, inf, NaN, 1e30); 3,112,960 outputs per kernel for 6 kernels over all 160 experts; 4,608 synthetic zero-row outputs per kernel; positive control (one flipped weight) detected in every kernel [M T2Z run_full4_quick160.json].
  - colibri CUDA kernels on the RTX 3070 (sm_86, NVRTC 12.9): 10 kernels (mv_fp4_grouped<1,2,4,8>, mv_fp4_expert_major<4> T4, mv<4>, mv_rows<4,4>, mv_rows<4,16>, mm_batch<4> T4, v4f_dot_fp4): 14,008,320 outputs per kernel (56,033,280 for the 4-token ones), 0 value, 0 sign-only, 0 NaN-payload mismatches, 0 nondeterministic reruns; 184,320 synthetic zero-row outputs per kernel; positive control fires [M T2Z gpu_all160.json]. sm_86 mma f16/bf16 tensor-core tiles: 0 differing D on 20,000 random tiles; all-zero A with random zero signs gives D = +0 in 2,560,000/2,560,000 [M T2Z mma_signedzero.json]; tc_fp16_gemm_bitdiff 0 [M T2Z gpu_probe.log].
  - Data: 0 all-zero 32-blocks and 0 all-zero rows in w1 of the 40 experts probed (14,745,600 blocks); 13 all-zero groups of 8 [M T2Z zero_probe.json].
- **Untested**: the sm_120 tensor-core paths (DeepGEMM sm120 FP4; cuBLASLt MXFP8, where code 8 becomes E4M3 0x80), Vulkan on other drivers [RIC section 2]. colibri's top-k, argmax and Engram treat -0 as +0 [RIC section 2, team 2].

**Required before any CZ use (proposal from team 2; it would replace the logprob check of REF:150, REF:250 - user decision)**:
1. **RTX 5090 bit test**: for every expert-matmul kernel KAIRA can dispatch on sm_120 (default CUDA FP32 kernels, DeepGEMM sm120 FP4, cuBLASLt MXFP8, Vulkan if enabled), run original vs canonicalised slabs with identical activations (the 9 adversarial sets above + real activations), memcmp the outputs: 0 differing bytes required; positive and negative controls must fire. Which kernels KAIRA dispatches is **OPEN-K**.
2. **Per-step logits memcmp**: same prompts, same deterministic settings, byte-exact container vs CZ container; memcmp the logits buffer at every decode step (e.g. the 4 W3-style prompts x >= 256 tokens): 0 differing bytes at every step.
If both give 0, CZ is bit-identical by construction on that machine and build.

**Tooling that breaks with CZ** (team 2, RIC section 2; all outside the arithmetic):
- the V4.1 deploy gate: `sample_records.py` + `test_decode_batch` compare decoded slabs byte for byte with the WEX1 slab [R REF:331-334, 348-349] -> must compare against Z(WEX1) for CZ records and require that differences sit exactly on the code-8 nibbles and number `cz_count`;
- exact-decode tests (`test_nr2_cuda`, `test_expert_store_wex`) -> canonical-exact variants;
- whole-file sha256 and `.verified` markers of the W3 procedure [R REF:42-44, 82, 139, 216] -> a CZ container has its own sha256; the builder must also record sha256 of the canonicalised source slabs so a readback can be verified;
- mirror pairing (header page + table byte-identical [R REF:468]) -> only byte copies of the same CZ container pair;
- **irreversibility**: a CZ container cannot recreate the original bytes. Deleting the WEX1 container or the safetensors after building a CZ container is irreversible. Rule: keep WEX1 (and the safetensors) until tests 1-2 passed and the decision is recorded. (`check_container` already takes `canonical_zero 0|1` [R REF:285]; byte-exact WNR3 incl. ceil-form needs none of this.)
- Engine policy unchanged from DRAFT 0.1 8.3: `COLI_WEX_ALLOW_CANONICAL_ZERO` (default 0) refuses a flag-64 container.

### 5.7 Length coding (feature d)
Section 4.3. Measured on ceil-form records: -25,791 B (CI -26,080..-25,483; densest-pair summary; g0 summary -25,764 B [CRIT dump_cr.txt:327]), 40/40 experts improved; with CZ the table is 13,366 B (weight residual std 8.11) [M CR]. The predictor reads coded scales only, so it is computed before any ceil fix-up. Not run on any GPU [section 0.4].

### 5.8 Start-state bits: HASH (v1 default) vs TAIL + DIGEST (optional, gated)

The 23 free bits of `x0` in `[L, 2L)` can carry either a per-stream hash (HASH, DRAFT 0.1 feature e) or raw data (TAIL). Measured and inferred properties:

| option | what x0 carries | bytes per V4.1 record | per-stream final-state check | silent single-bit flips | identity binding | implemented / measured |
|---|---|---|---|---|---|---|
| today (WNR2) | nothing, x0 = L | 0 | `x == L` | accepted, every accepted case one weight with its sign flipped (f[s] == f[s^8], CDF starts 8192 apart) [M T2A, RF]. Rates depend on the model: **team 2's model**, original form: w1 31/12,000 = 0.258%, w2 62/12,000 = 0.517% [M T2A corrupt_audit.json case A]; 392/69,120 = 0.567% [M T2A symperm_audit.json]. **Refuter's model** (sample groups {0,2}, 16 odd experts x 1,500 streams, 24,000 flips per matrix and form): original w1 63 (0.263%), w2 69 (0.288%); ceil-form w1 82 (0.342%), w2 139 (0.579%) [M RF flip_ceil_big.json]. Ceil vs original: w1 not significant (z = 1.58), w2 significant (z = 4.85); team 2 vs refuter original w2 differ (z = 3.09) [I CRIT consistency_out.txt section 9]. Last-byte flips: 0/12,000 accepted, w1 and w2 [M T2A case B] | none (but a valid stream at a wrong index with real contexts is caught 24,000/24,000 [M T2A test H]) | WNR2 |
| **HASH (v1 default)** | `L + (h_g & 0x7FFFFF)` | **+1,352 B** (+0.0072%) [M CR fge vs fg] | `x == L + h23(decoded)` | closed: 104/104 detected [M team 2 audit, MV]; 0/9,000 accepted [M ans-prior-art, MV]; residual 2^-23 per corrupted stream [I] | per stream | CPU: yes, with the FNV-1a + lowbias32 h23 of the ceil-ref lineage, not the normative H32 rule (CR fge/fgde, RF e2e 17/17 on coded symbols, +1,345 B/record; 5.1); GPU: no |
| TAIL alone (forbidden) | `L + sum_j tail_j << (b*j)`, 5 nibbles (b = 4) / 2 bytes (b = 8) | **-40,199 B** (-0.214%) [M CR fgt vs fg] | window `L <= x < L + 2^20` (weights) / `2^16` (scales): accepts ~2^-10.99 / ~2^-14.99 of random end states vs ~2^-31 for `x == L` [I CRIT consistency_out.txt section 7] | stays open (state unchanged by a sign flip) [I]; corruption ending in the tail bits of the final state (magnitudes too, not only signs) passes the window [I]; a changed tail magnitude can flip the block's M' == 5 decision, so one symbol error becomes a 32-weight error [I]; **rate not measured** | none | CPU: yes (window check, `wnr_common.py:398-405`); GPU: no |
| TAIL + DIGEST (optional) | tail as above; 32-bit record digest in header word 48 | **-41,548 B vs HASH (-0.221% of raw)** [M CR fgdt vs fgde, used bytes, densest-pair summary, CI -41,618..-41,495; g0 summary: -41,537 B, CI -41,588..-41,495, ratios 0.87800 vs 0.88021]; digest 0 B (reserved word of the fixed 64-B header) | window as above | intended closed: the digest covers every decoded coded symbol; residual 2^-32 per corrupted record [I, untested] | per record (ident in every h_g) | **DIGEST implemented nowhere**; no corruption audit |

**Decision: HASH is the v1 default.** It is measured on real records (+1,352 B/record) and in team 2's audit (104/104 sign flips caught), it keeps today's per-stream structure, and it localises a bad stream. TAIL + DIGEST would save 41.5 KB/record (-0.221% of raw, -0.246% of a WNR2 record), worth about +0.16% (U:) / +0.25% (E:) tok/s [Sim CU capacity_update_output.txt:41, 60, WNR3t vs WNR3], below the +-0.85% that an ABBA x2 A/B resolves [RIC:251]; its integrity side rests on code that does not exist yet. A missing or buggy digest path would be worse than today's WNR2.

**Gate for the optional TAIL + DIGEST profile** (all required before a builder may write flags 0xB4 / 0xA4 / 0xB6 / 0xA6, and recorded as a decision):
1. TAIL and DIGEST implemented in the CPU reference encoder and decoder, the GPU scale kernel (tail extraction, per-lane h_g, warp XOR + atomicXor), the GPU weight kernel (window check, tail extraction, completion of the last block, per-lane h_g fold), the GPU commit kernel (incl. the batch scratch word per record) and the tools (`check_container`, `sample_records.py` / `test_decode_batch`), with T2's TAIL/digest cases and T7 for the TAIL profile passing. This needs the GPU helper of rollout step R2 first (12 R2b).
2. Team 2's corruption audit (protocol `audit_corrupt.py`, cases A-J, 8 experts x 1,500 streams, 12,000 trials per case and matrix, w1 and w2) re-run on ceil-form records for {today's rule, HASH, TAIL alone, TAIL + DIGEST}: 0 accepted for TAIL + DIGEST in every case, including case B (last-byte flips) and tail-symbol magnitude changes; the TAIL-alone acceptance rate and the number of accepted corruptions that flip a block's M' == 5 decision are reported (test T3).
3. Reported, not gating: the GPU cost of per-lane H32 plus digest atomics (T9).

TAIL without DIGEST is forbidden (it is strictly weaker than today), and so is DIGEST without TAIL (the digest is defined only in the TAIL profile; 4.2 TAIL <=> DIGEST). Never both HASH and TAIL.

**TAIL definition** (optional profile) [C `wnr_common.py:62, 398-405, 675-676, 731-733`]: for a weight stream of n coded symbols, `kw = min(5, n)`, the last kw symbols `y_0..y_{kw-1}` (stream order) are not coded; `x0 = L + sum_j y_j << (b*j)` with b = 4 (layout 0) or 3 (layout 2, magnitudes; signs stay in the sign plane; layout-2 tail not measured); the encoder encodes the first n - kw symbols backwards from x0. The decoder decodes n - kw symbols, requires exact length consumption and `0 <= x - L < 2^(b*kw)`, and reads `y_j = ((x - L) >> (b*j)) & (2^b - 1)`. Scale streams: `ks = min(2, n)`, b = 8. Cost: the start state rises from L to at most L + 2^20, costing < 0.18 bit per stream [I]; the measured net saving is ~2.07 B per stream [M CR]. Measured only without CZ.

**DIGEST definition** (optional profile) [I, untested]: `record_digest = XOR over all S + W streams g of h_g` (full 32-bit H32 of section 5.1). XOR is order-independent, so lanes can fold it with warp shuffles and one `atomicXor` per warp into a per-record word that is zeroed together with the error word [I]. Swapped or misplaced streams change it because g and the identity are in every h_g [I]. Header damage is caught by `header_check` (which covers `record_digest`).

Other start-state uses measured but not adopted: Appendix A.4-A.6.

### 5.9 Sign-plane layout (feature s, optional, decode speed)
- Model layout 2 + record flag SIGNPLANE: rANS codes the 3-bit magnitude with 8-symbol CDFs (same 259 rows, same contexts); the sign of every weight is stored raw, one u32 per 32-block (bit j = sign of nibble j), fixed size, no lengths, as a separate plane between the scale and weight payloads (team 2 variant D [C `<RESEARCH-PC>\team2\gpu-proto\src\wnr2fmt.py:19-20, 192`]).
- Bytes: the sign given magnitude and context carries 0.9999990 bit, so the bypass costs 1.03e-6 bit/w [M T2G signent.json]; CFD 16,573,814 vs CF 16,573,694 B/record [M CG encode_report.json].
- Decode time on the RTX 3070 (**best config per variant**, final run, 41 paired rounds) [M CG best_final.txt]: 1 matrix A259 300.0 / D 280.6 us (-6.5%), CF5 304.1 / CFD5 276.5 (-9.1%); 1 record A259 590.8 / D 523.3 (-11.4%), CF5 565.2 / CFD5 540.7 (-4.3%); 8 records A259 4,171.8 / D 3,835.9 (-8.1%), CF5 4,185.1 / CFD5 3,848.2 (-8.1%). **These claims hold only best-vs-best.** At the same launch config, over the 8 configs, D/A259 is min/median/max 0.783/0.856/0.935 (1 matrix), 0.708/1.069/1.205 (1 record), 0.812/1.120/1.646 (8 records) [M CG ratios_final.json]; DRAFT 0.2's "0.708 (1 record)" was D's best config, where A259 takes 739 us vs its best 590.8 us. CF0/A259, the same kernel work on different stream bytes, ranges 0.9415-1.003 (median about 0.998) [M CG ratios_final.json], so same-config differences of +-3-6% are within the noise of stream content and config. The layout needs the staged-store path: at nt32 with plain stores D took 7,663.6 us per 8 records, 1.31x A [M T2G final_table.txt:13]. Shared memory: 4,160 B (layout 2) vs 8,304 B (layout 0) per block [M CG].
- Two chains per stream (DB2/CFDB2) are faster again at their best configs (CFDB2 413 us/record with 8 records in flight (3,307.5 us / 8), 431.1 us with 1 record in flight [M CG best_final.txt:35-36]) but cost +0.366% of the record (+0.32 pp of raw) vs CF [M CG; I CRIT consistency_out.txt section 6]: Appendix A.3.
- Constraints: SIGNPLANE => (HASH or TAIL) (TAIL implies DIGEST, 4.2), because the plain sign bits are protected by no rANS invariant [RIC section 5]; SIGNPLANE => not CZ in v1: with a fixed plane the zero signs are still stored, which gives back c's whole 0.1165 bit/w [I]. T and T^-1 act on magnitudes only (CFD5/CFDB2 byte-exact on the GPU [M CG]).
- Decision: not the v1 default. On the PCIe path the decode is roughly free and bytes are what count [R brief]; adopt layout 2 only if a compressed VRAM tier needs decode throughput, after a 5090 measurement.

### 5.10 Reference encoder order
[I, untested as a whole: CR's encoder ran the equivalent of steps 1 and 3-7 in its own format (48-B header, record-level gate and CEIL flag, one ceil model, FNV/lowbias32 hash), see 0.4.]
1. Load the source record (WEX1 slab); CZ mode: canonicalise.
2. Per matrix: gate (count max-5 blocks) and count even blocks; choose `set_id[m]`: the ceil set if the gate passes, the model has a ceil set and the matrix's even-block share is >= 1% (the threshold of 3.4); else the plain set [I policy; on V4.1 the even share is far above 1%: per expert 22.78-31.88% on the 160 sample, per type 30.31-31.12% [M RF refute_an.json]; per-matrix minima were not reported]. `ceil_mask` bit m = form of the chosen set; apply T to the matrices in ceil form. The build report records the choice per matrix (5.2).
3. `anchor[m] = anchor0` of the chosen set (g0); compute the densest-pair value of the coded scales for the build-report anchor line (5.3).
4. Contexts on coded data; per stream: forward pass for contexts and h_g; x0 (`L + h23` in the v1 default; `L` without HASH; `L + tail` in the TAIL profile); encode backwards; flush 4 B.
5. Lengths (u16 table or LENC with fitted `sbase`, `alpha`); TAIL profile only: DIGEST = XOR of h_g.
6. Header (header_check last), padding. If `total_bytes >= aligned_raw`, store raw (flags 0).
7. Verify with the CPU reference decoder (coded domain, then T^-1, then byte compare with the source, or canonical compare for CZ), and check `anchor[m] == sets[set_id[m]].anchor0[m]`, before writing [R REF:289].

---

## 6. Decoder order of operations (normative)

**Open (once per container)**: DRAFT 0.1 section 6 steps 1-5, with layout in {0, 2} and the set checks of 3.3.

[I, untested as a whole: CR's decoder ran the equivalent of steps 4-9 in its own format (48-B header, record-level CEIL flag, the encoder's in-memory tables, FNV/lowbias32 hash), see 0.4; per-matrix `set_id`/`ceil_mask`, LENC with base offsets, the H32 check and the 64-B header checks never ran.]

**Per record** (CPU reference `wex_nr3_decode` and GPU helper; any failure -> status -1, destination untouched):
1. Entry rules (section 2); read `total_bytes` into the slab end.
2. Header: magic; `header_check`; model id; flags == entry bits 8-15; all flag constraints of 4.2 (incl. set form vs `ceil_mask`, TAIL <=> DIGEST, and the fail-closed rejection of TAIL, DIGEST, CZ or layout 2 by decoders that do not implement them); chunks; `set_id < K`; owner == bind; reserved == 0; cz_count rule; size equalities; padding zero.
3. Geometry from the entry.
4. Length section, scale part (u16 table, or LENC with `sbase`); bounds; group continuity.
5. **Scale streams -> coded scales `cs`** (into a coded-scale buffer, never into the output); TAIL profile: tail bytes from the final state; final-state rule (`x == L + h23` in the v1 default, `x == L` without HASH, or the tail window); TAIL profile: h_g into the digest accumulator.
6. LENC: weight lengths from `cs` (coded domain); bounds; continuity.
7. **Weight streams**, per block: contexts from `cs` and coded nibbles; decode 32 coded symbols (TAIL profile: the stream's last block is 27 + 5 tail symbols, available only after step 8's final state); SIGNPLANE: merge the block's sign word into the coded words; CZ check; h_g over the coded words; **then** T^-1 for gated matrices (`M' == 5` -> D on the nibbles, final scale `cs_b - 1`, `cs_b == 0` -> invalid). **For every matrix: final scale = `cs_b` unless the matrix is gated and `M' == 5`.** Weights are written in output form; final scales go to the OUTPUT scale region, separate from `cs`.
8. Per stream: final-state rule; TAIL profile: window check, tail extraction, then completion of the stream's last block (CZ check, h_g, T^-1, store, as in step 7), then h_g into the digest.
9. TAIL profile: DIGEST: accumulated XOR == `record_digest`.
10. Commit the 6 regions (output scales + weights) only if no error (and, in the TAIL profile, the digest matches).

Order invariants: integrity checks run on the coded domain (steps 5, 7-9); T^-1 never feeds back into contexts, lengths or hashes; the scale fix-up of block b happens after block b's coded max is known and never before step 5 finished. The only dependencies are step 6 on step 5 and step 7 on step 5, as in WNR2 (the weight kernel already runs after the scale kernel because contexts need scales [R REF:435]).

---

## 7. GPU helper and kernels

**Scope of the GPU evidence** [M CG; C CG]: the prototype decodes WNR2-style streams (u16 lengths, x0 = L, 4-B flush) with the 259-row compacted tables, in-sample model, 8 experts, on the RTX 3070 (sm_86). It covers the **CEIL** part of the v1 profile only: no HASH, LENC, TAIL, DIGEST or CZ ran on any GPU, and nothing ran on the RTX 5090 or inside KAIRA's kernel. The decode cost of the rest of the v1 profile (H32 per lane, LENC warp scan; in the TAIL profile also tail completion of the last block and digest atomics) is unknown on any GPU.

### 7.1 Scale kernel
As `wex_cuda_nr2_scales` [R REF:425] plus: writes CODED scales to a coded-scale scratch buffer (`sum_m scount_m` B = 1,105,920 B on V4.1); per-lane H32 (HASH, and DIGEST in the TAIL profile); final-state rule; TAIL profile only: tail extraction and warp XOR + atomicXor of h_g (DIGEST); LENC scale part as DRAFT 0.1 7.1. Measured prototype scale-stream decode (WNR2-style streams, no hash): ~45 us per record, identical for plain and ceil-form scale streams (orig nt32 m3 46.1 us, ceil 45.3 us) [M CG].

### 7.2 Weight kernel: where the ceil inverse sits
Per lane (one stream), relative to KAIRA's kernel (one thread per stream, 32-thread blocks per matrix, shared CDF rows, cp.async ring, one uint4 store per 32 symbols [R REF:426-427]; exact code positions **OPEN-K**):
Per-block order (normative): decode 32 symbols -> merge the sign word (SIGNPLANE) -> h_g -> D / mask (gated matrices) -> store.
1. Unchanged decode chain: contexts read `cs` (via `__ldg` today), 16-way (layout 0) or 8-way (layout 2) compare, renormalisation. Ceil-form adds nothing to the dependency chain.
2. **After the 32nd symbol of a block, before the block's uint4 store**: the block max is the running `mx` (already tracked for contexts; for the last block in the TAIL profile, include the 5 tail nibbles); if the matrix is gated and `mx == 5`: apply `D` to the 8 nibbles of each of the 4 u32 register words with SWAR `f(m) = m + min(m, 2)` (sign bit kept) and set bit b of a per-lane block mask. Prototype: `wnr2dec.cu:183-188`, applied at `:310-315` [C CG]; cost CF3/CF0 median 0.995-0.999 (free) [M CG]. h_g is computed on the words before D.
3. **Final scales, deferred**: at stream end (or every 16 blocks for Cw > 2048) the lane writes its blocks' final scales (`cs_b - mask_b`, per-byte subtraction such as `__vsub4`, never a plain u32 subtraction) with aligned 16-B stores into the OUTPUT scale region. Every stream's output scales must land there, in one of two normative forms: (a) **full write**: each lane of a gated matrix writes all its scale words; non-gated matrices' output scales are a copy of `cs` written by their lanes or by a copy ordered between the scale and the weight kernel; (b) **mask form**: the output scale region is filled from `cs` **before the weight kernel starts** (the scale kernel writes both buffers, or a copy is ordered between the scale and the weight kernel); each lane of a gated matrix then overwrites only its changed 16-B words. The commit kernel copies the output scale region unconditionally and never reads `cs`, except in the optional bitmap fusion of item 5. Measured on the 3070 (own-cost vs the same streams with the inverse off, min/median/max over 8 configs): mask form CF5 m1 0.976/1.017/1.034, 1 record 0.986/1.007/1.022, 8 records 0.986/1.003/1.015; full 64-B write CF6 the same (8-record median 1.003); the literal per-block 1-byte store CF is expensive (1 record 1.012/1.157/1.205, 8 records 1.015/1.059/1.213) and is NOT allowed [M CG ratios_final.json]. CF5 vs A259 at the same config: median 1.014 (m1), 1.003 (1 record), 0.988 (8 records) [M CG]. Caveat: CF5 ("patch mode", form (b)) timings exclude the pre-fill of the output-scale buffer with the coded scales, which the harness does by a separate copy before the kernel [C `ceilbench.py:256`]; CF6 (full write, form (a)) includes every scale write and costs about the same, so the conclusion holds [M CG].
4. **No in-place patch of the coded scales**: the coded scales are a `const __restrict__` / `__ldg` input; patching them in place is undefined behaviour and makes a retry decode with wrong srel [C RF on `team2\gpu-proto\src\wnr2dec.cu:182, 237, 244`; C CG `wnr2dec.cu:218-223`]. The output region must be distinct from `cs`; this also keeps decodes idempotent. Note: the prototype's PIPE (scale -> weight pipeline) timings used exactly this forbidden in-place variant (`sc_in = osc = buf` [C `ceilbench.py:313-321`]); a separate-output pipeline was not timed.
5. Optional fusion (not implemented, not timed): the weight kernel writes only a per-block bitmap (138,240 B per V4.1 record) and the commit kernel, which already copies every scale byte [R REF:431], writes `cs - bit`. Saves the 1.05 MiB coded-scale copy per lane; to measure on the 5090.
6. TAIL profile only: after n - 5 steps, check the window, extract tail nibbles, complete the last block (step 2), then hash and store. Not implemented on any GPU.
7. HASH (default): finalise h_g and compare with the final state per lane, setting the error word. TAIL profile only: DIGEST: finalise h_g, warp XOR, one atomicXor per warp; the commit kernel compares the per-record word with `record_digest` (batch path: one digest word per record next to the error words in page-locked scratch [R REF:446]; **OPEN-K**). Neither implemented on any GPU.
8. SIGNPLANE: read the block's u32 sign word, merge sign bits into the 4 coded words before h_g and before D (h_g covers the sign bits as bit 3 of each nibble, 5.1; D keeps bit 3).

Prototype correctness [M CG bench_final.json; I CRIT]: byte-exact against the ORIGINAL safetensors weight and scale bytes for all non-diagnostic variants, 352 checks, 0 bad streams: **224 full checks** of 138,240 weight streams + 138,240 scale groups each (8 variants x 8 launch configs x {single launch, 8 launches on 8 streams} = 128, plus the scale -> weight pipeline for 6 variants x 8 configs x {single, multi} = 96) and **128 partial-launch rechecks** (8 variants x 8 configs x {1 matrix, 3 matrices}) covering 5,760 / 17,280 weight streams each; negative controls fail as expected (inverse off: 138,240/138,240 weight streams wrong; weight-only inverse: all 138,240 scale groups wrong).

### 7.3 Shared-memory and register budget (the a/b decision input)

| table (one matrix, per block) | rows | bytes | fits the ~99 KB/block budget [I, DRAFT 0.1 7.3 assumption] | measured |
|---|---|---|---|---|
| WNR2 today (567 rows, 32 B/row) | 567 | 18,144 | yes | A: 18,160 B smem [M T2G] |
| **v1 layout 0** (f+g, any K) | 259 | 8,288 | yes | 8,304 B smem; ceil inverse +0 B, regs +0..7 (nt32/out4 73 -> 80, nt128/256 out4 40 -> 44, occupancy unchanged) [M CG] |
| v1 layout 2 (sign plane) | 259 | 4,144 | yes | 4,160 B smem [M CG] |
| a on ceil-form, uncompacted (3-valued side x half) | 3,402 | 108,864 | **no** | blob 349,046 B [M CR] |
| a on ceil-form, structural compaction (max 5/6/7 groups x half + start) | 1,211 | 38,752 | yes, 4.7x v1 [I] | a 33,184-B table (team 2 variant C) ran 1.9-2.3x slower at 32 threads, about A at its best config [M T2G final_table.txt:11, 23, 35] |
| b (K sets) | 259 per set | 8,288 (only the selected set) | yes | device table K x 3 x 8,288 B [I] |

KAIRA adds a 64-word cp.async ring per lane (8,192 B per 32 lanes) [R REF:427]: v1 layout 0 at 32 threads = 16,480 B per block, 6 blocks/SM by DRAFT 0.1's arithmetic [I]; to confirm with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on sm_120 (**OPEN-K**). Extra device memory: coded-scale buffer 1.05 MiB per lane, 8.4 MiB per batch-calling thread with 8 lanes on top of ~287 MiB [R REF:416] [I]. Registers above were measured without H32 (no HASH on the GPU).

### 7.4 What the GPU catches
Group continuity, bounds, width, final-state rule (`x == L + h23` with HASH; the TAIL window in the TAIL profile), CZ symbol test, set-form check (host), `cs_b == 0` in a transformed block, digest compare (commit kernel, TAIL profile), header check and padding (host). Any one sets the error word; the commit writes nothing [R REF:431].

### 7.5 5090 transfer caveats (from CG, inferred)
One V4.1 record is 540 warps: ~11.7 warps/SM on the 3070's 46 SMs but ~3.2 on the 5090's 170 SMs [R REF:432], i.e. the latency-bound regime of the prototype's 1-matrix case, where CF5 cost +1.7% (median) [M/I CG]. L2 size, store behaviour, sm_120 register allocation and KAIRA's own kernel (cp.async ring, 567-row CTX 0 layout today, separate 88-KB-smem scale kernel, commit kernel) differ; the prototype's absolute times (A259 ~520-590 us/record) do not transfer (KAIRA: 0.2 ms/record kernel-only on the 5090 [R brief]). Byte deltas transfer unchanged (CG's absolute record sizes are not on CR's basis, 5.2); on the PCIe path (0.28 ms/record, PCIe-bound [R brief]) the v1 default needs 2.14% fewer bytes than production WNR2 (1 - 16,553,310.4 / 16,914,680; equivalently WNR2 is 2.18% larger, the WNR2/WNR3 - 1 figure of CRIT consistency_out.txt section 5, WNR3_g0) [I; record sizes: production WNR2 + paired CR delta, CU capacity_update_output.txt:224, 229], which shortens uploads by about 2.14% [I]. Nothing was measured on the 5090.

---

## 8. Feature interaction matrix (ceil-form vs everything)

| other feature | interaction | rule | evidence |
|---|---|---|---|
| g anchor | anchors are computed on coded-domain scales of the set's form; ceil-form makes the median nearly stable (2/690 flips vs 43/690 plain); a byte one step off the set's training anchor costs 1.09-1.16% of the record's raw size per matrix in the same form [M CR] and ~10% of raw with plain data on ceil tables [I] | g0: constant `anchor0` per set and type; fallback matrices use the plain set's anchor0; encoder anchor rule (4.2), build-report anchor line (5.2) and set-level stop (5.3) | [M CR, RF]; anchor-check cost argument [M CR test_ceil.json vs M RF refute_an.json] |
| d LENC | predictor uses coded scales (before fix-up) and per-set `etab` trained on coded data | section 4.3 | -25,791 B on ceil-form [M CR] |
| e HASH (v1 default) | computed on coded symbols; ident carries the ceil bit of m and the flags | 5.1, 5.8 | hash +1,352 B on ceil-form [M CR]; coded-domain hash e2e 17/17 [M RF] |
| t TAIL (optional, gated) | tail symbols are coded symbols of the stream's last block; T^-1 of that block waits for the tail; a corrupted tail magnitude can flip the block's M' == 5 decision | 5.8 gate; 6 step 7, 7.2 item 6 | -40,199 B on ceil-form, 40/40 byte-exact [M CR]; integrity not measured |
| DIGEST (optional, with TAIL) | computed on coded symbols, XOR of h_g | 5.8 | not implemented anywhere |
| c CZ | canonicalise before gate and T; commutes; CZ check valid in both domains | 5.6 | fg0c -2.738% of raw [M CR]; 0 commutation failures [M RF] |
| s SIGNPLANE | T acts on magnitudes, sign plane untouched | 5.9 | CFD5, CFDB2 byte-exact on GPU [M CG]; sign plane unchanged 708/708 [M RF] |
| a side symbol | if ever used: `bm3 = clip(max - 5, 0, 2)` on the coded block | not adopted | +10,642 B [M CR] |
| b K sets | sets carry `form` and `anchor0`; fallback = plain set; no k-means | 3.2, 5.2 | fallback with the ceil model: +397,312 B = +2.11% of raw with aligned anchors [M CR]; +11.7..+12.7% with misaligned anchors [M RF, estimate] |
| canonical index | unchanged 259 rows (coded max 5 is an ordinary tri row) | 3.2 | [I] |
| DEDUP | T deterministic and bijective: equal source records give equal coded records | none | 0 duplicate matrices among 708 [M RF] |
| raw fallback (codec 0) | unchanged: a record not smaller than raw is stored raw | 5.10 step 6 | [R REF:186] |
| CPU decode path (`wex_cuda_decode_to_host` fallback) | must implement T^-1 (C code; only the NumPy reference exists: f decode 3.10 s vs WNR2 2.78 s per record, not representative [M CR timing]) | 6 | **OPEN-K** |
| deploy gate (byte compare with WEX1) | passes after T^-1 (byte-exact) | none | [M RF e2e_real.json, CG] |
| RAM-cache verify, mirrors | operate on record bytes / header page: unaffected | none | [R REF:216, 468] |
| MTP records (if stored) | own sets (role 1) with their own anchor0; routed tables expand MTP records (ratio 1.110 plain, 1.079 ceil) | **open** | [M RF] |

---

## 9. Engine, helper API, migration

DRAFT 0.1 section 8 stays (helper feature bit 16 and the bind-taking entry points `wex_cuda_decode_to_host3`, `_to_device3`, `_batch_to_device3`; engine guard; old entry points return 1 for codec 5; compatibility matrix; disk note). Additions:
- Device tables per (model, device): K sets x (3 x 259 x 32-B rows + scale direct tables 3 x 16384 x 5 B) + etab; at K = 2: 2 x (24,864 + 245,760) + 600 = 541,848 B, ~0.54 MB (0.52 MiB) [I].
- Workspace: + coded-scale buffer per lane (7.3); TAIL profile only: digest words next to error words.
- Tools: `wex_nr3_tools` exports gain per-matrix ceil statistics and the anchor line; `nr3lib.py` record struct `'<8I8B2H5I'`; `check_container` learns codec 5, applies T^-1 and compares byte-exact, or canonical-exact for CZ records, and checks `anchor[m] == anchor0` while no WNM3 per-anchor feature bit is set (4.2) (TAIL profile: also the digest); `sample_records.py` / `test_decode_batch` learn codec 5 (byte-exact compare unchanged for non-CZ records) (**OPEN-K**: these live in `lab\v41-gpu-20261107\w1\` and `lab\wex-prod-20261105\` [R REF:273, 332]).
- Build: same procedure as W1 (retrain, encode, CPU-verify every record, every 43rd GPU/CPU byte compare, readbacks, whole-file sha256, per-record rehash [R REF:286-291]) plus the ceil statistics and anchor line of 5.2 and the anchor pre-pass with its set-level stop (5.3).

---

## 10. Expected sizes and value (V4.1, mean record)

### 10.1 Measured real records, ref-impl layout (one basis for the whole table)
CR, 40 held-out experts, g0 = constant anchor 120; ref-impl layout: 48-B header, length table without base offsets [M CR `heldout_summary_g0.json`; ratios, capacities and GB re-derived in CRIT consistency_out.txt section 2]. Capacity = 1/ratio - 1. Container = 7,680 x mean + 618,496 B, on two bases: the held-out mean record, and the production-anchored record (production WNR2 16,914,680 B + the paired CR delta) [I; CU, CRIT].

| config | mean total B | of raw | vs WNR2 (pp of raw) | capacity vs WEX1 [I] | 7,680 records, held-out mean [I] | 7,680 records, production-anchored [I] |
|---|---|---|---|---|---|---|
| WNR2 | 16,909,926.4 | 0.89943 | | +11.18% | 129.9 GB | 129.905 GB |
| WNR2-g0 (never encoded) | ~16,855,000..16,866,000 [I] | ~0.8965..0.8971 [I] | ~-0.24..-0.29 [I from RF] | ~+11.5% | ~129.5 GB | n/a |
| f | 16,578,764.8 | 0.88182 | -1.761 | +13.40% | 127.3 GB | 127.362 GB |
| f g0 | 16,573,132.8 | 0.88152 | -1.791 | +13.44% | 127.3 GB | 127.319 GB |
| f g0 d | 16,546,611.2 | 0.88011 | -1.932 | +13.62% | 127.1 GB | 127.115 GB |
| **f g0 d e (v1 default, HASH)** | **16,548,556.8** | **0.88021** | **-1.922** | **+13.61%** | **127.1 GB** | **127.130 GB** |
| f g0 d t (optional TAIL profile) | 16,506,880.0 | 0.87800 | -2.144 | +13.90% | 126.8 GB | 126.810 GB |
| f g0 c | 16,057,856.0 | 0.85411 | -4.532 | +17.08% | 123.3 GB | 123.361 GB |
| f g0 c d | 16,032,153.6 | 0.85275 | -4.669 | +17.27% | 123.1 GB | 123.164 GB |
| **f g0 c d e (v1 value-exact, HASH)** | **16,033,587.2** | **0.85282** | **-4.661** | **+17.26%** | **123.1 GB** | **123.175 GB** |
| f g0 c d t (value-exact, TAIL profile) | not run; ~15,991,955 [I: fg0cd - 40,199] | ~0.85061 [I] | ~-4.883 [I] | ~+17.56% | ~122.8 GB | n/a |

v1 default parts [M CR fgde g0, CRIT dump_cr.txt]: header 48, length table 13,118.6, scales 134,994.7, weights 16,398,072.8, padding 2,322.8 B; 3.7069 bit/w, 0.9765 bit/scale (incl. flush and hash). TAIL profile parts [M CR fgdt g0]: header 48, length table 13,121.2, scales 134,356.1, weights 16,357,171.4, padding 2,183.4 B; 3.6976 bit/w, 0.9719 bit/scale.

### 10.2 This spec's layout [I, untested; test T9]
The spec layout adds ~2,452 B per record (group base offsets 2,436 + header 16) to every row of 10.1 [I]:

| profile | flags | est. mean total B | of raw | vs WNR2 (pp of raw) | vs WNR2-g0 (pp of raw) |
|---|---|---|---|---|---|
| **v1 default** (f g0 d e) | 0x1C | ~16,551,009 | ~0.88034 | ~-1.909 | ~-1.62..-1.67 |
| TAIL profile (f g0 d t) | 0xB4 | ~16,509,332 | ~0.87813 | ~-2.131 | ~-1.84..-1.90 |
| **v1 value-exact** (f g0 c d e) | 0x1E | ~16,036,039 | ~0.85295 | ~-4.648 | ~-4.36..-4.41 |
| value-exact + TAIL (f g0 c d t) | 0xB6 | ~15,994,407 | ~0.85074 | ~-4.870 | ~-4.58..-4.64 |

(The DIGEST costs 0 B: it uses a reserved word of the fixed 64-B header. The tail saving was measured only without CZ.)

### 10.3 Value of the bytes (CU, simulated)
CU re-ran team 3's capacity-value Monte Carlo (400 draws, unchanged parameters and seeds) with CR's real per-expert record deltas applied to production WNR2 (16,914,680 B; record bootstrap per draw) [CU capacity_update_output.txt:1-26; inputs M CR]. **The model assumes G2 (V4.1 MoE on the GPU pipeline: VRAM mirrors, prefetch, copy worker, RAM record cache) and W2 (per-layer batched WNR decode) are built; neither exists yet** [CU capacity_update_output.txt:2-3; RIC:247]. Ceil-form decode cost is taken as 0 in the primary runs. tok/s gain vs WNR2, median [5%, 95%] [Sim CU capacity_update_output.txt:35-38, 41, 54-57, 60]; ratios and containers are production WNR2 + the paired CR delta [I from M CR; CU capacity_update_output.txt:20-23, 229-232]:

| profile | CU format | ratio (production-anchored) | container | U: tok/s vs WNR2 | E: tok/s vs WNR2 |
|---|---|---|---|---|---|
| **v1 default (HASH, g0)** | WNR3_g0 (fgde, g0) | 0.88047 | 127.130 GB | **+1.44%** [+0.75, +2.29] | **+2.35%** [+1.80, +2.80] |
| TAIL profile (g0) | WNR3t_g0 (fgdt, g0) | 0.87825 | 126.810 GB | +1.61% [+0.85, +2.56] | +2.63% [+2.01, +3.10] |
| **v1 value-exact (HASH, g0, c)** | WNR3z_g0 (fgcde, g0) | 0.85307 | 123.175 GB | **+3.51%** [+1.91, +5.61] | **+5.96%** [+4.51, +6.88] |
| value-exact + TAIL (estimate, not encoded; densest-pair anchor) | WNR3zt_est | 0.85113 | 122.894 GB | +3.65% [+2.02, +5.86] | +6.23% [+4.69, +7.15] |
| TAIL instead of HASH (difference) | WNR3t vs WNR3 | | | +0.16% [+0.08, +0.28] | +0.25% [+0.18, +0.36] |

- Resolution: an ABBA x2 A/B resolves about +-0.85% [RIC:251]; the TAIL-vs-HASH difference is below it. P(gain >= +3% vs WNR2) is 0.000 (U:) / 0.018 (E:) for the v1 default and 0.670 / 1.000 for v1 value-exact [Sim CU capacity_update_output.txt:35, 37, 54, 56].
- Sensitivity without W2 (single helper): v1 default U: +1.30%, E: +2.29%; v1 value-exact U: +3.15%, E: +5.84% [Sim CU capacity_update_output.txt:75, 77, 94, 96].
- Sensitivity to ceil decode cost (WNR3 with the densest-pair anchor, U:): dk = 0.034 (the largest measured CF5/CF0 own-cost, about +3.4%) -> +1.33% [+0.66, +2.24]; dk = 0.157 (the CF/CF0 m3 median of the naive per-block byte store, which 7.2 forbids; the measured worst CF/CF0 is 1.213 at m24 and 1.205 at m3 [M CG ratios_final.json]) -> +1.03% [+0.36, +2.16] [Sim CU capacity_update_output.txt:111, 113].
- These are model outputs, not measurements. CU's comparison with team 3's earlier entropy estimates (WNR3 vs T3_WNR3 +1.07% on U:, +1.76% on E: [Sim CU capacity_update_output.txt:46, 65]) measures a format change (team 3's WNR3 had no ceil-form), not an estimation error; real CR records match their code-length estimates within 5.1 B for the same configuration [M CR].
- Byte metric: the elasticity is defined per 1% of CU's `bytes_vs_WNR2_%` = WNR2/WNR3 - 1 (how much larger WNR2 is) [C `capacity_update.py:635`; Sim CU capacity_update_output.txt:164-202]. On that basis the CU cross-check gives the same result as elasticity x bytes_vs_WNR2_% (e.g. TAIL profile, WNR2/WNR3 - 1 = 2.44%, i.e. 2.38% fewer bytes [M CR heldout_summary_g0.json fgdt delta_total_pct_of_wnr2_record -2.383]: U: +1.58%, E: +2.65% [I CRIT consistency_out.txt section 4]); DRAFT 0.2's +1.4/+2.3% and +3.2/+5.3% used pp of raw instead and are withdrawn. The tok/s results above are unaffected.
- Disk: every WNR container here exceeds U:'s 122.6 GB free (v1 default margin -4.53 GB, value-exact -0.58 GB) [I: CU container sizes minus REF:139's free space, capacity_update_output.txt:229, 231; R REF:139]; those free-space figures are the KAIRA machine's, not this PC's.

---

## 11. Test plan

Each test (or test part) names the rollout step it gates: **[gates R1]** must pass before rollout step R1 counts as done; **[gates R2b]** belongs to the gate of the optional TAIL + DIGEST profile (5.8); **[gates R6]** / **[gates R7]** belong to the sign-plane layout / canonical zero; **[gates R4]** must pass before the full build. The list in 12 R1 is exactly the set of [gates R1] parts.

**T1 codec round trips** (CPU reference): valid combinations only (invalid ones are T2 cases): 3 chunk pairs (512:256, 2048:512, 4096:2048) x {non-CEIL flag sets (plain, LENC, HASH, LENC|HASH) x ceil_mask 0; CEIL-family flag sets (CEIL, CEIL|LENC, CEIL|HASH, **CEIL|LENC|HASH (v1 default)**) x ceil_mask in {1, 2, 4, 7} (plus 3, 5, 6 if wanted)}. **[gates R1]** for these sets. **[gates R7]**: the same sets + CZ. **[gates R6]**: + SIGNPLANE (layout 2, with HASH, without CZ). **[gates R2b]**: the TAIL profile sets CEIL|TAIL|DIGEST and CEIL|LENC|TAIL|DIGEST (and LENC|TAIL|DIGEST with ceil_mask 0). Synthetic matrices [gates R1 unless marked]: even blocks (all 6 magnitudes, signs, -0), even blocks with scale 0, 1, 253, 254 (coded 255), 255 (skipped); blocks with max 0..4 in a gated matrix (pass through, must not be rejected, 5.2); a matrix with one max-5 block (gate fails -> plain set, ceil bit 0, plain anchor0); a gated matrix with an even-block share below 1% (plain set, 5.10 step 2); all-zero blocks; all -0 (CZ, [gates R7]); uniform random nibbles (raw fallback); scale count not a multiple of 4 (H32 tail); the H32 per-stream hash over ident || coded bytes checked against T5's regenerated record-level vectors; scale streams with n < 2 (tail = n) [gates R2b]; a transformed last block of a stream (tail inside it) [gates R2b]; partial length groups; streams near the 2n+8 bound; a record whose anchor byte differs from anchor0 (encoder must refuse; `check_container` must flag). Exhaustive: the 65,536 4-nibble patterns x scales {0, 1, 120, 253, 254, 255}, padded to 32 nibbles, as RF edge_tests2, now through the full codec with `ceil_mask`/`set_id`. CPU decode byte-exact (canonical-exact for CZ). This is the first run of the per-matrix gate, skip-255, coded 255, the plain fallback set, the WNM3 blob and the 64-B header (section 0.4).

**T2 corruption, each rejected with the slab untouched** **[gates R1]**, except the cases marked [gates R2b] (DRAFT 0.1's >= 30 cases, minus BM7, plus): flag bit 0 set; HASH|TAIL; TAIL without DIGEST; DIGEST without TAIL (e.g. 0x94); HASH|DIGEST (0x9C); SIGNPLANE|DIGEST without TAIL; TAIL or DIGEST on a decoder without TAIL support; CZ or a layout-2 model on a decoder without CZ / layout-2 support; CEIL with `ceil_mask == 0`; `ceil_mask != 0` without CEIL; SIGNPLANE with CZ; `ceil_mask` bit flipped with header_check recomputed (form check); `set_id` changed to the other form's set with matching `ceil_mask` (must fail on hash/digest); transformed block with coded scale 0; `ceil_mask` bit 3 set; hash off by one bit; digest off by one bit (TAIL profile, [gates R2b]); tail window violated by one (TAIL profile, [gates R2b]); sign-plane bit flips (SIGNPLANE: hash or digest must catch, [gates R6]); LENC `alpha`/`sbase` changed (continuity).

**T3 misplaced data and silent acceptance** **[gates R1: HASH part; TAIL part gates R2b]**: DRAFT 0.1 T3 (bind, owner patch, swapped streams, DEDUP alias) with HASH; then re-run team 2's audit protocol (`audit_corrupt.py`, cases A-J, 8 experts x 1,500 streams, 12,000 trials per case and matrix, w1 and w2) on ceil-form records for {today's rule, HASH}, with the normative H32 rule of 5.1 (not the FNV/lowbias32 or CRC32 hashes that ran so far): required 0 accepted for HASH in every case. TAIL part, for the TAIL-profile gate (5.8): also {TAIL alone, TAIL + DIGEST}: required 0 accepted for TAIL + DIGEST in every case incl. case B and tail-magnitude changes; report the TAIL-alone acceptance and the count of accepted corruptions that flip a block's M' == 5 decision. State the model used, since acceptance rates depend on it (5.8).

**T4 canonical zero** **[gates R7]**: DRAFT 0.1 T4, plus the two 5090 tests of 5.6 and the tooling changes (Z(WEX1) compare, sha256 of canonical slabs).

**T5 model** **[gates R1]**: WNM3 open rejects layout 1, alphabet/layout mismatch, form 2, role 2, K = 0 or 17, bad set_bytes, etab > bound, plus DRAFT 0.1 T5; regenerate the record-level H32 vectors.

**T6 container** **[gates R1]**: DRAFT 0.1 T6 with entry bits 8-15.

**T7 GPU** (sm_86 and sm_120): byte-exact vs the CPU reference and vs the original source on 30+ records via host and device paths, with the full v1 profile (CEIL, LENC, HASH; not only CEIL as in CG) (gates R2), and **[gates R2b]** the same with the TAIL profile (TAIL + DIGEST: tail extraction, last-block completion, digest compare, a digest-off-by-one record inside a batch); batch sizes 1, 2, 3, 4, 6, 8, 10, 16; negative controls (inverse off; weight-only inverse; hash off) must fail everywhere (CG style); retry idempotence: decode the same device record twice -> identical output; compute-sanitizer memcheck/racecheck 0 errors (final-scale stores); a corrupted record inside a batch -> -1 and untouched, others 0.

**T8 fuzzing**: >= 300 mutations per flag set; CPU-only part (status and slab untouched for every mutation) **[gates R1]** for the R1 flag sets; differential CPU vs GPU status once the GPU helper exists (R2).

**T9 sizes and speed**: sizes **[gates R1]**: reproduce CR on the same 40 held-out experts with the spec layout for the v1 default (target: within +-0.02 pp of raw of CR's fgde g0 + 2,452 B). Sizes **[gates R7]**: the v1 value-exact profile (fgcde g0 + 2,452 B). Sizes **[gates R2b]**: the TAIL profile (fgdt g0 + 2,452 B) and fg0cdt, measured for the first time. Speed (reported, gates R2 on the 5090): kernel-only us/record at 1 and 8 concurrent records on the 5090 with KAIRA's kernel, with HASH (and, for the TAIL gate, with TAIL + DIGEST): gate WNR3 <= WNR2 + 5% (DRAFT 0.1 7.3), inverse own-cost <= +2% (CG measured +0.3..+1.7% median on sm_86, CEIL only). Report same-config ratios over all launch configs, not only best-vs-best.

**T10 build checks** **[gates R4]**: all records CPU-verified; every 43rd GPU/CPU byte compare; readbacks; sha256; per-record rehash; ceil statistics (max-5 blocks, skipped scale-255 even blocks, fallback matrices, gated matrices coded plain for an even share < 1%) and the anchor line (anchor byte == anchor0 for every matrix; per (set, type) count of matrices whose coded-scale densest pair differs from anchor0, report-only; the set-level stop of 5.3 not triggered or its override recorded) equal between build report and `check_container`.

**T11 inference** (part of R5; CZ part [gates R7]): byte-exact WNR3 -> short token-identity run (it is byte-identical to WEX1 after decode); CZ -> 5.6 tests 1-2.

**T12 generality (optional)**: gate and gain on V4-Flash, V4-Pro, gpt-oss (CO data in `<RESEARCH-PC>\team4-data\`), and a synthetic single-pass ceil checkpoint (expected ~22% max-5 blocks -> gate fails -> plain set) [I CO sim_double.json]; include the anchor line (constant anchor0 vs each checkpoint's scale distribution).

**T13 cross-decode conformance (new)**, in three parts. Every decoder used must share no code with team 3's `wnr_common.py` lineage and must parse the model from its blob, not receive the encoder's in-memory tables.
- **(a) CR records [gates R4; can run before R1]**: CR's 40 records are not in the spec format (team 3's 48-B header with flags in byte 27 [C `ceil-ref\wnr_common.py:22`], FNV-1a + lowbias32 h23 [C `:292-323`], WNM2-style 1,701-row model, record-level CEIL), so they need an independent decoder of the ceil-ref format, e.g. the GPU prototype extended to the 48-B header, LENC and the FNV/lowbias32 hash; byte-exact against the source safetensors bytes. This checks the measured numbers of section 10 for symmetric encoder/decoder bugs.
- **(b) spec-format records [gates R1]**: records from the new encoder (v1 default flags, ceil_mask in {0, 1, 2, 4, 7}, H32 rule of 5.1) decoded by the C reference decoder and by a second decoder written independently of it (for R1 a second CPU decoder; from R2 on also the GPU helper), both parsing the WNM3 blob: byte-exact output and identical status on T2's corrupted records.
- **(c) WNR2 encoder vs KAIRA [OPEN-K, 13.1 item 11; not gating R1 if R0 cannot resolve it]**: byte-compare the Python WNR2 encoder's records with KAIRA's C encoder on the same experts and model; today only sizes agree (MV 0.8999 vs production 0.8997 [MV]).

**T14 WNR2-g0 baseline (new)** (rollout step R0): retrain the WNM2 model with constant anchor 120 on the 80 even experts and encode CR's 40 test experts as plain WNR2 records with `median[m] = 120`; decode with the unchanged WNR2 reference decoder (and, once available, KAIRA's); report bytes vs WNR2 with CI. Its result replaces the [I] WNR2-g0 rows of section 10.

---

## 12. Ordered rollout

Steps in order; each step names the OPEN-K items (13.1) it needs.
1. **R0 (step 0) prerequisites and WNR2-g0 baseline**: resolve OPEN-K item 10 (whether anything re-derives `median[m]`, which decides whether WNR2-g0 and g0 records are legal); keep WEX1 and the safetensors; decide the disk for a ~127.1 GB container (U: had 122.6 GB free at 17:11 [R REF:139]). Then T14 on real records, no decoder change. If item 10 shows that nothing re-derives `median[m]`, a WNR2-g0 container is a deployable intermediate; from here on WNR3's value is stated against WNR2-g0. T13 (a) can run in parallel.
2. **R1 CPU reference** (`wex_neural_rans3.h`, tools): f, g0 with the anchor check, d, e (HASH) as the v1 default profile; TAIL, DIGEST, CZ and layout 2 may be left unimplemented (decoders then reject flag bits 5 and 7, bit 1, and layout-2 models; fail closed, 4.2). OPEN-K items 1, 2, 6, 11 (item 11 only if it can be resolved). Done only when every [gates R1] test part passes: T1 (R1 sets), T2 (except [gates R2b] / [gates R6] cases), T3 (HASH part), T5, T6, T8 (CPU part), T9 (v1 default sizes), T13 (b).
3. **R2 GPU helper** (feature 16): layout 0 kernels with HASH, LENC, the deferred scale write and separate output; T7 on the 3070, then the 5090; T9 speed gate on the 5090. OPEN-K items 3, 4, 5, 9.
4. **R2b optional TAIL + DIGEST** (starts only after R2): implement TAIL and DIGEST in the CPU encoder/decoder, the GPU scale, weight and commit kernels and the tools; pass the 5.8 gate (every [gates R2b] part: T1/T2 TAIL cases, T3 TAIL part, T7 TAIL profile, T9 TAIL sizes; T9 timing reported); record the decision. Until then no builder writes flags 0xB4 / 0xA4 / 0xB6 / 0xA6. OPEN-K item 5 (digest word).
5. **R3 64-record V4.1 sample** (`sample_records.py`, seed 20261107): deploy gate (byte compare with WEX1 after T^-1) with `test_decode_batch` extended for codec 5, CPU/GPU agreement, corruption statuses. OPEN-K item 8.
6. **R4 full byte-exact build** (flags 0x1C, K = 2, layout 0): T10 and T13 (a); 0 expected fallbacks; anchor byte == anchor0 in all matrices, the densest-pair-differs count per (set, type) reported and below the 1% threshold of 5.3 (or the override recorded); sha256 + `.verified`.
7. **R5 engine integration and A/B** vs WNR2 (and WNR2-g0 if built) on the same drive (token identity, tok/s). Expected by CU's model, which assumes G2 and W2 are built (neither exists yet): +1.44% on U: [+0.75, +2.29], +2.35% on E: [+1.80, +2.80] vs WNR2 [Sim CU capacity_update_output.txt:35, 54]; the U: median is above the +-0.85% A/B resolution only by a small margin.
8. **R6 optional sign-plane layout** only if a compressed VRAM tier needs decode speed (5090 measurement first, same-config ratios); [gates R6] tests.
9. **R7 canonical zero (separate decision)**: 5090 bit test + per-step logits memcmp; tooling changes; [gates R7] tests; CZ container (flags 0x1E) next to the byte-exact one; never delete WEX1/safetensors before the decision. OPEN-K item 7.

---

## 13. Open items

### 13.1 Needs the KAIRA sources (not on this PC)
1. `wex_validate_header` (v3) rejects unknown flag bits (REF:174 says only "any of flags 1|2|4|8|16").
2. No code path reads `entry.flags >> 8` as scale_pages for codecs other than 1 (entry bits 8-15).
3. KAIRA weight kernel: where the uint4 store and the running `mx` are, how the scale buffer is read (`__ldg`/`__restrict__`), real smem/registers/blocks per SM on sm_86 and sm_120, and whether the 567-row CTX 0 layout is replaced by the 259-row index.
4. Scale kernel: blocks per matrix (LENC), where it writes scales (needs the coded-scale buffer).
5. Commit kernel `wex_cuda_commit_regions` and the batch scratch (`WexCudaBatchHost`): room for a separate final-scale region or bitmap, and (TAIL profile) a digest word per record.
6. Registry room for the model kind (WNM2/WNM3).
7. Which expert-matmul kernels KAIRA dispatches on sm_120 (for the CZ bit test), and whether team 2's colibri line numbers (`deepseek_v4.c:20123-20129`, `quant.h:1266-1404`, `backend_cuda_dsv4.cu:99, 2250-2313`, `chain_moe_gv.comp:97-114`) match KAIRA's fork.
8. Code paths of `sample_records.py`, `precheck_records.py`, `test_decode_batch.cu`, `check_container.c`, `build_nr2_full.py` to extend for codec 5, T^-1, the anchor check and Z(WEX1) compares.
9. Helper lane memory: room for +1.05 MiB per lane.
10. Whether `wex_nr2_parse`, the CUDA plan or `check_container` re-check `median[m]` against the decoded scales (g0 writes a constant instead). This also decides whether WNR2-g0 (R0) needs no decoder change.
11. (new) Byte comparison of team 3's Python WNR2 encoder with KAIRA's C encoder on the same experts and model (T13); so far only the size agrees.

### 13.2 Other open questions and unmeasured configurations
- **Integrity**: DIGEST implemented nowhere; TAIL-alone and TAIL + DIGEST silent acceptance not measured (T3); which corruption classes TAIL lets pass (magnitudes and last-byte flips, not only signs) and how often a single symbol error becomes a 32-weight error through the M' == 5 decision when the check is weaker than `x == L` [I]. The refuter's "no damage amplification" holds only under `x == L` [M RF flip_ceil_big.json].
- GPU cost of per-lane H32 (HASH, or the TAIL profile's DIGEST) and of the digest atomics: not measured anywhere; no HASH, LENC, TAIL, DIGEST or CZ on any GPU; nothing on the RTX 5090 or with KAIRA's 567-row cp.async kernel.
- Real records in the spec layout (WNM3 K = 2 / 259 rows, 64-B header, base offsets, per-matrix gate, coded scale 255 legal): not encoded (T9). fg0cdt and the tail saving under CZ: not run. Layout-2 tail: not run.
- WNR2-g0: not encoded (T14); the anchor-fix share on CR's exact 40 experts: unmeasured (the refuter's P3 is a different set).
- Constant anchor (g0) on models other than V4.1 and on MTP (densest-pair anchors 113-114 in w1/w3 of 3 of the 6 MTP experts, 122 in the other 3 [M RF agg_extra.json]): evidence n = 1 on V4.1 (5.3); a one-step misalignment costs 1.09-1.16% of the record's raw size per matrix in the same form [M CR] and ~10% of raw with plain data on ceil tables [I] (5.2).
- Independent decoder: no decoder outside team 3's code lineage has decoded a ceil-ref / WNR3 record (T13).
- CPU-side T^-1 in C: not written; the NumPy reference (f decode 3.10 s vs WNR2 2.78 s per record, +11%) is not representative [M CR timing].
- Slot sizing: the maximum record over all 7,680 was not measured; over the 40 test experts max/mean is 1.0113 for the v1 default with g0 [M CR records, computed by CU, capacity_update_output.txt:26].
- Production-like training on all 7,680 experts: all deltas are held-out with 80 training experts, and the bootstrap CIs resample test experts only, not model training.
- MTP records: whether the container stores them (the index has 3 x 128 = 384 MTP routed experts [RF, team4_results.json:595, 776]); they need own sets and their own anchor0 [M RF].
- Shared experts were not checked against the ceil invariant [RF].
- A real single-pass-ceil checkpoint was not measured (gate failure predicted from simulation) [CO].
- No token / logprob check of CZ has run (5.6 tests 1-2).
- Commit-kernel fusion of the bitmap: not implemented.
- 3070 timings are a proxy for the 5090 (7.5).

---

## 14. Sources
- Measurements: CR, CG, RF, CO, CU, MV, T2A, T2Z, T2G, T3S, ANS (table at the top); team 3 DRAFT 0.1 and `wnr3_ref.py` / `lenres.json` (`<RESEARCH-PC>\team3\wnr3-spec\`); team 4 DRAFT 0.2 (`<RESEARCH-PC>\team4\spec-update\WNR3-DRAFT-v2.md`).
- Errata and consistency checks: CRIT (`<RESEARCH-PC>\team4\critic\critic_result.json`, `consistency_out.txt`, `dump_cr.txt`); team 4 member results `<RESEARCH-PC>\team4\critic\team4_results.json`.
- Reference: `D:\Desktop\WNR2-REFERENCE.md` (lines cited as REF:n) and `...\research\wnr2-brief.md`.
- Code read: `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py` (header line 3, transform 212-265, tail 62/398-405/675-676/731-733, LENC 499-599/771-789), `<RESEARCH-PC>\team4\ceil-ref\run_ceil.py:107`; `<RESEARCH-PC>\team4\ceil-gpu\src\wnr2dec.cu` (as cited by CG), `<RESEARCH-PC>\team4\ceil-gpu\src\ceilbench.py:252-267, 313-321`; `<RESEARCH-PC>\team4\ceil-refuter\anchor_an.py:7, 136`, `refute_count.py`; `<RESEARCH-PC>\team2\gpu-proto\src\wnr2fmt.py:12-22`; `<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c:4-19` (colibri source map); `<RESEARCH-PC>\team4\capacity-update\capacity_update.py:105-135` (format definitions).
- [S] MurmurHash3 / SMHasher (DRAFT 0.1 section 13); OCP MX v1.0 section 6.3 and DeepSeek `inference/kernel.py` (via CO and RIC sources).

---

## Appendix A. Measured, not adopted

**A.1 Feature a (block-max side symbol + block half) on ceil-form.** On ceil-form the block max takes 3 values (5, 6, 7), so the side symbol becomes `bm3 = clip(max - 5, 0, 2)`, one 3-symbol rANS step per block, contexts x6 (3,402 per type). Real records: fga vs fg +10,642 B (CI +10,135..+11,524), improved on 0 of 40 experts; side symbols cost 0.0332 bit/w; fgad -1.847% vs fgd -1.905% [M CR]. Estimate on top of ceil-form: +0.062% [M MV mxceil2]. Table: 3,402 rows uncompacted (108,864 B, does not fit), 1,211 rows compacted (38,752 B, 4.7x v1) [I]; a team 2 table of 33 KB ran 1.9-2.3x slower at 32 threads [M T2G]. On the plain form a gave -0.20% of the record [RIC section 7]; ceil-form absorbs it (the max-7 bit was a proxy for the doubled blocks) [M MV]. Verdict: costs bytes and speed. Reserved: record flag bit 0, model layout 1.

**A.2 Feature b (K k-means table sets).** fgaK2 vs fga -2,087 B (CI -4,564..-601, 27/40 improved), fgaK2 -1.722% vs fg -1.764%; fgaK2cde vs fgcde +8,585 B (CI +6,900..+9,625) [M CR]. Online-research team: K tables add ~0 on ceil-form (relayed in the task brief; RIC section 7 gives -0.02..-0.04%); plain-form K = 2..4 gains (-0.26..-0.28%) were mostly the anchor flip [M T3S: K2 "by flip flag" equals K2 EM, 3.77262 bit/w]. Shared memory: none (one set in smem); but no gain. The set mechanism stays only for the plain fallback set (and possible MTP sets).

**A.3 Two rANS chains per stream (DB2/CFDB2).** Fastest measured decode at the best configs (CFDB2 413 us/record with 8 records in flight (3,307.5 us / 8), 431.1 us with 1 record in flight, on the 3070 [M CG best_final.txt:35-36]) but +0.366% of the record (+0.32 pp of raw; CFDB2 0.88477 vs CF 0.88155) [M CG; I CRIT consistency_out.txt section 6]; CFDB2 vs DB2: +8-10 registers at nt128/256 (48 -> 56/57 regs with launch bounds, 56/58 with out4; occupancy 40 -> 36/32 warps); vs CF (40 regs) or CFD5 (40-46 regs) the difference is +10..+18 [M CG bench_final.json]. Not adopted: bytes dominate on the PCIe path.

**A.4 Split start state** (e.g. 12-bit hash + 2 tail nibbles): ~-0.12% estimated [RIC section 3], not measured; it shares the integrity questions of TAIL (5.8) and would need the same audit.

**A.5 Small initial state** (x0 = 2^k + h): -0.184..-0.188% of raw on 720 streams (k = 8 / 0), 24/24 round trips [M ANS flushx.json]; uses the same start-state slot as TAIL; measured on 720 real streams of 4 experts encoded with a real coder (bytes per stream, 24/24 round trips), not on full records [M ANS flushx.json]; TAIL is measured on real records (-0.214%).

**A.6 Alphabet order 2*mag + sign** (instead of codes 0..15): silent single-bit-flip acceptance 0.567% -> 0.058% at identical bits (3.759073 bit/w) [M T2A symperm_audit.json]. Made unnecessary by HASH (or DIGEST); would also change the CDF order the GPU compare relies on [I].

**A.7 Extra contexts on ceil-form** (held-out estimates, flush 3.5 B): full pm2 -0.0005% (42 KB table); 6-value pm2 bucket -0.0004%; position in block -0.019% (185 KB table) [M MV mxceil2]; row/column, running-mean, phase contexts ~0 (online-research team, relayed in the task brief; RIC section 7: learned context tree and row/column classes 0.002-0.005 bit/w). srel alignment on top of ceil-form -0.003 bit/w (online-research team) is about what g0 captures: -6,028 B = -0.0014 bit/w [M CR; I conversion].

**A.8 Non-ceil side-information model** (3-class side + half + fixed anchor + scale|class): -1.807% / -1.780% / -1.827% of raw (P2 odd 80 / P3 refuter group 1, which shares 20 of 40 experts with CR's set / P4 70 extra), 97% of the clean ceil gain; ceil-form adds -0.055% on top; needs 3,402 rows and a class stream decoded before scales [M RF anchor_an.json]. Ceil-form is preferred for simplicity at equal or better size.

## Appendix B. Worked example of T (one block)

Source block (scale s = 120, magnitudes of 32 nibbles, signs arbitrary): {7, 6, 0, 2, 4, 5, 0, 7, ... } using only {0, 2, 4, 5, 6, 7}, max 7 -> even. Values: magnitudes {0, 2, 4, 5, 6, 7} = {0, 1, 2, 3, 4, 6} x 2^(120-127).
T: magnitudes -> {5, 4, 0, 1, 2, 3, 0, 5, ...}, scale 121. Values: {3, 2, 0, 0.5, 1, 1.5, 0, 3} x 2^(121-127) = the same numbers (block max 3.0 at scale 121: the ceil rule's form).
Decoder: coded max 5 -> D: {7, 6, 0, 2, 4, 5, 0, 7, ...}, scale 121 - 1 = 120: the source bytes.
A non-even block {7, 3, 1, 6, ...} (contains 1 and 3) or {6, 4, 2, ...} (max 6) is not transformed; its coded max is 6 or 7, so the decoder leaves it alone.
