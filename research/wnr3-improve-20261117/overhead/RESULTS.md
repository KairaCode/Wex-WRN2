# WNR3: scales, record skeleton and integrity (lane "overhead", 2026-11-17)

This lane covers every byte of a WNR3-CZ record except the weight nibbles: the scale streams, the skeleton
(lengths, rANS start state and flush, padding) and what integrity costs. Every size below is **held out**: a
model trained on the 80 TRAIN records (`model_train.bin`, `scale_train.bin`) encodes the 41 TEST records (40
typical + L39 E120). Each figure comes from a **real encode followed by a decode** with this lane's decoder,
compared byte for byte with the reference decoder's output Z(source): **41/41 records x 28 profiles exact**.
Raw record = 18,800,640 B, and 0.01% of raw = 1,880 B/record.

## 0. Baseline and harness

- Stored production records, whole sample (121): mean 16,062,582 B = **0.85436 of raw** (`extract.py`). TEST
  41: 0.85534. The 40 typical TEST records: 0.85364.
- Harness: for 41/41 TEST records, `wex_nr3_encode_record` with the production model reproduces the stored
  record byte for byte. This lane's own encoder (profile `wnr3`) does too, and its decoder decodes the stored
  record to Z(source).
- WNR3 re-encoded with the TRAIN model is 835 B/record smaller than the stored records (typical). All savings
  below are measured against **WNR3 re-encoded with the same TRAIN model** (`wnr3_train`), so they are pure
  format effects. Section 4 also gives them against the stored records.
- Container facts (exact, not extrapolated): the 32-B header of all 7,680 records (`headers.bin`, fetched by
  `fetch_headers.py`, about 250 KB of HTTP range reads). Mean scale payload SP = 134,790 B. All 7,680
  records use chunks 2048:512.

## 1. Scale streams

### 1a. Why L39 E120 carries 1.08 MB of scales (`out/explain_outlier.txt`)
The production scale model is an **absolute** order-0 table per matrix. Every value other than 119..122 gets
the minimum frequency, 1/16384, which costs 14 bits. 7,642 of 7,680 records have rule-g anchors (120,120,120).
L39 E120 has anchors (121,122,121): its whole scale distribution sits one or two binades higher. For example,
matrix 1 has 239,532 scales at 123 (14.0 bit each) and 99,949 at 122 (9.0 bit). The ideal cost under the
production table is 1,071,922 B against a stored SP of 1,079,648 B. Its own order-0 entropy is 179 KB.
Across the container, the 38 records with a non-120 anchor have a mean SP of 317,833 B (others: 133,880 B).
L39 E154 has anchors of 120, but its wide matrix-1 distribution still costs 177 KB.

### 1b. Context models: ideal code lengths, held out (bounds, no stream overhead)
Source: `scale_bounds.py`, `scale_bounds_cs512.txt`. Contexts only use earlier symbols of the same stream,
an earlier matrix or the header anchor, so every model keeps the scales decodable before the weights.

| model (per matrix, 14-bit CDFs) | 40 typical, B/record | L39 E120 |
|---|---:|---:|
| today: absolute order-0 | 127,138 | 973,671 |
| relative to the anchor, order-0 (256 symbols) | 127,138 | 314,031 |
| + previous scale in the row | 126,738 | 251,847 |
| + scale above | 126,824 | 307,381 |
| + previous and above | 126,447 | 249,288 |
| + mean of up to 8 previous scales in the row ("mean8") | 125,976 | 184,979 |
| + mean8 and above | 125,739 | 184,054 |
| + sibling (w1 scale for w3) | 126,963 | 320,594 |
| per-record order-0, in-sample entropy (`scale_adaptive.py`) | 122,065 (+~28 B of table) | 179,358 |
| K = 2 / 4 / 16 global table sets per matrix | 126,047 / 125,631 / 125,426 | 295k / 281k / 278k |

**Typical records are near their floor.** Given its own distribution, a scale is close to independent of
its neighbours: the best context gains about 1.4 KB/record (0.007% of raw, noise level). The scale above and
the sibling matrix give nothing. What matters is (i) the record-to-record shift of the distribution and (ii)
the quantisation loss of a 256-symbol table, where every unused symbol takes at least 1/16384, about 1.5% of
the mass, worth about 3 KB/record.

### 1c. Real rANS encodes (held out, `out/report.txt`)
Scale symbol = scale − anchor[m] + 7 in 0..14, else an escape (15) followed by the raw byte as two flat
nibbles. Escapes never occurred on the 41 TEST records.

| profile | saved B/record (40 typical) | % raw | L39 E120 saved | container (extrap.) |
|---|---:|---:|---:|---:|
| `s_rel16`: global 16-symbol table, relative | 2,937 | 0.016 | 662,744 | 23 MB |
| `s_mean8`: + mean8 context (10 contexts) | 4,094 | 0.022 | 791,646 | 31 MB |
| `s_rec16`: per-record 16-symbol table in the record (25 B) | 5,036 (3,255..40,858) | 0.027 | 794,292 | 39 MB |
| `inl_rec16`: same, scales **inline** in the weight streams (no scale streams) | **17,041** | **0.091** | 806,252 | **131 MB** |
| `inl_mean8` | 16,045 | 0.085 | 801,487 | 123 MB |
| for comparison, scale chunk 2048 (`s_rec16_cs2048`) | 14,047 | 0.075 | 803,279 | 108 MB |

- Per-record tables are the robust fix. They gain 5 KB on a typical record and fix the tail: L39 E154
  −40.9 KB, L39 E120 1.08 MB → 187 KB. The table costs 24.8 B/record on average.
- **Inline scales** (one 16-symbol scale symbol at the start of each block, inside the weight stream that
  carries the block) remove the 2,160 scale streams: their 2 B lengths and their 3.57 B start/flush+hash
  each. That is worth more than raising the scale chunk to 2048, which costs single-record GPU latency
  (V4/WNR2 5090 figures: 0.200 → 0.261 ms). It also removes the scale kernel. The block's scale is decoded
  just before the block's 32 nibbles, so `srel` is still available, and the weights' contexts do not change.
- Container extrapolation = typical mean × 7,680. The 38 shifted-anchor and ~45 wide-scale records add at
  most about 5-6 MB more: their SP above 160 KB sums to 6.27 MB, exact from the headers. **The outlier is
  spectacular per record (−4.2% of that record's raw) and small for the container.**

## 2. Skeleton (held out, real bytes, `out/report.txt`; savings vs `wnr3_train`, 40 typical records)

| # | option alone | saved B/rec | % raw | container | integrity |
|---|---|---:|---:|---:|---|
| 1 | coded length table (`lenc`, wnr4-record's group-32) | 24,460 | 0.130 | 188 MB | unchanged |
| 2 | minimal 3/4-byte start state (`sd`) | 2,406 | 0.013 | 18.5 MB | unchanged |
| 3 | 16-bit renorm, L = 2^16, **23-bit hash kept** (`r16_k23`) | **−8,590** (worse) | −0.046 | −66 MB | unchanged |
| 3b | 16-bit renorm, 16-bit hash + record digest (`r16_k16_dig`) | 8,416 | 0.045 | 65 MB | 16 bits + digest |
| 4 | tail 2 nibbles in x0, 15-bit hash + digest | 15,408 | 0.082 | 118 MB | 15 bits + digest |
| 5 | tail 5 nibbles in x0, 3-bit hash + digest | 38,591 | 0.205 | 296 MB | 3 bits + digest |
| 6 | inline per-record scales (section 1) | 17,041 | 0.091 | 131 MB | unchanged |
| 7 | chunks 4096:1024 / 8192:2048 | 54,110 / 81,182 | 0.288 / 0.432 | 416 / 624 MB | unchanged |
| 8 | 64-B instead of 4096-B alignment | +1,630 on top of any row (sample mean padding) | 0.009 | 12.5 MB (wnr4-record, exact over the 7,680: 15.6 MB) | n/a |
| – | 40-B WNR4 header instead of 32 B | −8 | 0 | | |

**16-bit renormalisation is a negative result.** With the hash kept, it costs 0.44 B/stream. The final state
of a 16-bit coder is spread over [2^16, 2^32), mean log2 about 24, against [2^23, 2^31) and about 27 for
byte renormalisation, so the 4-B flush carries less free state. Its −0.045% in mxfp4-contexts came from
starting at x0 = 2^16 with no hash, that is from dropping 7 hash bits. At equal hash width, a tail in x0
buys 0.11 B per hash bit against 0.063 B for r16. The flush itself wastes only about 0.7 B/stream; the rest
of the 3.57 B is the 23-bit hash slot (2.875 B). **The only large skeleton levers are the length table,
the hash width and the stream count.**

### Combined profiles (all round trips exact; `out/report.txt`, `out/final_table.txt`)

| profile | contents | saved vs `wnr3_train` B/rec | % raw | L39 E120 | container (extrap.) | 64-B-aligned record / raw | vs stored, % raw |
|---|---|---:|---:|---:|---:|---:|---:|
| A | lenc + sd (= wnr4-record's lenc_sd) | 26,874 | 0.143 | 22,615 | 206 MB | 0.85208 | 0.157 |
| B | A + per-record scale tables | 31,951 | 0.170 | 817,811 | 245 MB | 0.85181 | 0.184 |
| **C** | **A + inline per-record scales** (hash unchanged) | **40,471** | **0.215** | 826,978 | **311 MB** | 0.85136 | 0.229 |
| D | C + tail2 (15-bit stream hash) + digest | 55,911 | 0.297 | 842,807 | 429 MB | 0.85053 | 0.311 |
| **E** | **C + tail5 (3-bit stream hash) + digest** | **79,088** | **0.421** | 866,486 | **607 MB** | 0.84930 | 0.434 |
| F | C with 16-bit renorm, 16-bit hash + digest | 45,564 | 0.242 | 835,828 | 350 MB | 0.85108 | 0.256 |
| G | C at 4096:1024 | 76,160 | 0.405 | 863,879 | 585 MB | 0.84946 | 0.419 |
| H | E at 4096:1024 | 95,471 | 0.508 | 883,610 | 733 MB | 0.84843 | 0.521 |
| I / J | C / E at 8192:2048 | 94,468 / 104,130 | 0.502 / 0.554 | | 726 / 800 MB | 0.84848 / 0.84797 | 0.516 / 0.567 |

"vs stored" = stored 4096-padded record minus the 64-B-padded profile record, typical mean (it includes the
835 B by which the TRAIN model beats the production model). After E, what is left outside the model code
length is about 2 B per weight stream (0.77 B of coded length, about 1.2 B of flush), about 34 KB/record
(0.18%). That is an estimate from the measured parts, not a separate measurement.

## 3. Integrity

### Design of the replacement check (profiles D, E, F)
- Per stream, as today: h = MurmurHash3_x86_32(ident {model, owner, g, word24} ‖ the stream's decoded coded
  bytes; inline: the block scales, then the packed weights), seed "WNR3". The low `23 − 4·tail` bits of h go
  in x0 (tail profiles), or 16 bits (F). The decoder checks them, plus exact length consumption, x0 window,
  CZ and the ceil-scale rule, as today.
- **Record digest** = XOR of the full 32-bit h of every stream, stored in the header (4 B). The decoder XORs
  each stream's h as it finishes and compares once at the end. Fail closed: the output is scratch until then.
  Cost: CPU, one XOR per stream. GPU, one `__shfl_xor` warp reduction plus one `atomicXor` per warp
  (540/record), compared in the commit kernel, which already reads the error word before it copies.
- **Strength (analysis):** a stream passes its k-bit check only if h' ≡ h (mod 2^k), so the digest change is
  h ⊕ h' with its low k bits zero and 32 − k random bits. P(false accept) = 2^−k · 2^−(32−k) = **2^−32 per
  corrupted record whatever k is**, against 2^−23 per corrupted stream today. Localisation is per record,
  not per stream, which is what the store acts on anyway. At-rest integrity (the WEX4 per-record SHA-256)
  does not change.

### Audit (`ovh_proto audit`, `out/audit_report.txt`)
4 TEST records (L10 E153, L20 E107, L30 E61, L39 E120) × 7 profiles × 6 corruption types × 2,500 trials, plus
every header bit: **61,280 corruptions per profile, 429k in total.** Types: random bit flip anywhere in the
record, random byte overwrite, 16-byte random burst, swap of two streams of the same matrix and size, stream
cut by one byte (lengths and SP/WP repaired), one stream replaced by the same stream encoded under another
owner, and all header bits. Verdicts come from an incremental decode: only changed streams, or streams
whose scales changed, are decoded again, and anything in the header goes through the full decoder. They
were cross-checked against the full decoder on 70 trials per profile: 0 mismatches.

| profile | stream check | accepted WRONG | 95% upper bound per corruption | caught only by the digest | accepted, identical output |
|---|---|---:|---:|---:|---:|
| WNR3 (today) | 23 bit | 0 / 61,024 | 4.9e-5 | – | 0 |
| A lenc_sd | 23 bit | 0 / 61,280 | 4.9e-5 | – | 0 |
| C inline rec16 | 23 bit | 0 / 61,280 | 4.9e-5 | – | 0 |
| D tail2 + digest | 15 bit | 0 / 61,280 | 4.9e-5 | 128 (all digest-word flips) | 0 |
| **E tail5 + digest** | 3 bit | **0 / 61,280** | 4.9e-5 | **331** (203 corruptions + 128 digest-word flips) | 1,264 (*) |
| F r16 k16 + digest | 16 bit | 0 / 61,280 | 4.9e-5 | 128 | 1 (*) |
| **negative control: E without digest** | 3 bit | **182 / 61,280** (8 bit flips, 18 bytes, 4 bursts, **152 swaps**) | – | – | 1,243 (*) |

(*) "other owner" trials: the substituted stream decodes to the same data, differs only in its hash bits,
and passes the k-bit check with probability 2^−k (1,264/10,000 ≈ 1/8 at k = 3). The output is correct
(identical), so this is not a false accept.

Reading: a 3-bit stream check alone lets 0.3% of corruptions through (1.5% of stream swaps). With the digest,
all 203 corruptions that passed every per-stream check were caught, and none of 61,280 was accepted wrong.
The audit can only bound the rate at about 5e-5 per corruption (0/203 conditional: 1.5% upper bound); the
2^−32 is the analysis above.

## 4. Decoder cost
- Measured (`out/bench.jsonl`): this lane's reference-style decoder (binary search, 1 thread, 9 interleaved
  rounds, machine at load 12). WNR3 1,177 / 1,123 ms; C 1,041 / 1,140; E 1,141 / 1,205; F 1,102 / 1,079 (two
  records, medians). The spread within one profile is ±15%, so **no difference is detectable**. The fast CPU
  decoder and the GPU kernels were not modified or run.
- By reading the code (`c/wex_cuda.cuh`):
  - lenc: the host offsets builder (8.5 → 36.5 µs per record, wnr4-record).
  - sd: 3 lines in the stream open.
  - per-record tables: up to 96 B parsed at open, three 17-entry CDFs staged as 2 `uint4` rows each.
  - **Inline scales: kernel A goes away**, with its 88 KB shared-memory direct table and the A→B dependency.
    Kernel B decodes one more 16-symbol symbol per 32 nibbles (+3.1% symbols) with the existing
    `wex_cuda_nr2_row_find`, and writes the scale it decodes instead of reading it.
  - tail5: the last block decodes 27 symbols and takes 5 nibbles from the final state; window check instead
    of equality.
  - digest: the warp XOR above.
- Transfer: sizes are properties of the format and hold on every machine. Chunk sizes 4096+ trade
  single-record GPU latency (earlier 5090/V4 measurements: 0.264 → 0.437 ms single, unchanged in batch-16).
  Nothing here uses AVX-512, so the i7-7700K (AVX2) and the 7950X3D are affected alike.

## 5. Recommendation
1. Only as part of a WNR4 (codec 7): **profile C** (lenc + sd + inline per-record 16-symbol scales,
   64-B alignment). −0.215% of raw (−0.229% vs stored), about 0.31 GB (+12 MB alignment). Integrity
   unchanged (23-bit stream hash), audit 0/61,280. It also fixes the scale outliers and removes one GPU
   kernel.
2. If the owner accepts record-level instead of stream-level integrity: **profile E** (C + tail5 +
   32-bit record digest). −0.421% of raw (−0.434% vs stored), about 0.61 GB, stronger overall (2^−32 per
   record against 2^−23 per stream), audited 0/61,280 with 203 digest-only catches. The digest is
   mandatory: without it, 0.3% of corruptions pass.
3. Chunk 4096:1024 (H: −0.508%) is a build choice, and only if single-record latency allows it.
4. Do not do: 16-bit renormalisation (worse at equal hash width); scale contexts beyond the anchor shift
   (≤0.007%, noise); sibling or above contexts (0); K table sets for scales (beaten by per-record tables).
