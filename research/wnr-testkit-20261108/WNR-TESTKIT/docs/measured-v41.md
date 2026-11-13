# Measured on real V4.1 REAP-192 MXFP4 experts (2026-11-07, this PC)

Data: 160 experts (4 per layer x 40 layers, random.Random(20261107).sample per layer) range-downloaded from
`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` @ 4ab649b2254afdcf6c2f7c1fbea1c05eda11828c into
`<RESEARCH-PC>\v41-sample` (L{layer}_E{expert}.bin = w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight back to back;
manifest.json has dtype/shape/bytes/sha256). Tensors: w1/w3 weight I8 [2304, 2560] (packed E2M1, low nibble = even column),
scale F8_E8M0 [2304, 160]; w2 weight I8 [5120, 1152], scale [5120, 72]. 3.0 GB total.

Scripts (research PC, temporary folder `<TMP>\wnr2\`):
mxcount.py (pass 1 counts; test_mxcount.py checks it against a plain loop of the WNR2 rule), mxanalyze.py (14-bit quantizer
exactly like wex_nr2_quantize, costs, held-out = train on half the experts, test on the other half, both ways),
mxcombo.py + mxcombo_an.py (combined contexts, clusters), mxstreams.py (length table), mxpairs.py (cross-expert),
rans_check.py (exact WNR2-style rANS, L = 2^23, 14-bit, byte renorm, 4-B big-endian flush). Results JSON in
`<RESEARCH-PC>\v41-results\` (analysis.json, combo.json, streams.json, pairs.json).

## Baseline reproduced (validates the harness)
- Shared static 567-context model, in-sample: weights 3.7821 bit/w (production measured 3.781), scales 1.0672 (1.066),
  record estimate 16,919,483 B = 0.8999 of raw (production 16,914,680 B = 0.8997). Held-out 3.7824.
- Ideal (unquantized) entropy 3.7665 + flush 0.0156 = 3.7821: the 14-bit CDF quantization costs ~0.
- Order-0 weights 3.895 bit/w. Per matrix: w1 3.7675, w2 3.7657, w3 3.7664.
- Record byte shares of raw: weights payload 88.99%, scales 0.78%, flush (4 B/stream) 0.41%, u16 length table 0.21%, padding 0.011%.

## Contexts added to the 567 (held-out, % of weight bits)
- srel +-7 instead of +-3: 0.000%; sign of previous symbol: 0.000%; upper row same column: +0.0005%;
  upper row after sorting rows by mean scale: 0.000%; w3 given w1 at the same position: -0.035%.
- position in block (x32): -0.148%; magnitude two back (x9): -0.284%.
- block max magnitude is ALWAYS 6 or 7 (100% of blocks; 44-46% are 6): OCP scale rule confirmed.
  Coding "block max == 7" as 1 side bit per block (cost 0.0155 bit/w, included) and using it as context: -0.161% (442 used rows, 13.8 KB table);
  + half of block: -0.213% (803 rows, 25 KB); + pm2 bucket + half: -0.226% (1,674 rows, 52 KB). All fit the ~99 KB/block shared memory
  once unreachable rows are compacted.
- Model granularity: per-layer models gain ~0.05% (LOO) and overfit with 4 experts/layer; layer groups G=8: -0.06%;
  per-expert own model: payload -0.28% but +57.8 KB table -> net +0.07% (worse).
- K models chosen per matrix (2-bit id per matrix), held-out: K=2 -0.26%, K=4 -0.28%, K=8 -0.285%.
  On top of block-max+half contexts: K=2 3.7696, K=4 3.7692 bit/w = -0.35% vs 3.7826.
- Scales: order-0 1.0672 bit/scale; left neighbour 1.0649; upper 1.0668; left+upper 1.0630; after row sort 1.0617 (-0.5% of scale bits = 0.004% of record).

## Big lever: signed zero
- +0 and -0 are EXACTLY equally frequent: 5.8255% and 5.8263% of all weights. The sign of a zero is a fair coin = 0.1165 bit/w of noise.
- WNR2z (-0 -> +0 before coding; value-exact, not byte-exact): 3.6657 bit/w in-sample on the base context (-3.08%);
  with block-max+pm2 contexts 3.6594 held-out; + 4 models 3.6544 (-3.38%). Record ratio ~0.868 (capacity vs WEX1 +15.2% instead of +11.1%).
- Production note from the reference: WNR2z "needs a logprob comparison before production; tokens identical in 4 runs only" (V4).

## Ceil-form (found by team 2/critic, re-checked independently with mxceil.py + mxceil_an.py, all 160 experts)
- 176,947,200 blocks: max magnitude code 6 = 79,342,654, code 7 = 97,604,546, anything below 6 = 0.
- 'Even' blocks (max code 7 and no magnitude 1 or 3, i.e. only 0, 1, 2, 3, 4, 6 used = doubled codes): 54,228,706 = 30.65%
  (w1 30.31%, w2 31.12%, w3 30.51%; per expert 22.8-30.9%). Under i.i.d. magnitudes the chance is 1.9e-4.
  Reading: the weights were quantized with DeepSeek's ceil scale rule (block max 3.0, 4.0 or 6.0) and then stored in OCP
  floor form, so ceil-max-3.0 blocks were moved one octave down with every code doubled.
- Transform: even block -> magnitudes halved (0,2,4,5,6,7 -> 0,1,2,3,4,5, sign bit kept), scale + 1. Inverse: block whose
  decoded max magnitude is 5 -> doubled, scale - 1. Bijective because no stored block has max < 6.
  Byte round trip on all 160 experts: 0 failures; scale + 1 never reaches 255.
- Held-out (same 567-context model, retrained on the transformed data, median on the transformed scales):
  weights 3.7824 -> 3.7084 bit/w (-1.96%); order-0 scales 1.0050 -> 0.9167 bit/scale.
  Record: -339.5 KB = -1.81% of raw, ratio ~0.8997 -> ~0.882, capacity vs WEX1 ~+11.1% -> ~+13.4%.
- With the zero-sign merge (WNR2z, value-exact) on top: 3.5920 bit/w (-5.03% of weight bits), record -4.55% of raw, ratio ~0.854.
- Extra contexts ON TOP of ceil-form (mxceil2.py + mxceil2_an.py, held-out, flush 3.5 B/stream, ceil.json/ceil2.json in v41-results):
  base 3.7065 bit/w; + full pm2 -0.0005% (42 KB table); + 6-value pm2 bucket -0.0004%; + block-max side symbol (3 values on
  ceil-form, 0.033 bit/w of side info) + half: +0.062% (WORSE); + both: +0.064%; + position in block -0.019% (185 KB table).
  So ceil-form absorbs essentially all of the block-max / pm2 / K-model gains: they were proxies for the doubled blocks.

## Team 4 (real rANS records and GPU; details in WNR-RICERCA-2026-11-07.md and <RESEARCH-PC>\team4\)
- ceil-ref, 40 held-out experts, 736 records encoded+decoded (0 errors, 0 fallbacks), % of raw vs WNR2 0.89943:
  f -1.761% (CI -1.948..-1.632); f+g0 (anchor 120) 0.88152 (-1.791%); f+g0+d+e ~0.8802 (-1.922%); f+g0+d+tail 0.87800 (-2.144%);
  f+g0+c+d+e 0.85282 (-4.661%). On ceil-form: 3-valued block-max side symbol +0.057% (worse on 40/40), K=2 -2.1 KB,
  densest-pair anchor a wash, d -25.8 KB, e +1,352 B, tail instead of hash -41.5 KB. Real delta = estimate within 5 B.
- ceil-gpu (RTX 3070): inverse byte-exact vs original safetensors bytes in every variant/config; deferred scale patch costs
  ~0 (-2.4..+3.4% vs same streams without inverse); 1 record CF5 565 us vs A259 591 us.
- ceil-refuter: invariant on 236 records / 261 M blocks incl. 70 extra and 6 MTP experts: 0 violations; ~0.27 pp of the
  headline is the anchor fix; best non-ceil side-symbol model gets 97% of the gain with bigger tables; MTP needs own models.
- ceil-origin: artifact present upstream (deepseek-ai/DeepSeek-V4.1-Flash-1017), in V4-Flash, V4-Pro, gpt-oss-20b/120b;
  absent in PTQ checkpoints. Mechanism: ceil quantization, then the scale is derived again from dequantized values.

## Overheads
- u16 length table (38,880 B/record): lengths can be predicted from the already-decoded scales (E[bits | srel] per block);
  residual entropy ~4.5-4.8 bits per weight stream, ~2.8-3.8 per scale stream -> ~11.0 KB instead of 38.9 KB: -0.148% of raw record.
- Flush 4 B/stream x 19,440 streams = 0.41%; unchanged by any context change.

## Cross-expert
- Mutual information between two experts of the same layer at the same position: ~1.8e-5 bit/w (= estimator bias); scales ~1e-5.
  0 duplicate rows out of 1.56 M, 0 duplicate matrices across the 160 experts. Delta/XOR/other-expert coding: no headroom.

## Integrity idea tested
- Start each stream's rANS encoder at x0 = L + h (h = 23-bit hash of the stream's symbols, its index and record identity);
  the decoder checks final state == L + hash(decoded). Cost on 300 real streams: +0.067 B/stream (~1.3 KB/record, 0.007%).
  40/40 exact round trips; random single-bit flips: 40/40 caught by both today's rule (final state == L) and the hash rule;
  a VALID stream decoded at the wrong index: hash rule 40/40, today's rule 0/40.
- CORRECTED by the team 2 audit (larger trials): today's rule silently accepts 0.26-0.63% of random single-bit flips, and
  every accepted case is one weight with its sign flipped (f[s] == f[s^8] and the two CDF starts differ by exactly 8192).
  The hash rule caught 104/104 (audit) and 0/9000 accepted (ans-prior-art). My 'wrong index 0/40' test was set up wrong
  (it reused the original stream's contexts); with the real contexts of the wrong index today's rule catches 24,000/24,000.
  So the hash's real value is (a) closing the silent sign-flip hole and (b) whole-record substitution, if the identity is hashed.
- Exact rANS flush is ~3.50 B/stream, not 4 B (audit): absolute sizes above are ~9.8 KB/record too high; deltas unchanged.

## Not measured here
- Any GPU decode speed (no CUDA toolkit installed; RTX 3070 sm_86, driver 621.01). Any token/logprob effect.
