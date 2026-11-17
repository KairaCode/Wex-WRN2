# Entropy floor of the WNR3-CZ expert records (lane: floor, 2026-11-17)

**Question.** How far is WNR3-CZ from the best lossless code for these records, with no limit on the model?

**Answer (TEST set, 40 records, one per layer, L39 E120 apart).** Stored records are **0.85364** of raw.
The best code length found here is **0.84673** of raw: the gap is **0.69% of raw** (range 0.60..1.61% per record).
Of that gap:

| Part | % of raw | Measured how |
|---|---|---|
| Framing: u16 lengths 0.207, weight-stream flush+hash 0.328, scale-stream flush+hash 0.041, 4096-B padding 0.010, header | **0.585** | stored bytes minus the ideal cost under the production WNM2 (decomp.py, all 121 records: 0.586) |
| Weight model (production static 567 contexts vs the best adaptive model) | **0.069** (layers 0-7: 0.157; layers 8-38: 0.028; L39 E154: 0.658) | cm.c, measured |
| Scale model (production order-0 vs the best adaptive model) | **0.037** | cm.c, measured |
| Learning cost still in the bound (pre-training on 1 TRAIN record removed 0.019% more on 14 records) | ~0.02 more | cm.c -p, measured on 14 records |

So **only ~0.11-0.13% of raw is left in the model**; ~85% of the remaining gap is framing (length table, per-stream
flush/hash, padding), which is the other lanes' topic (wnr4-record). In bit/weight: stored 3.5883, ideal under
the production model 3.5743, floor bound 3.5714 (TEST40 means; all weights including signs).

L39 E120 (the outlier): stored 0.92331, floor 0.82028 (gap 10.3% of raw): 5.07% in its scales (the global scale
table costs 7.75 bit/scale there, the record's own distribution ~1.4 bit) and 4.64% in its weights (w2 has strong
per-column (= per-neuron) magnitude structure: 3.16 bit/w adaptive vs 3.57 static).

Two static-table items take about half of the model headroom and were checked with real rANS round trips:

| Item | Effect (real bytes, held-out) | Round trip | Static-table decoder? |
|---|---|---|---|
| **Per-record scale tables** (order-0 table of the record's own coded scales, stored in the record, 1 + 3 B per used symbol per matrix, ~60 B) | **-0.0265% of raw** mean over 120 records (median -0.0186, every record smaller: -0.016..-0.26%); **L39 E120 -4.75%** (scales 1,079,648 -> 187,078 B + 60 B) | rt_scale.c: production tables reproduce all 121 x 2,160 stored scale streams byte for byte; own tables decode back exactly, final state = L + h23 | yes: same decoder, the scale CDF is read from the record instead of the model blob |
| **Sibling context for w3**: context `srel3 x mx x clamp(s1 - s3, -2..2) x a1` (a1 = coded magnitude of w1[i,j]), 2,520 contexts, two table sets (layers 0-7 / 8-39), trained on TRAIN | **-0.0316% of raw** mean over TEST40 (one record per layer = container estimate); layers 0-7 **-0.121%**, layers 8-39 -0.009% (of which -0.0016% is only retrained tables); max +0.005% | rt.c: production tables reproduce all 41 x 5,760 stored w3 streams byte for byte; the variant decodes all 41 TEST records exactly | yes, if w1 is decoded before w3 (same Cw and shape: w3 stream j needs w1 stream j only). Tables 2 x 2,520 x 34 B = 171 KB |

## Mutual information of every relation the task named (measured, real records)

`mi2.py`, all 121 records, in-sample plug-in with Miller-Madow bias correction, conditioned on srel (the bias term
is <= 1e-4 bit). Bits per weight of the named matrix (1/3 of the record's weights); 0.01 bit/w on one matrix =
0.078% of raw of a record. Mean over 120 records (L39 E120 apart) | layers 0-7 | layers 8-39 | max.

| Relation | mean | L0-7 | L8-39 | max |
|---|---|---|---|---|
| **w3[i,j] vs w1[i,j] magnitude, given srel3** | 0.00253 | **0.0103** | 0.00058 | 0.0287 (L0) |
| same, given srel3 and srel1 | 0.00381 | **0.0147** | 0.00110 | 0.0398 |
| full nibbles (adds the sign) | 0.00255 | 0.0103 | 0.00060 | 0.0288 |
| sign(w3) vs sign(w1), per weight | 0.00001 | 0.00002 | 0.00000 | 0.00011 |
| scale w3 vs scale w1 (bit/scale) | 0.0034 | 0.0039 | 0.0033 | 0.045 |
| same position, other expert of the same layer (w1/w2/w3) | 0.00000 | | | 0.00006 |
| scales, other expert of the same layer | 0.00000-0.00007 | | | 0.0016 |
| same expert index, adjacent layer (L0E7-L1E7, L22E21-L23E21; `adjacent.py`, 2 extra records fetched) | <= 3e-6 (weights), <= 1.1e-5 (scales) | | | |
| column mean magnitude, own matrix: w1 / w2 / w3 | 0.00057 / **0.00269** / 0.00052 | | | 0.0078 / 0.0227 / 0.0077 |
| column mean of another expert of the same layer: w1 / w2 / w3 | 0.00018 / 0.00002 / 0.00017 | | | 0.007 |
| row mean magnitude, own matrix | 0.00005-0.0006 | | | 0.0034 |
| exact position in block (32) | 0.00002-0.00012 | | | 0.0046 (L39 w2) |
| w2 column (neuron) vs magnitude of w1+w3 rows of that neuron | 0.00059 | | | 0.0035 |
| L39 E120: w2 column / neuron relation | 0.239 / 0.110 | | | |

Held-out version on top of the full WNR3 context (base567), 8 TEST records, 2-fold by row parity (`mi.py`,
`out/mi_weights.json`): sibling magnitude x sibling srel gains 0.029 (L0), 0.016 (L5), 0.0001-0.0002 (L11),
<0 from L16 on (the extra contexts cost more than they bring). Every other feature (position, row/column means,
other expert, nibble above) gains < 0.005 or loses.

**The sign is a coin against everything tried, including the sibling's sign (<= 1e-4 bit/weight).**
**Experts of the same layer and the same expert in the adjacent layer carry nothing** (independent tensors).
The only cross-tensor relation with real information is **w1 <-> w3 at the same [i,j]**, and only in the first
~8 layers (0.003-0.040 bit/w on w3; layers 8-39 ~0.001).

Scales (`mi.py scales`, held-out 2-fold, 9 records): order-0 ~0.88-0.98 bit/scale; previous scale in the row,
scale above, both, the sibling's scale, column/row statistics, other expert: every gain <= 0.04 bit/scale except
in L39 (E120 w3: left neighbour 0.57 bit/scale, row median 0.78; E154 w1/w3: 0.04-0.06). The block max of the block's own
nibbles carries 0.18-0.54 bit/scale, but that only matters if nibbles are coded before scales, which would cost
the nibbles their `srel` context (far more).

## The models (all held-out)

- `cm.c`: adaptive binary context mixing (PAQ-style): each nibble = 3 magnitude decisions (tree) + 1 sign decision;
  each scale = 8 decisions. 13 weight-context models (srel; WNR3's srel/pm/mx; position; adaptive column mean;
  nibble above + neighbours; two back; zero count; exact scale offsets; for w3: w1 magnitude, its neighbours j-1/j+1,
  w1 block max, w1 scale, w1 column means; for w2: w1+w3 neuron magnitude), 10 scale-context models (left, up,
  left2, up-left/up/up-right, sibling scale, adaptive column and row means, row max), counts with 1/(n+1.5) rate,
  two gated logistic mixers, one APM. Coding order: scales, w1, w3, w2. Nothing trained offline: an honest
  adaptive code length (an arithmetic coder adds a few bytes per record; the bound adds 32 + 8 + 1 B per record).
  `-f 0` = only the WNR3 contexts (learning-cost check: within +0.04..-0.06% of the production static model);
  `-p` = pre-trained on TRAIN record L20 E175.
- The floor per record = per part (scales, w1, w2, w3) the best of cm all / cm -f 0 / cm primed (14 records), and
  per group (weights, scales) the production static model if better. (Taking the production model per matrix
  instead of per group would lower the bound slightly; not done: the bound is conservative.)
- `nn.py`: MLP 175-128-64-8 over a wide causal neighbourhood (4 magnitudes back, row above j-1..j+1, column mean,
  exact scale, position; for w3 w1 at j-2..j+2, its sign, block max, scale offset), trained on 1.6 M weights of 20
  TRAIN records, scored on 82 K weights of each of 8 TEST records against static base567 trained on the same 20
  full records: w1 mean gain -0.0001 bit/w (nothing); w3 +0.009 bit/w in L0/L5, ~0 or negative after L10. A
  global network does not beat the static table where there is no sibling information. Weak evidence (small test
  samples); the adaptive CM is the stronger bound.

## Recommendation for WNR4 (from this lane)

1. **Do not expect more than ~0.12% of raw from any model change.** The model is within ~0.11-0.13% of the best
   adaptive code found; the rest of the gap to the floor (0.585%) is framing.
2. If a WNR4 is built anyway, two model items are cheap for a static-table decoder and verified by round trip:
   per-record scale tables (-0.027% mean, -4.75% on the L39 E120 outlier; removes the outlier's tail) and the
   w1->w3 sibling context with a layer split (-0.032%, -0.12% in layers 0-7). Together ~-0.058% of raw
   (~11 KB per record, ~84 MB over 7,680 records, extrapolated from 40-120 records). Not measured: how they stack
   with K table sets per stream (-0.024..-0.044%, prior work); they are different features, so likely mostly additive.
3. Anything cross-expert or cross-layer: drop (zero MI). The sign: keep as a coin. Scale contexts beyond a
   per-record table: <= 0.01% except in L39.
4. L39 w2 (per-neuron column structure) is the only other sizeable relation (0.02 bit/w on w2 in L39, 0.24 in
   E120); a per-column class as side info could use it, but over the container it is noise-level (estimate from
   MI, not round-tripped).

## Not checked / caveats

- 41 TEST records for the CM bound (one per layer + E120); primed CM only on 14. No MTP experts.
- CM, NN and MI numbers are ideal code lengths; the static items are real rANS bytes (rt.c / rt_scale.c using the
  production primitives of wex_neural_rans.h). The round trips cover the changed streams only; the other streams are
  unchanged by construction. Decoded coded nibbles/scales equal the coded slab, which is `wex_nr3_prepare(cz=1)` of
  the reference decoder's output; the record flags and anchors are unchanged, so the ceil inverse gives the same
  WNR3-CZ output (not run through a full WNR4 record parser: no such format exists).
- The record header/format for per-record scale tables and the second table set is not designed here.
- GPU cost of the sibling dependency (w3 stream j after w1 stream j: half as many independent chains per record,
  each twice as long) is inferred, not measured (no GPU). No timing was done in this lane; all sizes are
  machine-independent and transfer to every production machine.
- The extrapolation to 7,680 records assumes the one-record-per-layer TEST mean is representative; the container's
  entry table gives a mean of 0.85398 of raw for all 7,680 (sample 0.85436).

## Files and rerun (from this directory; ~2 h on 2 threads in total, mostly cm)

See README.md. Raw outputs are in `out/` (`*.jsonl`, `*.json`, logs); `out/coded/` holds the coded slabs (2.2 GB,
regenerated by extract.py).
