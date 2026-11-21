# check-floor: verification of the 'entropy floor' lane (\<TMP>/wnr3-research/floor)
All reruns in this directory (copies of the lane's scripts; own outputs in out/). Own scripts: indep.py (re-encode
all 121 coded slabs with the production encoder + production-counter ideal costs), sib_check.py (own context code +
own quantiser for the sibling tables, plus variant/split selection by 2-fold CV inside TRAIN), scale_indep.py,
mi_quick.py.
- extract.py: 121 coded slabs byte-identical to the lane's. indep.py: production wex_nr3_encode_record on each coded
  slab reproduces all 121 stored records byte for byte (so the coded slab is exactly the WNR3-CZ domain).
- decomp / rt_scale / scale_table / static_sib / rt / rt_v0train / cm full / cm f0 / cm primed: outputs identical
  (cm bit-exact) to the lane's; primed subset file holds 12 records, lane's output 14 -> floor 0.84674 (gap 0.6907%)
  vs reported 0.84673 (0.6916%).
- Independent: framing 0.5854% TEST40 (lens 0.2068, w 0.3281, s 0.0410, pad 0.0093); bpw 3.5883 / 3.5743.
  Sibling V3 split8 ideal TEST40 -0.03162% (rt real -0.03163%); CV inside TRAIN picks V3 + split 8 too (-0.0244%
  with 40-record tables). Scale tables: entropy bound -0.02652% vs real -0.02648% (120 recs), median -0.0186%.
  MI I(a3;a1|srel3,srel1) matches mi2.jsonl to 1e-5 on 7 records.
