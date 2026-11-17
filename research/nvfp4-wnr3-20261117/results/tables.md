## 1. Raw bytes of the 16 sampled experts

| checkpoint | format | weights B | scales B | global scalars B | total B | bits per weight |
|---|---|---:|---:|---:|---:|---:|
| ds-nvfp4 | nvfp4 | 283,115,520 | 35,389,440 | 384 | 318,505,344 | 4.5000 |
| ds-mxfp4 | mxfp4 | 283,115,520 | 17,694,720 | 0 | 300,810,240 | 4.2500 |
| ds-quark | mxfp4 | 283,115,520 | 17,694,720 | 0 | 300,810,240 | 4.2500 |
| qwen-nvfp4 | nvfp4 | 39,321,600 | 4,915,200 | 384 | 44,237,184 | 4.5000 |
| qwen-mxfp4 | mxfp4 | 39,321,600 | 2,457,600 | 0 | 41,779,200 | 4.2500 |
| glm-nvfp4 | nvfp4 | 201,326,592 | 25,165,824 | 384 | 226,492,800 | 4.5000 |
| glm-mxfp4 | mxfp4 | 201,326,592 | 12,582,912 | 0 | 213,909,504 | 4.2500 |
| glm-nz5 | nz5 (csf0_v1) | 201,326,592 | 6,422,528 | 384 | 207,749,504 | 4.1276 |

## 2. Properties of the nibbles and scales

| checkpoint | -0 nibbles | share | blocks with max magnitude 6 | even blocks | blocks with max 3 | scale codes used |
|---|---:|---:|---:|---:|---:|---:|
| ds-nvfp4 | 0 | 0.0000% | 0.4074 | 0.2502 | 4,863,781 | 6 |
| ds-mxfp4 | 33,013,398 | 5.8304% | 0.5211 | 0.3001 | 0 | 6 |
| ds-quark | 0 | 0.0000% | 0.5211 | 0.3001 | 0 | 6 |
| qwen-nvfp4 | 3,113,970 | 3.9596% | 0.6031 | 0.0056 | 0 | 51 |
| qwen-mxfp4 | 0 | 0.0000% | 0.4152 | 0.0001 | 0 | 7 |
| glm-nvfp4 | 13,773,073 | 3.4206% | 1.0000 | 0.0122 | 0 | 31 |
| glm-mxfp4 | 0 | 0.0000% | 0.5601 | 0.0000 | 0 | 4 |

## 3. Entropy estimates (in-sample, pooled per matrix over the 16 experts)

Nibbles, bits per nibble:

| checkpoint | order 0 | class only | WNR3 contexts (class, pm, mx) | same after CZ | full scale code, pm, mx (CZ) | + position in block (CZ) |
|---|---:|---:|---:|---:|---:|---:|
| ds-nvfp4 | 3.7787 | 3.6578 | 3.6419 | 3.6419 | 3.6605 | 3.6416 |
| ds-mxfp4 | 3.8942 | 3.7671 | 3.7474 | 3.6307 | 3.6307 | - |
| ds-quark | 3.7776 | 3.6504 | 3.6307 | 3.6307 | 3.6307 | 3.6280 |
| qwen-nvfp4 | 3.9446 | 3.9246 | 3.9206 | 3.8414 | 3.8422 | 3.8374 |
| qwen-mxfp4 | 3.7760 | 3.6890 | 3.6848 | 3.6848 | 3.6848 | 3.6822 |
| glm-nvfp4 | 3.9680 | 3.9574 | 3.9461 | 3.8777 | 3.8643 | 3.8692 |
| glm-mxfp4 | 3.7860 | 3.7356 | 3.7308 | 3.7308 | 3.7308 | 3.7295 |

Class: MXFP4 `srel7` = clamp(scale - anchor, -3, 3) (WNR3); NVFP4 `exponent srel7` = the same on the E4M3 exponent field.

NVFP4 nibbles under other scale classes (bits per nibble, WNR3 contexts with that class, no CZ):

| checkpoint | none | mantissa | exponent | exponent srel7 | code srel25 | full code |
|---|---:|---:|---:|---:|---:|---:|
| ds-nvfp4 | 3.7244 | 3.7244 | 3.6605 | 3.6419 | 3.6596 | 3.6605 |
| qwen-nvfp4 | 3.9392 | 3.9388 | 3.9268 | 3.9206 | 3.8978 | 3.9214 |
| glm-nvfp4 | 3.9580 | 3.9545 | 3.9451 | 3.9461 | 3.9300 | 3.9327 |

Scales, bits per scale:

| checkpoint | order 0 | order 1 (left) | left + parity | delta along row | order 1 (row above) |
|---|---:|---:|---:|---:|---:|
| ds-nvfp4 | 1.3016 | 0.8419 | 0.5546 | 1.0176 | 1.1107 |
| ds-mxfp4 | 1.0004 | 0.9988 | - | 1.4733 | 1.0002 |
| ds-quark | 1.0004 | 0.9988 | 0.9988 | 1.4733 | 1.0002 |
| qwen-nvfp4 | 4.5278 | 4.0992 | 4.0986 | 4.2908 | 4.1082 |
| qwen-mxfp4 | 0.9636 | 0.9605 | 0.9604 | 1.4247 | 0.9634 |
| glm-nvfp4 | 3.5395 | 3.4964 | 3.4961 | 3.9752 | 3.4940 |
| glm-mxfp4 | 0.7060 | 0.7055 | 0.7055 | 1.1638 | 0.7060 |

Estimated stored / raw (nibble entropy + scale entropy + raw scalars; no model, chunk or padding cost):

| checkpoint | WNR3-like (class srel7, scales order 0) | same + CZ | full scale code + best scale model, CZ |
|---|---:|---:|---:|
| ds-nvfp4 | 0.8274 | 0.8274 | 0.8251 |
| ds-mxfp4 | 0.8891 | 0.8616 | 0.8616 |
| ds-quark | 0.8616 | 0.8616 | 0.8616 |
| qwen-nvfp4 | 0.9341 | 0.9165 | 0.9108 |
| qwen-mxfp4 | 0.8741 | 0.8741 | 0.8741 |
| glm-nvfp4 | 0.9261 | 0.9109 | 0.9073 |
| glm-mxfp4 | 0.8830 | 0.8830 | 0.8830 |

## 4. WNR3 (the repository encoder) on the MXFP4 samples

| checkpoint | profile | byte-exact, cross-fold | byte-exact, in-sample | CZ, cross-fold | CZ, in-sample |
|---|---|---:|---:|---:|---:|
| ds-mxfp4 | deepseek | 0.8807 | 0.8804 | 0.8533 | 0.8529 |
| ds-quark | deepseek-quark | 0.8533 | 0.8529 | 0.8533 | 0.8529 |
| qwen-mxfp4 | qwen38 | 0.8810 | 0.8810 | 0.8810 | 0.8810 |
| glm-mxfp4 | qwen38 | 0.8892 | 0.8891 | 0.8892 | 0.8891 |
| glm-mxfp4, records of the existing container | (container model) | 0.8891 | | | |

Container records: 16 of the 16, WNR3 flags 0x07, decoded equal to the HF source in 16 of 16.

## 5. Prototype coder (nvwnr.c), cross-fold unless marked in-sample, records padded to 4096 B

| checkpoint | configuration | stored / raw | scale payload B | weight payload B | round trips | decode MB/s, 1 thread |
|---|---|---:|---:|---:|---:|---:|
| ds-nvfp4 | nv-exp7-o0 | 0.8419 | 8,593,575 | 258,841,991 | 16/16 | 26.0 |
| ds-nvfp4 | nv-exp7-o0-cz | 0.8419 | 8,593,575 | 258,841,991 | 16/16 | 26.5 |
| ds-nvfp4 | nv-exp7-o1-cz | 0.8308 | 5,053,913 | 258,841,991 | 16/16 | 26.5 |
| ds-nvfp4 | nv-exp7-o1p-cz | 0.8284 | 4,278,314 | 258,841,991 | 16/16 | 26.1 |
| ds-nvfp4 | nv-code12-o1-cz | 0.8403 | 5,053,913 | 261,839,029 | 16/16 | 26.9 |
| ds-nvfp4 | nv-exp7-o1-cz-ceil | 0.8308 | 5,053,913 | 258,841,991 | 16/16 | 26.8 |
| ds-nvfp4 | nv-exp7-o1 | 0.8308 | 5,053,913 | 258,841,991 | 16/16 | 26.7 |
| ds-nvfp4 | nv-code6-o1-cz | 0.8403 | 5,053,913 | 261,840,715 | 16/16 | 26.9 |
| ds-nvfp4 | nv-code24-o1-cz | 0.8403 | 5,053,913 | 261,839,123 | 16/16 | 27.0 |
| ds-nvfp4 | nv-code12-o1 | 0.8403 | 5,053,913 | 261,839,029 | 16/16 | 26.9 |
| ds-nvfp4 | nv-exp7-o0 (in-sample) | 0.8338 | 6,098,433 | 258,747,533 | 16/16 | 27.0 |
| ds-nvfp4 | nv-exp7-o0-cz (in-sample) | 0.8338 | 6,098,433 | 258,747,533 | 16/16 | 27.1 |
| ds-nvfp4 | nv-exp7-o1p-cz (in-sample) | 0.8234 | 2,794,603 | 258,747,533 | 16/16 | 27.1 |
| ds-nvfp4 | nv-code6-o1-cz (in-sample) | 0.8313 | 4,065,102 | 259,998,241 | 16/16 | 27.1 |
| ds-nvfp4 | nv-code12-o1-cz (in-sample) | 0.8313 | 4,065,102 | 259,995,634 | 16/16 | 27.1 |
| ds-mxfp4 | mx-wnr3 | 0.8806 | 2,024,488 | 262,217,623 | 16/16 | 24.5 |
| ds-mxfp4 | mx-wnr3-cz | 0.8532 | 2,024,488 | 253,969,107 | 16/16 | 25.5 |
| ds-quark | mx-wnr3 | 0.8532 | 2,024,488 | 253,969,107 | 16/16 | 25.4 |
| ds-quark | mx-wnr3-cz | 0.8532 | 2,024,488 | 253,969,107 | 16/16 | 25.4 |
| qwen-nvfp4 | nv-exp7-o0 | 0.9455 | 2,973,925 | 38,715,952 | 16/16 | 24.7 |
| qwen-nvfp4 | nv-exp7-o0-cz | 0.9278 | 2,973,925 | 37,938,308 | 16/16 | 25.0 |
| qwen-nvfp4 | nv-exp7-o1-cz | 0.9216 | 2,689,054 | 37,938,308 | 16/16 | 25.0 |
| qwen-nvfp4 | nv-exp7-o1p-cz | 0.9215 | 2,685,263 | 37,938,308 | 16/16 | 24.9 |
| qwen-nvfp4 | nv-code12-o1-cz | 0.9159 | 2,689,054 | 37,703,148 | 16/16 | 24.7 |
| qwen-nvfp4 | nv-exp7-o1-cz-ceil | 0.9213 | 2,690,278 | 37,929,742 | 16/16 | 24.0 |
| qwen-nvfp4 | nv-exp7-o1 | 0.9391 | 2,689,054 | 38,715,952 | 16/16 | 24.9 |
| qwen-nvfp4 | nv-code6-o1-cz | 0.9158 | 2,689,054 | 37,703,310 | 16/16 | 25.4 |
| qwen-nvfp4 | nv-code24-o1-cz | 0.9161 | 2,689,054 | 37,709,932 | 16/16 | 22.5 |
| qwen-nvfp4 | nv-code12-o1 | 0.9336 | 2,689,054 | 38,481,389 | 16/16 | 23.4 |
| qwen-nvfp4 | nv-exp7-o0 (in-sample) | 0.9412 | 2,827,128 | 38,675,616 | 16/16 | 25.6 |
| qwen-nvfp4 | nv-exp7-o0-cz (in-sample) | 0.9237 | 2,827,128 | 37,898,074 | 16/16 | 25.9 |
| qwen-nvfp4 | nv-exp7-o1p-cz (in-sample) | 0.9177 | 2,563,966 | 37,898,074 | 16/16 | 25.8 |
| qwen-nvfp4 | nv-code6-o1-cz (in-sample) | 0.9130 | 2,564,319 | 37,684,816 | 16/16 | 25.7 |
| qwen-nvfp4 | nv-code12-o1-cz (in-sample) | 0.9125 | 2,564,319 | 37,674,550 | 16/16 | 25.7 |
| qwen-mxfp4 | mx-wnr3 | 0.8810 | 321,100 | 36,365,180 | 16/16 | 25.1 |
| qwen-mxfp4 | mx-wnr3-cz | 0.8810 | 321,100 | 36,365,180 | 16/16 | 24.4 |
| glm-nvfp4 | nv-exp7-o0 | 0.9335 | 11,594,800 | 199,312,590 | 16/16 | 24.9 |
| glm-nvfp4 | nv-exp7-o0-cz | 0.9183 | 11,594,800 | 195,873,677 | 16/16 | 25.5 |
| glm-nvfp4 | nv-exp7-o1-cz | 0.9182 | 11,567,582 | 195,873,677 | 16/16 | 25.4 |
| glm-nvfp4 | nv-exp7-o1p-cz | 0.9182 | 11,577,941 | 195,873,677 | 16/16 | 25.5 |
| glm-nvfp4 | nv-code12-o1-cz | 0.9148 | 11,567,582 | 195,110,336 | 16/16 | 25.1 |
| glm-nvfp4 | nv-exp7-o1-cz-ceil | 0.9183 | 11,665,821 | 195,811,357 | 16/16 | 24.2 |
| glm-nvfp4 | nv-exp7-o1 | 0.9334 | 11,567,582 | 199,312,590 | 16/16 | 25.1 |
| glm-nvfp4 | nv-code6-o1-cz | 0.9149 | 11,567,582 | 195,117,490 | 16/16 | 25.3 |
| glm-nvfp4 | nv-code24-o1-cz | 0.9148 | 11,567,582 | 195,112,488 | 16/16 | 25.5 |
| glm-nvfp4 | nv-code12-o1 | 0.9300 | 11,567,582 | 198,549,662 | 16/16 | 25.1 |
| glm-nvfp4 | nv-exp7-o0 (in-sample) | 0.9325 | 11,370,913 | 199,301,465 | 16/16 | 25.6 |
| glm-nvfp4 | nv-exp7-o0-cz (in-sample) | 0.9173 | 11,370,913 | 195,862,541 | 16/16 | 25.9 |
| glm-nvfp4 | nv-exp7-o1p-cz (in-sample) | 0.9167 | 11,234,435 | 195,862,541 | 16/16 | 25.8 |
| glm-nvfp4 | nv-code6-o1-cz (in-sample) | 0.9131 | 11,235,291 | 195,066,698 | 16/16 | 25.8 |
| glm-nvfp4 | nv-code12-o1-cz (in-sample) | 0.9131 | 11,235,291 | 195,050,492 | 16/16 | 25.7 |
| glm-mxfp4 | mx-wnr3 | 0.8891 | 1,232,141 | 188,477,265 | 16/16 | 25.3 |
| glm-mxfp4 | mx-wnr3-cz | 0.8891 | 1,232,141 | 188,477,265 | 16/16 | 25.3 |

## 6. General-purpose baselines on the same slabs

| checkpoint | codec | stored / raw | decode MB/s, 1 thread |
|---|---|---:|---:|
| ds-nvfp4 | zstd -19 | 0.8493 | 862.2 |
| ds-nvfp4 | lz4 default | 0.9289 | 2170.8 |
| ds-nvfp4 | lz4 -12 (HC) | 0.9026 | 2327.5 |
| ds-nvfp4 | WBP1 1 plane | 0.9137 | 173.8 |
| ds-nvfp4 | WBP1 2 planes | 0.9137 | 173.4 |
| ds-mxfp4 | zstd -19 | 0.9263 | 834.9 |
| ds-mxfp4 | lz4 default | 0.9702 | 2340.9 |
| ds-mxfp4 | lz4 -12 (HC) | 0.9521 | 2346.0 |
| ds-mxfp4 | WBP1 1 plane | 0.9595 | 175.5 |
| ds-mxfp4 | WBP1 2 planes | 0.9595 | 174.8 |
| ds-quark | zstd -19 | 0.8990 | 827.5 |
| ds-quark | lz4 default | 0.9702 | 2213.6 |
| ds-quark | lz4 -12 (HC) | 0.9521 | 2372.5 |
| ds-quark | WBP1 1 plane | 0.9333 | 173.4 |
| ds-quark | WBP1 2 planes | 0.9333 | 172.4 |
| qwen-nvfp4 | zstd -19 | 0.9454 | 1454.0 |
| qwen-nvfp4 | lz4 default | 0.9989 | 2490.1 |
| qwen-nvfp4 | lz4 -12 (HC) | 0.9711 | 2321.5 |
| qwen-nvfp4 | WBP1 1 plane | 0.9791 | 175.9 |
| qwen-nvfp4 | WBP1 2 planes | 0.9792 | 176.1 |
| qwen-mxfp4 | zstd -19 | 0.9012 | 722.1 |
| qwen-mxfp4 | lz4 default | 0.9693 | 2385.1 |
| qwen-mxfp4 | lz4 -12 (HC) | 0.9523 | 2493.1 |
| qwen-mxfp4 | WBP1 1 plane | 0.9355 | 171.8 |
| qwen-mxfp4 | WBP1 2 planes | 0.9359 | 172.0 |
| glm-nvfp4 | zstd -19 | 0.9363 | 1622.2 |
| glm-nvfp4 | lz4 default | 0.9863 | 2328.1 |
| glm-nvfp4 | lz4 -12 (HC) | 0.9578 | 2234.3 |
| glm-nvfp4 | WBP1 1 plane | 0.9767 | 177.8 |
| glm-nvfp4 | WBP1 2 planes | 0.9768 | 177.4 |
| glm-mxfp4 | zstd -19 | 0.9011 | 801.3 |
| glm-mxfp4 | lz4 default | 0.9640 | 2313.2 |
| glm-mxfp4 | lz4 -12 (HC) | 0.9501 | 2328.9 |
| glm-mxfp4 | WBP1 1 plane | 0.9359 | 174.3 |
| glm-mxfp4 | WBP1 2 planes | 0.9359 | 174.1 |

## 7. Whole checkpoints (projection from the 16 experts: per-expert mean x routed experts)

| model | format | routed experts | raw GB | best coded ratio (source) | coded GB | saved GB |
|---|---|---:|---:|---|---:|---:|
| deepseek | ds-nvfp4 | 15,360 | 305.77 | 0.8308 (prototype nv-exp7-o1, byte-exact) | 254.04 | 51.72 |
| deepseek | ds-nvfp4 | 15,360 | 305.77 | 0.8284 (prototype nv-exp7-o1p-cz, value-exact) | 253.29 | 52.47 |
| deepseek | ds-mxfp4 | 15,360 | 288.78 | 0.8533 (WNR3 CZ, cross-fold) | 246.40 | 42.37 |
| deepseek | ds-mxfp4 | 15,360 | 288.78 | 0.8807 (WNR3 byte-exact, cross-fold) | 254.32 | 34.46 |
| deepseek | ds-quark | 15,360 | 288.78 | 0.8533 (WNR3 byte-exact, cross-fold) | 246.40 | 42.37 |
| qwen | qwen-nvfp4 | 24,576 | 67.95 | 0.9336 (prototype nv-code12-o1, byte-exact) | 63.44 | 4.51 |
| qwen | qwen-nvfp4 | 24,576 | 67.95 | 0.9158 (prototype nv-code6-o1-cz, value-exact) | 62.23 | 5.72 |
| qwen | qwen-mxfp4 | 24,576 | 64.17 | 0.8810 (WNR3 byte-exact, cross-fold) | 56.54 | 7.64 |
| glm | glm-nvfp4 | 12,096 | 171.23 | 0.9300 (prototype nv-code12-o1, byte-exact) | 159.24 | 11.99 |
| glm | glm-nvfp4 | 12,096 | 171.23 | 0.9148 (prototype nv-code12-o1-cz, value-exact) | 156.65 | 14.58 |
| glm | glm-mxfp4 | 12,096 | 161.72 | 0.8892 (WNR3 byte-exact, cross-fold) | 143.79 | 17.92 |
| glm | glm-nz5 | 12,096 | 157.06 | 0.9605 (entropy estimate only) | 150.86 | 6.20 |

## Decode speed of the repository decoders: speed-ds-quark.txt

| decoder | threads | stored / raw | median ms/record | min to max ms | MB/s |
|---|---|---:|---:|---|---:|
| WNR3 reference wex_nr2_decode | 1 | 0.8529 | 724.61 | 724.27 to 725.23 | 25.9 |
| WNR3 reference wex_nr2_decode | 8 records at once | 0.8529 | 124.18 | 122.59 to 125.11 | 151.4 |
| WNR3 fast, k1 4 streams fused (default) | 1 | 0.8529 | 94.62 | 94.53 to 94.83 | 198.7 |
| WNR3 fast, k1 4 streams fused (default) | 8 per record | 0.8529 | 22.35 | 22.26 to 22.40 | 841.3 |
| WNR3 fast, k1 4 streams fused (default) | 8 records at once | 0.8529 | 22.06 | 21.87 to 22.16 | 852.1 |
| WNR3 fast, k2 gather 16 streams fused | 1 | 0.8529 | 115.15 | 115.15 to 115.86 | 163.3 |
| WNR3 fast, k2 gather 16 streams fused | 8 per record | 0.8529 | 20.86 | 20.81 to 21.00 | 901.2 |


## Decode speed of the repository decoders: speed-glm-mxfp4.txt

| decoder | threads | stored / raw | median ms/record | min to max ms | MB/s |
|---|---|---:|---:|---|---:|
| WNR3 reference wex_nr2_decode | 1 | 0.8891 | 533.64 | 532.77 to 533.72 | 25.1 |
| WNR3 reference wex_nr2_decode | 8 records at once | 0.8891 | 89.62 | 88.88 to 91.82 | 149.2 |
| WNR3 fast, k1 4 streams fused (default) | 1 | 0.8891 | 65.62 | 65.34 to 65.88 | 203.7 |
| WNR3 fast, k1 4 streams fused (default) | 8 per record | 0.8891 | 15.64 | 15.48 to 15.78 | 854.6 |
| WNR3 fast, k1 4 streams fused (default) | 8 records at once | 0.8891 | 15.20 | 15.13 to 15.33 | 879.7 |
| WNR3 fast, k2 gather 16 streams fused | 1 | 0.8891 | 80.54 | 80.49 to 80.80 | 166.0 |
| WNR3 fast, k2 gather 16 streams fused | 8 per record | 0.8891 | 14.66 | 14.54 to 14.80 | 911.9 |


## Decode speed of the repository decoders: speed-qwen-mxfp4.txt

| decoder | threads | stored / raw | median ms/record | min to max ms | MB/s |
|---|---|---:|---:|---|---:|
| WNR3 reference wex_nr2_decode | 1 | 0.8810 | 103.84 | 103.82 to 104.00 | 25.1 |
| WNR3 reference wex_nr2_decode | 8 records at once | 0.8810 | 17.58 | 17.33 to 19.43 | 148.5 |
| WNR3 fast, k1 4 streams fused (default) | 1 | 0.8810 | 13.12 | 12.91 to 13.85 | 199.0 |
| WNR3 fast, k1 4 streams fused (default) | 8 per record | 0.8810 | 3.39 | 3.37 to 3.49 | 769.2 |
| WNR3 fast, k1 4 streams fused (default) | 8 records at once | 0.8810 | 3.10 | 3.04 to 3.13 | 843.6 |
| WNR3 fast, k2 gather 16 streams fused | 1 | 0.8810 | 16.38 | 16.00 to 17.16 | 159.5 |
| WNR3 fast, k2 gather 16 streams fused | 8 per record | 0.8810 | 3.47 | 3.30 to 3.51 | 752.7 |

