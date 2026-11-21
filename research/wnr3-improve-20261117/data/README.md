# Shared WNR3 research data (read-only for every agent)

Source: the published DeepSeek V4.1 Flash REAP-192 MXFP4 container
(huggingface.co/Kaira666/Kaira-WEX4-Experts, `v41-reap/v4_flash_experts.nr3.wex`, 123,312,910,336 B,
WEX4 upgraded in place from WEX3; production WNR3-CZ records, codec 5, record flags 0x0f, model 0x27e8a1e0).

- `v4_flash_experts.nr3.wex`: a SPARSE local copy of the full file, at its real size. Only these ranges hold
  data (`fetched.json`): page 0, the WEX4 index, the saved WEX3 header page, the WNM2 model, the WEX3 entry
  table, and the 121 sample records. Every other byte reads as zero. Tools that open the container
  (e.g. `wex4/research/*/fetch.py`, `extract.py`) work on it as long as they only touch the sample records.
  Do not run whole-container scans/verifies on it (they read zeros).
- `recs/`: what `wex4/research/wnr4-record/fetch.py ... records` writes: `header.bin` (WEX3 header page),
  `model.bin` (WNM2, 59,408 B), `entries.bin` (WEX3 entry table, 7,680 x 72 B, struct `<QIII6H6H3I3II`,
  index = layer*192 + expert), and `rec_LL_EEE.bin` = the stored record bytes (total_bytes) of (layer, expert).
- Sample (`sample.txt`): 3 experts per layer for all 40 layers, experts
  `(37*l+11)%192`, `(37*l+101)%192`, `(53*l+7)%192` (sorted), plus L39 E120 (the outlier record). 121 records,
  1.94 GB. All 121 decode with the CPU reference (`check_decode.py`; stream hashes pass).
- Suggested held-out split for anything trained: TRAIN = experts `(37*l+11)%192` and `(37*l+101)%192` of each
  layer (80 records); TEST = `(53*l+7)%192` of each layer (40) + L39 E120. Note that the production model was
  trained on a different 80 experts (`build_nr3.py --train-lab80`), so it is (mostly) out-of-sample on all 121.
- `libwex_nr3_tools.so`: `Wex-WRN2/c/tools/wex_nr3_tools.c` built for Linux (`gcc -O2 -shared -fPIC -I.`).
  Exports `wex_nr3_decode`, `wex_nr3_prepare`, `wex_nr3_count_expert`, `wex_nr3_build_model`,
  `wex_nr3_encode_record`, `wex_nr3_verify`, ... (see the source for signatures).
- `rangefetch.py records L:E,...` can fetch more records into the sparse file (about 15-50 MB/s); add them to
  `recs/` with fetch.py. Only do it if your question really needs more data, and say so in your report.

Raw record (decoded slab) = 18,800,640 B: scales w1, w2, w3 then packed FP4 weights w1, w2, w3.

## Rebuilding \<TMP>/wnr3-data

    mkdir -p <TMP>/wnr3-data && cp rangefetch.py check_decode.py <TMP>/wnr3-data/ && cd <TMP>/wnr3-data
    python3 rangefetch.py meta
    python3 rangefetch.py records "$(cat <this folder>/sample.txt)"
    python3 <Wex-WRN2>/wex4/research/wnr4-record/fetch.py v4_flash_experts.nr3.wex recs records "$(cat <this folder>/sample.txt)"
    (cd <Wex-WRN2>/c && gcc -O2 -shared -fPIC -I. tools/wex_nr3_tools.c -o <TMP>/wnr3-data/libwex_nr3_tools.so)
    python3 check_decode.py      # 121 decoded, failures: []
