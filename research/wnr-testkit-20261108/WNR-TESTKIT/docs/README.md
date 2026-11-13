# docs: reference documents and the old-path map

| file | what |
|---|---|
| `WNR3-DRAFT-v2.1.md` | WNR3 spec DRAFT 0.2.1 (cite as S:<line>); tests T1..T14 in section 11 |
| `WNR-RICERCA-2026-11-07.md` | consolidated research report (Italian) |
| `measured-v41.md` | raw measurements on the V4.1 sample |
| `WNR2-REFERENCE.md` | internal WNR2 reference (cite as REF:<line>) |
| `wnr2-brief.md` | short WNR2 brief |

These documents were written on the original research PC and cite its absolute paths as provenance. **Those paths do
not exist on the PC running the kit.** Use the table below; any old path not listed is provenance only and its content
is not in the kit (the numbers quoted in the docs are what remains of it).

| old path (original PC) | in the kit |
|---|---|
| `<RESEARCH-PC>\team4\ceil-ref\` | `partial\_shared\ceil-ref\` (coder, `models\even.npz`, `results\heldout_g0.jsonl`, summaries); the coder is also `tests\cpu\lib\wnr_common.py` (+ team 6 `set_chunks()`) and `tests\audit\wnr_common_cr.py` (byte copy) |
| `<RESEARCH-PC>\team4\ceil-gpu\` | `tests\gpu\src_bench\` (wnr2dec.cu, t4enc.py = ceilenc.py as a library), `tests\gpu\ref\bench_3070\` (best_final.txt, ratios_final.json, encode reports) |
| `<RESEARCH-PC>\team2\gpu-proto\` | `tests\gpu\src_bench\wnr2fmt.py`, `wnr2dec.cu`; pinned venv versions in `setup\requirements-gpu.txt` |
| `<RESEARCH-PC>\team2\zero-safety\`, `...\team2\zero-refuter\` | `tests\gpu\src_zero\` (colibri kernels copy, extra kernels), `tests\gpu\ref\zero_3070\` |
| `<RESEARCH-PC>\team2\audit\`, `...\team2\ans-prior-art\` | `tests\audit\` (kit audit built on team 5 audit-ceil); team 2 baseline numbers in `tests\audit\reference\team2_wnr2_corrupt_audit.json` |
| `<RESEARCH-PC>\team4\capacity-update\` | `partial\_shared\team4-capacity-update\` |
| `<RESEARCH-PC>\team3\capacity-value\` | `partial\_shared\team3-capacity-value\` |
| `<RESEARCH-PC>\team4\ceil-origin\` | `partial\_shared\team4-ceil-origin\` |
| `<RESEARCH-PC>\team3\` other-tensors | `partial\_shared\team3-other-tensors\` (scripts, specs, results, tensors_subset.json) |
| `<RESEARCH-PC>\team4\spec-update\WNR3-DRAFT-v2.1.md` | `docs\WNR3-DRAFT-v2.1.md` (DRAFT 0.2 and team 3's DRAFT 0.1 are not in the kit) |
| `D:\Desktop\WNR2-REFERENCE.md` | `docs\WNR2-REFERENCE.md` |
| `<RESEARCH-PC>\v41-sample\` (160 experts + manifest.json) | `<data>\v41-sample\` from `setup\fetch_data.py`; manifest `setup\manifests\v41-sample.json` |
| `<RESEARCH-PC>\team4-data\extra\` (70 routed + 6 MTP) | `<data>\extra\` from `setup\fetch_data.py --extra`; manifest `setup\manifests\extra.json` |
| `<RESEARCH-PC>\team4-data\origin\`, `...\other\` manifests | `partial\_shared\team4-data-manifests\` (manifests only, no data) |
| `<RESEARCH-PC>\team5\<member>\`, `...\team6\<member>\`, `...\team7\<member>\` | `partial\team5\`, `partial\team6\`, `partial\team7\` (copied members only; see `partial\STATUS.md` and `partial\SKIPPED_FILES.csv`) |
| temporary folder `<TMP>\wnr2\` (fetch_experts.py, mxcount.py, mxanalyze.py, mxceil.py) | `setup\fetch_data.py` (replaces fetch_experts.py); `partial\team5\headroom\scripts\` has copies of mxcount.py, mxanalyze.py, mxceil.py |
| `<RESEARCH-PC>\v41-results\` (analysis.json, ceil.json, ...) | not in the kit (numbers quoted in `measured-v41.md`) |
| `<RESEARCH-PC>\team4\critic\`, `...\team4\ceil-refuter\`, `...\team3\wnr3-spec\`, `...\team3\expert-stats\` | not in the kit (provenance only) |
