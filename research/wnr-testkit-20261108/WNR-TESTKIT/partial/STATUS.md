# partial/ — work of the stopped teams 5, 6, 7 (starting points for the kit workflows)

These folders are **copies** of what the stopped members had written on the original PC (i7-7700K + RTX 3070),
taken on 2026-11-08 when the user stopped all tests there. Only code and small result files were copied:
no records (`*.wnr3`, `*.rec`), no executables/DLLs, no `.npz`/`.bin` over 5 MB, no downloaded checkpoint data.
Every file left out is listed in `SKIPPED_FILES.csv` (member, relative path, bytes, reason).

The code may be incomplete or buggy: it is a starting point, not a validated runner. The workflows in
`../workflows/` tell each member to copy its folder into `results/wf-teamN/<member>/` and work on the copy:
**never edit files under partial/ in place.**

## Changes made while copying (kit version vs original)

Paths of the original PC were removed. Code now reads environment variables (PowerShell: `$env:NAME='value'; ...`
in the same command line, because shell state does not persist between agent commands):

| Variable | Meaning | Default when unset |
|---|---|---|
| `WNR_SAMPLE` | folder with the 160-expert sample (`manifest.json` + `L{LL}_E{EEE}.bin`) = `<data>\v41-sample` | `v41-sample` (relative to the cwd) |
| `WNR_EXTRA` | folder with the 70 extra routed + 6 MTP experts = `<data>\extra` | not used by default |
| `WNR_CEILREF_MODEL` | ceil-ref held-out models `even.npz` | `partial/_shared/ceil-ref/models/even.npz` (gpu-v1 v1enc.py) |
| `WNR_HEADROOM_FEAT` | headroom stride-16 feature files (regenerate with `hfeat.py`, 2.9 GB) | `feat\s16` (relative) |
| `WNR_NVRTC_ARCH` | NVRTC `-arch` for the GPU scripts | detected: `sm_` + device compute capability (sm_120 on the RTX 5090; was hard-coded sm_86) |
| `WNR_NVSMI` | nvidia-smi executable | `nvidia-smi` on PATH (was a DriverStore path) |
| `WNR_GCC_BIN` | folder with gcc.exe for `team6/cpu-simd/src/build.ps1` | gcc on PATH |
| `WNR_T4MAN` | team 4 data manifests (REAP keepset, other/*/manifest.json) | `partial/_shared/team4-data-manifests` |
| `WNR_T4_ORIGIN_UP` | folder with team 4's upstream experts `UP_L*_U*.bin` (re-fetch with `_shared/team4-ceil-origin/fetch_upstream.py`) | `origin-up` (relative) |
| `WNR_CEILREF_RESULTS` | ceil-ref results for `team4-capacity-update/capacity_update.py` | `partial/_shared/ceil-ref/results` |

Other edits: `team6/cpu-simd/src/build.ps1` and `team7/engram-format/scripts/build.bat` rewritten without absolute
paths (`build.ps1` defaults to `-march=native`; the original used `-march=haswell -mtune=skylake`);
`team6/cpu-simd/src/mklist.py` writes the sample path relative to the member folder and stops if it contains a
space (wnrdec.c reads the list with `fscanf %s`); the old `list_*.txt` files were removed (regenerate);
five ceil-ref helper scripts that need folders absent from the kit were removed from `_shared/ceil-ref`
(check_results.py, est_subset_check.py, count_ceil.py, count_pass.py, train_ceil.py). In result/log JSON files
original paths were replaced by `<orig-pc>/...` (provenance only). Docstring/comment references were updated.

## team5

### conf-encoder — DONE on the original PC (records not copied)
Spec-conformant WNR3 v1 encoder (T9 / encoder half of T13). `wnr3c.py` (codec, cites the spec as S:<line>),
`train.py` (WNM3 blob, K = 2 sets: set 0 ceil form, set 1 plain fallback), `encode.py` (records + `records/index.json`),
`corrupt.py`, `synth.py` (ceil_mask 0/1/2/4), `vectors.py` (H32 vectors), `signflip_all.py`.
Results: `encode.log`: 40 held-out records, mean ratio 0.8803159 vs ceil-ref f+g0+d+e 0.8802124 (+0.0103 pp,
CI95 +1331..+2560 B), T9 PASS, all byte-exact, flags 28, ceil_mask 7. Anchor pre-pass stops on L39_E156 w2
(densest pair 121 vs anchor0 120): run with `--override-anchor-stop`. `corrupt.log`: cases A-D all rejected.
`signflip_all.json`: 22-34 structure-accepted flips per stream, 0 accepted by HASH.
Kept: `records/index.json` (record sha256 values!), `records/model.wnm3`, `records/wnd3_descriptor.bin`,
corrupted/synthetic indexes. Regenerate the 40 records (~10-15 min) before use.
**c-decoder** (independent C decoder, T13) was never started.

### gpu-v1 — DONE on the RTX 3070, not run on the 5090
`src/v1enc.py` (CEIL|LENC|HASH streams for the 8 GPU experts), `src/wnr3dec.cu` (wdec3<NT,HASH,LENC,CEIL,DIAG>,
sdec3), `src/v1bench.py` (stages h32, scale, weight, neg, corrupt, lenc, timing, pipe), `src/v1summ.py`.
3070 results: `out/bench_v1_table.txt`, `out/bench_v1_summary.json`, `out/corrupt.json`, `out/lenc_corrupt.json`.
HASH +0.7..+1.8 %, LENC -0.8..0 %, v1 vs CF5 +0.6..+3.4 % (median paired ratios). The kit's `tests/gpu/src_v1`
holds another copy used by the GPU part. Needs T4_ENC_DIR = `tests/gpu/run_gpu_bench.py --encode-only --enc <dir>`.
The 530 MB of encoded streams (`enc/str_*.npz`, `common_v1.npz`) were not copied: re-run v1enc.py.

### headroom — PARTIAL
Greedy static context search done (`out/greedy_b0_base_k4.json`, `out/greedy_b3072_base_k4.json`): base
3.693190 bit/w held-out, + rowp4 3.691817 (-0.032 % of raw), nothing else helped. Not run: `hadapt.py` (adaptive
ceiling), `hmlp.py`, `hscale.py`, `hpath.py`. The 2.9 GB feature files `feat/s16` were not copied: regenerate with
`hfeat.py SAMPLE_DIR OUT_DIR 16 [workers]`. `mxcount.py`, `mxanalyze.py`, `mxceil.py` are the wnr2 tools.

## team6

### cpu-simd — PARTIAL (decoder written, no timing)
`src/wnrdec.c` (Windows; AVX2 interleaved rANS, LUT variants NBSH 5/6/7, OpenMP; modes h32/decode/bench),
`src/wdec_v8.h`, `src/wdec_il.h`, `src/gen_records.py` (copied ceil-ref coder `src/wnr_common.py`, `src/even.npz`),
`src/mklist.py`, `src/build.ps1`. Records for experts idx 1,23,41,63,81,103,121,143 (WNR2 + f+g0+e) were generated
and Python-verified (`out/gen_records.log`, `rec/records.json`, `rec/models/*.u16`) but the .rec files were not
copied: regenerate with `gen_records.py SAMPLE_DIR OUT_DIR 1,23,41,63,81,103,121,143`.
**cpu-path-value** was never started.

### signplane-z — PARTIAL
`src/signplane.py`, `train_z.py` (`out/zmodel.npz`, `out/train_z.json`), `run_z.py`, `summarize_z.py`,
`integrity_z.py`, `make_jobs.py`, `c/zdec.c`, `gpu/wnr2dec_z.cu`, `gpu/zbench.py`, `gpu/ceilbench_copy.py`.
`out/heldout_z.jsonl`: z1de and z2de for 18 of 40 held-out experts (value-exact). Not run: the other 22 experts,
summary, integrity trials, C timing, GPU timing.

## team7

### engram-format — PARTIAL (codec measured)
`scripts/fetch_segments.py`, `prep.py`, `models.py`, `erow.c`, `build.bat`, `codec_run.py`, `work/stubinc/malloc.h`.
`results/codec.json`: 237,568 rows, model 9,894 B, held-out 209.10 B/row (max 234) vs 264 raw, 0 mismatches;
`results/models.json`. Rows (72 MB npz) and the downloaded samples were not copied: re-fetch (team 3 samples with
`_shared/team3-other-tensors/scripts/fetch_tensors.py _shared/team3-other-tensors/tensors_subset.json OUT specs/engram.json`,
segment samples with `fetch_segments.py`), then `prep.py`, `build.bat`, `codec_run.py`.
**disk-plan** was never started.

### nvfp4 — PARTIAL
Scripts `hf_search.py`, `fetch_meta.py`, `fetch_nv.py`, `nvcount.py`, `nvanalyze.py`, `nvscales.py`, `nvpair_ds.py`,
`nvstruct.py`, `nvinspect.py`, `nvcommon.py`; results `res_ds41_1017.json`, `res_qwen36_nv.json`, `res_qwen36_rh.json`,
`hf_search.json`, fetch logs. The count files (`counts/*`, 222 MB) were not copied; nemo3s_nv was counted but not
analysed; glm53_nv never fetched.

### other-models — PARTIAL (stats done, analysis not run)
`fetch7.py`, `stats7.py`, `ana7.py`, `fetch_small.py`, `hf_info.py`, `inspect_meta.py`, logs, and `stats/<checkpoint>/`
(per-expert statistics, 10 MB, complete for 13 checkpoints; v41sample has 75 of 160). `ana7.py` was never run.

## _shared
- `ceil-ref/`: team 4 real-record coder (`wnr_common.py`, `run_ceil.py`, `run_fixed_anchor.py`, `train_models.py`,
  `summarize_ceil.py`, ...), `models/even.npz` + `train_summary.json` (trained on the 80 even-indexed experts),
  `results/heldout_g0.jsonl`, `heldout_summary.json`, `heldout_summary_g0.json` (reference values).
- `team3-capacity-value/`, `team4-capacity-update/`: tok/s / capacity models and their results.
- `team4-ceil-origin/`: fetch_upstream.py / fetch_other.py (HTTP Range, pinned revisions), mxfp.py, other_*.json.
- `team3-other-tensors/`: scripts, specs, results (Engram, FP8, BF16, MTP) and `tensors_subset.json` (header entries
  for every tensor in specs/*.json; replaces the 8.5 MB tensors.json for fetch_tensors.py).
- `team4-data-manifests/`: manifest.json of team 4's extra / origin/up / other/* slices and the REAP keepset.
