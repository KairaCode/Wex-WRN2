# WNR3 in production, 2026-11-10

> **Dated report (2026-11-10).** Since 2026-11-11 the same WNR3-CZ records sit in a WEX4 container, and the engine and helper builds are newer than the ones in "Production configuration": superseded for what runs today by [STATUS.md](../../STATUS.md). The rebuild steps below still apply; the WEX4 upgrade that follows them is [WEX4-SPEC.md](../WEX4-SPEC.md) section 13.1.

Since 2026-11-10 the maintainer's application runs DeepSeek V4.1 REAP-192 from a **WNR3-CZ** container (codec 5, header flags 97) on U: with a full mirror on C:. WEX1 and every WNR2 copy were deleted afterwards, so WNR3 is now the only V4.1 expert container on the machine. Format: [WNR3-SPEC.md](../WNR3-SPEC.md); verification on real records and the G7 proof: [WNR3-VERIFY.md](WNR3-VERIFY.md); container build: [`research/wnr3-20261110/full-build/RESULT-20261110.md`](../../research/wnr3-20261110/full-build/RESULT-20261110.md).

Paths: `lab\...`, `app\...` and `artifacts\...` are relative to the maintainer's tree (`<TREE>`); the run folders named below stay there.

## Production configuration

| item | value |
|---|---|
| model / profile | DeepSeek V4.1 REAP-192 MXFP4, engine profile `v41gpu` (`COLI_FAMILY_ENGINE=deepseek_v4`) |
| container | `U:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex`, 123,306,192,896 B (114.8 GiB), model 0x27e8a1e0, sha256 490e8208... |
| mirror | full byte copy `C:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex`, hash-verified identical (`.verified`) |
| WEX env | `COLI_WEX_PATH` = the U: container, `COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap`, `COLI_WEX_WEIGHTS=38,62`, `COLI_WEX_GPU_DIRECT=2`, `V4_WEX_BATCH=1` |
| engine / helper | installed now: `deepseek_v4.exe` b7c7b6a2 from the predictor-gates engine of 2026-11-10 with the refit predictor gates (section *Later the same night*; same WEX code as the WNR3 branch head); before that the loader-fix engine (2ab584d9, `lab\gpuload-20261110\ws-rel4`). Helper `kaira_wex_cuda.dll` 602e3f4d (feature 16 = WNR3). The switch itself (steps 3-6) used `deepseek_v4.exe` a55a4ce0 from the WNR3 branch head; that engine and the helper are copied in `artifacts\wnr3-deploy-20261110` |
| expected log lines | CUDA backend `deepgemm-sm120`, `Container v3 WNR3 value-exact active`, `Matching mirror container active`, decode on the gpu-pipeline |

Deployment note: after the switch the engine's main branch also got an engine loader fix (`COLI_DSV4_DLL` paths loaded as absolute files, `DSV4_CUDA_REQUIRE=1` makes the GPU mandatory instead of a silent CPU fallback). It is not WEX code; the profile sets `DSV4_CUDA_REQUIRE=1`.

## Results (2026-11-10, RTX 5090)

| measurement | before | after |
|---|---|---|
| container size, V4.1 REAP-192 | WEX1 144,389,472,256 B; WNR2 129,905,360,896 B | **WNR3-CZ 123,306,192,896 B** = 0.854 of WEX1, 0.949 of WNR2 |
| production-identical launch, 1,886-token turn, TTFT | WNR2 U:+C: 326 s (same morning) | **339-352 s** (two launches) |
| same turn, decode | WNR2 U:+C: 2.86 tok/s | **2.92-2.96 tok/s**, same output text as the WNR2 run |
| short run, 7 prompts x 64 tokens, final engine, GPU required | WNR2 reference logits sha a14761eb... | **2.7-3.3 tok/s (mean ~3.0)**, 8 RAM slots/layer, head resident, logits bit-identical to the WNR2 reference |

Measurements that led to the switch:

| A/B (7 prompts x 64 tokens, one pair) | A | B | result |
|---|---|---|---|
| mirror effect ([NOTE-5](../../research/wnr2-fastbench-20261109/NOTE-5-mirror-ab-20261110.md)) | WEX1, U: only, 2.02 tok/s | WNR2, U:+C: mirror 38/62, 2.99 tok/s | **+49% decode**, TTFT -22%, bit-identical; one pair, format and split not isolated |
| pre-switch, WNR3-capable engine (a55a4ce0 + helper 602e3f4d) | WNR2, U:+C: mirror, 3.05 tok/s | WNR3, U: only (no mirror yet), 2.15 tok/s | -30% without the mirror; logits bit-identical on all 8 prompts; 58,568 GPU-direct WNR3 decodes, 0 failures |

Read together: on this machine the second drive is the lever, not the format. WNR3 on U: only (2.15 tok/s) is close to WEX1 on U: only (2.02 tok/s, an earlier run); with the C: mirror it runs at the WNR2-mirror speed while being 5.1% smaller, value-exact and hash-checked per stream. WNR3 + mirror against WNR2 + mirror has no paired A/B; the numbers above (2.92-2.96 vs 2.86 tok/s on the long turn, ~3.0 mean on the short run) are unpaired.

## Switch procedure, as done

1. **Build.** `c/tools/build_nr3.py` container mode, all 40 layers, `--train-lab80 --cz`, source WEX1 on U:, target U:; 986 s. Every record CPU-verified against the canonical-zero WEX1 slab before writing; 7,680 codec-5 records, 0 raw, 0 gate failures.
2. **Read back.** `research/wnr3-20261110/full-build/readback.py`: every table entry read from disk and decoded, 7,680/7,680 equal to the builder's expected-slab sha256; whole-file sha256 into `<container>.verified`.
3. **Pre-switch A/B** (`lab\wnr3-impl-20261110\deploy\ab-wnr2mirror-vs-wnr3u`): WNR2 + mirror vs WNR3 on U: only, results above. Checked: logits bit-identical, 0 WNR3 decode failures.
4. **Mirror copy.** The container copied to `C:\kaira_wex\v41-reap\` under the same basename, sha256 compared with the U: file (identical), `.verified` written. The engine itself checks size, header and entry table of the mirror at open.
5. **Install** (application closed). Backup of the replaced files in `lab\app-before-wnr3-20261110-165802` (with `ROLLBACK.txt`): `deepseek_v4.exe` a55a4ce0 and `kaira_wex_cuda.dll` 602e3f4d into `app\colibri-optimized`; in the application config, profile `v41gpu`, `COLI_WEX_PATH` now names `v4_flash_experts.nr3.wex`; the mirror and split variables stay as they were for WNR2.
6. **production-identical launch** (gateway + the application config env; `lab\wnr3-impl-20261110\deploy\kaira-real`): the expected log lines of the table above, two launches of the 1,886-token turn with equal ids, text and logits bytes, and the same text as the WNR2 run of the morning.
7. **Engine update and short run** (the loader-fix engine, build 2ab584d9, GPU required; `lab\gpuload-20261110\real-rel4`): 7 prompts x 64 tokens, logits bit-identical to the WNR2 reference a14761eb.... Backup before the engine update: `lab\app-before-gpurequire-20261110-185752`.
8. **Deletions.** WEX1 (`U:\kaira_wex\v41-reap\v4_flash_experts.wex`) and every WNR2 copy (U:, C:, E:) deleted. Left on disk: the WNR3 container on U: and C:, and the REAP-192 safetensors on U:.

## Rollback limits

- The predictor-gates engine + gates update rolls back to the loader-fix engine with the backup `lab\app-before-v41gates-20261110-225421` (the container is unchanged).
- The loader-fix engine update rolls back to a55a4ce0 with `ROLLBACK.txt` in `lab\app-before-gpurequire-20261110-185752` (the container is unchanged).
- The engine and helper roll back to WNR2 with `ROLLBACK.txt` in `lab\app-before-wnr3-20261110-165802`, but that restores an application config pointing at a WNR2 container that no longer exists. Without a container the engine reads the safetensors (slower, see the README's M0 table).
- The installed engine also reads WEX1 and WNR2, so a format rollback needs no old binaries: regenerate the container and point `COLI_WEX_PATH` at it.
- WNR3-CZ cannot give back the WEX1 bytes (FP4 -0 is stored as +0; value-exact, not byte-exact). The only byte-exact source left is the safetensors on U:; do not delete them.

## Rebuilding WNR3 from the safetensors

`build_nr3.py` reads WEX1, so WEX1 comes first.

1. **WEX1** (deterministic: a v1 header has no uuid, so a rebuild reproduces sha256 `4C055238...`; [COLIBRI-INTEGRATION.md](../COLIBRI-INTEGRATION.md) 6.2):
   ```powershell
   python -I c\tools\repack_v4_wex.py --model-dir U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 `
       --output <dir>\v4_flash_experts.wex --format-version 1
   ```
   144,389,472,256 B, 115 s on 2026-11-07. Optional byte check against the safetensors: `scripts\verify_wex1.py` (7.1).
2. **WNR3-CZ** (encoder DLL `c/tools/wex_nr3_tools.c`, built in a colibri `c/` layout; see `scripts/build_cpu_check.sh`):
   ```powershell
   python -I c\tools\build_nr3.py <wex_nr3_tools.dll> <dir>\v4_flash_experts.wex container <target>\v4_flash_experts.nr3.wex `
       --layers 0,1,2,...,39 --train-lab80 --cz
   ```
   `--layers` lists all 40 layers. `--train-lab80` picks the lab's 80 training experts from a fixed seed; the 2026-11-10 build produced model 0x27e8a1e0, the same id as the lab CZ model. The builder refuses to start unless the target drive has the raw sum + 1 GiB free (about 135.5 GiB) although the file is 114.8 GiB. Whole-file sha256 reproduction across rebuilds was not tested: compare the model id and the per-record `sha256_expected` in the `.json` report.
3. **Read back**: `python -I research\wnr3-20261110\full-build\readback.py <wex_nr3_tools.dll> <container> <container>.json` (exit 0 = every record decodes to its expected slab).
4. Copy to the mirror drive and compare sha256 (COLIBRI-INTEGRATION.md 7.2), then delete WEX1 if space is needed.

## RAM: Windows memory compression does not help

Measured on 2026-11-10 with Windows' own compressors on 4 KB pages (ratio = compressed / original; 1.000 = no gain):

| engine RAM content | Xpress (what Windows memory compression uses) | Xpress-Huffman |
|---|---|---|
| FP4 expert weights | 1.000 | - |
| FP8 dense weights | 1.000 | 0.87 |
| BF16 head / embedding | 1.000 | 0.85 |
| WNR3 records | 0.998 | - |
| E8M0 scales | 0.31-0.35 | - |

Only the scales compress, and they are about 6% of a slab. Compressed memory therefore cannot stretch the engine's RAM budget. The compressed-record RAM cache measured later the same night (next section) does not either: neither OS compression nor the compressed record cache gives a speed gain at this RAM budget.

## Later the same night (2026-11-10)

### Engine updates

the engine's main branch was fast-forwarded twice after the switch: the loader-fix engine (build 2ab584d9, GPU required, see above) and then the predictor-gates engine (build b7c7b6a2) = the loader-fix engine + the V4.1 online predictor sized by `n_routed_experts`. Neither change touches WEX code: `git diff --stat the WNR3 branch head the predictor-gates engine` over the WEX/WNR files is empty.

the predictor-gates engine was installed with refit V4.1 predictor gates `v41_pred_gates.bin` (c2a280a0), fitted from a decode routing dump (6,834 decode forwards, 273,360 records): d1 recall@8 from 76-79% to **83.7%**, d2 77.5%. the application config, profile `v41gpu`, now sets `V4_NEXT_PREFETCH_GATES=app\colibri-optimized\v41_pred_gates.bin` and `V4_NP_ONLINE=1`. Backup: `lab\app-before-v41gates-20261110-225421`.

| measurement (WNR3 U:+C:) | before (previous install) | after (the predictor-gates engine + refit gates) |
|---|---|---|
| ABBA, 14 pairs | baseline | **+6.55% ± 1.81% decode**, logits bit-identical |
| production-identical 1,886-token turn | 2.83 tok/s | **3.06 tok/s**, logits identical |

### Other engine A/Bs (engine context, not WEX code)

| change | result | decision |
|---|---|---|
| prefetch off | -15% (2.58 vs 3.07 tok/s) | prefetch stays on |
| large usage history (1.72M selections) | -8% vs the small default history | default kept: large counts freeze live VRAM promotion |
| rescaled decode-only usage history | -4% vs the small default history | default kept |
| online predictor `V4_NP_ONLINE=2` | +1.5% ± 4.6% (noise) | stays at 1 |

### Compressed-record RAM cache: no gain, not adopted

`COLI_WEX_RAM_CACHE_GB` keeps WNR3 records compressed in pinned host RAM and decodes them on the GPU on a hit. The new variable-size version (8 equal-count size classes; engine branch `wnr3-ramcache-20261110`, not merged) holds, on the real V4.1 WNR3 record sizes (min 15.25, median 15.30, mean 15.31, max 16.61 MiB), **46 records** in one RAM step of 752,025,600 B, against 43 with max-size slots and 40 decoded slabs (+15% vs slabs), fill efficiency 0.991.

Equal-RAM A/B on V4.1 (A: 8 decoded slots/layer; B: 7 slots + 0.7004 GiB record cache; ABBA): **-0.19% ± 0.72%**, logits bit-identical. The cache served 52 hits against 61,157 misses: the routing spread is far larger than 46 records. Prefill does not use the record cache. Details: [`research/wnr3-20261110/ram-record-cache/RESULT-20261110.md`](../../research/wnr3-20261110/ram-record-cache/RESULT-20261110.md).
