# WNR kit: development workflows (Claude Code)

Some of the remaining WNR2/WNR3 work is development, not a fixed test: an independent decoder has to be written,
a SIMD decoder tuned for the CPU it runs on, a storage format designed. Those tasks are packaged as three
**Claude Code workflow scripts**, to be launched on the PC that received the kit (expected: the KAIRA machine,
Windows, RTX 5090 sm_120). They continue the stopped teams 5, 6 and 7 of the original PC. Their partial work is in
`partial/` (see `partial/STATUS.md`).

Members whose work is already covered by a plain runner of the kit were dropped from the workflows:

| Stopped member | Now covered by |
|---|---|
| team 5 wnr2-g0 | `tests/cpu/wnr2_g0.py` |
| team 5 audit-ceil | `tests/audit/run_audit.py` |
| team 5 mtp-training | `tests/cpu/train_curve_mtp.py` |
| team 6 chunk-size | `tests/cpu/chunk_sweep.py` |

## Before you launch

1. Unzip the kit, e.g. to `D:\wnr-testkit` (a path **without spaces** is safest: team 6's C decoder reads file lists
   with `fscanf %s`).
2. In PowerShell, from the kit root:
   ```powershell
   python -I setup\check_env.py --root D:\wnr-testkit --data D:\wnr-testkit\data
   python -I setup\fetch_data.py --root D:\wnr-testkit --data D:\wnr-testkit\data           # 160-expert sample, 3.0 GB
   python -I setup\fetch_data.py --root D:\wnr-testkit --data D:\wnr-testkit\data --extra   # optional, +1.4 GB
   ```
   (or `--local-shards <dir>` if the V4.1 safetensors shards are on a local drive; see `setup/README.md`).
3. Python 3.10+ with numpy (3.11+, better 3.12+, for the GPU members: CuPy + NVRTC >= 12.8 from
   `setup/requirements-gpu.txt`, in the kit's `venv`);
   a C compiler for c-decoder, cpu-simd, signplane-z and engram-format (MSVC Build Tools, or MSYS2/mingw-w64 gcc;
   `check_env.py` lists what it finds). Nothing is installed by the workflows.
4. Run **one workflow at a time**, and not while `setup\run_all.ps1` is running (the CPU and GPU timings assume a
   quiet machine). Inside a workflow, members run only the specific kit runners their task names (gpu-v1:
   run_gpu_v1_profile.py and run_gpu_bench.py --encode-only; signplane-z: run_gpu_bench.py --encode-only), always
   with `--out` in their own folder under `results\wf-teamN\`. Recommended order: team 5, team 6, team 7
   (team 7's disk-plan reads team 6's cpu-path-value result if it exists).
5. The docs cite the original PC's absolute paths as provenance; `docs\README.md` maps them to kit locations (the
   workflow prompts tell the agents to use it).

## How to launch

Open Claude Code in the kit root and ask it to run the workflow file with an `args` **object** (not a JSON string),
for example:

```
Run the workflow script workflows\wnr-kit-team5-conformance.js with
args {"root": "D:\\wnr-testkit", "data": "D:\\wnr-testkit\\data"}
```

which corresponds to the tool call

```
Workflow({ scriptPath: "D:\\wnr-testkit\\workflows\\wnr-kit-team5-conformance.js",
           args: { root: "D:\\wnr-testkit", data: "D:\\wnr-testkit\\data" } })
```

To resume an interrupted run: `Workflow({ scriptPath: "<same path>", args: <same args>, resumeFromRunId: "<runId>" })`
(finished agents are replayed from cache). If your Claude Code has no Workflow tool, each member's task text is a
plain string constant in the script (`CONF`, `CDEC`, `GPUV1`, `HEAD`, `SIMD`, `VAL`, `SIGN`, `ENGRAM`, `PLAN`,
`NVFP4`, `OTHER`, preceded by the shared `CTX`): paste CTX + the task into a normal session.

### Common args

| Key | Default | Meaning |
|---|---|---|
| `root` | **required** | absolute path of the kit root |
| `data` | `<root>\data` | data folder (`v41-sample\`, `extra\`, `downloads\`) |
| `out` | `<root>\results\wf-teamN` | results folder of the workflow (inside `results\` so `setup\pack_results.ps1` zips it) |
| `python` | `python` | interpreter with numpy (and CuPy for GPU members) |
| `members` | all | subset to run, e.g. `["c-decoder"]` |
| `machine` | — | free text about the PC passed to every agent (CPU, RAM, drives) |
| `kaira` | — | folder with KAIRA sources, read-only, if available |
| `gpu` | `true` | team 5/6: `false` skips GPU work |
| `drives` | — | team 6/7: free text about drive letters and roles (e.g. `U: expert NVMe`) |
| `downloads` | `true` | team 7: `false` forbids all network access |
| `download_gb` | `{nvfp4: 1.5, "other-models": 1.5, "engram-format": 1.0}` | team 7: per-member download caps (GB) |

Linux works with `root`/`data` given as POSIX paths (agents are told to read `\` as `/`).

## The workflows

Every member copies its `partial\...` folder into `<out>\<member>\` and works only there, writes
`<out>\<member>\member_result.json` (the object it returns), and tags each claim as measured / read-in-code /
source / inferred. The critic re-checks the numbers and writes `<out>\teamN_summary.json` (all member results +
review) and `<out>\teamN_summary.md` (one page, English). Nothing outside `<out>` (and, for team 7,
`<data>\downloads\<member>\`) is written; `docs\`, `tests\`, `partial\` and the sample are never modified.

### 1. `wnr-kit-team5-conformance.js` — 5 agents

| Member | Task | Inputs | Outputs (under `results\wf-team5\`) | Needs |
|---|---|---|---|---|
| conf-encoder | regenerate the spec-conformant WNR3 v1 encoder's 40 held-out records, model blob, index, corrupted and synthetic sets from `partial\team5\conf-encoder`; reproduce T9 (ref 0.8803159 vs 0.8802124, +0.0103 pp) and the record sha256 values; list spec ambiguities | sample, `partial\_shared\ceil-ref` (even.npz, heldout_g0.jsonl), spec | `conf-encoder\records\` (~660 MB), logs, member_result.json | CPU |
| c-decoder | write an **independent** C11 decoder from the spec text only (T13), decode every record byte-exact vs the original .bin, check corrupted cases, time decode with/without ceil inverse and HASH | spec, encoder records/index (runs after conf-encoder) | `c-decoder\` source, build script, results | C compiler |
| gpu-v1 | WNR3 v1 GPU profile CEIL\|LENC\|HASH on the RTX 5090: byte-exact, negative/corruption controls, paired timing vs CF5/A259, regs/smem/occupancy on sm_120, launch grid for 170 SMs | `tests\gpu` (run_gpu_bench.py `--encode-only` for T4_ENC_DIR; src_v1) or `partial\team5\gpu-v1`; 3070 reference in `partial\team5\gpu-v1\out` | `gpu-v1\` bench JSON/tables | RTX 5090, CuPy |
| headroom | remaining ratio headroom after ceil-form: regenerate features, reproduce the greedy result (-0.032 % of raw), then adaptive ceiling, MLP bound, scale contexts, exact eval with CIs, budget/causality checks | sample, `partial\team5\headroom\scripts` | `headroom\` (feature files 2.9 GB + JSON) | CPU, ~4 GB disk |
| critic | re-check all numbers, decoder independence, GPU method, leakage | member results | `team5_summary.json`, `team5_summary.md` | — |

Concurrency: conf-encoder -> c-decoder run in sequence; gpu-v1 and headroom run alongside (3 at once).
Expected runtime: 2-5 h wall clock (conf-encoder ~30-45 min of compute; c-decoder 1-3 h of development;
gpu-v1 ~1-2 h incl. the 5090 grid; headroom ~1.5-3 h, feature extraction and adaptive runs dominate).
Disk: ~5 GB under `results\wf-team5`.

### 2. `wnr-kit-team6-cpu-path.js` — 4 agents

| Member | Task | Inputs | Outputs (under `results\wf-team6\`) | Needs |
|---|---|---|---|---|
| cpu-simd | regenerate 16 test records (8 held-out experts, WNR2 + f+g0+e), build and tune the AVX2 (and AVX-512 if present) interleaved rANS decoder `wnrdec.c`, verify byte-exact, measure GB/s and ms/record for scalar / SIMD / 1..N threads, with and without ceil inverse and HASH; compare with NVMe read times | sample, `partial\team6\cpu-simd` | `cpu-simd\` records (~270 MB), binaries, timing JSON | C compiler (gcc or MSVC) |
| cpu-path-value | explicit tok/s model of the CPU-only V4.1 path for WEX1/WNR2/WNR3/WNR3z using the measured decode speed of THIS CPU, RAM cache and drive table; break-even decode GB/s; recommendation (runs after cpu-simd) | REF, cpu-simd result, `partial\_shared\team3-capacity-value`, `team4-capacity-update` | `cpu-path-value\` model + JSON | — (read-only drive info) |
| signplane-z | compacted sign plane layout (Z1/Z2): finish the 40-expert held-out encode with CIs, integrity trials, C timing of CF/D/Z1/Z2, GPU timing on the 5090 vs D/CFD5/CF5/A259 | sample, `partial\team6\signplane-z`, `partial\_shared\ceil-ref`, `tests\gpu\run_gpu_bench.py --encode-only` | `signplane-z\` JSONL/JSON | CPU, C compiler, RTX 5090 (optional) |
| critic | re-check numbers, load labelling of timings, model parameters | member results | `team6_summary.json`, `team6_summary.md` | — |

Concurrency: cpu-simd -> cpu-path-value in sequence, signplane-z alongside; signplane-z pauses heavy jobs while
`cpu-simd\TIMING_IN_PROGRESS` exists. Expected runtime: 2-4 h (cpu-simd 1-2.5 h; signplane-z encode of 40 experts
x 4 configs ~30-60 min plus timing). Disk: ~1.5 GB.

### 3. `wnr-kit-team7-portability.js` — 5 agents

| Member | Task | Inputs | Outputs (under `results\wf-team7\`) | Needs |
|---|---|---|---|---|
| engram-format | lossless row-random-access format for the 202.76 GB Engram tables: re-fetch row samples, rebuild erow.dll, reproduce 209.10 B/row (of 264), compare fixed slots / per-row rANS + index / row groups / byte planes, time decode per token (48 rows) on this CPU, reads per token; recommend one | `partial\team7\engram-format`, `partial\_shared\team3-other-tensors` (tensors_subset.json, specs) | `engram-format\` JSON; downloads in `data\downloads\engram-format\` (<= 1.0 GB) | network (HF, public), C compiler |
| disk-plan | placement plans for the full V4.1 deployment that fit the KAIRA drives (REF drive table; live free space if this is the KAIRA machine, read-only), with tok/s from team 4's capacity model; lists files the user would have to move, touches nothing (runs after engram-format) | REF, `partial\_shared\team4-capacity-update`, engram result, team 6 cpu-path-value if present | `disk-plan\` ranked plans | — |
| nvfp4 | WNR on NVFP4 (16-blocks, E4M3 scales): check the 3 copied results, fetch/analyse the missing families (nemo3s_nv, glm53_nv), entropy with WNR-style contexts, block-max / double-rounding artifacts, sign of zero, held-out ratio of a WNR-NV variant, WNR-NV sketch | `partial\team7\nvfp4`, sample | `nvfp4\` JSON; downloads in `data\downloads\nvfp4\` (<= 1.5 GB) | network |
| other-models | WNR3 on other MXFP4 MoE checkpoints: finish stats for the V4.1 sample, run ana7.py on the 14 copied stats sets, QAT vs PTQ, ceil gate, held-out and transfer ratios, per-checkpoint training needed | `partial\team7\other-models\stats`, `partial\_shared\team4-ceil-origin`, sample | `other-models\` table/JSON; optional downloads (<= 1.5 GB) | network optional |
| critic | re-check numbers, download caps, leakage, that no file was touched | member results | `team7_summary.json`, `team7_summary.md` | — |

Concurrency: engram-format -> disk-plan in sequence, nvfp4 and other-models alongside; at most 4 HTTP connections
per member; no GPU. Expected runtime: 2-4 h, dominated by downloads (up to ~4 GB in total at the caps) and the
nvfp4/other-models analyses. Disk: up to ~5 GB under `data\downloads` + ~1 GB of results.
With `"downloads": false` the team works from the copied stats/results only (~1-2 h).

## Sending the results back

```powershell
powershell -ExecutionPolicy Bypass -File setup\pack_results.ps1 -MaxFileMB 10
```
zips `results\` (plain runners and `wf-team5/6/7`; downloads live in `data\` and are not included). `-MaxFileMB 10`
leaves out the large regenerable intermediates (16 MB records, 18 MB feature files; listed in
SKIPPED_LARGE_FILES.txt); cache folders named records, feat, enc, v1enc, t4enc, parts, _cache are left out by default
anyway. The `member_result.json`, `teamN_summary.*`, logs and JSON results are what is needed;
`setup\collect_results.py` merges `teamN_summary.json` and every `member_result.json` into `results\ALL_RESULTS.json`.

## Notes and limits

- The runtimes are estimates from the original PC's measured script timings plus agent development time; agent
  token use is high (a team typically runs several hours of agent work).
- The scripts were syntax-checked (`node --check`) and dry-run with stubbed agents on the original PC; they have
  never been executed for real. The partial code is the stopped teams' work and may contain bugs: each member is
  told to verify before reusing.
- No member is allowed to modify `docs\`, `tests\`, `partial\` or the sample, to install anything system-wide,
  to post anything, push to git or use credentials. Only team 7 may download (public, ungated, pinned Hugging Face
  revisions, within the caps).
