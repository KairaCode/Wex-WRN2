# WNR2 / WNR3 MXFP4 test kit: start here

Portable tests for the WNR2 / WNR3 lossless format research on DeepSeek V4.1 REAP-192 MXFP4 expert records. Made
to run on another PC (expected: Windows, RTX 5090 sm_120). Nothing in the kit was run at full size where it was
built: only compile checks and short CPU smokes. Every runner compares with the original PC's values in its
`ref\` / `reference\` folder.

All commands: Windows PowerShell, from the kit root (the folder holding this file). On Linux use `/` and
`venv/bin/python`.

## 1. Python and packages (one venv for everything)

Python 3.12+ recommended. The CPU part needs 3.10+, the GPU part 3.11+ (3.12+ for the exact reference pins).

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r setup\requirements-gpu.txt    # CPU + GPU (CuPy, NVRTC wheels)
# no NVIDIA GPU:  .\venv\Scripts\python.exe -m pip install -r setup\requirements-cpu.txt
```

RTX 5090: NVIDIA driver R570 or newer (`nvidia-smi` shows CUDA Version >= 12.8). No CUDA toolkit install needed.

## 2. Environment check (read-only, launches no GPU kernel)

```powershell
.\venv\Scripts\python.exe -I setup\check_env.py
```

## 3. Data (3.0 GB, plus 1.4 GB with --extra)

```powershell
.\venv\Scripts\python.exe -I setup\fetch_data.py --extra                       # HTTP Range from the pinned HF revision
.\venv\Scripts\python.exe -I setup\fetch_data.py --local-shards D:\path\to\checkpoint --extra   # or from local shards
.\venv\Scripts\python.exe -I setup\fetch_data.py --verify-only --extra         # sha256 check only
```

Or copy the expert `.bin` files you already have into `data\v41-sample\` (160 files `L??_E???.bin`) and
`data\extra\` (76 files); a `manifest.json` next to them is optional, the kit's own `setup\manifests\*.json` are the
reference. Another drive: add `--data D:\wnr-data` here and `-Data D:\wnr-data` to run_all.

## 4. Run the tests

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -ListOnly      # what runs, in which order
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Quick         # first pass (estimated 30-60 min, not measured)
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1                # full (hours; audit alone 1-2 h)
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Quick -All    # + EXPERIMENTAL gpu_v1_profile
```

Runners (all discovered through their `KIT_META`): CPU `heldout_configs`, `wnr2_g0`, `chunk_sweep`,
`train_curve_mtp` (needs the extra experts), `audit`; GPU `nvrtc_check` (compile only), `zero_bittest`, `gpu_bench`;
optional `gpu_v1_profile`. `-NoGpu` skips the GPU ones except nvrtc_check; `-Only` / `-Skip` filter by name.
Each writes `results\<test>\summary.json` + `<test>.log` and prints `RESULT <test> PASS|FAIL|DONE ...`;
run_all merges everything into `results\ALL_RESULTS.json` and keeps going when a step fails.

## 5. Send the results back

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\pack_results.ps1
```

Creates `results_<PC>_<date>.zip` in the kit root, without data files and regenerable caches (`-IncludeCaches`
keeps the caches).

## 6. Research workflows (Claude Code, optional, after the tests)

Three Claude Code workflows continue the stopped teams' work (`workflows\README_WORKFLOWS.md`): run one at a time,
team5, team6, team7, never together with run_all.ps1. Their outputs go to `results\wf-teamN\` and are packed and
collected too.

## Where to read more

| folder | README |
|---|---|
| `setup\` | environment, data, run_all, pack (`setup\README.md`) |
| `tests\cpu\` | CPU record-coder tests, pass criteria, reference values (`tests\cpu\README.md`) |
| `tests\audit\` | T3 corruption audit (`tests\audit\README.md`) |
| `tests\gpu\` | GPU decoder bench, canonical-zero bit test, NVRTC check (`tests\gpu\README.md`, `CHECKLIST-kaira-gpu-paths.md`) |
| `workflows\` | Claude Code workflows (`workflows\README_WORKFLOWS.md`) |
| `partial\` | stopped teams' work (`partial\STATUS.md`) |
| `docs\` | spec, report, measurements, and the map from the original PC's paths to kit locations (`docs\README.md`) |
