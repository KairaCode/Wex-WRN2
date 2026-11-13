# setup: environment, data, run-all

All commands from the kit root in PowerShell. Nothing here needs admin rights.

## 1. Python

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r setup\requirements-gpu.txt   # or requirements-cpu.txt without GPU
```

GPU: RTX 5090 (sm_120) needs NVIDIA driver R570+ (nvidia-smi shows CUDA Version >= 12.8) and the CUDA 12.8+
NVRTC wheel pinned in requirements-gpu.txt (12.9.86). CuPy wheel: `cupy-cuda12x`. Python: 3.10+ for the CPU part,
3.11+ for the GPU part (3.12+ for the exact reference pins; the 3.10 GPU fallback line is untested). This is the
only GPU requirements file and `venv` the only venv name the kit uses (`tests\gpu\requirements-gpu.txt` forwards here).

## 2. Environment check (read-only, no kernel launches)

```powershell
.\venv\Scripts\python.exe -I setup\check_env.py
```

Reports OS, CPU name, cores, AVX2/AVX-512 (Windows: IsProcessorFeaturePresent by default; `--cpuid` adds the full
cpuid probe, a few bytes of machine code run in a child process from a W^X page; Linux: /proc/cpuinfo), RAM, GPUs, CUDA/NVRTC versions, an NVRTC
compile-only probe for the local arch and sm_120, the C compilers it finds (cl, gcc, clang, nvcc, MSVC via
vswhere), free disk, the data already present, and which runners can run.
Output: `results\check_env\summary.json`.

## 3. Data (about 3.0 GB, plus 1.4 GB with --extra)

```powershell
python -I setup\fetch_data.py                    # 160 experts over HTTP Range from the pinned HF revision
python -I setup\fetch_data.py --extra            # + 70 extra routed + 6 MTP experts
python -I setup\fetch_data.py --local-shards D:\path\to\checkpoint --extra   # copy from local safetensors shards
python -I setup\fetch_data.py --verify-only --extra   # sha256 check, read-only, no network
python -I setup\fetch_data.py --dry-run --extra       # print the plan only
```

Layout: `data\v41-sample\L{LL}_E{EEE}.bin` (160), `data\extra\L{LL}_E{EEE}.bin` + `M{LL}_E{EEE}.bin` (76). Each
expert is 18,800,640 bytes: w1.scale, w2.scale, w3.scale (E8M0), then w1.weight, w2.weight, w3.weight (packed FP4),
copied exactly from the shards. A `manifest.json` (copy of `setup\manifests\*.json`) is written next to the data.
Interrupted downloads resume from the `.part` file. Use `--data <dir>` for another drive; then pass the same
`--data` (or `-Data`) to the runners. `--local-shards` takes the folder holding `model-000NN-of-00048.safetensors`
of `patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` at revision `4ab649b2...`; a different checkpoint
fails the dtype/shape check or the sha256 check.

## 4. Run everything

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Quick        # first pass, minutes
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1               # full
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Fetch -Extra -TimeoutMin 240
.\setup\run_all.ps1 -ListOnly            # show the runners and their order
.\setup\run_all.ps1 -Only ceil -NoGpu    # filter by name/path substring; skip GPU runners
.\setup\run_all.ps1 -Quick -All          # also the EXPERIMENTAL optional runners (gpu_v1_profile)
powershell -ExecutionPolicy Bypass -File .\setup\pack_results.ps1         # zip results\ to send back
```

Default run order (`-ListOnly`): heldout_configs 20, wnr2_g0 21, chunk_sweep 23, train_curve_mtp 24 (needs
`-Extra` data), audit 40, nvrtc_check 55 (compile only: needs CuPy/NVRTC, no GPU, no data), zero_bittest 60,
gpu_bench 65; gpu_v1_profile 70 is optional (EXPERIMENTAL) and runs only with `-All` or `-Only gpu_v1`.
`-NoGpu` skips the runners whose needs contain `gpu` (nvrtc_check still runs). A full run takes a few hours
(audit alone 1-2 h, estimated); give `-TimeoutMin` 240 or more, or leave it at 0 (no timeout).
`pack_results.ps1` leaves out .bin/.part files, files over `-MaxFileMB` and the regenerable cache folders (enc,
v1enc, t4enc, _cache, records, parts, feat, downloads), all listed in SKIPPED_LARGE_FILES.txt; `-IncludeCaches` keeps
the caches. `collect_results.py` also merges the workflow outputs (results\wf-*\team*_summary.json and the members'
member_result.json).

run_all.ps1 runs check_env, (fetch_data if `-Fetch` or `-LocalShards`), fetch_data `--verify-only`, then each
runner as `python -X utf8 -u -I <runner> --root <kit> [--data] [--out] [--quick]`, keeps going when one fails, saves
stdout/stderr under `results\_runlogs\`, and merges every `results\*\summary.json` into `results\ALL_RESULTS.json`
(setup\collect_results.py).

Runner discovery: if `runners.txt` exists in the kit root it lists runner paths in order (one per line, `#`
comments). Otherwise every `run_*.py` with a `__main__` block outside setup/results/data/docs/partial/workflows/venv
is a runner, and so is any .py with a module-level literal `KIT_META = {'name': ..., 'order': 20, 'needs': [...]}`
(read with ast, not executed), which sets its name, order, needs (`data`, `extra`, `gpu`, `nvrtc`, `cc`),
`optional` and time estimates. Every kit runner (tests\cpu\*.py, tests\audit\run_audit.py, tests\gpu\run_*.py) has
one; without it the values are guessed from the path (a `gpu` folder means GPU) and the source. Folders holding a
`pyvenv.cfg` (any virtualenv) are skipped. `.\setup\run_all.ps1 -ListOnly` shows the result.
