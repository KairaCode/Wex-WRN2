# tests/gpu - portable GPU tests (WNR2 / WNR3 MXFP4 format research)

Runners for the GPU part of the WNR3 draft test plan (section 11: T7 GPU byte-exact, T9 speed, T4/5.6 canonical-zero
bit test). Built from team 2 `gpu-proto` + `zero-safety` + `zero-refuter`, team 4 `ceil-gpu`, team 5 `gpu-v1` (partial).
They were **not run** on the PC where the kit was built (only compile / short CPU smoke checks, see the end); the
reference numbers come from the RTX 3070 (sm_86) runs of teams 2/4/5 and are in `ref/`.

| runner | what | GPU | full run | `--quick` |
|---|---|---|---|---|
| `run_nvrtc_check.py` | compile-only smoke: every kernel, NVRTC to PTX (`compute_XY`) and cubin (`sm_XY`, ptxas registers/spills) for 86, 120 and the local device | not needed | ~3-5 min | ~1 min |
| `run_gpu_bench.py` | encode 8 experts (CPU, cached), GPU decode byte-exact vs the ORIGINAL safetensors bytes for every stream and config, paired timing, regs/smem per kernel, table vs 3070 | yes | ~10-20 min (+3 min encode once) | ~3-5 min |
| `run_zero_bittest.py` | canonical zero (-0 -> +0) bit test: 10 colibri CUDA FP4 kernels + FP16 tensor-core GEMM + mma tiles (+ sm_120 FP4 MMA, experimental), positive/negative controls | yes | 160 experts, ~4-10 min | 8 experts, ~1 min |
| `run_gpu_v1_profile.py` | **EXPERIMENTAL, untested**: WNR3 v1 profile CEIL\|LENC\|HASH decoded in-kernel (team 5) | yes | ~20-40 min | ~10 min |

`CHECKLIST-kaira-gpu-paths.md`: what this kit cannot test without the KAIRA/colibri build (DeepGEMM sm120 FP4,
cuBLASLt MXFP8 `tc_fp4_matvec`, Vulkan, end-to-end per-step logits memcmp), as a fill-in checklist.

## 1. Setup (Windows PowerShell; Linux: same commands with `/` and `venv/bin/python`)

Python 3.11+ for the GPU part (3.12+ for the exact pins; references were measured with 3.14.7), an NVIDIA driver
with CUDA 12.9 support (RTX 5090 / sm_120 needs NVRTC >= 12.8 and a Blackwell driver, R570+). No CUDA toolkit install
is needed: NVRTC comes as a pip wheel. The whole kit uses ONE venv named `venv` in the kit root (setup\run_all.ps1
picks it up automatically) and ONE requirements file, `setup\requirements-gpu.txt` (it includes the CPU set);
`tests\gpu\requirements-gpu.txt` only forwards to it.

```powershell
cd <kit root>
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r setup\requirements-gpu.txt
.\venv\Scripts\python.exe -I -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```
`setup\requirements-gpu.txt` pins the exact versions of the venv the 3070 references were measured with (CuPy 14.2.0,
NVRTC 12.9.86), with python_version markers for older interpreters. The warning "CUDA path could not be detected" is
harmless with the pip wheels.

## 2. Data

`gpu_bench` and `gpu_v1_profile` need the 8 test experts of team 2/4 (indices 17,0,40,80,120,159,60,100 of the
160-expert sample manifest `ref/manifest_v41_sample160.json`):

| file | shard |
|---|---|
| L04_E038.bin | model-00007-of-00048.safetensors |
| L00_E097.bin | model-00003-of-00048.safetensors |
| L10_E015.bin | model-00013-of-00048.safetensors |
| L20_E005.bin | model-00023-of-00048.safetensors |
| L30_E002.bin | model-00033-of-00048.safetensors |
| L39_E156.bin | model-00042-of-00048.safetensors |
| L15_E025.bin | model-00018-of-00048.safetensors |
| L25_E000.bin | model-00028-of-00048.safetensors |

`zero_bittest` full needs all 160 sample experts (quick: the same 8). Each `.bin` is one expert record of
`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` @ `4ab649b2254afdcf6c2f7c1fbea1c05eda11828c`: the six tensors
w1.scale, w2.scale, w3.scale, w1.weight, w2.weight, w3.weight back to back (18,800,640 bytes; fetch_experts.py layout).
Every file is sha256-checked against the manifest. Three ways to provide them (searched in this order):

1. files in `<data>\` or `<data>\v41-sample\` (default `<data>` = `<kit root>\data`), e.g. the kit's data part or a copy
   of the team's `v41-sample` folder;
2. `--shards D:\path\to\V4.1` - the local safetensors shards: the records are cut out by the manifest offsets
   (absolute file offsets) into `<root>\results\_cache\v41-sample\` and sha256-checked;
3. `--fetch` - HTTP Range download of just those tensors from Hugging Face (pinned revision; ~150 MB for 8 experts,
   ~3 GB for 160). Off unless you pass the flag.

## 3. Run

```powershell
$py = ".\venv\Scripts\python.exe"
& $py -I tests\gpu\run_nvrtc_check.py --root .                 # 1) compile check, no GPU work
& $py -I tests\gpu\run_gpu_bench.py --root . --quick           # 2) first check (encodes once, ~3 min CPU)
& $py -I tests\gpu\run_gpu_bench.py --root .                   #    full: 8 configs x 10 variants, 41 rounds
& $py -I tests\gpu\run_zero_bittest.py --root . --quick
& $py -I tests\gpu\run_zero_bittest.py --root .                #    full: 160 experts
& $py -I tests\gpu\run_gpu_v1_profile.py --root . --quick      # 3) EXPERIMENTAL (after gpu_bench: reuses its encode)
```
`run_gpu_v1_profile.py` is optional: `setup\run_all.ps1` runs it only with `-All` (or `-Only gpu_v1`). It looks for
gpu_bench's encode cache next to its own `--out` (`<parent of --out>\gpu_bench\enc`), then in
`<root>\results\gpu_bench\enc`, or takes `--t4enc DIR`. Run directly, `src_v1\v1bench.py` reads `WNRKIT_ARCH` or
`WNR_NVRTC_ARCH` (`sm_120` or `-arch=sm_120`; default: the device's arch) and needs `WNRKIT_SAMPLE`.
`--fetch` accepts only HTTP 206 replies with a matching Content-Range and streams at most the tensor's bytes, so a
server or proxy that ignores Range is refused instead of sending a whole shard.
Common options: `--data DIR`, `--out DIR`, `--shards DIR`, `--fetch`, `--device N`.
`run_gpu_bench.py`: `--rounds N` (>= 21), `--nts 32,64,128,256` (threads per block = launch grid: blocks per matrix =
ceil(5760/nt)), `--opts lb,out4` (plain stores / 128-B staged stores), `--variants`, `--pipe-variants`,
`--arch-mode sm|compute`, `--nvrtc-extra -DLDM=1`, `--encode-only` (CPU only), `--reencode`.
`run_zero_bittest.py`: `--max-experts N`, `--arch-mode compute|sm`, `--no-mxf4`.

Results go to `results\<test>\` : `summary.json` (machine-readable, with OS/CPU/GPU/driver/Python/numpy/CuPy/NVRTC),
`<test>.log`, plus `bench.json`, `bench_table.txt`, `compare_3070.txt` (gpu_bench), `zero_bittest.json`,
`nvrtc_check.json`, `bench_v1.json` / `bench_v1_table.txt` (v1). The last line printed is
`RESULT <test> PASS|FAIL|DONE <summary>`. Please send back the whole `results\` folder with
`setup\pack_results.ps1`, which leaves out `results\_cache` and the `enc` / `v1enc` encode caches by default (large:
~800 MB encoded streams, ~150 MB-3 GB expert copies).

The NVRTC target is detected from device 0: `-arch=sm_120` on the RTX 5090, `sm_86` on the 3070 (`--arch-mode compute`
-> `compute_120` PTX + driver JIT). The experimental FP4 MMA uses `sm_120a`.

## 4. What PASS means / reference values (RTX 3070, sm_86, CuPy 14.2.0, NVRTC 12.9)

**gpu_bench** PASS = sha256 of the 8 inputs ok; CPU encoder output identical (sha256 of every array) to team 4's
encoder output (`ref/t4enc_expected.json`); scale-stream decode byte-exact; every non-control variant byte-exact vs the
original weight AND scale bytes in every config, in the single launch, 8 launches on 8 streams, partial launches (1
matrix, 1 record) and the scale+weight pipeline; negative controls CF0 / CFD0 / CFDB20 (inverse off) NOT byte-exact;
all timings finite. Speed is reported, not judged. Reference sizes (8 records): A259 0.90227 of raw, CF 0.88155, CFD
0.88156. 3070 best config per variant (`ref/bench_3070/best_final.txt`, us, 41 rounds):

| variant | m1 | m3 (1 record) | m24 (8 records) | 8 streams |
|---|---|---|---|---|
| A259 | 300.0 (nt32/lb) | 590.8 | 4171.8 | 4226.0 (nt64/out4) |
| CF5 | 304.1 | 565.2 | 4185.1 | 4234.2 |
| CF6 | 303.1 | 565.2 | 4190.2 | 4231.2 |
| D | 280.6 | 523.3 | 3835.9 | 3894.2 |
| CFD5 | 276.5 | 540.7 | 3848.2 | 3908.6 |
| DB2 | 271.4 | 415.7 | 3230.7 | 3287.4 |
| CFDB2 | 270.3 | 431.1 | 3307.5 | 3360.0 |

Same-config paired ratios on the 3070 (median over 8 configs): CF5/CF0 (inverse own cost) m1 1.017, m3 1.007, m24
1.003, 8streams 1.003; full table in `ref/bench_3070/ratios_final.json`. `compare_3070.txt` puts the local numbers next
to these. Spec targets (T9, for KAIRA's kernel on the 5090): inverse own cost <= +2%, WNR3 <= WNR2 + 5%.

**zero_bittest** PASS = 0 differing output bits (value, sign-of-zero, NaN payload) between original and canonicalised
records for the 10 colibri kernels on real and synthetic zero-row data, all reruns identical, no unwritten output,
positive control (one weight changed) fires in every kernel and in the tensor-core GEMM, negative control kernel
(first-product-seeded accumulator) shows sign-only mismatches on synthetic zero rows, FP16 tensor-core GEMM 0 diff,
f16/bf16 mma tiles 0 diff. 3070 reference (`ref/zero_3070/gpu_all160.json`, 160 experts): 14,008,320 outputs per
kernel (56,033,280 for the 4-token kernels), 0 mismatches of any kind, positive control 1 (4) mismatch per kernel;
`mma_signedzero.json`: 0 differing D on 20,000 random tiles, all-zero A -> D = +0 in 2,560,000/2,560,000;
`gpu_probe.json`: tc_fp16_gemm_bitdiff 0. The sm_120 `mxf4` FP4 MMA part is EXPERIMENTAL (written for this kit,
never run): reported in the summary, not in the verdict.

**gpu_v1_profile** prints DONE (never PASS, experimental) with `weight_all_exact`, scale exactness and the H32 GPU
vector check in the summary; team 5's 3070 table of an earlier code version is `ref/v1_3070/bench_v1_table.txt`.

## 5. Smoke checks done on the build PC (RTX 3070 present, but NO kernel was launched there)

- `python -m py_compile` of every `.py` file.
- `run_nvrtc_check.py --archs 86,120`: NVRTC compile (PTX and cubin with ptxas) of every kernel set for compute_86,
  sm_86, compute_120, sm_120 (and sm_120a for the mxf4 kernel) - see `ref/nvrtc_check_buildpc.json`.
- CPU encoder (src_bench/t4enc.py) on one matrix (w1 of L04_E038): A259, CF, DB2 weight streams, sign plane, both scale
  stream sets and the ceil-form scales byte-identical to team 4's encoder output (3.8 s).
- Not run at all: every GPU launch path of the three runners, the full 8-expert encode in the kit code, `--shards`,
  `--fetch`, the v1 wrapper end to end.

## 6. Notes for whoever runs it (or for Claude Code on that PC)

- Run `run_nvrtc_check.py` first; if sm_120 fails there, the NVRTC wheel is too old.
- Close other GPU work; timing uses the paired method (all variants of a config alternate inside every round), so
  ratios are robust to clock drift, absolute us are not. `nvidia-smi` clocks at start/end are in `bench.json`.
- On the 5090 the m1 size (one matrix = 180 CUDA blocks at nt=32) does not fill 170 SMs; the 8-record sizes are the
  relevant ones.
- If a check fails, keep `results\` as it is and send it back; do not edit the kernels.
