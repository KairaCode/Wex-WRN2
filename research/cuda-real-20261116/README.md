# The CUDA helper built with nvcc, on real DeepSeek V4.1 records, on an RTX 3070 (2026-11-16)

The CUDA WNR3 decoder as the engine uses it: the helper `kaira_wex_cuda.dll` built by nvcc from
`c/tools/wex_cuda_bridge.cu`, driven by the CPU/GPU differential `c/tests/test_wex_nr3_gpu.cu` on one GPU.
Before this run the decoder had been checked as C++ on the CPU through an emulation and with its kernels
as PTX from clang on this GPU ([research/cuda-review-20261115](../cuda-review-20261115/README.md)); the
helper as nvcc builds it had not been run in this repository.

## Question

Does the helper built by nvcc give the CPU reference's bytes and verdicts on synthetic records, on
crafted records that only the WNR3 checks refuse, and on WNR3 records made from real DeepSeek V4.1 Flash
experts, with and without canonical zero, also when they are mutated?

## Setup

- **Machine:** NVIDIA GeForce RTX 3070, 8 GB, driver 621.01; Intel Core i7-7700K (4 cores, 8 threads,
  AVX2), 32 GB of RAM; Windows 10 (10.0.19045). What else ran during the run was not recorded.
- **Build:** this repository at commit ef71574 (`c/tools/wex_cuda_bridge.cu`, `c/tests/test_wex_nr3_gpu.cu`
  and the headers of `c/` did not change between ef71574 and `main` at b95de79). CUDA 13.5, MSVC of Visual
  Studio 2022 and Windows SDK 10.0.28000, for `sm_86`, with the options of
  [research/wnr3-20261110/build_helper.cmd](../wnr3-20261110/build_helper.cmd) and the architecture changed
  (from a prompt where `vcvars64.bat` of that MSVC has run, `C` the folder `c/`):

  ```bat
  nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler /O2,/EHsc,/arch:AVX2 -shared C\tools\wex_cuda_bridge.cu -o kaira_wex_cuda.dll
  nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler /O2,/EHsc C\tests\test_wex_nr3_gpu.cu -o test_wex_nr3_gpu.exe
  ```

- **Records** (parts 3 of the test): two bins in the layout of `c/tools/build_nr3.py bins`, 8 records each,
  made from 8 routed experts of the published DeepSeek V4.1 Flash checkpoint (layers 10 and 30, experts 0
  to 3, the expert shards 13 and 33 of 48, fetched by
  [research/real-experts-20261115](../real-experts-20261115/README.md)): `real.bin` without and
  `real--cz.bin` with canonical zero, each with a WNM2 model trained on those 8 experts. The expected slab
  of each record is the source slab (with `--cz`, the source with -0 stored as +0). The bins are 282,765,944
  and 278,628,984 bytes and are not in the repository. To make them again:

  ```sh
  # the 8 experts: research/real-experts-20261115/fetch_experts.py (its deepseek-v41 row), into M
  python -I c/tools/repack_v4_wex.py --model-dir M --output v41.wex
  # wex_nr3_tools.dll from c/tools/wex_nr3_tools.c, as scripts/build_cpu_check.sh builds it
  P=10:0,10:1,10:2,10:3,30:0,30:1,30:2,30:3
  python -I c/tools/build_nr3.py wex_nr3_tools.dll v41.wex bins real.bin --experts $P --train $P
  python -I c/tools/build_nr3.py wex_nr3_tools.dll v41.wex bins real--cz.bin --experts $P --train $P --cz
  ```

- **Command:** `test_wex_nr3_gpu.exe kaira_wex_cuda.dll 120 real.bin real--cz.bin`: 120 mutated copies
  per bin (single bit flips anywhere in a record, edits of header bytes).

## Results

| Part of the test | Outcome |
|---|---|
| 1. synthetic experts, canonical zero off and on, 4 chunk pairs (512:256 to 16384:16384), each through the 3 entry points (to host, to device, batch) | ALL OK: every output equal to the CPU decode, which equals the source or Z(source) |
| 2. crafted records only the WNR3 checks refuse (a canonical-zero record with -0, a ceil block at coded scale 0, a WNR3 record under codec 4) | refused by the GPU and by the CPU, the destination left untouched |
| 3. `real.bin`, 8 records of real experts, no canonical zero | 8 of 8 decoded by the batch API equal to the stored slab; 120 mutated copies: the same status on CPU and GPU for each, all 120 refused by both |
| 3. `real--cz.bin`, the same 8 experts with canonical zero | 8 of 8 equal to the stored slab; 120 mutated copies: the same status on CPU and GPU, all refused by both |

No difference of verdict or bytes between the helper on the GPU and the CPU reference. The log of this run
was not kept; the table is its summary as it was noted down.

## A sample of the published container's own records

The same helper and test, the same day (10:59 UTC, while the Vulkan run of
[research/vulkan-real-20261116](../vulkan-real-20261116/README.md) used the same GPU and one CPU thread), on
17 records of the published DeepSeek V4.1 Flash REAP-192 container (`v41-reap/v4_flash_experts.nr3.wex` of
<https://huggingface.co/Kaira666/Kaira-WEX4-Experts>, WNR3-CZ, flag byte 0x0f, the production model
`0x27e8a1e0`): one expert per fifth layer, expert `(53 L + 97) mod 192`, in two bins (layers 0, 5, ..., 35 and
4, 9, ..., 39), plus L39 E120, the record with the largest scale payload. `make_bin.py` of this folder wrote
the bins (278,764,152 and 315,460,292 bytes, not in the repository): each record as stored, with its WEX3
entry, and as expected slab the decode of the CPU reference, so this compares the GPU with the CPU reference
on production records; the source checkpoint is not involved. Command:
`test_wex_nr3_gpu.exe kaira_wex_cuda.dll 120 container-a.bin container-b.bin` (`container-sample.log`).

| Part | Outcome |
|---|---|
| synthetic parts, canonical zero off and on | 4 chunk pairs x 3 entry points equal to the CPU decode, crafted records rejected |
| `container-a.bin`, 8 records | 8 of 8 equal to the CPU reference's decode; 120 mutated copies with the same status on CPU and GPU, 0 accepted |
| `container-b.bin`, 9 records | 9 of 9 equal; 120 mutated copies with the same status on CPU and GPU, 0 accepted |
| last lines | `compared 14 (rejected by both 6)`, `ALL OK`; 25 s for the whole test |

To make the bins again (the DLL as `scripts/build_cpu_check.sh` builds it from `c/tools/wex_nr3_tools.c`):

```sh
python -I research/cuda-real-20261116/make_bin.py <MODELS>/v41-reap/v4_flash_experts.nr3.wex wex_nr3_tools.dll \
    container-a.bin 0:97,5:170,10:51,15:124,20:5,25:78,30:151,35:32
python -I research/cuda-real-20261116/make_bin.py <MODELS>/v41-reap/v4_flash_experts.nr3.wex wex_nr3_tools.dll \
    container-b.bin 4:117,9:190,14:71,19:144,24:25,29:98,34:171,39:52,39:120
```

## Limits

- One GPU, one driver, one build (sm_86). Not run under `compute-sanitizer`.
- 16 records made from 8 real experts (trained on themselves) and 17 records of the published container,
  out of 7,680. The whole container was not decoded through the helper.
- 120 mutations per bin; all were refused, so the case "both accept a mutated record" did not occur.
- The engine did not run: the test calls the helper's entry points directly.

## Raw files

| File | What it is |
|---|---|
| `make_bin.py` | writes a bin of the test from records of the published container (records as stored, expected slab = the CPU reference's decode) |
| `container-sample.log` | the run on the 17 records of the container; the paths of the bins are shortened to `<WORK>/` and the `ls -l` lines reduced to name and size |
