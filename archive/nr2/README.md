# nr2/

**Legacy.** The WNR2 lab code: GPU trainer and encoder, container builders, checkers and tests, copied from the KAIRA lab. It exists only in this repository. WNR2 was replaced by WNR3 on 2026-11-10 and every WNR2 container was deleted, so nothing in production depends on this folder; the WNR3 builder is `c/tools/build_nr3.py`. The WNR2 decoders the engine still carries are in [`../../c/`](../../c/). Reference: [docs/WNR2-REFERENCE.md](../../docs/WNR2-REFERENCE.md).

- **Build:** `integration\build\build_wex_windows.cmd` builds `archive\nr2\wex_nr2_tools.dll` (from `encode_nr2.cu`; `nr2lib.py` loads it from this folder) and `check_container.exe`. `build_tools.cmd` builds the same DLL with line info and the ptxas report; `build_test_nr2.cmd` builds `test_nr2_cuda.exe` into `archive\nr2\build\` (git-ignored). All need MSVC, CUDA and an NVIDIA GPU. Overrides: `VCVARS`, `NVCC`, `ARCH`.
- **Build a container:** `build_nr2_full.py` (a WNR2 container from a WEX1 one, on the GPU; it always retrains the model) and `build_nr2.py` (the pilot sample).
- **Check one:** `check_container.c` (every record against the WEX1 source, with the engine's headers only), `check_all.py` (every record through the CUDA helper), `precheck_records.py` and `sample_records.py` (record samples for `archive\bench\test_decode_batch.cu`).
- **Tests:** `test_nr2_tools.py` (GPU encoder against the CPU reference) and `run_test_nr2.cmd` (the helper's GPU decode; needs `WEX_PILOT_DIR`). Both read pilot containers that are not in the repository.
- **Without a real model:** `fixtures/make_synth_mxfp4.py` makes a synthetic checkpoint whose records have the shape `build_nr2_full.py` needs.
