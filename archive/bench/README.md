# bench/

**Legacy.** GPU micro-benchmarks from the WNR2 work of October and November 2026, copied from the KAIRA lab. They exist only in this repository, are not part of the engine and are not run by any test. They need an NVIDIA GPU and real WNR2 records; the WNR2 containers they were written for have been deleted.

- **Build one:** `archive\bench\build.cmd NAME` compiles `archive\bench\NAME.cu` into `archive\bench\build\NAME.exe` (git-ignored), with the headers of `..\..\c`. Overrides: `VCVARS`, `NVCC`, `ARCH` (default sm_120).
- **Sources:** `bench_prio` (decode against inference with and without stream priority), `h2d_bw` (pinned host-to-device bandwidth), `bench_nr2_batch` (batched decode prototype), `bench_nr2_kernel` (kernel-only throughput), `bench_nr2_rechunk` (latency by stream length), `wddm_graph` (stream launches against one CUDA graph under WDDM), `test_decode_batch` (the helper's batch decode against real records; `integration\build\build_wex_windows.cmd` builds it too).
- **Python:** `rechunk.py` and `reencode_bin.py` write the `records_*.bin` samples the benchmarks read. They import `..\nr2\nr2lib.py`, which is why `bench/` and `nr2/` stay side by side. `nvme_temp.py` reads drive temperatures without admin rights.
- **Results:** [docs/WNR2-REFERENCE.md](../../docs/WNR2-REFERENCE.md) and the reports of 2026-11-05 in [docs/reports/](../../docs/reports/).
