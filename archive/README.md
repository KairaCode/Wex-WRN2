# archive/

**Archived, kept for the record.** The lab code of the WNR2 period, copied from the KAIRA lab folder:

- [`nr2/`](nr2/README.md): the WNR2 GPU trainer and encoder, the container builders, the checkers and their tests;
- [`bench/`](bench/README.md): the decode, rechunk, host-to-device and stream-priority benchmarks of 2026-11-05.

Why it is archived: WNR2 was superseded by WNR3 on 2026-11-10 and every WNR2 container was deleted. No test of this repository runs this code, and it needs MSVC, the CUDA toolkit, an NVIDIA GPU and WNR2 containers that no longer exist. The WNR2 and WNR3 decoders that the engines still use are in [`../c/`](../c/); the WNR3 builder is `c/tools/build_nr3.py`.

The two folders moved here from the repository root on 2026-11-13. Only the paths that depend on the folder depth were changed in the code (the include path to `c/` in the `.cmd` and `.cu` files, the default helper path in `nr2/run_test_nr2.cmd`); `integration/build/build_wex_windows.cmd` still builds `nr2/wex_nr2_tools.dll` and the two checkers from here. The measurements made with this code are in [docs/reports/wnr2-history-20261110.md](../docs/reports/wnr2-history-20261110.md) and the reports of 2026-11-05.

`research/` is the archive of dated runs and is not moved here: frozen pages and mirrored code name its paths.
