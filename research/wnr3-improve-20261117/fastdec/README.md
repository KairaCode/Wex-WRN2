# fastdec: faster WNR3 CPU decode, same format

Rerun everything (from this directory): `./build.sh && ./run_equal2.sh && ./run_mutate.sh && python3 negcontrol.py 6000 99 /tmp/wnr3-data/recs/rec_{00_007,20_073,39_120}.bin && ./run_time.sh` (results in `logs/`; see RESULTS.md).

| file | what |
|---|---|
| `RESULTS.md` | the report: variant table, per-symbol time breakdown, correctness evidence, what transfers |
| `wex_fast_decode.patch` | the deliverable: patch against `Wex-WRN2/wex4/research/fast-cpu-decode/wex_fast_decode.h` (`patch -p1` from the repo root); result = `wex_fast_decode.h` here |
| `wex_fast_decode.h` | patched header (kernels 3 = AVX2 two-level tables, 4 = AVX-512 two-level tables + AVX-512 scale kernel) |
| `wex_fast_decode.orig.h` | unchanged copy of the repo header (byte-identical) |
| `test_fast_decode.c` | the repo harness + `ab` (interleaved A/B timing), `list`, entry read from `entries.bin` for bare `rec_LL_EEE.bin`, `EQ_THREADS`, coverage counters, larger mini-records (kinds 18/19) so mutate reaches the scale kernel |
| `abx.py` | interleaved A/B across binaries (rounds rotate the order; median, min, max, per-round ratio) |
| `build.sh` | builds out/test_z (AVX-512), out/test_h (AVX2), out/test_orig (unchanged header), scalar, UBSan |
| `run_equal2.sh`, `run_mutate.sh`, `negcontrol.py`, `run_time.sh` | correctness and timing runs, logs in `logs/` |
| `wex_fast_decode_research.h`, `test_research.c`, `count.c`, `flagstat.c` | research versions (kernel 3 scalar bucket table, kernel 4 u64-bucket AVX-512, counters) |
| `wex_fast_decode.abl.h` / `test_abl.c`, `wex_fast_decode.ablz.h` / `test_ablz.c` | ablation builds for the time breakdown (wrong output on purpose, timing only) |
| `gbench.c` | core-clock / gather-throughput calibration on this VM |
| `wex_fast_decode.v1.h`, `.k4v1.h`, `.k4v2.h` | intermediate snapshots (before the scale kernel / research steps); `wex_fast_decode.v1.h` is used by the table-equivalence check of the faster `wexf_two_level_init` (`out/cmp_*.c`: identical ta/tsub/tc) |
| `k3_4.s`, `k1_4.s` | gcc output of research kernel 3 (fully unrolled version, 4 streams) and kernel 1 with 4 streams: per-stream state spilled to the stack in both |
| `flagstat.c` / `flagstat.txt` | buckets with 1 / 2 / 3+ symbols in the production model for bucket sizes 2^3..2^9 |
| `pmu_probe.c` | shows `perf_event_open` is unavailable in this VM |
