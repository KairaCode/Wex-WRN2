# macOS read paths for expert records on GitHub Actions runners (2026-11-15 and 2026-11-16)

Runs of [`bench/wex_io.py`](../../bench/README.md) on macOS. All ran on GitHub-hosted `macos-15` runners:
virtual machines on Apple silicon hosts, with a virtual disk. **They are not the SSD of a Mac and say
nothing about it.** No Mac outside such a virtual machine was available.

Words used on this page. A **run** is one run of the script on one runner, one JSON file. A **round** is
one timed pass of every read path; `--runs N` makes N rounds. A **position** is the place of a pass in its
round (1 = first).

## Question

Do the read paths of `bench/wex_io.py` work on macOS through the calls macOS has (`F_NOCACHE` where
Linux has `O_DIRECT`, `fcntl` `F_RDAHEAD` and `F_RDADVISE` for the `posix_fadvise` hints, `pread` on 1
to 8 threads, no `io_uring`)? And is any difference between them large and steady enough to change the
store on macOS?

## Setup

- **Machines:** six `macos-15` arm64 runners of GitHub Actions, one per run, runner image
  `macos-15-arm64` 20261104.0514.1, a different worker id each time. What `sysctl` reported was the same
  on all six: CPU `Apple M1 (Virtual)`, model `VirtualMac2,1`, `kern.hv_vmm_present` 1 (a virtual
  machine), 3 logical CPUs, 7,516,192,768 bytes of RAM, 16 KiB pages, macOS 15.8.2 (24H232). The volume
  is APFS on a disk that `diskutil` names `Apple Inc. VirtIO Block Media` (solid state, internal,
  512-byte blocks), with 45 to 46 GiB free. The host's hardware, its load and how it backs the virtual
  disk cannot be seen from inside the virtual machine and are not known. Whether two runs shared a host
  is not known.
- **Container:** the synthetic container of [linux-io-20261114](../linux-io-20261114/README.md), written
  by the script with the reference writer: WEX4 4.0, 8 x 48 RAW expert records of 13,369,344 bytes of
  seeded random bytes, 5,134,127,104 bytes, root `13389132...8cd61d` in all six runs. No model weights.
- **The record: runs 1 to 3** (2026-11-16, started at 03:12, 03:18 and 03:23 UTC). This repository at
  commit b946969 (branch `io-macos`); `bench/wex_io.c` is the same as at 3fbac41. Command
  `WEX_CFLAGS="-std=c99 -pedantic" python3 -I bench/wex_io.py --dir "$RUNNER_TEMP" --drop-caches --runs 11`
  (job `macos-io-bench` of `.github/workflows/ci.yml`, dispatched by hand with `-f macos_io_bench=true`).
  The compiler is the runner's `gcc`, which is Apple clang 17.0.0 (clang-1700.0.13.5), `-O2`, C99; the
  build gave no warning (the script stops on any). 11 rounds, 48 records (641,728,512 bytes) per pass,
  every pass of a round reading the same records. The script rotates the order of the 11 timed rows by
  one from round to round, so with 11 rounds every row takes every position exactly once.
- **The earlier set: runs r5-1 to r5-3** (2026-11-15, started at 17:23, 17:32 and 17:37 UTC), commit
  3fbac41, the same command without `--runs` (5 rounds) and without `WEX_CFLAGS`. With 5 rounds the
  rotation uses 5 of the 11 start places, so the rows did not take every position (below, "Why 11
  rounds"). They are kept as recorded and are not the record.
- **What runs on macOS** (the blocks `__APPLE__` of `wex_io.c`; the Linux code is unchanged):

  | On Linux | On macOS |
  |---|---|
  | `O_DIRECT` | `open`, then `fcntl(F_NOCACHE, 1)` on the descriptor |
  | `POSIX_FADV_RANDOM` | `fcntl(F_RDAHEAD, 0)`: readahead off on the descriptor |
  | `POSIX_FADV_SEQUENTIAL` | none: skipped |
  | `POSIX_FADV_WILLNEED` on the record 1 or 4 reads ahead | `fcntl(F_RDADVISE)` on that record, an advisory read the caller does not wait for |
  | `POSIX_FADV_DONTNEED` on the file before a pass | `msync(MS_INVALIDATE)` on a mapping of the file |
  | `/proc/sys/vm/drop_caches` (`--drop-caches`) | `sync`, then `purge`, through `sudo -n` |
  | `io_uring`, 8 rows | none: skipped |
  | `mincore` in 4096-byte pages | `mincore` in 16 KiB system pages |

- **Cold cache:** `sudo -n true` succeeded on all six runners, so `purge` ran as root before every pass,
  and each cold pass first dropped the file's pages with `msync(MS_INVALIDATE)`. In runs 1 to 3 `mincore`
  found 0.0% of the pass's pages in the cache before each of the 330 cold passes and 100% before each of
  the 33 warm ones (r5 set: 150 and 15, the same shares). Neither call reaches a cache of the host under
  the virtual disk, and the first pass of each round (below) shows that one was at work.

## Results (runs 1 to 3)

**Identity.** In all six runs each of the 11 read paths that macOS has read 8 records whose SHA-256
matched the index before anything was timed. The other 9 (`POSIX_FADV_SEQUENTIAL` and the 8 `io_uring`
rows) printed `skipped`. The timed passes do not hash. The smoke run of the `macos` CI job checks the
same 11 paths on a 10 MB container, built in C99 with Homebrew GCC 15.3.0 and then with Apple clang
17.0.0 (the step prints both versions); it passed in the three CI runs that carried runs 1 to 3.

**Speed.** MB = 10^6 bytes. Each cell is the median MB/s of the 11 passes (minimum to maximum) and the
median against row 1 of the same run. "Rounds ahead" counts the rounds of the three runs in which the row
read faster than row 1 of the same round. "ms per record" is the median, over the 33 passes of the three
runs, of each pass's median record time; with several threads it grows, because the records wait for
each other.

| Read path | Run 1 | Run 2 | Run 3 | Rounds ahead of row 1 | ms per record |
|---|---|---|---|---|---|
| page cache, `pread`, 1 thread (`c/expert_store_wex.c` on macOS today) | 1537 (502 to 1956), 1.00x | 1495 (608 to 3240), 1.00x | 1428 (842 to 4406), 1.00x | | 6.8 (3.0 to 22.1) |
| page cache, `pread`, `F_RDAHEAD` off | 1535 (965 to 2003), 1.00x | 1461 (686 to 1840), 0.98x | 1404 (1036 to 3910), 0.98x | 18 of 33 | 7.6 (3.1 to 16.8) |
| page cache, `pread`, `F_RDADVISE` 1 record ahead | 1500 (912 to 2171), 0.98x | 1550 (764 to 2208), 1.04x | 1397 (661 to 3535), 0.98x | 16 of 33 | 7.0 (2.8 to 15.1) |
| page cache, `pread`, `F_RDADVISE` 4 records ahead | 1383 (603 to 3778), 0.90x | 1496 (892 to 3128), 1.00x | 1662 (936 to 3763), 1.16x | 19 of 33 | 6.2 (2.6 to 17.0) |
| page cache, `pread`, 4 threads | 2277 (1694 to 5104), 1.48x | 1731 (1187 to 5395), 1.16x | 2291 (796 to 4812), 1.60x | 26 of 33 | 19.2 (7.8 to 54.9) |
| page cache, `pread`, 8 threads | 2338 (845 to 3243), 1.52x | 2135 (875 to 6415), 1.43x | 2118 (1138 to 3443), 1.48x | 26 of 33 | 42.4 (16.0 to 110.9) |
| `F_NOCACHE`, `pread`, 1 thread | 5423 (1436 to 8470), 3.53x | 3885 (1838 to 6811), 2.60x | 3655 (2436 to 6765), 2.56x | 31 of 33 | 2.3 (1.5 to 6.2) |
| `F_NOCACHE`, `pread`, 2 threads | 6466 (1002 to 9695), 4.21x | 6389 (1107 to 9943), 4.27x | 5180 (1816 to 8641), 3.63x | 30 of 33 | 3.0 (2.5 to 14.7) |
| `F_NOCACHE`, `pread`, 4 threads | 9866 (1245 to 11777), 6.42x | 10179 (1072 to 12091), 6.81x | 9436 (2729 to 12149), 6.61x | 30 of 33 | 4.7 (4.2 to 35.6) |
| `F_NOCACHE`, `pread`, 8 threads | 10878 (2141 to 12053), 7.08x | 10584 (726 to 12408), 7.08x | 7942 (1771 to 13290), 5.56x | 31 of 33 | 9.0 (7.6 to 55.7) |
| page cache already warm, `pread`, 1 thread | 7561 (4207 to 8336), 4.92x | 7392 (5045 to 8216), 4.94x | 7411 (5456 to 8125), 5.19x | 33 of 33 | 1.7 (1.5 to 3.1) |

A cold pass lasted 0.05 to 1.28 s. All 11 rows of every run are in the JSON files, with every pass.

**What `mincore` saw.** After each of the 132 `F_NOCACHE` passes 0.0% of the pass's pages were in the
cache; after each of the 231 page-cache passes, 100%. In all 66 passes of the two `F_RDADVISE` rows,
100.0% of the pages of every announced record (47 per pass) were in the cache just before its read. On
the Linux cloud VMs one `POSIX_FADV_WILLNEED` had loaded 1.0% of such a record
([linux-io-20261114](../linux-io-20261114/README.md), run 3): another system and another disk. Whether
the `mincore` of macOS counts a page whose read is still under way was not checked, so the 100% says that
the advisory read had taken in hand the whole record, not that it had finished.

**The first pass of a round.** Every pass of a round reads the same 48 records, so the first pass of a
round is the only one whose records no earlier pass of that round had read. In the 33 rounds of runs 1
to 3 it was the slowest of the 11 passes of its row 23 times and one of the two slowest 30 times; if the
position did not matter, about 6 of 33 would be among the two slowest. That held for the `F_NOCACHE` rows
as much as for the page-cache rows (first passes of `F_NOCACHE` on 1 thread: 1,436, 1,838 and 2,574 MB/s,
against medians of 3,655 to 5,423). The median over all rows was 1,072 MB/s at position 1 and 2,012 to
3,443 MB/s at positions 2 to 11. A slow first pass, then faster passes over the same records with the
virtual machine's own cache empty before each, is what a cache below the virtual machine that keeps
recent reads would give: the host's. That is an interpretation; the host cannot be looked at from inside.
Two more numbers fit it: a 13.4 MB record through `F_NOCACHE` on 1 thread took 2.3 ms (median of 33
passes), close to the 1.7 ms of a record from the warm page cache of the virtual machine, and on 8 threads
`F_NOCACHE` reached up to 13.3 GB/s from the virtual disk. The warm row, whose timed pass reads
the virtual machine's RAM, was also the slowest of its row at position 1 in run 1 (4,207 MB/s against a
median of 7,669 for its other passes), but not in runs 2 and 3. Without the first pass of each round
(part 3 of `summary.py`), `F_NOCACHE` on 1 thread is 3.65x, 2.72x and 2.62x row 1, the two page-cache
thread rows 1.20x to 1.56x, and `F_RDAHEAD` off and `F_RDADVISE` 0.93x to 1.23x.

## Why 11 rounds (the earlier set)

The script rotates the order by one row per round, starting round r at row r. With the default 5 rounds
the first set (r5-1 to r5-3) used only 5 of the 11 start places: positions 1 and 2 held only page-cache
rows 1 to 6, and the `F_NOCACHE` rows sat only at positions 3 to 10. Over the 15 rounds of that set the
median MB/s by position was 1,045, 1,575, 1,770, 2,072, 4,678, 5,408, 6,431, 7,358, 6,439, 3,507 and
3,407 (positions 1 to 11; part 2 of `summary.py` prints which rows held each). So in that set a row's
median depended on where it sat as well as on its read path, and the gap between `F_NOCACHE` and the page
cache was partly a position effect. The set gave row 1 at 964, 3,377 and 2,239 MB/s (medians of runs r5-1
to r5-3), `F_NOCACHE` on 1 thread 1.59x to 3.52x row 1, and the page-cache variants 0.51x to 2.22x in
another order in each run. Runs 1 to 3 repeat the measurement with 11 rounds, where every row takes every
position once; their row 1 lies between 1,428 and 1,537 MB/s. The first pass of a round was still slow (above), but in runs 1 to 3 it falls on every row
once per run.

## What the runs show, and what they do not

- **The macOS port works on these virtual machines.** All 11 paths return the bytes the index names, and
  the stand-in calls did what `mincore` can show: `msync` and `purge` emptied the cache, `F_NOCACHE` left
  no page of its records in it, `F_RDADVISE` took in hand the whole announced record. The C99 build of
  Apple clang 17.0.0 and of GNU gcc 15.3.0 gave no warning.
- **"Cold" here is the cache of the virtual machine only.** At least part of the reads came from a cache
  below it (previous sections), so the table compares the macOS read paths over a virtual disk that answers
  partly from memory. It is not what the SSD of a Mac gives, cold or warm.
- **`F_NOCACHE` was faster than the page cache here.** With every row at every position once, its median
  on 1 thread was 2.56x to 3.53x row 1 in all three runs, and it read faster than row 1 of the same round
  in 31 of 33 rounds; on 2 to 8 threads 3.63x to 7.08x. The ranges still overlap row 1's in every run,
  because the first passes of a round are slow on every row. On a disk that answers from memory this says
  that the page-cache path of macOS costs more per record than a read that bypasses it (6.8 against 2.3 ms
  per record, medians over 33 passes). Whether that holds on an SSD, where the drive's own time counts,
  was not measured.
- **Threads through the page cache:** 4 threads 1.16x to 1.60x row 1, 8 threads 1.43x to 1.52x, ahead of
  row 1 in 26 of 33 rounds each: a smaller gain than `F_NOCACHE` on one thread.
- **No gain from the hints:** `F_RDAHEAD` off 0.98x to 1.00x, `F_RDADVISE` 0.90x to 1.16x, ahead of row 1
  in 16 to 19 of 33 rounds, about half. `F_RDADVISE` loaded the whole record, unlike the Linux hint, and
  still gave no steady gain here.
- **Warm page cache on 1 thread:** 7.4 to 7.6 GB/s (medians), the steadiest row.
- **The store is not changed.** `c/expert_store_wex.c` reads through the page cache on macOS, the slower
  kind of path on these machines, but a virtual disk that answers partly from the host's memory is no
  reason to change a store that runs on the SSD of a Mac. On macOS `c/qwen38_wex.h` opens through
  `compat_open_direct` of the engine, which is not in this repository, so which row it matches is not
  known.

## Limits

- One kind of virtual machine, three runs of 11 rounds started within 11 minutes (and three earlier runs of 5
  rounds), on six workers; the hosts are not known.
- The host's cache was not seen, not dropped and not measured apart: how much of each pass came from it
  is not known. The benchmark reads the same records on every path of a round (the design of the Linux
  benchmark), so on these machines every pass but the first of a round read records that the host may
  have kept. A run in which every pass reads records that no earlier pass read (a larger container, or
  fewer records per pass) was not made.
- 11 rounds balance the positions within a run; they do not remove the first-pass effect, which every
  row carries once per run.
- Whether `mincore` counts pages in flight on macOS was not checked (the `F_RDADVISE` share).
- A cold pass reads 642 MB and lasts 0.05 to 1.28 s; no longer pass was run.
- The benchmark reads records the way the stores do, but it is not the stores: no decode, no engine
  cache, no SHA-256 in the timed passes. No engine ran.
- Not run: a Mac outside a virtual machine, an Intel Mac, a macOS other than 15.8.2, a filesystem other
  than APFS, an external drive, the full run built with a compiler other than Apple clang 17.0.0 (the
  smoke run also builds with GNU gcc 15.3.0).
- An earlier run of the job on a commit that the branch replaced before the r5 set is not recorded here.

## Raw files

| File | What it is |
|---|---|
| `run1-wex_io.json` | `bench/out/wex_io.json` of run 1 (11 rounds), as the job printed it: every pass, the system, the disk, the table. From the log of the `macos-io-bench` job of <https://github.com/KairaCode/Wex-WRN2/actions/runs/43796778348> |
| `run2-wex_io.json` | the same for run 2, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43797333303> |
| `run3-wex_io.json` | the same for run 3, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43797901440> |
| `r5-run1-wex_io.json` | the earlier set, 5 rounds: run r5-1, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43740809358> |
| `r5-run2-wex_io.json` | run r5-2, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43741429336> |
| `r5-run3-wex_io.json` | run r5-3, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43742184549> |
| `summary.py` | prints the numbers of this page from the JSON files: `python3 -I summary.py run1-wex_io.json run2-wex_io.json run3-wex_io.json`, and the same with the three `r5-` files for the earlier set |

The script and the C program are `bench/wex_io.py` and `bench/wex_io.c` at commit b946969 (runs 1 to 3)
and 3fbac41 (the r5 set); `wex_io.c` and `wex_io.py` are the same at both. The job logs on GitHub expire;
the six JSON files are the record.
