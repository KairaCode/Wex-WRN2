# macOS read paths for expert records on GitHub Actions runners (2026-11-15)

The first three runs of [`bench/wex_io.py`](../../bench/README.md) on macOS. All ran on GitHub-hosted
`macos-15` runners: virtual machines on Apple silicon hosts, with a virtual disk. **They are not the SSD
of a Mac and say nothing about it.** No Mac outside such a virtual machine was available.

## Question

Do the read paths of `bench/wex_io.py` work on macOS through the calls macOS has (`F_NOCACHE` where
Linux has `O_DIRECT`, `fcntl` `F_RDAHEAD` and `F_RDADVISE` for the `posix_fadvise` hints, `pread` on 1
to 8 threads, no `io_uring`)? And is any difference between them large and steady enough to change the
store on macOS?

## Setup

- **Machines:** three `macos-15` arm64 runners of GitHub Actions, one per run, runner image
  `macos-15-arm64` 20261104.0514.1, a different worker id each time. What `sysctl` reported was the same
  on all three: CPU `Apple M1 (Virtual)`, model `VirtualMac2,1`, `kern.hv_vmm_present` 1 (a virtual
  machine), 3 logical CPUs, 7,516,192,768 bytes of RAM, 16 KiB pages, macOS 15.8.2 (24H232). The volume
  is APFS on a disk that `diskutil` names `Apple Inc. VirtIO Block Media` (solid state, internal,
  512-byte blocks), with 46 GiB free. The host's hardware, its load and how it backs the virtual disk
  cannot be seen from inside the virtual machine and are not known. Whether two runs shared a host is
  not known.
- **Container:** the synthetic container of [linux-io-20261114](../linux-io-20261114/README.md), written
  by the script with the reference writer: WEX4 4.0, 8 x 48 RAW expert records of 13,369,344 bytes of
  seeded random bytes, 5,134,127,104 bytes, root `13389132...8cd61d` in all three runs. No model weights.
- **Build:** this repository at commit 3fbac41 (branch `io-macos`). `bench/wex_io.py` built
  `bench/wex_io.c` with the runner's `gcc`, which is Apple clang 17.0.0 (clang-1700.0.13.5), `-O2`.
- **Command:** `python3 -I bench/wex_io.py --dir "$RUNNER_TEMP" --drop-caches` (job `macos-io-bench` of
  `.github/workflows/ci.yml`, dispatched by hand with `-f macos_io_bench=true`): 5 timed passes per read
  path, 48 records (641,728,512 bytes) per pass, the passes of all paths interleaved and their order
  rotated from run to run, a run reading the same records on every path. The three runs started at 17:23,
  17:32 and 17:37 UTC.
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

- **Cold cache:** `sudo -n true` succeeded on all three runners, so `purge` ran as root before every pass,
  and each cold pass first dropped the file's pages with `msync(MS_INVALIDATE)`. `mincore` found 0.0% of
  the pass's pages in the cache before each of the 150 cold passes (50 per run) and 100% before each of
  the 15 warm ones. Neither call reaches a cache of the host under the virtual disk, and the order of the
  passes (below) shows that one was at work.

## Results

**Identity.** In all three runs each of the 11 read paths that macOS has read 8 records whose SHA-256
matched the index before anything was timed. The other 9 (`POSIX_FADV_SEQUENTIAL` and the 8 `io_uring`
rows) printed `skipped`. The timed passes do not hash. The smoke run of the `macos` CI job checks the
same 11 paths on a 10 MB container, built with Homebrew GCC 15.3.0 and then with Apple clang 17.0.0; it
passed in the three CI runs that carried these benchmark runs.

**Speed.** MB = 10^6 bytes. Each cell is the median MB/s of the 5 passes (minimum to maximum) and the
median against row 1 of the same run. "ms per record" is the median, over the 15 passes of the three runs,
of each pass's median record time; with several threads it grows, because the records wait for each other.

| Read path | Run 1 | Run 2 | Run 3 | ms per record |
|---|---|---|---|---|
| page cache, `pread`, 1 thread (`c/expert_store_wex.c` on macOS today) | 964 (788 to 3407), 1.00x | 3377 (1029 to 3507), 1.00x | 2239 (162 to 3865), 1.00x | 4.4 (3.2 to 15.3) |
| page cache, `pread`, `F_RDAHEAD` off | 1214 (895 to 1575), 1.26x | 2796 (1083 to 3461), 0.83x | 2751 (418 to 4416), 1.23x | 8.2 (3.0 to 28.9) |
| page cache, `pread`, `F_RDADVISE` 1 record ahead | 1320 (1009 to 1384), 1.37x | 1712 (1076 to 3862), 0.51x | 1214 (768 to 4602), 0.54x | 6.4 (2.6 to 14.3) |
| page cache, `pread`, `F_RDADVISE` 4 records ahead | 1748 (647 to 2072), 1.81x | 1770 (1609 to 3836), 0.52x | 1473 (1302 to 1696), 0.66x | 5.2 (2.6 to 14.0) |
| page cache, `pread`, 4 threads | 2145 (991 to 4678), 2.22x | 2238 (614 to 3313), 0.66x | 1808 (943 to 3652), 0.81x | 18.8 (10.7 to 43.8) |
| page cache, `pread`, 8 threads | 1897 (1806 to 5666), 1.97x | 2382 (1485 to 4374), 0.71x | 1761 (1249 to 2578), 0.79x | 47.2 (20.1 to 58.3) |
| `F_NOCACHE`, `pread`, 1 thread | 3393 (1558 to 5407), 3.52x | 5385 (1514 to 6554), 1.59x | 4171 (1007 to 6798), 1.86x | 2.0 (1.7 to 3.0) |
| `F_NOCACHE`, `pread`, 2 threads | 6786 (2994 to 8519), 7.04x | 8416 (5416 to 9974), 2.49x | 7054 (1474 to 8921), 3.15x | 3.0 (2.5 to 3.5) |
| `F_NOCACHE`, `pread`, 4 threads | 9482 (5565 to 11703), 9.83x | 7164 (4017 to 11379), 2.12x | 8046 (4750 to 13135), 3.59x | 4.9 (4.0 to 5.9) |
| `F_NOCACHE`, `pread`, 8 threads | 7396 (1017 to 12051), 7.67x | 10012 (8615 to 11815), 2.96x | 11030 (3530 to 11783), 4.93x | 8.7 (7.6 to 30.9) |
| page cache already warm, `pread`, 1 thread | 7055 (6622 to 7551), 7.31x | 7358 (7199 to 7554), 2.18x | 7751 (6985 to 8403), 3.46x | 1.8 (1.6 to 2.0) |

A cold pass lasted 0.05 to 4.0 s. All 11 rows of every run are in the JSON files, with every pass.

**What `mincore` saw.** After each of the 60 `F_NOCACHE` passes 0.0% of the pass's pages were in the
cache; after each of the 105 page-cache passes, 100%. In all 30 passes of the two `F_RDADVISE` rows, 100.0%
of the pages of every announced record (47 per pass) were in the cache just before its read. On the Linux
cloud VMs one `POSIX_FADV_WILLNEED` had loaded 1.0% of such a record
([linux-io-20261114](../linux-io-20261114/README.md), run 3): another system and another disk. Whether
the `mincore` of macOS counts a page whose read is still under way was not checked, so the 100% says that
the advisory read had taken in hand the whole record, not that it had finished.

**The order of the passes.** A run of the script reads the 11 timed rows in the order of the table,
rotated by one row from run to run, and every pass of a run reads the same 48 records. The first pass of
a run is therefore the only one whose records no earlier pass of that run had read. With 5 runs and 11
rows the first passes fall on rows 1 to 5, all through the page cache. In the 15 runs of the three files
the first pass was the slowest of the 5 passes of its row 6 times, the second slowest 8 times and the
fastest once; if the position did not matter, about 6 of 15 would be among the two slowest. The 15 first
passes read at 1,045 MB/s, median (162 to 1,645). A slow first pass, then faster passes over the same
records with the virtual machine's own cache empty before each, is what a cache below the virtual
machine that keeps recent reads would give: the host's. That is an interpretation; the host cannot be
looked at from inside. Two more numbers fit it. The median time of one 13.4 MB record through
`F_NOCACHE` on 1 thread was 2.0 ms (median of 15 passes), close to the 1.8 ms of a record from the warm
page cache of the virtual machine. And with 2 to 8 threads `F_NOCACHE` reached up to 13.1 GB/s from the
virtual disk. Without the first pass of each
run (part 3 of `summary.py`), `F_NOCACHE` on 1 thread is 2.80x, 1.57x and 1.50x row 1, and the other
page-cache rows are 0.53x to 1.86x row 1.

## What the runs show, and what they do not

- **The macOS port works on these virtual machines.** All 11 paths return the bytes the index names, and
  the stand-in calls did what `mincore` can show: `msync` and `purge` emptied the cache, `F_NOCACHE` left
  no page of its records in it, `F_RDADVISE` took in hand the whole announced record.
- **"Cold" here is the cache of the virtual machine only.** At least part of the reads came from a cache
  below it (previous section), so the table compares the macOS read paths over a virtual disk that answers
  partly from memory. It is not what the SSD of a Mac gives, cold or warm.
- **`F_NOCACHE` was faster than the page cache on 1, 4 and 8 threads in every run** (medians). On 1
  thread it was 1.59x to 3.52x row 1, with ranges that overlap row 1's in every run, and 1.50x to 2.80x
  without the first passes. On a disk that answers from memory this says that the page-cache path of
  macOS costs more per record than a read that bypasses it (4.4 against 2.0 ms per record, medians over
  15 passes). Whether that holds on an SSD, where the drive's own time counts, was not measured.
- **No page-cache variant was steady against row 1:** `F_RDAHEAD` off 0.83x to 1.26x, `F_RDADVISE` 0.51x
  to 1.81x, 4 and 8 threads 0.66x to 2.22x, in another order in each run. `F_RDADVISE` loaded the whole
  record, unlike the Linux hint, and still gave no steady gain here.
- **Warm page cache on 1 thread:** 7.1 to 7.8 GB/s (6.6 to 8.4 over the 15 passes), the steadiest row.
- **The store is not changed.** `c/expert_store_wex.c` reads through the page cache on macOS, the slower
  kind of path on these machines, but a virtual disk that answers partly from the host's memory is no
  reason to change a store that runs on the SSD of a Mac. On macOS `c/qwen38_wex.h` opens through
  `compat_open_direct` of the engine, which is not in this repository, so which row it matches is not
  known.

## Limits

- One kind of virtual machine, three runs of the script within 17 minutes (5 passes per path in a run),
  on three workers; the hosts are not known.
- The host's cache was not seen, not dropped and not measured apart: how much of each pass came from it
  is not known. The benchmark reads the same records on every path of a run (the design of the Linux
  benchmark), so on these machines every pass but the first of a run read records that the host may have
  kept. A run in which every pass reads records that no earlier pass read (a larger container, or fewer
  records per pass) was not made.
- The rotation puts the first pass of a run on rows 1 to 5 only, so those page-cache rows carry the slow
  first passes and the `F_NOCACHE` rows none; part 3 of `summary.py` gives the rows without them.
- Whether `mincore` counts pages in flight on macOS was not checked (the `F_RDADVISE` share).
- A cold pass reads 642 MB and lasts 0.05 to 4.0 s; no longer pass was run.
- The benchmark reads records the way the stores do, but it is not the stores: no decode, no engine
  cache, no SHA-256 in the timed passes. No engine ran.
- Not run: a Mac outside a virtual machine, an Intel Mac, a macOS other than 15.8.2, a filesystem other
  than APFS, an external drive, the full run built with a compiler other than Apple clang 17.0.0 (the
  smoke run also builds with GNU gcc 15.3.0).
- An earlier run of the same job, on a commit that the branch replaced before these runs, is not
  recorded here.

## Raw files

| File | What it is |
|---|---|
| `run1-wex_io.json` | `bench/out/wex_io.json` of run 1, as the job printed it: every pass, the system, the disk, the table. From the log of the `macos-io-bench` job of <https://github.com/KairaCode/Wex-WRN2/actions/runs/43740809358> |
| `run2-wex_io.json` | the same for run 2, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43741429336> |
| `run3-wex_io.json` | the same for run 3, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43742184549> |
| `summary.py` | prints the numbers of this page from the three files: `python3 -I summary.py run1-wex_io.json run2-wex_io.json run3-wex_io.json` |

The script and the C program are `bench/wex_io.py` and `bench/wex_io.c` at commit 3fbac41. The job logs on
GitHub expire; the three JSON files are the record.
