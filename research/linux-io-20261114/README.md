# Linux read paths for expert records on GitHub Actions runners (2026-11-14)

The first three runs of [`bench/wex_io.py`](../../bench/README.md). All ran on GitHub-hosted runners:
shared cloud virtual machines with a virtual disk. **They are not NVMe hardware and say nothing about
it.** No Linux machine with a real drive was available. Runs 1 and 2 are the comparison of the read
paths; run 3 was made after the script learned to measure what `POSIX_FADV_WILLNEED` loads, landed on the
same kind of machine as run 2, and is reported in its own section.

## Question

Through which Linux read path should an engine read the expert records of a WEX4 container: `pread`
through the page cache (what `c/expert_store_wex.c` does on Linux), with or without `posix_fadvise` hints,
`pread` with `O_DIRECT` (what `c/qwen38_wex.h` does), several threads, or `io_uring` with several records
or sub-reads in flight? And is any difference large and steady enough to change the store?

## Setup

- **Machines:** `ubuntu-24.04` runners of GitHub Actions (Ubuntu 24.04.5, 4 logical CPUs, 16 GB of
  RAM, ext4), one per run. Run 1: AMD EPYC 7763, kernel 7.0.0-1017-azure, runner image 20261111.370.1;
  the block device reports model `Virtual Disk`, `read_ahead_kb` 128, `max_sectors_kb` 2048, scheduler
  `none`. Run 2: Intel Xeon 6975P-C, kernel 7.0.0-1014-aws, runner image 20261111.127.1; the block device
  reports `read_ahead_kb` 128 and `max_sectors_kb` 128, no model and no scheduler. Run 3: the same CPU
  model, kernel, runner image and block-device settings as run 2; whether it was the same host is not
  known. What else ran on the hosts is not known.
- **Container:** synthetic, written by the script with the reference writer: WEX4 4.0, 8 x 48 RAW expert
  records of 13,369,344 bytes of seeded random bytes, 5,134,127,104 bytes, root `13389132...8cd61d` in
  all three runs. No model weights.
- **Build:** this repository at commit 0971dc8 (runs 1 and 2) and 4c2ad6e (run 3), `gcc -O2` of the
  runner image (version not recorded). Between the two commits `wex_io.c` gained one `mincore` call per
  record in the two `WILLNEED` paths, and the `io_uring` ring is now set up before the clock starts, not
  after; the other paths read as before.
- **Command:** `python3 -I bench/wex_io.py --dir "$RUNNER_TEMP" --drop-caches` (job `linux-io-bench` of
  `.github/workflows/ci.yml`, dispatched by hand): 5 timed passes per read path, 48 records
  (641,728,512 bytes) per pass, the passes of all paths interleaved, a pass reading the same records on
  every path.
- **Cold cache:** `/proc/sys/vm/drop_caches` was written before every pass (the runner has `sudo`), and
  each pass dropped the file's pages with `POSIX_FADV_DONTNEED`. `mincore` found 0.0% of the pass's pages
  in the page cache before each of the 95 cold passes of each run, and 100% before the warm ones. A cache
  of the host under the virtual disk cannot be seen or dropped from inside the VM.

## Results

**Identity.** In all three runs each of the 20 read paths read 8 records whose SHA-256 matched the index
before anything was timed. The timed passes do not hash.

**Speed, runs 1 and 2.** MB = 10^6 bytes. "ms per record" is the median over the 5 passes of the pass's median record
time; with several records in flight it grows with the depth, because the records wait for each other.

| Read path | Run 1: MB/s, median (min to max) | against row 1 | ms per record | Run 2: MB/s, median (min to max) | against row 1 | ms per record |
|---|---|---|---|---|---|---|
| page cache, `pread`, 1 thread | 393 (380 to 395) | 1.00x | 45.5 | 543 (541 to 575) | 1.00x | 28.7 |
| page cache, `pread`, `POSIX_FADV_RANDOM` | 466 (455 to 492) | 1.19x | 41.1 | 517 (511 to 548) | 0.95x | 28.2 |
| page cache, `pread`, `POSIX_FADV_SEQUENTIAL` | 387 (374 to 390) | 0.98x | 45.6 | 501 (499 to 510) | 0.92x | 29.3 |
| page cache, `pread`, `WILLNEED` 1 record ahead | 395 (384 to 399) | 1.00x | 45.2 | 507 (499 to 518) | 0.93x | 28.7 |
| page cache, `pread`, `WILLNEED` 4 records ahead | 389 (378 to 395) | 0.99x | 45.1 | 508 (500 to 508) | 0.93x | 28.7 |
| page cache, `pread`, 4 threads | 418 (416 to 431) | 1.06x | 142.4 | 516 (509 to 519) | 0.95x | 97.8 |
| page cache, `pread`, 8 threads | 437 (434 to 451) | 1.11x | 253.1 | 528 (521 to 532) | 0.97x | 167.9 |
| page cache, `io_uring`, 8 records in flight | 424 (420 to 434) | 1.08x | 253.4 | 531 (522 to 535) | 0.98x | 227.7 |
| `O_DIRECT`, `pread`, 1 thread | 456 (450 to 468) | 1.16x | 44.2 | 520 (514 to 522) | 0.96x | 28.3 |
| `O_DIRECT`, `pread`, 2 threads | 458 (444 to 461) | 1.16x | 48.3 | 512 (504 to 515) | 0.94x | 43.4 |
| `O_DIRECT`, `pread`, 4 threads | 451 (448 to 459) | 1.15x | 137.1 | 516 (510 to 519) | 0.95x | 70.2 |
| `O_DIRECT`, `pread`, 8 threads | 469 (457 to 478) | 1.19x | 251.6 | 523 (515 to 528) | 0.96x | 135.9 |
| `O_DIRECT`, `io_uring`, 1 record in flight | 447 (446 to 462) | 1.14x | 43.8 | 512 (497 to 514) | 0.94x | 28.3 |
| `O_DIRECT`, `io_uring`, 1 record as 4 sub-reads | 456 (442 to 461) | 1.16x | 43.3 | 511 (506 to 512) | 0.94x | 28.3 |
| `O_DIRECT`, `io_uring`, 1 record as 16 sub-reads | 454 (444 to 455) | 1.15x | 43.8 | 512 (506 to 515) | 0.94x | 28.3 |
| `O_DIRECT`, `io_uring`, 4 records in flight | 460 (451 to 466) | 1.17x | 113.0 | 518 (512 to 520) | 0.95x | 113.0 |
| `O_DIRECT`, `io_uring`, 8 records in flight | 474 (457 to 476) | 1.21x | 251.9 | 524 (518 to 527) | 0.96x | 171.1 |
| `O_DIRECT`, `io_uring`, 16 records in flight | 481 (471 to 496) | 1.22x | 488.1 | 540 (533 to 544) | 0.99x | 396.8 |
| `O_DIRECT`, `io_uring`, 8 records x 4 sub-reads | 477 (458 to 480) | 1.21x | 251.4 | 527 (522 to 529) | 0.97x | 173.3 |
| page cache already warm, `pread`, 1 thread | 15335 (14854 to 15604) | 39.00x | 0.9 | 10242 (10097 to 10766) | 18.85x | 1.3 |

## Run 3: what `POSIX_FADV_WILLNEED` loaded

The two `WILLNEED` rows announce a whole record (13,369,344 bytes, 3,264 pages) 1 or 4 reads before it is
read. Runs 1 and 2 did not check what the kernel did with that. Run 3 did: just before each announced
record is read, one `mincore` call counts how many of its pages are in the page cache (47 announced
records per pass, 5 passes per row).

| Read path | Announced pages in the page cache before the read, median (min to max) | MB/s, median (min to max) | against row 1 |
|---|---|---|---|
| page cache, `pread`, `WILLNEED` 1 record ahead | 1.0% (1.0% to 1.0%) | 508 (502 to 520) | 0.93x |
| page cache, `pread`, `WILLNEED` 4 records ahead | 1.0% (1.0% to 1.0%) | 506 (502 to 510) | 0.93x |

1.0% of 3,264 pages is about 33 pages, 130 KB. The block device of run 3 reports `read_ahead_kb` 128 and
`max_sectors_kb` 128. So on this machine the hint loaded about one readahead window of each record, not
the record: **these two rows do not measure a prefetch of the next record**, they measure a hint that
the kernel cut to about 128 KB. That the kernel caps a `WILLNEED` request at the device's readahead or
request size is the reading that fits the number; the kernel source was not checked here. `mincore`
counts pages that are in the cache and read; pages whose read was still under way may be missed. Not
measured: the same share on the machine type of run 1 (`max_sectors_kb` 2048), and a prefetch that
really loads the record (for example `readahead(2)` or `WILLNEED` in chunks, or a second thread).

The other rows of run 3 repeat run 2 within a few MB/s (same machine type): row 1 546 (543 to 597),
`POSIX_FADV_RANDOM` 518 (510 to 553, 0.95x), `O_DIRECT` `pread` 1 thread 523 (517 to 527, 0.96x),
`O_DIRECT` `io_uring` 16 records in flight 545 (537 to 549, 1.00x), every row with more than one read in
flight between 513 and 545, warm page cache 9,775. All 20 rows are in `run3-wex_io.json`.

## What the runs show, and what they do not

- **No read path wins on both machines.** In run 1 the store's path of today (row 1) is among the
  slowest cold paths, and `POSIX_FADV_RANDOM` and `O_DIRECT` on one thread are 19% and 16% faster with
  ranges that do not overlap row 1's. In run 2 row 1 is the fastest cold path and the same two are 5% and 4% slower. A change
  of the store's Linux read path is therefore **not supported** by these runs, and none was made.
- **More reads in flight gained little.** Against one read of the same kind, the largest gain was 11%
  in run 1 (page cache, 8 threads against 1: 437 against 393 MB/s, ranges apart) and 4% in runs 2 and 3
  (`O_DIRECT`, `io_uring` with 16 records in flight against `pread` on 1 thread: 540 against 520 and 545
  against 523, ranges apart); through the page cache, more reads in flight were 2% to 6% slower than one
  in runs 2 and 3. Threads, `io_uring` depth and sub-reads all end between 418 and 481 MB/s in run 1,
  511 and 540 in run 2 and 513 and 545 in run 3, whatever the depth. Reading this as a throughput limit of the virtual disk is an interpretation: the limit of the
  runner's disk was not looked up or measured apart. An NVMe drive is expected to behave differently
  here; that is an expectation, not a result.
- **`POSIX_FADV_SEQUENTIAL` did not help** in any run (0.92x to 0.98x). **`POSIX_FADV_WILLNEED` on the
  next records did not help either** (0.93x to 1.00x), but run 3 shows why that says little: the kernel
  loaded about 1% of each announced record (section above). Whether a prefetch that loads the whole
  record helps was not measured.
- **`io_uring` with one record in flight equals `pread`** with `O_DIRECT` in all three runs (447 against 456,
  512 against 520, 516 against 523 MB/s): on these disks the system-call path is not what costs.
- **Warm page cache:** 15.3 and 10.2 GB/s, with no drive in it. A store that opens with `O_DIRECT` gives
  that second-level cache up; whether an engine wants it depends on its own RAM cache and was not
  measured.

## Limits

- Two kinds of machine, three runs of the script (5 passes per path in a run). All are cloud VMs with a
  virtual disk; runs 1 and 2 landed on different hosts with different CPUs, kernels and disks, so they
  are two samples, not a repeat. Run 3 is close to a repeat of run 2 (same machine type, a newer commit
  of the script).
- What `POSIX_FADV_WILLNEED` loaded was measured in run 3 only, on one machine type.
- The `io_uring` rows of runs 1 and 2 include the set-up of the ring (three `mmap` calls) in the timed
  pass; run 3 does not. The `io_uring` rows of run 3 are 2 to 5 MB/s above those of run 2 and its
  `pread` rows are 3 below to 5 above, so the set-up time is not visible in these numbers.
- The worker threads are created inside the timed pass in all runs.
- A cold pass reads 642 MB and lasts 1.1 to 1.7 s. A disk with a burst allowance may behave differently
  over longer reads; no longer pass was run (`--records 384` reads the whole file).
- The benchmark reads records the way the stores do, but it is not the stores: no decode, no engine
  cache, no eviction, no SHA-256 at the first read in the timed passes. No engine ran.
- Not run: any NVMe or SATA drive, ARM, a filesystem other than ext4, ZFS, two drives (mirror, stripe),
  `IORING_SETUP_SQPOLL`, registered buffers or files, macOS.
- The runner's `gcc` version and the load of the hosts are not recorded.

## Raw files

| File | What it is |
|---|---|
| `run1-wex_io.json` | `bench/out/wex_io.json` of run 1, as the job printed it: every pass, the system, the block-device settings, the table. From the log of the `linux-io-bench` job of <https://github.com/KairaCode/Wex-WRN2/actions/runs/43620349072> |
| `run2-wex_io.json` | the same for run 2, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43641677405> |
| `run3-wex_io.json` | the same for run 3 (JSON `version` 2: each `WILLNEED` pass has `ahead_cached`), <https://github.com/KairaCode/Wex-WRN2/actions/runs/43644613531> |

The script and the C program are `bench/wex_io.py` and `bench/wex_io.c` at commit 0971dc8 (runs 1 and 2)
and 4c2ad6e (run 3). The job logs on GitHub expire; the three JSON files are the record.
