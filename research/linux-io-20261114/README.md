# Linux read paths for expert records on two GitHub Actions runners (2026-11-14)

The first two runs of [`bench/wex_io.py`](../../bench/README.md). Both ran on GitHub-hosted runners: shared
cloud virtual machines with a virtual disk. **They are not NVMe hardware and say nothing about it.** No
Linux machine with a real drive was available.

## Question

Through which Linux read path should an engine read the expert records of a WEX4 container: `pread`
through the page cache (what `c/expert_store_wex.c` does on Linux), with or without `posix_fadvise` hints,
`pread` with `O_DIRECT` (what `c/qwen38_wex.h` does), several threads, or `io_uring` with several records
or sub-reads in flight? And is any difference large and steady enough to change the store?

## Setup

- **Machines:** two `ubuntu-24.04` runners of GitHub Actions (Ubuntu 24.04.5, 4 logical CPUs, 16 GB of
  RAM, ext4), one per run. Run 1: AMD EPYC 7763, kernel 7.0.0-1017-azure, runner image 20261111.370.1;
  the block device reports model `Virtual Disk`, `read_ahead_kb` 128, `max_sectors_kb` 2048, scheduler
  `none`. Run 2: Intel Xeon 6975P-C, kernel 7.0.0-1014-aws, runner image 20261111.127.1; the block device
  reports `read_ahead_kb` 128 and `max_sectors_kb` 128, no model and no scheduler. What else ran on the
  hosts is not known.
- **Container:** synthetic, written by the script with the reference writer: WEX4 4.0, 8 x 48 RAW expert
  records of 13,369,344 bytes of seeded random bytes, 5,134,127,104 bytes, root `13389132...8cd61d` in
  both runs. No model weights.
- **Build:** this repository at commit 0971dc8, `gcc -O2` of the runner image (version not recorded).
- **Command:** `python3 -I bench/wex_io.py --dir "$RUNNER_TEMP" --drop-caches` (job `linux-io-bench` of
  `.github/workflows/ci.yml`, dispatched by hand): 5 timed passes per read path, 48 records
  (641,728,512 bytes) per pass, the passes of all paths interleaved, a pass reading the same records on
  every path.
- **Cold cache:** `/proc/sys/vm/drop_caches` was written before every pass (the runner has `sudo`), and
  each pass dropped the file's pages with `POSIX_FADV_DONTNEED`. `mincore` found 0.0% of the pass's pages
  in the page cache before each of the 95 cold passes of each run, and 100% before the warm ones. A cache
  of the host under the virtual disk cannot be seen or dropped from inside the VM.

## Results

**Identity.** In both runs each of the 20 read paths read 8 records whose SHA-256 matched the index
before anything was timed. The timed passes do not hash.

**Speed.** MB = 10^6 bytes. "ms per record" is the median over the 5 passes of the pass's median record
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

## What the two runs show, and what they do not

- **No read path wins on both machines.** In run 1 the store's path of today (row 1) is among the
  slowest cold paths, and `POSIX_FADV_RANDOM` and `O_DIRECT` on one thread are 19% and 16% faster with
  ranges that do not overlap row 1's. In run 2 row 1 is the fastest cold path and the same two are 5% and 4% slower. A change
  of the store's Linux read path is therefore **not supported** by these runs, and none was made.
- **More reads in flight gave nothing that the disk could deliver.** Threads, `io_uring` depth and
  sub-reads all end between 418 and 481 MB/s in run 1 and between 511 and 540 MB/s in run 2, whatever the
  depth. Reading this as a throughput limit of the virtual disk is an interpretation: the limit of the
  runner's disk was not looked up or measured apart. An NVMe drive is expected to behave differently
  here; that is an expectation, not a result.
- **`POSIX_FADV_WILLNEED` on the next records and `POSIX_FADV_SEQUENTIAL` did not help** in either run
  (0.92x to 1.00x).
- **`io_uring` with one record in flight equals `pread`** with `O_DIRECT` in both runs (447 against 456,
  512 against 520 MB/s): on these disks the system-call path is not what costs.
- **Warm page cache:** 15.3 and 10.2 GB/s, with no drive in it. A store that opens with `O_DIRECT` gives
  that second-level cache up; whether an engine wants it depends on its own RAM cache and was not
  measured.

## Limits

- Two machines, one run of the script each (5 passes per path in a run). Both are cloud VMs with a
  virtual disk; the two runs landed on different hosts with different CPUs, kernels and disks, so they
  are two samples, not a repeat.
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

The script and the C program are `bench/wex_io.py` and `bench/wex_io.c` at commit 0971dc8. The job logs
on GitHub expire; the two JSON files are the record.
