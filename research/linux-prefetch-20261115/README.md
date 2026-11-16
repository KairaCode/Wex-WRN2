# Prefetch of the next expert record on Linux, GitHub Actions runners (2026-11-15)

Eleven runs of [`bench/wex_io.py`](../../bench/README.md) with three read paths that try to load the
whole next record into the page cache before it is read, and four rows with 20 ms of work between
records. All ran on GitHub-hosted runners: shared cloud virtual machines with a virtual disk. **They are
not NVMe hardware and say nothing about it.** No Linux machine with a real drive was available. The
eleven runs landed on six kinds of virtual machine, called A to F below. Runs 10 and 11 were started to
get the rows with work on kind A; they landed on kinds D and C, and are recorded like the others.

## Question

The runs of 2026-11-14 ([linux-io-20261114](../linux-io-20261114/README.md)) found that one
`POSIX_FADV_WILLNEED` on the whole next record left 1.0% of it in the page cache before its read, so they
did not measure a prefetch of the record. Here: do one `readahead(2)` on the whole next record,
`POSIX_FADV_WILLNEED` in chunks no larger than the device's readahead, or a helper thread that `pread`s
the next record load it before it is read? Does that make the reads faster, with no work between records
and with 20 ms of work after each record? Is any gain large and steady enough to change
`c/expert_store_wex.c`?

## Setup

- **Machines:** `ubuntu-24.04` runners of GitHub Actions, one per run: 4 logical CPUs, about 16 GB of RAM
  (`MemTotal` 16,078,424 to 16,367,528 kB), ext4. Runner image 20261111.127.1 (run 1) and 20261111.370.1
  (runs 2 to 11).

  | Run | Kind | CPU | Kernel | Block device: model, `rotational`, `read_ahead_kb`, `max_sectors_kb`, `nr_requests` |
  |---|---|---|---|---|
  | 1 | A | Intel Xeon 6975P-C | 7.0.0-1014-aws | no model, 0, 128, 128, not reported |
  | 2 | B | Intel Xeon Platinum 8573C | 7.0.0-1017-azure | `Virtual Disk`, 1, 128, 256, 316 |
  | 3 | C | AMD EPYC 9V74 | 7.0.0-1017-azure | `MSFT NVMe Accelerator v1.0`, 0, 128, 256, 255 |
  | 4 | D | AMD EPYC 7763 | 7.0.0-1017-azure | `Virtual Disk`, 1, 128, 2048, 316 |
  | 5 | D | AMD EPYC 7763 | 7.0.0-1017-azure | `Virtual Disk`, 1, 128, 2048, 316 |
  | 6 | E | Intel Xeon Platinum 8573C | 7.0.0-1017-azure | `MSFT NVMe Accelerator v1.0`, 0, 128, 256, 255 |
  | 7 | F | AMD EPYC 9V45 | 7.0.0-1017-azure | `MSFT NVMe Accelerator v1.0`, 0, 128, 256, 255 |
  | 8 | C | AMD EPYC 9V74 | 7.0.0-1017-azure | `MSFT NVMe Accelerator v1.0`, 0, 128, 256, 255 |
  | 9 | F | AMD EPYC 9V45 | 7.0.0-1017-azure | `MSFT NVMe Accelerator v1.0`, 0, 128, 256, 255 |
  | 10 | D | AMD EPYC 7763 | 7.0.0-1017-azure | `Virtual Disk`, 1, 128, 2048, 316 |
  | 11 | C | AMD EPYC 9V74 | 7.0.0-1017-azure | `MSFT NVMe Accelerator v1.0`, 0, 128, 256, 255 |

  A kind is a CPU model with a block device and its settings. Kind A has the CPU, kernel and block-device
  settings of runs 2 and 3 of 2026-11-14, kind D those of its run 1. `MSFT NVMe Accelerator v1.0` is the
  model that a virtual NVMe device of the host reports; what is behind it is not known. Whether two runs
  shared a host, and what else ran on the hosts, is not known.
- **Container:** synthetic, written by the script with the reference writer: WEX4 4.0, 8 x 48 RAW expert
  records of 13,369,344 bytes (3,264 pages) of seeded random bytes, 5,134,127,104 bytes, root
  `13389132...8cd61d` in all eleven runs, the same as on 2026-11-14. No model weights.
- **Build:** this repository at commit 5372032 (run 1), 13181df (runs 2 to 9) and 3842c0d (runs 10 and
  11), `gcc -O2` of the runner image (version not recorded). 13181df adds `gap_ms` and the four rows with
  work; the other paths read as at 5372032. 3842c0d changes only documentation and this folder:
  `bench/wex_io.c` and `bench/wex_io.py` are as at 13181df.
- **Command:** `python3 -I bench/wex_io.py --dir "$RUNNER_TEMP" --drop-caches` (job `linux-io-bench` of
  `.github/workflows/ci.yml`, dispatched by hand): 5 timed passes per read path, 48 records
  (641,728,512 bytes) per pass, the passes of all paths interleaved, a pass reading the same records on
  every path. 27 read paths (23 in run 1).
- **The new paths:** each announces a record when the read of the record before it starts (`ahead=1`).
  `readahead(2)`: one call on the whole record. `WILLNEED` in chunks: 102 calls of 128 KiB, the
  `read_ahead_kb` of the device on all six kinds. Helper thread: a second thread `pread`s each announced
  record on the same descriptor into a buffer of its own; it starts a record only after the one before
  it, and never one that is not announced yet. One `mincore` call just before each announced record is
  read counts its pages in the page cache (47 announced records per pass).
- **Work:** in the four rows with `gap_ms=20` the reading thread spins 20 ms on the clock after each
  record (inside the timed pass, outside the record's own time). It stands for a decode but is none, and
  20 ms is a chosen value, not a measured decode time.
- **Cold cache:** `/proc/sys/vm/drop_caches` was written before every pass, and each pass dropped the
  file's pages with `POSIX_FADV_DONTNEED`. `mincore` found 0.0% of the pass's pages in the page cache
  before each of the 110 (run 1) or 130 cold passes of each run, and 100% before the warm ones. A cache of
  the host under the virtual disk cannot be seen or dropped from inside the VM.

## What the kernel does with one hint (source, not a measurement)

From the source of Linux 7.0 (tag `v7.0` of the kernel's Git tree). The runners run the vendor kernels
7.0.0-1014-aws and 7.0.0-1017-azure, whose own patches were not read.

- `readahead(2)` (`ksys_readahead` in `mm/readahead.c`) checks the descriptor and the file type, then
  calls `vfs_fadvise(..., POSIX_FADV_WILLNEED)`: it is the same request as `posix_fadvise` with
  `POSIX_FADV_WILLNEED` on the same range.
- `vfs_fadvise` calls the filesystem's own `fadvise` when it has one; ext4 has none
  (`fs/ext4/file.c`), so it calls `generic_fadvise` (`mm/fadvise.c`).
- For `POSIX_FADV_WILLNEED`, `generic_fadvise` calls `force_page_cache_readahead`, and
  `force_page_cache_ra` cuts the request to `max(bdi->io_pages, ra->ra_pages)` pages. `io_pages` is the
  device's `max_sectors` (`block/blk-settings.c`) and `ra_pages` comes from `read_ahead_kb`. So one call
  loads at most the larger of `max_sectors_kb` and `read_ahead_kb`.

## Results

**Identity.** In every run each read path read 8 records whose SHA-256 matched the index before anything
was timed. The timed passes do not hash.

### What an announcement left in the page cache

Share of the announced records' pages in the page cache just before their read, median of the 5 passes
(minimum to maximum), with no work between records. The last column is the time the reading thread spent
in the 102 calls per record of the chunked path, median per pass.

| Run (kind) | larger of `read_ahead_kb` and `max_sectors_kb`, as a share of one record | `WILLNEED` on the whole record | `readahead(2)` | `WILLNEED` in chunks | helper thread | chunked calls, ms per pass |
|---|---|---|---|---|---|---|
| 1 (A) | 128 KiB, 0.98% | 1.0% (1.0 to 1.0) | 1.0% (1.0 to 1.0) | 25.5% (22.8 to 31.1) | 2.5% (2.5 to 2.6) | 1,124 |
| 2 (B) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 20.9% (18.5 to 23.5) | 2.3% (2.2 to 2.5) | 152 |
| 3 (C) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 37.4% (24.9 to 47.2) | 2.4% (1.7 to 2.4) | 135 |
| 4 (D) | 2 MiB, 15.69% | 15.8% (15.8 to 16.1) | 15.8% (15.8 to 16.1) | 65.8% (62.4 to 69.3) | 2.8% (2.2 to 3.5) | 191 |
| 5 (D) | 2 MiB, 15.69% | 15.8% (15.8 to 16.1) | 15.8% (15.8 to 16.1) | 61.5% (57.6 to 65.5) | 2.8% (2.0 to 3.7) | 186 |
| 6 (E) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 29.5% (26.1 to 48.5) | 2.4% (2.3 to 2.5) | 141 |
| 7 (F) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 22.5% (21.1 to 23.2) | 2.3% (2.2 to 2.4) | 121 |
| 8 (C) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 46.0% (40.2 to 52.1) | 2.2% (2.1 to 2.4) | 132 |
| 9 (F) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 31.5% (26.0 to 34.9) | 2.3% (2.2 to 2.4) | 123 |
| 10 (D) | 2 MiB, 15.69% | 15.8% (15.8 to 16.1) | 15.8% (15.8 to 16.1) | 59.8% (57.7 to 63.1) | 2.8% (2.0 to 3.5) | 186 |
| 11 (C) | 256 KiB, 1.96% | 2.0% (2.0 to 2.1) | 2.0% (2.0 to 2.1) | 30.9% (16.4 to 38.2) | 2.2% (1.8 to 2.4) | 133 |

- **One `WILLNEED` and one `readahead(2)` on the whole record left the same share in every run**, and
  that share is close to the limit that the kernel source above gives: about 33, 66 and 516 of the 3,264
  pages, against a limit of 32, 64 and 512 pages (128 KiB, 256 KiB, 2 MiB). Where the few pages above the limit come from
  was not looked into. The `WILLNEED` row with 4 records ahead left the same shares.
- **In chunks** the 102 calls asked for the whole record; 20.9% to 65.8% of it was in the page cache when
  its read began. On kinds B to F the calls took 121 to 191 ms of each pass (6% to 14% of it); on kind A
  they took 1,124 ms, 89% to 91% of each pass of about 1.25 s: there the reading thread spent most of
  the pass inside `posix_fadvise`. Why the calls take that long on kind A was not looked into.
- **The helper thread** had 2.2% to 2.8% of the next record loaded when its read began: with no work
  between records it was not ahead of the reading thread.

### Speed with no work between records

MB = 10^6 bytes. MB/s median of the 5 passes (minimum to maximum), and against row 1 (`pread` through the
page cache, no prefetch, what `c/expert_store_wex.c` does on Linux).

| Run (kind) | `pread`, no prefetch (row 1) | `WILLNEED` on the whole record | `readahead(2)` | `WILLNEED` in chunks | helper thread | `O_DIRECT`, `pread`, 1 thread |
|---|---|---|---|---|---|---|
| 1 (A) | 544 (542 to 600) | 507 (500 to 521), 0.93x | 508 (502 to 510), 0.93x | 516 (509 to 518), 0.95x | 519 (513 to 524), 0.95x | 521 (515 to 525), 0.96x |
| 2 (B) | 243 (227 to 244) | 242 (227 to 243), 1.00x | 242 (227 to 243), 1.00x | 250 (238 to 251), 1.03x | 242 (227 to 243), 1.00x | 251 (238 to 252), 1.03x |
| 3 (C) | 430 (328 to 480) | 298 (256 to 405), 0.69x | 410 (254 to 480), 0.95x | 439 (374 to 465), 1.02x | 269 (201 to 447), 0.63x | 456 (403 to 478), 1.06x |
| 4 (D) | 400 (388 to 401) | 389 (384 to 406), 0.97x | 390 (381 to 393), 0.97x | 461 (458 to 481), 1.15x | 394 (392 to 396), 0.98x | 472 (456 to 475), 1.18x |
| 5 (D) | 393 (391 to 396) | 392 (372 to 399), 1.00x | 386 (384 to 388), 0.98x | 458 (454 to 458), 1.16x | 388 (384 to 401), 0.99x | 467 (450 to 469), 1.19x |
| 6 (E) | 445 (441 to 459) | 415 (403 to 435), 0.93x | 418 (398 to 425), 0.94x | 430 (424 to 441), 0.97x | 424 (408 to 433), 0.95x | 439 (417 to 444), 0.99x |
| 7 (F) | 438 (431 to 475) | 421 (419 to 440), 0.96x | 421 (418 to 425), 0.96x | 442 (434 to 446), 1.01x | 442 (435 to 445), 1.01x | 444 (436 to 448), 1.01x |
| 8 (C) | 378 (333 to 461) | 417 (240 to 466), 1.10x | 405 (298 to 420), 1.07x | 441 (292 to 500), 1.17x | 383 (217 to 413), 1.01x | 452 (322 to 492), 1.20x |
| 9 (F) | 456 (448 to 460) | 434 (413 to 459), 0.95x | 438 (413 to 442), 0.96x | 447 (430 to 454), 0.98x | 454 (428 to 462), 1.00x | 448 (442 to 459), 0.98x |
| 10 (D) | 394 (382 to 411) | 385 (382 to 400), 0.98x | 387 (375 to 388), 0.98x | 456 (452 to 473), 1.16x | 391 (379 to 401), 0.99x | 456 (451 to 464), 1.15x |
| 11 (C) | 452 (416 to 475) | 418 (375 to 473), 0.93x | 419 (325 to 434), 0.93x | 431 (392 to 475), 0.95x | 427 (297 to 476), 0.94x | 451 (421 to 502), 1.00x |

- **No prefetch path was faster than row 1 on every kind.** In runs 1, 6, 9 and 11 (kinds A, E, F and
  C) row 1 was the fastest column of the table. `WILLNEED` in chunks was faster with ranges apart only
  on kind D (1.15x, 1.16x and 1.16x), where `POSIX_FADV_RANDOM` (1.17x, 1.15x and 1.18x) and `O_DIRECT`
  (1.18x, 1.19x and 1.15x) were as fast, as in run 1 of 2026-11-14. On kinds B and F every column is
  within 5% of row 1. On kind C the passes of every row spread widely (row 1 from 328 to 480, 333 to 461
  and 416 to 475 MB/s) and no range is apart from row 1's.
- `readahead(2)` and one `WILLNEED` on the whole record read at the same speed (within 2%) in every run
  but runs 3 and 8, two of the three on kind C, whose passes spread most.

### Speed with 20 ms of work after each record

MB/s median (minimum to maximum), against the row with the same work and no prefetch. The second table
gives the median time of one record (the median over the 5 passes of the pass's median) and the share of
the next record found in the page cache before its read. A record's time runs from the start of the
announcing calls for the record ahead to the end of this record's read, so in the prefetch columns it
includes those calls (`announce_ms`; in the chunked column they are most of it, see below). It does not
include the 20 ms of work. Kind A has no such rows: run 1 was at
commit 5372032, which does not have them, and none of the ten later runs landed on kind A.

| Run (kind) | no prefetch | `WILLNEED` on the whole record | `WILLNEED` in chunks | helper thread |
|---|---|---|---|---|
| 2 (B) | 191 (182 to 191) | 191 (183 to 191), 1.00x | 247 (237 to 248), 1.30x | 240 (224 to 240), 1.26x |
| 3 (C) | 207 (188 to 221) | 252 (243 to 314), 1.22x | 480 (400 to 494), 2.32x | 439 (426 to 457), 2.12x |
| 4 (D) | 372 (361 to 376) | 389 (379 to 393), 1.05x | 471 (470 to 474), 1.27x | 391 (388 to 399), 1.05x |
| 5 (D) | 361 (359 to 366) | 386 (378 to 393), 1.07x | 464 (457 to 465), 1.28x | 388 (382 to 390), 1.08x |
| 6 (E) | 334 (316 to 336) | 335 (311 to 336), 1.00x | 492 (469 to 494), 1.47x | 433 (407 to 439), 1.30x |
| 7 (F) | 322 (316 to 326) | 323 (319 to 324), 1.00x | 482 (478 to 484), 1.49x | 447 (443 to 450), 1.39x |
| 8 (C) | 248 (239 to 312) | 292 (133 to 300), 1.17x | 474 (268 to 481), 1.91x | 426 (292 to 467), 1.72x |
| 9 (F) | 333 (321 to 335) | 334 (318 to 335), 1.00x | 481 (474 to 494), 1.44x | 461 (428 to 477), 1.38x |
| 10 (D) | 367 (357 to 368) | 386 (378 to 389), 1.05x | 454 (453 to 465), 1.24x | 391 (383 to 400), 1.06x |
| 11 (C) | 280 (159 to 310) | 264 (212 to 311), 0.94x | 469 (390 to 492), 1.68x | 432 (275 to 456), 1.54x |

| Run (kind) | no prefetch: ms per record | `WILLNEED`: ms, share loaded | in chunks: ms, share loaded | helper thread: ms, share loaded |
|---|---|---|---|---|
| 2 (B) | 50.0 | 49.9, 2.1% | 30.2, 55.6% (55.1 to 58.4) | 38.0, 34.8% (33.9 to 35.8) |
| 3 (C) | 20.1 | 20.2, 2.1% | 3.8, 96.5% (93.3 to 98.9) | 6.8, 76.9% (70.0 to 80.4) |
| 4 (D) | 8.3 | 6.7, 15.8% | 5.6, 87.7% (84.7 to 93.2) | 9.5, 67.2% (65.7 to 68.0) |
| 5 (D) | 19.2 | 5.4, 15.8% | 5.2, 85.8% (80.8 to 87.4) | 9.9, 66.5% (65.5 to 70.0) |
| 6 (E) | 19.9 | 19.8, 2.1% | 4.8, 93.6% (89.6 to 94.6) | 7.5, 68.6% (67.6 to 70.2) |
| 7 (F) | 19.9 | 19.7, 2.1% | 3.4, 88.3% (86.4 to 91.3) | 7.5, 75.2% (71.9 to 76.4) |
| 8 (C) | 20.2 | 20.4, 2.1% | 3.8, 94.9% (93.2 to 96.7) | 7.2, 76.0% (71.2 to 76.4) |
| 9 (F) | 20.1 | 20.0, 2.1% | 3.6, 89.5% (88.1 to 93.1) | 7.5, 76.3% (72.5 to 78.6) |
| 10 (D) | 20.7 | 5.7, 15.8% | 5.1, 83.8% (82.2 to 86.6) | 9.6, 68.2% (65.5 to 71.7) |
| 11 (C) | 22.1 | 22.0, 2.1% | 4.6, 92.0% (88.2 to 99.2) | 5.7, 76.8% (72.6 to 77.3) |

- **`WILLNEED` in chunks had the highest median of the four rows in every run with them** (runs 2 to 11,
  five kinds): 1.24x to 2.32x the MB/s of the same work without prefetch, with ranges apart from it in
  every run but run 8 (kind C: its slowest pass, 268 MB/s, is below the fastest pass without prefetch,
  312). 83.8% to 96.5% of the next record was in the page cache before its read, except on kind B
  (55.6%). A record took 3.4 to 5.6 ms (kind B: 30.2 ms, against 50.0 without prefetch). Of that time
  the chunked calls for the next record were on average 2.5 to 4.1 ms (the median `announce_ms` of the
  5 passes over the 47 records announced in a pass; kind B: 3.2 ms), so the read itself was the smaller
  part. This average and the median time of a record are not the same statistic, so their difference is
  only a rough size of the read.
- **The helper thread** gained 1.05x to 2.12x, with 34.8% to 76.9% of the next record loaded. Its median
  is below that of the chunks in every run; the ranges are apart in runs 4 to 7 and 10.
- **One `WILLNEED` on the whole record** gained 1.00x on kinds B, E and F (2.1% loaded) and 1.05x to
  1.07x on kind D (15.8% loaded). On kind C, where it also loaded 2.1%, it gave 1.22x, 1.17x and 0.94x:
  what it loaded does not explain these; the passes of that kind spread most.
- On kind D the reads with work between them took 8.3, 19.2 and 20.7 ms without prefetch, against about
  45 ms per record with no work. Why is not known: a cache or a throughput limit of the host under the
  virtual disk would both fit, and neither can be seen from inside the VM.

### The other rows

The rows that were also in the runs of 2026-11-14 tell the same story. On kind D (runs 4, 5 and 10)
`POSIX_FADV_RANDOM` was 1.17x, 1.15x and 1.18x row 1, `O_DIRECT` `pread` on one thread 1.18x, 1.19x and
1.15x, and `O_DIRECT` `io_uring` with 16 records in flight 1.20x, 1.25x and 1.20x (run 1 of 2026-11-14,
same kind: 1.19x, 1.16x and 1.22x). On kind A (run 1) the fastest cold paths were row 1 and `O_DIRECT`
`io_uring` with 16 records in flight, both 544 MB/s (the latter 0.99x and 1.00x of row 1 in runs 2 and 3
of 2026-11-14). On kinds B, E and F every cold row without work was between 0.93x and 1.06x of row 1.
The warm page cache read 8,999 to 30,865 MB/s. All 27 rows (23 in run 1) of each run are in its JSON
file.

## What the runs show, and what they do not

- **One `POSIX_FADV_WILLNEED` or one `readahead(2)` on a whole record is no prefetch of the record** on
  these machines: it loaded about one request of the device (the larger of `read_ahead_kb` and
  `max_sectors_kb`: 128 KiB to 2 MiB of a 13.4 MB record), as the Linux 7.0 source says it does.
  `readahead(2)` is the same request. To load a whole record, the hint has to be cut into chunks, or the
  record read.
- **With no work between records, no prefetch helped on every kind** (medians from 0.63x to 1.17x of
  row 1, faster with ranges apart only on kind D, where the paths that switch the kernel's readahead off
  or bypass the page cache were as fast). That the disk is the limit here, and a prefetch only shares it
  with the read it should get ahead of, is an interpretation; it fits the helper thread, which with no
  work was not ahead of the reading thread.
- **With 20 ms of work after each record, `WILLNEED` in chunks was the fastest of the three prefetches
  in every run where it ran** (ten runs, five kinds): 1.24x to 2.32x. The helper thread came second
  (1.05x to 2.12x). The work is a spin on one CPU, not a decode, and the rows were not run on kind A.
- **The chunked calls can hold the thread that makes them:** on kind A the reading thread spent about
  90% of each pass in them. An engine would make them on a thread other than its decode thread; that was
  not measured.
- **The store was not changed.** No path won on every kind seen without work; with work, the rows did
  not run on kind A, and the work stands in for a decode that the benchmark does not do. Also,
  `c/expert_store_wex.c` has no page-cache prefetch to switch: its prefetch call reads the records into
  its own slots before it returns. Loading the next record through the page cache while the engine
  decodes would be a new engine feature, to be measured with an engine A/B.

## Limits

- Six kinds of machine, eleven runs of the script, 5 passes per path in a run. Kinds C and D were seen
  three times, F twice, the others once; kind A once, without the rows with work. All are cloud VMs with
  a virtual disk.
- The work between records is a spin of 20 ms on the reading thread. A real decode takes another time,
  may use more cores, memory bandwidth or a GPU, and was not run.
- One chunk size (128 KiB, the `read_ahead_kb` of all six kinds) and one record ahead (`ahead=1`) for the
  new paths. Not run: other chunk sizes (for example `max_sectors_kb`), the chunked hints from a thread
  of their own, a helper thread with its own descriptor, more than one record ahead.
- `mincore` counts the pages that are in the page cache and read; pages whose read is still under way
  may be missed, so the shares may be low.
- The kernel source read is that of Linux 7.0, not the vendor kernels of the runners.
- A cold pass reads 642 MB and lasts 1.1 to 4.8 s; no longer pass was run.
- The benchmark reads records the way the stores do, but it is not the stores: no decode, no engine
  cache, no eviction, no SHA-256 at the first read in the timed passes. No engine ran.
- Not run: any NVMe or SATA drive, ARM, a filesystem other than ext4, ZFS, two drives.
- The runner's `gcc` version and the load of the hosts are not recorded.

## Raw files

| File | What it is |
|---|---|
| `run1-wex_io.json` | `bench/out/wex_io.json` of run 1, as the job printed it: every pass, the system, the block-device settings, the table. From the log of the `linux-io-bench` job of <https://github.com/KairaCode/Wex-WRN2/actions/runs/43701295776> |
| `run2-wex_io.json` | the same for run 2, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43702116495> |
| `run3-wex_io.json` | run 3, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43713043280> |
| `run4-wex_io.json` | run 4, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43713665900> |
| `run5-wex_io.json` | run 5, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43714259119> |
| `run6-wex_io.json` | run 6, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43714997354> |
| `run7-wex_io.json` | run 7, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43715447113> |
| `run8-wex_io.json` | run 8, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43716110067> |
| `run9-wex_io.json` | run 9, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43716777105> |
| `run10-wex_io.json` | run 10, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43740267418> |
| `run11-wex_io.json` | run 11, <https://github.com/KairaCode/Wex-WRN2/actions/runs/43740848880> |

All eleven files are JSON `version` 3. The script and the C program are `bench/wex_io.py` and
`bench/wex_io.c` at commit 5372032 (run 1) and 13181df (runs 2 to 11; runs 10 and 11 were built at
3842c0d, where these two files are as at 13181df). The job logs on GitHub expire; the JSON files are the
record.
