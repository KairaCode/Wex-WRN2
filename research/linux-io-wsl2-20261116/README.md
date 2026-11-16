# Linux read paths for expert records under WSL2 on a desktop (2026-11-16)

One full run of [`bench/wex_io.py`](../../bench/README.md) under WSL2 on a Windows 10 desktop. It is the
first run of the benchmark outside GitHub Actions, and the first on a machine whose drive is an NVMe SSD.
**It is still not a run on a bare NVMe drive:** the file system is ext4 on the virtual disk of WSL2, a VHDX
file that Windows keeps on the SSD, so the Hyper-V virtual disk layer and the Windows storage stack sit
between Linux and the drive. The numbers say what a Linux engine under WSL2 would see on this machine,
not what Linux on that drive would see.

## Question

The question of [research/linux-io-20261114](../linux-io-20261114/README.md) and
[research/linux-prefetch-20261115](../linux-prefetch-20261115/README.md): through which Linux read path
should an engine read the expert records of a WEX4 container, and does a prefetch of the next record
help, with and without work between the reads? Here on a machine with a faster disk than the cloud VMs of
those runs.

## Setup

- **Machine:** Intel Core i7-7700K (4 cores, 8 threads, AVX2), 32 GB of RAM (26.2 GB given to WSL2,
  `MemTotal` of the JSON), NVIDIA RTX 3070 (not used by the benchmark), Windows 10 (10.0.19045). The
  virtual disk of the distribution lies on drive C:, a Samsung SSD 970 EVO Plus 500GB (NVMe, PCIe), read
  from Windows after the run; the drive's state (fill, temperature) was not recorded.
- **Linux:** Ubuntu 26.04.1 under WSL2, kernel 5.10.102.1-microsoft-standard-WSL2, Python 3.14.4, glibc
  2.43. Test file system: ext4 on the WSL2 virtual disk (a VHDX file on the NVMe SSD). The block device
  reports model `Virtual Disk`, `rotational` 1, `read_ahead_kb` 128, `max_sectors_kb` 512, `nr_requests`
  664, scheduler `none`.
- **Build:** this repository at commit ef71574, `bench/wex_io.c` built by the script with the gcc of the
  distribution (version not recorded).
- **Command:** `python3 -I bench/wex_io.py --dir DIR --drop-caches`, DIR a folder on that ext4 file
  system: 5 timed passes per read path (27 paths), 48 records (641,728,512 bytes) per pass, the passes of
  all paths interleaved.
- **Container:** synthetic, written by the script with the reference writer: WEX4 4.0, 8 x 48 RAW expert
  records of 13,369,344 bytes of seeded random bytes, 5,134,127,104 bytes, root `13389132...8cd61d`, the
  same file as in the runs on GitHub Actions. No model weights.
- **Cold cache:** `/proc/sys/vm/drop_caches` was written before every pass and each pass dropped the
  file's pages with `POSIX_FADV_DONTNEED`. `mincore` found 0.0% of the pass's pages in the page cache before
  each cold pass, and 100% before the warm ones. Windows keeps a cache of its own for the VHDX file; it
  cannot be seen or dropped from inside WSL2, so part of a "cold" read may have come from the RAM of the
  Windows side.
- **What else ran:** not recorded. The desktop was in use for other work that morning (the run started at
  06:32 local time).

## Results

**Identity.** Each of the 27 read paths read 8 records whose SHA-256 matched the index before anything was
timed. The timed passes do not hash.

**Speed.** MB = 10^6 bytes; 5 passes per path. "ms per record" is the median over the 5 passes of the
pass's median record time; with several reads in flight it grows with the depth.

| Read path | MB/s, median | min to max | against row 1 | ms per record |
|---|---:|---|---:|---:|
| page cache, `pread`, 1 thread (the store's path on Linux) | 1895 | 1579 to 1983 | 1.00x | 6.9 |
| page cache, `pread`, `POSIX_FADV_RANDOM` | 1118 | 1056 to 1168 | 0.59x | 11.8 |
| page cache, `pread`, `POSIX_FADV_SEQUENTIAL` | 1814 | 1506 to 1929 | 0.96x | 7.0 |
| page cache, `pread`, `WILLNEED` 1 record ahead | 1707 | 1530 to 1941 | 0.90x | 7.3 |
| page cache, `pread`, `WILLNEED` 4 records ahead | 1768 | 1623 to 1907 | 0.93x | 7.0 |
| page cache, `pread`, `readahead(2)` 1 record ahead | 1849 | 1726 to 1874 | 0.98x | 6.8 |
| page cache, `pread`, `WILLNEED` in chunks 1 record ahead | 2628 | 2527 to 2793 | 1.39x | 4.0 |
| page cache, `pread`, helper thread 1 record ahead | 1466 | 1282 to 1640 | 0.77x | 8.2 |
| page cache, `pread`, 4 threads | 3153 | 2791 to 3283 | 1.66x | 16.2 |
| page cache, `pread`, 8 threads | 3211 | 2835 to 3358 | 1.69x | 31.9 |
| page cache, `io_uring`, 8 records in flight | 3098 | 2856 to 3262 | 1.64x | 32.8 |
| `O_DIRECT`, `pread`, 1 thread | 2060 | 1924 to 2277 | 1.09x | 6.5 |
| `O_DIRECT`, `pread`, 2 threads | 3236 | 3053 to 3488 | 1.71x | 7.6 |
| `O_DIRECT`, `pread`, 4 threads | 3351 | 3284 to 3401 | 1.77x | 15.3 |
| `O_DIRECT`, `pread`, 8 threads | 3191 | 3155 to 3344 | 1.68x | 30.6 |
| `O_DIRECT`, `io_uring`, 1 record in flight | 2093 | 1836 to 2308 | 1.10x | 6.3 |
| `O_DIRECT`, `io_uring`, 1 record as 4 sub-reads | 2390 | 2242 to 2533 | 1.26x | 5.5 |
| `O_DIRECT`, `io_uring`, 1 record as 16 sub-reads | 2071 | 1991 to 2559 | 1.09x | 6.4 |
| `O_DIRECT`, `io_uring`, 4 records in flight | 3310 | 3245 to 3333 | 1.75x | 15.6 |
| `O_DIRECT`, `io_uring`, 8 records in flight | 3166 | 3120 to 3193 | 1.67x | 32.1 |
| `O_DIRECT`, `io_uring`, 16 records in flight | 3180 | 3104 to 3298 | 1.68x | 60.0 |
| `O_DIRECT`, `io_uring`, 8 records x 4 sub-reads | 3309 | 3139 to 3358 | 1.75x | 29.4 |
| page cache already warm, `pread`, 1 thread | 8405 | 7337 to 8863 | 4.44x | 1.5 |

**With 20 ms of work after each record** (a spin, not a decode), against the same work without a prefetch:

| Read path, 20 ms of work per record | MB/s, median (min to max) | against the same work without prefetch | announced pages in the cache before the read |
|---|---|---:|---|
| page cache, `pread`, 1 thread | 476 (474 to 485) | 1.00x | |
| `WILLNEED` 1 record ahead | 478 (470 to 486) | 1.01x | 3.9% |
| `WILLNEED` in chunks 1 record ahead | 572 (561 to 581) | 1.20x | 100% |
| helper thread 1 record ahead | 601 (598 to 603) | 1.26x | 100% |

**What the prefetches loaded** (one `mincore` call on the announced record just before its read). Without
work: `WILLNEED` on the whole record, 1 or 4 ahead, and `readahead(2)` 3.9% of its pages (about one
readahead window of 512 KiB, the `max_sectors_kb` of the device); `WILLNEED` in chunks of 128 KiB 19.0%
(9.9% to 20.1%), with 63.8 ms of the pass spent in the calls; the helper thread 2.5%. With 20 ms of work
both `WILLNEED` in chunks and the helper thread had 100% of the record in the cache at its read.

## What the run shows, and what it does not

- **More reads in flight pay here.** `O_DIRECT` on 4 threads read 3,351 MB/s, 1.77x the store's path
  (1,895), and `io_uring` with 4 records in flight 1.75x; every row with 2 or more reads in flight is
  between 1.64x and 1.77x, and beyond 4 in flight nothing gained more. On the cloud VMs of the earlier runs
  the same rows gained at most 11% (they ended between 418 and 545 MB/s). One read at a time leaves this
  disk path idle; that is the reading that fits, not a measurement of the disk apart.
- **`O_DIRECT` on one thread** is 1.09x the page cache, and `POSIX_FADV_RANDOM` is much slower (0.59x):
  with the kernel readahead off each 13.4 MB record is read in small requests.
- **A prefetch without work:** only `WILLNEED` in chunks gained (1.39x), and it is still slower than two
  plain reads in flight. `WILLNEED` on the whole record and `readahead(2)` loaded one request's worth and
  changed nothing; the helper thread was slower (0.77x), and why was not looked for.
- **A prefetch with 20 ms of work:** the helper thread 1.26x and `WILLNEED` in chunks 1.20x the same work
  without prefetch, both with the whole record in the cache at its read. On the cloud VMs `WILLNEED` in
  chunks gave 1.24x to 2.32x in the same rows.
- **Warm page cache:** 8.4 GB/s, less than on the cloud VMs (10 to 15 GB/s); the cause was not looked for.
- **Not shown:** anything about a bare NVMe drive under Linux. The VHDX layer and a Windows cache may both
  shape these numbers, and neither was measured apart. The store was not changed.

## Limits

- One run of the script, 5 passes per path, one machine. No interval is given beyond min to max.
- WSL2: ext4 on a virtual disk (VHDX) on the NVMe SSD; the Windows side may cache the file. Not a bare
  NVMe, not native Linux.
- The benchmark is not the stores: no decode, no engine cache, no eviction, no SHA-256 in the timed passes.
  The 20 ms of work is a spin, not a decode.
- The other load on the desktop during the run was not recorded; the gcc version was not recorded.

## Raw files

| File | What it is |
|---|---|
| `wex_io-wsl2.json` | `bench/out/wex_io.json` of the run, as written: every pass, the system, the block-device settings, the summary and the table. It holds no host or user name. |
