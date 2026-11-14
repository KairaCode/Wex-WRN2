# Hardware benchmark for testers

A short, self-contained benchmark of the things that decide how a streamed Mixture-of-Experts model
runs on a machine: RAM bandwidth, host ↔ GPU copies, reads from the volume the models live on, and the
repository's own CPU checks. It needs only `gcc`, `python3` (standard library) and, for the GPU copy
test, `nvcc`. It installs nothing and sends nothing anywhere.

## What you need first

Nothing is downloaded or installed by the benchmark itself. A standard Linux box needs:

| Needed for | Debian / Ubuntu | Fedora | Check |
|---|---|---|---|
| everything | `sudo apt install git gcc python3 make` | `sudo dnf install git gcc python3 make` | `gcc --version`, `python3 --version` (3.8 or later) |
| GPU copy test (NVIDIA only, optional) | the CUDA toolkit (`nvcc`); DGX OS already has it | same | `nvcc --version`; if it is installed but not found: `export PATH=/usr/local/cuda/bin:$PATH` |
| extra details in the report (optional) | `sudo apt install pciutils` (`lspci`); `zfsutils-linux` if you use ZFS | `pciutils` | not required, the fields just stay empty |

Also make sure that:
- the directory you pass to `--dir` has **about 20 GB free** (or use `--size-gb 4` / `--quick`);
- nothing heavy is running during the run (other benchmarks, downloads, model inference), since it would show up in the numbers.

No root, no Python packages, no model download: OpenMP comes with gcc, and the repository's checks generate their own small test files.

## Run it (Linux)

```bash
git clone https://github.com/KairaCode/Wex-WRN2.git
cd Wex-WRN2
python3 bench/bench.py --dir /path/on/your/model/volume
```

- `--dir`: a directory on the volume your models are on. A 16 GiB test file is written there and
  removed at the end (`--size-gb N` to change it).
- `--file BIG_FILE`: read an existing large file instead, for example a model file you have not read for
  a while. That is the closest thing to a cold read on ZFS, whose ARC cache cannot be dropped without
  root.
- `--quick`: about 2 minutes instead of about 10.
- `--skip h2d,wex_check`: skip sections.

The script prints each section as it goes and writes `bench/out/results.json`. **Read it before sharing
it.** It holds the CPU, GPU, RAM and filesystem details and the numbers, but no host name, user name or
paths. Then open a **Benchmark results** issue and attach the file.

## What each section measures

| Section | What | Why it matters for WEX |
|---|---|---|
| `sysinfo` | CPU (and AVX2/AVX-512), RAM, GPUs, kernel, filesystem type; ZFS recordsize, compression, primarycache and `direct` when it is ZFS | explains the numbers below |
| `ram_bw` | read / write / copy GB/s on 1 thread and on all threads | the RAM expert cache and CPU-computed experts are bound by it |
| `h2d` | pinned and pageable host ↔ device copies at 1, 8, 32 and 256 MiB | moving experts to VRAM; skipped without CUDA (Intel and AMD GPUs for now) |
| `disk` | sequential 8 MiB reads, random 1 MiB and 18 MiB (expert-sized) 4 KiB-aligned reads, with `O_DIRECT` when the filesystem accepts it; first and second pass | an expert miss is one such read; the second pass shows the cache (ZFS ARC, page cache) |
| `wex_check` | `scripts/check_all.sh`: pass/fail and seconds | the format code builds and works on your machine |

Notes:
- On ZFS, `O_DIRECT` is honoured only from OpenZFS 2.3 with `direct=standard` or `always`. Otherwise the
  reads go through the ARC, and the second pass mostly measures RAM. The JSON records which case you
  were in.
- On ARM (for example a DGX Spark) the `wex_check` section runs the scalar kernel of the fast CPU decoder: the AVX2 kernels are x86-64 only and there is no NEON version yet, so it checks correctness there, not speed.
- The GPU copy test is CUDA-only. On Intel Arc or AMD it reports `skipped`, which is expected.

## Read paths for expert records (Linux): `wex_io.py`

`bench.py` reads a plain file with one call per read. `wex_io.py` asks a narrower question: through which
Linux read path should an engine read the expert records of a WEX4 container? It needs `gcc` and `python3`
only, and Linux (`io_uring` is called through its system calls, so there is no liburing to install).

```bash
python3 -I bench/wex_io.py --dir /path/on/your/model/volume --drop-caches
```

What it does:

1. It writes a synthetic WEX4 container in `--dir` with the reference writer (`wex4/wex4.py`): 8 x 48 expert
   records of 13,369,344 bytes of seeded random bytes, 5.1 GB, no model weights. `--layers`, `--experts`,
   `--rows` and `--cols` change the size. The file is removed at the end unless `--keep` is given.
   `--file FILE` reads an existing container instead; `--dir` is then left out, and the filesystem and
   block device in the JSON are those of the file. CI runs `--file` on the smoke container; it was not
   run on a container with real weights.
2. It builds `bench/wex_io.c`, which opens the header page and the index with `c/wex4.h`, as the engine
   stores do, and reads whole expert records at their 4096-aligned offsets, in a seeded random order
   without repeats.
3. **Identity first:** every read path reads 8 records and compares their SHA-256 with the index. A path
   that the machine cannot run (no `O_DIRECT` on the filesystem, `io_uring` switched off or older than
   Linux 5.6) is reported as not run, with the reason; a path that fails the check is not timed and the
   script ends with `FAILED`. With `--strict` a path that was not run is a failure too (CI uses it,
   because all 20 paths run on its runners). The skip for an old kernel was not tried on one.
4. **Then speed:** `--runs` timed passes per path (default 5) of `--records` records each (default 48). The
   runs of all paths are interleaved and the order rotates, a run reads the same records on every path, and
   the buffers are allocated and touched and the `io_uring` ring is set up before the clock starts. The
   worker threads of the multi-thread paths are created after it starts.

The read paths:

| Path | What it stands for |
|---|---|
| page cache, `pread`, 1 thread | what `c/expert_store_wex.c` and `c/tools/wex4_verify.c` do on Linux today |
| the same with `POSIX_FADV_RANDOM` or `POSIX_FADV_SEQUENTIAL` | the kernel's readahead switched off or doubled |
| the same with `POSIX_FADV_WILLNEED` on the record 1 or 4 reads ahead | a hint for records the engine already knows it will need. The kernel may load only part of the record; the script measures how much (below) |
| page cache, `pread`, 4 and 8 threads; `io_uring`, 8 records in flight | several misses at once through the page cache |
| `O_DIRECT`, `pread`, 1 thread | what `c/qwen38_wex.h` does on Linux today for one read |
| `O_DIRECT`, `pread`, 2, 4 and 8 threads | one blocking read per thread |
| `O_DIRECT`, `io_uring`, 1 record in flight, whole or as 4 or 16 sub-reads | a lone miss; the sub-reads are what `V4_WEX_SUBREADS` does with overlapped I/O on Windows |
| `O_DIRECT`, `io_uring`, 4, 8 and 16 records in flight, and 8 records as 4 sub-reads each | a batch of misses from one thread |
| page cache already warm | the records are in RAM: a ceiling with no drive in it |

**Cold cache.** Before every pass `wex_io` asks the kernel to drop the file's pages
(`POSIX_FADV_DONTNEED`, no root needed), then measures with `mincore` how much of the pass is still in the
page cache; that share is the column "in page cache before" and must be about 0% for every row but the
last. `--drop-caches` also writes `/proc/sys/vm/drop_caches` before every pass when the user may (root, or
`sudo` without a password). Neither reaches a cache below the kernel: the ARC of ZFS, a RAID controller,
or the host of a virtual machine.

**What `WILLNEED` loads.** In the two `WILLNEED` paths one `mincore` call just before each announced
record is read counts how many of its pages are in the page cache (inside the timed pass, outside the
record's own time). The share is printed under the table and stored per pass as `ahead_cached`. In the
only full run that has it (a cloud VM, block device with `read_ahead_kb` 128) it was 1.0%: the kernel had
loaded about 130 KB of each 13.4 MB record, so those two rows did not measure a prefetch of the record.
Pages whose read is still under way may not be counted. The measurement needs 4096-byte pages.

The script prints a Markdown table (median, minimum and maximum MB/s over the runs, MB = 10^6 bytes, and
the median time of one record) and writes `bench/out/wex_io.json` with every pass, the CPU, the kernel,
the filesystem and the block-device queue settings (`read_ahead_kb`, scheduler), but no host name, user
name or path. Results from real NVMe drives are wanted: attach the JSON to a **Benchmark results** issue.

What it does not measure: the decode of a record, the engine's own caches, two drives (mirror, stripe),
and anything on Windows or macOS. The only runs so far are three on GitHub Actions runners, shared cloud
VMs with a virtual disk of two kinds, where no read path won on both kinds:
[research/linux-io-20261114](../research/linux-io-20261114/README.md).

What CI covers: the `linux`, `linux-arm` and `linux-sanitizers` jobs build `wex_io.c` and run
`--smoke --strict` (a 10 MB container, every path checked and run once; the `linux` job then reads the
same container again through `--file`). The full benchmark was not run on ARM. `scripts/check_all.sh`
does not build `wex_io.c`, so the Windows job does not see it: its stub for systems other than Linux
(it prints one `skipped` line) was built by hand with MinGW `gcc` and `clang`
(`-std=c99 -pedantic -Wall -Wextra`, no warning) and is not built by CI. The full run is the job
`linux-io-bench`, started by hand with `gh workflow run ci.yml --ref <branch> -f linux_io_bench=true`.
