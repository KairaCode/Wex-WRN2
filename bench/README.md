# Hardware benchmark for testers

A short, self-contained benchmark of the things that decide how a streamed Mixture-of-Experts model
runs on a machine: RAM bandwidth, host ↔ GPU copies, reads from the volume the models live on, and the
repository's own CPU checks. It needs only `gcc`, `python3` (standard library) and, for the GPU copy
test, `nvcc`. It installs nothing and sends nothing anywhere.

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
- The GPU copy test is CUDA-only. On Intel Arc or AMD it reports `skipped`, which is expected.
