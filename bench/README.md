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
- On ARM (for example a DGX Spark) the `wex_check` section runs the NEON kernel of the fast CPU decoder. It reports pass or fail and seconds, not a decode speed. For the decode speed of the NEON kernel against the scalar kernel on your CPU, run `bash wex4/research/fast-cpu-decode/synth.sh speed` after it (synthetic records, no model needed) and add its output to the issue.
- The GPU copy test is CUDA-only. On Intel Arc or AMD it reports `skipped`, which is expected.
