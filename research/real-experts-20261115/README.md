# Real expert tensors of three MXFP4 checkpoints: round trips, WNR3 and general-purpose codecs (2026-11-15)

> **Note of 2026-11-16, after the merges.** Section 4 measured the Vulkan prototype of commit `1af6c0d`. The
> prototype was changed the same day by [research/vulkan-gpu-20261115](../vulkan-gpu-20261115/README.md): its
> shader keeps the scale word in a register and no longer loses the ceil-form decrements on the RTX 3070
> (the fault of section 4, found there on the selftest). That shader was not run on the records of this
> folder. `vulkan_diag.sh` now takes `wnr3_vk.c` and `wnr3.comp` from commit `1af6c0d` (the patch does not
> apply to the later `wnr3_vk.c`), so it rebuilds the diagnostic of this run from any later checkout.
> `vulkan.sh` runs the prototype of the checkout: on a later checkout it runs the shader with the fix, and its
> step 3 makes the `coherent` copy of that shader. Neither was run after the fix. Nothing else of this folder
> was changed.

Eight routed experts (two layers, four experts each) of each of three published checkpoints, fetched with
HTTP range requests. **This is a partial checkpoint, not a run on a real checkpoint:** 24 experts in all,
no whole shard, no tensor other than the six of each chosen expert. `config.json` gives 40 layers x 384
routed experts for DeepSeek V4.1 Flash, 94 x 128 for Qwen3-235B and 62 x 256 for MiniMax-M2.1; 8 of each
were read. Everything below is measured on these 24 experts only.

## Question

1. Do the naming profiles `deepseek`, `qwen3-moe` and `mixtral` of `wex4/wex4_convert.py` find the experts
   of real checkpoints of these families, and do pack, verify and unpack round-trip byte for byte on real
   tensors, with RAW and with WNR3 expert records? ([HELP-WANTED](../../docs/HELP-WANTED.md), section 5)
2. On the same expert records, what ratio and what decode speed do zstd, lz4, brotli, lzma, zlib and bz2
   give, against WNR3 with the reference decoder and with the fast CPU decoder? (section 8)
3. Does the Vulkan prototype of `wex4/research/vulkan-decode/` decode these records on a real GPU with the
   reference's bytes? (section 6; byte identity only, no speed)

## Verdict

1. Yes, on these 24 experts: each profile found its 8 experts (0 kept as ordinary tensors), and every
   round trip (RAW, WNR3, and for DeepSeek V4.1 also WEX1 → WEX3 → WEX4 → back) gave the source bytes.
2. WNR3 gave the smallest files on all three checkpoints (0.8796, 0.8919 and 0.8856 of raw, against
   0.9222, 0.8951 and 0.9043 for the best general-purpose codec measured, always brotli), and on the CPU
   a slower decode than every codec but lzma, bz2 and (on one thread) brotli at quality 11: about 200 MB/s
   on one thread with the fast decoder, against about 1,500 MB/s for zstd, on a machine shared with other
   jobs. On Qwen3-235B the margin is 0.0032 of raw, and with a model trained on one layer WNR3 is larger
   than zstd.
3. No: on an RTX 3070 the prototype of commit `1af6c0d` fails its own selftest and gives other bytes for
   the 16 DeepSeek V4.1 records (19% of the scale decrements of the ceil-form inverse are lost, the weights
   are right). The 16 records of the two other checkpoints decode right. A copy of the shader with its
   output buffer declared `coherent` gives the reference's bytes on all of them.

## Setup

**Data.** `fetch_experts.py` (standard library only) read, at a pinned revision, `config.json`,
`model.safetensors.index.json`, the safetensors header of each shard that holds a chosen expert (two range
requests: the 8-byte length, then the header) and then one range request per tensor. Every response had to
be HTTP 206 with the asked `Content-Range` and length, and every tensor the size of its dtype and shape. It
wrote one small safetensors file per source shard with exactly the fetched tensors (names, dtypes, shapes
and bytes unchanged, in the order of the source shard) and `fetch-log.json`: every request with its
repository, revision, file, byte range, byte count and SHA-256, and every file written with its SHA-256.
Started 2026-11-15 09:19 UTC; 162 requests, 0 retries, 317,018,971 bytes received (budget 1,500,000,000
bytes). Before the script existed, the `config.json` and `model.safetensors.index.json` of the three
checkpoints and of `amd/Kimi-K2.6-MXFP4` (not used) were read once with `curl` to choose the layers:
39,884,664 bytes, not in the log; with a few small requests to the Hugging Face model API (revision and
licence tag), the run downloaded about 357 MB in all. Nothing fetched is in the repository; the default
output folder `data/` is git-ignored. The weights stay under the licence of each checkpoint; this run used
them for research only and redistributes none of them.

| Name here | Repository | Revision | Profile | Licence tag on Hugging Face | Layers, experts | Tensor bytes | Shards read |
|---|---|---|---|---|---|---:|---|
| `deepseek-v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `314113f8a7dadb388215396ba5fbee2c610a04d6` | `deepseek` | mit | 10 and 30; 0 to 3 | 150,405,120 | `model-00013-of-00048`, `model-00033-of-00048` |
| `qwen3-235b` | `amd/Qwen3-235B-A22B-Instruct-2507-MXFP4` | `381342aab13be9f91265d0f6b1760470bfd2704d` | `qwen3-moe` | apache-2.0 | 23 and 70; 0 to 3 | 80,216,064 | `model-00007-of-00026`, `model-00019-of-00026` |
| `minimax-m21` | `amd/MiniMax-M2.1-MXFP4` | `bbf15c45e91cfa2b4970f42619750415a214e24b` | `mixtral` | other (modified-mit) | 15 and 46; 0 to 3 | 60,162,048 | `model-00007-of-00026`, `model-00019-of-00026` |

The layers are N/4 and 3N/4 (rounded down) of the N layers of `config.json`; the experts are 0 to 3 of
each, not a random pick. One expert record (its decoded slab: the three scale tensors, then the three
weight tensors) is 18,800,640 bytes for DeepSeek V4.1 (hidden 5120, expert intermediate 2304), 10,027,008
for Qwen3-235B (4096, 1536) and 7,520,256 for MiniMax-M2.1 (3072, 1536). DeepSeek's own export stores the
scales as F8_E8M0 and the weights as I8; the two AMD exports store both as U8.

**Machine.** Intel Core i7-7700K (4 cores, 8 logical CPUs), 32 GB of RAM, Windows 10 Pro (build
19045), NVIDIA GeForce RTX 3070 (8 GB, driver 621.01, Vulkan 1.4.357). MSYS2 UCRT64: gcc 16.2.0,
Python 3.14.10, libzstd 1.5.7, lz4 1.11.1, brotli 1.2.1, zlib 1.4 (the one Python reports). This
repository at commit `1af6c0d` plus the files of this folder. Other heavy jobs ran on the machine during
this run; section 3 says how that was handled and what it means for the speeds.

**Commands**, from the repository root, after `bash scripts/check_all.sh` (which builds the DLLs that
`run.sh` uses):

```sh
python -I research/real-experts-20261115/fetch_experts.py DATA
bash research/real-experts-20261115/run.sh DATA WORK        # round trips, models, records (447 s here)
bash research/real-experts-20261115/vulkan.sh WORK          # needs a Vulkan device (157 s here)
bash research/real-experts-20261115/vulkan_diag.sh WORK     # the same, which bytes differ (46 s here)
WAIT_IDLE=20 bash research/real-experts-20261115/bench.sh WORK 11
```

`DATA` and `WORK` are folders outside the repository here (both default to git-ignored folders next to
the scripts). `WORK` took about 7 GB here.

## Results

### 1. Round trips (HELP-WANTED section 5)

`run.sh` did the following per checkpoint, and stops at the first step that fails; it ended with `run.sh:
all round trips passed` ([run.log](run.log)).

- `pack --experts raw --names PROFILE` printed `8 expert records; 0 experts kept as ordinary tensors
  (float or FP8, not MXFP4)` for each of the three profiles, so every one of the 48 tensors of each
  checkpoint went into an expert record.
- `wex4.py verify` checked every hash and decoded every record, with the DLL that holds the reference WNR3
  decoder and with the one that holds the fast decoder; both gave the same content digest.
  `c/tools/wex4_verify.c` (8 threads) printed the container's root.
- `unpack` printed `sha256 identical` for both files of each checkpoint, and `cmp` found each rebuilt file
  identical to its source.
- The same with WNR3 expert records (`pack --experts nr3`, no canonical zero), once with each of three
  models (below): verify as above, `unpack` with `sha256 identical`, `cmp` identical. The slabs decoded from
  the WNR3 records equal the slabs of the RAW records.

| Checkpoint, profile | Source files (2) | RAW container | WNR3 container, model trained on the 8 | Content digest of both, first 16 hex digits |
|---|---:|---:|---:|---|
| DeepSeek V4.1 Flash, `deepseek` | 150,410,224 B | 150,425,600 B | 132,382,720 B | `dc73a09559d7d5b9` |
| Qwen3-235B, `qwen3-moe` | 80,221,904 B | 80,236,544 B | 71,626,752 B | `84e8b52b7eda675a` |
| MiniMax-M2.1, `mixtral` | 60,168,176 B | 60,182,528 B | 53,362,688 B | `1cb9c920a8ae38fa` |

**DeepSeek V4.1 also through the production path.** A WEX1 container of the 8 experts was written by
`c/tools/repack_v4_wex.py` (with a `config.json` of 31 layers x 4 experts and the V4.1 dimensions, so that
layers 10 and 30 hold experts 0 to 3 and the other entries stay empty), WEX3 containers with WNR3 records by
`c/tools/build_nr3.py container --train` on the 8 experts, without and with `--cz`. Each was upgraded in
place with `upgrade --names deepseek`, verified as above, and downgraded: the WEX1 file and both WEX3 files
came back byte for byte (same SHA-256). `pack --experts nr3` with the model `build_nr3.py` saved (and `--cz`
for the second) wrote the same 8 records, size and SHA-256, as `build_nr3.py`. With `--cz` the container
is value-exact and `unpack` says `DIFFERS (value-exact container)` for both files, as it must when
the source has FP4 -0 nibbles; the content digest of that container differs from the source's for the same
reason.

Not covered here: the profiles `deepseek-quark`, `qwen38`, `kimi-k26`, `minimax-m3` and `mistral-large4`;
`upgrade` with the `qwen3-moe` and `mixtral` names (their tensors were not in a WEX1 or WEX3 container: the
tools that write those know the DeepSeek and Qwen3.8 names only, so `pack` is the way in for these
families); the refusal `incomplete expert` on a real checkpoint (every expert fetched was whole); a
checkpoint with experts of another dtype.

### 2. WNR3 ratio on these records

WNR3 needs a WNM2 model. For each checkpoint three were trained with the training of `build_nr3.py`
(`records.py model`): on the 8 experts, on the 4 of the first layer and on the 4 of the second. "Held out"
is a layer the model did not see. Ratio = stored bytes / decoded bytes; byte-exact (no canonical zero)
unless noted. All 80 WNR3 records of the table (10 rows of 8) have the ceil form on all three matrices
(record flag byte 0x07; 0x0f with canonical zero). The model (59,408 B) is stored once per container and is not in the
ratios; with it, the 8 records of the first row would be 0.0004 of raw larger (Qwen3-235B 0.0007,
MiniMax-M2.1 0.0010).

| Checkpoint | Model trained on | First layer | Second layer | All 8 records | Range of the 8 |
|---|---|---:|---:|---:|---|
| DeepSeek V4.1, L10 and L30 | the 8 | 0.8798 | 0.8795 | 0.8796 | 0.8789 to 0.8806 |
| | the 4 of L10 | 0.8798 | 0.8795 (held out) | 0.8797 | 0.8789 to 0.8806 |
| | the 4 of L30 | 0.8798 (held out) | 0.8794 | 0.8796 | 0.8786 to 0.8806 |
| | the 8, canonical zero (value-exact) | 0.8523 | 0.8520 | 0.8521 | 0.8512 to 0.8532 |
| Qwen3-235B, L23 and L70 | the 8 | 0.8882 | 0.8956 | 0.8919 | 0.8840 to 0.8979 |
| | the 4 of L23 | 0.8843 | 0.9195 (held out) | 0.9019 | 0.8795 to 0.9240 |
| | the 4 of L70 | 0.9469 (held out) | 0.8908 | 0.9189 | 0.8901 to 0.9624 |
| MiniMax-M2.1, L15 and L46 | the 8 | 0.8860 | 0.8852 | 0.8856 | 0.8824 to 0.8916 |
| | the 4 of L15 | 0.8856 | 0.8866 (held out) | 0.8861 | 0.8840 to 0.8922 |
| | the 4 of L46 | 0.8867 (held out) | 0.8848 | 0.8858 | 0.8818 to 0.8916 |

- DeepSeek V4.1 with canonical zero gives 0.8521 here; the production container (DeepSeek V4.1 REAP-192,
  7,680 records, a model trained on 80 of them) is 0.854 of raw
  ([docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)). Without canonical zero, 0.8796; the lab figure of
  2026-11-08 for the ceil form without canonical zero on 40 held-out V4.1 experts is 0.88179
  ([wnr3-5090-20261107/RESULTS.md](../wnr3-5090-20261107/RESULTS.md)).
- For DeepSeek V4.1 a model from one layer codes the other layer within 0.0001 of that layer's own model,
  for MiniMax-M2.1 within 0.0018 (0.8866 against 0.8848 on L46). For Qwen3-235B it does not: 0.9195 and
  0.9469 on the held-out layer against 0.8908 and 0.8843 with that layer's own model. So for that family a
  model trained on a few experts of one layer does not stand for the checkpoint; the 80 training experts
  spread over every layer that `build_nr3.py --train-lab80` picks would be the test, and it was not run.
- Canonical zero was not tried on the two AMD exports. Whether their weights hold FP4 -0 nibbles at all
  was not checked.

### 3. General-purpose codecs against WNR3 (HELP-WANTED section 8)

**What was measured.** `bench.sh` takes, per checkpoint, the 8 slabs and the 8 WNR3 records made with the
model trained on the 8 (the first row of each checkpoint above). `bench_real.c` decodes the WNR3 records
with the reference decoder (`wex_nr2_decode` of `c/`) and with the fast CPU decoder of
`wex4/research/fast-cpu-decode/` (its AVX2 kernels; kernel 1 is its default), and compresses and decodes
each slab with lz4, lz4hc, brotli (window 24) and zstd through their C libraries; `baselines.py` does the
same with `compression.zstd`, `lzma`, `zlib` and `bz2` of the Python standard library. Each slab is
compressed whole, as one frame. Before any timing every decoder's output was compared with the slab: all
equal. A row is 11 timed passes over the 8 records after one pass that is not counted; the table gives
MB/s = decoded bytes per record / median time per record (MB = 10^6 bytes). "1" is one thread; "8" is 8
records decoded at the same time on 8 threads, one per thread (for Python, a thread pool, which runs in
parallel only where the codec releases the GIL). "(a)" is the fast decoder with its own 8 threads on the
streams of one record, the records one after the other. Kernel 2 was run only on 1 thread and in form
(a). gcc 16.2.0 `-std=c99 -O3 -march=haswell -fopenmp`, Python 3.14.10. The compression time is printed
in the raw files but is not a speed measurement (8 threads, shared machine). Run of 11:11 to 11:28 local
time (UTC+1), raw files in [bench-1/](bench-1/).

| Decoder | Ratio, DeepSeek V4.1 | Qwen3-235B | MiniMax-M2.1 | MB/s, 1 / 8 threads, DeepSeek V4.1 | Qwen3-235B | MiniMax-M2.1 |
|---|---:|---:|---:|---:|---:|---:|
| WNR3, reference decoder `wex_nr2_decode` | 0.8796 | 0.8919 | 0.8856 | 25.2 / 148 | 25.1 / 149 | 24.9 / 147 |
| WNR3, fast decoder, kernel 1 (its default) | 0.8796 | 0.8919 | 0.8856 | 197 / 847 (834 (a)) | 200 / 864 (838 (a)) | 202 / 850 (823 (a)) |
| WNR3, fast decoder, kernel 2 (gathers) | 0.8796 | 0.8919 | 0.8856 | 162 / 891 (a) | 164 / 887 (a) | 163 / 881 (a) |
| lz4 (`LZ4_compress_default`) | 0.9733 | 0.9643 | 0.9727 | 9,567 / 14,428 | 10,244 / 14,466 | 9,915 / 13,985 |
| lz4hc level 9 | 0.9598 | 0.9547 | 0.9595 | 11,176 / 12,264 | 11,788 / 12,166 | 14,247 / 12,301 |
| lz4hc level 12 | 0.9557 | 0.9527 | 0.9556 | 11,521 / 12,302 | 11,545 / 12,357 | 15,853 / 12,063 |
| brotli quality 1, window 24 | 0.9300 | 0.8994 | 0.9082 | 270 / 1,432 | 312 / 1,616 | 314 / 1,692 |
| brotli quality 5, window 24 | 0.9266 | 0.8968 | 0.9049 | 282 / 1,650 | 331 / 1,732 | 343 / 1,878 |
| brotli quality 9, window 24 | 0.9259 | 0.8961 | 0.9043 | 284 / 1,701 | 334 / 1,749 | 345 / 1,870 |
| brotli quality 11, window 24 | 0.9222 | 0.8951 | 0.9047 | 145 / 933 | 159 / 973 | 157 / 989 |
| zstd level 1 (libzstd from C) | 0.9284 | 0.8985 | 0.9068 | 1,499 / 6,721 | 1,498 / 6,179 | 1,499 / 6,396 |
| zstd level 3 (libzstd from C) | 0.9301 | 0.8992 | 0.9077 | 1,481 / 5,487 | 1,481 / 6,782 | 1,484 / 6,440 |
| zstd level 19 (libzstd from C) | 0.9268 | 0.8966 | 0.9055 | 1,545 / 6,920 | 1,514 / 6,500 | 1,506 / 6,499 |
| zstd level 1 (`compression.zstd`) | 0.9284 | 0.8985 | 0.9068 | 929 / 2,396 | 906 / 2,267 | 896 / 2,216 |
| zstd level 3 (`compression.zstd`) | 0.9301 | 0.8992 | 0.9077 | 934 / 2,416 | 899 / 2,275 | 898 / 2,150 |
| zstd level 9 (`compression.zstd`) | 0.9270 | 0.8971 | 0.9054 | 901 / 2,287 | 852 / 2,083 | 813 / 2,014 |
| zstd level 19 (`compression.zstd`) | 0.9268 | 0.8966 | 0.9055 | 844 / 2,162 | 747 / 1,949 | 726 / 1,817 |
| zstd level 22 (`compression.zstd`) | 0.9268 | 0.8966 | 0.9055 | 735 / 1,935 | 737 / 1,877 | 721 / 1,817 |
| lzma preset 0 | 0.9403 | 0.9161 | 0.9244 | 47.9 / 276 | 48.1 / 279 | 47.8 / 276 |
| lzma preset 6 | 0.9306 | 0.9045 | 0.9117 | 48.8 / 287 | 48.9 / 287 | 48.5 / 282 |
| lzma preset 9e | 0.9306 | 0.9043 | 0.9117 | 48.5 / 284 | 48.9 / 285 | 48.5 / 281 |
| zlib level 1 | 0.9349 | 0.9052 | 0.9139 | 293 / 1,284 | 276 / 1,211 | 280 / 1,214 |
| zlib level 6 | 0.9302 | 0.9008 | 0.9089 | 301 / 1,313 | 284 / 1,237 | 288 / 1,237 |
| zlib level 9 | 0.9297 | 0.8999 | 0.9083 | 303 / 1,322 | 285 / 1,228 | 290 / 1,249 |
| bz2 level 9 | 0.9420 | 0.9180 | 0.9247 | 25.0 / 96.6 | 25.5 / 102 | 25.2 / 95.7 |

**Ratio.**

- WNR3 is the smallest on all three checkpoints. The best general-purpose codec was brotli each time:
  quality 11 on DeepSeek V4.1 (0.9222, WNR3 0.0426 of raw smaller) and on Qwen3-235B (0.8951, 0.0032
  smaller), quality 9 on MiniMax-M2.1 (0.9043, 0.0187 smaller). zstd at its best level: 0.9268, 0.8966
  and 0.9055.
- On Qwen3-235B the margin depends on the model: WNR3 with a model trained on the 4 experts of one layer
  (0.9019 and 0.9189 for the 8 records, half of them held out, section 2) is larger than zstd level 19
  (0.8966) and than brotli quality 1 (0.8994).
- The level hardly matters: zstd level 1 to 22 changes the ratio by at most 0.0033 (and level 3 is larger
  than level 1), lzma 9e is lzma 6 within 0.0002, brotli 11 against 9 gains 0.0037 at most and loses
  0.0004 on MiniMax-M2.1.
- The general-purpose codecs do better on the two AMD exports (zstd 19: 0.8966 and 0.9055) than on
  DeepSeek's own (0.9268), while WNR3 stays between 0.8796 and 0.8919. Why was not looked into. For
  comparison, an older measurement on 16 DeepSeek V4 slabs gives zstd 19 0.917 and lzma 6 0.920
  ([docs/WNR2-REFERENCE.md](../../docs/WNR2-REFERENCE.md), section 7; another checkpoint and tool, not
  repeated here).

**Decode speed.**

- One thread: the fast WNR3 decoder gives 197 to 202 MB/s (kernel 1) and 162 to 164 (kernel 2), the
  reference decoder about 25. zstd from C gives 1,481 to 1,545 MB/s, 7.3 to 7.8 times the fast WNR3
  decoder; lz4 and lz4hc 9,567 to 15,853; brotli 270 to 345 (quality 1 to 9) and 145 to 159 (quality
  11); zlib 276 to 303; lzma about 48; bz2 about 25. So on one thread the fast WNR3 decoder is faster
  than lzma, bz2 and brotli quality 11 only.
- 8 threads: the fast WNR3 decoder gives 823 to 891 MB/s, whether 8 records run at once or its own 8
  threads share one record; zstd from C 5,487 to 6,920; brotli 933 to 1,878; zlib 1,211 to 1,322; lzma
  276 to 287; bz2 96 to 102. On 8 threads kernel 1 is 4.1 to 4.3 times its one-thread speed and kernel 2
  5.4 to 5.5 times, on 4 cores with 2 logical CPUs each.
- `compression.zstd` decodes the same frames with the same libzstd (1.5.7) at 721 to 934 MB/s on one
  thread, against 1,481 to 1,545 from C. The C harness decodes into a buffer it reuses; Python returns a
  new object for every record. That this is the cause was not checked.
- lz4 runs at 10 to 16 GB/s on one thread and gains at most 1.5 times on 8 (lz4hc at most 1.1 times, and
  it is slower on 8 threads than on one for MiniMax-M2.1); its rows last 0.5 to 2 ms per record and are
  the least steady of the table.
- In production the DeepSeek V4.1 records are decoded on the GPU by the CUDA decoder
  ([STATUS.md](../../STATUS.md): `COLI_WEX_GPU_DIRECT=2`); this run measured no GPU decode speed and no
  GPU codec (nvCOMP, GDeflate).

**Conditions of the speeds.** Other jobs ran on the machine. `bench.sh` waited (`WAIT_IDLE=20`) until the
total CPU load of the machine, logged by `cpuload.ps1` about once a second, averaged below 20% over 10 s:
it waited 12 minutes, then the mean was 12% before each of the three checkpoints, about one of the 8
logical CPUs busy with other work. The log has one sample about every 1.3 s, too coarse for the short
rows. During the one-thread rows that last 9 s or more the total load averaged 20% to 29%, 7 to 17
points above the 12.5% of the benchmark's own thread; in some short rows a sample read up to 100%.
During the 8-thread rows that last 4 s or more it averaged 80% to 100%, the benchmark's own threads
included, so how much the other jobs took from them cannot be read from the log. The other jobs can
have made the rows slower; by how much is not known, more so for the 8-thread rows. The spread of the
11 passes: in the 114 rows whose median is 5 ms or more, the fastest pass was at most 7.9%
below the median; the slowest was up to 39% above it. One run of `bench.sh` (11 passes per row), not
repeated: a second run, started at 13:17, waited 18 minutes for the 10-second mean load to fall below
15%; it stayed between 53% and 100%, and that run was stopped before it measured anything.

Not measured: per-tensor compression (scales and weights apart, which is what WNR3 does); zstd with a
trained dictionary or long-distance matching; byte-shuffle filters (Blosc2) and OpenZL; the NEON and
scalar kernels of the fast decoder (no ARM machine here; `bench.sh` needs only the files of `run.sh`, so it
can run there); any other CPU.

### 4. The Vulkan prototype on the RTX 3070 (HELP-WANTED section 6, byte identity only)

`vulkan.sh` ran `wex4/research/vulkan-decode/run.sh` of commit `1af6c0d` with `MUTATIONS=0`, then its host
program in mode `decode` on the conformance vector that `run.sh` decodes and on four WNR3 containers of
`run.sh` above (32 real records), then the same with a copy of the shader changed by one word
([vulkan.log](vulkan.log)). `vulkan_diag.sh` built a copy of the host program with
[vulkan-diag.patch](vulkan-diag.patch), which prints per matrix which bytes of a differing record differ
([vulkan-diag.log](vulkan-diag.log)). Device line: `NVIDIA GeForce RTX 3070 (type 2, Vulkan 1.4.357)`,
driver 621.01, Windows 10. The prototype's files were not changed.

| Input | WNR3 records | Ceil-form blocks | `wnr3.comp` as committed | Decoded buffer declared `coherent` |
|---|---:|---:|---|---|
| `run.sh` selftest | 12 (5 valid, 7 to refuse) | 534 | FAILED: the 2 records in ceil form differ, 7 refused by both | 5 with the reference's bytes, 7 refused by both |
| `wex4/vectors/valid/tiny2-cz-upgraded.wex4` | 2 | 0 | 2 with the reference's bytes | the same |
| DeepSeek V4.1, `nr3-all.wex4` | 8 | 2,707,895 | 8 differ | 8 with the reference's bytes |
| DeepSeek V4.1, canonical zero, `pack-wnr3-cz.wex4` | 8 | 2,707,895 | 8 differ | 8 with the reference's bytes |
| Qwen3-235B, `nr3-all.wex4` | 8 | 158 | 8 with the reference's bytes | the same |
| MiniMax-M2.1, `nr3-all.wex4` | 8 | 165 | 8 with the reference's bytes | the same |

- **Which bytes differ** (`vulkan-diag.log`): in each DeepSeek V4.1 container 515,991 of the 8,847,360
  scale bytes of the 8 records, and every one of them holds the coded scale, the value before the
  decrement of the ceil-form inverse. 19.1% of the 2,707,895 decrements are missing (18.3% to 19.6% per
  matrix); 0 of the 141,557,760 weight bytes differ. The two containers have the same scales and give the
  same counts. In the selftest, 78 of the 534 decrements are missing.
- **The `coherent` copy** is a diagnostic, not a proposed change: `sed` added the word to the declaration
  of the buffer `Decoded` in a copy of `wnr3.comp` in the work folder, passed through `WNR3_VK_SHADER`.
  Whether it is the right fix, and what it costs in speed, was not studied.
- With the `coherent` copy, `run.sh` passed the selftest, the decodes and the step with a batch limit of
  256 KiB, then stopped with exit 2 at its reject step: `cannot open` for the three invalid vectors. Their
  relative paths, from this machine's deep checkout folder, are longer than the 260 characters of the
  Windows path limit (265 for the first; the valid vector, 227, opened). Step 4 of `vulkan.sh` ran the
  same reject step with paths through a short folder: the three vectors got the verdicts `run.sh` expects,
  `wnr3_vk: ok`.
- The Qwen3-235B and MiniMax-M2.1 records have few ceil-form blocks (158 and 165 in 8 records); that they
  decode right with the committed shader does not show that the shader is right for them.
- Not run: the mutation test (`MUTATIONS=0`), any speed, any other GPU or driver, a whole real container.

## Limits

- 8 experts of each checkpoint, two layers, experts 0 to 3. Ratios of other layers and experts were not
  measured; the Qwen3-235B result above shows that one layer can differ from another.
- The WNR3 models were trained on the records they code (or on the other layer, for the held-out rows),
  on at most 8 experts; production trains on 80 spread over the layers.
- One machine, one compiler, a machine shared with other jobs (section 3). The fast decoder ran with its
  AVX2 kernels only; NEON and the scalar kernel were not run on these records.
- Identity was checked on the CPU by both reference readers and by `cmp` against the source files; the GPU
  check is the Vulkan prototype's own comparison with `wex_nr2_decode`.
- Nothing here ran an engine: no expert of these families was used in a forward pass.

## Raw files

| File | What it is |
|---|---|
| `fetch_experts.py` | the fetch script (the plan of the three checkpoints is in it) |
| `fetch-log.json` | its log: every request (repository, revision, file, byte range, bytes, SHA-256) and every file written |
| `records.py` | `model`: trains a WNM2 model with `build_nr3.py`'s training; `extract`: the slabs, WNR3 records and model of a WEX4 file |
| `run.sh`, `run.log` | the round trips and their output |
| `bench_real.c`, `baselines.py`, `cpuload.ps1`, `bench.sh` | the speed harness (C), the standard-library codecs (Python), the CPU-load logger (Windows) and the driver |
| `bench-1/` | the output of `bench.sh` (`system.txt`, three files per checkpoint, `cpuload.csv`) |
| `vulkan.sh`, `vulkan.log` | the Vulkan runs and their output |
| `vulkan_diag.sh`, `vulkan-diag.patch`, `vulkan-diag.log` | the diagnostic copy of `wnr3_vk.c`, the patch it applies, and its output |

In the logs the work folder is written `<SCRATCH>/w` and the checkout `<Wex-WRN2>`. The models, records
and containers are not here: they are computed from the weights.
