# Help wanted

Things an outside contributor can pick up, roughly from easiest to hardest. Each item says what is
needed, where to start and how to tell it is done. Before writing code, read
[CONTRIBUTING.md](../CONTRIBUTING.md) (privacy rules, how results are stated, sign-off). For anything
bigger than a small fix, open an issue first so two people do not do the same work.

Labels you may see on issues: **good first issue** (a few hours, little context), **help wanted**
(larger, needs a design note first).

Several items of this page got a first answer on 2026-11-14. Where that is so, the item says what
exists now ("Today"), with a link, and what is still open. A thing listed as open was not run or not
measured; it is not known to fail.

## 1. Run the benchmark on your hardware (no code)

- **What:** run [`bench/`](../bench/README.md) and share `results.json` through the
  "Benchmark results" issue template. On Linux also run `bench/wex_io.py` (the read paths for expert
  records, section 7) and attach its `wex_io.json`.
- **Most wanted:** ARM machines (DGX Spark, Apple silicon under Linux, Ampere); AMD and Intel GPUs;
  filesystems other than NTFS (ZFS, btrfs, ext4, XFS); NVMe RAID or mirrors; machines with
  128 GB+ of RAM.
- **Done when:** the issue has the JSON and a note on the run conditions.

## 2. Documentation (good first issue)

- **Spec review:** read [WEX4-SPEC](WEX4-SPEC.md) and [WNR3-SPEC](WNR3-SPEC.md) as if you were
  writing a reader from scratch. Open an issue for every sentence that is ambiguous, every field
  whose range is not stated, and every place where the spec and the code disagree. This is the most
  valuable review there is.
  - **Today:** a first pass was made on 2026-11-14
    ([research/spec-review-20261114](../research/spec-review-20261114/README.md)): 16 clarifications
    went into the two specifications. The same day a reader was written from the pages alone
    (section 4); its [SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md) says which of its questions the
    text now answers.
  - **Still open:** the 4 findings the review left open (no vector has a WNR3 record with a CEIL
    matrix or without canonical zero; WNR3-SPEC has no conformance section; the error classes are
    defined in the code only; whether a decoder may accept any anchor byte), and the notes of
    SPEC-NOTES that the text does not answer yet. Not reviewed: WEX4-SPEC sections 13.1 and 14
    against `wex4/wex4_convert.py` and the numbers of its informative sections, WNR3-SPEC sections 5
    to 11, WNR2-REFERENCE, and the GPU decoder against the text.
- **Glossary:** add missing terms to [GLOSSARY.md](GLOSSARY.md). 32 terms of the specifications and of
  the tests were added on 2026-11-14. Still open: the engine and host words of
  [COLIBRI-INTEGRATION](COLIBRI-INTEGRATION.md) and of the dated reports, which were not gone
  through.
- **Translations:** a short README in your language (the Italian summary at the end of the README
  shows the expected size).

## 3. Tests and fuzzing

- **Fuzzing:**
  - **Today:** five libFuzzer harnesses are in [`wex4/fuzz/`](../wex4/fuzz/README.md): the C WEX4
    reader (`c/wex4.h`), `c/tools/wex4_verify.c`, the WNR3 decoders, and the two engine-side readers
    (`c/expert_store_wex.c`, `c/qwen38_wex.h`), which are built against stand-ins for the engine
    headers. Since 2026-11-15 every stand-in declaration follows colibri's headers and cites its line
    (colibri `dev` at `98327913`); `WEX_FUZZ_COLIBRI=fetch bash wex4/fuzz/run.sh --replay` builds the two
    harnesses against colibri's own headers instead and checks that both give the same declarations,
    and the opt-in `colibri-headers` CI job does that on Linux and macOS. A second pass of each engine
    harness reaches the mirrors, the read splits, the striped read and `WEX_READ_RETRY`, and on Windows,
    with the CPU stand-in of the CUDA helper, the helper's host paths and the load wave (replayed only).
    A custom mutator keeps the WNR3 stream hashes right. CI builds them with clang and runs
    each for 60 seconds on x86-64 Linux, and replays their seeds and mutators with gcc
    (`run.sh --replay`, no coverage guidance) on x86-64 Linux, ARM64 Linux, macOS (Apple silicon)
    and Windows.
  - **What is wanted now** is machine time: run `bash wex4/fuzz/run.sh 14400` (hours, not seconds)
    and report what stops it, with the reproducer it leaves.
  - **Still open:** seeds and a mutator for palette, WNR and WNR2 records (the store decodes them;
    in the one 60 s run that counted them, the store harness served none); the CUDA helper's paths
    fuzzed and not only replayed (they exist on Windows only, where libFuzzer does not run here), and
    with the real helper on a GPU; the paths the second passes cannot reach (a second read that
    succeeds after one that failed, a read that fails: [wex4/fuzz/README.md](../wex4/fuzz/README.md),
    "Limits"); `c/qwen38_wex.h` built inside colibri itself: no public colibri tree includes it, so the
    members it adds to colibri's expert index are still inferred; a fix in `c/qwen38_wex.h` for
    `COLI_WEX_WEIGHTS` with a weight of `inf`, which turns a NaN into an `int` (found on 2026-11-15,
    same "Limits"); a harness for the GPU decoder (`c/wex_cuda.cuh`); a coverage-guided run on ARM64 or
    on macOS (only `--replay` runs there).
- **Sanitizers:**
  - **Today:** CI runs the CPU checks and the fuzz replay under ASan and UBSan on Linux (the
    `linux-sanitizers` job of `.github/workflows/ci.yml`, which holds the exact `WEX_CFLAGS`, the
    sanitizer options and the preload of the ASan runtime). The jobs `linux-tsan` and `linux-msan`
    run the C test executables under ThreadSanitizer (clang, OpenMP on) and MemorySanitizer (clang,
    OpenMP off) through `bash scripts/sanitize_c.sh tsan|msan`, whose header lists what runs:
    `test_wex_format` (msan only), `test_wex4_verify`, `wex4_verify` (8 threads under tsan, one
    under msan), and the five fuzzing harnesses on their seeds, two of them also on a file of
    synthetic WNR3 records. `linux-tsan` also runs the smoke mode of `bench/wex_io.py`, whose
    `pread` paths have their own threads, under ThreadSanitizer.
  - **Still open:** the libraries that the tests load into Python (`wex4_tools.dll` with the fast
    decoder, the WNR3 encoder) under these two sanitizers, which needs the same tests driven from C
    or an instrumented Python; the engine-side readers inside an engine tree (here they run only
    through the fuzzing harnesses, under the sanitizers against the stand-ins; against colibri's
    headers without a sanitizer); any sanitizer on ARM64 or on macOS (the three
    sanitizer jobs are x86-64 Linux); and the GPU code.
- **More test vectors:** hostile files that [wex4/vectors](../wex4/vectors/README.md) does not cover
  yet, each with its `MANIFEST.tsv` row and the expected verdict from both reference readers; the
  `go-reader` job of CI then runs the Go reader on the new rows too. Valid files are wanted as
  well. Two lists of what the 112 rows do not exercise exist: finding 17 of the
  [spec review](../research/spec-review-20261114/README.md) (no WNR3 record with a CEIL matrix or
  without canonical zero) and part D of the Go reader's
  [SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md) (the byte 255 in rANS-coded data, WBP1 records of
  4 or 8 planes or of a size that is not a multiple of the plane count, a WBP1 record with a page
  of zeros too many, tensors of 0 bytes). The files of
  [impl/go-wex4/testdata](../impl/go-wex4/testdata/README.md) cover some of these and are not
  vectors yet. A new vector goes through an issue first.
- **macOS, what is left** (good first issue each):
  - **Today:** the `macos` job of CI (macos-15, Apple silicon) runs `scripts/check_all.sh` and
    `wex4/fuzz/run.sh --replay` with a Homebrew GNU gcc named in `CC`.
  - **Still open, not tried:** Apple clang with `libomp` (Apple clang ships no OpenMP runtime of
    its own); an Intel Mac; the libFuzzer harnesses of `wex4/fuzz/` on macOS; and the set-up lines
    of the README on a Mac of your own (`brew install gcc`, `CC=gcc-15`), which CI does not run as
    written.
- **NEON decoder, what is left** (help wanted):
  - **Today:** kernel 1 of the fast CPU decoder (`wex4/research/fast-cpu-decode/wex_fast_decode.h`)
    has a NEON version. In the `linux-arm` CI job it is byte-identical to the reference decoder and
    3.9 times faster than the scalar kernel on one thread, on synthetic records and on that runner
    only ([results](../wex4/research/fast-cpu-decode/RESULTS.md), section "NEON kernel"). The
    `macos` job builds the same kernel with GNU gcc and passes the comparisons with the reference
    decoder that `scripts/check_all.sh` and the fuzz replay make (not the other steps of
    `linux-arm`); no speed was measured there.
  - **Still open:** its speed on other ARM CPUs (Apple silicon, Ampere, DGX Spark), which needs no
    model (`bash scripts/check_all.sh`, then `bash wex4/research/fast-cpu-decode/synth.sh speed`);
    its speed on real records; a build with Apple clang; the number of interleaved streams that is
    best on each CPU; kernel 2 for SVE, which has gather loads (NEON has none, so kernel 2 is AVX2
    only).

## 4. A second, independent implementation

- **Today, a reader in Go:** [impl/go-wex4](../impl/go-wex4/README.md) is a reader and verifier
  written from the specification pages only, with the WBP1 and the WNR3 decoder. It gives the
  manifest's verdict, reject class and root on all 112 rows of
  [wex4/vectors](../wex4/vectors/README.md) (2026-11-14; the `go-reader` job of CI runs it). Passing
  the 112 rows was not enough: on two kinds of file outside the manifest its first version gave
  another verdict than the reference ([SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md), A6 and A9); it
  now gives the reference's.
- **Still wanted, a reader in another language** (Rust, C++, Zig…) written **from the spec only**
  (WEX4-SPEC, WNR3-SPEC and, for the WNR3 decoder, WNR2-REFERENCE), not by porting `wex4.py` or the
  Go reader. Done when it gives the same verdict as the reference on every file in
  [wex4/vectors](../wex4/vectors/README.md), and the same root for the valid ones. Every
  disagreement is either a bug in the new reader or an ambiguity in the spec, and both are worth an
  issue. Optional: the WNR3 decoder as well; the vectors alone do not prove it (they have no record
  in ceil form), so use the files of [impl/go-wex4/testdata](../impl/go-wex4/testdata/README.md)
  too, and ask the reference for its verdict on every file you are not sure of.
- **Still wanted, a writer from the spec only:** no writer exists outside the reference code. Done
  when the two reference readers and the Go reader accept its files and decode them to the source
  bytes. WEX4-SPEC section 5 says the index encoding is canonical, so a writer that makes the same
  choices as the reference writer should give the same bytes; nobody has checked that this can be
  done from the spec alone (the source tensors of `valid/generic.wex4` are not in the vectors
  folder).
- **Still wanted, the rest of the notes turned into text and vectors:** the spec review of
  2026-11-14 wrote part of the Go reader's notes into the specifications; the section "After the
  spec review" of [SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md) lists which. Open: a sentence of
  the specification for each note that is left, and a conformance vector for each gap of part D
  (none was added). A change of the specification or of the vectors goes through an issue first.

## 5. Converters for more models (help wanted)

- **Today:** [wex4_convert.py](../wex4/wex4_convert.py) has eight naming profiles (`--names`, for `pack`
  and for `upgrade`): `deepseek`, `deepseek-quark`, `qwen38`, `qwen3-moe` (also Kimi K2, GLM-4.7 and 5.1,
  DeepSeek R1 and V3.2), `kimi-k26`, `mixtral` (also MiniMax-M2.1), `minimax-m3` (also Kimi K3) and
  `mistral-large4`. The table, with the checkpoint each profile's names were read in, is in
  [wex4/README.md](../wex4/README.md#naming-profiles). Each has a synthetic fixture
  (`fixtures/make_synth_mxfp4.py --names`) on which pack → verify → unpack and upgrade → verify →
  downgrade round-trip byte for byte.
- **Still open, most wanted:** a run on a real checkpoint. The profiles were tested on the synthetic
  fixtures only, and the names were read in two shards per checkpoint. Pack a real MXFP4 checkpoint of
  one of these families with `pack --experts raw --names PROFILE`, run `wex4.py verify` and `unpack`, and
  report the line `pack` prints (expert records, experts kept as ordinary tensors), the sizes and
  whether every file came back with its SHA-256. If `pack` stops with `incomplete expert` although all
  the shards were given, the profile does not fit that export: report the names.
- **Still open:** which of w1, w2, w3 is the gate, the down and the up matrix for `mixtral`,
  `minimax-m3` and `mistral-large4`, checked against the model code. The slab order is by number; an
  engine needs the answer, the converter does not.
- **Still open:** a profile for a family that is not in the table. It has to be an MXFP4 checkpoint with
  three matrices per expert and one group of tensors per expert, which is what an expert record holds.
  Add its row to `PROFILES` in the converter and to `NAMES` in the fixture script (the test fails when
  one of the two is missing), and name the checkpoint and the shard the names were read in.
- **Does not fit without a format change** (open an issue first): expert records for float or FP8
  experts (Mixtral, Qwen3 MoE and GLM-4.5 as published in BF16, Kimi K2 as published in FP8), and for
  gpt-oss, which holds all the experts of a layer in one tensor and has two matrices per expert. Today
  `pack` stores those as ordinary tensors.
- **Done when:** convert → verify → downgrade round-trips byte for byte on the fixture, and for a real
  checkpoint on the checkpoint.

## 6. Port the WNR3 GPU decode to Vulkan (help wanted, priority)

- **Today:** WNR3 records are decoded on the GPU only through CUDA (`c/tools/wex_cuda_bridge.cu`). On
  Intel (Arc) and AMD GPUs the CPU decoder runs instead, so those cards cannot decode compressed experts
  straight into VRAM.
- **What:** a Vulkan compute version of the WNR3 decode into device memory, following the CUDA kernels
  (one record = many independent rANS streams, then the canonical-zero and scale handling of
  [WNR3-SPEC](WNR3-SPEC.md)). colibri already has a Vulkan backend to hook it into.
- **Already there:** the design note [WNR3-VULKAN](WNR3-VULKAN.md) (how streams map to workgroups, the
  buffers, the model tables, batching, what differs from CUDA) and a prototype in
  [`wex4/research/vulkan-decode/`](../wex4/research/vulkan-decode/README.md): a compute shader and a C99
  host program that build without the CUDA toolkit and agree with the CPU reference, in bytes and in
  verdicts, on synthetic records, on the tiny-model containers and a conformance vector, and on mutated
  copies of them. It ran only on Mesa lavapipe, a Vulkan device on the CPU (the opt-in `linux-vulkan` job
  of `.github/workflows/ci.yml`).
- **Still open, most wanted first:**
  - run `bash wex4/research/vulkan-decode/run.sh <container.wex4>` on a real GPU (Intel, AMD, NVIDIA,
    Apple through MoltenVK) and report the last lines and the device line; a real container with WNR3
    records is the test that counts, and none was decoded yet. The host program reads the file record
    by record and cuts its batches to the device's buffer limit, so the file size is not a limit (a
    5.4 GB test file was read; no record of a real model was). The slow part is the CPU reference,
    which decodes every record too: start with `MUTATIONS=0 WNR3_VK_EVERY=64` (no mutated copies, one
    record in 64), as the folder's README says;
  - a speed number against the CPU path on the same machine. The prototype looks symbols up with the
    reference binary search; the design note, section 5, lists the tables the CUDA kernels use instead;
  - the commit step (copy the decoded regions to the engine's buffers only when the record's error word
    is zero) and the hook into colibri's Vulkan backend (design note, sections 8 and 11).
- **Done when:** its output is byte-identical to the CPU reference on every record of a real container,
  there is a speed number against the CPU path on the same machine, and it builds without the CUDA
  toolkit. Today only the third holds.
- HIP and Metal ports are welcome after it, on the same tests.

## 7. I/O on Linux and macOS (help wanted)

- **Today:** the read paths were tuned on Windows (unbuffered reads, overlapped I/O). On Linux
  `c/expert_store_wex.c` reads one record per `pread` through the page cache, and `c/qwen38_wex.h` one
  per `pread` with `O_DIRECT`. The benchmark exists: [`bench/wex_io.py`](../bench/README.md) writes a
  synthetic WEX4 container and reads its expert records through 20 read paths (page cache or `O_DIRECT`,
  `posix_fadvise` hints, 1 to 8 threads, `io_uring` with records or sub-reads in flight), each checked
  against the index's SHA-256 and timed over several cold passes.
- **Measured so far:** only GitHub Actions runners, shared cloud VMs with a virtual disk
  ([research/linux-io-20261114](../research/linux-io-20261114/README.md): two kinds of VM, three recorded runs of
  5 passes per path). No read path won on both kinds: `POSIX_FADV_RANDOM` and `O_DIRECT` were 19% and
  16% faster than the store's path on the first and 5% and 4% slower on the second; more reads in flight
  gained at most 11% over one read of the same kind on the first and at most 4% on the second (418 to
  481 and 511 to 540 MB/s at every depth). So the store was not changed. The `POSIX_FADV_WILLNEED` rows
  do not show what a prefetch of the next record gives: where it was measured, the kernel had loaded 1.0%
  of the announced record before its read.
- **What:** run `bench/wex_io.py` on real drives (NVMe first; also SATA, ZFS, btrfs, XFS, ARM) and share
  `bench/out/wex_io.json` through the "Benchmark results" issue template. Then, if a path wins on real
  drives: the change in the store (candidates the two runs point at but do not support yet:
  `posix_fadvise(POSIX_FADV_RANDOM)` or `O_DIRECT` in `c/expert_store_wex.c`, and several records in
  flight for a prefetch batch), with an engine A/B. Still unmeasured: a prefetch through the page cache
  that loads the whole record (a read path to add to `bench/wex_io.c`: `readahead(2)` or `WILLNEED` in
  chunks), striping across drives with different speeds, and everything on macOS (the benchmark is
  Linux only; the `macos` CI job does not build it).
- **Done when:** the table of `bench/wex_io.py` from at least one NVMe machine is in the repository, and
  a change of the store comes with its before/after from the same benchmark, stated as CONTRIBUTING
  section 5 asks.

## 8. Compression research

- **Baselines:** compress real expert records with zstd, lz4 and others at several levels and compare
  ratio and decode speed with WNR3 on the same records. Honest negative results are welcome too.
- **Ideas for a next codec:** see [NEXT-FORMAT-SPEC](NEXT-FORMAT-SPEC.md) and the research reports in
  [docs/reports](reports/). Any proposal has to stay lossless (bit-identical weights).

## Roadmap: what is still open on the maintainer's side

So you can see where things are going and avoid duplicating work. Items marked *(open to you)* are
listed above as well.

- **Integration into colibri.** The format and its readers are being upstreamed into
  [colibri](https://github.com/JustVugg/colibri) `dev` as a series of small pull requests: Qwen3.8 MXFP4
  experts first (#2822-#2825 are open), then the WEX4 container, the WNR3 codec and its GPU decode, and
  the I/O switches. Reviewing or testing those PRs on your hardware helps directly.
- **WNR3 decode on Vulkan:** the design note and a prototype checked on a CPU Vulkan device are in; a run
  on a GPU, a speed number, a real container and the engine hook are open *(open to you, section 6)*.
- **ARM and macOS:** the ARM CI jobs (Linux, and macOS on Apple silicon) and a NEON version of kernel 1
  of the fast CPU decoder are in. Its speed on real records and on ARM CPUs other than the Linux CI
  runner, an SVE version of kernel 2, and Apple clang are still open *(open to you, section 3)*.
- **Linux I/O in the engine:** the fast read paths were built and measured on Windows. The Linux
  benchmark is in (`bench/wex_io.py`), run so far only on two kinds of cloud VM, where it gave no reason to
  change the store; runs on real drives and the tuning that follows from them are still open *(open to
  you, section 7)*.
- **Measurements not done yet** (they need the full models, so they stay with the maintainer, see
  [STATUS.md](../STATUS.md)):
  - the SHA-256 of each record at its first read on Qwen3.8;
  - a full ABBA set for it on DeepSeek V4.1;
  - tuning the two-drive split for Qwen3.8;
  - MTP together with MXFP4 experts.
- **The specifications after the review and the Go reader:** the review wrote 16 clarifications into
  WEX4-SPEC and WNR3-SPEC and left 4 findings open; the Go reader of `impl/go-wex4` found no wrong
  verdict in the 112 manifest rows, and a list of sentences that can be read two ways and of cases
  the vectors do not exercise. The review answers part of that list. The rest of it, the 4 open
  findings and the vectors that are missing are not done *(open to you, sections 2 to 4)*.
- **Fuzzing for hours:** the harnesses run for 60 seconds each in CI; no run longer than that is
  recorded in this repository *(open to you, section 3)*.
- **Converter profiles for more model families:** eight naming profiles are in, tested on synthetic
  fixtures; a run on a real checkpoint of the new families is open *(open to you, section 5)*.
- **A first tagged release** of the specifications (WEX4 4.0, WNR3 v1) with the conformance vectors, so
  other implementations can name the version they pass.

## Not in scope here

- **Engine features unrelated to the format** (sampling, chat templates, new architectures): those
  belong in [colibri](https://github.com/JustVugg/colibri).
- **Lossy quantization formats:** WEX stores the checkpoint's own bytes or a lossless coding of them.
- **Model weights:** never commit them; see the README's "Model weights" section.
