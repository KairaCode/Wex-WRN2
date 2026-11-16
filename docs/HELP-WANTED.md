# Help wanted

Things an outside contributor can pick up, roughly from easiest to hardest. Each item says what is
needed, where to start and how to tell it is done. Before writing code, read
[CONTRIBUTING.md](../CONTRIBUTING.md) (privacy rules, how results are stated, sign-off). For anything
bigger than a small fix, open an issue first so two people do not do the same work.

Labels you may see on issues: **good first issue** (a few hours, little context), **help wanted**
(larger, needs a design note first).

Several items of this page got a first answer on 2026-11-14 or 2026-11-15. Where that is so, the
item says what exists now ("Today"), with a link, and what is still open. A thing listed as open was
not run or not measured; it is not known to fail.

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
    text now answers. On 2026-11-15 the 4 findings the review had left open and the notes the text
    did not answer yet got sentences and 24 conformance vectors, after both reference readers were
    given files made for each point (they agreed on all, so no rule changed): WNR3-SPEC has a
    conformance section (section 12) and errata 13 and 14, the reject classes are informative
    (WEX4-SPEC section 17), and the table "After the sentences and vectors of 2026-11-15" of
    SPEC-NOTES says where each note went.
  - **Today, the parts the first pass left out:** a second pass on 2026-11-15
    ([research/spec-review-20261115](../research/spec-review-20261115/README.md)) read WEX4-SPEC
    sections 13.1 and 14 against `wex4/wex4_convert.py`, re-derived the numbers of the informative
    sections that rest on sizes and counts, and read WNR3-SPEC sections 5 to 11, WNR2-REFERENCE and
    the GPU decoder (`c/wex_cuda.cuh`, `c/tools/wex_cuda_bridge.cu`) against the code. 18
    clarifications went into the three pages; an upgrade and a pack written from the clarified
    sections give the converter's bytes.
  - **Still open:** 3 findings of the second pass, listed in that folder (the host batch of helper
    feature 32 writes its destination without a commit step, which WNR3-SPEC section 4.7 does not
    allow for; the converter upgrades a WEX1 file whose header flags are not 0; one extrapolated
    time); the GPU decoder run on a GPU against the text (it was read, and seven of its helper
    functions were run on the CPU only); and the numbers that need the full models.
- **Glossary:** add missing terms to [GLOSSARY.md](GLOSSARY.md). 32 terms of the specifications and of
  the tests were added on 2026-11-14, and 14 engine and host words of
  [COLIBRI-INTEGRATION](COLIBRI-INTEGRATION.md) and of the dated reports on 2026-11-15. Still open:
  words that only one dated report uses; the reports were searched for the words of the guide, not
  read through one by one.
- **Translations:** a short README in your language (the Italian summary at the end of the README
  shows the expected size).

## 3. Tests and fuzzing

- **Fuzzing:**
  - **Today:** five libFuzzer harnesses are in [`wex4/fuzz/`](../wex4/fuzz/README.md): the C WEX4
    reader (`c/wex4.h`), `c/tools/wex4_verify.c`, the WNR3 decoders, and the two engine-side readers
    (`c/expert_store_wex.c`, `c/qwen38_wex.h`), which are built against stand-ins for the engine
    headers. A custom mutator keeps the WNR3 stream hashes right. Since 2026-11-15 the store
    harness has a second run, `fuzz_store_wex-legacy`, on four seed containers with palette, WNR
    and WNR2 records, and a mutator that re-encodes those and scale-tile records so that they
    pass the checks of their codec. In three 60 s runs it served palette, WNR and WNR2 records,
    none of which it served before; the two runs with the mutator gave 127 to 215 different WNR
    and WNR2 records, the one without 3 of each
    ([counts and conditions](../wex4/fuzz/README.md#the-custom-mutators-and-the-dictionary)).
    Those runs were of commit dbc1391, before the merges of 2026-11-15, with six engine seed
    containers instead of the eight of today; the merged code ran only in 60 s Fuzz jobs.
    The Fuzz workflow builds the harnesses with clang and runs each for 60 seconds (the store
    harness twice) on x86-64 Linux, on ARM64 Linux and on macOS (Apple silicon, with the clang of a
    Homebrew LLVM, `llvm@18` on 2026-11-15: Apple clang ships no libFuzzer runtime); on both ARM64
    systems `fuzz_wnr3_record` compares the NEON kernel of the fast decoder with the reference. CI
    replays their seeds and mutators (`run.sh --replay`, no coverage guidance) on x86-64 Linux,
    ARM64 Linux, macOS (Apple silicon, GNU gcc and Apple clang) and Windows.
  - **Today, one run of an hour per harness:** started by hand on 2026-11-15 on commit `1af6c0d` of
    `main`, which had the five harnesses but not yet the second store run, the seeds of the older
    codecs or the ARM64 and macOS jobs: the Fuzz workflow on its `ubuntu-24.04` runner (x86-64), the
    distribution's clang, the custom mutators on, 3,600 s per harness, one run
    ([43672379360](https://github.com/KairaCode/Wex-WRN2/actions/runs/43672379360)). Every harness
    ran its hour without a crash and without a failed oracle, from 640,764 inputs
    (`fuzz_wnr3_record`) to 4,095,358 (`fuzz_wex4_file`); the counts are in
    [wex4/fuzz](../wex4/fuzz/README.md#a-run-of-an-hour-per-harness).
  - **What is wanted now** is more machine time: runs of several hours
    (`bash wex4/fuzz/run.sh 14400`) on the code of today, with the run `fuzz_store_wex-legacy`, and
    on ARM64 and macOS; report what stops them, with the reproducer they leave.
  - **Still open:**
    - the older codecs: WNR records from the repository's own WNR encoder, which needs CUDA (the
      seeds' WNR records come from an encoder written in the harness, 4 classes each, framed by
      `c/tools/repack_v4_wex.py`, and the mutation keeps the count of classes); WNR2 records from
      the CUDA encoder of `archive/nr2/`; a seed container that mixes the older codecs; runs of
      the older codecs longer than 60 s;
    - the store's mirror, striped-read and CUDA-helper paths; `c/qwen38_wex.h` built inside the
      real engine instead of against inferred types (on macOS it opens files through a stand-in for
      the engine's `compat_open_direct`, issue 22); a harness for the GPU decoder (`c/wex_cuda.cuh`);
    - a coverage-guided run on Windows (the MSYS2 clang has no libFuzzer, so only `--replay` runs
      there) and on an Intel Mac; any run longer than 60 s on ARM64 or on macOS, and longer than
      one hour per harness anywhere;
    - in `c/expert_store_wex.c`: the store allocates 24 slabs per layer when it opens, also for a
      layer whose entries are all absent. The slab size is bounded by the table since 2026-11-15
      (WEX1 and WEX2: at most twice the largest 4096-aligned decoded record), the number of layers
      is not, so a small file with many layers and one record still asks for memory out of
      proportion to its size, and the store harness does not open a valid container that needs
      more than 64 MiB of slabs. Allocating a slab at its first use, or weighing layers x slab
      against a budget, changes the engine's store and is not done.
- **Sanitizers:**
  - **Today:** CI runs the CPU checks and the fuzz replay under ASan and UBSan on x86-64 Linux (the
    `linux-sanitizers` job of `.github/workflows/ci.yml`, which holds the exact `WEX_CFLAGS`, the
    sanitizer options and the preload of the ASan runtime), on ARM64 Linux (`linux-arm-sanitizers`,
    gcc, the same flags) and on macOS with Apple clang (`macos-clang`, where the ASan runtime is put
    into the Python interpreter with `DYLD_INSERT_LIBRARIES`). On the two ARM64 systems the NEON
    kernel of the fast decoder runs under both, in the library the tests load into Python, in its
    own harness and in the fuzz replay; no sanitizer reported an error in the code in the green
    runs of the three jobs listed in [CHANGELOG.md](../CHANGELOG.md) (2026-11-15). The jobs
    `linux-tsan` and `linux-msan` run the C test executables under ThreadSanitizer (clang, OpenMP
    on) and MemorySanitizer (clang, OpenMP off) through `bash scripts/sanitize_c.sh tsan|msan`,
    whose header lists what runs: `test_wex_format` (msan only), `test_wex4_verify`, `wex4_verify`
    (8 threads under tsan, one under msan), and the five fuzzing harnesses on their seeds, two of
    them also on a file of synthetic WNR3 records. `linux-tsan` also runs the smoke mode of
    `bench/wex_io.py`, whose `pread` paths have their own threads, under ThreadSanitizer.
  - **Still open:** the libraries that the tests load into Python (`wex4_tools.dll` with the fast
    decoder, the WNR3 encoder) under these two sanitizers, which needs the same tests driven from C
    or an instrumented Python; the engine-side readers inside an engine tree (here they run only
    through the stand-ins of the fuzzing harnesses); ThreadSanitizer and MemorySanitizer on ARM64
    and on macOS (both jobs are x86-64 Linux); the SVE build of the fast decoder under any
    sanitizer (the sanitizer jobs build NEON on ARM64); any sanitizer on Windows; and the GPU code.
- **More test vectors:** hostile files that [wex4/vectors](../wex4/vectors/README.md) does not cover
  yet, each with its `MANIFEST.tsv` row and the expected verdict from both reference readers; the
  `go-reader` job of CI then runs the Go reader on the new rows too. Valid files are wanted as
  well. A new vector goes through an issue first.
  - **Today:** 136 rows since 2026-11-15. The 24 files added that day cover the two lists of what
    the 112 rows did not exercise (finding 17 of the
    [spec review](../research/spec-review-20261114/README.md) and part D of the Go reader's
    [SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md)): the byte 255 in rANS-coded data, WNR3
    records in ceil form, without canonical zero and with other chunk pairs, WBP1 records of 4 and
    8 planes and of sizes that are not a multiple of the plane count, a page of zeros too many,
    tensors of 0 bytes, and one refused file for each check of a WNR3 decoder that no vector had
    (among them a WNR3 record under its minimum size and a WNM2 blob with a table that is not
    increasing). The manifest gives the content digest of every accepted file.
  - **Still open:** the checks of a WNR3 decoder that WNR3-SPEC section 12 lists under "What no
    vector exercises" (an initial state outside `[L, 256L)`, a stream shorter than 4 bytes, a
    stream that ends early or runs out of bytes, each alone); `chunk_log2` other than 10, 11 and
    12 (13 to 15 are in files of the Go writer, which the reference readers accept, not in a
    vector); a WBP1 record with `transform = 1` on a size that is not a multiple of the plane count
    (WEX4-SPEC 7.2 makes it invalid; no vector has one).
- **macOS, what is left** (good first issue each):
  - **Today:** the `macos` job of CI (macos-15, Apple silicon) runs `scripts/check_all.sh` and
    `wex4/fuzz/run.sh --replay` with a Homebrew GNU gcc named in `CC`. The `macos-clang` job runs
    the same with Apple clang (`CC=clang`) and the `libomp` of Homebrew, whose flags go in
    `WEX_OPENMP` (Apple clang refuses `-fopenmp`; the line is in `scripts/build_cpu_check.sh`), and
    then both again under ASan and UBSan. The Fuzz workflow runs the libFuzzer harnesses there
    (section "Fuzzing" above).
  - **Still open, not tried:** an Intel Mac; ThreadSanitizer on macOS; and the set-up lines on a
    Mac of your own: those of the README (`brew install gcc`, `CC=gcc-15`, or `brew install libomp`
    and `WEX_OPENMP` for Apple clang) and the libFuzzer line of
    [wex4/fuzz](../wex4/fuzz/README.md#running-them) (`CC=$(brew --prefix llvm@18)/bin/clang`).
    CI does not run them on a fresh Mac: it takes the GNU gcc and the `llvm@18` of the runner image
    and installs `libomp` into the Homebrew of that image. The step of `libfuzzer-macos` that
    installs `llvm` when the image has no `llvm@N` has never run.
- **NEON decoder, what is left** (help wanted):
  - **Today:** kernel 1 of the fast CPU decoder (`wex4/research/fast-cpu-decode/wex_fast_decode.h`)
    has a NEON version. In the `linux-arm` CI job it is byte-identical to the reference decoder and
    3.9 times faster than the scalar kernel on one thread, on synthetic records and on that runner
    only ([results](../wex4/research/fast-cpu-decode/RESULTS.md), section "NEON kernel"). The
    `macos` job builds the same kernel with GNU gcc and the `macos-clang` job with Apple clang;
    both pass the comparisons with the reference decoder that `scripts/check_all.sh` and the fuzz
    replay make (not the other steps of `linux-arm`); no speed was measured there. The kernel also
    runs under ASan and UBSan on ARM64 Linux and on macOS, and `fuzz_wnr3_record` compares it with
    the reference under libFuzzer on both (60 s per run); no report so far.
  - **Today, SVE:** kernel 2 (the gathers, which NEON lacks) has an SVE version since 2026-11-15. It
    is built only with an SVE flag (`-march=armv8-a+sve`, or `-mcpu=native` on a CPU with SVE), and
    NEON kernel 1 stays the default. The `linux-arm` runner has SVE with 128-bit vectors: there the
    kernel is byte-identical to the reference decoder (also at 256, 512 and 2048 bits under QEMU,
    which checks bytes only; at 2048 bits the test records form no group of 3 vectors, so that code
    never ran, and groups of 2 vectors only in `mutate`), and its fastest variant is 1.23 times
    slower than NEON kernel 1 on one thread, on synthetic records
    ([results](../wex4/research/fast-cpu-decode/RESULTS.md), section "SVE kernel 2"). So it is an
    option, not the default.
  - **Still open:** its speed on other ARM CPUs (Apple silicon, Ampere, DGX Spark), which needs no
    model (`bash scripts/check_all.sh`, then `bash wex4/research/fast-cpu-decode/synth.sh speed`);
    its speed on real records; the number of interleaved streams that is
    best on each CPU; the SVE kernel 2 on a CPU with vectors longer than 128 bits (256 bits on
    Neoverse V1, 512 on A64FX), where it might win: `WEX4_MARCH=-march=armv8-a+sve` before the same
    `synth.sh speed` times both kernels; a choice at run time between a NEON and an SVE build (a
    binary built for SVE needs a CPU with SVE, and nothing checks for it).
- **Older scripts that call `gcc` by name:** `wex4/research/fast-cpu-decode/verify/run.sh`,
  `wex4/research/mxfp4-contexts/run_all.sh`, the two scripts of `wex4/research/wnr4-record/` and
  `integration/build/build_test_wex_planned.sh` do not read `CC` or `WEX_OPENMP`, so they need a
  GNU gcc named `gcc` on `PATH` (three of them also pass x86 flags: `-march=haswell`, `-mavx2`,
  `-march=x86-64-v3`). None of them was converted: each needs inputs that are not
  in this repository (records or the container of the real DeepSeek V4.1 model, its checkpoint,
  numpy, or an engine tree in the maintainer's layout), so a converted script could not be run to
  its end here or in CI. Open: the conversion, by someone who has those inputs and can show the
  last lines of a run.

## 4. A second, independent implementation

- **Today, a reader in Go:** [impl/go-wex4](../impl/go-wex4/README.md) is a reader and verifier
  written from the specification pages only, with the WBP1 and the WNR3 decoder (and, since
  2026-11-15, a writer: below). It gives the manifest's verdict, reject class and root on all 136
  rows of [wex4/vectors](../wex4/vectors/README.md), and since 2026-11-15 the content digest of the
  accepted ones (the `go-reader` job of CI runs it). Passing the 112 rows of 2026-11-14 was not
  enough: on two kinds of file outside the manifest its first version gave another verdict than the
  reference ([SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md), A6 and A9); it now gives the
  reference's, and since 2026-11-15 both kinds are vectors.
- **Still wanted, a reader in another language** (Rust, C++, Zig…) written **from the spec only**
  (WEX4-SPEC, WNR3-SPEC and, for the WNR3 decoder, WNR2-REFERENCE sections 6 and 7, which
  WNR3-SPEC section 12 names), not by porting `wex4.py` or the Go reader. Done when it gives the
  same verdict as the reference on every file in [wex4/vectors](../wex4/vectors/README.md), and the
  same root for the valid ones. Every disagreement is either a bug in the new reader or an
  ambiguity in the spec, and both are worth an issue. Optional: the WNR3 decoder as well; since
  2026-11-15 the vectors have WNR3 records in ceil form, without canonical zero and with other
  chunk pairs, and the manifest's content digest checks the decoded bytes. Ask the reference for
  its verdict on every file you are not sure of.
- **Today, a writer in Go, from the spec only:** the same program writes WEX4 files since
  2026-11-15 (`wex4check write`, from raw tensor files and a text manifest), with RAW, WBP1 and
  WNR3 records, trains the WNM2 model that WNR3 records need (`wex4check train`) and computes the
  content digest of WEX4-SPEC section 5. The two reference readers and the Go reader accept its
  files and decode them to the source bytes: the tests of
  [impl/go-wex4](../impl/go-wex4/README.md) ask all three in the `go-reader` job of CI, on fixed
  files and on 24 random ones in every run (400 in one local run). The canonical encoding of
  section 5 was tested for the first time: 9 files of the reference writers (6 valid vectors and 3
  test files), taken apart and written again by the Go writer with the choices read from each
  file, come out byte for byte, WNR3 records included, and the trainer gives the two WNM2 models of
  those files byte for byte. The digest matched the reference's on its first reading.
- **Still open for the writer:**
  - the sentence of section 5 lists fewer choices than a writer has to fix (version and feature
    words, the expert grid, the blobs, the reserved header area), section 7 does not give the
    converters' `chunk_log2`, and WNR2-REFERENCE section 7 calls the CDF of an empty context
    "near-uniform" where the files show it uniform: notes E1 to E6 and E8 of
    [SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md), each wanting a sentence of the specification;
  - a writer in another language, written the same way; a second one would show whether the notes
    of part E are enough;
  - the same tensors through the Go writer and through `wex4/wex4_convert.py pack`, compared byte
    for byte: it needs a safetensors reader in the Go writer and the converter's grouping rule
    (WEX4-SPEC section 14 gives it in two sentences), and was not tried. One part of it was: on
    `testdata/planes.wex4`, which `pack` wrote, the Go writer's default codec and parameters give
    the file byte for byte (the grouping and the metadata were taken from the file);
  - a real model: the writer was run on files of 843,776 B at most and on synthetic experts of
    13,056 B, the trainer on two training sets of 2 and 48 such experts; no speed was measured.
    The trainer holds one slab at a time; the writer holds one record, two or three times while
    it is encoded and checked.
- **Today, the notes turned into text and vectors:** the spec review of 2026-11-14 wrote part of
  the Go reader's notes into the specifications, and the sentences and vectors of 2026-11-15 the
  rest of parts A to D; [SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md) lists where each went. Left
  out on purpose: a vector for a record of an unknown codec, which a reader opens and cannot
  decode, so it has no verdict in the manifest (`wex4/test_wex4.py` checks both reference readers
  on such a file). The notes of part E, from the writer and the trainer, came after and want their
  sentences (the item above). A change of the specification or of the vectors goes through an
  issue first.

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
  copies of them. It runs on Mesa lavapipe, a Vulkan device on the CPU, in the opt-in `linux-vulkan` job
  of `.github/workflows/ci.yml`.
- **On a GPU (2026-11-15):** one GPU, an NVIDIA GeForce RTX 3070 with driver 621.01 on Windows 10
  ([research/vulkan-gpu-20261115](../research/vulkan-gpu-20261115/README.md)). The first run there found
  scale decrements of the ceil-form inverse lost by the driver; the shader now avoids that, and every check
  of `run.sh` passes there with the validation layer on, also with 1,000 mutated copies per record. On
  16 synthetic records of the size of a DeepSeek V4.1 expert (random bytes, not a model) the decode took
  1.23 ms of device time per record, 3.64 ms with the upload, against 24.9 ms for the fast CPU decoder
  (AVX2) on the 8 threads of the same desktop's Intel Core i7-7700K and 863 ms for the reference decoder on
  one; other jobs were loading that CPU. The table lookups of the design note's section 5 are in the shader
  since that run: 4% to 6% less device time than the binary search on those records, 32% less on the small
  records of the tiny-model container, with the same bytes. Not run: a container of a real model (none was on
  that machine), any other GPU or driver.
- **Still open, most wanted first:**
  - run `bash wex4/research/vulkan-decode/run.sh <container.wex4>` with a real container (WNR3 records of
    a real model) on any GPU, and report the last lines and the device line: that is the test that counts,
    and no record of a real model has been decoded yet. The host program reads the file record by record
    and cuts its batches to the device's buffer limit, so the file size is not a limit (a 5.4 GB test file
    was read). The slow part is the CPU reference, which decodes every record too: start with
    `MUTATIONS=0 WNR3_VK_EVERY=64` (no mutated copies, one record in 64), as the folder's README says.
    `WNR3_VK_DEVICE=<part of the name>` picks the device;
  - a run on other GPUs and drivers (Intel, AMD, Apple through MoltenVK, NVIDIA on Linux): the RTX 3070
    showed that a driver can get this shader wrong, and only `run.sh` tells;
  - the speed on real records: `bash wex4/research/vulkan-decode/bench.sh` measures the synthetic records
    above; a run of `wnr3_vk bench` on a real container, against the CPU path on the same machine, is open;
  - the commit step (copy the decoded regions to the engine's buffers only when the record's error word
    is zero) and the hook into colibri's Vulkan backend (design note, sections 8 and 11);
  - the llvmpipe crash of issue 21: on Mesa 25.2.8 with LLVM 20.1.2 it is reduced to a shader of about 30
    lines and a host of its own (`crash_min.comp`, `crash_min.c`) that crash only with a binding 0 in the
    descriptor set layout; the first form of the shader was valid SPIR-V and gave no message of the
    validation layer, so the crash is in the driver. Open: its cause inside Mesa, a run on a newer Mesa on
    Linux, and a report to Mesa (none was sent). The folder's README, "The llvmpipe crash", has the forms.
- **Done when:** its output is byte-identical to the CPU reference on every record of a real container,
  there is a speed number against the CPU path on the same machine, and it builds without the CUDA
  toolkit. Today the third holds, and the second holds for synthetic records of the size of a real expert
  on one GPU; the first, the speed on real records and the engine hook are open.
- HIP and Metal ports are welcome after it, on the same tests.

## 7. I/O on Linux and macOS (help wanted)

- **Today:** the read paths were tuned on Windows (unbuffered reads, overlapped I/O). On Linux
  `c/expert_store_wex.c` reads one record per `pread` through the page cache, and `c/qwen38_wex.h` one
  per `pread` with `O_DIRECT`. The benchmark exists: [`bench/wex_io.py`](../bench/README.md) writes a
  synthetic WEX4 container and reads its expert records through 27 read paths (page cache or `O_DIRECT`,
  `posix_fadvise` hints, three ways to prefetch the next record, 1 to 8 threads, `io_uring` with records
  or sub-reads in flight, and four rows with 20 ms of work after each record), each checked against the
  index's SHA-256 and timed over several cold passes.
- **Measured so far:** only GitHub Actions runners, shared cloud VMs with a virtual disk.
  [research/linux-io-20261114](../research/linux-io-20261114/README.md) (two kinds of VM, three recorded
  runs of 5 passes per path): no read path won on both kinds. `POSIX_FADV_RANDOM` and `O_DIRECT` were 19%
  and 16% faster than the store's path on the first and 5% and 4% slower on the second; more reads in
  flight gained at most 11% over one read of the same kind on the first and at most 4% on the second
  (418 to 481 and 511 to 540 MB/s at every depth).
  [research/linux-prefetch-20261115](../research/linux-prefetch-20261115/README.md) (six kinds of VM,
  eleven runs): one `POSIX_FADV_WILLNEED` or one `readahead(2)` on a whole record loaded only one request
  of the device (128 KiB to 2 MiB of 13.4 MB). With no work between records no prefetch beat the store's
  path on every kind. With 20 ms of work after each record (a spin, not a decode), `WILLNEED` in chunks
  of the device's readahead was the fastest prefetch in every run that had those rows: 1.24x to 2.32x
  the same work without prefetch, on five kinds; not run on the sixth. So the store was not changed.
- **What:** run `bench/wex_io.py` on real drives (NVMe first; also SATA, ZFS, btrfs, XFS, ARM) and share
  `bench/out/wex_io.json` through the "Benchmark results" issue template. Then, if a path wins on real
  drives: the change in the store (candidates the runs point at but do not support yet:
  `posix_fadvise(POSIX_FADV_RANDOM)` or `O_DIRECT` in `c/expert_store_wex.c`, several records in flight
  for a prefetch batch, and `WILLNEED` in chunks on the record the engine will need next while it
  decodes), with an engine A/B. A page-cache prefetch would be new engine code: the store's prefetch call
  reads records into its own slots before it returns. Still unmeasured: the chunked hints from a thread
  of their own (on one kind of VM the calls held the reading thread for about 90% of the pass), other
  chunk sizes, more than one record ahead for the new paths, the rows with work on that kind of VM, a
  real decode between the reads instead of the spin, striping across drives with different speeds, and
  everything on macOS (the benchmark is Linux only; the `macos` CI job does not build it).
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
- **WNR3 decode on Vulkan:** the design note and a prototype are in, checked on a CPU Vulkan device and on
  one GPU (an RTX 3070), with a speed number on synthetic records there; a real container, other GPUs and
  drivers, and the engine hook are open *(open to you, section 6)*.
- **ARM and macOS:** the ARM CI jobs (Linux, and macOS on Apple silicon with GNU gcc and with Apple
  clang), ASan and UBSan and the libFuzzer harnesses on both, a NEON version of kernel 1 of the fast
  CPU decoder and an SVE version of kernel 2 are in; the SVE kernel is not the default, because on
  the CI runner's 128-bit SVE it is slower than NEON kernel 1. Their speed on real records and on ARM
  CPUs other than the Linux CI runner, and the SVE kernel on longer vectors, are still open *(open
  to you, section 3)*.
- **Linux I/O in the engine:** the fast read paths were built and measured on Windows. The Linux
  benchmark is in (`bench/wex_io.py`), run so far only on six kinds of cloud VM, where it gave no reason to
  change the store. A prefetch of the next record through the page cache gained there only with work
  between the reads (`WILLNEED` in chunks, 1.24x to 2.32x with a 20 ms spin standing for a decode), which
  only an engine A/B can confirm; runs on real drives and the tuning that follows from them are still open
  *(open to you, section 7)*.
- **Measurements not done yet** (they need the full models, so they stay with the maintainer, see
  [STATUS.md](../STATUS.md)):
  - the SHA-256 of each record at its first read on Qwen3.8;
  - a full ABBA set for it on DeepSeek V4.1;
  - tuning the two-drive split for Qwen3.8;
  - MTP together with MXFP4 experts.
- **The specifications after the review and the Go reader:** the review wrote 16 clarifications into
  WEX4-SPEC and WNR3-SPEC and left 4 findings open; the Go reader of `impl/go-wex4` found no wrong
  verdict in the 112 manifest rows of that day, and a list of sentences that can be read two ways
  and of cases the vectors did not exercise. On 2026-11-15 the 4 findings and the rest of that list
  got sentences and 24 vectors, with no rule changed. The writer and the trainer of 2026-11-15
  added eight notes (part E), seven of which want a sentence of the specification. A second review
  pass of 2026-11-15 read the sections the first had left out and wrote 18 clarifications. Open:
  the seven notes, the 3 findings of the second pass, and the WNR3 checks no vector exercises alone
  *(open to you, sections 2 to 4)*.
- **Fuzzing for hours:** the harnesses run for 60 seconds each in CI; one run of an hour per harness,
  on x86-64 Linux and on the code of 2026-11-15 before the merges of that day, is recorded, and it
  stopped on nothing. Longer runs, runs on the code of today and on ARM64 and macOS are open *(open
  to you, section 3)*.
- **Converter profiles for more model families:** eight naming profiles are in, tested on synthetic
  fixtures; a run on a real checkpoint of the new families is open *(open to you, section 5)*.
- **A first tagged release** of the specifications (WEX4 4.0, WNR3 v1) with the conformance vectors, so
  other implementations can name the version they pass.

## Not in scope here

- **Engine features unrelated to the format** (sampling, chat templates, new architectures): those
  belong in [colibri](https://github.com/JustVugg/colibri).
- **Lossy quantization formats:** WEX stores the checkpoint's own bytes or a lossless coding of them.
- **Model weights:** never commit them; see the README's "Model weights" section.
