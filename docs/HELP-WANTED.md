# Help wanted

Things an outside contributor can pick up, roughly from easiest to hardest. Each item says what is
needed, where to start and how to tell it is done. Before writing code, read
[CONTRIBUTING.md](../CONTRIBUTING.md) (privacy rules, how results are stated, sign-off). For anything
bigger than a small fix, open an issue first so two people do not do the same work.

Labels you may see on issues: **good first issue** (a few hours, little context), **help wanted**
(larger, needs a design note first).

Several items of this page got a first answer from 2026-11-14 on. Where that is so, the item says what
exists now ("Today"), with a link, and what is still open. A thing listed as open was not run or not
measured; it is not known to fail.

## 1. Run the benchmark on your hardware (no code)

- **What:** run [`bench/`](../bench/README.md) (Linux, or Windows in the MSYS2 UCRT64 shell) and share
  `results.json` through the "Benchmark results" issue template. On Linux or macOS also run `bench/wex_io.py`
  (the read paths for expert records, section 7) and attach its `wex_io.json`.
- **Most wanted:** ARM machines (DGX Spark, Apple silicon under Linux, Ampere); AMD and Intel GPUs;
  filesystems other than NTFS (ZFS, btrfs, ext4, XFS); NVMe RAID or mirrors; machines with
  128 GB+ of RAM.
- **Today:** `bench/bench.py` runs on Windows too (2026-11-15; before, its `disk` section stopped
  there and `wex_check` started WSL), the CI job `windows` runs it with `--quick`, and every message
  in `results.json` has paths, user name and host name replaced by placeholders. One Windows machine
  ran it: Intel Core i7-7700K, 32 GB DDR4, NVIDIA RTX 3070, Windows 10 22H2, NTFS on one NVMe drive;
  six runs of the default command in two sets while other jobs kept its CPU busy (58% to 100% total
  CPU, median, in the 30 s before each run and in the `disk` section, which reads with one thread).
  Its results go to a "Benchmark results" issue, not into this repository.
- **Still open:** everything under "Most wanted"; `h2d` on Windows (it needs the CUDA toolkit, which
  that machine does not have; `nvcc` there also needs `cl.exe` of Visual Studio on `PATH`, which is
  installed there but was not tried with `nvcc`); a run of that machine with nothing
  else running, on its SATA drives, and with `--file` on a model file (a cold read of real weights);
  `bench.py` on macOS (not tried).
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
    time); the six questions of part E5 of the Rust reader's
    [SPEC-NOTES.md](../impl/rust-wex4/SPEC-NOTES.md) (2026-11-15); and the numbers that need the
    full models.
  - **The GPU decoder against the text:** reviewed on 2026-11-15
    ([research/cuda-review-20261115](../research/cuda-review-20261115/README.md)): `c/wex_cuda.cuh` and
    `c/tools/wex_cuda_bridge.cu` make every check of WNR3-SPEC that the CPU reference makes, with
    the same verdict; no difference of verdict or of decoded bytes was found, by reading, by the
    harness of section 3 on the CPU, or by its kernels on one GPU. Six differences of another kind
    are listed there with their lines: the status 1 for a model that is not registered, a stream
    decoded past its end before it is refused, the canonical-zero check per block, error bits that
    do not always name the cause, `wex_cuda_decode_batch_to_host`, which can write its destination
    for a record it refuses where section 4.7 says "destination untouched", and 3 bytes that the
    scale kernel may read while they are still being copied (never used by a stream that decodes
    right). **Still open:** a sentence in WNR3-SPEC 4.7 for `wex_cuda_decode_batch_to_host`; the
    WNR2 path of the same kernels and the WNR1 and palette kernels, which were not reviewed.
- **Glossary:** add missing terms to [GLOSSARY.md](GLOSSARY.md). 32 terms of the specifications and of
  the tests were added on 2026-11-14, and 14 engine and host words of
  [COLIBRI-INTEGRATION](COLIBRI-INTEGRATION.md) and of the dated reports on 2026-11-15. Still open:
  words that only one dated report uses; the reports were searched for the words of the guide, not
  read through one by one.
- **Translations:** a short README in your language (the Italian summary at the end of the README
  shows the expected size).
  - **Today:** short READMEs in [Spanish](i18n/README.es.md), [French](i18n/README.fr.md),
    [German](i18n/README.de.md) and [Simplified Chinese](i18n/README.zh-CN.md) are in `docs/i18n/`
    (2026-11-15), next to the Italian summary. They were not written by native speakers, and each
    says so at its top: native speakers are welcome to correct them, through an issue or a pull
    request.
  - **Still open:** a reading of each of the four by a native speaker; other languages, one file
    `docs/i18n/README.<lang>.md` each, linked from the top of the README and listed in
    [docs/README.md](README.md). A translation follows the English README of the date at its top;
    the English text is the reference ([CONTRIBUTING.md](../CONTRIBUTING.md), section 6).

## 3. Tests and fuzzing

- **Fuzzing:**
  - **Today:** six libFuzzer harnesses are in [`wex4/fuzz/`](../wex4/fuzz/README.md): the C WEX4
    reader (`c/wex4.h`), `c/tools/wex4_verify.c`, the WNR3 decoders, the two engine-side readers
    (`c/expert_store_wex.c`, `c/qwen38_wex.h`), which are built against stand-ins for the engine
    headers, and since 2026-11-15 the CUDA WNR3 decoder (`c/wex_cuda.cuh` and its bridge) built as
    C++ for the CPU through an emulation of the CUDA constructs it uses, against the CPU reference.
    That one checks the decoding logic and the order of the bridge's calls, not a GPU; a negative
    control (`wex4/fuzz/cuda_negative.py`, 23 broken copies of the CUDA source) shows that it fails
    when a check, a wait or a barrier is missing. With `run.sh --gpu` the same harness also runs
    the kernels on an NVIDIA GPU, from PTX that clang writes without the CUDA toolkit; that ran by
    hand on one GPU (an RTX 3070) on 2026-11-15, with no failed oracle. Since 2026-11-15 every stand-in declaration that colibri has follows its headers and
    cites its line (colibri `dev` at `98327913`); the members that no public colibri tree has stay
    inferred. `WEX_FUZZ_COLIBRI=fetch bash wex4/fuzz/run.sh --replay` builds the two harnesses against
    colibri's own headers instead (for `fuzz_q38_wex`, colibri's `compat.h` and `st.h`; its model types
    stay those of `engine/q38_engine.h`, checked against `qwen38_core.h` by reading, not by
    `engine_layout.c`) and checks that both give the same declarations, and the opt-in
    `colibri-headers` CI job does that on Linux and macOS. A second pass of each engine
    harness reaches the mirrors, the read splits, the striped read and `WEX_READ_RETRY`, and on Windows,
    with the CPU stand-in of the CUDA helper, the helper's host paths and the load wave (replayed only).
    A third pass fails reads of the engine file on purpose (opens, prefetches, lookups and loads, the
    second read of `WEX_READ_RETRY` that succeeds) and tries the root pin of `COLI_WEX_ROOT`.
    A custom mutator keeps the WNR3 stream hashes right. Since 2026-11-15 the store
    harness has a second run, `fuzz_store_wex-legacy`, on four seed containers with palette, WNR
    and WNR2 records, and a mutator that re-encodes those and scale-tile records so that they
    pass the checks of their codec. In three 60 s runs it served palette, WNR and WNR2 records,
    none of which it served before; the two runs with the mutator gave 127 to 215 different WNR
    and WNR2 records, the one without 3 of each
    ([counts and conditions](../wex4/fuzz/README.md#the-custom-mutators-and-the-dictionary)).
    Those runs were of commit dbc1391, before the merges of 2026-11-15, with six engine seed
    containers instead of the nine of today; the merged code ran only in 60 s Fuzz jobs.
    The Fuzz workflow builds the harnesses with clang and runs each for 60 seconds (the store
    harness twice; for the CUDA harness 421 and 256 inputs in the two x86-64 runs of 2026-11-15) on x86-64 Linux, on ARM64 Linux and on macOS (Apple silicon, with the clang of a
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
    on ARM64 and macOS; report what stops them, with the reproducer they leave; and
    `bash wex4/fuzz/run.sh --gpu 100` on other NVIDIA GPUs (Turing or Pascal for the code built
    below `sm_80`, Ada, Blackwell).
  - **Still open:**
    - the older codecs: WNR records from the repository's own WNR encoder, which needs CUDA (the
      seeds' WNR records come from an encoder written in the harness, 4 classes each, framed by
      `c/tools/repack_v4_wex.py`, and the mutation keeps the count of classes); WNR2 records from
      the CUDA encoder of `archive/nr2/`; a seed container that mixes the older codecs; runs of
      the older codecs longer than 60 s;
    - the CUDA helper's paths fuzzed and not only replayed (they exist on Windows only, where
      libFuzzer does not run here), and with the real helper on a GPU; reads that come up short or
      return other bytes than the file holds (the third pass fails a read outright), and the attach
      of `c/qwen38_wex.h` stopped by a failed read, whose refusing functions the harness calls
      itself ([wex4/fuzz/README.md](../wex4/fuzz/README.md), "Limits"); `c/qwen38_wex.h` built
      inside colibri itself: no public colibri tree includes it, so the members it adds to
      colibri's expert index are still inferred (on macOS it opens files through a stand-in for
      the engine's `compat_open_direct`, issue 22); a fix in `c/qwen38_wex.h` for
      `COLI_WEX_WEIGHTS` with a weight of `inf`, which turns a NaN into an `int` (found on
      2026-11-15, same "Limits"); the GPU decoder as nvcc builds it (the production
      helper), with the bridge's own code and the CUDA runtime, under `compute-sanitizer` and under
      coverage guidance (the GPU mode replays with the mutator, runs clang's PTX and its own copy
      of the bridge's sequence); its WNR2, WNR1 and palette paths, which no seed reaches;
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
    (8 threads under tsan, one under msan), and the six fuzzing harnesses on their seeds, two of
    them also on a file of synthetic WNR3 records. Since 2026-11-15 the same script also runs the libraries that the tests
    load into Python (`wex4_tools.dll` with the fast and with the reference WNR3 decoder, and
    `wex_nr3_tools.dll`, the WNR3 encoder) under both sanitizers: driven from C by
    `scripts/sanitize_libs.c`, which calls every function they export with the inputs of the Python
    tests, and under ThreadSanitizer also by the WEX4 tests themselves, with the runtime preloaded
    into Python. Neither sanitizer reported anything there. `linux-tsan` also runs the smoke mode of
    `bench/wex_io.py`, whose `pread` paths have their own threads, under ThreadSanitizer.
  - **Still open:** the Python tests under MemorySanitizer, which needs a Python (and its `libffi`)
    built with it: the C driver covers the functions of the libraries, not every path by which the
    tests reach them; the WEX4 tests under ThreadSanitizer at their default fuzz count (CI runs
    `--fuzz 100`; the default took 256 to 298 s in each of three runs on a GitHub Actions runner,
    without a report); the engine-side readers inside an engine tree (here they run only
    through the fuzzing harnesses, under the sanitizers against the stand-ins; against colibri's
    headers without a sanitizer); ThreadSanitizer and MemorySanitizer on ARM64
    and on macOS (both jobs are x86-64 Linux); the SVE build of the fast decoder under any
    sanitizer (the sanitizer jobs build NEON on ARM64); any sanitizer on Windows; and the GPU code on a GPU under `compute-sanitizer`, which needs the
    CUDA toolkit (its CUDA source, built for the CPU, runs under the four sanitizers in the fuzz
    replay, and its kernels ran on one GPU without a sanitizer; since 2026-11-16 the helper as nvcc
    builds it has run its CPU/GPU differential on one GPU too, without a sanitizer:
    [research/cuda-real-20261116](../research/cuda-real-20261116/README.md)).
- **More test vectors:** hostile files that [wex4/vectors](../wex4/vectors/README.md) does not cover
  yet, each with its `MANIFEST.tsv` row and the expected verdict from both reference readers; the
  `go-reader` and `rust-reader` jobs of CI then run the Go and the Rust reader on the new rows too.
  Valid files are wanted as well. A new vector goes through an issue first.
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
    (WEX4-SPEC 7.2 makes it invalid; no vector has one); and the gaps of the index that part C of
    the Rust reader's [SPEC-NOTES.md](../impl/rust-wex4/SPEC-NOTES.md) lists (no empty index, no
    blob of kind 18, no file above 4 GiB).
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
    the reference under libFuzzer on both (60 s per run); no report so far. On 2026-11-15 the `arm-speed` jobs (only in a CI run started by hand) built it
    with each number of interleaved streams (`-DWEX_FAST_LANES` = 1, 2, 3, 4, 6 or 8), compared
    each build with the reference decoder (byte-identical every time) and timed it: 18 calls of
    `synth.sh speed` per compiler on 6 VMs each of ubuntu-24.04-arm and of macos-15 (an Apple M1
    under virtualization), there with Homebrew GCC 15.3 and with Apple clang 17 and Homebrew's
    `libomp`. On the M1 VMs the default configuration was 4.8 (GCC) and 5.0 (Apple clang) times
    faster than the scalar kernel. The fastest number of streams was 6 on the Neoverse-N2 (0.8%
    under 4) and 8 on the M1 VMs (11% and 16% under 4 by the medians, the ranges of the calls
    overlapping); 4 is fastest on the two x86 machines, so the default stays 4 (RESULTS, "Streams
    per CPU, Apple silicon, Apple clang").
  - **Today, SVE:** kernel 2 (the gathers, which NEON lacks) has an SVE version since 2026-11-15. It
    is built only with an SVE flag (`-march=armv8-a+sve`, or `-mcpu=native` on a CPU with SVE), and
    NEON kernel 1 stays the default. The `linux-arm` runner has SVE with 128-bit vectors: there the
    kernel is byte-identical to the reference decoder (also at 256, 512 and 2048 bits under QEMU,
    which checks bytes only; at 2048 bits the test records form no group of 3 vectors, so that code
    never ran, and groups of 2 vectors only in `mutate`), and its fastest variant is 1.23 times
    slower than NEON kernel 1 on one thread, on synthetic records
    ([results](../wex4/research/fast-cpu-decode/RESULTS.md), section "SVE kernel 2"). So it is an
    option, not the default.
  - **Still open:** its speed on other ARM CPUs (Apple silicon outside a VM, M2 and later, Ampere,
    DGX Spark), which needs no model (`bash scripts/check_all.sh`, then
    `bash wex4/research/fast-cpu-decode/synth.sh speed` three times or more and `speed_runs.py` on
    the outputs); its speed on real records, and with them whether 6 or 8 streams beat 4 on an ARM
    CPU (no real record is in CI); the SVE kernel 2 on a CPU with vectors longer than 128 bits (256 bits on
    Neoverse V1, 512 on A64FX), where it might win: `WEX4_MARCH=-march=armv8-a+sve` before the same
    `synth.sh speed` times both kernels; a choice at run time between a NEON and an SVE build (a
    binary built for SVE needs a CPU with SVE, and nothing checks for it).
- **The older research scripts and the engine test, with another compiler:**
  `wex4/research/fast-cpu-decode/verify/run.sh`, `wex4/research/mxfp4-contexts/run_all.sh`, the two
  scripts of `wex4/research/wnr4-record/` and `integration/build/build_test_wex_planned.sh`. Today
  (2026-11-16): the five scripts read `CC` and `WEX_OPENMP` as `scripts/build_cpu_check.sh` does and pass
  their x86 flags on x86-64 only (`WEX4_MARCH` overrides); `verify/run.sh` ran to its end on 8 real
  WNR3-CZ records of DeepSeek V4.1 Flash with MSYS2 gcc 16.2 and clang 23.1.5 on Windows 10 and with gcc
  15.2 under WSL2 (0 mismatches); `build_test_wex_planned.sh` built with gcc on Windows and its test
  passed on the synthetic fixture and on 2 real experts. The same day `run_all.sh`, `wnr4-record/run.sh`
  and `wnr4-record/verify/run_verify.sh` ran to their end with gcc 16.2 on Windows 10 on the published 40
  x 192 WNR3-CZ container of DeepSeek V4.1 Flash, read through the WEX3 header page that its upgrade to
  WEX4 kept; every figure of their result files came out as committed except the times (the folders'
  `RESULTS.md`). Still open: any of the five on ARM; the two wnr4-record scripts with a clang that has
  an OpenMP runtime; the engine test outside Windows; and the first part of `verify/st_check.py`
  against the REAP-192 checkpoint the container was made from (the only shards on the machine were of
  the unpruned checkpoint, whose experts are not those of the container).

## 4. A second, independent implementation

- **Today, a reader in Go:** [impl/go-wex4](../impl/go-wex4/README.md) is a reader and verifier
  written from the specification pages only, with the WBP1 and the WNR3 decoder (and, since
  2026-11-15, a writer: below). It gives the manifest's verdict, reject class and root on all 136
  rows of [wex4/vectors](../wex4/vectors/README.md), and since 2026-11-15 the content digest of the
  accepted ones (the `go-reader` job of CI runs it). Passing the 112 rows of 2026-11-14 was not
  enough: on two kinds of file outside the manifest its first version gave another verdict than the
  reference ([SPEC-NOTES.md](../impl/go-wex4/SPEC-NOTES.md), A6 and A9); it now gives the
  reference's, and since 2026-11-15 both kinds are vectors.
- **Today, a reader in Rust:** [impl/rust-wex4](../impl/rust-wex4/README.md) is a second reader and
  verifier written from the pages only (WEX4-SPEC, WNR3-SPEC, WNR2-REFERENCE, the glossary and the
  vectors README), with its own SHA-256 and the WBP1 and WNR3 decoders; the reference code and the Go
  reader's code and notes were opened only after it passed the 112 rows of 2026-11-14. Since the merges
  (2026-11-16) it gives the manifest's verdict, reject class, root and content digest on all 136 rows
  (one class was moved for a vector of 2026-11-15: part F of its notes), and it accepts three files of
  [impl/go-wex4/testdata](../impl/go-wex4/testdata/README.md) (one with WNR3 records in ceil form)
  with the reference's content digest and refuses the fourth (the `rust-reader` job of CI runs it and
  a differential run). On 35 hand-made files and 20,596 mutated ones (one run on
  Windows, seed 20261115; the CI job runs 6,866 mutated files) it gave the C reader's verdict on
  every file, and so does its first version, the one that passed the 112 rows.
  The reject classes differ from the C reader's on 3 hand-made and 15 mutated files, and the C reader
  gives two classes to one sentence of 5.3 ([SPEC-NOTES.md](../impl/rust-wex4/SPEC-NOTES.md),
  part E).
- **Still open after the two readers:** a reader in a third language (C++, Zig…) is welcome on the
  same terms: written **from the spec only** (WEX4-SPEC, WNR3-SPEC and, for the WNR3 decoder,
  WNR2-REFERENCE sections 6 and 7, which WNR3-SPEC section 12 names), not by porting `wex4.py` or
  another reader; every disagreement is a bug or an ambiguity, both worth an issue. More useful
  first: the six questions of part E5 of the Rust reader's notes (the reject classes, the order of
  the overflow check of a shape, the bound on `16 * cols`, the empty index at 4096, what a verifier
  reports for an unknown codec, the anchor byte 255), a run of the Rust reader on ARM, macOS or a
  big-endian host (it ran on x86-64 only), and a coverage-guided fuzzer for it (it has none).
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
  sentences (the item above), and so do the questions of part E5 of the Rust reader's notes; the
  gaps of part C of its notes want a vector each. A change of the specification or of the vectors
  goes through an issue first.

## 5. Converters for more models (help wanted)

- **Today:** [wex4_convert.py](../wex4/wex4_convert.py) has sixteen naming profiles (`--names`, for
  `pack` and for `upgrade`): `deepseek`, `deepseek-quark`, `qwen38`, `qwen3-moe` (also Kimi K2, GLM-4.7
  and 5.1, DeepSeek R1 and V3.2, K2 Horizon, K-EXAONE, Hydrogen 1.0 and Beam 1.0), `kimi-k26`, `mixtral`
  (also MiniMax-M2.1 and M2.7, and an MXFP4 Mixtral 8x7B), `minimax-m3` (also Kimi K3), `mistral-large4`,
  `step3p5` (Step 3.5 Flash), and seven profiles for exports in the compressed-tensors format, where the
  weight is `weight_packed`: `deepseek-ct`, `qwen38-ct`, `qwen3-moe-ct` (also GLM-5.3, Hy3, Ling 3.0
  flash, Laguna S 2.1, LongCat-Flash-Lite and Kolibri-1), `kimi-k26-ct`, `mixtral-ct`, `minimax-m3-ct`
  and `step3p7-ct` (Step 3.7 Flash). The eight that end the list were added on 2026-11-15. The table,
  with the checkpoint and the shard each profile's names were read in, is in
  [wex4/README.md](../wex4/README.md#naming-profiles). Each has a synthetic fixture
  (`fixtures/make_synth_mxfp4.py --names`) on which pack → verify → unpack and upgrade → verify →
  downgrade round-trip byte for byte. Under a `-ct` profile, an expert whose weight is a plain `.weight`
  that is not U8 or I8 (an MXFP8 or BF16 expert of a mixed compressed-tensors export) stays ordinary
  tensors; this is tested on a synthetic fixture only.
- **Today, the matrix roles:** w1 is the gate, w2 the down and w3 the up matrix in `mixtral`,
  `minimax-m3`, `mistral-large4`, the DeepSeek profiles and their `-ct` forms. This was read on
  2026-11-15 in the model code of Hugging Face transformers (commit 3a134bc3) and of the checkpoints'
  own repositories (MiniMax-M2.1, Kimi K3); file and lines are in
  [wex4/README.md](../wex4/README.md#matrix-roles). So in every profile the slab order is gate, down,
  up, as WNR3-SPEC section 1 names the matrices. Not checked: the weights (none was read), and the
  model code of MiniMax-M3.1 Flash and of DeepSeek V4.1 (the roles given for them are those of
  MiniMax-M3 and DeepSeek V4, which use the same names).
- **Today, on real tensors (2026-11-15):** 8 routed experts (2 layers x 4) of each of three published
  checkpoints, fetched with range requests: DeepSeek V4.1 Flash (`deepseek`), Qwen3-235B
  (`qwen3-moe`) and MiniMax-M2.1 (`mixtral`)
  ([research/real-experts-20261115](../research/real-experts-20261115/README.md)). For each,
  `pack --experts raw` and `--experts nr3` printed 8 expert records and 0 experts kept as ordinary
  tensors, both readers verified the files, and `unpack` gave both source files back with their
  SHA-256. The DeepSeek experts also went WEX1 → WEX3 (WNR3) → `upgrade --names deepseek` → verify →
  `downgrade`, back to the same bytes. This is a partial checkpoint (24 experts, no whole shard), not a
  run on a real checkpoint, and it was run with the eight profiles of before the merges of that day.
- **Today, on a whole real checkpoint (2026-11-16):** `amd/Qwen3.5-35B-A3B-MXFP4` (commit `2e19c657`,
  14 shards, 24,600,620,848 B) with `pack --experts raw --names qwen38`: 10,240 expert records (40 x 256),
  0 experts kept as ordinary tensors; `wex4.py verify` OK (file 22,070,321,152 B, ratio 0.8966: RAW for
  the experts, WBP1 for the 1,303 other records); `unpack` gave all 14 files back with their SHA-256. One
  run on an Intel Core i7-7700K with the files on a SATA SSD and two downloads running, so the times (pack
  607 s, verify 257 s, unpack 450 s) are upper bounds
  ([research/converter-qwen35-20261116](../research/converter-qwen35-20261116/README.md)).
- **Still open, most wanted:** runs on whole real checkpoints of the other profiles. Fifteen of the
  sixteen have met no whole checkpoint, twelve of them only the synthetic fixtures, and their names
  were read in one or two shards per checkpoint; a row in `PLAN` of
  `research/real-experts-20261115/fetch_experts.py` fetches a few experts of another checkpoint. Pack a
  real MXFP4 checkpoint of one of these families with `pack --experts raw --names PROFILE`, run
  `wex4.py verify` and `unpack`, and report the line `pack` prints (expert records, experts kept as
  ordinary tensors), the sizes and whether every file came back with its SHA-256. If `pack` stops with
  `incomplete expert` although all the shards were given, the profile does not fit that export: report
  the names. A WNR3 pack (`--experts nr3`) of a whole checkpoint is open too.
- **Still open:** a profile for a family or an export that is not in the table. It has to be an MXFP4
  checkpoint with three matrices per expert and one group of tensors per expert, which is what an expert
  record holds. Add its row to `PROFILES` in the converter and to `NAMES` in the fixture script (the
  test fails when one of the two is missing), and name the checkpoint and the shard the names were read
  in. The search of 2026-11-15 ("Limits" in [wex4/README.md](../wex4/README.md#naming-profiles)) found
  only checkpoints whose name contains MXFP4, did not open the GGUF and MLX exports, and could not read
  `amd/Qwen3-VL-235B-A22B-Instruct-MXFP4`, which is gated (its files need an accepted agreement).
- **Does not fit without a format change** (open an issue first): expert records for float or FP8
  experts (Mixtral, Qwen3 MoE and GLM-4.5 as published in BF16, Kimi K2 as published in FP8), for
  gpt-oss, ZAYA1 and Inkling, which hold all the experts of a layer in one tensor, and for experts with
  two matrices (`amd/gpt-oss-20b-w-mxfp4-a-bf16`, and `Inferact/Qwen3.8-2.4T-A95B-MXFP4`, whose scales
  are also F32). Today `pack` stores those as ordinary tensors.
- **Done when:** convert → verify → downgrade round-trips byte for byte on the fixture, and for a real
  checkpoint on the checkpoint. Today the first holds for all sixteen profiles, the round trip holds on 8
  real experts of three of them, and pack → verify → unpack holds on one whole checkpoint for `qwen38`
  (RAW expert records; `upgrade` and `downgrade` were not run on it).

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
  verdicts, on synthetic records, on the tiny-model containers and the three accepted conformance
  vectors with WNR3 records (since the merges of 2026-11-15; before, one), and on mutated copies of
  them. It runs on Mesa lavapipe, a Vulkan device on the CPU, in the opt-in `linux-vulkan` job
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
  records of the tiny-model container, with the same bytes. Not run with the shader of that day: a
  container of a real model (none was on that machine).
- **On a real container (2026-11-16):** the same GPU and driver, the shader of `main` at b95de79, the
  published DeepSeek V4.1 Flash REAP-192 container (7,680 WNR3-CZ records, flag byte 0x0f)
  ([research/vulkan-real-20261116](../research/vulkan-real-20261116/README.md)). `run.sh` with
  `MUTATIONS=0 WNR3_VK_EVERY=64` and the validation layer on: 120 records, device and reference agree on
  120 accepted (40,623,950 blocks undone by the ceil-form inverse), 0 differences, in both shader forms,
  0 messages of the validation layer, `vulkan-decode: all checks passed`. `wnr3_vk bench` on its first 16
  records: 1.056 ms of device time per record, 3.386 ms with the upload, against 24.1 ms for the fast CPU
  decoder (AVX2) on the 8 threads of the i7-7700K, 99.5 ms on one and 775 ms for the reference decoder;
  other work loaded that CPU (19% to 48% in the readings), so the CPU times are upper bounds.
- **Records made from real experts, on the same GPU, with the prototype of before the fix (2026-11-15,
  commit `1af6c0d`, byte identity only):** of 32 WNR3 records made from real experts (8 each of DeepSeek
  V4.1 with and without canonical zero, Qwen3-235B and MiniMax-M2.1, partial containers), the 16 of
  DeepSeek V4.1, which have many ceil-form blocks, decoded to other bytes than the reference's: 19% of
  the scale decrements of the ceil-form inverse were missing, the fault the run above found on the
  selftest; every weight byte was right. The 16 others, with few ceil-form blocks, gave the reference's
  bytes. With the decoded buffer of a copy of that `wnr3.comp` declared `coherent`, all 32 gave the
  reference's bytes ([research/real-experts-20261115](../research/real-experts-20261115/README.md),
  section 4). The shader of today, which fixes the fault another way, was not run on these records.
- **Still open, most wanted first:**
  - run `bash wex4/research/vulkan-decode/run.sh <container.wex4>` with a real container (WNR3 records of
    a real model) on any other GPU or driver (Intel, AMD, Apple through MoltenVK, NVIDIA on Linux), and
    report the last lines and the device line: that is the test that counts, and so far it has passed on
    one GPU only. The RTX 3070 showed that a driver can get this shader wrong, and only `run.sh` tells.
    The container of the run above is public
    ([Kaira666/Kaira-WEX4-Experts](https://huggingface.co/Kaira666/Kaira-WEX4-Experts), 123 GB); the
    records of `research/real-experts-20261115` (made by its `run.sh` from tensors its
    `fetch_experts.py` fetches) are a small test that needs no download of that size. The host program
    reads the file record by record and cuts its batches to the device's buffer limit, so the file size
    is not a limit. The slow part is the CPU reference, which decodes every record too (about 1.6 s per
    V4.1 record on that machine, reading included): start with `MUTATIONS=0 WNR3_VK_EVERY=64` (no
    mutated copies, one record in 64), as the folder's README says. `WNR3_VK_DEVICE=<part of the name>`
    picks the device;
  - the speed on real records on other GPUs, and on the RTX 3070 on other layers than layer 0 (`wnr3_vk
    bench RUNS FILE RECORDS` takes the first records of a file);
  - the commit step (copy the decoded regions to the engine's buffers only when the record's error word
    is zero) and the hook into colibri's Vulkan backend (design note, sections 8 and 11);
  - the llvmpipe crash of issue 21: on Mesa 25.2.8 with LLVM 20.1.2 it is reduced to a shader of about 30
    lines and a host of its own (`crash_min.comp`, `crash_min.c`) that crash only with a binding 0 in the
    descriptor set layout; the first form of the shader was valid SPIR-V and gave no message of the
    validation layer, so the crash is in the driver. On a newer Mesa on Linux it did not crash: on
    2026-11-16 (run 43828157734, opt-in job `linux-vulkan-newmesa`, one run in each container) Mesa
    26.2.7 with LLVM 23.1.4 (`archlinux:latest`) and Mesa 26.2.6 with LLVM 23.1.2 (`fedora:latest`,
    Fedora 45) ran 0 crashes in the 50 runs of the probe, `crash_min` and the first form of the shader
    included, and passed every `run.sh` check (decode, reject, mutate, the conformance vectors; 0
    differences, 0 messages of the validation layer), while `linux-vulkan` in the same run (Mesa 25.2.8,
    LLVM 20.1.2) crashed in 38 of 61. Open: its cause inside Mesa, which change ended it, and a report to
    Mesa (none was sent). The folder's README, "The llvmpipe crash", has the forms.
- **Done when:** its output is byte-identical to the CPU reference on every record of a real container,
  there is a speed number against the CPU path on the same machine, and it builds without the CUDA
  toolkit. Today the second and the third hold, on one GPU (the speed on 16 real records of the V4.1
  container); the first holds on one record in 64 of that container, on that GPU; the other records and
  the engine hook are open.
- HIP and Metal ports are welcome after it, on the same tests.

## 7. I/O on Linux and macOS (help wanted)

- **Today:** the read paths were tuned on Windows (unbuffered reads, overlapped I/O). On Linux
  `c/expert_store_wex.c` reads one record per `pread` through the page cache, and `c/qwen38_wex.h` one
  per `pread` with `O_DIRECT`. The benchmark exists: [`bench/wex_io.py`](../bench/README.md) writes a
  synthetic WEX4 container and reads its expert records through 27 read paths (page cache or `O_DIRECT`,
  `posix_fadvise` hints, three ways to prefetch the next record, 1 to 8 threads, `io_uring` with records
  or sub-reads in flight, and four rows with 20 ms of work after each record), each checked against the
  index's SHA-256 and timed over several cold passes.
- **Measured so far:** GitHub Actions runners, shared cloud VMs with a virtual disk, and one desktop
  under WSL2 (last sentence of this item); no bare Linux on a real drive.
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
  One full run under WSL2 on a desktop (2026-11-16, ext4 on the WSL2 virtual disk, a VHDX file on the
  machine's NVMe SSD: not a bare NVMe) gave the first large gains from reads in flight: `O_DIRECT` on 4
  threads 3,351 MB/s, 1.77x the store's path, `io_uring` with 4 records in flight 1.75x, `WILLNEED` in
  chunks 1.39x without work and the helper thread 1.26x with 20 ms of work
  ([research/linux-io-wsl2-20261116](../research/linux-io-wsl2-20261116/README.md)).
- **macOS:** the same benchmark runs on macOS through the calls macOS has (`F_NOCACHE` for `O_DIRECT`,
  `F_RDAHEAD` and `F_RDADVISE` for the hints, `purge` for `drop_caches`; `io_uring`,
  `POSIX_FADV_SEQUENTIAL` and `readahead(2)` print `skipped`, so 17 of the 27 paths run there). The
  `macos` CI job runs its smoke run; the job
  `macos-io-bench` is the full run, started by hand. Measured so far: three runs of 11 rounds on GitHub
  Actions `macos-15` runners, virtual machines on Apple silicon hosts with a virtual disk
  ([research/macos-io-20261115](../research/macos-io-20261115/README.md); a round is one timed pass of
  every path, and with 11 rounds every path takes every place in a round once; those runs had the 11
  paths of before the prefetch rows, which have run on macOS only in the smoke run). All 11 paths read the
  right bytes. `F_NOCACHE` on 1 thread was 2.56x to 3.53x the page-cache `pread` and faster than it in
  31 of the 33 rounds; page-cache threads gained 1.16x to 1.60x; `F_RDAHEAD` off and `F_RDADVISE` gave
  no steady gain. But a cache below the virtual machine served part of the reads (the first pass of a
  round was one of the two slowest of its row in 30 of the 33 rounds), so these numbers say nothing
  about the SSD of a Mac. The store was not changed.
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
  macOS on a real Mac (the SSD of an Apple silicon Mac, an Intel Mac, an external drive: only virtual
  machines with a virtual disk ran it), with the prefetch rows too (on macOS they have run only in the
  smoke run of CI).
- **Done when:** the table of `bench/wex_io.py` from at least one NVMe machine is in the repository, and
  a change of the store comes with its before/after from the same benchmark, stated as CONTRIBUTING
  section 5 asks. Today one table from a machine with an NVMe SSD is in, but read through the virtual
  disk of WSL2; a table of Linux on the drive itself, and a change of the store, are open.

## 8. Compression research

- **Baselines:** compress real expert records with zstd, lz4 and others at several levels and compare
  ratio and decode speed with WNR3 on the same records. Honest negative results are welcome too.
  - **Today (2026-11-15):** on 24 expert records made from real experts (8 each of DeepSeek V4.1
    Flash, Qwen3-235B and MiniMax-M2.1, two layers of each, partial checkpoints), WNR3 (byte-exact, with
    a model trained on the same 8 records) was smaller than every general-purpose codec measured: 0.8796,
    0.8919 and 0.8856 of raw, against 0.9222, 0.8951 and 0.9043 for the best of zstd (levels 1 to 22),
    lz4, lz4hc, brotli (quality 1 to 11), lzma, zlib and bz2, which was brotli each time. On Qwen3-235B
    a model trained on one layer made WNR3 larger than zstd (0.9019 and 0.9189 against 0.8966). Decoding
    was slower: on an Intel Core i7-7700K shared with other jobs, the fast CPU decoder gave about 200 MB/s
    on one thread and 820 to 890 on 8, zstd about 1,500 and 5,500 to 6,900; only lzma, bz2 and (on one
    thread) brotli at quality 11 were slower than it
    ([research/real-experts-20261115](../research/real-experts-20261115/README.md), section 3).
  - **Still open:** the same on whole checkpoints, with a model trained as in production (80 experts
    spread over the layers); other CPUs, and the NEON kernel on these records (the scripts of that folder
    fetch them, about 320 MB, and need no model); the GPU decoders against GPU codecs (nvCOMP, GDeflate);
    per-tensor compression (scales and weights apart); zstd with a trained dictionary or long-distance
    matching; byte-shuffle filters (Blosc2) and OpenZL.
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
  one GPU (an RTX 3070). On that GPU, on 2026-11-16, the shader of today gave the reference's bytes on one
  record in 64 of the published DeepSeek V4.1 container (120 records), and decoded its records in 1.06 ms
  of device time each against 24.1 ms for the fast CPU decoder on 8 threads. The other records of that
  container, other GPUs and drivers, and the engine hook are open *(open to you, section 6)*.
- **ARM and macOS:** the ARM CI jobs (Linux, and macOS on Apple silicon with GNU gcc and with Apple
  clang), ASan and UBSan and the libFuzzer harnesses on both, a NEON version of kernel 1 of the fast
  CPU decoder and an SVE version of kernel 2 are in; the NEON kernel's speed on synthetic records was
  measured on the Linux runner and on the macOS runner's Apple M1 VM, also built with Apple clang, for
  every number of interleaved streams (the default stays 4); the SVE kernel is not the default, because on
  the CI runner's 128-bit SVE it is slower than NEON kernel 1. Their speed on real records and on ARM
  CPUs other than the Linux CI runner, and the SVE kernel on longer vectors, are still open *(open
  to you, section 3)*.
- **Linux I/O in the engine:** the fast read paths were built and measured on Windows. The Linux
  benchmark is in (`bench/wex_io.py`), run on six kinds of cloud VM, where it gave no reason to change the
  store, and once under WSL2 on a desktop with an NVMe SSD (through the virtual disk of WSL2), where 2 to
  16 reads in flight gave 1.64x to 1.77x the store's path. A prefetch of the next record through the page
  cache gained on the VMs only with work between the reads (`WILLNEED` in chunks, 1.24x to 2.32x with a
  20 ms spin standing for a decode), which only an engine A/B can confirm; runs of Linux on real drives
  and the tuning that follows from them are still open *(open to you, section 7)*. The same benchmark now runs on macOS, so far only on GitHub `macos-15`
  virtual machines (three runs of 11 rounds), where a cache below the virtual machine served part of the
  reads and the store was not changed; a run on a real Mac is open *(open to you, section 7)*.
- **Measurements not done yet** (they need the full models, so they stay with the maintainer, see
  [STATUS.md](../STATUS.md)):
  - the SHA-256 of each record at its first read on Qwen3.8;
  - a full ABBA set for it on DeepSeek V4.1;
  - tuning the two-drive split for Qwen3.8;
  - MTP together with MXFP4 experts.
- **The specifications after the review and the two readers:** the review wrote 16 clarifications
  into WEX4-SPEC and WNR3-SPEC and left 4 findings open; the Go reader of `impl/go-wex4` found no
  wrong verdict in the 112 manifest rows of that day, and a list of sentences that can be read two
  ways and of cases the vectors did not exercise. On 2026-11-15 the 4 findings and the rest of that
  list got sentences and 24 vectors, with no rule changed. The writer and the trainer of 2026-11-15
  added eight notes (part E), seven of which want a sentence of the specification. A second review
  pass of 2026-11-15 read the sections the first had left out and wrote 18 clarifications. The Rust
  reader of `impl/rust-wex4`, written the same day, found no wrong verdict in the 112 rows and left
  six questions (part E5 of its notes) and four gaps of the index (part C). Open: the seven notes,
  the six questions, the 3 findings of the second pass, the gaps of the index and the WNR3 checks no
  vector exercises alone *(open to you, sections 2 to 4)*.
- **Fuzzing for hours:** the harnesses run for 60 seconds each in CI; one run of an hour per harness,
  on x86-64 Linux and on the code of 2026-11-15 before the merges of that day, is recorded, and it
  stopped on nothing. Longer runs, runs on the code of today and on ARM64 and macOS are open *(open
  to you, section 3)*.
- **The CUDA decoder:** reviewed against WNR3-SPEC and fuzzed as C++ on the CPU, through an
  emulation of the CUDA constructs, on 2026-11-15, and its kernels replayed with the mutator on one
  GPU (an RTX 3070, PTX from clang, no CUDA toolkit); no difference of verdict or bytes from the CPU
  reference was found. On 2026-11-16 the helper as nvcc builds it (CUDA 13.5, `sm_86`) passed its
  CPU/GPU differential on the same GPU, also on 16 records made from real V4.1 experts and on 17 records
  of the published V4.1 container, with 120 mutated copies per bin
  ([research/cuda-real-20261116](../research/cuda-real-20261116/README.md)). That helper under
  `compute-sanitizer`, other GPUs and a sentence of WNR3-SPEC for the host batch are open *(open to
  you, sections 2 and 3)*.
- **The engine-side readers inside colibri:** the two engine harnesses follow colibri's headers and
  build against them (a pinned `dev` commit, the opt-in `colibri-headers` CI job). `c/qwen38_wex.h` and
  the members it adds to colibri's expert index are in no public colibri tree yet, so its harness keeps
  those members inferred until the upstreaming above brings them in; the CUDA helper's paths are only
  replayed, on Windows, with a CPU stand-in *(open to you, section 3)*.
- **Converter profiles for more model families:** sixteen naming profiles are in, tested on synthetic
  fixtures, among them seven for compressed-tensors exports; the gate, down and up roles of w1, w2, w3
  were read in the model code; three profiles (`deepseek`, `qwen3-moe`, `mixtral`) also round-tripped
  8 real experts each, and `qwen38` round-tripped a whole real checkpoint (Qwen3.5-35B-A3B MXFP4,
  RAW expert records). Runs on whole real checkpoints of the other profiles are open *(open to you,
  section 5)*.
- **Compression baselines:** WNR3 against zstd, lz4, brotli, lzma, zlib and bz2 on 24 real expert
  records is in (smaller files, slower CPU decode); whole checkpoints, other CPUs and GPU codecs are
  open *(open to you, section 8)*.
- **A first tagged release** of the specifications (WEX4 4.0, WNR3 v1) with the conformance vectors, so
  other implementations can name the version they pass.

## Not in scope here

- **Engine features unrelated to the format** (sampling, chat templates, new architectures): those
  belong in [colibri](https://github.com/JustVugg/colibri).
- **Lossy quantization formats:** WEX stores the checkpoint's own bytes or a lossless coding of them.
- **Model weights:** never commit them; see the README's "Model weights" section.
