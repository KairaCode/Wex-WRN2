# wex4/fuzz/

**Current.** Coverage-guided fuzzing harnesses for the C readers and decoders, for [libFuzzer](https://llvm.org/docs/LibFuzzer.html) (and AFL++ through its libFuzzer driver). They complement the in-process mutation fuzzer of [`../wex4_tools.c`](../wex4_tools.c) (`wex4_fuzz_run`, run by [`../build_and_test.sh`](../build_and_test.sh)): that one applies a fixed number of random edits to three files on every run of the tests; these keep the inputs that reach new code and can run for hours.

| Harness | Input | Code under test | What must hold, besides the sanitizers |
|---|---|---|---|
| `fuzz_wex4_file.c` | a whole WEX4 file | `c/wex4.h`: header page, index, RAW / WBP1 / WNR3 record decoding, through `check_impl` of `wex4_tools.c` | each input is checked as it is (every hash verified) and again with the header and index hashes recomputed and the records decoded without their hash, so that a mutation reaches the validators and the decoders. A file accepted with its hashes decodes to the same content digest without them; decoding twice gives the same digest; a file whose records decode has a header page and an index that validate |
| `fuzz_wnr3_record.c` | the stored bytes of one WNR3 record | `wex_nrx_parse` and `wex_nr2_decode` (`c/wex_neural_rans.h`), `wex_fast_decode` (`../research/fast-cpu-decode/wex_fast_decode.h`) | the reference and the fast decoder give the same verdict, and on success the same bytes. The model and the layout come from the first WNR3 record of a real container (`WEX_FUZZ_BASE`, default `wex4/vectors/valid/tiny2-cz-upgraded.wex4`); the input is zero-padded to a multiple of 4096 bytes, up to that record's decoded size rounded up to 4096 |
| `fuzz_wex4_verify.c` | a whole WEX4 file | `c/tools/wex4_verify.c` (through a temporary file: the checker takes a path) | it answers "does not open as WEX4" exactly when `wex4_parse_header` or `wex4_open` refuses the same bytes, and the root of a file it passes is the SHA-256 of bytes [0, 4064) |
| `fuzz_store_wex.c` | a whole legacy container (WEX1, WEX2, WEX3) | `c/expert_store_wex.c` opened without an engine, with `c/wex_decode.c` and `c/wex_format.h` (through a temporary file named by `COLI_WEX_PATH`): open, lookup, release, prefetch, statistics, close | the store opens the file exactly when `wex_validate_header`, `wex_validate_table` and the model checks accept the same bytes in memory; a lookup succeeds exactly when the record's bytes pass the owner check and decode with `wex_decode_record` or `wex_decode_nr2_record`, and the three matrices it hands out are those decoded bytes; an expert that loaded loads again, and a cache hit reads no byte; the request, hit and miss counters add up |
| `fuzz_q38_wex.c` | a whole container (WEX1, legacy WNR3, WEX4) | the container reader of the Qwen3.8 engine, `c/qwen38_wex.h`: `q38_wex_manifest`, `q38_wex4_manifest`, `q38_wex_entry_ok`, the model registration, `q38_expert_mx4_attach`, `q38_mx4_load` (a WEX4 file twice, without and with `COLI_WEX_VERIFY=1`) | the attach does not stop on a container whose manifest, entries and model the same functions accepted; every record of an attached container lies inside the file and its read fits its slab; a load stops the engine exactly when the record fails the first-read hash (when it is on), the owner check or `wex_decode_nr2_record`, and otherwise the slot holds exactly the decoded bytes |

## The two engine harnesses

`c/expert_store_wex.c` and `c/qwen38_wex.h` are engine files: they include headers that are not in this repository, and their own tests (`c/tests/test_expert_store_wex.c`, `c/tests/test_qwen38_mxfp4.c`) build only inside an engine tree. The harnesses build them here against the stand-ins of [`engine/`](engine/):

- `engine/expert_store.h`, `engine/deepseek_v4_internal.h` and `engine/compat.h` declare what `c/expert_store_wex.c` uses of the engine headers of those names (the tensor and expert views, the store's function table, the open options, `Sleep`). The first two follow colibri's headers. The harness defines the two engine functions the store calls: `coli_v4_expert_store_open_planned` (never opens anything) and `coli_expert_store_backend_register` (records nothing). On 2026-11-14 the harness also built against the headers of a colibri checkout (commit `0a5504a`) in place of the stand-ins and replayed its seeds (Windows, MSYS2 UCRT64 gcc 16.2, `--replay` only; not run on Linux, where that tree's `compat.h` has no `Sleep`).
- `engine/q38_engine.h` declares the engine types `c/qwen38_wex.h` names: the configuration, the model, the expert slot, the weight, the expert index with its `wex_*` and `mx_*` members. Those members are in no public tree, so their types are inferred from how `c/qwen38_wex.h` uses them. A difference between a stand-in type and the engine's (a width, a signedness) is a difference this harness cannot see. `c/qwen38_wex.h` cannot be built against the real engine from this repository. On macOS, which has no `O_DIRECT`, `c/qwen38_wex.h` opens the file through the engine's `compat_open_direct`; the stand-in opens it and sets `F_NOCACHE`, which is a guess at what the engine's function does.
- The engine stops with `exit(1)` on a container it refuses. In `fuzz_q38_wex.c` an `exit` inside `c/qwen38_wex.h` returns to the harness (`longjmp`), which first runs the same checks without the exit and calls the real attach only on a container that passed them. The geometry (layers, experts, hidden and intermediate size) is taken from the input, where the engine takes it from the model's configuration.
- Both harnesses empty the model registry of `c/wex_decode.c` (16 models for the life of the process) before every input, and send the log lines of the code under test nowhere.

Every line of `c/expert_store_wex.c`, `c/qwen38_wex.h`, `c/wex_decode.c`, `c/wex_format.h` and `c/wex4.h` that the harnesses run is the real one.

## The WNR3-aware mutator and the dictionary

A WNR3 stream starts at a state that holds a 23-bit hash of the bytes it decodes to, and the decoder requires that state at its end. A byte changed in a stored record fails the first stream it touches, so with libFuzzer's own mutations the fuzzer reaches the decode loops and nothing behind an accepted stream. `wnr3_mutator.h` mutates on the other side of the codec: it decodes the record, edits the decoded scales and weights and the record's parameters (chunk sizes, owner, anchors, flags), and encodes the result with the reference encoder, which writes the right hash into every stream. Two calls in three it encodes the canonical way (canonical zero, ceil form); one in three it draws the flag bits at random, so that a record with right hashes is still refused later (a -0 under the canonical-zero flag, a scale of 0 in a block the ceil inverse has to undo).

Four harnesses define `LLVMFuzzerCustomMutator` with it; every other call is libFuzzer's own mutation (`LLVMFuzzerMutate`):

| Harness | Structured calls | What they change |
|---|---|---|
| `fuzz_wnr3_record` | 2 in 3 | the input record, or the base record when the input does not decode |
| `fuzz_store_wex` | 1 in 4 | one WNR3 record of a legacy container that opens, in place, when the new record takes the same pages |
| `fuzz_q38_wex` | 1 in 4 | the same, and one WNR3 record of a WEX4 file that opens; its SHA-256 in the index is left alone, so the record loads without `COLI_WEX_VERIFY` and stops the engine with it |
| `fuzz_wex4_file` | 1 in 4 | one WNR3 record of the input as the second check sees it (header and index hashes recomputed) |

`fuzz_wex4_verify` has none: the checker verifies the SHA-256 of every record, which a mutator cannot keep right without rewriting the index. `WEX_FUZZ_MUTATOR=0` in the environment turns the structured calls off.

`wex.dict` is a libFuzzer dictionary with the magic numbers of the containers, records and blobs and a few field values; `run.sh` passes it to the four harnesses whose input is a whole file.

Each harness with a mutator prints one line at exit with what the run reached. Measured on 2026-11-14 on GitHub Actions (`ubuntu-24.04` runner, the distribution's `clang`, `-fsanitize=fuzzer,address,undefined`, 60 s per harness from the seeds, an empty corpus), one run with the mutators ([43641976259](https://github.com/KairaCode/Wex-WRN2/actions/runs/43641976259)) and one with `WEX_FUZZ_MUTATOR=0` ([43642516736](https://github.com/KairaCode/Wex-WRN2/actions/runs/43642516736)), same source, each run once:

| Harness | What is counted | With the mutators | `WEX_FUZZ_MUTATOR=0` |
|---|---|---|---|
| `fuzz_wnr3_record` | inputs run | 6,595 | 123,139 |
| | records both decoders accepted | 4,297 | 30 |
| | different decoded records among them | 4,242 | 2 |
| | `cov` / `ft` | 287 / 924 | 246 / 582 |
| `fuzz_store_wex` | inputs run | 32,556 | 64,423 |
| | WNR3 records the store served | 2,270 | 626 |
| | different decoded records among them | 405 | 2 |
| | `cov` / `ft` | 409 / 1,001 | 403 / 924 |
| `fuzz_q38_wex` | inputs run | 38,487 | 60,321 |
| | WNR3 records loaded | 2,329 | 422 |
| | different decoded records among them | 392 | 2 |
| | `cov` / `ft` | 580 / 1,610 | 575 / 1,534 |
| `fuzz_wex4_file` | inputs run | 57,715 | 87,804 |
| | files accepted with the header and index hashes recomputed | 614 | 633 |
| | different content digests among them | 444 | 442 |
| | `cov` / `ft` | 513 / 2,056 | 506 / 1,987 |

`cov` is libFuzzer's count of instrumented edges reached and `ft` its count of features, in the harness and the code under test together. What the numbers show: without the mutator no WNR3 record other than the two of the seeds was accepted, in any of the three harnesses that count them; with it the record harness decoded 4,242 different accepted records and reached 41 more edges. In the three file harnesses the edge counts differ by 5 to 7, which is inside the variation between runs: a second run with the mutators ([43645082256](https://github.com/KairaCode/Wex-WRN2/actions/runs/43645082256), same conditions, once, made after `fuzz_store_wex` got its counts by codec, which add one check to the harness) gave `fuzz_wex4_file`, `fuzz_store_wex` and `fuzz_q38_wex` 500, 406 and 577 edges where the table has 513, 409 and 580. The runner is shared hardware, and a structured call costs a decode and an encode under the sanitizers, so a run with the mutators executes fewer inputs. Not measured: line coverage of the code under test, more than two runs of a setting, and runs longer than 60 s. GitHub deletes the log of a run after its retention period; the figures here were copied from the logs.

On ARM64 and on macOS, 2026-11-15: the same 60 s runs, with the mutators, on `ubuntu-24.04-arm` (Ubuntu clang 18.1.3) and on `macos-15` (Apple silicon, Homebrew clang 18.1.8 of `llvm@18`), once each, in one run of the Fuzz workflow ([43698184361](https://github.com/KairaCode/Wex-WRN2/actions/runs/43698184361)). There `fuzz_wnr3_record` compares the NEON kernel of the fast decoder with the reference: it ran 7,691 inputs on ARM64 Linux and 10,442 on macOS, of which both decoders accepted 4,980 and 6,900 (4,916 and 6,820 different decoded records), with `cov` 291 and 295. No harness stopped, on either. The x86-64 job of the same run, which compares the AVX2 kernels, ran 15,492 inputs and accepted 10,140 (`cov` 292); the three jobs build different kernels, so their edge counts are not comparable. The figures are those of one run; an earlier run of the same two jobs that day ([43672893056](https://github.com/KairaCode/Wex-WRN2/actions/runs/43672893056)) also ended with no harness stopped. No run without the mutators was made there.

## Running them

```sh
bash wex4/fuzz/run.sh            # 60 s per harness
bash wex4/fuzz/run.sh 14400      # 4 hours per harness
bash wex4/fuzz/run.sh --replay   # no fuzzing: gcc, every seed once through every harness
```

The first two need `clang` with libFuzzer, AddressSanitizer and UndefinedBehaviorSanitizer (`-fsanitize=fuzzer,address,undefined`): on Linux they are the compiler-rt runtimes (Debian and Ubuntu: `libclang-rt-N-dev`, which `clang` installs unless recommended packages are turned off; Fedora: `compiler-rt`); the MSYS2 UCRT64 clang does not accept `-fsanitize=fuzzer`, so on Windows use WSL or only `--replay`. Apple clang ships no libFuzzer runtime: on macOS name the clang of a Homebrew LLVM (`CC=$(brew --prefix llvm@18)/bin/clang`, which brings libFuzzer and both sanitizers; the Fuzz workflow takes the highest `llvm@N` of the runner image). `CC` names the compiler (`CC=clang-18`), `PYTHON` the interpreter (`PYTHON=python3` on Linux and macOS), as in the other scripts. `--replay` needs only `gcc` (or the compiler named in `CC`, for example a Homebrew `gcc-15` or Apple clang on macOS): it builds each harness with `-DWEX_FUZZ_STANDALONE`, which adds a `main` that runs the files named on its command line, and runs the seeds through it. A harness with a custom mutator also runs that mutator on every seed in `--replay` (200 inputs from each seed of `fuzz_wnr3_record`, 100 for `fuzz_store_wex`, 40 for `fuzz_wex4_file` and `fuzz_q38_wex`), with a small deterministic stand-in for `LLVMFuzzerMutate`: the mutator and the oracles behind it are tested without clang. Seven jobs of `.github/workflows/ci.yml` run `--replay` after the CPU checks: gcc on x86-64 Linux, on 64-bit ARM Linux and on macOS (Apple silicon, a Homebrew GNU gcc), on Windows, and under AddressSanitizer and UndefinedBehaviorSanitizer on x86-64 Linux and on 64-bit ARM Linux (`linux-sanitizers`, `linux-arm-sanitizers`); and Apple clang on macOS, once as it is and once under both sanitizers (`macos-clang`). `--replay` prints the compiler and the kernel of the fast decoder it built, and every ARM64 job fails unless it is NEON. One `--replay` took 28 s on 2026-11-14 (Windows 10 desktop, MSYS2 UCRT64 gcc 16.2, one run).

The seeds are the conformance vectors ([`../vectors/`](../vectors/README.md)): every file of the manifest for the two WEX4 file harnesses, and the WNR3 records of the accepted ones for the record harness (`make_seeds.py`). The two engine harnesses get six small expert containers that `make_seeds.py` writes: a WEX1 file and a WEX2 file made there, the legacy vector `tiny2-cz.wex3`, the same with its grid cut to its two experts, and the WEX4 form of two of them; `fuzz_q38_wex` also gets every file of the manifest. No seed holds palette (codec 2), WNR (codec 3) or WNR2 (codec 4) records. Everything is written under `wex4/fuzz/out/`, which is git-ignored: `corpus/<harness>/` holds the inputs libFuzzer kept and is reused by the next run, until the folder is deleted.

## When a harness stops

`run.sh` prints the end of the harness's log and exits 1. The input is in `wex4/fuzz/out/crashes/`, named `<harness>-crash-<sha1>` (or `-oom-`, `-timeout-`), and the log names what happened: a sanitizer report with a stack, or a line `ORACLE FAILED: ...` for one of the properties of the table. To see it again:

```sh
wex4/fuzz/out/fuzz_wex4_file wex4/fuzz/out/crashes/fuzz_wex4_file-crash-<sha1>
```

A crash file is a hostile input by construction. One found on your own machine is reported as [SECURITY.md](../../SECURITY.md) says, not in a public issue, unless it is only a failed oracle with no memory error. One found by the CI run is public already: the job prints the end of the log and keeps the reproducer as the artifact `fuzz-crashes` of a public run. A file that the readers must refuse and that no vector covers is a candidate for [`../vectors/`](../vectors/README.md).

## Limits

- `fuzz_wex4_file.c` does not decode a file whose header claims more than 64 MiB for one record (`FUZZ_MAX_DECODED`). The format allows 2^30 bytes; the harness would spend its time filling memory. Such a file is still validated up to that point.
- `fuzz_store_wex.c` does not open a container whose header would make the store allocate more than 64 MiB of slabs (`FUZZ_MAX_SLABS`: 24 slabs per layer of `aligned_expert_bytes` each, which a WEX1 or WEX2 header may set to anything up to 4 GiB whatever its records are). `fuzz_q38_wex.c` does not attach a grid of more than 4096 experts or a record of more than 16 MiB. It looks up or loads at most 64 experts of a container.
- Not fuzzed in the engine files: the mirror files and the striped read, `WEX_READ_RETRY`, the load wave and every path through the CUDA helper (`COLI_WEX_CUDA_LIB=0` is set), the Quark shards of `Q38_MXFP4_DIR`, unbuffered reads (`O_DIRECT` is not requested on Linux; Windows is only replayed), and the store opened through an engine (`coli_v4_expert_store_open_planned`). Palette, WNR and WNR2 records have decoders in `c/wex_format.h` that `fuzz_store_wex.c` can reach, but no seed and no mutator. `fuzz_store_wex` prints the records the store served by codec in its exit line: in the one 60 s run with the mutators made since it does ([43645082256](https://github.com/KairaCode/Wex-WRN2/actions/runs/43645082256), conditions as in the table above) they were 326 raw, 148 scale-tile, 2,218 WNR3 and no palette, WNR or WNR2 record. Not counted: the run without the mutators, and any longer run.
- The mutator does not write WNR2 records (the same streams without the hash), and it replaces a record inside a container only when the new one takes the same number of pages.
- The harnesses run on one thread and are built without `-fopenmp`: the parallel paths of `wex4_verify.c` and of the fast decoder are not fuzzed. They are run once on the seeds, not fuzzed, under ThreadSanitizer with `-fopenmp` and under MemorySanitizer by [`../../scripts/sanitize_c.sh`](../../scripts/README.md), which calls `run.sh --replay` with `CC=clang` and its flags in `WEX_CFLAGS`.
- The GPU decoder (`c/wex_cuda.cuh`) has no harness.
- CI (`.github/workflows/fuzz.yml`) runs each harness for 60 seconds: enough to notice a harness that no longer builds or a shallow bug, not a substitute for a long run. It builds with clang on x86-64 Linux and on 64-bit ARM Linux (the distribution's clang) and on macOS (Apple silicon, the clang of the highest `llvm@N` of Homebrew on the runner image, `llvm@18` on 2026-11-15), one job each; on Windows the harnesses only run `--replay`, which has no coverage guidance. No Intel Mac.
