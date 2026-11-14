# wex4/fuzz/

**Current.** Coverage-guided fuzzing harnesses for the C readers and decoders, for [libFuzzer](https://llvm.org/docs/LibFuzzer.html) (and AFL++ through its libFuzzer driver). They complement the in-process mutation fuzzer of [`../wex4_tools.c`](../wex4_tools.c) (`wex4_fuzz_run`, run by [`../build_and_test.sh`](../build_and_test.sh)): that one applies a fixed number of random edits to three files on every run of the tests; these keep the inputs that reach new code and can run for hours.

| Harness | Input | Code under test | What must hold, besides the sanitizers |
|---|---|---|---|
| `fuzz_wex4_file.c` | a whole WEX4 file | `c/wex4.h`: header page, index, RAW / WBP1 / WNR3 record decoding, through `check_impl` of `wex4_tools.c` | each input is checked as it is (every hash verified) and again with the header and index hashes recomputed and the records decoded without their hash, so that a mutation reaches the validators and the decoders. A file accepted with its hashes decodes to the same content digest without them; decoding twice gives the same digest; a file whose records decode has a header page and an index that validate |
| `fuzz_wnr3_record.c` | the stored bytes of one WNR3 record | `wex_nrx_parse` and `wex_nr2_decode` (`c/wex_neural_rans.h`), `wex_fast_decode` (`../research/fast-cpu-decode/wex_fast_decode.h`) | the reference and the fast decoder give the same verdict, and on success the same bytes. The model, the layout and the record size come from the first WNR3 record of a real container (`WEX_FUZZ_BASE`, default `wex4/vectors/valid/tiny2-cz-upgraded.wex4`) |
| `fuzz_wex4_verify.c` | a whole WEX4 file | `c/tools/wex4_verify.c` (through a temporary file: the checker takes a path) | it answers "does not open as WEX4" exactly when `wex4_parse_header` or `wex4_open` refuses the same bytes, and the root of a file it passes is the SHA-256 of bytes [0, 4064) |

## Running them

```sh
bash wex4/fuzz/run.sh            # 60 s per harness
bash wex4/fuzz/run.sh 14400      # 4 hours per harness
bash wex4/fuzz/run.sh --replay   # no fuzzing: gcc, every seed once through every harness
```

The first two need `clang` with libFuzzer, AddressSanitizer and UndefinedBehaviorSanitizer (`-fsanitize=fuzzer,address,undefined`): on Linux they are the compiler-rt runtimes (Debian and Ubuntu: `libclang-rt-N-dev`, which `clang` installs unless recommended packages are turned off; Fedora: `compiler-rt`); the MSYS2 UCRT64 clang does not accept `-fsanitize=fuzzer`, so on Windows use WSL or only `--replay`. `CC` names the compiler (`CC=clang-18`), `PYTHON` the interpreter (`PYTHON=python3` on Linux), as in the other scripts. `--replay` needs only `gcc`: it builds each harness with `-DWEX_FUZZ_STANDALONE`, which adds a `main` that runs the files named on its command line, and runs the seeds through it.

The seeds are the conformance vectors ([`../vectors/`](../vectors/README.md)): every file of the manifest for the two file harnesses, and the WNR3 records of the accepted ones for the record harness (`make_seeds.py`). Everything is written under `wex4/fuzz/out/`, which is git-ignored: `corpus/<harness>/` holds the inputs libFuzzer kept and is reused by the next run, until the folder is deleted.

## When a harness stops

`run.sh` prints the end of the harness's log and exits 1. The input is in `wex4/fuzz/out/crashes/`, named `<harness>-crash-<sha1>` (or `-oom-`, `-timeout-`), and the log names what happened: a sanitizer report with a stack, or a line `ORACLE FAILED: ...` for one of the properties of the table. To see it again:

```sh
wex4/fuzz/out/fuzz_wex4_file wex4/fuzz/out/crashes/fuzz_wex4_file-crash-<sha1>
```

A crash file is a hostile input by construction. One found on your own machine is reported as [SECURITY.md](../../SECURITY.md) says, not in a public issue, unless it is only a failed oracle with no memory error. One found by the CI run is public already: the job prints the end of the log and keeps the reproducer as the artifact `fuzz-crashes` of a public run. A file that the readers must refuse and that no vector covers is a candidate for [`../vectors/`](../vectors/README.md).

## Limits

- `fuzz_wex4_file.c` does not decode a file whose header claims more than 64 MiB for one record (`FUZZ_MAX_DECODED`). The format allows 2^30 bytes; the harness would spend its time filling memory. Such a file is still validated up to that point.
- A WNR3 stream carries a 23-bit hash of its decoded symbols, checked after the stream is decoded; a mutated stream is therefore decoded and then refused. The fuzzer reaches the decode loops, rarely what follows a stream that passed.
- The harnesses run on one thread and are built without `-fopenmp`: the parallel paths of `wex4_verify.c` and of the fast decoder are not fuzzed.
- The GPU decoder (`c/wex_cuda.cuh`) and the engine store (`c/expert_store_wex.c`, `c/qwen38_wex.h`) have no harness.
- CI (`.github/workflows/fuzz.yml`) runs each harness for 60 seconds: enough to notice a harness that no longer builds or a shallow bug, not a substitute for a long run.
