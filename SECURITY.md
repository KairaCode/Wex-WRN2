# Security policy

A WEX file is untrusted input: it can be damaged, or built by someone who has read the reader's code. This page says which code is meant to read such files, how to report a problem, and what the formats do not protect against.

## Supported

| Code | Untrusted files | Fixes |
|---|---|---|
| WEX4 4.0 readers: `c/wex4.h` (C) and `wex4/wex4.py` (Python reference); the checker `c/tools/wex4_verify.c`; the converter `wex4/wex4_convert.py` | yes: this is what they are written and fuzzed for ([WEX4-SPEC](docs/WEX4-SPEC.md), section 9) | yes |
| The Go reader `impl/go-wex4/`, with its WBP1 and WNR3 decoders | yes: it does the checks of [WEX4-SPEC](docs/WEX4-SPEC.md) section 6 and checks every hash. It is tested with the conformance vectors, with seeded mutations and with one differential run against the two readers above, not with a coverage-guided fuzzer. Its first version gave another verdict than those readers on two kinds of file outside the vectors, since fixed; where the specification can be read two ways it may still differ ([its README](impl/go-wex4/README.md), "Limits"). `wex4check export` verifies the whole file before it creates or writes anything, then reads it again to write it, and names the files it writes by number, never by a tensor name of the file. The manifest of `wex4check write` and `wex4check train` is not input to take from someone else: it names the files they read | yes |
| WNR3 record decoders (`c/wex_neural_rans.h`, `c/wex_cuda.cuh`), when reached through a WEX4 reader | yes: every stream carries a 23-bit hash, and the record is covered by its SHA-256 | yes |
| WEX1, WEX2 and WEX3 readers (`c/wex_format.h`, `c/expert_store_wex.c`, `scripts/verify_wex1.py`) | **no.** These containers have no hashes, and WNR2 records accepted 1.8-1.9% of random bit flips without noticing. Open only files you built yourself | crashes are fixed as bugs |
| `archive/`, `research/` | no: lab code, kept for the record | no |
| WNR3 decoder on Vulkan, `wex4/research/vulkan-decode/` (prototype, in no engine) | no. The shader does not bound-check: it is safe only on records and models that its host program validated first, and that program is a test tool that no fuzzer drives ([docs/WNR3-VULKAN.md](docs/WNR3-VULKAN.md), section 6) | no |

The engines that read these files in production (the DeepSeek loader and the maintainer's application host) are not in this repository; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reporting a vulnerability

A file that makes a supported reader read or write out of bounds, crash, hang, allocate without limit, or accept bytes that do not match the hashes is a vulnerability. Please report it privately:

1. Use GitHub's private vulnerability reporting: the **Security** tab of the repository, then **Report a vulnerability**.
2. If that button is not there, open an ordinary issue that says only that you have a security report and asks for a private channel. Put no details, files or code in it.

Include the smallest file that shows the problem (or the script that builds it from `fixtures/`), the reader and the commit you used, your OS, compiler and Python version, and what happens. Never attach real containers or third-party model weights.

## What to expect

There is one maintainer and no fixed response time; reports are handled on a best-effort basis. A confirmed problem is fixed in the two reference readers together (they must keep agreeing on every verdict) and in the Go reader when it is affected, gets a hostile test case in `wex4/test_wex4.py` or a conformance vector in [wex4/vectors/](wex4/vectors/README.md), and is published as a GitHub security advisory once the fix is in `main`.

## What the formats do not protect against

From [WEX4-SPEC](docs/WEX4-SPEC.md), sections 8.4 and 9:

- **Integrity, not authenticity.** The 32-byte root covers the whole file, but whoever can rewrite the file can rewrite its hashes too. A root only protects you when you pin it from a source the attacker cannot write, such as your own configuration.
- **Hash collisions inside a record.** The hash of a WNR3 stream is 23 bits: a corruption passes it with probability 2^-23 per stream. The SHA-256 of the record catches it when the reader checks it.
- **Denial of service.** A valid but huge file is bounded only by the reader's own budget.
- **Free-form content.** Metadata, unknown blobs, the reserved header area and tensor data can hold anything, so a file that is also valid as another format (a polyglot) is possible.
- Side channels are out of scope.

The older containers have one more case, from the code and not from a specification. The standalone store (`c/expert_store_wex.c`, not a supported reader of untrusted files) allocates 24 slabs of the header's `aligned_expert_bytes` per layer when it opens. Since 2026-11-15 the open path refuses a WEX1 or WEX2 header that states more than twice the largest 4096-aligned decoded record of its table; before, it accepted any multiple of 4096 below 4 GiB that no record exceeded, and a failed allocation was handled. WEX3 already required the field to equal the largest 4096-aligned decoded record. The number of layers is still not weighed against the file: a table of many layers with one record makes the store ask for 24 slabs for each of them.

## Testing, and its gaps

- `bash wex4/build_and_test.sh` runs 110 hand-made hostile files, mutation fuzzing in three modes in C and in Python, and a differential test in which the C and Python readers must agree on every verdict and content digest.
- `bash impl/go-wex4/run.sh` (the `go-reader` job of CI) runs the Go reader on the conformance vectors and on 35,000 seeded mutations of nine valid files: bit flips with the hashes left alone, header and index bytes with the hashes recomputed, bytes of compressed records with the record hash recomputed. It also gives the files of the Go writer, fixed ones and 24 random ones in every run, to the two reference readers and to `wex4_verify`.
- `bash wex4/fuzz/run.sh [SECONDS]` runs five coverage-guided harnesses (libFuzzer with AddressSanitizer and UndefinedBehaviorSanitizer) on the C reader, the WNR3 decoders, `wex4_verify`, the standalone expert store (`c/expert_store_wex.c`) and the container reader of the Qwen3.8 engine (`c/qwen38_wex.h`), seeded with the conformance vectors and, for the two engine harnesses, eight small expert containers that `make_seeds.py` writes (four more, with palette, WNR and WNR2 records, for a second run of the store harness); a custom mutator re-encodes WNR3 records so that their stream hashes stay right, and in the store harness also scale-tile, palette, WNR and WNR2 records ([wex4/fuzz/README.md](wex4/fuzz/README.md)). The two engine files are built against stand-ins for the engine headers, not inside an engine. CI runs each for 60 seconds (the store harness twice) on every pull request, on x86-64 Linux, 64-bit ARM Linux and macOS (Apple silicon), and replays the seeds and the mutators (`run.sh --replay`, no coverage guidance) on x86-64 Linux, 64-bit ARM Linux, macOS (Apple silicon, with GNU gcc and with Apple clang) and Windows; nothing runs them for hours on a schedule, and on Windows no coverage-guided run is made.
- An adversarial review of the spec and the prototype was made on 2026-11-11: [wex4/review/REVIEW.md](wex4/review/REVIEW.md).
- The guard pages that turn an out-of-bounds access into a crash during fuzzing exist only in the Windows build of `wex4/wex4_tools.c` (`VirtualAlloc`); elsewhere the fuzzer runs without them. On x86-64 Linux, 64-bit ARM Linux and macOS (Apple clang), CI runs every CPU check, the fuzzer included, and the fuzz replay under AddressSanitizer and UndefinedBehaviorSanitizer (the jobs `linux-sanitizers`, `linux-arm-sanitizers` and `macos-clang`); on Windows no sanitizer runs. The jobs `linux-tsan` and `linux-msan` run the C test executables, not the libraries loaded into Python, under ThreadSanitizer and MemorySanitizer (`scripts/sanitize_c.sh`). No fuzzing runs for hours on a schedule.
- The Python tools load whatever library path they are given (`wex4.py --dll`, `c/tools/build_nr3.py`): only pass them a library you built yourself.
