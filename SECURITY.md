# Security policy

A WEX file is untrusted input: it can be damaged, or built by someone who has read the reader's code. This page says which code is meant to read such files, how to report a problem, and what the formats do not protect against.

## Supported

| Code | Untrusted files | Fixes |
|---|---|---|
| WEX4 4.0 readers: `c/wex4.h` (C) and `wex4/wex4.py` (Python reference); the checker `c/tools/wex4_verify.c`; the converter `wex4/wex4_convert.py` | yes: this is what they are written and fuzzed for ([WEX4-SPEC](docs/WEX4-SPEC.md), section 9) | yes |
| WNR3 record decoders (`c/wex_neural_rans.h`, `c/wex_cuda.cuh`), when reached through a WEX4 reader | yes: every stream carries a 23-bit hash, and the record is covered by its SHA-256 | yes |
| WEX1, WEX2 and WEX3 readers (`c/wex_format.h`, `c/expert_store_wex.c`, `scripts/verify_wex1.py`) | **no.** These containers have no hashes, and WNR2 records accepted 1.8-1.9% of random bit flips without noticing. Open only files you built yourself | crashes are fixed as bugs |
| `archive/`, `research/` | no: lab code, kept for the record | no |

The engines that read these files in production (the DeepSeek loader and the maintainer's application host) are not in this repository; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reporting a vulnerability

A file that makes a supported reader read or write out of bounds, crash, hang, allocate without limit, or accept bytes that do not match the hashes is a vulnerability. Please report it privately:

1. Use GitHub's private vulnerability reporting: the **Security** tab of the repository, then **Report a vulnerability**.
2. If that button is not there, open an ordinary issue that says only that you have a security report and asks for a private channel. Put no details, files or code in it.

Include the smallest file that shows the problem (or the script that builds it from `fixtures/`), the reader and the commit you used, your OS, compiler and Python version, and what happens. Never attach real containers or third-party model weights.

## What to expect

There is one maintainer and no fixed response time; reports are handled on a best-effort basis. A confirmed problem is fixed in both WEX4 readers together (they must keep agreeing on every verdict), gets a hostile test case in `wex4/test_wex4.py` or a conformance vector in [wex4/vectors/](wex4/vectors/README.md), and is published as a GitHub security advisory once the fix is in `main`.

## What the formats do not protect against

From [WEX4-SPEC](docs/WEX4-SPEC.md), sections 8.4 and 9:

- **Integrity, not authenticity.** The 32-byte root covers the whole file, but whoever can rewrite the file can rewrite its hashes too. A root only protects you when you pin it from a source the attacker cannot write, such as your own configuration.
- **Hash collisions inside a record.** The hash of a WNR3 stream is 23 bits: a corruption passes it with probability 2^-23 per stream. The SHA-256 of the record catches it when the reader checks it.
- **Denial of service.** A valid but huge file is bounded only by the reader's own budget.
- **Free-form content.** Metadata, unknown blobs, the reserved header area and tensor data can hold anything, so a file that is also valid as another format (a polyglot) is possible.
- Side channels are out of scope.

## Testing, and its gaps

- `bash wex4/build_and_test.sh` runs 94 hand-made hostile files, mutation fuzzing in three modes in C and in Python, and a differential test in which the C and Python readers must agree on every verdict and content digest.
- An adversarial review of the spec and the prototype was made on 2026-11-11: [wex4/review/REVIEW.md](wex4/review/REVIEW.md).
- The guard pages that turn an out-of-bounds access into a crash during fuzzing exist only in the Windows build of `wex4/wex4_tools.c` (`VirtualAlloc`); elsewhere the fuzzer runs without them. There is no sanitizer build and no continuous fuzzing.
- The Python tools load whatever library path they are given (`wex4.py --dll`, `c/tools/build_nr3.py`): only pass them a library you built yourself.
