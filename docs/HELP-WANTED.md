# Help wanted

Things an outside contributor can pick up, roughly from easiest to hardest. Each item says what is
needed, where to start and how to tell it is done. Before writing code, read
[CONTRIBUTING.md](../CONTRIBUTING.md) (privacy rules, how results are stated, sign-off). For anything
bigger than a small fix, open an issue first so two people do not do the same work.

Labels you may see on issues: **good first issue** (a few hours, little context), **help wanted**
(larger, needs a design note first).

## 1. Run the benchmark on your hardware (no code)

- **What:** run [`bench/`](../bench/README.md) and share `results.json` through the
  "Benchmark results" issue template.
- **Most wanted:** ARM machines (DGX Spark, Apple silicon under Linux, Ampere); AMD and Intel GPUs;
  filesystems other than NTFS (ZFS, btrfs, ext4, XFS); NVMe RAID or mirrors; machines with
  128 GB+ of RAM.
- **Done when:** the issue has the JSON and a note on the run conditions.

## 2. Documentation (good first issue)

- **Spec review:** read [WEX4-SPEC](WEX4-SPEC.md) and [WNR3-SPEC](WNR3-SPEC.md) as if you were
  writing a reader from scratch. Open an issue for every sentence that is ambiguous, every field
  whose range is not stated, and every place where the spec and the code disagree. This is the most
  valuable review there is.
- **Glossary:** add missing terms to [GLOSSARY.md](GLOSSARY.md).
- **Translations:** a short README in your language (the Italian summary at the end of the README
  shows the expected size).

## 3. Tests and fuzzing

- **Fuzzing harnesses:** a libFuzzer/AFL++ harness for the C WEX4 reader (`c/wex4.h`, `c/tools/wex4_verify.c`)
  and the WNR3 decoder. Run it for hours, not seconds. The in-process fuzzer in `wex4/wex4_tools.c` is a
  starting point. Done when it builds in CI and any crash comes with a reproducer.
- **Sanitizers in CI:** run the CPU checks under ASan/UBSan on Linux (`scripts/check_all.sh` with
  `-fsanitize=address,undefined`).
- **More test vectors:** hostile files that [wex4/vectors](../wex4/vectors/README.md) does not cover yet
  (each with its `MANIFEST.tsv` row and the expected verdict from both readers).
- **macOS and ARM jobs in CI:** today CI runs on x86-64 Linux and Windows only, and the fast CPU
  decoder in `wex4/build_and_test.sh` is built with `-march=haswell` (AVX2). An ARM build needs that
  path guarded (reference decoder only, or a NEON version), then a CI job. Good first issue for the
  guard, help wanted for NEON.

## 4. A second, independent implementation

- **What:** a WEX4 reader and verifier in another language (Rust, Go, C++, Zig…) written **from the
  spec only**, not by porting `wex4.py`.
- **Done when:** it gives the same verdict as the reference on every file in
  [wex4/vectors](../wex4/vectors/README.md), and the same root for the valid ones. Every disagreement
  is either a bug in the new reader or an ambiguity in the spec, and both are worth an issue.
- **Optional:** the WNR3 decoder as well, checked byte for byte against the reference on the vectors.

## 5. Converters for more models (help wanted)

- **Today:** [wex4_convert.py](../wex4/wex4_convert.py) knows the tensor names of two families
  (`--names deepseek|qwen38`).
- **What:** naming profiles for other MoE checkpoints in safetensors (for example Mixtral, Qwen MoE
  variants, GLM, Kimi-style models), with a tiny synthetic fixture for each, as
  `fixtures/make_synth_mxfp4.py` does.
- **Done when:** convert → verify → downgrade round-trips byte for byte on the fixture.

## 6. GPU decode beyond CUDA (help wanted, larger)

- **Today:** WNR3 records are decoded on the GPU only through CUDA (`c/tools/wex_cuda_bridge.cu`). On
  Intel and AMD GPUs the CPU decoder runs.
- **What:** a Vulkan compute (or HIP, or Metal) version of the WNR3 decode into device memory.
- **Done when:** its output is byte-identical to the CPU reference on every record of a real container,
  and there is a speed number against the CPU path on the same machine.
- **Start with:** a design note in an issue, including how the rANS state is split across threads.

## 7. I/O on Linux and macOS (help wanted)

- **What:** the read paths were tuned on Windows (unbuffered reads, overlapped I/O). Measure and
  improve the Linux side: `O_DIRECT` vs buffered, `io_uring` batches of expert records, readahead
  hints, striping across drives with different speeds.
- **Done when:** there is a reproducible benchmark (`bench/` or a script beside it) and a before/after
  table stated as CONTRIBUTING section 5 asks.

## 8. Compression research

- **Baselines:** compress real expert records with zstd, lz4 and others at several levels and compare
  ratio and decode speed with WNR3 on the same records. Honest negative results are welcome too.
- **Ideas for a next codec:** see [NEXT-FORMAT-SPEC](NEXT-FORMAT-SPEC.md) and the research reports in
  [docs/reports](reports/). Any proposal has to stay lossless (bit-identical weights).

## Not in scope here

- **Engine features unrelated to the format** (sampling, chat templates, new architectures): those
  belong in [colibri](https://github.com/JustVugg/colibri).
- **Lossy quantization formats:** WEX stores the checkpoint's own bytes or a lossless coding of them.
- **Model weights:** never commit them; see the README's "Model weights" section.
