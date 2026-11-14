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
- **macOS job in CI, NEON decoder:** CI runs on Linux (x86-64 and ARM64) and Windows. On a CPU that is
  not x86 `wex4/build_and_test.sh` drops `-march=haswell`, so the fast CPU decoder builds its scalar
  kernel only. Still open: a macOS job (the scripts call `gcc` and link with `-fopenmp`), good first
  issue; a NEON version of the AVX2 kernels in `wex4/research/fast-cpu-decode/wex_fast_decode.h`,
  byte-identical to the reference on the vectors and with a speed number against the scalar kernel,
  help wanted.

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

## 6. Port the WNR3 GPU decode to Vulkan (help wanted, priority)

- **Today:** WNR3 records are decoded on the GPU only through CUDA (`c/tools/wex_cuda_bridge.cu`). On
  Intel (Arc) and AMD GPUs the CPU decoder runs instead, so those cards cannot decode compressed experts
  straight into VRAM.
- **What:** a Vulkan compute version of the WNR3 decode into device memory, following the CUDA kernels
  (one record = many independent rANS streams, then the canonical-zero and scale handling of
  [WNR3-SPEC](WNR3-SPEC.md)). colibri already has a Vulkan backend to hook it into.
- **Done when:** its output is byte-identical to the CPU reference on every record of a real container,
  there is a speed number against the CPU path on the same machine, and it builds without the CUDA
  toolkit.
- **Start with:** a design note in an issue (how streams map to workgroups, where the model tables live,
  how records are batched). HIP and Metal ports are welcome after it, on the same tests.

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

## Roadmap: what is still open on the maintainer's side

So you can see where things are going and avoid duplicating work. Items marked *(open to you)* are
listed above as well.

- **Integration into colibri.** The format and its readers are being upstreamed into
  [colibri](https://github.com/JustVugg/colibri) `dev` as a series of small pull requests: Qwen3.8 MXFP4
  experts first (#2822-#2825 are open), then the WEX4 container, the WNR3 codec and its GPU decode, and
  the I/O switches. Reviewing or testing those PRs on your hardware helps directly.
- **WNR3 decode on Vulkan** *(open to you, section 6)*.
- **ARM:** the CPU decoder path without AVX2 and the ARM CI job are in (scalar kernel); a NEON version
  is still open *(open to you, section 3)*.
- **Linux I/O in the engine:** the fast read paths were built and measured on Windows; Linux needs its
  own measurements and tuning *(open to you, section 7)*.
- **Measurements not done yet** (they need the full models, so they stay with the maintainer, see
  [STATUS.md](../STATUS.md)):
  - the SHA-256 of each record at its first read on Qwen3.8;
  - a full ABBA set for it on DeepSeek V4.1;
  - tuning the two-drive split for Qwen3.8;
  - MTP together with MXFP4 experts.
- **A first tagged release** of the specifications (WEX4 4.0, WNR3 v1) with the conformance vectors, so
  other implementations can name the version they pass.

## Not in scope here

- **Engine features unrelated to the format** (sampling, chat templates, new architectures): those
  belong in [colibri](https://github.com/JustVugg/colibri).
- **Lossy quantization formats:** WEX stores the checkpoint's own bytes or a lossless coding of them.
- **Model weights:** never commit them; see the README's "Model weights" section.
