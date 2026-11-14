# WNR3 decode on Vulkan compute (prototype)

**Question.** Can a Vulkan compute shader decode today's WNR3 records, with no CUDA and no format change, and give exactly the bytes and the verdicts of the CPU reference `wex_nr2_decode` (`c/wex_neural_rans.h`)?

**Answer (2026-11-14).** Yes on every record it was run on, on Mesa lavapipe, which is a Vulkan device that executes on the CPU. **Nothing was run on a GPU: there is no speed number, and the decoder is not hooked into any engine.** The design, and what a port still has to decide, are in [docs/WNR3-VULKAN.md](../../../docs/WNR3-VULKAN.md). The task this belongs to is section 6 of [docs/HELP-WANTED.md](../../../docs/HELP-WANTED.md), which stays open for the parts named under "Limits".

## Files

| File | What |
|---|---|
| `wnr3.comp` | the compute shader, GLSL 4.50, 32-bit unsigned arithmetic only, no extension. One invocation decodes one rANS stream. It builds twice: `scales.spv` (pass A, the scale streams) and, with `-DWEIGHTS`, `weights.spv` (pass B, the weight streams: contexts, stream hash, canonical-zero check, ceil-form inverse) |
| `wnr3_vk.c` | the host program, C99, Vulkan 1.0 core API, no feature and no extension enabled. It validates each record and its model on the CPU as `wex4_decode_record` does, uploads them, runs both passes for a batch of up to 64 records in one command buffer, and compares the device's verdict and bytes with the reference. Modes `selftest`, `decode`, `reject`, `mutate` (its header comment) |
| `run.sh` | builds both into `out/` (git-ignored) and runs the four modes. Last line on success: `vulkan-decode: all checks passed` |
| `negative_control.sh` | changes one line of a copy of the shader at a time, seven changes, and requires `run.sh` to fail on each. Last line on success: `negative control: 7 of 7 broken shaders noticed` |

## Run it

It needs `gcc`, `glslangValidator`, the Vulkan headers and loader, and any Vulkan device. It is not part of `scripts/check_all.sh`.

```sh
# Debian, Ubuntu: sudo apt-get install libvulkan-dev glslang-tools mesa-vulkan-drivers vulkan-validationlayers
bash wex4/build_and_test.sh          # optional: leaves wex4/out/tiny_nr3_cz.wex4 and wex4/out/synth.wex4
bash wex4/research/vulkan-decode/run.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
bash wex4/research/vulkan-decode/negative_control.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
```

From the repository root. Without arguments `run.sh` uses the conformance vectors only. Any WEX4 file with WNR3 records can be named, a real container too: every WNR3 record of it must decode, and `MUTATIONS` copies of each record (default 20) are decoded with one byte changed. `WNR3_VK_DEVICE=N` picks another device than the first (the program lists them), `WNR3_VK_VALIDATE=1` turns the Khronos validation layer on and makes any message of it an error. On MSYS2 UCRT64 the packages are `vulkan-headers`, `vulkan-loader`, `glslang` and `mesa`, and `VK_DRIVER_FILES` must name Mesa's `lvp_icd.x86_64.json` when no other driver is installed.

In CI: the job `linux-vulkan` of `.github/workflows/ci.yml`, which only a run started by hand has.

## Result

"Agree" means the same verdict as `wex_nr2_decode`, and for an accepted record the same bytes in all six regions (three of scales, three of packed weights). Counts are those of one run of the CI job on an ubuntu-24.04 runner, device `llvmpipe (LLVM 20.1.2, 256 bits)`, Vulkan 1.4.318, validation layer on with 0 messages. One run on Windows 10 (MSYS2 UCRT64, Mesa llvmpipe with LLVM 23.1.4, Vulkan 1.4.354) gave the same counts.

| Check | Records | Outcome |
|---|---:|---|
| `selftest`: synthetic records from the reference encoder and a synthetic model; flag bytes 0x00, 0x08, 0x07, 0x0f; chunk pairs from 512:256 to 16384:16384; scale counts that are not multiples of 4 | 8 | 5 accepted with the reference's bytes, 534 blocks of them undone by the ceil-form inverse; 3 refused by both with the expected error bit (ceil scale 0: 4, nibble 8 under canonical zero: 8, changed payload byte: 1); 0 differences |
| `decode` `wex4/vectors/valid/tiny2-cz-upgraded.wex4` | 2 | 2 accepted, 0 differences |
| `decode` `wex4/out/tiny_nr3_cz.wex4` (tiny model, WNR3-CZ, built by `wex4/build_and_test.sh`) | 48 | 48 accepted, 0 differences |
| `decode` `wex4/out/synth.wex4` (synthetic, built by the same script) | 4 | 4 accepted, 0 differences |
| `reject`: the three vectors of `wex4/vectors/invalid/` whose WNR3 data a decoding reader refuses | 6 | 3 refused by the host checks, 1 refused by the device, 2 accepted; the same as the reference for each |
| `mutate`: 20 copies of each of the 54 records above, one byte of the used part changed, seed 20261114 | 1,080 | 1,075 refused by the device and by the reference, 5 refused by the host checks, 0 accepted, 0 differences |
| `negative_control.sh` | 7 broken shaders | 7 noticed |

The four modes together take about 7 seconds on the runner and the negative control about 14; these are times of a CPU rasteriser on shared hardware and say nothing about a GPU.

## Limits

- **No GPU.** Only llvmpipe was run. A GPU driver compiles the shader with another compiler and runs the invocations truly at once; neither was exercised. No Intel, AMD, NVIDIA, Apple or mobile driver was tried.
- **No speed number.** The shader uses the reference binary search on the model blob (4 probes per weight, 8 per scale); the faster lookups of the CUDA kernels are described in the design note and not built.
- **No real model.** The containers are the tiny-model fixture, a synthetic one and a conformance vector: 54 records in all, flag bytes 0x08 and 0x00. A record in ceil form was decoded only in `selftest` (synthetic data and a synthetic model). The largest synthetic record holds 1,903 blocks; a real V4.1 record holds 1,105,920 (computed from its size), so stream counts and buffer sizes of that order were never on the device.
- **No commit step and no engine hook.** The decoded bytes stay in a device buffer; nothing copies them to an engine's buffers. The prototype reads them back only to compare them.
- **WNR3 only.** A WNR2 record (codec 4) is not handled.
- The host program was not built under AddressSanitizer or UndefinedBehaviorSanitizer, and no fuzzer drives it; the mutation mode changes one byte per copy.
