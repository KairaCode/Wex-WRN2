# WNR3 decode on Vulkan compute (prototype)

**Question.** Can a Vulkan compute shader decode today's WNR3 records, with no CUDA and no format change, and give exactly the bytes and the verdicts of the CPU reference `wex_nr2_decode` (`c/wex_neural_rans.h`)?

**Answer (2026-11-14).** Yes on every record it was run on, on Mesa lavapipe, which is a Vulkan device that executes on the CPU. **Nothing was run on a GPU: there is no speed number, and the decoder is not hooked into any engine.** The design, and what a port still has to decide, are in [docs/WNR3-VULKAN.md](../../../docs/WNR3-VULKAN.md). The task this belongs to is section 6 of [docs/HELP-WANTED.md](../../../docs/HELP-WANTED.md), which stays open for the parts named under "Limits".

## Files

| File | What |
|---|---|
| `wnr3.comp` | the compute shader, GLSL 4.50, 32-bit unsigned arithmetic only, no extension. One invocation decodes one rANS stream. It builds twice: `scales.spv` (pass A, the scale streams) and, with `-DWEIGHTS`, `weights.spv` (pass B, the weight streams: contexts, stream hash, canonical-zero check, ceil-form inverse) |
| `wnr3_vk.c` | the host program, C99, Vulkan 1.0 core API. The device is created with no feature and no extension; the instance enables `VK_EXT_debug_utils` only when the validation layer is asked for. It reads a WEX4 file record by record (64-bit offsets, the file is never loaded whole), validates each record and its model on the CPU as `wex4_decode_record` does, uploads them, runs both passes for a batch in one command buffer, and compares the device's verdict and bytes with the reference. A batch is at most 64 records and at most the batch limit in stored and in decoded bytes. Modes `selftest`, `decode`, `reject`, `mutate` (its header comment) |
| `run.sh` | builds both into `out/` (git-ignored) and runs the four modes. Last line on success: `vulkan-decode: all checks passed` |
| `negative_control.sh` | changes one line of a copy of the shader at a time, eleven changes, and requires `run.sh` to fail on each. Last line on success: `negative control: 11 of 11 broken shaders noticed` |
| `big_file.py` | makes a WEX4 file above 4 GiB out of a small one (blobs of zero bytes in front of the records), to check the 64-bit offsets of the host program |

## Run it

It needs `gcc`, `glslangValidator`, the Vulkan headers and loader, and any Vulkan device. It is not part of `scripts/check_all.sh`.

```sh
# Debian, Ubuntu: sudo apt-get install libvulkan-dev glslang-tools mesa-vulkan-drivers vulkan-validationlayers
bash wex4/build_and_test.sh          # optional: leaves wex4/out/tiny_nr3_cz.wex4 and wex4/out/synth.wex4
bash wex4/research/vulkan-decode/run.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
bash wex4/research/vulkan-decode/negative_control.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
```

From the repository root. Without arguments `run.sh` uses the conformance vectors only. Every WNR3 record of a file named on the command line must decode, and `MUTATIONS` copies of each record (default 20) are decoded with one byte changed. `WNR3_VK_DEVICE=N` picks another device than the first (the program lists them), `WNR3_VK_VALIDATE=1` turns the Khronos validation layer on and makes any message of it an error. On MSYS2 UCRT64 the packages are `vulkan-headers`, `vulkan-loader`, `glslang` and `mesa`, and `VK_DRIVER_FILES` must name Mesa's `lvp_icd.x86_64.json` when no other driver is installed.

**A container of a real model.** None was run (see "Limits"), so what follows is what the program is built for, not a report. The file size is not a limit: records are read one at a time. The batch limit is `WNR3_VK_BATCH_BYTES` (default 2^28), lowered to the device's `maxStorageBufferRange`; with 128 MiB, which lavapipe reports, that is 7 records of DeepSeek V4.1 per batch by their size. The slow part is the CPU reference, which decodes every record and every mutated copy too, at about half a second per V4.1 record on one thread ([WEX4-SPEC](../../../docs/WEX4-SPEC.md) section 11.3, not measured here): 7,680 records are about an hour, and the default 20 mutated copies of each would be about 20 more. A first run should be short:

```sh
MUTATIONS=0 WNR3_VK_EVERY=64 bash wex4/research/vulkan-decode/run.sh /path/to/container.wex4   # one record in 64, no mutated copies
MUTATIONS=0 bash wex4/research/vulkan-decode/run.sh /path/to/container.wex4                    # every record
```

In CI: the job `linux-vulkan` of `.github/workflows/ci.yml`, which only a run started by hand has.

## Result

"Agree" means the same verdict as `wex_nr2_decode`, and for an accepted record the same bytes in all six regions (three of scales, three of packed weights). Counts are those of one run of the CI job on an ubuntu-24.04 runner, device `llvmpipe (LLVM 20.1.2, 256 bits)`, Mesa 25.2.8, Vulkan 1.4.318, validation layer on with 0 messages. One run on Windows 10 (MSYS2 UCRT64, Mesa llvmpipe with LLVM 23.1.4, Vulkan 1.4.354) gave the same counts.

| Check | Records | Outcome |
|---|---:|---|
| `selftest`, valid: synthetic records from the reference encoder and a synthetic model; flag bytes 0x00, 0x08, 0x07, 0x0f; chunk pairs from 512:256 to 16384:16384; scale counts that are not multiples of 4 | 5 | 5 accepted with the reference's bytes, 534 blocks of them undone by the ceil-form inverse; 0 differences |
| `selftest`, refused: coded scale 0 in a max-5 block; nibble 8 under canonical zero; one changed payload byte; the owner word changed (every stream decodes, every hash is wrong); a scale stream with one byte after its last symbol; a scale stream that opens below `L`; one that opens at `256 L` or more. The last three are built so that a decoder without that one check decodes the reference's bytes with the right hash | 7 | 7 refused by both, each with the error word it must have: 4, 8, 1 (bit 1 or 2 allowed), 2, 1, 1, 1 |
| `decode` `wex4/vectors/valid/tiny2-cz-upgraded.wex4` | 2 | 2 accepted, 0 differences |
| `decode` `wex4/out/tiny_nr3_cz.wex4` (tiny model, WNR3-CZ, built by `wex4/build_and_test.sh`) | 48 | 48 accepted, 0 differences |
| `decode` `wex4/out/synth.wex4` (synthetic, built by the same script) | 4 | 4 accepted, 0 differences |
| the same three files with a batch limit of 256 KiB | 54 | 1, 3 and 4 batches; 54 accepted, 0 differences |
| `reject`: the three vectors of `wex4/vectors/invalid/` whose WNR3 data a decoding reader refuses | 6 | 3 refused by the host checks, 1 refused by the device, 2 accepted; the same as the reference for each |
| `mutate`: 20 copies of each of the 54 records above, one byte of the used part changed, seed 20261114 | 1,080 | 1,075 refused by the device and by the reference, 5 refused by the host checks, 0 accepted, 0 differences |
| `negative_control.sh` | 11 broken shaders | 11 noticed |
| `big_file.py`: the 4 records of `synth.wex4` behind 5 blobs of 1 GiB (file of 5,369,438,208 B, first record at offset 5,368,782,848) | 4 | 4 accepted, 0 differences |
| all of `run.sh` again with the host program built under AddressSanitizer and UndefinedBehaviorSanitizer (the driver is not instrumented, leak detection off) | as above | passed, no report |

Times of that one CI run (run 43647269411 of the repository's Actions, on the commit before this page was last edited; whole seconds from the step log, build included): `run.sh` 10 s, the negative control 18 s, the 5 GiB file 14 s of which 13 to write it, `run.sh` under the sanitizers 8 s. These are times of a CPU rasteriser on shared hardware and say nothing about a GPU.

Local only, Windows 10 with the Mesa above, one run each: `MUTATIONS=1000` (54,000 mutated copies of the same 54 records) gave 53,769 refused by both, 231 refused by the host checks, 0 accepted, 0 differences; and the file made by `big_file.py` passes `wex4/wex4.py verify` (every hash checked, every record decoded).

## Limits

- **No GPU.** Only llvmpipe was run. A GPU driver compiles the shader with another compiler and runs the invocations truly at once; neither was exercised. No Intel, AMD, NVIDIA, Apple or mobile driver was tried.
- **No speed number.** The shader uses the reference binary search on the model blob (4 probes per weight, 8 per scale); the faster lookups of the CUDA kernels are described in the design note and not built.
- **No real model.** The containers are the tiny-model fixture, a synthetic one and a conformance vector: 54 records in all, flag bytes 0x08 and 0x00. A record in ceil form was decoded only in `selftest` (synthetic data and a synthetic model). The largest synthetic record holds 14,000 blocks; a real V4.1 record holds 1,105,920 (computed from its size), so stream counts and buffer sizes of that order were never on the device. The batch cut was exercised only with small records and a small limit, and the 64-bit offsets only with the file of `big_file.py`.
- **A driver crash was found and worked around, not explained.** With the first form of the shader, llvmpipe of Mesa 25.2.8 with LLVM 20.1.2 crashed when scale stream 0 or 8 of a record had an initial state out of range. The shader now decodes such a stream to its end and refuses the record there, which does not crash; design note, section 6. The same may exist elsewhere in another shape.
- **One check of the shader is not pinned by a test:** the stop at the end of a stream inside the renormalisation loop. Without it a damaged stream reads on into the next one and is refused by the end-of-stream check, so no verdict changes (one local run with the line removed: `run.sh` passed).
- **No commit step and no engine hook.** The decoded bytes stay in a device buffer; nothing copies them to an engine's buffers. The prototype reads them back only to compare them.
- **WNR3 only.** A WNR2 record (codec 4) is not handled.
- **Not for untrusted files** ([SECURITY.md](../../../SECURITY.md)). The shader does not bound-check; it relies on the host checks. No fuzzer drives the host program, and the mutation mode changes one byte per copy. The sanitizer build runs only in the CI job started by hand.
