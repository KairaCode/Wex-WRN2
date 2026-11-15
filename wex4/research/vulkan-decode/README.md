# WNR3 decode on Vulkan compute (prototype)

**Question.** Can a Vulkan compute shader decode today's WNR3 records, with no CUDA and no format change, and give exactly the bytes and the verdicts of the CPU reference `wex_nr2_decode` (`c/wex_neural_rans.h`)?

**Answer (2026-11-15).** Yes on every record it was run on: on Mesa lavapipe, a Vulkan device that executes on the CPU, and on one GPU, an NVIDIA GeForce RTX 3070 with driver 621.01 on Windows 10. On that GPU the first form of the shader lost scale decrements of the ceil-form inverse, a driver fault that the shader now avoids; since then every check below passes there too, with the validation layer on. On synthetic records of the size of a DeepSeek V4.1 expert the GPU decoded a record in 1.23 ms of device time, against 24.9 ms for the fast CPU decoder on the 8 threads of the same desktop ([research/vulkan-gpu-20261115](../../../research/vulkan-gpu-20261115/README.md)). **No record of a real model was decoded, and the decoder is not hooked into any engine.** The design, and what a port still has to decide, are in [docs/WNR3-VULKAN.md](../../../docs/WNR3-VULKAN.md). The task this belongs to is section 6 of [docs/HELP-WANTED.md](../../../docs/HELP-WANTED.md), which stays open for the parts named under "Limits".

## Files

| File | What |
|---|---|
| `wnr3.comp` | the compute shader, GLSL 4.50, 32-bit unsigned arithmetic only, no extension. One invocation decodes one rANS stream. It builds twice: `scales.spv` (pass A, the scale streams) and, with `-DWEIGHTS`, `weights.spv` (pass B, the weight streams: contexts, stream hash, canonical-zero check, ceil-form inverse). Two forms of the symbol lookup: the default reads the scale symbol from a table of one byte per slot and compares a whole weight row with the slot at once; `-DSEARCH` is the reference binary search of the first version |
| `wnr3_vk.c` | the host program, C99, Vulkan 1.0 core API. The device is created with no extension and no feature (`robustBufferAccess` only when asked for); the instance enables `VK_EXT_debug_utils` only when the validation layer is asked for. It reads a WEX4 file record by record (64-bit offsets, the file is never loaded whole), validates each record and its model on the CPU as `wex4_decode_record` does, builds the scale symbol table from the model and checks it against the binary search on every slot, uploads them (through staging buffers into device-local memory on a GPU), runs both passes for a batch in one command buffer, and compares the device's verdict and bytes with the reference. A batch is at most 64 records and at most the batch limit in stored and in decoded bytes. Modes `selftest`, `decode`, `reject`, `mutate`, `bench` (its header comment) |
| `run.sh` | builds both into `out/` (git-ignored) and runs the four checking modes, the first two also with the `-DSEARCH` form, and the bench mode once on two small records (both forms must give the reference's bytes; no time is read). Last line on success: `vulkan-decode: all checks passed` |
| `negative_control.sh` | changes one line of a copy of the shader at a time, eleven changes, and requires `run.sh` to fail on each. Last line on success: `negative control: 11 of 11 broken shaders noticed` |
| `big_file.py` | makes a WEX4 file above 4 GiB out of a small one (blobs of zero bytes in front of the records), to check the 64-bit offsets of the host program |
| `bench.sh` | a measurement, not a check: writes a container of synthetic records of the size of a DeepSeek V4.1 expert (`fixtures/make_synth_mxfp4.py --hidden 5120 --inter 2304`), then times the device against the CPU reference and the fast CPU decoder of `wex4/research/fast-cpu-decode/` with `wnr3_vk bench`, for both shader forms alone and in pairs. Last line: `vulkan-decode bench: done` |
| `crash_probe.sh`, `crash.comp`, `crash_min.comp`, `crash_min.c` | the probe of the llvmpipe crash (section "The llvmpipe crash" below): forms of the scale pass that stop a stream at once, a greedy reduction of them, the smallest form written out, and a host of its own for that form (a few small buffers and one dispatch, no WNR3 code). Not a check: it ends with status 0 whatever the driver does. Last line: `crash probe: N of M runs crashed` |

## Run it

It needs `gcc`, `glslangValidator`, the Vulkan headers and loader, and any Vulkan device. It is not part of `scripts/check_all.sh`.

```sh
# Debian, Ubuntu: sudo apt-get install libvulkan-dev glslang-tools mesa-vulkan-drivers vulkan-validationlayers
bash wex4/build_and_test.sh          # optional: leaves wex4/out/tiny_nr3_cz.wex4 and wex4/out/synth.wex4
bash wex4/research/vulkan-decode/run.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
bash wex4/research/vulkan-decode/negative_control.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
```

From the repository root. Without arguments `run.sh` uses the conformance vectors only. Every WNR3 record of a file named on the command line must decode, and `MUTATIONS` copies of each record (default 20) are decoded with one byte changed. The program prints one line per Vulkan device and marks the one it uses with `*`. `WNR3_VK_DEVICE` picks another device than the first: its index, or a part of its name with letters compared without case (`WNR3_VK_DEVICE=NVIDIA`, `WNR3_VK_DEVICE=llvmpipe`). `WNR3_VK_VALIDATE=1` turns the Khronos validation layer on and makes any message of it an error; `WNR3_VK_ROBUST=1` enables `robustBufferAccess`; `WNR3_VK_STAGE=0` or `1` chooses between the shader reading the inputs where the host wrote them and a copy into device-local memory first (default: the copy, when the device has memory the host cannot map). On MSYS2 UCRT64 the packages are `vulkan-headers`, `vulkan-loader`, `glslang`, `mesa` and `vulkan-validation-layers`; `VK_DRIVER_FILES` must name Mesa's `lvp_icd.x86_64.json` when no other driver is installed, `VK_ADD_DRIVER_FILES` adds it next to a GPU driver, and `VK_LAYER_PATH=/ucrt64/bin` lets the loader find the validation layer.

```sh
WNR3_VK_DEVICE=NVIDIA WNR3_VK_VALIDATE=1 bash wex4/research/vulkan-decode/run.sh wex4/out/tiny_nr3_cz.wex4 wex4/out/synth.wex4
WNR3_VK_DEVICE=NVIDIA bash wex4/research/vulkan-decode/bench.sh 21      # the speed; about 5 minutes, 300 MB of disk
```

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

**On a GPU** (2026-11-15, [research/vulkan-gpu-20261115](../../../research/vulkan-gpu-20261115/README.md), one run each): `run.sh` with the validation layer on, `run.sh` with `MUTATIONS=1000` and `negative_control.sh` on an NVIDIA GeForce RTX 3070, driver 621.01, Windows 10 (`WNR3_VK_DEVICE=NVIDIA`), gave the counts of the table above (54,000 mutated copies: the counts of the local run above) and 0 messages of the validation layer. The 5 GiB file and the sanitizer build were not run there. The first run there, with the shader of 2026-11-14, gave other bytes than the reference for the two valid ceil-form records of the selftest: 28 and 50 decrements of the ceil-form inverse lost by the driver (design note, section 6). The speed, from `bench.sh` on 16 synthetic records of the V4.1 geometry, is in section 2 of that page.

## The llvmpipe crash

With the first form of the shader, which stopped a stream at once when its initial state was out of range, llvmpipe of Mesa 25.2.8 with LLVM 20.1.2 (the ubuntu-24.04 runner) crashed when scale stream 0 or 8 of a record stopped that way (issue 21 of the repository). The shader has since decoded such a stream to its end and refused the record there (design note, section 6). `crash_probe.sh` runs record 10 of the selftest (scale stream 0 opens below `L`) with forms of the scale pass and says for each whether the driver crashed, then the smallest form with a host of its own, `crash_min.c`; the CI job `linux-vulkan` runs it after `run.sh`. What its log in run 43742350289 of the repository's Actions shows (2026-11-15; runs 43701920171 and 43741961696 gave the same verdicts for the forms they had), device `llvmpipe (LLVM 20.1.2, 256 bits)`, driver 25.2.8, glslang 15.1.0, SPIRV-Tools v2025.1, 38 of 61 runs crashed:

| Form | Crashed |
|---|---|
| the shader as it is, with the tables and with `-DSEARCH` | no |
| the first form (return at the state check, binary search): record 10; with the validation layer; with `robustBufferAccess`; all twelve records | yes, each time; 0 messages of the validation layer before the crash |
| the first form with the scale table instead of the search | no |
| no return, the symbol count set to 0 | yes |
| `crash.comp` (the first form without the stream hash), then a greedy reduction: one switch after the other turned off for as long as the driver still crashes | crashes with 9 of its 12 parts taken away (the stop on the data, the error bit, the stores, the renormalisation, the state read from the record, the blob found through the plan, the bound on the stream count, the matrix from the plan, the stream's own symbol count); taking away the symbol count from the plan, the nested search or the reads of the blob stops it |
| that smallest form, with the validation layer and with `robustBufferAccess`, and on record 9 | yes |
| the same with another invocation returning at once instead of 0: 1, 2, 4, 7, 9, 15 / 8, 16, 24 | no / yes |
| `crash_min.comp` (the smallest form written out), on record 10 and on record 0, with the validation layer and with `robustBufferAccess`; with invocation 8 returning; with whole words read instead of 16-bit values; with a next state that is the state times a constant plus the value found | yes |
| `crash_min.comp` with invocation 1 returning; with the constant 256 as the loop count; with a next state that reads nothing | no |
| `crash_min.comp` with `crash_min.c`: no record, no WNR3 code; three small buffers the shader reads and writes (a CDF of 257 even steps, a loop count of 256, one word), one workgroup of 32, the descriptor set layout of `wnr3_vk.c` (bindings 0 to 5; the shader declares 1, 2 and 5); also with the validation layer, with `robustBufferAccess`, with 2 workgroups, without the push constant range; with invocation 8 returning; with whole words read | yes |
| the same with invocation 1 returning; with the constant loop count | no |
| `crash_min.c` with other set layouts: bindings 0, 1, 2, 5 / bindings 1, 2, 5 (also without the push constant range); 1, 2, 3, 5; 1, 2, 4, 5; 1 to 5 | yes / no |

`spirv-val --target-env vulkan1.0` accepts every module the probe builds, the first form included. In every gdb stop the faulting instruction loads 4 bytes at offset 8 of an address held in a register, and where gdb printed that register it held 0 (`rsi` in the first form and `rdi` in the smallest form of `crash.comp` in run 43701920171, `rdi` in one form of `crash_min.c` in run 43742350289; the others used `r9`, which the probe does not print). In the code of run 43742350289 (a runner whose generated code has AVX-512 instructions) the next instructions load 8 bytes at offset 0 of the same address and use the first value, divided by 4, to bound the lane indices of a masked gather from the second. **What was established:** the first form was valid SPIR-V (`spirv-val`), used the API without a message of the validation layer, and its crash does not depend on out-of-bounds accesses (`robustBufferAccess` changes nothing; `crash_min.comp` reads only in bounds). The crash is in the driver. It needs, in every form that was run: an invocation that is the first of a group of 8 (0, 8, 16 or 24: the driver's 256 bits are 8 invocations side by side) to have left before a loop whose count is read from a buffer, a search in another buffer inside that loop, and a binding 0 in the descriptor set layout, that is a set in which the buffer of the search (binding 1) is not the first binding; a layout with binding 0 is valid whether the shader declares it or not. Then the generated code loads through a null pointer what looks like the address and the byte size of a buffer for a bounds-checked load; that reading of the machine code was not checked against the Mesa sources. `crash_min.comp` with `crash_min.c` (and `CRASH_MIN_BINDINGS`, `CRASH_MIN_PUSH` for the set-ups above) is the smallest reproducer found: a shader of about 30 lines of code, three buffers with fixed contents, one dispatch. **Not established:** the cause inside Mesa or LLVM, whether a layout with binding 0 and another buffer for the search crashes, and whether a newer Mesa on Linux still crashes. Mesa 26.2.6 with LLVM 23.1.4 (MSYS2, Windows) and the NVIDIA driver of the GPU run ran all 50 runs of the probe without a crash ([research/vulkan-gpu-20261115](../../../research/vulkan-gpu-20261115/README.md)). No report was sent to Mesa. To reproduce: `bash run.sh`, then `bash crash_probe.sh` (needs glslang; `spirv-val` and gdb are used when installed); the reproducer alone: `gcc -std=c99 crash_min.c -o crash_min -lvulkan`, `glslangValidator -V crash_min.comp -o min.spv`, `./crash_min min.spv`.

## Limits

- **One GPU.** Besides llvmpipe, one GPU was run: an NVIDIA GeForce RTX 3070 with driver 621.01 on Windows 10 ([research/vulkan-gpu-20261115](../../../research/vulkan-gpu-20261115/README.md)). Its first run found decrements of the ceil-form inverse lost by the driver, worked around in the shader (design note, section 6). No Intel, AMD, Apple or mobile driver, no other NVIDIA driver and no NVIDIA driver on Linux was tried.
- **The speed number is on synthetic records.** 16 records of the V4.1 geometry whose bytes are random draws from a skewed distribution (flag byte 0x00: no canonical zero, no ceil form), on one GPU, against a CPU that other jobs were loading. Not measured: real records, another GPU, a batch whose upload overlaps the decode of the one before.
- **No real model.** The containers are the tiny-model fixture, a synthetic one and a conformance vector: 54 records in all, flag bytes 0x08 and 0x00; and, in the bench only, 16 synthetic records of 18,800,640 decoded bytes each (1,105,920 blocks, 19,440 streams), flag byte 0x00. No container of a real model was on the machine of the GPU run. A record in ceil form was decoded only in `selftest` (synthetic data and a synthetic model). The batch cut was exercised only with small records and a small limit, and the 64-bit offsets only with the file of `big_file.py`.
- **A driver crash was found, worked around and reduced, not explained** (section "The llvmpipe crash"). The same may exist elsewhere in another shape.
- **One check of the shader is not pinned by a test:** the stop at the end of a stream inside the renormalisation loop. Without it a damaged stream reads on into the next one and is refused by the end-of-stream check, so no verdict changes (one local run with the line removed: `run.sh` passed).
- **No commit step and no engine hook.** The decoded bytes stay in a device buffer; nothing copies them to an engine's buffers. The prototype reads them back only to compare them.
- **WNR3 only.** A WNR2 record (codec 4) is not handled.
- **Not for untrusted files** ([SECURITY.md](../../../SECURITY.md)). The shader does not bound-check; it relies on the host checks. No fuzzer drives the host program, and the mutation mode changes one byte per copy. The sanitizer build runs only in the CI job started by hand.
