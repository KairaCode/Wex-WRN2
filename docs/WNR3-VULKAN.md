# WNR3 decode on Vulkan compute: design note

**Status (2026-11-14):** design note with a prototype. The prototype in [`wex4/research/vulkan-decode/`](../wex4/research/vulkan-decode/README.md) decodes WNR3 records with a Vulkan compute shader and gives the bytes and the verdicts of the CPU reference `wex_nr2_decode` (`c/wex_neural_rans.h`) on every record it was run on. It was run only on Mesa lavapipe, a Vulkan device that executes on the CPU. **No run on a GPU exists, so there is no speed number, and nothing is hooked into the engine.** Section 11 lists what is open.

This page answers the "start with" question of [HELP-WANTED](HELP-WANTED.md) section 6: how streams map to workgroups, where the model tables live, how records are batched. It names what the prototype does today ("prototype") and what a port into an engine still has to decide ("port"). The record format is [WNR3-SPEC](WNR3-SPEC.md) and is not changed by anything here.

## 1. What the decoder is given

One record is one routed expert (WNR3-SPEC sections 1 and 3):

- a 32-byte header: model id, payload sizes, the chunk sizes `Cw` (weights, a power of two in 512..16384) and `Cs` (scales, 256..16384), three anchor bytes, the flag byte (bits 0-2 ceil form of matrix m, bit 3 canonical zero) and the owner;
- `S + W` u16 stream lengths, scale streams first, in the global stream order g;
- the scale payload and the weight payload: one rANS stream after the other, each at least 4 bytes long, each ending exactly at its length.

Every stream is independent of every other stream of the record, with one exception: a weight stream needs the decoded scales of its own blocks for its contexts. So the decode is two passes, **A** (all scale streams) and **B** (all weight streams), and inside a pass every stream can run at once. A stream is short: at most `Cs` scale bytes or `Cw` nibbles (`Cw / 32` blocks). For the sizes of DeepSeek V4.1 (raw record 18,800,640 B, `Cw` = 2048, `Cs` = 512) a record has 2,160 scale streams and 17,280 weight streams; these two counts are computed from the record size and the chunk sizes, not read from a real record here.

The decoded slab has six regions: scales of matrix 0, 1, 2, then packed 4-bit weights of matrix 0, 1, 2.

## 2. Streams, invocations and workgroups

**One invocation decodes one stream.** There is no communication between invocations, no shared memory and no barrier inside a pass.

- The shader `wnr3.comp` is compiled twice from one source: `scales.spv` (pass A) and `weights.spv` (pass B, `-DWEIGHTS`). Each is one compute pipeline; both use the same descriptor set layout.
- `local_size_x = 32`, as the CUDA kernels use 32 lanes per block. A dispatch for one record has `ceil(streams / 32)` workgroups in x, 1 in y and z, where `streams` is `S` in pass A and `W` in pass B. Invocation `j = gl_GlobalInvocationID.x` is stream j of the pass; invocations with `j >= streams` return at once.
- The record is named by a 4-byte push constant (its index in the batch). Everything else the invocation needs comes from the record's plan (section 3).
- The matrix m of stream j comes from three comparisons with the first stream of each matrix (`P_SCALE_BASE`, `P_WEIGHT_BASE`), so streams are numbered across the three matrices without gaps. The CUDA kernels instead start every matrix on a new block, because a block loads the tables of one matrix into shared memory.
- Global stream index: `g = j` in pass A and `g = S + j` in pass B. g selects the stream's byte range in the offsets buffer and is one of the four identity words of the stream hash.
- Limits: `maxComputeWorkGroupCount[0]` is at least 65,535 on every Vulkan device, so one dispatch covers at least 2,097,120 streams; the host checks the device's value. `maxComputeWorkGroupInvocations` is at least 128.

What one invocation does, pass A: read the 4-byte big-endian initial state, note whether it is in `[L, 256 L)` (section 6 says why a state out of range is reported at the end and not at once), decode `count = min(Cs, scales of m - j * Cs)` symbols with the scale CDF of m, store them, fold them into the stream hash, then require that the stream ended exactly at its length and that the final state is `L + h23`.

Pass B: the same for `count / 32` blocks. For each block it reads the block's coded scale (written by pass A), forms `srel`, decodes 32 symbols with the context `(srel * 9 + previous) * 9 + maximum`, packs them into four words, and then applies the three WNR3 rules of section 6 before it stores the four words.

## 3. Buffers

Six storage buffers in one descriptor set (set 0), all declared as arrays of `uint` (std430). Byte values are read and written through shifts and masks of 32-bit words, so the shader needs neither 8-bit nor 16-bit storage types.

| binding | name | access | content | host memory |
|---|---|---|---|---|
| 0 | `rec` | read | the stored bytes of every record of the batch, one after the other. A record size is a multiple of 4096, so each record starts on a word | host-visible, written by the host |
| 1 | `tab` | read | the WNM2 model blobs the batch uses, 59,408 B each (a multiple of 4), unchanged | host-visible |
| 2 | `plan` | read | 32 words per record (table below) | host-visible |
| 3 | `offs` | read | per record `S + W + 1` words: the absolute byte offset in `rec` of every stream, in g order, and the end of the last one | host-visible |
| 4 | `outw` | read and write | the decoded bytes: six regions per record, each starting on a 16-byte boundary | **not mapped by the host**: device-local memory that is not host-visible where the device has such a type |
| 5 | `err` | read and write | one error word per record | host-visible |

All multi-byte values are little-endian words. The host writes every plan and offset word byte by byte, so the layout does not depend on the host's byte order (a big-endian host was not run).

The plan of a record (word index, count, meaning):

| word | n | meaning |
|---|---|---|
| 0 | 3 | byte offset in `outw` of the scales of matrix m (multiple of 16) |
| 3 | 3 | byte offset in `outw` of the packed weights of matrix m (multiple of 16) |
| 6 | 3 | scales of matrix m |
| 9 | 3 | 4-bit weights of matrix m (a multiple of 32) |
| 12 | 1 | `Cs` |
| 13 | 1 | `Cw` |
| 14 | 1 | model id (record bytes 4..7) |
| 15 | 1 | owner (record bytes 28..31) |
| 16 | 1 | record bytes 24..27 as one word: the three anchors and the flag byte |
| 17 | 1 | word index in `tab` of the record's model blob |
| 18 | 1 | index in `offs` of the record's stream 0 |
| 19 | 4 | first scale stream of matrix 0, 1, 2; then `S` |
| 23 | 4 | first weight stream of matrix 0, 1, 2, counted from 0; then `W` |
| 27 | 5 | zero, reserved |

The plan is the Vulkan form of `WexCudaNr2Plan` (`c/wex_cuda.cuh`), which CUDA passes to the kernel by value. A push constant block of that size would also fit (the guaranteed minimum is 128 bytes), but a buffer lets one command buffer hold a whole batch with a 4-byte push constant per dispatch.

**Why every write is race-free.** A storage buffer has no byte stores, so an invocation that writes a byte rewrites its whole word. Two invocations must therefore never own bytes of the same word:

- Scales: stream j of matrix m owns the scale bytes `[j * Cs, j * Cs + count)` of a region that starts on a 16-byte boundary. `Cs` is a multiple of 4, so every stream starts on a word; only the last stream of a matrix can end inside a word, and the rest of that word is padding before the next region.
- Weights: a block is 16 bytes, four whole words.
- Scale fix-up of the ceil-form inverse (section 6): weight stream j owns the scale bytes `[j * Cw / 32, ...)`, and `Cw / 32` is at least 16, so again whole words, except the padded tail of the matrix. Pass B is the only writer of scales while it runs.

## 4. Arithmetic: 32 bits are enough

The shader uses only `uint` (and one `int` for `scale - anchor`). No 64-bit integers (`shaderInt64`), no 16-bit or 8-bit types, no extension, no optional feature: GLSL 4.50 compiled to SPIR-V 1.0, Vulkan 1.0 core.

- rANS step: `state = f * (state >> 14) + slot - start`. The state is below `2^31` (`256 L`), so `state >> 14` is below `2^17`; a frequency f is at most `2^14`; `slot - start` is below f. The result is below `2^14 * 2^17 = 2^31`: no overflow, the same bound the C reference relies on.
- Renormalisation: `while (state < L) state = state << 8 | next byte`. The state is below `2^23` before the shift, so below `2^31` after it.
- The one exception: a stream whose initial state is outside `[L, 256 L)` is decoded all the same (section 6) and its record is refused at the end. For such a state the product can wrap modulo `2^32`. That changes no index: the slot is 14 bits of the state, the symbol search stays inside the CDF row, the read position never passes the end of the stream, and the output position depends only on the symbol count.
- MurmurHash3 needs 32-bit multiplication modulo `2^32` and rotations: GLSL `uint` arithmetic wraps, which is exactly that.
- The ceil-form inverse and the nibble-8 test work on 8 nibbles per word (the SWAR forms of `wex_cuda_ceil_dbl8` and `wex_cuda_has_nibble8`).
- Byte offsets into `rec` and `outw` are 32-bit, so one bound range is limited to 4 GiB; the host also checks `maxStorageBufferRange` (guaranteed minimum 2^27 bytes, 128 MiB). See section 7.

## 5. Model tables

**Prototype.** The validated WNM2 blob is uploaded as it is, and the shader does the reference binary search on its u16 CDF values: 4 probes per weight symbol (17 values per context row, at `32 + row * 34` bytes of the blob) and 8 per scale symbol (257 values per matrix, after the `3 * 567` weight rows). A u16 is read as half of a `uint`; the blob header is 32 bytes and every row offset is even, so no value straddles a word. No derived table exists, so there is nothing to build per device and nothing that can disagree with the blob. One buffer can hold several models; plan word 17 selects the record's one.

The host must validate the blob first (`wex_nr2_model_open`: magic, version, id, every CDF starts at 0, ends at `2^14` and is strictly increasing). The shader relies on that for termination of the search and for the range of every frequency.

**Port, not built and not measured.** The binary search is the simplest correct form, not the fastest. The CUDA helper builds three derived tables once per model and device, and a port that wants its speed has the same choices:

| CUDA table | size | use | Vulkan equivalent |
|---|---|---|---|
| `weight_rows`: each context row as 16 u16 in two `uint4` | 3 x 567 x 32 B | compare all 16 CDF values with the slot at once (SWAR), no loop; the rows of one matrix are copied into shared memory | a `uvec4` storage buffer read directly, or a `shared` array. One matrix is 18,144 B, above the guaranteed `maxComputeSharedMemorySize` of 16,384 B, so the shader must either read the limit or leave the rows in the buffer |
| `scale_direct` + `scale_symbol`: per slot the frequency, the bias and the symbol | 3 x 16,384 x 5 B | one lookup per scale symbol instead of 8 probes | the same two tables as storage buffers; they are built by a small compute pass or on the host, from the validated blob |
| `weight_buckets` (lookup 0, not the default) | 3 x 567 x 1024 x 4 B | coarse bucket, then a walk | not recommended: CUDA measured it slower than the shared-memory rows |

Where tables live in a port: one device buffer per (model, device), created when the engine registers the model (the place where `wex_cuda_register_model` is called today), kept until the model is dropped, and bound read-only to every decode. Each derived table must be checked against the binary search on all `2^14` slots when it is built or in a test, because a wrong table decodes wrong bytes that only the stream hash would catch.

The CUDA kernels also prefetch the payload into a per-lane ring in shared memory (`cp.async` on Ampere and later) and renormalise without a branch. Vulkan has no asynchronous copy; the prototype reads the record words directly and uses the plain loop.

## 6. Hash, canonical zero, ceil form, errors

Order as in WNR3-SPEC section 4.7; everything that is checked is checked on the **coded** bytes.

- **Stream hash.** At stream open the four identity words `{model id, owner, g, bytes 24..27}` are folded into MurmurHash3 with seed `0x33524E57`. Pass A folds every four decoded scale bytes as one little-endian word and the last 1 to 3 bytes as the tail; pass B folds the four words of every block before the inverse. Length is `16 + count` (scales) or `16 + 16 * blocks` (weights). The final state must be `L + (hash & 0x7FFFFF)`.
- **Canonical zero.** In a record with flag bit 3, a block that holds a nibble 8 is an error. Tested on the four words of the block.
- **Ceil-form inverse.** In a matrix with its ceil bit set, a block whose coded maximum magnitude is 5 gets `D` on every magnitude (0, 1, 2, 3, 4, 5 become 0, 2, 4, 5, 6, 7; sign bits kept) and its scale byte is decremented **in place** in `outw`, after the block's scale was read for its context. A coded scale 0 in such a block is an error. This is erratum 1 of WNR3-SPEC, the way the CUDA weight kernel does it; section 3 shows why the in-place write does not race.
- **Errors.** `atomicOr(err[record], bit)` with the bits of `c/wex_cuda.cuh`: 1 structure (initial state out of range, stream shorter or longer than its length), 2 hash, 4 ceil scale 0, 8 canonical zero. A stream that runs out of bytes, finds a nibble 8 or a coded scale 0 stops there; a stream with an initial state out of range runs to its end and sets bit 1 then; other streams of the record run on. A nonzero word means the record is refused, whatever the bits. The bits are diagnostic: on a damaged record the set of bits depends on which streams notice first, only "zero or not" is compared with the reference. The exact word is compared only for the synthetic records of the selftest that are built to fail one check (section 9).

**A driver crash that shaped the shader (2026-11-14).** The first form of the shader returned at once when the initial state was out of range. On llvmpipe of Mesa 25.2.8 with LLVM 20.1.2 (the ubuntu-24.04 CI runner) that form crashed the driver, a read at address 8 inside the generated code, whenever the stream was scale stream 0 or 8 of its record; scale streams 1, 7 and 14 did not crash it (weight streams were not tried). A form with no return that set the symbol count to 0 crashed the same way. Two forms that let the stream decode (from the state as it is, or from `L`) and report at the end did not crash. Mesa llvmpipe with LLVM 23.1.4 on Windows ran the first and the last form without a crash. The device name on the runner says 256 bits, which would be 8 invocations side by side, so what the crashing forms seem to share is that the first of 8 never enters the symbol loop; the cause inside the driver was not looked for, and no driver bug was reported. The shader therefore decodes such a stream to its end. The selftest keeps a record whose scale stream 0 opens below `L`: with the first form of the shader that record crashed the CI job. The CI runs of the first version were green although the mutation test could have drawn this case (1,080 copies, one seed); it apparently did not. Whether a GPU driver has a similar corner is unknown.

**What the host must do before the device sees a record** (the prototype's `container_job`, the same checks as `wex4_decode_record` in `c/wex4.h`): the WEX4 index entry gives the layout; `wex_nrx_parse` checks magic, flags, chunk sizes, stream counts, every length at least 4, the length sums, the padding; the owner in the record equals the owner in the index; the canonical-zero flag of the record equals the value-exact flag of its index entry; the model blob opens and has the id the record names. The shader trusts every index it forms from the plan, the offsets and the model. A Vulkan device does not bound-check buffer accesses unless `robustBufferAccess` is enabled, so these checks are part of the safety of the decode, not an optimisation.

## 7. Batching

**Prototype.** The host program reads a file record by record (64-bit offsets), not as a whole. A batch is cut at 64 records, or earlier when the next record would take the stored bytes or the decoded bytes of the batch above the batch limit: `WNR3_VK_BATCH_BYTES` (default 2^28), lowered to the device's `maxStorageBufferRange`. One batch is one command buffer, one submit, one wait:

1. clear `err` and `outw` (`vkCmdFillBuffer`; clearing `outw` is not needed for the result, it gives the comparison a known start);
2. bind pipeline A; for each record: push constant = record index, `vkCmdDispatch(ceil(S / 32), 1, 1)`;
3. a buffer memory barrier on `outw` (shader write to shader read and write): pass B reads what pass A wrote;
4. bind pipeline B; for each record: push constant, `vkCmdDispatch(ceil(W / 32), 1, 1)`;
5. a barrier on `outw` (shader write to transfer read), then, in the prototype only, a copy to a host-visible buffer for the comparison.

Running A for every record before B for any record needs one barrier per batch instead of one per record, and lets the device overlap the dispatches of a pass. The buffers are created per batch and sized from it; a port would keep them and reuse them.

**Limits the host checks.** The total of record bytes and of decoded bytes of a batch must fit `maxStorageBufferRange` and 32 bits; the batch cut above keeps a batch inside them, and a single record above the device's range stops the program with a message. On a device with the guaranteed minimum of 128 MiB (lavapipe reports exactly that) the cut gives 7 V4.1 records per batch: 18,800,640 decoded bytes each, a number computed from the record size, since no V4.1 record was run. The cut itself was exercised only on small records, with the limit set to 256 KiB (the 48 records of the tiny-model container in 3 batches, the 4 of the synthetic one in 4). Desktop drivers usually report 4 GiB, which was not checked on any GPU here. The host holds several times the batch limit in memory while a batch runs: the records, the mapped buffers, and two copies of the decoded bytes for the comparison. A port can bind a range of a larger buffer per record (dynamic storage buffer offsets), which removes the limit at the cost of one bind per record.

**Port, open.** (a) One dispatch for a whole pass, with a prefix sum of stream counts to find the record of an invocation, instead of one dispatch per record: fewer commands, more table reads; whether it is faster is unmeasured. (b) The batch size and the number of batches in flight (the CUDA helper pipelines a batch over 8 lanes with separate upload and decode streams, `WEX_CUDA_BATCH_LANES`). (c) The upload: the prototype memcpys records into a host-visible storage buffer, which on a discrete GPU is system memory read over the bus by the shader; a staging buffer and `vkCmdCopyBuffer` into a device-local record buffer is the usual alternative. Which one is faster depends on the device and is unmeasured.

## 8. Where the output lands

The decoded bytes are written by the shader into `outw`, which the host never maps: on a device with dedicated memory they never leave VRAM, which is the point of the port. The prototype asks for a memory type that is device-local and not host-visible; lavapipe has no such type, so there the buffer is ordinary memory, and the property "the bytes stay on the device" is true by construction but was not observed on a GPU.

`outw` is the equivalent of the CUDA helper's device raw buffer: a scratch slab. WNR3-SPEC section 4.7 requires that a failed decode leaves the destination untouched, and the CUDA helper meets it with a commit kernel (`wex_cuda_commit_regions`) that copies the six regions to the caller's buffers only when the error word is zero. **The prototype has no commit step.** A port has three ways:

1. a third compute pass that reads `err[record]` and copies the six regions into the engine's expert buffers when it is zero (the CUDA way, no host round trip);
2. the host waits for the batch, reads the error words and records `vkCmdCopyBuffer` for the accepted records (simple, one more submit);
3. decode straight into the destination and publish the expert only when the error word is zero. This saves the copy but the destination is written on a failure, so it is allowed only where nothing can read the destination before it is published.

The six regions are 16-byte aligned in `outw`; the canonical slab has the weights right after the scales with no padding. Which layout the engine's Vulkan buffers want decides between a plain copy and a decode with the plan offsets set to the destination layout.

## 9. Validation strategy

The reference is `wex_nr2_decode` in `c/wex_neural_rans.h`. "Agree" below means: same verdict (accepted or refused), and for an accepted record the same bytes in all six regions.

What `wex4/research/vulkan-decode/run.sh` runs (details and counts in its [README](../wex4/research/vulkan-decode/README.md)):

- **selftest:** twelve synthetic records made with the reference encoder and a synthetic model with uneven frequencies. Five are valid and cover every flag combination that matters (plain, canonical zero, ceil form, both), the smallest and the largest chunk sizes, and scale counts that are not multiples of 4 (hash tail); the test also requires that the ceil-form inverse really changed blocks. Seven must be refused, and the error word of each is compared: coded scale 0 in a max-5 block (bit 4), nibble 8 under canonical zero (bit 8), one changed payload byte (bit 1, bit 2 or both: which stream notices first is not fixed), and four records built so that exactly one check can refuse them. Those four are: the owner word changed after encoding, so every stream decodes and only its hash is wrong (bit 2 alone); a scale stream with one byte after its last symbol (bit 1); a scale stream re-encoded to open with a state below `L` (bit 1); and one re-encoded to open with a state of `256 L` or more (bit 1). A decoder without the end-of-stream check, the lower bound or the upper bound would decode the last three to the reference's bytes with the right hash, which the negative control confirms.
- **decode:** every WNR3 record of the valid conformance vector that has WNR3 records (`wex4/vectors/valid/tiny2-cz-upgraded.wex4`) and of any WEX4 file named on the command line.
- **reject:** the three invalid conformance vectors whose WNR3 data a decoding reader refuses. Two are refused by the host checks, one by the device.
- **mutate:** copies of every record with one byte of its used part changed. The verdict must agree with the reference for every copy.
- **negative control** (`negative_control.sh`): eleven one-line changes of the shader (hash rotation, hash tail, final state check, a wrong hash reported as a structure error, canonical-zero check, ceil scale-0 error, ceil doubling, ceil scale decrement, end-of-stream check, lower bound and upper bound of the initial state); the run must fail on each. A check that passes with a broken shader proves nothing. One check of the shader is not pinned this way: the test inside the renormalisation loop that stops a stream at its end. Without it the stream reads on into the next stream and the end-of-stream check refuses it all the same, so no verdict changes (run once with that line removed: `run.sh` passed); it only bounds how far a damaged stream reads.
- **A file above 4 GiB:** `big_file.py` puts 5 GiB of blobs in front of the records of a small container; the host program must find the records at their 64-bit offsets.
- **Validation layer:** with `WNR3_VK_VALIDATE=1` the Khronos validation layer is on and any message fails the run.

CI: the job `linux-vulkan` of `.github/workflows/ci.yml` runs all of this on Mesa lavapipe on an ubuntu-24.04 runner, and `run.sh` once more with the host program built under AddressSanitizer and UndefinedBehaviorSanitizer (the driver is not instrumented). It is opt-in (runs started by hand only) and not part of `scripts/check_all.sh`, because it needs Vulkan packages.

What this does **not** show: correctness on a GPU driver (different compilers, real concurrency, device-local memory), any speed, and any real model container. The records that were decoded from containers are canonical-zero records with no ceil-form matrix (flag byte 0x08) and plain ones (0x00); ceil form is covered only by the synthetic records.

## 10. Differences from the CUDA kernels

| | CUDA (`c/wex_cuda.cuh`, `c/tools/wex_cuda_bridge.cu`) | Vulkan prototype |
|---|---|---|
| codecs | WNR2 (codec 4) and WNR3 (codec 5) in these kernels; the older codecs in others | WNR3 only |
| lanes | 32 threads per block; a block belongs to one matrix | 32 invocations per workgroup; streams numbered across matrices |
| plan | struct passed by value to the kernel | 32 words in a buffer, record index as push constant |
| weight symbol | SWAR compare of a 16-value row held in shared memory | binary search in the WNM2 blob, 4 probes |
| scale symbol | one lookup in a 16,384-entry direct table | binary search, 8 probes |
| derived tables | built per model and device by three kernels | none |
| payload | per-lane ring in shared memory, asynchronous copies, branch-free renormalisation of 0 to 2 bytes | direct word reads, loop |
| byte stores | `unsigned char` and `uint4` stores | whole `uint` words; ownership of words argued in section 3 |
| output | device raw buffer, then a commit kernel into caller buffers when the error word is 0 | device buffer; no commit step |
| batch | pipelined over 8 lanes, upload and decode on separate streams | one command buffer per batch (at most 64 records and the batch limit of section 7), pass A for all, then pass B for all |
| integer width | 32-bit state; pointers and sizes 64-bit | 32-bit only |
| error word | one per record, `atomicOr`, bits 1, 2, 4, 8 | the same |
| build | nvcc, CUDA toolkit | gcc, glslangValidator, Vulkan headers and loader |
| measured | 0.366 to 0.373 ms per V4.1 WNR3-CZ record in batches of 16 on an RTX 5090 ([WNR3-VERIFY](reports/WNR3-VERIFY.md) section 3; not re-measured here) | no GPU run, no number |

## 11. Open

- **Speed on a GPU.** Nothing was run on a GPU. The "done when" of HELP-WANTED section 6 asks for a speed number against the CPU path on the same machine: open. The table lookups of section 5 are the first thing to try when the number is poor.
- **A real container.** "Byte-identical on every record of a real container" is open: the prototype ran on the conformance vector, on the tiny-model and synthetic containers that `wex4/build_and_test.sh` builds, and on synthetic records. No real expert record, and no real record in ceil form, was decoded. The host program reads a file of any size record by record, but the CPU reference decodes every record too, at about half a second per V4.1 record on one thread (505-590 ms on a Ryzen 9 7950X3D, [WEX4-SPEC](WEX4-SPEC.md) section 11.3; not measured here). By that figure the 7,680 expert records of that model take about an hour of reference decoding, and each mutated copy of a record costs the same again. `MUTATIONS=0` and `WNR3_VK_EVERY=N` (one record in N) keep a first run short.
- **Other drivers.** Only llvmpipe (Mesa lavapipe) was run: Intel, AMD, NVIDIA, Apple (MoltenVK) and mobile drivers were not.
- **Commit step** (section 8) and the destination layout.
- **The hook into the engine.** colibri has a Vulkan backend; nothing here is connected to it. The entry points to mirror are the ones of `c/wex_decode.h` that the engine calls for CUDA (`coli_wex_cuda_decode_to_device`, `coli_wex_cuda_decode_batch_to_device`), with Vulkan buffers and a queue in place of device pointers and a stream.
- **Upload path and batch shape** (section 7).
- **WNR2 records** (codec 4) are not handled; they differ in the final-state rule and have none of the three WNR3 rules.
- **HIP and Metal** ports can reuse the validation of section 9 unchanged: it compares against the CPU reference and does not depend on the API.
