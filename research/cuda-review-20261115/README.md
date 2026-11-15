# The CUDA WNR3 decoder against WNR3-SPEC (2026-11-15)

The review that [docs/HELP-WANTED.md](../../docs/HELP-WANTED.md) section 2 listed as "the GPU decoder against the text", and the runs of the harness that section 3 asked for ([wex4/fuzz/fuzz_wnr3_cuda.cpp](../../wex4/fuzz/README.md#the-cuda-decoder-on-the-cpu)).

## Question

Does the CUDA WNR3 decoder (`c/wex_cuda.cuh`, `c/tools/wex_cuda_bridge.cu`) make every check that [WNR3-SPEC](../../docs/WNR3-SPEC.md) asks of a decoder, the way the CPU reference makes it (`wex_nr2_decode` of `c/wex_neural_rans.h`, `wex_decode_nr2_record` of `c/wex_format.h`)? And does it give the same verdict and the same bytes on the WNR3 records of the conformance vectors and on mutated copies of them?

## Setup

- Tree: this repository at the commit that adds this folder. No file under `c/` was changed. Line numbers below are those of `c/` and of `docs/WNR3-SPEC.md` at commit `1af6c0d`, from which the branch of this folder starts.
- Reading: the plan (`wex_cuda_plan_nr2`), the scale kernel, the weight kernel, the commit kernel and the four WNR3 entry points of the bridge, line by line, against WNR3-SPEC sections 1 to 4 and errata 1, 9, 10 and 11, and against the CPU reference.
- Running: no GPU and no CUDA toolkit. There was none to use: no `nvcc` on the path, no `CUDA_PATH`, and none was installed (its licence). The machine has an NVIDIA GeForce RTX 3070, which was not used. The CUDA source ran as host C++ through the emulation of `wex4/fuzz/cuda_host.h`, built twice (an `__CUDA_ARCH__` 860 build with the weight lookup 1, and a 750 build with lookup 0). Each record went through the four decode calls of the bridge and through the kernels alone with both lookups, against `wex_decode_nr2_record`.
- Machine: Windows 10 Pro, 8 logical CPUs, MSYS2 UCRT64 g++ 16.2, Python 3.14. Other builds ran on the machine at the same time: the wall times below are not measurements.
- Command: `bash research/cuda-review-20261115/run.sh` from the repository root, once. It runs `wex4/fuzz/run.sh --replay`, then the harness without mutator rounds on the files of the vectors and of `impl/go-wex4/testdata` (`vectors.log`), with 100 rounds of the custom mutator on every seed (`mutations.log`), and `wex4/fuzz/cuda_negative.py` (`negative.log`).

## Result

No difference of verdict or of decoded bytes between the GPU path and the CPU reference was found, by reading or by running. Every check that WNR3-SPEC asks of a decoder is made on the GPU path; the table below gives each with its lines. Six differences of another kind are listed after it. None was changed: they are not wrong against the text, except the fifth, which the text does not mention.

| Run (log) | Records | Result |
|---|---|---|
| the 111 files of `wex4/vectors/MANIFEST.tsv`, no mutation (`vectors.log`) | 122: the 15 own; the 14 WNR3 records of the manifest, in 7 of the 18 files that open as WEX4; the 93 files that do not open | 20 accepted by the CPU and by every GPU path with the same bytes (7 own, 13 of the manifest; 5 different decoded records), 9 refused on the device (8 own, built to be refused, and the record with a damaged stream of `wnr3-stream-damaged-record-hash-fixed.wex4`), 93 refused by the host checks; no failed oracle |
| the 4 files of `impl/go-wex4/testdata` (`vectors.log`) | 17: the 15 own and the 2 of `ceil-form.wex4` | 9 accepted with the same bytes (the 2 ceil-form records among them), 8 refused on the device (the own records built to be refused); no failed oracle |
| every seed with 100 inputs of the custom mutator each, 11,817 inputs (`mutations.log`) | 11,744 | 5,873 accepted with the same bytes (5,217 different decoded records), 2,033 refused on the device (error bit 1 in 1,523, bit 2 in 747, bit 4 in 37, bit 8 in 505), 3,741 refused by the host checks, 97 naming a model that is not registered (status 1 on every path); no failed oracle |
| negative control, 23 copies of the CUDA source with one check, wait or barrier broken (`negative.log`) | | 23 of 23 noticed |

Every record above went through 2 builds x (4 calls of the bridge + the kernels alone with 2 lookups). "Records" counts, in each run, the 15 records that the harness builds at start (`selftest` in `fuzz_wnr3_cuda.cpp`; "own" above). A file of the manifest that does not open as WEX4 (header or index refused) goes in as the bytes of one record against the layout of `tiny2-cz-upgraded.wex4`, as in `fuzz_wnr3_record`, and the host checks refuse it. The 14 WNR3 records of the manifest all have the CZ flag and no CEIL flag: they are the 2 records of `tiny2-cz-upgraded.wex4` and copies of them in 6 invalid files, one copy with a damaged stream. The 2 records of `impl/go-wex4/testdata/ceil-form.wex4` have the CEIL flags 7 and 5. The 100 mutator rounds took 390 s of wall time while other builds ran on the machine (one run).

### The checks, side by side

CPU means `c/wex_neural_rans.h` unless another file is named; GPU means `c/wex_cuda.cuh`; bridge means `c/tools/wex_cuda_bridge.cu`.

| Rule of WNR3-SPEC | CPU | GPU path | Same? |
|---|---|---|---|
| Entry: codec 5 (or 4), valid sizes, `total_bytes` below the aligned raw size (2.2; 4.7 step 1) | `c/wex_format.h:521-524` | GPU 426-428; bridge 258-260 (to host), 323-324 (to device), 457-458 (batch) | yes |
| Parse: magic of the codec, chunks, every length >= 4, sums, `total == align4096(used)`, zero padding, flag bits 4-7 zero (3) | `c/wex_format.h:526` calls `wex_nrx_parse` (287-337) | GPU 430 calls the same function, with model id 0 | yes, the same function |
| Model id registered (3; 4.7 step 2) | the id is passed to the parser (`c/wex_format.h:526`) and compared again at 473 | bridge 183-184: an id that is not registered gives status 1, not -1 | verdict differs by design: difference 1 |
| Initial state in `[L, 256L)` (1; 4.5; erratum 10) | 379 | GPU 590 (scales), 708 (weights): error bit 1 | yes |
| A stream that needs a byte it does not have is invalid (erratum 10) | 367, at once | no test per byte: the branch-free renormalization (GPU 549-555) reads on, and the position past the end fails the end test below | same verdict: difference 2 |
| Every stream ends exactly at its u16 length (1) | 488, 515 (`ptr != end`) | GPU 635, 767 (`s.pos != end`): bit 1 | yes |
| Final state `L + h23`, ident `{model_id, owner, g, bytes 24..27}`, data the stream's coded bytes, length `16 + n` (4.5) | 488 and 515 through `wex_nr3_stream_x0` (231-233) | scales: ident GPU 594-596, data 612-615 (32 at a time) and 621-624 (byte by byte), tail 632, length 633, test 636; weights: ident 713-715, 16 bytes per block 749-750, length 766, test 768: bit 2 | yes |
| Contexts from the coded scale and the coded nibbles: `srel = clamp(cs - anchor, -3, 3) + 3`, previous and maximum "none" at every block start (4.1; erratum 9) | 503-505, 511-512 | GPU 719-722, 744-746 (maximum starts at 0 with the context of "none" already chosen, then `max`) | yes |
| Canonical zero: no symbol 8 in a CZ record (4.4) | 508, at the symbol | GPU 751-754, after the block's 32 symbols: bit 8 | same verdict: difference 3 |
| Ceil form, only in a matrix with its CEIL bit: a block of coded maximum 5 at coded scale 0 is invalid, else D on its nibbles and scale - 1 (4.2; erratum 1) | 446-461, run at 520 after every stream decoded | GPU 711, 755-760, in the weight kernel, per block, in place: bit 4 | yes, same bytes |
| Hashes and contexts in the coded domain; the inverse never feeds back (4.7) | 520 comes after every hash | GPU 749-750 hash the block before the inverse at 755-760; the next block's scale is read at 719, before the write at 759 | yes |
| Any failure: status -1, destination untouched (4.7) | `c/wex_format.h:528-535`: decode into scratch, copy on success | to host: bridge 250 and 268 (copy after a clean decode); to device and batch to device: the commit kernel copies only when the error word is 0 (GPU 849; bridge 236, 522); batch to host: no commit, bridge 558-560 | differs for the batch to host: difference 5 |
| Error reporting: 1 structure, 2 hash, 4 ceil, 8 CZ; the helper returns -1 and prints once per process for 2, 4, 8 (4.8) | -1 | bits as above; bridge 190-191 (the message), 250 and 536 (the status) | yes; the bits are not always the cause: difference 4 |

The owner (record bytes 28..31) and the CZ bit against header flag 64 are the caller's checks on both paths (`wex_nr3_owner_ok`, `c/wex_format.h:189-194`; WNR3-SPEC 4.7 step 1, erratum 2); both paths put the owner into every stream's hash. The anchor bytes are checked by neither (erratum 11).

### Differences that are not of verdict or bytes

1. **A model that is not registered: status 1, not -1** (bridge 183-184). The GPU path parses with model id 0 (GPU 430) and then asks its registry. Status 1 means "helper unavailable" (WNR3-SPEC 4.8 and erratum 4): `c/wex_decode.c:339-350` then decodes the record with the CPU decoder, which refuses it. The harness requires 1 exactly for a record that parses and names an id it did not register.
2. **A stream is decoded past its end before it is refused.** The CPU stops at the first byte a stream does not have. The GPU renormalizes without a branch (GPU 549-555), reads ring slots past the stream's last word, which hold stale bytes (GPU 507-509 say so), and refuses the stream at its end because the position has passed it (635, 767). The position never goes back, so the verdict is the same. The emulation fills shared and device memory with 0xA5 for that reason, also under MemorySanitizer, which would report those reads.
3. **Canonical zero is checked per block** (GPU 751-754), after its 32 symbols, where the CPU checks each symbol (508). A block never spans two streams, so the verdict is the same.
4. **The error bits do not always name the cause.** After an error the other streams run on; a scale stream that fails leaves its scales unwritten, and the weight kernel then decodes with what the buffer held before. A scale stream opened below L gives the error word 0x9 (1 and 8) on both builds in the harness's own record: bit 8 comes from weights decoded with stale scales. The bridge then prints its "failed its integrity check (error 0x9)" line (bridge 190-191) for a structural defect. The status, -1, is right.
5. **`wex_cuda_decode_batch_to_host` can write its destination for a record it refuses.** WNR3-SPEC 4.7 says "Any failure: status -1, destination untouched ... GPU commit kernel copies only when the error word is 0". This call, added after the text (bit 32, section 2.1), has no commit kernel: it copies every record back and its comment says that the destination of a refused record may hold partial or wrong bytes (bridge 558-560). Its one caller decodes such a record again alone (`c/qwen38_wex.h:562-564`). The text does not mention the exception; a sentence in 4.7 is open (HELP-WANTED section 2).
6. **Kernel A may read 3 bytes that are still being copied.** The bridge starts the scale kernel when the head of the record (up to the end of the scale payload) is on the device (bridge 224-229). The kernel's ring loads whole words up to the word that holds a stream's last byte (GPU 515), so for the last scale stream up to 3 bytes of the weight payload, which the copy stream may still be writing. A stream that decodes right never uses them; one that reads them is refused at its end. The emulation runs a copy whole, so it cannot show this.

### What else the runs show

- The negative control: each of the 23 broken copies listed in `wex4/fuzz/cuda_negative.py` (a check of 4.2, 4.4, 4.5 or of the stream bounds removed, the wait of `cp.async` one group short, the ring refilled by 8 words instead of 16, a barrier removed, a wait between the bridge's two streams removed, the commit kernel without its test, ...) makes the replay fail (`negative.log`). Before the record at 14 bits per symbol was added to the harness, a first version of the control did not notice the `cp.async` wait one group short. By the refill rule (GPU 525-536: at every 32 symbols the ring is topped up to 40 words ahead of the read position, at most 16 words at a time, and only the newest group may still be in flight), the short wait leaves the two newest groups in flight, and a lane reads one of their words only when three blocks in a row take about 39 words (156 bytes) or more; the records of the vectors take far less. The record at 14 bits per symbol takes 14 words per block. By the same rule the real wait is safe at any load: a lane would read a word in flight only if two blocks in a row took about 39 words, and a block takes at most 16 (2 bytes per symbol, GPU 547-555); a refill overwrites the slot of a word 64 words back, which was read already.
- The two builds give the same verdict and the same bytes on every record. So the code for `__CUDA_ARCH__` below 800 (no `cp.async`, the scale table in global memory; STATUS.md, "Pre-Ampere fallback", says it has not run on such a GPU) ran too, on the CPU.
- The batch to host was run the way `c/qwen38_wex.h` calls it, each record in the tail of its own slab and decoded over it; the bytes were right in every accepted case.

## Limits

- This is not a run on a GPU. nvcc did not compile anything; the device code that nvcc makes, the GPU's memory model, warps, two blocks or two operations at the same time, timing and speed are outside it. `c/tests/test_wex_nr3_gpu.cu` (nvcc, a GPU) was not run.
- Only the WNR3 path was reviewed and run. The WNR2 path of the same kernels (`plan.nr3` 0), the WNR1 and palette kernels and their bridge paths were not: no seed holds such a record.
- The vector records are few: 14 WNR3 records in 7 files, copies of 2 records, none with a CEIL flag. The ceil inverse on recorded data ran on the 2 records of `impl/go-wex4/testdata/ceil-form.wex4` and on the harness's own records.
- No coverage-guided run was made here (the MSYS2 clang has no libFuzzer); the mutations of `mutations.log` are the custom mutator's, 100 per seed, with a deterministic stand-in for libFuzzer's own. The Fuzz workflow runs the harness under libFuzzer for 60 s on Linux.
- The negative control shows that the harness notices 23 kinds of broken decoder, not that it notices every kind.
- One run of each.

## Raw files

- `run.sh`: the command of this folder.
- `vectors.log`, `mutations.log`, `negative.log`: its output. The bridge prints its once-per-process line once per build; `run.sh` leaves it out.
