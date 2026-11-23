# impl/go-wex4/testdata/

Four WEX4 files for the Go reader's tests (and, since 2026-11-23, the codec-8 files of the [last section](#codec-8-wnr4-files-2026-11-23), which no test reads yet). The Rust reader reads them too (2026-11-15): its `run.sh` checks the content digest of the first three and that the fourth is refused, and its `differential.py` mutates the first three ([impl/rust-wex4](../../rust-wex4/README.md)), so a change of these files changes what both readers expect. They hold what the conformance vectors did not hold until 2026-11-15: WNR3 records in ceil form, WBP1 records of 4 and 8 planes, the byte 255 in rANS-coded data, WBP1 records whose size is not a multiple of the plane count, and a WBP1 record with one more page of zeros than it needs, which a reader must refuse ([SPEC-NOTES.md](../SPEC-NOTES.md), part D). They are not conformance vectors: they have no row in `wex4/vectors/MANIFEST.tsv` and the tests of the two reference readers do not read them. Since 2026-11-15 the vectors have each of these cases, in files that `wex4/test_wex4.py` makes from data of its own (the table of SPEC-NOTES, "After the sentences and vectors of 2026-11-15"); these four stay as they are.

The data is synthetic (the three scripts here, fixed seeds). The files were written on 2026-11-14 by the repository's own tools, so the Go reader is tested on bytes it did not produce. The tools were used as they are, with one exception: for `extra-page.wex4` the script wraps the reference encoder and appends 4096 zero bytes to the record it returns. the expected SHA-256 values in `wex4_test.go` are those of the sources, printed by the scripts.

| File | Bytes | Root | Content |
|---|---|---|---|
| `ceil-form.wex4` | 102,400 | `9c2b563bd0dc288bc5eee3c94a5c751cf1ad963aa529f161ba04c9f93fa13f32` | the in-place upgrade of a WNR3 container without canonical zero: 2 experts of 13,056 B, record flags 0x07 and 0x05 (five matrices in ceil form, one plain because it has blocks of maximum 5), with blocks at coded scale 255 |
| `planes.wex4` | 221,184 | `7327b116925c58312ca6ce93f489659ab7cc94412ff403f204e76268c1e23bfa` | 13 tensors packed from a safetensors file: 5 WBP1 records (4 planes with the rotation, 8 planes with it, 8 planes, 4 planes, 1 plane) and 1 RAW record |
| `ragged.wex4` | 40,960 | `ad170c58136f6f16a51fae1f274efd498e7679abbe0d60187fd733ca8b0bdc40` | 4 WBP1 records of U8 tensors whose size is not a multiple of the plane count: 5,001 B in 2 planes, 10,001 B in 4, 9,999 B in 8, 20,003 B in 8; `chunk_log2 = 11`; one plane of random bytes in three of them, stored raw |
| `extra-page.wex4` | 16,384 | `9f098a3dfb618a37ab6508010c3d9d3804f5abf5ac70984d9d123feae370cc10` | **must be refused.** One WBP1 record of 1 plane, 10,000 B decoded, 3,356 B used, `stored_bytes` 8,192 instead of 4,096. Every hash of the file is right; the root is the one in its header |

How they were made, from the repository root, in a scratch folder `<TMP>` (MSYS2 UCRT64, gcc 16.2, Python 3.14):

```sh
gcc -O2 -shared c/tools/wex_nr3_tools.c -o '<TMP>'/wex_nr3_tools.dll
python -I impl/go-wex4/testdata/make_ceil_form.py '<TMP>'/src
python -I c/tools/repack_v4_wex.py --model-dir '<TMP>'/src --output '<TMP>'/ceil.wex
python -I c/tools/build_nr3.py '<TMP>'/wex_nr3_tools.dll '<TMP>'/ceil.wex container '<TMP>'/ceil-form.wex4 \
    --layers 0 --train 0:0,0:1 --save-model '<TMP>'/ceil.wnm2
python -I wex4/wex4_convert.py upgrade '<TMP>'/ceil-form.wex4

python -I impl/go-wex4/testdata/make_planes.py '<TMP>'/planes.safetensors
python -I wex4/wex4_convert.py pack '<TMP>'/planes.wex4 '<TMP>'/planes.safetensors

python -I impl/go-wex4/testdata/make_edge.py '<TMP>'      # writes ragged.wex4 and extra-page.wex4
```

`build_nr3.py` recorded, for each record, the SHA-256 of the slab it must decode to; they are the two values `make_ceil_form.py` prints. `python -I wex4/wex4.py verify FILE --dll wex4/out/wex4_tools.dll` accepted `ceil-form.wex4`, `planes.wex4` and `ragged.wex4` and printed the roots above (one run each). It refused `extra-page.wex4` ("WBP1 record rejected" with the C codec, "WBP1 padding" without `--dll`), which is what the file is for.

A second run of the commands gave every file again byte for byte (same machine, same day, one re-run). On another compiler or Python version this was not tried.

## Codec-8 (WNR4) files, 2026-11-23

The 20 `wnr4-*.wex4` files are WEX4 containers with codec-8 records of WNR4 kind GG (2) and kind I4 (3) ([WNR4-SPEC](../../../docs/WNR4-SPEC.md) sections 5 and 6, [WEX4-SPEC](../../../docs/WEX4-SPEC.md) 7.3), for readers that decode codec 8. Neither reader of `impl/` decodes codec 8 yet. Like the files above they are not conformance vectors: WEX4-SPEC 7.3 keeps codec 8 out of `wex4/vectors/` until the draft is adopted. Every file and the reason for its verdict is a row of [wnr4-MANIFEST.tsv](wnr4-MANIFEST.tsv): file, bytes, root (header bytes 4064..4095), verdict, content digest or reason, and what the file holds or what was changed. [wnr4-records.tsv](wnr4-records.tsv) gives every record of the four valid files the changes did not touch: how it is stored (codec, and for codec 8 the kind, flags, G, ntiles and, for I4, the bases), its decoded size and the SHA-256 of its decoded bytes, so a decoder can tell which record it gets wrong.

**Valid files.** The content digest in the manifest is the one of the RAW container the file was packed from, as `python -I wex4/wex4.py verify` prints it for that container. A codec-8 reader must print that digest. The codec-8 tools (`wex4/wnr4gg.py verify`, `wex4/wnr4i4.py verify`) accept each of these files with and without `--source`, and decode every record identical to the source.

| File | Bytes | What it holds |
|---|---|---|
| `wnr4-gg-mix.wex4` | 122,880 | 9 records. 6 GG records: IQ2_XXS gate and Q8_0 up stored verbatim (234 B, so the verbatim area ends with 2 zero bytes) plus a Q2_0 down slice of 444 blocks, at G 32 and at G 7; merged IQ4_NL gate_up (387 blocks) and Q2_0 down (69 blocks), at G 32 and at G 1 (tiles of fewer than 32 blocks); one Q2_0 down slice of 460 blocks at G 1 (1 segment, last tile of 12 blocks); 4 segments at G 7 (Q2_0 gate of 20 blocks, a tile with lanes 20..31 empty; IQ2_XXS gate_up verbatim; IQ4_NL up; Q2_0 down). 3 RAW records: 2 routed experts of random IQ4_NL blocks, which the tool did not code because coding them was not smaller, and one F32 tensor that is not a routed expert |
| `wnr4-gg-one.wex4` | 53,248 | 1 GG record at G 32, the first geometry above; the `wnr4-gg-bad-*` and `wnr4-gg-mut-*` files are changed copies of it |
| `wnr4-i4-mix.wex4` | 262,144 | 6 records. 5 I4 records: 3 matrices (rows 129, 128, 127; cols 64, 64, 128) planar at G 32; 1 matrix of 8 rows x 2048 cols (32 groups per row) in pairs packing at G 64; 2 matrices (4 x 2112, 33 groups per row; 129 x 64) planar at G 100, above the 64 of the CUDA decoder; 1 matrix of 256 x 64 planar at G 1 whose scales include NaN, NaN with a payload, +inf, -inf, -0, -0.5, two subnormals and 1e-7 off the lattice, and +0 (index 0 of any lattice), and whose second block (rows 128..255, negative scales only) has base 0; 130 x 128 in pairs packing at G 32 with bases 1.375 x 2^-120 and 1.0625 x 2^-118, so that most of its lattice scales are subnormal f32 values. 1 RAW record of random codes and scales, not smaller when coded |
| `wnr4-i4-one.wex4` | 217,088 | 1 I4 record, planar at G 32, of the 256 x 64 geometry with escapes and a base-0 block above (bases 0x3b9d43c1 and 0); the `wnr4-i4-bad-*` and `wnr4-i4-mut-*` files are changed copies of it |

**Changed copies.** One byte of the record or of the index changed, then the record's SHA-256, the index hash and the header hash recomputed, so that only the codec-8 decoder or the codec-8 bindings of WEX4-SPEC 7.3 can find the change; `wnr4-gg-bad-sha.wex4` alone keeps its old hashes.

| File | Change | Verdict of the codec-8 tool |
|---|---|---|
| `wnr4-gg-bad-word.wex4` | one byte of tile 0's 16-bit words | refused (record rejected) |
| `wnr4-gg-bad-kind0.wex4` | record byte 12, kind 2 -> 0 (MX, which WEX4-SPEC 7.3 does not allow in a codec-8 record of this kind of container) | refused |
| `wnr4-gg-bad-flags.wex4` | record byte 13, flags 0 -> 1 | refused |
| `wnr4-gg-bad-reserved.wex4` | record byte 40 (of bytes 28..63, which must be 0) -> 1 | refused |
| `wnr4-gg-bad-ntiles.wex4` | record bytes 20..23, ntiles 1 -> 2 | refused |
| `wnr4-gg-bad-owner.wex4` | record bytes 8..11, owner 0 -> 1 (the index says 0) | refused |
| `wnr4-gg-bad-pad.wex4` | the last of the 2 zero bytes after the verbatim slices -> 1 | refused |
| `wnr4-gg-mut-verbatim.wex4` | byte 5 of the verbatim gate slice | **accepted**, with another content digest: a verbatim byte is in no tile hash (WNR4-SPEC 5.5), only the record's SHA-256, which was recomputed. `verify --source` refuses it (record 0 differs from the source) |
| `wnr4-gg-bad-sha.wex4` | the change of `bad-word`, hashes not recomputed | refused (record hash mismatch); today's readers refuse it too |
| `wnr4-gg-bad-modelid.wex4` | index: the id of the WNM4 blob entry + 1, so it is not the blob's FNV-1a nor the record's model id | refused (WNM4 blob id is not its model id) |
| `wnr4-gg-bad-blob0.wex4` | index: the blob field of the codec-8 record entry -> 0 | refused (codec 8 without a WNM4 blob) |
| `wnr4-i4-bad-word.wex4` | one byte of tile 0's 16-bit words | refused |
| `wnr4-i4-bad-base.wex4` | base of block 0 -> 0x7f800000 (+inf, not 0 nor a positive normal finite f32) | refused |
| `wnr4-i4-bad-flags.wex4` | record byte 13, packing planar -> pairs | refused |
| `wnr4-i4-bad-anchor.wex4` | anchor[1] (bytes 30..31) -> 1 in a record of one matrix | refused |
| `wnr4-i4-mut-base.wex4` | base of block 1 0 -> 0x3f800000, a valid base; every scale of that block escapes, so the base is never read | **accepted with the content digest of `wnr4-i4-one.wex4`** (WNR4-SPEC 6.5, "What the tile hash covers") |

Which whole-file verdict and exit code a reader gives for a refused codec-8 record (a rejected file, or an undecodable record) is not decided here; the manifest says only accepted or refused. Readers that do not decode codec 8, on 2026-11-23 (one run each): the Go reader (`wex4check FILE`) and the Rust reader (`wex4r check FILE`) printed the manifest's root and "undecodable" (exit 3, the records of codec 8 named) for 19 files, and rejected `wnr4-gg-bad-sha.wex4` with `WEX4_E_HASH` (exit 1); on the four RAW sources both printed the content digests of the manifest. `python -I wex4/wex4.py verify` refused the 19 with "unsupported codec 8" and `bad-sha` with "record hash mismatch". So `bad-modelid` and `bad-blob0` pass every check of today's readers: only a reader that applies the codec-8 bindings of WEX4-SPEC 7.3 refuses them.

How they were made, from the repository root, in a scratch folder `<TMP>` (MSYS2 UCRT64, gcc 16.2, Python 3.14):

```sh
gcc -O3 -march=haswell -fopenmp -static -shared -o '<TMP>'/wnr4_gg_tools.dll wnr4/tools/wnr4_gg_tools.c
gcc -O3 -march=haswell -fopenmp -static -shared -o '<TMP>'/wnr4_i4_tools.dll wnr4/tools/wnr4_i4_tools.c
python -I impl/go-wex4/testdata/make_wnr4.py --gg-dll '<TMP>'/wnr4_gg_tools.dll --i4-dll '<TMP>'/wnr4_i4_tools.dll \
    '<TMP>'/work impl/go-wex4/testdata
```

`make_wnr4.py` writes the RAW sources with `wex4/wex4.py` (`<TMP>/work/src-*.wex4`, not kept here), trains one WNM4 model per kind on the larger source (`wnr4gg.train`, `wnr4i4.train`, every record), and packs each source once per tile depth and packing it needs (`wnr4gg.pack`, `wnr4i4.pack`). A pack uses one G and one packing for all its records; to have several in one file, the final file takes each record's stored bytes from the pack chosen for it and is written by `wex4.write` as the packs write it (the script checks that a file assembled with every record from one pack is that pack, byte for byte). The changed copies and the verdicts come last. The tools were used through their public calls only; no decoder was written. A second run gave every file again byte for byte (same machine, same day). The sources' content digests were also printed by `python -I wex4/wex4.py verify` on the RAW sources: the four digests of the manifest.
