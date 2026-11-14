# impl/go-wex4/testdata/

Four WEX4 files for the Go reader's tests. They hold what the conformance vectors do not: WNR3 records in ceil form, WBP1 records of 4 and 8 planes, the byte 255 in rANS-coded data, WBP1 records whose size is not a multiple of the plane count, and a WBP1 record with one more page of zeros than it needs, which a reader must refuse ([SPEC-NOTES.md](../SPEC-NOTES.md), part D). They are not conformance vectors: they have no row in `wex4/vectors/MANIFEST.tsv` and the repository's own tests do not read them.

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
