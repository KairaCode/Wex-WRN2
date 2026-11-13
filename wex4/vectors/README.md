# WEX4 conformance vectors

Small WEX4 files with the verdict a conforming reader must give ([WEX4-SPEC](../../docs/WEX4-SPEC.md), section 17). Use them to test a reader or a writer written from the specification, in any language.

| Folder | Files | What a reader must do |
|---|---|---|
| `valid/` | 7 | open the file, verify every hash, decode every record (the WNR3 records need a WNR3 decoder, [WNR3-SPEC](../../docs/WNR3-SPEC.md)), and report the root given in the manifest |
| `invalid/` | 46 | refuse the file, without reading or writing out of bounds, allocating without limit or hanging |
| `legacy/` | 1 | a WEX3 file with WNR3 records: not WEX4, so a WEX4 reader refuses it (wrong magic); it is the input of the upgrade test below |
| `../review/security-c/findings/` | 10 | the files of the 2026-11-11 security review, which stay in the review folder; their current verdicts are listed too |

`MANIFEST.tsv` has one row per file, tab-separated:

| Column | Meaning |
|---|---|
| `path` | relative to this folder |
| `verdict` | `accept` or `reject` |
| `class` | for a rejected file, the status the C reader (`c/wex4.h`, `wex4_check` with hashes on) returns, as its `WEX4_E_*` name. Another reader only has to refuse the file; the class says which rule the file breaks first in the C reader's order |
| `root` | for an accepted file, the root: SHA-256 of bytes [0, 4064) of the header page, also stored at byte 4064 |
| `sha256` | of the whole file, so that a damaged copy of a vector is not mistaken for a reader bug |
| `upgrade_of` | for `valid/tiny2-cz-upgraded.wex4`: `legacy/tiny2-cz.wex3`. The in-place upgrade (`wex4/wex4_convert.py upgrade`) of that file must give this file byte for byte; a converter written elsewhere can be checked the same way |

What the valid files cover:

- `generic.wex4`: written from scratch, index right after the header page; 12 tensors (one record holding several tensors, a split tensor, a zero-size tensor, a scalar, an opaque dtype), RAW records and WBP1 records (codec 6), free-form metadata.
- `generic-index-last.wex4`: the same content with the index after the records; same content digest, different root.
- `tiny2-cz-upgraded.wex4`: the in-place upgrade of a WNR3 container with canonical zero (the layout of the DeepSeek V4.1 production file), 2 experts, with the WNM2 model blob and the old header page and entry table kept as blobs.
- `tiny-raw-upgraded.wex4`: the in-place upgrade of a WEX1 container of RAW records (the layout of the Qwen3.8 production files), 6 layers x 8 experts.
- `minor-version-9.wex4`, `unknown-optional-feature-bits.wex4`, `bytes-in-the-reserved-header-area.wex4`: forward-compatible changes of `generic.wex4` that a reader must still accept (section 10).

The invalid files are `generic.wex4` with one field changed each, the header and index hashes recomputed so that the change reaches the validators (except where the file name says the hash was left alone): wrong magic, major version 5, unknown required feature, sizes and offsets past the end or overflowing, overlapping records and tensors, a decompression bomb, duplicate or invalid names, a truncated file, appended bytes, and so on. Their names say what was changed; `wex4/test_wex4.py` (`test_hostile`) builds them.

## How the vectors are checked here

`bash wex4/build_and_test.sh` (and `bash scripts/check_all.sh`) ends with a check of every row: the file hash, the C reader's verdict and class, the Python reader's verdict, the root, and the upgrade. The vectors are written by `wex4/test_wex4.py` itself, from the same files its tests build; after a change of the format, a maintainer runs `bash wex4/build_and_test.sh --write-vectors "$PWD/wex4/vectors"` and commits the new files with the specification change. A vector whose verdict changes is a format change.

Size: 54 files in this folder (the manifest has 64 rows: these and the 10 review files), 8,620,032 B; the invalid files are near copies of `generic.wex4`, so git packs the folder into about 0.74 MB.
