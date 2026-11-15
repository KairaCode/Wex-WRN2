# A WEX4 reader in Rust

`wex4r` reads and verifies WEX4 containers. It is written in Rust with the standard library only (no
crate, also no SHA-256 crate), from the specification pages and not from the reference code or the
Go reader of [impl/go-wex4](../go-wex4/README.md). It is the second reader written that way.

**Written from:** [WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md) and, for
the WNR3 decoder and the WNM2 model, [WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md), with
[GLOSSARY](../../docs/GLOSSARY.md), [wex4/vectors/README.md](../../wex4/vectors/README.md) and
`MANIFEST.tsv`. `wex4/wex4.py`, `wex4/wex4_tools.c`, `wex4/test_wex4.py`, `c/` and the Go reader's
code and notes were not opened until this reader gave the manifest's verdict, class and root on all
112 rows. What the text did not decide is in [SPEC-NOTES.md](SPEC-NOTES.md); its part E is the
comparison with the other readers that came after, with [differential.py](differential.py).

## What it checks

`wex4r check FILE` runs the three steps of WEX4-SPEC section 6 on the whole file:

- **step 1**, the header page (section 4), in the order of section 6;
- **step 2**, the index: its SHA-256 and padding, every tensor entry, the name order, every blob and
  record entry, the expert records (5.5), split tensors, the two totals, and the tiling of the file by
  its extents (section 3);
- **step 3**, every blob (SHA-256, zero padding, the WNM2 model of WNR3-SPEC erratum 12 and its FNV-1a
  id) and every record: its SHA-256, then a decode into scratch memory with its codec. RAW: zero
  padding. WBP1: section 7.2 with every stream and chunk hash. WNR3: the cross-checks of section 7.1,
  then WNR3-SPEC sections 3 and 4 with errata 9 to 11 (stream hashes, canonical zero, the inverse of
  ceil form).

When every record decodes it also prints the content digest of section 5. One line per file:

```text
accept FILE root <64 hex> digest <64 hex> records R (D decoded, E expert) tensors T blobs B
reject FILE WEX4_E_<CLASS>: <the rule>
undecodable FILE root <64 hex>: records [...] have a codec this reader does not know
```

The classes carry the `WEX4_E_*` names of the manifest. The specifications do not define them; part B
of SPEC-NOTES says how this reader assigns them. SHA-256 (FIPS 180-4), MurmurHash3_x86_32 and FNV-1a
are written here too and tested against published values (the FIPS 180-2 examples, the vectors of
WNR3-SPEC section 4.5); the constants of SHA-256 are also recomputed from their definition in a test.

## Build and run

```bash
bash impl/rust-wex4/run.sh             # last line on success: rust reader: all checks passed
bash impl/rust-wex4/run.sh FILE.wex4   # the same, then checks FILE.wex4 too
```

`run.sh` needs `cargo` and nothing else; it builds into `impl/rust-wex4/out/target` (git-ignored),
checks the formatting when `rustfmt` is installed, runs the unit tests and every row of
`wex4/vectors/MANIFEST.tsv` (file SHA-256, verdict, class, root). Then the valid files and three files
of [impl/go-wex4/testdata](../go-wex4/testdata/README.md) (ceil form, 4 and 8 planes, sizes that are
not a multiple of the plane count) must be accepted with the content digest that the Python reader
with the C codec printed for them, and `extra-page.wex4` of that folder must be refused. It is not
part of `scripts/check_all.sh`, so that Rust is not needed for the repository's checks; the
`rust-reader` job of `.github/workflows/ci.yml` runs it on ubuntu-24.04 with the runner's own cargo,
then the differential run below. Other commands:

```bash
wex4r check [-v] [--no-decode] FILE...   # -v: SHA-256 of every decoded record and tensor; --no-decode: hashes only
wex4r manifest [-v] wex4/vectors/MANIFEST.tsv
wex4r dump FILE                           # header page and index tables as read, after step 1
```

Exit status: 0 all accepted, or every row as the manifest says; 1 a refusal or a row that disagrees;
2 usage; 3 a file with a record of an unknown codec.

The differential run compares this reader's verdicts with those of the C and the Python reader (and of
the Go reader with `--go`) on 35 hand-made files and on as many mutated ones as `--count` asks for
(7,000 by default; files equal to their base are dropped). It needs the library that
`wex4/build_and_test.sh` builds:

```bash
bash wex4/build_and_test.sh
python -I impl/rust-wex4/differential.py [--count 7000] [--go impl/go-wex4/out/wex4check] \
    impl/rust-wex4/out/target/release/wex4r OUTDIR
```

It fails when a verdict differs, when the C and the Python reader disagree or when this reader panics.
It counts the classes that differ and does not fail on them.

## Results

- **Conformance vectors** (2026-11-15): the manifest's verdict on all 112 rows, its class on all 104
  rejected rows and its root on all 8 accepted ones; every record of the valid files decoded, the two
  WNR3 records with all their stream hashes. The five valid files made from `generic.wex4` (index
  first, index last, minor 9, unknown optional bits, bytes in the reserved area) give one content
  digest, as section 5 says the two layouts must.
- **After the rows passed** (2026-11-15, SPEC-NOTES part E, one run each): the content digest is the
  Python reader's on 18 files (the valid vectors, three files of `impl/go-wex4/testdata`, the eight
  containers that `wex4/build_and_test.sh` writes), and every record and tensor of those testdata files
  has the SHA-256 that the Go reader's tests expect. On 35 hand-made files and on 20,596 mutated copies
  of seven valid files (seed 20261115), the verdict is the C reader's on every file, as is the Go
  reader's; the class differs from the C reader's on 3 of the 27 and 15 of the 18,378 files that both
  refuse. No panic.
- **Where it ran:** Windows 10 (Intel Core i7-7700K) with MSYS2 UCRT64 and Rust 1.99.1; ubuntu-24.04
  x86-64 in CI with the runner's Rust, also 1.99.1. In GitHub Actions run 43715915144 (2026-11-15)
  the `rust-reader` job passed `run.sh` and gave the C reader's verdict on 35 hand-made and 6,866
  mutated files (C library built with gcc 13.3). Not run on macOS, on ARM or on a big-endian host.

## Limits

- One thread, a binary search per rANS symbol, and every record read whole into memory: it is meant
  for checking, not for speed. No time was measured, and no real container was read.
- It reads; it does not write, convert or upgrade. The `upgrade_of` column of the manifest is not
  checked, and there is no option to compare the root with a pinned one.
- A record of a codec it does not know makes the file `undecodable` (section 10, rule 6), neither
  accepted nor refused.
- The test encoders of `src/testenc.rs` are not canonical: their records show that this encoder and
  this decoder agree, not that they agree with the reference.
- The reject classes are this reader's choice where the text does not decide (SPEC-NOTES, parts B and
  E). Six of them were moved after the comparison to the class the C reader gives; no verdict was
  changed by it.
- Tested with the vectors, hand-made and mutated files and its unit tests, not with a coverage-guided
  fuzzer. It is not in the support table of [SECURITY.md](../../SECURITY.md), so no fix is promised
  for it there.
- By reading, the code uses nothing newer than Rust 1.61 (`std::process::ExitCode`); it was compiled
  only with 1.99.1 (MSYS2, and the CI runner's).
