# A WEX4 reader in Rust

`wex4r` reads and verifies WEX4 containers. It is written in Rust with the standard library only (no
crate, also no SHA-256 crate), from the specification pages and not from the reference code or the
Go reader of [impl/go-wex4](../go-wex4/README.md). It is the second reader written that way.

**Written from:** [WEX4-SPEC](../../docs/WEX4-SPEC.md), [WNR3-SPEC](../../docs/WNR3-SPEC.md) and, for
the WNR3 decoder and the WNM2 model, [WNR2-REFERENCE](../../docs/WNR2-REFERENCE.md), with
[GLOSSARY](../../docs/GLOSSARY.md), [wex4/vectors/README.md](../../wex4/vectors/README.md) and
`MANIFEST.tsv`. `wex4/wex4.py`, `wex4/wex4_tools.c`, `wex4/test_wex4.py`, `c/` and the Go reader's
code and notes were not opened until this reader gave the manifest's verdict, class and root on all
112 rows. What the text did not decide is in [SPEC-NOTES.md](SPEC-NOTES.md).

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
checks the formatting when `rustfmt` is installed, runs the unit tests, every row of
`wex4/vectors/MANIFEST.tsv` (file SHA-256, verdict, class, root) and the valid files. It is not part of
`scripts/check_all.sh`, so that Rust is not needed for the repository's checks; the `rust-reader` job
of `.github/workflows/ci.yml` runs it on ubuntu-24.04 with the runner's own cargo. Other commands:

```bash
wex4r check [--no-decode] FILE...   # --no-decode: hashes only, like wex4_verify
wex4r manifest [-v] wex4/vectors/MANIFEST.tsv
wex4r dump FILE                      # header page and index tables as read, after step 1
```

Exit status: 0 all accepted, or every row as the manifest says; 1 a refusal or a row that disagrees;
2 usage; 3 a file with a record of an unknown codec.

## Results

- **Conformance vectors** (2026-11-15): the manifest's verdict on all 112 rows, its class on all 104
  rejected rows and its root on all 8 accepted ones; every record of the valid files decoded, the two
  WNR3 records with all their stream hashes. The five valid files made from `generic.wex4` (index
  first, index last, minor 9, unknown optional bits, bytes in the reserved area) give one content
  digest, as section 5 says the two layouts must.
- **Where it ran:** Windows 10 with MSYS2 UCRT64 and Rust 1.99.1; ubuntu-24.04 in CI. Not run on
  macOS, on ARM or on a big-endian host.

## Limits

- One thread, a binary search per rANS symbol, and every record read whole into memory: it is meant
  for checking, not for speed. No time was measured, and no real container was read.
- It reads; it does not write, convert or upgrade. The `upgrade_of` column of the manifest is not
  checked, and there is no option to compare the root with a pinned one.
- A record of a codec it does not know makes the file `undecodable` (section 10, rule 6), neither
  accepted nor refused.
- The test encoders of `src/testenc.rs` are not canonical: their records show that this encoder and
  this decoder agree, not that they agree with the reference.
- By reading, the code uses nothing newer than Rust 1.61 (`std::process::ExitCode`); it was compiled
  only with 1.99.1 and with the CI runner's version.
