#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Builds the Rust reader, runs its tests and checks every row of wex4/vectors/MANIFEST.tsv with it, then the
# files of impl/go-wex4/testdata, the codec-8 ones by their wnr4-MANIFEST.tsv. Needs cargo and rustc on PATH and nothing else: the crate has no dependency,
# so nothing is downloaded. Warnings fail the build unless RUSTFLAGS is set; rustfmt is run when it is installed.
# Usage: bash impl/rust-wex4/run.sh     (last line on success: rust reader: all checks passed)
#        bash impl/rust-wex4/run.sh FILE...   also checks these WEX4 files at the end
set -eu
here=$(cd "$(dirname "$0")" && pwd)
files=()
for f in "$@"; do files+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"); done
cd "$here"
command -v cargo >/dev/null || { echo "cargo is not on PATH: the Rust reader was not checked"; exit 1; }
cargo --version
rustc --version
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$here/out/target}"
export RUSTFLAGS="${RUSTFLAGS--D warnings}"
if cargo fmt --version >/dev/null 2>&1; then
  cargo fmt --check
else
  echo "rustfmt is not installed: formatting not checked"
fi
cargo build --release --offline
cargo test --release --offline
bin="$CARGO_TARGET_DIR/release/wex4r"
[ -x "$bin" ] || bin="$bin.exe"
# every row: the file's SHA-256, the verdict, the class of a refused file, and of an accepted one the root and the
# content digest of WEX4-SPEC section 5 that the manifest gives (its 7th column)
"$bin" manifest ../../wex4/vectors/MANIFEST.tsv
# three files that the vectors did not cover until 2026-11-15 (ceil form, 4 and 8 planes, sizes that are not a
# multiple of the plane count): each must be accepted with the content digest that wex4/wex4.py with the C codec
# (wex4/out/wex4_tools.dll) printed for it on 2026-11-15; these files have no row in the manifest
out=$("$bin" check ../go-wex4/testdata/{ceil-form,planes,ragged}.wex4) || { echo "$out"; exit 1; }
echo "$out"
while read -r name digest; do
  echo "$out" | grep -q "^accept [^ ]*/$name root [0-9a-f]* digest $digest " ||
    { echo "$name: not accepted with the content digest $digest"; exit 1; }
done <<'EOF'
ceil-form.wex4 8807c0ab7ba3bc555107c9e4749d70fc97cd93dd83c316c6717c44b633678d82
planes.wex4 26ccf6f46b0ae36cef9889f6945e53e8b622f4913ca32da5f8692a071e879bd6
ragged.wex4 c30449ba23b441c38aed2b6805e39251b2b9531595ba06a6eb8cdb029c389346
EOF
# a WBP1 record with one page of zeros more than it needs: the reference refuses it, so must this reader, with
# exit status 1 and a reject line of class WEX4_E_DATA (a panic, a usage error or an unknown codec is not that)
status=0
out=$("$bin" check ../go-wex4/testdata/extra-page.wex4) || status=$?
echo "$out"
[ "$status" -eq 1 ] && echo "$out" | grep -q '^reject [^ ]*extra-page.wex4 WEX4_E_DATA:' ||
  { echo "extra-page.wex4: not refused as WEX4_E_DATA (exit status $status)"; exit 1; }
# the codec-8 (WNR4) files of impl/go-wex4/testdata, row by row of its wnr4-MANIFEST.tsv (WEX4-SPEC 7.3): a valid
# file is accepted (exit status 0) with the manifest's root and the content digest of its RAW source;
# wnr4-gg-bad-sha, whose record SHA-256 does not match, is refused as WEX4_E_HASH (exit status 1); every other
# changed copy opens with the manifest's root and record 0 undecodable (exit status 3), since a codec-8 record
# that does not decode makes that record undecodable, not the file invalid
rows=0
while IFS=$'\t' read -r name _ root verdict what _; do
  case $name in '#'*|'') continue ;; esac
  rows=$((rows + 1))
  status=0
  out=$("$bin" check "../go-wex4/testdata/$name") || status=$?
  case $verdict:$what in
    accept:*) want=0; text="^accept [^ ]*/$name root $root digest ${what%% *} " ;;
    'refuse:record hash mismatch') want=1; text="^reject [^ ]*/$name WEX4_E_HASH: record 0: SHA-256 does not match\$" ;;
    *) want=3; text="^undecodable [^ ]*/$name root $root: records \[0\] not decoded (record 0: " ;;
  esac
  if [ "$status" -ne "$want" ] || ! printf '%s\n' "$out" | grep -q -- "$text"; then
    echo "$name: exit status $status and '$out', expected $want and '$text'"; exit 1
  fi
done < ../go-wex4/testdata/wnr4-MANIFEST.tsv
[ "$rows" -eq 23 ] || { echo "wnr4-MANIFEST.tsv: $rows rows, 23 expected"; exit 1; }
echo "the codec-8 files of impl/go-wex4/testdata: $rows rows of wnr4-MANIFEST.tsv as expected (6 accepted, 1 refused, 16 with an undecodable record)"
if [ ${#files[@]} -gt 0 ]; then "$bin" check "${files[@]}"; fi
echo "rust reader: all checks passed"
