#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Builds the Rust reader, runs its tests and checks every row of wex4/vectors/MANIFEST.tsv with it.
# Needs cargo and rustc on PATH and nothing else: the crate has no dependency, so nothing is downloaded.
# Compiled with Rust 1.99.1 (MSYS2) and the CI runner's version only. Warnings fail the build unless
# RUSTFLAGS is set; rustfmt is run when it is installed.
# Usage: bash impl/rust-wex4/run.sh     (last line on success: rust reader: all checks passed)
#        bash impl/rust-wex4/run.sh FILE...   also checks these WEX4 files after the vectors
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
"$bin" manifest ../../wex4/vectors/MANIFEST.tsv
"$bin" check ../../wex4/vectors/valid/*.wex4
if [ ${#files[@]} -gt 0 ]; then "$bin" check "${files[@]}"; fi
echo "rust reader: all checks passed"
