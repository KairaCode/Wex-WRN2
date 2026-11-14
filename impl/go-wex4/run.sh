#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Builds the Go reader, runs its tests and checks every row of wex4/vectors/MANIFEST.tsv with it.
# Needs Go 1.21 or newer on PATH and nothing else (MSYS2: export GOROOT=/ucrt64/lib/go first).
# Usage: bash impl/go-wex4/run.sh     (last line on success: go reader: all checks passed)
#        bash impl/go-wex4/run.sh FILE...   also verifies these WEX4 files after the vectors
set -eu
here=$(cd "$(dirname "$0")" && pwd)
files=()
for f in "$@"; do files+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"); done
cd "$here"
command -v go >/dev/null || { echo "go is not on PATH: the Go reader was not checked"; exit 1; }
go version
unformatted=$(gofmt -l .)
if [ -n "$unformatted" ]; then echo "gofmt would change: $unformatted"; exit 1; fi
go vet ./...
go test -count=1 ./...
mkdir -p out
bin="out/wex4check$(go env GOEXE)"
go build -o "$bin" .
"$bin" -manifest ../../wex4/vectors/MANIFEST.tsv
"$bin" ../../wex4/vectors/valid/*.wex4
if [ ${#files[@]} -gt 0 ]; then "$bin" -v "${files[@]}"; fi
echo "go reader: all checks passed"
