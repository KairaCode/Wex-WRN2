#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Builds the Go reader and writer, runs their tests and checks every row of wex4/vectors/MANIFEST.tsv.
# Needs Go on PATH and nothing else (the code uses the min and max builtins of Go 1.21; run with Go 1.24.13
# and 1.27.4 only, the reader alone also with 1.21.13). MSYS2: export GOROOT=/ucrt64/lib/go first.
# With Python 3 and a C compiler on PATH (PYTHON and CC name them; default python or python3, and gcc) the
# tests also give the files of the Go writer to the two reference readers: wex4/wex4.py alone and with the C
# codec of wex4/wex4_tools.c, and c/tools/wex4_verify.c. Without them that part is skipped and a line says so;
# WEX4_NEED_REFERENCE=1 (CI) makes their absence a failure. WEX4_RANDOM_FILES (default 24) is the number of
# random files of TestWriterRandom, WEX4_RANDOM_SEED its seed.
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
mkdir -p out

# the reference readers, built here from the repository's sources and used as they are
py=""
for c in "${PYTHON:-}" python python3; do
  if [ -n "$c" ] && command -v "$c" >/dev/null 2>&1; then py=$(command -v "$c"); break; fi
done
cc=${CC:-gcc}
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) so="-shared" ;;
  *) so="-shared -fPIC" ;;
esac
exe=$(go env GOEXE)
rm -f out/reference.log
if [ -n "$py" ] && command -v "$cc" >/dev/null 2>&1 &&
   TMP="$here/out" TEMP="$here/out" "$cc" -O2 $so -I ../../c ../../wex4/wex4_tools.c -o out/wex4_tools_ref.dll &&
   TMP="$here/out" TEMP="$here/out" "$cc" -O2 -fopenmp ../../c/tools/wex4_verify.c -o "out/wex4_verify$exe"; then
  export WEX4_REF_PYTHON="$py" WEX4_REF_DLL="$here/out/wex4_tools_ref.dll" WEX4_REF_VERIFY="$here/out/wex4_verify$exe"
  export WEX4_REF_LOG="$here/out/reference.log"
elif [ "${WEX4_NEED_REFERENCE:-0}" = 1 ]; then
  echo "WEX4_NEED_REFERENCE=1 and the reference readers could not be built (python: '$py', C compiler: '$cc')"; exit 1
fi
go test -count=1 ./...
if [ -f out/reference.log ]; then
  echo "the writer's files: accepted by the reference readers with the root and the content digest of the Go writer, $(tr -d '\r\n' < out/reference.log) runs"
else
  echo "the reference readers were not asked (no Python or no C compiler): the writer's files were read by the Go reader only"
fi

bin="out/wex4check$exe"
go build -o "$bin" .
"$bin" -manifest ../../wex4/vectors/MANIFEST.tsv
"$bin" ../../wex4/vectors/valid/*.wex4
if [ ${#files[@]} -gt 0 ]; then "$bin" -v "${files[@]}"; fi
# the command line of the writer: a vector taken apart and written again has the root of the vector
rm -rf out/export
"$bin" export ../../wex4/vectors/valid/generic.wex4 out/export
"$bin" write out/export/manifest.txt out/generic-again.wex4
root=$(awk -F'\t' '$1 == "valid/generic.wex4" { print $4 }' ../../wex4/vectors/MANIFEST.tsv)
"$bin" -root "$root" out/generic-again.wex4
# the command line of the trainer: the model of testdata/ceil-form.wex4, trained again on its two experts,
# is the model blob of that file
rm -rf out/ceil
"$bin" export testdata/ceil-form.wex4 out/ceil
n=$(grep -o ' model=[0-9]*' out/ceil/manifest.txt | head -n 1 | cut -d= -f2)
"$bin" train out/ceil/manifest.txt "$n" out/ceil.wnm2
if [ "$(cksum < out/ceil.wnm2)" != "$(cksum < "out/ceil/b$((n - 1)).bin")" ]; then
  echo "the trained model is not the model of testdata/ceil-form.wex4"; exit 1
fi
echo "go reader: all checks passed"
