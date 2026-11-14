#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Every check that needs no GPU, no model and no engine tree, in one command; the entry point of README,
# CONTRIBUTING and CI. Windows: MSYS2 UCRT64 or MINGW64 gcc and python on PATH. Linux: gcc and PYTHON=python3.
# Usage: bash scripts/check_all.sh     (last line on success: all checks passed)
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" -I check_repo.py --self-test
"$PY" -I check_repo.py
for t in scripts/test_verify_wex1.py c/tests/test_repack_v4_wex.py c/tests/test_build_nr3_picks.py; do
  echo "== $t"; "$PY" -I "$t"
done
bash scripts/build_cpu_check.sh
bash wex4/build_and_test.sh
echo "all checks passed"
