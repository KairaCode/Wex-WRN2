#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Does the mutation test notice a wrong decoder? Each case removes one check from a copy of
# wex_fast_decode.h, rebuilds the harness against the copy and expects "MISMATCH" from a short mutate run.
# Usage: sh negative_control.sh MODEL.wnm2 REC...   (after build.sh; exit 0 = every broken decoder was caught)
# Built with the SIMD kernels of this CPU: -march=haswell on x86 (AVX2), no flag elsewhere (NEON on AArch64).
# The kernel 2 control is skipped where kernel 2 does not exist (it is AVX2 only).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${WEX_SRC:-$HERE/../../../c}
MODEL=$1; shift
N=$HERE/out/neg
export TMP="$HERE/out" TEMP="$HERE/out"
cd "$HERE"
case $(uname -m) in
  x86_64|amd64|i?86) MARCH=-march=haswell ;;
  *) MARCH= ;;
esac
SIMD=$(printf '#include "wex_fast_decode.h"\nsimd WEXF_SIMD\n' | gcc $MARCH -E -P -I"$SRC" -I"$HERE" -x c - | sed -n 's/^simd //p')
fail=0
check() {   # name, sed expression, records
    name=$1; edit=$2; shift 2
    rm -rf "$N" && mkdir -p "$N"
    sed "$edit" wex_fast_decode.h > "$N/wex_fast_decode.h"
    if cmp -s wex_fast_decode.h "$N/wex_fast_decode.h"; then echo "$name: the edit did not apply"; fail=1; return; fi
    cp test_fast_decode.c "$N/"
    gcc -std=c11 -O3 $MARCH -fopenmp -I"$SRC" "$N/test_fast_decode.c" -o "$N/t.exe"
    if "$N/t.exe" mutate "$MODEL" 6000 99 "$@" | grep -q MISMATCH; then echo "$name: caught"; else echo "$name: NOT caught"; fail=1; fi
}
R="$*"
check "no canonical-zero check (kernel 1)" 's/    if (bad \&\& J->cz) return -1;/    (void)bad;/' $R
check "no ceil underflow check (fused)" 's/    if (!\*scale) return -1;/    (void)0;/' $R
check "no end-of-stream check (kernel 1 weights)" 's/        if (q\[k\] != (uintptr_t)e\[k\] || x\[k\] != x0) return -1;/        if (x[k] != x0) return -1;/' $R
[ "$SIMD" != 1 ] || check "no end-of-stream check (kernel 2 weights)" 's/            if (pp\[k\] != ee\[k\] || st\[k\] != x0) return -1;/            if (st[k] != x0) return -1;/' $R
check "no end-of-stream check (scales)" 's/        if (p\[k\] != e\[k\] || x\[k\] != wex_nr3_stream_x0(ident, out\[k\], count)) return -1;/        if (x[k] != wex_nr3_stream_x0(ident, out[k], count)) return -1;/' $R
check "no stream hash (kernel 1 weights, fused)" 's/        if (fuse) x0 = WEX_NR_L + (wex_nr3_h32_final(h\[k\], 16u + blocks \* 16u) \& WEX_NR3_HASH_MASK);/        if (fuse) x0 = x[k];/' $R
rm -rf "$N"
exit $fail
