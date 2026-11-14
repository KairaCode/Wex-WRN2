#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Negative control of run.sh: one line of a copy of wnr3.comp is changed at a time, and run.sh must fail on
# each copy. A check that still passes with a broken shader would prove nothing about the real one.
# Usage: bash wex4/research/vulkan-decode/negative_control.sh [FILE.wex4 ...]   (the arguments of run.sh)
# Last line on success: negative control: 11 of 11 broken shaders noticed
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/out"
COPY=$HERE/out/broken.comp
noticed=0; total=0
control() {   # name, sed expression
  total=$((total + 1))
  sed "$2" "$HERE/wnr3.comp" > "$COPY"
  if [ "$(cat "$HERE/wnr3.comp")" = "$(cat "$COPY")" ]; then echo "control '$1': the expression changes nothing"; return; fi
  if WNR3_VK_SHADER="$COPY" bash "$HERE/run.sh" "$@" > "$HERE/out/control-$total.log" 2>&1; then
    echo "control '$1': NOT noticed (out/control-$total.log)"
  else
    noticed=$((noticed + 1)); echo "control '$1': noticed"
  fi
}
A=("$@")
control "hash: another rotation" 's/rotl(k, 15u)/rotl(k, 14u)/' "${A[@]+"${A[@]}"}"
control "hash: the tail of a scale stream is not folded" 's/hash = h32_tail(hash, pending);//' "${A[@]+"${A[@]}"}"
control "the final state is not compared with the hash" 's/else if (state != expect) fail(ERR_HASH);//' "${A[@]+"${A[@]}"}"
control "a wrong hash is reported as a structure error" 's/else if (state != expect) fail(ERR_HASH);/else if (state != expect) fail(ERR_STRUCT);/' "${A[@]+"${A[@]}"}"
control "no canonical-zero check" 's/if (cz && /if (false \&\& /' "${A[@]+"${A[@]}"}"
control "no error for a coded scale 0 in a max-5 block" 's/if (scale == 0u) { fail(ERR_CEIL); return; }//' "${A[@]+"${A[@]}"}"
control "ceil-form inverse: the weights are not doubled back" 's/w\[i\] = ceil_double(w\[i\])/w[i] = w[i]/' "${A[@]+"${A[@]}"}"
control "ceil-form inverse: the scale is not decremented" 's/((scale - 1u) << shift)/(scale << shift)/' "${A[@]+"${A[@]}"}"
control "a stream may be longer than its symbols need" 's/else if (position != end) fail(ERR_STRUCT);/else if (false) fail(ERR_STRUCT);/' "${A[@]+"${A[@]}"}"
control "no lower bound on the initial state" 's/bool bad = state < L || state >= (L << 8);/bool bad = state >= (L << 8);/' "${A[@]+"${A[@]}"}"
control "no upper bound on the initial state" 's/bool bad = state < L || state >= (L << 8);/bool bad = state < L;/' "${A[@]+"${A[@]}"}"
rm -f "$COPY"
bash "$HERE/run.sh" "$@" > "$HERE/out/control-0.log" 2>&1 || { echo "the unchanged shader fails (out/control-0.log)"; exit 1; }
echo "negative control: $noticed of $total broken shaders noticed"
[ "$noticed" = "$total" ]
