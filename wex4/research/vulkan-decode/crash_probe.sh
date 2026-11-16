#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Probe of a driver crash (docs/WNR3-VULKAN.md, section 6; README, "The llvmpipe crash"). The scale pass of
# wnr3.comp is built in forms that stop a stream at once when its initial state is out of range, and one
# record of the selftest is run with each: record 10, whose scale stream 0 opens below L. The script says for
# each form whether the driver crashed. It is not a check: it ends with status 0 whatever the forms do, and
# the verdict of a form that does not crash is of no interest (crash.comp decodes nothing useful).
#
# Forms: `current` (wnr3.comp as it is: the stream is decoded to its end, then refused), with the scale table
# and with -DSEARCH; `first` (the first form of the shader: fail and return at the state check, and the binary
# search for scale symbols, -DSEARCH), also with the validation layer and with robustBufferAccess, and once
# with the scale table; `count0` (no return: the symbol count is set to 0); then crash.comp, the first form
# without the stream hash, from which one switch after the other is turned off for as long as the driver
# still crashes (a greedy reduction: a switch whose removal stops the crash is turned on again). The smallest
# form found is run once more with the validation layer, with robustBufferAccess and, when the stream that
# stops no longer depends on the data, with other values of STOP and on another record. Last, crash_min.comp
# (the smallest form found on the CI runner, written out) runs through wnr3_vk and then with crash_min.c, a
# host of its own that binds small buffers and dispatches one workgroup: no record and no WNR3 code, with the
# descriptor set layout of wnr3_vk.c and without parts of it.
# Needs: what run.sh needs, after run.sh (it uses out/wnr3_vk and out/weights.spv). spirv-val and gdb are used
# when they are installed: spirv-val checks every module, gdb prints where a form that crashed stopped.
# Usage: bash wex4/research/vulkan-decode/crash_probe.sh [CASE]     (default 10)
#   WNR3_VK_DEVICE, WNR3_VK_STAGE: see wnr3_vk.c. CC, VULKAN_LIB as in run.sh.
# Last lines: the runs of crash_min.c, then: crash probe: N of M runs crashed
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/out
P=$OUT/probe
CASE=${1:-10}
X=$OUT/wnr3_vk
if [ ! -x "$X" ] && [ ! -x "$X.exe" ]; then echo "$X is missing: run run.sh first" >&2; exit 1; fi
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) LIB=${VULKAN_LIB:-vulkan-1} ;;
  *) LIB=${VULKAN_LIB:-vulkan} ;;
esac
mkdir -p "$P"
echo "glslangValidator: $(glslangValidator --version 2>&1 | sed -n 1p)"
if command -v spirv-val > /dev/null 2>&1; then echo "spirv-val: $(spirv-val --version 2>&1 | sed -n 1p)"; fi
crashed=0; total=0
probe() {   # name, then the environment of the run and the arguments of wnr3_vk; status 0 when it crashed
  local name=$1 status
  shift
  total=$((total + 1))
  env WNR3_VK_SHADERS="$P/shaders" "$@" > "$P/run.log" 2>&1
  status=$?
  if [ "$status" -ge 128 ]; then
    crashed=$((crashed + 1))
    echo "$name: CRASHED (status $status, signal $((status - 128)); $(grep -c '^validation:' "$P/run.log") messages of the validation layer before it)"
    if command -v gdb > /dev/null 2>&1 && [ "${GDB:-1}" = 1 ]; then
      gdb -q -batch -ex run -ex bt -ex 'x/6i $pc' -ex 'info registers rip rax rbx rcx rdx rsi rdi' \
          --args env WNR3_VK_SHADERS="$P/shaders" "$@" 2>&1 | grep -v '^\[' | grep -v 'libthread_db\|executing new program' |
          tail -20 | sed 's/^/    gdb: /'
    fi
    return 0
  fi
  echo "$name: no crash (status $status; $(grep -c . "$P/run.log") lines, the last: $(tail -1 "$P/run.log" | cut -c1-80))"
  return 1
}
form() {   # name, shader source, glslang options: the scale pass of this form, the weight pass as it is
  rm -rf "$P/shaders"; mkdir -p "$P/shaders"
  cp "$OUT/weights.spv" "$P/shaders/weights.spv"
  if ! glslangValidator -V $3 "$2" -o "$P/shaders/scales.spv" > "$P/glslang.log" 2>&1; then cat "$P/glslang.log"; echo "$1: does not compile"; return 1; fi
  if command -v spirv-val > /dev/null 2>&1; then
    if spirv-val --target-env vulkan1.0 "$P/shaders/scales.spv"; then echo "$1: spirv-val (vulkan1.0) accepts the module"; else echo "$1: spirv-val REFUSES the module"; fi
  fi
}
form current "$HERE/wnr3.comp" "" && probe "current, record $CASE" "$X" selftest "$CASE"
form "current, -DSEARCH" "$HERE/wnr3.comp" "-DSEARCH" && probe "current, -DSEARCH, record $CASE" "$X" selftest "$CASE"
# the first form had the binary search for scale symbols too: -DSEARCH
sed 's/bool bad = state < L || state >= (L << 8);/if (state < L || state >= (L << 8)) { fail(ERR_STRUCT); return; }\n    bool bad = false;/' \
    "$HERE/wnr3.comp" > "$P/first.comp"
sed 's/bool bad = state < L || state >= (L << 8);/&\n    if (bad) count = 0u;/' "$HERE/wnr3.comp" > "$P/count0.comp"
if form first "$P/first.comp" "-DSEARCH"; then
  probe "first, record $CASE" "$X" selftest "$CASE"
  GDB=0 probe "first, record $CASE, validation layer" WNR3_VK_VALIDATE=1 "$X" selftest "$CASE"
  GDB=0 probe "first, record $CASE, robustBufferAccess" WNR3_VK_ROBUST=1 "$X" selftest "$CASE"
  GDB=0 probe "first, all twelve records" "$X" selftest
fi
form "first, scale table" "$P/first.comp" "" && GDB=0 probe "first with the scale table, record $CASE" "$X" selftest "$CASE"
form count0 "$P/count0.comp" "-DSEARCH" && GDB=0 probe "count0, record $CASE" "$X" selftest "$CASE"

# the greedy reduction of crash.comp
SWITCHES="DATA_STOP ATOMIC STORES RENORM REC_STATE PLAN_TABLE BOUND MATRIX LANE_COUNT PLAN_COUNT NESTED SEARCH"
declare -A on
for s in $SWITCHES; do on[$s]=1; done
defines() { local s d=""; for s in $SWITCHES; do d="$d -DF_$s=${on[$s]}"; done; echo "$d${1:+ -DSTOP=$1}"; }
off() { local s d=""; for s in $SWITCHES; do if [ "${on[$s]}" = 0 ]; then d="$d $s"; fi; done; echo "${d:- nothing}"; }
form "crash.comp, all switches on" "$HERE/crash.comp" "$(defines)" &&
  GDB=0 probe "crash.comp, all switches on, record $CASE" "$X" selftest "$CASE"
for s in $SWITCHES; do
  # without the bound an invocation past the last stream would store and read past the record's own parts
  if [ "$s" = BOUND ] && [ "${on[STORES]}${on[RENORM]}${on[REC_STATE]}" != 000 ]; then continue; fi
  on[$s]=0
  if ! { form "crash.comp without$(off)" "$HERE/crash.comp" "$(defines)" &&
         GDB=0 probe "crash.comp without$(off), record $CASE" "$X" selftest "$CASE"; }; then
    on[$s]=1
  fi
done
echo "smallest form of crash.comp:$(defines) (taken away:$(off)); after the preprocessor:"
glslangValidator -E $(defines) "$HERE/crash.comp" 2>&1 | grep -v '^ *$\|^ *//' | sed 's/^/    | /'
if form "smallest form" "$HERE/crash.comp" "$(defines)"; then
  probe "smallest form, record $CASE" "$X" selftest "$CASE"
  GDB=0 probe "smallest form, record $CASE, validation layer" WNR3_VK_VALIDATE=1 "$X" selftest "$CASE"
  GDB=0 probe "smallest form, record $CASE, robustBufferAccess" WNR3_VK_ROBUST=1 "$X" selftest "$CASE"
  if [ "${on[DATA_STOP]}" = 0 ]; then GDB=0 probe "smallest form, record 9" "$X" selftest 9; fi
fi
if [ "${on[DATA_STOP]}" = 0 ]; then
  # the invocation that stops no longer depends on the record: other invocations (record 10 has 15 scale
  # streams: with the bound, STOP=15 stops none of them)
  for stop in 1 2 4 7 8 9 15 16 24; do
    form "smallest form, STOP=$stop" "$HERE/crash.comp" "$(defines "$stop")" &&
      GDB=0 probe "smallest form, STOP=$stop, record $CASE" "$X" selftest "$CASE"
  done
fi
# crash_min.comp: the smallest form found on the CI runner, written out
for v in "" "-DSTOP=1" "-DSTOP=8" "-DF_VALUES=0" "-DF_STATE=1" "-DF_STATE=0" "-DF_COUNT=0" \
         "-DF_VALUES=0 -DF_STATE=1" "-DF_VALUES=0 -DF_STATE=1 -DSTOP=1"; do
  form "crash_min.comp${v:+ $v}" "$HERE/crash_min.comp" "$v" &&
    GDB=0 probe "crash_min.comp${v:+ $v}, record $CASE" "$X" selftest "$CASE"
done
if form "crash_min.comp" "$HERE/crash_min.comp" ""; then
  GDB=0 probe "crash_min.comp, record 0" "$X" selftest 0
  GDB=0 probe "crash_min.comp, record $CASE, validation layer" WNR3_VK_VALIDATE=1 "$X" selftest "$CASE"
  GDB=0 probe "crash_min.comp, record $CASE, robustBufferAccess" WNR3_VK_ROBUST=1 "$X" selftest "$CASE"
fi
# crash_min.comp with crash_min.c: small buffers, one workgroup of 32, invocation STOP returns at once; the set
# layout of wnr3_vk.c (bindings 0 to 5, a push constant range) unless CRASH_MIN_BINDINGS or CRASH_MIN_PUSH say
if TMP="$OUT" TEMP="$OUT" "${CC:-gcc}" -std=c99 -O2 -Wall -Wextra "$HERE/crash_min.c" -o "$P/crash_min" -l"$LIB"; then
  for v in "" "-DSTOP=1" "-DSTOP=8" "-DF_COUNT=0" "-DF_VALUES=0 -DF_STATE=1"; do
    form "crash_min.c, crash_min.comp${v:+ $v}" "$HERE/crash_min.comp" "$v" &&
      probe "crash_min.c, crash_min.comp${v:+ $v}" "$P/crash_min" "$P/shaders/scales.spv"
  done
  if form "crash_min.c, crash_min.comp" "$HERE/crash_min.comp" ""; then
    GDB=0 probe "crash_min.c, crash_min.comp, validation layer" WNR3_VK_VALIDATE=1 "$P/crash_min" "$P/shaders/scales.spv"
    GDB=0 probe "crash_min.c, crash_min.comp, robustBufferAccess" WNR3_VK_ROBUST=1 "$P/crash_min" "$P/shaders/scales.spv"
    GDB=0 probe "crash_min.c, crash_min.comp, 2 workgroups" "$P/crash_min" "$P/shaders/scales.spv" 256 2
    GDB=0 probe "crash_min.c, crash_min.comp, no push constant range" CRASH_MIN_PUSH=0 "$P/crash_min" "$P/shaders/scales.spv"
    # which bindings of the set layout it takes: the shader declares 1, 2 and 5
    for b in 125 0125 12345 1235 1245; do
      GDB=0 probe "crash_min.c, crash_min.comp, set layout bindings $b" CRASH_MIN_BINDINGS=$b "$P/crash_min" "$P/shaders/scales.spv"
    done
    GDB=0 probe "crash_min.c, crash_min.comp, set layout bindings 125, no push constant range" CRASH_MIN_BINDINGS=125 \
        CRASH_MIN_PUSH=0 "$P/crash_min" "$P/shaders/scales.spv"
  fi
else
  echo "crash_min.c: does not build"
fi
echo "crash probe: $crashed of $total runs crashed"
