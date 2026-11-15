#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# The Vulkan prototype of wex4/research/vulkan-decode on the WNR3 containers of run.sh, byte identity only:
#   1. bash wex4/research/vulkan-decode/run.sh with no file (selftest, conformance vectors), as its README says,
#      without mutated copies; its output is kept whatever its exit status;
#   2. the host program that step 1 built, in mode decode, on the conformance vector run.sh decodes and on each
#      container (every WNR3 record must decode with the reference's bytes), one file per call;
#   3. a diagnostic, not a change of the prototype: run.sh again, with WNR3_VK_SHADER naming a copy of wnr3.comp
#      in WORK_DIR whose decoded buffer is declared coherent (one word added by sed), on the same containers.
#      Step 3 leaves the shaders of that copy in wex4/research/vulkan-decode/out;
#   4. the reject step of run.sh with those shaders, the files named by absolute paths.
# No speed is measured here. Needs what run.sh of vulkan-decode needs (gcc, glslangValidator, the Vulkan headers
# and loader, a Vulkan device).
# Usage: bash research/real-experts-20261115/vulkan.sh [WORK_DIR]   (default out/ next to this script)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
W=$(cd "${1:-$HERE/out}" && pwd)
V=$ROOT/wex4/research/vulkan-decode
FILES="$W/deepseek-v41/nr3-all.wex4 $W/deepseek-v41/pack-wnr3-cz.wex4 $W/qwen3-235b/nr3-all.wex4 $W/minimax-m21/nr3-all.wex4"
echo "== 1. bash wex4/research/vulkan-decode/run.sh (MUTATIONS=0)"
MUTATIONS=0 bash "$V/run.sh"
echo "run.sh exit status $?"
fail=0
for f in "$ROOT/wex4/vectors/valid/tiny2-cz-upgraded.wex4" $FILES; do
  echo "== 2. wnr3_vk decode $f"
  WNR3_VK_SHADERS="$V/out" "$V/out/wnr3_vk" decode "$f"
  status=$?
  echo "exit status $status"
  [ $status = 0 ] || fail=1
done
echo "steps 1 and 2: $([ $fail = 0 ] && echo 'every container decoded with the reference'"'"'s bytes' || echo 'differences, see above')"
echo "== 3. diagnostic: run.sh with the decoded buffer of a copy of wnr3.comp declared coherent"
mkdir -p "$W/vk-coherent"
sed 's/^layout(std430, set = 0, binding = 4) buffer Decoded/layout(std430, set = 0, binding = 4) coherent buffer Decoded/' \
    "$V/wnr3.comp" > "$W/vk-coherent/wnr3.comp"
diff "$V/wnr3.comp" "$W/vk-coherent/wnr3.comp"
WNR3_VK_SHADER="$W/vk-coherent/wnr3.comp" MUTATIONS=0 bash "$V/run.sh" $FILES
echo "run.sh with the coherent copy: exit status $?"
# the reject step of run.sh again, with absolute paths: on the Windows machine of this run the host program
# could not open these files by the relative paths run.sh gives (from the deep checkout folder of that machine
# their full paths are longer than the 260 characters of the Windows path limit; the valid vector of step 2 has
# a shorter name). This script ran from a short folder linked to the checkout, so ROOT is short here.
echo "== 4. the reject step of run.sh, the shaders of step 3, absolute paths"
I=$ROOT/wex4/vectors/invalid
case $(uname -s) in MINGW*|MSYS*|CYGWIN*) I=$(cygpath -m "$I") ;; esac
WNR3_VK_SHADERS="$V/out" "$V/out/wnr3_vk" reject "$I/canonical-zero-wnr3-data-without-the-value-exact-flag.wex4" \
    "$I/wnr3-record-owner-differs-from-the-owner-in-its-data.wex4" "$I/wnr3-stream-damaged-record-hash-fixed.wex4"
echo "reject: exit status $?"
