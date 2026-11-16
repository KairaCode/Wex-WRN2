#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# A diagnostic of the differences that vulkan.sh found, not a change of the prototype. It builds, in WORK_DIR/vk-diag,
# a copy of wex4/research/vulkan-decode/wnr3_vk.c with vulkan-diag.patch applied (at a record whose decoded bytes
# differ from the reference's, it prints per matrix how many scale bytes and how many weight bytes differ, how
# many of the differing scale bytes hold the coded scale, the value before the ceil-form inverse, and how many
# scales that inverse changes), and the two shaders of wnr3.comp. Then it decodes the two DeepSeek V4.1 containers
# of run.sh with it and runs its selftest. No speed is measured. Needs gcc, git (for git show and git apply, in a
# clone that has commit 1af6c0d), glslangValidator, the Vulkan loader, a device.
# wnr3_vk.c and wnr3.comp are taken from commit 1af6c0d (WNR3_DIAG_COMMIT names another), the prototype this run
# measured: the patch applies to that wnr3_vk.c and not to the later one, whose shader no longer has the fault
# (research/vulkan-gpu-20261115). Until 2026-11-16 the script took both files from the checkout, which was 1af6c0d
# when the logs of this folder were written.
# Usage: bash research/real-experts-20261115/vulkan_diag.sh [WORK_DIR]   (default out/ next to this script)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
W=$(cd "${1:-$HERE/out}" && pwd)
C=${WNR3_DIAG_COMMIT:-1af6c0d}
D=$W/vk-diag
mkdir -p "$D"
export TMP="$D" TEMP="$D"
case $(uname -s) in MINGW*|MSYS*|CYGWIN*) LIB=vulkan-1 ;; *) LIB=vulkan ;; esac
echo "== wnr3_vk.c with vulkan-diag.patch and the shaders of wnr3.comp, both of commit $C; the c/ headers of $(git -C "$ROOT" rev-parse --short HEAD)"
git -C "$ROOT" show "$C:wex4/research/vulkan-decode/wnr3_vk.c" > "$D/wnr3_vk.c" || exit 1
git -C "$ROOT" show "$C:wex4/research/vulkan-decode/wnr3.comp" > "$D/wnr3.comp" || exit 1
(cd "$D" && git apply "$HERE/vulkan-diag.patch") || exit 1
mv "$D/wnr3_vk.c" "$D/wnr3_vk_diag.c"
"${CC:-gcc}" -std=c99 -O2 -Wall -Wextra -Wno-unused-function -I "$ROOT/c" "$D/wnr3_vk_diag.c" -o "$D/wnr3_vk_diag" -l"$LIB" || exit 1
glslangValidator -V "$D/wnr3.comp" -o "$D/scales.spv" > /dev/null || exit 1
glslangValidator -V -DWEIGHTS "$D/wnr3.comp" -o "$D/weights.spv" > /dev/null || exit 1
export WNR3_VK_SHADERS="$D"
for f in "$W/deepseek-v41/nr3-all.wex4" "$W/deepseek-v41/pack-wnr3-cz.wex4"; do
  echo "== wnr3_vk_diag decode $f"
  "$D/wnr3_vk_diag" decode "$f"
  echo "exit status $?"
done
echo "== wnr3_vk_diag selftest"
"$D/wnr3_vk_diag" selftest
echo "exit status $?"
