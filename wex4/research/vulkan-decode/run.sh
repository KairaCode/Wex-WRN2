#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# WNR3 decode on Vulkan compute (prototype): build the two shaders and the host program into out/, then check
# the device against the CPU reference on synthetic records, on the conformance vectors and on any WEX4 file
# named on the command line. Not part of scripts/check_all.sh: it needs a Vulkan device.
#
# Needs: gcc, glslangValidator, the Vulkan headers and loader, and a Vulkan device. Any driver will do; Mesa
# lavapipe runs it on the CPU. Debian and Ubuntu: apt install libvulkan-dev glslang-tools mesa-vulkan-drivers
# (and vulkan-validationlayers for WNR3_VK_VALIDATE=1). MSYS2 UCRT64: the packages vulkan-headers,
# vulkan-loader, glslang and mesa, with VK_DRIVER_FILES naming lvp_icd.x86_64.json when no other driver is installed.
#
# Usage: bash wex4/research/vulkan-decode/run.sh [FILE.wex4 ...]
#   paths relative to the repository root, or absolute. Every WNR3 record of the extra files must decode (for
#   example wex4/out/tiny_nr3_cz.wex4 and wex4/out/synth.wex4, which wex4/build_and_test.sh leaves behind);
#   they are also the input of the mutation test.
#   MUTATIONS = mutated copies per record (default 20). WNR3_VK_DEVICE, WNR3_VK_VALIDATE: see wnr3_vk.c.
#   WNR3_VK_SHADER = another shader source (negative_control.sh uses it). VULKAN_LIB = the loader's library name.
# Last line on success: vulkan-decode: all checks passed
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
OUT=$HERE/out
mkdir -p "$OUT"
cd "$ROOT"
V=wex4/vectors
export TMP="$OUT" TEMP="$OUT"   # gcc from Git Bash cannot create temporary files under the Windows directory
case $(uname -s) in
  MINGW*|MSYS*|CYGWIN*) LIB=${VULKAN_LIB:-vulkan-1} ;;
  *) LIB=${VULKAN_LIB:-vulkan} ;;
esac
SHADER=${WNR3_VK_SHADER:-$HERE/wnr3.comp}
glslangValidator -V "$SHADER" -o "$OUT/scales.spv" > "$OUT/glslang.log"
glslangValidator -V -DWEIGHTS "$SHADER" -o "$OUT/weights.spv" >> "$OUT/glslang.log"
gcc -std=c99 -O2 -Wall -Wextra -Wno-unused-function ${WEX_CFLAGS-} -I "$ROOT/c" "$HERE/wnr3_vk.c" -o "$OUT/wnr3_vk" -l"$LIB"
export WNR3_VK_SHADERS="$OUT"
X="$OUT/wnr3_vk"
"$X" selftest
"$X" decode "$V/valid/tiny2-cz-upgraded.wex4" "$@"
# the three vectors whose WNR3 data is refused by a reader that decodes (WEX4_E_DATA); the first two are
# refused by the host checks, the third by the device
"$X" reject "$V/invalid/canonical-zero-wnr3-data-without-the-value-exact-flag.wex4" \
            "$V/invalid/wnr3-record-owner-differs-from-the-owner-in-its-data.wex4" \
            "$V/invalid/wnr3-stream-damaged-record-hash-fixed.wex4"
"$X" mutate "${MUTATIONS:-20}" 20261114 "$V/valid/tiny2-cz-upgraded.wex4" "$@"
echo "vulkan-decode: all checks passed"
