#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBWEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CC="${CC:-gcc}"

BUILD_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t libwex-build)"
cleanup() {
    rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

UNAME_S="$(uname -s 2>/dev/null || echo Unknown)"
case "${UNAME_S}" in
    MINGW*|MSYS*|CYGWIN*)
        IO_SRC="${LIBWEX_DIR}/src/wex_io_win.c"
        LIB_EXT=".dll"
        EXE_EXT=".exe"
        DLL_CFLAGS="-DLIBWEX_BUILDING_DLL"
        GGML_DLL_CFLAGS="-DGGML_WEX_BUILDING_DLL"
        SHARED_FLAGS="-shared"
        SYS_LDFLAGS=""
        IS_WIN=1
        ;;
    Darwin)
        IO_SRC="${LIBWEX_DIR}/src/wex_io_posix.c"
        LIB_EXT=".dylib"
        EXE_EXT=""
        DLL_CFLAGS="-fPIC"
        GGML_DLL_CFLAGS="-fPIC"
        SHARED_FLAGS="-dynamiclib"
        SYS_LDFLAGS="-lpthread -ldl"
        IS_WIN=0
        ;;
    Linux)
        IO_SRC="${LIBWEX_DIR}/src/wex_io_linux.c"
        LIB_EXT=".so"
        EXE_EXT=""
        DLL_CFLAGS="-fPIC"
        GGML_DLL_CFLAGS="-fPIC"
        SHARED_FLAGS="-shared"
        SYS_LDFLAGS="-lpthread -ldl"
        IS_WIN=0
        ;;
    *)
        IO_SRC="${LIBWEX_DIR}/src/wex_io_posix.c"
        LIB_EXT=".so"
        EXE_EXT=""
        DLL_CFLAGS="-fPIC"
        GGML_DLL_CFLAGS="-fPIC"
        SHARED_FLAGS="-shared"
        SYS_LDFLAGS="-lpthread -ldl"
        IS_WIN=0
        ;;
esac

BASE_CFLAGS="-std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -I${LIBWEX_DIR}/include -I${LIBWEX_DIR}/src -I${LIBWEX_DIR}/../c ${WEX_CFLAGS:-}"
CORE_SRCS=(
    "${LIBWEX_DIR}/src/wex_store.c"
    "${LIBWEX_DIR}/src/wex_cache.c"
    "${LIBWEX_DIR}/src/wex_verify.c"
    "${IO_SRC}"
)

LIB_PATH="${BUILD_DIR}/libwex${LIB_EXT}"
GGML_LIB_PATH="${BUILD_DIR}/libggml-wex${LIB_EXT}"
NATIVE_BUILD_DIR="${BUILD_DIR}"
NATIVE_LIB_PATH="${LIB_PATH}"
NATIVE_GGML_PATH="${GGML_LIB_PATH}"
if [ "${IS_WIN}" -eq 1 ] && command -v cygpath >/dev/null 2>&1; then
    NATIVE_BUILD_DIR="$(cygpath -w "${BUILD_DIR}")"
    NATIVE_LIB_PATH="$(cygpath -w "${LIB_PATH}")"
    NATIVE_GGML_PATH="$(cygpath -w "${GGML_LIB_PATH}")"
fi

echo "[libwex] Building libwex (${CC}, ${UNAME_S}) in ${BUILD_DIR}..."
# shellcheck disable=SC2086
"${CC}" ${BASE_CFLAGS} ${DLL_CFLAGS} ${SHARED_FLAGS} "${CORE_SRCS[@]}" ${SYS_LDFLAGS} -o "${LIB_PATH}"

echo "[libwex] Building & running test_libwex_io..."
# shellcheck disable=SC2086
"${CC}" ${BASE_CFLAGS} "${LIBWEX_DIR}/tests/test_libwex_io.c" "${IO_SRC}" "${LIBWEX_DIR}/src/wex_verify.c" ${SYS_LDFLAGS} -o "${BUILD_DIR}/test_libwex_io${EXE_EXT}"
"${BUILD_DIR}/test_libwex_io${EXE_EXT}"

echo "[libwex] Building & running test_libwex_c..."
# shellcheck disable=SC2086
"${CC}" ${BASE_CFLAGS} "${LIBWEX_DIR}/tests/test_libwex_c.c" "${CORE_SRCS[@]}" ${SYS_LDFLAGS} -o "${BUILD_DIR}/test_libwex_c${EXE_EXT}"
(cd "${BUILD_DIR}" && "./test_libwex_c${EXE_EXT}" --lib "./libwex${LIB_EXT}")

echo "[libwex] Building & running colibri shim test..."
# shellcheck disable=SC2086
"${CC}" ${BASE_CFLAGS} -I"${LIBWEX_DIR}/adapters/colibri" "${LIBWEX_DIR}/adapters/colibri/test_colibri_shim.c" ${SYS_LDFLAGS} -o "${BUILD_DIR}/test_colibri_shim${EXE_EXT}"
"${BUILD_DIR}/test_colibri_shim${EXE_EXT}" --lib "${NATIVE_LIB_PATH}" --cuda 0 --file "${NATIVE_BUILD_DIR}/colibri_shim_test_mx.wex4"

echo "[libwex] Building & running ggml-wex plugin test..."
MATH_LD=""
if [ "${IS_WIN}" -eq 0 ]; then MATH_LD="-lm"; fi
# shellcheck disable=SC2086
"${CC}" ${BASE_CFLAGS} ${GGML_DLL_CFLAGS} ${SHARED_FLAGS} -I"${LIBWEX_DIR}/adapters/ggml-wex" "${LIBWEX_DIR}/adapters/ggml-wex/ggml_wex.c" ${SYS_LDFLAGS} ${MATH_LD} -o "${GGML_LIB_PATH}"
# shellcheck disable=SC2086
"${CC}" ${BASE_CFLAGS} -I"${LIBWEX_DIR}/adapters/ggml-wex" "${LIBWEX_DIR}/adapters/ggml-wex/test_ggml_wex.c" ${SYS_LDFLAGS} ${MATH_LD} -o "${BUILD_DIR}/test_ggml_wex${EXE_EXT}"
"${BUILD_DIR}/test_ggml_wex${EXE_EXT}" --plugin "${NATIVE_GGML_PATH}" --lib "${NATIVE_LIB_PATH}" --cuda 0 --file "${NATIVE_BUILD_DIR}/ggml_wex_test_wnr4gg.wex4"

PY_BIN="${PYTHON:-}"
if [ -z "${PY_BIN}" ]; then
    for cand in python3 python; do
        if command -v "${cand}" >/dev/null 2>&1; then
            if "${cand}" -c "import sys; raise SystemExit(0 if sys.version_info[0] >= 3 else 1)" >/dev/null 2>&1; then
                PY_BIN="${cand}"
                break
            fi
        fi
    done
fi

if [ -z "${PY_BIN}" ]; then
    echo "[libwex] ERROR: Python 3 interpreter not found (set PYTHON=...)" >&2
    exit 1
fi

TEST_LIB_ENV="${LIB_PATH}"
CLI_TEST_PATH="${LIBWEX_DIR}/tests/test_wex_cli.py"
PYWEX_TEST_PATH="${LIBWEX_DIR}/tests/test_pywex.py"
if [ "${IS_WIN}" -eq 1 ] && command -v cygpath >/dev/null 2>&1; then
    if "${PY_BIN}" -c "import sys; raise SystemExit(0 if sys.platform == 'win32' else 1)" >/dev/null 2>&1; then
        TEST_LIB_ENV="$(cygpath -w "${LIB_PATH}")"
        CLI_TEST_PATH="$(cygpath -w "${CLI_TEST_PATH}")"
        PYWEX_TEST_PATH="$(cygpath -w "${PYWEX_TEST_PATH}")"
    fi
fi

echo "[libwex] Running CLI, HTTP server, IPC, and MCP test suite..."
PYTHONDONTWRITEBYTECODE=1 WEX_LIB_PATH="${TEST_LIB_ENV}" WEX_CUDA_LIB="0" "${PY_BIN}" -B -I "${CLI_TEST_PATH}"

echo "[libwex] Running pywex, Strata bridge, and Hermes provider test suite..."
PYTHONDONTWRITEBYTECODE=1 WEX_LIB_PATH="${TEST_LIB_ENV}" WEX_CUDA_LIB="0" "${PY_BIN}" -B -I "${PYWEX_TEST_PATH}"

echo "[libwex] ALL SUITES PASSED."
