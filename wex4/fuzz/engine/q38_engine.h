/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_ENGINE_Q38_H
#define WEX_FUZZ_ENGINE_Q38_H
/* STAND-IN, for wex4/fuzz/fuzz_q38_wex.c only. c/qwen38_wex.h is not a header that stands alone: the engine
 * includes it once from qwen38_core.h, in the middle of qwen38.c, and it uses the engine's model, its expert
 * slots, its safetensors reader and its compat.h. None of them is in this repository, and the fields the WEX
 * path adds to them (the wex_* and mx_* members, the MXFP4 weight kind) are in no public tree. This file
 * declares what c/qwen38_wex.h names, with types inferred from how it uses them, so that the functions that
 * read a container build without the engine. A difference between these types and the engine's (a width, a
 * signedness) is a difference the harness cannot see.
 *
 * What is real: every line of c/qwen38_wex.h, c/wex_decode.c, c/wex_format.h and c/wex4.h.
 * What is a stand-in: the types below; q38_weight_free and q38_slab_unpin (nothing to free or unpin here);
 * the safetensors reader behind Q38_MXFP4_DIR (the Quark shards are not a container: that path is not
 * fuzzed and its stand-ins stop the process); on Windows the positional read and the aligned allocation of
 * the engine's compat.h. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <malloc.h>
#else
#include <strings.h>
#include <unistd.h>
#endif

typedef struct {
    int hidden, layers, experts, inter;
} Cfg;

typedef enum { Q38_WEIGHT_F32 = 0, Q38_WEIGHT_MXFP4 } Q38WeightKind;

typedef struct {
    void *data;
    const uint8_t *e8;   /* MXFP4: one E8M0 exponent per row and 32 inputs */
    int rows, cols;
    int64_t elements, scale_count;
    Q38WeightKind kind;
} Q38Weight;

typedef struct {
    int eid;
    Q38Weight gate, up, down;
    void *int4_slab;
    int64_t int4_slab_bytes;
} Slot;

typedef struct { int dtype, rank, fd; int64_t shape[4], nbytes, off; } st_tensor;
typedef struct { int unused; } shards;

typedef struct {
    shards S;
    int mx;                                   /* 1: MXFP4 records */
    int64_t record_bytes, mx_raw, mx_scale_off[3], mx_weight_off[3];
    int wex4, nr3, wex_n, wex_stripe;
    int wex_fd[4], wex_cut[4];
    uint32_t wex_flags;
    WexEntry *wex_entry;
    unsigned char *wex_seen, *wex_sha;
} Q38Int4Experts;

typedef struct {
    Cfg c;
    int range_begin, range_end;
    Q38Int4Experts *x4;
} Model;

static void q38_weight_free(Q38Weight *weight) { (void)weight; }
static void q38_slab_unpin(Slot *slot) { (void)slot; }

/* Q38_MXFP4_DIR: not a container, not fuzzed */
static void q38_name(Model *m, char *out, size_t cap, int layer, const char *suffix) {
    (void)m; (void)layer;
    snprintf(out, cap, "%s", suffix);
}
static st_tensor *st_find(shards *S, const char *name) { (void)S; (void)name; return NULL; }
static void st_init(shards *S, const char *dir) { (void)S; (void)dir; abort(); }
static void st_read_range_raw_cap(shards *S, int fd, int64_t off, int64_t bytes, void *dst, int64_t cap, int strict,
                                  const char *what) {
    (void)S; (void)fd; (void)off; (void)bytes; (void)dst; (void)cap; (void)strict; (void)what;
    abort();
}

#ifdef _WIN32
/* the parts of the engine's compat.h that c/qwen38_wex.h uses on Windows */
static DWORD compat_pread_lasterr;
static int64_t compat_fsize(int fd) {
    LARGE_INTEGER size;
    return GetFileSizeEx((HANDLE)_get_osfhandle(fd), &size) ? (int64_t)size.QuadPart : -1;
}
/* the file is open for overlapped I/O (q38_wex_open): the offset travels in the OVERLAPPED */
static ssize_t fuzz_q38_pread(int fd, void *buffer, size_t bytes, int64_t offset) {
    HANDLE file = (HANDLE)_get_osfhandle(fd);
    OVERLAPPED ov;
    DWORD got = 0;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)((uint64_t)offset & 0xffffffffu);
    ov.OffsetHigh = (DWORD)((uint64_t)offset >> 32);
    if (!ReadFile(file, buffer, (DWORD)bytes, &got, &ov) &&
        (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult(file, &ov, &got, TRUE))) {
        compat_pread_lasterr = GetLastError();
        errno = EIO;
        return -1;
    }
    return (ssize_t)got;
}
#define pread(fd, buffer, bytes, offset) fuzz_q38_pread(fd, buffer, bytes, (int64_t)(offset))
static int fuzz_q38_memalign(void **out, size_t alignment, size_t bytes) {
    *out = _aligned_malloc(bytes ? bytes : 1u, alignment);
    return *out ? 0 : ENOMEM;
}
#define posix_memalign(out, alignment, bytes) fuzz_q38_memalign(out, alignment, bytes)
#define compat_aligned_free _aligned_free
#else
#define compat_aligned_free free
#endif

#endif /* WEX_FUZZ_ENGINE_Q38_H */
