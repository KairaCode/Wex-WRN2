/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_ENGINE_PATHS_H
#define WEX_FUZZ_ENGINE_PATHS_H
/* Shared by the passes of store_paths.h (fuzz_store_wex.c) and q38_paths.h (fuzz_q38_wex.c) over the read paths
 * of the two engine stores that their main pass does not take: mirror copies, the read split, the striped
 * read, the CUDA helper. Included after fuzz_common.h.
 *
 *  - A mirror is a copy of the input in a directory of its own (one per mirror and process, made at first use
 *    and removed at exit), under the input's file name, in one of the forms below. The form decides whether a
 *    store must take the copy.
 *  - Both stores send an expert to one drive by the byte (flat * 2654435761) >> 16 & 255 of its flat index
 *    flat = layer * experts + expert (fuzz_split_byte).
 *  - WEX_FUZZ_CUDA_LIB names a CUDA helper for c/wex_decode.c (a full path): run.sh --replay on Windows replays
 *    the two engine harnesses a second time with the CPU stand-in c/tests/fake_wex_cuda.c, built as a DLL.
 *    Without it the harnesses set COLI_WEX_CUDA_LIB=0 (the CPU decoders). c/wex_decode.c loads a helper on
 *    Windows only, so elsewhere the variable changes nothing.
 * Which form, setting and variant an input gets depends on its bytes only (fuzz_paths_hash), so that a crash
 * file replays the same way. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

/* FNV-1a over the input */
static uint64_t fuzz_paths_hash(const unsigned char *data, size_t size) {
    uint64_t h = UINT64_C(14695981039346656037);
    size_t i;
    for (i = 0; i < size; i++) { h ^= data[i]; h *= UINT64_C(1099511628211); }
    return h;
}
/* another 64-bit value from a hash and a counter (splitmix64) */
static uint64_t fuzz_paths_mix(uint64_t h, uint64_t k) {
    uint64_t z = h + (k + 1u) * UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static unsigned fuzz_split_byte(uint32_t flat) { return (unsigned)((flat * 2654435761u) >> 16) & 255u; }

static const char *fuzz_cuda_lib(void) {
    const char *lib = getenv("WEX_FUZZ_CUDA_LIB");
    return lib && *lib ? lib : "0";
}

/* The directory of mirror k (0..3) of this process: <temp>/wex-fuzz-<pid>-mirror<k> */
#define FUZZ_MIRRORS 4
static char fuzz_mirror_dirs[FUZZ_MIRRORS][1024], fuzz_mirror_files[FUZZ_MIRRORS][2048];
static const char *fuzz_mirror_dir(int k) {
    if (!fuzz_mirror_dirs[k][0]) {
        char name[32];
        snprintf(name, sizeof name, "mirror%d", k);
        fuzz_temp_path(fuzz_mirror_dirs[k], sizeof fuzz_mirror_dirs[k], name);
#ifdef _WIN32
        (void)_mkdir(fuzz_mirror_dirs[k]);
#else
        (void)mkdir(fuzz_mirror_dirs[k], 0700);
#endif
    }
    return fuzz_mirror_dirs[k];
}
/* at exit: the copies and their directories */
static void fuzz_mirror_cleanup(void) {
    int k;
    for (k = 0; k < FUZZ_MIRRORS; k++) {
        if (fuzz_mirror_files[k][0]) remove(fuzz_mirror_files[k]);
        if (fuzz_mirror_dirs[k][0]) {
#ifdef _WIN32
            (void)_rmdir(fuzz_mirror_dirs[k]);
#else
            (void)rmdir(fuzz_mirror_dirs[k]);
#endif
        }
    }
}
/* the part of a path after its last '/' or '\\' */
static const char *fuzz_basename(const char *path) {
    const char *base = path + strlen(path);
    while (base > path && base[-1] != '/' && base[-1] != '\\') base--;
    return base;
}

/* The forms of a copy. SAME: the input's bytes. RECORDS: one byte changed in every 4096-byte page of every record
 * (the extents passed), nothing else. PAGE: one byte of the first 4096 changed. SIZE: one page of zeros appended. TABLE: one byte
 * changed in the region passed (the table, or a WEX4 index). MISSING: no file in the directory. */
enum { FUZZ_COPY_SAME, FUZZ_COPY_RECORDS, FUZZ_COPY_PAGE, FUZZ_COPY_SIZE, FUZZ_COPY_TABLE, FUZZ_COPY_MISSING,
       FUZZ_COPY_FORMS };
static const char *const fuzz_copy_names[FUZZ_COPY_FORMS] = {"same", "records", "page", "size", "table", "missing"};

typedef struct { uint64_t offset, bytes; } FuzzExtent;

/* Writes mirror k of `file` in form `form` under the file name of `primary`; returns the copy's bytes (malloc'd,
 * *copy_size of them), or NULL for MISSING. A changed byte is XORed with a value from 1 to 255. */
static unsigned char *fuzz_mirror_write(int k, const char *primary, const unsigned char *file, size_t size, int form,
                                        uint64_t h, const FuzzExtent *records, size_t record_count,
                                        FuzzExtent table) {
    unsigned char *copy;
    size_t i, n = size + (form == FUZZ_COPY_SIZE ? 4096u : 0u);
    snprintf(fuzz_mirror_files[k], sizeof fuzz_mirror_files[k], "%.1000s/%.1000s", fuzz_mirror_dir(k),
             fuzz_basename(primary));
    if (form == FUZZ_COPY_MISSING) { remove(fuzz_mirror_files[k]); return NULL; }
    copy = (unsigned char *)calloc(n ? n : 1u, 1u);
    if (!copy) abort();
    memcpy(copy, file, size);
    if (form == FUZZ_COPY_RECORDS) {
        for (i = 0; i < record_count; i++) {
            uint64_t at;
            if (!records[i].bytes || records[i].offset > size || records[i].bytes > size - records[i].offset) continue;
            for (at = 0; at < records[i].bytes; at += 4096u) {   /* every page: a read cut on a page shows */
                uint64_t r = fuzz_paths_mix(h, ((uint64_t)i << 20) + at / 4096u);
                uint64_t span = records[i].bytes - at < 4096u ? records[i].bytes - at : 4096u;
                copy[records[i].offset + at + r % span] ^= (unsigned char)(1u + (r >> 40) % 255u);
            }
        }
    } else if (form == FUZZ_COPY_PAGE && size) {
        copy[h % (size < 4096u ? size : 4096u)] ^= (unsigned char)(1u + (h >> 40) % 255u);
    } else if (form == FUZZ_COPY_TABLE && table.bytes && table.offset < size && table.bytes <= size - table.offset) {
        copy[table.offset + h % table.bytes] ^= (unsigned char)(1u + (h >> 40) % 255u);
    }
    fuzz_write_file(fuzz_mirror_files[k], copy, n);
    return copy;
}

#endif /* WEX_FUZZ_ENGINE_PATHS_H */
