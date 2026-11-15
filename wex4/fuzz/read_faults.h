/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_READ_FAULTS_H
#define WEX_FUZZ_READ_FAULTS_H
/* Reads that fail on demand, for the passes of store_paths.h and q38_paths.h that fail one read of the engine
 * file under test. A harness includes this file right before c/expert_store_wex.c or c/qwen38_wex.h, after the
 * headers that declare the reads (on Windows, for c/qwen38_wex.h, after the compat.h whose pread it uses). From
 * there on the file's reads go through two wrappers: fuzz_fault_pread for pread (on Windows only where compat.h
 * defines pread as compat_pread, which c/qwen38_wex.h uses), and on Windows fuzz_fault_ReadFile for ReadFile
 * (c/expert_store_wex.c, and the striped read of c/qwen38_wex.h). Nothing else changes: compat_pread keeps the
 * real ReadFile, because compat.h defines it before this file.
 *
 * Nothing fails until a pass arms the fault: fuzz_fault_arm(skip, fail) lets `skip` reads through, then fails
 * the next `fail` ones, and fuzz_fault_disarm() ends it. A failed pread returns -1 with errno EIO (on Windows
 * also compat_pread_lasterr = ERROR_READ_FAULT, as compat_pread reports a failed ReadFile); a failed ReadFile
 * returns FALSE with ERROR_READ_FAULT and 0 bytes read, before it reads anything. fuzz_fault_reads counts every
 * read, so a pass can count the reads of a clean call before it fails one of them; fuzz_fault_failed counts
 * the reads failed. The engine files read on the calling thread here, so the counts are exact. */
#include <errno.h>
#include <stddef.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

static long long fuzz_fault_skip = -1, fuzz_fault_left;
static unsigned long long fuzz_fault_reads, fuzz_fault_failed;

static inline void fuzz_fault_arm(long long skip, long long fail) { fuzz_fault_skip = skip; fuzz_fault_left = fail; }
static inline void fuzz_fault_disarm(void) { fuzz_fault_skip = -1; fuzz_fault_left = 0; }
/* 1: this read fails */
static inline int fuzz_fault_now(void) {
    fuzz_fault_reads++;
    if (fuzz_fault_skip < 0) return 0;
    if (fuzz_fault_skip > 0) { fuzz_fault_skip--; return 0; }
    if (fuzz_fault_left <= 0) return 0;
    fuzz_fault_left--;
    fuzz_fault_failed++;
    return 1;
}

#ifdef _WIN32
static inline BOOL fuzz_fault_ReadFile(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD got, LPOVERLAPPED ov) {
    if (fuzz_fault_now()) {
        if (got) *got = 0;
        SetLastError(ERROR_READ_FAULT);
        return FALSE;
    }
    return ReadFile(file, buffer, bytes, got, ov);
}
#define ReadFile fuzz_fault_ReadFile
#ifdef pread
static inline ssize_t fuzz_fault_pread(int fd, void *buffer, size_t bytes, off_t offset) {
    if (fuzz_fault_now()) {
        compat_pread_lasterr = ERROR_READ_FAULT;
        errno = EIO;
        return -1;
    }
    return compat_pread(fd, buffer, bytes, offset);
}
#undef pread
#define pread(fd, buffer, bytes, offset) fuzz_fault_pread(fd, buffer, bytes, offset)
#endif
#else
static inline ssize_t fuzz_fault_pread(int fd, void *buffer, size_t bytes, off_t offset) {
    if (fuzz_fault_now()) {
        errno = EIO;
        return -1;
    }
    return pread(fd, buffer, bytes, offset);
}
#define pread fuzz_fault_pread
#endif

#endif /* WEX_FUZZ_READ_FAULTS_H */
