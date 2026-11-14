/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_COMMON_H
#define WEX_FUZZ_COMMON_H
/* Shared by the fuzzing harnesses of wex4/fuzz/. Each harness defines the libFuzzer entry point
 * LLVMFuzzerTestOneInput (AFL++ takes the same function through its libFuzzer driver).
 *
 * With -DWEX_FUZZ_STANDALONE the harness gets a main() that runs every file named on the command line once:
 * no fuzzing engine and no clang needed, so a crash file can be replayed with gcc (and under a debugger), and
 * the seed files double as a regression test. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* An oracle: a property that must hold for every input. A failure aborts, so the engine keeps the input. */
#define FUZZ_REQUIRE(condition, what) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "ORACLE FAILED: %s (%s:%d)\n", what, __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)

/* The whole file in a malloc'd buffer of exactly its size, or NULL. */
static unsigned char *fuzz_read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *data = NULL;
    long n;
    if (!f) return NULL;
    if (!fseek(f, 0, SEEK_END) && (n = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET)) {
        data = (unsigned char *)malloc(n ? (size_t)n : 1u);
        if (data && fread(data, 1, (size_t)n, f) != (size_t)n) { free(data); data = NULL; }
        *size = (size_t)n;
    }
    fclose(f);
    return data;
}

/* The input in a heap buffer of exactly `size` bytes, so that AddressSanitizer sees a read one byte past it. */
static unsigned char *fuzz_exact_copy(const uint8_t *data, size_t size) {
    unsigned char *copy = (unsigned char *)malloc(size ? size : 1u);
    if (!copy) abort();
    if (size) memcpy(copy, data, size);
    return copy;
}

#ifdef WEX_FUZZ_STANDALONE
int main(int argc, char **argv) {
    int i, ran = 0;
    for (i = 1; i < argc; i++) {
        size_t size = 0;
        unsigned char *data = fuzz_read_file(argv[i], &size);
        if (!data) { fprintf(stderr, "%s: cannot be read\n", argv[i]); return 2; }
        LLVMFuzzerTestOneInput(data, size);
        free(data);
        ran++;
    }
    printf("%d inputs replayed, no crash and no failed oracle\n", ran);
    return 0;
}
#endif

#endif /* WEX_FUZZ_COMMON_H */
