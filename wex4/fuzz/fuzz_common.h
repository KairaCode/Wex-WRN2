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
#include <unistd.h>   /* getpid; MinGW has it too */

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

/* A count of the different byte strings a harness passes here (decoded records that were accepted), told apart
 * by a 64-bit FNV-1a hash: what a run reached behind the stream hashes. The count stops at FUZZ_SEEN_MAX. */
#define FUZZ_SEEN_SLOTS 65536u
#define FUZZ_SEEN_MAX (FUZZ_SEEN_SLOTS / 2u)
static unsigned long long fuzz_distinct;
static void fuzz_note_distinct(const unsigned char *bytes, size_t n) {
    static uint64_t seen[FUZZ_SEEN_SLOTS];
    uint64_t h = UINT64_C(14695981039346656037);
    size_t i, slot;
    if (fuzz_distinct >= FUZZ_SEEN_MAX) return;
    for (i = 0; i < n; i++) { h ^= bytes[i]; h *= UINT64_C(1099511628211); }
    if (!h) h = 1u;
    for (slot = (size_t)(h % FUZZ_SEEN_SLOTS); seen[slot]; slot = (slot + 1u) % FUZZ_SEEN_SLOTS)
        if (seen[slot] == h) return;
    seen[slot] = h;
    fuzz_distinct++;
}

/* For the harnesses whose code under test opens a path: the input goes to one temporary file per process, in
 * $TMPDIR, $TEMP or /tmp. `name` ends the file name; the harness removes the file at exit. */
static void fuzz_temp_path(char *out, size_t cap, const char *name) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !dir[0]) dir = getenv("TEMP");
    if (!dir || !dir[0]) dir = "/tmp";
    snprintf(out, cap, "%s/wex-fuzz-%ld-%s", dir, (long)getpid(), name);
}
static void fuzz_write_file(const char *path, const uint8_t *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, size, f) != size || fclose(f)) {
        fprintf(stderr, "%s: cannot be written\n", path);
        exit(2);
    }
}
/* One variable of the process environment. An empty value: the code under test reads it as unset. */
static void fuzz_setenv(const char *name, const char *value) {
#ifdef _WIN32
    if (_putenv_s(name, value)) abort();
#else
    if (setenv(name, value, 1)) abort();
#endif
}

/* A harness with a custom mutator defines FUZZ_MUTATOR_MAX_LEN (the -max_len it is run with) before it includes
 * this file. libFuzzer then calls LLVMFuzzerCustomMutator instead of its own mutations, and the mutator calls
 * LLVMFuzzerMutate when it wants one of those. WEX_FUZZ_MUTATOR=0 in the environment turns the structured
 * mutations off (the mutator then only forwards to LLVMFuzzerMutate): the control for a measurement. */
#ifdef FUZZ_MUTATOR_MAX_LEN
size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned int seed);
size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);

static int fuzz_structured(void) {
    static int state = -1;
    if (state < 0) {
        const char *v = getenv("WEX_FUZZ_MUTATOR");
        state = !(v && v[0] == '0' && !v[1]);
    }
    return state;
}
#endif

#ifdef WEX_FUZZ_STANDALONE
#ifdef FUZZ_MUTATOR_MAX_LEN
#ifndef FUZZ_MUTATOR_ROUNDS
#define FUZZ_MUTATOR_ROUNDS 200
#endif
/* Stand-in for libFuzzer's own mutation in the standalone build: one to four bytes overwritten, or the input
 * cut or grown by up to 16 bytes. Deterministic. */
size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size) {
    static uint32_t state = UINT32_C(0x2545f491);
    uint32_t edits, r;
#define FUZZ_NEXT() (state ^= state << 13, state ^= state >> 17, state ^= state << 5, state)
    r = FUZZ_NEXT();
    if (!(r % 8u) && size > 16u) return size - 1u - (r >> 8) % 16u;
    if (r % 8u == 1u && size < max_size) {
        size_t grown = size + 1u + (r >> 8) % 16u;
        if (grown > max_size) grown = max_size;
        memset(data + size, (int)(r >> 16 & 255u), grown - size);
        return grown;
    }
    for (edits = 1u + FUZZ_NEXT() % 4u; size && edits; edits--) {
        r = FUZZ_NEXT();
        data[r % size] = (uint8_t)(r >> 24);
    }
#undef FUZZ_NEXT
    return size;
}
#endif

int main(int argc, char **argv) {
    int i, ran = 0, mutated = 0;
    for (i = 1; i < argc; i++) {
        size_t size = 0;
        unsigned char *data = fuzz_read_file(argv[i], &size);
        if (!data) { fprintf(stderr, "%s: cannot be read\n", argv[i]); return 2; }
        LLVMFuzzerTestOneInput(data, size);
        ran++;
#ifdef FUZZ_MUTATOR_MAX_LEN
        if (size <= FUZZ_MUTATOR_MAX_LEN) {
            /* the custom mutator on this seed: each round changes the result of the round before it, and every
             * fourth round starts again from the seed. The mutators choose between their structured path and
             * LLVMFuzzerMutate by the seed number modulo 2 or 3: the numbers 6k .. 6k + 3 of each start take
             * both paths of each of them, and the file harnesses start with the structured one */
            uint8_t *work = (uint8_t *)malloc(FUZZ_MUTATOR_MAX_LEN);
            size_t n = 0;
            int round;
            if (!work) abort();
            for (round = 0; round < FUZZ_MUTATOR_ROUNDS; round++) {
                if (!(round % 4)) { memcpy(work, data, size); n = size; }
                n = LLVMFuzzerCustomMutator(work, n, FUZZ_MUTATOR_MAX_LEN,
                                            (unsigned int)(i * 12000 + round / 4 * 6 + round % 4));
                LLVMFuzzerTestOneInput(work, n);
                mutated++;
            }
            free(work);
        }
#endif
        free(data);
    }
    printf("%d inputs replayed", ran);
    if (mutated) printf(" and %d inputs of the custom mutator run", mutated);
    printf(", no crash and no failed oracle\n");
    return 0;
}
#endif

#endif /* WEX_FUZZ_COMMON_H */
