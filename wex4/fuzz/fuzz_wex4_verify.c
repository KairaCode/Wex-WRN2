/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Fuzzing harness for the WEX4 checker c/tools/wex4_verify.c: the input is a whole WEX4 file.
 *
 * wex4_verify works on a path, so every input is written to one temporary file (in $TMPDIR, $TEMP or /tmp,
 * named after the process id and removed at exit) and checked with wex4_verify_file on one thread.
 *
 * Besides what the sanitizers catch, the checker must agree with the reader on the same bytes in memory:
 * it returns 2 ("does not open as WEX4") exactly when wex4_parse_header or wex4_open (c/wex4.h) refuses the
 * file, and when it passes a file (exit 0) the root it prints is the SHA-256 of bytes [0, 4064) of the
 * header page. */
#define WEX4_VERIFY_NO_MAIN
#include "../../c/tools/wex4_verify.c"
#include "fuzz_common.h"
#include <unistd.h>

static char temp_path[1024];
static FILE *sink;

static void fuzz_cleanup(void) { remove(temp_path); }

static void fuzz_setup(void) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !dir[0]) dir = getenv("TEMP");
    if (!dir || !dir[0]) dir = "/tmp";
    snprintf(temp_path, sizeof temp_path, "%s/wex4-fuzz-verify-%ld.wex4", dir, (long)getpid());
#ifdef _WIN32
    sink = fopen("NUL", "w");
#else
    sink = fopen("/dev/null", "w");
#endif
    if (!sink) { fprintf(stderr, "the null device cannot be opened\n"); exit(2); }
    atexit(fuzz_cleanup);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    unsigned char *file;
    char root[65], expected[65];
    Wex4Header h;
    Wex4View v;
    FILE *f;
    int status, opens, i;
    if (!sink) fuzz_setup();
    f = fopen(temp_path, "wb");
    if (!f || fwrite(data, 1, size, f) != size || fclose(f)) {
        fprintf(stderr, "%s: cannot be written\n", temp_path);
        exit(2);
    }
    status = wex4_verify_file(temp_path, 1, root, sink);

    file = fuzz_exact_copy(data, size);
    opens = size >= WEX4_PAGE && !wex4_parse_header(file, size, &h) &&
            !wex4_open(&h, file + h.index_offset, (size_t)wex4_align(h.index_bytes), &v);
    FUZZ_REQUIRE((status != 2) == opens, "wex4_verify and the reader disagree on whether the file opens");
    if (!status) {
        unsigned char digest[32];
        wex4_sha256(file, WEX4_HEADER_HASHED, digest);
        for (i = 0; i < 32; i++) snprintf(expected + i * 2, 3, "%02x", digest[i]);
        FUZZ_REQUIRE(!strcmp(root, expected), "wex4_verify passed a file with a root that is not its header hash");
    }
    free(file);
    return 0;
}
