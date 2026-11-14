/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Fuzzing harness for the C WEX4 reader (c/wex4.h): the input is a whole WEX4 file.
 *
 * Every input is checked twice, through the same code the tests use (check_impl of wex4/wex4_tools.c):
 *
 *  1. as it is, with every hash verified. A mutated byte is then stopped by the first SHA-256 that covers
 *     it, which is the path a damaged or tampered file takes.
 *  2. cut to a multiple of 4096 bytes, with file_bytes set to that size and the header and index hashes
 *     recomputed, records decoded without their SHA-256. This is the path of a hostile writer, who computes
 *     the hashes last: the mutation reaches the header and index validators and the record decoders
 *     (RAW padding, WBP1, WNR3).
 *
 * Besides what the sanitizers catch, three properties must hold: a file accepted with its hashes decodes to
 * the same content digest without them; an accepted file gives the same digest when it is decoded again;
 * a file whose records decode has a header page and an index that wex4_validate accepts.
 *
 * A header may claim up to 2^30 decoded bytes for one record (WEX4-SPEC section 6). That is legal, and it
 * would make the fuzzer allocate and fill 1 GiB per input, so a file that claims more than FUZZ_MAX_DECODED
 * is not decoded here. It is a limit of this harness, not of the format. */
#include "../wex4_tools.c"
#include "fuzz_common.h"

#define FUZZ_MAX_DECODED (64u << 20)

/* check_impl, or 1 when the file claims a record larger than the harness decodes */
static int fuzz_check(const unsigned char *file, size_t len, int flags, unsigned char *digest) {
    Wex4Header h;
    if (len >= WEX4_PAGE && !wex4_parse_header(file, len, &h) && h.max_decoded > FUZZ_MAX_DECODED) return 1;
    return check_impl(file, len, flags, digest, 0);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    unsigned char first[32], second[32], *file;
    size_t len = size & ~(size_t)(WEX4_PAGE - 1u);
    if (size < WEX4_PAGE) return 0;

    file = fuzz_exact_copy(data, size);
    if (!fuzz_check(file, size, 3, first)) {
        FUZZ_REQUIRE(!fuzz_check(file, size, 2, second) && !memcmp(first, second, 32u),
                     "a file accepted with its hashes decodes differently without them");
    }
    free(file);

    file = fuzz_exact_copy(data, len);
    put_u64(file + 24u, len);
    reseal(file, len);
    if (!fuzz_check(file, len, 2, first)) {
        uint64_t index_offset = wex4_u64(file + 32u), index_bytes = wex4_u64(file + 40u);
        FUZZ_REQUIRE(!fuzz_check(file, len, 2, second) && !memcmp(first, second, 32u),
                     "the same file decoded twice gives two content digests");
        FUZZ_REQUIRE(!wex4_validate(file, len, file + index_offset, (size_t)wex4_align(index_bytes)),
                     "a file whose records decode has a header page or an index that does not validate");
    }
    free(file);
    return 0;
}
