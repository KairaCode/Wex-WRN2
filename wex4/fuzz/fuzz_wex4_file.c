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
 * is not decoded here. It is a limit of this harness, not of the format.
 *
 * The custom mutator: one call in two, when the input (with its header and index hashes recomputed, as in the
 * second check) is a WEX4 file with WNR3 records, one of them is mutated with wnr3_mutator.h: decoded, edited
 * and encoded again, so that its stream hashes are right and the second check decodes a record that changed.
 * Every other call is libFuzzer's own mutation. At exit the harness prints how many inputs each check accepted
 * and how many different content digests the second check gave. */
#include "../wex4_tools.c"
#define FUZZ_MUTATOR_MAX_LEN 1048576u
#define FUZZ_MUTATOR_ROUNDS 40
#include "fuzz_common.h"
#include "wnr3_mutator.h"

#define FUZZ_MAX_DECODED (64u << 20)

/* what the run reached, printed at exit */
static unsigned long long count_inputs, count_hashed, count_resealed;

static void fuzz_report(void) {
    fprintf(stderr, "fuzz_wex4_file: %llu inputs, %llu accepted with their hashes, %llu accepted with the header and "
            "index hashes recomputed, %llu%s different content digests (structured mutations %s)\n", count_inputs,
            count_hashed, count_resealed, fuzz_distinct, fuzz_distinct >= FUZZ_SEEN_MAX ? " or more" : "",
            fuzz_structured() ? "on" : "off");
}

size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned int seed) {
    size_t len = size & ~(size_t)(WEX4_PAGE - 1u);
    if (fuzz_structured() && !(seed % 2u) && len >= WEX4_PAGE) {
        unsigned char *sealed = fuzz_exact_copy(data, len);
        int replaced;
        put_u64(sealed + 24u, len);
        reseal(sealed, len);
        replaced = fuzz_wnr3_mutate_wex4(data, sealed, len, seed);
        free(sealed);
        if (replaced) return size;
    }
    return LLVMFuzzerMutate(data, size, max_size);
}

/* check_impl, or 1 when the file claims a record larger than the harness decodes */
static int fuzz_check(const unsigned char *file, size_t len, int flags, unsigned char *digest) {
    Wex4Header h;
    if (len >= WEX4_PAGE && !wex4_parse_header(file, len, &h) && h.max_decoded > FUZZ_MAX_DECODED) return 1;
    return check_impl(file, len, flags, digest, 0);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    unsigned char first[32], second[32], *file;
    size_t len = size & ~(size_t)(WEX4_PAGE - 1u);
    if (!count_inputs++) atexit(fuzz_report);
    if (size < WEX4_PAGE) return 0;

    file = fuzz_exact_copy(data, size);
    if (!fuzz_check(file, size, 3, first)) {
        count_hashed++;
        FUZZ_REQUIRE(!fuzz_check(file, size, 2, second) && !memcmp(first, second, 32u),
                     "a file accepted with its hashes decodes differently without them");
    }
    free(file);

    file = fuzz_exact_copy(data, len);
    put_u64(file + 24u, len);
    reseal(file, len);
    if (!fuzz_check(file, len, 2, first)) {
        uint64_t index_offset = wex4_u64(file + 32u), index_bytes = wex4_u64(file + 40u);
        count_resealed++;
        fuzz_note_distinct(first, 32u);
        FUZZ_REQUIRE(!fuzz_check(file, len, 2, second) && !memcmp(first, second, 32u),
                     "the same file decoded twice gives two content digests");
        FUZZ_REQUIRE(!wex4_validate(file, len, file + index_offset, (size_t)wex4_align(index_bytes)),
                     "a file whose records decode has a header page or an index that does not validate");
    }
    free(file);
    return 0;
}
