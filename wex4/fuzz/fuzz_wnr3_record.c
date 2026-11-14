/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Fuzzing harness for the WNR3 record decoders: the input is the stored bytes of one WNR3 record.
 *
 * The model, the layout and the size of the record come from a real container: the first WNR3 record of the
 * WEX4 file named by the environment variable WEX_FUZZ_BASE (default: the conformance vector
 * wex4/vectors/valid/tiny2-cz-upgraded.wex4, relative to the working directory). The input replaces the
 * stored bytes of that record: it is cut or zero-padded to the record's stored size, parsed with
 * wex_nrx_parse and decoded twice, by the reference decoder (wex_nr2_decode, c/wex_neural_rans.h) and by the
 * fast CPU decoder (wex_fast_decode, wex4/research/fast-cpu-decode/wex_fast_decode.h).
 *
 * Besides what the sanitizers catch, the two decoders must agree: the same verdict, and on success the same
 * decoded bytes. The fast decoder states that contract ("same bytes in, same bytes out, same accept/reject
 * as the reference"); the outputs are not compared after a failure, where both may be partially written.
 * Built with -march=haswell the fast decoder runs its AVX2 kernels, without it the scalar kernel. */
#include "wex_fast_decode.h"
#include "wex4.h"
#include "fuzz_common.h"

static unsigned char *base_file;
static Wex4Record base_record;
static WexNr2Model base_model;
static WexNr2Layout base_layout;
static WexEntry base_entry;

static void fuzz_setup(void) {
    const char *path = getenv("WEX_FUZZ_BASE");
    Wex4Header h;
    Wex4View v;
    Wex4Blob blob;
    size_t size = 0;
    uint32_t r;
    if (!path || !path[0]) path = "wex4/vectors/valid/tiny2-cz-upgraded.wex4";
    base_file = fuzz_read_file(path, &size);
    if (!base_file || size < WEX4_PAGE || wex4_parse_header(base_file, size, &h) ||
        wex4_open(&h, base_file + h.index_offset, (size_t)wex4_align(h.index_bytes), &v)) {
        fprintf(stderr, "%s: not a WEX4 file that opens (set WEX_FUZZ_BASE, or run from the repository root)\n", path);
        exit(2);
    }
    for (r = 0; r < h.records; r++) {
        wex4_get_record(&v, r, &base_record);
        if (base_record.codec == WEX4_CODEC_WNR3) break;
    }
    if (r == h.records) { fprintf(stderr, "%s: no WNR3 record\n", path); exit(2); }
    wex4_get_blob(&v, base_record.blob - 1u, &blob);
    if (wex_nr2_model_open(base_file + blob.offset, (size_t)blob.bytes, &base_model) || base_model.id != blob.id ||
        wex4_expert_entry(&v, r, &base_entry)) {
        fprintf(stderr, "%s: the model blob or the expert entry of record %u is not valid\n", path, r);
        exit(2);
    }
    wex_nr2_layout(&base_entry, &base_layout);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    unsigned char *stored, *reference, *fast, *scales_a[3], *weights_a[3], *scales_b[3], *weights_b[3];
    WexNr2Record view;
    int a, b;
    unsigned int m;
    if (!base_file) fuzz_setup();
    stored = (unsigned char *)calloc(base_record.stored, 1u);
    if (!stored) abort();
    memcpy(stored, data, size < base_record.stored ? size : base_record.stored);
    if (wex_nrx_parse(stored, base_record.stored, &base_layout, base_model.id, WEX_NR3_MAGIC, &view)) {
        free(stored);
        return 0;
    }
    reference = (unsigned char *)calloc(base_record.decoded, 1u);
    fast = (unsigned char *)calloc(base_record.decoded, 1u);
    if (!reference || !fast) abort();
    for (m = 0; m < 3u; m++) {
        scales_a[m] = reference + base_entry.scale_offsets[m]; weights_a[m] = reference + base_entry.weight_offsets[m];
        scales_b[m] = fast + base_entry.scale_offsets[m]; weights_b[m] = fast + base_entry.weight_offsets[m];
    }
    a = wex_nr2_decode(&view, &base_layout, &base_model, scales_a, weights_a);
    b = wex_fast_decode(&view, &base_layout, &base_model, scales_b, weights_b);
    FUZZ_REQUIRE(!a == !b, "the reference and the fast WNR3 decoder give different verdicts");
    FUZZ_REQUIRE(a || !memcmp(reference, fast, base_record.decoded),
                 "the reference and the fast WNR3 decoder accept the record with different bytes");
    free(fast);
    free(reference);
    free(stored);
    return 0;
}
