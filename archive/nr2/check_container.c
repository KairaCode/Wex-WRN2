/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Independent check of a WNR2 container with the engine's toolchain (MSYS2
 * gcc, C11) and only the engine headers: wex_validate_header/_table,
 * wex_nr2_model_location/_model_open, every entry against the WEX1 source
 * (all fields except file_offset/total_bytes/flags), and every record decoded
 * from disk by the engine's slab path (wex_decode_nr2_record, or the raw rule
 * for flags 0) and compared with the source's raw slab (canonicalized -0 -> +0
 * when asked).
 * usage: check_container <wnr2.wex> <source-v1.wex> <canonical_zero 0|1> [stride]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wex_format.h"

static int read_at(FILE *f, void *dst, size_t bytes, uint64_t offset) {
    return _fseeki64(f, (long long)offset, SEEK_SET) || fread(dst, 1, bytes, f) != bytes ? -1 : 0;
}

static uint64_t file_size(FILE *f) {
    _fseeki64(f, 0, SEEK_END);
    return (uint64_t)_ftelli64(f);
}

static void canonicalize(const WexEntry *e, unsigned char *slab) {
    for (unsigned m = 0; m < 3u; m++) {
        unsigned char *w = slab + e->weight_offsets[m];
        size_t n = (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1];
        for (size_t i = 0; i < n; i++) {
            if ((w[i] & 15u) == 8u) w[i] &= 0xf0u;
            if ((w[i] >> 4) == 8u) w[i] &= 0x0fu;
        }
    }
}

#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); return 1; } while (0)

int main(int argc, char **argv) {
    if (argc < 4) FAIL("usage: %s <wnr2.wex> <source-v1.wex> <canonical_zero> [stride]", argv[0]);
    int cz = atoi(argv[3]), stride = argc > 4 ? atoi(argv[4]) : 1;
    FILE *f = fopen(argv[1], "rb"), *s = fopen(argv[2], "rb");
    if (!f || !s || stride < 1) FAIL("cannot open inputs");
    unsigned char *page = malloc(WEX_ALIGN), *spage = malloc(WEX_ALIGN);
    WexHeader hdr, shdr;
    size_t tb, atb, stb, satb;
    uint64_t bytes = file_size(f), sbytes = file_size(s);
    if (read_at(f, page, WEX_ALIGN, 0) || read_at(s, spage, WEX_ALIGN, 0)) FAIL("header read");
    memcpy(&hdr, page, sizeof hdr); memcpy(&shdr, spage, sizeof shdr);
    if (wex_validate_header(&hdr, bytes, 0, 0, &tb, &atb)) FAIL("wex_validate_header refused");
    if (wex_validate_header(&shdr, sbytes, (int)hdr.num_layers, (int)hdr.num_experts, &stb, &satb) || stb != tb)
        FAIL("source layout differs");
    WexEntry *t = malloc(atb), *st = malloc(satb);
    if (read_at(f, t, atb, hdr.table_offset) || read_at(s, st, satb, shdr.table_offset)) FAIL("table read");
    if (wex_validate_table(&hdr, t, bytes)) FAIL("wex_validate_table refused");
    uint64_t at; uint32_t mbytes, id;
    if (wex_nr2_model_location(&hdr, page, &at, &mbytes, &id)) FAIL("wex_nr2_model_location refused");
    unsigned char *blob = malloc(mbytes);
    WexNr2Model model;
    if (read_at(f, blob, mbytes, at) || wex_nr2_model_open(blob, mbytes, &model) || model.id != id)
        FAIL("model invalid or id mismatch");
    size_t count = tb / sizeof(WexEntry), present = 0, decoded = 0, wnr2 = 0, raw_entries = 0;
    size_t capacity = (size_t)hdr.aligned_expert_bytes;
    unsigned char *slab = malloc(capacity + 64), *ref = malloc(capacity);
    for (size_t i = 0; i < count; i++) {
        WexEntry a = t[i], b = st[i];
        if (!b.total_bytes) {
            static const WexEntry zero;
            if (memcmp(&a, &zero, sizeof a)) FAIL("entry %zu present only in the WNR2 file", i);
            continue;
        }
        present++;
        a.file_offset = b.file_offset; a.total_bytes = b.total_bytes; a.flags = b.flags;
        if (memcmp(&a, &b, sizeof a)) FAIL("entry %zu differs from the source beyond offset/size/flags", i);
        if (t[i].flags != WEX_ENTRY_NEURAL_RANS2 && t[i].flags != 0) FAIL("entry %zu codec %u", i, t[i].flags);
        if ((present - 1) % (size_t)stride) continue;
        const WexEntry *e = &t[i];
        uint64_t raw, aligned;
        if (wex_entry_raw_sizes(e, &raw, &aligned)) FAIL("raw sizes %zu", i);
        if (read_at(s, ref, (size_t)raw, b.file_offset)) FAIL("source read %zu", i);
        if (cz) canonicalize(e, ref);
        memset(slab, 0xa5, capacity + 64);
        size_t off = wex_record_read_offset(e);
        if (read_at(f, slab + off, e->total_bytes, e->file_offset)) FAIL("record read %zu", i);
        if (e->flags == WEX_ENTRY_NEURAL_RANS2) {
            if (wex_read_u32_le(slab + off + 4u) != model.id) FAIL("record %zu names another model", i);
            if (wex_decode_nr2_record(e, slab, capacity, &model)) FAIL("wex_decode_nr2_record refused %zu", i);
            wnr2++;
        } else {
            if (wex_decode_record(e, slab, capacity)) FAIL("raw entry %zu refused", i);
            raw_entries++;
        }
        if (memcmp(slab, ref, (size_t)raw)) FAIL("expert %zu decodes to different bytes", i);
        decoded++;
    }
    printf("{\"file\": \"%s\", \"file_bytes\": %llu, \"model_id\": \"0x%08x\", \"present\": %zu, "
           "\"decoded_and_equal\": %zu, \"wnr2\": %zu, \"raw\": %zu, \"canonical_zero\": %d, \"stride\": %d}\n",
           argv[1], (unsigned long long)bytes, model.id, present, decoded, wnr2, raw_entries, cz, stride);
    return 0;
}
