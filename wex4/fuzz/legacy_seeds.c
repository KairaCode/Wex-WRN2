/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Seed containers with WNR (codec 3) or WNR2 (codec 4) records for fuzz_store_wex, from a WEX1 container.
 *
 *   legacy_seeds SOURCE.wex wnr|wnr2 OUT.wex
 *
 * wex4/fuzz/run.sh builds it and wex4/fuzz/make_seeds.py runs it. The tools of this repository that write
 * such records need a GPU (c/tools/wex_neural_rans.py loads a CUDA encoder, archive/nr2/ a CUDA library), so
 * the records are encoded here on the CPU:
 *  - wnr2: a WNM2 model trained on the experts of SOURCE (wex_nr2_count, wex_nr2_model_build) and the reference
 *    encoder wex_nr2_encode of c/wex_neural_rans.h. Layout as archive/nr2/ and c/tools/build_nr3.py write it:
 *    the model descriptor at byte 64, the model at 4096, then the table and the records.
 *  - wnr: the encoder of legacy_mutator.h (fuzz_wnr1_encode). Each record carries its own model: at most four
 *    classes of scale values, CDFs from the counts of that expert.
 * The chunk sizes go round three settings from one expert to the next; every second WNR record is scale-only.
 * An expert whose bytes equal an earlier one becomes an alias of its entry (WEX3 deduplication), and one whose
 * record would not be smaller than its raw slab is stored raw.
 *
 * Before it writes OUT it checks the container with wex_validate_table and decodes every record with the
 * decoders of c/wex_format.h: each must give the bytes of SOURCE. Exit 0 written, 1 a check failed, 2 usage
 * or a file error. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "legacy_mutator.h"

static int fail(const char *what) {
    fprintf(stderr, "legacy_seeds: %s\n", what);
    return 1;
}

/* CDF of 16 symbols, total 4096, every frequency 1 or more: the counts in proportion, the rest to the most
 * frequent symbol. */
static void seeds_cdf(const uint64_t counts[16], unsigned char *row) {
    uint64_t total = 0;
    uint32_t frequency[16], used = 0, s, best = 0;
    for (s = 0; s < 16u; s++) total += counts[s];
    for (s = 0; s < 16u; s++) {
        frequency[s] = 1u + (uint32_t)(total ? counts[s] * 4080u / total : 255u);
        used += frequency[s];
        if (counts[s] > counts[best]) best = s;
    }
    frequency[best] += 4096u - used;
    used = 0;
    for (s = 0; s < 16u; s++) { fuzz_put_u16(row + s * 2u, used); used += frequency[s]; }
    fuzz_put_u16(row + 32u, used);
}

/* One WNR record of a raw slab (scales, then weights). 0 and its unpadded size, or -1. */
static int seeds_wnr(const WexEntry *e, const unsigned char *slab, uint32_t chunk, uint32_t scale_only,
                     unsigned char *out, size_t capacity, size_t *written) {
    const unsigned char *scales = slab, *weights = slab + e->scale_bytes;
    uint32_t lo = 255u, hi = 0, classes, s, c, p;
    uint64_t (*counts)[17][16];
    unsigned char model[512u + 4u * WEX_NR_CDF_BYTES], *stream;
    size_t i, packed;
    int result;
    for (i = 0; i < e->scale_bytes; i++) {
        if (scales[i] < lo) lo = scales[i];
        if (scales[i] > hi) hi = scales[i];
    }
    classes = hi - lo + 1u < 4u ? hi - lo + 1u : 4u;
    counts = (uint64_t (*)[17][16])calloc(classes, sizeof *counts);
    stream = (unsigned char *)malloc(fuzz_tiles_bound(e->scale_bytes));
    if (!counts || !stream) abort();
    for (s = 0; s < 256u; s++)
        fuzz_put_u16(model + s * 2u, s < lo || s > hi ? 65535u : (s - lo) * classes / (hi - lo + 1u));
    for (i = 0; i < (size_t)e->weight_bytes * 2u; i++) {
        uint32_t x = (weights[i / 2u] >> ((i & 1u) * 4u)) & 15u;
        uint32_t previous = !(i % 32u) || scale_only ? 16u : (weights[(i - 1u) / 2u] >> (((i - 1u) & 1u) * 4u)) & 15u;
        counts[wex_nr_u16(model + (size_t)scales[i / 32u] * 2u)][previous][x]++;
    }
    for (c = 0; c < classes; c++)
        for (p = 0; p < 17u; p++)
            seeds_cdf(counts[c][scale_only ? 16u : p], model + 512u + ((size_t)c * 17u + p) * 34u);
    packed = fuzz_tiles_pack(scales, e->scale_bytes, stream, NULL);
    result = fuzz_wnr1_encode(model, classes, chunk, scale_only, stream, (uint32_t)packed, scales, weights,
                              e->weight_bytes, out, capacity, written);
    free(stream);
    free(counts);
    return result;
}

int main(int argc, char **argv) {
    static const uint32_t wnr_chunk[3] = {512u, 256u, 1024u};
    static const uint32_t nr2_weight_chunk[3] = {2048u, 512u, 16384u}, nr2_scale_chunk[3] = {512u, 256u, 16384u};
    unsigned char *source = NULL, *out, *slab;
    const WexHeader *sh;
    const WexEntry *se;
    WexHeader h;
    WexEntry *entries;
    WexNr2Model model;
    FILE *f;
    long source_bytes = 0;
    size_t table_bytes, aligned_table, count, i, j, offset, capacity, coded = 0, raw_records = 0, aliases = 0;
    uint64_t *weight_counts = NULL, *scale_counts = NULL;
    int nr2;
    if (argc != 4 || (strcmp(argv[2], "wnr") && strcmp(argv[2], "wnr2"))) {
        fprintf(stderr, "usage: legacy_seeds SOURCE.wex wnr|wnr2 OUT.wex\n");
        return 2;
    }
    nr2 = !strcmp(argv[2], "wnr2");
    memset(&model, 0, sizeof model);
    f = fopen(argv[1], "rb");
    if (f && !fseek(f, 0, SEEK_END) && (source_bytes = ftell(f)) > 0 && !fseek(f, 0, SEEK_SET)) {
        source = (unsigned char *)malloc((size_t)source_bytes);
        if (source && fread(source, 1, (size_t)source_bytes, f) != (size_t)source_bytes) { free(source); source = NULL; }
    }
    if (f) fclose(f);
    if (!source) { fprintf(stderr, "legacy_seeds: %s: cannot be read\n", argv[1]); return 2; }
    sh = (const WexHeader *)source;
    if ((size_t)source_bytes < WEX_ALIGN || sh->magic != WEX_MAGIC_V1 ||
        wex_validate_header(sh, (uint64_t)source_bytes, 0, 0, &table_bytes, &aligned_table) ||
        wex_validate_table(sh, (const WexEntry *)(source + sh->table_offset), (uint64_t)source_bytes))
        return fail("the source is not a valid WEX1 container");
    se = (const WexEntry *)(source + sh->table_offset);
    count = table_bytes / sizeof(*se);

    memset(&h, 0, sizeof h);
    h.magic = WEX_MAGIC_V3; h.version = 3u;
    h.num_layers = sh->num_layers; h.num_experts = sh->num_experts;
    h.aligned_expert_bytes = sh->aligned_expert_bytes;
    h.table_offset = WEX_ALIGN + (nr2 ? WEX_ALIGN_UP(WEX_NR2_MODEL_BYTES, WEX_ALIGN) : 0u);
    h.flags = WEX_HEADER_SCALE_TILES | (nr2 ? WEX_HEADER_NEURAL_RANS2 : WEX_HEADER_NEURAL_RANS);
    offset = (size_t)h.table_offset + aligned_table;
    capacity = offset;
    for (i = 0; i < count; i++) capacity += se[i].total_bytes;   /* a WEX1 record is its aligned raw slab */
    out = (unsigned char *)calloc(1, capacity);
    entries = (WexEntry *)calloc(count ? count : 1u, sizeof *entries);
    slab = (unsigned char *)malloc((size_t)sh->aligned_expert_bytes);
    if (!out || !entries || !slab) abort();

    if (nr2) {
        weight_counts = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof *weight_counts);
        scale_counts = (uint64_t *)calloc(3u * 256u, sizeof *scale_counts);
        if (!weight_counts || !scale_counts) abort();
        for (i = 0; i < count; i++) {
            const unsigned char *scales[3], *weights[3];
            WexNr2Layout layout;
            unsigned int m;
            if (!se[i].total_bytes) continue;
            wex_nr2_layout(&se[i], &layout);
            for (m = 0; m < 3u; m++) {
                scales[m] = source + se[i].file_offset + se[i].scale_offsets[m];
                weights[m] = source + se[i].file_offset + se[i].weight_offsets[m];
            }
            wex_nr2_count(&layout, scales, weights, weight_counts, scale_counts);
        }
        wex_nr2_model_build(weight_counts, scale_counts, out + WEX_ALIGN);
        if (wex_nr2_model_open(out + WEX_ALIGN, WEX_NR2_MODEL_BYTES, &model)) return fail("the trained model does not open");
        fuzz_put_u32(out + WEX_NR2_DESCRIPTOR_OFFSET, WEX_NR2_DESCRIPTOR_MAGIC);
        fuzz_put_u32(out + WEX_NR2_DESCRIPTOR_OFFSET + 4u, WEX_NR2_MODEL_BYTES);
        fuzz_put_u32(out + WEX_NR2_DESCRIPTOR_OFFSET + 8u, WEX_ALIGN);
        fuzz_put_u32(out + WEX_NR2_DESCRIPTOR_OFFSET + 16u, model.id);
    }

    for (i = 0; i < count; i++) {
        const WexEntry *e = &se[i];
        const unsigned char *raw = source + e->file_offset;
        size_t raw_bytes = (size_t)e->scale_bytes + e->weight_bytes, written = 0, padded;
        int rc;
        if (!e->total_bytes) continue;   /* an absent expert: 72 zero bytes */
        for (j = 0; j < i; j++)
            if (se[j].total_bytes && !memcmp((const char *)&se[j] + 8u, (const char *)e + 8u, sizeof(*e) - 8u) &&
                !memcmp(source + se[j].file_offset, raw, raw_bytes)) break;
        if (j < i) {
            entries[i] = entries[j];
            h.flags |= WEX_HEADER_DEDUP;
            aliases++;
            continue;
        }
        if (nr2) {
            const unsigned char *scales[3], *weights[3];
            WexNr2Layout layout;
            unsigned int m;
            wex_nr2_layout(e, &layout);
            for (m = 0; m < 3u; m++) { scales[m] = raw + e->scale_offsets[m]; weights[m] = raw + e->weight_offsets[m]; }
            rc = wex_nr2_encode(&model, &layout, scales, weights, nr2_weight_chunk[coded % 3u], nr2_scale_chunk[coded % 3u],
                                out + offset, e->total_bytes, &written);
        } else {
            rc = seeds_wnr(e, raw, wnr_chunk[coded % 3u], (uint32_t)(coded & 1u), out + offset, e->total_bytes, &written);
        }
        padded = (written + WEX_ALIGN - 1u) & ~(size_t)(WEX_ALIGN - 1u);
        entries[i] = *e;
        entries[i].file_offset = offset;
        if (rc || padded >= e->total_bytes) {
            memset(out + offset, 0, e->total_bytes);
            memcpy(out + offset, raw, raw_bytes);
            raw_records++;
        } else {
            memset(out + offset + written, 0, e->total_bytes - written);
            entries[i].total_bytes = (uint32_t)padded;
            entries[i].flags = nr2 ? WEX_ENTRY_NEURAL_RANS2 : WEX_ENTRY_NEURAL_RANS;
            coded++;
        }
        offset += entries[i].total_bytes;
    }
    memcpy(out, &h, sizeof h);
    memcpy(out + h.table_offset, entries, table_bytes);

    /* the checks: the container as the store validates it, and every record against its source */
    if (wex_validate_header((const WexHeader *)out, offset, 0, 0, NULL, NULL) ||
        wex_validate_table((const WexHeader *)out, (const WexEntry *)(out + h.table_offset), offset))
        return fail("the new container does not validate");
    for (i = 0; i < count; i++) {
        const WexEntry *e = &entries[i];
        size_t at = wex_record_read_offset(e);
        if (!e->total_bytes) continue;
        memset(slab, 0xa5, (size_t)h.aligned_expert_bytes);
        memcpy(slab + at, out + e->file_offset, e->total_bytes);
        if ((wex_entry_nr_model(e) ? wex_decode_nr2_record(e, slab, (size_t)h.aligned_expert_bytes, &model)
                                   : wex_decode_record(e, slab, (size_t)h.aligned_expert_bytes)) ||
            memcmp(slab, source + se[i].file_offset, (size_t)e->scale_bytes + e->weight_bytes))
            return fail("a record does not decode to its source");
    }
    if (!coded) return fail("no record was smaller than its raw slab");

    f = fopen(argv[3], "wb");
    if (!f || fwrite(out, 1, offset, f) != offset || fclose(f)) {
        fprintf(stderr, "legacy_seeds: %s: cannot be written\n", argv[3]);
        return 2;
    }
    printf("legacy_seeds: %s container of %lu bytes: %lu coded records, %lu raw, %lu aliases, all decoded back\n",
           nr2 ? "WNR2" : "WNR", (unsigned long)offset, (unsigned long)coded, (unsigned long)raw_records,
           (unsigned long)aliases);
    free(slab); free(entries); free(out); free(source); free(weight_counts); free(scale_counts);
    return 0;
}
