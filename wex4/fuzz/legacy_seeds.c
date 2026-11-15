/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* The CPU encoders behind the seed containers of WNR (codec 3) and WNR2 (codec 4) records of fuzz_store_wex.
 *
 *   legacy_seeds wnr2 SOURCE.wex OUT.wex                         a WEX3 container of WNR2 records
 *   legacy_seeds model SOURCE.wex context|scale-only OUT.cdf     a WNR model trained on the experts of SOURCE
 *   legacy_seeds record MODEL.cdf CHUNK SLAB STREAM OUT          one WNR record
 *
 * SOURCE is a WEX1 container. wex4/fuzz/run.sh builds this program and wex4/fuzz/make_seeds.py runs it. The
 * tools of this repository that write such records need a GPU (c/tools/wex_neural_rans.py loads a CUDA
 * encoder, archive/nr2/ a CUDA library), so the records are encoded here on the CPU:
 *  - wnr2: a WNM2 model trained on the experts of SOURCE (wex_nr2_count, wex_nr2_model_build) and the
 *    reference encoder wex_nr2_encode of c/wex_neural_rans.h. Layout as archive/nr2/ and c/tools/build_nr3.py
 *    write it: the model descriptor at byte 64, the model at 4096, then the table and the records. The chunk
 *    sizes go round three settings from one record to the next. An expert whose bytes equal an earlier one
 *    becomes an alias of its entry (WEX3 deduplication), and one whose record would not be smaller than its
 *    raw slab is stored raw. No tool of this repository writes WNR2 containers on the CPU, so this mode also
 *    writes the container. Before it writes OUT it checks the container as the store does when it opens it
 *    (wex_validate_header, wex_validate_table, the model descriptor, the model and its id) and decodes every
 *    record with the model read back from the new container: each must give the bytes of SOURCE.
 *  - model: the compiled CDF model that c/tools/repack_v4_wex.py takes (--neural-cdf): a class per scale value
 *    (4 classes over the scale values of SOURCE below 255; 255 has none, so the writer keeps an expert with a
 *    scale of 255 out of the WNR codec), 17 CDFs of 16 symbols per class from the counts of every expert. With
 *    scale-only the 17 rows of a class are the same CDF (the counts without the symbol before).
 *  - record: what the CUDA encoder does for that writer (wex_neural_rans.Encoder.encode): the record of one
 *    expert, SLAB its scales then its weights (16 weight bytes per scale), STREAM the tile stream of its scales
 *    as the writer packed it, CHUNK 256, 512 or 1024 symbols per rANS stream; scale-only when every class of
 *    the model has 17 equal rows, as that encoder decides. It is encoded with fuzz_wnr1_encode of
 *    legacy_mutator.h, and decoded back with the decoder of c/wex_neural_rans.h before OUT is written
 *    (unpadded, as the CUDA encoder returns it).
 *
 * Exit 0 written, 1 a check failed, 2 usage or a file error. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "legacy_mutator.h"

static int fail(const char *what) {
    fprintf(stderr, "legacy_seeds: %s\n", what);
    return 1;
}

/* The whole file in a malloc'd buffer, or NULL (with a message). */
static unsigned char *seeds_read(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *data = NULL;
    long n = 0;
    if (f && !fseek(f, 0, SEEK_END) && (n = ftell(f)) > 0 && !fseek(f, 0, SEEK_SET)) {
        data = (unsigned char *)malloc((size_t)n);
        if (data && fread(data, 1, (size_t)n, f) != (size_t)n) { free(data); data = NULL; }
    }
    if (f) fclose(f);
    if (!data) fprintf(stderr, "legacy_seeds: %s: cannot be read\n", path);
    *size = data ? (size_t)n : 0;
    return data;
}

static int seeds_write(const char *path, const unsigned char *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, size, f) != size || fclose(f)) {
        fprintf(stderr, "legacy_seeds: %s: cannot be written\n", path);
        return 2;
    }
    return 0;
}

/* SOURCE: a valid WEX1 container. 0 and its table, or a message and -1. */
static int seeds_source(const unsigned char *source, size_t bytes, const WexEntry **entries, size_t *count,
                        size_t *aligned_table) {
    const WexHeader *sh = (const WexHeader *)source;
    size_t table_bytes;
    if (bytes < WEX_ALIGN || sh->magic != WEX_MAGIC_V1 ||
        wex_validate_header(sh, (uint64_t)bytes, 0, 0, &table_bytes, aligned_table) ||
        wex_validate_table(sh, (const WexEntry *)(source + sh->table_offset), (uint64_t)bytes)) {
        fail("the source is not a valid WEX1 container");
        return -1;
    }
    *entries = (const WexEntry *)(source + sh->table_offset);
    *count = table_bytes / sizeof(**entries);
    return 0;
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

static int seeds_model(const char *source_path, const char *kind, const char *out_path) {
    enum { CLASSES = 4 };
    static uint64_t counts[CLASSES][17][16];
    unsigned char model[512u + CLASSES * WEX_NR_CDF_BYTES], *source;
    const WexEntry *se;
    size_t bytes, count, aligned_table, i, k;
    uint32_t lo = 255u, hi = 0, s, c, p, scale_only;
    int rc;
    if (strcmp(kind, "context") && strcmp(kind, "scale-only")) return fail("the model kind is context or scale-only");
    scale_only = !strcmp(kind, "scale-only");
    if (!(source = seeds_read(source_path, &bytes))) return 2;
    if (seeds_source(source, bytes, &se, &count, &aligned_table)) { free(source); return 1; }
    for (i = 0; i < count; i++)
        for (k = 0; se[i].total_bytes && k < se[i].scale_bytes; k++) {
            s = source[se[i].file_offset + k];
            if (s < 255u && s < lo) lo = s;
            if (s < 255u && s > hi) hi = s;
        }
    if (lo > hi || hi - lo + 1u < CLASSES) { free(source); return fail("the source has fewer than 4 scale values below 255"); }
    for (s = 0; s < 256u; s++)
        fuzz_put_u16(model + s * 2u, s < lo || s > hi ? 65535u : (s - lo) * CLASSES / (hi - lo + 1u));
    for (i = 0; i < count; i++) {
        const unsigned char *scales = source + se[i].file_offset, *weights = scales + se[i].scale_bytes;
        if (!se[i].total_bytes) continue;
        for (k = 0; k < (size_t)se[i].weight_bytes * 2u; k++) {
            uint32_t x = (weights[k / 2u] >> ((k & 1u) * 4u)) & 15u, cls = wex_nr_u16(model + (size_t)scales[k / 32u] * 2u);
            uint32_t previous = !(k % 32u) || scale_only ? 16u : (weights[(k - 1u) / 2u] >> (((k - 1u) & 1u) * 4u)) & 15u;
            if (cls < CLASSES) counts[cls][previous][x]++;
        }
    }
    for (c = 0; c < CLASSES; c++)
        for (p = 0; p < 17u; p++)
            seeds_cdf(counts[c][scale_only ? 16u : p], model + 512u + ((size_t)c * 17u + p) * 34u);
    rc = seeds_write(out_path, model, sizeof model);
    if (!rc) printf("legacy_seeds: %s WNR model, %u classes over the scales %u to %u\n", kind, (unsigned)CLASSES,
                    (unsigned)lo, (unsigned)hi);
    free(source);
    return rc;
}

static int seeds_record(const char *model_path, const char *chunk_text, const char *slab_path,
                        const char *stream_path, const char *out_path) {
    unsigned char *model, *slab, *stream, *out = NULL, *decoded = NULL;
    size_t model_bytes, slab_bytes, stream_bytes, capacity, written = 0, padded;
    uint32_t classes, chunk = (uint32_t)strtoul(chunk_text, NULL, 10), scale_only = 1u, c, p, scale_bytes, weight_bytes;
    WexNeuralRans view;
    int rc = 1;
    model = seeds_read(model_path, &model_bytes);
    slab = seeds_read(slab_path, &slab_bytes);
    stream = seeds_read(stream_path, &stream_bytes);
    if (!model || !slab || !stream) { free(model); free(slab); free(stream); return 2; }
    classes = (uint32_t)((model_bytes - 512u) / WEX_NR_CDF_BYTES);
    scale_bytes = (uint32_t)(slab_bytes / 17u);
    weight_bytes = scale_bytes * 16u;
    if (model_bytes < 512u + WEX_NR_CDF_BYTES || (model_bytes - 512u) % WEX_NR_CDF_BYTES || classes > 256u ||
        !scale_bytes || slab_bytes != (size_t)scale_bytes * 17u || slab_bytes > UINT32_MAX) {
        fail("the model or the slab has a wrong size");
        goto done;
    }
    for (c = 0; c < classes; c++)   /* scale-only when the 17 rows of every class are equal, as the CUDA encoder decides */
        for (p = 1; p < 17u; p++)
            if (memcmp(model + 512u + ((size_t)c * 17u + p) * 34u, model + 512u + (size_t)c * 17u * 34u, 34u)) scale_only = 0;
    capacity = 32u + model_bytes + ((size_t)weight_bytes * 2u / 256u + 2u) * 4u + stream_bytes + (size_t)weight_bytes * 4u;
    out = (unsigned char *)calloc(1, capacity + WEX_ALIGN);
    decoded = (unsigned char *)malloc(slab_bytes);
    if (!out || !decoded) abort();
    if (fuzz_wnr1_encode(model, classes, chunk, scale_only, stream, (uint32_t)stream_bytes, slab, slab + scale_bytes,
                         weight_bytes, out, capacity, &written)) {
        fail("the record cannot be encoded (chunk size, model or capacity)");
        goto done;
    }
    padded = (written + WEX_ALIGN - 1u) & ~(size_t)(WEX_ALIGN - 1u);   /* out is zeroed past `written` */
    if (wex_nr_parse(out, padded, weight_bytes, &view) ||
        wex_unpack_scale_tiles(view.scales, view.scale_encoded, decoded, scale_bytes) ||
        wex_nr_decode_weights(&view, decoded, weight_bytes, decoded + scale_bytes) || memcmp(decoded, slab, slab_bytes)) {
        fail("the record does not decode to its slab");
        goto done;
    }
    rc = seeds_write(out_path, out, written);
done:
    free(decoded); free(out); free(stream); free(slab); free(model);
    return rc;
}

static int seeds_wnr2(const char *source_path, const char *out_path) {
    static const uint32_t weight_chunk[3] = {2048u, 512u, 16384u}, scale_chunk[3] = {512u, 256u, 16384u};
    unsigned char *source, *out, *slab;
    const WexHeader *sh;
    const WexEntry *se;
    WexHeader h;
    WexEntry *entries;
    WexNr2Model model, read_back;
    uint64_t model_offset, model_aligned;
    uint32_t model_bytes, model_id;
    size_t bytes, table_bytes, aligned_table, count, i, j, offset, capacity, coded = 0, raw_records = 0, aliases = 0;
    uint64_t *weight_counts, *scale_counts;
    int rc;
    if (!(source = seeds_read(source_path, &bytes))) return 2;
    if (seeds_source(source, bytes, &se, &count, &aligned_table)) { free(source); return 1; }
    sh = (const WexHeader *)source;
    table_bytes = count * sizeof(*se);

    memset(&h, 0, sizeof h);
    h.magic = WEX_MAGIC_V3; h.version = 3u;
    h.num_layers = sh->num_layers; h.num_experts = sh->num_experts;
    h.aligned_expert_bytes = sh->aligned_expert_bytes;
    h.table_offset = WEX_ALIGN + WEX_ALIGN_UP(WEX_NR2_MODEL_BYTES, WEX_ALIGN);
    h.flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS2;
    offset = (size_t)h.table_offset + aligned_table;
    capacity = offset;
    for (i = 0; i < count; i++) capacity += se[i].total_bytes;   /* a WEX1 record is its aligned raw slab */
    out = (unsigned char *)calloc(1, capacity);
    entries = (WexEntry *)calloc(count ? count : 1u, sizeof *entries);
    slab = (unsigned char *)malloc((size_t)sh->aligned_expert_bytes);
    weight_counts = (uint64_t *)calloc(3u * WEX_NR2_CONTEXTS * 16u, sizeof *weight_counts);
    scale_counts = (uint64_t *)calloc(3u * 256u, sizeof *scale_counts);
    if (!out || !entries || !slab || !weight_counts || !scale_counts) abort();

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

    for (i = 0; i < count; i++) {
        const WexEntry *e = &se[i];
        const unsigned char *raw = source + e->file_offset, *scales[3], *weights[3];
        size_t raw_bytes = (size_t)e->scale_bytes + e->weight_bytes, written = 0, padded;
        WexNr2Layout layout;
        unsigned int m;
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
        wex_nr2_layout(e, &layout);
        for (m = 0; m < 3u; m++) { scales[m] = raw + e->scale_offsets[m]; weights[m] = raw + e->weight_offsets[m]; }
        rc = wex_nr2_encode(&model, &layout, scales, weights, weight_chunk[coded % 3u], scale_chunk[coded % 3u],
                            out + offset, e->total_bytes, &written);
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
            entries[i].flags = WEX_ENTRY_NEURAL_RANS2;
            coded++;
        }
        offset += entries[i].total_bytes;
    }
    memcpy(out, &h, sizeof h);
    memcpy(out + h.table_offset, entries, table_bytes);

    /* the checks: the container as the store opens it, and every record, decoded with the model read back from
     * the container, against its source */
    if (wex_validate_header((const WexHeader *)out, offset, 0, 0, NULL, NULL) ||
        wex_validate_table((const WexHeader *)out, (const WexEntry *)(out + h.table_offset), offset) ||
        wex_nr2_model_location((const WexHeader *)out, out, &model_offset, &model_bytes, &model_id) ||
        wex_aligned_size(model_bytes, &model_aligned) ||
        wex_nr2_model_open(out + model_offset, model_bytes, &read_back) || read_back.id != model_id)
        return fail("the new container does not open as the store opens it");
    for (i = 0; i < count; i++) {
        const WexEntry *e = &entries[i];
        size_t at = wex_record_read_offset(e);
        if (!e->total_bytes) continue;
        memset(slab, 0xa5, (size_t)h.aligned_expert_bytes);
        memcpy(slab + at, out + e->file_offset, e->total_bytes);
        if ((wex_entry_nr_model(e) ? wex_read_u32_le(slab + at + 4u) != read_back.id ||
                                         wex_decode_nr2_record(e, slab, (size_t)h.aligned_expert_bytes, &read_back)
                                   : wex_decode_record(e, slab, (size_t)h.aligned_expert_bytes)) ||
            memcmp(slab, source + se[i].file_offset, (size_t)e->scale_bytes + e->weight_bytes))
            return fail("a record does not decode to its source");
    }
    if (!coded) return fail("no record was smaller than its raw slab");
    if ((rc = seeds_write(out_path, out, offset))) return rc;
    printf("legacy_seeds: WNR2 container of %lu bytes: %lu coded records, %lu raw, %lu aliases, all decoded back\n",
           (unsigned long)offset, (unsigned long)coded, (unsigned long)raw_records, (unsigned long)aliases);
    free(slab); free(entries); free(out); free(source); free(weight_counts); free(scale_counts);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "wnr2")) return seeds_wnr2(argv[2], argv[3]);
    if (argc == 5 && !strcmp(argv[1], "model")) return seeds_model(argv[2], argv[3], argv[4]);
    if (argc == 7 && !strcmp(argv[1], "record")) return seeds_record(argv[2], argv[3], argv[4], argv[5], argv[6]);
    fprintf(stderr, "usage: legacy_seeds wnr2 SOURCE.wex OUT.wex\n"
                    "       legacy_seeds model SOURCE.wex context|scale-only OUT.cdf\n"
                    "       legacy_seeds record MODEL.cdf CHUNK SLAB STREAM OUT\n");
    return 2;
}
