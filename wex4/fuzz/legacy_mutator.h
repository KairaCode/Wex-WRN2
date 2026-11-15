/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_LEGACY_MUTATOR_H
#define WEX_FUZZ_LEGACY_MUTATOR_H
/* Mutations that know the coded records of a legacy container (WEX2, WEX3), for the custom mutator of
 * fuzz_store_wex.c, and the two encoders they need, which legacy_seeds.c uses too.
 *
 * What a changed byte does to each record (c/wex_format.h, c/wex_neural_rans.h, c/wex_palette.h):
 *  - scale tiles (codec 1) and palette (codec 2): the scales are a tile stream, per 256 scales a base, a bit
 *    width and the packed differences. It must end exactly at its stored length, so a changed width refuses
 *    the record. The palette tables and the 3-bit weights have no check at all: every byte decodes.
 *  - WNR (codec 3): the record holds its own model (a class per scale value, 17 CDFs of 16 symbols per class),
 *    a tile stream for the scales and one rANS stream per 256, 512 or 1024 weights. Each stream must end at
 *    its length with the state back at 2^23, so a changed payload byte refuses the record.
 *  - WNR2 (codec 4): the WNR3 streams without the hash (wnr3_mutator.h); the same end condition.
 *
 * The mutations here work on the other side of each codec, as wnr3_mutator.h does for WNR3: they decode the
 * record, change the decoded scales and weights (and, for WNR, the model, the chunk size and the scale-only
 * flag; for a palette record, the tables and the packed weights), and encode it again, so the new record
 * passes the checks that its codec has. One tile in four of the tile stream is written loosely (a wider bit
 * width, a lower base: the decoder's checked path for a base near 255). A few WNR edits make a record the
 * decoder refuses, on purpose: a scale value taken out of the model, or the scale-only flag set over rows that
 * differ. A record is replaced in place, only when the new one takes the same pages and differs from the old.
 *
 * No tool of this repository encodes WNR records on the CPU (c/tools/wex_neural_rans.py calls a CUDA
 * library): fuzz_wnr1_encode is written here from the decoder wex_nr_decode_weights. */
#include "wnr3_mutator.h"

static void fuzz_put_u16(unsigned char *p, uint32_t value) {
    p[0] = (unsigned char)value; p[1] = (unsigned char)(value >> 8);
}
static void fuzz_put_u32(unsigned char *p, uint32_t value) {
    fuzz_put_u16(p, value); fuzz_put_u16(p + 2, value >> 16);
}

/* The most bytes fuzz_tiles_pack writes for `count` scales: two per tile and one per scale. */
static size_t fuzz_tiles_bound(size_t count) { return count + 2u * ((count + 255u) / 256u); }

/* Tile stream of `count` scales into out (fuzz_tiles_bound(count) bytes); the result is its size. rng NULL:
 * the form c/tools/repack_v4_wex.py writes (base = the minimum of the tile, the narrowest width). Else one
 * tile in four gets a wider width and a lower base, which decode to the same scales. */
static size_t fuzz_tiles_pack(const unsigned char *scales, size_t count, unsigned char *out, uint32_t *rng) {
    size_t at = 0, first, i;
    for (first = 0; first < count; first += WEX_SCALE_TILE_VALUES) {
        size_t n = count - first < WEX_SCALE_TILE_VALUES ? count - first : WEX_SCALE_TILE_VALUES;
        uint32_t lo = 255u, hi = 0, bits = 0, base, accumulator = 0, have = 0;
        for (i = 0; i < n; i++) {
            if (scales[first + i] < lo) lo = scales[first + i];
            if (scales[first + i] > hi) hi = scales[first + i];
        }
        while ((hi - lo) >> bits) bits++;
        base = lo;
        if (rng && !(fuzz_rng(rng) % 4u)) {
            uint32_t r = fuzz_rng(rng), room;
            bits += r % (9u - bits);
            room = ((1u << bits) - 1u) - (hi - lo);
            if (room > base) room = base;
            base -= (r >> 8) % (room + 1u);
        }
        out[at++] = (unsigned char)base;
        out[at++] = (unsigned char)bits;
        if (!bits) continue;
        for (i = 0; i < n; i++) {
            accumulator |= (scales[first + i] - base) << have;
            have += bits;
            while (have >= 8u) { out[at++] = (unsigned char)accumulator; accumulator >>= 8; have -= 8u; }
        }
        if (have) out[at++] = (unsigned char)accumulator;
    }
    return at;
}

/* One edit of `count` decoded scales. */
static void fuzz_scales_edit(unsigned char *scales, size_t count, uint32_t *rng) {
    static const unsigned char edge[4] = {0, 1, 254, 255};
    size_t i = fuzz_rng(rng) % count, n;
    uint32_t r = fuzz_rng(rng);
    switch (fuzz_rng(rng) % 6u) {
    case 0: scales[i] = scales[r % count]; break;                       /* a value the record has already */
    case 1: scales[i] = (unsigned char)(scales[i] + (r & 1u ? 1u : 255u)); break;
    case 2: scales[i] = (unsigned char)r; break;
    case 3: scales[i] = edge[r % 4u]; break;
    case 4:                                                             /* a run of one value: tiles of width 0 */
        n = 1u + r % 512u;
        if (n > count - i) n = count - i;
        memset(scales + i, scales[i], n);
        break;
    default:                                                            /* a run copied from elsewhere */
        n = 1u + r % 32u;
        if (n > count - i) n = count - i;
        memmove(scales + i, scales + (r >> 8) % (count - n + 1u), n);
        break;
    }
}

/* A scale-tile record (codec 1): "WCS1", the decoded size, the stream size and the stream in the first
 * `flags >> 8` pages, then the weights as they are. The scales are edited and packed again; the record is
 * rewritten when the new stream still fits those pages. The result is `total`, or 0 with the record unchanged. */
static size_t fuzz_tiles_mutate(unsigned char *record, size_t total, const WexEntry *e, uint32_t seed) {
    size_t prefix = (size_t)(e->flags >> WEX_ENTRY_SCALE_PAGES_SHIFT) * WEX_ALIGN, packed, result = 0;
    uint32_t rng = seed, stream, edits;
    unsigned char *scales, *out;
    if (prefix < WEX_SCALE_STREAM_HEADER_BYTES || prefix > total || !e->scale_bytes ||
        wex_read_u32_le(record) != WEX_SCALE_STREAM_MAGIC || wex_read_u32_le(record + 4u) != e->scale_bytes) return 0;
    stream = wex_read_u32_le(record + 8u);
    if (!stream || stream > prefix - WEX_SCALE_STREAM_HEADER_BYTES) return 0;
    scales = (unsigned char *)malloc(e->scale_bytes);
    out = (unsigned char *)malloc(fuzz_tiles_bound(e->scale_bytes));
    if (!scales || !out) abort();
    if (!wex_unpack_scale_tiles(record + WEX_SCALE_STREAM_HEADER_BYTES, stream, scales, e->scale_bytes)) {
        for (edits = 1u + fuzz_rng(&rng) % 4u; edits; edits--) fuzz_scales_edit(scales, e->scale_bytes, &rng);
        packed = fuzz_tiles_pack(scales, e->scale_bytes, out, &rng);
        if (packed <= prefix - WEX_SCALE_STREAM_HEADER_BYTES) {
            memset(record + WEX_SCALE_STREAM_HEADER_BYTES, 0, prefix - WEX_SCALE_STREAM_HEADER_BYTES);
            memcpy(record + WEX_SCALE_STREAM_HEADER_BYTES, out, packed);
            fuzz_put_u32(record + 8u, (uint32_t)packed);
            result = total;
        }
    }
    free(out);
    free(scales);
    return result;
}

/* A palette record (codec 2): 16 header bytes, three tables of 1024 bytes, the tile stream of the scales, three
 * packed bytes per four weight bytes. The scales are edited and packed again, and a few bytes of the tables
 * and of the packed weights are changed (a scale picks its row of a table, so a new scale reads another row).
 * The result is `total`, or 0 with the record unchanged when the new record takes other pages. */
static size_t fuzz_palette_mutate(unsigned char *record, size_t total, const WexEntry *e, uint32_t seed) {
    size_t packed, disk, result = 0;
    uint32_t rng = seed, stream, weights, edits;
    unsigned char *scales, *out;
    if (total < WEX_PALETTE3_PREFIX_BYTES || !e->scale_bytes || wex_read_u32_le(record) != WEX_PALETTE3_MAGIC) return 0;
    stream = wex_read_u32_le(record + 4u);
    weights = wex_read_u32_le(record + 8u);
    if (!stream || !weights || weights != e->weight_bytes / 4u * 3u ||
        (uint64_t)WEX_PALETTE3_PREFIX_BYTES + stream + weights > total) return 0;
    scales = (unsigned char *)malloc(e->scale_bytes);
    out = (unsigned char *)malloc(WEX_PALETTE3_PREFIX_BYTES + fuzz_tiles_bound(e->scale_bytes) + weights);
    if (!scales || !out) abort();
    if (!wex_unpack_scale_tiles(record + WEX_PALETTE3_PREFIX_BYTES, stream, scales, e->scale_bytes)) {
        unsigned char *new_weights;
        for (edits = fuzz_rng(&rng) % 4u; edits; edits--) fuzz_scales_edit(scales, e->scale_bytes, &rng);
        memcpy(out, record, WEX_PALETTE3_PREFIX_BYTES);
        packed = fuzz_tiles_pack(scales, e->scale_bytes, out + WEX_PALETTE3_PREFIX_BYTES, &rng);
        new_weights = out + WEX_PALETTE3_PREFIX_BYTES + packed;
        memcpy(new_weights, record + WEX_PALETTE3_PREFIX_BYTES + stream, weights);
        fuzz_put_u32(out + 4u, (uint32_t)packed);
        for (edits = 1u + fuzz_rng(&rng) % 4u; edits; edits--) {
            uint32_t r = fuzz_rng(&rng);
            if (r & 1u) out[WEX_PALETTE3_HEADER_BYTES + (r >> 1) % WEX_PALETTE3_TABLE_BYTES] = (unsigned char)(r >> 16);
            else new_weights[(r >> 1) % weights] = (unsigned char)(r >> 16);
        }
        disk = WEX_PALETTE3_PREFIX_BYTES + packed + weights;
        if (((disk + WEX_ALIGN - 1u) & ~(size_t)(WEX_ALIGN - 1u)) == total) {
            memset(record, 0, total);
            memcpy(record, out, disk);
            result = total;
        }
    }
    free(out);
    free(scales);
    return result;
}

/* WNR encoder: the record wex_nr_parse and wex_nr_decode_weights (c/wex_neural_rans.h) read. model: the 512
 * mapping bytes (a u16 class per scale value, 65535 for none) and `classes` x 17 CDFs of 17 u16 each, total
 * 4096, every symbol with a frequency of 1 or more. The weights are coded backwards from the state 2^23, a
 * stream per `chunk` weights (4-bit symbols, two per byte); the context of a symbol is the symbol before it,
 * or 16 at the start of a block of 32 and everywhere in a scale-only record. scale_stream: the tile stream of
 * the scales. A scale whose class is not in the model is coded with class 0: the decoder refuses the record
 * when it reaches that block. Writes the unpadded record and its size; -1 when it does not fit `capacity`. */
static int fuzz_wnr1_encode(const unsigned char *model, uint32_t classes, uint32_t chunk, uint32_t scale_only,
                            const unsigned char *scale_stream, uint32_t scale_encoded, const unsigned char *scales,
                            const unsigned char *weights, uint32_t weight_bytes, unsigned char *out, size_t capacity,
                            size_t *written) {
    uint64_t symbols = (uint64_t)weight_bytes * 2u, first = 0;
    uint32_t streams = (uint32_t)((symbols + chunk - 1u) / chunk), stream, payload = 0;
    size_t model_bytes = 512u + (size_t)classes * WEX_NR_CDF_BYTES;
    size_t meta = 32u + model_bytes + ((size_t)streams + 1u) * 4u, cursor;
    unsigned char *offsets;
    if (!classes || classes > 256u || !streams || (chunk != 256u && chunk != 512u && chunk != 1024u) ||
        capacity < meta || capacity - meta < scale_encoded) return -1;
    memcpy(out + 32u, model, model_bytes);
    offsets = out + 32u + model_bytes;
    fuzz_put_u32(offsets, 0);
    memcpy(out + meta, scale_stream, scale_encoded);
    cursor = meta + scale_encoded;
    for (stream = 0; stream < streams; stream++, first += chunk) {
        unsigned char tmp[2u * 1024u + 8u], *end = tmp + sizeof tmp, *p = end;
        uint32_t count = symbols - first < chunk ? (uint32_t)(symbols - first) : chunk, state = WEX_NR_L, j, length;
        for (j = count; j-- > 0;) {
            size_t position = (size_t)(first + j);
            uint32_t x = (weights[position / 2u] >> ((position & 1u) * 4u)) & 15u;
            uint32_t previous = !(j % 32u) || scale_only ? 16u
                                : (weights[(position - 1u) / 2u] >> (((position - 1u) & 1u) * 4u)) & 15u;
            uint32_t cls = wex_nr_u16(model + (size_t)scales[position / 32u] * 2u), start, frequency, maximum;
            const unsigned char *cdf;
            if (cls >= classes) cls = 0;
            cdf = model + 512u + ((size_t)cls * 17u + previous) * 34u;
            start = wex_nr_u16(cdf + x * 2u);
            frequency = wex_nr_u16(cdf + x * 2u + 2u) - start;
            maximum = ((WEX_NR_L >> 12) << 8) * frequency;
            while (state >= maximum) {
                if (p == tmp) return -1;
                *--p = (unsigned char)state; state >>= 8;
            }
            state = ((state / frequency) << 12) + state % frequency + start;
        }
        for (j = 0; j < 4u; j++) {
            if (p == tmp) return -1;
            *--p = (unsigned char)state; state >>= 8;
        }
        length = (uint32_t)(end - p);
        if (capacity - cursor < length) return -1;
        memcpy(out + cursor, p, length);
        cursor += length;
        payload += length;
        fuzz_put_u32(offsets + ((size_t)stream + 1u) * 4u, payload);
    }
    fuzz_put_u32(out, WEX_NR_MAGIC);
    fuzz_put_u32(out + 4u, scale_encoded);
    fuzz_put_u32(out + 8u, payload);
    fuzz_put_u32(out + 12u, classes);
    fuzz_put_u32(out + 16u, streams);
    fuzz_put_u32(out + 20u, chunk);
    fuzz_put_u32(out + 24u, scale_only ? 1u : 0u);
    fuzz_put_u32(out + 28u, 0);
    *written = cursor;
    return 0;
}

/* A valid CDF row of a WNR model (17 u16, from 0 to 4096, increasing): even frequencies, one symbol with
 * nearly all of them, or a few random shares. */
static void fuzz_wnr1_cdf(unsigned char *row, uint32_t *rng) {
    uint32_t frequency[16], left = 4096u - 16u, s, used = 0, r = fuzz_rng(rng);
    for (s = 0; s < 16u; s++) frequency[s] = 1u;
    if (r % 3u == 0) {
        for (s = 0; s < 16u; s++) frequency[s] += left / 16u;   /* 4080 = 16 * 255 */
    } else if (r % 3u == 1u) {
        frequency[(r >> 8) % 16u] += left;
    } else {
        for (s = 0; s < 4u; s++) {
            uint32_t q = fuzz_rng(rng), part = (q >> 4) % (left + 1u);
            frequency[q % 16u] += part;
            left -= part;
        }
        frequency[(r >> 8) % 16u] += left;
    }
    for (s = 0; s < 16u; s++) { fuzz_put_u16(row + s * 2u, used); used += frequency[s]; }
    fuzz_put_u16(row + 32u, used);
}

/* A WNR record (codec 3): decoded, edited and encoded again. The edits: weight bytes and blocks, scales, the
 * class of a scale the record uses (to another class, or one time in 8 to none), a CDF row, the chunk size,
 * the scale-only flag (set with the rows of every class made equal, cleared, or one time in 4 only flipped:
 * the parser refuses a scale-only record whose rows differ). A scale value that an edit of the scales brings
 * in and that the model has no class for gets a class 7 times in 8 (the encoder codes a scale without one
 * with class 0, and the decoder refuses the record when it reaches it). The result is `total`, or 0 with the
 * record unchanged: the record does not decode, or the new one takes other pages. */
static size_t fuzz_wnr1_mutate(unsigned char *record, size_t total, const WexEntry *e, uint32_t seed) {
    WexNeuralRans view;
    unsigned char *data, *scales, *weights, *model, *stream, *out, decided[256];
    size_t model_bytes, written = 0, packed, result = 0, i;
    uint32_t rng = seed, classes, chunk, scale_only, edits, c, p, v;
    if (!e->scale_bytes || !e->weight_bytes || wex_nr_parse(record, total, e->weight_bytes, &view)) return 0;
    memset(decided, 0, sizeof decided);   /* the scale values whose class this call set to none, or chose to keep so */
    classes = view.classes; chunk = view.chunk_symbols; scale_only = view.scale_only;
    model_bytes = 512u + (size_t)classes * WEX_NR_CDF_BYTES;
    data = (unsigned char *)malloc((size_t)e->scale_bytes + e->weight_bytes);
    model = (unsigned char *)malloc(model_bytes);
    stream = (unsigned char *)malloc(fuzz_tiles_bound(e->scale_bytes));
    out = (unsigned char *)malloc(total);
    if (!data || !model || !stream || !out) abort();
    scales = data; weights = data + e->scale_bytes;
    memcpy(model, view.mapping, model_bytes);
    if (wex_unpack_scale_tiles(view.scales, view.scale_encoded, scales, e->scale_bytes) ||
        wex_nr_decode_weights(&view, scales, e->weight_bytes, weights)) goto done;
    for (edits = 1u + fuzz_rng(&rng) % 4u; edits; edits--) {
        uint32_t r = fuzz_rng(&rng);
        switch (fuzz_rng(&rng) % 10u) {
        case 0: weights[r % e->weight_bytes] = (unsigned char)(r >> 16); break;
        case 1: weights[r % e->weight_bytes] ^= (unsigned char)(1u << ((r >> 16) % 8u)); break;
        case 2:   /* a block of 32 weights copied from another block */
            if (e->weight_bytes >= 32u)
                memmove(weights + (size_t)(r % (e->weight_bytes / 16u)) * 16u,
                        weights + (size_t)((r >> 12) % (e->weight_bytes / 16u)) * 16u, 16u);
            break;
        case 3: case 4: fuzz_scales_edit(scales, e->scale_bytes, &rng); break;
        case 5:
            v = scales[r % e->scale_bytes];
            if ((r >> 12) % 8u) fuzz_put_u16(model + (size_t)v * 2u, (r >> 16) % classes);
            else { fuzz_put_u16(model + (size_t)v * 2u, 65535u); decided[v] = 1; }
            break;
        case 6:
            c = r % classes; p = (r >> 8) % 17u;
            fuzz_wnr1_cdf(model + 512u + ((size_t)c * 17u + p) * 34u, &rng);
            if (scale_only)
                for (p = 0; p < 17u; p++)
                    memmove(model + 512u + ((size_t)c * 17u + p) * 34u, model + 512u + ((size_t)c * 17u + (r >> 8) % 17u) * 34u, 34u);
            break;
        case 7: chunk = 256u << (r % 3u); break;
        case 8:
            if (!(r % 4u)) scale_only = !scale_only;
            else if (r % 4u == 1u) scale_only = 0;
            else {
                scale_only = 1u;
                for (c = 0; c < classes; c++)
                    for (p = 0; p < 16u; p++)
                        memcpy(model + 512u + ((size_t)c * 17u + p) * 34u, model + 512u + ((size_t)c * 17u + 16u) * 34u, 34u);
            }
            break;
        default: weights[r % e->weight_bytes] = (unsigned char)(r & 0x10000u ? 0x88u : 0u); break;
        }
    }
    for (i = 0; i < e->scale_bytes; i++) {   /* a new scale value without a class: one, 7 times in 8 */
        v = scales[i];
        if (decided[v] || wex_nr_u16(model + (size_t)v * 2u) != 65535u) continue;
        decided[v] = 1;
        if (fuzz_rng(&rng) % 8u) fuzz_put_u16(model + (size_t)v * 2u, fuzz_rng(&rng) % classes);
    }
    packed = fuzz_tiles_pack(scales, e->scale_bytes, stream, &rng);
    if (!fuzz_wnr1_encode(model, classes, chunk, scale_only, stream, (uint32_t)packed, scales, weights, e->weight_bytes,
                          out, total, &written) &&
        ((written + WEX_ALIGN - 1u) & ~(size_t)(WEX_ALIGN - 1u)) == total) {
        memset(record, 0, total);
        memcpy(record, out, written);
        result = total;
    }
done:
    free(out);
    free(stream);
    free(model);
    free(data);
    return result;
}

/* The records each structured mutation replaced, by codec (1 scale tiles .. 5 WNR3), and those of them that
 * the store's checks refuse, for the exit line. A record whose decoded size is above FUZZ_LEGACY_CHECK_MAX is
 * not checked (and counted in neither). */
static unsigned long long fuzz_legacy_replaced[6], fuzz_legacy_refused[6];
#define FUZZ_LEGACY_CHECK_MAX (UINT64_C(64) << 20)

/* 1 when entry `index` of a container that opens would be served with `record` as its stored bytes: the owner
 * check, then wex_decode_record or wex_decode_nr2_record, as fuzz_store_wex.c expects of the store. 0 refused,
 * -1 not checked. */
static int fuzz_legacy_accepted(const WexHeader *h, const WexEntry *e, size_t index, const unsigned char *record,
                                const WexNr2Model *model, int has_model) {
    uint64_t raw, aligned;
    size_t at;
    unsigned char *slab;
    int ok;
    if (wex_entry_raw_sizes(e, &raw, &aligned) || aligned > FUZZ_LEGACY_CHECK_MAX) return -1;
    at = wex_record_read_offset(e);
    if (at > aligned || e->total_bytes > aligned - at) return 0;
    slab = (unsigned char *)calloc(1, (size_t)aligned);
    if (!slab) abort();
    memcpy(slab + at, record, e->total_bytes);
    if (!wex_nr3_owner_ok(e, slab + at, (uint32_t)(index / h->num_experts), (uint32_t)(index % h->num_experts), h->flags))
        ok = 0;
    else if (!wex_entry_nr_model(e)) ok = !wex_decode_record(e, slab, (size_t)aligned);
    else ok = has_model && wex_read_u32_le(slab + at + 4u) == model->id && !wex_decode_nr2_record(e, slab, (size_t)aligned, model);
    free(slab);
    return ok;
}

/* One coded record of a legacy container held in memory, picked by the seed and mutated in place with the
 * mutation of its codec: scale tiles, palette, WNR, WNR2 or WNR3. The container must validate as the store
 * requires (header, table, and the model of a WNR2 or WNR3 container). 1 when a record was replaced. 0, with
 * the file unchanged: not such a container, no coded record, the new record takes other pages (the records
 * of a WEX3 file are contiguous, so its size is fixed), or it is the same bytes. A record that two entries
 * share (WEX3 deduplication) changes for both. */
static int fuzz_legacy_mutate(unsigned char *file, size_t size, uint32_t seed) {
    const WexHeader *h = (const WexHeader *)file;
    const WexEntry *entries, *entry = NULL;
    WexNr2Model model;
    WexNr2Layout layout;
    unsigned char *work;
    uint64_t offset, aligned;
    uint32_t bytes, id, codec;
    size_t table_bytes, aligned_table, count, i, picked = 0, n = 0;
    int has_model = 0;
    if (size < WEX_ALIGN || wex_validate_header(h, size, 0, 0, &table_bytes, &aligned_table) ||
        wex_validate_table(h, (const WexEntry *)(file + h->table_offset), size)) return 0;
    if (h->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3)) {
        if (wex_nr2_model_location(h, file, &offset, &bytes, &id) || wex_aligned_size(bytes, &aligned) ||
            wex_nr2_model_open(file + offset, bytes, &model) || model.id != id) return 0;
        has_model = 1;
    }
    entries = (const WexEntry *)(file + h->table_offset);
    count = table_bytes / sizeof(*entries);
    for (i = 0; i < count; i++)   /* one of the coded entries, picked by the seed */
        if (entries[i].total_bytes && entries[i].flags && !((seed / 4u + i * 2654435761u) % ++picked)) entry = &entries[i];
    if (!entry) return 0;
    codec = entry->flags & WEX_ENTRY_CODEC_MASK;
    work = (unsigned char *)malloc(entry->total_bytes);
    if (!work) abort();
    memcpy(work, file + entry->file_offset, entry->total_bytes);
    if (wex_entry_nr_model(entry)) {
        if (has_model) {
            wex_nr2_layout(entry, &layout);
            n = fuzz_nrx_mutate(work, entry->total_bytes, entry->total_bytes, &layout, &model, wex_entry_nr_magic(entry), seed);
        }
    } else if (entry->flags == WEX_ENTRY_NEURAL_RANS) n = fuzz_wnr1_mutate(work, entry->total_bytes, entry, seed);
    else if (entry->flags == WEX_ENTRY_PALETTE3) n = fuzz_palette_mutate(work, entry->total_bytes, entry, seed);
    else if (codec == WEX_ENTRY_SCALE_TILES) n = fuzz_tiles_mutate(work, entry->total_bytes, entry, seed);
    if (n != entry->total_bytes || !memcmp(work, file + entry->file_offset, n)) n = 0;
    else {
        int accepted = fuzz_legacy_accepted(h, entry, (size_t)(entry - entries), work, &model, has_model);
        memcpy(file + entry->file_offset, work, n);
        if (accepted >= 0) {
            fuzz_legacy_replaced[codec]++;
            fuzz_legacy_refused[codec] += !accepted;
        }
    }
    free(work);
    return n != 0;
}

#endif /* WEX_FUZZ_LEGACY_MUTATOR_H */
