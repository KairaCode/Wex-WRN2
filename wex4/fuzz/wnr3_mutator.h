/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_WNR3_MUTATOR_H
#define WEX_FUZZ_WNR3_MUTATOR_H
/* A mutation that knows the WNR3 record (c/wex_neural_rans.h), for the custom mutators of the harnesses.
 *
 * Every stream of a WNR3 record starts at the state L + h23, where h23 is a 23-bit hash of the bytes the
 * stream decodes to, and the decoder requires that state at the end of the stream. A byte changed in the stored
 * record therefore fails the first stream it touches (the final state would have to equal a hash of the new
 * symbols), and the fuzzer sees no record that is accepted after a change. This mutation works on the other side of the codec: it decodes
 * the record, changes the decoded scales and weights and the record's parameters, and encodes the result again
 * with the reference encoder (wex_nrx_encode), which writes the right hash into every stream.
 *
 * What it changes, a few edits per call: one weight byte; one scale (any value, the edge values 0, 1, 254, 255,
 * or one step up or down); a block of 32 weights rewritten as an "even" block of maximum 7 (the ceil form), as
 * a block of maximum 5 (which keeps a matrix out of the ceil form), as all +0 or all -0, or as a copy of another
 * block; the weight and scale chunk sizes; the owner; the anchors.
 *
 * Two ways to encode. Canonical, 2 times in 3: canonical zero when the record has the flag, then
 * wex_nr3_ceil_forward, whose mask becomes the record's ceil flags; the decoders must accept such a record.
 * Loose, 1 time in 3: the four flag bits are drawn at random and the data is encoded as it is, so the record
 * may carry a -0 under the canonical-zero flag (refused at that symbol, after the streams before it passed)
 * or a ceil flag over data that was never halved (the inverse then runs on it, and refuses a scale of 0).
 *
 * fuzz_wnr3_mutate works on one record; fuzz_wnr3_mutate_legacy and fuzz_wnr3_mutate_wex4 pick a WNR3 record
 * inside a whole container and replace it in place. */
#include "wex4.h"   /* and through it wex_format.h and wex_neural_rans.h */

static uint32_t fuzz_rng(uint32_t *state) {
    uint32_t x = *state ? *state : UINT32_C(0x9e3779b9);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}

/* One edit of the decoded record. */
static void fuzz_wnr3_edit(const WexNr2Layout *layout, unsigned char *const scales[3], unsigned char *const weights[3],
                           uint32_t *rng) {
    static const unsigned char even[6] = {0, 2, 4, 5, 6, 7}, edge[4] = {0, 1, 254, 255};
    uint32_t m = fuzz_rng(rng) % 3u, b = fuzz_rng(rng) % layout->scale_count[m], r = fuzz_rng(rng), k;
    unsigned char *block = weights[m] + (size_t)b * 16u;
    switch (fuzz_rng(rng) % 9u) {
    case 0: block[r % 16u] = (unsigned char)(r >> 8); break;
    case 1: scales[m][b] = (unsigned char)r; break;
    case 2: scales[m][b] = edge[r % 4u]; break;
    case 3: scales[m][b] = (unsigned char)(scales[m][b] + (r & 1u ? 1u : 255u)); break;
    case 4:   /* an even block of maximum 7: the encoder stores it halved, with its scale + 1 */
        for (k = 0; k < 16u; k++) {
            uint32_t q = fuzz_rng(rng);
            block[k] = (unsigned char)(even[q % 6u] | (q & 8u) | (even[(q >> 8) % 6u] | ((q >> 4) & 8u)) << 4);
        }
        block[r % 16u] |= 7u;
        break;
    case 5:   /* a block of maximum 5: no block of its matrix is halved */
        for (k = 0; k < 16u; k++) {
            uint32_t q = fuzz_rng(rng);
            block[k] = (unsigned char)(q % 6u | (q & 8u) | ((q >> 8) % 6u | ((q >> 4) & 8u)) << 4);
        }
        block[r % 16u] = (unsigned char)((block[r % 16u] & 0xf8u) | 5u);
        break;
    case 6: memset(block, r & 1u ? 0x88 : 0, 16u); break;   /* all -0 or all +0 */
    case 7: {
        uint32_t other = r % layout->scale_count[m];
        memmove(block, weights[m] + (size_t)other * 16u, 16u);
        scales[m][b] = scales[m][other];
        break;
    }
    default: block[r % 16u] ^= (unsigned char)(1u << ((r >> 8) % 8u)); break;
    }
}

/* record: a WNR3 record of `stored` bytes in a buffer of `capacity` bytes (both multiples of 4096). On success
 * the buffer holds another record that the reference encoder wrote, zero-padded, and the result is its stored
 * size. 0, with the buffer unchanged: the input is not a record that decodes with this layout and model, or
 * the new record does not fit the capacity. */
static size_t fuzz_wnr3_mutate(unsigned char *record, size_t stored, size_t capacity, const WexNr2Layout *layout,
                               const WexNr2Model *model, uint32_t seed) {
    static const uint32_t chunks[7] = {256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u};
    unsigned char *data, *out, *scales[3], *weights[3], anchor[3];
    const unsigned char *const_scales[3], *const_weights[3];
    size_t bytes = 0, at = 0, written = 0, aligned = 0, weight_bytes = 0;
    uint32_t rng = seed, m, edits, weight_chunk, scale_chunk, flags, owner;
    WexNr2Record view;
    for (m = 0; m < 3u; m++) bytes += (size_t)layout->scale_count[m] + layout->symbol_count[m] / 2u;
    if (wex_nrx_parse(record, stored, layout, model->id, WEX_NR3_MAGIC, &view)) return 0;
    data = (unsigned char *)malloc(bytes ? bytes : 1u);
    out = (unsigned char *)malloc(capacity ? capacity : 1u);
    if (!data || !out) abort();
    for (m = 0; m < 3u; m++) { scales[m] = data + at; at += layout->scale_count[m]; }
    for (m = 0; m < 3u; m++) { weights[m] = data + at; at += layout->symbol_count[m] / 2u; weight_bytes += layout->symbol_count[m] / 2u; }
    if (wex_nr2_decode(&view, layout, model, scales, weights)) goto done;
    weight_chunk = view.weight_chunk; scale_chunk = view.scale_chunk;
    flags = view.flags; owner = view.owner;
    for (edits = 1u + fuzz_rng(&rng) % 4u; edits; edits--) fuzz_wnr3_edit(layout, scales, weights, &rng);
    if (!(fuzz_rng(&rng) % 4u)) weight_chunk = chunks[1u + fuzz_rng(&rng) % 6u];
    if (!(fuzz_rng(&rng) % 4u)) scale_chunk = chunks[fuzz_rng(&rng) % 7u];
    if (!(fuzz_rng(&rng) % 8u)) owner = fuzz_rng(&rng) & 1u ? fuzz_rng(&rng) : owner ^ (1u << fuzz_rng(&rng) % 32u);
    if (fuzz_rng(&rng) % 3u) {
        if (flags & WEX_NR3_FLAG_CZ) wex_nr3_canonical_zero(weights[0], weight_bytes);   /* the three matrices are contiguous */
        flags = (flags & WEX_NR3_FLAG_CZ) | wex_nr3_ceil_forward(layout, scales, weights, NULL);
    } else {
        flags = fuzz_rng(&rng) & (WEX_NR3_FLAG_CEIL_MASK | WEX_NR3_FLAG_CZ);
        if ((flags & WEX_NR3_FLAG_CZ) && fuzz_rng(&rng) % 4u) wex_nr3_canonical_zero(weights[0], weight_bytes);
    }
    for (m = 0; m < 3u; m++) {
        anchor[m] = fuzz_rng(&rng) % 8u ? wex_nr3_anchor(scales[m], layout->scale_count[m]) : (unsigned char)fuzz_rng(&rng);
        const_scales[m] = scales[m]; const_weights[m] = weights[m];
    }
    if (!wex_nrx_encode(model, layout, const_scales, const_weights, weight_chunk, scale_chunk, anchor, WEX_NR3_MAGIC,
                        flags, owner, out, capacity, &written)) {
        aligned = (written + 4095u) & ~(size_t)4095u;
        if (aligned > capacity) aligned = 0;
    }
    if (aligned) {
        memset(record, 0, aligned);
        memcpy(record, out, written);
    }
done:
    free(out);
    free(data);
    return aligned;
}

/* The same for one WNR3 record of a legacy container (WEX3 with WEX_HEADER_NEURAL_RANS3) held in memory: when
 * the header, the table and the model validate, one record picked by the seed is mutated in place. 1 when a
 * record was replaced. 0, with the file unchanged: not such a container, or the new record no longer takes
 * the same number of pages (the records of a WEX3 file are contiguous, so its size is fixed). */
static int fuzz_wnr3_mutate_legacy(unsigned char *file, size_t size, uint32_t seed) {
    const WexHeader *h = (const WexHeader *)file;
    const WexEntry *entries, *entry = NULL;
    WexNr2Model model;
    WexNr2Layout layout;
    unsigned char *work;
    uint64_t offset, aligned;
    uint32_t bytes, id;
    size_t table_bytes, aligned_table, count, i, picked = 0, n;
    if (size < WEX_ALIGN || wex_validate_header(h, size, 0, 0, &table_bytes, &aligned_table) ||
        !(h->flags & WEX_HEADER_NEURAL_RANS3) ||
        wex_validate_table(h, (const WexEntry *)(file + h->table_offset), size) ||
        wex_nr2_model_location(h, file, &offset, &bytes, &id) || wex_aligned_size(bytes, &aligned) ||
        wex_nr2_model_open(file + offset, bytes, &model) || model.id != id) return 0;
    entries = (const WexEntry *)(file + h->table_offset);
    count = table_bytes / sizeof(*entries);
    for (i = 0; i < count; i++)   /* one of the WNR3 entries, picked by the seed */
        if (entries[i].flags == WEX_ENTRY_NEURAL_RANS3 && !((seed / 4u + i * 2654435761u) % ++picked)) entry = &entries[i];
    if (!entry) return 0;
    wex_nr2_layout(entry, &layout);
    work = (unsigned char *)malloc(entry->total_bytes);
    if (!work) abort();
    memcpy(work, file + entry->file_offset, entry->total_bytes);
    n = fuzz_wnr3_mutate(work, entry->total_bytes, entry->total_bytes, &layout, &model, seed);
    if (n == entry->total_bytes) memcpy(file + entry->file_offset, work, n);
    free(work);
    return n == entry->total_bytes;
}

/* The same for one WNR3 record of a WEX4 file. `sealed` holds the same `size` bytes with a header page and an
 * index that validate: the file itself, or a copy whose header and index hashes were recomputed. The record is
 * read from `file` and replaced there; its SHA-256 in the index is left alone, so the new record is accepted
 * only where that hash is not checked (the second pass of fuzz_wex4_file, the Qwen3.8 reader without
 * COLI_WEX_VERIFY). 1 when a record was replaced, 0 with the file unchanged. */
static int fuzz_wnr3_mutate_wex4(unsigned char *file, const unsigned char *sealed, size_t size, uint32_t seed) {
    Wex4Header h;
    Wex4View v;
    Wex4Record rec, pick;
    Wex4Blob blob;
    WexEntry entry;
    WexNr2Model model;
    WexNr2Layout layout;
    unsigned char *work;
    uint32_t r, at = 0, picked = 0;
    size_t n;
    memset(&pick, 0, sizeof pick);
    if (size < WEX4_PAGE || wex4_parse_header(sealed, size, &h) ||
        wex4_open(&h, sealed + h.index_offset, (size_t)wex4_align(h.index_bytes), &v)) return 0;
    for (r = 0; r < h.records; r++) {
        wex4_get_record(&v, r, &rec);
        if (rec.codec == WEX4_CODEC_WNR3 && !((seed / 4u + r * 2654435761u) % ++picked)) { pick = rec; at = r; }
    }
    if (!picked || !pick.blob || wex4_expert_entry(&v, at, &entry)) return 0;
    wex4_get_blob(&v, pick.blob - 1u, &blob);
    if (wex_nr2_model_open(sealed + blob.offset, (size_t)blob.bytes, &model) || model.id != blob.id) return 0;
    wex_nr2_layout(&entry, &layout);
    work = (unsigned char *)malloc(pick.stored);
    if (!work) abort();
    memcpy(work, file + pick.offset, pick.stored);
    n = fuzz_wnr3_mutate(work, pick.stored, pick.stored, &layout, &model, seed);
    if (n == pick.stored) memcpy(file + pick.offset, work, n);
    free(work);
    return n == pick.stored;
}

#endif /* WEX_FUZZ_WNR3_MUTATOR_H */
