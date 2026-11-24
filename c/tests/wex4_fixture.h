/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* tests/wex4_fixture.h -- the WEX4 form of a WEX1 (raw) or WEX3/WNR3 expert container, built in
 * memory the way the in-place upgrade (Wex-WRN2 wex4_convert.py upgrade) lays it out: the old
 * bytes unchanged, then the old header page (blob kind 16) and the index, and a new header page.
 * The front sections are blobs in file order: the old table (kind 17), a WNR3 file's WNM2 model
 * (kind 1) and any gap before them or the records (kind 18). Every slot with a record becomes one
 * EXPERT record of 6 U8 tensors (w1..w3 scales, then w1..w3 weights): RAW, or WNR3 (codec 5,
 * naming the model blob; value-exact under canonical zero). Needs the records contiguous behind
 * the front sections, in slot order. For tests only. */
#ifndef KAIRA_TESTS_WEX4_FIXTURE_H
#define KAIRA_TESTS_WEX4_FIXTURE_H
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../wex4.h"

static const char *wex4_fixture_names;
static const uint32_t *wex4_fixture_name_at, *wex4_fixture_name_len;
static int wex4_fixture_by_name(const void *a, const void *b) {
    uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b, li = wex4_fixture_name_len[i], lj = wex4_fixture_name_len[j];
    int c = memcmp(wex4_fixture_names + wex4_fixture_name_at[i], wex4_fixture_names + wex4_fixture_name_at[j], li < lj ? li : lj);
    return c ? c : (li > lj) - (li < lj);
}

/* old: a WEX1 or WNR3 file of old_bytes; returns the WEX4 file (malloc) and its root as 64 hex digits. */
static unsigned char *wex4_fixture_from_wex1(const unsigned char *old, size_t old_bytes, size_t *bytes, char root[65]) {
    const WexHeader *h1 = (const WexHeader *)old;
    const WexEntry *table = (const WexEntry *)(old + h1->table_offset);
    uint32_t slots = h1->num_layers * h1->num_experts, n = 0, t, i, m;
    size_t table_bytes = (size_t)slots * sizeof(WexEntry), at = 0, name_bytes = 0;
    uint64_t total = 0, maximum = 0, raw = 0, aligned = 0, first_record = 0;
    char *names = malloc((size_t)slots * 6u * 48u);
    uint32_t *name_at = malloc((size_t)slots * 6u * 4u), *name_len = malloc((size_t)slots * 6u * 4u),
             *order = malloc((size_t)slots * 6u * 4u), *slot_of = malloc((size_t)slots * 4u);
    int nr3 = h1->magic == WEX_MAGIC_V3 && h1->version == 3u && (h1->flags & WEX_HEADER_NEURAL_RANS3);
    int cz = nr3 && (h1->flags & WEX_HEADER_CANONICAL_ZERO);
    assert((h1->magic == WEX_MAGIC_V1 || nr3) && names && name_at && name_len && order && slot_of);
    for (i = 0; i < slots; i++) {
        if (!table[i].total_bytes) continue;
        raw = (uint64_t)table[i].scale_bytes + table[i].weight_bytes;   /* decoded bytes */
        assert(nr3 || !wex_entry_raw_sizes(&table[i], &raw, &aligned));
        if (!n) first_record = table[i].file_offset;
        total += raw; if (raw > maximum) maximum = raw;
        for (m = 0; m < 6u; m++) {
            t = n * 6u + m;
            name_at[t] = (uint32_t)name_bytes;
            name_len[t] = (uint32_t)snprintf(names + name_bytes, (size_t)slots * 6u * 48u - name_bytes,
                                             "layers.%u.ffn.experts.%u.w%u.%s", i / h1->num_experts,
                                             i % h1->num_experts, m % 3u + 1u, m < 3u ? "scale" : "weight");
            name_bytes += name_len[t];
            order[t] = t;
        }
        slot_of[n++] = i;
    }
    wex4_fixture_names = names; wex4_fixture_name_at = name_at; wex4_fixture_name_len = name_len;
    qsort(order, (size_t)n * 6u, 4u, wex4_fixture_by_name);
    /* blobs: the front sections in file order, a gap blob wherever one does not start where the last
     * ended (up to the first record), then the old header page */
    struct { uint32_t kind, id; uint64_t offset, size; } blobs[6], front[3] = {{17u, 0u, h1->table_offset, table_bytes}};
    uint32_t n_front = 1, n_blobs = 0, model_blob = 0;
    if (nr3) {
        const unsigned char *d = old + WEX_NR2_DESCRIPTOR_OFFSET;
        assert(wex_read_u32_le(d) == WEX_NR2_DESCRIPTOR_MAGIC && wex_read_u32_le(d + 4) == WEX_NR2_MODEL_BYTES);
        front[1].kind = WEX4_BLOB_WNM2; front[1].id = wex_read_u32_le(d + 16);
        memcpy(&front[1].offset, d + 8, 8); front[1].size = WEX_NR2_MODEL_BYTES;
        if (front[1].offset < front[0].offset) { front[2] = front[0]; front[0] = front[1]; front[1] = front[2]; }
        n_front = 2;
    }
    front[n_front].kind = 0; front[n_front].id = 0; front[n_front].offset = first_record; front[n_front].size = 0;
    for (i = 0, at = WEX4_PAGE; i <= n_front; i++) {
        assert(front[i].offset >= at && !(front[i].offset % WEX4_PAGE));
        if (front[i].offset > at) {
            blobs[n_blobs].kind = 18u; blobs[n_blobs].id = 0; blobs[n_blobs].offset = at; blobs[n_blobs++].size = front[i].offset - at;
        }
        if (i == n_front) break;
        blobs[n_blobs++] = front[i];
        if (front[i].kind == WEX4_BLOB_WNM2) model_blob = n_blobs;   /* 1-based */
        at = (size_t)(front[i].offset + wex4_align(front[i].size));
    }
    blobs[n_blobs].kind = 16u; blobs[n_blobs].id = 0; blobs[n_blobs].offset = old_bytes; blobs[n_blobs++].size = WEX4_PAGE;
    at = 0;
    size_t index_bytes = (size_t)n_blobs * WEX4_BLOB_BYTES +(size_t)n * WEX4_RECORD_BYTES + (size_t)n * 6u * (WEX4_TENSOR_BYTES + 4u) + name_bytes;
    size_t index_offset = old_bytes + WEX4_PAGE, file = index_offset + (size_t)wex4_align(index_bytes);
    unsigned char *out = calloc(1, file), *x;
    assert(out && !(old_bytes % WEX4_PAGE));
    memcpy(out, old, old_bytes);
    memcpy(out + old_bytes, old, WEX4_PAGE);
    x = out + index_offset;
#define W4F_PUT(value, type) do { type t_ = (type)(value); memcpy(x + at, &t_, sizeof t_); at += sizeof t_; } while (0)
    for (i = 0; i < n_blobs; i++) {
        W4F_PUT(blobs[i].kind, uint32_t); W4F_PUT(blobs[i].id, uint32_t); W4F_PUT(blobs[i].offset, uint64_t); W4F_PUT(blobs[i].size, uint64_t);
        wex4_sha256(out + blobs[i].offset, (size_t)blobs[i].size, x + at); at += 32; W4F_PUT(0, uint64_t);
    }
    for (i = 0; i < n; i++) {
        const WexEntry *e = &table[slot_of[i]];
        int wnr3 = e->flags == WEX_ENTRY_NEURAL_RANS3;
        assert(wnr3 ? nr3 : !e->flags && !wex_entry_raw_sizes(e, &raw, &aligned) && aligned == e->total_bytes);
        assert(!i || e->file_offset == table[slot_of[i - 1]].file_offset + table[slot_of[i - 1]].total_bytes);
        W4F_PUT(e->file_offset, uint64_t); W4F_PUT(e->total_bytes, uint32_t);
        W4F_PUT((uint64_t)e->scale_bytes + e->weight_bytes, uint32_t); W4F_PUT(wnr3 ? WEX4_CODEC_WNR3 : WEX4_CODEC_RAW, uint16_t);
        W4F_PUT(WEX4_REC_EXPERT | (wnr3 && cz ? WEX4_REC_VALUE_EXACT : 0u), uint16_t); W4F_PUT(wnr3 ? model_blob : 0u, uint32_t);
        W4F_PUT((slot_of[i] / h1->num_experts) << 16 | slot_of[i] % h1->num_experts, uint32_t);
        W4F_PUT(i * 6u, uint32_t); W4F_PUT(6, uint32_t);
        wex4_sha256(out + e->file_offset, e->total_bytes, x + at); at += 32; W4F_PUT(0, uint32_t);
    }
    for (i = 0; i < n; i++) {
        const WexEntry *e = &table[slot_of[i]];
        for (m = 0; m < 6u; m++) {
            const uint16_t *shape = m < 3u ? e->scale_shapes[m] : e->weight_shapes[m - 3u];
            t = i * 6u + m;
            W4F_PUT(name_at[t], uint32_t); W4F_PUT(name_len[t], uint16_t); W4F_PUT(2, uint8_t); W4F_PUT(2, uint8_t);
            W4F_PUT(shape[0], uint64_t); W4F_PUT(shape[1], uint64_t);
            for (int d = 2; d < 8; d++) W4F_PUT(0, uint64_t);
            W4F_PUT(i, uint32_t); W4F_PUT(1, uint32_t);
            W4F_PUT(m < 3u ? e->scale_offsets[m] : e->weight_offsets[m - 3u], uint64_t);
            W4F_PUT((uint64_t)shape[0] * shape[1], uint64_t);
        }
    }
    for (t = 0; t < n * 6u; t++) W4F_PUT(order[t], uint32_t);
    memcpy(x + at, names, name_bytes); at += name_bytes;
    assert(at == index_bytes);
    x = out; at = 0;
    memset(out, 0, WEX4_PAGE);
    W4F_PUT(WEX4_MAGIC, uint32_t); W4F_PUT(4, uint16_t); W4F_PUT(0, uint16_t);
    W4F_PUT(cz ? WEX4_REQ_VALUE_EXACT : 0u, uint64_t); W4F_PUT(0, uint64_t);
    W4F_PUT(file, uint64_t); W4F_PUT(index_offset, uint64_t); W4F_PUT(index_bytes, uint64_t);
    W4F_PUT(n_blobs, uint32_t); W4F_PUT(n, uint32_t); W4F_PUT(n * 6u, uint32_t); W4F_PUT(h1->num_layers, uint32_t);
    W4F_PUT(h1->num_experts, uint32_t); W4F_PUT(maximum, uint32_t); W4F_PUT(name_bytes, uint64_t); W4F_PUT(0, uint64_t);
    W4F_PUT(total, uint64_t);
    wex4_sha256(out + index_offset, index_bytes, out + at);
    wex4_sha256(out, WEX4_HEADER_HASHED, out + WEX4_HEADER_HASHED);
#undef W4F_PUT
    for (i = 0; i < 32u; i++) snprintf(root + i * 2u, 3, "%02x", out[WEX4_HEADER_HASHED + i]);
    free(names); free(name_at); free(name_len); free(order); free(slot_of);
    *bytes = file;
    return out;
}

/* Reseal after an edit of the index: index hash, then header hash; root gets the new root. */
static void wex4_fixture_reseal(unsigned char *w4, char root[65]) {
    uint64_t index_offset, index_bytes;
    memcpy(&index_offset, w4 + 32, 8); memcpy(&index_bytes, w4 + 40, 8);
    wex4_sha256(w4 + index_offset, (size_t)index_bytes, w4 + 96);
    wex4_sha256(w4, WEX4_HEADER_HASHED, w4 + WEX4_HEADER_HASHED);
    for (int i = 0; i < 32; i++) snprintf(root + i * 2, 3, "%02x", w4[WEX4_HEADER_HASHED + i]);
}
#endif
