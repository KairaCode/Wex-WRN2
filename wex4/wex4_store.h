/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX4_STORE_H
#define WEX4_STORE_H

/* Loader of docs/WEX4-SPEC.md section 13.2: opens a WEX4 expert
 * container and gives the engine store exactly what it keeps today (the
 * layers x experts table of legacy WexEntry, the registered WNM2 model, the
 * legacy header flags, the slab size) plus the per-record SHA-256 for the
 * first-read check of section 8.2. Prototype of the change to
 * v4_wex_load_manifest; nothing here is on the per-token path. */
#include <stdio.h>
#include "wex4.h"

typedef struct {
    uint32_t layers, experts, legacy_flags;   /* WEX_HEADER_NEURAL_RANS3 / _CANONICAL_ZERO as the old checks expect */
    uint64_t slab_bytes;                      /* largest align(decoded) over the expert records */
    WexEntry *entries;                        /* layers * experts, absent experts all zero */
    unsigned char *hashes;                    /* 32 bytes per slot: SHA-256 of the stored record */
    unsigned char *model_blob;                /* WNM2 bytes, NULL when no record is WNR3 */
    WexNr2Model model;
    unsigned char root[32];
} Wex4Store;

static inline int wex4_store_read(FILE *f, uint64_t at, unsigned char *out, size_t n) {
#ifdef _WIN32
    if (_fseeki64(f, (long long)at, SEEK_SET)) return -1;
#else
    if (fseeko(f, (off_t)at, SEEK_SET)) return -1;
#endif
    return fread(out, 1, n, f) == n ? 0 : -1;
}
static inline void wex4_store_close(Wex4Store *s) {
    free(s->entries); free(s->hashes); free(s->model_blob);
    memset(s, 0, sizeof(*s));
}
/* expected_root: 32 bytes pinned in the configuration, or NULL. max_table_bytes: the caller's budget for the entry table. */
static inline int wex4_store_open(FILE *f, uint64_t file_bytes, const unsigned char *expected_root,
                                  uint64_t max_table_bytes, Wex4Store *s) {
    unsigned char page[WEX4_PAGE], *index = NULL;
    Wex4Header h;
    Wex4View v;
    uint64_t slots;
    uint32_t r, model = 0;
    int status;
    memset(s, 0, sizeof(*s));
    if (wex4_store_read(f, 0, page, sizeof(page))) return WEX4_E_HEADER;
    if ((status = wex4_parse_header(page, file_bytes, &h)) != 0) return status;
    if (expected_root && memcmp(expected_root, h.root, 32u)) return WEX4_E_HASH;
    slots = (uint64_t)h.layers * h.experts;
    if (!slots || slots * sizeof(WexEntry) > max_table_bytes) return WEX4_E_EXPERT;
    index = (unsigned char *)malloc((size_t)wex4_align(h.index_bytes));
    status = WEX4_E_INDEX;
    if (!index || wex4_store_read(f, h.index_offset, index, (size_t)wex4_align(h.index_bytes))) goto fail;
    if ((status = wex4_open(&h, index, (size_t)wex4_align(h.index_bytes), &v)) != 0) goto fail;
    s->layers = h.layers; s->experts = h.experts;
    memcpy(s->root, h.root, 32u);
    s->entries = (WexEntry *)calloc((size_t)slots, sizeof(WexEntry));
    s->hashes = (unsigned char *)calloc((size_t)slots, 32u);
    status = WEX4_E_ARG;
    if (!s->entries || !s->hashes) goto fail;
    for (r = 0; r < h.records; r++) {
        Wex4Record rec;
        size_t slot;
        wex4_get_record(&v, r, &rec);
        if (!(rec.flags & WEX4_REC_EXPERT)) continue;
        slot = (size_t)(rec.owner >> 16) * h.experts + (rec.owner & 0xffffu);
        if ((status = wex4_expert_entry(&v, r, &s->entries[slot])) != 0) goto fail;
        memcpy(s->hashes + slot * 32u, rec.sha, 32u);
        if (wex4_align(rec.decoded) > s->slab_bytes) s->slab_bytes = wex4_align(rec.decoded);
        if (rec.flags & WEX4_REC_VALUE_EXACT) s->legacy_flags |= WEX_HEADER_CANONICAL_ZERO;
        if (rec.codec == WEX4_CODEC_WNR3) {
            s->legacy_flags |= WEX_HEADER_NEURAL_RANS3;
            status = WEX4_E_BLOB;
            if (model && model != rec.blob) goto fail;   /* the store registers one model */
            model = rec.blob;
        }
    }
    /* the old store applies canonical zero to the whole container: a mixed file would be misread */
    if (s->legacy_flags & WEX_HEADER_CANONICAL_ZERO)
        for (r = 0; r < h.records; r++) {
            Wex4Record rec;
            wex4_get_record(&v, r, &rec);
            status = WEX4_E_RECORD;
            if (rec.codec == WEX4_CODEC_WNR3 && !(rec.flags & WEX4_REC_VALUE_EXACT)) goto fail;
        }
    if (model) {
        Wex4Blob b;
        unsigned char digest[32];
        uint64_t at;
        wex4_get_blob(&v, model - 1u, &b);
        s->model_blob = (unsigned char *)malloc((size_t)wex4_align(b.bytes));
        status = WEX4_E_BLOB;
        if (!s->model_blob || wex4_store_read(f, b.offset, s->model_blob, (size_t)wex4_align(b.bytes))) goto fail;
        wex4_sha256(s->model_blob, (size_t)b.bytes, digest);
        if (memcmp(digest, b.sha, 32u)) { status = WEX4_E_HASH; goto fail; }
        for (at = b.bytes; at < wex4_align(b.bytes); at++) if (s->model_blob[at]) goto fail;
        if (wex_nr2_model_open(s->model_blob, (size_t)b.bytes, &s->model) || s->model.id != b.id) goto fail;
    }
    free(index);
    return WEX4_OK;
fail:
    free(index);
    wex4_store_close(s);
    return status;
}
/* Section 8.2 first-read check: 0 when the stored bytes of (layer, expert) match the index. */
static inline int wex4_store_check(const Wex4Store *s, uint32_t layer, uint32_t expert, const unsigned char *stored) {
    const WexEntry *e;
    unsigned char digest[32];
    if (layer >= s->layers || expert >= s->experts) return WEX4_E_ARG;
    e = &s->entries[(size_t)layer * s->experts + expert];
    if (!e->total_bytes) return WEX4_E_ARG;
    wex4_sha256(stored, e->total_bytes, digest);
    return memcmp(digest, s->hashes + ((size_t)layer * s->experts + expert) * 32u, 32u) ? WEX4_E_HASH : WEX4_OK;
}

#endif /* WEX4_STORE_H */
