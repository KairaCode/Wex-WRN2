/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WEX4 prototype tools for wex4/wex4.py and wex4/test_wex4.py (ctypes). CPU only.
 * Build: gcc -O2 -shared -I c wex4/wex4_tools.c -o wex4_tools.dll
 * Everything works on a whole file held in memory: the tests use small files;
 * a real reader reads the header page, the index and single records instead.
 * With -DWEX4_FAST_NR3 -I wex4/research/fast-cpu-decode (and -O3 -march=haswell -fopenmp)
 * wex4_decode_record decodes WNR3 records with wex_fast_decode instead of the reference. */
#ifdef WEX4_FAST_NR3
/* c/wex4.h is the engine-side file and always names the reference decoder (WEX4_NR3_DECODE = wex_nr2_decode),
 * so the fast one is put under that name here. wex_fast_decode.h is included first: its own fallback
 * call to wex_nr2_decode stays the reference. build_and_test.sh checks that the call site changed. */
#include "wex_fast_decode.h"
#define wex_nr2_decode wex_fast_decode
#endif
#include "wex4.h"
#include "wex4_store.h"
#ifdef _WIN32
#include <windows.h>
#define WEX4_EXPORT __declspec(dllexport)
#else
#define WEX4_EXPORT __attribute__((visibility("default")))
#endif

WEX4_EXPORT uint32_t wex4_tools_abi(void) { return 1u; }
WEX4_EXPORT void wex4_sha256_bytes(const unsigned char *p, size_t n, unsigned char *out) { wex4_sha256(p, n, out); }
WEX4_EXPORT int wex4_bp_encode_record(const unsigned char *data, uint32_t decoded, uint32_t planes, uint32_t chunk_log2,
                                      uint32_t transform, uint32_t owner, unsigned char *out, size_t capacity,
                                      size_t *written) {
    return wex4_bp_encode(data, decoded, planes, chunk_log2, transform, owner, out, capacity, written);
}
WEX4_EXPORT int wex4_bp_decode_record(const unsigned char *record, size_t total, uint32_t decoded, uint32_t owner,
                                      unsigned char *out) {
    return wex4_bp_decode(record, total, decoded, owner, out);
}

/* Fuzzing: a buffer that ends right before an inaccessible page, so that any
 * read or write past its end faults instead of passing silently. */
static unsigned char *guard_alloc(size_t n, int guard) {
#ifdef _WIN32
    if (guard) {
        size_t body = (n + 4095u) & ~(size_t)4095u;
        unsigned char *base = (unsigned char *)VirtualAlloc(NULL, body + 4096u, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        DWORD old;
        if (!base) return NULL;
        VirtualProtect(base + body, 4096u, PAGE_NOACCESS, &old);
        return base + (body - n);
    }
#endif
    (void)guard;
    return (unsigned char *)malloc(n ? n : 1u);
}
static void guard_free(unsigned char *p, int guard) {
#ifdef _WIN32
    if (guard) { if (p) VirtualFree((void *)((uintptr_t)p & ~(uintptr_t)4095u), 0, MEM_RELEASE); return; }
#endif
    (void)guard;
    free(p);
}

static void put_u64(unsigned char *p, uint64_t v) { unsigned int i; for (i = 0; i < 8u; i++) p[i] = (unsigned char)(v >> (i * 8u)); }

/* flags: 1 = verify every blob and record SHA-256 (and zero blob padding),
 * 2 = decode every record and write the content digest: SHA-256 over, per
 * tensor in table order, {u16 name_len, name, u8 dtype, u8 rank, u64 shape[8],
 * u64 nbytes, data}. Returns 0 or a WEX4_E_* code. */
static int check_impl(const unsigned char *file, uint64_t len, int flags, unsigned char *digest, int guard) {
    Wex4Header h;
    Wex4View v;
    Wex4Sha content;
    WexNr2Model *models = NULL;
    unsigned char *out = NULL;
    uint32_t i, r;
    int status;
    if (!file || len < WEX4_PAGE) return WEX4_E_HEADER;
    if ((status = wex4_parse_header(file, len, &h)) != 0) return status;
    if ((status = wex4_open(&h, file + h.index_offset, (size_t)wex4_align(h.index_bytes), &v)) != 0) return status;
    models = (WexNr2Model *)calloc(h.blobs ? h.blobs : 1u, sizeof(*models));
    out = guard_alloc(h.max_decoded, guard);
    status = WEX4_E_ARG;
    if (!models || !out) goto finish;
    for (i = 0; i < h.blobs; i++) {
        Wex4Blob b;
        wex4_get_blob(&v, i, &b);
        if (flags & 1) {
            unsigned char d[32];
            uint64_t at;
            wex4_sha256(file + b.offset, (size_t)b.bytes, d);
            status = WEX4_E_HASH;
            if (memcmp(d, b.sha, 32u)) goto finish;
            status = WEX4_E_BLOB;
            for (at = b.bytes; at < wex4_align(b.bytes); at++) if (file[b.offset + at]) goto finish;
        }
        status = WEX4_E_BLOB;
        if (b.kind == WEX4_BLOB_WNM2 &&
            (wex_nr2_model_open(file + b.offset, (size_t)b.bytes, &models[i]) || models[i].id != b.id)) goto finish;
    }
    wex4_sha_init(&content);
    for (r = 0; r < h.records; r++) {
        Wex4Record rec;
        uint32_t k;
        wex4_get_record(&v, r, &rec);
        if (!(flags & 2)) {
            if (flags & 1) {
                unsigned char d[32];
                wex4_sha256(file + rec.offset, rec.stored, d);
                status = WEX4_E_HASH;
                if (memcmp(d, rec.sha, 32u)) goto finish;
            }
            continue;
        }
        if (guard) {   /* record and output each end at an inaccessible page: any overrun faults */
            unsigned char *in = guard_alloc(rec.stored, 1);
            guard_free(out, 1);
            out = guard_alloc(rec.decoded, 1);
            status = WEX4_E_ARG;
            if (in && out) {
                memcpy(in, file + rec.offset, rec.stored);
                status = wex4_decode_record(&v, r, in, out, rec.blob ? &models[rec.blob - 1u] : NULL, flags & 1);
            }
            guard_free(in, 1);
        } else
            status = wex4_decode_record(&v, r, file + rec.offset, out, rec.blob ? &models[rec.blob - 1u] : NULL, flags & 1);
        if (status) goto finish;
        if (!rec.tensor_count) wex4_sha_update(&content, out, rec.decoded);   /* next part of a split tensor */
        for (k = 0; k < rec.tensor_count; k++) {
            Wex4Tensor t;
            unsigned char meta[2 + 2 + 72];
            unsigned int d;
            wex4_get_tensor(&v, rec.first_tensor + k, &t);
            meta[0] = (unsigned char)t.name_len; meta[1] = (unsigned char)(t.name_len >> 8);
            wex4_sha_update(&content, meta, 2u);
            wex4_sha_update(&content, t.name, t.name_len);
            meta[0] = (unsigned char)t.dtype; meta[1] = (unsigned char)t.rank;
            for (d = 0; d < 8u; d++) put_u64(meta + 2u + d * 8u, t.shape[d]);
            put_u64(meta + 66u, t.nbytes);
            wex4_sha_update(&content, meta, 74u);
            wex4_sha_update(&content, out + t.offset, t.record_count == 1u ? (size_t)t.nbytes : rec.decoded);
        }
    }
    if (flags & 2) {
        Wex4Tensor t;
        uint32_t first_empty = h.tensors;
        /* zero-size tensors close the table: find where they start */
        while (first_empty && (wex4_get_tensor(&v, first_empty - 1u, &t), !t.nbytes)) first_empty--;
        for (i = first_empty; i < h.tensors; i++) {
            unsigned char meta[76];
            unsigned int d;
            wex4_get_tensor(&v, i, &t);
            meta[0] = (unsigned char)t.name_len; meta[1] = (unsigned char)(t.name_len >> 8);
            wex4_sha_update(&content, meta, 2u);
            wex4_sha_update(&content, t.name, t.name_len);
            meta[0] = (unsigned char)t.dtype; meta[1] = (unsigned char)t.rank;
            for (d = 0; d < 8u; d++) put_u64(meta + 2u + d * 8u, t.shape[d]);
            put_u64(meta + 66u, 0);
            wex4_sha_update(&content, meta, 74u);
        }
        if (digest) wex4_sha_final(&content, digest);
    }
    status = WEX4_OK;
finish:
    guard_free(out, guard);
    free(models);
    return status;
}
WEX4_EXPORT int wex4_check(const unsigned char *file, uint64_t len, int flags, unsigned char *digest) {
    return check_impl(file, len, flags, digest, 0);
}
/* Same, on a guard-paged copy with a guard-paged decode buffer. */
WEX4_EXPORT int wex4_fuzz_check(const unsigned char *file, uint64_t len, int flags, unsigned char *digest) {
    unsigned char *copy = guard_alloc((size_t)len, 1);
    int status;
    if (!copy) return WEX4_E_ARG;
    memcpy(copy, file, (size_t)len);
    status = check_impl(copy, len, flags, digest, 1);
    guard_free(copy, 1);
    return status;
}

/* Reader-style entry points: the caller holds the header page, the index (as
 * read from disk, 4096-aligned) and one stored record, never the whole file. */
static int view_of(const unsigned char *page, uint64_t file_bytes, const unsigned char *index, size_t index_len, Wex4View *v) {
    Wex4Header h;
    int status = wex4_parse_header(page, file_bytes, &h);
    return status ? status : wex4_open(&h, index, index_len, v);
}
/* 0 when the header page and the index are valid. */
WEX4_EXPORT int wex4_validate(const unsigned char *page, uint64_t file_bytes, const unsigned char *index, size_t index_len) {
    Wex4View v;
    return view_of(page, file_bytes, index, index_len, &v);
}
/* Decode record r from its stored bytes; blob = the bytes of its model blob or NULL. */
WEX4_EXPORT int wex4_decode_record_buf(const unsigned char *page, uint64_t file_bytes, const unsigned char *index,
                                       size_t index_len, uint32_t r, const unsigned char *stored,
                                       const unsigned char *blob, int check_sha, unsigned char *out) {
    Wex4View v;
    Wex4Record rec;
    WexNr2Model model, *use = NULL;
    int status = view_of(page, file_bytes, index, index_len, &v);
    if (status) return status;
    if (r >= v.h.records) return WEX4_E_ARG;
    wex4_get_record(&v, r, &rec);
    if (rec.blob && blob) {
        Wex4Blob b;
        wex4_get_blob(&v, rec.blob - 1u, &b);
        if (b.kind == WEX4_BLOB_WNM2) {
            if (wex_nr2_model_open(blob, (size_t)b.bytes, &model) || model.id != b.id) return WEX4_E_BLOB;
            use = &model;
        }
    }
    return wex4_decode_record(&v, r, stored, out, use, check_sha);
}
/* Legacy WEX3 entry table (layers * experts entries of 72 B, absent experts
 * all zero) rebuilt from the index: what the engine store keeps in memory. */
WEX4_EXPORT int wex4_expert_table(const unsigned char *page, uint64_t file_bytes, const unsigned char *index,
                                  size_t index_len, unsigned char *out) {
    Wex4View v;
    uint32_t r;
    int status = view_of(page, file_bytes, index, index_len, &v);
    if (status || !out) return status ? status : WEX4_E_ARG;
    memset(out, 0, (size_t)v.h.layers * v.h.experts * sizeof(WexEntry));
    for (r = 0; r < v.h.records; r++) {
        Wex4Record rec;
        WexEntry e;
        wex4_get_record(&v, r, &rec);
        if (!(rec.flags & WEX4_REC_EXPERT)) continue;
        if ((status = wex4_expert_entry(&v, r, &e)) != 0) return status;
        memcpy(out + ((size_t)(rec.owner >> 16) * v.h.experts + (rec.owner & 0xffffu)) * sizeof(e), &e, sizeof(e));
    }
    return WEX4_OK;
}

/* ---- In-process mutation fuzzer ------------------------------------------ */
static uint64_t rng_next(uint64_t *s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return *s * UINT64_C(2685821657736338717);
}
static void reseal(unsigned char *file, uint64_t len) {
    /* recompute index_sha256 and header_sha256 so that a mutated header or index reaches the validators */
    uint64_t at = wex4_u64(file + 32u), bytes = wex4_u64(file + 40u);
    if (at <= len && bytes <= len - at) wex4_sha256(file + at, (size_t)bytes, file + 96u);
    wex4_sha256(file, WEX4_HEADER_HASHED, file + WEX4_HEADER_HASHED);
}
/* mode 0: any byte of the file, hashes left alone, full verification: must be rejected (or be a no-op).
 * mode 1: bytes of compressed records only, record SHA-256 NOT checked: the codec's own checks must reject.
 * mode 2: header fields or index bytes, then both hashes are recomputed: the validators must hold (no fault);
 *         an accepted file must still decode with every record hash right.
 * counts: [0] rejected, [1] accepted with the original content digest, [2] accepted with another digest
 * (legal only in mode 2: e.g. a renamed tensor), [3] mutations that changed nothing. Returns 0, or -1 on setup error. */
WEX4_EXPORT int wex4_fuzz_run(const unsigned char *file, uint64_t len, uint64_t seed, uint32_t iterations, int mode,
                              uint64_t *counts) {
    unsigned char *work = guard_alloc((size_t)len, 1), base[32], digest[32];
    Wex4Header h;
    Wex4View v;
    uint64_t state = seed * UINT64_C(0x9e3779b97f4a7c15) + 1u, *spans = NULL, span_total = 0;
    uint32_t it, nspans = 0, r;
    int flags = mode == 1 ? 2 : 3;
    if (!work || !counts || check_impl(file, len, 3, base, 1)) { guard_free(work, 1); return -1; }
    memcpy(work, file, (size_t)len);
    wex4_parse_header(file, len, &h);
    wex4_open(&h, file + h.index_offset, (size_t)wex4_align(h.index_bytes), &v);
    spans = (uint64_t *)malloc(((size_t)h.records + 2u) * 2u * sizeof(*spans));
    if (!spans) { guard_free(work, 1); return -1; }
    if (mode == 0) { spans[0] = 0; spans[1] = len; nspans = 1; }
    else if (mode == 2) {
        spans[0] = 0; spans[1] = 128u; spans[2] = h.index_offset; spans[3] = h.index_bytes; nspans = 2;
    } else for (r = 0; r < h.records; r++) {
        Wex4Record rec;
        wex4_get_record(&v, r, &rec);
        if (rec.codec == WEX4_CODEC_RAW) continue;   /* raw data has no codec check: only the SHA-256 covers it */
        spans[nspans * 2u] = rec.offset; spans[nspans * 2u + 1u] = rec.stored; nspans++;
    }
    for (r = 0; r < nspans; r++) span_total += spans[r * 2u + 1u];
    memset(counts, 0, 4u * sizeof(*counts));
    if (!span_total) { free(spans); guard_free(work, 1); return -1; }
    for (it = 0; it < iterations; it++) {
        uint32_t edits = 1u + (uint32_t)(rng_next(&state) % 3u), e, changed = 0;
        uint64_t where[3];
        unsigned char saved[3][8];
        int status;
        for (e = 0; e < edits; e++) {
            uint64_t pick = rng_next(&state) % span_total, x = rng_next(&state);
            uint32_t kind = (uint32_t)(x % 5u), width = 1, j;
            for (r = 0; pick >= spans[r * 2u + 1u]; r++) pick -= spans[r * 2u + 1u];
            pick += spans[r * 2u];
            if (kind >= 3u) width = pick + 4u <= len ? 4u : 1u;
            if (pick + 8u > len) pick = len - 8u;
            where[e] = pick;
            memcpy(saved[e], work + pick, 8u);
            if (kind == 0u) work[pick] ^= (unsigned char)(1u << (x >> 8 & 7u));
            else if (kind == 1u) work[pick] = (unsigned char)(x >> 8);
            else if (kind == 2u) work[pick] = (unsigned char)(work[pick] + ((x >> 8 & 1u) ? 1u : 255u));
            else {   /* a 32-bit field: boundary values or a small delta */
                static const uint32_t magic[8] = {0u, 1u, 0xffffffffu, 0x7fffffffu, 0x80000000u, 4096u, 4095u, 0x10000u};
                uint32_t old = wex_nr_u32(work + pick), value = kind == 3u ? magic[x >> 8 & 7u] : old + (uint32_t)(x >> 8 & 15u) - 8u;
                for (j = 0; j < width; j++) work[pick + j] = (unsigned char)(value >> (j * 8u));
            }
        }
        if (mode == 2) reseal(work, len);
        changed = memcmp(work, file, (size_t)len) != 0;
        status = check_impl(work, len, flags, digest, 1);
        if (!changed) counts[3]++;
        else if (status) counts[0]++;
        else if (!memcmp(digest, base, 32u)) counts[1]++;
        else counts[2]++;
        for (e = edits; e-- > 0;) memcpy(work + where[e], saved[e], 8u);
        if (mode == 2) { memcpy(work, file, 128u); memcpy(work + WEX4_HEADER_HASHED, file + WEX4_HEADER_HASHED, 32u); }
    }
    free(spans);
    guard_free(work, 1);
    return 0;
}

/* Engine-style open (wex4_store.h) of a file on disk: copies the legacy table to out (capacity bytes),
 * info = {legacy flags, model id, slab bytes low, slab bytes high, records whose first-read SHA-256 check passed}.
 * root: 32 pinned bytes or NULL. Returns 0 or a WEX4_E_* code. */
WEX4_EXPORT int wex4_store_probe(const char *path, const unsigned char *root, unsigned char *out, uint64_t capacity,
                                 uint32_t *info) {
    FILE *f = fopen(path, "rb");
    Wex4Store s;
    uint64_t size;
    uint32_t i, good = 0;
    unsigned char *record;
    int status;
    if (!f) return WEX4_E_ARG;
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END); size = (uint64_t)_ftelli64(f);
#else
    fseeko(f, 0, SEEK_END); size = (uint64_t)ftello(f);
#endif
    status = wex4_store_open(f, size, root, capacity, &s);
    if (status) { fclose(f); return status; }
    memcpy(out, s.entries, (size_t)s.layers * s.experts * sizeof(WexEntry));
    record = (unsigned char *)malloc((size_t)s.slab_bytes);
    for (i = 0; record && i < s.layers * s.experts; i++) {
        const WexEntry *e = &s.entries[i];
        if (!e->total_bytes) continue;
        if (wex4_store_read(f, e->file_offset, record, e->total_bytes) ||
            wex4_store_check(&s, i / s.experts, i % s.experts, record)) { status = WEX4_E_HASH; break; }
        good++;
    }
    info[0] = s.legacy_flags; info[1] = s.model_blob ? s.model.id : 0u;
    info[2] = (uint32_t)s.slab_bytes; info[3] = (uint32_t)(s.slab_bytes >> 32); info[4] = good;
    free(record);
    wex4_store_close(&s);
    fclose(f);
    return status;
}
