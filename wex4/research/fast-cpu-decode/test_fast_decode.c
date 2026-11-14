/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Test and benchmark harness for wex_fast_decode.h against the reference wex_nr2_decode.
 *
 *   test_fast_decode equal  MODEL.wnm2 REC...        every variant, 1 and 8 threads, byte-identical to the reference
 *   test_fast_decode mutate MODEL.wnm2 N SEED REC... N mini-records at the decoders: valid by construction (some with
 *                                                    a semantic twist) or byte-mutated; same status, same bytes
 *   test_fast_decode bench  MODEL.wnm2 REPS REC...   timing table (best of REPS)
 *   test_fast_decode speed  MODEL.wnm2 RUNS REC...   one thread: median and best of RUNS passes over the records,
 *                                                    and the ratio of each median to the scalar kernel's (k0)
 *
 * REC = 72-byte WexEntry followed by the record bytes (extract.py writes them).
 * Every buffer a decoder touches sits between two pages without access (Windows: PAGE_NOACCESS; elsewhere
 * mmap + mprotect PROT_NONE): the record is copied WITHOUT its zero padding flush against the upper guard page
 * and outputs are flush against the upper or the lower one, so a read or write one byte outside faults. */
#if !defined(_WIN32) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE   /* glibc with -std=c11: clock_gettime, MAP_ANONYMOUS */
#endif
#include "wex_fast_decode.h"
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

typedef struct { unsigned char *lo, *hi; } Guard;   /* usable [lo, hi) */

static Guard guard_alloc(size_t bytes) {
    Guard g;
#ifdef _WIN32
    const size_t page = 4096u, pages = (bytes + page - 1u) / page + !bytes;
    DWORD old;
    unsigned char *base = (unsigned char *)VirtualAlloc(NULL, (pages + 2u) * 4096u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!base || !VirtualProtect(base, 4096, PAGE_NOACCESS, &old) ||
        !VirtualProtect(base + (pages + 1u) * 4096u, 4096, PAGE_NOACCESS, &old)) { fprintf(stderr, "guard_alloc\n"); exit(2); }
#else
    const size_t page = (size_t)sysconf(_SC_PAGESIZE), pages = (bytes + page - 1u) / page + !bytes;   /* 4, 16 or 64 KiB */
    unsigned char *base = (unsigned char *)mmap(NULL, (pages + 2u) * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == (unsigned char *)MAP_FAILED || mprotect(base, page, PROT_NONE) ||
        mprotect(base + (pages + 1u) * page, page, PROT_NONE)) { fprintf(stderr, "guard_alloc\n"); exit(2); }
#endif
    g.lo = base + page; g.hi = g.lo + pages * page;
    return g;
}

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

static uint64_t rng_state = 1;
static uint64_t rnd(void) {   /* xorshift64* */
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return rng_state * UINT64_C(2685821657736338717);
}
static uint32_t rnd_below(uint32_t n) { return (uint32_t)((rnd() >> 32) % n); }

static unsigned char *read_file(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    unsigned char *data;
    long n;
    if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    data = (unsigned char *)malloc((size_t)n + 1u);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    fclose(f);
    *bytes = (size_t)n;
    return data;
}

static int poison = 1;   /* cleared by bench */
static const WexFastModel *tables;   /* built once in main (NULL in a scalar build) */

/* ---- decoders under test ------------------------------------------------ */
typedef struct { const char *name; WexFastOptions o; } Variant;
static const Variant variants[] = {
    {"k0: reference loop per stream", {1, 0, 1, 0}},
#if WEXF_SIMD
    {"k1: " WEXF_KERNELS " search, 1 stream", {1, 1, 1, 0}},
    {"k1: 2 streams interleaved", {1, 1, 2, 0}},
    {"k1: 3 streams interleaved", {1, 1, 3, 0}},
    {"k1: 4 streams interleaved", {1, 1, 4, 0}},
    {"k1: 6 streams interleaved", {1, 1, 6, 0}},
    {"k1: 8 streams interleaved", {1, 1, 8, 0}},
    {"k1: 1 stream, fused", {1, 1, 1, 1}},
    {"k1: 2 streams, fused", {1, 1, 2, 1}},
    {"k1: 3 streams, fused", {1, 1, 3, 1}},
    {"k1: 4 streams, fused", {1, 1, 4, 1}},
    {"k1: 6 streams, fused", {1, 1, 6, 1}},
    {"k1: 8 streams, fused", {1, 1, 8, 1}},
#endif
#if WEXF_SIMD == 1
    {"k2: gather 8 streams", {1, 2, 1, 0}},
    {"k2: gather 16 streams", {1, 2, 2, 0}},
    {"k2: gather 24 streams", {1, 2, 3, 0}},
    {"k2: gather 8 streams, fused", {1, 2, 1, 1}},
    {"k2: gather 16 streams, fused", {1, 2, 2, 1}},
    {"k2: gather 24 streams, fused", {1, 2, 3, 1}},
#endif
};
#define VARIANTS (sizeof(variants) / sizeof(variants[0]))

/* Output buffers of one decoder: six guarded regions, used flush high or flush low. */
typedef struct { Guard gs[3], gw[3]; unsigned char *scales[3], *weights[3]; } Out;

static void out_alloc(Out *o, const WexNr2Layout *max) {
    unsigned m;
    for (m = 0; m < 3u; m++) { o->gs[m] = guard_alloc(max->scale_count[m]); o->gw[m] = guard_alloc(max->symbol_count[m] / 2u); }
}
static void out_place(Out *o, const WexNr2Layout *l, int high) {
    unsigned m;
    for (m = 0; m < 3u; m++) {
        o->scales[m] = high ? o->gs[m].hi - l->scale_count[m] : o->gs[m].lo;
        o->weights[m] = high ? o->gw[m].hi - l->symbol_count[m] / 2u : o->gw[m].lo;
    }
}
static int out_equal(const Out *a, const Out *b, const WexNr2Layout *l) {
    unsigned m;
    for (m = 0; m < 3u; m++)
        if (memcmp(a->scales[m], b->scales[m], l->scale_count[m]) ||
            memcmp(a->weights[m], b->weights[m], l->symbol_count[m] / 2u)) return 0;
    return 1;
}

/* Copy the unpadded record flush against the guard page and point the view at the copy. */
static void tighten(WexNr2Record *v, const unsigned char *record, Guard g) {
    size_t used = (size_t)v->metadata_bytes + v->scale_payload_bytes + v->weight_payload_bytes;
    unsigned char *copy = g.hi - used;
    memcpy(copy, record, used);
    v->lengths = copy + WEX_NR2_HEADER_BYTES;
    v->scale_payload = copy + v->metadata_bytes;
    v->weight_payload = v->scale_payload + v->scale_payload_bytes;
}

static int decode_ref(const WexNr2Record *v, const WexNr2Layout *l, const WexNr2Model *model, Out *o) {
    return wex_nr2_decode(v, l, model, o->scales, o->weights);
}
static int decode_fast(const WexNr2Record *v, const WexNr2Layout *l, const WexNr2Model *model, Out *o, WexFastOptions opt, int threads) {
    unsigned m;
    for (m = 0; m < 3u && poison; m++) {   /* no stale bytes from the previous variant */
        memset(o->scales[m], 0xa5, l->scale_count[m]); memset(o->weights[m], 0xa5, l->symbol_count[m] / 2u);
    }
    opt.threads = threads;
    return wex_fast_decode_with(tables, v, l, model, o->scales, o->weights, &opt);
}

typedef struct { WexEntry entry; WexNr2Layout layout; unsigned char *record; size_t total; } Real;

static void load_real(const char *path, Real *r) {
    size_t bytes;
    unsigned char *data = read_file(path, &bytes);
    uint64_t raw, aligned;
    if (bytes < sizeof(WexEntry)) exit(2);
    memcpy(&r->entry, data, sizeof(WexEntry));
    if (wex_entry_raw_sizes(&r->entry, &raw, &aligned) || r->entry.flags != WEX_ENTRY_NEURAL_RANS3 ||
        bytes != sizeof(WexEntry) + r->entry.total_bytes) { fprintf(stderr, "%s: not an entry + WNR3 record\n", path); exit(2); }
    wex_nr2_layout(&r->entry, &r->layout);
    r->record = data + sizeof(WexEntry); r->total = r->entry.total_bytes;
}

/* ---- equal: real records, every variant ---------------------------------- */
static int cmd_equal(const WexNr2Model *model, int count, char **paths) {
    int i, failures = 0;
    for (i = 0; i < count; i++) {
        Real r;
        WexNr2Record view;
        Out ref, fast;
        Guard rec;
        size_t v;
        int status, threads;
        load_real(paths[i], &r);
        if (wex_nr3_parse(r.record, r.total, &r.layout, model->id, &view)) { printf("%s: parse failed\n", paths[i]); return 1; }
        out_alloc(&ref, &r.layout); out_alloc(&fast, &r.layout);
        out_place(&ref, &r.layout, 1);
        rec = guard_alloc(r.total);
        tighten(&view, r.record, rec);
        status = decode_ref(&view, &r.layout, model, &ref);
        printf("%s: reference status %d, %u scale + %u weight streams, flags 0x%02x\n", paths[i], status,
               view.scale_streams, view.weight_streams, view.flags);
        if (status) return 1;
        for (v = 0; v < VARIANTS; v++)
            for (threads = 1; threads <= 8; threads += 7) {
                out_place(&fast, &r.layout, (int)((v + (size_t)threads) & 1u));
                status = decode_fast(&view, &r.layout, model, &fast, variants[v].o, threads);
                if (status || !out_equal(&ref, &fast, &r.layout)) {
                    printf("  MISMATCH %s, %d threads: status %d\n", variants[v].name, threads, status);
                    failures++;
                }
            }
        out_place(&fast, &r.layout, 1);   /* the default entry point */
        if (wex_fast_decode(&view, &r.layout, model, fast.scales, fast.weights) || !out_equal(&ref, &fast, &r.layout)) {
            printf("  MISMATCH wex_fast_decode default\n"); failures++;
        }
    }
    printf("equal: %d records x %d variants x {1, 8} threads + default, %d mismatches\n", count, (int)VARIANTS, failures);
    return failures != 0;
}

/* ---- mutate ---------------------------------------------------------------- */
#define MINI_MAX_BLOCKS 2500u
#define MINI_CAP ((3u * MINI_MAX_BLOCKS * 17u) * 4u + (1u << 17))
typedef struct {
    WexNr2Layout layout;
    unsigned char record[MINI_CAP];
    size_t total;
    int expect;   /* 0 accept, 1 reject, 2 whatever the reference says */
} Mini;

static unsigned char *src_scales[16][3], *src_weights[16][3];
static WexNr2Layout src_layout;
static int src_count;

static void put32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24); }
static void put16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }

/* A valid-by-construction WNR3 record over a random cut of real data, with optional semantic twists whose
 * hashes are still right (so the decoders run to the end and reach the CZ / ceil checks). */
static int make_mini(const WexNr2Model *model, Mini *mini, unsigned char *plain_s[3], unsigned char *plain_w[3]) {
    static unsigned char cs[3][MINI_MAX_BLOCKS], cw[3][MINI_MAX_BLOCKS * 16u];
    unsigned char *scales[3], *weights[3], anchor[3];
    const uint32_t kind = rnd_below(20);
    uint32_t m, i, flags, cz = 1, wchunk, schunk;
    size_t written;
    mini->expect = 0;
    for (m = 0; m < 3u; m++) {
        const uint32_t blocks = rnd_below(10) ? 40u + rnd_below(MINI_MAX_BLOCKS - 40u) : 1u + rnd_below(40);
        const uint32_t r = rnd_below((uint32_t)src_count), sm = rnd_below(3), at = rnd_below(src_layout.scale_count[sm] - blocks);
        mini->layout.scale_count[m] = blocks; mini->layout.symbol_count[m] = blocks * 32u;
        memcpy(cs[m], src_scales[r][sm] + at, blocks);
        memcpy(cw[m], src_weights[r][sm] + (size_t)at * 16u, (size_t)blocks * 16u);
        scales[m] = cs[m]; weights[m] = cw[m];
        if (kind == 10 || kind == 11)   /* noise: every context, long streams */
            for (i = 0; i < blocks * 16u; i++) {
                uint32_t v = (uint32_t)rnd() & 0xffu;
                if ((v & 15u) == 8u) v &= 0xf0u;
                if ((v >> 4) == 8u) v &= 0x0fu;
                cw[m][i] = (unsigned char)v;
                if (!(i % 16u)) cs[m][i / 16u] = (unsigned char)(kind == 10 ? 118u + rnd_below(12) : rnd());
            }
        if (kind == 12) memset(cw[m], rnd_below(2) ? 0 : 0x11, (size_t)blocks * 16u);   /* near-zero entropy */
        if (kind == 13 || kind == 14)   /* -0 nibbles: legal without the CZ flag (13), a CZ violation with it (14) */
            for (i = 0; i < 1u + rnd_below(4); i++) {
                const uint32_t at8 = rnd_below(blocks * 32u);
                cw[m][at8 / 2u] = (unsigned char)((cw[m][at8 / 2u] & (at8 & 1u ? 0x0fu : 0xf0u)) | (at8 & 1u ? 0x80u : 0x08u));
            }
    }
    if (kind == 13) cz = 0;
    if (kind == 14) mini->expect = 1;
    for (m = 0; m < 3u; m++) {
        memcpy(plain_s[m], cs[m], mini->layout.scale_count[m]);
        memcpy(plain_w[m], cw[m], mini->layout.symbol_count[m] / 2u);
    }
    flags = wex_nr3_ceil_forward(&mini->layout, scales, weights, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
    if (kind == 15) {   /* ceil underflow: a coded block of maximum 5 with coded scale 0 in a ceil matrix */
        const uint32_t block = rnd_below(mini->layout.scale_count[m = rnd_below(3)]);
        for (i = 0; i < 16u; i++) cw[m][block * 16u + i] = (unsigned char)(rnd_below(6) | rnd_below(6) << 4);
        cw[m][block * 16u + rnd_below(16)] = 0x15;
        cs[m][block] = 0;
        flags |= 1u << m;
        mini->expect = 1;
    }
    if (kind == 16) { flags ^= 1u + rnd_below(7); mini->expect = 2; }   /* ceil flags that do not match the data */
    for (m = 0; m < 3u; m++) anchor[m] = kind == 17 ? (unsigned char)rnd() : wex_nr3_anchor(scales[m], mini->layout.scale_count[m]);
    wchunk = 512u << rnd_below(rnd_below(8) ? 3 : 6);
    schunk = 256u << rnd_below(rnd_below(8) ? 3 : 7);
    if (wex_nrx_encode(model, &mini->layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                       wchunk, schunk, anchor, WEX_NR3_MAGIC, flags, (uint32_t)rnd(), mini->record, MINI_CAP - 4096u, &written)) return -1;
    mini->total = (written + 4095u) & ~(size_t)4095u;
    memset(mini->record + written, 0, mini->total - written);
    return 0;
}

/* Byte-level damage to a valid record; the parser decides whether the decoders get to see it. */
static void mutate(Mini *x) {
    WexNr2Record v;
    unsigned char *r = x->record, *lengths = r + WEX_NR2_HEADER_BYTES;
    uint32_t streams, used, meta, s, k, a, b, n;
    if (wex_nr3_parse(r, x->total, &x->layout, 0, &v)) return;
    streams = v.scale_streams + v.weight_streams; meta = v.metadata_bytes;
    used = meta + v.scale_payload_bytes + v.weight_payload_bytes;
    switch (rnd_below(16)) {
    case 0: r[rnd_below(used)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 1: r[rnd_below(used)] = (unsigned char)rnd(); break;
    case 2: r[4u + rnd_below(28)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 3: r[WEX_NR2_HEADER_BYTES + rnd_below(2u * streams)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 4:   /* move the boundary between two neighbouring streams */
        if (streams < 2u) break;
        s = rnd_below(streams - 1u); k = 1u + rnd_below(8);
        a = wex_nr_u16(lengths + 2u * s); b = wex_nr_u16(lengths + 2u * s + 2u);
        if (rnd_below(2)) { if (a < 4u + k || b + k > 65535u) break; a -= k; b += k; }
        else { if (b < 4u + k || a + k > 65535u) break; a += k; b -= k; }
        if (s + 1u == v.scale_streams) {   /* the boundary between the two payloads moves too */
            put32(r + 8u, v.scale_payload_bytes + a - wex_nr_u16(lengths + 2u * s));
            put32(r + 12u, v.weight_payload_bytes + b - wex_nr_u16(lengths + 2u * s + 2u));
        }
        put16(lengths + 2u * s, a); put16(lengths + 2u * s + 2u, b);
        break;
    case 5:   /* truncate the last stream */
        a = wex_nr_u16(lengths + 2u * (streams - 1u)); k = 1u + rnd_below(6);
        if (a < 4u + k) break;
        put16(lengths + 2u * (streams - 1u), a - k); put32(r + 12u, v.weight_payload_bytes - k);
        memset(r + used - k, 0, k);
        x->total = ((size_t)used - k + 4095u) & ~(size_t)4095u;
        break;
    case 6:   /* extend the last stream into the zero padding */
        a = wex_nr_u16(lengths + 2u * (streams - 1u)); k = 1u + rnd_below(6);
        if (a + k > 65535u || used + k > x->total) break;
        put16(lengths + 2u * (streams - 1u), a + k); put32(r + 12u, v.weight_payload_bytes + k);
        break;
    case 7: {   /* swap two neighbouring streams (bytes and lengths) */
        unsigned char tmp[65536];
        uint32_t at = meta;
        if (streams < 2u) break;
        s = rnd_below(streams - 1u);
        if (s + 1u == v.scale_streams) break;
        for (k = 0; k < s; k++) at += wex_nr_u16(lengths + 2u * k);
        a = wex_nr_u16(lengths + 2u * s); b = wex_nr_u16(lengths + 2u * s + 2u);
        memcpy(tmp, r + at, a); memmove(r + at, r + at + a, b); memcpy(r + at + b, tmp, a);
        put16(lengths + 2u * s, b); put16(lengths + 2u * s + 2u, a);
        break;
    }
    case 8:   /* header fields the parser accepts: anchors, flags, owner */
        k = rnd_below(4);
        if (k < 3u) r[24u + k] = (unsigned char)rnd();
        else if (rnd_below(2)) r[27] ^= (unsigned char)(1u << rnd_below(4));
        else put32(r + 28u, (uint32_t)rnd());
        break;
    case 9: {   /* a stream's initial state */
        uint32_t at = meta;
        s = rnd_below(streams);
        for (k = 0; k < s; k++) at += wex_nr_u16(lengths + 2u * k);
        if (rnd_below(2)) put32(r + at, (uint32_t)rnd());
        else r[at] = (unsigned char)(rnd_below(2) ? 0x80u | rnd() : 0);   /* state >= 2^31 or < 2^24 */
        break;
    }
    case 10:   /* a run of bytes in the payload */
        n = 1u + rnd_below(64); a = meta + rnd_below(used - meta);
        k = rnd_below(3);
        for (b = a; b < used && b < a + n; b++) r[b] = (unsigned char)(k == 0 ? 0 : k == 1 ? 0xff : rnd());
        break;
    case 11: for (n = 2u + rnd_below(4); n; n--) r[meta + rnd_below(used - meta)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 12: {   /* stray bytes appended to one stream: it decodes to the right state, only "pointer at end" fails */
        uint32_t at = meta;
        s = rnd_below(streams); k = 1u + rnd_below(3);
        a = wex_nr_u16(lengths + 2u * s);
        if (a + k > 65535u || (((size_t)used + k + 4095u) & ~(size_t)4095u) > MINI_CAP) break;
        for (b = 0; b <= s; b++) at += wex_nr_u16(lengths + 2u * b);
        memmove(r + at + k, r + at, used - at);
        for (b = 0; b < k; b++) r[at + b] = (unsigned char)(rnd_below(2) ? 0 : rnd());
        put16(lengths + 2u * s, a + k);
        if (s < v.scale_streams) put32(r + 8u, v.scale_payload_bytes + k); else put32(r + 12u, v.weight_payload_bytes + k);
        used += k;
        x->total = ((size_t)used + 4095u) & ~(size_t)4095u;
        memset(r + used, 0, x->total - used);
        break;
    }
    case 13: case 14: {   /* drop the last byte of a stream whose successor starts with the same byte: the reference
                             runs out of bytes; a decoder that reads ahead finds the right byte in the wrong stream */
        uint32_t at = meta, pick = rnd_below(streams), found = 0, found_at = 0;
        for (s = 0; s + 1u < streams; s++) {
            a = wex_nr_u16(lengths + 2u * s);
            at += a;
            if (a > 4u && r[at - 1u] == r[at] && (!found_at || s <= pick)) { found = s; found_at = at; }
        }
        if (!found_at) break;
        memmove(r + found_at - 1u, r + found_at, used - found_at);
        put16(lengths + 2u * found, wex_nr_u16(lengths + 2u * found) - 1u);
        if (found < v.scale_streams) put32(r + 8u, v.scale_payload_bytes - 1u); else put32(r + 12u, v.weight_payload_bytes - 1u);
        used--;
        r[used] = 0;
        x->total = ((size_t)used + 4095u) & ~(size_t)4095u;
        break;
    }
    default:   /* the last bytes of the record: the checked path and the hard end */
        r[used - 1u - rnd_below(used - meta < 80u ? used - meta : 80u)] ^= (unsigned char)(1u << rnd_below(8));
    }
}

static int cmd_mutate(const WexNr2Model *model, long wanted, int count, char **paths) {
    enum { POOL = 64 };
    static Mini pool[POOL], work;
    WexNr2Layout max;
    Out ref, fast;
    Guard rec = guard_alloc(MINI_CAP), plain[6];
    unsigned char *plain_s[3], *plain_w[3];
    long reached = 0, parse_rejected = 0, accepted = 0, rejected = 0, valid_made = 0, runs = 0, kinds[3] = {0, 0, 0};
    long unchanged = 0, mutated = 0, mutated_rejected = 0;
    int i, pool_used = 0;
    unsigned m;
    for (i = 0; i < count && i < 16; i++) {   /* real decoded data to cut mini-records from */
        Real r;
        WexNr2Record view;
        load_real(paths[i], &r);
        for (m = 0; m < 3u; m++) {
            src_scales[i][m] = (unsigned char *)malloc(r.layout.scale_count[m]);
            src_weights[i][m] = (unsigned char *)malloc(r.layout.symbol_count[m] / 2u);
        }
        if (wex_nr3_parse(r.record, r.total, &r.layout, model->id, &view) ||
            wex_fast_decode(&view, &r.layout, model, src_scales[i], src_weights[i])) { printf("%s: decode failed\n", paths[i]); return 1; }
        for (m = 0; m < 3u; m++) {   /* a record without the CZ flag can hold -0 nibbles: make_mini wants none in its source */
            size_t k;
            for (k = 0; k < r.layout.symbol_count[m] / 2u; k++) {
                unsigned char *b = src_weights[i][m] + k;
                if ((*b & 15u) == 8u) *b &= 0xf0u;
                if ((*b >> 4) == 8u) *b &= 0x0fu;
            }
        }
        src_layout = r.layout;
        free(r.record - sizeof(WexEntry));
    }
    src_count = i;
    for (m = 0; m < 3u; m++) {
        max.scale_count[m] = MINI_MAX_BLOCKS; max.symbol_count[m] = MINI_MAX_BLOCKS * 32u;
        plain[m] = guard_alloc(MINI_MAX_BLOCKS); plain[3 + m] = guard_alloc(MINI_MAX_BLOCKS * 16u);
        plain_s[m] = plain[m].lo; plain_w[m] = plain[3 + m].lo;
    }
    out_alloc(&ref, &max); out_alloc(&fast, &max);
    while (reached < wanted) {
        WexNr2Record view;
        int status, fresh = pool_used < POOL || !rnd_below(4);
        size_t v;
        if (fresh) {   /* a new record, valid by construction (possibly with a semantic twist) */
            if (make_mini(model, &work, plain_s, plain_w)) continue;
            valid_made++;
        } else {
            const Mini *base = &pool[rnd_below((uint32_t)pool_used)];
            work = *base;
            mutate(&work);
            if (work.total == base->total && !memcmp(work.record, base->record, work.total)) { unchanged++; continue; }   /* not counted */
        }
        if (wex_nr3_parse(work.record, work.total, &work.layout, model->id, &view)) {
            if (fresh) { printf("a constructed record did not parse\n"); return 1; }
            parse_rejected++;
            continue;
        }
        reached++;
        tighten(&view, work.record, rec);
        out_place(&ref, &work.layout, (int)(reached & 1));
        status = decode_ref(&view, &work.layout, model, &ref);
        if (status) rejected++; else accepted++;
        if (!fresh) { mutated++; mutated_rejected += status != 0; }
        if (fresh) {
            kinds[work.expect]++;
            if ((work.expect == 0 && status) || (work.expect == 1 && !status)) { printf("constructed record: reference status %d, expected %d\n", status, work.expect); return 1; }
            if (work.expect == 0) {   /* the round trip must give the data back */
                for (m = 0; m < 3u; m++)
                    if (memcmp(ref.scales[m], plain_s[m], work.layout.scale_count[m]) ||
                        memcmp(ref.weights[m], plain_w[m], work.layout.symbol_count[m] / 2u)) { printf("round trip differs\n"); return 1; }
                pool[pool_used < POOL ? pool_used++ : (int)rnd_below(POOL)] = work;
            }
        }
        out_place(&fast, &work.layout, (int)(reached >> 1 & 1));   /* the default entry point: tables built per call */
        if (wex_fast_decode(&view, &work.layout, model, fast.scales, fast.weights) != status ||
            (!status && !out_equal(&ref, &fast, &work.layout))) { printf("MISMATCH at case %ld (wex_fast_decode default)\n", reached); return 1; }
        for (v = 0; v < VARIANTS; v++) {
            const int threads = (reached + (long)v) % 5 ? 1 : 3;
            int got;
            out_place(&fast, &work.layout, (int)((reached + (long)v) & 1));
            got = decode_fast(&view, &work.layout, model, &fast, variants[v].o, threads);
            runs++;
            if (got != status || (!status && !out_equal(&ref, &fast, &work.layout))) {
                printf("MISMATCH at case %ld (%s, %d threads): reference %d, fast %d\n", reached, variants[v].name, threads, status, got);
                return 1;
            }
        }
    }
    printf("mutate: %ld records reached the decoders (%ld accepted, %ld rejected by the reference)\n", reached, accepted, rejected);
    printf("        %ld of them byte-mutated (%ld rejected, %ld accepted by the reference); %ld more mutated records stopped at the parser,\n"
           "        %ld mutations that changed nothing were dropped\n", mutated, mutated_rejected, mutated - mutated_rejected, parse_rejected, unchanged);
    printf("        %ld constructed records (expected accept %ld, expected reject %ld, either %ld), %ld fast decodes over %d variants, 0 mismatches\n",
           valid_made, kinds[0], kinds[1], kinds[2], runs, (int)VARIANTS);
    return 0;
}

/* ---- bench ----------------------------------------------------------------- */
static int cmd_bench(const WexNr2Model *model, int reps, int count, char **paths) {
    Real real[16];
    WexNr2Record view[16];
    Out out, par[8];
    double raw = 0, best, t;
    int i, rep, threads, n = count < 16 ? count : 16;
    size_t v;
    poison = 0;
    for (i = 0; i < n; i++) {
        load_real(paths[i], &real[i]);
        if (wex_nr3_parse(real[i].record, real[i].total, &real[i].layout, model->id, &view[i])) return 1;
        raw += real[i].entry.scale_bytes + real[i].entry.weight_bytes;
    }
    out_alloc(&out, &real[0].layout); out_place(&out, &real[0].layout, 0);
    for (i = 0; i < 8; i++) { out_alloc(&par[i], &real[0].layout); out_place(&par[i], &real[0].layout, 0); }
    raw /= n;
    printf("%d records, %.0f decoded bytes each, best of %d; MB = 10^6 bytes\n", n, raw, reps);
    printf("| decoder | 1 thread: ms/record | MB/s | 8 threads: ms/record | MB/s |\n|---|---:|---:|---:|---:|\n");
    for (best = 1e30, rep = 0; rep < reps; rep++) {
        t = now_ms();
        for (i = 0; i < n; i++) if (decode_ref(&view[i], &real[i].layout, model, &out)) return 1;
        t = (now_ms() - t) / n;
        if (t < best) best = t;
    }
    printf("| reference wex_nr2_decode | %.2f | %.1f |", best, raw / best / 1000.0);
#ifdef _OPENMP
    for (best = 1e30, rep = 0; rep < reps; rep++) {   /* the reference has no threads: 8 records at once instead */
        int bad = 0;
        t = now_ms();
#pragma omp parallel for num_threads(8) reduction(|:bad) schedule(static, 1)
        for (i = 0; i < 8; i++) bad |= decode_ref(&view[i % n], &real[i % n].layout, model, &par[i]);
        t = (now_ms() - t) / 8.0;
        if (bad) return 1;
        if (t < best) best = t;
    }
    printf(" %.2f (8 records at once) | %.1f |\n", best, raw / best / 1000.0);
#else
    printf(" - | - |\n");
#endif
    for (v = 0; v < VARIANTS; v++) {
        printf("| %s |", variants[v].name);
        for (threads = 1; threads <= 8; threads += 7) {
            for (best = 1e30, rep = 0; rep < reps; rep++) {
                t = now_ms();
                for (i = 0; i < n; i++) if (decode_fast(&view[i], &real[i].layout, model, &out, variants[v].o, threads)) return 1;
                t = (now_ms() - t) / n;
                if (t < best) best = t;
            }
            printf(" %.2f | %.1f |", best, raw / best / 1000.0);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("| wex_fast_decode_opt (k1, 4 streams, fused; tables built per call) |");
    for (threads = 1; threads <= 8; threads += 7) {
        const WexFastOptions def = {threads, 1, 4, 1};
        for (best = 1e30, rep = 0; rep < reps; rep++) {
            t = now_ms();
            for (i = 0; i < n; i++) if (wex_fast_decode_opt(&view[i], &real[i].layout, model, out.scales, out.weights, &def)) return 1;
            t = (now_ms() - t) / n;
            if (t < best) best = t;
        }
        printf(" %.2f | %.1f |", best, raw / best / 1000.0);
    }
    printf("\n");
    return 0;
}

/* ---- speed ----------------------------------------------------------------- */
static int by_value(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* One thread. A run is one pass over the records; a row is the reference (row 0) or a variant. */
static int cmd_speed(const WexNr2Model *model, int runs, int count, char **paths) {
    Real real[16];
    WexNr2Record view[16];
    Out out;
    double raw = 0, median[VARIANTS + 1u], best[VARIANTS + 1u], *t;
    int i, run, n = count < 16 ? count : 16;
    size_t row;
    if (runs < 1 || !(t = (double *)malloc((size_t)runs * sizeof(*t)))) return 2;
    poison = 0;
    for (i = 0; i < n; i++) {
        load_real(paths[i], &real[i]);
        if (wex_nr3_parse(real[i].record, real[i].total, &real[i].layout, model->id, &view[i])) return 1;
        raw += real[i].entry.scale_bytes + real[i].entry.weight_bytes;
    }
    out_alloc(&out, &real[0].layout); out_place(&out, &real[0].layout, 0);
    raw /= n;
    for (row = 0; row <= VARIANTS; row++) {
        for (run = -1; run < runs; run++) {   /* run -1 warms up and is not counted */
            const double start = now_ms();
            for (i = 0; i < n; i++)
                if (row ? decode_fast(&view[i], &real[i].layout, model, &out, variants[row - 1u].o, 1)
                        : decode_ref(&view[i], &real[i].layout, model, &out)) return 1;
            if (run >= 0) t[run] = (now_ms() - start) / n;
        }
        qsort(t, (size_t)runs, sizeof(*t), by_value);
        best[row] = t[0];
        median[row] = runs & 1 ? t[runs / 2] : (t[runs / 2 - 1] + t[runs / 2]) / 2.0;
    }
    printf("%d records, %.0f decoded bytes each, 1 thread, %d runs (one run = one pass over the records, after one\n"
           "pass that is not counted); MB = 10^6 bytes\n", n, raw, runs);
    printf("| decoder | median ms/record | MB/s | best ms/record | k0 median / this median |\n|---|---:|---:|---:|---:|\n");
    for (row = 0; row <= VARIANTS; row++)
        printf("| %s | %.3f | %.1f | %.3f | %.2f |\n", row ? variants[row - 1u].name : "reference wex_nr2_decode",
               median[row], raw / median[row] / 1000.0, best[row], median[1] / median[row]);
    free(t);
    return 0;
}

int main(int argc, char **argv) {
    WexNr2Model model;
    size_t bytes;
    unsigned char *blob;
    Guard g;
    if (argc < 4) { fprintf(stderr, "usage: %s equal|mutate|bench|speed MODEL.wnm2 [N SEED | REPS | RUNS] REC...\n", argv[0]); return 2; }
    blob = read_file(argv[2], &bytes);
    g = guard_alloc(bytes);   /* the model flush against a guard page too */
    memcpy(g.hi - bytes, blob, bytes);
    if (wex_nr2_model_open(g.hi - bytes, bytes, &model)) { fprintf(stderr, "bad model\n"); return 2; }
#if WEXF_SIMD
    {
        static WexFastModel built;
        double t = now_ms();
        int i;
        for (i = 0; i < 100; i++) wex_fast_model_init(&built, &model);
        printf("wex_fast_model_init: %.1f us per call (mean of 100), tables %u bytes\n", (now_ms() - t) * 10.0, (unsigned)sizeof(built));
        tables = &built;
    }
#endif
    printf("build: %s, %s; model id 0x%08x\n", WEXF_SIMD ? WEXF_KERNELS " kernels" : "scalar only",
#ifdef _OPENMP
           "OpenMP",
#else
           "no OpenMP",
#endif
           model.id);
    if (!strcmp(argv[1], "equal")) return cmd_equal(&model, argc - 3, argv + 3);
    if (!strcmp(argv[1], "mutate") && argc >= 6) { rng_state = strtoull(argv[4], NULL, 0) | 1u; return cmd_mutate(&model, atol(argv[3]), argc - 5, argv + 5); }
    if (!strcmp(argv[1], "bench") && argc >= 5) return cmd_bench(&model, atoi(argv[3]), argc - 4, argv + 4);
    if (!strcmp(argv[1], "speed") && argc >= 5) return cmd_speed(&model, atoi(argv[3]), argc - 4, argv + 4);
    return 2;
}
