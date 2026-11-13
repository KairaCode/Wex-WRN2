/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Verifier's own harness (independent of test_fast_decode.c) for ../wex_fast_decode.h.
 *   vh dump  MODEL REC OUT        fast default decode -> raw slab file (scales w1..w3, weights w1..w3); ref compared
 *   vh time  MODEL REPS REC...    own timing: warm, thread scaling, default entry, production drop-in path
 *   vh reenc MODEL REC            full-size re-encoded records with twists + full-size boundary mutations
 * REC = 72-byte WexEntry + record bytes. */
#include "wex_fast_decode.h"
#include <stdio.h>
#include <windows.h>

static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
static unsigned char *read_file(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    unsigned char *d; long n;
    if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    d = (unsigned char *)malloc((size_t)n + 1u);
    if (!d || fread(d, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *bytes = (size_t)n; return d;
}
typedef struct { WexEntry entry; WexNr2Layout layout; unsigned char *record; size_t total; uint64_t raw, aligned; } Real;
static void load_real(const char *path, Real *r) {
    size_t bytes; unsigned char *d = read_file(path, &bytes);
    memcpy(&r->entry, d, sizeof(WexEntry));
    if (wex_entry_raw_sizes(&r->entry, &r->raw, &r->aligned) || bytes != sizeof(WexEntry) + r->entry.total_bytes) { fprintf(stderr, "bad rec %s\n", path); exit(2); }
    wex_nr2_layout(&r->entry, &r->layout);
    r->record = d + sizeof(WexEntry); r->total = r->entry.total_bytes;
}
static void point(const WexEntry *e, unsigned char *slab, unsigned char *s[3], unsigned char *w[3]) {
    unsigned m; for (m = 0; m < 3u; m++) { s[m] = slab + e->scale_offsets[m]; w[m] = slab + e->weight_offsets[m]; }
}
#if WEXF_SIMD
static WexFastModel fm_storage;
static const WexFastModel *fm = &fm_storage;
#else
static const WexFastModel *fm = NULL;
#endif

/* wex_decode_nr2_record (c/wex_format.h) with only the decode call replaced. */
static int decode_record_fast(const WexEntry *entry, unsigned char *slab, size_t capacity, const WexNr2Model *model, int threads) {
    uint64_t raw, aligned_raw; size_t read_offset; WexNr2Layout layout; WexNr2Record view;
    unsigned char *decoded, *scales[3], *weights[3]; int result;
    WexFastOptions o = {threads, 1, 4, 1};
    if (!entry || !slab || !model || !wex_entry_nr_model(entry) || wex_entry_raw_sizes(entry, &raw, &aligned_raw) || aligned_raw > capacity) return -1;
    read_offset = wex_record_read_offset(entry);
    if (!read_offset || read_offset > capacity || entry->total_bytes > capacity - read_offset) return -1;
    wex_nr2_layout(entry, &layout);
    if (wex_nrx_parse(slab + read_offset, entry->total_bytes, &layout, model->id, wex_entry_nr_magic(entry), &view)) return -1;
    decoded = (unsigned char *)malloc((size_t)raw);
    if (!decoded) return -1;
    point(entry, decoded, scales, weights);
    result = wex_fast_decode_with(fm, &view, &layout, model, scales, weights, &o);
    if (!result) memcpy(slab, decoded, (size_t)raw);
    free(decoded);
    return result;
}

static int cmd_dump(const WexNr2Model *model, const char *rec, const char *outp) {
    Real r; WexNr2Record v; unsigned char *a, *b, *s[3], *w[3]; FILE *f;
    load_real(rec, &r);
    if (wex_nr3_parse(r.record, r.total, &r.layout, model->id, &v)) return 1;
    a = (unsigned char *)malloc(r.raw); b = (unsigned char *)malloc(r.raw);
    memset(a, 0x5a, r.raw); memset(b, 0xa5, r.raw);
    point(&r.entry, a, s, w);
    if (wex_fast_decode(&v, &r.layout, model, s, w)) { printf("fast decode failed\n"); return 1; }
    point(&r.entry, b, s, w);
    if (wex_nr2_decode(&v, &r.layout, model, s, w)) { printf("ref decode failed\n"); return 1; }
    printf("%s: owner 0x%08x flags 0x%02x, fast == ref over the whole %llu-byte slab: %s\n", rec, v.owner, v.flags,
           (unsigned long long)r.raw, memcmp(a, b, r.raw) ? "NO" : "yes");
    f = fopen(outp, "wb"); fwrite(a, 1, r.raw, f); fclose(f);
    return memcmp(a, b, r.raw) != 0;
}

static int cmd_time(const WexNr2Model *model, int reps, int n, char **paths) {
    static const int tl[] = {1, 2, 4, 8, 16, 32};
    Real real[16]; WexNr2Record view[16];
    unsigned char *out, *s[3], *w[3], *slab;
    double best, sum, t; int i, rep, k;
    if (n > 16) n = 16;
    for (i = 0; i < n; i++) { load_real(paths[i], &real[i]); if (wex_nr3_parse(real[i].record, real[i].total, &real[i].layout, model->id, &view[i])) return 1; }
    out = (unsigned char *)malloc(real[0].raw); memset(out, 0, real[0].raw);
    point(&real[0].entry, out, s, w);
    printf("%d records, best of %d (mean in brackets), ms per record\n", n, reps);
    for (best = 1e30, sum = 0, rep = 0; rep < (reps < 3 ? reps : 3); rep++) {
        t = now_ms(); for (i = 0; i < n; i++) if (wex_nr2_decode(&view[i], &real[i].layout, model, s, w)) return 1;
        t = (now_ms() - t) / n; sum += t; if (t < best) best = t;
    }
    printf("reference wex_nr2_decode, warm output: %.2f [%.2f]\n", best, sum / rep);
    fflush(stdout);
#if WEXF_SIMD
    for (k = 0; k < 6; k++) {
        WexFastOptions o = {tl[k], 1, 4, 1};
        for (best = 1e30, sum = 0, rep = 0; rep < reps; rep++) {
            t = now_ms(); for (i = 0; i < n; i++) if (wex_fast_decode_with(fm, &view[i], &real[i].layout, model, s, w, &o)) return 1;
            t = (now_ms() - t) / n; sum += t; if (t < best) best = t;
        }
        printf("fast k1 x4 fused, cached tables, warm output, %2d threads: %.2f [%.2f]\n", tl[k], best, sum / reps);
        fflush(stdout);
    }
    for (best = 1e30, sum = 0, rep = 0; rep < reps; rep++) {
        t = now_ms(); for (i = 0; i < n; i++) if (wex_fast_decode(&view[i], &real[i].layout, model, s, w)) return 1;
        t = (now_ms() - t) / n; sum += t; if (t < best) best = t;
    }
#ifdef _OPENMP
    printf("wex_fast_decode default entry (tables per call, %d threads = runtime default): %.2f [%.2f]\n", omp_get_max_threads(), best, sum / reps);
#endif
    /* fresh (untouched) output buffer per record */
    for (k = 0; k < 4; k += 3) {
        WexFastOptions o = {tl[k], 1, 4, 1};
        for (best = 1e30, sum = 0, rep = 0; rep < reps; rep++) {
            double acc = 0;
            for (i = 0; i < n; i++) {
                unsigned char *fresh, *fs[3], *fw[3];
                t = now_ms();
                fresh = (unsigned char *)VirtualAlloc(NULL, real[i].raw, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                point(&real[i].entry, fresh, fs, fw);
                if (wex_fast_decode_with(fm, &view[i], &real[i].layout, model, fs, fw, &o)) return 1;
                acc += now_ms() - t;
                VirtualFree(fresh, 0, MEM_RELEASE);
            }
            t = acc / n; sum += t; if (t < best) best = t;
        }
        printf("fast, fresh VirtualAlloc output per record (first touch inside the timer), %d threads: %.2f [%.2f]\n", tl[k], best, sum / reps);
    }
    /* production drop-in path: wex_decode_nr2_record with the decode call swapped */
    slab = (unsigned char *)malloc(real[0].aligned);
    for (k = 0; k < 4; k += 3) {
        for (best = 1e30, sum = 0, rep = 0; rep < reps; rep++) {
            double acc = 0;
            for (i = 0; i < n; i++) {
                memcpy(slab + wex_record_read_offset(&real[i].entry), real[i].record, real[i].total);   /* stands for the disk read */
                t = now_ms();
                if (decode_record_fast(&real[i].entry, slab, real[i].aligned, model, tl[k])) return 1;
                acc += now_ms() - t;
            }
            t = acc / n; sum += t; if (t < best) best = t;
        }
        printf("drop-in wex_decode_nr2_record with the fast call (parse + malloc + decode + memcpy + free), %d threads: %.2f [%.2f]\n", tl[k], best, sum / reps);
    }
#endif
    slab = (unsigned char *)malloc(real[0].aligned);
    for (best = 1e30, sum = 0, rep = 0; rep < 2; rep++) {
        double acc = 0;
        for (i = 0; i < n; i++) {
            memcpy(slab + wex_record_read_offset(&real[i].entry), real[i].record, real[i].total);
            t = now_ms();
            if (wex_decode_nr2_record(&real[i].entry, slab, real[i].aligned, model)) return 1;
            acc += now_ms() - t;
        }
        t = acc / n; sum += t; if (t < best) best = t;
    }
    printf("wex_decode_nr2_record as shipped (reference): %.2f [%.2f]\n", best, sum / 2);
    return 0;
}


/* decode with an idle gap before each record (the engine waits on NVMe between decodes) */
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static int cmd_gap(const WexNr2Model *model, int gap_ms, int n, char **paths) {
    Real real[16]; WexNr2Record view[16]; unsigned char *out, *s[3], *w[3]; double v[64], t, sum; int i, rep, c, th;
    if (n > 16) n = 16;
    for (i = 0; i < n; i++) { load_real(paths[i], &real[i]); if (wex_nr3_parse(real[i].record, real[i].total, &real[i].layout, model->id, &view[i])) return 1; }
    out = (unsigned char *)malloc(real[0].raw); memset(out, 0, real[0].raw); point(&real[0].entry, out, s, w);
    for (th = 1; th <= 8; th += 7) {
        WexFastOptions o = {th, 1, 4, 1};
        for (c = 0, sum = 0, rep = 0; rep < 4; rep++)
            for (i = 0; i < n && c < 64; i++) {
                Sleep((DWORD)gap_ms);
                t = now_ms(); if (wex_fast_decode_with(fm, &view[i], &real[i].layout, model, s, w, &o)) return 1;
                v[c] = now_ms() - t; sum += v[c++];
            }
        qsort(v, (size_t)c, sizeof(double), cmp_d);
        printf("fast k1 x4 fused, %d threads, %d ms idle before each decode, %d decodes: min %.2f median %.2f mean %.2f max %.2f ms\n", th, gap_ms, c, v[0], v[c / 2], sum / c, v[c - 1]);
    }
    return 0;
}


/* does the placement of the output buffers change the speed? one slab from malloc, one page-aligned slab,
 * six separate page-aligned regions (what ../test_fast_decode.c bench uses) */
static int cmd_align(const WexNr2Model *model, int reps, int n, char **paths) {
    Real real[16]; WexNr2Record view[16]; unsigned char *s[3][3], *w[3][3], *base; double best, sum, t; int i, rep, th, mode; unsigned m;
    static const char *names[3] = {"one malloc slab", "one VirtualAlloc slab", "six VirtualAlloc regions"};
    if (n > 16) n = 16;
    for (i = 0; i < n; i++) { load_real(paths[i], &real[i]); if (wex_nr3_parse(real[i].record, real[i].total, &real[i].layout, model->id, &view[i])) return 1; }
    base = (unsigned char *)malloc(real[0].raw); memset(base, 0, real[0].raw); point(&real[0].entry, base, s[0], w[0]);
    printf("malloc slab address mod 4096 = %u, mod 64 = %u\n", (unsigned)((uintptr_t)base & 4095u), (unsigned)((uintptr_t)base & 63u));
    base = (unsigned char *)VirtualAlloc(NULL, real[0].raw, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); memset(base, 0, real[0].raw); point(&real[0].entry, base, s[1], w[1]);
    for (m = 0; m < 3u; m++) {
        s[2][m] = (unsigned char *)VirtualAlloc(NULL, real[0].layout.scale_count[m], MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        w[2][m] = (unsigned char *)VirtualAlloc(NULL, real[0].layout.symbol_count[m] / 2u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        memset(s[2][m], 0, real[0].layout.scale_count[m]); memset(w[2][m], 0, real[0].layout.symbol_count[m] / 2u);
    }
    for (th = 1; th <= 8; th += 7)
        for (mode = 0; mode < 3; mode++) {
            WexFastOptions o = {th, 1, 4, 1};
            for (best = 1e30, sum = 0, rep = 0; rep < reps; rep++) {
                t = now_ms(); for (i = 0; i < n; i++) if (wex_fast_decode_with(fm, &view[i], &real[i].layout, model, s[mode], w[mode], &o)) return 1;
                t = (now_ms() - t) / n; sum += t; if (t < best) best = t;
            }
            printf("%d threads, %-26s best %.2f mean %.2f ms/record\n", th, names[mode], best, sum / reps);
            fflush(stdout);
        }
    return 0;
}

#if WEXF_SIMD
#include "tailcut.inc"
#endif
/* ---- full-size checks ------------------------------------------------------ */
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }
static const WexFastOptions VAR[] = {{1, 0, 1, 0}, {1, 1, 1, 0}, {3, 1, 4, 1}, {8, 1, 4, 1}, {8, 1, 8, 0}, {8, 1, 3, 1}, {1, 2, 1, 1}, {8, 2, 3, 0}, {5, 2, 2, 1}, {8, 1, 6, 1}, {2, 1, 2, 0}};
#define NVAR (sizeof(VAR) / sizeof(VAR[0]))
static long cases = 0, accepts = 0, rejects = 0, mism = 0;

/* reference vs every variant + the default entry on one padded record buffer */
static void diff(const char *what, const WexNr2Model *model, const WexNr2Layout *L, const unsigned char *rec, size_t total,
                 unsigned char *a, unsigned char *b, const WexEntry *e, int expect) {
    WexNr2Record v; unsigned char *s[3], *w[3], *fs[3], *fw[3]; size_t k; int st, got;
    uint64_t raw = (uint64_t)e->scale_bytes + e->weight_bytes;
    if (wex_nr3_parse(rec, total, L, model->id, &v)) { printf("  %s: stopped at the parser\n", what); return; }
    point(e, a, s, w); point(e, b, fs, fw);
    st = wex_nr2_decode(&v, L, model, s, w);
    cases++; if (st) rejects++; else accepts++;
    if (expect >= 0 && (st != 0) != expect) printf("  NOTE %s: reference status %d, expected %s\n", what, st, expect ? "reject" : "accept");
    memset(b, 0x33, raw);
    got = wex_fast_decode(&v, L, model, fs, fw);
    if (got != st || (!st && memcmp(a, b, raw))) { printf("  MISMATCH %s default: ref %d fast %d\n", what, st, got); mism++; }
    for (k = 0; k < NVAR; k++) {
        memset(b, 0x33, raw);
        got = wex_fast_decode_with(fm, &v, L, model, fs, fw, &VAR[k]);
        if (got != st || (!st && memcmp(a, b, raw))) { printf("  MISMATCH %s variant %d: ref %d fast %d\n", what, (int)k, st, got); mism++; }
    }
    printf("  %-58s ref %s, %d variants + default agree: %s\n", what, st ? "reject" : "accept", (int)NVAR, mism ? "SEE ABOVE" : "yes");
    fflush(stdout);
}

static int cmd_reenc(const WexNr2Model *model, const char *rec) {
    Real r; WexNr2Record v; unsigned char *plain, *coded, *a, *b, *buf, *ps[3], *pw[3], *cs[3], *cw[3], anchor[3];
    size_t cap, written, total; uint32_t m, flags, i; unsigned t;
    static const uint32_t chunks[][2] = {{512, 256}, {16384, 16384}, {8192, 4096}, {1024, 2048}, {2048, 512}, {4096, 256}};
    load_real(rec, &r);
    if (wex_nr3_parse(r.record, r.total, &r.layout, model->id, &v)) return 1;
    plain = (unsigned char *)malloc(r.raw); coded = (unsigned char *)malloc(r.raw); a = (unsigned char *)malloc(r.raw); b = (unsigned char *)malloc(r.raw);
    cap = r.aligned * 2u; buf = (unsigned char *)malloc(cap);
    point(&r.entry, plain, ps, pw); point(&r.entry, coded, cs, cw);
    if (wex_fast_decode(&v, &r.layout, model, ps, pw)) return 1;
    printf("%s: full-size re-encodes (twist, chunks)\n", rec);
    for (t = 0; t < 12; t++) {
        char what[96]; int expect = 0; uint32_t wc = chunks[t % 6][0], sc = chunks[t % 6][1];
        memcpy(coded, plain, r.raw);
        flags = WEX_NR3_FLAG_CZ;
        if (t == 6 || t == 7)   /* -0 nibbles: 6 without CZ (accept), 7 with CZ (reject) */
            for (i = 0; i < 5; i++) { uint32_t mm = rnd() % 3u, at = rnd() % r.layout.symbol_count[mm];
                cw[mm][at / 2u] = (unsigned char)((cw[mm][at / 2u] & (at & 1u ? 0x0fu : 0xf0u)) | (at & 1u ? 0x80u : 0x08u)); }
        if (t == 6) flags = 0;
        if (t == 7) expect = 1;
        flags |= wex_nr3_ceil_forward(&r.layout, cs, cw, NULL);
        if (t >= 8 && t <= 10) {   /* ceil underflow: coded max-5 block with coded scale 0; first / last / random block */
            uint32_t mm = t - 8u, blk = t == 8 ? 0u : t == 9 ? r.layout.scale_count[mm] - 1u : rnd() % r.layout.scale_count[mm];
            for (i = 0; i < 16u; i++) cw[mm][(size_t)blk * 16u + i] = (unsigned char)(rnd() % 6u | (rnd() % 6u) << 4);
            cw[mm][(size_t)blk * 16u + 3u] = 0x52; cs[mm][blk] = 0; flags |= 1u << mm; expect = 1;
        }
        if (t == 11) { flags &= ~7u; expect = -1; }   /* ceil flags cleared although the data is in ceil form */
        for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(cs[m], r.layout.scale_count[m]);
        if (wex_nrx_encode(model, &r.layout, (const unsigned char *const *)cs, (const unsigned char *const *)cw, wc, sc, anchor,
                           WEX_NR3_MAGIC, flags, v.owner, buf, cap - 4096u, &written)) { printf("  encode failed t=%u\n", t); continue; }
        total = (written + 4095u) & ~(size_t)4095u; memset(buf + written, 0, total - written);
        sprintf(what, "twist %u, weight chunk %u, scale chunk %u, %u B", t, wc, sc, (unsigned)written);
        diff(what, model, &r.layout, buf, total, a, b, &r.entry, expect);
        if (!expect && t < 6 && memcmp(a, plain, r.raw)) { printf("  round trip differs from the original decode\n"); mism++; }
    }
    /* full-size boundary mutations of the ORIGINAL record (production chunks): drop a stream's last byte when the
     * next stream starts with the same byte; append a stray byte; move a boundary */
    printf("%s: full-size boundary mutations\n", rec);
    for (t = 0; t < 24; t++) {
        char what[96]; uint32_t streams = v.scale_streams + v.weight_streams, s, at, len, used = v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes;
        unsigned char *lengths = buf + WEX_NR2_HEADER_BYTES; uint32_t pick, found = 0, found_at = 0, kind = t % 3u;
        memcpy(buf, r.record, r.total); total = r.total;
        /* early streams keep the reference fast (it stops at the first bad stream); every 4th case anywhere */
        pick = t % 4u == 3u ? rnd() % streams : (t & 1u ? v.scale_streams : 0u) + rnd() % 600u;
        if (kind == 0) {
            for (at = v.metadata_bytes, s = 0; s + 1u < streams; s++) {
                len = wex_nr_u16(lengths + 2u * s); at += len;
                if (len > 4u && buf[at - 1u] == buf[at] && s >= pick) { found = s; found_at = at; break; }
            }
            if (!found_at) continue;
            memmove(buf + found_at - 1u, buf + found_at, used - found_at);
            len = wex_nr_u16(lengths + 2u * found) - 1u; lengths[2u * found] = (unsigned char)len; lengths[2u * found + 1u] = (unsigned char)(len >> 8);
            len = (found < v.scale_streams ? v.scale_payload_bytes : v.weight_payload_bytes) - 1u;
            memcpy(buf + (found < v.scale_streams ? 8u : 12u), &len, 4);
            used--; buf[used] = 0; total = ((size_t)used + 4095u) & ~(size_t)4095u;
            sprintf(what, "last byte of stream %u dropped (next starts with it)", found);
        } else if (kind == 1) {
            for (at = v.metadata_bytes, s = 0; s <= pick; s++) at += wex_nr_u16(lengths + 2u * s);
            memmove(buf + at + 1u, buf + at, used - at);
            buf[at] = (unsigned char)rnd();
            len = wex_nr_u16(lengths + 2u * pick) + 1u; lengths[2u * pick] = (unsigned char)len; lengths[2u * pick + 1u] = (unsigned char)(len >> 8);
            len = (pick < v.scale_streams ? v.scale_payload_bytes : v.weight_payload_bytes) + 1u;
            memcpy(buf + (pick < v.scale_streams ? 8u : 12u), &len, 4);
            used++; total = ((size_t)used + 4095u) & ~(size_t)4095u; memset(buf + used, 0, total - used);
            sprintf(what, "stray byte appended to stream %u", pick);
        } else {
            uint32_t la, lb;
            if (pick + 1u >= streams || pick + 1u == v.scale_streams) continue;
            la = wex_nr_u16(lengths + 2u * pick) + 1u; lb = wex_nr_u16(lengths + 2u * pick + 2u) - 1u;
            if (lb < 4u || la > 65535u) continue;
            lengths[2u * pick] = (unsigned char)la; lengths[2u * pick + 1u] = (unsigned char)(la >> 8);
            lengths[2u * pick + 2u] = (unsigned char)lb; lengths[2u * pick + 3u] = (unsigned char)(lb >> 8);
            sprintf(what, "boundary after stream %u moved by 1", pick);
        }
        diff(what, model, &r.layout, buf, total, a, b, &r.entry, 1);
    }
    printf("reenc: %ld cases at the decoders (%ld accepted, %ld rejected by the reference), %ld mismatches\n", cases, accepts, rejects, mism);
    return mism != 0;
}

int main(int argc, char **argv) {
    WexNr2Model model; size_t bytes; unsigned char *blob;
    if (argc < 4) return 2;
    blob = read_file(argv[2], &bytes);
    if (wex_nr2_model_open(blob, bytes, &model)) return 2;
#if WEXF_SIMD
    wex_fast_model_init(&fm_storage, &model);
#endif
    printf("build: %s\n", WEXF_SIMD ? "AVX2 kernels" : "scalar only");
    if (!strcmp(argv[1], "dump") && argc == 5) return cmd_dump(&model, argv[3], argv[4]);
    if (!strcmp(argv[1], "time")) return cmd_time(&model, atoi(argv[3]), argc - 4, argv + 4);
    if (!strcmp(argv[1], "reenc")) return cmd_reenc(&model, argv[3]);
#if WEXF_SIMD
    if (!strcmp(argv[1], "align")) return cmd_align(&model, atoi(argv[3]), argc - 4, argv + 4);
    if (!strcmp(argv[1], "tailcut")) return cmd_tailcut(&model, argv[3]);
    if (!strcmp(argv[1], "gap")) return cmd_gap(&model, atoi(argv[3]), argc - 4, argv + 4);
#endif
    return 2;
}
