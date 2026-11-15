/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Decode speed on real expert slabs: WNR3 (the reference decoder and the fast CPU decoder) and the
 * general-purpose codecs that the Python standard library does not have (lz4, brotli), with zstd as a check
 * of the Python numbers.
 *
 *   bench_real wnr3   RUNS MODEL.wnm2 NAME...   NAME.rec (72-byte entry + WNR3 record) and NAME.slab (its slab)
 *   bench_real codecs RUNS NAME...              NAME.slab
 *
 * Before any timing, every decoder's output is compared with NAME.slab byte for byte, and every compressed
 * form is decompressed and compared. A run is one pass over the records:
 *   "1 thread"             the records one after the other on one thread;
 *   "T records at once"    T OpenMP threads, each decodes whole records (T = the logical CPUs; with fewer
 *                          records than T, the records are taken again until there are T);
 *   "T threads per record" the fast WNR3 decoder with its own T threads over the streams of one record,
 *                          the records one after the other.
 * One pass that is not counted comes first. A row gives the median, the minimum and the maximum over RUNS of
 * the time per record (the pass time divided by the records of the pass) and MB/s = decoded bytes per record /
 * median time, MB = 10^6 bytes. Output: a Markdown table and one line "json {...}" per row, with the local
 * time at which the row ended and how long it ran (to match the row with a log of the machine's CPU load).
 *
 * Build (MSYS2 UCRT64: packages lz4, brotli, zstd; the fast decoder needs AVX2 on x86):
 *   gcc -std=c99 -O3 -march=haswell -fopenmp -Wall -Wextra -Wno-unused-function -I c \
 *       -I wex4/research/fast-cpu-decode bench_real.c -o bench_real -llz4 -lbrotlienc -lbrotlidec -lzstd */
#include "wex_fast_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lz4.h>
#include <lz4hc.h>
#include <brotli/decode.h>
#include <brotli/encode.h>
#include <zstd.h>
#include <omp.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define MAX_RECORDS 64

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

static unsigned char *read_named(const char *name, const char *suffix, size_t *bytes) {
    char path[1024];
    if ((size_t)snprintf(path, sizeof path, "%s%s", name, suffix) >= sizeof path) { fprintf(stderr, "path too long\n"); exit(2); }
    return read_file(path, bytes);
}

/* ---- the records and the per-thread output buffers ---------------------------------------------------- */
typedef struct {
    unsigned char *slab, *packed;   /* source slab; WNR3 record or compressed form */
    size_t slab_bytes, packed_bytes;
    WexEntry entry;
    WexNr2Layout layout;
    WexNr2Record view;
} Rec;

static Rec recs[MAX_RECORDS];
static int nrec, threads;
static unsigned char *outbuf[256];
static size_t maxslab;
static WexNr2Model model;
static WexFastModel tables;

typedef int (*Decode)(int rec, unsigned char *out, int arg);

static void place(const Rec *r, unsigned char *out, unsigned char *s[3], unsigned char *w[3]) {
    int m;
    for (m = 0; m < 3; m++) { s[m] = out + r->entry.scale_offsets[m]; w[m] = out + r->entry.weight_offsets[m]; }
}

static int dec_ref(int i, unsigned char *out, int arg) {
    unsigned char *s[3], *w[3];
    (void)arg;
    place(&recs[i], out, s, w);
    return wex_nr2_decode(&recs[i].view, &recs[i].layout, &model, s, w);
}

static int dec_fast(int i, unsigned char *out, int arg) {   /* arg: threads * 16 + kernel */
    unsigned char *s[3], *w[3];
    WexFastOptions o;
    o.threads = arg >> 4; o.kernel = arg & 15; o.lanes = o.kernel == 2 ? 2 : 4; o.fuse = 1;
    place(&recs[i], out, s, w);
    return wex_fast_decode_with(&tables, &recs[i].view, &recs[i].layout, &model, s, w, &o);
}

static int dec_lz4(int i, unsigned char *out, int arg) {
    (void)arg;
    return LZ4_decompress_safe((const char *)recs[i].packed, (char *)out, (int)recs[i].packed_bytes, (int)maxslab) !=
           (int)recs[i].slab_bytes;
}

static int dec_brotli(int i, unsigned char *out, int arg) {
    size_t n = maxslab;
    (void)arg;
    return BrotliDecoderDecompress(recs[i].packed_bytes, recs[i].packed, &n, out) != BROTLI_DECODER_RESULT_SUCCESS ||
           n != recs[i].slab_bytes;
}

static int dec_zstd(int i, unsigned char *out, int arg) {
    size_t n;
    (void)arg;
    n = ZSTD_decompress(out, maxslab, recs[i].packed, recs[i].packed_bytes);
    return ZSTD_isError(n) || n != recs[i].slab_bytes;
}

/* the output of every record is the slab */
static int check(Decode fn, int arg, const char *what) {
    int i;
    for (i = 0; i < nrec; i++) {
        memset(outbuf[0], 0xa5, maxslab);
        if (fn(i, outbuf[0], arg) || memcmp(outbuf[0], recs[i].slab, recs[i].slab_bytes)) {
            printf("MISMATCH %s, record %d\n", what, i);
            return 1;
        }
    }
    return 0;
}

static int by_value(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* mode 0: 1 thread; 1: T records at once; 2: records one after the other (the decoder has its own threads) */
static void timed(const char *row, const char *mode_name, int mode, Decode fn, int arg, int runs, double ratio) {
    double *t = (double *)malloc((size_t)runs * sizeof(double)), raw = 0, median, began = now_ms();
    int run, i, items = mode == 1 && nrec < threads ? threads : nrec, bad = 0;
    for (i = 0; i < nrec; i++) raw += (double)recs[i].slab_bytes;
    raw /= nrec;
    for (run = -1; run < runs; run++) {
        const double start = now_ms();
        if (mode == 1) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(|:bad)
            for (i = 0; i < items; i++) bad |= fn(i % nrec, outbuf[omp_get_thread_num()], arg);
        } else {
            for (i = 0; i < items; i++) bad |= fn(i, outbuf[0], arg);
        }
        if (run >= 0) t[run] = (now_ms() - start) / items;
    }
    if (bad) { printf("DECODE FAILED in %s\n", row); exit(1); }
    qsort(t, (size_t)runs, sizeof(double), by_value);
    median = runs & 1 ? t[runs / 2] : (t[runs / 2 - 1] + t[runs / 2]) / 2.0;
    printf("| %s | %s | %.4f | %.2f | %.2f to %.2f | %.1f |\n", row, mode_name, ratio, median, t[0], t[runs - 1],
           raw / median / 1000.0);
    {
        char clock[16];
        const time_t now = time(NULL);
        strftime(clock, sizeof clock, "%H:%M:%S", localtime(&now));
        printf("json {\"row\":\"%s\",\"mode\":\"%s\",\"threads\":%d,\"records\":%d,\"runs\":%d,\"ratio\":%.6f,"
               "\"median_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f,\"mb_s\":%.2f,\"decoded_bytes_per_record\":%.0f,"
               "\"ended\":\"%s\",\"seconds\":%.1f}\n",
               row, mode_name, mode ? threads : 1, nrec, runs, ratio, median, t[0], t[runs - 1], raw / median / 1000.0,
               raw, clock, (now_ms() - began) / 1000.0);
    }
    fflush(stdout);
    free(t);
}

static void header(void) {
    printf("| decoder | threads | stored / raw | median ms/record | min to max ms | MB/s |\n|---|---|---:|---:|---|---:|\n");
}

static int cmd_wnr3(int runs, const char *model_path, int count, char **names) {
    size_t bytes, rec_bytes;
    unsigned char *blob = read_file(model_path, &bytes);
    double stored = 0, raw = 0;
    char mode[64];
    int i;
    if (wex_nr2_model_open(blob, bytes, &model)) { fprintf(stderr, "bad model\n"); return 2; }
    wex_fast_model_init(&tables, &model);
    for (i = 0; i < count; i++) {
        Rec *r = &recs[i];
        unsigned char *data = read_named(names[i], ".rec", &rec_bytes);
        r->slab = read_named(names[i], ".slab", &r->slab_bytes);
        if (rec_bytes < sizeof(WexEntry)) return 2;
        memcpy(&r->entry, data, sizeof(WexEntry));
        r->packed = data + sizeof(WexEntry); r->packed_bytes = rec_bytes - sizeof(WexEntry);
        if (r->entry.flags != WEX_ENTRY_NEURAL_RANS3 || r->entry.total_bytes != r->packed_bytes ||
            (size_t)r->entry.scale_bytes + r->entry.weight_bytes != r->slab_bytes) { fprintf(stderr, "%s: entry\n", names[i]); return 2; }
        wex_nr2_layout(&r->entry, &r->layout);
        if (wex_nr3_parse(r->packed, r->packed_bytes, &r->layout, model.id, &r->view)) { fprintf(stderr, "%s: parse\n", names[i]); return 1; }
        if (r->slab_bytes > maxslab) maxslab = r->slab_bytes;
        stored += (double)r->packed_bytes; raw += (double)r->slab_bytes;
    }
    nrec = count;
    for (i = 0; i < threads; i++) outbuf[i] = (unsigned char *)malloc(maxslab);
    if (check(dec_ref, 0, "reference") || check(dec_fast, 1 << 4 | 1, "fast k1 1 thread") ||
        check(dec_fast, threads << 4 | 1, "fast k1 T threads") || check(dec_fast, 1 << 4 | 2, "fast k2 1 thread") ||
        check(dec_fast, threads << 4 | 2, "fast k2 T threads")) return 1;
    printf("wnr3: %d records, model id 0x%08x, %s kernels, every decoder's output equal to the slab\n", nrec, model.id, WEXF_KERNELS);
    header();
    snprintf(mode, sizeof mode, "%d records at once", threads);
    timed("WNR3 reference wex_nr2_decode", "1", 0, dec_ref, 0, runs, stored / raw);
    timed("WNR3 reference wex_nr2_decode", mode, 1, dec_ref, 0, runs, stored / raw);
    timed("WNR3 fast, k1 4 streams fused (default)", "1", 0, dec_fast, 1 << 4 | 1, runs, stored / raw);
    snprintf(mode, sizeof mode, "%d per record", threads);
    timed("WNR3 fast, k1 4 streams fused (default)", mode, 2, dec_fast, threads << 4 | 1, runs, stored / raw);
    snprintf(mode, sizeof mode, "%d records at once", threads);
    timed("WNR3 fast, k1 4 streams fused (default)", mode, 1, dec_fast, 1 << 4 | 1, runs, stored / raw);
    timed("WNR3 fast, k2 gather 16 streams fused", "1", 0, dec_fast, 1 << 4 | 2, runs, stored / raw);
    snprintf(mode, sizeof mode, "%d per record", threads);
    timed("WNR3 fast, k2 gather 16 streams fused", mode, 2, dec_fast, threads << 4 | 2, runs, stored / raw);
    return 0;
}

typedef struct { const char *name; int kind, level; } Codec;   /* kind 0 lz4, 1 lz4hc, 2 brotli, 3 zstd */
static const Codec codecs[] = {
    {"lz4 (LZ4_compress_default)", 0, 1}, {"lz4hc level 9", 1, 9}, {"lz4hc level 12", 1, 12},
    {"brotli quality 1, window 24", 2, 1}, {"brotli quality 5, window 24", 2, 5}, {"brotli quality 9, window 24", 2, 9},
    {"brotli quality 11, window 24", 2, 11},
    {"zstd level 1 (libzstd)", 3, 1}, {"zstd level 3 (libzstd)", 3, 3}, {"zstd level 19 (libzstd)", 3, 19},
};

static int compress_one(const Codec *c, Rec *r) {
    size_t cap;
    if (c->kind <= 1) {
        int n;
        cap = (size_t)LZ4_compressBound((int)r->slab_bytes);
        r->packed = (unsigned char *)malloc(cap);
        n = c->kind ? LZ4_compress_HC((const char *)r->slab, (char *)r->packed, (int)r->slab_bytes, (int)cap, c->level)
                    : LZ4_compress_default((const char *)r->slab, (char *)r->packed, (int)r->slab_bytes, (int)cap);
        r->packed_bytes = (size_t)n;
        return n <= 0;
    }
    if (c->kind == 2) {
        cap = BrotliEncoderMaxCompressedSize(r->slab_bytes);
        r->packed = (unsigned char *)malloc(cap);
        r->packed_bytes = cap;
        return !BrotliEncoderCompress(c->level, 24, BROTLI_MODE_GENERIC, r->slab_bytes, r->slab, &r->packed_bytes, r->packed);
    }
    cap = ZSTD_compressBound(r->slab_bytes);
    r->packed = (unsigned char *)malloc(cap);
    r->packed_bytes = ZSTD_compress(r->packed, cap, r->slab, r->slab_bytes, c->level);
    return ZSTD_isError(r->packed_bytes);
}

static int cmd_codecs(int runs, int count, char **names) {
    size_t k;
    int i;
    char mode[64];
    for (i = 0; i < count; i++) {
        recs[i].slab = read_named(names[i], ".slab", &recs[i].slab_bytes);
        if (recs[i].slab_bytes > maxslab) maxslab = recs[i].slab_bytes;
    }
    nrec = count;
    for (i = 0; i < threads; i++) outbuf[i] = (unsigned char *)malloc(maxslab);
    printf("codecs: %d records; lz4 %s, brotli %u.%u.%u, zstd %s\n", nrec, LZ4_versionString(),
           BrotliDecoderVersion() >> 24, BrotliDecoderVersion() >> 12 & 0xfff, BrotliDecoderVersion() & 0xfff,
           ZSTD_versionString());
    header();
    snprintf(mode, sizeof mode, "%d records at once", threads);
    for (k = 0; k < sizeof codecs / sizeof codecs[0]; k++) {
        const Codec *c = &codecs[k];
        const Decode fn = c->kind <= 1 ? dec_lz4 : c->kind == 2 ? dec_brotli : dec_zstd;
        double stored = 0, raw = 0, t = now_ms();
        int bad = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(|:bad)
        for (i = 0; i < nrec; i++) bad |= compress_one(c, &recs[i]);
        if (bad) { printf("COMPRESS FAILED %s\n", c->name); return 1; }
        for (i = 0; i < nrec; i++) { stored += (double)recs[i].packed_bytes; raw += (double)recs[i].slab_bytes; }
        printf("compressed with %s: %.1f s for the %d records, %d threads (not a speed measurement)\n", c->name,
               (now_ms() - t) / 1000.0, nrec, threads);
        if (check(fn, 0, c->name)) return 1;
        timed(c->name, "1", 0, fn, 0, runs, stored / raw);
        timed(c->name, mode, 1, fn, 0, runs, stored / raw);
        for (i = 0; i < nrec; i++) free(recs[i].packed);
    }
    return 0;
}

int main(int argc, char **argv) {
    int runs;
    if (argc < 4) { fprintf(stderr, "usage: %s wnr3 RUNS MODEL.wnm2 NAME... | codecs RUNS NAME...\n", argv[0]); return 2; }
    threads = omp_get_num_procs();
    if (threads > 256) threads = 256;
    runs = atoi(argv[2]);
    if (runs < 1) return 2;
    printf("bench_real: %d logical CPUs, %d runs per row, one pass not counted first\n", threads, runs);
    if (!strcmp(argv[1], "wnr3") && argc >= 5 && argc - 4 <= MAX_RECORDS) return cmd_wnr3(runs, argv[3], argc - 4, argv + 4);
    if (!strcmp(argv[1], "codecs") && argc - 3 <= MAX_RECORDS) return cmd_codecs(runs, argc - 3, argv + 3);
    return 2;
}
