/* tdrv: one timing / equality driver for every decoder variant of the fastdec and fastfmt lanes.
 * Compiled several times (see build.sh): the wex_fast_decode.h found first on the include path is either the
 * unchanged Wex-WRN2 header (wex4/research/fast-cpu-decode) or the fastdec lane's header; wnx_fast.h is the
 * fastfmt lane's decoder. The source is identical for every binary.
 *
 *   tdrv equal VARIANT ID...             decode each record at 1 and 4 threads, memcmp with Z(source) (whole slab)
 *   tdrv time PASSES THREADS VARIANT ID...   1 warm-up pass + PASSES timed passes over the records;
 *                                        THREADS > 0: threads per record; THREADS < 0: -THREADS records at once, 1 thread each
 *                                        prints "PASS wall_ms_per_record cpu_ms_per_record" per pass
 * VARIANT: ref | w3:KERNEL,LANES,FUSE | wnx:DIR:KERNEL:G     ID: LL_EEE
 * Z(source) = <TMP>/wnr3-research/fastfmt/data/Z_LL_EEE.bin (checked against wex_nr2_decode by "tdrv equal ref"). */
#define _GNU_SOURCE
#include "wnx_fast.h"
#include "wex_fast_decode.h"
#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static double cpu_ms(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static unsigned char *read_file(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    unsigned char *d;
    long n;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    d = (unsigned char *)aligned_alloc(64, ((size_t)n + 63u) & ~(size_t)63u);
    if (!d || fread(d, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    fclose(f);
    *bytes = (size_t)n;
    return d;
}

typedef struct { int kind, k, l, f; char dir[512]; } Var;   /* kind 0 ref, 1 w3 fast, 2 wnx fast */
static Var V;
static WexNr2Model model3;
static WexFastModel *fm3;
static WnxModel wm;
static WnxfTables *wt;

typedef struct {
    int layer, expert;
    WexEntry e;
    WexNr2Layout lay;
    unsigned char *rec, *z;
    size_t n, raw;
    WexNr2Record v3;
    WnxView vx;
    WnxStreamInfo *si;
    uint32_t nsi;
} Rec;

static void setup(const char *var) {
    size_t b;
    unsigned char *blob = read_file("<TMP>/wnr3-data/recs/model.bin", &b);
    memset(&V, 0, sizeof V);
    if (wex_nr2_model_open(blob, b, &model3)) exit(2);
    if (!strcmp(var, "ref")) V.kind = 0;
    else if (!strncmp(var, "w3:", 3)) {
        V.kind = 1;
        if (sscanf(var + 3, "%d,%d,%d", &V.k, &V.l, &V.f) != 3) exit(2);
#ifndef WEXF_DEFAULT_KERNEL
        if (V.k > 2) { fprintf(stderr, "kernel %d needs the fastdec header\n", V.k); exit(2); }
#else
        if (V.k == 4 && !WEXF_AVX512) { fprintf(stderr, "kernel 4 needs an AVX-512 build\n"); exit(2); }
#endif
        fm3 = (WexFastModel *)aligned_alloc(64, (sizeof(WexFastModel) + 63u) & ~(size_t)63u);
        wex_fast_model_init(fm3, &model3);
    } else if (!strncmp(var, "wnx:", 4)) {
        const char *c = strrchr(var, ':'), *d;
        unsigned char *cnt;
        V.kind = 2;
        V.f = atoi(c + 1);
        for (d = c - 1; *d != ':'; d--) {}
        V.k = atoi(d + 1);
        if (V.k == 2 && !WNXF_512) { fprintf(stderr, "WNX kernel 2 needs an AVX-512 build\n"); exit(2); }
        snprintf(V.dir, sizeof V.dir, "%.*s", (int)(d - (var + 4)), var + 4);
        cnt = read_file("<TMP>/wnr3-research/fastfmt/out_train.counts", &b);
        {   /* the same model build as fastfmt bench.c / ff.c (counts layout of ff.c) */
            const uint64_t (*nib)[7][16] = (const uint64_t(*)[7][16])(const void *)cnt;
            const uint64_t (*sc)[256] = (const uint64_t(*)[256])(const void *)(cnt + 8u * (3 * 7 * 16 + 3 * 7 * 32 * 16));
            wnx_model_build(&wm, strstr(V.dir, "joint") != NULL, nib, sc);
        }
        wt = (WnxfTables *)aligned_alloc(64, (sizeof(WnxfTables) + 63u) & ~(size_t)63u);
        wnxf_tables_init(wt, &wm);
    } else { fprintf(stderr, "bad variant %s\n", var); exit(2); }
}

static void load_rec(const char *id, Rec *r, int want_z) {
    char p[1024];
    size_t b;
    unsigned char *eb;
    uint64_t raw, al;
    memset(r, 0, sizeof *r);
    if (sscanf(id, "%d_%d", &r->layer, &r->expert) != 2) exit(2);
    snprintf(p, sizeof p, "<TMP>/wnr3-research/fastfmt/data/E_%02d_%03d.bin", r->layer, r->expert);
    eb = read_file(p, &b); memcpy(&r->e, eb, 72); free(eb);
    wex_entry_raw_sizes(&r->e, &raw, &al); r->raw = raw;
    wex_nr2_layout(&r->e, &r->lay);
    if (want_z) { snprintf(p, sizeof p, "<TMP>/wnr3-research/fastfmt/data/Z_%02d_%03d.bin", r->layer, r->expert); r->z = read_file(p, &b); if (b != raw) exit(2); }
    if (V.kind < 2) {
        snprintf(p, sizeof p, "<TMP>/wnr3-data/recs/rec_%02d_%03d.bin", r->layer, r->expert);
        r->rec = read_file(p, &r->n);
        if (r->n != r->e.total_bytes || wex_nr3_parse(r->rec, r->n, &r->lay, model3.id, &r->v3)) { fprintf(stderr, "wnr3 parse %s\n", p); exit(2); }
    } else {
        snprintf(p, sizeof p, "%s/X_%02d_%03d.wnx", V.dir, r->layer, r->expert);
        r->rec = read_file(p, &r->n);
        if (wnx_parse(r->rec, r->n, &r->lay, &wm, &r->vx)) { fprintf(stderr, "wnx parse %s\n", p); exit(2); }
        r->si = (WnxStreamInfo *)malloc(sizeof(WnxStreamInfo) * r->vx.streams);
        r->nsi = wnx_streams(&r->vx, &r->lay, r->si);
    }
}
static void free_rec(Rec *r) { free(r->rec); free(r->z); free(r->si); }

static int run(Rec *r, unsigned char *slab, int threads) {
    unsigned char *so[3], *wo[3];
    uint32_t m;
    for (m = 0; m < 3u; m++) { so[m] = slab + r->e.scale_offsets[m]; wo[m] = slab + r->e.weight_offsets[m]; }
    if (V.kind == 0) return wex_nr2_decode(&r->v3, &r->lay, &model3, so, wo);
    if (V.kind == 1) {
        WexFastOptions o;
        memset(&o, 0, sizeof o);
        o.threads = threads; o.kernel = V.k; o.lanes = V.l; o.fuse = V.f;
        return wex_fast_decode_with(fm3, &r->v3, &r->lay, &model3, so, wo, &o);
    }
    {
        const WnxfOpt o = {threads, V.k, V.f, 0};
        return wnxf_decode(wt, &wm, &r->vx, r->si, r->nsi, so, wo, &o, r->rec + r->n);
    }
}

int main(int argc, char **argv) {
    int i;
    if (argc < 3) return 2;
#ifdef WEXF_DEFAULT_KERNEL
    fprintf(stderr, "header: fastdec lane wex_fast_decode.h, WNX AVX-512 %d\n", WNXF_512);
#else
    fprintf(stderr, "header: unchanged Wex-WRN2 wex_fast_decode.h, WNX AVX-512 %d\n", WNXF_512);
#endif
    if (!strcmp(argv[1], "equal")) {
        unsigned char *slab = (unsigned char *)aligned_alloc(64, 18800640u + 4096u);
        int fails = 0, th;
        setup(argv[2]);
        for (i = 3; i < argc; i++) {
            Rec r;
            load_rec(argv[i], &r, 1);
            for (th = 1; th <= 4; th += 3) {
                int st, eq;
                if (V.kind == 0 && th > 1) continue;
                memset(slab, 0xa5, r.raw);
                st = run(&r, slab, th);
#ifdef NEGCTL
                slab[(size_t)(i * 7919u) % r.raw] ^= 1u;   /* negative control: one flipped output bit must be caught */
#endif
                eq = !st && !memcmp(slab, r.z, r.raw);
                printf("%s %s threads %d: %s\n", argv[2], argv[i], th, eq ? "equal" : "MISMATCH");
                fails += !eq;
            }
            free_rec(&r);
        }
        printf("SUMMARY %s: %d records, %d mismatches\n", argv[2], argc - 3, fails);
        return fails != 0;
    }
    if (!strcmp(argv[1], "time") && argc >= 6) {
        const int passes = atoi(argv[2]), threads = atoi(argv[3]), nz = argc - 5;
        Rec *r = (Rec *)calloc((size_t)nz, sizeof(Rec));
        unsigned char *slab[4];
        int p;
        setup(argv[4]);
        for (i = 0; i < 4; i++) slab[i] = (unsigned char *)aligned_alloc(64, 18800640u + 4096u);
        for (i = 0; i < nz; i++) load_rec(argv[5 + i], &r[i], 0);
        for (p = -1; p < passes; p++) {
            double s = now_ms(), c = cpu_ms();
            int bad = 0;
            if (threads > 0) { for (i = 0; i < nz; i++) bad |= run(&r[i], slab[i & 3], threads); }
            else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(-threads) schedule(static, 1) reduction(|:bad)
#endif
                for (i = 0; i < nz; i++) bad |= run(&r[i], slab[i & 3], 1);
            }
            if (bad) { printf("decode failed\n"); return 1; }
            if (p >= 0) printf("PASS %.3f %.3f\n", (now_ms() - s) / nz, (cpu_ms() - c) / nz);
        }
        return 0;
    }
    return 2;
}
