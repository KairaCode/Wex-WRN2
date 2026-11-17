/* bench: equality of every fast WNX decoder and of wex_fast_decode with Z(source), and interleaved A/B timing.
 *   bench equal OUTDIR Z...              every WNX kernel (and WNR3 wex_fast_decode) on every record == Z
 *   bench time ROUNDS THREADS OUTDIR... -- Z...   interleaved rounds over decoders; median / min / max ms per record
 *   bench mutate OUTDIR N SEED Z...      byte-mutated WNX records: fast and reference verdicts (and bytes) agree
 * OUTDIR holds model.wnxm (rebuilt from out_train.counts by name: mag / joint) and X_LL_EEE.wnx. */
#define _GNU_SOURCE
#include "wnx_fast.h"
#include "wex_fast_decode.h"
#include <stdio.h>
#include <time.h>
#include <sys/mman.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static double cpu_ms(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static unsigned char *read_file(const char *path, size_t *bytes, size_t slack) {
    FILE *f = fopen(path, "rb");
    unsigned char *d;
    long n;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    d = (unsigned char *)aligned_alloc(64, ((size_t)n + slack + 63u) & ~(size_t)63u);
    if (!d || fread(d, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    memset(d + n, 0, slack);
    fclose(f);
    *bytes = (size_t)n;
    return d;
}

typedef struct {
    int layer, expert;
    WexEntry e;
    WexNr2Layout lay;
    unsigned char *z, *w3, *wx[4];
    size_t w3n, wxn[4], raw;
    WexNr2Record v3;
    WnxView vx[4];
    WnxStreamInfo *si[4];
    uint32_t nsi[4];
} Rec;

static WexNr2Model model3;
static WexFastModel *fm3;
static WnxModel models[4];
static WnxfTables *tabs[4];
static int nout;
static const char *outdirs[4];

static void load_models(int n, char **dirs) {
    size_t b;
    unsigned char *blob = read_file("/tmp/wnr3-data/recs/model.bin", &b, 0);
    int i;
    if (wex_nr2_model_open(blob, b, &model3)) exit(2);
    fm3 = (WexFastModel *)aligned_alloc(64, (sizeof(WexFastModel) + 63u) & ~(size_t)63u);
    wex_fast_model_init(fm3, &model3);
    {
        unsigned char *c = read_file("/tmp/wnr3-research/fastfmt/out_train.counts", &b, 0);
        /* Counts layout of ff.c: nib[3][7][16] first, then pos, then sc[3][256] */
        const uint64_t (*nib)[7][16] = (const uint64_t(*)[7][16])(const void *)c;
        const uint64_t (*sc)[256] = (const uint64_t(*)[256])(const void *)(c + 8u * (3 * 7 * 16 + 3 * 7 * 32 * 16));
        for (i = 0; i < n; i++) {
            outdirs[i] = dirs[i];
            wnx_model_build(&models[i], strstr(dirs[i], "joint") != NULL, nib, sc);
            tabs[i] = (WnxfTables *)aligned_alloc(64, (sizeof(WnxfTables) + 63u) & ~(size_t)63u);
            wnxf_tables_init(tabs[i], &models[i]);
        }
        free(c);
    }
    nout = n;
}

static void load_rec(const char *zpath, Rec *r) {
    char p[1024];
    size_t b;
    const char *base = strrchr(zpath, '/');
    unsigned char *eb;
    uint64_t raw, al;
    int i;
    base = base ? base + 1 : zpath;
    sscanf(base, "Z_%d_%d.bin", &r->layer, &r->expert);
    snprintf(p, sizeof p, "/tmp/wnr3-research/fastfmt/data/E_%02d_%03d.bin", r->layer, r->expert);
    eb = read_file(p, &b, 0); memcpy(&r->e, eb, 72); free(eb);
    wex_entry_raw_sizes(&r->e, &raw, &al); r->raw = raw;
    wex_nr2_layout(&r->e, &r->lay);
    r->z = read_file(zpath, &b, 0);
    snprintf(p, sizeof p, "/tmp/wnr3-data/recs/rec_%02d_%03d.bin", r->layer, r->expert);
    r->w3 = read_file(p, &r->w3n, 0);
    if (wex_nr3_parse(r->w3, r->w3n, &r->lay, model3.id, &r->v3)) { fprintf(stderr, "wnr3 parse\n"); exit(2); }
    for (i = 0; i < nout; i++) {
        snprintf(p, sizeof p, "%s/X_%02d_%03d.wnx", outdirs[i], r->layer, r->expert);
        r->wx[i] = read_file(p, &r->wxn[i], 0);
        if (wnx_parse(r->wx[i], r->wxn[i], &r->lay, &models[i], &r->vx[i])) { fprintf(stderr, "wnx parse %s\n", p); exit(2); }
        r->si[i] = (WnxStreamInfo *)malloc(sizeof(WnxStreamInfo) * r->vx[i].streams);
        r->nsi[i] = wnx_streams(&r->vx[i], &r->lay, r->si[i]);
    }
}
static void free_rec(Rec *r) {
    int i;
    free(r->z); free(r->w3);
    for (i = 0; i < nout; i++) { free(r->wx[i]); free(r->si[i]); }
}

/* decoder ids: -1 WNR3 wex_fast_decode (k1, 4 streams, fused); -2 WNR3 reference; (i, kernel, G) WNX */
typedef struct { int fmt, kernel, G; char name[96]; } Dec;

static int run_dec(const Dec *d, Rec *r, unsigned char *slab, int threads) {
    unsigned char *so[3], *wo[3];
    uint32_t m;
    for (m = 0; m < 3u; m++) { so[m] = slab + r->e.scale_offsets[m]; wo[m] = slab + r->e.weight_offsets[m]; }
    if (d->fmt == -1) {
        const WexFastOptions o = {threads, 1, 4, 1};
        return wex_fast_decode_with(fm3, &r->v3, &r->lay, &model3, so, wo, &o);
    }
    if (d->fmt == -2) return wex_nr2_decode(&r->v3, &r->lay, &model3, so, wo);
    if (d->kernel == 0) return wnx_decode_ref(&r->vx[d->fmt], &r->lay, &models[d->fmt], so, wo);
    {
        const WnxfOpt o = {threads, d->kernel, d->G, d->name[0] == 'S'};
        return wnxf_decode(tabs[d->fmt], &models[d->fmt], &r->vx[d->fmt], r->si[d->fmt], r->nsi[d->fmt], so, wo, &o,
                           r->wx[d->fmt] + r->wxn[d->fmt]);
    }
}

static int make_decs(Dec *d, int with_ref) {
    int n = 0, i, k, G;
    d[n].fmt = -1; d[n].kernel = 1; d[n].G = 4; snprintf(d[n].name, 96, "WNR3 wex_fast_decode (AVX2 k1, 4 streams, fused)"); n++;
    if (with_ref) { d[n].fmt = -2; d[n].kernel = 0; d[n].G = 1; snprintf(d[n].name, 96, "WNR3 reference wex_nr2_decode"); n++; }
    for (i = 0; i < nout; i++) {
        const char *nm = strrchr(outdirs[i], '/') ? strrchr(outdirs[i], '/') + 1 : outdirs[i];
        if (with_ref) { d[n].fmt = i; d[n].kernel = 0; d[n].G = 1; snprintf(d[n].name, 96, "WNX %s reference", nm); n++; }
        for (k = 1; k <= (WNXF_512 ? 2 : 1); k++)
            for (G = 1; G <= 2; G++) {
                d[n].fmt = i; d[n].kernel = k; d[n].G = G;
                snprintf(d[n].name, 96, "WNX %s %s G=%d", nm, k == 2 ? "AVX-512" : "AVX2", G); n++;
            }
        if (getenv("WNX_SCALES_ONLY")) { d[n].fmt = i; d[n].kernel = 1; d[n].G = 1; snprintf(d[n].name, 96, "Scales only, WNX %s", nm); n++; }
    }
    return n;
}

static int cmd_equal(int nz, char **zs) {
    Dec d[64];
    const int n = make_decs(d, 1);
    int i, j, fails = 0;
    unsigned char *slab = (unsigned char *)malloc(18800640u + 4096u);
    for (i = 0; i < nz; i++) {
        Rec r;
        load_rec(zs[i], &r);
        printf("L%02d E%03d:", r.layer, r.expert);
        for (j = 0; j < n; j++) {
            int st, eq, th;
            if (d[j].fmt == -2 && i % 10) continue;   /* the WNR3 reference is slow: every 10th record */
            for (th = 1; th <= 4; th += 3) {
                if (d[j].kernel == 0 && th > 1) continue;
                memset(slab, 0xa5, r.raw);
                st = run_dec(&d[j], &r, slab, th);
                eq = !st && !memcmp(slab, r.z, r.raw);
                if (!eq) { printf(" MISMATCH[%s, %d thr, status %d]", d[j].name, th, st); fails++; }
            }
        }
        printf(" %s\n", fails ? "" : "all equal");
        fflush(stdout);
        free_rec(&r);
    }
    printf("equal: %d records x %d decoders (fast ones at 1 and 4 threads): %d mismatches\n", nz, n, fails);
    return fails != 0;
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

static int cmd_time(int rounds, int threads, int nz, char **zs) {
    Dec d[64];
    const int n = make_decs(d, 0);
    Rec *r = (Rec *)calloc((size_t)nz, sizeof(Rec));
    unsigned char *slab[4];
    double *t = (double *)calloc((size_t)(n * rounds), sizeof(double)), *ra = (double *)calloc((size_t)(n * rounds), sizeof(double));
    double *ta = (double *)calloc((size_t)(n * rounds), sizeof(double)), raw = 0, wts = 0;
    double *c = (double *)calloc((size_t)(n * rounds), sizeof(double)), *rc = (double *)calloc((size_t)(n * rounds), sizeof(double));
    double *ca = (double *)calloc((size_t)(n * rounds), sizeof(double));
    int i, j, k, q, na = 0;
    for (i = 0; i < 4; i++) slab[i] = (unsigned char *)aligned_alloc(64, 18800640u + 4096u);
    for (i = 0; i < nz; i++) { load_rec(zs[i], &r[i]); raw += (double)r[i].raw; wts += (double)r[i].lay.symbol_count[0] + r[i].lay.symbol_count[1] + r[i].lay.symbol_count[2]; }
    raw /= nz; wts /= nz;
    for (j = 0; j < n; j++) for (i = 0; i < nz; i++) if (run_dec(&d[j], &r[i], slab[0], threads > 0 ? threads : 1)) { printf("decode failed\n"); return 1; }   /* warm up */
    /* every WNX decoder B is timed right next to the baseline A (wex_fast_decode): A B or B A, alternating;
       the ratio of that pair is the speed-up sample, so slow drifts of the shared machine's load cancel */
    for (q = 0; q < rounds; q++)
        for (k = 1; k < n; k++) {
            const int j2 = 1 + (k - 1 + q) % (n - 1);
            double s, s2, tA = 0, tB = 0, cA = 0, cB = 0;
            int pass;
            for (pass = 0; pass < 2; pass++) {
                const int isA = ((q + k) & 1) ? pass == 1 : pass == 0;
                s = now_ms(); s2 = cpu_ms();
                if (threads > 0)
                    for (i = 0; i < nz; i++) run_dec(&d[isA ? 0 : j2], &r[i], slab[i & 3], threads);
                else {   /* throughput: -threads records at once, one thread each */
#pragma omp parallel for num_threads(-threads) schedule(static, 1)
                    for (i = 0; i < nz; i++) run_dec(&d[isA ? 0 : j2], &r[i], slab[i & 3], 1);
                }
                if (isA) { tA = (now_ms() - s) / nz; cA = (cpu_ms() - s2) / nz; } else { tB = (now_ms() - s) / nz; cB = (cpu_ms() - s2) / nz; }
            }
            t[j2 * rounds + q] = tB; ra[j2 * rounds + q] = tA / tB; c[j2 * rounds + q] = cB; rc[j2 * rounds + q] = cA / cB;
            ca[na] = cA; ta[na++] = tA;
        }
    printf("%d records (%.0f decoded bytes, %.0f weights each), %s %d, %d rounds; each decoder timed next to wex_fast_decode (A B / B A alternating); ms/record = wall time / records\n", nz, raw, wts, threads > 0 ? "threads per record" : "records at once (1 thread each):", threads > 0 ? threads : -threads, rounds);
    printf("| decoder | wall ms/record: median (min..max) | MB/s | ns/weight | wall speed-up, median of paired ratios (min..max) | CPU ms/record (median) | CPU MB/s | CPU ns/weight | CPU-time speed-up, paired (min..max) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    qsort(ta, (size_t)na, sizeof(double), cmp_d);
    qsort(ca, (size_t)na, sizeof(double), cmp_d);
    {
        const double med = ta[na / 2], cm = ca[na / 2];
        printf("| %s | %.2f (%.2f..%.2f) | %.1f | %.3f | 1 | %.2f | %.1f | %.3f | 1 |\n", d[0].name, med, ta[0], ta[na - 1], raw / med / 1e3, med * 1e6 / wts,
               cm, raw / cm / 1e3, cm * 1e6 / wts);
    }
    for (j = 1; j < n; j++) {
        double *row = t + j * rounds, *rr = ra + j * rounds, *cr = c + j * rounds, *rcr = rc + j * rounds, med, rmed, cmed, crm;
        qsort(row, (size_t)rounds, sizeof(double), cmp_d); qsort(rr, (size_t)rounds, sizeof(double), cmp_d);
        qsort(cr, (size_t)rounds, sizeof(double), cmp_d); qsort(rcr, (size_t)rounds, sizeof(double), cmp_d);
#define MED(a) (rounds & 1 ? (a)[rounds / 2] : ((a)[rounds / 2 - 1] + (a)[rounds / 2]) / 2)
        med = MED(row); rmed = MED(rr); cmed = MED(cr); crm = MED(rcr);
        printf("| %s | %.2f (%.2f..%.2f) | %.1f | %.3f | %.2fx (%.2f..%.2f) | %.2f | %.1f | %.3f | %.2fx (%.2f..%.2f) |\n", d[j].name, med, row[0], row[rounds - 1],
               raw / med / 1e3, med * 1e6 / wts, rmed, rr[0], rr[rounds - 1], cmed, raw / cmed / 1e3, cmed * 1e6 / wts, crm, rcr[0], rcr[rounds - 1]);
    }
    return 0;
}

/* mutate: random byte damage to WNX records; both the fast decoder and the reference must agree */
static uint64_t rs = 1;
static uint64_t rnd(void) { rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * UINT64_C(2685821657736338717); }
static int cmd_mutate(long N, uint64_t seed, int nz, char **zs) {
    Dec d[64];
    const int n = make_decs(d, 1);
    long c, acc = 0, rej = 0, parse = 0;
    unsigned char *slabr = (unsigned char *)malloc(18800640u), *slabf = (unsigned char *)malloc(18800640u);
    int fmt;
    rs = seed | 1;
    for (c = 0; c < N; c++) {
        Rec r;
        unsigned char *buf;
        WnxView v;
        WnxStreamInfo *si;
        uint32_t ns, pick, s, len;
        size_t at;
        int sr, j, kind;
        load_rec(zs[(size_t)(rnd() >> 33) % (size_t)nz], &r);
        fmt = (int)((rnd() >> 33) % (uint64_t)nout);
        buf = r.wx[fmt];
        /* damage inside one random stream (header / offset damage is the parser's job) */
        si = r.si[fmt]; ns = r.nsi[fmt];
        pick = (uint32_t)((rnd() >> 33) % ns);
        at = (size_t)(r.vx[fmt].payload - buf) + si[pick].begin;
        len = si[pick].end - si[pick].begin;
        kind = (int)((rnd() >> 33) % 6);
        if (kind == 0) buf[at + (rnd() >> 33) % len] ^= (unsigned char)(1u << (rnd() >> 61));          /* one bit */
        else if (kind == 1) buf[at + (rnd() >> 33) % 128] ^= (unsigned char)(1u << (rnd() >> 61));    /* a lane state */
        else if (kind == 2) buf[at + len - 1 - (rnd() >> 33) % 64] ^= (unsigned char)(1u << (rnd() >> 61)); /* sign words */
        else if (kind == 3) buf[at + (rnd() >> 33) % len] = (unsigned char)rnd();                      /* a byte */
        else if (kind == 4) { for (s = 0; s < 4; s++) buf[at + (rnd() >> 33) % len] ^= (unsigned char)(1u << (rnd() >> 61)); }
        else {   /* swap two neighbouring streams' bodies when they have the same length: hash / ident catches it */
            if (pick + 1 < ns && si[pick + 1].end - si[pick + 1].begin == len) {
                unsigned char *tmp = (unsigned char *)malloc(len);
                memcpy(tmp, buf + at, len); memcpy(buf + at, buf + at + len, len); memcpy(buf + at + len, tmp, len); free(tmp);
            } else buf[at + 128 + (rnd() >> 33) % (len - 128)] ^= 0x80;
        }
        if (wnx_parse(buf, r.wxn[fmt], &r.lay, &models[fmt], &v)) { parse++; free_rec(&r); continue; }
        memset(slabr, 0, r.raw);
        sr = wnx_decode_ref(&v, &r.lay, &models[fmt], (unsigned char *[3]){slabr + r.e.scale_offsets[0], slabr + r.e.scale_offsets[1], slabr + r.e.scale_offsets[2]},
                            (unsigned char *[3]){slabr + r.e.weight_offsets[0], slabr + r.e.weight_offsets[1], slabr + r.e.weight_offsets[2]});
        if (sr) rej++; else acc++;
        if (!sr && memcmp(slabr, r.z, r.raw)) { printf("case %ld: reference ACCEPTED A WRONG OUTPUT (kind %d)\n", c, kind); return 1; }
        for (j = 0; j < n; j++) {
            int sf;
            if (d[j].fmt != fmt || d[j].kernel == 0) continue;
            memset(slabf, 0, r.raw);
            sf = run_dec(&d[j], &r, slabf, 1 + (int)(c % 3));
            if ((sf != 0) != (sr != 0) || (!sf && memcmp(slabf, slabr, r.raw))) {
                printf("case %ld (%s, kind %d): reference %d, fast %d\n", c, d[j].name, kind, sr, sf); return 1;
            }
        }
        free_rec(&r);
    }
    printf("mutate: %ld damaged records: %ld rejected by the reference, %ld accepted (accepted ones decode to Z), %ld stopped at the parser; fast kernels agree on all\n",
           N, rej, acc, parse);
    return 0;
}

static int cmd_scal(int nz, char **zs) {
    Dec d[64];
    const int n = make_decs(d, 0);
    Rec r;
    unsigned char *slab = (unsigned char *)aligned_alloc(64, 18800640u + 4096u);
    int j, th, rep;
    (void)nz;
    load_rec(zs[0], &r);
    for (j = 0; j < n; j++) {
        if (j && !strstr(d[j].name, "G=2")) continue;
        printf("%-60s", d[j].name);
        for (th = 1; th <= 4; th++) {
            double best = 1e30;
            for (rep = 0; rep < 7; rep++) { double s = now_ms(); run_dec(&d[j], &r, slab, th); s = now_ms() - s; if (s < best) best = s; }
            printf(" %d thr %.1f ms |", th, best);
        }
        printf("\n");
    }
    return 0;
}
int main(int argc, char **argv) {
    int i, sep = -1;
    for (i = 1; i < argc; i++) if (!strcmp(argv[i], "--")) sep = i;
    printf("build: WNX kernels AVX2%s; wex_fast_decode %s\n", WNXF_512 ? " + AVX-512" : "", WEXF_KERNELS);
    if (argc > 3 && !strcmp(argv[1], "time") && sep > 4) {
        load_models(sep - 4, argv + 4);
        return cmd_time(atoi(argv[2]), atoi(argv[3]), argc - sep - 1, argv + sep + 1);
    }
    if (argc > 2 && !strcmp(argv[1], "scal") && sep > 1) { load_models(sep - 2, argv + 2); return cmd_scal(argc - sep - 1, argv + sep + 1); }
    if (argc > 3 && !strcmp(argv[1], "equal") && sep > 2) { load_models(sep - 2, argv + 2); return cmd_equal(argc - sep - 1, argv + sep + 1); }
    if (argc > 3 && !strcmp(argv[1], "mutate") && sep > 4) {
        load_models(sep - 4, argv + 4);
        return cmd_mutate(atol(argv[2]), strtoull(argv[3], NULL, 0), argc - sep - 1, argv + sep + 1);
    }
    fprintf(stderr, "usage: bench equal OUTDIR... -- Z... | time ROUNDS THREADS OUTDIR... -- Z... | mutate N SEED OUTDIR... -- Z...\n");
    return 2;
}
