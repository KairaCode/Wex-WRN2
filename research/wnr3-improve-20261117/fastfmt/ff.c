/* fastfmt driver: train / eval / enc (see README.md). Build: see build.sh */
#define _GNU_SOURCE
#include "wnx.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static unsigned char *read_file(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    unsigned char *d;
    long n;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    d = (unsigned char *)malloc((size_t)n + 4096u);
    if (!d || fread(d, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    fclose(f);
    *bytes = (size_t)n;
    return d;
}
static void write_file(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fclose(f);
}

typedef struct {
    int layer, expert;
    WexEntry e;
    WexNr2Layout lay;
    unsigned char *z;        /* Z(source) slab (raw layout) */
    unsigned char *coded;    /* coded slab */
    unsigned char *cs[3], *cw[3];
    uint32_t ceil;
    size_t raw;
} Expert;

/* path = data/Z_LL_EEE.bin; the entry is data/E_LL_EEE.bin */
static void load_expert(const char *zpath, Expert *x, int want_coded) {
    char ep[1024];
    size_t n, en;
    unsigned char *eb;
    const char *base = strrchr(zpath, '/');
    uint64_t raw, aligned;
    uint32_t m;
    base = base ? base + 1 : zpath;
    if (sscanf(base, "Z_%d_%d.bin", &x->layer, &x->expert) != 2) { fprintf(stderr, "bad name %s\n", zpath); exit(2); }
    snprintf(ep, sizeof ep, "%.*sE_%02d_%03d.bin", (int)(base - zpath), zpath, x->layer, x->expert);
    eb = read_file(ep, &en);
    if (en != 72) exit(2);
    memcpy(&x->e, eb, 72); free(eb);
    x->z = read_file(zpath, &n);
    if (wex_entry_raw_sizes(&x->e, &raw, &aligned) || n != raw) { fprintf(stderr, "size %s\n", zpath); exit(2); }
    x->raw = raw;
    wex_nr2_layout(&x->e, &x->lay);
    x->coded = NULL;
    if (want_coded) {
        x->coded = (unsigned char *)malloc(raw);
        memcpy(x->coded, x->z, raw);
        if (wex_nr3_canonical_zero(x->coded + x->e.scale_bytes, x->e.weight_bytes)) { fprintf(stderr, "Z has -0?\n"); exit(2); }
        for (m = 0; m < 3u; m++) { x->cs[m] = x->coded + x->e.scale_offsets[m]; x->cw[m] = x->coded + x->e.weight_offsets[m]; }
        x->ceil = wex_nr3_ceil_forward(&x->lay, x->cs, x->cw, NULL);
    }
}
static void free_expert(Expert *x) { free(x->z); free(x->coded); }

/* counts of one coded expert: nib[3][7][16] per srel; pos[3][7][32][16] per srel and position; scales */
typedef struct { uint64_t nib[3][7][16], pos[3][7][32][16], sc[3][256], wnr3[3][567][16]; } Counts;
static void count_expert(const Expert *x, Counts *C) {
    uint32_t m, b, j;
    for (m = 0; m < 3u; m++) {
        const unsigned char a = wex_nr3_anchor(x->cs[m], x->lay.scale_count[m]);
        for (b = 0; b < x->lay.scale_count[m]; b++) {
            const uint32_t r = wex_nr2_srel(x->cs[m][b], a);
            uint32_t prev = 8, mx = 8;
            C->sc[m][x->cs[m][b]]++;
            for (j = 0; j < 32u; j++) {
                const uint32_t nib = (x->cw[m][b * 16u + j / 2u] >> ((j & 1u) * 4u)) & 15u;
                C->nib[m][r][nib]++; C->pos[m][r][j][nib]++;
                C->wnr3[m][wex_nr2_context(r, prev, mx)][nib]++;
                prev = nib & 7u; if (mx == 8u || prev > mx) mx = prev;
            }
        }
    }
}

static int cmd_train(const char *out, int n, char **paths) {
    Counts *C = (Counts *)calloc(1, sizeof(Counts));
    int i;
    for (i = 0; i < n; i++) {
        Expert x;
        load_expert(paths[i], &x, 1);
        count_expert(&x, C);
        free_expert(&x);
    }
    write_file(out, C, sizeof(*C));
    printf("train: %d experts -> %s\n", n, out);
    return 0;
}

/* ideal bits of counts T under CDF c (total 2^P) */
static double bits_of(const uint64_t *T, const uint16_t *c, uint32_t nsym, uint32_t P) {
    double b = 0;
    uint32_t s;
    for (s = 0; s < nsym; s++) if (T[s]) b += (double)T[s] * (P - log2((double)(c[s + 1] - c[s])));
    return b;
}

/* eval TRAINCOUNTS Z...: ideal code lengths (bounds) on the given experts with models from the train counts */
static int cmd_eval(const char *trainc, int n, char **paths) {
    size_t bytes;
    Counts *Tr = (Counts *)read_file(trainc, &bytes), *Te = (Counts *)calloc(1, sizeof(Counts));
    uint32_t m, r, j, s, P;
    double weights = 0;
    int i;
    if (bytes != sizeof(Counts)) return 2;
    for (i = 0; i < n; i++) {
        Expert x;
        load_expert(paths[i], &x, 1);
        count_expert(&x, Te);
        free_expert(&x);
    }
    for (m = 0; m < 3u; m++) for (r = 0; r < 7u; r++) for (s = 0; s < 16u; s++) weights += (double)Te->nib[m][r][s];
    printf("eval: %d experts, %.0f weights (%.0f per expert)\n", n, weights, weights / n);
    for (P = 10; P <= 16; P += 2) {
        double mag = 0, joint = 0, signs = 0, posmag = 0, wnr3 = 0, sc = 0;
        for (m = 0; m < 3u; m++) {
            for (r = 0; r < 7u; r++) {
                uint64_t cm[8] = {0}, tm[8] = {0}, cj[15], tj[15];
                uint16_t cdf[257];
                for (s = 0; s < 16u; s++) { cm[s & 7u] += Tr->nib[m][r][s]; tm[s & 7u] += Te->nib[m][r][s]; if (s & 7u) signs += (double)Te->nib[m][r][s]; }
                wnx_quantize(cm, 8, 1u << P, cdf); mag += bits_of(tm, cdf, 8, P);
                for (s = 0; s < 16u; s++) if (s != 8u) { cj[wnx_jsym_of(s)] = Tr->nib[m][r][s]; tj[wnx_jsym_of(s)] = Te->nib[m][r][s]; }
                wnx_quantize(cj, 15, 1u << P, cdf); joint += bits_of(tj, cdf, 15, P);
                for (j = 0; j < 32u; j++) {
                    uint64_t pm[8] = {0}, qm[8] = {0};
                    for (s = 0; s < 16u; s++) { pm[s & 7u] += Tr->pos[m][r][j][s]; qm[s & 7u] += Te->pos[m][r][j][s]; }
                    wnx_quantize(pm, 8, 1u << P, cdf); posmag += bits_of(qm, cdf, 8, P);
                }
            }
            for (r = 0; r < 567u; r++) {
                uint16_t cdf[17];
                if (P > 14) break;
                wnx_quantize(Tr->wnr3[m][r], 16, 1u << P, cdf); wnr3 += bits_of(Te->wnr3[m][r], cdf, 16, P);
            }
            {
                uint16_t cdf[257];
                wnx_quantize(Tr->sc[m], 256, 1u << P, cdf); sc += bits_of(Te->sc[m], cdf, 256, P);
            }
        }
        printf("P=%2u bit/w: mag7+sign %.5f  joint15 %.5f  pos(7x32)mag+sign %.5f  wnr3-567ctx %.5f | scales %.0f B/expert\n", P,
               (mag + signs) / weights, joint / weights, (posmag + signs) / weights, P <= 14 ? wnr3 / weights : 0.0, sc / 8 / n);
    }
    return 0;
}

/* enc MODELKIND(mag|joint) TRAINCOUNTS Bw Bs OUTDIR Z...: encode each, reference-decode, compare with Z(source) */
static int cmd_enc(const char *kind, const char *trainc, uint32_t Bw, uint32_t Bs, const char *outdir, int n, char **paths) {
    size_t bytes;
    Counts *Tr = (Counts *)read_file(trainc, &bytes);
    WnxModel *M = (WnxModel *)malloc(sizeof(WnxModel));
    double sum_used = 0, sum_total = 0, sum_wnr3 = 0;
    int i, bad = 0;
    char p[1024];
    if (bytes != sizeof(Counts)) return 2;
    wnx_model_build(M, !strcmp(kind, "joint"), (const uint64_t(*)[7][16])Tr->nib, (const uint64_t(*)[256])Tr->sc);
    snprintf(p, sizeof p, "%s/model.wnxm", outdir);
    write_file(p, M->blob, M->blob_bytes);
    printf("model %s id 0x%08x, %u bytes; Bw %u Bs %u\n", kind, M->id, M->blob_bytes, Bw, Bs);
    printf("| expert | WNR3 stored B | WNX used B | WNX padded B | delta padded vs WNR3, %% of raw | ref decode == Z |\n|---|---:|---:|---:|---:|---|\n");
    for (i = 0; i < n; i++) {
        Expert x;
        size_t cap, total, used, wb;
        unsigned char *rec, *outbuf, *so[3], *wo[3];
        WnxView v;
        uint32_t m;
        int ok;
        load_expert(paths[i], &x, 1);
        cap = x.raw * 2u;
        rec = (unsigned char *)malloc(cap);
        used = wnx_encode(M, &x.lay, x.cs, x.cw, x.ceil, (uint32_t)x.layer << 16 | (uint32_t)x.expert, Bw, Bs, !strcmp(kind, "magt"), rec, cap, &total);
        if (!used) { printf("encode failed %s\n", paths[i]); return 1; }
        snprintf(p, sizeof p, "/tmp/wnr3-data/recs/rec_%02d_%03d.bin", x.layer, x.expert);
        { unsigned char *w3 = read_file(p, &wb); free(w3); }
        snprintf(p, sizeof p, "%s/X_%02d_%03d.wnx", outdir, x.layer, x.expert);
        write_file(p, rec, total);
        outbuf = (unsigned char *)malloc(x.raw);
        memset(outbuf, 0xa5, x.raw);
        for (m = 0; m < 3u; m++) { so[m] = outbuf + x.e.scale_offsets[m]; wo[m] = outbuf + x.e.weight_offsets[m]; }
        ok = !wnx_parse(rec, total, &x.lay, M, &v) && !wnx_decode_ref(&v, &x.lay, M, so, wo) && !memcmp(outbuf, x.z, x.raw);
        bad += !ok;
        printf("| L%02d E%03d | %zu | %zu | %zu | %+.4f | %s |\n", x.layer, x.expert, wb, used, total,
               100.0 * ((double)total - (double)wb) / (double)x.raw, ok ? "yes" : "NO");
        fflush(stdout);
        sum_used += (double)used; sum_total += (double)total; sum_wnr3 += (double)wb;
        free(rec); free(outbuf); free_expert(&x);
    }
    printf("mean over %d: WNR3 %.1f B (%.5f of raw), WNX used %.1f B, padded %.1f B (%.5f of raw); delta padded %+.4f%% of raw, used vs WNR3 %+.4f%%; round trips failed: %d\n",
           n, sum_wnr3 / n, sum_wnr3 / n / 18800640.0, sum_used / n, sum_total / n, sum_total / n / 18800640.0,
           100.0 * (sum_total - sum_wnr3) / n / 18800640.0, 100.0 * (sum_used - sum_wnr3) / n / 18800640.0, bad);
    return bad != 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "train")) return cmd_train(argv[2], argc - 3, argv + 3);
    if (argc >= 4 && !strcmp(argv[1], "eval")) return cmd_eval(argv[2], argc - 3, argv + 3);
    if (argc >= 8 && !strcmp(argv[1], "enc")) return cmd_enc(argv[2], argv[3], (uint32_t)atoi(argv[4]), (uint32_t)atoi(argv[5]), argv[6], argc - 7, argv + 7);
    fprintf(stderr, "usage: ff train OUT Z... | eval TRAINC Z... | enc mag|magt|joint TRAINC Bw Bs OUTDIR Z...\n");
    return 2;
}
