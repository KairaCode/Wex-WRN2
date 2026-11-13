/* Corruption audit on REAL V4.1 records (lab tool, KAIRA-private).
 * Every single-bit flip is decoded like the engine does it (owner/profile check, then the CPU
 * decoder) and classified: rejected (slab untouched), harmless (decodes to exactly the expected
 * slab) or WRONG (accepted with other bytes).
 *   exhaustive: every bit of the record header (32 B), the first 64 B of the length table, and
 *   the first scale stream and the first weight stream of each matrix;
 *   random: R flips uniform over the used bytes (header + lengths + payloads), seeded.
 * usage: audit_real BIN HEADER_FLAGS R SEED idx:layer:expert ...   (one JSON line per record)
 * build: gcc -O2 -fopenmp -I<worktree>/src/colibri/c audit_real.c -o audit_real.exe */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wex_format.h"

typedef struct { WexEntry e; unsigned char *expected, *record; uint32_t raw; } Rec;
typedef struct { uint64_t flips, rejected, harmless, wrong, touched; } Tally;

static WexNr2Model model;
static uint32_t header_flags;

/* engine order: owner/profile check, then decode into the slab */
static int decode(const Rec *r, const unsigned char *rec, unsigned char *slab, size_t cap, uint32_t L, uint32_t E) {
    size_t at = wex_record_read_offset(&r->e);
    memset(slab, 0x5a, cap);
    memcpy(slab + at, rec, r->e.total_bytes);
    if (!wex_nr3_owner_ok(&r->e, rec, L, E, header_flags)) return -1;
    return wex_decode_nr2_record(&r->e, slab, cap, &model);
}

static void one(const Rec *r, size_t byte, int bit, uint32_t L, uint32_t E, unsigned char *rec, unsigned char *slab,
                size_t cap, Tally *t) {
    rec[byte] ^= (unsigned char)(1u << bit);
    int st = decode(r, rec, slab, cap, L, E);
    rec[byte] ^= (unsigned char)(1u << bit);
    t->flips++;
    if (st) { t->rejected++; if (slab[0] != 0x5a) t->touched++; }
    else if (!memcmp(slab, r->expected, r->raw)) t->harmless++;
    else t->wrong++;
}

static void add(Tally *a, const Tally *b) {
    a->flips += b->flips; a->rejected += b->rejected; a->harmless += b->harmless; a->wrong += b->wrong;
    a->touched += b->touched;
}

/* every bit of bytes [lo, hi) of the record (or n random bits of [0, lo) when n > 0) */
static Tally run(const Rec *r, size_t lo, size_t hi, uint64_t n, uint64_t seed, uint32_t L, uint32_t E) {
    Tally all = {0};
    uint64_t raw, aligned;
    long long count = n ? (long long)n : (long long)(hi - lo) * 8;
    wex_entry_raw_sizes(&r->e, &raw, &aligned);
    #pragma omp parallel
    {
        Tally t = {0};
        size_t cap = (size_t)aligned;
        unsigned char *slab = malloc(cap), *rec = malloc(r->e.total_bytes);
        memcpy(rec, r->record, r->e.total_bytes);
        #pragma omp for schedule(dynamic, 16)
        for (long long k = 0; k < count; k++) {
            if (n) {
                uint64_t x = seed + (uint64_t)k * 0x9E3779B97F4A7C15ull;   /* splitmix64 */
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; x ^= x >> 31;
                one(r, (size_t)((x >> 3) % lo), (int)(x & 7), L, E, rec, slab, cap, &t);
            } else one(r, lo + (size_t)(k / 8), (int)(k % 8), L, E, rec, slab, cap, &t);
        }
        #pragma omp critical
        add(&all, &t);
        free(slab); free(rec);
    }
    return all;
}

static void print_tally(const char *name, const Tally *t) {
    printf("\"%s\":{\"flips\":%llu,\"rejected\":%llu,\"harmless\":%llu,\"wrong\":%llu,\"slab_touched_on_reject\":%llu},",
           name, (unsigned long long)t->flips, (unsigned long long)t->rejected, (unsigned long long)t->harmless,
           (unsigned long long)t->wrong, (unsigned long long)t->touched);
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: audit_real BIN HEADER_FLAGS R SEED idx:L:E ...\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    FILE *f = fopen(argv[1], "rb");
    uint32_t mb, n;
    if (!f || fread(&mb, 4, 1, f) != 1) return 1;
    unsigned char *blob = malloc(mb);
    if (fread(blob, 1, mb, f) != mb || fread(&n, 4, 1, f) != 1 || wex_nr2_model_open(blob, mb, &model)) return 1;
    header_flags = (uint32_t)strtoul(argv[2], NULL, 0);
    uint64_t R = strtoull(argv[3], NULL, 0), seed = strtoull(argv[4], NULL, 0);
    /* only the requested records are kept in memory */
    unsigned want[64][3], nw = 0;
    for (int a = 5; a < argc && nw < 64; a++, nw++)
        if (sscanf(argv[a], "%u:%u:%u", &want[nw][0], &want[nw][1], &want[nw][2]) != 3 || want[nw][0] >= n) return 2;
    Rec *recs = calloc(n, sizeof(Rec));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rb, keep = 0;
        for (unsigned w = 0; w < nw; w++) keep |= want[w][0] == i;
        if (fread(&recs[i].e, sizeof(WexEntry), 1, f) != 1 || fread(&rb, 4, 1, f) != 1) return 1;
        recs[i].raw = rb;
        if (!keep) { _fseeki64(f, (long long)rb + recs[i].e.total_bytes, SEEK_CUR); continue; }
        recs[i].expected = malloc(rb); recs[i].record = malloc(recs[i].e.total_bytes);
        if (fread(recs[i].expected, 1, rb, f) != rb ||
            fread(recs[i].record, 1, recs[i].e.total_bytes, f) != recs[i].e.total_bytes) return 1;
    }
    fclose(f);
    for (unsigned w = 0; w < nw; w++) {
        unsigned idx = want[w][0], L = want[w][1], E = want[w][2];
        const Rec *r = &recs[idx];
        uint64_t raw, aligned;
        wex_entry_raw_sizes(&r->e, &raw, &aligned);
        unsigned char *slab = malloc((size_t)aligned);
        double t0 = omp_get_wtime();
        if (decode(r, r->record, slab, (size_t)aligned, L, E) || memcmp(slab, r->expected, r->raw)) {
            fprintf(stderr, "record %u L%u E%u: clean decode failed\n", idx, L, E); return 1;
        }
        double clean_s = omp_get_wtime() - t0;
        free(slab);
        WexNr2Layout layout; WexNr2Record view;
        wex_nr2_layout(&r->e, &layout);
        if (wex_nrx_parse(r->record, r->e.total_bytes, &layout, model.id, wex_entry_nr_magic(&r->e), &view)) return 1;
        size_t used = view.metadata_bytes + view.scale_payload_bytes + view.weight_payload_bytes;
        uint32_t ns = view.scale_streams + view.weight_streams, s, m;
        size_t *off = malloc((ns + 1) * sizeof(size_t));
        off[0] = view.metadata_bytes;
        for (s = 0; s < ns; s++) off[s + 1] = off[s] + wex_nr_u16(view.lengths + s * 2u);
        uint32_t picks[12], np = 0, base = 0;
        for (m = 0; m < 3u; m++) { picks[np++] = base; base += wex_nr2_streams(layout.scale_count[m], view.scale_chunk); }
        for (m = 0; m < 3u; m++) {
            uint32_t per = wex_nr2_streams(layout.symbol_count[m], view.weight_chunk);
            picks[np++] = base;   /* shortcut: first weight stream per matrix only (CPU cost); the GPU audit sweeps more */
            base += per;
        }
        Tally ex = {0}, sc = {0}, wt = {0}, hdr, len, t, rnd;
        size_t sbytes = 0, wbytes = 0;
        t0 = omp_get_wtime();
        hdr = run(r, 0, 32, 0, 0, L, E); add(&ex, &hdr);
        len = run(r, 32, 96, 0, 0, L, E); add(&ex, &len);
        for (uint32_t p = 0; p < np; p++) {
            size_t a = off[picks[p]], b = off[picks[p] + 1];
            t = run(r, a, b, 0, 0, L, E);
            if (p < 3) { add(&sc, &t); sbytes += b - a; } else { add(&wt, &t); wbytes += b - a; }
            add(&ex, &t);
        }
        double ex_s = omp_get_wtime() - t0;
        t0 = omp_get_wtime();
        rnd = run(r, used, 0, R, seed * 1000003u + idx, L, E);
        double rnd_s = omp_get_wtime() - t0;
        printf("{\"record\":%u,\"layer\":%u,\"expert\":%u,\"codec\":%u,\"cz\":%d,\"total\":%u,\"used\":%zu,"
               "\"streams\":%u,\"clean_decode_s\":%.3f,\"exhaustive_scale_bytes\":%zu,\"exhaustive_weight_bytes\":%zu,",
               idx, L, E, r->e.flags, r->e.flags == WEX_ENTRY_NEURAL_RANS3 && (r->record[27] & WEX_NR3_FLAG_CZ) ? 1 : 0,
               r->e.total_bytes, used, ns, clean_s, sbytes, wbytes);
        print_tally("ex_header", &hdr); print_tally("ex_lengths64", &len); print_tally("ex_scale_streams", &sc);
        print_tally("ex_weight_streams", &wt); print_tally("exhaustive", &ex); print_tally("random", &rnd);
        printf("\"exhaustive_s\":%.1f,\"random_s\":%.1f}\n", ex_s, rnd_s);
        free(off);
    }
    return 0;
}
