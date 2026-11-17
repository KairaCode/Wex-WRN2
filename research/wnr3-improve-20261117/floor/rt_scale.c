/* rt_scale.c - real rANS round trip of the SCALE streams of one record with a per-record order-0 table per matrix
 * (built from the record's own coded scales: wex_nr2_quantize's rule over the USED symbols only, unused symbols
 * get frequency 0), against the production WNM2 scale tables. Same primitives as WNR3 (wex_neural_rans.h).
 * Check 1: production tables -> every scale stream byte-identical to the stored record's.
 * Check 2: own tables -> decode gives the coded scales back, final state == L + h23.
 * Table cost to add to a record: 1 count byte + 3 B (symbol, u16 frequency) per used symbol, per matrix.
 * Build: gcc -O2 -I/workspace/Wex-WRN2/c -o rt_scale rt_scale.c
 * Run:   ./rt_scale CODED RECORD MODEL.wnm2 soff0..2 woff0..2 R0 NB0 R1 NB1 R2 NB2 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wex_neural_rans.h"

static unsigned char *readall(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b;
}
static uint32_t rd32(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

/* used-symbol quantiser: frequency 1 + floor(c * spare / total) for used symbols, remainder by largest remainder */
static int own_cdf(const uint64_t *c, unsigned char *cdf) {
    uint64_t total = 0, rem[256]; uint32_t f[256] = {0}, used = 0, sum = 0, nu = 0; unsigned char taken[256] = {0};
    for (int s = 0; s < 256; s++) if (c[s]) { total += c[s]; nu++; }
    uint32_t spare = 16384 - nu;
    for (int s = 0; s < 256; s++) if (c[s]) { f[s] = 1 + (uint32_t)(c[s] * spare / total); rem[s] = c[s] * spare % total; sum += f[s]; }
    for (uint32_t extra = 16384 - sum; extra; extra--) {
        int best = -1;
        for (int s = 0; s < 256; s++) if (c[s] && !taken[s] && (best < 0 || rem[s] > rem[best])) best = s;
        taken[best] = 1; f[best]++;
    }
    for (int s = 0; s <= 256; s++) { cdf[2 * s] = (unsigned char)used; cdf[2 * s + 1] = (unsigned char)(used >> 8); if (s < 256) used += f[s]; }
    return (int)nu;
}

int main(int argc, char **argv) {
    if (argc != 16) { fprintf(stderr, "usage: see source\n"); return 2; }
    size_t n, rn, mn;
    unsigned char *coded = readall(argv[1], &n), *rec = readall(argv[2], &rn), *blob = readall(argv[3], &mn);
    long so[3]; int R[3], NB[3];
    for (int m = 0; m < 3; m++) { so[m] = atol(argv[4 + m]); R[m] = atoi(argv[10 + 2 * m]); NB[m] = atoi(argv[11 + 2 * m]); }
    WexNr2Model model;
    if (wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) return 1;
    uint32_t Cw = rd32(rec + 16), Cs = rd32(rec + 20), ident[4] = {rd32(rec + 4), rd32(rec + 28), 0, rd32(rec + 24)};
    uint32_t nstreams = 0;
    for (int m = 0; m < 3; m++) nstreams += (R[m] * NB[m] + Cs - 1) / Cs + (R[m] * NB[m] * 32 + Cw - 1) / Cw;
    const unsigned char *lens = rec + 32, *sp = rec + 32 + 2 * nstreams;
    unsigned char tmp[2 * WEX_NR2_MAX_CHUNK + 8], owncdf[3][514];
    uint64_t bytes[2] = {0, 0}, stored = 0; long mism = 0, bad = 0, tbytes = 0; int equal = 1;
    unsigned char *dec = malloc(WEX_NR2_MAX_CHUNK);
    for (int m = 0; m < 3; m++) {
        uint64_t c[256] = {0}; uint32_t cnt = (uint32_t)R[m] * NB[m];
        for (uint32_t i = 0; i < cnt; i++) c[coded[so[m] + i]]++;
        tbytes += 1 + 3 * own_cdf(c, owncdf[m]);
    }
    uint32_t g = 0;
    for (int m = 0; m < 3; m++) {
        const unsigned char *s = coded + so[m]; uint32_t cnt_all = (uint32_t)R[m] * NB[m];
        for (uint32_t first = 0; first < cnt_all; first += Cs, g++) {
            uint32_t cnt = cnt_all - first < Cs ? cnt_all - first : Cs, slen = wex_nr_u16(lens + 2 * g);
            stored += slen;
            for (int v = 0; v < 2; v++) {
                const unsigned char *cdf = v ? owncdf[m] : blob + 57866 + m * 514;
                ident[2] = g;
                uint32_t state = wex_nr3_stream_x0(ident, s + first, cnt), x0 = state;
                unsigned char *end = tmp + sizeof tmp, *p = end;
                for (uint32_t j = cnt; j-- > 0;) if (wex_nr2_put(&state, &p, tmp, cdf, s[first + j])) return 1;
                wex_nr2_flush(state, &p, tmp);
                uint32_t len = (uint32_t)(end - p); bytes[v] += len;
                if (!v && (len != slen || memcmp(p, sp, len))) mism++;
                const unsigned char *ptr, *e2; uint32_t st, sym;
                if (wex_nr2_stream_open(p, len, &ptr, &e2, &st)) { bad++; continue; }
                for (uint32_t j = 0; j < cnt; j++) { if (wex_nr2_decode_symbol(cdf, 256, &st, &ptr, e2, &sym)) { bad++; break; } dec[j] = (unsigned char)sym; }
                if (ptr != e2 || st != x0) bad++;
                if (memcmp(dec, s + first, cnt)) equal = 0;
            }
            sp += slen;
        }
    }
    printf("{\"scale_streams\": %u, \"prod_bytes\": %llu, \"stored_bytes\": %llu, \"own_bytes\": %llu, \"own_table_bytes\": %ld, "
           "\"streams_not_identical_to_stored\": %ld, \"decode_failures\": %ld, \"roundtrip_equal\": %d}\n", g,
           (unsigned long long)bytes[0], (unsigned long long)stored, (unsigned long long)bytes[1], tbytes, mism, bad, equal);
    return !(equal && !bad && !mism);
}
