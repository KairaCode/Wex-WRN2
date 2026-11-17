/* rt.c - real rANS round trip of the w3 (matrix 2) weight streams of one record under a static-table context
 * variant that may use the sibling w1 (matrix 0). Uses the production codec primitives of
 * Wex-WRN2/c/wex_neural_rans.h unchanged (wex_nr2_put / wex_nr2_flush / wex_nr3_stream_x0 /
 * wex_nr2_stream_open / wex_nr2_decode_symbol): 32-bit state, L = 2^23, 14-bit CDFs, byte renormalisation,
 * Cw-symbol streams, x0 = L + h23 (MurmurHash3 over the record's real ident), 4-B flush.
 *
 * Check 1 (variant 0 with the production WNM2 tables): every w3 stream is byte-identical to the stored record's.
 * Check 2 (any variant): encode -> decode (contexts from the decoded w3 nibbles and the w1 nibbles/scales, which
 * WNR3 decodes unchanged before) -> decoded coded nibbles == the coded slab, final state == L + h23.
 * Since w1, w2, scales and their streams are untouched, the rest of the record is identical, and equal coded
 * nibbles + scales + flags give the same WNR3-CZ output through the unchanged ceil inverse.
 *
 * Build: gcc -O2 -I/workspace/Wex-WRN2/c -o rt rt.c
 * Run:   ./rt CODED RECORD MODEL.wnm2 TABLE.u16|- VARIANT soff0..2 woff0..2 R0 NB0 R1 NB1 R2 NB2
 * Prints one JSON line. */
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
static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

static int V;  /* variant */
static uint32_t ctx_of(int srel, int pm, int mx, int sibd, int sa) {
    int base = (srel * 9 + pm) * 9 + mx;
    switch (V) {
    case 0: return base;
    case 1: return base * 8 + sa;
    case 2: return (base * 5 + sibd) * 8 + sa;
    case 3: return ((srel * 9 + mx) * 5 + sibd) * 8 + sa;
    case 4: return (srel * 5 + sibd) * 8 + sa;
    case 5: return base * 5 + sibd;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 18) { fprintf(stderr, "usage: see source\n"); return 2; }
    size_t n, rn, mn, tn = 0;
    unsigned char *coded = readall(argv[1], &n), *rec = readall(argv[2], &rn), *blob = readall(argv[3], &mn);
    unsigned char *table = strcmp(argv[4], "-") ? readall(argv[4], &tn) : NULL;
    V = atoi(argv[5]);
    long so[3], wo[3]; int R[3], NB[3];
    for (int m = 0; m < 3; m++) { so[m] = atol(argv[6 + m]); wo[m] = atol(argv[9 + m]); R[m] = atoi(argv[12 + 2 * m]); NB[m] = atoi(argv[13 + 2 * m]); }
    WexNr2Model model;
    if (wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) { fprintf(stderr, "model\n"); return 1; }
    uint32_t Cw = rd32(rec + 16), Cs = rd32(rec + 20);
    uint32_t ident[4] = {rd32(rec + 4), rd32(rec + 28), 0, rd32(rec + 24)};
    int anc[3] = {rec[24], rec[25], rec[26]};
    if (ident[0] != model.id) { fprintf(stderr, "model id mismatch\n"); return 1; }
    uint32_t S = 0, Wst[3];
    for (int m = 0; m < 3; m++) { S += (R[m] * NB[m] + Cs - 1) / Cs; Wst[m] = (R[m] * NB[m] * 32 + Cw - 1) / Cw; }
    uint32_t g0 = S + Wst[0] + Wst[1], nstreams = S + Wst[0] + Wst[1] + Wst[2];
    /* stored w3 streams: offset in the weight payload */
    const unsigned char *lens = rec + 32, *wpay = rec + 32 + 2 * nstreams + rd32(rec + 8);
    size_t off = 0; for (uint32_t g = S; g < g0; g++) off += wex_nr_u16(lens + 2 * g);
    const unsigned char *stored = wpay + off;
    const unsigned char *s1 = coded + so[0], *s3 = coded + so[2], *w1 = coded + wo[0], *w3 = coded + wo[2];
    uint32_t count3 = (uint32_t)R[2] * NB[2] * 32;
    unsigned char tmp[2 * WEX_NR2_MAX_CHUNK + 8];
    static uint32_t ctx[WEX_NR2_MAX_CHUNK];
    unsigned char *dec = calloc(count3 / 2, 1);
    uint64_t bytes = 0, stored_bytes = 0; long mism_stream = 0, bad = 0;
    const unsigned char *sp = stored;
    for (uint32_t first = 0, g = g0; first < count3; first += Cw, g++) {
        uint32_t cnt = count3 - first < Cw ? count3 - first : Cw, srel = 0, pm = 8, mx = 8, sibd = 2;
        /* contexts, forward */
        for (uint32_t j = 0; j < cnt; j++) {
            uint32_t i = first + j, x = (w3[i / 2] >> ((i & 1) * 4)) & 15, sa = (w1[i / 2] >> ((i & 1) * 4)) & 7;
            if (!(i % 32)) { srel = wex_nr2_srel(s3[i / 32], (uint32_t)anc[2]); pm = mx = 8; sibd = (uint32_t)(clampi((int)s1[i / 32] - (int)s3[i / 32], -2, 2) + 2); }
            ctx[j] = ctx_of((int)srel, (int)pm, (int)mx, (int)sibd, (int)sa);
            pm = x & 7; if (mx == 8 || pm > mx) mx = pm;
        }
        ident[2] = g;
        uint32_t state = wex_nr3_stream_x0(ident, w3 + first / 2, cnt / 2), x0 = state;
        unsigned char *end = tmp + sizeof tmp, *p = end;
        for (uint32_t j = cnt; j-- > 0;) {
            uint32_t i = first + j, x = (w3[i / 2] >> ((i & 1) * 4)) & 15;
            const unsigned char *cdf = table ? table + (size_t)ctx[j] * 34 : wex_nr2_weight_cdf(&model, 2, ctx[j]);
            if (wex_nr2_put(&state, &p, tmp, cdf, x)) { fprintf(stderr, "put\n"); return 1; }
        }
        wex_nr2_flush(state, &p, tmp);
        uint32_t len = (uint32_t)(end - p), slen = wex_nr_u16(lens + 2 * g);
        bytes += len; stored_bytes += slen;
        if (len != slen || memcmp(p, sp, len)) mism_stream++;
        sp += slen;
        /* decode the stream just written */
        const unsigned char *ptr, *e2; uint32_t st, sym;
        if (wex_nr2_stream_open(p, len, &ptr, &e2, &st)) { bad++; continue; }
        srel = 0; pm = mx = 8; sibd = 2;
        for (uint32_t j = 0; j < cnt; j++) {
            uint32_t i = first + j, sa = (w1[i / 2] >> ((i & 1) * 4)) & 7;
            if (!(i % 32)) { srel = wex_nr2_srel(s3[i / 32], (uint32_t)anc[2]); pm = mx = 8; sibd = (uint32_t)(clampi((int)s1[i / 32] - (int)s3[i / 32], -2, 2) + 2); }
            uint32_t c = ctx_of((int)srel, (int)pm, (int)mx, (int)sibd, (int)sa);
            const unsigned char *cdf = table ? table + (size_t)c * 34 : wex_nr2_weight_cdf(&model, 2, c);
            if (wex_nr2_decode_symbol(cdf, 16, &st, &ptr, e2, &sym) || sym == 8) { bad++; break; }
            if (!(i & 1)) dec[i / 2] = (unsigned char)sym; else dec[i / 2] |= (unsigned char)(sym << 4);
            pm = sym & 7; if (mx == 8 || pm > mx) mx = pm;
        }
        if (ptr != e2 || st != x0) bad++;
    }
    int equal = !memcmp(dec, w3, count3 / 2);
    printf("{\"variant\": %d, \"table\": \"%s\", \"w3_streams\": %u, \"w3_bytes\": %llu, \"stored_w3_bytes\": %llu, "
           "\"streams_not_identical_to_stored\": %ld, \"decode_failures\": %ld, \"roundtrip_equal\": %d}\n",
           V, argv[4], Wst[2], (unsigned long long)bytes, (unsigned long long)stored_bytes, mism_stream, bad, equal);
    return !(equal && !bad);
}
