/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 record candidates, CPU prototype (research only; c/ is not touched).
 *
 * Reads cached production WNR3-CZ records (fetch.py), validates the harness
 * (reference decode -> wex_nr3_prepare(cz=1) -> wex_nr3_encode_record must
 * reproduce the stored record byte for byte), then re-encodes each expert
 * with every candidate profile below, decodes it again and compares with the
 * reference slab Z(source). The coder is a copy of wex_nrx_encode /
 * wex_nr2_decode with four switches; contexts, CDFs, model and the
 * independent-stream structure are unchanged.
 *
 * Candidate record: 40-B header {magic, model id, scale payload B, weight
 * payload B, weight chunk, scale chunk, anchors|flags<<24, owner, record
 * digest, length-section B}, length section, scale payload, weight payload.
 * (The baseline profile writes the WNR3 32-B header so it can be compared
 * with the production record.)
 *
 * Switches:
 *   lenc  0: u16 length per stream (WNR3).
 *         1: streams in g order, groups of 32: u16 min, u8 width w (0..16),
 *            then the residuals len - min packed LSB-first at w bits (4w B for
 *            a full group). The host turns it into the same u32 offsets table
 *            it builds today by prefix sum; the kernels do not see it.
 *   sd    0: 4-B big-endian initial state (WNR3).
 *         1: the initial state is stored in its minimal 3 or 4 bytes. The
 *            decoder starts from state 0 and runs the normal renormalisation
 *            loop (read a byte while state < L): a first byte >= 0x80 gives a
 *            state >= L after 3 bytes, anything else needs the 4th.
 *   k     the encoder starts at x0 = 2^k + payload, payload < 2^k. k = 23 is
 *         WNR3 (x0 in [L, 2L)). k < 23 is the "small initial state": the
 *         decoder may hold a state < L once the stream is exhausted, so its
 *         renormalisation stops at the stream end instead of failing there.
 *   tail  the last `tail` symbols of a stream (4-bit nibbles for weights,
 *         8-bit scales) are not coded but stored raw in the low bits of the
 *         payload; the remaining k - bits(tail) payload bits hold the low
 *         bits of the stream's MurmurHash3 (WNR3 = 23 hash bits, no tail).
 * Decoder check per stream: exact length consumption, 2^k <= final state <
 * 2^(k+1), tail symbols read from the low payload bits, hash field equal to
 * the low bits of the hash of the whole decoded stream. The XOR of every
 * stream's full 32-bit hash is the record digest (header word), which is what
 * covers a record whose per-stream hash field is short or absent.
 *
 * Build and run: see run.sh. */
#include "wex_nr3_tools.c"   /* staged copy: wex_format.h, wex_neural_rans.h and the encoder DLL functions */
#include <stdio.h>
#include <math.h>
#include <omp.h>

typedef struct {
    const char *name;
    uint32_t wchunk, schunk, lenc, sd, k, tail_w, tail_s;
} Cfg;

static const Cfg cfgs[] = {
    {"base_2048_512", 2048, 512, 0, 0, 23, 0, 0},            /* = WNR3, must equal the production record */
    {"lenc", 2048, 512, 1, 0, 23, 0, 0},
    {"sd", 2048, 512, 0, 1, 23, 0, 0},
    {"w4096", 4096, 512, 0, 0, 23, 0, 0},
    {"w8192", 8192, 512, 0, 0, 23, 0, 0},
    {"w16384", 16384, 512, 0, 0, 23, 0, 0},
    {"s1024", 2048, 1024, 0, 0, 23, 0, 0},
    {"s2048", 2048, 2048, 0, 0, 23, 0, 0},
    {"w4096_s1024", 4096, 1024, 0, 0, 23, 0, 0},
    {"w8192_s2048", 8192, 2048, 0, 0, 23, 0, 0},
    {"tail1_hash19", 2048, 512, 0, 0, 23, 1, 0},
    {"tail2_hash15", 2048, 512, 0, 0, 23, 2, 0},
    {"tail3_hash11", 2048, 512, 0, 0, 23, 3, 0},
    {"tail4_hash7", 2048, 512, 0, 0, 23, 4, 0},
    {"tail5_hash3", 2048, 512, 0, 0, 23, 5, 0},
    {"tail5_stail2", 2048, 512, 0, 0, 23, 5, 2},            /* the drafts' TAIL profile */
    {"small_k16", 2048, 512, 0, 0, 16, 0, 0},
    {"small_k8", 2048, 512, 0, 0, 8, 0, 0},
    {"small_k0", 2048, 512, 0, 0, 0, 0, 0},
    {"small_k20_tail5", 2048, 512, 0, 0, 20, 5, 0},
    {"lenc_sd", 2048, 512, 1, 1, 23, 0, 0},                 /* integrity and stream count unchanged */
    {"lenc_sd_tail2", 2048, 512, 1, 1, 23, 2, 0},
    {"lenc_sd_tail5", 2048, 512, 1, 1, 23, 5, 0},
    {"lenc_sd_w4096_s1024", 4096, 1024, 1, 1, 23, 0, 0},
    {"lenc_sd_w4096_s1024_tail5", 4096, 1024, 1, 1, 23, 5, 0},
    {"lenc_sd_w8192_s2048", 8192, 2048, 1, 1, 23, 0, 0},
    {"lenc_sd_w8192_s2048_tail5", 8192, 2048, 1, 1, 23, 5, 0},
};
#define NCFG (sizeof(cfgs) / sizeof(cfgs[0]))
#define HDR4 40u
#define MAGIC4 UINT32_C(0x34524e57)

static void put32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24); }
static uint32_t mask(uint32_t bits) { return bits >= 32u ? UINT32_MAX : (1u << bits) - 1u; }

/* ---- length section ------------------------------------------------------ */
static size_t lenc_pack(const uint32_t *len, uint32_t n, unsigned char *out) {
    size_t at = 0;
    uint32_t g, i;
    for (g = 0; g < n; g += 32u) {
        uint32_t cnt = n - g < 32u ? n - g : 32u, lo = len[g], hi = len[g], w = 0;
        uint64_t acc = 0;
        uint32_t have = 0;
        for (i = 1; i < cnt; i++) { if (len[g + i] < lo) lo = len[g + i]; if (len[g + i] > hi) hi = len[g + i]; }
        while ((hi - lo) >> w) w++;
        out[at++] = (unsigned char)lo; out[at++] = (unsigned char)(lo >> 8); out[at++] = (unsigned char)w;
        for (i = 0; i < cnt; i++) {
            acc |= (uint64_t)(len[g + i] - lo) << have; have += w;
            while (have >= 8u) { out[at++] = (unsigned char)acc; acc >>= 8; have -= 8u; }
        }
        if (have) out[at++] = (unsigned char)acc;
    }
    return at;
}
/* Returns the section size, 0 on a malformed section. */
static size_t lenc_unpack(const unsigned char *in, size_t cap, uint32_t n, uint32_t *len) {
    size_t at = 0;
    uint32_t g, i;
    for (g = 0; g < n; g += 32u) {
        uint32_t cnt = n - g < 32u ? n - g : 32u, lo, w, have = 0;
        uint64_t acc = 0;
        if (cap - at < 3u) return 0;
        lo = wex_nr_u16(in + at); w = in[at + 2u]; at += 3u;
        if (w > 16u || cap - at < (cnt * w + 7u) / 8u) return 0;
        for (i = 0; i < cnt; i++) {
            while (have < w) { acc |= (uint64_t)in[at++] << have; have += 8u; }
            len[g + i] = lo + (uint32_t)(acc & mask(w)); acc >>= w; have -= w;
        }
        if (acc) return 0;   /* unused high bits of the last byte must be 0 */
    }
    return at;
}

/* ---- candidate encoder (copy of wex_nrx_encode with the switches) -------- */
typedef struct { uint64_t scale_bytes, weight_bytes, len_bytes, used; uint32_t streams, short_flush; } Sizes;

static int v_encode(const Cfg *c, const WexNr2Model *model, const WexNr2Layout *layout,
                    unsigned char *const scales[3], unsigned char *const weights[3],
                    uint32_t flags, uint32_t owner, unsigned char *out, size_t cap, Sizes *sz) {
    int base = c == &cfgs[0];
    uint32_t m, streams = 0, stream = 0, ident[4], digest = 0, header = base ? WEX_NR2_HEADER_BYTES : HDR4;
    unsigned char anchor[3], tmp[2u * WEX_NR2_MAX_CHUNK + 8u];
    uint32_t *len;
    unsigned char *payload;
    size_t cursor = 0;
    memset(sz, 0, sizeof(*sz));
    for (m = 0; m < 3u; m++) {
        anchor[m] = wex_nr3_anchor(scales[m], layout->scale_count[m]);
        streams += wex_nr2_streams(layout->scale_count[m], c->schunk) + wex_nr2_streams(layout->symbol_count[m], c->wchunk);
    }
    ident[0] = model->id; ident[1] = owner;
    ident[3] = (uint32_t)anchor[0] | (uint32_t)anchor[1] << 8 | (uint32_t)anchor[2] << 16 | flags << 24;
    len = (uint32_t *)malloc(streams * sizeof(*len));
    payload = (unsigned char *)malloc(cap);
    if (!len || !payload) return -1;
    for (m = 0; m < 6u; m++) {
        uint32_t matrix = m % 3u, scale_pass = m < 3u, first;
        uint32_t count_all = scale_pass ? layout->scale_count[matrix] : layout->symbol_count[matrix];
        uint32_t chunk = scale_pass ? c->schunk : c->wchunk, sym_bits = scale_pass ? 8u : 4u;
        uint32_t tail = scale_pass ? c->tail_s : c->tail_w, tail_bits = tail * sym_bits;
        if (tail_bits > c->k) return -1;
        for (first = 0; first < count_all; first += chunk, stream++) {
            uint32_t count = count_all - first < chunk ? count_all - first : chunk, j, state, h, kw = tail < count ? tail : count;
            uint32_t coded = count - kw, tailv = 0, length;
            unsigned char *end = tmp + sizeof(tmp), *p = end;
            static _Thread_local uint16_t contexts[WEX_NR2_MAX_CHUNK];
            if (!scale_pass) {
                uint32_t srel = 0, previous = 8u, maximum = 8u;
                for (j = 0; j < count; j++) {
                    uint32_t i = first + j, x = (weights[matrix][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                    if (!(i % 32u)) { srel = wex_nr2_srel(scales[matrix][i / 32u], anchor[matrix]); previous = 8u; maximum = 8u; }
                    contexts[j] = (uint16_t)wex_nr2_context(srel, previous, maximum);
                    previous = x & 7u;
                    if (maximum == 8u || previous > maximum) maximum = previous;
                }
            }
            ident[2] = stream;
            h = scale_pass ? wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), scales[matrix] + first, count, 16u)
                           : wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), weights[matrix] + first / 2u, count / 2u, 16u);
            digest ^= h;
            for (j = 0; j < kw; j++) {
                uint32_t i = first + coded + j;
                uint32_t x = scale_pass ? scales[matrix][i] : (weights[matrix][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                tailv |= x << (j * sym_bits);
            }
            state = (1u << c->k) + (tailv | (h & mask(c->k - tail_bits)) << tail_bits);
            for (j = coded; j-- > 0;) {
                uint32_t i = first + j;
                if (scale_pass) {
                    if (wex_nr2_put(&state, &p, tmp, wex_nr2_scale_cdf(model, matrix), scales[matrix][i])) return -1;
                } else {
                    uint32_t x = (weights[matrix][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                    if (wex_nr2_put(&state, &p, tmp, wex_nr2_weight_cdf(model, matrix, contexts[j]), x)) return -1;
                }
            }
            if (c->sd) {   /* minimal big-endian state: the decoder reads bytes while its state is < L */
                uint32_t nb = state >> 24 ? 4u : state >> 16 ? 3u : state >> 8 ? 2u : 1u, b;
                for (b = 0; b < nb; b++) { *--p = (unsigned char)state; state >>= 8; }
                sz->short_flush += nb < 4u;
            } else if (wex_nr2_flush(state, &p, tmp)) return -1;
            length = (uint32_t)(end - p);
            if (length > 65535u || cap - cursor < length) return -1;
            memcpy(payload + cursor, p, length);
            cursor += length; len[stream] = length;
            if (scale_pass) sz->scale_bytes += length; else sz->weight_bytes += length;
        }
    }
    if (c->lenc) sz->len_bytes = lenc_pack(len, streams, out + header);
    else {
        for (stream = 0; stream < streams; stream++) {
            out[header + stream * 2u] = (unsigned char)len[stream];
            out[header + stream * 2u + 1u] = (unsigned char)(len[stream] >> 8);
        }
        sz->len_bytes = 2u * (size_t)streams;
    }
    memcpy(out + header + sz->len_bytes, payload, cursor);
    put32(out, base ? WEX_NR3_MAGIC : MAGIC4); put32(out + 4, model->id);
    put32(out + 8, (uint32_t)sz->scale_bytes); put32(out + 12, (uint32_t)sz->weight_bytes);
    put32(out + 16, c->wchunk); put32(out + 20, c->schunk); put32(out + 24, ident[3]); put32(out + 28, owner);
    if (!base) { put32(out + 32, digest); put32(out + 36, (uint32_t)sz->len_bytes); }
    sz->streams = streams; sz->used = header + sz->len_bytes + cursor;
    free(len); free(payload);
    return 0;
}

/* ---- candidate decoder (copy of wex_nr2_decode with the switches) -------- */
static int v_decode(const Cfg *c, const WexNr2Model *model, const WexNr2Layout *layout,
                    const unsigned char *rec, size_t used, unsigned char *const scales[3], unsigned char *const weights[3]) {
    uint32_t header = HDR4, m, streams = 0, stream = 0, ident[4], digest = 0, cz, flags;
    uint32_t *len;
    const unsigned char *payload;
    size_t lsec, total = 0;
    if (used < header || wex_nr_u32(rec) != MAGIC4 || wex_nr_u32(rec + 4) != model->id ||
        wex_nr_u32(rec + 16) != c->wchunk || wex_nr_u32(rec + 20) != c->schunk) return -1;
    for (m = 0; m < 3u; m++)
        streams += wex_nr2_streams(layout->scale_count[m], c->schunk) + wex_nr2_streams(layout->symbol_count[m], c->wchunk);
    len = (uint32_t *)malloc(streams * sizeof(*len));
    if (!len) return -1;
    lsec = wex_nr_u32(rec + 36);
    if (lsec > used - header) return -1;
    if (c->lenc) { if (lenc_unpack(rec + header, lsec, streams, len) != lsec) return -1; }
    else {
        if (lsec != 2u * (size_t)streams) return -1;
        for (stream = 0; stream < streams; stream++) len[stream] = wex_nr_u16(rec + header + stream * 2u);
    }
    for (stream = 0; stream < streams; stream++) total += len[stream];
    if (header + lsec + total != used || total != (size_t)wex_nr_u32(rec + 8) + wex_nr_u32(rec + 12)) return -1;
    ident[0] = model->id; ident[1] = wex_nr_u32(rec + 28); ident[3] = wex_nr_u32(rec + 24);
    flags = rec[27]; cz = flags & WEX_NR3_FLAG_CZ;
    payload = rec + header + lsec;
    stream = 0;
    for (m = 0; m < 6u; m++) {
        uint32_t matrix = m % 3u, scale_pass = m < 3u, first;
        uint32_t count_all = scale_pass ? layout->scale_count[matrix] : layout->symbol_count[matrix];
        uint32_t chunk = scale_pass ? c->schunk : c->wchunk, sym_bits = scale_pass ? 8u : 4u;
        uint32_t tail = scale_pass ? c->tail_s : c->tail_w, tail_bits = tail * sym_bits;
        for (first = 0; first < count_all; first += chunk, stream++) {
            uint32_t count = count_all - first < chunk ? count_all - first : chunk, kw = tail < count ? tail : count;
            uint32_t coded = count - kw, state = 0, j, symbol, srel = 0, previous = 8u, maximum = 8u, v, h;
            const unsigned char *ptr = payload, *end = payload + len[stream];
            if (c->sd) { while (state < WEX_NR_L && ptr < end) state = state << 8 | *ptr++; }
            else {
                if (len[stream] < 4u) return -1;
                state = (uint32_t)ptr[0] << 24 | (uint32_t)ptr[1] << 16 | (uint32_t)ptr[2] << 8 | ptr[3]; ptr += 4;
            }
            if (c->k == 23u && (state < WEX_NR_L || state >= (WEX_NR_L << 8))) return -1;
            for (j = 0; j < count; j++) {
                uint32_t i = first + j;
                if (!scale_pass && !(i % 32u)) { srel = wex_nr2_srel(scales[matrix][i / 32u], rec[24u + matrix]); previous = 8u; maximum = 8u; }
                if (j < coded) {
                    const unsigned char *cdf = scale_pass ? wex_nr2_scale_cdf(model, matrix)
                                                          : wex_nr2_weight_cdf(model, matrix, wex_nr2_context(srel, previous, maximum));
                    uint32_t slot = state & (WEX_NR2_TOTAL - 1u), lo = 0, hi = scale_pass ? 256u : 16u, start;
                    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid; }
                    start = wex_nr_u16(cdf + lo * 2u);
                    state = (wex_nr_u16(cdf + lo * 2u + 2u) - start) * (state >> WEX_NR2_PRECISION) + slot - start;
                    while (state < WEX_NR_L) {
                        if (ptr == end) { if (c->k == 23u) return -1; break; }   /* small initial state: stop at the end */
                        state = state << 8 | *ptr++;
                    }
                    symbol = lo;
                } else {   /* raw tail symbol from the final state */
                    if (j == coded && (ptr != end || state >> c->k != 1u)) return -1;
                    symbol = ((state - (1u << c->k)) >> ((j - coded) * sym_bits)) & mask(sym_bits);
                }
                if (scale_pass) scales[matrix][i] = (unsigned char)symbol;
                else {
                    if (cz && symbol == 8u) return -1;
                    if (!(i & 1u)) weights[matrix][i / 2u] = (unsigned char)symbol;
                    else weights[matrix][i / 2u] |= (unsigned char)(symbol << 4);
                    previous = symbol & 7u;
                    if (maximum == 8u || previous > maximum) maximum = previous;
                }
            }
            if (ptr != end || state >> c->k != 1u) return -1;
            v = state - (1u << c->k);
            if (kw < tail && (v & mask(tail_bits)) >> (kw * sym_bits)) return -1;   /* unused tail slots are 0 */
            ident[2] = stream;
            h = scale_pass ? wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), scales[matrix] + first, count, 16u)
                           : wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), weights[matrix] + first / 2u, count / 2u, 16u);
            if (v >> tail_bits != (h & mask(c->k - tail_bits))) return -1;
            digest ^= h;
            payload += len[stream];
        }
    }
    free(len);
    if (digest != wex_nr_u32(rec + 32)) return -1;
    return wex_nr3_ceil_inverse(layout, flags & WEX_NR3_FLAG_CEIL_MASK, scales, weights);
}

/* Model code length (bits) of the coded slab, and the share of magnitude-0 nibbles. */
static void ideal_bits(const WexNr2Model *model, const WexNr2Layout *layout, unsigned char *const scales[3],
                       unsigned char *const weights[3], const unsigned char anchor[3], double *wbits, double *sbits,
                       uint64_t *zeros) {
    uint32_t m, i;
    *wbits = *sbits = 0; *zeros = 0;
    for (m = 0; m < 3u; m++) {
        const unsigned char *cdf = wex_nr2_scale_cdf(model, m);
        uint32_t srel = 0, previous = 8u, maximum = 8u;
        for (i = 0; i < layout->scale_count[m]; i++) {
            uint32_t s = scales[m][i];
            *sbits += 14.0 - log2((double)(wex_nr_u16(cdf + s * 2u + 2u) - wex_nr_u16(cdf + s * 2u)));
        }
        for (i = 0; i < layout->symbol_count[m]; i++) {
            uint32_t x = (weights[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
            if (!(i % 32u)) { srel = wex_nr2_srel(scales[m][i / 32u], anchor[m]); previous = 8u; maximum = 8u; }
            cdf = wex_nr2_weight_cdf(model, m, wex_nr2_context(srel, previous, maximum));
            *wbits += 14.0 - log2((double)(wex_nr_u16(cdf + x * 2u + 2u) - wex_nr_u16(cdf + x * 2u)));
            *zeros += !(x & 7u);
            previous = x & 7u;
            if (maximum == 8u || previous > maximum) maximum = previous;
        }
    }
}

static unsigned char *slurp(const char *dir, const char *name, size_t *n) {
    char path[1024];
    FILE *f;
    unsigned char *p;
    long size;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
    p = (unsigned char *)malloc((size_t)size);
    if (!p || fread(p, 1, (size_t)size, f) != (size_t)size) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    fclose(f);
    *n = (size_t)size;
    return p;
}

typedef struct {
    int layer, expert, harness_ok, ok[NCFG], flip_rejected[NCFG];
    uint32_t total, used, flags;
    Sizes sz[NCFG];
    double wbits, sbits, ref_decode_s, offsets_u16_us, offsets_lenc_us;
    uint64_t zeros, symbols, scale_count;
} Result;

int main(int argc, char **argv) {
    size_t n, model_bytes, table_bytes;
    unsigned char *page, *blob, *table;
    WexHeader hdr;
    WexNr2Model model;
    int count = argc - 3, x, bad = 0;
    Result *res;
    FILE *out;
    if (argc < 4) { fprintf(stderr, "usage: wnr4_proto CACHE_DIR OUT.json L:E...\n"); return 1; }
    page = slurp(argv[1], "header.bin", &n);
    blob = slurp(argv[1], "model.bin", &model_bytes);
    table = slurp(argv[1], "entries.bin", &table_bytes);
    memcpy(&hdr, page, sizeof(hdr));
    if (wex_nr2_model_open(blob, model_bytes, &model) || !(hdr.flags & WEX_HEADER_CANONICAL_ZERO)) { fprintf(stderr, "bad container\n"); return 1; }
    res = (Result *)calloc((size_t)count, sizeof(*res));
#pragma omp parallel for schedule(dynamic, 1)
    for (x = 0; x < count; x++) {
        Result *r = &res[x];
        WexEntry e;
        WexNr2Layout layout;
        char name[64];
        size_t rec_bytes, used = 0, total = 0, cap;
        unsigned char *rec, *slab, *coded, *buf, *back, *scales[3], *weights[3], *bs[3], *bw[3];
        unsigned char anchor[3];
        uint64_t raw = 0, aligned = 0;
        uint32_t m, stats[16], owner;
        unsigned int c;
        int flags;
        double t0;
        sscanf(argv[3 + x], "%d:%d", &r->layer, &r->expert);
        memcpy(&e, table + ((size_t)r->layer * hdr.num_experts + r->expert) * sizeof(e), sizeof(e));
        snprintf(name, sizeof(name), "rec_%02d_%03d.bin", r->layer, r->expert);
        rec = slurp(argv[1], name, &rec_bytes);
        wex_entry_raw_sizes(&e, &raw, &aligned);
        wex_nr2_layout(&e, &layout);
        cap = (size_t)raw * 2u;
        slab = (unsigned char *)malloc((size_t)raw); coded = (unsigned char *)malloc((size_t)raw);
        back = (unsigned char *)malloc((size_t)raw); buf = (unsigned char *)malloc(cap);
        owner = (uint32_t)r->layer << 16 | (uint32_t)r->expert;
        r->total = e.total_bytes;
        /* 1. harness: reference decode, prepare, re-encode with the container's model, compare */
        t0 = omp_get_wtime();
        r->harness_ok = rec_bytes == e.total_bytes && !wex_nr3_decode(blob, &e, rec, (uint32_t)r->layer, (uint32_t)r->expert, slab);
        r->ref_decode_s = omp_get_wtime() - t0;
        flags = wex_nr3_prepare(&e, slab, 1, coded, stats);
        r->flags = (uint32_t)flags;
        r->harness_ok = r->harness_ok && flags == rec[27] && !stats[13] && !stats[14] &&   /* slab is already Z(source) */
            !wex_nr3_encode_record(blob, &e, coded, (uint32_t)flags, owner, wex_nr_u32(rec + 16), wex_nr_u32(rec + 20),
                                   buf, cap, &used, &total) &&
            total == rec_bytes && !memcmp(buf, rec, rec_bytes);
        r->used = (uint32_t)used;
        for (m = 0; m < 3u; m++) {
            scales[m] = coded + e.scale_offsets[m]; weights[m] = coded + e.weight_offsets[m];
            bs[m] = back + e.scale_offsets[m]; bw[m] = back + e.weight_offsets[m];
            anchor[m] = rec[24u + m];
            r->symbols += layout.symbol_count[m]; r->scale_count += layout.scale_count[m];
        }
        ideal_bits(&model, &layout, scales, weights, anchor, &r->wbits, &r->sbits, &r->zeros);
        /* 2. candidates: encode, decode, compare with the reference slab */
        for (c = 0; c < NCFG; c++) {
            if (v_encode(&cfgs[c], &model, &layout, scales, weights, (uint32_t)flags, owner, buf, cap, &r->sz[c])) continue;
            if (!c) { r->ok[c] = r->sz[c].used == used && !memcmp(buf, rec, used); continue; }   /* my copy == production bytes */
            memset(back, 0xA5, (size_t)raw);
            r->ok[c] = !v_decode(&cfgs[c], &model, &layout, buf, (size_t)r->sz[c].used, bs, bw) && !memcmp(back, slab, (size_t)raw);
            if (cfgs[c].lenc && cfgs[c].sd && cfgs[c].wchunk == 2048u) {   /* smoke test, not an audit: one flipped payload bit must be rejected */
                size_t at = HDR4 + (size_t)(r->sz[c].len_bytes + r->sz[c].scale_bytes + r->sz[c].weight_bytes / 2u);
                buf[at] ^= 0x10u;
                r->flip_rejected[c] = v_decode(&cfgs[c], &model, &layout, buf, (size_t)r->sz[c].used, bs, bw) != 0;   /* leaks the length array of the failed decode */
                buf[at] ^= 0x10u;
            }
            if (c == 1u) {   /* host cost of the offsets table: u16 prefix sum vs coded section */
                uint32_t streams = r->sz[c].streams, *len = (uint32_t *)malloc((streams + 1u) * sizeof(*len)), rep, s;
                volatile uint32_t sink = 0;
                t0 = omp_get_wtime();
                for (rep = 0; rep < 2000u; rep++) {
                    uint32_t at = 0;
                    for (s = 0; s < streams; s++) { len[s] = at; at += wex_nr_u16(rec + 32u + s * 2u); }
                    sink += at;
                }
                r->offsets_u16_us = (omp_get_wtime() - t0) / 2000.0 * 1e6;
                t0 = omp_get_wtime();
                for (rep = 0; rep < 2000u; rep++) {
                    uint32_t at = 0;
                    lenc_unpack(buf + HDR4, (size_t)r->sz[c].len_bytes, streams, len);
                    for (s = 0; s < streams; s++) { uint32_t l = len[s]; len[s] = at; at += l; }
                    sink += at;
                }
                r->offsets_lenc_us = (omp_get_wtime() - t0) / 2000.0 * 1e6;
                (void)sink;
                free(len);
            }
        }
        free(rec); free(slab); free(coded); free(back); free(buf);
#pragma omp critical
        fprintf(stderr, "L%02d E%03d harness %s\n", r->layer, r->expert, r->harness_ok ? "ok" : "FAIL");
    }
    out = fopen(argv[2], "w");
    fprintf(out, "{\"raw_bytes\": 18800640, \"configs\": [");
    for (x = 0; x < (int)NCFG; x++)
        fprintf(out, "%s{\"name\": \"%s\", \"wchunk\": %u, \"schunk\": %u, \"lenc\": %u, \"sd\": %u, \"k\": %u, \"tail_w\": %u, \"tail_s\": %u}",
                x ? ", " : "", cfgs[x].name, cfgs[x].wchunk, cfgs[x].schunk, cfgs[x].lenc, cfgs[x].sd, cfgs[x].k, cfgs[x].tail_w, cfgs[x].tail_s);
    fprintf(out, "],\n \"experts\": [\n");
    for (x = 0; x < count; x++) {
        Result *r = &res[x];
        unsigned int c;
        fprintf(out, "  {\"layer\": %d, \"expert\": %d, \"harness_ok\": %d, \"total_bytes\": %u, \"used_bytes\": %u, \"flags\": %u, "
                "\"ideal_weight_bits\": %.1f, \"ideal_scale_bits\": %.1f, \"symbols\": %llu, \"scales\": %llu, \"zero_magnitudes\": %llu, "
                "\"ref_decode_s\": %.3f, \"offsets_u16_us\": %.2f, \"offsets_lenc_us\": %.2f,\n   \"cfg\": [",
                r->layer, r->expert, r->harness_ok, r->total, r->used, r->flags, r->wbits, r->sbits,
                (unsigned long long)r->symbols, (unsigned long long)r->scale_count, (unsigned long long)r->zeros,
                r->ref_decode_s, r->offsets_u16_us, r->offsets_lenc_us);
        for (c = 0; c < NCFG; c++) {
            fprintf(out, "%s[%d, %llu, %llu, %llu, %llu, %u, %u, %d]", c ? ", " : "", r->ok[c], (unsigned long long)r->sz[c].used,
                    (unsigned long long)r->sz[c].len_bytes, (unsigned long long)r->sz[c].scale_bytes,
                    (unsigned long long)r->sz[c].weight_bytes, r->sz[c].streams, r->sz[c].short_flush, r->flip_rejected[c]);
            bad += !r->ok[c];
        }
        fprintf(out, "]}%s\n", x + 1 < count ? "," : "");
        bad += !r->harness_ok;
    }
    fprintf(out, " ],\n \"cfg_columns\": [\"round_trip_ok\", \"used_bytes\", \"length_section_bytes\", \"scale_payload_bytes\", \"weight_payload_bytes\", \"streams\", \"streams_with_3_byte_state\", \"bit_flip_rejected\"]}\n");
    fclose(out);
    fprintf(stderr, "%d experts, %d failures\n", count, bad);
    return bad != 0;
}
