/* erow.c -- lossless per-row static-context rANS codec for DeepSeek V4.1 Engram rows.
 *
 * Row = 256 B FP8 E4M3 weights + 8 B E8M0 scales (one per 32 weights) = 264 B.
 * One rANS stream per row (32-bit state, byte renorm, L = 2^23, PROB_BITS = 12), every op is a
 * 16-symbol alphabet, so decode = one 16-wide compare against the context's cumulative table.
 *   scales  : 8 ops, symbol = class(v) = v-112 for 112<=v<127 else 15 (escape, + raw 8-bit op); ctx = class of left scale (16 = none)
 *   weights : per element: exp (4 bits) with ctx = scale class of its block x previous exp in block (16 = first)
 *             then if exp >= 14: mantissa coded with ctx = exp (8-symbol), sign raw 1 bit
 *                  else        : sign|mantissa raw 4 bits
 *   The signs of the last NEMB=30 elements are not coded: they are put in the encoder's initial state
 *   x0 = 2^30 + bits (the decoder ends exactly there), which removes the ~3 B per-row start-state cost.
 * Contexts: 0..271 exp (sc*17+pe), 272/273 mantissa (e=14/15), 274..290 scales (left class).
 * Model = cum[291][17] u16 (cum[c][16] = 4096).
 * Group mode (format c): N rows in one stream (decoder state carried over), only the first row embeds signs.
 * Build: cl /O2 /LD erow.c
 */
#include <stdint.h>
#include <emmintrin.h>
/* CRT-free build (no Windows SDK on this PC): own memcpy/memcmp/memset */
#pragma function(memcpy, memset, memcmp)
void *memcpy(void *d, const void *s, size_t n) { uint8_t *a = d; const uint8_t *b = s; while (n--) *a++ = *b++; return d; }
void *memset(void *d, int v, size_t n) { uint8_t *a = d; while (n--) *a++ = (uint8_t)v; return d; }
int memcmp(const void *x, const void *y, size_t n) { const uint8_t *a = x, *b = y; for (; n; n--, a++, b++) if (*a != *b) return *a < *b ? -1 : 1; return 0; }
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define PB 12
#define PMASK ((1u << PB) - 1)
#define RL (1u << 23)
#define NCTX 291
#define CX_M 272
#define CX_S 274
#define NEMB 30

static inline int sclass(int v) { return (v >= 112 && v < 127) ? v - 112 : 15; }

/* ---------------- counting (training) ---------------- */
EXPORT void erow_count(const uint8_t *W, const uint8_t *S, int64_t nrows, int64_t *cnt /* [NCTX][16] */) {
    for (int64_t r = 0; r < nrows; r++) {
        const uint8_t *w = W + r * 256, *s = S + r * 8;
        int left = 16;
        for (int b = 0; b < 8; b++) { int c = sclass(s[b]); cnt[(CX_S + left) * 16 + c]++; left = c; }
        for (int b = 0; b < 8; b++) {
            int sc = sclass(s[b]), pe = 16;
            for (int j = 0; j < 32; j++) {
                int v = w[b * 32 + j], e = (v >> 3) & 15, m = v & 7;
                cnt[(sc * 17 + pe) * 16 + e]++;
                if (e >= 14) cnt[(CX_M + e - 14) * 16 + m]++;
                pe = e;
            }
        }
    }
}

/* ---------------- encoder ---------------- */
typedef struct { uint32_t start, freq; } Op;

static inline void enc_put(uint32_t *x, uint8_t **p, uint32_t start, uint32_t freq) {
    uint64_t xmax = (uint64_t)((RL >> PB) << 8) * freq;
    while (*x >= xmax) { *--(*p) = (uint8_t)(*x & 0xFF); *x >>= 8; }
    *x = ((*x / freq) << PB) + (*x % freq) + start;
}

/* builds the op list of one row in DECODE order; returns n ops; *emb gets embedded sign bits */
static int row_ops(const uint16_t *cum, const uint8_t *w, const uint8_t *s, Op *ops, uint32_t *emb, int embed) {
    int n = 0, left = 16;
    for (int b = 0; b < 8; b++) {
        int c = sclass(s[b]); const uint16_t *t = cum + (CX_S + left) * 17;
        ops[n].start = t[c]; ops[n].freq = t[c + 1] - t[c]; n++;
        if (c == 15) { ops[n].start = (uint32_t)s[b] << (PB - 8); ops[n].freq = 1u << (PB - 8); n++; }
        left = c;
    }
    uint32_t eb = 0;
    for (int b = 0; b < 8; b++) {
        int sc = sclass(s[b]), pe = 16;
        for (int j = 0; j < 32; j++) {
            int i = b * 32 + j, v = w[i], sg = v >> 7, e = (v >> 3) & 15, m = v & 7;
            const uint16_t *t = cum + (sc * 17 + pe) * 17;
            ops[n].start = t[e]; ops[n].freq = t[e + 1] - t[e]; n++;
            int emb_i = embed && i >= 256 - NEMB;
            if (emb_i) eb |= (uint32_t)sg << (i - (256 - NEMB));
            if (e >= 14) {
                const uint16_t *tm = cum + (CX_M + e - 14) * 17;
                ops[n].start = tm[m]; ops[n].freq = tm[m + 1] - tm[m]; n++;
                if (!emb_i) { ops[n].start = (uint32_t)sg << (PB - 1); ops[n].freq = 1u << (PB - 1); n++; }
            } else if (!emb_i) {
                ops[n].start = (uint32_t)(sg << 3 | m) << (PB - 4); ops[n].freq = 1u << (PB - 4); n++;
            } else {
                ops[n].start = (uint32_t)m << (PB - 3); ops[n].freq = 1u << (PB - 3); n++;
            }
            pe = e;
        }
    }
    *emb = eb;
    return n;
}

/* Encode nrows rows as groups of g rows (g=1: per-row streams).  out: concatenated streams, lens[ngroups].
 * Returns total bytes, or -1 if a group's stream overflows cap bytes. */
EXPORT int64_t erow_encode(const uint16_t *cum, const uint8_t *W, const uint8_t *S, int64_t nrows, int g,
                           uint8_t *out, int32_t *lens, int64_t cap) {
    static Op ops[64 * 1200];
    static uint8_t tmp[64 * 2048];
    int64_t pos = 0;
    for (int64_t r0 = 0, gi = 0; r0 < nrows; r0 += g, gi++) {
        int gn = (int)((nrows - r0) < g ? (nrows - r0) : g), n = 0;
        uint32_t emb = 0;
        for (int k = 0; k < gn; k++) {
            uint32_t e2;
            n += row_ops(cum, W + (r0 + k) * 256, S + (r0 + k) * 8, ops + n, &e2, k == 0);
            if (k == 0) emb = e2;
        }
        uint32_t x = (1u << NEMB) + emb;   /* x0 in [2^30, 2^31) subset of [L, 256L) */
        uint8_t *p = tmp + sizeof(tmp);
        for (int i = n - 1; i >= 0; i--) enc_put(&x, &p, ops[i].start, ops[i].freq);
        *--p = (uint8_t)(x); *--p = (uint8_t)(x >> 8); *--p = (uint8_t)(x >> 16); *--p = (uint8_t)(x >> 24);
        int len = (int)(tmp + sizeof(tmp) - p);
        if (pos + len > cap) return -1;
        memcpy(out + pos, p, len); pos += len; lens[gi] = len;
    }
    return pos;
}


/* ---------------- decoder ---------------- */
#include <nmmintrin.h>
static inline int find16(const uint16_t *t, uint32_t slot) {
    /* symbol s with t[s] <= slot < t[s+1]  ==  16 - #{k in 1..16 : t[k] > slot}  (t[16] = 4096 > slot) */
    __m128i v = _mm_set1_epi16((short)slot);
    __m128i a = _mm_loadu_si128((const __m128i *)(t + 1));
    __m128i b = _mm_loadu_si128((const __m128i *)(t + 9));
    int ma = _mm_movemask_epi8(_mm_cmpgt_epi16(a, v)), mb = _mm_movemask_epi8(_mm_cmpgt_epi16(b, v));
    return 16 - (int)((_mm_popcnt_u32((unsigned)ma) + _mm_popcnt_u32((unsigned)mb)) >> 1);
}
#define RENORM while (x < RL) x = (x << 8) | *p++
#define DSYM(t, s) do { uint32_t sl_ = x & PMASK; s = find16((t), sl_); uint32_t st_ = (t)[s]; \
    x = ((t)[s + 1] - st_) * (x >> PB) + sl_ - st_; RENORM; } while (0)
#define DRAW(k, v) do { uint32_t sl_ = x & PMASK; v = sl_ >> (PB - (k)); \
    x = (x >> PB) << (PB - (k)) | (sl_ & ((1u << (PB - (k))) - 1)); RENORM; } while (0)

/* decode gn rows from one stream at in; returns bytes consumed (or -1 on final-state mismatch) */
static int64_t dec_group(const uint16_t *cum, const uint8_t *in, int gn, uint8_t *W, uint8_t *S) {
    const uint8_t *p = in;
    uint32_t x = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; p += 4;
    for (int k = 0; k < gn; k++) {
        uint8_t *w = W + k * 256, *s = S + k * 8;
        int left = 16, sc[8], embed = (k == 0);
        for (int b = 0; b < 8; b++) {
            int c; DSYM(cum + (CX_S + left) * 17, c);
            if (c == 15) { uint32_t v; DRAW(8, v); s[b] = (uint8_t)v; } else s[b] = (uint8_t)(c + 112);
            sc[b] = c; left = c;
        }
        for (int b = 0; b < 8; b++) {
            const uint16_t *te = cum + sc[b] * 17 * 17;
            int pe = 16;
            for (int j = 0; j < 32; j++) {
                int i = b * 32 + j, e; uint32_t v, sg = 0;
                DSYM(te + pe * 17, e);
                int emb_i = embed && i >= 256 - NEMB;
                if (e >= 14) {
                    int m; DSYM(cum + (CX_M + e - 14) * 17, m);
                    if (!emb_i) { DRAW(1, sg); }
                    v = (uint32_t)m;
                } else if (!emb_i) { uint32_t r; DRAW(4, r); sg = r >> 3; v = r & 7; }
                else { DRAW(3, v); }
                w[i] = (uint8_t)(sg << 7 | (uint32_t)e << 3 | v);
                pe = e;
            }
        }
    }
    if ((x >> NEMB) != 1) return -1;
    for (int i = 256 - NEMB; i < 256; i++) W[i] |= (uint8_t)(((x >> (i - (256 - NEMB))) & 1) << 7);
    return (int64_t)(p - in);
}

/* verify: decode every group (sizes g) of the concatenated streams and compare; returns #mismatching groups */
EXPORT int64_t erow_verify(const uint16_t *cum, const uint8_t *buf, const int32_t *lens, int64_t nrows, int g,
                           const uint8_t *W, const uint8_t *S) {
    static uint8_t w[64 * 256], s[64 * 8];
    int64_t pos = 0, bad = 0;
    for (int64_t r0 = 0, gi = 0; r0 < nrows; r0 += g, gi++) {
        int gn = (int)((nrows - r0) < g ? (nrows - r0) : g);
        int64_t used = dec_group(cum, buf + pos, gn, w, s);
        if (used != lens[gi] || memcmp(w, W + r0 * 256, gn * 256) || memcmp(s, S + r0 * 8, gn * 8)) bad++;
        pos += lens[gi];
    }
    return bad;
}

/* bench: decode n listed rows (per-row streams at offs[]); g>1: decode the whole group gidx then copy row.
 * mode 0 = rANS decode, 1 = memcpy of a 264-B raw row (baseline). Returns checksum. */
EXPORT uint64_t erow_bench(const uint16_t *cum, const uint8_t *buf, const int64_t *offs, const int64_t *idx, int64_t n,
                           int g, int mode, uint8_t *Wout, uint8_t *Sout) {
    static uint8_t w[64 * 256], s[64 * 8];
    uint64_t ck = 0;
    for (int64_t q = 0; q < n; q++) {
        int64_t r = idx[q];
        uint8_t *wo = Wout + (q % 48) * 256, *so = Sout + (q % 48) * 8;
        if (mode == 1) { memcpy(wo, buf + r * 264, 256); memcpy(so, buf + r * 264 + 256, 8); }
        else if (g == 1) { dec_group(cum, buf + offs[r], 1, wo, so); }
        else { int64_t gi = r / g; dec_group(cum, buf + offs[gi], g, w, s); memcpy(wo, w + (r % g) * 256, 256); memcpy(so, s + (r % g) * 8, 8); }
        ck += wo[q & 255] + so[q & 7];
    }
    return ck;
}



