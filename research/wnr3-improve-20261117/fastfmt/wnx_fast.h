/* Fast SIMD decoders for WNX records (wnx.h). Same bytes and same verdict as wnx_stream_ref / wnx_decode_ref.
 *
 * Kernels (compile time): AVX2 (-march=haswell: AVX2 + BMI2 + LZCNT + POPCNT) and, with -mavx512f -mavx512bw
 * -mavx512vl, AVX-512 weight kernels (16 lanes per vector: 2 vectors per 32-lane step). Scales use the AVX2
 * gather kernel in both builds. G = 1 or 2 streams are decoded in one loop to overlap their dependency chains.
 *
 * Memory safety: a kernel only runs on a stream whose end is >= WNX_SLACK bytes before the end of the payload
 * buffer; each step reads at most 64 (AVX2: 4 x 16) or 96 bytes past the word pointer and one u32 below the
 * sign pointer, and the kernel stops when the word pointer passes the stream end or the sign pointer reaches
 * the stream start. Other streams take wnx_stream_ref.
 * Equivalence with the reference: both read the same words (in lane order from begin + 128) and the same sign
 * words (end - 4, end - 8, ...); the kernel may load one sign word more than it uses (prefetch), so it checks
 * ptr == end - 4 * ceil(S / 32) for the S sign bits it used, and that the unused bits of the last used word
 * are 0; since ptr only grows and the sign pointer only falls, that equality implies the reference's per-read
 * checks all hold. */
#ifndef WNX_FAST_H
#define WNX_FAST_H
#include "wnx.h"
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && !defined(WNX_NO512)
#define WNXF_512 1
#else
#define WNXF_512 0
#endif
#define WNXF_INLINE static inline __attribute__((always_inline))

typedef struct {
    /* AVX2 weight tables per (m, srel) */
    __m256i thr[3][7][14];        /* 16-bit (C[i+1] - 1), i = 0..nsym-2 */
    __m256i cv[3][7][2], fv[3][7][2];   /* u32 C[s], f[s], s = 0..7 and 8..15 */
#if WNXF_512
    __m512i thr5[3][7][14];       /* 16-bit C[i+1] */
    __m512i cv5[3][7], fv5[3][7]; /* u32 C[s], f[s], s = 0..15 */
#endif
    __m128i sl16[256];            /* AVX2: bit i of the index -> 16-bit lane i = 8 */
    __m256i rlut[256];            /* AVX2 renorm: pshufb control, needing lane i <- word rank(i) of the window */
    __m256i rsh[256];             /* AVX2 renorm: per-lane shift, 16 where the lane needs a word */
    uint32_t sdir[3][WNX_TS];     /* scales: (f - 1) | (slot - start) << 16 */
    unsigned char ssym[3][WNX_TS + 4];
    uint32_t nsym, joint, id;
} WnxfTables;

static inline void wnxf_tables_init(WnxfTables *X, const WnxModel *M) {
    uint32_t m, r, s, i, k;
    memset(X, 0, sizeof(*X));
    X->nsym = M->nsym; X->joint = (uint32_t)M->joint; X->id = M->id;
    for (m = 0; m < 3u; m++) {
        for (r = 0; r < 7u; r++) {
            const uint16_t *c = M->wc[m][r];
            uint32_t cv[16] = {0}, fv[16] = {0};
            for (i = 0; i + 1u < M->nsym; i++) {
                X->thr[m][r][i] = _mm256_set1_epi16((short)(c[i + 1u] - 1u));
#if WNXF_512
                X->thr5[m][r][i] = _mm512_set1_epi16((short)c[i + 1u]);
#endif
            }
            for (s = 0; s < M->nsym; s++) { cv[s] = c[s]; fv[s] = (uint32_t)(c[s + 1u] - c[s]); }
            X->cv[m][r][0] = _mm256_loadu_si256((const __m256i *)(const void *)cv);
            X->cv[m][r][1] = _mm256_loadu_si256((const __m256i *)(const void *)(cv + 8));
            X->fv[m][r][0] = _mm256_loadu_si256((const __m256i *)(const void *)fv);
            X->fv[m][r][1] = _mm256_loadu_si256((const __m256i *)(const void *)(fv + 8));
#if WNXF_512
            X->cv5[m][r] = _mm512_loadu_si512(cv); X->fv5[m][r] = _mm512_loadu_si512(fv);
#endif
        }
        for (s = 0; s < 256u; s++) {
            const uint32_t lo = M->sc[m][s], hi = M->sc[m][s + 1u];
            for (i = lo; i < hi; i++) { X->sdir[m][i] = (hi - lo - 1u) | (i - lo) << 16; X->ssym[m][i] = (unsigned char)s; }
        }
    }
    for (i = 0; i < 256u; i++) {
        uint32_t lanes[8];
        for (k = 0; k < 8u; k++) { lanes[k] = (i >> k & 1u) ? 8u : 0u; }
        {
            uint16_t l16[8];
            for (k = 0; k < 8u; k++) l16[k] = (uint16_t)lanes[k];
            X->sl16[i] = _mm_loadu_si128((const __m128i *)(const void *)l16);
        }
        {
            unsigned char c[32];
            uint32_t rank2 = 0;
            for (k = 0; k < 8u; k++) {
                const uint32_t at = 4u * k;
                if (i >> k & 1u) { c[at] = (unsigned char)(2u * rank2); c[at + 1u] = (unsigned char)(2u * rank2 + 1u); rank2++; }
                else { c[at] = c[at + 1u] = 0x80; }
                c[at + 2u] = c[at + 3u] = 0x80;
            }
            X->rlut[i] = _mm256_loadu_si256((const __m256i *)(const void *)c);
        }
        {
            uint32_t sh[8];
            for (k = 0; k < 8u; k++) sh[k] = (i >> k & 1u) ? 16u : 0u;
            X->rsh[i] = _mm256_loadu_si256((const __m256i *)(const void *)sh);
        }
    }
}

typedef struct {
    const WnxView *v;
    const WnxModel *M;
    const WnxfTables *X;
    unsigned char *const *so, *const *wo;
    unsigned char srel[3][256];
    uint32_t ceil;
} WnxfJob;

typedef struct { const unsigned char *begin, *end; uint32_t m, first, T, g; } WnxfS;

/* ---- shared pieces ---- */
WNXF_INLINE __m256i wnxf_mur8(__m256i h, __m256i k) {   /* wex_nr3_h32_block on 8 lanes */
    __m256i q = _mm256_mullo_epi32(k, _mm256_set1_epi32((int)0xcc9e2d51u));
    q = _mm256_mullo_epi32(_mm256_or_si256(_mm256_slli_epi32(q, 15), _mm256_srli_epi32(q, 17)), _mm256_set1_epi32(0x1b873593));
    q = _mm256_xor_si256(h, q);
    q = _mm256_or_si256(_mm256_slli_epi32(q, 13), _mm256_srli_epi32(q, 19));
    return _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(q, 2), q), _mm256_set1_epi32((int)0xe6546b64u));
}
static const uint32_t wnxf_golden[32] = {
#define G4(k) 0x9e3779b9u * (k + 1u), 0x9e3779b9u * (k + 2u), 0x9e3779b9u * (k + 3u), 0x9e3779b9u * (k + 4u)
    G4(0), G4(4), G4(8), G4(12), G4(16), G4(20), G4(24), G4(28)
#undef G4
};

/* AVX2 shared-pointer renormalisation of one 8-lane vector: the 8 words at ptr, broadcast to both halves, are
 * moved by one pshufb into the lanes that need a word (zero elsewhere) */
WNXF_INLINE __m256i wnxf_renorm8(const WnxfTables *X, __m256i x, const unsigned char **ptr) {
    const __m256i need = _mm256_cmpeq_epi32(_mm256_srli_epi32(x, 16), _mm256_setzero_si256());
    const uint32_t mk = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(need));
    const __m256i win = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)*ptr));
    const __m256i w = _mm256_shuffle_epi8(win, X->rlut[mk]);
    *ptr += 2u * (uint32_t)__builtin_popcount(mk);
    return _mm256_or_si256(_mm256_sllv_epi32(x, X->rsh[mk]), w);
}

/* end of a stream: 16 chain hashes (canonical chain order) -> expected states, compare */
static inline int wnxf_close(const WnxfJob *J, uint32_t g, const uint32_t *hc, const uint32_t *xl, uint32_t T) {
    const uint32_t s0 = wnx_ident_hash(J->v->model_id, J->v->owner, g, J->v->word16);
    uint32_t h[16], x0[32], k;
    for (k = 0; k < 16u; k++) h[k] = wex_nr3_h32_final(hc[k], T);
    wnx_expect(s0, h, x0);
    for (k = 0; k < 32u; k++) if (xl[k] != x0[k]) return -1;
    return 0;
}
/* AVX2 16-bit packed order: packus_epi32(lanes 0-7, lanes 8-15) holds the chains below in its 8 dwords */
static const uint32_t wnxf_chain_a[8] = {0, 1, 4, 5, 2, 3, 6, 7};
static inline void wnxf_chains_avx2(__m256i h01, __m256i h23, uint32_t hc[16]) {
    uint32_t t[16], i;
    _mm256_storeu_si256((__m256i *)(void *)t, h01); _mm256_storeu_si256((__m256i *)(void *)(t + 8), h23);
    for (i = 0; i < 8u; i++) { hc[wnxf_chain_a[i]] = t[i]; hc[8u + wnxf_chain_a[i]] = t[8u + i]; }
}
static inline __m256i wnxf_seed_avx2(uint32_t s0, int half) {   /* chain seeds in the packed order */
    uint32_t t[8], i;
    for (i = 0; i < 8u; i++) t[i] = wnx_lane_seed(s0, 8u * (uint32_t)half + wnxf_chain_a[i]);
    return _mm256_loadu_si256((const __m256i *)(const void *)t);
}

/* ---- scales (AVX2 gathers), 1 stream, cnt % 32 == 0 ---- */
static inline int wnxf_scales(const WnxfJob *J, const WnxfS *S) {
    const WnxfTables *X = J->X;
    const uint32_t m = S->m, T = S->T;
    const int *dir = (const int *)(const void *)X->sdir[m];
    const int *sym = (const int *)(const void *)X->ssym[m];
    const __m256i lowp = _mm256_set1_epi32((int)(WNX_TS - 1u)), low16 = _mm256_set1_epi32(0xffff), lowb = _mm256_set1_epi32(0xff);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const unsigned char *ptr = S->begin + 128;
    unsigned char *out = J->so[m] + S->first;
    const uint32_t s0 = wnx_ident_hash(J->v->model_id, J->v->owner, S->g, J->v->word16);
    __m256i x[4], sy[4], h01 = wnxf_seed_avx2(s0, 0), h23 = wnxf_seed_avx2(s0, 1), a01 = _mm256_setzero_si256(), a23 = a01;
    uint32_t v, t, xl[32], hc[16];
    for (v = 0; v < 4u; v++) {
        x[v] = _mm256_loadu_si256((const __m256i *)(const void *)(S->begin + 32u * v));
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_srli_epi32(x[v], 16), _mm256_setzero_si256()))) return -1;
    }
    for (t = 0; t < T; t++) {
        const __m128i cnt = _mm_cvtsi32_si128((int)(8u * (t & 1u)));
        __m256i p01, p23;
        if (ptr > S->end) return -1;
        for (v = 0; v < 4u; v++) {
            const __m256i slot = _mm256_and_si256(x[v], lowp);
            const __m256i e = _mm256_i32gather_epi32(dir, slot, 4);
            sy[v] = _mm256_and_si256(_mm256_i32gather_epi32(sym, slot, 1), lowb);
            x[v] = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_and_si256(e, low16), _mm256_set1_epi32(1)),
                                                       _mm256_srli_epi32(x[v], WNX_PS)), _mm256_srli_epi32(e, 16));
        }
        for (v = 0; v < 4u; v++) x[v] = wnxf_renorm8(X, x[v], &ptr);
        p01 = _mm256_packus_epi32(sy[0], sy[1]); p23 = _mm256_packus_epi32(sy[2], sy[3]);
        _mm256_storeu_si256((__m256i *)(void *)(out + 32u * t), _mm256_permutevar8x32_epi32(_mm256_packus_epi16(p01, p23), perm));
        a01 = _mm256_or_si256(a01, _mm256_sll_epi16(p01, cnt)); a23 = _mm256_or_si256(a23, _mm256_sll_epi16(p23, cnt));
        if (t & 1u) { h01 = wnxf_mur8(h01, a01); h23 = wnxf_mur8(h23, a23); a01 = a23 = _mm256_setzero_si256(); }
    }
    if (T & 1u) { h01 = wnxf_mur8(h01, a01); h23 = wnxf_mur8(h23, a23); }
    if (ptr != S->end) return -1;
    for (v = 0; v < 4u; v++) _mm256_storeu_si256((__m256i *)(void *)(xl + 8u * v), x[v]);
    wnxf_chains_avx2(h01, h23, hc);
    return wnxf_close(J, S->g, hc, xl, T);
}

/* ---- TAIL form: the pool of 480 sign bits in lanes 2..31's final states ---- */
typedef struct { uint64_t w[8]; uint32_t pp; } WnxfPool;
static inline int wnxf_pool_open(WnxfPool *P, const uint32_t *xl) {
    uint32_t k;
    memset(P, 0, sizeof(*P));
    for (k = 2; k < 32u; k++) {
        const uint32_t v = xl[k] - WNX_L, at = 16u * (k - 2u);
        if (xl[k] >= 2u * WNX_L) return -1;
        P->w[at / 64u] |= (uint64_t)v << (at % 64u);
    }
    return 0;
}
static inline uint32_t wnxf_pool_bits(const WnxfPool *P, uint32_t at, uint32_t n) {   /* n <= 32, at + n <= 480 */
    const uint32_t w = at / 64u, off = at % 64u;
    uint64_t v = P->w[w] >> off;
    if (off + n > 64u) v |= P->w[w + 1u] << (64u - off);
    return (uint32_t)(v & ((UINT64_C(1) << n) - 1u));
}
/* n sign bits: from the pool while it lasts, then from the region (on demand) */
WNXF_INLINE int wnxf_take(WnxfPool *P, uint64_t *bb, uint32_t *nb, uint32_t *used, const unsigned char **sp,
                          const unsigned char *begin, uint32_t n, uint32_t *out) {
    uint32_t a = WNX_POOL_BITS - P->pp, bits = 0, r;
    if (a > n) a = n;
    if (a) { bits = wnxf_pool_bits(P, P->pp, a); P->pp += a; }
    r = n - a;
    if (r) {
        if (*nb < r) {
            uint32_t w;
            if (*sp - begin < 4) return -1;
            *sp -= 4; memcpy(&w, *sp, 4); *bb |= (uint64_t)w << *nb; *nb += 32u;
        }
        bits |= (uint32_t)(*bb & ((UINT64_C(1) << r) - 1u)) << a;
        *bb >>= r; *nb -= r; *used += r;
    }
    *out = bits;
    return 0;
}
static inline int wnxf_pool_rest_zero(const WnxfPool *P) {
    uint32_t at = P->pp;
    while (at < WNX_POOL_BITS) {
        const uint32_t n = WNX_POOL_BITS - at < 32u ? WNX_POOL_BITS - at : 32u;
        if (wnxf_pool_bits(P, at, n)) return 0;
        at += n;
    }
    return 1;
}
/* canonical non-zero mask of 32 lanes from the AVX2 16-bit packed symbols */
WNXF_INLINE uint32_t wnxf_nz16(__m256i s01, __m256i s23) {
    const __m256i z = _mm256_setzero_si256();
    const uint32_t m01 = (uint32_t)_pext_u32((uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(s01, z)), 0x55555555u);
    const uint32_t m23 = (uint32_t)_pext_u32((uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(s23, z)), 0x55555555u);
    const uint32_t p = m01 | m23 << 16;   /* packed order per 16 lanes: 0-3, 8-11, 4-7, 12-15 */
    const uint32_t c = (p & 0xf00ff00fu) | ((p & 0x00f000f0u) << 4) | ((p & 0x0f000f00u) >> 4);
    return ~c;
}

/* ---- weights, AVX2, G streams of one matrix and the same T ---- */
WNXF_INLINE int wnxf_w2(const WnxfJob *J, const WnxfS *S, const uint32_t G, const int joint, const int tail) {
    const WnxfTables *X = J->X;
    const uint32_t m = S[0].m, T = S[0].T, ceil = J->ceil >> m & 1u, nthr = joint ? 14u : 7u;
    const uint32_t Tm = tail ? (T > WNX_TAIL_STEPS ? T - WNX_TAIL_STEPS : 0u) : T;
    const __m256i low12 = _mm256_set1_epi32((int)(WNX_T - 1u)), zero = _mm256_setzero_si256(), seven = _mm256_set1_epi32(7);
    const __m256i seven16 = _mm256_set1_epi16(7), five16 = _mm256_set1_epi16(5), two16 = _mm256_set1_epi16(2);
    const __m256i one16 = _mm256_set1_epi16(1), perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const unsigned char *ptr[2], *sp[2];
    unsigned char *out[2], *sc[2];
    uint64_t bb[2] = {0, 0};
    uint32_t nb[2] = {0, 0}, used[2] = {0, 0}, g, v, t, bad = 0;
    __m256i x[2][4], h[2][2], acc[2][2], dbuf[2][tail ? WNX_TAIL_STEPS : 1][2];
#define W2_FINISH(g, t, n01, n23) do { \
        switch ((t) & 3u) {   /* immediate shifts (a 2-uop shift by register is avoided) */ \
        case 0: acc[g][0] = _mm256_or_si256(acc[g][0], n01); acc[g][1] = _mm256_or_si256(acc[g][1], n23); break; \
        case 1: acc[g][0] = _mm256_or_si256(acc[g][0], _mm256_slli_epi16(n01, 4)); acc[g][1] = _mm256_or_si256(acc[g][1], _mm256_slli_epi16(n23, 4)); break; \
        case 2: acc[g][0] = _mm256_or_si256(acc[g][0], _mm256_slli_epi16(n01, 8)); acc[g][1] = _mm256_or_si256(acc[g][1], _mm256_slli_epi16(n23, 8)); break; \
        default: acc[g][0] = _mm256_or_si256(acc[g][0], _mm256_slli_epi16(n01, 12)); acc[g][1] = _mm256_or_si256(acc[g][1], _mm256_slli_epi16(n23, 12)); \
        } \
        if (((t) & 3u) == 3u) { h[g][0] = wnxf_mur8(h[g][0], acc[g][0]); h[g][1] = wnxf_mur8(h[g][1], acc[g][1]); acc[g][0] = acc[g][1] = zero; } \
        if (ceil) {   /* block of coded maximum 5: D on the magnitudes, scale - 1 (branch-free) */ \
            const __m256i a0_ = _mm256_and_si256(n01, seven16), a1_ = _mm256_and_si256(n23, seven16); \
            const __m256i mx_ = _mm256_max_epu16(a0_, a1_); \
            const uint32_t e5_ = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(mx_, five16)); \
            const uint32_t g5_ = (uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi16(mx_, five16)); \
            const uint32_t is5_ = (e5_ != 0) & (g5_ == 0); \
            const __m256i msk_ = _mm256_set1_epi16((short)(0u - is5_)); \
            n01 = _mm256_add_epi16(n01, _mm256_and_si256(msk_, _mm256_min_epu16(a0_, two16))); \
            n23 = _mm256_add_epi16(n23, _mm256_and_si256(msk_, _mm256_min_epu16(a1_, two16))); \
            bad |= is5_ & (sc[g][t] == 0); \
            sc[g][t] = (unsigned char)(sc[g][t] - is5_); \
        } \
        { \
            const __m256i P_ = _mm256_or_si256(n01, _mm256_slli_epi16(n23, 4)); \
            const __m256i Q_ = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(P_, P_), perm); \
            _mm_storeu_si128((__m128i *)(void *)(out[g] + (size_t)(t) * 16u), _mm256_castsi256_si128(Q_)); \
        } \
    } while (0)
#define W2_SIGNS(n01, n23, s01, s23, sg) do { \
        const uint32_t sh_ = (sg) >> 16; \
        n01 = _mm256_or_si256(s01, _mm256_loadu2_m128i(&X->sl16[(((sg) >> 4) & 0xfu) | (((sg) >> 8) & 0xf0u)], &X->sl16[((sg) & 0xfu) | (((sg) >> 4) & 0xf0u)])); \
        n23 = _mm256_or_si256(s23, _mm256_loadu2_m128i(&X->sl16[((sh_ >> 4) & 0xfu) | ((sh_ >> 8) & 0xf0u)], &X->sl16[(sh_ & 0xfu) | ((sh_ >> 4) & 0xf0u)])); \
    } while (0)
    for (g = 0; g < G; g++) {
        const uint32_t s0 = wnx_ident_hash(J->v->model_id, J->v->owner, S[g].g, J->v->word16);
        for (v = 0; v < 4u; v++) {
            x[g][v] = _mm256_loadu_si256((const __m256i *)(const void *)(S[g].begin + 32u * v));
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_srli_epi32(x[g][v], 16), zero))) return -1;
        }
        h[g][0] = wnxf_seed_avx2(s0, 0); h[g][1] = wnxf_seed_avx2(s0, 1); acc[g][0] = acc[g][1] = zero;
        ptr[g] = S[g].begin + 128; sp[g] = S[g].end;
        out[g] = J->wo[m] + (size_t)S[g].first * 16u; sc[g] = J->so[m] + S[g].first;
    }
    for (t = 0; t < T; t++) {
        for (g = 0; g < G; g++) {
            const uint32_t r = J->srel[m][sc[g][t]];
            const __m256i *thr = X->thr[m][r];
            __m256i slot[4], sym[4], p01, p23, s01, s23, n01, n23;
            if (ptr[g] > S[g].end) return -1;
            if (!joint && t < Tm && nb[g] < 32u) {
                uint32_t w;
                if (sp[g] - S[g].begin < 4) return -1;
                sp[g] -= 4; memcpy(&w, sp[g], 4); bb[g] |= (uint64_t)w << nb[g]; nb[g] += 32u;
            }
            for (v = 0; v < 4u; v++) slot[v] = _mm256_and_si256(x[g][v], low12);
            p01 = _mm256_packus_epi32(slot[0], slot[1]); p23 = _mm256_packus_epi32(slot[2], slot[3]);
            {   /* symbol = number of thresholds <= slot, 16 lanes at a time (two chains for latency) */
                __m256i a01 = _mm256_cmpgt_epi16(p01, thr[0]), a23 = _mm256_cmpgt_epi16(p23, thr[0]);
                __m256i b01 = _mm256_cmpgt_epi16(p01, thr[1]), b23 = _mm256_cmpgt_epi16(p23, thr[1]);
                uint32_t i;
                for (i = 2; i < nthr; i += 2) {
                    a01 = _mm256_add_epi16(a01, _mm256_cmpgt_epi16(p01, thr[i]));
                    a23 = _mm256_add_epi16(a23, _mm256_cmpgt_epi16(p23, thr[i]));
                    if (i + 1u < nthr) {
                        b01 = _mm256_add_epi16(b01, _mm256_cmpgt_epi16(p01, thr[i + 1u]));
                        b23 = _mm256_add_epi16(b23, _mm256_cmpgt_epi16(p23, thr[i + 1u]));
                    }
                }
                s01 = _mm256_sub_epi16(zero, _mm256_add_epi16(a01, b01)); s23 = _mm256_sub_epi16(zero, _mm256_add_epi16(a23, b23));
            }
            sym[0] = _mm256_unpacklo_epi16(s01, zero); sym[1] = _mm256_unpackhi_epi16(s01, zero);
            sym[2] = _mm256_unpacklo_epi16(s23, zero); sym[3] = _mm256_unpackhi_epi16(s23, zero);
            for (v = 0; v < 4u; v++) {
                __m256i st, fq;
                if (joint) {
                    const __m256i hi = _mm256_cmpgt_epi32(sym[v], seven);
                    st = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(X->cv[m][r][0], sym[v]), _mm256_permutevar8x32_epi32(X->cv[m][r][1], sym[v]), hi);
                    fq = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(X->fv[m][r][0], sym[v]), _mm256_permutevar8x32_epi32(X->fv[m][r][1], sym[v]), hi);
                } else {
                    st = _mm256_permutevar8x32_epi32(X->cv[m][r][0], sym[v]);
                    fq = _mm256_permutevar8x32_epi32(X->fv[m][r][0], sym[v]);
                }
                x[g][v] = _mm256_add_epi32(_mm256_mullo_epi32(fq, _mm256_srli_epi32(x[g][v], WNX_P)), _mm256_sub_epi32(slot[v], st));
            }
            for (v = 0; v < 4u; v++) x[g][v] = wnxf_renorm8(X, x[g][v], &ptr[g]);
            if (tail && t >= Tm) { dbuf[g][t - Tm][0] = s01; dbuf[g][t - Tm][1] = s23; continue; }
            if (joint) {   /* joint symbol 8..14 -> nibble 9..15 */
                n01 = _mm256_add_epi16(s01, _mm256_and_si256(_mm256_cmpgt_epi16(s01, seven16), one16));
                n23 = _mm256_add_epi16(s23, _mm256_and_si256(_mm256_cmpgt_epi16(s23, seven16), one16));
            } else {
                const uint32_t nz = wnxf_nz16(s01, s23), n = (uint32_t)__builtin_popcount(nz);
                const uint32_t sg = _pdep_u32((uint32_t)bb[g], nz);
                bb[g] >>= n; nb[g] -= n; used[g] += n;
                W2_SIGNS(n01, n23, s01, s23, sg);
            }
            W2_FINISH(g, t, n01, n23);
        }
    }
    for (g = 0; g < G; g++) {
        uint32_t xl[32], hc[16];
        for (v = 0; v < 4u; v++) _mm256_storeu_si256((__m256i *)(void *)(xl + 8u * v), x[g][v]);
        if (tail) {
            WnxfPool P;
            if (wnxf_pool_open(&P, xl)) return -1;
            for (t = Tm; t < T; t++) {
                const __m256i s01 = dbuf[g][t - Tm][0], s23 = dbuf[g][t - Tm][1];
                __m256i n01, n23;
                const uint32_t nz = wnxf_nz16(s01, s23);
                uint32_t bits, sg;
                if (wnxf_take(&P, &bb[g], &nb[g], &used[g], &sp[g], S[g].begin, (uint32_t)__builtin_popcount(nz), &bits)) return -1;
                sg = _pdep_u32(bits, nz);
                W2_SIGNS(n01, n23, s01, s23, sg);
                W2_FINISH(g, t, n01, n23);
            }
            if (!wnxf_pool_rest_zero(&P)) return -1;
        }
        if (T & 3u) { h[g][0] = wnxf_mur8(h[g][0], acc[g][0]); h[g][1] = wnxf_mur8(h[g][1], acc[g][1]); }
        if (!joint) {
            const uint32_t words = (used[g] + 31u) / 32u, spare = (32u - used[g] % 32u) % 32u;
            if (ptr[g] != S[g].end - 4u * words || (bb[g] & ((UINT64_C(1) << spare) - 1u))) return -1;
        } else if (ptr[g] != S[g].end) return -1;
        wnxf_chains_avx2(h[g][0], h[g][1], hc);
        if (tail) {
            const uint32_t s0 = wnx_ident_hash(J->v->model_id, J->v->owner, S[g].g, J->v->word16);
            uint32_t hh[16], x0[32], k;
            for (k = 0; k < 16u; k++) hh[k] = wex_nr3_h32_final(hc[k], T);
            wnx_expect(s0, hh, x0);
            if (xl[0] != x0[0] || xl[1] != x0[1]) return -1;
        } else if (wnxf_close(J, S[g].g, hc, xl, T)) return -1;
    }
    return bad ? -1 : 0;
#undef W2_FINISH
#undef W2_SIGNS
}

#if WNXF_512
WNXF_INLINE __m512i wnxf_mur16(__m512i h, __m512i k) {
    __m512i q = _mm512_mullo_epi32(k, _mm512_set1_epi32((int)0xcc9e2d51u));
    q = _mm512_mullo_epi32(_mm512_rol_epi32(q, 15), _mm512_set1_epi32(0x1b873593));
    q = _mm512_rol_epi32(_mm512_xor_si512(h, q), 13);
    return _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(q, 2), q), _mm512_set1_epi32((int)0xe6546b64u));
}
/* ---- weights, AVX-512: lanes 0..15 and 16..31 in two vectors; 16-bit work on all 32 lanes at once ---- */
WNXF_INLINE int wnxf_w5(const WnxfJob *J, const WnxfS *S, const uint32_t G, const int joint, const int tail) {
    const WnxfTables *X = J->X;
    const uint32_t m = S[0].m, T = S[0].T, ceil = J->ceil >> m & 1u, nthr = joint ? 14u : 7u;
    const uint32_t Tm = tail ? (T > WNX_TAIL_STEPS ? T - WNX_TAIL_STEPS : 0u) : T;
    const __m512i low12 = _mm512_set1_epi32((int)(WNX_T - 1u)), L = _mm512_set1_epi32((int)WNX_L);
    const __m512i z16 = _mm512_setzero_si512(), seven16 = _mm512_set1_epi16(7), five16 = _mm512_set1_epi16(5);
    const __m512i two16 = _mm512_set1_epi16(2), eight16 = _mm512_set1_epi16(8), one16 = _mm512_set1_epi16(1);
    const unsigned char *ptr[2], *sp[2];
    unsigned char *out[2], *sc[2];
    uint64_t bb[2] = {0, 0};
    uint32_t nb[2] = {0, 0}, used[2] = {0, 0}, g, t, bad = 0;
    __m512i x[2][2], h[2], acc[2], dbuf[2][tail ? WNX_TAIL_STEPS : 1];
#define W5_FINISH(g, t, nib16) do { \
        switch ((t) & 3u) { \
        case 0: acc[g] = _mm512_or_si512(acc[g], nib16); break; \
        case 1: acc[g] = _mm512_or_si512(acc[g], _mm512_slli_epi16(nib16, 4)); break; \
        case 2: acc[g] = _mm512_or_si512(acc[g], _mm512_slli_epi16(nib16, 8)); break; \
        default: acc[g] = _mm512_or_si512(acc[g], _mm512_slli_epi16(nib16, 12)); \
        } \
        if (((t) & 3u) == 3u) { h[g] = wnxf_mur16(h[g], acc[g]); acc[g] = z16; } \
        if (ceil) { \
            const __m512i a16_ = _mm512_and_si512(nib16, seven16); \
            const uint32_t e5_ = (uint32_t)_mm512_cmpeq_epi16_mask(a16_, five16), g5_ = (uint32_t)_mm512_cmpgt_epu16_mask(a16_, five16); \
            const uint32_t is5_ = (e5_ != 0) & (g5_ == 0); \
            nib16 = _mm512_mask_add_epi16(nib16, (__mmask32)(0u - is5_), nib16, _mm512_min_epu16(a16_, two16)); \
            bad |= is5_ & (sc[g][t] == 0); \
            sc[g][t] = (unsigned char)(sc[g][t] - is5_); \
        } \
        { \
            const __m256i lo_ = _mm512_castsi512_si256(nib16), hi_ = _mm512_extracti64x4_epi64(nib16, 1); \
            _mm_storeu_si128((__m128i *)(void *)(out[g] + (size_t)(t) * 16u), _mm256_cvtepi16_epi8(_mm256_or_si256(lo_, _mm256_slli_epi16(hi_, 4)))); \
        } \
    } while (0)
    for (g = 0; g < G; g++) {
        const uint32_t s0 = wnx_ident_hash(J->v->model_id, J->v->owner, S[g].g, J->v->word16);
        x[g][0] = _mm512_loadu_si512(S[g].begin); x[g][1] = _mm512_loadu_si512(S[g].begin + 64);
        if (_mm512_cmplt_epu32_mask(x[g][0], L) | _mm512_cmplt_epu32_mask(x[g][1], L)) return -1;
        h[g] = _mm512_xor_si512(_mm512_set1_epi32((int)s0), _mm512_loadu_si512(wnxf_golden));
        acc[g] = z16;
        ptr[g] = S[g].begin + 128; sp[g] = S[g].end;
        out[g] = J->wo[m] + (size_t)S[g].first * 16u; sc[g] = J->so[m] + S[g].first;
    }
    for (t = 0; t < T; t++) {
        for (g = 0; g < G; g++) {
            const uint32_t r = J->srel[m][sc[g][t]];
            const __m512i *thr = X->thr5[m][r];
            __m512i slot0, slot1, p, s16, nib16, sym0, sym1, a, b;
            uint32_t i;
            if (ptr[g] > S[g].end) return -1;
            if (!joint && t < Tm && nb[g] < 32u) {
                uint32_t w;
                if (sp[g] - S[g].begin < 4) return -1;
                sp[g] -= 4; memcpy(&w, sp[g], 4); bb[g] |= (uint64_t)w << nb[g]; nb[g] += 32u;
            }
            slot0 = _mm512_and_si512(x[g][0], low12); slot1 = _mm512_and_si512(x[g][1], low12);
            p = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(slot0)), _mm512_cvtepi32_epi16(slot1), 1);
            a = _mm512_maskz_mov_epi16(_mm512_cmpge_epu16_mask(p, thr[0]), one16);
            b = _mm512_maskz_mov_epi16(_mm512_cmpge_epu16_mask(p, thr[1]), one16);
            for (i = 2; i < nthr; i += 2) {   /* two masked-add chains */
                a = _mm512_mask_add_epi16(a, _mm512_cmpge_epu16_mask(p, thr[i]), a, one16);
                if (i + 1u < nthr) b = _mm512_mask_add_epi16(b, _mm512_cmpge_epu16_mask(p, thr[i + 1u]), b, one16);
            }
            s16 = _mm512_add_epi16(a, b);
            sym0 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(s16));
            sym1 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(s16, 1));
            {
                const __m512i st0 = _mm512_permutexvar_epi32(sym0, X->cv5[m][r]), fq0 = _mm512_permutexvar_epi32(sym0, X->fv5[m][r]);
                const __m512i st1 = _mm512_permutexvar_epi32(sym1, X->cv5[m][r]), fq1 = _mm512_permutexvar_epi32(sym1, X->fv5[m][r]);
                x[g][0] = _mm512_add_epi32(_mm512_mullo_epi32(fq0, _mm512_srli_epi32(x[g][0], WNX_P)), _mm512_sub_epi32(slot0, st0));
                x[g][1] = _mm512_add_epi32(_mm512_mullo_epi32(fq1, _mm512_srli_epi32(x[g][1], WNX_P)), _mm512_sub_epi32(slot1, st1));
            }
            {   /* shared-pointer renormalisation: lanes in order take the next words */
                const __mmask16 k0 = _mm512_cmplt_epu32_mask(x[g][0], L), k1 = _mm512_cmplt_epu32_mask(x[g][1], L);
                const unsigned char *q1 = ptr[g] + 2u * (uint32_t)__builtin_popcount((uint32_t)k0);
                const __m512i w0 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(const void *)ptr[g]));
                const __m512i w1 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(const void *)q1));
                x[g][0] = _mm512_mask_or_epi32(x[g][0], k0, _mm512_slli_epi32(x[g][0], 16), _mm512_maskz_expand_epi32(k0, w0));
                x[g][1] = _mm512_mask_or_epi32(x[g][1], k1, _mm512_slli_epi32(x[g][1], 16), _mm512_maskz_expand_epi32(k1, w1));
                ptr[g] = q1 + 2u * (uint32_t)__builtin_popcount((uint32_t)k1);
            }
            if (tail && t >= Tm) { dbuf[g][t - Tm] = s16; continue; }
            if (joint) nib16 = _mm512_mask_add_epi16(s16, _mm512_cmpgt_epu16_mask(s16, seven16), s16, one16);
            else {
                const uint32_t nz = (uint32_t)_mm512_test_epi16_mask(s16, s16), n = (uint32_t)__builtin_popcount(nz);
                const uint32_t sg = _pdep_u32((uint32_t)bb[g], nz);
                bb[g] >>= n; nb[g] -= n; used[g] += n;
                nib16 = _mm512_mask_add_epi16(s16, (__mmask32)sg, s16, eight16);   /* mag <= 7: + 8 = | 8 */
            }
            W5_FINISH(g, t, nib16);
        }
    }
    for (g = 0; g < G; g++) {
        uint32_t xl[32], hc[16];
        _mm512_storeu_si512(xl, x[g][0]); _mm512_storeu_si512(xl + 16, x[g][1]);
        if (tail) {
            WnxfPool P;
            if (wnxf_pool_open(&P, xl)) return -1;
            for (t = Tm; t < T; t++) {
                const __m512i s16 = dbuf[g][t - Tm];
                const uint32_t nz = (uint32_t)_mm512_test_epi16_mask(s16, s16);
                uint32_t bits;
                __m512i nib16;
                if (wnxf_take(&P, &bb[g], &nb[g], &used[g], &sp[g], S[g].begin, (uint32_t)__builtin_popcount(nz), &bits)) return -1;
                nib16 = _mm512_mask_add_epi16(s16, (__mmask32)_pdep_u32(bits, nz), s16, eight16);
                W5_FINISH(g, t, nib16);
            }
            if (!wnxf_pool_rest_zero(&P)) return -1;
        }
        if (T & 3u) h[g] = wnxf_mur16(h[g], acc[g]);
        if (!joint) {
            const uint32_t words = (used[g] + 31u) / 32u, spare = (32u - used[g] % 32u) % 32u;
            if (ptr[g] != S[g].end - 4u * words || (bb[g] & ((UINT64_C(1) << spare) - 1u))) return -1;
        } else if (ptr[g] != S[g].end) return -1;
        _mm512_storeu_si512(hc, h[g]);
        if (tail) {
            const uint32_t s0 = wnx_ident_hash(J->v->model_id, J->v->owner, S[g].g, J->v->word16);
            uint32_t hh[16], x0[32], k;
            for (k = 0; k < 16u; k++) hh[k] = wex_nr3_h32_final(hc[k], T);
            wnx_expect(s0, hh, x0);
            if (xl[0] != x0[0] || xl[1] != x0[1]) return -1;
        } else if (wnxf_close(J, S[g].g, hc, xl, T)) return -1;
    }
    return bad ? -1 : 0;
#undef W5_FINISH
}
#endif

/* kernel: 1 = AVX2, 2 = AVX-512 (if compiled), 0 = reference loop per stream */
static int wnxf_group(const WnxfJob *J, const WnxfS *S, uint32_t G, int kernel) {
    const int joint = J->M->joint, tail = (J->v->flags & WNX_FLAG_TAIL) != 0;
#if WNXF_512
    if (kernel == 2) {
        if (G == 2u) return joint ? wnxf_w5(J, S, 2, 1, 0) : tail ? wnxf_w5(J, S, 2, 0, 1) : wnxf_w5(J, S, 2, 0, 0);
        return joint ? wnxf_w5(J, S, 1, 1, 0) : tail ? wnxf_w5(J, S, 1, 0, 1) : wnxf_w5(J, S, 1, 0, 0);
    }
#endif
    if (G == 2u) return joint ? wnxf_w2(J, S, 2, 1, 0) : tail ? wnxf_w2(J, S, 2, 0, 1) : wnxf_w2(J, S, 2, 0, 0);
    return joint ? wnxf_w2(J, S, 1, 1, 0) : tail ? wnxf_w2(J, S, 1, 0, 1) : wnxf_w2(J, S, 1, 0, 0);
}

typedef struct { int threads, kernel, G, scales_only; } WnxfOpt;

/* Decode a parsed record. si = wnx_streams() table. Returns 0 / -1. */
/* hard = end of the readable record buffer (the record with its zero padding). */
static inline int wnxf_decode(const WnxfTables *X, const WnxModel *M, const WnxView *v, const WnxStreamInfo *si, uint32_t n,
                              unsigned char *const so[3], unsigned char *const wo[3], const WnxfOpt *o, const unsigned char *hard) {
    WnxfJob J;
    uint32_t m, nscale = v->scale_streams, i;
    int bad = 0, s;
    if (X->id != M->id || v->model_id != M->id) return -1;
    J.v = v; J.M = M; J.X = X; J.so = so; J.wo = wo; J.ceil = v->flags & 7u;
    for (m = 0; m < 3u; m++) for (i = 0; i < 256u; i++) J.srel[m][i] = (unsigned char)wex_nr2_srel(i, v->anchor[m]);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(|:bad) num_threads(o->threads > 0 ? o->threads : 1)
#endif
    for (s = 0; s < (int)nscale; s++) {
        const WnxStreamInfo *q = si + s;
        const unsigned char *b = v->payload + q->begin, *e = v->payload + q->end;
        if (o->kernel && !(q->cnt % 32u) && e + WNX_SLACK <= hard) {
            WnxfS S = {b, e, q->m, q->first, q->cnt / 32u, q->g};
            bad |= wnxf_scales(&J, &S) != 0;
        } else bad |= wnx_stream_ref(v, M, 0, q->m, q->first, q->cnt, q->g, b, e, so, wo) != 0;
    }
    if (bad) return -1;
    if (o->scales_only) return 0;
    {
        const int units = (int)((n - nscale + 1u) / 2u);   /* pairs of weight streams */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(|:bad) num_threads(o->threads > 0 ? o->threads : 1)
#endif
        for (s = 0; s < units; s++) {
            const uint32_t a = nscale + 2u * (uint32_t)s, z = a + 2u <= n ? a + 2u : n;
            WnxfS S[2];
            uint32_t k, ok = 1;
            for (k = a; k < z; k++) {
                const WnxStreamInfo *q = si + k;
                S[k - a].begin = v->payload + q->begin; S[k - a].end = v->payload + q->end;
                S[k - a].m = q->m; S[k - a].first = q->first; S[k - a].T = q->cnt; S[k - a].g = q->g;
                ok &= S[k - a].end + WNX_SLACK <= hard;
            }
            if (o->kernel && ok && o->G == 2 && z - a == 2u && S[0].m == S[1].m && S[0].T == S[1].T)
                bad |= wnxf_group(&J, S, 2, o->kernel) != 0;
            else
                for (k = a; k < z; k++) {
                    if (o->kernel && S[k - a].end + WNX_SLACK <= hard) bad |= wnxf_group(&J, S + (k - a), 1, o->kernel) != 0;
                    else bad |= wnx_stream_ref(v, M, 1, si[k].m, si[k].first, si[k].cnt, si[k].g, S[k - a].begin, S[k - a].end, so, wo) != 0;
                }
        }
    }
    return bad ? -1 : 0;
}
#endif
