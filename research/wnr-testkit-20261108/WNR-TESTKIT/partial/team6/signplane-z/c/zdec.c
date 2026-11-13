// signplane-z (team 6): CPU decoder for WNR2-style weight streams in four layouts, paired timing on one expert.
//   CF : 16-symbol CDF rows (canonical zero, team 4 fgcde streams)
//   D  : 8-symbol magnitude rows + fixed sign plane (u32 per 32-block)
//   Z1 : 8-symbol magnitude rows + compacted sign plane at the stream tail, read backwards (this member's layout)
//   Z2 : Z1 + the last r = min(nsg & 7, nonzeros of the last block) signs carried in x0 above the 23-bit hash
// rANS: 32-bit state, L = 2^23, 14-bit CDF, byte renorm, 4-B big-endian initial state; contexts (srel*9+pm)*9+mx,
// 567 rows per matrix type, pm = mx = 8 at a block start. Per-stream hash (team 4 ceil-ref E): FNV-1a over the identity
// (seed given) then every coded nibble, lowbias32, & (2^23-1); final state == L + h23 (Z2: low 23 bits of x - L).
// Symbol search: SSE2/AVX2 compare of the row's cum[1..alpha] against the slot, popcount (like the GPU dsym8/dsym16).
// Signs (D, Z1, Z2) are merged per block with BMI2: Z: s = pdep(bits, nonzero mask); spread to bit 3 of each nibble.
// NI streams are decoded interleaved by one thread (independent chains for ILP).
// Build: gcc -O3 -march=native -o zdec.exe zdec.c      Run: zdec JOB REPS NI [nohash]
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define RL (1u << 23)
#define M23 ((1u << 23) - 1u)
#define FNVP 16777619u
#define NCTX 567

typedef struct {
    uint32_t var, S, R, alpha, stride;
    uint64_t paylen;
    uint16_t *tab;
    uint32_t *off, *len, *seed;
    uint8_t *mat, *srel, *pay;  // pay points into a buffer with 64 B front and 64 B tail padding
    uint32_t *signs;
    uint8_t *expect;
} Job;

static void *xread(FILE *f, size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p || fread(p, 1, n, f) != n) { fprintf(stderr, "read failed\n"); exit(2); }
    return p;
}

static Job load(const char *path) {
    Job j;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
    char mg[4];
    uint32_t ver;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "ZJOB", 4)) { fprintf(stderr, "magic\n"); exit(2); }
    uint32_t h[5];
    if (fread(&ver, 4, 1, f) != 1 || fread(h, 4, 5, f) != 5 || fread(&j.paylen, 8, 1, f) != 1) exit(2);
    // header after the magic: version, var, S, R, alpha, stride (u32), paylen (u64)
    j.var = h[0]; j.S = h[1]; j.R = h[2]; j.alpha = h[3]; j.stride = h[4];
    j.tab = xread(f, (size_t)j.R * j.stride * 2);
    j.off = xread(f, 4 * (size_t)j.S); j.len = xread(f, 4 * (size_t)j.S); j.seed = xread(f, 4 * (size_t)j.S);
    j.mat = xread(f, j.S); j.srel = xread(f, 64 * (size_t)j.S);
    uint8_t *pb = calloc(j.paylen + 256, 1);
    if (fread(pb + 128, 1, j.paylen, f) != j.paylen) exit(2);
    j.pay = pb + 128;
    j.signs = (j.var == 1) ? xread(f, 4 * 64 * (size_t)j.S) : NULL;
    j.expect = xread(f, 1024 * (size_t)j.S);
    fclose(f);
    return j;
}

static inline uint32_t lowbias32(uint32_t h) {
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return h;
}

static inline uint32_t dsym8(const uint16_t *r, uint32_t *x) {
    uint32_t x_ = *x, slot = x_ & 16383u;
    __m128i c = _mm_loadu_si128((const __m128i *)(r + 1));
    uint32_t m = (uint32_t)_mm_movemask_epi8(_mm_cmpgt_epi16(c, _mm_set1_epi16((short)slot)));
    uint32_t s = 8u - ((uint32_t)_mm_popcnt_u32(m) >> 1);
    uint32_t st = r[s];
    *x = ((uint32_t)r[s + 1] - st) * (x_ >> 14) + slot - st;
    return s;
}

static inline uint32_t dsym16(const uint16_t *r, uint32_t *x) {
    uint32_t x_ = *x, slot = x_ & 16383u;
    __m256i c = _mm256_loadu_si256((const __m256i *)(r + 1));
    uint32_t m = (uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi16(c, _mm256_set1_epi16((short)slot)));
    uint32_t s = 16u - ((uint32_t)_mm_popcnt_u32(m) >> 1);
    uint32_t st = r[s];
    *x = ((uint32_t)r[s + 1] - st) * (x_ >> 14) + slot - st;
    return s;
}

#define RENORM(x, p)                                       \
    do {                                                   \
        uint32_t c_ = (x) < RL, y_ = ((x) << 8) | (p)[0];  \
        (x) = c_ ? y_ : (x); (p) += c_;                    \
        c_ = (x) < RL; y_ = ((x) << 8) | (p)[0];           \
        (x) = c_ ? y_ : (x); (p) += c_;                    \
    } while (0)

// spread 32 sign bits (bit j = sign of nibble j) to bit 3 of the nibbles of the block's 128-bit word (lo = nibbles 0..15)
static inline void spread_signs(uint32_t sb, uint64_t *lo, uint64_t *hi) {
    *lo |= _pdep_u64((uint64_t)(sb & 0xFFFFu), 0x8888888888888888ull);
    *hi |= _pdep_u64((uint64_t)(sb >> 16), 0x8888888888888888ull);
}

static inline uint32_t hash_block(uint32_t h, uint64_t lo, uint64_t hi) {
    for (int q = 0; q < 16; q++) h = (h ^ (uint32_t)((lo >> (4 * q)) & 15u)) * FNVP;
    for (int q = 0; q < 16; q++) h = (h ^ (uint32_t)((hi >> (4 * q)) & 15u)) * FNVP;
    return h;
}

// nonzero mask of 16 magnitude nibbles (bit 3 clear) -> 16 bits
static inline uint32_t nzmask16(uint64_t w) {
    uint64_t t = (w | (w >> 1) | (w >> 2)) & 0x1111111111111111ull;
    return (uint32_t)_pext_u64(t, 0x1111111111111111ull);
}

typedef struct { long bad_out, bad_state, bad_len, bad_pad; } Err;

// Decode streams [s0, s0+NI) of job j into out (S*1024 bytes). VAR 0 CF, 1 D, 2 Z1, 3 Z2. HASH 0/1.
#define DEFINE_DECODER(NAME, VAR, NI, HASH)                                                                           \
static void NAME(const Job *j, uint32_t s0, uint8_t *out, Err *er) {                                                  \
    const uint32_t ST = j->stride;                                                                                    \
    const uint16_t *T = j->tab;                                                                                       \
    uint32_t x[NI], h[NI], pm[NI], mx[NI], cnt[NI];                                                                   \
    const uint8_t *p[NI], *sp[NI], *end[NI];                                                                          \
    const uint16_t *mb[NI], *rb[NI];                                                                                  \
    uint64_t buf[NI], lo[NI], hi[NI];                                                                                 \
    uint32_t nsg[NI];                                                                                                 \
    for (int n = 0; n < NI; n++) {                                                                                    \
        uint32_t s = s0 + n;                                                                                          \
        const uint8_t *q = j->pay + j->off[s];                                                                        \
        x[n] = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) | ((uint32_t)q[2] << 8) | q[3];                        \
        p[n] = q + 4; end[n] = q + j->len[s]; sp[n] = end[n];                                                         \
        h[n] = j->seed[s]; buf[n] = 0; cnt[n] = 0; nsg[n] = 0;                                                        \
        mb[n] = T + (size_t)j->mat[s] * NCTX * ST;                                                                    \
        if (x[n] < RL) er->bad_state++;                                                                               \
    }                                                                                                                 \
    for (int b = 0; b < 64; b++) {                                                                                    \
        for (int n = 0; n < NI; n++) { rb[n] = mb[n] + (size_t)j->srel[(size_t)(s0 + n) * 64 + b] * 81 * ST;          \
                                       lo[n] = 0; hi[n] = 0; pm[n] = 8; mx[n] = 8; }                                  \
        for (int jj = 0; jj < 32; jj++) {                                                                             \
            for (int n = 0; n < NI; n++) {                                                                            \
                const uint16_t *row = rb[n] + (pm[n] * 9 + mx[n]) * ST;                                               \
                uint32_t sy = (VAR == 0) ? dsym16(row, &x[n]) : dsym8(row, &x[n]);                                    \
                RENORM(x[n], p[n]);                                                                                   \
                uint32_t mg = sy & 7u;                                                                                \
                mx[n] = (jj == 0) ? mg : (mg > mx[n] ? mg : mx[n]);                                                   \
                pm[n] = mg;                                                                                           \
                if (jj < 16) lo[n] |= (uint64_t)sy << (4 * jj); else hi[n] |= (uint64_t)sy << (4 * (jj - 16));       \
            }                                                                                                         \
        }                                                                                                             \
        for (int n = 0; n < NI; n++) {                                                                                \
            uint32_t s = s0 + n;                                                                                      \
            if (VAR == 1) spread_signs(j->signs[(size_t)s * 64 + b], &lo[n], &hi[n]);                                 \
            if (VAR >= 2) {                                                                                           \
                uint32_t nz = nzmask16(lo[n]) | (nzmask16(hi[n]) << 16);                                             \
                while (cnt[n] <= 56) { buf[n] |= (uint64_t)(*--sp[n]) << cnt[n]; cnt[n] += 8; }                       \
                uint32_t k = (uint32_t)_mm_popcnt_u32(nz);                                                            \
                uint32_t sb = (uint32_t)_pdep_u64(buf[n], (uint64_t)nz);                                              \
                buf[n] >>= k; cnt[n] -= k; nsg[n] += k;                                                               \
                if (VAR == 3 && b == 63) { /* keep raw block; signs of the last r nonzeros fixed after the state */  \
                    uint32_t nzl = k, r = (nsg[n] & 7u) < nzl ? (nsg[n] & 7u) : nzl;                                  \
                    uint32_t v = x[n] - RL, t = v >> 23;                                                              \
                    if (t >> r) er->bad_state++;                                                                      \
                    /* the last r set bits of nz take bits of t (lowest index first) */                               \
                    uint32_t tailmask = 0, mm = nz;                                                                   \
                    for (uint32_t i = 0; i < nzl - r; i++) mm &= mm - 1;                                              \
                    tailmask = mm;                                                                                    \
                    sb = (sb & ~tailmask) | (uint32_t)_pdep_u64(t, tailmask);                                         \
                    /* plane bits actually used: nsg - r; un-consume the r guessed bits */                            \
                    nsg[n] -= r;                                                                                      \
                }                                                                                                     \
                spread_signs(sb, &lo[n], &hi[n]);                                                                     \
            }                                                                                                         \
            if (HASH) h[n] = hash_block(h[n], lo[n], hi[n]);                                                          \
            uint64_t *o = (uint64_t *)(out + (size_t)s * 1024 + (size_t)b * 16);                                      \
            o[0] = lo[n]; o[1] = hi[n];                                                                               \
        }                                                                                                             \
    }                                                                                                                 \
    for (int n = 0; n < NI; n++) {                                                                                    \
        uint32_t want = HASH ? (lowbias32(h[n]) & M23) : 0u;                                                          \
        uint32_t v = x[n] - RL;                                                                                       \
        if (VAR == 3) { if ((v & M23) != want) er->bad_state++; }                                                     \
        else if (v != want) er->bad_state++;                                                                          \
        const uint8_t *rend = end[n];                                                                                 \
        if (VAR >= 2) {                                                                                               \
            uint32_t nb = (nsg[n] + 7u) >> 3;                                                                         \
            rend = end[n] - nb;                                                                                       \
            if (nsg[n] & 7u) { uint8_t lb = end[n][-1 - (int)(nsg[n] >> 3)]; if (lb >> (nsg[n] & 7u)) er->bad_pad++; }\
        }                                                                                                             \
        if (p[n] != rend) er->bad_len++;                                                                              \
    }                                                                                                                 \
}

DEFINE_DECODER(dec_cf_1, 0, 1, 1)
DEFINE_DECODER(dec_cf_2, 0, 2, 1)
DEFINE_DECODER(dec_cf_4, 0, 4, 1)
DEFINE_DECODER(dec_cf_8, 0, 8, 1)
DEFINE_DECODER(dec_d_1, 1, 1, 1)
DEFINE_DECODER(dec_d_2, 1, 2, 1)
DEFINE_DECODER(dec_d_4, 1, 4, 1)
DEFINE_DECODER(dec_d_8, 1, 8, 1)
DEFINE_DECODER(dec_z1_1, 2, 1, 1)
DEFINE_DECODER(dec_z1_2, 2, 2, 1)
DEFINE_DECODER(dec_z1_4, 2, 4, 1)
DEFINE_DECODER(dec_z1_8, 2, 8, 1)
DEFINE_DECODER(dec_z2_1, 3, 1, 1)
DEFINE_DECODER(dec_z2_2, 3, 2, 1)
DEFINE_DECODER(dec_z2_4, 3, 4, 1)
DEFINE_DECODER(dec_z2_8, 3, 8, 1)
// hash off (sensitivity of the FNV per nibble; state check then expects x == L, which fails: count only timing)
DEFINE_DECODER(dec_cf_4n, 0, 4, 0)
DEFINE_DECODER(dec_d_4n, 1, 4, 0)
DEFINE_DECODER(dec_z1_4n, 2, 4, 0)

typedef void (*DecFn)(const Job *, uint32_t, uint8_t *, Err *);

static DecFn pick(uint32_t var, int ni, int hash) {
    if (!hash) {
        if (ni != 4) return NULL;
        return var == 0 ? dec_cf_4n : var == 1 ? dec_d_4n : var == 2 ? dec_z1_4n : NULL;
    }
    DecFn t[4][4] = {{dec_cf_1, dec_cf_2, dec_cf_4, dec_cf_8}, {dec_d_1, dec_d_2, dec_d_4, dec_d_8},
                     {dec_z1_1, dec_z1_2, dec_z1_4, dec_z1_8}, {dec_z2_1, dec_z2_2, dec_z2_4, dec_z2_8}};
    int k = ni == 1 ? 0 : ni == 2 ? 1 : ni == 4 ? 2 : ni == 8 ? 3 : -1;
    return k < 0 ? NULL : t[var][k];
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// usage: zdec REPS NI HASH JOB [JOB ...]   (HASH 1/0). Paired rounds: in round r every job is decoded once, the job order
// rotated by r; per job min/median time, per pair the median over rounds of the per-round time ratio.
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: zdec REPS NI HASH JOB [JOB ...]\n"); return 1; }
    int reps = atoi(argv[1]), ni = atoi(argv[2]), hash = atoi(argv[3]);
    int nj = argc - 4;
    Job *J = calloc(nj, sizeof(Job));
    DecFn *F = calloc(nj, sizeof(DecFn));
    uint8_t **O = calloc(nj, sizeof(uint8_t *));
    double *T = calloc((size_t)nj * reps, sizeof(double));
    Err *E0 = calloc(nj, sizeof(Err));
    for (int k = 0; k < nj; k++) {
        J[k] = load(argv[4 + k]);
        F[k] = pick(J[k].var, ni, hash);
        if (!F[k] || J[k].S % ni) { fprintf(stderr, "no decoder for var %u ni %d hash %d\n", J[k].var, ni, hash); return 1; }
        O[k] = calloc((size_t)J[k].S * 1024, 1);
    }
    LARGE_INTEGER fq, t0, t1;
    QueryPerformanceFrequency(&fq);
    for (int r = 0; r < reps; r++) {
        for (int q = 0; q < nj; q++) {
            int k = (r + q) % nj;
            Err er = {0};
            QueryPerformanceCounter(&t0);
            for (uint32_t s = 0; s < J[k].S; s += ni) F[k](&J[k], s, O[k], &er);
            QueryPerformanceCounter(&t1);
            T[(size_t)k * reps + r] = (double)(t1.QuadPart - t0.QuadPart) / (double)fq.QuadPart;
            if (r == 0) E0[k] = er;
        }
    }
    printf("{\"ni\": %d, \"hash\": %d, \"reps\": %d, \"jobs\": [\n", ni, hash, reps);
    for (int k = 0; k < nj; k++) {
        long badout = 0;
        for (uint32_t s = 0; s < J[k].S; s++)
            badout += memcmp(O[k] + (size_t)s * 1024, J[k].expect + (size_t)s * 1024, 1024) != 0;
        double *ts = malloc(reps * sizeof(double));
        memcpy(ts, T + (size_t)k * reps, reps * sizeof(double));
        qsort(ts, reps, sizeof(double), cmpd);
        double tmin = ts[0], tmed = ts[reps / 2], nsym = (double)J[k].S * 2048.0;
        printf("  {\"job\": \"%s\", \"var\": %u, \"S\": %u, \"payload\": %llu, \"bad_out\": %ld, \"bad_state\": %ld, "
               "\"bad_len\": %ld, \"bad_pad\": %ld, \"t_min_s\": %.6f, \"t_med_s\": %.6f, \"ns_per_sym_min\": %.4f, "
               "\"ns_per_sym_med\": %.4f, \"out_MBps_min\": %.2f, \"ratios_vs\": {",
               argv[4 + k], J[k].var, J[k].S, (unsigned long long)J[k].paylen, badout, E0[k].bad_state, E0[k].bad_len,
               E0[k].bad_pad, tmin, tmed, tmin * 1e9 / nsym, tmed * 1e9 / nsym, (double)J[k].S * 1024.0 / tmin / 1e6);
        for (int k2 = 0; k2 < nj; k2++) {
            double *rr = malloc(reps * sizeof(double));
            for (int r = 0; r < reps; r++) rr[r] = T[(size_t)k * reps + r] / T[(size_t)k2 * reps + r];
            qsort(rr, reps, sizeof(double), cmpd);
            printf("%s\"%u\": %.4f", k2 ? ", " : "", J[k2].var, rr[reps / 2]);
            free(rr);
        }
        printf("}}%s\n", k + 1 < nj ? "," : "");
        free(ts);
    }
    printf("]}\n");
    return 0;
}
