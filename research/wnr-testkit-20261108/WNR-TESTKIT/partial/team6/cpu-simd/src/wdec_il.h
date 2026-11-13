/* Interleaved scalar weight decoder: ILK streams per call, state in GPRs, symbol lookup by one AVX2 compare of the
   16-entry u16 cum_excl row (vpcmpgtw + vpmovmskb + popcnt), branchless byte renorm.
   Included twice from wnrdec.c with ILK = 4 / 8 and ILNAME set. */
static FINL int ILNAME(const Dec *d, int g0, const int HASH, const int CEIL)
{
    const Model *M = d->M;
    const uint16_t *cx16 = M->cx16;
    const uint32_t *fc32 = M->fc32;
    uint32_t x[ILK], h[ILK], mx[ILK], pmx[ILK], rb[ILK], mrow[ILK], wd[ILK];
    const uint8_t *p[ILK], *pend[ILK];
    uint8_t *out[ILK], *sc[ILK];
    int anc[ILK];
    uint8_t bm[ILK][BPS];
    int bad = 0;
#define IL_INIT(l) { int g = g0 + l; int m = d->wmat[g], j = d->widx[g]; \
        p[l] = d->rec + d->woff[g]; pend[l] = p[l] + d->wlen[g]; \
        x[l] = ld_be32(p[l]); p[l] += 4; bad += (x[l] < RL) | (x[l] >= 0x80000000u) | (d->wlen[g] < 4); \
        if (x[l] < RL || x[l] >= 0x80000000u) x[l] = RL; \
        h[l] = HASH ? seed_of(d, 1, m, j) : 0; \
        out[l] = d->slab + d->w_base[m] + (size_t)j * (WCH / 2); \
        sc[l] = d->slab + d->sc_base[m] + (size_t)j * BPS; \
        anc[l] = d->anchor[m]; mrow[l] = (uint32_t)(m * NCTX); }
    IL_LANES(IL_INIT)
    for (int b = 0; b < BPS; b++) {
#define IL_BLK(l) { int dd = (int)sc[l][b] - anc[l]; dd = dd < -3 ? -3 : (dd > 3 ? 3 : dd); \
        rb[l] = mrow[l] + (uint32_t)(dd + 3) * 81; pmx[l] = 80; mx[l] = 0; }
        IL_LANES(IL_BLK)
        for (int q = 0; q < 4; q++) {
#define IL_ZW(l) wd[l] = 0;
            IL_LANES(IL_ZW)
            for (int t = 0; t < 8; t++) {
#define IL_STEP(l) { \
                uint32_t slot = x[l] & PMASK; uint32_t row = rb[l] + pmx[l]; \
                __m256i r = _mm256_load_si256((const __m256i *)(cx16 + (size_t)row * 16)); \
                uint32_t msk = (uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi16(r, _mm256_set1_epi16((short)slot))); \
                uint32_t s = 15 - (POPCNT(msk) >> 1); \
                uint32_t fc = fc32[(size_t)row * 16 + s]; \
                uint32_t xx = (fc & 0xFFFF) * (x[l] >> PB) + slot - (fc >> 16); \
                uint32_t v = ld_be32(p[l]); uint32_t k = (xx < RL) + (xx < (RL >> 8)); \
                x[l] = (uint32_t)(((uint64_t)xx << (k * 8)) | ((uint64_t)v >> (32 - k * 8))); p[l] += k; \
                if (HASH) h[l] = (h[l] ^ s) * FNV_PRIME; \
                { uint32_t mag = s & 7; mx[l] = mx[l] > mag ? mx[l] : mag; pmx[l] = mag * 9 + mx[l]; } \
                wd[l] |= s << (4 * t); }
                IL_LANES(IL_STEP)
            }
#define IL_ST(l) memcpy(out[l] + b * 16 + q * 4, &wd[l], 4);
            IL_LANES(IL_ST)
        }
#define IL_BM(l) bm[l][b] = (uint8_t)mx[l];
        IL_LANES(IL_BM)
    }
#define IL_FIN(l) { uint32_t want = RL + (HASH ? (lowbias32(h[l]) & 0x7FFFFFu) : 0); \
        if (HASH || !d->hash_in_rec) bad += (p[l] != pend[l]) | (x[l] != want); \
        else bad += (p[l] != pend[l]) | (x[l] < RL) | (x[l] >= RL + 0x800000u); \
        if (CEIL) bad += ceil_inv(out[l], sc[l], bm[l]); }
    IL_LANES(IL_FIN)
    return bad;
#undef IL_INIT
#undef IL_BLK
#undef IL_ZW
#undef IL_STEP
#undef IL_ST
#undef IL_BM
#undef IL_FIN
}
