/* AVX2 8-lane weight decoder: 8 streams per __m256i (one 32-bit rANS state per lane), VG independent groups
   interleaved per call (8*VG streams). Symbol lookup by gathers:
     VLUT == 0: 4-step binary search over the u32 cum_excl row (4 gathers) + 1 gather of (f | c << 16);
     VLUT == 1: bucket LUT (u8, 2^(14-NBSH) buckets per row) gives the symbol at the bucket start, then
                gather (f|c), and advance while slot >= c + f (loop until no lane needs it).
   Renorm: one unaligned dword gather at each lane's read pointer, byte-swapped, variable shifts by 0/8/16.
   Included from wnrdec.c with VG, VLUT, VNAME set. */
static FINL int VNAME(const Dec *d, int g0, const int HASH, const int CEIL, uint64_t *nfix)
{
    const Model *M = d->M;
    const int *cx32 = (const int *)M->cx32;
    const int *fc32 = (const int *)M->fc32;
    const int *lut = (const int *)M->lut;
    const int *base = (const int *)d->rec;
    const __m256i vmask = _mm256_set1_epi32(PMASK), v15 = _mm256_set1_epi32(15), vffff = _mm256_set1_epi32(0xFFFF);
    const __m256i vL = _mm256_set1_epi32(RL), vL8 = _mm256_set1_epi32(RL >> 8), v32 = _mm256_set1_epi32(32);
    const __m256i v7 = _mm256_set1_epi32(7), vprime = _mm256_set1_epi32((int)FNV_PRIME), vone = _mm256_set1_epi32(1);
    const __m256i v8 = _mm256_set1_epi32(8), v4 = _mm256_set1_epi32(4), v2 = _mm256_set1_epi32(2);
    const __m256i vff = _mm256_set1_epi32(0xFF), v80 = _mm256_set1_epi32(80);
    const __m256i bswap = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                                           3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m256i X[VG], PTR[VG], H[VG], MX[VG], PMX[VG], RB[VG], WD[VG];
    ALIGNED(32) uint32_t wb[VG][256][8];
    ALIGNED(32) uint32_t bmb[VG][BPS][8];
    ALIGNED(32) uint32_t tmp[8], tmp2[8], tmp3[8];
    uint8_t bm8[8][BPS];
    int bad = 0;
    uint64_t fixes = 0;
    for (int gi = 0; gi < VG; gi++) {
        for (int l = 0; l < 8; l++) {
            int g = g0 + gi * 8 + l;
            const uint8_t *pp = d->rec + d->woff[g];
            uint32_t xx = ld_be32(pp);
            bad += (xx < RL) | (xx >= 0x80000000u) | (d->wlen[g] < 4);
            if (xx < RL || xx >= 0x80000000u) xx = RL;
            tmp[l] = xx;
            tmp2[l] = d->woff[g] + 4;
            tmp3[l] = HASH ? seed_of(d, 1, d->wmat[g], d->widx[g]) : 0;
        }
        X[gi] = _mm256_load_si256((const __m256i *)tmp);
        PTR[gi] = _mm256_load_si256((const __m256i *)tmp2);
        H[gi] = _mm256_load_si256((const __m256i *)tmp3);
    }
    for (int b = 0; b < BPS; b++) {
        for (int gi = 0; gi < VG; gi++) {
            for (int l = 0; l < 8; l++) {
                int g = g0 + gi * 8 + l;
                int m = d->wmat[g];
                int dd = (int)d->slab[d->sc_base[m] + (size_t)d->widx[g] * BPS + b] - d->anchor[m];
                dd = dd < -3 ? -3 : (dd > 3 ? 3 : dd);
                tmp[l] = (uint32_t)(m * NCTX + (dd + 3) * 81);
            }
            RB[gi] = _mm256_load_si256((const __m256i *)tmp);
            PMX[gi] = v80;
            MX[gi] = _mm256_setzero_si256();
        }
        for (int q = 0; q < 4; q++) {
            for (int gi = 0; gi < VG; gi++) WD[gi] = _mm256_setzero_si256();
            for (int t = 0; t < 8; t++) {
                const __m256i vsh = _mm256_set1_epi32(4 * t);
                for (int gi = 0; gi < VG; gi++) {
                    __m256i slot = _mm256_and_si256(X[gi], vmask);
                    __m256i row = _mm256_add_epi32(RB[gi], PMX[gi]);
                    __m256i idx = _mm256_slli_epi32(row, 4);
                    __m256i fc;
#if VLUT
                    __m256i li = _mm256_add_epi32(_mm256_slli_epi32(row, PB - NBSH), _mm256_srli_epi32(slot, NBSH));
                    __m256i s0 = _mm256_and_si256(_mm256_i32gather_epi32(lut, li, 1), vff);
                    idx = _mm256_add_epi32(idx, s0);
                    fc = _mm256_i32gather_epi32(fc32, idx, 4);
                    for (;;) {
                        __m256i cf = _mm256_add_epi32(_mm256_srli_epi32(fc, 16), _mm256_and_si256(fc, vffff));
                        __m256i need = _mm256_cmpgt_epi32(_mm256_add_epi32(slot, vone), cf);
                        if (_mm256_testz_si256(need, need)) break;
                        fixes++;
                        idx = _mm256_sub_epi32(idx, need);
                        fc = _mm256_i32gather_epi32(fc32, idx, 4);
                    }
#else
                    __m256i t8 = _mm256_i32gather_epi32(cx32, _mm256_add_epi32(idx, v8), 4);
                    idx = _mm256_add_epi32(idx, _mm256_andnot_si256(_mm256_cmpgt_epi32(t8, slot), v8));
                    __m256i t4 = _mm256_i32gather_epi32(cx32, _mm256_add_epi32(idx, v4), 4);
                    idx = _mm256_add_epi32(idx, _mm256_andnot_si256(_mm256_cmpgt_epi32(t4, slot), v4));
                    __m256i t2 = _mm256_i32gather_epi32(cx32, _mm256_add_epi32(idx, v2), 4);
                    idx = _mm256_add_epi32(idx, _mm256_andnot_si256(_mm256_cmpgt_epi32(t2, slot), v2));
                    __m256i t1 = _mm256_i32gather_epi32(cx32, _mm256_add_epi32(idx, vone), 4);
                    idx = _mm256_add_epi32(idx, _mm256_andnot_si256(_mm256_cmpgt_epi32(t1, slot), vone));
                    fc = _mm256_i32gather_epi32(fc32, idx, 4);
#endif
                    __m256i s = _mm256_and_si256(idx, v15);
                    __m256i f = _mm256_and_si256(fc, vffff), c = _mm256_srli_epi32(fc, 16);
                    __m256i xx = _mm256_add_epi32(_mm256_mullo_epi32(f, _mm256_srli_epi32(X[gi], PB)), _mm256_sub_epi32(slot, c));
                    __m256i v = _mm256_shuffle_epi8(_mm256_i32gather_epi32(base, PTR[gi], 1), bswap);
                    __m256i k = _mm256_sub_epi32(_mm256_setzero_si256(),
                                                 _mm256_add_epi32(_mm256_cmpgt_epi32(vL, xx), _mm256_cmpgt_epi32(vL8, xx)));
                    __m256i sh = _mm256_slli_epi32(k, 3);
                    X[gi] = _mm256_or_si256(_mm256_sllv_epi32(xx, sh), _mm256_srlv_epi32(v, _mm256_sub_epi32(v32, sh)));
                    PTR[gi] = _mm256_add_epi32(PTR[gi], k);
                    if (HASH) H[gi] = _mm256_mullo_epi32(_mm256_xor_si256(H[gi], s), vprime);
                    __m256i mag = _mm256_and_si256(s, v7);
                    MX[gi] = _mm256_max_epu32(MX[gi], mag);
                    PMX[gi] = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(mag, 3), mag), MX[gi]);
                    WD[gi] = _mm256_or_si256(WD[gi], _mm256_sllv_epi32(s, vsh));
                }
            }
            for (int gi = 0; gi < VG; gi++) _mm256_store_si256((__m256i *)wb[gi][b * 4 + q], WD[gi]);
        }
        for (int gi = 0; gi < VG; gi++) _mm256_store_si256((__m256i *)bmb[gi][b], MX[gi]);
    }
    for (int gi = 0; gi < VG; gi++) {
        _mm256_store_si256((__m256i *)tmp, X[gi]);
        _mm256_store_si256((__m256i *)tmp2, PTR[gi]);
        _mm256_store_si256((__m256i *)tmp3, H[gi]);
        for (int l = 0; l < 8; l++) {
            int g = g0 + gi * 8 + l;
            int m = d->wmat[g], j = d->widx[g];
            uint32_t *o = (uint32_t *)(d->slab + d->w_base[m] + (size_t)j * (WCH / 2));
            for (int i = 0; i < 256; i++) o[i] = wb[gi][i][l];
            uint32_t want = RL + (HASH ? (lowbias32(tmp3[l]) & 0x7FFFFFu) : 0);
            uint32_t pend = d->woff[g] + d->wlen[g];
            if (HASH || !d->hash_in_rec) bad += (tmp2[l] != pend) | (tmp[l] != want);
            else bad += (tmp2[l] != pend) | (tmp[l] < RL) | (tmp[l] >= RL + 0x800000u);
            if (CEIL) {
                for (int b = 0; b < BPS; b++) bm8[l][b] = (uint8_t)bmb[gi][b][l];
                bad += ceil_inv((uint8_t *)o, d->slab + d->sc_base[m] + (size_t)j * BPS, bm8[l]);
            }
        }
    }
    if (nfix) *nfix += fixes;
    return bad;
}
