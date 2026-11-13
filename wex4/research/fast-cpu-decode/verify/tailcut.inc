/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* ---- tailcut: a stream whose tail bytes were deleted, followed by fewer bytes than were deleted --------------
 * The reference runs out of bytes and rejects. A read-ahead decoder keeps reading the following bytes; without
 * its 32-symbol bound check it would run past the end of the weight payload. The record is flush against a
 * PAGE_NOACCESS page, so that would fault. Included by vh.c. */
static unsigned char *guard_hi(size_t bytes) {   /* [p, p + bytes) usable, p + bytes not accessible */
    size_t pages = (bytes + 4095u) / 4096u + 1u; DWORD old;
    unsigned char *base = (unsigned char *)VirtualAlloc(NULL, (pages + 1u) * 4096u, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!base || !VirtualProtect(base + pages * 4096u, 4096, PAGE_NOACCESS, &old)) exit(3);
    return base + pages * 4096u - bytes;
}
static const WexFastOptions TV[] = {{1, 0, 1, 0}, {1, 1, 1, 1}, {1, 1, 1, 0}, {1, 1, 2, 0}, {1, 1, 4, 1}, {1, 1, 8, 1}, {1, 1, 8, 0}, {1, 2, 1, 1}, {1, 2, 1, 0}, {3, 1, 4, 1}, {1, 1, 3, 1}, {1, 1, 6, 0}};
#define NTV (sizeof(TV) / sizeof(TV[0]))
static long tc_cases = 0, tc_mism = 0, tc_refacc = 0;
static void tc_run(const char *what, const WexNr2Model *model, const WexNr2Layout *L, unsigned char *rec, size_t total) {
    WexNr2Record v; unsigned char *so[3], *wo[3], *copy; unsigned m; size_t used, k; int st, got;
    if (wex_nr3_parse(rec, total, L, model->id, &v)) { printf("  %s: stopped at the parser\n", what); return; }
    used = (size_t)v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes;
    copy = guard_hi(used); memcpy(copy, rec, used);
    v.lengths = copy + WEX_NR2_HEADER_BYTES; v.scale_payload = copy + v.metadata_bytes; v.weight_payload = v.scale_payload + v.scale_payload_bytes;
    for (m = 0; m < 3u; m++) { so[m] = guard_hi(L->scale_count[m]); wo[m] = guard_hi(L->symbol_count[m] / 2u); }
    st = wex_nr2_decode(&v, L, model, so, wo);
    tc_cases++; tc_refacc += !st;
    got = wex_fast_decode(&v, L, model, so, wo);
    if (got != st) { tc_mism++; printf("  MISMATCH %s default: ref %d fast %d\n", what, st, got); }
    for (k = 0; k < NTV; k++) {
        got = wex_fast_decode_with(fm, &v, L, model, so, wo, &TV[k]);
        if (got != st) { tc_mism++; printf("  MISMATCH %s variant %d: ref %d fast %d\n", what, (int)k, st, got); }
    }
    printf("  %-64s ref %s, all fast variants same status, no fault\n", what, st ? "reject" : "ACCEPT");
    fflush(stdout);
}
/* delete D bytes at the tail of stream T of a valid record (in place); returns the new padded size */
static size_t tail_delete(unsigned char *r, size_t total, const WexNr2Layout *L, uint32_t T, uint32_t D) {
    WexNr2Record v; unsigned char *lengths = r + WEX_NR2_HEADER_BYTES; uint32_t s, at, len, used, pb;
    if (wex_nr3_parse(r, total, L, 0, &v)) return 0;
    used = v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes;
    for (at = v.metadata_bytes, s = 0; s <= T; s++) at += wex_nr_u16(lengths + 2u * s);   /* end of stream T */
    len = wex_nr_u16(lengths + 2u * T);
    if (D + 4u > len) return 0;
    memmove(r + at - D, r + at, used - at);
    len -= D; lengths[2u * T] = (unsigned char)len; lengths[2u * T + 1u] = (unsigned char)(len >> 8);
    pb = (T < v.scale_streams ? v.scale_payload_bytes : v.weight_payload_bytes) - D;
    memcpy(r + (T < v.scale_streams ? 8u : 12u), &pb, 4);
    used -= D; total = ((size_t)used + 4095u) & ~(size_t)4095u; memset(r + used, 0, total - used);
    return total;
}
static int cmd_tailcut(const WexNr2Model *model, const char *rec) {
    Real r; WexNr2Record v; unsigned char *plain, *ps[3], *pw[3], *buf, *work, *cs[3], *cw[3], anchor[3];
    WexNr2Layout L; size_t written, total, cap = 1u << 22; uint32_t m, i, flags, T, lenT, follow; unsigned d;
    load_real(rec, &r);
    if (wex_nr3_parse(r.record, r.total, &r.layout, model->id, &v)) return 1;
    plain = (unsigned char *)malloc(r.raw); point(&r.entry, plain, ps, pw);
    if (wex_fast_decode(&v, &r.layout, model, ps, pw)) return 1;
    buf = (unsigned char *)malloc(cap); work = (unsigned char *)malloc(cap);
    /* W: weight chunk 2048; matrix 2 = 8 full streams + a short last stream; target = the 8th full stream */
    {
        static const uint32_t lastblocks[] = {12, 8, 16, 6};
        unsigned lb;
        for (lb = 0; lb < 4; lb++) {
            L.scale_count[0] = 8; L.scale_count[1] = 8; L.scale_count[2] = 8u * 64u + lastblocks[lb];
            for (m = 0; m < 3u; m++) {
                L.symbol_count[m] = L.scale_count[m] * 32u;
                cs[m] = (unsigned char *)malloc(L.scale_count[m]); cw[m] = (unsigned char *)malloc(L.scale_count[m] * 16u);
                memcpy(cs[m], ps[0] + 1000u * (m + 1u) + lb * 7777u, L.scale_count[m]);
                memcpy(cw[m], pw[0] + (size_t)(1000u * (m + 1u) + lb * 7777u) * 16u, (size_t)L.scale_count[m] * 16u);
            }
            flags = wex_nr3_ceil_forward(&L, cs, cw, NULL) | WEX_NR3_FLAG_CZ;
            for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(cs[m], L.scale_count[m]);
            if (wex_nrx_encode(model, &L, (const unsigned char *const *)cs, (const unsigned char *const *)cw, 2048u, 256u, anchor,
                               WEX_NR3_MAGIC, flags, 7u, buf, cap - 4096u, &written)) return 1;
            total = (written + 4095u) & ~(size_t)4095u; memset(buf + written, 0, total - written);
            if (wex_nr3_parse(buf, total, &L, model->id, &v)) return 1;
            T = v.scale_streams + 2u + 7u; lenT = wex_nr_u16(v.lengths + 2u * T); follow = wex_nr_u16(v.lengths + 2u * (T + 1u));
            printf("W: %u weight streams, target stream %u has %u B, %u B follow it up to the end of the payload (%s the fast path, slack 104)\n",
                   v.weight_streams, T, lenT, follow, follow >= 104u ? "target takes" : "target does NOT take");
            tc_run("W unmodified", model, &L, buf, total);
            {
                const uint32_t ds[] = {1, 7, 60, 100, follow > 4 ? follow - 4u : 1u, follow + 1u, follow + 40u, follow + 200u, lenT / 2u, lenT - 4u};
                for (d = 0; d < 10; d++) {
                    char what[96]; size_t t2;
                    memcpy(work, buf, total);
                    if (!(t2 = tail_delete(work, total, &L, T, ds[d]))) continue;
                    sprintf(what, "W last-blocks %u: %u tail bytes of stream %u deleted", lastblocks[lb], ds[d], T);
                    tc_run(what, model, &L, work, t2);
                }
            }
        }
    }
    /* S: scales of matrix 2 are one rare value (expensive), weights follow the most probable symbol (cheap), so the
     * last scale stream is longer than the whole weight payload behind it */
    {
        uint32_t best_srel = 0, sym, lanes, rare = 0, rare_f = 99999, srel, j;
        double best_cost = 1e30; unsigned char blk[7][16];
        for (srel = 0; srel < 7u; srel++) {
            uint32_t previous = 8, maximum = 8; double cost = 0;
            memset(blk[srel], 0, 16);
            for (j = 0; j < 32u; j++) {
                const unsigned char *cdf = wex_nr2_weight_cdf(model, 2, wex_nr2_context(srel, previous, maximum));
                uint32_t bs = 0, bf = 0;
                for (sym = 0; sym < 16u; sym++) {
                    uint32_t f = wex_nr_u16(cdf + 2u * sym + 2u) - wex_nr_u16(cdf + 2u * sym);
                    if (sym != 8u && f > bf) { bf = f; bs = sym; }
                }
                cost += 14.0 - __builtin_log2((double)bf);
                blk[srel][j / 2u] |= (unsigned char)(bs << (4u * (j & 1u)));
                previous = bs & 7u; if (maximum == 8u || previous > maximum) maximum = previous;
            }
            if (cost < best_cost) { best_cost = cost; best_srel = srel; }
        }
        for (sym = 4; sym < 250u; sym++) {
            const unsigned char *cdf = wex_nr2_scale_cdf(model, 2) + 2u * sym;
            uint32_t f = wex_nr_u16(cdf + 2u) - wex_nr_u16(cdf);
            if (f < rare_f) { rare_f = f; rare = sym; }
        }
        printf("S: cheapest weight block: srel %u, %.2f bits per 32 symbols; rarest scale value %u (frequency %u of 16384)\n", best_srel, best_cost, rare, rare_f);
        for (lanes = 8; lanes >= 1u; lanes /= 2u) {
            const uint32_t n2 = 256u * lanes;
            L.scale_count[0] = 1; L.scale_count[1] = 1; L.scale_count[2] = n2;
            for (m = 0; m < 3u; m++) {
                L.symbol_count[m] = L.scale_count[m] * 32u;
                cs[m] = (unsigned char *)malloc(L.scale_count[m]); cw[m] = (unsigned char *)malloc(L.scale_count[m] * 16u);
                memset(cs[m], (int)rare, L.scale_count[m]);
                for (i = 0; i < L.scale_count[m]; i++) memcpy(cw[m] + (size_t)i * 16u, blk[best_srel], 16);
                anchor[m] = (unsigned char)(rare + 3u - best_srel);
            }
            if (wex_nrx_encode(model, &L, (const unsigned char *const *)cs, (const unsigned char *const *)cw, 16384u, 256u, anchor,
                               WEX_NR3_MAGIC, WEX_NR3_FLAG_CZ, 9u, buf, cap - 4096u, &written)) return 1;
            total = (written + 4095u) & ~(size_t)4095u; memset(buf + written, 0, total - written);
            if (wex_nr3_parse(buf, total, &L, model->id, &v)) return 1;
            T = v.scale_streams - 1u; lenT = wex_nr_u16(v.lengths + 2u * T); follow = v.weight_payload_bytes;
            printf("S: %u equal scale streams in matrix 2, last one (stream %u) has %u B, weight payload behind it %u B (%s)\n", lanes, T, lenT, follow,
                   follow >= 104u && lenT > follow + 8u ? "usable: fast path and deletable tail > payload" :
                   follow < 104u ? "target does NOT take the fast path" : "tail not longer than the payload: no overrun possible");
            tc_run("S unmodified", model, &L, buf, total);
            {
                const uint32_t ds[] = {1, 30, 100, follow + 1u, follow + 60u, lenT - 4u};
                for (d = 0; d < 6; d++) {
                    char what[96]; size_t t2;
                    memcpy(work, buf, total);
                    if (ds[d] + 4u > lenT || !(t2 = tail_delete(work, total, &L, T, ds[d]))) continue;
                    sprintf(what, "S x%u: %u tail bytes of scale stream %u deleted", lanes, ds[d], T);
                    tc_run(what, model, &L, work, t2);
                }
            }
        }
    }
    printf("tailcut: %ld cases (%ld accepted by the reference), %ld status mismatches, no access violation\n", tc_cases, tc_refacc, tc_mism);
    return tc_mism != 0;
}
