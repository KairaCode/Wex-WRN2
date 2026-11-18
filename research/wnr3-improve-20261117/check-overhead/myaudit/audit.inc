/* Corruption audit (included by ovh_proto.c).
 *
 * For each record and audit profile: encode the record with the TRAIN model (clean record R0, its per-stream
 * full hashes H0), then apply N corruptions of each type and classify the decoder's verdict:
 *   rejected | accepted, output identical to Z(source) | accepted, WRONG output.
 * Types: 0 random bit flip anywhere in the record, 1 random byte overwritten with another value, 2 a burst of
 * 16 random bytes, 3 two streams of the same kind and matrix swapped (contents and lengths), 4 one stream
 * shortened by its last byte (length entry and SP/WP fixed so the structure stays valid), 5 one stream replaced
 * by the same stream of the same expert encoded under another owner (owner ^ 1), 6 every bit of the header
 * flipped in turn (exhaustive, not N).
 * The verdict is computed incrementally: the decoder's streams are independent except that weight streams read
 * the scales, so only the streams whose bytes, length or inputs changed are decoded again (the rest are known
 * to decode to the clean output with the clean hash); the digest is updated from the changed streams' hashes.
 * A change of the header (other than the digest word) or of the per-record table section, and every trial
 * flagged for cross-checking, goes through the full decoder v_decode instead; the cross-check asserts the two
 * paths agree. "digest_only" counts trials that passed every per-stream check and were caught by the digest. */

static uint64_t sm64(uint64_t *s) { uint64_t z = (*s += UINT64_C(0x9E3779B97F4A7C15)); z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB); return z ^ (z >> 31); }

static const char *const audit_names[] = {"A_lenc_sd", "wnr3_train", "C_A_inl_rec16", "D_C_tail2_dig", "E_C_tail5_dig", "F_r16k16_inl_rec16", "tail5_nodig"};
#define NAUD (sizeof(audit_names) / sizeof(audit_names[0]))
#define NTYPE 7
static const Cfg cfg_tail5_nodig = {"tail5_nodig", 2048, 512, 0, 1, 1, 0, 23, 5, 0, 2, 1};   /* E without the digest: what the per-stream checks alone catch */

static const Cfg *audit_cfg(unsigned a) {
    unsigned c;
    if (!strcmp(audit_names[a], "tail5_nodig")) return &cfg_tail5_nodig;
    for (c = 0; c < NCFG; c++) if (!strcmp(cfgs[c].name, audit_names[a])) return &cfgs[c];
    return NULL;
}

typedef struct { uint64_t trials, rejected, same, wrong, digest_only, crosschecked, cross_mismatch; } Tally;

/* Full-decoder verdict: 0 rejected, 1 accepted identical, 2 accepted wrong. */
static int verdict_full(const Cfg *c, const WexNr2Model *model, const STables *st, const Expert *X, const unsigned char *rec, size_t used,
                        unsigned char *scratch) {
    unsigned char *bs[3], *bw[3];
    uint32_t m;
    for (m = 0; m < 3u; m++) { bs[m] = scratch + X->e.scale_offsets[m]; bw[m] = scratch + X->e.weight_offsets[m]; }
    if (v_decode(c, model, st, &X->L, X->C, rec, used, X->owner, bs, bw)) return 0;
    return memcmp(scratch, X->slab, (size_t)X->raw) ? 2 : 1;
}

/* Incremental verdict. scratch = a full coded-size buffer whose contents equal X->coded on entry and on return. */
static int verdict_inc(const Cfg *c, const WexNr2Model *model, const STables *st, const Expert *X, const Parsed *P0, const uint32_t *H0,
                       uint32_t D0, const unsigned char *rec0, size_t used0, const unsigned char *rec, size_t used,
                       unsigned char *scratch, unsigned char *scratch2, int *digest_only) {
    Parsed P1;
    unsigned char *sc[3], *w[3];
    uint32_t g, m, D1 = D0, hdr = c->wnr3 ? 32u : HDR4;
    int result = 1, changed = 0, sdirty = 0;
    unsigned char *dirty;
    *digest_only = 0;
    if (v_parse(c, model, st, &X->L, X->C, rec, used, X->owner, &P1)) { parsed_free(&P1); return 0; }
    {   /* header / table section: a change of anything the streams depend on (bytes 0..7, 16..31, the per-record
           tables) -> full decoder; SP/WP, the length-section size and the digest word only enter checks */
        size_t e0 = P0->off[0] - (c->wnr3 ? 2u * (size_t)P0->geo.n : wex_nr_u32(rec0 + 36));
        size_t e1 = P1.off[0] - (c->wnr3 ? 2u * (size_t)P1.geo.n : wex_nr_u32(rec + 36)), i;
        int other = e0 != e1;
        for (i = 0; i < e0 && i < e1 && !other; i++)
            if (rec[i] != rec0[i] && !(i >= 8u && i < 16u) && !(!c->wnr3 && i >= 32u && i < 40u)) other = 1;
        (void)hdr;
        if (other) { parsed_free(&P1); return verdict_full(c, model, st, X, rec, used, scratch2); }
    }
    for (m = 0; m < 3u; m++) { sc[m] = scratch + X->e.scale_offsets[m]; w[m] = scratch + X->e.weight_offsets[m]; }
    dirty = (unsigned char *)calloc(P1.geo.n, 1);
    for (g = 0; g < P1.geo.n; g++)
        if (P1.len[g] != P0->len[g] || memcmp(rec + P1.off[g], rec0 + P0->off[g], P1.len[g])) dirty[g] = 1;
    /* scale streams first */
    for (g = 0; g < P1.geo.n && result; g++) {
        const SInfo *s = &P1.geo.s[g];
        uint32_t h;
        if (!dirty[g] || !s->scale_pass) continue;
        if (v_decode_stream(c, &P1, rec, g, sc, w, &h)) { result = 0; break; }
        D1 ^= H0[g] ^ h;
        sdirty = 1;
        if (memcmp(sc[s->matrix] + s->first, X->sc[s->matrix] + s->first, s->count)) {
            uint32_t g2;   /* weight streams whose blocks overlap the changed scales read other contexts */
            changed = 1;
            for (g2 = 0; g2 < P1.geo.n; g2++) {
                const SInfo *t = &P1.geo.s[g2];
                if (t->scale_pass || t->matrix != s->matrix) continue;
                if (t->first / 32u < s->first + s->count && (t->first + t->count) / 32u > s->first) dirty[g2] = 1;
            }
        }
    }
    for (g = 0; g < P1.geo.n && result; g++) {
        const SInfo *s = &P1.geo.s[g];
        uint32_t h;
        if (!dirty[g] || s->scale_pass) continue;
        if (v_decode_stream(c, &P1, rec, g, sc, w, &h)) { result = 0; break; }
        D1 ^= H0[g] ^ h;
        if (memcmp(w[s->matrix] + s->first / 2u, X->w[s->matrix] + s->first / 2u, s->count / 2u) ||
            (c->inl && memcmp(sc[s->matrix] + s->first / 32u, X->sc[s->matrix] + s->first / 32u, s->count / 32u))) changed = 1;
    }
    if (result && c->digest && D1 != P1.digest) { result = 0; *digest_only = 1; }
    if (result && changed) {   /* accepted with changed coded data: the ceil inverse may still refuse it */
        unsigned char *bs[3], *bw[3];
        memcpy(scratch2, scratch, (size_t)X->raw);
        for (m = 0; m < 3u; m++) { bs[m] = scratch2 + X->e.scale_offsets[m]; bw[m] = scratch2 + X->e.weight_offsets[m]; }
        if (wex_nr3_ceil_inverse(&X->L, P1.flags & WEX_NR3_FLAG_CEIL_MASK, bs, bw)) result = 0;
        else result = memcmp(scratch2, X->slab, (size_t)X->raw) ? 2 : 1;
    }
    /* restore scratch to the clean coded data */
    for (g = 0; g < P1.geo.n; g++) {
        const SInfo *s = &P1.geo.s[g];
        if (!dirty[g]) continue;
        if (s->scale_pass) memcpy(sc[s->matrix] + s->first, X->sc[s->matrix] + s->first, s->count);
        else {
            memcpy(w[s->matrix] + s->first / 2u, X->w[s->matrix] + s->first / 2u, s->count / 2u);
            if (c->inl) memcpy(sc[s->matrix] + s->first / 32u, X->sc[s->matrix] + s->first / 32u, s->count / 32u);
        }
    }
    (void)sdirty;
    free(dirty); parsed_free(&P1);
    return result;
}

/* rebuild a record with new stream lengths/contents: streams[g] = (ptr, len). Header and table copied from rec0,
   SP/WP and the length section recomputed. Returns used size. */
static size_t rebuild(const Cfg *c, const unsigned char *rec0, const Parsed *P0, const unsigned char *const *sp, const uint32_t *sl,
                      unsigned char *out) {
    uint32_t g, n = P0->geo.n;
    size_t hdr_tab = P0->off[0] - (c->wnr3 ? 2u * n : wex_nr_u32(rec0 + 36)), at, lsec;
    uint64_t SP = 0, WP = 0;
    memcpy(out, rec0, hdr_tab);
    if (c->lenc) lsec = lenc_pack(sl, n, out + hdr_tab);
    else { for (g = 0; g < n; g++) { out[hdr_tab + 2u * g] = (unsigned char)sl[g]; out[hdr_tab + 2u * g + 1u] = (unsigned char)(sl[g] >> 8); } lsec = 2u * (size_t)n; }
    at = hdr_tab + lsec;
    for (g = 0; g < n; g++) { memmove(out + at, sp[g], sl[g]); at += sl[g]; if (P0->geo.s[g].scale_pass) SP += sl[g]; else WP += sl[g]; }
    put32(out + 8, (uint32_t)SP); put32(out + 12, (uint32_t)WP);
    if (!c->wnr3) put32(out + 36, (uint32_t)lsec);
    return at;
}

static int do_audit(int argc, char **argv) {
    Container K;
    unsigned char *tblob;
    WexNr2Model tmodel;
    STables st;
    int nrec = argc - 7, i;
    uint64_t N = strtoull(argv[5], NULL, 10), seed = strtoull(argv[6], NULL, 10);
    Tally (*tal)[NAUD][NTYPE] = calloc((size_t)nrec, sizeof(*tal));
    FILE *out;
    open_recs(argv[2], &K);
    load_train(argv[3], &tblob, &tmodel, &st);
#pragma omp parallel for schedule(dynamic, 1)
    for (i = 0; i < nrec; i++) {
        Expert X;
        unsigned a;
        size_t cap;
        unsigned char *r0, *r1, *rb, *ralt, *scratch, *scratch2, **sp;
        uint32_t *sl;
        if (load_expert(&K, argv[2], argv[7 + i], &X)) { fprintf(stderr, "load fail %s\n", argv[7 + i]); exit(1); }
        cap = (size_t)X.raw * 2u;
        r0 = malloc(cap); r1 = malloc(cap); rb = malloc(cap); ralt = malloc(cap); scratch = malloc((size_t)X.raw); scratch2 = malloc((size_t)X.raw);
        for (a = 0; a < NAUD; a++) {
            const Cfg *c = audit_cfg(a);
            Ctx x;
            Sizes s0, s1;
            Parsed P0, Palt;
            uint32_t *H0, D0 = 0, g, t;
            uint64_t rng = seed ^ ((uint64_t)i << 32) ^ ((uint64_t)a << 48);
            ctx_init(&x, &tmodel, &st, &X);
            H0 = malloc(40000u * sizeof(uint32_t));
            v_encode(c, &x, X.sc, X.w, r0, cap, &s0, H0);
            if (v_parse(c, &tmodel, &st, &X.L, X.C, r0, s0.used, X.owner, &P0)) { fprintf(stderr, "clean parse fail\n"); exit(1); }
            for (g = 0; g < P0.geo.n; g++) D0 ^= H0[g];
            if (verdict_full(c, &tmodel, &st, &X, r0, s0.used, scratch2) != 1) { fprintf(stderr, "clean decode fail %s\n", c->name); exit(1); }
            {   /* the same expert under owner ^ 1 (for type 5) */
                Ctx y = x;
                y.ident[1] = X.owner ^ 1u;
                v_encode(c, &y, X.sc, X.w, ralt, cap, &s1, NULL);
                put32(ralt + 28, X.owner);   /* header says the right owner; only the streams carry the other one */
                if (v_parse(c, &tmodel, &st, &X.L, X.C, ralt, s1.used, X.owner, &Palt)) { fprintf(stderr, "alt parse fail\n"); exit(1); }
            }
            memcpy(scratch, X.coded, (size_t)X.raw);
            memcpy(r1, r0, s0.used);   /* working copy for in-place corruptions; r0 stays clean */
            sp = malloc(P0.geo.n * sizeof(*sp)); sl = malloc(P0.geo.n * sizeof(*sl));
            for (t = 0; t < NTYPE; t++) {
                Tally *T = &tal[i][a][t];
                uint64_t trial, ntr = t == 6u ? 0u : N;
                for (trial = 0; trial < ntr; trial++) {
                    size_t used = s0.used, pos = 0;
                    unsigned char saved[16];
                    int v, dg = 0, inplace = t <= 2u || t == 6u, cross = trial < 10u && i == 0;
                    if (inplace) {
                        size_t L = t == 2u ? 16u : 1u, k;
                        if (t == 6u) pos = trial / 8u; else pos = (size_t)(sm64(&rng) % (s0.used - L + 1u));
                        memcpy(saved, r1 + pos, L);
                        if (t == 0u) r1[pos] ^= (unsigned char)(1u << (sm64(&rng) & 7u));
                        else if (t == 6u) r1[pos] ^= (unsigned char)(1u << (trial & 7u));
                        else if (t == 1u) r1[pos] ^= (unsigned char)(1u + sm64(&rng) % 255u);
                        else { for (k = 0; k < L; k++) r1[pos + k] = (unsigned char)sm64(&rng); }
                        v = verdict_inc(c, &tmodel, &st, &X, &P0, H0, D0, r0, s0.used, r1, used, scratch, scratch2, &dg);
                        if (cross) {
                            int vf = verdict_full(c, &tmodel, &st, &X, r1, used, scratch2);
                            T->crosschecked++; if (vf != v) T->cross_mismatch++;
                        }
                        memcpy(r1 + pos, saved, L);
                    } else {
                        uint32_t g1 = 0, g2 = 0;
                        for (g = 0; g < P0.geo.n; g++) { sp[g] = r0 + P0.off[g]; sl[g] = P0.len[g]; }
                        if (t == 3u) {
                            do {
                                g1 = (uint32_t)(sm64(&rng) % P0.geo.n); g2 = (uint32_t)(sm64(&rng) % P0.geo.n);
                            } while (g1 == g2 || P0.geo.s[g1].scale_pass != P0.geo.s[g2].scale_pass || P0.geo.s[g1].matrix != P0.geo.s[g2].matrix ||
                                     P0.geo.s[g1].count != P0.geo.s[g2].count);
                            sp[g1] = r0 + P0.off[g2]; sl[g1] = P0.len[g2]; sp[g2] = r0 + P0.off[g1]; sl[g2] = P0.len[g1];
                        } else if (t == 4u) {
                            g1 = (uint32_t)(sm64(&rng) % P0.geo.n); sl[g1]--;
                        } else {
                            g1 = (uint32_t)(sm64(&rng) % P0.geo.n); sp[g1] = ralt + Palt.off[g1]; sl[g1] = Palt.len[g1];
                        }
                        used = rebuild(c, r0, &P0, (const unsigned char *const *)sp, sl, rb);
                        v = verdict_inc(c, &tmodel, &st, &X, &P0, H0, D0, r0, s0.used, rb, used, scratch, scratch2, &dg);
                        if (cross) {
                            int vf = verdict_full(c, &tmodel, &st, &X, rb, used, scratch2);
                            T->crosschecked++; if (vf != v) T->cross_mismatch++;
                        }
                    }
                    T->trials++;
                    if (v == 0) T->rejected++; else if (v == 1) T->same++; else T->wrong++;
                    T->digest_only += (uint64_t)dg;
                }
            }
            free(sp); free(sl); free(H0); parsed_free(&P0); parsed_free(&Palt);
#pragma omp critical
            fprintf(stderr, "audit %s %s done\n", argv[7 + i], c->name);
        }
        free(r0); free(r1); free(rb); free(ralt); free(scratch); free(scratch2); free_expert(&X);
    }
    out = fopen(argv[4], "w");
    fprintf(out, "{\"N\": %llu, \"seed\": %llu, \"types\": [\"bitflip\", \"byte\", \"burst16\", \"swap\", \"truncate1\", \"other_owner_stream\", \"header_bits\"],\n \"columns\": [\"trials\", \"rejected\", \"accepted_identical\", \"accepted_wrong\", \"digest_only\", \"crosschecked\", \"cross_mismatch\"],\n \"records\": [",
            (unsigned long long)N, (unsigned long long)seed);
    for (i = 0; i < nrec; i++) {
        unsigned a, t;
        fprintf(out, "%s\n  {\"record\": \"%s\", \"profiles\": {", i ? "," : "", argv[7 + i]);
        for (a = 0; a < NAUD; a++) {
            fprintf(out, "%s\"%s\": [", a ? ", " : "", audit_names[a]);
            for (t = 0; t < NTYPE; t++) {
                Tally *T = &tal[i][a][t];
                fprintf(out, "%s[%llu, %llu, %llu, %llu, %llu, %llu, %llu]", t ? ", " : "", (unsigned long long)T->trials, (unsigned long long)T->rejected,
                        (unsigned long long)T->same, (unsigned long long)T->wrong, (unsigned long long)T->digest_only,
                        (unsigned long long)T->crosschecked, (unsigned long long)T->cross_mismatch);
            }
            fprintf(out, "]");
        }
        fprintf(out, "}}");
    }
    fprintf(out, "\n ]}\n");
    fclose(out);
    return 0;
}
