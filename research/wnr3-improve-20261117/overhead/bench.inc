/* Decoder cost (included by ovh_proto.c): single thread, this file's reference-style decoder (binary-search
 * symbol lookup like wex_nr2_decode; not the fast decoder), interleaved A/B over the given profiles:
 * every round decodes each profile's record once, in rotating order; reports the median and min..max per profile.
 *   ovh_proto bench RECS OUTDIR ROUNDS L:E   */
static double now_s(void) { return omp_get_wtime(); }
static int do_bench(int argc, char **argv) {
    static const char *const prof[] = {"wnr3_train", "A_lenc_sd", "C_A_inl_rec16", "E_C_tail5_dig", "F_r16k16_inl_rec16"};
    enum { NP = sizeof(prof) / sizeof(prof[0]) };
    Container K; unsigned char *tblob; WexNr2Model tmodel; STables st; Expert X;
    int rounds = atoi(argv[4]), r, p, q;
    const Cfg *cf[NP]; unsigned char *rec[NP]; Sizes sz[NP]; double t[NP][64];
    unsigned char *back, *bs[3], *bw[3]; uint32_t m; size_t cap;
    open_recs(argv[2], &K); load_train(argv[3], &tblob, &tmodel, &st);
    if (load_expert(&K, argv[2], argv[5], &X)) return 1;
    cap = (size_t)X.raw * 2u; back = malloc((size_t)X.raw);
    for (m = 0; m < 3u; m++) { bs[m] = back + X.e.scale_offsets[m]; bw[m] = back + X.e.weight_offsets[m]; }
    for (p = 0; p < NP; p++) {
        Ctx x; unsigned c;
        for (c = 0; c < NCFG; c++) if (!strcmp(cfgs[c].name, prof[p])) cf[p] = &cfgs[c];
        ctx_init(&x, &tmodel, &st, &X); rec[p] = malloc(cap);
        v_encode(cf[p], &x, X.sc, X.w, rec[p], cap, &sz[p], NULL);
    }
    for (r = 0; r < rounds && r < 64; r++)
        for (q = 0; q < NP; q++) {
            double t0;
            p = (q + r) % NP;
            t0 = now_s();
            if (v_decode(cf[p], &tmodel, &st, &X.L, X.C, rec[p], (size_t)sz[p].used, X.owner, bs, bw) || memcmp(back, X.slab, (size_t)X.raw)) { fprintf(stderr, "bench decode fail\n"); return 1; }
            t[p][r] = now_s() - t0;
        }
    printf("{\"record\": \"%s\", \"rounds\": %d, \"decoder\": \"ovh_proto reference-style (binary search), 1 thread\", \"profiles\": {", argv[5], rounds);
    for (p = 0; p < NP; p++) {
        double v[64], tmp; int i, j;
        for (i = 0; i < rounds; i++) v[i] = t[p][i];
        for (i = 0; i < rounds; i++) for (j = i + 1; j < rounds; j++) if (v[j] < v[i]) { tmp = v[i]; v[i] = v[j]; v[j] = tmp; }
        printf("%s\"%s\": {\"median_ms\": %.1f, \"min_ms\": %.1f, \"max_ms\": %.1f}", p ? ", " : "", prof[p], 1e3 * v[rounds / 2], 1e3 * v[0], 1e3 * v[rounds - 1]);
    }
    printf("}}\n");
    return 0;
}
