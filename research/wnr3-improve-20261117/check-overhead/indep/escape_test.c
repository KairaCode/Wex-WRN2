/* Escape-path round trip (never exercised by the lane's TEST run: 0 escapes). Includes the lane's ovh_proto.c
 * unchanged; injects out-of-window scales into one record's coded domain and checks that every scale-mode
 * profile decodes the coded domain back exactly (before the ceil inverse, which would reject edited data). */
#define main ovh_main
#include "../ovh_proto.c"
#undef main
int main(int argc, char **argv) {
    Container K; unsigned char *tblob; WexNr2Model tmodel; STables st; Expert X; unsigned c, m, g;
    const char *names[] = {"s_rel16", "s_rec16", "s_mean8", "inl_rel16", "inl_rec16", "inl_mean8", "B_A_rec16", "C_A_inl_rec16", "E_C_tail5_dig", "F_r16k16_inl_rec16"};
    int i, fails = 0;
    for (i = 0; i < 16; i++) { flat16[i * 2] = (unsigned char)(i * 1024); flat16[i * 2 + 1] = (unsigned char)((i * 1024) >> 8); }
    flat16[32] = 0; flat16[33] = 0x40;
    open_recs("/tmp/wnr3-data/recs", &K); load_train("..", &tblob, &tmodel, &st);
    if (load_expert(&K, "/tmp/wnr3-data/recs", argc > 1 ? argv[1] : "10:153", &X)) return 1;
    /* inject escapes: values far outside anchor-7..anchor+7, at stream/block boundaries and inside */
    X.sc[0][0] = (unsigned char)(X.anchor[0] + 8); X.sc[0][5] = (unsigned char)(X.anchor[0] + 40); X.sc[0][511] = 0; X.sc[0][512] = 255;
    X.sc[1][63] = (unsigned char)(X.anchor[1] - 8); X.sc[2][X.L.scale_count[2] - 1] = 1;
    for (i = 0; i < 6; i++) X.sc[1][1000 + i] = (unsigned char)(X.anchor[1] + 7 + i);   /* run crossing the window edge */
    for (c = 0; c < sizeof(names) / sizeof(names[0]); c++) {
        const Cfg *cf = NULL; Ctx x; Sizes sz; Parsed P; unsigned char *buf = malloc((size_t)X.raw * 2u), *back = calloc((size_t)X.raw, 1), *bs[3], *bw[3];
        uint32_t h, dig = 0; int ok = 1;
        for (g = 0; g < NCFG; g++) if (!strcmp(cfgs[g].name, names[c])) cf = &cfgs[g];
        ctx_init(&x, &tmodel, &st, &X);
        v_encode(cf, &x, X.sc, X.w, buf, (size_t)X.raw * 2u, &sz, NULL);
        for (m = 0; m < 3u; m++) { bs[m] = back + X.e.scale_offsets[m]; bw[m] = back + X.e.weight_offsets[m]; }
        if (v_parse(cf, &tmodel, &st, &X.L, X.C, buf, sz.used, X.owner, &P)) ok = 0;
        for (g = 0; ok && g < P.geo.n; g++) { if (v_decode_stream(cf, &P, buf, g, bs, bw, &h)) { ok = 0; printf("  stream %u fails\n", g); } dig ^= h; }
        if (ok && cf->digest && dig != P.digest) ok = 0;
        if (ok && memcmp(back, X.coded, (size_t)X.raw)) ok = 0;
        printf("%-20s used %llu  escape round trip %s\n", names[c], (unsigned long long)sz.used, ok ? "OK" : "FAIL");
        fails += !ok; parsed_free(&P); free(buf); free(back);
    }
    return fails;
}
