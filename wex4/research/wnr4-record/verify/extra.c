/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Verification extras on top of the UNMODIFIED ../wnr4_proto.c (its main is renamed):
 * single-thread timings (reference decode, host offsets table) and a small corruption test
 * (exhaustive header bit flips, random flips in the length section, the payload and stream starts). */
#define main proto_main
#include "../wnr4_proto.c"
#undef main

static uint64_t rng_s = UINT64_C(0x9E3779B97F4A7C15);
static uint64_t rng(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }

int main(int argc, char **argv) {
    size_t n, model_bytes, table_bytes, rec_bytes;
    unsigned char *page, *blob, *table, *rec, *slab, *coded, *buf, *scales[3], *weights[3];
    WexHeader hdr; WexNr2Model model; WexEntry e; WexNr2Layout layout;
    int layer, expert, flags, nflip, rep;
    uint64_t raw = 0, aligned = 0;
    uint32_t stats[16], owner, m;
    char name[64];
    double t0, best;
    static const unsigned picks[] = {20, 22, 14};   /* lenc_sd, lenc_sd_tail5, tail5_hash3 */
    unsigned pi;
    if (argc < 4) return 1;
    page = slurp(argv[1], "header.bin", &n); blob = slurp(argv[1], "model.bin", &model_bytes); table = slurp(argv[1], "entries.bin", &table_bytes);
    memcpy(&hdr, page, sizeof(hdr));
    if (wex_nr2_model_open(blob, model_bytes, &model)) return 1;
    sscanf(argv[2], "%d:%d", &layer, &expert); nflip = atoi(argv[3]);
    memcpy(&e, table + ((size_t)layer * hdr.num_experts + expert) * sizeof(e), sizeof(e));
    snprintf(name, sizeof(name), "rec_%02d_%03d.bin", layer, expert);
    rec = slurp(argv[1], name, &rec_bytes);
    wex_entry_raw_sizes(&e, &raw, &aligned); wex_nr2_layout(&e, &layout);
    slab = malloc(raw); coded = malloc(raw); buf = malloc(raw * 2);
    owner = (uint32_t)layer << 16 | (uint32_t)expert;
    best = 1e9;
    for (rep = 0; rep < 3; rep++) {
        t0 = omp_get_wtime();
        if (wex_nr3_decode(blob, &e, rec, (uint32_t)layer, (uint32_t)expert, slab)) return 2;
        t0 = omp_get_wtime() - t0; if (t0 < best) best = t0;
    }
    printf("reference CPU decode, single thread, best of 3: %.3f s/record (%.1f MB/s of raw)\n", best, raw / best / 1e6);
    flags = wex_nr3_prepare(&e, slab, 1, coded, stats);
    for (m = 0; m < 3u; m++) { scales[m] = coded + e.scale_offsets[m]; weights[m] = coded + e.weight_offsets[m]; }
    {   /* host offsets table: u16 prefix sum (today) vs coded section + prefix sum */
        Sizes sz; uint32_t streams, *len, s, r; volatile uint32_t sink = 0; double a = 1e9, b = 1e9;
        if (v_encode(&cfgs[1], &model, &layout, scales, weights, (uint32_t)flags, owner, buf, raw * 2, &sz)) return 3;
        streams = sz.streams; len = malloc((streams + 1u) * sizeof(*len));
        for (rep = 0; rep < 7; rep++) {
            t0 = omp_get_wtime();
            for (r = 0; r < 5000u; r++) { uint32_t at = 0; for (s = 0; s < streams; s++) { len[s] = at; at += wex_nr_u16(rec + 32u + s * 2u); } sink += at; }
            t0 = (omp_get_wtime() - t0) / 5000.0 * 1e6; if (t0 < a) a = t0;
            t0 = omp_get_wtime();
            for (r = 0; r < 5000u; r++) { uint32_t at = 0; lenc_unpack(buf + HDR4, (size_t)sz.len_bytes, streams, len);
                for (s = 0; s < streams; s++) { uint32_t l = len[s]; len[s] = at; at += l; } sink += at; }
            t0 = (omp_get_wtime() - t0) / 5000.0 * 1e6; if (t0 < b) b = t0;
        }
        printf("host offsets table, single thread, best of 7: u16 prefix sum %.2f us, lenc unpack + prefix sum %.2f us (len section %llu B)\n",
               a, b, (unsigned long long)sz.len_bytes);
        free(len);
    }
    for (pi = 0; pi < sizeof(picks) / sizeof(picks[0]); pi++) {
        const Cfg *c = &cfgs[picks[pi]];
        Sizes sz; uint32_t streams, *len, *off, s; size_t pay0; int total, k, rejected[4] = {0, 0, 0, 0}, tried[4] = {0, 0, 0, 0}, wrong = 0;
        size_t *pos; unsigned char *bit; int *region;
        if (v_encode(c, &model, &layout, scales, weights, (uint32_t)flags, owner, buf, raw * 2, &sz)) return 3;
        streams = sz.streams; len = malloc(streams * sizeof(*len)); off = malloc((streams + 1) * sizeof(*off));
        if (c->lenc) { if (lenc_unpack(buf + HDR4, sz.len_bytes, streams, len) != sz.len_bytes) return 4; }
        else for (s = 0; s < streams; s++) len[s] = wex_nr_u16(buf + HDR4 + s * 2u);
        pay0 = HDR4 + sz.len_bytes;
        off[0] = 0; for (s = 0; s < streams; s++) off[s + 1] = off[s] + len[s];
        total = 320 + 3 * nflip;
        pos = malloc(total * sizeof(*pos)); bit = malloc(total); region = malloc(total * sizeof(*region));
        for (k = 0; k < total; k++) {
            if (k < 320) { region[k] = 0; pos[k] = k / 8; bit[k] = (unsigned char)(1u << (k % 8)); }
            else if (k < 320 + nflip) { region[k] = 1; pos[k] = HDR4 + rng() % sz.len_bytes; bit[k] = (unsigned char)(1u << (rng() % 8)); }
            else if (k < 320 + 2 * nflip) { region[k] = 2; pos[k] = pay0 + rng() % (sz.scale_bytes + sz.weight_bytes); bit[k] = (unsigned char)(1u << (rng() % 8)); }
            else { s = (uint32_t)(rng() % streams); region[k] = 3; pos[k] = pay0 + off[s] + rng() % 4; bit[k] = (unsigned char)(1u << (rng() % 8)); }
        }
#pragma omp parallel
        {
            unsigned char *mine = malloc(sz.used), *back = malloc(raw), *bs[3], *bw[3];
            uint32_t mm; int kk;
            memcpy(mine, buf, sz.used);
            for (mm = 0; mm < 3u; mm++) { bs[mm] = back + e.scale_offsets[mm]; bw[mm] = back + e.weight_offsets[mm]; }
#pragma omp single
            {   /* sanity: the untouched record decodes to the reference slab */
                if (v_decode(c, &model, &layout, mine, (size_t)sz.used, bs, bw) || memcmp(back, slab, raw)) { printf("%s: clean decode FAILED\n", c->name); exit(5); }
            }
#pragma omp for schedule(dynamic, 1)
            for (kk = 0; kk < total; kk++) {
                int r;
                mine[pos[kk]] ^= bit[kk];
                r = v_decode(c, &model, &layout, mine, (size_t)sz.used, bs, bw);
                mine[pos[kk]] ^= bit[kk];
#pragma omp critical
                { tried[region[kk]]++; if (r) rejected[region[kk]]++; else if (memcmp(back, slab, raw)) wrong++; }
            }
            free(mine); free(back);
        }
        printf("%-16s used %llu B: single-bit flips rejected: header %d/%d, length section %d/%d, payload %d/%d, stream start bytes %d/%d; accepted with wrong output: %d\n",
               c->name, (unsigned long long)sz.used, rejected[0], tried[0], rejected[1], tried[1], rejected[2], tried[2], rejected[3], tried[3], wrong);
        fflush(stdout);
        free(len); free(off); free(pos); free(bit); free(region);
    }
    return 0;
}
