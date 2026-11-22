// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
/* The WNR4-I4 CUDA decoder (wnr4/wnr4_i4_cuda.cuh) through the helper API (wnr4/tools/wnr4_cuda_bridge.cu) on
 * sampled int4-g64 expert records. Needs nvcc and a GPU, run by hand:
 *   nvcc -O3 -std=c++17 -arch=sm_120 wnr4/tests/test_wnr4_i4_gpu.cu -o test_wnr4_i4_gpu
 *   test_wnr4_i4_gpu GEOMETRY planar|pairs SAMPLE_DIR [G]
 * GEOMETRY: the record's matrices as ROWSxCOLS joined by commas (Qwen3.8-Flash-Next: 640x2560,640x2560,2560x640).
 * SAMPLE_DIR: raw int4-g64 records (*.rec, each the codes of every matrix then their f32 scales, as WNR4-SPEC section
 * 6.1); those named train_* train the model (else every even file), all are coded at G (default 32, the default
 * of wex4/wnr4i4.py) and checked on the CPU. Then, through the helper: each record decoded on the GPU and compared
 * with its raw bytes (synchronous batches of up to 32, asynchronous batches of 16 with 2 in flight), and per-record
 * isolation (one payload bit flipped in each odd record: those refused, the even ones bit-exact with status 0).
 * Synthetic groups: two matrices 130x192 and 70x320, pairs, G 3 (partial tiles, partial 128-row blocks) with 3 % of
 * the scales escaped, and every scale random (all escape), G 1; again at 8 and 16 warps per block. Refused: a record
 * at G 128 (above the CUDA limit), a destination that is not 4-byte aligned, a wrong owner, an unknown model, a bad
 * geometry, a bad blob. Kernel-only GB/s (decoded bytes) for batches of 1, 8, 16, 32 records resident in VRAM, L2
 * flushed before each run, at G and at G 16 and 64. */
#include "../tools/wnr4_cuda_bridge.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define REQ(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

struct Rec { uint32_t owner; std::vector<unsigned char> raw, enc; };
struct Group { uint32_t nmat, rows[3], cols[3], pairs, G; Wnr4I4Geo g; std::vector<Rec> recs; };

static std::vector<unsigned char> slurp(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb"); REQ(f);
    std::vector<unsigned char> v;
    unsigned char buf[1 << 16];
    size_t k;
    while ((k = fread(buf, 1, sizeof(buf), f)) > 0) v.insert(v.end(), buf, buf + k);
    fclose(f);
    return v;
}
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }

/* a synthetic record as the CPU test makes them (lattice scales, escapes, nibbles leaning to 8); all_escape: every
 * scale random */
static std::vector<unsigned char> synth(const Wnr4I4Geo &g, bool all_escape) {
    std::vector<unsigned char> s((size_t)g.raw);
    std::vector<uint32_t> B(g.nbases + 1u);
    for (uint32_t i = 0; i < g.nbases; i++) B[i] = 0x3a000000u + rnd() % 0x01000000u;
    for (uint32_t m = 0; m < g.nmat; m++)
        for (uint32_t i = 0; i < g.cells[m]; i++) {
            uint32_t k = 126u - rnd() % 16u - rnd() % 16u, r = rnd() % 100u, v[64], sc;
            sc = all_escape || r < 3u ? rnd() << 16 ^ rnd() : wnr4_i4_lat(k, B[wnr4_i4_block(g.bbase[m], g.gc[m], g.nbc[m], i)]);
            wnr4_put32(s.data() + g.soff[m] + 4u * (size_t)i, sc);
            for (uint32_t e = 0; e < 64u; e++) v[e] = 8u + rnd() % 5u - 2u;
            for (uint32_t e = 0; e < 32u; e++)
                s[g.coff[m] + 32u * (size_t)i + e] = (unsigned char)(g.pairs ? v[2 * e] | v[2 * e + 1] << 4 : v[e] | v[e + 32] << 4);
        }
    return s;
}
static void encode_all(Group &gr, const Wnr4I4Model &model) {
    const size_t cap = (size_t)gr.g.raw * 4u + 65536u;
    std::vector<unsigned char> out((size_t)gr.g.raw);
    double st = 0, rw = 0;
    for (auto &r : gr.recs) {
        r.enc.resize(cap);
        size_t n = wnr4_i4_encode(&model, &gr.g, r.raw.data(), r.owner, gr.G, r.enc.data(), cap);
        REQ(n);
        r.enc.resize(n);
        REQ(wnr4_i4_decode(&model, &gr.g, r.enc.data(), n, r.owner, out.data(), 8) == 0 && out == r.raw);
        st += (double)n; rw += (double)gr.g.raw;
    }
    printf("group of %zu records, G %u: CPU encode + decode OK, stored/raw %.5f\n", gr.recs.size(), gr.G, st / rw);
}

/* every record of the group through the helper: sync, async, one bit flipped in each odd record */
static void gpu_check(Group &gr, uint32_t id, const char *name) {
    const uint32_t n = (uint32_t)gr.recs.size();
    const size_t raw = (size_t)gr.g.raw, slab = (raw + 255u) & ~(size_t)255u;
    size_t all = 0;
    std::vector<size_t> at(n), total(n);
    std::vector<uint32_t> owner(n);
    for (uint32_t i = 0; i < n; i++) { at[i] = all; total[i] = gr.recs[i].enc.size(); owner[i] = gr.recs[i].owner; all += total[i]; }
    unsigned char *host, *dst;
    CK(cudaMallocHost(&host, all));
    CK(cudaMalloc(&dst, slab * n));
    std::vector<const unsigned char *> rp(n);
    std::vector<unsigned char *> dp(n);
    for (uint32_t i = 0; i < n; i++) { memcpy(host + at[i], gr.recs[i].enc.data(), total[i]); rp[i] = host + at[i]; dp[i] = dst + slab * i; }
    std::vector<int> st(n, 7);
    std::vector<unsigned char> back(raw);
    auto exact = [&](uint32_t i) {
        CK(cudaMemcpy(back.data(), dp[i], raw, cudaMemcpyDeviceToHost));
        return !memcmp(back.data(), gr.recs[i].raw.data(), raw);
    };
    auto reset = [&]() { CK(cudaMemset(dst, 0x5A, slab * n)); CK(cudaDeviceSynchronize()); std::fill(st.begin(), st.end(), 7); };
    reset();
    for (uint32_t b0 = 0; b0 < n; b0 += 32) {
        int cnt = (int)std::min(32u, n - b0);
        REQ(wnr4_cuda_i4_decode_batch(cnt, gr.nmat, gr.rows, gr.cols, gr.pairs, rp.data() + b0, total.data() + b0,
                                      owner.data() + b0, id, 0, dp.data() + b0, st.data() + b0) == 0);
    }
    uint32_t good = 0;
    for (uint32_t i = 0; i < n; i++) good += st[i] == 0 && exact(i);
    printf("%s: %u records, G %u, %zu B each, sync batches of 32: bit-exact %u/%u\n", name, n, gr.G, raw, good, n);
    REQ(good == n);
    reset();
    int prev = -1, bad = 0;
    uint32_t prev0 = 0;
    for (uint32_t b0 = 0; b0 < n; b0 += 16) {
        int cnt = (int)std::min(16u, n - b0);
        int t = wnr4_cuda_i4_batch_submit(cnt, gr.nmat, gr.rows, gr.cols, gr.pairs, rp.data() + b0, total.data() + b0,
                                          owner.data() + b0, id, 0, dp.data() + b0, nullptr);
        REQ(t >= 0);
        if (prev >= 0) { int w = wnr4_cuda_batch_wait(prev, st.data() + prev0); REQ(w >= 0); bad += w; }
        prev = t; prev0 = b0;
    }
    { int w = wnr4_cuda_batch_wait(prev, st.data() + prev0); REQ(w >= 0); bad += w; }
    good = 0;
    for (uint32_t i = 0; i < n; i++) good += st[i] == 0 && exact(i);
    printf("  async batches of 16 (2 in flight): failures %d, bit-exact with status 0 %u/%u\n", bad, good, n);
    REQ(bad == 0 && good == n);
    reset();
    std::vector<size_t> pos(n);
    for (uint32_t i = 1; i < n; i += 2) {
        Wnr4I4Record r;
        REQ(!wnr4_i4_parse(rp[i], total[i], &gr.g, &r));
        pos[i] = at[i] + (size_t)(r.data - rp[i]) + (size_t)((i * 2654435761u + 12345u) % r.payload);
        host[pos[i]] ^= (unsigned char)(1u << (i % 8u));
    }
    for (uint32_t b0 = 0; b0 < n; b0 += 32) {
        int cnt = (int)std::min(32u, n - b0);
        REQ(wnr4_cuda_i4_decode_batch(cnt, gr.nmat, gr.rows, gr.cols, gr.pairs, rp.data() + b0, total.data() + b0,
                                      owner.data() + b0, id, 0, dp.data() + b0, st.data() + b0) == cnt / 2);
    }
    uint32_t refused = 0, kept = 0;
    for (uint32_t i = 0; i < n; i++)
        if (i & 1u) { refused += st[i] == -1; host[pos[i]] ^= (unsigned char)(1u << (i % 8u)); }
        else kept += st[i] == 0 && exact(i);
    printf("  one payload bit flipped in each odd record: refused %u/%u, the even ones bit-exact with status 0 %u/%u\n",
           refused, n / 2, kept, n - n / 2);
    REQ(refused == n / 2 && kept == n - n / 2);
    CK(cudaFreeHost(host)); CK(cudaFree(dst));
}

/* kernel-only time of batches of b records already in VRAM, L2 flushed before each run */
static void bench(Group &gr, const Wnr4I4Model &model) {
    const uint32_t n = (uint32_t)std::min<size_t>(gr.recs.size(), 32u);
    const size_t raw = (size_t)gr.g.raw, slab = (raw + 255u) & ~(size_t)255u;
    size_t all = 0, stored = 0;
    std::vector<size_t> at(n);
    for (uint32_t i = 0; i < n; i++) { at[i] = all; all += (gr.recs[i].enc.size() + 255u) & ~(size_t)255u; stored += gr.recs[i].enc.size(); }
    unsigned char *drec, *dst, *dtab, *flush;
    uint32_t *derr;
    int l2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t flush_bytes = 2u * (size_t)l2 + (1u << 20);
    CK(cudaMalloc(&drec, all)); CK(cudaMalloc(&dst, slab * n)); CK(cudaMalloc(&derr, 4u * n)); CK(cudaMalloc(&flush, flush_bytes));
    for (uint32_t i = 0; i < n; i++) CK(cudaMemcpy(drec + at[i], gr.recs[i].enc.data(), gr.recs[i].enc.size(), cudaMemcpyHostToDevice));
    std::vector<unsigned char> img(wnr4_i4_cuda_table_bytes());
    wnr4_i4_cuda_table_image(&model, img.data());
    CK(cudaMalloc(&dtab, img.size())); CK(cudaMemcpy(dtab, img.data(), img.size(), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("kernel only, records resident in VRAM, L2 (%d MiB) flushed before each run (G %u, %zu B decoded per record,"
           " stored/raw %.5f, median of 20):\n", l2 >> 20, gr.G, raw, (double)stored / ((double)raw * n));
    for (uint32_t b : {1u, 8u, 16u, 32u}) {
        if (b > n) break;
        std::vector<Wnr4I4CudaJob> by_m[3], jobs;
        for (uint32_t i = 0; i < b; i++)
            REQ(!wnr4_i4_cuda_plan(&gr.g, &model, gr.recs[i].enc.data(), gr.recs[i].enc.size(), gr.recs[i].owner, drec + at[i], i,
                                   dst + slab * i, by_m));
        const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
        const uint32_t W = wnr4_i4_cuda_jobs(by_m, jobs);
        Wnr4I4CudaJob *djob;
        CK(cudaMalloc(&djob, sizeof(Wnr4I4CudaJob) * jobs.size()));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4I4CudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(derr, 0, 4u * b)); CK(cudaMemset(dst, 0x5A, slab * b));
        std::vector<float> td;
        for (int it = 0; it < 21; it++) {
            CK(cudaMemsetAsync(flush, it, flush_bytes, 0));
            CK(cudaEventRecord(e0, 0));
            CK(wnr4_i4_cuda_launch(dtab, model.id, djob, (uint32_t)jobs.size(), W, derr, 0));
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
            float a;
            CK(cudaEventElapsedTime(&a, e0, e1));
            if (it) td.push_back(a);   /* the first run warms up */
        }
        std::sort(td.begin(), td.end());
        std::vector<uint32_t> err(b);
        CK(cudaMemcpy(err.data(), derr, 4u * b, cudaMemcpyDeviceToHost));
        std::vector<unsigned char> back(raw);
        uint32_t good = 0;
        for (uint32_t i = 0; i < b; i++) {
            CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost));
            good += !err[i] && !memcmp(back.data(), gr.recs[i].raw.data(), raw);
        }
        const double ms = td[td.size() / 2];
        printf("  batch %2u: %4zu tiles, %2u warps/block: %.3f ms = %6.1f GB/s of decoded records; bit-exact %u/%u\n",
               b, tiles, W, ms, (double)raw * b / ms / 1e6, good, b);
        REQ(good == b);
        CK(cudaFree(djob));
    }
    CK(cudaFree(drec)); CK(cudaFree(dst)); CK(cudaFree(derr)); CK(cudaFree(dtab)); CK(cudaFree(flush));
}

static Group regroup(const Group &gr, const Wnr4I4Model &model, uint32_t G) {
    Group o = gr;
    o.G = G;
    o.recs.resize(std::min<size_t>(gr.recs.size(), 32u));
    encode_all(o, model);
    return o;
}

static void refusals(Group &gr, uint32_t id, const Wnr4I4Model &model) {
    Rec &r = gr.recs[0];
    const size_t raw = (size_t)gr.g.raw;
    unsigned char *host, *dst;
    std::vector<unsigned char> big((size_t)gr.g.raw * 4u + 65536u);
    size_t n128 = wnr4_i4_encode(&model, &gr.g, r.raw.data(), r.owner, 128u, big.data(), big.size());
    REQ(n128);
    const size_t all = std::max(r.enc.size(), n128);
    CK(cudaMallocHost(&host, all)); CK(cudaMalloc(&dst, raw + 256u));
    int st = 7;
    const unsigned char *rp = host;
    unsigned char *dp = dst;
    size_t tot = n128;
    memcpy(host, big.data(), n128);
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, gr.cols, gr.pairs, &rp, &tot, &r.owner, id, 0, &dp, &st) == 1 && st == -1);
    memcpy(host, r.enc.data(), r.enc.size()); tot = r.enc.size();
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, gr.cols, gr.pairs, &rp, &tot, &r.owner, id, 0, &dp, &st) == 0 && st == 0);
    unsigned char *odd = dst + 2;
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, gr.cols, gr.pairs, &rp, &tot, &r.owner, id, 0, &odd, &st) == 1 && st == -1);
    uint32_t wrong = r.owner ^ 1u;
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, gr.cols, gr.pairs, &rp, &tot, &wrong, id, 0, &dp, &st) == 1 && st == -1);
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, gr.cols, gr.pairs, &rp, &tot, &r.owner, id ^ 1u, 0, &dp, &st) == 1 && st == 1);
    uint32_t badcols[3] = {gr.cols[0] + 1u, gr.cols[1], gr.cols[2]};
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, badcols, gr.pairs, &rp, &tot, &r.owner, id, 0, &dp, &st) == 1 && st == -1);
    REQ(wnr4_cuda_i4_decode_batch(1, gr.nmat, gr.rows, gr.cols, gr.pairs ^ 1u, &rp, &tot, &r.owner, id, 0, &dp, &st) == 1 && st == -1);
    uint32_t got = 5;
    unsigned char junk[64] = {0};
    REQ(wnr4_cuda_i4_register_model(junk, sizeof(junk), 0, &got) == -1 && got == 0);
    printf("G 128 -> -1, unaligned destination -> -1, wrong owner -> -1, unknown model -> 1, bad geometry -> -1, other"
           " packing -> -1, bad blob -> -1: OK\n");
    CK(cudaFreeHost(host)); CK(cudaFree(dst));
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: test_wnr4_i4_gpu GEOMETRY planar|pairs SAMPLE_DIR [G]\n"); return 2; }
    Group real = {};
    {
        const char *p = argv[1];
        while (*p && real.nmat < 3u) {
            char *e;
            real.rows[real.nmat] = (uint32_t)strtoul(p, &e, 10); REQ(*e == 'x');
            real.cols[real.nmat++] = (uint32_t)strtoul(e + 1, &e, 10);
            p = *e == ',' ? e + 1 : e;
        }
    }
    real.pairs = !strcmp(argv[2], "pairs");
    real.G = argc > 4 ? (uint32_t)atoi(argv[4]) : 32u;
    REQ(!wnr4_i4_geo(&real.g, real.nmat, real.rows, real.cols, real.pairs));
    std::vector<std::string> files, train;
    for (auto &de : std::filesystem::directory_iterator(argv[3])) {
        std::string f = de.path().filename().string();
        if (f.size() > 4 && f.compare(f.size() - 4, 4, ".rec") == 0) files.push_back(std::string(argv[3]) + "/" + f);
    }
    std::sort(files.begin(), files.end());
    for (size_t i = 0; i < files.size(); i++)
        if (files[i].find("/train_") != std::string::npos) train.push_back(files[i]);
    if (train.empty()) for (size_t i = 0; i < files.size(); i += 2) train.push_back(files[i]);
    std::vector<uint64_t> counts(wnr4_i4_counts_words());
    for (auto &f : train) { auto v = slurp(f); REQ(v.size() == real.g.raw); REQ(!wnr4_i4_count(&real.g, v.data(), counts.data())); }
    /* the synthetic records count too, so the model has seen escapes */
    Group A = {}, B = {};
    A.nmat = B.nmat = 2; A.rows[0] = B.rows[0] = 130; A.rows[1] = B.rows[1] = 70; A.cols[0] = B.cols[0] = 192;
    A.cols[1] = B.cols[1] = 320; A.pairs = B.pairs = 1; A.G = 3; B.G = 1;
    REQ(!wnr4_i4_geo(&A.g, 2, A.rows, A.cols, 1)); B.g = A.g;
    for (uint32_t i = 0; i < 21u; i++) {
        Rec ra, rb;
        ra.owner = 0x00050000u + i; ra.raw = synth(A.g, false); rb.owner = 0x00060000u + i; rb.raw = synth(B.g, true);
        if (!(i & 1u)) REQ(!wnr4_i4_count(&A.g, ra.raw.data(), counts.data()));
        A.recs.push_back(ra); B.recs.push_back(rb);
    }
    std::vector<unsigned char> blob(WNR4_I4_MODEL_BYTES);
    wnr4_i4_model_build(counts.data(), blob.data());
    Wnr4I4Model model;
    REQ(!wnr4_i4_model_open(blob.data(), blob.size(), &model));
    for (size_t i = 0; i < files.size(); i++) { Rec r; r.owner = (uint32_t)i; r.raw = slurp(files[i]); real.recs.push_back(r); }
    printf("%zu records (%zu train), model %08x\n", files.size(), train.size(), model.id);
    encode_all(real, model); encode_all(A, model); encode_all(B, model);
    uint32_t id = 0;
    REQ(wnr4_cuda_abi() == 6u);
    REQ(wnr4_cuda_i4_register_model(blob.data(), blob.size(), 0, &id) == 0 && id == model.id);
    printf("model id 0x%08x registered\n", id);
    gpu_check(real, id, "real");
    for (uint32_t w : {0u, 8u, 16u}) {
        wnr4_cuda_warps = w;
        gpu_check(A, id, w == 0 ? "synthetic" : w == 8 ? "synthetic, 8 warps/block" : "synthetic, 16 warps/block");
        gpu_check(B, id, w == 0 ? "synthetic, all scales escaped" : w == 8 ? "all escaped, 8 warps/block" : "all escaped, 16 warps/block");
    }
    wnr4_cuda_warps = 0;
    refusals(real, id, model);
    bench(real, model);
    for (uint32_t G : {16u, 64u}) if (G != real.G) { Group o = regroup(real, model, G); bench(o, model); }
    wnr4_cuda_release_thread();
    wnr4_i4_model_free(&model);
    printf("ALL OK\n");
    return 0;
}
