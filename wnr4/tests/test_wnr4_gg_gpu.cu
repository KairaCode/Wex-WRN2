// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
/* The WNR4-GG CUDA decoder (wnr4/wnr4_gg_cuda.cuh) through the helper API (wnr4/tools/wnr4_cuda_bridge.cu) on
 * sampled GGUF routed-expert records. Needs nvcc and a GPU, run by hand:
 *   nvcc -O3 -std=c++17 -arch=sm_120 wnr4/tests/test_wnr4_gg_gpu.cu -o test_wnr4_gg_gpu
 *   python -I wnr4/tests/sample_gg_experts.py PART.gguf SAMPLE_DIR 0,6,12,24,36,47 0-15
 *   test_wnr4_gg_gpu SAMPLE_DIR [G]
 * SAMPLE_DIR: raw expert records (gguf2wex4's record: the expert's slices back to back) and manifest.json,
 * {"experts": [{"file": F, "layer": L, "expert": E, "segments": [{"type": ggml type, "bytes": n, ...}, ...]}, ...]}.
 * Trains a WNM4 kind-2 model on the even experts of every group with the CPU codec, encodes every record (G, default 32, the
 * container default) and checks the CPU decode. Then, through the helper, for every group: each record decoded on
 * the GPU and compared with its raw bytes (synchronous batches of up to 32, asynchronous batches of 16 with 2 in
 * flight, every wait and status checked), and per-record isolation (one payload bit flipped in each odd record:
 * those refused, the even ones bit-exact with status 0). Groups: the real records, and two synthetic ones cut from
 * them: A = Q2_0 of 1000 blocks, an odd 4097-byte verbatim slice (the coded slices after it sit at odd offsets),
 * IQ4_NL of 37 blocks, Q2_0 of 5 blocks, G 3 (partial tiles of 40, 37 and 5 blocks); B = tiles of one block, G 1.
 * The synthetic groups run again at 8 and 16 warps per block. Refused: a tile with one unused word, a wrong owner,
 * an unknown model, a bad geometry, a bad blob. Kernel-only GB/s for batches of 1, 8, 16, 32 records resident in
 * VRAM, L2 flushed before each run, at G and at G = 128. */
#include "../tools/wnr4_cuda_bridge.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define REQ(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

struct Rec { uint32_t owner, expert; std::vector<unsigned char> raw, enc; };
struct Group { uint32_t nseg, type[4], bytes[4], G; Wnr4GgGeo g; std::vector<Rec> recs; };

static std::vector<unsigned char> slurp(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb"); REQ(f);
    std::vector<unsigned char> v;
    unsigned char buf[1 << 16];
    size_t k;
    while ((k = fread(buf, 1, sizeof(buf), f)) > 0) v.insert(v.end(), buf, buf + k);
    fclose(f);
    return v;
}
/* the integer after the first "key": in [from, to) */
static long long num(const std::string &s, size_t from, size_t to, const char *key) {
    size_t k = s.find(key, from); REQ(k < to);
    size_t c = s.find(':', k); REQ(c < to);
    return strtoll(s.c_str() + c + 1, nullptr, 10);
}
static uint32_t coded_bytes(const Group &gr) {
    uint32_t b = 0;
    for (uint32_t s = 0; s < gr.nseg; s++) if (gr.g.sub[s] >= 0) b += gr.bytes[s];
    return b;
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
    /* synchronous, batches of up to 32 */
    reset();
    for (uint32_t b0 = 0; b0 < n; b0 += 32) {
        int cnt = (int)std::min(32u, n - b0);
        REQ(wnr4_cuda_gg_decode_batch(cnt, gr.nseg, gr.type, gr.bytes, rp.data() + b0, total.data() + b0, owner.data() + b0,
                                      id, 0, dp.data() + b0, st.data() + b0) == 0);
    }
    uint32_t good = 0;
    for (uint32_t i = 0; i < n; i++) good += st[i] == 0 && exact(i);
    printf("%s: %u records, G %u, %zu B each (%u coded), sync batches of 32: bit-exact %u/%u\n", name, n, gr.G, raw,
           coded_bytes(gr), good, n);
    REQ(good == n);
    /* asynchronous, batches of 16, two in flight */
    reset();
    int prev = -1, bad = 0;
    uint32_t prev0 = 0;
    for (uint32_t b0 = 0; b0 < n; b0 += 16) {
        int cnt = (int)std::min(16u, n - b0);
        int t = wnr4_cuda_gg_batch_submit(cnt, gr.nseg, gr.type, gr.bytes, rp.data() + b0, total.data() + b0, owner.data() + b0,
                                          id, 0, dp.data() + b0, nullptr);
        REQ(t >= 0);
        if (prev >= 0) { int w = wnr4_cuda_batch_wait(prev, st.data() + prev0); REQ(w >= 0); bad += w; }
        prev = t; prev0 = b0;
    }
    { int w = wnr4_cuda_batch_wait(prev, st.data() + prev0); REQ(w >= 0); bad += w; }
    good = 0;
    for (uint32_t i = 0; i < n; i++) good += st[i] == 0 && exact(i);
    printf("  async batches of 16 (2 in flight): failures %d, bit-exact with status 0 %u/%u\n", bad, good, n);
    REQ(bad == 0 && good == n);
    /* one flipped payload bit in each odd record: those refused, the others decoded as if alone */
    reset();
    std::vector<size_t> pos(n);
    for (uint32_t i = 1; i < n; i += 2) {
        Wnr4GgRecord r;
        REQ(!wnr4_gg_parse(rp[i], total[i], &gr.g, &r));
        pos[i] = at[i] + (size_t)(r.data - rp[i]) + (size_t)((i * 2654435761u + 12345u) % r.payload);
        host[pos[i]] ^= (unsigned char)(1u << (i % 8u));
    }
    for (uint32_t b0 = 0; b0 < n; b0 += 32) {   /* b0 even: cnt / 2 odd records in the batch */
        int cnt = (int)std::min(32u, n - b0);
        REQ(wnr4_cuda_gg_decode_batch(cnt, gr.nseg, gr.type, gr.bytes, rp.data() + b0, total.data() + b0, owner.data() + b0,
                                      id, 0, dp.data() + b0, st.data() + b0) == cnt / 2);
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

/* kernel-only time of batches of b records already in VRAM (decode kernel; decode + verbatim copy), L2 flushed */
static void bench(Group &gr, const Wnr4GgModel &model) {
    const uint32_t n = (uint32_t)gr.recs.size();
    const size_t raw = (size_t)gr.g.raw, slab = (raw + 255u) & ~(size_t)255u;
    size_t all = 0;
    std::vector<size_t> at(n);
    for (uint32_t i = 0; i < n; i++) { at[i] = all; all += (gr.recs[i].enc.size() + 255u) & ~(size_t)255u; }
    unsigned char *drec, *dst, *dtab, *flush;
    uint32_t *derr;
    int l2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t flush_bytes = 2u * (size_t)l2 + (1u << 20);   /* written before each run: evicts the records and dst */
    CK(cudaMalloc(&drec, all)); CK(cudaMalloc(&dst, slab * n)); CK(cudaMalloc(&derr, 4u * n)); CK(cudaMalloc(&flush, flush_bytes));
    for (uint32_t i = 0; i < n; i++) CK(cudaMemcpy(drec + at[i], gr.recs[i].enc.data(), gr.recs[i].enc.size(), cudaMemcpyHostToDevice));
    std::vector<unsigned char> img(wnr4_gg_cuda_table_bytes());
    wnr4_gg_cuda_table_image(&model, img.data());
    CK(cudaMalloc(&dtab, img.size())); CK(cudaMemcpy(dtab, img.data(), img.size(), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1, e2;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
    printf("kernel only, records resident in VRAM, L2 (%d MiB) flushed before each run (G %u, %u coded + %zu verbatim B per record, median of 20):\n",
           l2 >> 20, gr.G, coded_bytes(gr), raw - coded_bytes(gr));
    for (uint32_t b : {1u, 8u, 16u, 32u}) {
        if (b > n) break;
        std::vector<Wnr4GgCudaJob> by_sub[2], jobs;
        std::vector<Wnr4GgCudaCopy> copies;
        size_t max_copy = 0;
        for (uint32_t i = 0; i < b; i++)
            REQ(!wnr4_gg_cuda_plan(&gr.g, &model, gr.recs[i].enc.data(), gr.recs[i].enc.size(), gr.recs[i].owner, drec + at[i], i,
                                   dst + slab * i, by_sub, copies));
        const size_t tiles = by_sub[0].size() + by_sub[1].size();
        const uint32_t W = wnr4_gg_cuda_jobs(by_sub, jobs);
        for (auto &c : copies) max_copy = std::max(max_copy, c.bytes);
        Wnr4GgCudaJob *djob; Wnr4GgCudaCopy *dcp;
        CK(cudaMalloc(&djob, sizeof(Wnr4GgCudaJob) * jobs.size())); CK(cudaMalloc(&dcp, sizeof(Wnr4GgCudaCopy) * (copies.size() + 1u)));
        CK(cudaMemcpy(djob, jobs.data(), sizeof(Wnr4GgCudaJob) * jobs.size(), cudaMemcpyHostToDevice));
        if (!copies.empty()) CK(cudaMemcpy(dcp, copies.data(), sizeof(Wnr4GgCudaCopy) * copies.size(), cudaMemcpyHostToDevice));
        CK(cudaMemset(derr, 0, 4u * b)); CK(cudaMemset(dst, 0x5A, slab * b));
        std::vector<float> td, tt;
        for (int it = 0; it < 21; it++) {
            CK(cudaMemsetAsync(flush, it, flush_bytes, 0));
            CK(cudaEventRecord(e0, 0));
            CK(wnr4_gg_cuda_launch(dtab, model.id, djob, (uint32_t)jobs.size(), W, derr, 0));
            CK(cudaEventRecord(e1, 0));
            CK(wnr4_gg_cuda_copy(dcp, (uint32_t)copies.size(), max_copy, 0));
            CK(cudaEventRecord(e2, 0)); CK(cudaEventSynchronize(e2));
            float a, c;
            CK(cudaEventElapsedTime(&a, e0, e1)); CK(cudaEventElapsedTime(&c, e0, e2));
            if (it) { td.push_back(a); tt.push_back(c); }   /* the first run warms up */
        }
        std::sort(td.begin(), td.end()); std::sort(tt.begin(), tt.end());
        std::vector<uint32_t> err(b);
        CK(cudaMemcpy(err.data(), derr, 4u * b, cudaMemcpyDeviceToHost));
        std::vector<unsigned char> back(raw);
        uint32_t good = 0;
        for (uint32_t i = 0; i < b; i++) {
            CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost));
            good += !err[i] && !memcmp(back.data(), gr.recs[i].raw.data(), raw);
        }
        const double ms_d = td[td.size() / 2], ms_t = tt[tt.size() / 2];
        printf("  batch %2u: %4zu tiles, %2u warps/block: decode %.3f ms = %6.1f GB/s of coded slices; with the verbatim copy %.3f ms = %6.1f GB/s of records; bit-exact %u/%u\n",
               b, tiles, W, ms_d, (double)coded_bytes(gr) * b / ms_d / 1e6, ms_t, (double)raw * b / ms_t / 1e6, good, b);
        REQ(good == b);
        CK(cudaFree(djob)); CK(cudaFree(dcp));
    }
    CK(cudaFree(drec)); CK(cudaFree(dst)); CK(cudaFree(derr)); CK(cudaFree(dtab)); CK(cudaFree(flush));
}

/* the group's first 32 records encoded again at another G (the bench at another tile size) */
static Group regroup(const Group &gr, const Wnr4GgModel &model, uint32_t G) {
    Group o = {};
    o.nseg = gr.nseg; o.G = G; o.g = gr.g;
    memcpy(o.type, gr.type, sizeof(o.type)); memcpy(o.bytes, gr.bytes, sizeof(o.bytes));
    for (size_t i = 0; i < gr.recs.size() && i < 32u; i++) {
        Rec r;
        r.owner = gr.recs[i].owner; r.expert = gr.recs[i].expert; r.raw = gr.recs[i].raw;
        r.enc.resize(r.raw.size() * 2u + (1u << 20));
        const size_t m = wnr4_gg_encode(&model, &o.g, r.raw.data(), r.owner, G, r.enc.data(), r.enc.size());
        REQ(m);
        r.enc.resize(m);
        o.recs.push_back(std::move(r));
    }
    return o;
}

/* records cut from the first k of src: coded slices from consecutive blocks of its Q2_0 slice, verbatim ones from
 * its byte 7 on */
static Group synth(const Group &src, uint32_t nseg, const uint32_t *type, const uint32_t *bytes, uint32_t G, size_t k) {
    Group o = {};
    o.nseg = nseg; o.G = G;
    memcpy(o.type, type, 4u * nseg); memcpy(o.bytes, bytes, 4u * nseg);
    REQ(!wnr4_gg_geo(&o.g, nseg, type, bytes));
    uint32_t cs = 0;
    while (cs < src.nseg && src.type[cs] != WNR4_GG_Q2_0) cs++;
    REQ(cs < src.nseg);
    for (size_t i = 0; i < src.recs.size() && i < k; i++) {
        const unsigned char *b = src.recs[i].raw.data();
        Rec r;
        r.owner = src.recs[i].owner; r.expert = src.recs[i].expert;
        size_t cur = src.g.off[cs];
        for (uint32_t s = 0; s < nseg; s++) {
            const size_t from = o.g.sub[s] >= 0 ? cur : 7u;
            REQ(o.g.sub[s] < 0 ? from + bytes[s] <= src.g.raw : from + bytes[s] <= (size_t)src.g.off[cs] + src.bytes[cs]);
            r.raw.insert(r.raw.end(), b + from, b + from + bytes[s]);
            if (o.g.sub[s] >= 0) cur += bytes[s];
        }
        o.recs.push_back(std::move(r));
    }
    return o;
}

/* enc with 2 zero bytes after the words of tile t - 1 (a word it never reads), tile t on and the payload moved by 2:
 * still a well-formed record, refused only by the exact word count of tile t - 1 */
static std::vector<unsigned char> widen(const std::vector<unsigned char> &enc, const Wnr4GgGeo &g, uint32_t t) {
    Wnr4GgRecord r;
    REQ(!wnr4_gg_parse(enc.data(), enc.size(), &g, &r) && t >= 1u && t < r.ntiles);
    const size_t data_at = (size_t)(r.data - enc.data()), cut = data_at + wnr4_u32(r.offsets + 4u * t), end = data_at + r.payload;
    std::vector<unsigned char> o(enc.begin(), enc.begin() + cut);
    o.push_back(0); o.push_back(0);
    o.insert(o.end(), enc.begin() + cut, enc.begin() + end);
    o.resize((end + 2u + 4095u) & ~(size_t)4095u, 0);
    wnr4_put32(o.data() + 24, r.payload + 2u);
    for (uint32_t k = t; k <= r.ntiles; k++) {
        unsigned char *p = o.data() + WNR4_HEADER_BYTES + 4u * k;
        wnr4_put32(p, wnr4_u32(p) + 2u);
    }
    REQ(!wnr4_gg_parse(o.data(), o.size(), &g, &r));
    return o;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_wnr4_gg_gpu SAMPLE_DIR [G]\n"); return 2; }
    const std::string dir = argv[1];
    const uint32_t G = argc > 2 ? (uint32_t)atoi(argv[2]) : 32u;
    REQ(G >= 1u && G <= WNR4_MAX_G);
    std::vector<unsigned char> mj = slurp(dir + "/manifest.json");
    const std::string man(mj.begin(), mj.end());
    std::vector<Group> groups;
    for (size_t p = man.find("\"file\""); p != std::string::npos;) {
        size_t next = man.find("\"file\"", p + 6);
        const size_t end = next == std::string::npos ? man.size() : next;
        size_t q1 = man.find('"', man.find(':', p) + 1), q2 = man.find('"', q1 + 1);
        Group key = {};
        /* the entry's "segments" array, to its matching ']' (the manifest's other sections also have "type" keys) */
        size_t sb = man.find('[', man.find("\"segments\"", p)), se = sb;
        REQ(sb < end);
        for (int depth = 0; se < man.size(); se++)
            if (man[se] == '[') depth++;
            else if (man[se] == ']' && !--depth) break;
        for (size_t a = man.find("\"type\"", sb); a < se; a = man.find("\"type\"", a + 6)) {
            REQ(key.nseg < WNR4_GG_MAX_SEG);
            key.type[key.nseg] = (uint32_t)num(man, a, se, "\"type\"");
            key.bytes[key.nseg] = (uint32_t)num(man, a, se, "\"bytes\"");
            key.nseg++;
        }
        Rec r;
        r.expert = (uint32_t)num(man, p, end, "\"expert\"");
        r.owner = (uint32_t)num(man, p, end, "\"layer\"") << 16 | r.expert;
        r.raw = slurp(dir + "/" + man.substr(q1 + 1, q2 - q1 - 1));
        Group *gr = nullptr;
        for (auto &x : groups)
            if (x.nseg == key.nseg && !memcmp(x.type, key.type, 4u * key.nseg) && !memcmp(x.bytes, key.bytes, 4u * key.nseg)) gr = &x;
        if (!gr) {
            REQ(!wnr4_gg_geo(&key.g, key.nseg, key.type, key.bytes));
            key.G = G;
            groups.push_back(key); gr = &groups.back();
        }
        REQ(r.raw.size() == gr->g.raw);
        gr->recs.push_back(std::move(r));
        p = next;
    }
    REQ(!groups.empty());
    const size_t nreal = groups.size();
    {   /* 21 records each: at 4, 8 and 16 warps at least one per-subtype job list of each group needs padding */
        const uint32_t ta[4] = {WNR4_GG_Q2_0, 16u, WNR4_GG_IQ4_NL, WNR4_GG_Q2_0}, ba[4] = {18000u, 4097u, 18u * 37u, 18u * 5u};
        const uint32_t tb[3] = {WNR4_GG_Q2_0, 16u, WNR4_GG_IQ4_NL}, bb[3] = {18u, 3u, 18u};
        Group a = synth(groups[0], 4u, ta, ba, 3u, 21u), b = synth(groups[0], 3u, tb, bb, 1u, 21u);
        groups.push_back(std::move(a)); groups.push_back(std::move(b));
    }
    /* train on the even experts of every group (the synthetic ones give subtype 1, IQ4_NL, real classes),
       encode everything, CPU decode */
    std::vector<uint64_t> counts(wnr4_gg_counts_words());
    for (size_t k = 0; k < groups.size(); k++)
        for (auto &r : groups[k].recs) if (!(r.expert & 1u)) wnr4_gg_count(&groups[k].g, r.raw.data(), counts.data());
    std::vector<unsigned char> blob(WNR4_GG_MODEL_BYTES);
    REQ(wnr4_gg_model_build(counts.data(), blob.data()) == 0);
    Wnr4GgModel model;
    REQ(!wnr4_gg_model_open(blob.data(), blob.size(), &model));
    for (auto &gr : groups) {
        size_t raw_sum = 0, enc_sum = 0;
        for (auto &r : gr.recs) {
            std::vector<unsigned char> back(r.raw.size());
            r.enc.resize(r.raw.size() * 2u + (1u << 20));
            size_t m = wnr4_gg_encode(&model, &gr.g, r.raw.data(), r.owner, gr.G, r.enc.data(), r.enc.size());
            REQ(m);
            r.enc.resize(m);
            REQ(wnr4_gg_decode(&model, &gr.g, r.enc.data(), m, r.owner, back.data(), 8) == 0 && back == r.raw);
            raw_sum += (r.raw.size() + 4095u) & ~(size_t)4095u; enc_sum += m;
        }
        printf("group of %zu records, G %u: CPU encode + decode OK, stored/raw %.4f (all records, odd and even)\n",
               gr.recs.size(), gr.G, (double)enc_sum / (double)raw_sum);
    }
    /* the helper */
    REQ(wnr4_cuda_abi() == 5u);
    uint32_t id = 7;
    REQ(wnr4_cuda_gg_register_model(blob.data(), blob.size(), 0, &id) == 0 && id == model.id);
    id = 7;
    REQ(wnr4_cuda_gg_register_model(blob.data(), blob.size(), 0, &id) == 0 && id == model.id);   /* again: same id */
    blob[0] ^= 1u;
    REQ(wnr4_cuda_gg_register_model(blob.data(), blob.size(), 0, &id) == -1 && id == 0);
    blob[0] ^= 1u;
    printf("model id 0x%08x registered\n", model.id);
    for (size_t k = 0; k < groups.size(); k++) gpu_check(groups[k], model.id, k < nreal ? "real" : "synthetic");
    for (uint32_t w : {8u, 16u}) {   /* the block sizes few tiles never pick */
        wnr4_cuda_warps = w;
        for (size_t k = nreal; k < groups.size(); k++) gpu_check(groups[k], model.id, w == 8u ? "synthetic, 8 warps/block" : "synthetic, 16 warps/block");
    }
    wnr4_cuda_warps = 0;
    {   /* a tile with one unused word, wrong owner, unknown model, bad geometry, no status array */
        Group &gr = groups[0], &sa = groups[nreal];
        const std::vector<unsigned char> wide = widen(sa.recs[0].enc, sa.g, 1u);
        std::vector<unsigned char> back(sa.recs[0].raw.size());
        REQ(wnr4_gg_decode(&model, &sa.g, wide.data(), wide.size(), sa.recs[0].owner, back.data(), 1) == WNR4_ERR_STRUCT &&
            back == sa.recs[0].raw);   /* the CPU reference: every byte right, only the word count wrong */
        const size_t cap = std::max(gr.recs[0].enc.size(), wide.size());
        unsigned char *host, *dst;
        CK(cudaMallocHost(&host, cap)); CK(cudaMalloc(&dst, std::max((size_t)gr.g.raw, (size_t)sa.g.raw)));
        const unsigned char *rp = host;
        size_t total = wide.size();
        memcpy(host, wide.data(), wide.size());
        int st = 7;
        REQ(wnr4_cuda_gg_decode_batch(1, sa.nseg, sa.type, sa.bytes, &rp, &total, &sa.recs[0].owner, model.id, 0, &dst, &st) == 1 && st == -1);
        memcpy(host, gr.recs[0].enc.data(), gr.recs[0].enc.size());
        total = gr.recs[0].enc.size();
        uint32_t wrong = gr.recs[0].owner ^ 1u, zero[WNR4_GG_MAX_SEG];
        memcpy(zero, gr.bytes, sizeof(zero));
        zero[0] = 0;   /* an empty slice */
        REQ(wnr4_cuda_gg_decode_batch(1, gr.nseg, gr.type, gr.bytes, &rp, &total, &wrong, model.id, 0, &dst, &st) == 1 && st == -1);
        REQ(wnr4_cuda_gg_decode_batch(1, gr.nseg, gr.type, gr.bytes, &rp, &total, &gr.recs[0].owner, model.id ^ 1u, 0, &dst, &st) == 1 &&
            st == 1);
        REQ(wnr4_cuda_gg_decode_batch(1, 0, gr.type, gr.bytes, &rp, &total, &gr.recs[0].owner, model.id, 0, &dst, &st) == 1 && st == -1);
        REQ(wnr4_cuda_gg_decode_batch(1, gr.nseg, gr.type, zero, &rp, &total, &gr.recs[0].owner, model.id, 0, &dst, &st) == 1 && st == -1);
        REQ(wnr4_cuda_gg_decode_batch(1, gr.nseg, gr.type, gr.bytes, &rp, &total, &gr.recs[0].owner, model.id, 0, &dst, nullptr) == 1);
        REQ(wnr4_cuda_gg_decode_batch(1, gr.nseg, gr.type, gr.bytes, &rp, &total, &gr.recs[0].owner, model.id, 0, &dst, &st) == 0 && st == 0);
        CK(cudaFreeHost(host)); CK(cudaFree(dst));
        printf("one unused word in a tile -> -1, wrong owner -> -1, unknown model -> 1, bad geometry -> -1, no status array -> n: OK\n");
    }
    size_t big = 0;
    for (size_t k = 0; k < nreal; k++) if (groups[k].recs.size() > groups[big].recs.size()) big = k;
    bench(groups[big], model);
    if (G != 128u) { Group g128 = regroup(groups[big], model, 128u); bench(g128, model); }   /* the speed at G 128, for the table */
    wnr4_cuda_release_thread();
    wnr4_gg_model_free(&model);
    printf("all GG checks passed\n");
    return 0;
}
