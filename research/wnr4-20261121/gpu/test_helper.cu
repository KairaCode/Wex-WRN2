// SPDX-License-Identifier: Apache-2.0
/* The WNR4 helper API (wnr4/tools/wnr4_cuda_bridge.cu) on a pack: register the model, decode the whole
 * pack in batches from page-locked host memory into device slabs, compare every record with its
 * expected bytes; time per batch size (end to end: upload + plan + kernel + error copy, synchronous);
 * a corrupted record in a batch must get status -1 and the others 0.
 *   test_helper PACK */
#include "../../../wnr4/tools/wnr4_cuda_bridge.cu"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define REQ(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); REQ(f);
    char magic[4]; uint32_t ver, n, kind, G, geo[13], l4, l2;
    REQ(fread(magic, 4, 1, f) == 1 && fread(&ver, 4, 1, f) == 1 && fread(&n, 4, 1, f) == 1 && fread(&kind, 4, 1, f) == 1 && fread(&G, 4, 1, f) == 1);
    REQ(fread(geo, 4, 13, f) == 13 && fread(&l4, 4, 1, f) == 1 && fread(&l2, 4, 1, f) == 1);
    unsigned char e72[72]; REQ(fread(e72, 72, 1, f) == 1);
    std::vector<unsigned char> wnm4(l4), wnm2(l2);
    REQ(fread(wnm4.data(), 1, l4, f) == l4); if (l2) REQ(fread(wnm2.data(), 1, l2, f) == l2);
    std::vector<uint32_t> owner(n); std::vector<size_t> total(n), at(n); std::vector<std::vector<unsigned char>> want(n);
    std::vector<std::vector<unsigned char>> recs(n);
    size_t all = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t h[4]; REQ(fread(h, 4, 4, f) == 4);
        owner[i] = h[0]; recs[i].resize(h[1]); want[i].resize(h[3]);
        REQ(fread(recs[i].data(), 1, h[1], f) == h[1]);
        REQ(fseek(f, h[2], SEEK_CUR) == 0);
        REQ(i == 0 || h[3] == want[0].size());
        REQ(fread(want[i].data(), 1, h[3], f) == h[3]);
        total[i] = h[1]; at[i] = all; all += h[1];
    }
    fclose(f);
    Wnr4Geo g; REQ(!wnr4_geo(&g, geo[0], geo + 1, geo + 4, geo + 7, geo + 10));
    Wnr4Model mh; REQ(!wnr4_model_open(wnm4.data(), l4, &mh));
    REQ(wnr4_cuda_register_model(wnm4.data(), l4, 0) == 0);
    unsigned char *host; CK(cudaMallocHost(&host, all));
    std::vector<const unsigned char *> rp(n);
    for (uint32_t i = 0; i < n; i++) { memcpy(host + at[i], recs[i].data(), total[i]); rp[i] = host + at[i]; }
    const size_t raw = (size_t)g.raw, slab = (raw + 255u) & ~(size_t)255u;
    REQ(want[0].size() == raw);
    unsigned char *dst; CK(cudaMalloc(&dst, slab * n));
    std::vector<unsigned char *> ds(3 * n), dw(3 * n);
    for (uint32_t i = 0; i < n; i++) for (int m = 0; m < 3; m++) { ds[3 * i + m] = dst + slab * i + g.soff[m]; dw[3 * i + m] = dst + slab * i + g.woff[m]; }
    std::vector<int> st(n);
    printf("helper on %s: %u records, %zu B each\n", argv[1], n, raw);
    for (uint32_t b : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
        if (b > n) break;
        double best = 1e30;
        for (int it = 0; it < 4; it++) {
            CK(cudaMemset(dst, 0x5A, slab * n)); CK(cudaDeviceSynchronize());   /* the helper's streams are non-blocking */
            double t0 = now(), dt = 0;
            for (uint32_t b0 = 0; b0 < n; b0 += b) {
                int cnt = (int)std::min(b, n - b0);
                REQ(wnr4_cuda_decode_batch(cnt, geo, rp.data() + b0, total.data() + b0, owner.data() + b0, mh.id, 0,
                                           ds.data() + 3 * b0, dw.data() + 3 * b0, st.data() + b0) == 0);
            }
            dt = now() - t0; best = std::min(best, dt);
        }
        std::vector<unsigned char> back(raw); uint32_t good = 0;
        for (uint32_t i = 0; i < n; i++) { CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost)); good += !memcmp(back.data(), want[i].data(), raw); }
        printf("  batch %3u: %7.2f ms for %u records, %6.1f GB/s decoded, bit-exact %u/%u\n", b, best * 1e3, n,
               (double)raw * n / best / 1e9, good, n);
    }
    /* asynchronous: submit batch k+1 before waiting for batch k (2 in flight), helper's own stream */
    for (uint32_t b : {4u, 8u, 16u, 32u}) {
        if (b > n) break;
        double best = 1e30; uint32_t bad = 0;
        for (int it = 0; it < 4; it++) {
            CK(cudaMemset(dst, 0x5A, slab * n)); CK(cudaDeviceSynchronize());   /* check the async path's own output */
            double t0 = now();
            int prev = -1; uint32_t prev0 = 0;
            for (uint32_t b0 = 0; b0 < n; b0 += b) {
                int cnt = (int)std::min(b, n - b0);
                int t = wnr4_cuda_batch_submit(cnt, geo, rp.data() + b0, total.data() + b0, owner.data() + b0, mh.id, 0,
                                               ds.data() + 3 * b0, dw.data() + 3 * b0, nullptr);
                REQ(t >= 0);
                if (prev >= 0) bad += wnr4_cuda_batch_wait(prev, st.data() + prev0);
                prev = t; prev0 = b0;
            }
            bad += wnr4_cuda_batch_wait(prev, st.data() + prev0);
            best = std::min(best, now() - t0);
        }
        std::vector<unsigned char> back(raw); uint32_t good = 0;
        for (uint32_t i = 0; i < n; i++) { CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost)); good += !memcmp(back.data(), want[i].data(), raw); }
        printf("  async batch %3u (2 in flight): %7.2f ms for %u records, %6.1f GB/s decoded, failures %u, bit-exact %u/%u\n",
               b, best * 1e3, n, (double)raw * n / best / 1e9, bad, good, n);
    }
    /* one corrupted record in a batch of 8 */
    uint32_t k = std::min(8u, n);
    host[at[3 % k] + total[3 % k] / 2] ^= 0x10;
    int bad = wnr4_cuda_decode_batch((int)k, geo, rp.data(), total.data(), owner.data(), mh.id, 0, ds.data(), dw.data(), st.data());
    printf("corrupted record %u of a batch of %u: failures %d\n", 3 % k, k, bad);
    for (uint32_t i = 0; i < k; i++) printf("  status[%u] = %d\n", i, st[i]);
    REQ(bad == 1 && st[3 % k] == -1);
    for (uint32_t i = 0; i < k; i++) if (i != 3 % k) {
        std::vector<unsigned char> back(raw);
        REQ(st[i] == 0);
        CK(cudaMemcpy(back.data(), dst + slab * i, raw, cudaMemcpyDeviceToHost));
        REQ(!memcmp(back.data(), want[i].data(), raw));
    }
    host[at[3 % k] + total[3 % k] / 2] ^= 0x10;   /* restore */
    /* unknown model, wrong owner */
    REQ(wnr4_cuda_decode_batch(1, geo, rp.data(), total.data(), owner.data(), mh.id ^ 1u, 0, ds.data(), dw.data(), st.data()) == 1 && st[0] == 1);
    uint32_t wrong = owner[1] ^ 1u;
    REQ(wnr4_cuda_decode_batch(1, geo, rp.data() + 1, total.data() + 1, &wrong, mh.id, 0, ds.data(), dw.data(), st.data()) == 1 && st[0] == -1);
    printf("unknown model -> 1, wrong owner -> -1: OK\n");
    /* the slot ring: 4 batches in flight are accepted, a 5th before any wait is refused (-4), none is dropped */
    {
        int t[5];
        for (int i = 0; i < 4; i++) REQ((t[i] = wnr4_cuda_batch_submit(1, geo, rp.data() + i, total.data() + i, owner.data() + i, mh.id, 0, ds.data() + 3 * i, dw.data() + 3 * i, nullptr)) >= 0);
        REQ((t[4] = wnr4_cuda_batch_submit(1, geo, rp.data() + 4, total.data() + 4, owner.data() + 4, mh.id, 0, ds.data() + 12, dw.data() + 12, nullptr)) == -4);
        for (int i = 3; i >= 0; i--) { int s1 = 7; REQ(wnr4_cuda_batch_wait(t[i], &s1) == 0 && s1 == 0); }
        REQ(wnr4_cuda_batch_wait(t[0], st.data()) == -1);
        wnr4_cuda_release_thread();
        printf("slot ring: 4 in flight accepted, 5th refused, out-of-order waits OK, stale ticket -1, release OK\n");
    }
    return 0;
}
