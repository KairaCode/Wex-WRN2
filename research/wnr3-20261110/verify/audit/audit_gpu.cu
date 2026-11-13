/* GPU corruption audit on REAL V4.1 records (lab tool, KAIRA-private): the same classification as
 * audit_real.c (rejected / harmless / WRONG), decoded by the helper's batch API
 * (wex_cuda_decode_batch_to_device, 16 flipped copies per call) after the engine's host owner/profile
 * check. Rejected: the destination must stay untouched (0xA5); accepted: compared on the device with
 * the expected slab. Regions (exhaustive, every bit): record header 32 B, the first 4 KiB of the
 * length table, every scale stream of matrix 0, and the first, middle and last weight stream of
 * each matrix.
 * usage: audit_gpu.exe HELPER.dll BIN HEADER_FLAGS idx:layer:expert ...   (one JSON line per record) */
#include <windows.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "wex_format.h"

typedef int (*RegFn)(const unsigned char *, size_t);
typedef int (*BatchFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                       unsigned char *const *, unsigned char *const *, int *);
typedef unsigned int (*FeatFn)(void);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static RegFn reg; static BatchFn batch;
struct Rec { WexEntry e; std::vector<unsigned char> expected, record; };
struct Tally { unsigned long long flips = 0, rejected = 0, harmless = 0, wrong = 0, touched = 0; };

__global__ void differs(const unsigned char *a, const unsigned char *b, size_t n, int *flag) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) { *flag = 1; return; }
}
__global__ void not_fill(const unsigned char *a, size_t n, int *flag) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != 0xA5) { *flag = 1; return; }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 5) { fprintf(stderr, "usage: audit_gpu HELPER.dll BIN HEADER_FLAGS idx:L:E ...\n"); return 2; }
    HMODULE h = LoadLibraryA(argv[1]);
    if (!h) return 1;
    FeatFn feat = (FeatFn)GetProcAddress(h, "wex_cuda_helper_features");
    reg = (RegFn)GetProcAddress(h, "wex_cuda_register_model");
    batch = (BatchFn)GetProcAddress(h, "wex_cuda_decode_batch_to_device");
    if (!feat || !reg || !batch) return 1;
    uint32_t header_flags = (uint32_t)strtoul(argv[3], NULL, 0);
    FILE *f = fopen(argv[2], "rb");
    uint32_t mb, n;
    if (!f || fread(&mb, 4, 1, f) != 1) return 1;
    std::vector<unsigned char> blob(mb);
    WexNr2Model model;
    if (fread(blob.data(), 1, mb, f) != mb || fread(&n, 4, 1, f) != 1 || wex_nr2_model_open(blob.data(), mb, &model)) return 1;
    CK(cudaSetDevice(0));
    if (reg(blob.data(), mb)) { fprintf(stderr, "model registration failed\n"); return 1; }
    std::vector<unsigned> want_idx, want_l, want_e;
    for (int a = 4; a < argc; a++) {
        unsigned i, l, e;
        if (sscanf(argv[a], "%u:%u:%u", &i, &l, &e) != 3 || i >= n) return 2;
        want_idx.push_back(i); want_l.push_back(l); want_e.push_back(e);
    }
    std::vector<Rec> recs(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rb; bool keep = false;
        for (unsigned w : want_idx) keep |= w == i;
        if (fread(&recs[i].e, sizeof(WexEntry), 1, f) != 1 || fread(&rb, 4, 1, f) != 1) return 1;
        if (!keep) { _fseeki64(f, (long long)rb + recs[i].e.total_bytes, SEEK_CUR); continue; }
        recs[i].expected.resize(rb); recs[i].record.resize(recs[i].e.total_bytes);
        if (fread(recs[i].expected.data(), 1, rb, f) != rb ||
            fread(recs[i].record.data(), 1, recs[i].e.total_bytes, f) != recs[i].e.total_bytes) return 1;
    }
    fclose(f);
    const int B = 16;
    for (size_t w = 0; w < want_idx.size(); w++) {
        const Rec &r = recs[want_idx[w]];
        uint32_t L = want_l[w], E = want_e[w];
        size_t ms[3], mw[3], per = 0;
        for (int m = 0; m < 3; m++) {
            ms[m] = (size_t)r.e.scale_shapes[m][0] * r.e.scale_shapes[m][1];
            mw[m] = (size_t)r.e.weight_shapes[m][0] * r.e.weight_shapes[m][1];
            per += ms[m] + mw[m];
        }
        /* expected on the device in the arena's slot layout: scales w1..w3 then weights w1..w3 */
        std::vector<unsigned char> exp_slot(per);
        { size_t p = 0;
          for (int m = 0; m < 3; m++) { memcpy(&exp_slot[p], r.expected.data() + r.e.scale_offsets[m], ms[m]); p += ms[m]; }
          for (int m = 0; m < 3; m++) { memcpy(&exp_slot[p], r.expected.data() + r.e.weight_offsets[m], mw[m]); p += mw[m]; } }
        unsigned char *base, *dexp; int *flags;
        CK(cudaMalloc(&base, per * B)); CK(cudaMalloc(&dexp, per)); CK(cudaMalloc(&flags, 2 * B * sizeof(int)));
        CK(cudaMemcpy(dexp, exp_slot.data(), per, cudaMemcpyHostToDevice));
        std::vector<unsigned char *> ds(3 * B), dw(3 * B), pin(B);
        for (int b = 0; b < B; b++) {
            unsigned char *p = base + per * b;
            for (int m = 0; m < 3; m++) { ds[3 * b + m] = p; p += ms[m]; }
            for (int m = 0; m < 3; m++) { dw[3 * b + m] = p; p += mw[m]; }
            CK(cudaMallocHost((void **)&pin[b], r.record.size()));
            memcpy(pin[b], r.record.data(), r.record.size());
        }
        WexNr2Layout layout; WexNr2Record view;
        wex_nr2_layout(&r.e, &layout);
        if (wex_nrx_parse(r.record.data(), r.record.size(), &layout, model.id, wex_entry_nr_magic(&r.e), &view)) return 1;
        uint32_t ns = view.scale_streams + view.weight_streams;
        std::vector<size_t> off(ns + 1);
        off[0] = view.metadata_bytes;
        for (uint32_t s = 0; s < ns; s++) off[s + 1] = off[s] + wex_nr_u16(view.lengths + s * 2u);
        uint32_t s0 = wex_nr2_streams(layout.scale_count[0], view.scale_chunk), base_w = view.scale_streams;
        std::vector<std::pair<size_t, size_t>> weight_ranges;
        for (int m = 0; m < 3; m++) {
            uint32_t pw = wex_nr2_streams(layout.symbol_count[m], view.weight_chunk);
            for (uint32_t s : {base_w, base_w + pw / 2u, base_w + pw - 1u}) weight_ranges.push_back({off[s], off[s + 1]});
            base_w += pw;
        }
        struct Region { const char *name; std::vector<std::pair<size_t, size_t>> r; };
        size_t len_end = view.metadata_bytes < 32u + 4096u ? view.metadata_bytes : 32u + 4096u;
        std::vector<Region> regions = {{"header", {{0, 32}}}, {"lengths4k", {{32, len_end}}},
                                       {"scale_streams_m0", {{off[0], off[s0]}}}, {"weight_streams_9", weight_ranges}};
        /* clean decode */
        std::vector<int> st(B, 7);
        std::vector<const WexEntry *> es(B, &r.e);
        std::vector<const unsigned char *> ps(pin.begin(), pin.end());
        CK(cudaMemset(base, 0xA5, per * B));
        batch(1, es.data(), ps.data(), 0, ds.data(), dw.data(), st.data());
        CK(cudaMemset(flags, 0, 2 * B * sizeof(int)));
        differs<<<256, 256>>>(base, dexp, per, flags);
        int clean_flag = 1;
        CK(cudaMemcpy(&clean_flag, flags, sizeof(int), cudaMemcpyDeviceToHost));
        if (st[0] != 0 || clean_flag) { fprintf(stderr, "record %u: clean GPU decode failed (%d, %d)\n", want_idx[w], st[0], clean_flag); return 1; }
        printf("{\"record\":%u,\"layer\":%u,\"expert\":%u,\"codec\":%u,\"total\":%u,\"streams\":%u", want_idx[w], L, E,
               r.e.flags, r.e.total_bytes, ns);
        LARGE_INTEGER fq, t0, t1; QueryPerformanceFrequency(&fq);
        for (auto &reg_ : regions) {
            Tally t;
            std::vector<std::pair<size_t, int>> bits;
            for (auto &rg : reg_.r) for (size_t at = rg.first; at < rg.second; at++) for (int b = 0; b < 8; b++) bits.push_back({at, b});
            QueryPerformanceCounter(&t0);
            for (size_t k = 0; k < bits.size(); k += B) {
                int cnt = (int)std::min<size_t>(B, bits.size() - k), dec = 0;
                std::vector<int> host_reject(cnt, 0), slot_of(cnt, -1);
                std::vector<const WexEntry *> be; std::vector<const unsigned char *> bp;
                std::vector<unsigned char *> bds, bdw;
                for (int j = 0; j < cnt; j++) {
                    pin[j][bits[k + j].first] ^= (unsigned char)(1u << bits[k + j].second);
                    if (!wex_nr3_owner_ok(&r.e, pin[j], L, E, header_flags)) { host_reject[j] = 1; continue; }
                    slot_of[j] = dec;
                    be.push_back(&r.e); bp.push_back(pin[j]);
                    for (int m = 0; m < 3; m++) { bds.push_back(ds[3 * dec + m]); bdw.push_back(dw[3 * dec + m]); }
                    dec++;
                }
                if (dec) {
                    CK(cudaMemset(base, 0xA5, per * dec));
                    st.assign(dec, 7);
                    batch(dec, be.data(), bp.data(), 0, bds.data(), bdw.data(), st.data());
                    CK(cudaMemset(flags, 0, 2 * B * sizeof(int)));
                    for (int d = 0; d < dec; d++) {
                        if (st[d] == 0) differs<<<256, 256>>>(base + per * d, dexp, per, flags + d);
                        else not_fill<<<256, 256>>>(base + per * d, per, flags + B + d);
                    }
                    std::vector<int> hf(2 * B);
                    CK(cudaMemcpy(hf.data(), flags, 2 * B * sizeof(int), cudaMemcpyDeviceToHost));
                    for (int j = 0; j < cnt; j++) {
                        int d = slot_of[j];
                        if (d < 0) continue;
                        if (st[d]) { t.rejected++; t.touched += hf[B + d] != 0; }
                        else if (hf[d]) t.wrong++;
                        else t.harmless++;
                    }
                }
                for (int j = 0; j < cnt; j++) {
                    pin[j][bits[k + j].first] ^= (unsigned char)(1u << bits[k + j].second);
                    t.rejected += host_reject[j];
                }
                t.flips += cnt;
            }
            QueryPerformanceCounter(&t1);
            size_t bytes = 0;
            for (auto &rg : reg_.r) bytes += rg.second - rg.first;
            printf(",\"%s\":{\"bytes\":%zu,\"flips\":%llu,\"rejected\":%llu,\"harmless\":%llu,\"wrong\":%llu,"
                   "\"slab_touched_on_reject\":%llu,\"s\":%.1f}", reg_.name, bytes, t.flips, t.rejected, t.harmless, t.wrong,
                   t.touched, (double)(t1.QuadPart - t0.QuadPart) / fq.QuadPart);
        }
        printf("}\n");
        for (auto p : pin) cudaFreeHost(p);
        cudaFree(base); cudaFree(dexp); cudaFree(flags);
    }
    return 0;
}
