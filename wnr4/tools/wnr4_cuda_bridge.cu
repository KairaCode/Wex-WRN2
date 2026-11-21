/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 CUDA helper (prototype, separate from c/tools/wex_cuda_bridge.cu): model registry and batched decode
 * straight into caller-owned device buffers. Per batch: one upload of its records, one kernel launch for
 * every tile of every record, one copy of the error words back.
 *   nvcc -O3 -std=c++17 -arch=sm_120 -shared wnr4/tools/wnr4_cuda_bridge.cu -o wnr4_cuda.dll
 * geo = u32[13]: kind, scount[3], scols[3], soff[3], woff[3] (wnr4_geo).
 * Asynchronous use: wnr4_cuda_batch_submit() enqueues a batch and returns a ticket; up to WNR4_CUDA_SLOTS
 * batches per calling thread and device may be in flight; every ticket must be waited for (on the thread that
 * submitted it) with wnr4_cuda_batch_wait(), and the record buffers must stay valid and unchanged until then.
 * The helper's own streams (stream == NULL) are non-blocking: they are not ordered after work the caller
 * queued on the legacy default stream; synchronise first, or pass your own stream.
 * Every call leaves the caller's current CUDA device as it found it. */
#include "../wnr4_cuda.cuh"
#include <mutex>
#include <vector>

#ifdef _WIN32
#define WNR4_API extern "C" __declspec(dllexport)
#else
#define WNR4_API extern "C" __attribute__((visibility("default")))
#endif

#define WNR4_CUDA_SLOTS 4
#define WNR4_E_ARGS (-1)       /* submit: bad arguments or geometry */
#define WNR4_E_MODEL (-2)      /* submit: model not registered on that device */
#define WNR4_E_CUDA (-3)       /* submit: CUDA or memory error, nothing queued */
#define WNR4_E_BUSY (-4)       /* submit: the next slot holds a batch not yet waited for */

namespace {
struct Model { Wnr4Model host; int device; unsigned char *dev; Wnr4CudaTables tab; };
template<typename T> bool grow_dev(T *&p, size_t &cap, size_t n) {
    if (n <= cap) return true;
    if (p) cudaFree(p);
    p = nullptr; cap = 0;
    if (cudaMalloc((void **)&p, n * sizeof(T)) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    cap = n; return true;
}
template<typename T> bool grow_host(T *&p, size_t &cap, size_t n) {   /* page-locked: async copies from it */
    if (n <= cap) return true;
    if (p) cudaFreeHost(p);
    p = nullptr; cap = 0;
    if (cudaMallocHost((void **)&p, n * sizeof(T)) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    cap = n; return true;
}
struct Slot {
    unsigned char *drec = nullptr; size_t drec_cap = 0;
    Wnr4CudaJob *djob = nullptr, *hjob = nullptr; size_t djob_cap = 0, hjob_cap = 0;
    Wnr4CudaRec *dinfo = nullptr, *hinfo = nullptr; size_t dinfo_cap = 0, hinfo_cap = 0;
    uint32_t *derr = nullptr, *herr = nullptr; size_t derr_cap = 0, herr_cap = 0;
    cudaEvent_t done = nullptr;
    cudaStream_t own = nullptr;   /* this slot's stream when the caller gives none: batches in flight overlap */
    std::vector<int> status;      /* host-side verdicts before the GPU (0 = queued) */
    bool busy = false, broken = false;
    int ticket = -1;
    void release() {
        if (busy && done) (void)cudaEventSynchronize(done);
        if (drec) cudaFree(drec);
        if (djob) cudaFree(djob);
        if (dinfo) cudaFree(dinfo);
        if (derr) cudaFree(derr);
        if (hjob) cudaFreeHost(hjob);
        if (hinfo) cudaFreeHost(hinfo);
        if (herr) cudaFreeHost(herr);
        if (done) cudaEventDestroy(done);
        if (own) cudaStreamDestroy(own);
        (void)cudaGetLastError();
        *this = Slot();
    }
};
struct Ring { int device = -1; Slot slot[WNR4_CUDA_SLOTS]; };
struct Rings {   /* one ring per device used by this thread; released when the thread exits */
    std::vector<Ring *> rings;
    unsigned next = 0;   /* tickets are unique per thread across devices */
    Ring *get(int device) {
        for (Ring *r : rings) if (r->device == device) return r;
        Ring *r = new Ring(); r->device = device; rings.push_back(r); return r;
    }
    void release() {
        int cur = 0;
        if (cudaGetDevice(&cur) != cudaSuccess) { (void)cudaGetLastError(); cur = -1; }
        for (Ring *r : rings) {
            if (cudaSetDevice(r->device) == cudaSuccess) for (Slot &s : r->slot) s.release();
            (void)cudaGetLastError();
            delete r;
        }
        rings.clear();
        if (cur >= 0) (void)cudaSetDevice(cur);
    }
    ~Rings() { release(); }
};
struct DeviceGuard {   /* restores the caller's current device */
    int saved = -1;
    DeviceGuard() { if (cudaGetDevice(&saved) != cudaSuccess) { (void)cudaGetLastError(); saved = -1; } }
    ~DeviceGuard() { if (saved >= 0) (void)cudaSetDevice(saved); }
};
std::mutex lock_models;
std::vector<Model *> models;
thread_local Rings rings;

Model *find_locked(uint32_t id, int device) {
    for (Model *m : models) if (m->host.id == id && m->device == device) return m;
    return nullptr;
}
Model *find(uint32_t id, int device) {
    std::lock_guard<std::mutex> g(lock_models);
    return find_locked(id, device);
}
int finish(Slot &s, int *status) {   /* waits for the slot's batch and writes the verdicts */
    int n = (int)s.status.size(), bad = 0;
    bool ok = !s.broken && cudaEventSynchronize(s.done) == cudaSuccess;
    if (!ok) (void)cudaGetLastError();
    for (int i = 0; i < n; i++) {
        int v = s.status[(size_t)i];
        if (!v) v = !ok ? 1 : s.herr[i] ? -1 : 0;
        if (status) status[i] = v;
        bad += v != 0;
    }
    s.busy = false; s.ticket = -1;
    return bad;
}
}

WNR4_API unsigned int wnr4_cuda_abi(void) { return 4u; }

/* Validates the WNM4 blob and uploads its tables to `device`. 0 ok (also when already registered),
 * -1 invalid blob, 1 CUDA error. */
WNR4_API int wnr4_cuda_register_model(const unsigned char *blob, size_t bytes, int device) {
    DeviceGuard guard;
    Model *m = new Model();
    if (wnr4_model_open(blob, bytes, &m->host)) { delete m; return -1; }
    std::lock_guard<std::mutex> g(lock_models);   /* find and insert under one lock */
    if (find_locked(m->host.id, device)) { wnr4_model_free(&m->host); delete m; return 0; }
    std::vector<unsigned char> img(wnr4_cuda_table_bytes(&m->host));
    wnr4_cuda_table_image(&m->host, img.data());
    m->device = device; m->dev = nullptr;
    if (cudaSetDevice(device) != cudaSuccess || cudaMalloc(&m->dev, img.size()) != cudaSuccess ||
        cudaMemcpy(m->dev, img.data(), img.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError();
        if (m->dev) cudaFree(m->dev);
        wnr4_model_free(&m->host); delete m; return 1;
    }
    wnr4_cuda_tables_view(&m->host, m->dev, &m->tab);
    models.push_back(m);
    return 0;
}

/* Enqueues n records of one geometry (page-locked host memory; records[i] holds totals[i] bytes) for
 * decoding into dst_scale[3i+m] / dst_weight[3i+m] (device, weights 16-byte aligned) on `stream`
 * (a cudaStream_t; NULL = the slot's own non-blocking stream). Returns a ticket >= 0, or WNR4_E_ARGS,
 * WNR4_E_MODEL, WNR4_E_CUDA (nothing queued) or WNR4_E_BUSY (the slot this batch would use still holds
 * a batch that was not waited for: wait for older tickets first). */
WNR4_API int wnr4_cuda_batch_submit(int n, const uint32_t *geo, const unsigned char *const *records,
                                    const size_t *totals, const uint32_t *owners, uint32_t model_id, int device,
                                    unsigned char *const *dst_scale, unsigned char *const *dst_weight, void *stream) {
    Wnr4Geo g;
    if (n <= 0 || !geo || !records || !totals || !owners || !dst_scale || !dst_weight ||
        wnr4_geo(&g, geo[0], geo + 1, geo + 4, geo + 7, geo + 10)) return WNR4_E_ARGS;
    Model *m = find(model_id, device);
    if (!m) return WNR4_E_MODEL;
    DeviceGuard guard;
    if (cudaSetDevice(device) != cudaSuccess) { (void)cudaGetLastError(); return WNR4_E_CUDA; }
    Ring *R = rings.get(device);
    const int ticket = (int)(rings.next & 0x7fffffffu);
    Slot &s = R->slot[(unsigned)ticket % WNR4_CUDA_SLOTS];
    if (s.busy) return WNR4_E_BUSY;   /* never drop verdicts the caller has not read */
    if (!s.done && cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) != cudaSuccess) { (void)cudaGetLastError(); s.done = nullptr; return WNR4_E_CUDA; }
    if (!s.own && cudaStreamCreateWithFlags(&s.own, cudaStreamNonBlocking) != cudaSuccess) { (void)cudaGetLastError(); s.own = nullptr; return WNR4_E_CUDA; }
    cudaStream_t st = stream ? (cudaStream_t)stream : s.own;
    size_t bytes = 0;
    std::vector<size_t> at((size_t)n);
    for (int i = 0; i < n; i++) { at[(size_t)i] = bytes; bytes += (totals[i] + 255u) & ~(size_t)255u; }
    if (!grow_dev(s.drec, s.drec_cap, bytes) || !grow_dev(s.dinfo, s.dinfo_cap, (size_t)n) ||
        !grow_host(s.hinfo, s.hinfo_cap, (size_t)n) || !grow_dev(s.derr, s.derr_cap, (size_t)n) ||
        !grow_host(s.herr, s.herr_cap, (size_t)n)) return WNR4_E_CUDA;
    s.status.assign((size_t)n, -1);
    std::vector<Wnr4CudaJob> by_m[3], jobs;
    for (int i = 0; i < n; i++) {
        unsigned char *ds[3] = {dst_scale[3 * i], dst_scale[3 * i + 1], dst_scale[3 * i + 2]};
        unsigned char *dw[3] = {dst_weight[3 * i], dst_weight[3 * i + 1], dst_weight[3 * i + 2]};
        bool ok = records[i] && ds[0] && ds[1] && ds[2] && dw[0] && dw[1] && dw[2] &&
                  !(((uintptr_t)dw[0] | (uintptr_t)dw[1] | (uintptr_t)dw[2]) & 15u);
        std::vector<Wnr4CudaJob> mine[3];
        if (ok && !wnr4_cuda_plan(&g, &m->host, records[i], totals[i], owners[i], s.drec + at[(size_t)i], (uint32_t)i,
                                  ds, dw, &s.hinfo[i], mine)) {
            for (int k = 0; k < 3; k++) by_m[k].insert(by_m[k].end(), mine[k].begin(), mine[k].end());
            s.status[(size_t)i] = 0;
        }
    }
    wnr4_cuda_jobs(by_m, jobs);
    if (!grow_dev(s.djob, s.djob_cap, jobs.size() + 1u) || !grow_host(s.hjob, s.hjob_cap, jobs.size() + 1u)) return WNR4_E_CUDA;
    if (!jobs.empty()) memcpy(s.hjob, jobs.data(), sizeof(Wnr4CudaJob) * jobs.size());
    cudaError_t e = cudaSuccess;
    for (int i = 0; i < n && e == cudaSuccess; i++)
        if (!s.status[(size_t)i]) e = cudaMemcpyAsync(s.drec + at[(size_t)i], records[i], totals[i], cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = cudaMemcpyAsync(s.dinfo, s.hinfo, sizeof(Wnr4CudaRec) * (size_t)n, cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess && !jobs.empty()) e = cudaMemcpyAsync(s.djob, s.hjob, sizeof(Wnr4CudaJob) * jobs.size(), cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = cudaMemsetAsync(s.derr, 0, 4u * (size_t)n, st);
    if (e == cudaSuccess) e = wnr4_cuda_launch(m->tab, s.djob, (uint32_t)jobs.size(), s.dinfo, s.derr, st);
    if (e == cudaSuccess) e = cudaMemcpyAsync(s.herr, s.derr, 4u * (size_t)n, cudaMemcpyDeviceToHost, st);
    /* the event is recorded even after a failed step, so wait always waits for what was queued */
    cudaError_t ev = cudaEventRecord(s.done, st);
    s.broken = e != cudaSuccess || ev != cudaSuccess;   /* every queued record then reports 1 */
    if (s.broken) (void)cudaGetLastError();
    s.busy = true; s.ticket = ticket;
    rings.next++;
    return ticket;
}

/* Waits for a submitted batch and writes status[i]: 0 decoded, -1 invalid record or failed integrity
 * check (its destinations may hold partial data: do not use them), 1 CUDA error. Returns the number of
 * records with status != 0, or -1 for a ticket that is unknown on this thread or already waited for. */
WNR4_API int wnr4_cuda_batch_wait(int ticket, int *status) {
    if (ticket < 0) return -1;
    for (Ring *R : rings.rings) {
        Slot &s = R->slot[(unsigned)ticket % WNR4_CUDA_SLOTS];
        if (s.busy && s.ticket == ticket) {
            DeviceGuard guard;
            (void)cudaSetDevice(R->device);
            return finish(s, status);
        }
    }
    return -1;
}

/* Synchronous: submit on the slot's own stream, then wait. status[i]: 0 decoded; -1 invalid record or
 * failed integrity check, also every record when the arguments or geometry are invalid; 1 CUDA or memory
 * error, or model not registered. Returns the number of records with status != 0. */
WNR4_API int wnr4_cuda_decode_batch(int n, const uint32_t *geo, const unsigned char *const *records,
                                    const size_t *totals, const uint32_t *owners, uint32_t model_id, int device,
                                    unsigned char *const *dst_scale, unsigned char *const *dst_weight, int *status) {
    if (n <= 0) return 0;
    if (!status) return n;
    int t = wnr4_cuda_batch_submit(n, geo, records, totals, owners, model_id, device, dst_scale, dst_weight, nullptr);
    if (t < 0) {
        const int v = t == WNR4_E_ARGS ? -1 : 1;
        for (int i = 0; i < n; i++) status[i] = v;
        return n;
    }
    return wnr4_cuda_batch_wait(t, status);
}

/* Frees this thread's slots on every device (waiting for batches in flight first). */
WNR4_API void wnr4_cuda_release_thread(void) { rings.release(); }
