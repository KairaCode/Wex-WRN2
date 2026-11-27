/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 CUDA helper (prototype, separate from c/tools/wex_cuda_bridge.cu): model registry and batched decode
 * straight into caller-owned device buffers. Per batch: one coalesced upload of contiguous valid records, one
 * kernel launch for every tile of every record (kind I4 uniform-G batches pass record metadata directly with no
 * per-tile job upload; kind GG adds one launch that copies verbatim slices), one copy of the error words back.
 *   nvcc -O3 -std=c++17 -arch=sm_120 -shared wnr4/tools/wnr4_cuda_bridge.cu -o wnr4_cuda.dll
 * On Windows MSVC hosts, pass -Xcompiler=/O2,/EHsc so host C++ is optimised alongside device code.
 * geo = u32[13]: kind, scount[3], scols[3], soff[3], woff[3] (wnr4_geo).
 * Asynchronous use: wnr4_cuda_batch_submit() enqueues a batch and returns a ticket; up to WNR4_CUDA_SLOTS
 * batches per calling thread and device may be in flight; every ticket must be waited for (on the thread that
 * submitted it) with wnr4_cuda_batch_wait(), and the record buffers must stay valid and unchanged until then.
 * The helper's own streams (stream == NULL) are non-blocking: they are not ordered after work the caller
 * queued on the legacy default stream; synchronise first, or pass your own stream.
 * Every call leaves the caller's current CUDA device as it found it.
 * Kind GG (GGUF Q2_0 / IQ4_NL slices, wnr4/wnr4_gg.h): wnr4_cuda_gg_register_model(), then
 * wnr4_cuda_gg_batch_submit() (waited for with wnr4_cuda_batch_wait(), same slots) or the synchronous
 * wnr4_cuda_gg_decode_batch(). The geometry (nseg, types[], bytes[]) is the same for every record of a batch;
 * each record decodes to its whole slab at dst[i] (device, any alignment), verbatim slices copied there too.
 * Kind I4 (int4-g64 experts, wnr4/wnr4_i4.h): wnr4_cuda_i4_register_model(), then wnr4_cuda_i4_batch_submit()
 * (waited for with wnr4_cuda_batch_wait()) or wnr4_cuda_i4_decode_batch(). The geometry (nmat, rows[], cols[],
 * pairs) is the same for every record of a batch; each record decodes to its whole record bytes at dst[i] (device,
 * 4-byte aligned); records with G above 64 are refused (status -1: decode them on the CPU). */
#include "../wnr4_cuda.cuh"
#include "../wnr4_gg_cuda.cuh"
#include "../wnr4_i4_cuda.cuh"
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
struct GgModel { Wnr4GgModel host; int device; unsigned char *dev; };
struct I4Model { Wnr4I4Model host; int device; unsigned char *dev; };
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
    Wnr4GgCudaJob *dgjob = nullptr, *hgjob = nullptr; size_t dgjob_cap = 0, hgjob_cap = 0;   /* kind GG */
    Wnr4GgCudaCopy *dcopy = nullptr, *hcopy = nullptr; size_t dcopy_cap = 0, hcopy_cap = 0;
    Wnr4I4CudaJob *dijob = nullptr, *hijob = nullptr; size_t dijob_cap = 0, hijob_cap = 0;   /* kind I4 mixed-G */
    Wnr4I4CudaRec *direc = nullptr, *hirec = nullptr; size_t direc_cap = 0, hirec_cap = 0;   /* kind I4 >64 recs */
    cudaEvent_t done = nullptr;
    cudaStream_t own = nullptr;   /* this slot's stream when the caller gives none: batches in flight overlap */
    std::vector<int> status;      /* host-side verdicts before the GPU (0 = queued) */
    std::vector<size_t> at;       /* reusable per-record device offset scratch */
    std::vector<Wnr4I4CudaRec> irecs;
    std::vector<Wnr4I4CudaJob> iby_m[3], ijobs;
    bool busy = false, broken = false, err_dirty = true, has_gpu_work = false;
    int ticket = -1;
    void release() {
        if (busy && has_gpu_work && done) (void)cudaEventSynchronize(done);
        if (drec) cudaFree(drec);
        if (djob) cudaFree(djob);
        if (dinfo) cudaFree(dinfo);
        if (derr) cudaFree(derr);
        if (hjob) cudaFreeHost(hjob);
        if (hinfo) cudaFreeHost(hinfo);
        if (herr) cudaFreeHost(herr);
        if (dgjob) cudaFree(dgjob);
        if (dcopy) cudaFree(dcopy);
        if (hgjob) cudaFreeHost(hgjob);
        if (hcopy) cudaFreeHost(hcopy);
        if (dijob) cudaFree(dijob);
        if (hijob) cudaFreeHost(hijob);
        if (direc) cudaFree(direc);
        if (hirec) cudaFreeHost(hirec);
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
struct DeviceGuard {   /* switches only when needed and restores the caller's current device only if switched */
    int saved = -1;
    bool switched = false;
    DeviceGuard() { if (cudaGetDevice(&saved) != cudaSuccess) { (void)cudaGetLastError(); saved = -1; } }
    bool set(int dev) {
        if (saved == dev) return true;
        if (cudaSetDevice(dev) != cudaSuccess) { (void)cudaGetLastError(); return false; }
        switched = true;
        return true;
    }
    ~DeviceGuard() { if (switched && saved >= 0) (void)cudaSetDevice(saved); }
};
std::mutex lock_models;
std::vector<Model *> models;
std::vector<GgModel *> gg_models;
std::vector<I4Model *> i4_models;
thread_local Rings rings;

template<typename M> M *find_locked(const std::vector<M *> &list, uint32_t id, int device) {
    for (M *m : list) if (m->host.id == id && m->device == device) return m;
    return nullptr;
}
template<typename M> M *find(const std::vector<M *> &list, uint32_t id, int device) {
    thread_local uint32_t last_id = 0;
    thread_local int last_dev = -1;
    thread_local M *last_m = nullptr;
    if (last_m && last_id == id && last_dev == device) return last_m;
    std::lock_guard<std::mutex> g(lock_models);
    M *m = find_locked(list, id, device);
    if (m) { last_id = id; last_dev = device; last_m = m; }
    return m;
}
/* The calling thread's next slot on `device` (already current): 0 and *out, *ticket, or WNR4_E_BUSY / WNR4_E_CUDA. */
int take_slot(int device, Slot **out, int *ticket) {
    Ring *R = rings.get(device);
    *ticket = (int)(rings.next & 0x7fffffffu);
    Slot &s = R->slot[(unsigned)*ticket % WNR4_CUDA_SLOTS];
    if (s.busy) return WNR4_E_BUSY;   /* never drop verdicts the caller has not read */
    if (!s.done && cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming) != cudaSuccess) { (void)cudaGetLastError(); s.done = nullptr; return WNR4_E_CUDA; }
    if (!s.own && cudaStreamCreateWithFlags(&s.own, cudaStreamNonBlocking) != cudaSuccess) { (void)cudaGetLastError(); s.own = nullptr; return WNR4_E_CUDA; }
    *out = &s;
    return 0;
}
bool grow_derr(Slot &s, size_t n) {
    const size_t prev = s.derr_cap;
    if (!grow_dev(s.derr, s.derr_cap, n)) return false;
    if (s.derr_cap != prev) s.err_dirty = true;
    return true;
}
cudaError_t clear_derr_if_dirty(Slot &s, cudaStream_t st) {
    if (!s.err_dirty) return cudaSuccess;
    cudaError_t e = cudaMemsetAsync(s.derr, 0, 4u * s.derr_cap, st);
    if (e == cudaSuccess) s.err_dirty = false;
    return e;
}
cudaError_t copy_records_coalesced(Slot &s, int n, const unsigned char *const *records, const size_t *totals, cudaStream_t st) {
    cudaError_t e = cudaSuccess;
    for (int i = 0; i < n && e == cudaSuccess; ) {
        if (s.status[(size_t)i]) { i++; continue; }
        const unsigned char *src0 = records[i];
        const size_t dst0 = s.at[(size_t)i];
        size_t len = totals[i];
        int j = i + 1;
        while (j < n && !s.status[(size_t)j] && records[j] == src0 + len && s.at[(size_t)j] == dst0 + len) {
            len += totals[j];
            j++;
        }
        if (len) e = cudaMemcpyAsync(s.drec + dst0, src0, len, cudaMemcpyHostToDevice, st);
        i = j;
    }
    return e;
}
/* After a batch's work was queued on st (e = the first error): the error words back, the event, the slot busy. */
int close_slot(Slot &s, cudaError_t e, int n, cudaStream_t st, int ticket, bool gpu_work) {
    s.has_gpu_work = gpu_work;
    if (gpu_work) {
        if (e == cudaSuccess) e = cudaMemcpyAsync(s.herr, s.derr, 4u * (size_t)n, cudaMemcpyDeviceToHost, st);
        /* the event is recorded even after a failed step, so wait always waits for what was queued */
        cudaError_t ev = cudaEventRecord(s.done, st);
        s.broken = e != cudaSuccess || ev != cudaSuccess;   /* every queued record then reports 1 */
        if (s.broken) { s.err_dirty = true; (void)cudaGetLastError(); }
    } else {
        s.broken = false;
    }
    s.busy = true; s.ticket = ticket;
    rings.next++;
    return ticket;
}
int finish(Slot &s, int *status) {   /* waits for the slot's batch and writes the verdicts */
    int n = (int)s.status.size(), bad = 0;
    bool ok = true;
    if (s.has_gpu_work) {
        ok = !s.broken && cudaEventSynchronize(s.done) == cudaSuccess;
        if (!ok) { s.err_dirty = true; (void)cudaGetLastError(); }
    }
    bool gpu_err = false;
    for (int i = 0; i < n; i++) {
        int v = s.status[(size_t)i];
        if (!v) {
            if (!ok) v = 1;
            else if (s.herr[i]) { v = -1; gpu_err = true; }
            else v = 0;
        }
        if (status) status[i] = v;
        bad += v != 0;
    }
    if (gpu_err) s.err_dirty = true;
    s.busy = false; s.ticket = -1;
    return bad;
}
}

WNR4_API unsigned int wnr4_cuda_abi(void) { return 6u; }   /* 5: kind GG entry points; 6: kind I4 entry points */

/* Validates the WNM4 blob and uploads its tables to `device`. 0 ok (also when already registered),
 * -1 invalid blob, 1 CUDA error. */
WNR4_API int wnr4_cuda_register_model(const unsigned char *blob, size_t bytes, int device) {
    DeviceGuard guard;
    Model *m = new Model();
    if (wnr4_model_open(blob, bytes, &m->host)) { delete m; return -1; }
    std::lock_guard<std::mutex> g(lock_models);   /* find and insert under one lock */
    if (find_locked(models, m->host.id, device)) { wnr4_model_free(&m->host); delete m; return 0; }
    std::vector<unsigned char> img(wnr4_cuda_table_bytes(&m->host));
    wnr4_cuda_table_image(&m->host, img.data());
    m->device = device; m->dev = nullptr;
    if (!guard.set(device) || cudaMalloc(&m->dev, img.size()) != cudaSuccess ||
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
    Model *m = find(models, model_id, device);
    if (!m) return WNR4_E_MODEL;
    DeviceGuard guard;
    if (!guard.set(device)) return WNR4_E_CUDA;
    Slot *sp = nullptr;
    int ticket = 0, busy = take_slot(device, &sp, &ticket);
    if (busy) return busy;
    Slot &s = *sp;
    cudaStream_t st = stream ? (cudaStream_t)stream : s.own;
    size_t bytes = 0;
    s.at.resize((size_t)n);
    for (int i = 0; i < n; i++) { s.at[(size_t)i] = bytes; bytes += (totals[i] + 255u) & ~(size_t)255u; }
    if (!grow_dev(s.drec, s.drec_cap, bytes) || !grow_dev(s.dinfo, s.dinfo_cap, (size_t)n) ||
        !grow_host(s.hinfo, s.hinfo_cap, (size_t)n) || !grow_derr(s, (size_t)n) ||
        !grow_host(s.herr, s.herr_cap, (size_t)n)) return WNR4_E_CUDA;
    s.status.assign((size_t)n, -1);
    std::vector<Wnr4CudaJob> by_m[3];
    bool any_valid = false;
    for (int i = 0; i < n; i++) {
        unsigned char *ds[3] = {dst_scale[3 * i], dst_scale[3 * i + 1], dst_scale[3 * i + 2]};
        unsigned char *dw[3] = {dst_weight[3 * i], dst_weight[3 * i + 1], dst_weight[3 * i + 2]};
        bool ok = records[i] && ds[0] && ds[1] && ds[2] && dw[0] && dw[1] && dw[2] &&
                  !(((uintptr_t)dw[0] | (uintptr_t)dw[1] | (uintptr_t)dw[2]) & 15u);
        std::vector<Wnr4CudaJob> mine[3];
        if (ok && !wnr4_cuda_plan(&g, &m->host, records[i], totals[i], owners[i], s.drec + s.at[(size_t)i], (uint32_t)i,
                                  ds, dw, &s.hinfo[i], mine)) {
            for (int k = 0; k < 3; k++) by_m[k].insert(by_m[k].end(), mine[k].begin(), mine[k].end());
            s.status[(size_t)i] = 0;
            any_valid = true;
        }
    }
    if (!any_valid) return close_slot(s, cudaSuccess, n, st, ticket, false);
    const size_t njobs_cap = wnr4_cuda_jobs_cap(by_m, nullptr);
    if (!grow_dev(s.djob, s.djob_cap, njobs_cap + 1u) || !grow_host(s.hjob, s.hjob_cap, njobs_cap + 1u)) return WNR4_E_CUDA;
    size_t njobs = 0;
    const uint32_t W = njobs_cap ? wnr4_cuda_jobs_into(by_m, s.hjob, &njobs) : 16u;
    cudaError_t e = copy_records_coalesced(s, n, records, totals, st);
    if (e == cudaSuccess) e = cudaMemcpyAsync(s.dinfo, s.hinfo, sizeof(Wnr4CudaRec) * (size_t)n, cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess && njobs) e = cudaMemcpyAsync(s.djob, s.hjob, sizeof(Wnr4CudaJob) * njobs, cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = clear_derr_if_dirty(s, st);
    if (e == cudaSuccess) e = wnr4_cuda_launch_w(m->tab, s.djob, (uint32_t)njobs, s.dinfo, s.derr, W, st);
    return close_slot(s, e, n, st, ticket, true);
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
            if (s.has_gpu_work) (void)guard.set(R->device);
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

/* ---- kind GG ------------------------------------------------------------------ */
/* Validates the WNM4 kind-2 blob and uploads its tables to `device`, as wnr4_cuda_register_model(): 0 ok
 * (also when already registered) and *id = the model id, -1 invalid blob, 1 CUDA error (*id = 0 on -1 and 1; 0 is never a model id). */
WNR4_API int wnr4_cuda_gg_register_model(const unsigned char *blob, size_t bytes, int device, uint32_t *id) {
    DeviceGuard guard;
    if (id) *id = 0;   /* never a valid model id */
    GgModel *m = new GgModel();
    if (wnr4_gg_model_open(blob, bytes, &m->host)) { delete m; return -1; }
    const uint32_t mid = m->host.id;
    std::lock_guard<std::mutex> g(lock_models);   /* find and insert under one lock */
    if (find_locked(gg_models, mid, device)) { wnr4_gg_model_free(&m->host); delete m; if (id) *id = mid; return 0; }
    std::vector<unsigned char> img(wnr4_gg_cuda_table_bytes());
    wnr4_gg_cuda_table_image(&m->host, img.data());
    m->device = device; m->dev = nullptr;
    if (!guard.set(device) || cudaMalloc(&m->dev, img.size()) != cudaSuccess ||
        cudaMemcpy(m->dev, img.data(), img.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError();
        if (m->dev) cudaFree(m->dev);
        wnr4_gg_model_free(&m->host); delete m; return 1;
    }
    gg_models.push_back(m);
    if (id) *id = mid;
    return 0;
}

/* Enqueues n records of one geometry (nseg slices of ggml types[] and sizes bytes[]; records[i]: totals[i]
 * bytes of page-locked host memory) for decoding into dst[i] (device, the record's whole decoded slab) on
 * `stream` (NULL = the slot's own non-blocking stream). Returns a ticket for wnr4_cuda_batch_wait(), or
 * WNR4_E_ARGS, WNR4_E_MODEL, WNR4_E_CUDA (nothing queued) or WNR4_E_BUSY, as wnr4_cuda_batch_submit(). */
WNR4_API int wnr4_cuda_gg_batch_submit(int n, uint32_t nseg, const uint32_t *types, const uint32_t *bytes,
                                       const unsigned char *const *records, const size_t *totals, const uint32_t *owners,
                                       uint32_t model_id, int device, unsigned char *const *dst, void *stream) {
    Wnr4GgGeo g;
    if (n <= 0 || !types || !bytes || !records || !totals || !owners || !dst || wnr4_gg_geo(&g, nseg, types, bytes))
        return WNR4_E_ARGS;
    GgModel *m = find(gg_models, model_id, device);
    if (!m) return WNR4_E_MODEL;
    DeviceGuard guard;
    if (!guard.set(device)) return WNR4_E_CUDA;
    Slot *sp = nullptr;
    int ticket = 0, busy = take_slot(device, &sp, &ticket);
    if (busy) return busy;
    Slot &s = *sp;
    cudaStream_t st = stream ? (cudaStream_t)stream : s.own;
    size_t all = 0, max_copy = 0;
    s.at.resize((size_t)n);
    for (int i = 0; i < n; i++) { s.at[(size_t)i] = all; all += (totals[i] + 255u) & ~(size_t)255u; }
    if (!grow_dev(s.drec, s.drec_cap, all) || !grow_derr(s, (size_t)n) ||
        !grow_host(s.herr, s.herr_cap, (size_t)n)) return WNR4_E_CUDA;
    s.status.assign((size_t)n, -1);
    std::vector<Wnr4GgCudaJob> by_sub[2], jobs;
    std::vector<Wnr4GgCudaCopy> copies;
    bool any_valid = false;
    for (int i = 0; i < n; i++)
        if (records[i] && dst[i] && !wnr4_gg_cuda_plan(&g, &m->host, records[i], totals[i], owners[i], s.drec + s.at[(size_t)i],
                                                       (uint32_t)i, dst[i], by_sub, copies)) {
            s.status[(size_t)i] = 0;
            any_valid = true;
        }
    if (!any_valid) return close_slot(s, cudaSuccess, n, st, ticket, false);
    const uint32_t W = wnr4_gg_cuda_jobs(by_sub, jobs);
    if (!grow_dev(s.dgjob, s.dgjob_cap, jobs.size() + 1u) || !grow_host(s.hgjob, s.hgjob_cap, jobs.size() + 1u) ||
        !grow_dev(s.dcopy, s.dcopy_cap, copies.size() + 1u) || !grow_host(s.hcopy, s.hcopy_cap, copies.size() + 1u))
        return WNR4_E_CUDA;
    if (!jobs.empty()) memcpy(s.hgjob, jobs.data(), sizeof(Wnr4GgCudaJob) * jobs.size());
    if (!copies.empty()) memcpy(s.hcopy, copies.data(), sizeof(Wnr4GgCudaCopy) * copies.size());
    for (auto &c : copies) max_copy = c.bytes > max_copy ? c.bytes : max_copy;
    cudaError_t e = copy_records_coalesced(s, n, records, totals, st);
    if (e == cudaSuccess && !jobs.empty()) e = cudaMemcpyAsync(s.dgjob, s.hgjob, sizeof(Wnr4GgCudaJob) * jobs.size(), cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess && !copies.empty()) e = cudaMemcpyAsync(s.dcopy, s.hcopy, sizeof(Wnr4GgCudaCopy) * copies.size(), cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = clear_derr_if_dirty(s, st);
    if (e == cudaSuccess) e = wnr4_gg_cuda_launch(m->dev, m->host.id, s.dgjob, (uint32_t)jobs.size(), W, s.derr, st);
    if (e == cudaSuccess) e = wnr4_gg_cuda_copy(s.dcopy, (uint32_t)copies.size(), max_copy, st);
    return close_slot(s, e, n, st, ticket, true);
}

/* Synchronous GG batch: status[i] and the return value as wnr4_cuda_decode_batch() (the number of records with
 * status != 0; when the batch could not be queued, n with every status -1 for bad arguments or geometry, else 1). */
WNR4_API int wnr4_cuda_gg_decode_batch(int n, uint32_t nseg, const uint32_t *types, const uint32_t *bytes,
                                       const unsigned char *const *records, const size_t *totals, const uint32_t *owners,
                                       uint32_t model_id, int device, unsigned char *const *dst, int *status) {
    if (n <= 0) return 0;
    if (!status) return n;
    int t = wnr4_cuda_gg_batch_submit(n, nseg, types, bytes, records, totals, owners, model_id, device, dst, nullptr);
    if (t < 0) {
        const int v = t == WNR4_E_ARGS ? -1 : 1;
        for (int i = 0; i < n; i++) status[i] = v;
        return n;
    }
    return wnr4_cuda_batch_wait(t, status);
}

/* ---- kind I4 ------------------------------------------------------------------ */
/* Validates the WNM4 kind-3 blob and uploads its tables to `device`, as wnr4_cuda_gg_register_model(): 0 ok (also
 * when already registered) and *id = the model id, -1 invalid blob, 1 CUDA error (*id = 0 on -1 and 1). */
WNR4_API int wnr4_cuda_i4_register_model(const unsigned char *blob, size_t bytes, int device, uint32_t *id) {
    DeviceGuard guard;
    if (id) *id = 0;   /* never a valid model id */
    I4Model *m = new I4Model();
    if (wnr4_i4_model_open(blob, bytes, &m->host)) { delete m; return -1; }
    const uint32_t mid = m->host.id;
    std::lock_guard<std::mutex> g(lock_models);   /* find and insert under one lock */
    if (find_locked(i4_models, mid, device)) { wnr4_i4_model_free(&m->host); delete m; if (id) *id = mid; return 0; }
    std::vector<unsigned char> img(wnr4_i4_cuda_table_bytes());
    wnr4_i4_cuda_table_image(&m->host, img.data());
    m->device = device; m->dev = nullptr;
    if (!guard.set(device) || cudaMalloc(&m->dev, img.size()) != cudaSuccess ||
        cudaMemcpy(m->dev, img.data(), img.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError();
        if (m->dev) cudaFree(m->dev);
        wnr4_i4_model_free(&m->host); delete m; return 1;
    }
    i4_models.push_back(m);
    if (id) *id = mid;
    return 0;
}

/* Enqueues n records of one geometry (nmat matrices of rows[] x cols[], packing pairs; records[i]: totals[i] bytes
 * of page-locked host memory) for decoding into dst[i] (device, the record's decoded bytes, 4-byte aligned) on
 * `stream` (NULL = the slot's own non-blocking stream). A record with G above 64, an unaligned destination or an
 * invalid header gets status -1. Returns a ticket for wnr4_cuda_batch_wait(), or WNR4_E_ARGS, WNR4_E_MODEL,
 * WNR4_E_CUDA (nothing queued) or WNR4_E_BUSY, as wnr4_cuda_batch_submit(). */
WNR4_API int wnr4_cuda_i4_batch_submit(int n, uint32_t nmat, const uint32_t *rows, const uint32_t *cols, uint32_t pairs,
                                       const unsigned char *const *records, const size_t *totals, const uint32_t *owners,
                                       uint32_t model_id, int device, unsigned char *const *dst, void *stream) {
    Wnr4I4Geo g;
    if (n <= 0 || !rows || !cols || !records || !totals || !owners || !dst || wnr4_i4_geo(&g, nmat, rows, cols, pairs))
        return WNR4_E_ARGS;
    I4Model *m = find(i4_models, model_id, device);
    if (!m) return WNR4_E_MODEL;
    DeviceGuard guard;
    if (!guard.set(device)) return WNR4_E_CUDA;
    Slot *sp = nullptr;
    int ticket = 0, busy = take_slot(device, &sp, &ticket);
    if (busy) return busy;
    Slot &s = *sp;
    cudaStream_t st = stream ? (cudaStream_t)stream : s.own;
    size_t all = 0;
    s.at.resize((size_t)n);
    for (int i = 0; i < n; i++) { s.at[(size_t)i] = all; all += (totals[i] + 255u) & ~(size_t)255u; }
    if (!grow_dev(s.drec, s.drec_cap, all) || !grow_derr(s, (size_t)n) ||
        !grow_host(s.herr, s.herr_cap, (size_t)n)) return WNR4_E_CUDA;
    s.status.assign((size_t)n, -1);
    s.irecs.clear();
    uint32_t batch_G = 0u, ntiles_rec = 0u, tile_base[4] = {};
    bool uniform_G = true;
    for (int i = 0; i < n; i++) {
        Wnr4I4Record r;
        if (!records[i] || !dst[i] || ((uintptr_t)dst[i] & 3u) ||
            wnr4_i4_parse(records[i], totals[i], &g, &r) ||
            r.model_id != m->host.id || r.owner != owners[i] || r.G > WNR4_I4_CUDA_MAX_G)
            continue;
        s.status[(size_t)i] = 0;
        if (s.irecs.empty()) {
            batch_G = r.G;
            ntiles_rec = r.ntiles;
            for (uint32_t k = 0; k < 4u; k++) tile_base[k] = r.tile_base[k];
        } else if (r.G != batch_G) {
            uniform_G = false;
        }
        Wnr4I4CudaRec rec = {};
        rec.rec = s.drec + s.at[(size_t)i];
        rec.dst = dst[i];
        rec.owner = r.owner;
        rec.rec_idx = (uint32_t)i;
        rec.bases_at = (uint32_t)(r.bases - records[i]);
        rec.data_at = (uint32_t)(r.data - records[i]);
        rec.payload = r.payload;
        for (uint32_t k = 0; k < 3u; k++) rec.anchor[k] = (uint16_t)r.anchor[k];
        s.irecs.push_back(rec);
    }
    const size_t nvalid = s.irecs.size();
    if (!nvalid) return close_slot(s, cudaSuccess, n, st, ticket, false);

    if (uniform_G) {
        const uint32_t W = wnr4_i4_cuda_choose_warps((size_t)ntiles_rec * nvalid);
        if (W != 1u && W != 2u && W != 4u && W != 8u && W != 16u)
            return close_slot(s, cudaErrorInvalidValue, n, st, ticket, true);
        Wnr4I4CudaBatchGeo bg = {};
        bg.nmat = g.nmat; bg.G = batch_G; bg.pairs = g.pairs; bg.nrecs = (uint32_t)nvalid;
        for (uint32_t k = 0; k < 3u; k++) {
            bg.cells[k] = g.cells[k]; bg.gc[k] = g.gc[k]; bg.nbc[k] = g.nbc[k];
            bg.bbase[k] = g.bbase[k]; bg.coff[k] = g.coff[k]; bg.soff[k] = g.soff[k];
        }
        for (uint32_t k = 0; k < 4u; k++) bg.tile_base[k] = tile_base[k];
        uint32_t blk = 0u;
        for (uint32_t k = 0; k < 3u; k++) {
            bg.mat_blocks[k] = blk;
            if (k < g.nmat) {
                const uint32_t mt = (bg.tile_base[k + 1u] - bg.tile_base[k]) * (uint32_t)nvalid;
                blk += (mt + W - 1u) / W;
            }
        }
        bg.mat_blocks[3] = blk;
        if (nvalid > WNR4_I4_CUDA_INLINE_RECS) {
            if (!grow_dev(s.direc, s.direc_cap, nvalid) || !grow_host(s.hirec, s.hirec_cap, nvalid))
                return WNR4_E_CUDA;
            memcpy(s.hirec, s.irecs.data(), sizeof(Wnr4I4CudaRec) * nvalid);
        }
        cudaError_t e = copy_records_coalesced(s, n, records, totals, st);
        if (e == cudaSuccess && nvalid > WNR4_I4_CUDA_INLINE_RECS)
            e = cudaMemcpyAsync(s.direc, s.hirec, sizeof(Wnr4I4CudaRec) * nvalid, cudaMemcpyHostToDevice, st);
        if (e == cudaSuccess) e = clear_derr_if_dirty(s, st);
        if (e == cudaSuccess) {
            if (nvalid <= WNR4_I4_CUDA_INLINE_RECS) {
                Wnr4I4InlineRecs irecs = {};
                memcpy(irecs.r, s.irecs.data(), sizeof(Wnr4I4CudaRec) * nvalid);
                e = wnr4_i4_cuda_launch_recs(m->dev, m->host.id, bg, irecs, W, s.derr, st);
            } else {
                e = wnr4_i4_cuda_launch_recs_ptr(m->dev, m->host.id, bg, s.direc, W, s.derr, st);
            }
        }
        return close_slot(s, e, n, st, ticket, true);
    }

    /* Mixed-G fallback: plan per-tile jobs across records with different G values. */
    for (uint32_t k = 0; k < 3u; k++) s.iby_m[k].clear();
    s.ijobs.clear();
    for (int i = 0; i < n; i++)
        if (!s.status[(size_t)i] &&
            wnr4_i4_cuda_plan(&g, &m->host, records[i], totals[i], owners[i], s.drec + s.at[(size_t)i], (uint32_t)i, dst[i], s.iby_m))
            s.status[(size_t)i] = -1;
    const uint32_t W = wnr4_i4_cuda_jobs(s.iby_m, s.ijobs);
    if (!grow_dev(s.dijob, s.dijob_cap, s.ijobs.size() + 1u) || !grow_host(s.hijob, s.hijob_cap, s.ijobs.size() + 1u))
        return WNR4_E_CUDA;
    if (!s.ijobs.empty()) memcpy(s.hijob, s.ijobs.data(), sizeof(Wnr4I4CudaJob) * s.ijobs.size());
    cudaError_t e = copy_records_coalesced(s, n, records, totals, st);
    if (e == cudaSuccess && !s.ijobs.empty())
        e = cudaMemcpyAsync(s.dijob, s.hijob, sizeof(Wnr4I4CudaJob) * s.ijobs.size(), cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = clear_derr_if_dirty(s, st);
    if (e == cudaSuccess) e = wnr4_i4_cuda_launch(m->dev, m->host.id, s.dijob, (uint32_t)s.ijobs.size(), W, s.derr, st);
    return close_slot(s, e, n, st, ticket, true);
}

/* Synchronous I4 batch: status[i] and the return value as wnr4_cuda_gg_decode_batch(). */
WNR4_API int wnr4_cuda_i4_decode_batch(int n, uint32_t nmat, const uint32_t *rows, const uint32_t *cols, uint32_t pairs,
                                       const unsigned char *const *records, const size_t *totals, const uint32_t *owners,
                                       uint32_t model_id, int device, unsigned char *const *dst, int *status) {
    if (n <= 0) return 0;
    if (!status) return n;
    int t = wnr4_cuda_i4_batch_submit(n, nmat, rows, cols, pairs, records, totals, owners, model_id, device, dst, nullptr);
    if (t < 0) {
        const int v = t == WNR4_E_ARGS ? -1 : 1;
        for (int i = 0; i < n; i++) status[i] = v;
        return n;
    }
    return wnr4_cuda_batch_wait(t, status);
}

/* Frees this thread's slots on every device (waiting for batches in flight first). */
WNR4_API void wnr4_cuda_release_thread(void) { rings.release(); }
