/* replay.exe: replays a decode's expert misses (make_trace.py output) against a WEX1 or
 * WNR2 container, the way deepseek_v4.exe serves them, without running the model.
 *
 *   replay.exe --wex CONTAINER --trace RUN.trace [--helper kaira_wex_cuda.dll]
 *              [--lanes 8] [--pf-threads 8] [--batch] [--split 1] [--io-only]
 *              [--tokens N] [--skip N] [--layer-ms X] [--device 0] [--json OUT.json]
 *   replay.exe --wex WNR2 --ref WEX1 --verify N [--helper DLL]     (byte check, no trace)
 *
 * Per token, per layer (in trace order), on the main ("decode") thread:
 *   1. the layer's demand reads D go out: per-record mode, one job per record to the
 *      demand lanes (V4_LOADER_LANES); --batch, all of them as overlapped reads here
 *      (V4_WEX_BATCH's reserve/start);
 *   2. the next layer's prefetch reads (needed P, then wasted W) are queued to the
 *      prefetch threads (V4_NEXT_PREFETCH_THREADS); layer 0's go out at token start
 *      (TOKSTART). --batch: one job per layer, up to 8 records (the prefetch twins);
 *   3. --batch: wait the reads, one wex_cuda_decode_batch_to_device; then wait until
 *      every D and P record of this layer is on the device. Wasted reads are never waited
 *      for (they run on into the next layers, as the engine's do);
 *   4. --layer-ms: spin that long (the layer's attention + GPU MoE compute).
 * --pf-threads 0 replays the demand reads only. --io-only reads and stops (no H2D, no decode).
 * --skip N leaves the first N tokens out of the per-token stats and the wall GB/s.
 * A record is "on the device" after: WNR2 (codec 4) wex_cuda_decode_to_device (per-record)
 * or the batch call; WEX1 (codec 0) cudaMemcpyAsync H2D of total_bytes + stream sync.
 * Reads: one overlapped unbuffered ReadFile per record (--split N: N 4096-aligned chunks
 * in flight at once) into a 4096-aligned page-locked buffer, on a handle opened with
 * FILE_FLAG_NO_BUFFERING | OVERLAPPED | SEQUENTIAL_SCAN, as the engine's v4_wex_read_at.
 *
 * shortcut: no VRAM/RAM cache, no host slab path (the engine's WNR2 guesses with rank >= 6
 * go through decode_to_host + D2H; here every record goes to the device), per-thread staging
 * instead of the engine's shared 16-buffer ring; the trace decides what is read. */
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "wex_format.h"

typedef unsigned int (*FeatFn)(void);
typedef int (*RegFn)(const unsigned char *, size_t);
typedef int (*DevFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);
typedef int (*BatchFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                       unsigned char *const *, unsigned char *const *, int *);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define DIE(...) do { fprintf(stderr, "replay: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

static const int MAXB = 8, MAXSPLIT = 16;

static struct {
    const char *wex = nullptr, *trace = nullptr, *ref = nullptr, *json = nullptr;
    const char *helper = "<KAIRA>/lab/v41-gpu-20261107/w1/build/kaira_wex_cuda.dll";
    int lanes = 8, pf = 8, batch = 0, split = 1, io_only = 0, tokens = 0, skip = 0, device = 0, verify = 0;
    double layer_ms = 0.0;
} opt;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ---- container ------------------------------------------------------------------- */
struct Container {
    HANDLE h = INVALID_HANDLE_VALUE;
    WexHeader hdr;
    std::vector<WexEntry> table;
    size_t max_total = 0;
    int codec4 = 0, raw = 0;
    unsigned char *model = nullptr;
    uint32_t model_bytes = 0;
};

static int read_sync(HANDLE f, uint64_t offset, unsigned char *dst, uint32_t bytes) {
    OVERLAPPED ov = {};
    ov.Offset = (DWORD)offset;
    ov.OffsetHigh = (DWORD)(offset >> 32);
    ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    DWORD got = 0;
    BOOL ok = ReadFile(f, dst, bytes, &got, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(f, &ov, &got, TRUE);
    CloseHandle(ov.hEvent);
    return ok && got == bytes ? 0 : -1;
}

static void open_container(const char *path, Container &c) {
    c.h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                      FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (c.h == INVALID_HANDLE_VALUE) DIE("cannot open %s (error %lu)", path, GetLastError());
    LARGE_INTEGER size;
    GetFileSizeEx(c.h, &size);
    unsigned char *page = (unsigned char *)_aligned_malloc(WEX_ALIGN, WEX_ALIGN);
    if (read_sync(c.h, 0, page, WEX_ALIGN)) DIE("%s: header read failed", path);
    memcpy(&c.hdr, page, sizeof(c.hdr));
    size_t table_bytes = 0, aligned_table = 0;
    if (wex_validate_header(&c.hdr, (uint64_t)size.QuadPart, 0, 0, &table_bytes, &aligned_table))
        DIE("%s: not a valid WEX container", path);
    unsigned char *tb = (unsigned char *)_aligned_malloc(aligned_table, WEX_ALIGN);
    if (read_sync(c.h, c.hdr.table_offset, tb, (uint32_t)aligned_table)) DIE("%s: table read failed", path);
    c.table.resize(table_bytes / sizeof(WexEntry));
    memcpy(c.table.data(), tb, table_bytes);
    _aligned_free(tb);
    if (wex_validate_table(&c.hdr, c.table.data(), (uint64_t)size.QuadPart)) DIE("%s: invalid entry table", path);
    for (const WexEntry &e : c.table) {
        if (!e.total_bytes) continue;
        c.max_total = std::max(c.max_total, (size_t)e.total_bytes);
        if (e.flags == WEX_ENTRY_NEURAL_RANS2) c.codec4++;
        else if (e.flags == 0) c.raw++;
        else DIE("%s: codec %u records are not handled (WEX1 raw and WNR2 only)", path, e.flags & 0xff);
    }
    if (c.hdr.flags & WEX_HEADER_NEURAL_RANS2) {
        uint64_t at; uint32_t id;
        if (wex_nr2_model_location(&c.hdr, page, &at, &c.model_bytes, &id)) DIE("%s: no WNM2 descriptor", path);
        uint64_t aligned = WEX_ALIGN_UP((uint64_t)c.model_bytes, WEX_ALIGN);
        c.model = (unsigned char *)_aligned_malloc((size_t)aligned, WEX_ALIGN);
        if (read_sync(c.h, at, c.model, (uint32_t)aligned)) DIE("%s: model read failed", path);
    }
    _aligned_free(page);
}

static Container box;
static FeatFn h_feat; static RegFn h_reg; static DevFn h_dev; static BatchFn h_batch;

static void load_helper() {
    HMODULE m = LoadLibraryA(opt.helper);
    if (!m) DIE("cannot load helper %s (error %lu)", opt.helper, GetLastError());
    h_feat = (FeatFn)GetProcAddress(m, "wex_cuda_helper_features");
    h_reg = (RegFn)GetProcAddress(m, "wex_cuda_register_model");
    h_dev = (DevFn)GetProcAddress(m, "wex_cuda_decode_to_device");
    h_batch = (BatchFn)GetProcAddress(m, "wex_cuda_decode_batch_to_device");
    if (!h_feat || (h_feat() & 13u) != 13u || !h_reg || !h_dev || !h_batch)
        DIE("helper %s lacks features 1|4|8 (to_device, WNR2, batch)", opt.helper);
}

static void register_model(const Container &c) {
    if (c.model && h_reg(c.model, c.model_bytes) != 0) DIE("helper refused the container's WNM2 model");
}

/* ---- per-thread resources and one group of records --------------------------------- */
struct Lane {
    int slots = 0;
    unsigned char *host[MAXB] = {}, *dev[MAXB] = {};
    HANDLE ev[MAXB * MAXSPLIT] = {};
    cudaStream_t stream = nullptr;
};

static void lane_init(Lane &L, int slots) {
    CK(cudaSetDevice(opt.device));
    L.slots = slots;
    for (int i = 0; i < slots; i++) {
        L.host[i] = (unsigned char *)_aligned_malloc(box.max_total, WEX_ALIGN);
        if (!L.host[i]) DIE("out of host memory");
        CK(cudaHostRegister(L.host[i], box.max_total, cudaHostRegisterPortable));
        CK(cudaMalloc((void **)&L.dev[i], (size_t)box.hdr.aligned_expert_bytes));
    }
    for (int i = 0; i < slots * MAXSPLIT; i++) L.ev[i] = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    CK(cudaStreamCreateWithFlags(&L.stream, cudaStreamNonBlocking));
}

struct Span { double a, b; };
static std::mutex stat_mu;
static std::vector<Span> read_spans, fin_spans;
static std::vector<double> read_lat[2];          /* per record, issue -> done; [0] demand [1] prefetch */
static double decode_ms, h2d_ms;
static long long decoded, uploaded, bytes_read, failures;

struct Group {
    int layer = 0, n = 0, prefetch = 0;
    int ids[MAXB];
    const WexEntry *e[MAXB];
    OVERLAPPED ov[MAXB][MAXSPLIT];
    int parts[MAXB];
    double t0 = 0;
};

static void group_issue(Lane &L, Group &g) {
    g.t0 = now_ms();
    for (int i = 0; i < g.n; i++) {
        const WexEntry *e = g.e[i] = &box.table[(size_t)g.layer * box.hdr.num_experts + g.ids[i]];
        uint32_t chunk = (uint32_t)WEX_ALIGN_UP((uint64_t)(e->total_bytes + opt.split - 1) / opt.split, WEX_ALIGN);
        int k = 0;
        for (uint32_t at = 0; at < e->total_bytes; at += chunk, k++) {
            OVERLAPPED &ov = g.ov[i][k];
            memset(&ov, 0, sizeof ov);
            uint64_t off = e->file_offset + at;
            ov.Offset = (DWORD)off;
            ov.OffsetHigh = (DWORD)(off >> 32);
            ov.hEvent = L.ev[i * MAXSPLIT + k];
            ResetEvent(ov.hEvent);
            if (!ReadFile(box.h, L.host[i] + at, std::min(chunk, e->total_bytes - at), nullptr, &ov) &&
                GetLastError() != ERROR_IO_PENDING)
                DIE("ReadFile layer %d expert %d failed (error %lu)", g.layer, g.ids[i], GetLastError());
        }
        g.parts[i] = k;
    }
}

static void group_complete(Lane &L, Group &g) {
    double done[MAXB];
    uint64_t bytes = 0;
    for (int i = 0; i < g.n; i++) {
        DWORD got, sum = 0;
        for (int k = 0; k < g.parts[i]; k++) {
            if (!GetOverlappedResult(box.h, &g.ov[i][k], &got, TRUE)) DIE("read wait failed (error %lu)", GetLastError());
            sum += got;
        }
        if (sum != g.e[i]->total_bytes) DIE("short read layer %d expert %d", g.layer, g.ids[i]);
        done[i] = now_ms();
        bytes += sum;
    }
    double f0 = now_ms(), f1 = f0;
    int dec = 0, up = 0, bad = 0;
    if (!opt.io_only) {
        const WexEntry *be[MAXB]; const unsigned char *br[MAXB];
        unsigned char *bs[3 * MAXB], *bw[3 * MAXB];
        int st[MAXB];
        for (int i = 0; i < g.n; i++) {
            const WexEntry *e = g.e[i];
            if (e->flags == 0) {   /* raw record (WEX1, or a WNR2 record stored raw): H2D */
                CK(cudaMemcpyAsync(L.dev[i], L.host[i], e->total_bytes, cudaMemcpyHostToDevice, L.stream));
                up++;
                continue;
            }
            unsigned char *s3[3], *w3[3];
            for (int m = 0; m < 3; m++) {
                s3[m] = bs[3 * dec + m] = L.dev[i] + e->scale_offsets[m];
                w3[m] = bw[3 * dec + m] = L.dev[i] + e->weight_offsets[m];
            }
            if (opt.batch) { be[dec] = e; br[dec] = L.host[i]; }
            else bad += h_dev(e, L.host[i], opt.device, s3, w3) != 0;
            dec++;
        }
        if (opt.batch && dec) bad += h_batch(dec, be, br, opt.device, bs, bw, st);
        if (up) CK(cudaStreamSynchronize(L.stream));
        f1 = now_ms();
    }
    std::lock_guard<std::mutex> lock(stat_mu);
    for (int i = 0; i < g.n; i++) {
        read_spans.push_back({g.t0, done[i]});
        read_lat[g.prefetch].push_back(done[i] - g.t0);
    }
    if (dec + up) fin_spans.push_back({f0, f1});
    if (dec) decode_ms += f1 - f0; else h2d_ms += f1 - f0;   /* a mixed group counts as decode */
    decoded += dec; uploaded += up; bytes_read += (long long)bytes; failures += bad;
}

/* the helper allocates its thread-local workspaces on first use: do that before timing */
static const WexEntry *warm_entry;
static unsigned char *warm_record;
static void lane_warm(Lane &L) {
    if (opt.io_only || !warm_entry) return;
    if (warm_entry->flags == 0) {
        CK(cudaMemcpyAsync(L.dev[0], warm_record, warm_entry->total_bytes, cudaMemcpyHostToDevice, L.stream));
        CK(cudaStreamSynchronize(L.stream));
        return;
    }
    const WexEntry *be[MAXB]; const unsigned char *br[MAXB];
    unsigned char *bs[3 * MAXB], *bw[3 * MAXB];
    int st[MAXB];
    for (int i = 0; i < L.slots; i++) {
        be[i] = warm_entry; br[i] = warm_record;
        for (int m = 0; m < 3; m++) {
            bs[3 * i + m] = L.dev[i] + warm_entry->scale_offsets[m];
            bw[3 * i + m] = L.dev[i] + warm_entry->weight_offsets[m];
        }
    }
    if ((opt.batch ? h_batch(L.slots, be, br, opt.device, bs, bw, st) : h_dev(be[0], br[0], opt.device, bs, bw)) != 0)
        DIE("warm-up decode failed");
}

/* ---- job pools ----------------------------------------------------------------------- */
struct Job { int pos, layer, n, needed, prefetch; int ids[MAXB]; };
struct Pool {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> q;
    bool stop = false;
    std::vector<std::thread> threads;
};
static Pool demand_pool, pf_pool;
static std::mutex wait_mu;
static std::condition_variable wait_cv;
static int pending[8];   /* needed records not yet on the device, per position (ring) */
static std::atomic<int> ready_workers{0};

static void worker(Pool *pool, int slots) {
    Lane L;
    lane_init(L, slots);
    lane_warm(L);
    ready_workers++;
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lock(pool->mu);
            pool->cv.wait(lock, [&] { return pool->stop || !pool->q.empty(); });
            if (pool->q.empty()) break;
            j = pool->q.front();
            pool->q.pop_front();
        }
        Group g;
        g.layer = j.layer; g.n = j.n; g.prefetch = j.prefetch;
        memcpy(g.ids, j.ids, sizeof(int) * j.n);
        group_issue(L, g);
        group_complete(L, g);
        if (j.needed) {
            std::lock_guard<std::mutex> lock(wait_mu);
            pending[j.pos & 7] -= j.needed;
            wait_cv.notify_all();
        }
    }
}

static void submit(Pool &pool, const Job &j) {
    if (j.needed) {
        std::lock_guard<std::mutex> lock(wait_mu);
        pending[j.pos & 7] += j.needed;
    }
    std::lock_guard<std::mutex> lock(pool.mu);
    pool.q.push_back(j);
    pool.cv.notify_one();
}

static void pool_start(Pool &pool, int threads, int slots) {
    for (int i = 0; i < threads; i++) pool.threads.emplace_back(worker, &pool, slots);
}

static void pool_stop(Pool &pool) {
    { std::lock_guard<std::mutex> lock(pool.mu); pool.stop = true; }
    pool.cv.notify_all();
    for (auto &t : pool.threads) t.join();
}

/* ---- trace ---------------------------------------------------------------------------- */
struct Line { int t, l, nd, np, nw; size_t off; };
static std::vector<Line> lines;
static std::vector<int> ids;

static void load_trace(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) DIE("cannot open trace %s", path);
    char buf[1 << 16];
    while (fgets(buf, sizeof buf, f)) {
        if (buf[0] == '#' || buf[0] == '\n' || buf[0] == '\r') continue;
        char *p = buf, *q;
        long v[5];
        for (int i = 0; i < 5; i++) {
            v[i] = strtol(p, &q, 10);
            if (q == p) DIE("trace %s: bad line: %s", path, buf);
            p = q;
        }
        Line ln = {(int)v[0], (int)v[1], (int)v[2], (int)v[3], (int)v[4], ids.size()};
        if (ln.l < 0 || (uint32_t)ln.l >= box.hdr.num_layers || ln.nd < 0 || ln.np < 0 || ln.nw < 0)
            DIE("trace %s: layer %d outside the container's %u layers", path, ln.l, box.hdr.num_layers);
        for (int i = 0; i < ln.nd + ln.np + ln.nw; i++) {
            long e = strtol(p, &q, 10);
            if (q == p || e < 0 || (uint32_t)e >= box.hdr.num_experts) DIE("trace %s: bad expert id in: %s", path, buf);
            if (!box.table[(size_t)ln.l * box.hdr.num_experts + e].total_bytes)
                DIE("trace %s: layer %d expert %ld is absent from the container", path, ln.l, e);
            ids.push_back((int)e);
            p = q;
        }
        lines.push_back(ln);
    }
    fclose(f);
    if (lines.empty()) DIE("trace %s has no lines", path);
}

/* queue the prefetch reads of line i (needed first, then wasted) */
static void submit_prefetch(size_t i, int pos) {
    const Line &ln = lines[i];
    const int *pid = ids.data() + ln.off + ln.nd;
    int total = ln.np + ln.nw;
    for (int at = 0; at < total;) {
        Job j = {};
        j.pos = pos; j.layer = ln.l; j.prefetch = 1;
        int take = opt.batch ? std::min(MAXB, total - at) : 1;
        for (int k = 0; k < take; k++, at++) {
            j.ids[j.n++] = pid[at];
            j.needed += at < ln.np;
        }
        submit(pf_pool, j);
    }
}

/* ---- stats ---------------------------------------------------------------------------- */
static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
}
static double mean(const std::vector<double> &v) {
    double s = 0; for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
}
static std::vector<Span> merged(std::vector<Span> s) {
    std::sort(s.begin(), s.end(), [](const Span &a, const Span &b) { return a.a < b.a; });
    std::vector<Span> out;
    for (const Span &x : s) {
        if (!out.empty() && x.a <= out.back().b) out.back().b = std::max(out.back().b, x.b);
        else out.push_back(x);
    }
    return out;
}
static double length(const std::vector<Span> &s) { double t = 0; for (auto &x : s) t += x.b - x.a; return t; }
static double overlap(const std::vector<Span> &a, const std::vector<Span> &b) {   /* both merged */
    double t = 0;
    for (size_t i = 0, j = 0; i < a.size() && j < b.size();) {
        double lo = std::max(a[i].a, b[j].a), hi = std::min(a[i].b, b[j].b);
        if (hi > lo) t += hi - lo;
        if (a[i].b < b[j].b) i++; else j++;
    }
    return t;
}

/* ---- --verify: WNR2 records decoded by the helper against the WEX1 bytes --------------- */
static int verify(int n) {
    Container ref;
    open_container(opt.ref, ref);
    if (!box.codec4) DIE("--verify needs a WNR2 container as --wex");
    if (ref.codec4 || ref.hdr.num_layers != box.hdr.num_layers || ref.hdr.num_experts != box.hdr.num_experts)
        DIE("--ref must be the raw WEX1 of the same geometry");
    std::vector<size_t> cells;
    for (size_t i = 0; i < box.table.size(); i++)
        if (box.table[i].total_bytes) cells.push_back(i);
    if (n > (int)cells.size()) n = (int)cells.size();
    Lane L;
    lane_init(L, MAXB);
    std::vector<unsigned char *> want(MAXB);
    for (auto &p : want) p = (unsigned char *)_aligned_malloc(ref.max_total, WEX_ALIGN);
    std::vector<unsigned char> got((size_t)box.hdr.aligned_expert_bytes);
    int ok_single = 0, ok_batch = 0, raw_ok = 0, checked = 0;
    for (int at = 0; at < n; at += MAXB) {
        int m = std::min(MAXB, n - at);
        const WexEntry *be[MAXB]; const unsigned char *br[MAXB];
        unsigned char *bs[3 * MAXB], *bw[3 * MAXB];
        int st[MAXB], slot[MAXB], dec = 0;
        size_t rawb[MAXB];
        for (int i = 0; i < m; i++) {
            size_t cell = cells[(size_t)(at + i) * cells.size() / n];
            const WexEntry *e = &box.table[cell], *r = &ref.table[cell];
            uint64_t raw, aligned;
            if (wex_entry_raw_sizes(r, &raw, &aligned) ||
                memcmp(&e->scale_bytes, &r->scale_bytes, offsetof(WexEntry, flags) - offsetof(WexEntry, scale_bytes)))
                DIE("entry %zu: shapes/offsets differ between the containers", cell);
            rawb[i] = (size_t)raw;
            if (read_sync(box.h, e->file_offset, L.host[i], e->total_bytes) ||
                read_sync(ref.h, r->file_offset, want[i], r->total_bytes)) DIE("verify read failed at entry %zu", cell);
            if (e->flags == 0) { raw_ok += !memcmp(L.host[i], want[i], r->total_bytes); slot[i] = -1; continue; }
            slot[i] = dec;
            be[dec] = e; br[dec] = L.host[i];
            for (int k = 0; k < 3; k++) {
                bs[3 * dec + k] = L.dev[i] + e->scale_offsets[k];
                bw[3 * dec + k] = L.dev[i] + e->weight_offsets[k];
            }
            dec++;
        }
        /* single-record API, then the batch API, each into sentinel-filled buffers */
        for (int pass = 0; pass < 2; pass++) {
            for (int i = 0; i < m; i++) CK(cudaMemset(L.dev[i], 0xA5, (size_t)box.hdr.aligned_expert_bytes));
            if (pass == 0) {
                for (int i = 0; i < m; i++)
                    if (slot[i] >= 0) st[slot[i]] = h_dev(be[slot[i]], br[slot[i]], opt.device, bs + 3 * slot[i], bw + 3 * slot[i]);
            } else if (dec) h_batch(dec, be, br, opt.device, bs, bw, st);
            for (int i = 0; i < m; i++) {
                if (slot[i] < 0) continue;
                CK(cudaMemcpy(got.data(), L.dev[i], rawb[i], cudaMemcpyDeviceToHost));
                int ok = st[slot[i]] == 0 && !memcmp(got.data(), want[i], rawb[i]);
                (pass ? ok_batch : ok_single) += ok;
            }
        }
        checked += m;
    }
    int codec = 0;
    for (int at = 0; at < n; at++) codec += box.table[cells[(size_t)at * cells.size() / n]].flags != 0;
    printf("verify: records=%d wnr2=%d single_ok=%d batch_ok=%d raw=%d raw_ok=%d -> %s\n", checked, codec,
           ok_single, ok_batch, checked - codec, raw_ok,
           ok_single == codec && ok_batch == codec && raw_ok == checked - codec ? "ALL OK" : "MISMATCH");
    return ok_single == codec && ok_batch == codec && raw_ok == checked - codec ? 0 : 1;
}

/* ---- main ----------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> const char * { if (i + 1 >= argc) DIE("%s needs a value", a.c_str()); return argv[++i]; };
        if (a == "--wex") opt.wex = val();
        else if (a == "--trace") opt.trace = val();
        else if (a == "--helper") opt.helper = val();
        else if (a == "--ref") opt.ref = val();
        else if (a == "--json") opt.json = val();
        else if (a == "--lanes") opt.lanes = atoi(val());
        else if (a == "--pf-threads") opt.pf = atoi(val());
        else if (a == "--split") opt.split = atoi(val());
        else if (a == "--tokens") opt.tokens = atoi(val());
        else if (a == "--skip") opt.skip = atoi(val());
        else if (a == "--device") opt.device = atoi(val());
        else if (a == "--verify") opt.verify = atoi(val());
        else if (a == "--layer-ms") opt.layer_ms = atof(val());
        else if (a == "--batch") opt.batch = 1;
        else if (a == "--io-only") opt.io_only = 1;
        else DIE("unknown option %s (see the header of replay.cu)", a.c_str());
    }
    if (!opt.wex || (!opt.trace && !opt.verify)) DIE("usage: replay.exe --wex FILE --trace FILE [options] | --wex WNR2 --ref WEX1 --verify N");
    if (opt.lanes < 1 || opt.lanes > 64 || opt.pf < 0 || opt.pf > 64 || opt.split < 1 || opt.split > MAXSPLIT)
        DIE("--lanes 1..64, --pf-threads 0..64, --split 1..%d", MAXSPLIT);
    open_container(opt.wex, box);
    CK(cudaSetDevice(opt.device));
    if (box.codec4 && (!opt.io_only || opt.verify)) { load_helper(); register_model(box); }
    if (opt.verify) {
        if (!opt.ref) DIE("--verify needs --ref WEX1");
        int rc = verify(opt.verify);
        if (!opt.trace || rc) return rc;
    }
    load_trace(opt.trace);
    std::vector<std::pair<size_t, size_t>> tok;   /* [begin, end) lines per token */
    for (size_t i = 0; i < lines.size(); i++)
        if (!i || lines[i].t != lines[i - 1].t) tok.push_back({i, i + 1}); else tok.back().second = i + 1;
    if (opt.tokens > 0 && (size_t)opt.tokens < tok.size()) tok.resize(opt.tokens);
    if (opt.skip >= (int)tok.size()) DIE("--skip %d leaves no token of %zu", opt.skip, tok.size());

    /* warm-up record: a WNR2 one when there is any, so the helper's workspaces get allocated */
    for (const WexEntry &e : box.table)
        if (e.total_bytes && (!box.codec4 || e.flags == WEX_ENTRY_NEURAL_RANS2)) { warm_entry = &e; break; }
    warm_record = (unsigned char *)_aligned_malloc(box.max_total, WEX_ALIGN);
    CK(cudaHostRegister(warm_record, box.max_total, cudaHostRegisterPortable));
    if (read_sync(box.h, warm_entry->file_offset, warm_record, warm_entry->total_bytes)) DIE("warm-up read failed");
    Lane main_lane;
    int workers = opt.pf + (opt.batch ? 0 : opt.lanes);
    if (opt.batch) { lane_init(main_lane, MAXB); lane_warm(main_lane); }
    else pool_start(demand_pool, opt.lanes, 1);
    if (opt.pf) pool_start(pf_pool, opt.pf, opt.batch ? MAXB : 1);
    while (ready_workers.load() < workers) Sleep(10);   /* allocations and warm-ups stay outside the clock */

    std::vector<double> tok_ms, wait_ms;
    long long counts[3] = {0, 0, 0};
    double t_start = now_ms(), t_measured = 0;
    long long bytes_at_skip = 0;
    for (size_t t = 0; t < tok.size(); t++) {
        if ((int)t == opt.skip) {
            std::lock_guard<std::mutex> lock(stat_mu);
            t_measured = now_ms();
            bytes_at_skip = bytes_read;
        }
        double t0 = now_ms(), waited = 0;
        for (size_t i = tok[t].first; i < tok[t].second; i++) {
            const Line &ln = lines[i];
            int pos = (int)i;
            if (i == tok[t].first && opt.pf) submit_prefetch(i, pos);   /* TOKSTART */
            Group g;
            g.layer = ln.l; g.n = std::min(ln.nd, MAXB);
            if (ln.nd > MAXB) DIE("more than %d demand reads in one layer", MAXB);
            memcpy(g.ids, ids.data() + ln.off, sizeof(int) * g.n);
            if (opt.batch) { if (g.n) group_issue(main_lane, g); }
            else for (int k = 0; k < ln.nd; k++) {
                Job j = {};
                j.pos = pos; j.layer = ln.l; j.n = 1; j.needed = 1; j.ids[0] = ids[ln.off + k];
                submit(demand_pool, j);
            }
            if (i + 1 < tok[t].second && opt.pf) submit_prefetch(i + 1, pos + 1);
            double w0 = now_ms();
            if (opt.batch && g.n) group_complete(main_lane, g);
            {
                std::unique_lock<std::mutex> lock(wait_mu);
                wait_cv.wait(lock, [&] { return pending[pos & 7] <= 0; });
            }
            waited += now_ms() - w0;
            if (opt.layer_ms > 0) { double until = now_ms() + opt.layer_ms; while (now_ms() < until) {} }
            if ((int)t >= opt.skip && opt.pf) { counts[1] += ln.np; counts[2] += ln.nw; }
            if ((int)t >= opt.skip) counts[0] += ln.nd;
        }
        if ((int)t >= opt.skip) { tok_ms.push_back(now_ms() - t0); wait_ms.push_back(waited); }
    }
    double t_end = now_ms();
    long long bytes_at_end;
    { std::lock_guard<std::mutex> lock(stat_mu); bytes_at_end = bytes_read; }
    pool_stop(demand_pool);
    pool_stop(pf_pool);   /* drains the wasted reads still queued */
    double t_drained = now_ms();

    std::vector<Span> R = merged(read_spans), F = merged(fin_spans);
    double r_busy = length(R), f_busy = length(F), both = overlap(R, F);
    double measured_s = (t_end - t_measured) / 1000.0;
    double gbs_wall = (bytes_at_end - bytes_at_skip) / 1e9 / measured_s;
    double gbs_busy = r_busy > 0 ? bytes_read / 1e9 / (r_busy / 1000.0) : 0.0;
    long long fin_records = decoded + uploaded;
    const char *fmt = box.codec4 ? "wnr2" : "wex1";
    printf("replay %s %s: %zu tokens (skip %d), lanes=%d pf_threads=%d batch=%d split=%d io_only=%d layer_ms=%.1f\n",
           fmt, opt.wex, tok_ms.size(), opt.skip, opt.lanes, opt.pf, opt.batch, opt.split, opt.io_only, opt.layer_ms);
    printf("  ms/token mean=%.2f p50=%.2f p95=%.2f | layer waits mean=%.2f ms/token\n",
           mean(tok_ms), pct(tok_ms, 0.5), pct(tok_ms, 0.95), mean(wait_ms));
    printf("  reads/token demand=%.2f needed=%.2f wasted=%.2f | %.1f MB/token | read %.2f GB/s wall, %.2f GB/s while busy\n",
           counts[0] / (double)tok_ms.size(), counts[1] / (double)tok_ms.size(), counts[2] / (double)tok_ms.size(),
           (bytes_at_end - bytes_at_skip) / 1e6 / tok_ms.size(), gbs_wall, gbs_busy);
    printf("  read latency ms demand p50=%.2f p95=%.2f | prefetch p50=%.2f p95=%.2f\n",
           pct(read_lat[0], 0.5), pct(read_lat[0], 0.95), pct(read_lat[1], 0.5), pct(read_lat[1], 0.95));
    printf("  decode %lld records %.1f ms (%.3f ms/record) | h2d %lld records %.1f ms (%.3f ms/record) | "
           "overlap: %.0f%% of finish time under reads | failures=%lld | drain %.0f ms\n",
           decoded, decode_ms, decoded ? decode_ms / decoded : 0.0, uploaded, h2d_ms, uploaded ? h2d_ms / uploaded : 0.0,
           f_busy > 0 ? 100.0 * both / f_busy : 0.0, failures, t_drained - t_end);
    if (opt.json) {
        FILE *f = fopen(opt.json, "w");
        if (!f) DIE("cannot write %s", opt.json);
        std::string wex = opt.wex, tr = opt.trace;
        std::replace(wex.begin(), wex.end(), '\\', '/');
        std::replace(tr.begin(), tr.end(), '\\', '/');
        fprintf(f, "{\"format\": \"%s\", \"wex\": \"%s\", \"trace\": \"%s\", \"tokens\": %zu, \"skip\": %d, "
                   "\"lanes\": %d, \"pf_threads\": %d, \"batch\": %d, \"split\": %d, \"io_only\": %d, \"layer_ms\": %.2f,\n"
                   " \"ms_per_token\": {\"mean\": %.3f, \"p50\": %.3f, \"p95\": %.3f}, \"wait_ms_per_token\": %.3f,\n"
                   " \"reads_per_token\": {\"demand\": %.3f, \"needed\": %.3f, \"wasted\": %.3f}, \"mb_per_token\": %.2f,\n"
                   " \"read_gbs_wall\": %.3f, \"read_gbs_busy\": %.3f,\n"
                   " \"read_ms\": {\"demand_p50\": %.3f, \"demand_p95\": %.3f, \"prefetch_p50\": %.3f, \"prefetch_p95\": %.3f},\n"
                   " \"decode\": {\"records\": %lld, \"ms\": %.2f}, \"h2d\": {\"records\": %lld, \"ms\": %.2f},\n"
                   " \"read_busy_ms\": %.1f, \"finish_busy_ms\": %.1f, \"finish_under_reads_ms\": %.1f, \"failures\": %lld, "
                   "\"records_finished\": %lld, \"wall_ms\": %.1f}\n",
                fmt, wex.c_str(), tr.c_str(), tok_ms.size(), opt.skip, opt.lanes, opt.pf, opt.batch, opt.split, opt.io_only,
                opt.layer_ms, mean(tok_ms), pct(tok_ms, 0.5), pct(tok_ms, 0.95), mean(wait_ms),
                counts[0] / (double)tok_ms.size(), counts[1] / (double)tok_ms.size(), counts[2] / (double)tok_ms.size(),
                (bytes_at_end - bytes_at_skip) / 1e6 / tok_ms.size(), gbs_wall, gbs_busy,
                pct(read_lat[0], 0.5), pct(read_lat[0], 0.95), pct(read_lat[1], 0.5), pct(read_lat[1], 0.95),
                decoded, decode_ms, uploaded, h2d_ms, r_busy, f_busy, both, failures, fin_records, t_drained - t_start);
        fclose(f);
    }
    return failures ? 2 : 0;
}
