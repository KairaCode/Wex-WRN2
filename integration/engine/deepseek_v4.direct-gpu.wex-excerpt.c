/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: the maintainer's engine tree, src/colibri/c/deepseek_v4.c as of 2026-11-07, as committed.
 * Line numbers are of that version; the live tree moves them. Code between ranges is omitted.
 * GPU unit (COLI_V4_UNIT_GPU) for GPU-direct: mirror-cache busy/epoch/pool extensions, the fill producer,
 * v4_wex_fill_mirror, the attach/prefetch entry points, and their three call sites.
 * Uses helpers of the maintainer's tree that upstream colibri lacks: v4_vram_first_enabled, coli_v4_gpu_expert_peek_routed,
 * coli_v4_expert_store_frequency, coli_v4_gpu_mirror_bitmap, v4_cuda_priority_mark, v4ap_* (attach profiler).
 */

/* ==== deepseek_v4.c:6623-6671  v4_job_run (decode loader lane): mode 1 hook after a VRAM-first miss ==== */
/* Every loader lane runs its job through here. A job flagged vram_first is
 * answered from the VRAM mirror cache when the expert is resident there; the
 * view then carries only device pointers and no host lease. */
static void v4_job_run(ExpertLoadJob *job) {
    int tap = v4tap_on();
    double tap_t0 = tap ? v4tap_now() : 0.0;
    job->tap_tier = V4TAP_T_UNKNOWN;
    job->tap_flags = 0;
    job->tap_lookup_us = 0;
    job->gpu_only = 0;
#ifdef COLI_V4_GPU_TIER
    if (job->vram_first && job->store && job->store->gpu &&
        v4_vram_first_enabled() && !coli_v4_hybrid_enabled()) {
        memset(&job->view, 0, sizeof(job->view));
        job->view.key = job->key;
        if (coli_v4_gpu_expert_peek_routed(job->store, &job->view) == 0) {
            job->gpu_only = 1;
            job->result = 0;
            if (tap) {
                job->tap_tier = V4TAP_T_VRAM;
                job->tap_lookup_us = v4tap_us16(v4tap_now() - tap_t0);
            }
            return;
        }
        memset(&job->view, 0, sizeof(job->view));
        job->view.key = job->key;
        /* COLI_WEX_GPU_DIRECT: a Neural-rANS miss outside RAM goes disk -> VRAM */
        if (coli_v4_gpu_expert_attach_wex(job->store, &job->view) == 0) {
            job->gpu_only = 1;
            job->result = 0;
            if (tap) {
                job->tap_tier = V4TAP_T_DISK;
                job->tap_flags |= V4TAP_EF_UPLOADED;
                job->tap_lookup_us = v4tap_us16(v4tap_now() - tap_t0);
            }
            return;
        }
        memset(&job->view, 0, sizeof(job->view));
    }
#endif
    if (tap) (void)coli_v4_expert_lookup_take_kind();   /* drop a stale kind */
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    if (tap) {
        int kind = coli_v4_expert_lookup_take_kind();
        job->tap_tier = kind == V4TAP_T_UNKNOWN ? V4TAP_T_UNKNOWN : (uint8_t)(kind & 0x7f);
        if (kind != V4TAP_T_UNKNOWN && (kind & 0x80)) job->tap_flags |= V4TAP_EF_PINNED;
        job->tap_lookup_us = v4tap_us16(v4tap_now() - tap_t0);
    }
}

/* ==== deepseek_v4.c:7332-7347  v4_np_worker (next-layer prefetcher): mode 2 hook ==== */
#ifdef COLI_V4_GPU_TIER
        /* Already on the GPU: decode will not touch the host store for it. */
        if (item.store->gpu && v4_vram_first_enabled() &&
            !coli_v4_hybrid_enabled() &&
            coli_v4_gpu_expert_peek(item.store, &view) == 0)
            continue;
        memset(&view, 0, sizeof(view));
        /* COLI_WEX_GPU_DIRECT=2: a predicted Neural-rANS expert outside RAM is
         * reconstructed straight into its mirror instead of a RAM slab */
        if (item.store->gpu && v4_vram_first_enabled() && !coli_v4_hybrid_enabled() &&
            coli_v4_gpu_wex_direct_prefetch(item.rank)) {
            view.key = item.key;
            if (coli_v4_gpu_expert_prefetch_wex(item.store, &view) == 0) continue;
            memset(&view, 0, sizeof(view));
        }
#endif

/* ==== deepseek_v4.c:9703-9704  moe_token_pipeline, after the layer's compute + drain: advance the mirror epoch ==== */
    /* this layer's mirrors are no longer read: WEX direct loads may recycle them */
    if (store->gpu) coli_v4_gpu_mirror_step(store);

/* ==== deepseek_v4.c:17259-17307  V4GpuExpertMirror (busy/epoch/pool) and V4GpuExpertMirrorCache (published/epoch/counters), v4_gpu_mirror_host_guard ==== */
typedef struct {
    int layer;
    int expert;
    Dsv4CudaTensor *gate;
    Dsv4CudaTensor *up;
    Dsv4CudaTensor *down;
    uint64_t clock;
    uint64_t frequency;
    int busy;   /* a WEX direct load writes it outside the mutex: not a victim,
                 * not a hit; readers of the same key wait for "published" */
    uint32_t epoch;   /* last step epoch a view of it was handed out in */
    int pool;   /* COLI_WEX_GPU_DIRECT=2: only prefetch loads rewrite it */
    int promo;  /* V4_PROMOTE_VRAM: written by coli_v4_gpu_expert_promote (peek sees it) */
} V4GpuExpertMirror;

enum { V4_GPU_STREAM_SLOTS = 16 };

struct V4GpuExpertMirrorCache {
    V4GpuExpertMirror *entries;
    int capacity;
    int count;
    int resident_count;
    int stream_cursor;
    int promote_cursor;     /* V4_PROMOTE_VRAM ring cursor: never moves stream_cursor, so a
                             * layer's own decode picks stay on distinct ring slots */
    uint64_t clock;
    int device;
    pthread_mutex_t mutex;
    pthread_cond_t published;   /* a busy entry was filled or released */
    uint32_t epoch;   /* advanced when a decode MoE layer finished computing */
    uint64_t guard_skips;   /* choices that passed over a mirror of the current epoch */
    uint64_t refusals;      /* direct loads that found no mirror to take */
    int pool_count;         /* resident entries flagged pool */
    uint64_t pool_fills;    /* prefetch loads written into a pool mirror */
    uint64_t pool_hits;     /* routed uses of a pool mirror */
};

#ifdef _WIN32
static int v4_wex_gpu_direct_enabled(void);
#endif
/* COLI_WEX_GPU_DIRECT=2: the prefetcher writes mirrors beside the decode thread,
 * so the host path keeps off the current epoch's mirrors too. */
static int v4_gpu_mirror_host_guard(void) {
#ifdef _WIN32
    return v4_wex_gpu_direct_enabled() >= 2;
#else
    return 0;
#endif
}

/* ==== deepseek_v4.c:17314-17397  victim choice that skips busy and pool entries, v4_gpu_mirror_victim_safe, v4_gpu_mirror_pool_slot (COLI_WEX_GPU_DIRECT_POOL) ==== */
/* Frequency determines residency, with a short recency window protecting
 * the views attached by the current forward. LRU breaks frequency ties. */
static int v4_gpu_mirror_victim_range(const V4GpuExpertMirrorCache *cache,
                                      int start, int limit) {
    if (limit <= start) return start >= 0 ? start : 0;
    int victim = -1, oldest = -1;
    int count = limit - start;
    uint64_t protect = count > 40 ? 32 : (uint64_t)(count > 1 ? count - 1 : 0);
    for (int i = start; i < limit; i++) {
        const V4GpuExpertMirror *entry = &cache->entries[i];
        if (entry->busy || entry->pool) continue;
        if (oldest < 0 || entry->clock < cache->entries[oldest].clock) oldest = i;
        if (protect && cache->clock - entry->clock < protect) continue;
        if (victim < 0 || entry->frequency < cache->entries[victim].frequency ||
            (entry->frequency == cache->entries[victim].frequency &&
             entry->clock < cache->entries[victim].clock)) victim = i;
    }
    return victim >= 0 ? victim : oldest;
}

/* The producer's victim: as v4_gpu_mirror_victim_range (frequency, then LRU) or
 * plain LRU, never a busy entry nor one handed out in the current epoch. */
static int v4_gpu_mirror_victim_safe(const V4GpuExpertMirrorCache *cache, int limit, int by_frequency) {
    int victim = -1;
    for (int i = 0; i < limit; i++) {
        const V4GpuExpertMirror *e = &cache->entries[i];
        if (e->busy || e->pool || e->epoch == cache->epoch) continue;
        if (victim < 0) { victim = i; continue; }
        const V4GpuExpertMirror *v = &cache->entries[victim];
        if (by_frequency ? (e->frequency < v->frequency ||
                            (e->frequency == v->frequency && e->clock < v->clock))
                         : e->clock < v->clock)
            victim = i;
    }
    return victim;
}

/* COLI_WEX_GPU_DIRECT=2: the mirror a prefetch load colder than every resident
 * one rewrites. The pool grows, up to COLI_WEX_GPU_DIRECT_POOL entries, by taking
 * over the coldest resident mirror (VRAM-neutral); once full, its least recently
 * used entry is reused. Never busy, never handed out in the current epoch.
 * Called with the mutex held; -1 when no entry qualifies. */
static int v4_gpu_mirror_pool_slot(V4GpuExpertMirrorCache *cache) {
    static int capacity = -1;
    if (capacity < 0) {
        const char *setting = getenv("COLI_WEX_GPU_DIRECT_POOL");
        capacity = setting && *setting ? atoi(setting) : 32;
        if (capacity < 0) capacity = 0;
        if (capacity > 256) capacity = 256;
    }
    int limit = cache->resident_count;
    if (cache->pool_count < capacity) {
        int by_frequency = v4_gpu_hot_cache_wanted(), taken = -1;
        for (int i = 0; i < limit; i++) {
            const V4GpuExpertMirror *e = &cache->entries[i];
            if (e->busy || e->pool || e->epoch == cache->epoch ||
                !e->gate || !e->up || !e->down) continue;
            const V4GpuExpertMirror *t = taken >= 0 ? &cache->entries[taken] : NULL;
            if (!t || (by_frequency ? (e->frequency < t->frequency ||
                                       (e->frequency == t->frequency && e->clock < t->clock))
                                    : e->clock < t->clock))
                taken = i;
        }
        if (taken >= 0) {
            V4GpuExpertMirror *e = &cache->entries[taken];
            v4_cuda_priority_mark(cache, e->layer, e->expert, 0);
            e->pool = 1;
            cache->pool_count++;
            return taken;
        }
    }
    int victim = -1;
    for (int i = 0; i < limit; i++) {
        V4GpuExpertMirror *e = &cache->entries[i];
        if (!e->pool || e->busy || e->epoch == cache->epoch) continue;
        if (!e->gate || !e->up || !e->down) {   /* emptied by a failed load */
            e->pool = 0;
            cache->pool_count--;
            continue;
        }
        if (victim < 0 || e->clock < cache->entries[victim].clock) victim = i;
    }
    return victim;
}

/* ==== deepseek_v4.c:17430-17454  test hook coli_v4_test_gpu_pool_slot (inside #ifdef COLI_V4_TEST_HOOKS; used by tests/test_v4_gpu_priority.c) ==== */
/* v4_gpu_mirror_pool_slot on a fake resident cache: loaded[i] says whether
 * entry i holds tensors; pool[] and *pool_count are updated in place. */
int coli_v4_test_gpu_pool_slot(const uint64_t *frequency, const uint64_t *clock,
                               const uint32_t *epoch, const int *busy, const int *loaded,
                               int *pool, int count, uint32_t now_epoch, int *pool_count) {
    if (count < 1 || !pool_count) return -2;
    V4GpuExpertMirrorCache cache = {0};
    cache.count = cache.resident_count = count;
    cache.epoch = now_epoch;
    cache.pool_count = *pool_count;
    cache.entries = calloc((size_t)count, sizeof(*cache.entries));
    if (!cache.entries) return -2;
    for (int i = 0; i < count; i++) {
        V4GpuExpertMirror *e = &cache.entries[i];
        e->layer = 0; e->expert = i;
        e->frequency = frequency[i]; e->clock = clock[i]; e->epoch = epoch[i];
        e->busy = busy[i]; e->pool = pool[i];
        if (loaded[i]) e->gate = e->up = e->down = (Dsv4CudaTensor *)(uintptr_t)(i + 1);
    }
    int slot = v4_gpu_mirror_pool_slot(&cache);
    for (int i = 0; i < count; i++) pool[i] = cache.entries[i].pool;
    *pool_count = cache.pool_count;
    free(cache.entries);
    return slot;
}

/* ==== deepseek_v4.c:17465-17508  mirror cache create/free: cond 'published' init, v4_gpu_mirror_epochs exit line ==== */
static V4GpuExpertMirrorCache *v4_gpu_expert_mirrors_create_capacity(
    int device, int capacity) {
    V4GpuExpertMirrorCache *cache = calloc(1, sizeof(*cache));
    if (!cache) return NULL;
    cache->entries = calloc((size_t)capacity, sizeof(*cache->entries));
    if (!cache->entries) {
        free(cache);
        return NULL;
    }
    cache->capacity = capacity;
    cache->device = device;
    pthread_mutex_init(&cache->mutex, NULL);
    pthread_cond_init(&cache->published, NULL);
    return cache;
}

static V4GpuExpertMirrorCache *v4_gpu_expert_mirrors_create(int device,
                                                            int suggested) {
    const char *setting = getenv("DSV4_CUDA_EXPERT_MIRRORS");
    int capacity = setting ? atoi(setting) : suggested;
    if (capacity < 1) capacity = 1;
    return v4_gpu_expert_mirrors_create_capacity(device, capacity);
}

static void v4_gpu_expert_mirrors_free(V4GpuExpertMirrorCache *cache) {
    if (!cache) return;
    if (v4_gpu_mirror_host_guard())
        fprintf(stderr, "v4_gpu_mirror_epochs steps=%u guard_skips=%llu direct_refusals=%llu "
                        "pool=%d pool_fills=%llu pool_hits=%llu\n",
                cache->epoch, (unsigned long long)cache->guard_skips,
                (unsigned long long)cache->refusals, cache->pool_count,
                (unsigned long long)cache->pool_fills, (unsigned long long)cache->pool_hits);
    for (int i = 0; i < cache->count; i++) {
        if (cache->entries[i].gate)
            dsv4_cuda_tensor_free(cache->entries[i].gate);
        if (cache->entries[i].up)
            dsv4_cuda_tensor_free(cache->entries[i].up);
        if (cache->entries[i].down)
            dsv4_cuda_tensor_free(cache->entries[i].down);
    }
    pthread_mutex_destroy(&cache->mutex);
    pthread_cond_destroy(&cache->published);
    free(cache->entries);
    free(cache);

/* ==== deepseek_v4.c:17510-17583  V4GpuMirrorFill, v4_gpu_mirror_refill(fill), v4_gpu_mirror_create(fill) ==== */

/* A producer that writes an expert's fp4 bytes straight into its mirror's
 * device buffers (WEX direct-to-VRAM) instead of the upload from host views.
 * fn returns 0 once gate, down and up hold the expert. */
typedef struct {
    int (*fn)(void *context, Dsv4CudaTensor *gate, Dsv4CudaTensor *down, Dsv4CudaTensor *up);
    void *context;
    int no_stream;   /* a prefetch guess: never a transient slot (the pool instead) */
} V4GpuMirrorFill;

/* Recycle entry e for view: refill from the host views, or let the producer
 * write it. The producer runs on another CUDA runtime's stream, so kernels
 * already queued on the backend stream that may still read the victim's old
 * bytes are drained first (the host refill is stream-ordered instead). */
static int v4_gpu_mirror_refill(V4GpuExpertMirrorCache *cache, V4GpuExpertMirror *e,
                                const ColiExpertView *view, int sync,
                                const V4GpuMirrorFill *fill) {
    if (!e->gate || !e->up || !e->down) return 0;
    if (fill)
        return dsv4_cuda_stream_drain(cache->device) &&
               fill->fn(fill->context, e->gate, e->down, e->up) == 0;
    return view->gate.data && view->gate.scales &&
           view->up.data && view->up.scales &&
           view->down.data && view->down.scales &&
           dsv4_cuda_tensor_refill_fp4(
               e->gate, (const uint8_t *)view->gate.data,
               (const uint8_t *)view->gate.scales, (int)view->gate.rows,
               (int)view->gate.columns, 0) &&
           dsv4_cuda_tensor_refill_fp4(
               e->up, (const uint8_t *)view->up.data,
               (const uint8_t *)view->up.scales, (int)view->up.rows,
               (int)view->up.columns, 0) &&
           dsv4_cuda_tensor_refill_fp4(
               e->down, (const uint8_t *)view->down.data,
               (const uint8_t *)view->down.scales, (int)view->down.rows,
               (int)view->down.columns, sync);
}

/* A new entry: upload the host views, or allocate and let the producer fill. */
static int v4_gpu_mirror_create(V4GpuExpertMirrorCache *cache, const ColiExpertView *view,
                                Dsv4CudaTensor **gate, Dsv4CudaTensor **up,
                                Dsv4CudaTensor **down, const V4GpuMirrorFill *fill) {
    if (fill)
        return view->gate.rows > 0 && view->gate.columns > 0 &&
               view->up.rows > 0 && view->up.columns > 0 &&
               view->down.rows > 0 && view->down.columns > 0 &&
               dsv4_cuda_tensor_alloc_fp4(gate, (int)view->gate.rows, (int)view->gate.columns, cache->device) &&
               dsv4_cuda_tensor_alloc_fp4(up, (int)view->up.rows, (int)view->up.columns, cache->device) &&
               dsv4_cuda_tensor_alloc_fp4(down, (int)view->down.rows, (int)view->down.columns, cache->device) &&
               fill->fn(fill->context, *gate, *down, *up) == 0;
    int ok = (view->gate.data && view->gate.scales &&
              view->gate.rows > 0 && view->gate.columns > 0 &&
              dsv4_cuda_upload_fp4(
                  gate, (const uint8_t *)view->gate.data,
                  (const uint8_t *)view->gate.scales,
                  (int)view->gate.rows, (int)view->gate.columns,
                  cache->device));
    if (ok)
        ok = (view->up.data && view->up.scales && view->up.rows > 0 &&
              view->up.columns > 0 &&
              dsv4_cuda_upload_fp4(
                  up, (const uint8_t *)view->up.data,
                  (const uint8_t *)view->up.scales, (int)view->up.rows,
                  (int)view->up.columns, cache->device));
    if (ok)
        ok = (view->down.data && view->down.scales &&
              view->down.rows > 0 && view->down.columns > 0 &&
              dsv4_cuda_upload_fp4(
                  down, (const uint8_t *)view->down.data,
                  (const uint8_t *)view->down.scales,
                  (int)view->down.rows, (int)view->down.columns,
                  cache->device));
    return ok;
}

/* ==== deepseek_v4.c:17944-18006  v4_gpu_mirror_grown_bytes, v4_gpu_mirror_fill_unlocked (reserve busy, fill outside the mutex, publish) ==== */
/* Bytes of the mirrors appended (new slots, not recycled ones) since the last
 * free-VRAM probe: the growth guard counts them against the reserve, because
 * the probe answer alone is stale for up to 64 misses. Guarded by cache->mutex. */
static long long v4_gpu_mirror_grown_bytes;

/* fp4 nibbles plus one UE8M0 scale per 32, as uploaded */
static long long v4_gpu_mirror_bytes(const ColiExpertView *view) {
    long long elements = (long long)view->gate.rows * view->gate.columns +
                         (long long)view->up.rows * view->up.columns +
                         (long long)view->down.rows * view->down.columns;
    return elements / 2 + elements / 32;
}

/* The producer path, entered with cache->mutex held on the chosen entry: it is
 * reserved (key set, busy), the mutex is released while the backend stream
 * drains (a recycled victim) or the buffers are allocated (a new entry) and the
 * producer writes them, then the entry is published or emptied. Peeks and
 * attaches of other experts proceed meanwhile; one of the same expert waits.
 * Returns with the mutex released. */
static int v4_gpu_mirror_fill_unlocked(V4GpuExpertMirrorCache *cache, int index,
                                       ColiExpertView *view, uint64_t frequency,
                                       const V4GpuMirrorFill *fill, int recycle, int mark) {
    V4GpuExpertMirror *e = &cache->entries[index];
    if (recycle && mark) v4_cuda_priority_mark(cache, e->layer, e->expert, 0);
    Dsv4CudaTensor *gate = recycle ? e->gate : NULL;
    Dsv4CudaTensor *up = recycle ? e->up : NULL;
    Dsv4CudaTensor *down = recycle ? e->down : NULL;
    e->layer = view->key.layer;
    e->expert = view->key.expert;
    e->clock = ++cache->clock;
    e->frequency = frequency;
    e->promo = 0;
    e->busy = 1;
    pthread_mutex_unlock(&cache->mutex);
    int ok = recycle ? dsv4_cuda_stream_drain(cache->device) &&
                       fill->fn(fill->context, gate, down, up) == 0
                     : v4_gpu_mirror_create(cache, view, &gate, &up, &down, fill);
    V4AP_LOCK(&cache->mutex);
    if (!ok) {
        if (gate) dsv4_cuda_tensor_free(gate);
        if (up) dsv4_cuda_tensor_free(up);
        if (down) dsv4_cuda_tensor_free(down);
        e->gate = e->up = e->down = NULL;
        e->layer = e->expert = -1;
        e->busy = 0;
        pthread_cond_broadcast(&cache->published);
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    e->gate = gate;
    e->up = up;
    e->down = down;
    if (!recycle) v4_gpu_mirror_grown_bytes += v4_gpu_mirror_bytes(view);   /* a new slot */
    e->busy = 0;
    e->epoch = cache->epoch;
    if (mark) v4_cuda_priority_mark(cache, view->key.layer, view->key.expert, 1);
    view->gate.gpu = gate;
    view->up.gpu = up;
    view->down.gpu = down;
    pthread_cond_broadcast(&cache->published);
    pthread_mutex_unlock(&cache->mutex);
    return 0;
}

/* ==== deepseek_v4.c:18092-18359  v4_gpu_expert_attach_cached_fill (the attach path with the fill producer) and the _ex wrapper ==== */
static int v4_gpu_expert_attach_cached_fill(V4GpuExpertMirrorCache *cache,
                                            ColiExpertView *view, int sync,
                                            uint64_t frequency, int warm_only,
                                            const V4GpuMirrorFill *fill) {
    if (!cache || !view) return -1;
    if (view->gate.block_rows != 1 || view->up.block_rows != 1 ||
        view->down.block_rows != 1)
        return -1;
    /* A layer attaches its 6 routed experts before computing them; a cache
     * smaller than that would recycle a slot still referenced by an earlier
     * view of the same token (measured: DSV4_CUDA_EXPERT_MIRRORS=1 produces
     * garbage). Below 8 entries, stay on the CPU. */
    if (cache->capacity < 8) return -1;
    int guard = fill || v4_gpu_mirror_host_guard();
    const int post = !fill && sync == 0 && !warm_only && coli_v4_copy_worker_enabled();
    V4AP_LOCK(&cache->mutex);
    int found;
find_entry:
    found = -1;
    for (int i = 0; i < cache->count; i++) {
        if (cache->entries[i].layer == view->key.layer &&
            cache->entries[i].expert == view->key.expert) {
            if (cache->entries[i].busy) {   /* a WEX direct load of this expert */
                V4AP_COND_WAIT(&cache->published, &cache->mutex);
                goto find_entry;
            }
            found = i;
            break;
        }
    }
    if (found < 0) {
        v4ap_tl_path = fill ? V4AP_PATH_FILL : V4AP_PATH_GROW;
        int stream_slots = cache->capacity > 32 ? V4_GPU_STREAM_SLOTS : 0;
        int resident_cap = cache->capacity > stream_slots ? cache->capacity - stream_slots : cache->capacity;
        int grow = cache->count < resident_cap;
        int appended = 0;   /* a new slot, not a recycled one (v4_gpu_mirror_grown_bytes) */
        static long long reserve_mb = -1;
        if (reserve_mb < 0) {
            const char *env = getenv("DSV4_CUDA_VRAM_RESERVE_MB");
            reserve_mb = env ? atoll(env)
                             : (coli_v4_gpu_moe_batch_wanted() ? 2800 : 600);
            if (reserve_mb < 256) reserve_mb = 256;
        }
        static int unified = -1;
        if (unified < 0) {
            /* Opt-in (V4_DEVICE_UNIFIED_PROBE=1): dsv4.def now exports the
             * probe, so a freshly built DLL would answer it on every card. The
             * 5090 (WDDM, discrete) is expected to answer 0, but that is not
             * validated; off, the guard stays on exactly as before the export. */
            const char *probe = getenv("V4_DEVICE_UNIFIED_PROBE");
            unified = (probe && atoi(probe) != 0 &&
                       dsv4_cuda_device_unified(cache->device)) ? 1 : 0;
            if (unified)
                fprintf(stderr, "v4_gpu mirror-cache: unified memory, VRAM reserve "
                                "guard off (free memory is the system's, not a "
                                "card's); capacity %d bounds the cache\n",
                        cache->capacity);
        }
        if (grow && cache->count > 0 && !unified) {
            static long long last_free_mb = -1;
            static unsigned probes;
            if (warm_only || last_free_mb < 0 || (probes++ & 63) == 0) {
                last_free_mb = dsv4_cuda_mem_free_mb(cache->device);
                v4_gpu_mirror_grown_bytes = 0;
            }
            long long needed = reserve_mb + (warm_only ? 256 : 0);
            /* from 8 live mirrors on, the probe less the mirrors grown since it
             * (this one included) must also clear the reserve: the stale answer
             * alone let the cache grow up to 64 mirrors past it */
            long long grown_mb = (v4_gpu_mirror_grown_bytes + v4_gpu_mirror_bytes(view) +
                                  (1LL << 20) - 1) >> 20;
            if (last_free_mb >= 0 &&
                (last_free_mb < needed ||
                 (cache->count >= 8 && last_free_mb - grown_mb < needed)))
                grow = 0;
        }
        if (!grow && warm_only) {
            if (cache->resident_count <= 0) cache->resident_count = cache->count;
            pthread_mutex_unlock(&cache->mutex);
            return 1;
        }
        if (!grow && cache->count == 0) {
            pthread_mutex_unlock(&cache->mutex);
            return -1;
        }
        if (!grow && cache->count < 8) {
            static int warned;
            if (!warned) {
                warned = 1;
                fprintf(stderr, "v4_gpu mirror-cache: frozen below the VRAM reserve "
                                "with %d live entries; a layer attaches up to 8 "
                                "before computing, so experts stay on the CPU until "
                                "the cache can grow (raise capacity or lower "
                                "DSV4_CUDA_VRAM_RESERVE_MB)\n", cache->count);
            }
            pthread_mutex_unlock(&cache->mutex);
            return -1;
        }
        if (!grow || cache->count >= resident_cap) {
            if (cache->resident_count <= 0)
                cache->resident_count = cache->count < resident_cap ? cache->count : resident_cap;
            int victim = -1;
            if (v4_gpu_hot_cache_wanted())
                victim = v4_gpu_mirror_victim_range(cache, 0, cache->resident_count);
            else {
                for (int i = 0; i < cache->resident_count; i++)
                    if (!cache->entries[i].busy && !cache->entries[i].pool &&
                        (victim < 0 || cache->entries[i].clock < cache->entries[victim].clock))
                        victim = i;
            }
            if (guard && victim >= 0 && cache->entries[victim].epoch == cache->epoch) {
                int safe = v4_gpu_mirror_victim_safe(cache, cache->resident_count,
                                                     v4_gpu_hot_cache_wanted());
                if (safe >= 0 || fill) victim = safe;   /* the host path may keep it */
                cache->guard_skips++;
            }
            if (victim < 0) {   /* every resident mirror is being loaded or in use */
                if (fill) cache->refusals++;
                pthread_mutex_unlock(&cache->mutex);
                return fill ? 2 : -1;
            }
            uint64_t victim_freq = (victim >= 0 && victim < cache->count) ? cache->entries[victim].frequency : 0;
            if (frequency > victim_freq || stream_slots <= 0) {
                /* PROMOTION (frequency > victim): promote incoming expert to permanent VRAM mirror,
                 * demoting the victim expert from VRAM -> RAM. */
                found = victim;
                v4ap_tl_path = fill ? V4AP_PATH_FILL : V4AP_PATH_RECYCLE;
                if (fill && cache->entries[found].gate && cache->entries[found].up &&
                    cache->entries[found].down)
                    return v4_gpu_mirror_fill_unlocked(cache, found, view, frequency, fill, 1, 1);
                if (post ? v4_gpu_mirror_refill_post(cache->device, &cache->entries[found], view)
                         : v4_gpu_mirror_refill(cache, &cache->entries[found], view, sync, fill)) {
                    v4_cuda_priority_mark(cache, cache->entries[found].layer, cache->entries[found].expert, 0);
                    cache->entries[found].layer = view->key.layer;
                    cache->entries[found].expert = view->key.expert;
                    cache->entries[found].promo = 0;   /* a decode reuse is not a promotion */
                    v4_cuda_priority_mark(cache, view->key.layer, view->key.expert, 1);
                    cache->entries[found].clock = ++cache->clock;
                    cache->entries[found].epoch = cache->epoch;
                    cache->entries[found].frequency = frequency;
                    view->gate.gpu = cache->entries[found].gate;
                    view->up.gpu = cache->entries[found].up;
                    view->down.gpu = cache->entries[found].down;
                    pthread_mutex_unlock(&cache->mutex);
                    return 0;
                }
                v4_cuda_priority_mark(cache, cache->entries[found].layer, cache->entries[found].expert, 0);
                if (cache->entries[found].gate) dsv4_cuda_tensor_free(cache->entries[found].gate);
                if (cache->entries[found].up) dsv4_cuda_tensor_free(cache->entries[found].up);
                if (cache->entries[found].down) dsv4_cuda_tensor_free(cache->entries[found].down);
                cache->entries[found].gate = NULL;
                cache->entries[found].up = NULL;
                cache->entries[found].down = NULL;
            } else {
                /* TRANSIENT STREAMING (frequency <= victim): expert is colder than resident mirrors,
                 * so its persistent residence stays in RAM. It executes on GPU via a transient streaming slot. */
                int sidx = -1;
                v4ap_tl_path = fill ? V4AP_PATH_FILL : V4AP_PATH_STREAM;
                if (fill && fill->no_stream) {   /* a prefetch guess: the pool or nothing */
                    int pidx = v4_gpu_mirror_pool_slot(cache);
                    if (pidx < 0) {
                        cache->refusals++;
                        pthread_mutex_unlock(&cache->mutex);
                        return 2;
                    }
                    cache->pool_fills++;
                    return v4_gpu_mirror_fill_unlocked(cache, pidx, view, frequency, fill, 1, 0);
                }
                /* pass 0 keeps off the current epoch's slots; the host path
                 * falls back to any idle slot (pass 1), a direct load never */
                for (int pass = guard ? 0 : 1; pass < 2 && sidx < 0; pass++) {
                    if (pass && fill) break;
                    for (int tries = 0; tries < stream_slots && sidx < 0; tries++) {
                        int candidate = cache->resident_count + (cache->stream_cursor++ % stream_slots);
                        if (candidate >= cache->count) sidx = candidate;
                        else if (cache->entries[candidate].busy) continue;
                        /* a promoted expert attached in this epoch is read by this layer:
                         * no pass may overwrite it (pass 1 used to, when the guard was off) */
                        else if (cache->entries[candidate].promo && cache->entries[candidate].epoch == cache->epoch)
                            cache->guard_skips++;
                        else if (pass || cache->entries[candidate].epoch != cache->epoch) sidx = candidate;
                        else cache->guard_skips++;
                    }
                }
                if (sidx < 0) {
                    if (fill) cache->refusals++;
                    pthread_mutex_unlock(&cache->mutex);
                    return fill ? 2 : -1;
                }
                if (sidx >= cache->count) cache->count = sidx + 1;
                found = sidx;
                if (fill && cache->entries[found].gate && cache->entries[found].up &&
                    cache->entries[found].down)
                    return v4_gpu_mirror_fill_unlocked(cache, found, view, frequency, fill, 1, 0);
                if (post ? v4_gpu_mirror_refill_post(cache->device, &cache->entries[found], view)
                         : v4_gpu_mirror_refill(cache, &cache->entries[found], view, sync, fill)) {
                    cache->entries[found].layer = view->key.layer;
                    cache->entries[found].expert = view->key.expert;
                    cache->entries[found].promo = 0;   /* transient streaming: never a promotion */
                    cache->entries[found].clock = ++cache->clock;
                    cache->entries[found].epoch = cache->epoch;
                    cache->entries[found].frequency = frequency;
                    view->gate.gpu = cache->entries[found].gate;
                    view->up.gpu = cache->entries[found].up;
                    view->down.gpu = cache->entries[found].down;
                    pthread_mutex_unlock(&cache->mutex);
                    return 0;
                }
                if (cache->entries[found].gate) dsv4_cuda_tensor_free(cache->entries[found].gate);
                if (cache->entries[found].up) dsv4_cuda_tensor_free(cache->entries[found].up);
                if (cache->entries[found].down) dsv4_cuda_tensor_free(cache->entries[found].down);
                cache->entries[found].gate = NULL;
                cache->entries[found].up = NULL;
                cache->entries[found].down = NULL;
            }
        } else {
            found = cache->count++;
            appended = 1;
            if (!warm_only && cache->resident_count < found) cache->resident_count = found;
        }
        cache->entries[found].layer = view->key.layer;
        cache->entries[found].expert = view->key.expert;
        cache->entries[found].frequency = frequency;
        cache->entries[found].promo = 0;
        cache->entries[found].gate = NULL;
        cache->entries[found].up = NULL;
        cache->entries[found].down = NULL;
        if (fill)
            return v4_gpu_mirror_fill_unlocked(cache, found, view, frequency, fill, 0,
                                               found < resident_cap);
        Dsv4CudaTensor *gate = NULL, *up = NULL, *down = NULL;
        int ok = post ? v4_gpu_mirror_create_post(cache, view, &gate, &up, &down)
                      : v4_gpu_mirror_create(cache, view, &gate, &up, &down, fill);
        if (!ok) {
            if (gate) dsv4_cuda_tensor_free(gate);
            if (up) dsv4_cuda_tensor_free(up);
            if (down) dsv4_cuda_tensor_free(down);
            pthread_mutex_unlock(&cache->mutex);
            return -1;
        }
        cache->entries[found].gate = gate;
        cache->entries[found].up = up;
        cache->entries[found].down = down;
        if (appended) v4_gpu_mirror_grown_bytes += v4_gpu_mirror_bytes(view);
        if (found < resident_cap) {
            v4_cuda_priority_mark(cache, view->key.layer, view->key.expert, 1);
        }
    }
    cache->entries[found].clock = ++cache->clock;
    cache->entries[found].epoch = cache->epoch;
    if (cache->entries[found].pool && !(fill && fill->no_stream)) cache->pool_hits++;
    if (frequency > cache->entries[found].frequency)
        cache->entries[found].frequency = frequency;
    if (found < (cache->resident_count > 0 ? cache->resident_count : cache->count)) {
        v4_cuda_priority_mark(cache, view->key.layer, view->key.expert, 1);
    }
    view->gate.gpu = cache->entries[found].gate;
    view->up.gpu = cache->entries[found].up;
    view->down.gpu = cache->entries[found].down;
    pthread_mutex_unlock(&cache->mutex);
    return 0;
}

static int v4_gpu_expert_attach_cached_ex(V4GpuExpertMirrorCache *cache,
                                          ColiExpertView *view, int sync,
                                          uint64_t frequency, int warm_only) {
    return v4_gpu_expert_attach_cached_fill(cache, view, sync, frequency, warm_only, NULL);
}

/* ==== deepseek_v4.c:18361-18510  v4_wex_fill_mirror, v4_wex_gpu_direct_enabled, coli_v4_gpu_wex_direct_prefetch, coli_v4_gpu_expert_attach_wex/_prefetch_wex, v4_gpu_expert_attach_wex_credit ==== */
#ifdef _WIN32
/* WEX direct-to-VRAM: the producer that reconstructs one Neural-rANS record
 * (compressed bytes in host staging) into a mirror's three fp4 tensors. */
typedef struct {
    ColiExpertStore *store;
    ColiExpertKey key;
    const WexEntry *entry;
    unsigned char *record;   /* page-locked staging of this thread */
    int device;
} V4WexDirectFill;

static int v4_wex_fill_mirror(void *context, Dsv4CudaTensor *gate, Dsv4CudaTensor *down,
                              Dsv4CudaTensor *up) {
    const V4WexDirectFill *f = context;
    Dsv4CudaTensor *tensors[3] = {gate, down, up};   /* WEX matrix order: w1, w2, w3 */
    unsigned char *scale[3], *weight[3];
    for (int m = 0; m < 3; m++) {
        void *w = NULL, *sc = NULL;
        int rows = 0, columns = 0;
        if (!dsv4_cuda_tensor_fp4_buffers(tensors[m], &w, &sc, &rows, &columns) ||
            rows != (int)f->entry->weight_shapes[m][0] ||
            columns != (int)f->entry->weight_shapes[m][1] * 2)
            return -1;
        scale[m] = sc;
        weight[m] = w;
    }
    /* the mirror is reserved: nobody else reads this record meanwhile. The RAM
     * cache of compressed records serves it when it can; otherwise the record
     * comes from disk into a cache slot (kept) or this thread's staging. */
    unsigned char *record = f->record;
    int hit = 0, slot = coli_v4_wex_crc_get(f->store, f->key, f->entry, &record, &hit);
    if (slot < 0) record = f->record;
    if (!hit && coli_v4_expert_store_wex_read(f->store, f->key, f->entry, record)) {
        if (slot >= 0) coli_v4_wex_crc_done(f->store, slot, 0);
        return -1;
    }
    int status = coli_wex_cuda_decode_to_device(f->entry, record, f->device, scale, weight);
    if (slot >= 0) coli_v4_wex_crc_done(f->store, slot, hit || status == 0);
    return status;
}

static int v4_wex_gpu_direct_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("COLI_WEX_GPU_DIRECT");
        int wanted = setting && atoi(setting) != 0;
        enabled = wanted && coli_wex_cuda_device_available() && dsv4_cuda_wex_direct_available()
            ? atoi(setting) : 0;
        if (wanted && !enabled)
            fprintf(stderr, "[v4_wex] COLI_WEX_GPU_DIRECT needs kaira_wex_cuda.dll with "
                            "wex_cuda_decode_to_device and a CUDA backend with "
                            "dsv4_cuda_tensor_alloc_fp4; staying on the host slab path\n");
        else if (enabled)
            fprintf(stderr, "[v4_wex] GPU-direct Neural-rANS reconstruction enabled: "
                            "VRAM-first misses outside RAM skip the host slab%s\n",
                    enabled >= 2 ? "; the next-layer prefetcher loads them to VRAM too" : "");
    }
    return enabled;
}
#endif

/* COLI_WEX_GPU_DIRECT=2: prefetch guesses below COLI_WEX_GPU_DIRECT_PREFETCH_RANK
 * (default 6, the experts a token routes per layer) go straight to VRAM; weaker
 * guesses keep the RAM prefetch, so a miss costs a RAM slot, not a VRAM mirror. */
int coli_v4_gpu_wex_direct_prefetch(int rank) {
#ifdef _WIN32
    static int limit = -1;
    if (limit < 0) {
        const char *setting = getenv("COLI_WEX_GPU_DIRECT_PREFETCH_RANK");
        limit = setting && *setting ? atoi(setting) : 6;
        if (limit < 0) limit = 0;
    }
    return v4_wex_gpu_direct_enabled() >= 2 && rank < limit;
#else
    (void)rank;
    return 0;
#endif
}

static int v4_gpu_expert_attach_wex_credit(ColiExpertStore *store, ColiExpertView *view, int credit);

int coli_v4_gpu_expert_attach_wex(ColiExpertStore *store, ColiExpertView *view) {
    return v4_gpu_expert_attach_wex_credit(store, view, 1);
}

int coli_v4_gpu_expert_prefetch_wex(ColiExpertStore *store, ColiExpertView *view) {
    return v4_gpu_expert_attach_wex_credit(store, view, 0);
}

static int v4_gpu_expert_attach_wex_credit(ColiExpertStore *store, ColiExpertView *view, int credit) {
#ifdef _WIN32
    if (!store || !view || !store->gpu || !v4_wex_gpu_direct_enabled()) return -1;
    V4GpuExpertMirrorCache *cache = (V4GpuExpertMirrorCache *)store->gpu;
    /* one page-locked staging buffer per loader lane: the whole direct load,
     * read and reconstruction, runs synchronously on this thread */
    static __thread unsigned char *staging;
    static __thread size_t staging_bytes;
    size_t need = (size_t)coli_v4_expert_store_record_bytes(store);
    need = (need + WEX_ALIGN - 1u) & ~(size_t)(WEX_ALIGN - 1u);
    if (!need) return -1;
    if (staging_bytes < need) {
        /* the record size is fixed per model: this happens once per thread */
        staging = _aligned_malloc(need, WEX_ALIGN);
        staging_bytes = staging ? need : 0;
        if (!staging) return -1;
        (void)coli_wex_cuda_host_register(staging, need);   /* pageable still works */
    }
    ColiExpertKey key = view->key;
    const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, staging_bytes);
    if (!entry) return -1;
    memset(view, 0, sizeof(*view));
    view->key = key;
    ColiTensorView *tensors[3] = {&view->gate, &view->down, &view->up};
    for (int m = 0; m < 3; m++) {
        ColiTensorView *t = tensors[m];
        t->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
        t->scale_format = COLI_SCALE_UE8M0;
        t->rows = entry->weight_shapes[m][0];
        t->columns = (size_t)entry->weight_shapes[m][1] * 2u;
        t->data_bytes = (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1];
        t->scale_bytes = (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
        t->block_rows = 1;
        t->block_columns = 32;
    }
    V4WexDirectFill context = {store, key, entry, staging, cache->device};
    V4GpuMirrorFill fill = {v4_wex_fill_mirror, &context, !credit};
    int status = v4_gpu_expert_attach_cached_fill(cache, view, 1,
        coli_v4_expert_store_frequency(store, key), 0, &fill);
    if (status != 0 || !view->gate.gpu || !view->up.gpu || !view->down.gpu) {
        if (status != 2) coli_v4_wex_direct_note(0, 0);   /* 2: no free mirror, nothing read */
        memset(view, 0, sizeof(*view));
        view->key = key;
        return -1;
    }
    /* a host-free view, like a VRAM hit: device handles only, no lease */
    ColiTensorView *clear[3] = {&view->gate, &view->down, &view->up};
    for (int m = 0; m < 3; m++) {
        void *gpu = clear[m]->gpu;
        memset(clear[m], 0, sizeof(*clear[m]));
        clear[m]->gpu = gpu;
    }
    if (credit)   /* a prefetch is a guess, not a routing */
        coli_v4_expert_store_note_routed(store, key);
    coli_v4_wex_direct_note(1, entry->total_bytes);
    return 0;
#else
    (void)store; (void)view; (void)credit;
    return -1;
#endif
}

/* ==== deepseek_v4.c:18538-18544  coli_v4_gpu_mirror_step ==== */
void coli_v4_gpu_mirror_step(ColiExpertStore *store) {
    if (!store || !store->gpu) return;
    V4GpuExpertMirrorCache *cache = (V4GpuExpertMirrorCache *)store->gpu;
    pthread_mutex_lock(&cache->mutex);
    cache->epoch++;
    pthread_mutex_unlock(&cache->mutex);
}

