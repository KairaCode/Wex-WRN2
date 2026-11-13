/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: KAIRA src/colibri/c/deepseek_v4.c at commit 88ddb07 (2026-11-07), as committed.
 * Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
 * Store side of GPU-direct (COLI_WEX_GPU_DIRECT=1/2) and the compressed-record RAM cache
 * (COLI_WEX_RAM_CACHE_GB). Same unit as deepseek_v4.mode0-store.wex-excerpt.c (level A); needs it.
 */

/* ==== deepseek_v4.c:13937-14257  direct counters, v4_wex_read_at, coli_v4_expert_store_wex_entry/_read/_compressed, coli_v4_wex_direct_note, RAM cache (v4_wex_crc*, coli_v4_wex_crc_get/_done/_prewarm, coli_v4_wex_flock_retry, v4_wex_crc_report) ==== */
#ifdef _WIN32
/* WEX direct-to-VRAM, store side (COLI_WEX_GPU_DIRECT): a GPU-destined
 * Neural-rANS expert that the RAM cache does not hold is read compressed into
 * the caller's staging buffer; the GPU tier reconstructs it into its mirror,
 * so it never takes the disk -> RAM -> GPU -> RAM -> GPU round trip. */
static uint64_t v4_wex_direct_reads;      /* compressed records read for the GPU */
static uint64_t v4_wex_direct_bytes;
static uint64_t v4_wex_direct_ram_skips;  /* RAM held it (or was loading it): host path */
static uint64_t v4_wex_direct_failures;   /* read or reconstruction failed: host path */

static int v4_wex_read_at(HANDLE file, uint64_t offset, unsigned char *dst, uint32_t bytes) {
    static __thread HANDLE event = NULL;
    if (!event) event = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!event) return -1;
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(offset & 0xFFFFFFFFULL);
    ov.OffsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFFULL);
    ov.hEvent = event;
    ResetEvent(event);
    DWORD got = 0;
    BOOL ok = ReadFile(file, dst, bytes, &got, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING)
        ok = GetOverlappedResult(file, &ov, &got, TRUE);
    return ok && got == bytes ? 0 : -1;
}

const WexEntry *coli_v4_expert_store_wex_entry(ColiExpertStore *store, ColiExpertKey key,
                                               size_t capacity) {
    if (!store || !store->state) return NULL;
    V4ExpertStoreState *state = store->state;
    if (!state->wex_entries || state->wex_handle == INVALID_HANDLE_VALUE || !get_record(state, key))
        return NULL;
    const WexEntry *entry =
        &state->wex_entries[(size_t)key.layer * state->experts_per_layer + key.expert];
    if (!(entry->flags == WEX_ENTRY_NEURAL_RANS ||
          (entry->flags == WEX_ENTRY_NEURAL_RANS2 && coli_wex_cuda_nr2_available())) ||
        !entry->total_bytes || entry->total_bytes > capacity)
        return NULL;
    pthread_mutex_lock(&state->mutex);
    int in_ram = indexed_expert_slot(state, key) != NULL;
    pthread_mutex_unlock(&state->mutex);
    if (in_ram) {
        __atomic_fetch_add(&v4_wex_direct_ram_skips, UINT64_C(1), __ATOMIC_RELAXED);
        return NULL;
    }
    return entry;
}

int coli_v4_expert_store_wex_read(ColiExpertStore *store, ColiExpertKey key,
                                  const WexEntry *entry, unsigned char *dst) {
    if (!store || !store->state || !entry || !dst || !WEX_IS_ALIGNED(dst, WEX_ALIGN)) return -1;
    V4ExpertStoreState *state = store->state;
    if (state->wex_handle == INVALID_HANDLE_VALUE) return -1;
    HANDLE file = v4_wex_pick(state, key.layer, key.expert);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (v4_wex_read_at(file, entry->file_offset, dst, entry->total_bytes)) {
        const char *rt = getenv("WEX_READ_RETRY");
        if (rt && *rt == '1') {   /* one retry after a short wait (remote NTFS flock / allocation-starved machines) */
            Sleep(16);
            if (v4_wex_read_at(file, entry->file_offset, dst, entry->total_bytes) == 0)
                goto retry_ok;
        }
        __atomic_fetch_add(&v4_wex_direct_failures, UINT64_C(1), __ATOMIC_RELAXED);
        return -1;
    }
retry_ok:
    (void)0;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    pthread_mutex_lock(&state->mutex);
    state->disk_sec += (double)(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    pthread_mutex_unlock(&state->mutex);
    __atomic_fetch_add(&v4_wex_direct_reads, UINT64_C(1), __ATOMIC_RELAXED);
    __atomic_fetch_add(&v4_wex_direct_bytes, entry->total_bytes, __ATOMIC_RELAXED);
    return 0;
}

const WexEntry *coli_v4_expert_store_wex_compressed(ColiExpertStore *store, ColiExpertKey key,
                                                    unsigned char *dst, size_t capacity) {
    const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, capacity);
    return entry && !coli_v4_expert_store_wex_read(store, key, entry, dst) ? entry : NULL;
}

void coli_v4_wex_direct_note(int ok, uint64_t bytes) {
    (void)bytes;   /* reads are counted where they happen */
    if (!ok) __atomic_fetch_add(&v4_wex_direct_failures, UINT64_C(1), __ATOMIC_RELAXED);
}

/* Compressed record cache: the RAM tier of the WEX direct path. Slots hold whole
 * compressed records; a VRAM miss decodes straight from one, and a record read
 * from disk stays here, so the direct path no longer rereads the disk for every
 * expert that falls out of VRAM. */
typedef struct { int cell; int refs; int ready; uint32_t bytes; uint64_t clock, sum; } V4WexCrcSlot;
typedef struct {
    unsigned char *arena;
    size_t slot_bytes, arena_bytes;
    int slots, used, pinned, cells;
    int *index;                 /* cell -> slot, -1 */
    V4WexCrcSlot *slot;
    uint64_t clock, hits, misses, evictions, refused, prewarmed;
    int verify;                 /* COLI_WEX_RAM_CACHE_VERIFY: checksum on fill, check on hit */
    uint64_t verify_checks, verify_bad;
    pthread_mutex_t mutex;
} V4WexCrc;

static uint64_t v4_wex_crc_sum(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(0x9E3779B97F4A7C15) ^ n, w;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        memcpy(&w, p + i, 8);
        h = (h ^ w) * UINT64_C(0x100000001B3);
        h ^= h >> 29;
    }
    for (; i < n; i++) h = (h ^ p[i]) * UINT64_C(0x100000001B3);
    return h;
}

static V4WexCrc *v4_wex_crc(V4ExpertStoreState *state) {
    if (__atomic_load_n(&state->wex_crc_tried, __ATOMIC_ACQUIRE)) return (V4WexCrc *)state->wex_crc;
    pthread_mutex_lock(&state->mutex);
    if (!state->wex_crc_tried) {
        const char *gb = getenv("COLI_WEX_RAM_CACHE_GB"), *direct = getenv("COLI_WEX_GPU_DIRECT");
        double budget = gb && *gb ? atof(gb) * 1073741824.0 : 0.0;
        size_t slot_bytes = 0, cells = (size_t)state->layers * state->experts_per_layer;
        if (budget > 0 && direct && atoi(direct) >= 1 && state->wex_entries &&
            state->wex_handle != INVALID_HANDLE_VALUE)
            for (size_t i = 0; i < cells; i++) {
                const WexEntry *e = &state->wex_entries[i];
                if ((e->flags == WEX_ENTRY_NEURAL_RANS || e->flags == WEX_ENTRY_NEURAL_RANS2) &&
                    e->total_bytes > slot_bytes)
                    slot_bytes = e->total_bytes;
            }
        int slots = slot_bytes ? (int)(budget / (double)slot_bytes) : 0;
        V4WexCrc *c = slots > 0 ? calloc(1, sizeof(*c)) : NULL;
        if (c) {
            c->slot_bytes = slot_bytes; c->slots = slots; c->cells = (int)cells;
            c->arena_bytes = slot_bytes * (size_t)slots;
            c->arena = _aligned_malloc(c->arena_bytes, WEX_ALIGN);
            c->index = malloc(cells * sizeof(int));
            c->slot = calloc((size_t)slots, sizeof(*c->slot));
            if (!c->arena || !c->index || !c->slot) {
                _aligned_free(c->arena); free(c->index); free(c->slot); free(c); c = NULL;
            } else {
                for (size_t i = 0; i < cells; i++) c->index[i] = -1;
                for (int i = 0; i < slots; i++) c->slot[i].cell = -1;
                pthread_mutex_init(&c->mutex, NULL);
                const char *verify = getenv("COLI_WEX_RAM_CACHE_VERIFY");
                c->verify = verify && *verify && atoi(verify);
                const char *pin = getenv("COLI_WEX_RAM_CACHE_PIN");
                if (!(pin && *pin && !atoi(pin)))
                    c->pinned = !coli_wex_cuda_host_register(c->arena, c->arena_bytes);
                fprintf(stderr, "[v4_wex] RAM cache of compressed records: %d slots x %.2f MiB = %.1f GiB, "
                                "page-locked=%d\n", slots, slot_bytes / 1048576.0,
                        c->arena_bytes / 1073741824.0, c->pinned);
            }
        }
        state->wex_crc = c;
        __atomic_store_n(&state->wex_crc_tried, 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&state->mutex);
    return (V4WexCrc *)state->wex_crc;
}

static void v4_wex_crc_free(V4ExpertStoreState *state) {
    V4WexCrc *c = state ? (V4WexCrc *)state->wex_crc : NULL;
    if (!c) return;
    if (c->pinned) coli_wex_cuda_host_unregister(c->arena);
    _aligned_free(c->arena); free(c->index); free(c->slot);
    pthread_mutex_destroy(&c->mutex);
    free(c);
    state->wex_crc = NULL;
}

/* A slot for key: a hit (*hit = 1, bytes valid) or a fresh claim the caller fills
 * (*hit = 0). Either way the slot is leased until coli_v4_wex_crc_done. -1: no
 * cache, the record does not fit, it is being filled elsewhere, or every slot is
 * leased; the caller then reads into its own staging. */
int coli_v4_wex_crc_get(ColiExpertStore *store, ColiExpertKey key, const WexEntry *entry,
                        unsigned char **record, int *hit) {
    if (!store || !store->state || !entry || !record || !hit) return -1;
    V4ExpertStoreState *state = store->state;
    V4WexCrc *c = v4_wex_crc(state);
    if (!c || entry->total_bytes > c->slot_bytes || key.layer < 0 || key.expert < 0 ||
        key.layer >= state->layers || key.expert >= state->experts_per_layer) return -1;
    int cell = key.layer * state->experts_per_layer + key.expert, s;
    pthread_mutex_lock(&c->mutex);
    s = c->index[cell];
    if (s >= 0) {
        if (!c->slot[s].ready) { pthread_mutex_unlock(&c->mutex); return -1; }
        c->slot[s].refs++;
        c->slot[s].clock = ++c->clock;
        c->hits++;
        uint64_t sum = c->slot[s].sum;
        uint32_t bytes = c->slot[s].bytes;
        pthread_mutex_unlock(&c->mutex);
        *record = c->arena + (size_t)s * c->slot_bytes;
        *hit = 1;
        if (c->verify) {
            int bad = bytes != entry->total_bytes || v4_wex_crc_sum(*record, bytes) != sum;
            __atomic_fetch_add(&c->verify_checks, UINT64_C(1), __ATOMIC_RELAXED);
            if (bad && __atomic_fetch_add(&c->verify_bad, UINT64_C(1), __ATOMIC_RELAXED) < 16)
                fprintf(stderr, "[v4_wex] RAM cache VERIFY FAILED: layer %d expert %d slot %d\n",
                        key.layer, key.expert, s);
        }
        return s;
    }
    if (c->used < c->slots) s = c->used++;
    else {
        s = -1;
        for (int i = 0; i < c->slots; i++) {
            const V4WexCrcSlot *v = &c->slot[i];
            if (v->refs) continue;
            if (v->cell < 0) { s = i; break; }
            if (s < 0 || v->clock < c->slot[s].clock) s = i;
        }
        if (s < 0) { c->refused++; pthread_mutex_unlock(&c->mutex); return -1; }
        if (c->slot[s].cell >= 0) { c->index[c->slot[s].cell] = -1; c->evictions++; }
    }
    c->slot[s].cell = cell; c->slot[s].refs = 1; c->slot[s].ready = 0; c->slot[s].clock = ++c->clock;
    c->slot[s].bytes = entry->total_bytes; c->slot[s].sum = 0;
    c->index[cell] = s;
    c->misses++;
    pthread_mutex_unlock(&c->mutex);
    *record = c->arena + (size_t)s * c->slot_bytes;
    *hit = 0;
    return s;
}

/* Transient Windows flock/allocation blip: the store retries once when
 * WEX_READ_RETRY=1. Both readers consult this single accessor. */
int coli_v4_wex_flock_retry(void) {
    static __thread int tried = 0;
    const char *rt = getenv("WEX_READ_RETRY");
    (void)tried;
    return rt && *rt == '1';
}

/* Release a lease; filled says the claimed slot now holds the record (ignored for hits). */
void coli_v4_wex_crc_done(ColiExpertStore *store, int s, int filled) {
    V4WexCrc *c = store && store->state ? (V4WexCrc *)((V4ExpertStoreState *)store->state)->wex_crc : NULL;
    if (!c || s < 0 || s >= c->slots) return;
    V4WexCrcSlot *v = &c->slot[s];
    /* only the claimant of a slot that is not ready yet gets here with ready == 0,
     * and nobody else touches its bytes meanwhile */
    uint64_t sum = c->verify && filled && !__atomic_load_n(&v->ready, __ATOMIC_ACQUIRE)
        ? v4_wex_crc_sum(c->arena + (size_t)s * c->slot_bytes, v->bytes) : 0;
    pthread_mutex_lock(&c->mutex);
    if (!v->ready) {
        if (filled) { v->sum = sum; v->ready = 1; }
        else { if (v->cell >= 0) c->index[v->cell] = -1; v->cell = -1; }
    }
    if (v->refs > 0) v->refs--;
    pthread_mutex_unlock(&c->mutex);
}

/* Fill the cache with the most used compressed experts that neither VRAM nor
 * the RAM slabs hold (called by the tier prewarm, before any worker runs). */
static int v4_wex_crc_rank_compare(const void *a, const void *b) {
    const uint64_t *x = a, *y = b;   /* {usage, cell} pairs, usage descending */
    if (x[0] != y[0]) return x[0] > y[0] ? -1 : 1;
    return (x[1] > y[1]) - (x[1] < y[1]);
}
int coli_v4_wex_crc_prewarm(ColiExpertStore *store, const uint64_t *usage, const uint8_t *vram_bits) {
    if (!store || !store->state || !usage) return 0;
    V4ExpertStoreState *state = store->state;
    V4WexCrc *c = v4_wex_crc(state);
    if (!c) return 0;
    size_t cells = (size_t)c->cells, n = 0;
    uint64_t *ranked = malloc(cells * 2 * sizeof(uint64_t));
    if (!ranked) return -1;
    for (size_t i = 0; i < cells; i++) {
        const WexEntry *e = &state->wex_entries[i];
        if (!(e->flags == WEX_ENTRY_NEURAL_RANS || e->flags == WEX_ENTRY_NEURAL_RANS2) || !usage[i]) continue;
        if (vram_bits && (vram_bits[i >> 3] & (1u << (i & 7)))) continue;
        ColiExpertKey key = {(int)(i / state->experts_per_layer), (int)(i % state->experts_per_layer)};
        pthread_mutex_lock(&state->mutex);
        int in_ram = indexed_expert_slot(state, key) != NULL;
        pthread_mutex_unlock(&state->mutex);
        if (in_ram) continue;
        ranked[2 * n] = usage[i]; ranked[2 * n + 1] = i; n++;
    }
    qsort(ranked, n, 2 * sizeof(uint64_t), v4_wex_crc_rank_compare);
    size_t want = n < (size_t)c->slots ? n : (size_t)c->slots;
    double began = hot_now();
    int filled = 0;
    #pragma omp parallel for schedule(dynamic, 4) reduction(+:filled) num_threads(8)
    for (size_t k = 0; k < want; k++) {
        size_t i = (size_t)ranked[2 * k + 1];
        ColiExpertKey key = {(int)(i / state->experts_per_layer), (int)(i % state->experts_per_layer)};
        const WexEntry *e = &state->wex_entries[i];
        unsigned char *record; int hit = 0;
        int s = coli_v4_wex_crc_get(store, key, e, &record, &hit);
        if (s < 0) continue;
        int ok = hit || !v4_wex_read_at(v4_wex_pick(state, key.layer, key.expert),
                                        e->file_offset, record, e->total_bytes);
        coli_v4_wex_crc_done(store, s, ok);
        filled += ok && !hit;
    }
    free(ranked);
    pthread_mutex_lock(&c->mutex);
    c->prewarmed = (uint64_t)filled;
    c->hits = c->misses = c->evictions = c->refused = 0;   /* count only what decoding does */
    pthread_mutex_unlock(&c->mutex);
    fprintf(stderr, "[v4_wex] RAM cache prewarm: %d compressed experts (%.1f GiB) in %.1f s\n",
            filled, filled * (double)c->slot_bytes / 1073741824.0, hot_now() - began);
    return 0;
}

static void v4_wex_crc_report(V4ExpertStoreState *state) {
    V4WexCrc *c = state ? (V4WexCrc *)state->wex_crc : NULL;
    if (!c) return;
    fprintf(stderr, "v4_wex_ram_cache slots=%d slot_mib=%.2f hits=%llu misses=%llu evictions=%llu "
                    "refused=%llu prewarmed=%llu pinned=%d\n", c->slots, c->slot_bytes / 1048576.0,
            (unsigned long long)c->hits, (unsigned long long)c->misses, (unsigned long long)c->evictions,
            (unsigned long long)c->refused, (unsigned long long)c->prewarmed, c->pinned);
    if (c->verify)
        fprintf(stderr, "v4_wex_ram_cache_verify checks=%llu bad=%llu\n",
                (unsigned long long)c->verify_checks, (unsigned long long)c->verify_bad);
}
#endif

/* ==== deepseek_v4.c:15015-15027  store destroy (ROWS16): exit lines v4_wex_gpu_direct + v4_wex_ram_cache ==== */
#ifdef _WIN32
        if (__atomic_load_n(&v4_wex_direct_reads, __ATOMIC_RELAXED) ||
            __atomic_load_n(&v4_wex_direct_failures, __ATOMIC_RELAXED))
            fprintf(stderr,
                    "v4_wex_gpu_direct reads=%llu payload_bytes=%llu ram_skips=%llu "
                    "failures=%llu device_decodes=%llu\n",
                    (unsigned long long)__atomic_load_n(&v4_wex_direct_reads, __ATOMIC_RELAXED),
                    (unsigned long long)__atomic_load_n(&v4_wex_direct_bytes, __ATOMIC_RELAXED),
                    (unsigned long long)__atomic_load_n(&v4_wex_direct_ram_skips, __ATOMIC_RELAXED),
                    (unsigned long long)__atomic_load_n(&v4_wex_direct_failures, __ATOMIC_RELAXED),
                    (unsigned long long)coli_wex_cuda_device_decode_count());
        v4_wex_crc_report(store->state);
#endif

/* ==== deepseek_v4.c:15235-15243  coli_v4_expert_store_prewarm_tiers: RAM-cache prewarm after the VRAM and RAM tiers ==== */
#if defined(COLI_V4_GPU_TIER) && defined(_WIN32)
    {
        size_t crc_cells = (size_t)state->layers * state->experts_per_layer;
        uint8_t *vram = store->gpu ? calloc((crc_cells + 7) / 8, 1) : NULL;
        if (vram) coli_v4_gpu_mirror_bitmap(store, vram, state->layers, state->experts_per_layer, NULL);
        coli_v4_wex_crc_prewarm(store, policy->usage, vram);
        free(vram);
    }
#endif

