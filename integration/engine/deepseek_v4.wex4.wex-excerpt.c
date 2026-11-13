/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: KAIRA src/colibri/c/deepseek_v4.c at commit 24f7e72 (master, 2026-11-13), as committed.
 * Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
 * The WEX4 side of the DeepSeek expert store after the three kaira-* patches: opening a WEX4
 * container and pinning its root, the first-read SHA-256 of every record and its hash pool
 * (V4_WEX_SHA_ASYNC), the refusal of a legacy container under a pinned root, the sub-reads
 * (V4_WEX_SUBREADS) and the batched VRAM tier prewarm (V4_PREWARM_BATCH). Needs the level A +
 * direct excerpts (deepseek_v4.mode0-store / direct-store / direct-gpu) and the WNR3 one; the
 * GPU-side batch entry points it calls (coli_v4_gpu_expert_wex_*) are not quoted.
 */

/* ==== deepseek_v4.c:1871-1876  build_runtime_plan: V4_WEX_SHA_ASYNC=1 reserves 4 record copies of RAM for the hash pool ==== */
    /* V4_WEX_SHA_ASYNC=1: the hash pool keeps up to V4_SHA_THREADS (4) record copies for the rest
     * of the run (v4_sha.spare, reused, not freed after READY). Reserved, as above. */
    const char *sha_async = getenv("V4_WEX_SHA_ASYNC");
    if (sha_async && atoi(sha_async) == 1) {
        runtime_other += (uint64_t)4 * record;
        fprintf(stderr, "v4_sha_pool_reserve 4 x %.1f MiB record copies\n", record / (double)MIB);

/* ==== deepseek_v4.c:15032-15124  v4_wex4_load_manifest: WEX4 header page and index (c/wex4.h), the root pinned with COLI_WEX_ROOT, the record hashes kept behind the entry table ==== */
/* WEX4 container (wex4.h): validate the header page and the index, then hand the
 * caller what a WEX3 container gives it: the layers x experts WexEntry table and a
 * header block with the legacy fields (version 3, flags, slab size). The record
 * bytes and offsets are those of WEX3, so nothing after the open changes. The
 * block keeps the container's root at byte 4064: equal blocks <=> same container.
 * COLI_WEX_ROOT (64 hex digits) pins the root. */
static int v4_wex4_load_manifest(HANDLE file, uint64_t size, int layers, int experts,
                                  unsigned char *block, WexEntry **entries_out,
                                  size_t *table_span) {
    Wex4Header h;
    Wex4View v;
    Wex4Record rec;
    unsigned char *index = NULL, *model = NULL;
    WexEntry *entries = NULL;
    uint64_t span, slab = 0;
    uint32_t r, flags = 0, model_blob = 0;
    const char *pin = getenv("COLI_WEX_ROOT");
    if (wex4_parse_header(block, size, &h) || h.layers != (uint32_t)layers ||
        h.experts != (uint32_t)experts) return -1;
    if (pin && *pin) {
        char hex[65];
        for (r = 0; r < 32u; r++) snprintf(hex + r * 2u, 3, "%02x", h.root[r]);
        if (_stricmp(pin, hex)) {
            fprintf(stderr, "[v4_wex] WEX4 root %s is not the pinned COLI_WEX_ROOT\n", hex);
            return -1;
        }
    }
    span = wex4_align((uint64_t)h.layers * h.experts * sizeof(WexEntry));
    index = (unsigned char *)_aligned_malloc((size_t)wex4_align(h.index_bytes), WEX_ALIGN);
    /* the record hashes sit right behind the table (span bytes), 32 per slot */
    entries = (WexEntry *)_aligned_malloc((size_t)(span + wex4_align((uint64_t)h.layers * h.experts * 32u)), WEX_ALIGN);
    if (!index || !entries ||
        v4_wex_read_exact(file, index, (DWORD)wex4_align(h.index_bytes), h.index_offset) ||
        wex4_open(&h, index, (size_t)wex4_align(h.index_bytes), &v)) goto fail;
    memset(entries, 0, (size_t)(span + wex4_align((uint64_t)h.layers * h.experts * 32u)));
    for (r = 0; r < h.records; r++) {
        size_t slot;
        wex4_get_record(&v, r, &rec);
        if (!(rec.flags & WEX4_REC_EXPERT)) continue;
        slot = (size_t)(rec.owner >> 16) * h.experts + (rec.owner & 0xffffu);
        if (wex4_expert_entry(&v, r, &entries[slot])) goto fail;
        memcpy((unsigned char *)entries + span + slot * 32u, rec.sha, 32u);
        if (wex4_align(rec.decoded) > slab) slab = wex4_align(rec.decoded);
        if (rec.flags & WEX4_REC_VALUE_EXACT) flags |= WEX_HEADER_CANONICAL_ZERO;
        if (rec.codec == WEX4_CODEC_WNR3) {
            flags |= WEX_HEADER_NEURAL_RANS3;
            if (model_blob && model_blob != rec.blob) goto fail;   /* one registered model */
            model_blob = rec.blob;
        }
    }
    /* the store applies canonical zero per container: refuse a mixed file */
    for (r = 0; (flags & WEX_HEADER_CANONICAL_ZERO) && r < h.records; r++) {
        wex4_get_record(&v, r, &rec);
        if (rec.codec == WEX4_CODEC_WNR3 && !(rec.flags & WEX4_REC_VALUE_EXACT)) goto fail;
    }
    if (model_blob) {
        Wex4Blob blob;
        unsigned char digest[32];
        uint32_t registered = 0;
        uint64_t at;
        wex4_get_blob(&v, model_blob - 1u, &blob);
        model = (unsigned char *)_aligned_malloc((size_t)wex4_align(blob.bytes), WEX_ALIGN);
        if (!model || v4_wex_read_exact(file, model, (DWORD)wex4_align(blob.bytes), blob.offset)) goto fail;
        wex4_sha256(model, (size_t)blob.bytes, digest);
        if (memcmp(digest, blob.sha, 32u)) goto fail;
        for (at = blob.bytes; at < wex4_align(blob.bytes); at++) if (model[at]) goto fail;
        if (coli_wex_register_model(model, (size_t)blob.bytes, &registered) || registered != blob.id) goto fail;
        if (!coli_wex_cuda_nr2_available() || !coli_wex_cuda_nr3_available()) {
            fprintf(stderr, "[v4_wex] WNR3 container needs kaira_wex_cuda.dll with WNR3 support "
                            "(helper feature 16) next to the engine\n");
            goto fail;
        }
        if (coli_wex_container_refused(flags, "v4_wex")) goto fail;
    }
    {
        WexHeader legacy = {WEX_MAGIC_V3, 3u, h.layers, h.experts, slab, WEX_ALIGN, flags, 0u};
        memset(block, 0, WEX_ALIGN);
        uint32_t marker = WEX4_MAGIC;
        memcpy(block, &legacy, sizeof(legacy));
        memcpy(block + 56u, &marker, 4u);   /* tells the caller the hashes are behind the table */
        memcpy(block + WEX4_HEADER_HASHED, h.root, 32u);
    }
    _aligned_free(index);
    _aligned_free(model);
    *entries_out = entries;
    *table_span = (size_t)span;
    return 0;
fail:
    _aligned_free(index);
    _aligned_free(model);
    _aligned_free(entries);
    return -1;
}

/* ==== deepseek_v4.c:15126-15184  first-read SHA-256 (COLI_WEX_VERIFY): seen bits per slot and read layout, the check, the inline path ==== */
/* WEX4 first-read check (COLI_WEX_VERIFY, on by default when COLI_WEX_ROOT pins the root): the
 * stored bytes of a record are hashed the first time they are read in this process and compared with
 * the index, which the pinned root covers. 1 = fine (or nothing to check), 0 = reject the record. */
/* first-read SHA-256 counters (wexprof and the tier prewarm print them): records hashed, their
 * hash time on any thread, and the time a reader waited for hashing (inline hash, copy for the
 * hash pool, join) = the hash's cost on the critical path */
static unsigned long long v4_sha_records, v4_sha_hash_ns, v4_sha_crit_ns;
static double v4_now_mono(void) {   /* this unit's copy of the runtime's clock */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void v4_sha_crit(double seconds) {
    __atomic_fetch_add(&v4_sha_crit_ns, (unsigned long long)(seconds * 1e9), __ATOMIC_RELAXED);
}

/* -1 rejected before, 1 passed before (or nothing to check), 0 due */
static int v4_wex_first_read_seen(V4ExpertStoreState *state, int layer, int expert, int striped) {
    if (!state->wex_sha) return 1;
    unsigned char layout = striped ? 2u : 1u;   /* a striped read takes its tail from another drive */
    unsigned char seen = __atomic_load_n(&state->wex_seen[(size_t)layer * state->experts_per_layer + expert],
                                         __ATOMIC_RELAXED);
    if (seen & layout) return 1;
    return seen & (layout << 2) ? -1 : 0;   /* rejected: that layout's stored bytes do not change, so not hashed again */
}

/* the check itself, on whichever thread hashes */
static int v4_wex_first_read_check(V4ExpertStoreState *state, int layer, int expert,
                                   const WexEntry *entry, const unsigned char *record, int striped) {
    unsigned char digest[32];
    size_t slot = (size_t)layer * (size_t)state->experts_per_layer + (size_t)expert;
    unsigned char layout = striped ? 2u : 1u;
    int seen = v4_wex_first_read_seen(state, layer, expert, striped);
    if (seen) return seen > 0;
    double t0 = v4_now_mono();
    wex4_sha256(record, entry->total_bytes, digest);
    __atomic_fetch_add(&v4_sha_hash_ns, (unsigned long long)((v4_now_mono() - t0) * 1e9), __ATOMIC_RELAXED);
    __atomic_fetch_add(&v4_sha_records, 1ULL, __ATOMIC_RELAXED);
    if (memcmp(digest, state->wex_sha + slot * 32u, 32u)) {
        static int said;
        if (!__atomic_exchange_n(&said, 1, __ATOMIC_RELAXED))
            fprintf(stderr, "[v4_wex] record of layer %d expert %d does not match its SHA-256; rejected "
                            "for this process\n", layer, expert);
        __atomic_fetch_or(&state->wex_seen[slot], (unsigned char)(layout << 2), __ATOMIC_RELAXED);
        return 0;
    }
    __atomic_fetch_or(&state->wex_seen[slot], layout, __ATOMIC_RELAXED);
    return 1;
}

/* the inline check: its whole time is on the reader's critical path */
static int v4_wex_first_read_ok(V4ExpertStoreState *state, int layer, int expert,
                                const WexEntry *entry, const unsigned char *record, int striped) {
    if (v4_wex_first_read_seen(state, layer, expert, striped)) return v4_wex_first_read_check(state, layer, expert, entry, record, striped);
    double t0 = v4_now_mono();
    int ok = v4_wex_first_read_check(state, layer, expert, entry, record, striped);
    v4_sha_crit(v4_now_mono() - t0);
    return ok;
}

/* ==== deepseek_v4.c:15186-15332  V4_WEX_SHA_ASYNC=1: the hash pool, its jobs, and the deferred verdicts drained before READY ==== */
/* V4_WEX_SHA_ASYNC=1: a due first-read hash goes to a small pool of hash threads while the
 * reader decodes/uploads the same record; the reader joins before it publishes the record
 * (v4_wex_first_read_finish), so nothing uses bytes whose hash has not passed. A host-slab
 * read hands the pool a copy (the in-place decode overwrites the record). The tier prewarm
 * defers the join to its end, before READY (v4_wex_sha_drain). A job goes to the pool only
 * when a hash thread is free (inflight < threads), so none ever queues behind another; with
 * every thread busy the reader hashes inline, as with the switch off. */
enum { V4_SHA_THREADS = 4 };   /* so at most 4 copies of an 18.8 MB V4.1 record */
_Static_assert(V4_SHA_THREADS == 4, "build_runtime_plan reserves 4 record copies for the hash pool");
typedef struct V4ShaJob {
    V4ExpertStoreState *state;
    const WexEntry *entry;
    const unsigned char *record;
    unsigned char *copy;   /* a pool buffer holding the record, or NULL: the reader keeps it intact */
    int layer, expert, striped, done, ok;
    struct V4ShaJob *next;
} V4ShaJob;
static struct {
    pthread_mutex_t mutex;
    pthread_cond_t work, done;
    V4ShaJob *head, *tail;
    int threads, tried, inflight, spares;
    unsigned char *spare[V4_SHA_THREADS];
    size_t spare_bytes;
    V4ShaJob **deferred;
    size_t deferred_n, deferred_cap;
} v4_sha = {.mutex = PTHREAD_MUTEX_INITIALIZER, .work = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER};
static __thread int v4_sha_defer_tls;   /* set by the tier prewarm: joins wait for its drain */

static void *v4_sha_worker(void *arg) {
    (void)arg;
    pthread_mutex_lock(&v4_sha.mutex);
    for (;;) {
        while (!v4_sha.head) pthread_cond_wait(&v4_sha.work, &v4_sha.mutex);
        V4ShaJob *j = v4_sha.head;
        v4_sha.head = j->next;
        if (!v4_sha.head) v4_sha.tail = NULL;
        pthread_mutex_unlock(&v4_sha.mutex);
        int ok = v4_wex_first_read_check(j->state, j->layer, j->expert, j->entry,
                                          j->copy ? j->copy : j->record, j->striped);
        pthread_mutex_lock(&v4_sha.mutex);
        if (j->copy) v4_sha.spare[v4_sha.spares++] = j->copy;
        j->copy = NULL;
        j->ok = ok;
        j->done = 1;
        v4_sha.inflight--;
        pthread_cond_broadcast(&v4_sha.done);
    }
    return NULL;
}

/* 0: rejected (before, or by the inline check). 1: passed, or *job (NULL on entry) hashes it on
 * the pool; with copy set the record may change as soon as this returns. */
static int v4_wex_first_read_start(V4ExpertStoreState *state, int layer, int expert, const WexEntry *entry,
                                   const unsigned char *record, int striped, int copy, V4ShaJob **job) {
    int seen = v4_wex_first_read_seen(state, layer, expert, striped);
    if (seen) return seen > 0;
    if (!state->wex_sha_async) return v4_wex_first_read_ok(state, layer, expert, entry, record, striped);
    double t0 = v4_now_mono();
    V4ShaJob *j = calloc(1, sizeof(*j));
    unsigned char *buffer = NULL;
    pthread_mutex_lock(&v4_sha.mutex);
    if (!v4_sha.tried) {
        v4_sha.tried = 1;
        for (int i = 0; i < V4_SHA_THREADS; i++) {
            pthread_t thread;
            if (!pthread_create(&thread, NULL, v4_sha_worker, NULL)) {
                pthread_detach(thread);
                v4_sha.threads++;
            }
        }
        fprintf(stderr, "[v4_wex] V4_WEX_SHA_ASYNC=1: first-read SHA-256 on %d hash threads, "
                        "overlapped with the decode; records published after their hash passes\n", v4_sha.threads);
    }
    int free_thread = j && v4_sha.inflight < v4_sha.threads;
    if (free_thread && copy) {
        if (!v4_sha.spare_bytes) v4_sha.spare_bytes = (size_t)state->record_bytes;   /* fixed per model */
        if (entry->total_bytes <= v4_sha.spare_bytes)
            buffer = v4_sha.spares ? v4_sha.spare[--v4_sha.spares] : malloc(v4_sha.spare_bytes);
    }
    int pooled = free_thread && (!copy || buffer);
    if (pooled) v4_sha.inflight++;
    pthread_mutex_unlock(&v4_sha.mutex);
    if (!pooled) {   /* no free thread, no memory: the inline check, as with the switch off */
        free(j);
        v4_sha_crit(v4_now_mono() - t0);
        return v4_wex_first_read_ok(state, layer, expert, entry, record, striped);
    }
    if (buffer) memcpy(buffer, record, entry->total_bytes);
    *j = (V4ShaJob){state, entry, record, buffer, layer, expert, striped, 0, 0, NULL};
    pthread_mutex_lock(&v4_sha.mutex);
    if (v4_sha.tail) v4_sha.tail->next = j; else v4_sha.head = j;
    v4_sha.tail = j;
    pthread_cond_signal(&v4_sha.work);
    int deferred = 0;
    if (v4_sha_defer_tls && copy) {   /* the record is the pool's own copy: the reader may go on */
        if (v4_sha.deferred_n == v4_sha.deferred_cap) {
            size_t cap = v4_sha.deferred_cap ? 2 * v4_sha.deferred_cap : 1024;
            V4ShaJob **grown = realloc(v4_sha.deferred, cap * sizeof(*grown));
            if (grown) { v4_sha.deferred = grown; v4_sha.deferred_cap = cap; }
        }
        if (v4_sha.deferred_n < v4_sha.deferred_cap) {
            v4_sha.deferred[v4_sha.deferred_n++] = j;
            deferred = 1;
        }
    }
    pthread_mutex_unlock(&v4_sha.mutex);
    v4_sha_crit(v4_now_mono() - t0);
    if (!deferred) *job = j;
    return 1;
}

/* joins *job (NULL: nothing pending, 1) and clears it: 1 when the hash passed */
static int v4_wex_first_read_finish(V4ShaJob **job) {
    V4ShaJob *j = *job;
    if (!j) return 1;
    *job = NULL;
    double t0 = v4_now_mono();
    pthread_mutex_lock(&v4_sha.mutex);
    while (!j->done) pthread_cond_wait(&v4_sha.done, &v4_sha.mutex);
    pthread_mutex_unlock(&v4_sha.mutex);
    v4_sha_crit(v4_now_mono() - t0);
    int ok = j->ok;
    free(j);
    return ok;
}

/* the deferred verdicts: waits for every one, puts the failed records' keys in rejected[]
 * (up to max) and returns their count */
static int v4_wex_sha_drain(ColiExpertKey *rejected, int max) {
    int count = 0;
    pthread_mutex_lock(&v4_sha.mutex);
    V4ShaJob **list = v4_sha.deferred;
    size_t n = v4_sha.deferred_n;
    v4_sha.deferred = NULL;
    v4_sha.deferred_n = v4_sha.deferred_cap = 0;
    pthread_mutex_unlock(&v4_sha.mutex);
    for (size_t i = 0; i < n; i++) {
        ColiExpertKey key = {list[i]->layer, list[i]->expert};
        if (!v4_wex_first_read_finish(&list[i]) && count < max) rejected[count++] = key;
    }
    free(list);
    pthread_mutex_lock(&v4_sha.mutex);   /* the prewarm's copies go back to the system */
    while (!v4_sha.inflight && v4_sha.spares) free(v4_sha.spare[--v4_sha.spares]);
    pthread_mutex_unlock(&v4_sha.mutex);
    return count;
}

/* ==== deepseek_v4.c:15357-15363  v4_wex_load_manifest: a pinned root refuses a container that is not WEX4 (W06) ==== */
    {   /* a pinned root refuses a legacy file: it has no root, and no first-read SHA-256 (W06) */
        const char *pin = getenv("COLI_WEX_ROOT");
        if (pin && *pin) {
            fprintf(stderr, "[v4_wex] a container that is not WEX4 is not the pinned COLI_WEX_ROOT\n");
            goto fail;
        }
    }

/* ==== deepseek_v4.c:15516-15530  store open: COLI_WEX_VERIFY (default on when the root is pinned) and V4_WEX_SHA_ASYNC ==== */
            state->wex_flags = hdr->flags;
            {
                uint32_t marker;
                const char *verify = getenv("COLI_WEX_VERIFY"), *pin = getenv("COLI_WEX_ROOT");
                memcpy(&marker, wex_header_block + 56u, 4u);
                if (marker == WEX4_MAGIC && (verify ? atoi(verify) != 0 : pin && *pin)) {
                    state->wex_seen = (unsigned char *)calloc((size_t)hdr->num_layers * hdr->num_experts, 1u);
                    if (state->wex_seen) state->wex_sha = (const unsigned char *)state->wex_entries + wex_table_span;
                    const char *async = getenv("V4_WEX_SHA_ASYNC");
                    state->wex_sha_async = state->wex_sha && async && atoi(async) == 1;
                    fprintf(stderr, "[v4_wex] WEX4 first-read SHA-256 of every record: %s%s\n",
                            state->wex_sha ? "on" : "off (no memory)",
                            state->wex_sha_async ? ", off the critical path (V4_WEX_SHA_ASYNC=1)" : "");
                }
            }

/* ==== deepseek_v4.c:16475-16545  V4_WEX_SUBREADS: sub-read count and cuts, the split read across primary and mirror ==== */
/* V4_WEX_SUBREADS=N (2..V4_WEX_SUBS; unset, 0 or 1 = today: one read a part): each drive's part of a
 * WEX record read goes out as N sub-reads in flight at once, cut on WEX_ALIGN pages, so the drive sees
 * a deeper queue (repo-scout-2 R6). The bytes land in place: verify and decode run once all landed.
 * With a mirror open the host path and the batches also cut the record across two drives where
 * V4_WEX_STRIPE cuts it, so its bytes are the striped layout's and that layout's first-read SHA-256
 * covers them; the single GPU-direct read stays on its routed drive (its caller checks layout whole).
 * Read at every call, as the tests switch it. */
enum { V4_WEX_SUBS = 16 };
#ifdef COLI_V4_TEST_HOOKS
static uint64_t v4_wex_sub_issued;   /* tests: ReadFile calls of WEX record reads */
uint64_t coli_v4_test_wex_sub_reads(void) { return __atomic_load_n(&v4_wex_sub_issued, __ATOMIC_RELAXED); }
#define V4_WEX_SUB_ISSUED() __atomic_fetch_add(&v4_wex_sub_issued, UINT64_C(1), __ATOMIC_RELAXED)
#else
#define V4_WEX_SUB_ISSUED() ((void)0)
#endif
static int v4_wex_subreads(void) {
    const char *text = getenv("V4_WEX_SUBREADS");
    int n = text && *text ? atoi(text) : 1;
    return n < 1 ? 1 : n > V4_WEX_SUBS ? V4_WEX_SUBS : n;
}

/* sub-reads of a part of `bytes`: n, or one a page when it has fewer pages */
static int v4_wex_sub_count(uint32_t bytes, int n) {
    uint32_t pages = bytes / WEX_ALIGN;
    return pages >= (uint32_t)n ? n : pages ? (int)pages : 1;
}

/* where sub-read k of n of a part of `bytes` starts (k = n: the part's end); every cut on a page */
static uint32_t v4_wex_sub_at(uint32_t bytes, int n, int k) {
    return k >= n ? bytes : (uint32_t)((uint64_t)(bytes / WEX_ALIGN) * (uint64_t)k / (uint64_t)n) * WEX_ALIGN;
}

/* bytes of a record at offset into dst: [0, head) from file, [head, bytes) from tail_file, each part as
 * up to subs sub-reads in flight at once. 0 when every byte landed; all issued reads are waited for */
static int v4_wex_read_split(HANDLE file, HANDLE tail_file, uint64_t offset, unsigned char *dst,
                             uint32_t bytes, uint32_t head, int subs) {
    static __thread HANDLE events[2 * V4_WEX_SUBS];
    OVERLAPPED ov[2 * V4_WEX_SUBS];
    HANDLE on[2 * V4_WEX_SUBS];
    DWORD want[2 * V4_WEX_SUBS];
    int pending[2 * V4_WEX_SUBS], n = 0, failed = 0;
    for (int part = 0; part < 2 && !failed; part++) {
        uint32_t at = part ? head : 0u, len = (part ? bytes : head) - at;
        int k_n = len ? v4_wex_sub_count(len, subs) : 0;
        for (int k = 0; k < k_n && !failed; k++) {
            uint32_t s = at + v4_wex_sub_at(len, k_n, k);
            uint64_t where = offset + s;
            if (!events[n] && !(events[n] = CreateEventA(NULL, TRUE, FALSE, NULL))) { failed = 1; break; }
            memset(&ov[n], 0, sizeof(ov[n]));
            ov[n].Offset = (DWORD)(where & 0xFFFFFFFFULL);
            ov[n].OffsetHigh = (DWORD)((where >> 32) & 0xFFFFFFFFULL);
            ov[n].hEvent = events[n];
            ResetEvent(events[n]);
            on[n] = part ? tail_file : file;
            want[n] = at + v4_wex_sub_at(len, k_n, k + 1) - s;
            DWORD got = 0;
            V4_WEX_SUB_ISSUED();
            if (ReadFile(on[n], dst + s, want[n], &got, &ov[n])) {
                pending[n] = 0;
                failed |= got != want[n];
            } else if (GetLastError() == ERROR_IO_PENDING) pending[n] = 1;
            else { failed = 1; break; }
            n++;
        }
    }
    for (int i = 0; i < n; i++) {
        DWORD got = 0;
        if (pending[i]) failed |= !GetOverlappedResult(on[i], &ov[i], &got, TRUE) || got != want[i];
    }
    return failed ? -1 : 0;
}

/* ==== deepseek_v4.c:16732-16735  v4_wex_read_at: every WEX record read goes through the sub-read path ==== */
/* bytes at offset from one drive, as V4_WEX_SUBREADS sub-reads (one when unset) */
static int v4_wex_read_at(HANDLE file, uint64_t offset, unsigned char *dst, uint32_t bytes) {
    return v4_wex_read_split(file, file, offset, dst, bytes, bytes, v4_wex_subreads());
}

/* ==== deepseek_v4.c:18514-18552  V4_PREWARM_BATCH=1: the batched tier prewarm, two batches deep ==== */
/* V4_PREWARM_BATCH=1: the default loop's records in its order, COLI_V4_WEX_BATCH_MAX at a time
 * through the WEX batch path, two batches deep: the next batch's mirrors are reserved and its
 * reads issued (both drives, each record from its routed drive) before the batch ahead is
 * waited for, hashed on parallel threads and decoded into its mirrors in one call. A record
 * is published only after its first-read SHA-256 passed; one the batch cannot take, or that
 * failed in it, takes the default loop's per-record path. */
static void hot_tier_prewarm_batched(ColiExpertStore *store, V4HotPolicy *policy,
                                     const V4TierRank *ranked, size_t count, V4TierFill *fill) {
    int per = COLI_V4_WEX_BATCH_MAX, stop = 0, lane = 0;
#ifdef COLI_V4_TEST_HOOKS
    if (coli_v4_test_prewarm_batch > 0 && coli_v4_test_prewarm_batch < per) per = coli_v4_test_prewarm_batch;
#endif
    void *pending = NULL;
    size_t at = 0;
    while (pending || (!stop && at < count)) {
        void *next = NULL;
        int single = 0;
        if (!stop && at < count) {
            ColiExpertKey keys[COLI_V4_WEX_BATCH_MAX];
            int want = count - at < (size_t)per ? (int)(count - at) : per, taken = 0;
            for (int j = 0; j < want; j++) keys[j] = ranked[at + j].key;
            next = coli_v4_gpu_expert_wex_warm_reserve(store, keys, want, lane, &taken);
            if (next) {
                coli_v4_gpu_expert_wex_start(next);
                fill->batches++;
                lane ^= 1;   /* the batch ahead still reads on the other event set */
            }
            at += (size_t)taken;
            single = taken < want;
        }
        if (pending) stop = hot_tier_batch_finish(store, policy, pending, stop, fill);
        pending = next;
        if (single && !stop) {   /* the record the reservation stopped at, after the ones ahead */
            stop = hot_tier_warm_host(store, policy, ranked[at].key, NULL, 0);
            if (!stop) hot_tier_tally(fill, ranked[at].frequency);
            at++;
        }
    }
}

/* ==== deepseek_v4.c:18574-18603  coli_v4_expert_store_prewarm_tiers: COLI_V4_TIER_PREWARM_AHEAD and the V4_PREWARM_BATCH choice ==== */
        /* COLI_V4_TIER_PREWARM_AHEAD=N (2..8, at most a layer's slots): the next N ranked records
         * are read, first-read hashed and decoded on parallel threads, the uploads stay serial and
         * in rank order. Off by default (1 = the one-at-a-time loop): each extra thread's GPU
         * decode allocates a thread-local CUDA workspace while the mirrors still fill, and the
         * fill stops on free VRAM, so it can mirror fewer experts. Compare vram_experts=. */
        enum { AHEAD_MAX = 8 };
        const char *ahead_env = getenv("COLI_V4_TIER_PREWARM_AHEAD");
        int ahead = ahead_env ? atoi(ahead_env) : 1;
        if (ahead > AHEAD_MAX) ahead = AHEAD_MAX;
        if (ahead > state->slots_per_layer) ahead = state->slots_per_layer;
        if (ahead < 1) ahead = 1;
        /* V4_PREWARM_BATCH=1 (default off): hot_tier_prewarm_batched instead of the loop below */
        const char *batch_env = getenv("V4_PREWARM_BATCH");
        int batched = batch_env && atoi(batch_env) != 0;
#ifdef _WIN32
        if (batched && !coli_v4_gpu_wex_batch_enabled()) {
            fprintf(stderr, "v4_tier_prewarm V4_PREWARM_BATCH needs V4_WEX_BATCH=1 and "
                            "COLI_WEX_GPU_DIRECT>=1; one record at a time\n");
            batched = 0;
        }
        if (batched) {
            V4TierFill fill = {0};
            double began = hot_now();
            hot_tier_prewarm_batched(store, policy, ranked, count, &fill);
            uploaded = fill.uploaded;
            highest = fill.highest;
            lowest = fill.lowest;
            fprintf(stderr, "v4_tier_prewarm_batch batches=%d direct=%d host=%d seconds=%.2f\n",
                    fill.batches, fill.direct, fill.uploaded - fill.direct, hot_now() - began);
        }
