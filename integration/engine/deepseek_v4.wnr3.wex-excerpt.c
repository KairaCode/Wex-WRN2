/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4.c and c/deepseek_v4_internal.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: the maintainer's engine tree, src/colibri/c/deepseek_v4.c and deepseek_v4_internal.h
 * on its WNR3 branch as of 2026-11-10, as committed.
 * Line numbers are of that version; the live tree moves them. Code between ranges is omitted.
 * WNR3 (header flag 32, entry codec 5) and canonical zero (header flag 64) in the engine's own
 * expert store: container checks at open (helper feature 16, coli_wex_container_refused in
 * c/wex_decode.c), the record owner/profile check (wex_nr3_owner_ok in c/wex_format.h)
 * before every decode (host slab, GPU-direct fill, batch), a RAM-cache record that is rejected
 * leaves the cache, codec 5 in cache sizing / prewarm / batch, and the lab-only
 * V4_LAB_CANON_ZERO switch (WNR3 gate G7, never set by the application). Needs the level A + direct
 * excerpts (deepseek_v4.mode0-store / direct-store / direct-gpu).
 */

/* ==== deepseek_v4_internal.h:1314-1318  coli_v4_expert_store_wex_flags prototype ==== */
/* the record when it is Neural-rANS, fits capacity and RAM does not hold it */
const WexEntry *coli_v4_expert_store_wex_entry(ColiExpertStore *store, ColiExpertKey key,
                                               size_t capacity);
/* the container's header flags (wex_nr3_owner_ok's canonical-zero check) */
uint32_t coli_v4_expert_store_wex_flags(ColiExpertStore *store);

/* ==== deepseek_v4.c:13306-13312  V4ExpertStoreState: wex_flags (container header flags) ==== */
    WexEntry *wex_entries;
    uint32_t wex_flags;   /* container header flags (the WNR3 records' canonical-zero check) */
#ifdef _WIN32
    void *wex_crc;        /* compressed record cache (COLI_WEX_RAM_CACHE_GB) */
    int wex_crc_tried;
#endif
} V4ExpertStoreState;

/* ==== deepseek_v4.c:13945-13973  v4_wex_load_manifest: WNR2/WNR3 model check, flag 32 needs helper feature 16, coli_wex_container_refused (flag 64 under DSV4_CUDA_MOE_MASKED) ==== */
    if (header->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3)) {
        /* WNR2/WNR3: the container's model must load and match the id its
         * header names before any record can be decoded */
        uint64_t model_offset;
        uint32_t model_bytes, model_id, registered = 0;
        uint64_t aligned_model;
        unsigned char *model;
        if (wex_nr2_model_location(header, block, &model_offset, &model_bytes, &model_id) ||
            wex_aligned_size(model_bytes, &aligned_model) || aligned_model > MAXDWORD) goto fail;
        model = (unsigned char *)_aligned_malloc((size_t)aligned_model, WEX_ALIGN);
        if (!model) goto fail;
        int bad = v4_wex_read_exact(file, model, (DWORD)aligned_model, model_offset) ||
                  coli_wex_register_model(model, model_bytes, &registered) || registered != model_id;
        _aligned_free(model);
        if (bad) goto fail;
        /* without the helper every WNR2 record would go to the CPU decoder (~0.3 s
         * per expert): refuse the container so the caller keeps the safetensors path */
        if (!coli_wex_cuda_nr2_available()) {
            fprintf(stderr, "[v4_wex] WNR2 container needs kaira_wex_cuda.dll with WNR2 support "
                            "(helper feature 4) next to the engine\n");
            goto fail;
        }
        if ((header->flags & WEX_HEADER_NEURAL_RANS3) && !coli_wex_cuda_nr3_available()) {
            fprintf(stderr, "[v4_wex] WNR3 container needs kaira_wex_cuda.dll with WNR3 support "
                            "(helper feature 16) next to the engine\n");
            goto fail;
        }
        if (coli_wex_container_refused(header->flags, "v4_wex")) goto fail;
    }

/* ==== deepseek_v4.c:14077-14088  coli_deepseek_v4_expert_store_open: keep the header flags, log WNR2 / WNR3 / WNR3 value-exact ==== */
                  &wex_header_block, &state->wex_entries,
                  &wex_table_span, &wex_file_bytes)) {
            const WexHeader *hdr = (const WexHeader *)wex_header_block;
            state->wex_handle = hWex;
            state->wex_flags = hdr->flags;
            fprintf(stderr, "[v4_wex] Container v%u%s active: %s (%u layers, %u experts, %llu decoded bytes/expert, direct_io=on)\n",
                    hdr->version,
                    hdr->flags & WEX_HEADER_NEURAL_RANS3
                        ? (hdr->flags & WEX_HEADER_CANONICAL_ZERO ? " WNR3 value-exact" : " WNR3")
                        : hdr->flags & WEX_HEADER_NEURAL_RANS2 ? " WNR2" : "",
                    wex_path, hdr->num_layers, hdr->num_experts,
                    (unsigned long long)hdr->aligned_expert_bytes);

/* ==== deepseek_v4.c:14455-14461  test hook coli_v4_test_wex_active (tests/test_v4_wex_nr3.c) ==== */
#if defined(COLI_V4_TEST_HOOKS) && defined(_WIN32)
/* tests/test_v4_wex_nr3.c: 1 when the store serves a WEX container */
int coli_v4_test_wex_active(ColiExpertStore *store) {
    V4ExpertStoreState *state = store ? store->state : NULL;
    return state && state->wex_entries && state->wex_handle != INVALID_HANDLE_VALUE;
}
#endif

/* ==== deepseek_v4.c:14680-14720  V4_LAB_CANON_ZERO (lab only): v4_read_expert_record wraps the unchanged reader (now v4_read_expert_record_io) ==== */
static int v4_read_expert_record_io(V4ExpertStoreState *state,
                                    const V4ExpertRecord *record,
                                    V4ExpertSlot *slot, int rep,
                                    uint64_t *payload_bytes);

/* V4_LAB_CANON_ZERO=1: LAB ONLY (WNR3 gate G7, never set by KAIRA). Every FP4 -0 nibble
 * (0x8) of an expert slab becomes +0 right after the read (WEX1, WNR2 or safetensors), so
 * a WEX1 run computes on exactly what a WNR3 canonical-zero container decodes to. Both
 * callers (lease load, read_private) come through here; the GPU-direct fill does not, so
 * the proof runs with COLI_WEX_GPU_DIRECT=0 on a WEX1 container (raw records only). */
static int v4_read_expert_record(V4ExpertStoreState *state,
                                 const V4ExpertRecord *record,
                                 V4ExpertSlot *slot, int rep,
                                 uint64_t *payload_bytes) {
    int result = v4_read_expert_record_io(state, record, slot, rep, payload_bytes);
#ifdef _WIN32   /* wex_format.h (wex_nr3_canonical_zero) is a Windows-only include here */
    static int canon = -1;
    static uint64_t experts, nibbles;
    if (canon < 0) {
        canon = getenv("V4_LAB_CANON_ZERO") && atoi(getenv("V4_LAB_CANON_ZERO")) ? 1 : 0;
        if (canon) fprintf(stderr, "v4_lab_canon_zero: ON (lab only): expert -0 nibbles -> +0 after every read\n");
    }
    if (result || !canon) return result;
    uint64_t n = wex_nr3_canonical_zero(slot->slab + record->scale_bytes, (size_t)record->weight_bytes);
    uint64_t e = __atomic_add_fetch(&experts, 1, __ATOMIC_RELAXED);
    uint64_t total = __atomic_add_fetch(&nibbles, n, __ATOMIC_RELAXED);
    if (e == 1 || e % 4096 == 0)
        fprintf(stderr, "v4_lab_canon_zero: experts=%llu nibbles=%llu (last L%d E%d: %llu)\n",
                (unsigned long long)e, (unsigned long long)total, record->layer, record->expert,
                (unsigned long long)n);
#endif
    return result;
}

static int v4_read_expert_record_io(V4ExpertStoreState *state,
                                    const V4ExpertRecord *record,
                                    V4ExpertSlot *slot, int rep,
                                    uint64_t *payload_bytes) {
    if (payload_bytes) *payload_bytes = record->record_bytes;
#ifdef _WIN32
    if (state->wex_entries != NULL && state->wex_handle != INVALID_HANDLE_VALUE) {

/* ==== deepseek_v4.c:14742-14758  v4_read_expert_record_io, host slab path: owner/profile check before the decode ==== */
            BOOL ok = ReadFile(hFile, slot->slab + read_offset,
                               entry->total_bytes, &bytes_read, &ov);
            if (!ok && GetLastError() == ERROR_IO_PENDING) {
                ok = GetOverlappedResult(hFile, &ov, &bytes_read, TRUE);
            }
            if (ok && bytes_read == entry->total_bytes &&
                wex_nr3_owner_ok(entry, slot->slab + read_offset, (uint32_t)record->layer,
                                 (uint32_t)record->expert, state->wex_flags) &&
                !coli_wex_decode_record(entry, slot->slab,
                                     (size_t)record->record_bytes)) {
                __atomic_fetch_add(&v4_direct_reads, UINT64_C(1), __ATOMIC_RELAXED);
                __atomic_fetch_add(&v4_direct_flock_reads, UINT64_C(1), __ATOMIC_RELAXED);
                __atomic_fetch_add(&v4_direct_payload_bytes, entry->total_bytes, __ATOMIC_RELAXED);
                if (payload_bytes) *payload_bytes = entry->total_bytes;
                return 0;
            }
            __atomic_fetch_add(&v4_direct_fallbacks, UINT64_C(1), __ATOMIC_RELAXED);

/* ==== deepseek_v4.c:14861-14886  coli_v4_expert_store_wex_entry accepts codec 5 with helper feature 16; coli_v4_expert_store_wex_flags ==== */
const WexEntry *coli_v4_expert_store_wex_entry(ColiExpertStore *store, ColiExpertKey key,
                                               size_t capacity) {
    if (!store || !store->state) return NULL;
    V4ExpertStoreState *state = store->state;
    if (!state->wex_entries || state->wex_handle == INVALID_HANDLE_VALUE || !get_record(state, key))
        return NULL;
    const WexEntry *entry =
        &state->wex_entries[(size_t)key.layer * state->experts_per_layer + key.expert];
    if (!(entry->flags == WEX_ENTRY_NEURAL_RANS ||
          (entry->flags == WEX_ENTRY_NEURAL_RANS2 && coli_wex_cuda_nr2_available()) ||
          (entry->flags == WEX_ENTRY_NEURAL_RANS3 && coli_wex_cuda_nr3_available())) ||
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

uint32_t coli_v4_expert_store_wex_flags(ColiExpertStore *store) {
    return store && store->state ? ((V4ExpertStoreState *)store->state)->wex_flags : 0u;
}

/* ==== deepseek_v4.c:15039-15046  v4_wex_crc: slot sizing counts codec 4 and 5 (wex_entry_nr_model) ==== */
        if (budget > 0 && direct && atoi(direct) >= 1 && state->wex_entries &&
            state->wex_handle != INVALID_HANDLE_VALUE)
            for (size_t i = 0; i < cells; i++) {
                const WexEntry *e = &state->wex_entries[i];
                if ((e->flags == WEX_ENTRY_NEURAL_RANS || wex_entry_nr_model(e)) &&
                    e->total_bytes > slot_bytes)
                    slot_bytes = e->total_bytes;
            }

/* ==== deepseek_v4.c:15152-15173  coli_v4_wex_crc_done: a hit released with filled == 0 leaves the index ==== */
/* Release a lease; filled says the claimed slot now holds the record. For a
 * hit, filled == 0 means its bytes did not decode: the slot leaves the index
 * (kept until its last lease ends), so the next use rereads the disk. */
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
    } else if (!filled && v->cell >= 0) {
        c->index[v->cell] = -1;
        v->cell = -1;
    }
    if (v->refs > 0) v->refs--;
    pthread_mutex_unlock(&c->mutex);
}

/* ==== deepseek_v4.c:15190-15193  coli_v4_wex_crc_prewarm: codec 4 and 5 ==== */
    for (size_t i = 0; i < cells; i++) {
        const WexEntry *e = &state->wex_entries[i];
        if (!(e->flags == WEX_ENTRY_NEURAL_RANS || wex_entry_nr_model(e)) || !usage[i]) continue;
        if (vram_bits && (vram_bits[i >> 3] & (1u << (i & 7)))) continue;

/* ==== deepseek_v4.c:19629-19674  v4_wex_fill_targets (owner check, only status -1 drops the cached record), v4_wex_fill_mirror, test hook coli_v4_test_wex_fill ==== */
/* One record into a mirror's resolved buffers (the tests call it with host buffers). */
static int v4_wex_fill_targets(const V4WexDirectFill *f, unsigned char *const scale[3],
                               unsigned char *const weight[3]) {
    /* the mirror is reserved: nobody else reads this record meanwhile. The RAM
     * cache of compressed records serves it when it can; otherwise the record
     * comes from disk into a cache slot (kept) or this thread's staging. */
    double t0 = v4ap_now();
    unsigned char *record = f->record;
    int hit = 0, slot = coli_v4_wex_crc_get(f->store, f->key, f->entry, &record, &hit);
    if (slot < 0) record = f->record;
    if ((!hit && coli_v4_expert_store_wex_read(f->store, f->key, f->entry, record)) ||
        !wex_nr3_owner_ok(f->entry, record, (uint32_t)f->key.layer, (uint32_t)f->key.expert,
                          coli_v4_expert_store_wex_flags(f->store))) {
        if (slot >= 0) coli_v4_wex_crc_done(f->store, slot, 0);
        return -1;
    }
    double t1 = v4ap_now();
    int status = coli_wex_cuda_decode_to_device(f->entry, record, f->device, scale, weight);
    unsigned long long *c = v4_wexb_record[f->prefetch ? 1 : 0];
    __atomic_fetch_add(&c[0], 1ULL, __ATOMIC_RELAXED);
    __atomic_fetch_add(&c[1], (unsigned long long)((t1 - t0) * 1e9), __ATOMIC_RELAXED);
    __atomic_fetch_add(&c[2], (unsigned long long)((v4ap_now() - t1) * 1e9), __ATOMIC_RELAXED);
    /* only a rejected record (-1) leaves the RAM cache: status 1 says nothing about its bytes */
    if (slot >= 0) coli_v4_wex_crc_done(f->store, slot, status != -1);
    return status;
}

static int v4_wex_fill_mirror(void *context, Dsv4CudaTensor *gate, Dsv4CudaTensor *down,
                              Dsv4CudaTensor *up) {
    const V4WexDirectFill *f = context;
    unsigned char *scale[3], *weight[3];
    if (!v4_wex_mirror_targets(f->entry, gate, down, up, scale, weight)) return -1;
    return v4_wex_fill_targets(f, scale, weight);
}

#ifdef COLI_V4_TEST_HOOKS
/* tests/test_v4_wex_nr3.c: the GPU-direct fill of one expert into host buffers
 * (a fake helper decodes there); -2 when the store would not serve it direct. */
int coli_v4_test_wex_fill(ColiExpertStore *store, ColiExpertKey key, unsigned char *staging,
                          size_t staging_bytes, unsigned char *const scale[3], unsigned char *const weight[3]) {
    const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, staging_bytes);
    if (!entry) return -2;
    V4WexDirectFill f = {store, key, entry, staging, 0, 0};
    return v4_wex_fill_targets(&f, scale, weight);
}
#endif

/* ==== deepseek_v4.c:19912-19913  coli_v4_gpu_expert_wex_reserve: batch takes codec 4 and 5 ==== */
        const WexEntry *entry = coli_v4_expert_store_wex_entry(store, key, need);
        if (!wex_entry_nr_model(entry)) continue;   /* WNR1: per record */

/* ==== deepseek_v4.c:19992-19992  coli_v4_gpu_expert_wex_complete: hits[] no longer decides the cache release ==== */
    int *slots = b->slots, *ok = b->ok;   /* a rejected hit leaves the RAM cache too (keep below) */

/* ==== deepseek_v4.c:20003-20030  coli_v4_gpu_expert_wex_complete: owner check per record, keep[] (rejected records leave the RAM cache) ==== */
    int keep[MAX];   /* record bytes fit for the RAM cache: read and not rejected */
    uint32_t wex_flags = coli_v4_expert_store_wex_flags(store);
    for (int i = 0; i < n; i++) {
        keep[i] = ok[i];
        if (!ok[i]) continue;
        if (!wex_nr3_owner_ok(entries[i], records[i], (uint32_t)b->keys[i].layer, (uint32_t)b->keys[i].expert,
                              wex_flags)) {
            ok[i] = keep[i] = 0;
            continue;
        }
        if (!v4_wex_mirror_targets(entries[i], holds[i].gate, holds[i].down, holds[i].up,
                                   scale + 3 * m, weight + 3 * m)) {
            ok[i] = 0;
            continue;
        }
        batch_entries[m] = entries[i];
        batch_records[m] = records[i];
        batch_of[m++] = i;
    }
    if (m) coli_wex_cuda_decode_batch_to_device(m, batch_entries, batch_records, cache->device,
                                                scale, weight, status);
    for (int j = 0; j < m; j++) {
        ok[batch_of[j]] = status[j] == 0;
        if (status[j] == -1) keep[batch_of[j]] = 0;
    }
    double t2 = v4ap_now();
    for (int i = 0; i < n; i++)
        if (slots[i] >= 0) coli_v4_wex_crc_done(store, slots[i], keep[i]);

