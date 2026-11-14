/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: the maintainer's engine tree, src/colibri/c/deepseek_v4.c as of 2026-11-07, as committed.
 * Line numbers are of that version; the live tree moves them. Code between ranges is omitted.
 * WEX store hooks for mode 0 (host slab): container open, WNR2 guard, mirrors, drive pick, record read.
 * All of it sits in the expert-store unit (COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16), Windows only.
 * Enough on its own for WEX1/WEX2/WEX3/WNR2 decode into the RAM slab (GPU-direct off).
 */

/* ==== deepseek_v4.c:12376-12376  include the WEX store header (pulls wex_format.h and wex_decode.h) ==== */
#include "expert_store_wex.h"

/* ==== deepseek_v4.c:12459-12476  V4ExpertStoreState: WEX fields + forward declaration of the RAM-cache free ==== */
#ifdef _WIN32
    HANDLE wex_handle;
    HANDLE wex_mirror;          /* wex_mirrors[0], kept for the single-mirror code */
    HANDLE wex_mirrors[3];      /* every matching mirror container, in COLI_MODEL_MIRROR order */
    int wex_nmir;
#else
    int wex_fd;
    int wex_mirror_fd;
#endif
    WexEntry *wex_entries;
#ifdef _WIN32
    void *wex_crc;        /* compressed record cache (COLI_WEX_RAM_CACHE_GB) */
    int wex_crc_tried;
#endif
} V4ExpertStoreState;
#ifdef _WIN32
static void v4_wex_crc_free(V4ExpertStoreState *state);
#endif

/* ==== deepseek_v4.c:12546-12548  build_record: a WEX container forces the per-matrix record layout ==== */
    if (state->wex_entries != NULL || !scale_range_contiguous || !weight_range_contiguous) {
        /* WEX Direct I/O layout or REAP-style packed checkpoint: per-matrix layout. */
        record->per_matrix = 1;

/* ==== deepseek_v4.c:13045-13058  destroy: close the container and its mirrors, free the table and the RAM cache ==== */
#ifdef _WIN32
        if (state->wex_handle && state->wex_handle != INVALID_HANDLE_VALUE)
            CloseHandle(state->wex_handle);
        for (int i = 0; i < state->wex_nmir; i++)
            if (state->wex_mirrors[i] && state->wex_mirrors[i] != INVALID_HANDLE_VALUE)
                CloseHandle(state->wex_mirrors[i]);
        if (state->wex_entries)
            _aligned_free(state->wex_entries);
        v4_wex_crc_free(state);
#else
        if (state->wex_fd >= 0) close(state->wex_fd);
        if (state->wex_mirror_fd >= 0) close(state->wex_mirror_fd);
        if (state->wex_entries) free(state->wex_entries);
#endif

/* ==== deepseek_v4.c:13064-13159  v4_wex_read_exact, v4_wex_load_manifest (header/table validation + WNR2 model guard), v4_wex_matches_record ==== */
#ifdef _WIN32
/* WEX v1/v2 share the decoded MXFP4 layout. Read and validate the manifest
 * before accepting any disk-controlled offsets or allocating cache slabs. */
static int v4_wex_read_exact(HANDLE file, void *buffer, DWORD bytes,
                              uint64_t offset) {
    OVERLAPPED ov = {0};
    DWORD read_bytes = 0;
    ov.Offset = (DWORD)offset;
    ov.OffsetHigh = (DWORD)(offset >> 32);
    BOOL ok = ReadFile(file, buffer, bytes, &read_bytes, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING)
        ok = GetOverlappedResult(file, &ov, &read_bytes, TRUE);
    return ok && read_bytes == bytes ? 0 : -1;
}

static int v4_wex_load_manifest(HANDLE file, int layers, int experts,
                                 unsigned char **header_block,
                                 WexEntry **entries_out, size_t *table_span,
                                 uint64_t *file_bytes) {
    LARGE_INTEGER length;
    size_t table_bytes = 0, aligned_table_bytes = 0;
    unsigned char *block = NULL;
    WexEntry *entries = NULL;
    *header_block = NULL;
    *entries_out = NULL;
    if (!GetFileSizeEx(file, &length) || length.QuadPart < WEX_ALIGN)
        return -1;
    block = (unsigned char *)_aligned_malloc(WEX_ALIGN, WEX_ALIGN);
    if (!block || v4_wex_read_exact(file, block, WEX_ALIGN, 0)) goto fail;
    const WexHeader *header = (const WexHeader *)block;
    if (wex_validate_header(header, (uint64_t)length.QuadPart,
                            (uint32_t)layers, (uint32_t)experts,
                            &table_bytes, &aligned_table_bytes) ||
        aligned_table_bytes > MAXDWORD)
        goto fail;
    entries = (WexEntry *)_aligned_malloc(aligned_table_bytes, WEX_ALIGN);
    if (!entries || v4_wex_read_exact(file, entries,
                     (DWORD)aligned_table_bytes, header->table_offset))
        goto fail;
    if (wex_validate_table(header, entries, (uint64_t)length.QuadPart)) goto fail;
    if (header->flags & WEX_HEADER_NEURAL_RANS2) {
        /* WNR2: the container's model must load and match the id its header
         * names before any record can be decoded */
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
    }
    *header_block = block;
    *entries_out = entries;
    *table_span = aligned_table_bytes;
    *file_bytes = (uint64_t)length.QuadPart;
    return 0;
fail:
    _aligned_free(block);
    _aligned_free(entries);
    return -1;
}

static int v4_wex_matches_record(const WexEntry *entry,
                                  const V4ExpertRecord *record) {
    uint64_t scale_cursor = 0, weight_cursor = 0;
    if (!entry->total_bytes) return 1; /* absent WEX entry uses safetensors */
    for (int matrix = 0; matrix < V4_MATRIX_COUNT; matrix++) {
        const ColiSafetensorsTensor *scale = record->scale[matrix];
        const ColiSafetensorsTensor *weight = record->weight[matrix];
        if (entry->scale_shapes[matrix][0] != scale->shape[0] ||
            entry->scale_shapes[matrix][1] != scale->shape[1] ||
            entry->weight_shapes[matrix][0] != weight->shape[0] ||
            entry->weight_shapes[matrix][1] != weight->shape[1] ||
            entry->scale_offsets[matrix] != scale_cursor ||
            entry->weight_offsets[matrix] != entry->scale_bytes + weight_cursor)
            return 0;
        scale_cursor += (uint64_t)scale->nbytes;
        weight_cursor += (uint64_t)weight->nbytes;
    }
    return entry->scale_bytes == scale_cursor &&
           entry->weight_bytes == weight_cursor &&
           record->record_bytes == scale_cursor + weight_cursor;
}
#endif

/* ==== deepseek_v4.c:13189-13198  coli_deepseek_v4_expert_store_open: initial values of the WEX fields ==== */
#ifdef _WIN32
    state->wex_handle = INVALID_HANDLE_VALUE;
    state->wex_mirror = INVALID_HANDLE_VALUE;
    for (int i = 0; i < 3; i++) state->wex_mirrors[i] = INVALID_HANDLE_VALUE;
    state->wex_nmir = 0;
#else
    state->wex_fd = -1;
    state->wex_mirror_fd = -1;
#endif
    state->wex_entries = NULL;

/* ==== deepseek_v4.c:13201-13307  store open: COLI_WEX_PATH / default path, CreateFileA unbuffered, manifest, mirrors (COLI_MODEL_MIRROR + COLI_WEX_EXTRA_MIRRORS) ==== */
    /* DUAL-SSD: register COLI_MODEL_MIRROR copies and derive the read split
     * before any expert load, so pin warmup and demand reads stream from all
     * drives. */
    if (!options->skip_mirror_setup)
        coli_st_mirror_setup(state->index, options->model_dir,
                             options->experts_per_layer);

    /* Open WEX Direct I/O container if present */
    const char *wex_path = getenv("COLI_WEX_PATH");
    char wex_buf[1024];
    if (!wex_path || !*wex_path) {
        snprintf(wex_buf, sizeof(wex_buf), "%s\\v4_flash_experts.wex", options->model_dir);
        wex_path = wex_buf;
    }
#ifdef _WIN32
    unsigned char *wex_header_block = NULL;
    size_t wex_table_span = 0;
    uint64_t wex_file_bytes = 0;
    HANDLE hWex = CreateFileA(
        wex_path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );
    if (hWex != INVALID_HANDLE_VALUE) {
        if (!v4_wex_load_manifest(hWex, state->layers, state->experts_per_layer,
                  &wex_header_block, &state->wex_entries,
                  &wex_table_span, &wex_file_bytes)) {
            const WexHeader *hdr = (const WexHeader *)wex_header_block;
            state->wex_handle = hWex;
            fprintf(stderr, "[v4_wex] Container v%u active: %s (%u layers, %u experts, %llu decoded bytes/expert, direct_io=on)\n",
                    hdr->version, wex_path, hdr->num_layers, hdr->num_experts,
                    (unsigned long long)hdr->aligned_expert_bytes);
        } else {
            CloseHandle(hWex);
            fprintf(stderr, "[v4_wex] Invalid or incompatible container: %s; using safetensors\n", wex_path);
        }
    }
    const char *mirror_list = getenv("COLI_MODEL_MIRROR");
    char mirror_buf[4096];
    /* COLI_WEX_EXTRA_MIRRORS: more dirs holding only the container (no shards),
     * after the COLI_MODEL_MIRROR ones */
    const char *extra_list = getenv("COLI_WEX_EXTRA_MIRRORS");
    snprintf(mirror_buf, sizeof(mirror_buf), "%s;%s", mirror_list ? mirror_list : "",
             extra_list ? extra_list : "");
    /* COLI_MODEL_MIRROR may list several dirs (';' or ','): every one holding a
     * matching container becomes a WEX replica, up to three */
    for (char *mirror_dir = mirror_buf, *next = NULL; mirror_dir && *mirror_dir &&
         state->wex_handle != INVALID_HANDLE_VALUE && state->wex_nmir < 3; mirror_dir = next) {
        char *sep = mirror_dir;
        while (*sep && *sep != ';' && *sep != ',') sep++;
        next = *sep ? sep + 1 : NULL;
        *sep = 0;
        while (*mirror_dir == ' ') mirror_dir++;
        if (!*mirror_dir) continue;
        char mirror_wex[1024];
        const char *basename = strrchr(wex_path, '\\');
        const char *slash = strrchr(wex_path, '/');
        if (!basename || (slash && slash > basename)) basename = slash;
        basename = basename ? basename + 1 : wex_path;
        size_t dir_bytes = strlen(mirror_dir), name_bytes = strlen(basename);
        mirror_wex[0] = '\0';
        if (dir_bytes < sizeof(mirror_wex) - 1 &&
            name_bytes < sizeof(mirror_wex) - dir_bytes - 1) {
            memcpy(mirror_wex, mirror_dir, dir_bytes);
            mirror_wex[dir_bytes] = '\\';
            memcpy(mirror_wex + dir_bytes + 1, basename, name_bytes + 1);
        }
        if (*mirror_wex && _stricmp(wex_path, mirror_wex) != 0) {
            HANDLE hMirror = CreateFileA(
                mirror_wex,
                GENERIC_READ,
                FILE_SHARE_READ,
                NULL,
                OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                NULL
            );
            if (hMirror != INVALID_HANDLE_VALUE) {
                unsigned char *mirror_header = NULL;
                WexEntry *mirror_entries = NULL;
                size_t mirror_table_span = 0;
                uint64_t mirror_bytes = 0;
                if (!v4_wex_load_manifest(hMirror, state->layers,
                        state->experts_per_layer, &mirror_header,
                        &mirror_entries, &mirror_table_span, &mirror_bytes) &&
                    mirror_bytes == wex_file_bytes &&
                    mirror_table_span == wex_table_span &&
                    !memcmp(mirror_header, wex_header_block, WEX_ALIGN) &&
                    !memcmp(mirror_entries, state->wex_entries, wex_table_span)) {
                    state->wex_mirrors[state->wex_nmir++] = hMirror;
                    if (state->wex_nmir == 1) state->wex_mirror = hMirror;
                    fprintf(stderr, "[v4_wex] Matching mirror container active: %s\n", mirror_wex);
                } else {
                    CloseHandle(hMirror);
                    fprintf(stderr, "[v4_wex] Ignoring mismatched mirror: %s\n", mirror_wex);
                }
                _aligned_free(mirror_header);
                _aligned_free(mirror_entries);
            }
        }
    }
    _aligned_free(wex_header_block);
#endif

/* ==== deepseek_v4.c:13320-13328  store open, per-expert loop: the container layout must match the safetensors (hard failure) ==== */
#ifdef _WIN32
            if (state->wex_entries && !v4_wex_matches_record(
                    &state->wex_entries[(size_t)layer * state->experts_per_layer + expert], record)) {
                set_error(error, error_size,
                          "WEX layout does not match native expert: layer=%d expert=%d",
                          layer, expert);
                goto fail;
            }
#endif

/* ==== deepseek_v4.c:13408-13418  store open fail path: release the WEX handles ==== */
#ifdef _WIN32
    if (state->wex_handle && state->wex_handle != INVALID_HANDLE_VALUE) CloseHandle(state->wex_handle);
    for (int i = 0; i < state->wex_nmir; i++)
        if (state->wex_mirrors[i] && state->wex_mirrors[i] != INVALID_HANDLE_VALUE)
            CloseHandle(state->wex_mirrors[i]);
    if (state->wex_entries) _aligned_free(state->wex_entries);
#else
    if (state->wex_fd >= 0) close(state->wex_fd);
    if (state->wex_mirror_fd >= 0) close(state->wex_mirror_fd);
    if (state->wex_entries) free(state->wex_entries);
#endif

/* ==== deepseek_v4.c:13774-13817  v4_wex_pick: which drive serves an expert (COLI_WEX_WEIGHTS, dual-SSD split) ==== */
#ifdef _WIN32
/* The WEX file of one expert. With one mirror: the dual-SSD replica split
 * (coli_st_expert_route, COLI_DISK_WEIGHTS), as before. With more mirrors, or
 * COLI_WEX_WEIGHTS=<primary>,<mirror1>[,...] set: the same deterministic hash of
 * (layer, expert) split over primary + mirrors by those weights (equal shares
 * when unset), so an expert is always read from the same drive. */
static HANDLE v4_wex_pick(const V4ExpertStoreState *state, int layer, int expert) {
    if (state->wex_nmir <= 0) return state->wex_handle;
    static int cut[4], n = -1;
    if (n < 0) {
        int want = 1 + state->wex_nmir;
        double w[4] = {1, 1, 1, 1}, W = 0;
        const char *e = getenv("COLI_WEX_WEIGHTS");
        int custom = 0;
        if (e && *e) {
            char b[128]; snprintf(b, sizeof(b), "%s", e);
            int k = 0, bad = 0;
            for (char *t = strtok(b, ", "); t; t = strtok(NULL, ", ")) {
                double v = atof(t);
                if (v <= 0 || k >= 4) { bad = 1; break; }
                w[k++] = v;
            }
            custom = !bad && k == want;
            if (!custom) fprintf(stderr, "[v4_wex] COLI_WEX_WEIGHTS '%s' ignored (want %d positive weights)\n", e, want);
        }
        if (!custom && want == 2) { n = 0; }   /* the dual-SSD split, unchanged */
        else {
            for (int r = 0; r < want; r++) W += w[r];
            double acc = 0;
            for (int r = 0; r < want; r++) { acc += w[r]; cut[r] = (int)(acc / W * 256.0 + 0.5); }
            cut[want - 1] = 256;
            n = want;
            fprintf(stderr, "[v4_wex] read split over %d drives:", want);
            for (int r = 0; r < want; r++) fprintf(stderr, " %.0f%%", 100.0 * w[r] / W);
            fprintf(stderr, "\n");
        }
    }
    if (n == 0) return coli_st_expert_route(layer, expert) == 1 ? state->wex_mirrors[0] : state->wex_handle;
    uint32_t h = (uint32_t)(layer * state->experts_per_layer + expert) * 2654435761u;
    int hv = (int)((h >> 16) & 255), r = 0;
    while (hv >= cut[r]) r++;
    return r == 0 ? state->wex_handle : state->wex_mirrors[r - 1];
}
#endif

/* ==== deepseek_v4.c:13819-13864  v4_read_expert_record: the WEX branch (read compressed record into the slab, decode in place, else fall back to safetensors) ==== */
static int v4_read_expert_record(V4ExpertStoreState *state,
                                 const V4ExpertRecord *record,
                                 V4ExpertSlot *slot, int rep,
                                 uint64_t *payload_bytes) {
    if (payload_bytes) *payload_bytes = record->record_bytes;
#ifdef _WIN32
    if (state->wex_entries != NULL && state->wex_handle != INVALID_HANDLE_VALUE) {
        size_t idx = (size_t)record->layer * state->experts_per_layer + record->expert;
        const WexEntry *entry = &state->wex_entries[idx];
        size_t read_offset = wex_record_read_offset(entry);
        /* All public callers guarantee only record_bytes of writable memory.
         * Non-page-sized legacy records fall back to safetensors rather than
         * relying on the hot cache's extra direct-I/O padding. */
        if (slot->aligned_slab && entry->total_bytes > 0 &&
            read_offset <= record->record_bytes &&
            entry->total_bytes <= record->record_bytes - read_offset &&
            WEX_IS_ALIGNED(slot->slab + read_offset, WEX_ALIGN)) {
            HANDLE hFile = v4_wex_pick(state, record->layer, record->expert);
            (void)rep;
            static __thread HANDLE tl_wex_ev = NULL;
            if (!tl_wex_ev) tl_wex_ev = CreateEventA(NULL, TRUE, FALSE, NULL);
            OVERLAPPED ov;
            memset(&ov, 0, sizeof(ov));
            ov.Offset = (DWORD)(entry->file_offset & 0xFFFFFFFFULL);
            ov.OffsetHigh = (DWORD)((entry->file_offset >> 32) & 0xFFFFFFFFULL);
            ov.hEvent = tl_wex_ev;
            ResetEvent(tl_wex_ev);
            DWORD bytes_read = 0;
            BOOL ok = ReadFile(hFile, slot->slab + read_offset,
                               entry->total_bytes, &bytes_read, &ov);
            if (!ok && GetLastError() == ERROR_IO_PENDING) {
                ok = GetOverlappedResult(hFile, &ov, &bytes_read, TRUE);
            }
            if (ok && bytes_read == entry->total_bytes &&
                !coli_wex_decode_record(entry, slot->slab,
                                     (size_t)record->record_bytes)) {
                __atomic_fetch_add(&v4_direct_reads, UINT64_C(1), __ATOMIC_RELAXED);
                __atomic_fetch_add(&v4_direct_flock_reads, UINT64_C(1), __ATOMIC_RELAXED);
                __atomic_fetch_add(&v4_direct_payload_bytes, entry->total_bytes, __ATOMIC_RELAXED);
                if (payload_bytes) *payload_bytes = entry->total_bytes;
                return 0;
            }
            __atomic_fetch_add(&v4_direct_fallbacks, UINT64_C(1), __ATOMIC_RELAXED);
        }
    }
#endif

/* ==== deepseek_v4.c:15287-15294  ROWS16 store open: the v4_ssd_io log line (NOTE: reads wex_handle outside #ifdef _WIN32, guard it for non-Windows) ==== */
    V4ExpertStoreState *state = (*output)->state;
    int direct_io = (state->wex_entries != NULL && state->wex_handle != INVALID_HANDLE_VALUE) ||
        (state->layers > 0 && state->experts_per_layer > 0 &&
         coli_st_streaming_direct_available(
            state->index, state->records[0].weight_shard));
    fprintf(stderr, "v4_ssd_io mode=%s fallback=buffered-pread\n",
            (state->wex_entries != NULL) ? "wex-direct-aligned" :
            (direct_io ? "direct-aligned" : "buffered-pread"));

