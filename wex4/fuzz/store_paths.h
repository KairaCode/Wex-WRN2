/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_STORE_PATHS_H
#define WEX_FUZZ_STORE_PATHS_H
/* fuzz_store_wex.c: a second pass over every input the store opened, through the paths of c/expert_store_wex.c
 * that the main pass leaves alone. The store is opened again with:
 *  - COLI_MODEL_MIRROR naming a directory with a copy of the container under its file name, in one of the
 *    forms of engine_paths.h, or the input's own directory, and the option skip_mirror_setup 0 or 1;
 *  - COLI_DISK_WEIGHTS from a table of settings, each with the two weights the store must take from it (a
 *    setting it must refuse leaves its default, 58 and 42);
 *  - DSV4_CUDA_MOE_MASKED=1 for one input in 8;
 *  - WEX_READ_RETRY=1 for one input in 8, with at most 2 lookups, of the first experts that have a record
 *    (the 16 ms pause before the second read of a failed load is skipped by read_faults.h).
 * Then it prefetches the first experts it will look up (at most 4), looks up experts (all of them up to
 * FUZZ_PATHS_LOOKUPS, else that many spread over the table) and closes.
 *
 * Besides what the sanitizers catch:
 *  - the store opens exactly when the container is not canonical-zero under DSV4_CUDA_MOE_MASKED=1 and, when
 *    it takes a copy (skip_mirror_setup 0, a file in the directory, a path other than the input's), the copy
 *    has the input's size, its first 4096 bytes and its table [table_offset, table_offset + aligned table);
 *  - with a copy, an expert is read from it when cut < 256 and the split byte of its flat index is at or above
 *    cut = (256 * primary + total / 2) / total (total = primary + mirror, cut at most 256), else from the
 *    input; a lookup succeeds exactly when the record's bytes of that file pass the owner check and decode,
 *    and the three matrices are the decoded bytes. A copy whose records differ (form RECORDS) shows a read
 *    from the wrong file. With WEX_READ_RETRY=1 a failed load fails again;
 *  - the request, hit and miss counters add up to the lookups made.
 * fuzz_store_faults, at the end of this file, is a third pass: the same store with reads that fail. */

#define FUZZ_PATHS_LOOKUPS 64u

static const struct { const char *text; int primary, mirror; } fuzz_disk_weights[] = {
    {"", 58, 42},          {"1,1", 1, 1},     {"3, 1", 3, 1},     {"1 255", 1, 255},
    {"200,1", 200, 1},     {"1,2147483647", 1, 2147483647},       {"2147483647,1", 2147483647, 1},
    {"0,5", 58, 42},       {"7", 58, 42},     {"5,-1", 58, 42},   {"x,2", 58, 42}};

/* what the pass reached, printed at exit */
static unsigned long long paths_passes, paths_opened, paths_with_mirror, paths_lookups, paths_served,
    paths_from_mirror, paths_retried, paths_masked;
static unsigned long long paths_forms[FUZZ_COPY_FORMS + 2];

static void fuzz_store_paths_report(void) {
    fuzz_mirror_cleanup();
    fprintf(stderr, "fuzz_store_wex other paths: %llu passes (copies: same %llu, records %llu, page %llu, size %llu, "
            "table %llu, missing %llu, records skipped %llu, own directory %llu), %llu opened (%llu with a mirror), "
            "%llu refused for DSV4_CUDA_MOE_MASKED, %llu lookups, %llu records served (%llu read from the mirror), "
            "%llu failed loads read again (WEX_READ_RETRY); CUDA helper %s, %llu records decoded through it\n",
            paths_passes, paths_forms[0], paths_forms[1], paths_forms[2], paths_forms[3], paths_forms[4],
            paths_forms[5], paths_forms[6], paths_forms[7], paths_opened, paths_with_mirror, paths_masked,
            paths_lookups, paths_served, paths_from_mirror, paths_retried,
            strcmp(fuzz_cuda_lib(), "0") ? "on" : "off", (unsigned long long)coli_wex_cuda_decode_count());
}

/* The record of entry e (layer, expert) as the store must serve it when it reads `bytes`: 0 and the decoded
 * slab, or non-zero. The main pass has checked that the record and its decode fit the slab. */
static int fuzz_store_expect(const unsigned char *bytes, const WexHeader *h, const WexEntry *e, int layer, int expert,
                             const WexNr2Model *model, int has_model, unsigned char *slab) {
    size_t at = wex_record_read_offset(e);
    if (!e->total_bytes) return -1;
    memcpy(slab + at, bytes + e->file_offset, e->total_bytes);
    if (!wex_nr3_owner_ok(e, slab + at, (uint32_t)layer, (uint32_t)expert, h->flags)) return -1;
    if (!wex_entry_nr_model(e)) return wex_decode_record(e, slab, (size_t)h->aligned_expert_bytes);
    if (has_model && wex_read_u32_le(slab + at + 4u) == model->id)
        return wex_decode_nr2_record(e, slab, (size_t)h->aligned_expert_bytes, model);
    return -1;
}

static void fuzz_store_paths(const char *primary, const unsigned char *file, size_t size, const WexNr2Model *model,
                             int has_model) {
    static int reported;
    const WexHeader *h = (const WexHeader *)file;
    const WexEntry *entries = (const WexEntry *)(file + h->table_offset);
    uint64_t hash = fuzz_paths_hash(file, size);
    int retry = (hash >> 24) % 8u == 0;
    /* 0-5: the forms of engine_paths.h; 6: RECORDS with skip_mirror_setup = 1; 7: the input's own directory. Under
     * WEX_READ_RETRY one pass in 2 takes RECORDS: a coded record read from that copy fails, and is read again */
    int variant = retry && (hash >> 28) & 1u ? FUZZ_COPY_RECORDS : (int)(hash % 8u), skip = variant == 6, own = variant == 7;
    int form = skip ? FUZZ_COPY_RECORDS : own ? FUZZ_COPY_MISSING : variant;
    int masked = (hash >> 16) % 8u == 0, with_mirror, rc, opens, same = 0;
    size_t w = (size_t)((hash >> 8) % (sizeof fuzz_disk_weights / sizeof fuzz_disk_weights[0]));
    size_t table_bytes = 0, aligned_table = 0, count = (size_t)h->num_layers * h->num_experts, i, step, copy_size = 0;
    size_t limit = retry ? 2u : FUZZ_PATHS_LOOKUPS, looked_up = 0, record_count = 0;
    unsigned cut = 256u;
    char directory[1024], error[256] = {0};
    unsigned char *copy, *slab;
    FuzzExtent *records, table;
    ColiDeepSeekV4ExpertStoreOptions options;
    ColiExpertStore *store = NULL;
    ColiExpertStoreStats stats;
    ColiExpertKey keys[4];
    size_t prefetched = 0;

    if (!reported) { reported = 1; atexit(fuzz_store_paths_report); }
    paths_passes++;
    paths_forms[own ? 7 : skip ? 6 : form]++;
    (void)wex_validate_header(h, size, 0, 0, &table_bytes, &aligned_table);   /* the main pass validated it */
    records = (FuzzExtent *)calloc(count ? count : 1u, sizeof *records);
    if (!records) abort();
    for (i = 0; i < count; i++)
        if (entries[i].total_bytes) {
            records[record_count].offset = entries[i].file_offset;
            records[record_count++].bytes = entries[i].total_bytes;
        }
    table.offset = h->table_offset; table.bytes = aligned_table;
    copy = fuzz_mirror_write(0, primary, file, size, form, hash, records, record_count, table);
    free(records);
    if (copy) copy_size = size + (form == FUZZ_COPY_SIZE ? 4096u : 0u);
    if (own) {   /* the input's directory: the mirror path is the input's own path, or names the same file */
        size_t length = (size_t)(fuzz_basename(primary) - primary);
        snprintf(directory, sizeof directory, "%.*s", (int)(length ? length - 1u : 0u), primary);
    } else {
        snprintf(directory, sizeof directory, "%s", fuzz_mirror_dir(0));
    }
    /* the bytes the store compares: size, header page, table */
    same = copy && copy_size == size && !memcmp(copy, file, WEX_ALIGN) &&
           !memcmp(copy + h->table_offset, file + h->table_offset, aligned_table);
    with_mirror = !skip && copy != NULL;
    opens = !(masked && (h->flags & WEX_HEADER_CANONICAL_ZERO)) && (!with_mirror || same);

    fuzz_setenv("COLI_MODEL_MIRROR", directory);
    fuzz_setenv("COLI_DISK_WEIGHTS", fuzz_disk_weights[w].text);
    fuzz_setenv("DSV4_CUDA_MOE_MASKED", masked ? "1" : "");
    fuzz_setenv("WEX_READ_RETRY", retry ? "1" : "");
    memset(&options, 0, sizeof options);
    options.model_dir = ".";
    options.skip_mirror_setup = skip;
    rc = coli_v4_expert_store_open_wex(NULL, NULL, &options, &store, error, sizeof error);
    fuzz_setenv("COLI_MODEL_MIRROR", "");
    fuzz_setenv("COLI_DISK_WEIGHTS", "");
    fuzz_setenv("DSV4_CUDA_MOE_MASKED", "");
    FUZZ_REQUIRE(rc ? !store : store != NULL, "the store's open result and its output pointer disagree (other paths)");
    FUZZ_REQUIRE((rc == 0) == opens, "the store takes a mirror copy, or refuses one, against the bytes it compares");
    paths_masked += masked && (h->flags & WEX_HEADER_CANONICAL_ZERO);
    if (rc) { fuzz_setenv("WEX_READ_RETRY", ""); free(copy); return; }
    paths_opened++;
    if (with_mirror) {
        int64_t total = (int64_t)fuzz_disk_weights[w].primary + fuzz_disk_weights[w].mirror;
        uint64_t c = ((uint64_t)fuzz_disk_weights[w].primary * 256u + (uint64_t)(total / 2)) / (uint64_t)total;
        cut = c > 256u ? 256u : (unsigned)c;
        paths_with_mirror++;
    }

    slab = (unsigned char *)malloc((size_t)h->aligned_expert_bytes);
    if (!slab) abort();
    /* under WEX_READ_RETRY the first experts that have a record: a lookup of an empty entry reads nothing */
    step = retry || count <= limit ? 1u : count / limit;
    for (i = 0; i < count && prefetched < 4u && prefetched < limit; i += step) {   /* the first keys of the loop below */
        if (retry && !entries[i].total_bytes) continue;
        keys[prefetched].layer = (int)(i / h->num_experts);
        keys[prefetched++].expert = (int)(i % h->num_experts);
    }
    store->ops->prefetch(store, keys, prefetched);
    for (i = 0; i < count && looked_up < limit; i += step) {
        const WexEntry *e = &entries[i];
        int layer = (int)(i / h->num_experts), expert = (int)(i % h->num_experts), want, got;
        int from_mirror = with_mirror && cut < 256u && fuzz_split_byte((uint32_t)i) >= cut;
        ColiExpertKey key;
        ColiExpertView view;
        if (retry && !e->total_bytes) continue;
        key.layer = layer; key.expert = expert;
        want = fuzz_store_expect(from_mirror ? copy : file, h, e, layer, expert, model, has_model, slab);
        memset(&view, 0, sizeof view);
        got = coli_expert_lookup(store, key, &view);
        looked_up++;
        FUZZ_REQUIRE(!got == !want, "the store and the decoders disagree on a record read through the split");
        if (got) { paths_retried += retry && e->total_bytes; continue; }
        {
            const ColiTensorView *matrix[3];
            unsigned int m;
            matrix[0] = &view.gate; matrix[1] = &view.down; matrix[2] = &view.up;
            for (m = 0; m < 3u; m++) {
                size_t scale_bytes = (size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
                size_t weight_bytes = (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1];
                FUZZ_REQUIRE(matrix[m]->scale_bytes == scale_bytes && matrix[m]->data_bytes == weight_bytes &&
                             !memcmp(matrix[m]->scales, slab + e->scale_offsets[m], scale_bytes) &&
                             !memcmp(matrix[m]->data, slab + e->weight_offsets[m], weight_bytes),
                             "the store hands out a matrix that is not the record of the file the split names");
            }
        }
        coli_expert_release(store, &view);
        paths_served++;
        paths_from_mirror += from_mirror;
    }
    paths_lookups += looked_up;
    store->ops->stats(store, &stats);
    FUZZ_REQUIRE(stats.requests == looked_up && stats.hits + stats.misses == looked_up,
                 "the store's counters do not add up to the lookups (other paths)");
    store->ops->destroy(store);
    fuzz_setenv("WEX_READ_RETRY", "");
    free(slab);
    free(copy);
}

/* The third pass, over every input the store opened: reads that fail (read_faults.h). For one input in 2,
 * COLI_MODEL_MIRROR names a copy of the same bytes, with COLI_DISK_WEIGHTS=1,1, so that a load reads from
 * either file. The store is opened once to count its reads, opened again with one of them failed, then opened
 * a third time: for one input in 2 the first expert with a record (from a point the input's hash picks) is
 * prefetched with its read failed, then that expert is looked up with its read failed, WEX_READ_RETRY off
 * (6 inputs in 8) or on with one read failed (1 in 8, so the second read succeeds) or two (1 in 8), then
 * looked up again with no read failed. The 16 ms pause of WEX_READ_RETRY is skipped and counted (read_faults.h).
 *
 * Besides what the sanitizers catch (LeakSanitizer, under libFuzzer, sees what the failed open leaves):
 *  - the open reads the header page, the table, the model of a WNR2 or WNR3 container, and the mirror's
 *    header page and table, each once; with one of those reads failed it fails, with no store;
 *  - a prefetch whose read failed does not publish the record;
 *  - a lookup whose read failed returns no record, unless WEX_READ_RETRY gave it a second read that did not
 *    fail: then it succeeds exactly when the record passes the owner check and decodes, with the decoded
 *    matrices; the lookup after it gives the record's own verdict and matrices (a failed load left its slot
 *    reusable);
 *  - every read the pass failed was a read the store made, and the counters add up to the two lookups. */

static unsigned long long faults_passes, faults_with_mirror, faults_open_reads, faults_open_refused, faults_prefetch,
    faults_lookups, faults_retry_one, faults_retry_served, faults_retry_two, faults_served_after;

static void fuzz_store_faults_report(void) {
    fprintf(stderr, "fuzz_store_wex failing reads: %llu passes (%llu with a mirror), %llu reads to open, %llu opens "
            "with a failed read refused, %llu prefetches with a failed read, %llu lookups with a failed read (%llu "
            "read again by WEX_READ_RETRY, %llu of them served; %llu with the second read failed too), %llu "
            "records served by the lookup after one; %llu pauses of WEX_READ_RETRY skipped in both passes\n",
            faults_passes, faults_with_mirror, faults_open_reads, faults_open_refused, faults_prefetch, faults_lookups,
            faults_retry_one, faults_retry_served, faults_retry_two, faults_served_after, fuzz_fault_pauses);
}

static ColiExpertStore *fuzz_store_open_plain(int *rc) {
    ColiDeepSeekV4ExpertStoreOptions options;
    ColiExpertStore *store = NULL;
    char error[256] = {0};
    memset(&options, 0, sizeof options);
    options.model_dir = ".";
    *rc = coli_v4_expert_store_open_wex(NULL, NULL, &options, &store, error, sizeof error);
    return store;
}

/* 1 when the three matrices of a view are those of entry e in the decoded slab */
static int fuzz_store_view_is(const ColiExpertView *view, const WexEntry *e, const unsigned char *slab) {
    const ColiTensorView *matrix[3];
    unsigned int m;
    matrix[0] = &view->gate; matrix[1] = &view->down; matrix[2] = &view->up;
    for (m = 0; m < 3u; m++) {
        size_t scale_bytes = (size_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
        size_t weight_bytes = (size_t)e->weight_shapes[m][0] * e->weight_shapes[m][1];
        if (matrix[m]->scale_bytes != scale_bytes || matrix[m]->data_bytes != weight_bytes ||
            memcmp(matrix[m]->scales, slab + e->scale_offsets[m], scale_bytes) ||
            memcmp(matrix[m]->data, slab + e->weight_offsets[m], weight_bytes)) return 0;
    }
    return 1;
}

static void fuzz_store_faults(const char *primary, const unsigned char *file, size_t size, const WexNr2Model *model,
                              int has_model) {
    static int reported;
    const WexHeader *h = (const WexHeader *)file;
    const WexEntry *entries = (const WexEntry *)(file + h->table_offset);
    uint64_t hash = fuzz_paths_mix(fuzz_paths_hash(file, size), 1000u);   /* not the second pass's choices */
    size_t count = (size_t)h->num_layers * h->num_experts, i, pick = count;
    int mirror = (int)(hash & 1u), mode = (int)((hash >> 24) % 8u), rc, fails;
    unsigned long long reads, failed, want_reads, pauses;
    ColiExpertStore *store;
    FuzzExtent none = {0, 0};

    if (!reported) { reported = 1; atexit(fuzz_store_faults_report); }
    faults_passes++;
    faults_with_mirror += (unsigned long long)mirror;
    if (mirror) free(fuzz_mirror_write(0, primary, file, size, FUZZ_COPY_SAME, hash, NULL, 0, none));
    fuzz_setenv("COLI_MODEL_MIRROR", mirror ? fuzz_mirror_dir(0) : "");
    fuzz_setenv("COLI_DISK_WEIGHTS", "1,1");

    /* the reads of an open, then the open with one of them failed */
    want_reads = 2u + ((h->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3)) ? 1u : 0u) + 2u * (unsigned)mirror;
    reads = fuzz_fault_reads;
    store = fuzz_store_open_plain(&rc);
    FUZZ_REQUIRE(!rc && store, "the store does not open again a container it opened (failing reads)");
    reads = fuzz_fault_reads - reads;
    FUZZ_REQUIRE(reads == want_reads, "the store's open does not read its header page, table, model and mirror once each");
    store->ops->destroy(store);
    failed = fuzz_fault_failed;
    fuzz_fault_arm((long long)((hash >> 8) % reads), 1);
    store = fuzz_store_open_plain(&rc);
    fuzz_fault_disarm();
    FUZZ_REQUIRE(fuzz_fault_failed == failed + 1u, "a read the pass failed was not made by the store's open");
    FUZZ_REQUIRE(rc && !store, "the store opens although one of its reads failed");
    faults_open_reads += reads;
    faults_open_refused++;

    /* a prefetch and a lookup whose read fails, then a lookup with no failure */
    store = fuzz_store_open_plain(&rc);
    FUZZ_REQUIRE(!rc && store, "the store does not open again a container it opened (failing reads)");
    for (i = 0; i < count && pick == count; i++)
        if (entries[((hash >> 32) + i) % count].total_bytes) pick = (size_t)(((hash >> 32) + i) % count);
    if (pick < count) {
        const WexEntry *e = &entries[pick];
        ColiExpertKey key;
        ColiExpertView view;
        ColiExpertStoreStats stats;
        unsigned char *slab = (unsigned char *)malloc((size_t)h->aligned_expert_bytes);
        int want, got;
        if (!slab) abort();
        key.layer = (int)(pick / h->num_experts);
        key.expert = (int)(pick % h->num_experts);
        want = fuzz_store_expect(file, h, e, key.layer, key.expert, model, has_model, slab);
        if ((hash >> 28) & 1u) {
            failed = fuzz_fault_failed;
            fuzz_fault_arm(0, 1);
            got = store->ops->prefetch(store, &key, 1u);
            fuzz_fault_disarm();
            FUZZ_REQUIRE(fuzz_fault_failed == failed + 1u, "a prefetch of an expert with a record made no read");
            FUZZ_REQUIRE(got == 0, "a prefetch whose read failed published the record");
            faults_prefetch++;
        }
        fails = mode == 7 ? 2 : 1;
        fuzz_setenv("WEX_READ_RETRY", mode >= 6 ? "1" : "");
        failed = fuzz_fault_failed;
        pauses = fuzz_fault_pauses;
        fuzz_fault_arm(0, fails);
        memset(&view, 0, sizeof view);
        got = coli_expert_lookup(store, key, &view);
        fuzz_fault_disarm();
        fuzz_setenv("WEX_READ_RETRY", "");
        FUZZ_REQUIRE(fuzz_fault_failed == failed + (unsigned)fails, "a lookup did not make the reads the pass failed");
        FUZZ_REQUIRE(fuzz_fault_pauses == pauses + (mode >= 6 ? 1u : 0u),
                     "a lookup whose read failed did not pause exactly once before its second read (WEX_READ_RETRY), "
                     "or paused without it");
        if (mode == 6)
            FUZZ_REQUIRE(!got == !want, "a lookup read again after a failed read (WEX_READ_RETRY) and the decoders "
                                        "disagree on a record");
        else
            FUZZ_REQUIRE(got != 0, "a lookup whose read failed, and was not read again without a failure, returns a record");
        if (!got) {
            FUZZ_REQUIRE(fuzz_store_view_is(&view, e, slab), "a lookup read again hands out a matrix that is not the decoded record");
            coli_expert_release(store, &view);
            faults_retry_served++;
        }
        memset(&view, 0, sizeof view);
        got = coli_expert_lookup(store, key, &view);
        FUZZ_REQUIRE(!got == !want, "a lookup after a failed read and the decoders disagree on a record");
        if (!got) {
            FUZZ_REQUIRE(fuzz_store_view_is(&view, e, slab), "a lookup after a failed read hands out a matrix that is "
                                                             "not the decoded record");
            coli_expert_release(store, &view);
            faults_served_after++;
        }
        store->ops->stats(store, &stats);
        FUZZ_REQUIRE(stats.requests == 2u && stats.hits + stats.misses == 2u,
                     "the store's counters do not add up to the lookups (failing reads)");
        faults_lookups++;
        faults_retry_one += mode == 6;
        faults_retry_two += mode == 7;
        free(slab);
    }
    store->ops->destroy(store);
    fuzz_setenv("COLI_MODEL_MIRROR", "");
    fuzz_setenv("COLI_DISK_WEIGHTS", "");
}

#endif /* WEX_FUZZ_STORE_PATHS_H */
