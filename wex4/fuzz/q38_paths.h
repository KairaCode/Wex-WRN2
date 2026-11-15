/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_Q38_PATHS_H
#define WEX_FUZZ_Q38_PATHS_H
/* fuzz_q38_wex.c: a second pass over every container that attached, through the read paths of c/qwen38_wex.h
 * that the main pass leaves alone. Included after fuzz_common.h and engine_paths.h. The container is attached
 * again with:
 *  - COLI_WEX_EXTRA_MIRRORS naming one to four directories, each with a copy of the input under its file name
 *    in one of the forms of engine_paths.h, or the input's own directory; the separators are ';', ',' or ", ",
 *    with an empty entry for one input in 4;
 *  - COLI_WEX_WEIGHTS from a table of settings (fuzz_q38_weights);
 *  - COLI_WEX_STRIPE=1 for one input in 2, and COLI_WEX_VERIFY=1 for one WEX4 input in 2.
 * Then it loads experts (all of them up to FUZZ_Q38_PATHS_LOADS, else that many spread over the grid), each one
 * alone (the striped read, when it is on) or not, by a bit of the input's hash. When the CUDA helper offers the
 * load wave (q38_mx4_wave: a WNR3 container and helper features 16 and 32, so Windows with WEX_FUZZ_CUDA_LIB) it
 * also runs one wave of up to FUZZ_Q38_WAVE experts of one layer, as the engine does: deferred loads, then
 * q38_mx4_decode_wave. For one input in 2 FAKE_WEX_BATCH_FAIL names one record of that wave: the stand-in
 * helper c/tests/fake_wex_cuda.c then reports it unavailable after writing over its slab, and the wave must
 * read and decode it again alone.
 *
 * Besides what the sanitizers catch:
 *  - the attach does not stop; it takes, in the order listed and at most three, the mirrors whose copy has the
 *    input's bytes or differs in record bytes only (forms SAME and RECORDS, and the input's own directory), and
 *    ignores the others (PAGE, SIZE, TABLE, MISSING);
 *  - the split's cuts are the shares of 256 of the setting's weights when it has one positive weight per drive,
 *    else equal shares; the stripe is on exactly with COLI_WEX_STRIPE=1 and a mirror taken;
 *  - an expert is read from the first drive whose cut is above the split byte of its flat index; a striped read
 *    takes the record's first `cut` bytes from that drive and the rest from the next one, `cut` being the two
 *    drives' shares of the record's 4096-byte pages, at least one page each. A load stops the engine exactly
 *    when the bytes so put together fail the first-read hash (when it is on), the owner check or the decode,
 *    and otherwise the slot holds the decoded record. A copy in form RECORDS shows a read from the wrong drive
 *    or across the wrong cut;
 *  - a wave stops the engine exactly when one of its WNR3 records, which passed its deferred load, does not
 *    decode; otherwise every slot of the wave holds its decoded record.
 *
 * A wave takes only records that name the container's model: the stand-in helper keeps the models of every
 * input of the process, where the harness empties the registry of c/wex_decode.c before each one. A wave the
 * engine stops in leaks its five arrays (the longjmp skips their free); that happens on Windows only, where
 * no leak checker runs.
 *
 * fuzz_q38_faults, at the end of this file, is a third pass: reads that fail (read_faults.h) and the root pin
 * of COLI_WEX_ROOT. */

#define FUZZ_Q38_PATHS_LOADS 32
#define FUZZ_Q38_WAVE 8

/* COLI_WEX_WEIGHTS settings. count: the number of weights it gives; -1, a setting c/qwen38_wex.h must ignore */
static const struct { const char *text; int count; double w[4]; } fuzz_q38_weights[] = {
    {"", 0, {0, 0, 0, 0}},            {"1,1", 2, {1, 1, 0, 0}},          {"38,62", 2, {38, 62, 0, 0}},
    {"3 1", 2, {3, 1, 0, 0}},         {"0.25, 4", 2, {0.25, 4, 0, 0}},   {"1e300,1", 2, {1e300, 1, 0, 0}},
    {"1,1e300", 2, {1, 1e300, 0, 0}}, {"1,2,3", 3, {1, 2, 3, 0}},        {"5,1,1", 3, {5, 1, 1, 0}},
    {"1,1,1,1", 4, {1, 1, 1, 1}},     {"7,1,1,2", 4, {7, 1, 1, 2}},      {"1,0", -1, {0, 0, 0, 0}},
    {"2,-1", -1, {0, 0, 0, 0}},       {"x,1", -1, {0, 0, 0, 0}},         {"nan,1", -1, {0, 0, 0, 0}},
    {"1,1,1,1,1", -1, {0, 0, 0, 0}}};
#define FUZZ_Q38_SETTINGS (sizeof fuzz_q38_weights / sizeof fuzz_q38_weights[0])
/* the forms of a mirror: those of engine_paths.h, and the input's own directory */
#define FUZZ_Q38_OWN FUZZ_COPY_FORMS

/* what the pass reached, printed at exit */
static unsigned long long q38p_passes, q38p_listed, q38p_taken, q38p_forms[FUZZ_COPY_FORMS + 1], q38p_drives[5],
    q38p_custom, q38p_loads, q38p_striped, q38p_loaded, q38p_from_mirror, q38p_stops, q38p_waves, q38p_wave_records,
    q38p_wave_stops, q38p_wave_fail;

static void fuzz_q38_paths_report(void) {
    fuzz_mirror_cleanup();
    fprintf(stderr, "fuzz_q38_wex other paths: %llu passes, %llu mirrors listed (same %llu, records %llu, page %llu, "
            "size %llu, table %llu, missing %llu, own directory %llu), %llu taken; drives 1/2/3/4: %llu/%llu/%llu/%llu, "
            "%llu with the setting's weights; %llu loads (%llu striped), %llu records loaded (%llu with bytes from a "
            "mirror), %llu stops; %llu load waves of %llu records, %llu of them stopped, %llu with a record the helper "
            "failed on purpose; CUDA helper %s, %llu records decoded through it\n",
            q38p_passes, q38p_listed, q38p_forms[0], q38p_forms[1], q38p_forms[2], q38p_forms[3], q38p_forms[4],
            q38p_forms[5], q38p_forms[6], q38p_taken, q38p_drives[1], q38p_drives[2], q38p_drives[3], q38p_drives[4],
            q38p_custom, q38p_loads, q38p_striped, q38p_loaded, q38p_from_mirror, q38p_stops, q38p_waves,
            q38p_wave_records, q38p_wave_stops, q38p_wave_fail, strcmp(fuzz_cuda_lib(), "0") ? "on" : "off",
            (unsigned long long)coli_wex_cuda_decode_count());
}

/* the cuts of the read split over n drives: drive r takes the split bytes below cut[r] */
static void fuzz_q38_cuts(size_t setting, int n, int cut[4]) {
    double w[4] = {1, 1, 1, 1}, sum = 0, acc = 0;
    int r;
    if (fuzz_q38_weights[setting].count == n)
        for (r = 0; r < n; r++) w[r] = fuzz_q38_weights[setting].w[r];
    for (r = 0; r < n; r++) sum += w[r];
    for (r = 0; r < n; r++) { acc += w[r]; cut[r] = (int)(acc / sum * 256.0 + 0.5); }
    cut[n - 1] = 256;
}
static int fuzz_q38_drive(const int *cut, uint32_t flat) {
    int r = 0, byte = (int)fuzz_split_byte(flat);
    while (byte >= cut[r]) r++;
    return r;
}
/* the bytes of a striped read of n bytes that come from drive r, the rest from drive s; 0: not striped */
static int64_t fuzz_q38_stripe(const int *cut, int r, int s, int64_t n) {
    double wr = cut[r] - (r ? cut[r - 1] : 0), ws = cut[s] - (s ? cut[s - 1] : 0);
    int64_t pages = n / WEX_ALIGN, first;
    if (n % WEX_ALIGN || pages < 2 || !(wr + ws > 0)) return 0;
    first = (int64_t)((double)pages * wr / (wr + ws) + 0.5);
    if (first < 1) first = 1;
    if (first > pages - 1) first = pages - 1;
    return first * WEX_ALIGN;
}

/* 1 when the engine stopped (exit) inside the call */
static int fuzz_q38_attach_as(Model *m) {
    fuzz_armed = 1;
    if (setjmp(fuzz_stop)) { fuzz_armed = 0; return 1; }
    q38_expert_mx4_attach(m);
    fuzz_armed = 0;
    return 0;
}
static int fuzz_q38_load_as(Model *m, int layer, int expert, Slot *slot, int defer, int alone) {
    fuzz_armed = 1;
    if (setjmp(fuzz_stop)) { fuzz_armed = 0; return 1; }
    q38_mx4_load(m, layer, expert, slot, defer, alone);
    fuzz_armed = 0;
    return 0;
}
static int fuzz_q38_wave_run(Model *m, int layer, int n, const int *experts, Slot *const *slots) {
    fuzz_armed = 1;
    if (setjmp(fuzz_stop)) { fuzz_armed = 0; return 1; }
    q38_mx4_decode_wave(m, layer, n, experts, slots);
    fuzz_armed = 0;
    return 0;
}

/* Puts together in slab the bytes a load of entry i reads: from drive r, or its first `cut` bytes from drive r and
 * the rest from drive s. Returns 1 when the engine must stop on them; `deferred`: the WNR3 decode is left to a
 * wave (the hash and the owner check are not). */
static int fuzz_q38_expect(const Q38Int4Experts *x, size_t i, int layer, int expert, unsigned char *const *drive,
                           int r, int s, int64_t cut, const WexNr2Model *model, unsigned char *slab, int deferred) {
    const WexEntry *e = &x->wex_entry[i];
    size_t tail = wex_record_read_offset(e);
    memcpy(slab + tail, drive[r] + e->file_offset, cut ? (size_t)cut : e->total_bytes);
    if (cut) memcpy(slab + tail + cut, drive[s] + e->file_offset + cut, e->total_bytes - (size_t)cut);
    if (x->wex_sha) {
        unsigned char digest[32];
        wex4_sha256(slab + tail, e->total_bytes, digest);
        if (memcmp(digest, x->wex_sha + i * 32u, 32u)) return 1;
    }
    if (!e->flags) return 0;
    if (!wex_nr3_owner_ok(e, slab + tail, (uint32_t)layer, (uint32_t)expert, x->wex_flags)) return 1;
    if (deferred) return 0;
    return wex_read_u32_le(slab + tail + 4u) != model->id ||
           wex_decode_nr2_record(e, slab, (size_t)x->record_bytes, model) != 0;
}

static int fuzz_q38_overlaps(uint64_t a, uint64_t n, uint64_t b, uint64_t m) { return a < b + m && b < a + n; }

static void fuzz_q38_paths(Model *m, const char *primary, const unsigned char *file, size_t size,
                           const WexNr2Model *model) {
    static int reported;
    uint64_t hash = fuzz_paths_hash(file, size);
    int listed = 1 + (int)(hash % 4u), stripe = (hash >> 8) & 1u, verify = (hash >> 9) & 1u, k, n, i, step;
    int form[4], cut[4], want_cut[4], taken = 0, want_n, wex4, fd, stopped, alone;
    size_t setting = (size_t)((hash >> 16) % FUZZ_Q38_SETTINGS), table_bytes = 0, record_count = 0, length;
    int64_t file_bytes = 0;
    unsigned char *drive[4] = {NULL, NULL, NULL, NULL}, *copies[4] = {NULL, NULL, NULL, NULL}, *page = NULL,
                  *table = NULL, *slab;
    FuzzExtent *records, changed, keep[4];
    char list[8192], own[1024];
    const char *sep;
    Q38Int4Experts *x;
    Slot slot;

    if (!reported) { reported = 1; atexit(fuzz_q38_paths_report); }
    /* the attach splits the list at ';' and ','; a directory name with one cannot be listed */
    for (k = 0; k < 4; k++) if (strpbrk(fuzz_mirror_dir(k), ";,")) return;
    length = (size_t)(fuzz_basename(primary) - primary);
    snprintf(own, sizeof own, "%.*s", (int)(length ? length - 1u : 0u), primary);
    if (strpbrk(own, ";,") || !own[0]) return;
    q38p_passes++;

    /* the regions a mirror is compared on, and the extents of the records, from the manifest itself */
    fd = q38_wex_open(primary);
    if (fd < 0 || q38_wex_manifest(fd, &m->c, &page, &table, &table_bytes, &file_bytes, &wex4)) abort();
    close(fd);
    n = m->c.layers * m->c.experts;
    memset(keep, 0, sizeof keep);
    keep[0].offset = 0; keep[0].bytes = WEX_ALIGN;
    if (wex4) {
        Wex4Header h4;
        uint64_t at = 0, aligned = 0;
        uint32_t bytes = 0, id = 0;
        if (wex4_parse_header(file, size, &h4)) abort();
        keep[1].offset = h4.index_offset; keep[1].bytes = wex4_align(h4.index_bytes);
        changed.offset = h4.index_offset; changed.bytes = h4.index_bytes;
        if (!wex_nr2_model_location((const WexHeader *)page, page, &at, &bytes, &id) && !wex_aligned_size(bytes, &aligned)) {
            keep[2].offset = at; keep[2].bytes = aligned;
        }
    } else {
        const WexHeader *h = (const WexHeader *)file;
        keep[1].offset = h->table_offset; keep[1].bytes = wex4_align(table_bytes);
        changed.offset = h->table_offset; changed.bytes = table_bytes;
    }
    records = (FuzzExtent *)calloc((size_t)n, sizeof *records);
    if (!records) abort();
    for (i = 0; i < n; i++) {   /* form RECORDS changes record bytes only, never a compared region */
        const WexEntry *e = &((const WexEntry *)table)[i];
        int clear = e->total_bytes != 0;
        for (k = 0; clear && k < 4; k++)
            clear = !keep[k].bytes || !fuzz_q38_overlaps(e->file_offset, e->total_bytes, keep[k].offset, keep[k].bytes);
        if (clear) { records[record_count].offset = e->file_offset; records[record_count++].bytes = e->total_bytes; }
    }
    compat_aligned_free(page);
    compat_aligned_free(table);

    /* the mirrors, the list and what the attach must take */
    drive[0] = (unsigned char *)file;
    sep = (hash >> 12) % 3u == 0 ? ";" : (hash >> 12) % 3u == 1 ? "," : ", ";
    list[0] = 0;
    if (((hash >> 20) & 3u) == 0) snprintf(list, sizeof list, ";");   /* an empty first entry */
    for (k = 0; k < listed; k++) {
        uint64_t hk = fuzz_paths_mix(hash, (uint64_t)k);
        size_t used = strlen(list);
        form[k] = (int)(hk % (FUZZ_COPY_FORMS + 1u));
        if (!k && stripe && (hk >> 8) & 1u) form[k] = FUZZ_COPY_RECORDS;   /* a stripe across drives that differ */
        q38p_forms[form[k]]++;
        q38p_listed++;
        if (form[k] == FUZZ_Q38_OWN) {
            snprintf(list + used, sizeof list - used, "%s%s", k ? sep : "", own);
        } else {
            copies[k] = fuzz_mirror_write(k, primary, file, size, form[k], hk, records, record_count, changed);
            snprintf(list + used, sizeof list - used, "%s%s", k ? sep : "", fuzz_mirror_dir(k));
        }
        if (taken < 3 && (form[k] == FUZZ_COPY_SAME || form[k] == FUZZ_COPY_RECORDS || form[k] == FUZZ_Q38_OWN))
            drive[1 + taken++] = form[k] == FUZZ_Q38_OWN ? (unsigned char *)file : copies[k];
    }
    free(records);
    want_n = 1 + taken;
    fuzz_q38_cuts(setting, want_n, want_cut);

    fuzz_setenv("COLI_WEX_EXTRA_MIRRORS", list);
    fuzz_setenv("COLI_WEX_WEIGHTS", fuzz_q38_weights[setting].text);
    fuzz_setenv("COLI_WEX_STRIPE", stripe ? "1" : "");
    fuzz_setenv("COLI_WEX_VERIFY", wex4 && verify ? "1" : "0");
    m->x4 = NULL;
    stopped = fuzz_q38_attach_as(m);
    fuzz_setenv("COLI_WEX_EXTRA_MIRRORS", "");
    fuzz_setenv("COLI_WEX_WEIGHTS", "");
    fuzz_setenv("COLI_WEX_STRIPE", "");
    FUZZ_REQUIRE(!stopped && m->x4, "the attach stops on a container it took before, with mirrors listed");
    x = m->x4;
    FUZZ_REQUIRE(x->wex_n == want_n, "the attach takes another set of mirrors than the copies' forms say");
    for (k = 0; k < want_n; k++) cut[k] = x->wex_cut[k];
    FUZZ_REQUIRE(!memcmp(cut, want_cut, (size_t)want_n * sizeof cut[0]), "the read split is not the setting's shares");
    FUZZ_REQUIRE(x->wex_stripe == (stripe && want_n > 1), "the stripe is on or off against COLI_WEX_STRIPE");
    q38p_taken += (unsigned long long)taken;
    q38p_drives[want_n]++;
    q38p_custom += fuzz_q38_weights[setting].count == want_n;

    /* loads, each from the drive the split names */
    slab = (unsigned char *)malloc((size_t)x->record_bytes);
    if (!slab) abort();
    memset(&slot, 0, sizeof slot);
    step = n > FUZZ_Q38_PATHS_LOADS ? n / FUZZ_Q38_PATHS_LOADS : 1;
    for (i = 0; i < n; i += step) {
        const WexEntry *e = &x->wex_entry[i];
        int layer = i / m->c.experts, expert = i % m->c.experts, r = fuzz_q38_drive(want_cut, (uint32_t)i);
        int s = (r + 1) % want_n, stop, got;
        int64_t c;
        alone = (int)(fuzz_paths_mix(hash, 100u + (uint64_t)i) & 1u);
        c = alone && x->wex_stripe ? fuzz_q38_stripe(want_cut, r, s, (int64_t)e->total_bytes) : 0;
        stop = fuzz_q38_expect(x, (size_t)i, layer, expert, drive, r, s, c, model, slab, 0);
        got = fuzz_q38_load_as(m, layer, expert, &slot, 0, alone);
        FUZZ_REQUIRE(got == stop, "a load through the split and the checks of wex_format.h disagree on a record");
        FUZZ_REQUIRE(got || !memcmp(q38_mx4_record(&slot), slab, (size_t)x->mx_raw),
                     "a slot does not hold the record of the drives the split names");
        q38p_loads++;
        q38p_striped += c != 0;
        q38p_stops += got != 0;
        if (!got) { q38p_loaded++; q38p_from_mirror += r != 0 || c != 0; }
    }
    free(slot.int4_slab);

    /* one load wave, where the helper offers it */
    if (x->nr3 && q38_mx4_wave(m, 2)) {
        int layer = (int)((hash >> 24) % (uint64_t)m->c.layers), count = 0, experts[FUZZ_Q38_WAVE], j, failed;
        int coded = 0, fail_at = -1, want_stop = 0;
        Slot wave[FUZZ_Q38_WAVE], *slots[FUZZ_Q38_WAVE];
        unsigned char *expected[FUZZ_Q38_WAVE];
        memset(wave, 0, sizeof wave);
        for (i = 0; i < m->c.experts && count < FUZZ_Q38_WAVE; i++) {
            int flat = layer * m->c.experts + i, r = fuzz_q38_drive(want_cut, (uint32_t)flat), stop;
            const WexEntry *e = &x->wex_entry[flat];
            size_t tail = wex_record_read_offset(e);
            unsigned char *bytes = (unsigned char *)malloc((size_t)x->record_bytes);
            if (!bytes) abort();
            stop = fuzz_q38_expect(x, (size_t)flat, layer, i, drive, r, 0, 0, model, bytes, 1);
            if (!stop && e->flags && wex_read_u32_le(bytes + tail + 4u) != model->id) {   /* not in this wave */
                free(bytes);
                continue;
            }
            failed = fuzz_q38_load_as(m, layer, i, &wave[count], 1, 0);
            FUZZ_REQUIRE(failed == stop, "a deferred load and the checks of wex_format.h disagree on a record");
            if (failed) { free(bytes); free(wave[count].int4_slab); memset(&wave[count], 0, sizeof wave[count]); continue; }
            if (e->flags) {   /* the wave hands the helper its WNR3 records, in this order */
                coded++;
                want_stop |= wex_decode_nr2_record(e, bytes, (size_t)x->record_bytes, model) != 0;
            }
            expected[count] = bytes;
            experts[count] = i;
            slots[count] = &wave[count];
            count++;
        }
        if (count > 1) {
            char at[16] = "";
            if ((hash >> 32) & 1u && coded) {
                fail_at = (int)((hash >> 33) % (uint64_t)coded);
                snprintf(at, sizeof at, "%d", fail_at);
            }
            fuzz_setenv("FAKE_WEX_BATCH_FAIL", at);
            stopped = fuzz_q38_wave_run(m, layer, count, experts, slots);
            fuzz_setenv("FAKE_WEX_BATCH_FAIL", "");
            FUZZ_REQUIRE(stopped == want_stop, "a load wave and the decoder disagree on its records");
            for (j = 0; !stopped && j < count; j++)
                FUZZ_REQUIRE(!memcmp(q38_mx4_record(slots[j]), expected[j], (size_t)x->mx_raw),
                             "a slot of a load wave does not hold its decoded record");
            q38p_waves++;
            q38p_wave_records += (unsigned long long)count;
            q38p_wave_stops += stopped != 0;
            q38p_wave_fail += fail_at >= 0;
        }
        for (j = 0; j < count; j++) { free(expected[j]); free(wave[j].int4_slab); }
    }
    free(slab);
    q38_mx4_free(x);
    free(x);
    m->x4 = NULL;
    for (k = 0; k < 4; k++) free(copies[k]);
}

/* The third pass, over every container that attached: reads that fail (read_faults.h) and the root pin.
 *  - q38_wex_manifest is run once to count its reads, then with one of them failed. The harness calls it
 *    itself: through the attach a refusal stops the engine, which exits there with the page, the table and
 *    the descriptor it holds, and the harness, which goes on, would leak them;
 *  - for a WNR3 container, q38_wex_model with its read failed;
 *  - COLI_WEX_ROOT: q38_wex_manifest under the WEX4 file's own root (in capitals for one input in 2: the
 *    comparison ignores case), under that root with one digit changed, and for a legacy file under a root of
 *    zeros;
 *  - the container is attached again, for one input in 2 with a copy of the same bytes as a mirror and
 *    COLI_WEX_STRIPE=1, and one expert (from the input's hash) is loaded alone with one of its reads failed
 *    (either half of a striped read), then loaded again with none failed.
 *
 * Besides what the sanitizers catch:
 *  - the manifest reads the header page and its table or index (2 reads at least), and refuses the container
 *    when one of its reads fails; the model's read failed stops the engine;
 *  - the manifest takes a WEX4 file under its own root and refuses it under another, and refuses a legacy file
 *    under any root;
 *  - a load whose read failed stops the engine; the load after it stops exactly when the bytes of the drives
 *    the split names fail the owner check or the decode, and otherwise the slot holds the decoded record;
 *  - every read the pass failed was a read of c/qwen38_wex.h. */

static unsigned long long q38f_passes, q38f_manifest_reads, q38f_manifest_refused, q38f_model_stops, q38f_pins_taken,
    q38f_pins_refused, q38f_loads, q38f_loads_striped, q38f_loaded_after;

static void fuzz_q38_faults_report(void) {
    fprintf(stderr, "fuzz_q38_wex failing reads: %llu passes, %llu reads of a manifest, %llu manifests with a failed "
            "read refused, %llu model reads failed (the engine stopped), COLI_WEX_ROOT: %llu pins taken, %llu refused; "
            "%llu loads with a failed read stopped (%llu of them striped), %llu records loaded after one\n",
            q38f_passes, q38f_manifest_reads, q38f_manifest_refused, q38f_model_stops, q38f_pins_taken,
            q38f_pins_refused, q38f_loads, q38f_loads_striped, q38f_loaded_after);
}

/* 1 when the engine stopped (exit) inside the call */
static int fuzz_q38_model_as(int fd, const unsigned char *page, const char *path) {
    fuzz_armed = 1;
    if (setjmp(fuzz_stop)) { fuzz_armed = 0; return 1; }
    q38_wex_model(fd, page, path);
    fuzz_armed = 0;
    return 0;
}
/* q38_wex_manifest of fd, with its buffers freed: its result */
static int fuzz_q38_manifest_rc(int fd, const Cfg *c) {
    unsigned char *page = NULL, *table = NULL;
    size_t table_bytes = 0;
    int64_t file_bytes = 0;
    int wex4 = 0, rc = q38_wex_manifest(fd, c, &page, &table, &table_bytes, &file_bytes, &wex4);
    if (page) compat_aligned_free(page);
    if (table) compat_aligned_free(table);
    return rc;
}

static void fuzz_q38_faults(Model *m, const char *primary, const unsigned char *file, size_t size,
                            const WexNr2Model *model) {
    static int reported;
    uint64_t hash = fuzz_paths_mix(fuzz_paths_hash(file, size), 1000u);   /* not the second pass's choices */
    int fd, wex4 = 0, nr3, rc, stopped, want, i, layer, expert, r, s, n = m->c.layers * m->c.experts;
    int striped = (int)(hash & 1u) && !strpbrk(fuzz_mirror_dir(0), ";,");
    unsigned long long reads, failed;
    unsigned char *page = NULL, *table = NULL, *copy = NULL, *slab, *drive[4];
    size_t table_bytes = 0;
    int64_t file_bytes = 0, cut;
    const WexEntry *e;
    Q38Int4Experts *x;
    Slot slot;
    FuzzExtent none = {0, 0};
    char hex[65];

    if (!reported) { reported = 1; atexit(fuzz_q38_faults_report); }
    q38f_passes++;
    fd = q38_wex_open(primary);
    if (fd < 0) abort();

    /* the manifest's reads, then the manifest with one of them failed */
    reads = fuzz_fault_reads;
    FUZZ_REQUIRE(!q38_wex_manifest(fd, &m->c, &page, &table, &table_bytes, &file_bytes, &wex4),
                 "the manifest of a container that attached fails when it is read again");
    reads = fuzz_fault_reads - reads;
    FUZZ_REQUIRE(reads >= 2u, "the manifest takes a container without reading its header page and its table or index");
    nr3 = !!(((const WexHeader *)page)->flags & WEX_HEADER_NEURAL_RANS3);
    hex[0] = 0;
    for (i = 0; wex4 && i < 32; i++)   /* the root, which the rewritten page keeps at byte 4064 */
        snprintf(hex + 2 * i, 3, (hash >> 40) & 1u ? "%02X" : "%02x", page[WEX4_HEADER_HASHED + i]);
    failed = fuzz_fault_failed;
    fuzz_fault_arm((long long)((hash >> 8) % reads), 1);
    rc = fuzz_q38_manifest_rc(fd, &m->c);
    fuzz_fault_disarm();
    FUZZ_REQUIRE(fuzz_fault_failed == failed + 1u, "a read the pass failed was not made by the manifest");
    FUZZ_REQUIRE(rc != 0, "the manifest takes a container although one of its reads failed");
    q38f_manifest_reads += reads;
    q38f_manifest_refused++;

    if (nr3) {   /* the model's read failed */
        failed = fuzz_fault_failed;
        fuzz_fault_arm(0, 1);
        stopped = fuzz_q38_model_as(fd, page, primary);
        fuzz_fault_disarm();
        FUZZ_REQUIRE(fuzz_fault_failed == failed + 1u, "a read the pass failed was not made by the model's load");
        FUZZ_REQUIRE(stopped, "the model of a WNR3 container registers although its read failed");
        q38f_model_stops++;
    }

    if (wex4) {   /* COLI_WEX_ROOT */
        int right, wrong;
        fuzz_setenv("COLI_WEX_ROOT", hex);
        right = fuzz_q38_manifest_rc(fd, &m->c);
        i = (int)((hash >> 44) % 64u);
        hex[i] = hex[i] == '0' ? '1' : '0';
        fuzz_setenv("COLI_WEX_ROOT", hex);
        wrong = fuzz_q38_manifest_rc(fd, &m->c);
        fuzz_setenv("COLI_WEX_ROOT", "");
        FUZZ_REQUIRE(right == 0, "a WEX4 container is refused under its own root (COLI_WEX_ROOT)");
        FUZZ_REQUIRE(wrong != 0, "a WEX4 container is taken under another root (COLI_WEX_ROOT)");
        q38f_pins_taken++;
        q38f_pins_refused++;
    } else {
        fuzz_setenv("COLI_WEX_ROOT", "0000000000000000000000000000000000000000000000000000000000000000");
        rc = fuzz_q38_manifest_rc(fd, &m->c);
        fuzz_setenv("COLI_WEX_ROOT", "");
        FUZZ_REQUIRE(rc != 0, "a container that is not WEX4 is taken under a pinned root (COLI_WEX_ROOT)");
        q38f_pins_refused++;
    }
    compat_aligned_free(page);
    compat_aligned_free(table);
    close(fd);

    /* one load with a failed read, then the same load with none */
    if (striped) copy = fuzz_mirror_write(0, primary, file, size, FUZZ_COPY_SAME, hash, NULL, 0, none);
    fuzz_setenv("COLI_WEX_EXTRA_MIRRORS", striped ? fuzz_mirror_dir(0) : "");
    fuzz_setenv("COLI_WEX_STRIPE", striped ? "1" : "");
    fuzz_setenv("COLI_WEX_VERIFY", "0");
    m->x4 = NULL;
    stopped = fuzz_q38_attach_as(m);
    fuzz_setenv("COLI_WEX_EXTRA_MIRRORS", "");
    fuzz_setenv("COLI_WEX_STRIPE", "");
    FUZZ_REQUIRE(!stopped && m->x4, "the attach stops on a container it took before (failing reads)");
    x = m->x4;
    FUZZ_REQUIRE(x->wex_n == 1 + striped, "the attach does not take a copy of the same bytes as its mirror");
    i = (int)((hash >> 16) % (uint64_t)n);
    layer = i / m->c.experts;
    expert = i % m->c.experts;
    e = &x->wex_entry[i];
    r = fuzz_q38_drive(x->wex_cut, (uint32_t)i);
    s = (r + 1) % x->wex_n;
    cut = x->wex_stripe ? fuzz_q38_stripe(x->wex_cut, r, s, (int64_t)e->total_bytes) : 0;
    memset(&slot, 0, sizeof slot);
    failed = fuzz_fault_failed;
    fuzz_fault_arm(cut ? (long long)((hash >> 20) & 1u) : 0, 1);   /* a striped read: either half */
    stopped = fuzz_q38_load_as(m, layer, expert, &slot, 0, 1);
    fuzz_fault_disarm();
    FUZZ_REQUIRE(fuzz_fault_failed == failed + 1u, "a read the pass failed was not made by the load");
    FUZZ_REQUIRE(stopped, "a load goes on although its read failed");
    slab = (unsigned char *)malloc((size_t)x->record_bytes);
    if (!slab) abort();
    drive[0] = drive[2] = drive[3] = (unsigned char *)file;
    drive[1] = striped ? copy : (unsigned char *)file;
    want = fuzz_q38_expect(x, (size_t)i, layer, expert, drive, r, s, cut, model, slab, 0);
    stopped = fuzz_q38_load_as(m, layer, expert, &slot, 0, 1);
    FUZZ_REQUIRE(stopped == want, "a load after a failed read and the checks of wex_format.h disagree on a record");
    FUZZ_REQUIRE(stopped || !memcmp(q38_mx4_record(&slot), slab, (size_t)x->mx_raw),
                 "a load after a failed read does not leave the decoded record in its slot");
    q38f_loads++;
    q38f_loads_striped += cut != 0;
    q38f_loaded_after += !stopped;
    free(slab);
    free(slot.int4_slab);
    q38_mx4_free(x);
    free(x);
    m->x4 = NULL;
    free(copy);
}

#endif /* WEX_FUZZ_Q38_PATHS_H */
