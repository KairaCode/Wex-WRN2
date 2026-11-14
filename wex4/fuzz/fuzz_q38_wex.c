/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Fuzzing harness for the container reader of the Qwen3.8 engine, c/qwen38_wex.h: the input is a whole
 * container (WEX1, a legacy WNR3 container, or WEX4).
 *
 * c/qwen38_wex.h is a part of the engine's qwen38.c and does not build alone. wex4/fuzz/engine/q38_engine.h
 * holds stand-ins for the engine types it names (read that file: the types are inferred, and what follows
 * from that). The code that reads the container is the real one: q38_wex_manifest and q38_wex4_manifest
 * (header page, table, WEX4 index, the legacy page a WEX4 file is turned into), q38_wex_entry_ok, the model
 * registration, q38_mx4_attach_wex and q38_mx4_load (the read of a record into its slab, the first-read
 * SHA-256 of COLI_WEX_VERIFY=1, the owner check and the decode).
 *
 * The engine takes the model's geometry from its configuration and stops (exit 1) on a container that does not
 * fit it. Here the geometry comes from the input: layers and experts from its header, hidden and intermediate
 * size from its first expert entry, so that a container of any geometry attaches. The harness first runs the
 * same checks the attach runs (manifest, every entry, the model); only when they pass does it call the real
 * q38_expert_mx4_attach, and then q38_mx4_load on its experts (all of them up to FUZZ_MAX_LOADS, else that
 * many spread over the grid). A WEX4 input is attached and loaded twice, without and with COLI_WEX_VERIFY=1.
 * exit() inside c/qwen38_wex.h is caught (a longjmp back to the harness), so "the engine stops" is a verdict.
 *
 * Besides what the sanitizers catch:
 *  - the attach does not stop on a container whose manifest, entries and model the same functions accepted;
 *  - every record of an attached container lies inside the file and its read fits its slab;
 *  - a load stops the engine exactly when the record's bytes, copied from the input, fail the first-read hash
 *    (when it is on), the owner check or wex_decode_nr2_record; otherwise the slot holds exactly those decoded
 *    bytes (for a raw record: the stored bytes).
 *
 * Limits of this harness, not of the reader: a grid of more than FUZZ_MAX_GRID experts or a record larger than
 * FUZZ_MAX_RECORD is not attached (the reader allocates what the model's own geometry asks for); the model
 * registry of c/wex_decode.c is emptied before every input; the log lines go nowhere; reads go through the
 * page cache (no O_DIRECT); no mirror, no striped read, no load wave, no CUDA helper, no Quark shards.
 *
 * The custom mutator: one call in two, when the input is a legacy container or a WEX4 file that opens and has
 * WNR3 records, one of them is mutated with wnr3_mutator.h (decoded, edited, encoded again, so its stream
 * hashes are right) and written back if it still takes the same pages. In a WEX4 file the record's SHA-256 in
 * the index is left alone: the new record loads without COLI_WEX_VERIFY and stops the engine with it. Every
 * other call is libFuzzer's own mutation. */
#define _FILE_OFFSET_BITS 64
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include "wex_format.h"
#include "engine/q38_engine.h"

static jmp_buf fuzz_stop;
static int fuzz_armed;
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
static void fuzz_q38_exit(int code) {
    if (fuzz_armed) longjmp(fuzz_stop, 1);
    exit(code);
}
static int fuzz_quiet(FILE *stream, const char *format, ...) { (void)stream; (void)format; return 0; }
#define fprintf fuzz_quiet
#define exit(code) fuzz_q38_exit(code)
#include "../../c/qwen38_wex.h"
#undef exit
#undef fprintf
#define FUZZ_MUTATOR_MAX_LEN 1048576u
#define FUZZ_MUTATOR_ROUNDS 40
#include "fuzz_common.h"
#include "wnr3_mutator.h"

#define FUZZ_MAX_GRID 4096u
#define FUZZ_MAX_RECORD (INT64_C(16) << 20)
#define FUZZ_MAX_LOADS 64

static char temp_path[1024];

/* what the run reached, printed at exit */
static unsigned long long count_inputs, count_attached, count_attached_wex4, count_loads, count_loaded, count_loaded_coded;

static void fuzz_cleanup(void) {
    remove(temp_path);
    fprintf(stderr, "fuzz_q38_wex: %llu inputs, %llu attached (%llu of them WEX4), %llu loads, %llu records loaded, %llu of "
            "them decoded from a WNR3 record, %llu%s different ones (structured mutations %s)\n", count_inputs,
            count_attached, count_attached_wex4, count_loads, count_loaded, count_loaded_coded, fuzz_distinct,
            fuzz_distinct >= FUZZ_SEEN_MAX ? " or more" : "", fuzz_structured() ? "on" : "off");
}

static void fuzz_setup(void) {
    static const char *const unset[] = {"Q38_MXFP4_DIR", "COLI_WEX_ROOT", "COLI_WEX_EXTRA_MIRRORS", "COLI_WEX_WEIGHTS",
                                        "COLI_WEX_STRIPE", "COLI_WEX_GPU_DIRECT", "DSV4_CUDA_MOE_MASKED"};
    size_t i;
    fuzz_temp_path(temp_path, sizeof temp_path, "q38.wex");
    fuzz_setenv("COLI_WEX_PATH", temp_path);
    fuzz_setenv("COLI_WEX_CUDA_LIB", "0");   /* the CPU decoder: no helper is looked for */
    for (i = 0; i < sizeof unset / sizeof unset[0]; i++) fuzz_setenv(unset[i], "");
    atexit(fuzz_cleanup);
}

/* the registry of c/wex_decode.c, emptied: its models are heap copies that nothing else holds */
static void fuzz_forget_models(void) {
    int i;
    for (i = 0; i < wex_nr2_model_count; i++) free((void *)wex_nr2_models[i].blob);
    wex_nr2_model_count = 0;
#ifdef _WIN32
    memset(wex_nr2_helper_ids, 0, sizeof wex_nr2_helper_ids);
#endif
}

/* 1 when the engine stopped (exit) inside the call */
static int fuzz_attach(Model *m) {
    fuzz_armed = 1;
    if (setjmp(fuzz_stop)) { fuzz_armed = 0; return 1; }
    q38_expert_mx4_attach(m);
    fuzz_armed = 0;
    return 0;
}
static int fuzz_load(Model *m, int layer, int expert, Slot *slot) {
    fuzz_armed = 1;
    if (setjmp(fuzz_stop)) { fuzz_armed = 0; return 1; }
    q38_mx4_load(m, layer, expert, slot, 0, 1);
    fuzz_armed = 0;
    return 0;
}

/* layers and experts as the input's own header states them; 1 x 1 when it states none the harness takes */
static void fuzz_grid(const unsigned char *file, size_t size, Cfg *c) {
    uint64_t layers = 0, experts = 0;
    if (size >= WEX4_PAGE && wex_read_u32_le(file) == WEX4_MAGIC) {
        Wex4Header h;
        if (!wex4_parse_header(file, size, &h)) { layers = h.layers; experts = h.experts; }
    } else if (size >= sizeof(WexHeader)) {
        const WexHeader *h = (const WexHeader *)file;
        layers = h->num_layers; experts = h->num_experts;
    }
    if (!layers || !experts || layers > FUZZ_MAX_GRID || experts > FUZZ_MAX_GRID || layers * experts > FUZZ_MAX_GRID)
        layers = experts = 1u;
    c->layers = (int)layers;
    c->experts = (int)experts;
}

/* The checks of q38_mx4_attach_wex, without its exit: 1 when the container must attach. Fills the rest of the
 * geometry, and the container's model when it has one. */
static int fuzz_attaches(const unsigned char *file, size_t size, Cfg *c, WexNr2Model *model) {
    unsigned char *page = NULL, *table = NULL;
    size_t table_bytes = 0;
    int64_t file_bytes = 0, scale_off[3], weight_off[3], raw = 0, record;
    int fd = q38_wex_open(temp_path), wex4 = 0, ok = 0, nr3, i, n = c->layers * c->experts;
    if (fd < 0) { fprintf(stderr, "%s: cannot be opened\n", temp_path); exit(2); }
    if (!q38_wex_manifest(fd, c, &page, &table, &table_bytes, &file_bytes, &wex4)) {
        const WexHeader *h = (const WexHeader *)page;
        const WexEntry *entries = (const WexEntry *)table;
        nr3 = !!(h->flags & WEX_HEADER_NEURAL_RANS3);
        for (i = 0; i < n && !entries[i].total_bytes; i++) {}
        if (i < n && entries[i].weight_shapes[0][0] && entries[i].weight_shapes[0][1]) {
            c->inter = entries[i].weight_shapes[0][0];
            c->hidden = entries[i].weight_shapes[0][1] * 2;
            record = q38_mx4_layout(c, scale_off, weight_off, &raw);
            ok = record <= FUZZ_MAX_RECORD;
            for (i = 0; ok && i < n; i++) ok = q38_wex_entry_ok(&entries[i], c, scale_off, weight_off, raw, record, nr3);
            if (ok && nr3) {
                uint64_t at = 0, aligned = 0;
                uint32_t bytes = 0, id = 0;
                ok = !wex_nr2_model_location(h, page, &at, &bytes, &id) && !wex_aligned_size(bytes, &aligned) &&
                     at <= size && aligned <= size - at && !wex_nr2_model_open(file + at, bytes, model) && model->id == id;
            }
        }
    }
    compat_aligned_free(page);
    compat_aligned_free(table);
    close(fd);
    return ok;
}

size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size, unsigned int seed) {
    if (fuzz_structured() && !(seed % 2u) &&
        (fuzz_wnr3_mutate_legacy(data, size, seed) || fuzz_wnr3_mutate_wex4(data, data, size, seed))) return size;
    return LLVMFuzzerMutate(data, size, max_size);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    unsigned char *file, *slab;
    WexNr2Model model;
    Model m;
    Slot slot;
    int pass, passes = 1;
    if (!temp_path[0]) fuzz_setup();
    count_inputs++;
    file = fuzz_exact_copy(data, size);
    fuzz_write_file(temp_path, data, size);
    fuzz_forget_models();
    memset(&m, 0, sizeof m);
    memset(&model, 0, sizeof model);
    fuzz_grid(file, size, &m.c);
    m.range_end = m.c.layers;
    if (!fuzz_attaches(file, size, &m.c, &model)) { free(file); return 0; }

    for (pass = 0; pass < passes; pass++) {
        Q38Int4Experts *x;
        int n = m.c.layers * m.c.experts, step = n > FUZZ_MAX_LOADS ? n / FUZZ_MAX_LOADS : 1, i;
        fuzz_setenv("COLI_WEX_VERIFY", pass ? "1" : "0");
        m.x4 = NULL;
        FUZZ_REQUIRE(!fuzz_attach(&m) && m.x4,
                     "the attach stops on a container whose manifest, entries and model the same functions accepted");
        x = m.x4;
        FUZZ_REQUIRE(!pass || x->wex_sha, "COLI_WEX_VERIFY=1 on a WEX4 container left the first-read hash off");
        if (x->wex4) passes = 2;
        if (!pass) { count_attached++; count_attached_wex4 += x->wex4 != 0; }
        slab = (unsigned char *)malloc((size_t)x->record_bytes);
        if (!slab) abort();
        memset(&slot, 0, sizeof slot);
        for (i = 0; i < n; i += step) {
            const WexEntry *e = &x->wex_entry[i];
            size_t tail = wex_record_read_offset(e);
            int layer = i / m.c.experts, expert = i % m.c.experts, stop = 0, stopped;
            FUZZ_REQUIRE(e->file_offset <= size && e->total_bytes <= size - e->file_offset,
                         "a record of an attached container lies outside the file");
            FUZZ_REQUIRE(tail <= (uint64_t)x->record_bytes && e->total_bytes <= (uint64_t)x->record_bytes - tail,
                         "a record of an attached container is read past its slab");
            memcpy(slab + tail, file + e->file_offset, e->total_bytes);
            if (x->wex_sha) {
                unsigned char digest[32];
                wex4_sha256(slab + tail, e->total_bytes, digest);
                stop = memcmp(digest, x->wex_sha + (size_t)i * 32u, 32u) != 0;
            }
            if (!stop && e->flags)
                stop = !wex_nr3_owner_ok(e, slab + tail, (uint32_t)layer, (uint32_t)expert, x->wex_flags) ||
                       wex_read_u32_le(slab + tail + 4u) != model.id ||
                       wex_decode_nr2_record(e, slab, (size_t)x->record_bytes, &model) != 0;
            stopped = fuzz_load(&m, layer, expert, &slot);
            FUZZ_REQUIRE(stopped == stop, "a load and the checks of wex_format.h disagree on a record");
            FUZZ_REQUIRE(stopped || !memcmp(q38_mx4_record(&slot), slab, (size_t)x->mx_raw),
                         "a loaded slot does not hold the decoded record");
            count_loads++;
            if (!stopped) {
                count_loaded++;
                if (e->flags) { count_loaded_coded++; fuzz_note_distinct(slab, (size_t)x->mx_raw); }
            }
        }
        free(slot.int4_slab);
        free(slab);
        q38_mx4_free(x);
        free(x);
    }
    free(file);
    return 0;
}
