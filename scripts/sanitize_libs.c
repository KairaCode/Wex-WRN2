/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* The libraries that the Python tests load through ctypes, driven from C instead, so that every object of the
 * process can be built under ThreadSanitizer or MemorySanitizer (scripts/sanitize_c.sh, step 5). The program loads
 * them the way Python does (dlopen, LoadLibrary on Windows) and calls the exported functions that
 * wex4/test_wex4.py, wex4/wex4.py, wex4/wex4_convert.py and c/tools/build_nr3.py call:
 *
 *   sanitize_libs FAST REF NR3 WEX1 WNR3 WEX4 VECTORS [THREADS [MUTATIONS]]
 *   sanitize_libs --control LIB    calls race() of LIB on 2 threads at once and uninit() once: with a library that
 *                                  has a defect there, the sanitizer must report it (scripts/sanitize_c.sh)
 *
 *   FAST, REF  wex4/wex4_tools.c built with the fast WNR3 decoder (-DWEX4_FAST_NR3) and with the reference one
 *   NR3        c/tools/wex_nr3_tools.c, the WNR3 encoder
 *   WEX1       a WEX1 container of raw experts (c/tools/repack_v4_wex.py)
 *   WNR3       the WEX3 container that c/tools/build_nr3.py made of WEX1, trained on all its experts, without --cz
 *   WEX4       the same file after wex4/wex4_convert.py upgrade
 *   VECTORS    the folder wex4/vectors (MANIFEST.tsv)
 *   THREADS    default 4: the threads of the parts that the Python tools run on a thread pool
 *   MUTATIONS  default 200: mutations of wex4_fuzz_run per file and mode, a fifth in mode 1 and fewer for a file
 *              above 150,000 bytes, as test_fuzz scales them
 *
 * What runs, each part named after the Python code it follows:
 *   1. test_sha: wex4_sha256_bytes on 203 lengths (FAST and REF agree) and three known answers
 *   2. test_bp: WBP1 records encoded by FAST and REF (the same bytes), decoded by both, refused for another owner
 *   3. test_vectors: every row of MANIFEST.tsv, and WEX4, through wex4_check and wex4_fuzz_check of FAST and REF:
 *      the listed verdict and error class, the SHA-256 of the file and its root
 *   4. wex4.Reader and test_upgrade: on every accepted file wex4_validate, wex4_decode_record_buf of every record
 *      (FAST and REF give the same bytes) and wex4_expert_table, which for WEX4 must give the table of WNR3
 *   5. test_store: wex4_store_probe on the files with an expert grid: the table, the records that pass their
 *      first-read SHA-256, a wrong pinned root, a table over budget, and (WEX4) a record with a flipped bit
 *   6. test_fuzz: wex4_fuzz_run in its 3 modes on 3 files with FAST and REF, 18 calls on THREADS threads at once
 *      (WEX4 is the third file: its records give the fast decoder 4 OpenMP tasks per weight section). As in the
 *      Python tests, the OpenMP runtime has already been started by the main thread (part 3): on clang 18 with
 *      Archer, a correct OpenMP loop that 4 threads started for the first time at once crashed in libomp
 *      (__kmpc_end_critical) in some runs, in a C program as in Python
 *   7. build_nr3.py: the WNM2 model trained on every expert of WEX1 and each expert encoded, verified and decoded,
 *      on THREADS threads; the model and the records must be those of WNR3 byte for byte. The same experts again
 *      with canonical zero (--cz). wex_nr3_check_container on WNR3
 *   8. the records encoded in 7 decoded by FAST and REF through wex4_decode_record_buf: the source slabs of WEX1
 *
 * Output buffers that a function promises to fill are left uninitialised (malloc), so that MemorySanitizer reports
 * a byte that the library did not write when it is compared. Exit 0 and the last line "sanitize_libs: all checks
 * passed" when every check holds; exit 1 and a "FAILED" line otherwise. */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include "wex4.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef __GNUC__
__attribute__((noreturn, format(printf, 2, 3)))
#endif
static void fail(int line, const char *format, ...) {
    va_list args;
    fprintf(stderr, "sanitize_libs: FAILED (line %d): ", line);
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}
#define CHECK(condition, ...) do { if (!(condition)) fail(__LINE__, __VA_ARGS__); } while (0)

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1u);
    CHECK(p != NULL, "out of memory (%lu bytes)", (unsigned long)n);
    return p;
}
static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1u, size);
    CHECK(p != NULL, "out of memory");
    return p;
}
static unsigned char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    unsigned char *data;
    long n = 0;
    CHECK(f != NULL, "%s: cannot be opened", path);
    CHECK(!fseek(f, 0, SEEK_END) && (n = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET), "%s: no size", path);
    data = (unsigned char *)xmalloc((size_t)n);
    CHECK(fread(data, 1, (size_t)n, f) == (size_t)n, "%s: short read", path);
    fclose(f);
    *size = (size_t)n;
    return data;
}

/* ---- the libraries, bound as ctypes binds them ------------------------------------------------------------- */
typedef void (*Fn)(void);   /* a function pointer of any type converts to this one and back */
static Fn lib_sym(void *lib, const char *path, const char *name) {
    Fn f = NULL;
#ifdef _WIN32
    f = (Fn)GetProcAddress((HMODULE)lib, name);
#else
    void *p = dlsym(lib, name);
    memcpy(&f, &p, sizeof f);   /* POSIX: an object pointer from dlsym holds a function pointer */
#endif
    CHECK(f != NULL, "%s: no symbol %s", path, name);
    return f;
}
static void *lib_open(const char *path) {
#ifdef _WIN32
    void *lib = (void *)LoadLibraryA(path);
#else
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);   /* ctypes.CDLL: RTLD_LOCAL */
#endif
    CHECK(lib != NULL, "%s: cannot be loaded", path);
    return lib;
}
#define BIND(lib, path, field, name) \
    do { Fn f_ = lib_sym(lib, path, name); memcpy(&(field), &f_, sizeof f_); } while (0)

typedef struct {
    const char *path;
    uint32_t (*abi)(void);
    void (*sha)(const unsigned char *, size_t, unsigned char *);
    int (*bp_encode)(const unsigned char *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, unsigned char *, size_t,
                     size_t *);
    int (*bp_decode)(const unsigned char *, size_t, uint32_t, uint32_t, unsigned char *);
    int (*check)(const unsigned char *, uint64_t, int, unsigned char *);
    int (*fuzz_check)(const unsigned char *, uint64_t, int, unsigned char *);
    int (*validate)(const unsigned char *, uint64_t, const unsigned char *, size_t);
    int (*decode_buf)(const unsigned char *, uint64_t, const unsigned char *, size_t, uint32_t, const unsigned char *,
                      const unsigned char *, int, unsigned char *);
    int (*expert_table)(const unsigned char *, uint64_t, const unsigned char *, size_t, unsigned char *);
    int (*fuzz_run)(const unsigned char *, uint64_t, uint64_t, uint32_t, int, uint64_t *);
    int (*store_probe)(const char *, const unsigned char *, unsigned char *, uint64_t, uint32_t *);
} Tools;
static void tools_load(Tools *t, const char *path) {
    void *lib = lib_open(path);
    t->path = path;
    BIND(lib, path, t->abi, "wex4_tools_abi");
    BIND(lib, path, t->sha, "wex4_sha256_bytes");
    BIND(lib, path, t->bp_encode, "wex4_bp_encode_record");
    BIND(lib, path, t->bp_decode, "wex4_bp_decode_record");
    BIND(lib, path, t->check, "wex4_check");
    BIND(lib, path, t->fuzz_check, "wex4_fuzz_check");
    BIND(lib, path, t->validate, "wex4_validate");
    BIND(lib, path, t->decode_buf, "wex4_decode_record_buf");
    BIND(lib, path, t->expert_table, "wex4_expert_table");
    BIND(lib, path, t->fuzz_run, "wex4_fuzz_run");
    BIND(lib, path, t->store_probe, "wex4_store_probe");
    CHECK(t->abi() == 1u, "%s: wex4_tools ABI", path);
}

typedef struct {
    uint32_t (*abi)(void);
    int (*prepare)(const WexEntry *, const unsigned char *, int, unsigned char *, uint32_t *);
    int (*count)(const WexEntry *, const unsigned char *, uint64_t *, uint64_t *);
    int (*build_model)(const uint64_t *, const uint64_t *, unsigned char *, uint32_t *);
    int (*model_id)(const unsigned char *, uint32_t *);
    int (*encode)(const unsigned char *, const WexEntry *, const unsigned char *, uint32_t, uint32_t, uint32_t, uint32_t,
                  unsigned char *, size_t, size_t *, size_t *);
    int (*decode)(const unsigned char *, const WexEntry *, const unsigned char *, uint32_t, uint32_t, unsigned char *);
    int (*verify)(const unsigned char *, const WexEntry *, const unsigned char *, uint32_t, uint32_t,
                  const unsigned char *);
    uint64_t (*canonicalize)(const WexEntry *, unsigned char *);
    int (*check_container)(const unsigned char *, const unsigned char *, const WexEntry *, uint64_t);
} Nr3;
static void nr3_load(Nr3 *t, const char *path) {
    void *lib = lib_open(path);
    BIND(lib, path, t->abi, "wex_nr3_tools_abi");
    BIND(lib, path, t->prepare, "wex_nr3_prepare");
    BIND(lib, path, t->count, "wex_nr3_count_expert");
    BIND(lib, path, t->build_model, "wex_nr3_build_model");
    BIND(lib, path, t->model_id, "wex_nr3_model_id");
    BIND(lib, path, t->encode, "wex_nr3_encode_record");
    BIND(lib, path, t->decode, "wex_nr3_decode");
    BIND(lib, path, t->verify, "wex_nr3_verify");
    BIND(lib, path, t->canonicalize, "wex_nr3_canonicalize");
    BIND(lib, path, t->check_container, "wex_nr3_check_container");
    CHECK(t->abi() == 1u, "%s: wex_nr3_tools ABI", path);
}

/* ---- a thread pool: THREADS workers take the jobs 0 .. jobs-1 in order (concurrent.futures) -------------------- */
typedef struct {
    pthread_mutex_t lock;
    int next, jobs;
    void (*run)(void *context, int job);
    void *context;
} Pool;
static void *pool_worker(void *arg) {
    Pool *p = (Pool *)arg;
    for (;;) {
        int job;
        pthread_mutex_lock(&p->lock);
        job = p->next < p->jobs ? p->next++ : -1;
        pthread_mutex_unlock(&p->lock);
        if (job < 0) return NULL;
        p->run(p->context, job);
    }
}
static void pool_run(int threads, int jobs, void (*run)(void *, int), void *context) {
    Pool p;
    pthread_t id[64];
    int i;
    if (threads > 64) threads = 64;
    if (threads < 1) threads = 1;
    p.next = 0; p.jobs = jobs; p.run = run; p.context = context;
    CHECK(!pthread_mutex_init(&p.lock, NULL), "pthread_mutex_init");
    for (i = 0; i < threads; i++) CHECK(!pthread_create(&id[i], NULL, pool_worker, &p), "pthread_create");
    for (i = 0; i < threads; i++) pthread_join(id[i], NULL);
    pthread_mutex_destroy(&p.lock);
}

/* ---- inputs ---------------------------------------------------------------------------------------------------- */
static uint64_t rng = UINT64_C(20261111);
static uint32_t rnd(void) {   /* xorshift64, main thread only */
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static void hex_of(const unsigned char *p, size_t n, char *out) {
    static const char digits[] = "0123456789abcdef";
    size_t i;
    for (i = 0; i < n; i++) { out[2 * i] = digits[p[i] >> 4]; out[2 * i + 1] = digits[p[i] & 15u]; }
    out[2 * n] = 0;
}

typedef struct { char path[1024]; unsigned char *data; size_t size; } File;

/* ---- 1. SHA-256 ------------------------------------------------------------------------------------------------ */
static void part_sha(const Tools *fast, const Tools *ref) {
    static const char *const known[3][2] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"}};
    unsigned char *data = (unsigned char *)xmalloc(65537u), a[32], b[32];
    char text[65];
    size_t lengths[203], i;
    int k;
    for (i = 0; i < 200u; i++) lengths[i] = i;
    lengths[200] = 4064u; lengths[201] = 4096u; lengths[202] = 65537u;
    for (i = 0; i < 65537u; i++) data[i] = (unsigned char)rnd();
    for (i = 0; i < 203u; i++) {
        fast->sha(data, lengths[i], a);
        ref->sha(data, lengths[i], b);
        CHECK(!memcmp(a, b, 32u), "sha256 of %lu bytes: the two libraries differ", (unsigned long)lengths[i]);
    }
    for (k = 0; k < 3; k++) {
        fast->sha((const unsigned char *)known[k][0], strlen(known[k][0]), a);
        hex_of(a, 32u, text);
        CHECK(!strcmp(text, known[k][1]), "sha256 of \"%s\" is %s", known[k][0], text);
    }
    free(data);
    printf("sha256: 203 lengths, the same digest from both libraries; 3 known answers (FIPS 180-4 examples)\n");
}

/* ---- 2. WBP1 --------------------------------------------------------------------------------------------------- */
static void fill_skewed(unsigned char *p, size_t n) {   /* around 120, a geometric spread: compressible */
    size_t i;
    for (i = 0; i < n; i++) {
        uint32_t r = rnd(), k = 0;
        while (r & 1u && k < 20u) { r >>= 1; k++; }
        p[i] = (unsigned char)(r & 2u ? 120u + k : 120u - k);
    }
}
static void fill_bf16(unsigned char *p, size_t values, float sigma) {   /* bf16 of an approximate Gaussian */
    size_t i;
    for (i = 0; i < values; i++) {
        float x = 0.0f, f;
        uint32_t bits;
        int k;
        for (k = 0; k < 4; k++) x += (float)rnd() / 4294967296.0f - 0.5f;
        f = x * sigma * 1.7320508f;
        memcpy(&bits, &f, 4u);
        p[2u * i] = (unsigned char)(bits >> 16); p[2u * i + 1u] = (unsigned char)(bits >> 24);
    }
}
static void fill_f32(unsigned char *p, size_t values) {
    size_t i;
    for (i = 0; i < values; i++) {
        float f = ((float)rnd() / 4294967296.0f - 0.5f) * 4.0f;
        memcpy(p + 4u * i, &f, 4u);
    }
}
static void part_bp(const Tools *fast, const Tools *ref) {
    enum { CASES = 17 };
    static const size_t sizes[CASES] = {30000, 18000, 20000, 5000, 70000, 1, 7, 80000, 8000, 2, 3, 4, 5, 6, 7, 8, 9};
    static const uint32_t planes[4] = {1, 2, 4, 8}, chunks[3] = {10, 12, 15};
    int c, count = 0;
    for (c = 0; c < CASES; c++) {
        const size_t n = sizes[c], cap = (size_t)wex4_align((uint64_t)n * 2u + 8192u);
        unsigned char *data = (unsigned char *)xmalloc(n), *a = (unsigned char *)xmalloc(cap);
        unsigned char *b = (unsigned char *)xmalloc(cap), *out = (unsigned char *)xmalloc(n);
        int p, k;
        uint32_t t;
        if (c == 0 || c == 5 || c == 6 || c >= 9) fill_skewed(data, n);
        else if (c == 1) fill_bf16(data, n / 2u, 0.05f);
        else if (c == 2) { size_t i; for (i = 0; i < n; i++) data[i] = (unsigned char)rnd(); }
        else if (c == 3) memset(data, 0, n);
        else if (c == 4) memset(data, 0x7f, n);
        else if (c == 7) fill_bf16(data, n / 2u, 3.0f);
        else fill_f32(data, n / 4u);
        for (p = 0; p < 4; p++)
            for (k = 0; k < 3; k++)
                for (t = 0; t < (planes[p] > 1u && n % planes[p] == 0 ? 2u : 1u); t++) {
                    size_t wa = 0, wb = 0;
                    CHECK(!fast->bp_encode(data, (uint32_t)n, planes[p], chunks[k], t, 7u, a, cap, &wa) &&
                          !ref->bp_encode(data, (uint32_t)n, planes[p], chunks[k], t, 7u, b, cap, &wb),
                          "WBP1 encode of case %d (planes %u, chunk 2^%u, transform %u)", c, planes[p], chunks[k], t);
                    CHECK(wa == wb && !memcmp(a, b, wa), "WBP1 case %d: the two libraries wrote different records", c);
                    CHECK(!fast->bp_decode(a, wa, (uint32_t)n, 7u, out) && !memcmp(out, data, n),
                          "WBP1 case %d: the fast library", c);
                    CHECK(!ref->bp_decode(a, wa, (uint32_t)n, 7u, out) && !memcmp(out, data, n),
                          "WBP1 case %d: the reference library", c);
                    CHECK(fast->bp_decode(a, wa, (uint32_t)n, 8u, out) && ref->bp_decode(a, wa, (uint32_t)n, 8u, out),
                          "WBP1 case %d: a record of owner 7 accepted for owner 8", c);
                    count++;
                }
        free(data); free(a); free(b); free(out);
    }
    printf("WBP1: %d records, the same bytes from both libraries, decoded exactly by both, refused for another owner\n",
           count);
}

/* ---- 3. the conformance vectors -------------------------------------------------------------------------------- */
static const char *class_name(int status) {
    static const char *const names[16] = {"-", "WEX4_E_ARG", "WEX4_E_MAGIC", "WEX4_E_HASH", "WEX4_E_VERSION",
        "WEX4_E_FEATURE", "WEX4_E_HEADER", "WEX4_E_INDEX", "WEX4_E_BLOB", "WEX4_E_RECORD", "WEX4_E_TENSOR",
        "WEX4_E_NAME", "WEX4_E_LAYOUT", "WEX4_E_EXPERT", "WEX4_E_DATA", "WEX4_E_UNSUPPORTED"};
    return status <= 0 && status >= -15 ? names[-status] : "?";
}
/* wex4_check and wex4_fuzz_check of both libraries: one status (and one digest when 0) */
static int four_checks(const Tools *fast, const Tools *ref, const unsigned char *data, size_t size, const char *what) {
    unsigned char d[4][32];
    int s[4], i;
    s[0] = fast->check(data, size, 3, d[0]);
    s[1] = fast->fuzz_check(data, size, 3, d[1]);
    s[2] = ref->check(data, size, 3, d[2]);
    s[3] = ref->fuzz_check(data, size, 3, d[3]);
    for (i = 1; i < 4; i++) {
        CHECK(s[i] == s[0], "%s: status %d %d %d %d (fast, fast guarded, reference, reference guarded)", what,
              s[0], s[1], s[2], s[3]);
        CHECK(s[0] || !memcmp(d[i], d[0], 32u), "%s: the content digests differ", what);
    }
    return s[0];
}
static int part_vectors(const Tools *fast, const Tools *ref, const char *dir, File *valid, int max_valid, int *rows) {
    char path[1024], line[1024];
    FILE *f;
    int accepted = 0;
    snprintf(path, sizeof path, "%s/MANIFEST.tsv", dir);
    CHECK((f = fopen(path, "rb")) != NULL, "%s: cannot be opened", path);
    *rows = 0;
    while (fgets(line, sizeof line, f)) {
        char *field[6], *at = line, sha_hex[65];
        unsigned char digest[32];
        File file;
        int n = 0, status;
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0] || line[0] == '#') continue;
        while (n < 6) { field[n++] = at; at = strchr(at, '\t'); if (!at) break; *at++ = 0; }
        CHECK(n == 6, "%s: a row without 6 fields", path);
        snprintf(file.path, sizeof file.path, "%s/%s", dir, field[0]);
        file.data = read_file(file.path, &file.size);
        fast->sha(file.data, file.size, digest);
        hex_of(digest, 32u, sha_hex);
        CHECK(!strcmp(sha_hex, field[4]), "%s: SHA-256 %s, the manifest says %s", field[0], sha_hex, field[4]);
        status = four_checks(fast, ref, file.data, file.size, field[0]);
        CHECK(!strcmp(class_name(status), field[2]) && (status == 0) == !strcmp(field[1], "accept"),
              "%s: status %d (%s), the manifest says %s %s", field[0], status, class_name(status), field[1], field[2]);
        if (!status) {
            char root[65];
            fast->sha(file.data, WEX4_HEADER_HASHED, digest);
            hex_of(file.data + WEX4_HEADER_HASHED, 32u, root);
            CHECK(!memcmp(digest, file.data + WEX4_HEADER_HASHED, 32u) && !strcmp(root, field[3]),
                  "%s: root %s, the manifest says %s", field[0], root, field[3]);
            CHECK(accepted < max_valid, "more than %d accepted vectors", max_valid);
            valid[accepted++] = file;
        } else free(file.data);
        (*rows)++;
    }
    fclose(f);
    return accepted;
}

/* ---- 4. the reader entry points -------------------------------------------------------------------------------- */
typedef struct { Wex4Header h; Wex4View v; const unsigned char *index; size_t index_len; } Opened;
static void open_file(const File *file, Opened *o) {
    CHECK(file->size >= WEX4_PAGE && !wex4_parse_header(file->data, file->size, &o->h), "%s: header", file->path);
    o->index = file->data + o->h.index_offset;
    o->index_len = (size_t)wex4_align(o->h.index_bytes);
    CHECK(!wex4_open(&o->h, o->index, o->index_len, &o->v), "%s: index", file->path);
}
static const unsigned char *blob_of(const File *file, const Opened *o, const Wex4Record *rec) {
    Wex4Blob b;
    if (!rec->blob) return NULL;
    wex4_get_blob(&o->v, rec->blob - 1u, &b);
    return file->data + b.offset;
}
/* both libraries: 0 and the same table; returns the table (malloc) or NULL without an expert grid */
static unsigned char *expert_tables(const Tools *fast, const Tools *ref, const File *file, const Opened *o) {
    const size_t bytes = (size_t)o->h.layers * o->h.experts * sizeof(WexEntry);
    unsigned char *a, *b;
    if (!o->h.layers) return NULL;
    a = (unsigned char *)xmalloc(bytes); b = (unsigned char *)xmalloc(bytes);
    CHECK(!fast->expert_table(file->data, file->size, o->index, o->index_len, a) &&
          !ref->expert_table(file->data, file->size, o->index, o->index_len, b) && !memcmp(a, b, bytes),
          "%s: wex4_expert_table", file->path);
    free(b);
    return a;
}
static int has_nv_expert(const Opened *o) {
    uint32_t r;
    for (r = 0; r < o->h.records; r++) {
        Wex4Record rec;
        Wex4ExpertGeo g;
        wex4_get_record(&o->v, r, &rec);
        if ((rec.flags & WEX4_REC_EXPERT) && !wex4_expert_geometry(&o->v, r, &g) && g.kind != 0u) return 1;
    }
    return 0;
}
static int part_reader(const Tools *fast, const Tools *ref, const File *file) {
    Opened o;
    unsigned char *a, *b, *table = NULL;
    uint32_t r;
    open_file(file, &o);
    CHECK(!fast->validate(file->data, file->size, o.index, o.index_len) &&
          !ref->validate(file->data, file->size, o.index, o.index_len), "%s: wex4_validate", file->path);
    a = (unsigned char *)xmalloc(o.h.max_decoded); b = (unsigned char *)xmalloc(o.h.max_decoded);
    for (r = 0; r < o.h.records; r++) {
        Wex4Record rec;
        const unsigned char *blob;
        int sa, sb;
        wex4_get_record(&o.v, r, &rec);
        blob = blob_of(file, &o, &rec);
        sa = fast->decode_buf(file->data, file->size, o.index, o.index_len, r, file->data + rec.offset, blob, 1, a);
        sb = ref->decode_buf(file->data, file->size, o.index, o.index_len, r, file->data + rec.offset, blob, 1, b);
        CHECK(!sa && !sb && !memcmp(a, b, rec.decoded), "%s record %u: status %d and %d, or other bytes", file->path,
              r, sa, sb);
    }
    if (!has_nv_expert(&o)) table = expert_tables(fast, ref, file, &o);
    free(table); free(a); free(b);
    return (int)o.h.records;
}

/* ---- 5. the engine loader -------------------------------------------------------------------------------------- */
static int count_experts(const Opened *o) {
    uint32_t r;
    int n = 0;
    for (r = 0; r < o->h.records; r++) {
        Wex4Record rec;
        wex4_get_record(&o->v, r, &rec);
        n += (rec.flags & WEX4_REC_EXPERT) != 0;
    }
    return n;
}
/* 1 when both libraries' loader refuses a valid file with an expert grid, as wex4/wex4_store.h refuses a file whose
 * WNR3 records mix canonical zero and plain ones (WEX4_E_RECORD: the old store applies canonical zero to the whole
 * container); any other refusal, or two different answers, fails */
static int store_refused(const Tools *fast, const Tools *ref, const File *file, const Opened *o) {
    const uint64_t cap = (uint64_t)o->h.layers * o->h.experts * sizeof(WexEntry);
    unsigned char *table = (unsigned char *)xmalloc((size_t)cap);
    uint32_t info[5];
    int sa = fast->store_probe(file->path, file->data + WEX4_HEADER_HASHED, table, cap, info);
    int sb = ref->store_probe(file->path, file->data + WEX4_HEADER_HASHED, table, cap, info);
    CHECK(sa == sb && (sa == 0 || sa == WEX4_E_RECORD), "%s: wex4_store_probe gives %d and %d", file->path, sa, sb);
    free(table);
    return sa != 0;
}
/* flip: the byte to flip for the test of the first-read SHA-256, or 0; legacy: the expected legacy flags or -1 */
static void part_store(const Tools *lib, const File *file, size_t flip, long legacy) {
    Opened o;
    unsigned char *table, *expect, zero[32];
    uint32_t info[5];
    uint64_t cap;
    int present;
    open_file(file, &o);
    cap = (uint64_t)o.h.layers * o.h.experts * sizeof(WexEntry);
    expect = expert_tables(lib, lib, file, &o);
    present = count_experts(&o);
    table = (unsigned char *)xmalloc((size_t)cap);
    CHECK(!lib->store_probe(file->path, file->data + WEX4_HEADER_HASHED, table, cap, info), "%s: wex4_store_probe",
          file->path);
    CHECK(!memcmp(table, expect, (size_t)cap) && info[4] == (uint32_t)present && info[2] % 4096u == 0,
          "%s: table, %u records checked of %d, slab %u", file->path, info[4], present, info[2]);
    CHECK(legacy < 0 || (info[0] == (uint32_t)legacy && info[1] != 0), "%s: legacy flags %u, model %08x", file->path,
          info[0], info[1]);
    memset(zero, 0, sizeof zero);
    CHECK(lib->store_probe(file->path, zero, table, cap, info) == WEX4_E_HASH, "%s: another root pinned", file->path);
    CHECK(lib->store_probe(file->path, file->data + WEX4_HEADER_HASHED, table, cap - sizeof(WexEntry), info) ==
          WEX4_E_EXPERT, "%s: a table over budget", file->path);
    if (flip) {
        char path[1100];
        FILE *f;
        snprintf(path, sizeof path, "%s.flip", file->path);
        CHECK((f = fopen(path, "wb")) != NULL, "%s: cannot be written", path);
        file->data[flip] ^= 1u;
        CHECK(fwrite(file->data, 1, file->size, f) == file->size && !fclose(f), "%s: cannot be written", path);
        file->data[flip] ^= 1u;
        CHECK(lib->store_probe(path, file->data + WEX4_HEADER_HASHED, table, cap, info) == WEX4_E_HASH &&
              info[4] < (uint32_t)present, "%s: a record with a flipped bit was served", path);
        remove(path);
    }
    free(table); free(expect);
}

/* ---- 6. the mutation fuzzer of wex4_tools.c, all calls at once ------------------------------------------------- */
typedef struct {
    const Tools *lib[2];
    const File *files[3];
    uint32_t n[3][3];
    uint64_t counts[3][3][2][4];
    int status[3][3][2];
} Fuzz;
static void fuzz_job(void *context, int job) {
    Fuzz *z = (Fuzz *)context;
    const int file = job / 6, mode = job / 2 % 3, lib = job % 2;
    const File *f = z->files[file];
    /* the seed of test_fuzz: 1000 + mode */
    z->status[file][mode][lib] = z->lib[lib]->fuzz_run(f->data, f->size, 1000u + (uint64_t)mode, z->n[file][mode], mode,
                                                       z->counts[file][mode][lib]);
}
static void part_fuzz(const Tools *fast, const Tools *ref, const File *files[3], int threads, uint32_t mutations) {
    Fuzz *z = (Fuzz *)xcalloc(1, sizeof(Fuzz));
    uint64_t total = 0;
    int f, m;
    z->lib[0] = fast; z->lib[1] = ref;
    for (f = 0; f < 3; f++) {
        z->files[f] = files[f];
        for (m = 0; m < 3; m++) {   /* test_fuzz: the same time per file; mode 1 decodes, so it gets a fifth */
            uint64_t n = (uint64_t)mutations * (m == 1 ? 1u : 5u) / 5u * 150000u /
                         (files[f]->size > 150000u ? files[f]->size : 150000u);
            z->n[f][m] = n < 20u ? 20u : (uint32_t)n;
        }
    }
    pool_run(threads, 18, fuzz_job, z);
    for (f = 0; f < 3; f++)
        for (m = 0; m < 3; m++) {
            const uint64_t *c = z->counts[f][m][0];
            CHECK(!z->status[f][m][0] && !z->status[f][m][1], "%s mode %d: wex4_fuzz_run failed", files[f]->path, m);
            CHECK(!memcmp(c, z->counts[f][m][1], 4u * sizeof(uint64_t)),
                  "%s mode %d: the fast and the reference decoder disagree", files[f]->path, m);
            CHECK(m == 2 || !c[2], "%s mode %d: %llu corrupted files accepted with another content", files[f]->path,
                  m, (unsigned long long)c[2]);
            CHECK(m != 0 || !c[1], "%s: %llu corrupted files accepted", files[f]->path, (unsigned long long)c[1]);
            CHECK(c[0] + c[1] + c[2] + c[3] == z->n[f][m], "%s mode %d: counts", files[f]->path, m);
            total += z->n[f][m];
        }
    printf("wex4_fuzz_run: 3 modes on 3 files, %llu mutations through each library, 18 calls on %d threads at once; "
           "the same counts from the fast and the reference decoder, no corrupted file accepted\n",
           (unsigned long long)total, threads);
    free(z);
}

/* ---- 7. the WNR3 encoder --------------------------------------------------------------------------------------- */
#define WC_COUNT (3u * WEX_NR2_CONTEXTS * 16u)
#define SC_COUNT (3u * 256u)
typedef struct {
    const Nr3 *nr3;
    const unsigned char *source, *blob;
    const WexEntry *entries;   /* the experts of WEX1, in table order */
    const uint32_t *owners;
    int cz;
    uint64_t *wc[1024], *sc[1024];
    unsigned char *record[1024];
    size_t total[1024];
    int rc[1024];
    uint64_t mapped[1024];
    const char *failed[1024];
} Encode;
static WexEntry raw_entry(const WexEntry *e, uint32_t total, uint32_t flags) {   /* build_nr3.py entry_bytes */
    WexEntry x = *e;
    x.file_offset = 0; x.total_bytes = total; x.flags = flags;
    return x;
}
static void train_job(void *context, int j) {
    Encode *w = (Encode *)context;
    const WexEntry *e = &w->entries[j];
    const uint32_t raw = e->scale_bytes + e->weight_bytes;
    const WexEntry eb = raw_entry(e, (uint32_t)wex4_align(raw), 0);
    unsigned char *coded = (unsigned char *)xmalloc(raw);
    w->wc[j] = (uint64_t *)xcalloc(WC_COUNT, sizeof(uint64_t));
    w->sc[j] = (uint64_t *)xcalloc(SC_COUNT, sizeof(uint64_t));
    if (w->nr3->prepare(&eb, w->source + e->file_offset, w->cz, coded, NULL) < 0) w->failed[j] = "wex_nr3_prepare";
    else if (w->nr3->count(&eb, coded, w->wc[j], w->sc[j])) w->failed[j] = "wex_nr3_count_expert";
    free(coded);
}
static void encode_job(void *context, int j) {   /* build_nr3.py encode_one */
    Encode *w = (Encode *)context;
    const WexEntry *e = &w->entries[j];
    const uint32_t raw = e->scale_bytes + e->weight_bytes, layer = w->owners[j] >> 16, expert = w->owners[j] & 0xffffu;
    const WexEntry eb = raw_entry(e, (uint32_t)wex4_align(raw), 0);
    const size_t cap = 4u * (size_t)raw + ((size_t)1 << 20);
    unsigned char *coded = (unsigned char *)xmalloc(raw), *expected = (unsigned char *)xmalloc(raw);
    unsigned char *out = (unsigned char *)xmalloc(cap), *back = (unsigned char *)xmalloc(raw);
    uint32_t stats[16];
    size_t used = 0, total = 0;
    int flags = w->nr3->prepare(&eb, w->source + e->file_offset, w->cz, coded, stats);
    memcpy(expected, w->source + e->file_offset, raw);
    if (w->cz) w->mapped[j] = w->nr3->canonicalize(&eb, expected);
    if (flags < 0 || (uint32_t)flags != stats[12]) w->failed[j] = "wex_nr3_prepare";
    else if ((w->rc[j] = w->nr3->encode(w->blob, &eb, coded, (uint32_t)flags, w->owners[j], 2048u, 512u, out, cap,
                                        &used, &total)) == 0) {
        const WexEntry vb = raw_entry(e, (uint32_t)total, WEX_ENTRY_NEURAL_RANS3);
        if (w->nr3->verify(w->blob, &vb, out, layer, expert, expected)) w->failed[j] = "wex_nr3_verify";
        else if (w->nr3->decode(w->blob, &vb, out, layer, expert, back) || memcmp(back, expected, raw))
            w->failed[j] = "wex_nr3_decode";
        else if (!w->nr3->verify(w->blob, &vb, out, layer, expert + 1u, expected))
            w->failed[j] = "wex_nr3_verify accepted the record for another expert";
        w->record[j] = out; w->total[j] = total;
        out = NULL;
    } else if (w->rc[j] != 1) w->failed[j] = "wex_nr3_encode_record";
    free(coded); free(expected); free(out); free(back);
}
/* ---- the control: a library with a deliberate defect, loaded and called the same way --------------------------- */
static void (*control_race)(void);
static void control_job(void *context, int job) { (void)context; (void)job; control_race(); }
static int control(const char *path) {   /* the run must end with the sanitizer's report: else nothing above counts */
    void *lib = lib_open(path);
    int (*uninit)(void);
    BIND(lib, path, control_race, "race");
    BIND(lib, path, uninit, "uninit");
    pool_run(2, 2, control_job, NULL);   /* two threads in race() at once */
    printf("control: race() on 2 threads, uninit() returned %d\n", uninit());
    return 0;
}

typedef struct { unsigned char *data; size_t size; WexHeader h; const WexEntry *table; } Legacy;
static void open_legacy(Legacy *l, const char *path, uint32_t magic, uint32_t version) {
    size_t bytes;
    l->data = read_file(path, &l->size);
    CHECK(l->size >= WEX_ALIGN, "%s: too short", path);
    memcpy(&l->h, l->data, sizeof l->h);
    CHECK(l->h.magic == magic && l->h.version == version &&
          !wex_validate_header(&l->h, l->size, 0, 0, &bytes, NULL), "%s: not a valid WEX%u container", path, version);
    l->table = (const WexEntry *)(const void *)(l->data + l->h.table_offset);   /* packed: any address will do */
}

int main(int argc, char **argv) {
    Tools fast, ref;
    Nr3 nr3;
    Legacy wex1, wnr3;
    File valid[128], wex4;
    const File *corpus[3] = {NULL, NULL, NULL};
    Opened o4;
    Encode *w;
    WexEntry entries[1024];
    unsigned char *kept[1024];   /* the records encoded without canonical zero, for part 8 */
    size_t kept_total[1024];
    uint32_t owners[1024], id = 0, id2 = 0, model_bytes = 0, model_id = 0, i;
    uint64_t *wc, *sc, model_at = 0;
    unsigned char *blob;
    const int threads = argc > 8 ? atoi(argv[8]) : 4;
    const uint32_t mutations = argc > 9 ? (uint32_t)strtoul(argv[9], NULL, 10) : 200u;
    int nvalid, rows, records = 0, experts = 0, coded = 0, stores = 0, store_refusals = 0, round_trip = 0, k, cz;
    uint64_t mapped = 0;
    setvbuf(stdout, NULL, _IONBF, 0);   /* each line when it is written, in order with the sanitizer's reports */
    if (argc == 3 && !strcmp(argv[1], "--control")) return control(argv[2]);
    if (argc < 8) {
        fprintf(stderr, "usage: sanitize_libs FAST REF NR3 WEX1 WNR3 WEX4 VECTORS [THREADS [MUTATIONS]]\n"
                        "       sanitize_libs --control LIB\n");
        return 2;
    }
    tools_load(&fast, argv[1]);
    tools_load(&ref, argv[2]);
    nr3_load(&nr3, argv[3]);
    part_sha(&fast, &ref);
    part_bp(&fast, &ref);

    nvalid = part_vectors(&fast, &ref, argv[7], valid, 128, &rows);
    snprintf(wex4.path, sizeof wex4.path, "%s", argv[6]);
    wex4.data = read_file(wex4.path, &wex4.size);
    CHECK(four_checks(&fast, &ref, wex4.data, wex4.size, wex4.path) == 0, "%s: refused", wex4.path);
    printf("files: %d rows of MANIFEST.tsv with their verdict, error class, SHA-256 and root, and the WEX4 file, "
           "through wex4_check and wex4_fuzz_check of both libraries\n", rows);

    for (k = 0; k < nvalid; k++) records += part_reader(&fast, &ref, &valid[k]);
    records += part_reader(&fast, &ref, &wex4);
    open_file(&wex4, &o4);
    open_legacy(&wnr3, argv[5], WEX_MAGIC_V3, 3u);
    {   /* test_upgrade: the legacy table rebuilt from the index is the table of the file before its upgrade */
        unsigned char *table = expert_tables(&fast, &ref, &wex4, &o4);
        CHECK(table && o4.h.layers == wnr3.h.num_layers && o4.h.experts == wnr3.h.num_experts &&
              !memcmp(table, wnr3.table, (size_t)o4.h.layers * o4.h.experts * sizeof(WexEntry)),
              "%s: the legacy table differs from the table of %s", wex4.path, argv[5]);
        free(table);
    }
    printf("reader entry points: wex4_validate and wex4_decode_record_buf on %d records of %d files, the same "
           "bytes from both libraries; wex4_expert_table gives the table of the file before its upgrade\n",
           records, nvalid + 1);

    for (k = 0; k < nvalid; k++) {
        Opened o;
        open_file(&valid[k], &o);
        if (!o.h.layers) continue;
        if (store_refused(&fast, &ref, &valid[k], &o)) { store_refusals++; continue; }
        part_store(&fast, &valid[k], 0, -1);
        part_store(&ref, &valid[k], 0, -1);
        stores++;
    }
    part_store(&fast, &wex4, wnr3.size / 2u, (long)(wnr3.h.flags & ~WEX_HEADER_SCALE_TILES));
    part_store(&ref, &wex4, wnr3.size / 2u, (long)(wnr3.h.flags & ~WEX_HEADER_SCALE_TILES));
    printf("engine loader: wex4_store_probe on %d files of both libraries: their table, every record checked, a "
           "wrong root and a table over budget refused; a flipped bit in a record of the WEX4 file refused; %d valid "
           "file(s) with WNR3 records with and without canonical zero refused by both, as the loader must\n",
           stores + 1, store_refusals);

    for (k = 0; k < nvalid; k++) {
        if (strstr(valid[k].path, "/valid/generic.wex4")) corpus[0] = &valid[k];
        if (strstr(valid[k].path, "/valid/tiny2-cz-upgraded.wex4")) corpus[1] = &valid[k];
    }
    corpus[2] = &wex4;
    CHECK(corpus[0] && corpus[1], "valid/generic.wex4 or valid/tiny2-cz-upgraded.wex4 missing in the vectors");
    part_fuzz(&fast, &ref, corpus, threads, mutations);

    /* 7. the encoder, as build_nr3.py container mode */
    open_legacy(&wex1, argv[4], WEX_MAGIC_V1, 1u);
    CHECK(wex1.h.num_layers == wnr3.h.num_layers && wex1.h.num_experts == wnr3.h.num_experts, "WEX1 and WNR3 grids");
    for (i = 0; i < wex1.h.num_layers * wex1.h.num_experts; i++) {
        const WexEntry *e = &wex1.table[i];
        uint64_t raw, aligned;
        if (!e->total_bytes) continue;
        CHECK(experts < 1024, "more than 1024 experts");
        CHECK((e->flags & WEX_ENTRY_CODEC_MASK) == 0 && !wex_entry_raw_sizes(e, &raw, &aligned) &&
              e->file_offset <= wex1.size && raw <= wex1.size - e->file_offset, "WEX1 entry %u", i);
        entries[experts] = *e;
        owners[experts++] = i / wex1.h.num_experts << 16 | i % wex1.h.num_experts;
    }
    CHECK(experts > 0, "%s: no expert", argv[4]);
    w = (Encode *)xcalloc(1, sizeof(Encode));
    w->nr3 = &nr3; w->source = wex1.data; w->entries = entries; w->owners = owners; w->cz = 0;
    pool_run(threads, experts, train_job, w);
    wc = (uint64_t *)xcalloc(WC_COUNT, sizeof(uint64_t));
    sc = (uint64_t *)xcalloc(SC_COUNT, sizeof(uint64_t));
    for (k = 0; k < experts; k++) {
        CHECK(!w->failed[k], "training, expert %d: %s", k, w->failed[k]);
        for (i = 0; i < WC_COUNT; i++) wc[i] += w->wc[k][i];
        for (i = 0; i < SC_COUNT; i++) sc[i] += w->sc[k][i];
        free(w->wc[k]); free(w->sc[k]);
    }
    blob = (unsigned char *)xmalloc(WEX_NR2_MODEL_BYTES);
    CHECK(!nr3.build_model(wc, sc, blob, &id) && !nr3.model_id(blob, &id2) && id == id2, "wex_nr3_build_model");
    CHECK(!wex_nr2_model_location(&wnr3.h, wnr3.data, &model_at, &model_bytes, &model_id) &&
          model_bytes == WEX_NR2_MODEL_BYTES && model_at + model_bytes <= wnr3.size, "%s: no model", argv[5]);
    CHECK(model_id == id && !memcmp(blob, wnr3.data + model_at, WEX_NR2_MODEL_BYTES),
          "the model trained here (%08x) is not the model of %s (%08x)", id, argv[5], model_id);
    w->blob = blob;
    CHECK(!(wnr3.h.flags & WEX_HEADER_CANONICAL_ZERO), "%s: built with --cz", argv[5]);
    for (cz = 0; cz < 2; cz++) {   /* as the container, then with canonical zero (build_nr3.py --cz) */
        memset(w->rc, 0, sizeof w->rc);
        w->cz = cz;
        pool_run(threads, experts, encode_job, w);
        for (k = 0; k < experts; k++) {
            const WexEntry *t = &wnr3.table[(owners[k] >> 16) * wnr3.h.num_experts + (owners[k] & 0xffffu)];
            CHECK(!w->failed[k], "expert %08x%s: %s", owners[k], cz ? " (CZ)" : "", w->failed[k]);
            coded += !w->rc[k];
            mapped += w->mapped[k];
            if (cz) { free(w->record[k]); continue; }
            if (w->rc[k])   /* not smaller than raw: build_nr3.py stored the raw slab (codec 0) */
                CHECK((t->flags & WEX_ENTRY_CODEC_MASK) == 0, "expert %08x: %s holds a record", owners[k], argv[5]);
            else
                CHECK((t->flags & WEX_ENTRY_CODEC_MASK) == WEX_ENTRY_NEURAL_RANS3 && t->total_bytes == w->total[k] &&
                      t->file_offset <= wnr3.size && w->total[k] <= wnr3.size - t->file_offset &&
                      !memcmp(wnr3.data + t->file_offset, w->record[k], w->total[k]),
                      "expert %08x: the record encoded here is not the record of %s", owners[k], argv[5]);
            kept[k] = w->rc[k] ? NULL : w->record[k];
            kept_total[k] = w->total[k];
        }
        memset(w->record, 0, sizeof w->record);
    }
    CHECK(!nr3.check_container(wnr3.data, blob, wnr3.table, wnr3.size), "%s: wex_nr3_check_container", argv[5]);
    printf("encoder: model trained on %d experts and every expert encoded on %d threads, model and records "
           "identical to %s; %d WNR3 records with and without canonical zero (%llu -0 mapped), each verified and "
           "decoded exactly, refused for another expert; wex_nr3_check_container passes\n",
           experts, threads, argv[5], coded, (unsigned long long)mapped);

    /* 8. the records encoded here (without canonical zero) through the WEX4 decoders */
    for (k = 0; k < experts; k++) {
        uint32_t r;
        if (!kept[k]) continue;
        for (r = 0; r < o4.h.records; r++) {
            Wex4Record rec;
            unsigned char *a, *b;
            const uint32_t raw = entries[k].scale_bytes + entries[k].weight_bytes;
            wex4_get_record(&o4.v, r, &rec);
            if (!(rec.flags & WEX4_REC_EXPERT) || rec.owner != owners[k]) continue;
            CHECK(rec.codec == WEX4_CODEC_WNR3 && rec.stored == kept_total[k] && rec.decoded == raw,
                  "WEX4 record %u: codec %u, %u stored bytes", r, rec.codec, rec.stored);
            a = (unsigned char *)xmalloc(raw); b = (unsigned char *)xmalloc(raw);
            CHECK(!fast.decode_buf(wex4.data, wex4.size, o4.index, o4.index_len, r, kept[k], blob_of(&wex4, &o4, &rec),
                                   1, a) && !memcmp(a, wex1.data + entries[k].file_offset, raw),
                  "WEX4 record %u: the fast decoder", r);
            CHECK(!ref.decode_buf(wex4.data, wex4.size, o4.index, o4.index_len, r, kept[k], blob_of(&wex4, &o4, &rec),
                                  1, b) && !memcmp(b, a, raw),
                  "WEX4 record %u: the reference decoder", r);
            free(a); free(b);
            round_trip++;
        }
        free(kept[k]);
    }
    CHECK(round_trip > 0, "no WNR3 record to decode");
    printf("round trip: %d records encoded here, decoded by both libraries through wex4_decode_record_buf to the "
           "slabs of %s\n", round_trip, argv[4]);
    printf("sanitize_libs: all checks passed\n");
    return 0;
}
