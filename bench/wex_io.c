/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* wex_io -- one timed pass of expert-record reads from a WEX4 container through one Linux read path.
 *
 *   wex_io FILE [key=value ...]        (driven by bench/wex_io.py, which builds it, repeats it and sums up)
 *
 * The header page and the index are opened and validated with c/wex4.h, as the engine stores do. The
 * expert records are then read whole, each at its own 4096-aligned offset, in a seeded random order
 * without repeats (an expert miss is one such read). One call = one pass = one JSON line on stdout.
 *
 *   path=sync|uring   sync: pread, one record per call, on `threads` threads (what c/expert_store_wex.c
 *                     and c/qwen38_wex.h do today). uring: one thread and one io_uring, `depth` records
 *                     in flight, each cut into `split` page-aligned sub-reads that are in flight together
 *   direct=0|1        open the file with O_DIRECT (1) or read through the page cache (0)
 *   advise=none|random|sequential     posix_fadvise on the whole file before the pass (page cache only)
 *   ahead=K           sync, 1 thread, page cache: POSIX_FADV_WILLNEED on the record K reads ahead
 *   records=N seed=S  N records of the permutation S (0 = every expert record)
 *   evict=0|1         1 (default): POSIX_FADV_DONTNEED on the whole file before the pass
 *   warm=0|1          1: the same records are read once through the page cache before the timed pass
 *   verify=0|1        1: SHA-256 of every record read against the index (the timing then includes the
 *                     hash and is not a result; it shows that the path returns the right bytes)
 *
 * cached_before / cached_after: the share of the pass's pages that mincore finds in the page cache,
 * so a cold pass can be told from a warm one without root. A path this machine cannot run prints
 * {"skipped": "<why>"} and exits 0. io_uring is used through its three system calls, no liburing.
 * Build: gcc -O2 -pthread bench/wex_io.c -o wex_io */
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__linux__) || defined(__CYGWIN__)
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../c/wex4.h"

#define WIO_MAX_SPLIT 64u
#define WIO_MAX_SLOTS 256u

typedef struct { uint64_t offset; uint32_t stored; const unsigned char *sha; } WioRec;

typedef struct {
    int fd;                    /* the data descriptor of the pass */
    const WioRec *recs;
    const uint32_t *plan;      /* record numbers in read order */
    uint32_t n, next;
    int verify, ahead;
    double *lat;               /* seconds per record, by plan position */
} WioPass;

typedef struct { WioPass *pass; unsigned char *buf; uint64_t bytes; uint32_t bad; } WioWorker;

static double wio_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int wio_fail(const char *what) {
    fprintf(stderr, "wex_io: %s (%s)\n", what, errno ? strerror(errno) : "no errno");
    return 1;
}

static int wio_skip(const char *why) {
    printf("{\"skipped\": \"%s\"}\n", why);
    return 0;
}

/* n bytes at off into buf, both 4096-aligned; a short read is continued, end of file is an error */
static int wio_read(int fd, unsigned char *buf, uint32_t n, uint64_t off) {
    while (n) {
        ssize_t got = pread(fd, buf, n, (off_t)off);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return -1;
        buf += got; off += (uint64_t)got; n -= (uint32_t)got;
    }
    return 0;
}

static int wio_sha_ok(const WioRec *r, const unsigned char *buf) {
    unsigned char digest[32];
    wex4_sha256(buf, r->stored, digest);
    return !memcmp(digest, r->sha, 32u);
}

static void *wio_worker(void *arg) {
    WioWorker *w = (WioWorker *)arg;
    WioPass *p = w->pass;
    for (;;) {
        uint32_t i = __sync_fetch_and_add(&p->next, 1u);
        if (i >= p->n) break;
        const WioRec *r = &p->recs[p->plan[i]];
        double t0 = wio_now();
        if (p->ahead && i + (uint32_t)p->ahead < p->n) {
            const WioRec *a = &p->recs[p->plan[i + (uint32_t)p->ahead]];
            posix_fadvise(p->fd, (off_t)a->offset, (off_t)a->stored, POSIX_FADV_WILLNEED);
        }
        if (wio_read(p->fd, w->buf, r->stored, r->offset)) { w->bad++; continue; }
        p->lat[i] = wio_now() - t0;
        w->bytes += r->stored;
        if (p->verify && !wio_sha_ok(r, w->buf)) w->bad++;
    }
    return NULL;
}

/* pread on `threads` threads, each with its own buffer; the threads take the next record of the plan */
static int wio_sync(WioPass *p, unsigned char **bufs, unsigned threads, uint64_t *bytes, uint32_t *bad) {
    pthread_t tid[WIO_MAX_SLOTS];
    WioWorker w[WIO_MAX_SLOTS];
    unsigned t, started = 0;
    for (uint32_t k = 1; p->ahead && k < (uint32_t)p->ahead && k < p->n; k++) {
        const WioRec *a = &p->recs[p->plan[k]];
        posix_fadvise(p->fd, (off_t)a->offset, (off_t)a->stored, POSIX_FADV_WILLNEED);
    }
    for (t = 0; t < threads; t++) {
        w[t].pass = p; w[t].buf = bufs[t]; w[t].bytes = 0; w[t].bad = 0;
        if (t + 1u < threads) {
            if (pthread_create(&tid[t], NULL, wio_worker, &w[t])) break;
            started++;
        }
    }
    if (t == threads) wio_worker(&w[threads - 1u]);   /* the calling thread is the last worker */
    for (unsigned k = 0; k < started; k++) pthread_join(tid[k], NULL);
    if (t != threads) return -1;
    for (t = 0; t < threads; t++) { *bytes += w[t].bytes; *bad += w[t].bad; }
    return 0;
}

#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/syscall.h>
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif

typedef struct {
    int fd;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    unsigned to_submit;
} WioRing;

typedef struct {
    uint32_t plan_i, open, bad;
    double t0;
    uint32_t pos[WIO_MAX_SPLIT], end[WIO_MAX_SPLIT];
} WioSlot;

static int wio_ring_open(WioRing *r, unsigned entries) {
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    memset(r, 0, sizeof *r);
    r->fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (r->fd < 0) return -1;
    size_t sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_len = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    unsigned char *sq = (unsigned char *)mmap(NULL, sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd,
                                              IORING_OFF_SQ_RING);
    unsigned char *cq = (unsigned char *)mmap(NULL, cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd,
                                              IORING_OFF_CQ_RING);
    void *sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) return -1;
    r->sq_head = (unsigned *)(sq + p.sq_off.head); r->sq_tail = (unsigned *)(sq + p.sq_off.tail);
    r->sq_mask = (unsigned *)(sq + p.sq_off.ring_mask); r->sq_array = (unsigned *)(sq + p.sq_off.array);
    r->cq_head = (unsigned *)(cq + p.cq_off.head); r->cq_tail = (unsigned *)(cq + p.cq_off.tail);
    r->cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
    r->cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    r->sqes = (struct io_uring_sqe *)sqes;
    return 0;
}

/* queue one read; never more are queued than the ring has entries (depth * split) */
static void wio_ring_read(WioRing *r, int fd, void *buf, uint32_t len, uint64_t off, uint64_t tag) {
    unsigned tail = *r->sq_tail, at = tail & *r->sq_mask;
    struct io_uring_sqe *s = &r->sqes[at];
    memset(s, 0, sizeof *s);
    s->opcode = IORING_OP_READ; s->fd = fd; s->addr = (uint64_t)(uintptr_t)buf; s->len = len; s->off = off;
    s->user_data = tag;
    r->sq_array[at] = at;
    __atomic_store_n(r->sq_tail, tail + 1u, __ATOMIC_RELEASE);
    r->to_submit++;
}

/* submit what is queued and wait for at least one completion */
static int wio_ring_enter(WioRing *r) {
    for (;;) {
        long n = syscall(__NR_io_uring_enter, r->fd, r->to_submit, 1u, IORING_ENTER_GETEVENTS, NULL, 0);
        if (n >= 0) { r->to_submit -= (unsigned)n; return 0; }
        if (errno != EINTR) return -1;
    }
}

/* one thread, `depth` records in flight, each as `split` sub-reads; a short sub-read is continued */
static int wio_uring(WioPass *p, unsigned char **bufs, unsigned depth, unsigned split, uint64_t *bytes, uint32_t *bad) {
    WioRing ring;
    WioSlot *slots = (WioSlot *)calloc(depth, sizeof *slots);
    unsigned free_slots[WIO_MAX_SLOTS], nfree = depth;
    uint32_t done = 0;
    if (!slots) return -1;
    if (wio_ring_open(&ring, depth * split)) { free(slots); return -2; }
    for (unsigned k = 0; k < depth; k++) free_slots[k] = depth - 1u - k;
    while (done < p->n) {
        while (nfree && p->next < p->n) {
            unsigned s = free_slots[--nfree];
            WioSlot *slot = &slots[s];
            const WioRec *r = &p->recs[p->plan[p->next]];
            uint32_t pages = r->stored / WEX4_PAGE, parts = pages < split ? pages : split;
            slot->plan_i = p->next++; slot->open = parts; slot->bad = 0; slot->t0 = wio_now();
            for (uint32_t k = 0; k < parts; k++) {
                slot->pos[k] = (uint32_t)((uint64_t)pages * k / parts) * WEX4_PAGE;
                slot->end[k] = (uint32_t)((uint64_t)pages * (k + 1u) / parts) * WEX4_PAGE;
                wio_ring_read(&ring, p->fd, bufs[s] + slot->pos[k], slot->end[k] - slot->pos[k], r->offset + slot->pos[k],
                              (uint64_t)s * WIO_MAX_SPLIT + k);
            }
        }
        if (wio_ring_enter(&ring)) { free(slots); return -1; }
        unsigned head = *ring.cq_head;
        while (head != __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE)) {
            const struct io_uring_cqe *c = &ring.cqes[head & *ring.cq_mask];
            unsigned s = (unsigned)(c->user_data / WIO_MAX_SPLIT), k = (unsigned)(c->user_data % WIO_MAX_SPLIT);
            WioSlot *slot = &slots[s];
            const WioRec *r = &p->recs[p->plan[slot->plan_i]];
            head++;
            if (c->res <= 0) { errno = -c->res; slot->bad = 1; slot->open--; }
            else if ((slot->pos[k] += (uint32_t)c->res) < slot->end[k])
                wio_ring_read(&ring, p->fd, bufs[s] + slot->pos[k], slot->end[k] - slot->pos[k], r->offset + slot->pos[k],
                              c->user_data);
            else slot->open--;
            if (slot->open) continue;
            p->lat[slot->plan_i] = wio_now() - slot->t0;
            if (slot->bad || (p->verify && !wio_sha_ok(r, bufs[s]))) (*bad)++;
            else *bytes += r->stored;
            free_slots[nfree++] = s;
            done++;
        }
        __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
    }
    free(slots);
    close(ring.fd);
    return 0;
}

/* share of the pass's pages that are in the page cache, or -1 */
static double wio_cached(int fd, uint64_t file_bytes, const WioPass *p) {
    uint64_t pages = file_bytes / WEX4_PAGE, in = 0, all = 0;
    unsigned char *vec = (unsigned char *)malloc((size_t)pages + 1u);
    void *map = mmap(NULL, (size_t)file_bytes, PROT_READ, MAP_SHARED, fd, 0);
    double share = -1.0;
    if (vec && map != MAP_FAILED && !mincore(map, (size_t)file_bytes, vec)) {
        for (uint32_t i = 0; i < p->n; i++) {
            const WioRec *r = &p->recs[p->plan[i]];
            for (uint64_t k = r->offset / WEX4_PAGE; k < (r->offset + r->stored) / WEX4_PAGE; k++) in += vec[k] & 1u;
            all += r->stored / WEX4_PAGE;
        }
        share = all ? (double)in / (double)all : 0.0;
    }
    if (map != MAP_FAILED) munmap(map, (size_t)file_bytes);
    free(vec);
    return share;
}
#else
static int wio_uring(WioPass *p, unsigned char **bufs, unsigned depth, unsigned split, uint64_t *bytes, uint32_t *bad) {
    (void)p; (void)bufs; (void)depth; (void)split; (void)bytes; (void)bad;
    return -2;
}
static double wio_cached(int fd, uint64_t file_bytes, const WioPass *p) {
    (void)fd; (void)file_bytes; (void)p;
    return -1.0;
}
#endif

static int wio_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static const char *wio_opt(int argc, char **argv, const char *key, const char *fallback) {
    size_t len = strlen(key);
    for (int i = 2; i < argc; i++)
        if (!strncmp(argv[i], key, len) && argv[i][len] == '=') return argv[i] + len + 1;
    return fallback;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s FILE [path=sync|uring] [direct=0|1] [threads=N] [depth=N] [split=N]\n"
                        "       [advise=none|random|sequential] [ahead=K] [records=N] [seed=S] [evict=0|1] [warm=0|1] [verify=0|1]\n",
                argv[0]);
        return 2;
    }
    const char *path = wio_opt(argc, argv, "path", "sync"), *advise = wio_opt(argc, argv, "advise", "none");
    int uring = !strcmp(path, "uring"), direct = atoi(wio_opt(argc, argv, "direct", "0"));
    unsigned threads = (unsigned)atoi(wio_opt(argc, argv, "threads", "1"));
    unsigned depth = (unsigned)atoi(wio_opt(argc, argv, "depth", "1")), split = (unsigned)atoi(wio_opt(argc, argv, "split", "1"));
    int ahead = atoi(wio_opt(argc, argv, "ahead", "0")), evict = atoi(wio_opt(argc, argv, "evict", "1"));
    int warm = atoi(wio_opt(argc, argv, "warm", "0")), verify = atoi(wio_opt(argc, argv, "verify", "0"));
    uint32_t want = (uint32_t)strtoul(wio_opt(argc, argv, "records", "0"), NULL, 10);
    uint64_t seed = strtoull(wio_opt(argc, argv, "seed", "1"), NULL, 10);
    if ((strcmp(path, "sync") && !uring) || threads < 1 || threads > WIO_MAX_SLOTS || depth < 1 || depth > WIO_MAX_SLOTS ||
        split < 1 || split > WIO_MAX_SPLIT || ahead < 0 || (ahead && (uring || threads != 1 || direct)) ||
        (strcmp(advise, "none") && strcmp(advise, "random") && strcmp(advise, "sequential"))) {
        fprintf(stderr, "wex_io: bad option\n");
        return 2;
    }

    /* header page and index, as the stores open them */
    int cfd = open(argv[1], O_RDONLY);
    struct stat st;
    Wex4Header h;
    Wex4View v;
    void *page = NULL, *index = NULL;
    if (cfd < 0 || fstat(cfd, &st)) return wio_fail("cannot open the container");
    uint64_t file_bytes = (uint64_t)st.st_size;
    if (posix_memalign(&page, WEX4_PAGE, WEX4_PAGE) || file_bytes < WEX4_PAGE ||
        wio_read(cfd, (unsigned char *)page, WEX4_PAGE, 0) || wex4_parse_header((const unsigned char *)page, file_bytes, &h))
        return wio_fail("not a WEX4 header page");
    uint32_t index_len = (uint32_t)wex4_align(h.index_bytes);
    if (!h.records || posix_memalign(&index, WEX4_PAGE, (size_t)index_len + WEX4_PAGE) ||
        wio_read(cfd, (unsigned char *)index, index_len, h.index_offset) ||
        wex4_open(&h, (const unsigned char *)index, index_len, &v))
        return wio_fail("the index does not open");

    /* the expert records, then a seeded permutation of them */
    WioRec *recs = (WioRec *)malloc((size_t)h.records * sizeof *recs);
    uint32_t *plan = (uint32_t *)malloc((size_t)h.records * sizeof *plan);
    uint32_t count = 0, max_stored = 0;
    if (!recs || !plan) return wio_fail("out of memory");
    for (uint32_t r = 0; r < h.records; r++) {
        Wex4Record rec;
        wex4_get_record(&v, r, &rec);
        if (!(rec.flags & WEX4_REC_EXPERT)) continue;
        recs[count].offset = rec.offset; recs[count].stored = rec.stored; recs[count].sha = rec.sha;
        if (rec.stored > max_stored) max_stored = rec.stored;
        plan[count] = count;
        count++;
    }
    if (!count) return wio_fail("the container has no expert record");
    uint64_t x = seed * UINT64_C(0x9E3779B97F4A7C15) + UINT64_C(0x632BE59BD9B4E019);
    for (uint32_t i = count - 1u; i > 0; i--) {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;   /* xorshift64* */
        uint32_t j = (uint32_t)(((x * UINT64_C(0x2545F4914F6CDD1D)) >> 32) % (i + 1u)), t = plan[i];
        plan[i] = plan[j]; plan[j] = t;
    }
    uint32_t n = want && want < count ? want : count;

    int fd = open(argv[1], O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) return direct ? wio_skip("the filesystem refuses O_DIRECT") : wio_fail("cannot open the container");
    WioPass pass;
    memset(&pass, 0, sizeof pass);
    pass.fd = fd; pass.recs = recs; pass.plan = plan; pass.n = n; pass.verify = verify; pass.ahead = ahead;
    pass.lat = (double *)calloc(n, sizeof(double));

    /* one buffer per thread or per record in flight, touched before the clock starts */
    unsigned nbuf = uring ? depth : threads;
    unsigned char *bufs[WIO_MAX_SLOTS];
    for (unsigned k = 0; k < nbuf; k++) {
        void *b = NULL;
        if (!pass.lat || posix_memalign(&b, WEX4_PAGE, max_stored)) return wio_fail("out of memory");
        memset(b, 1, max_stored);
        bufs[k] = (unsigned char *)b;
    }

    if (evict && !warm) posix_fadvise(cfd, 0, 0, POSIX_FADV_DONTNEED);
    if (warm)
        for (uint32_t i = 0; i < n; i++)
            if (wio_read(cfd, bufs[0], recs[plan[i]].stored, recs[plan[i]].offset)) return wio_fail("warm-up read");
    double cached_before = wio_cached(cfd, file_bytes, &pass);
    if (!strcmp(advise, "random")) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    if (!strcmp(advise, "sequential")) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    uint64_t bytes = 0;
    uint32_t bad = 0;
    errno = 0;
    double t0 = wio_now();
    int rc = uring ? wio_uring(&pass, bufs, depth, split, &bytes, &bad) : wio_sync(&pass, bufs, threads, &bytes, &bad);
    double seconds = wio_now() - t0;
    if (rc == -2) {
        char why[96];
        snprintf(why, sizeof why, "io_uring is not available here (%s)", errno ? strerror(errno) : "not built in");
        return wio_skip(why);
    }
    if (rc || bad) {
        fprintf(stderr, "wex_io: %u of %u records failed%s (%s)\n", bad, n, verify ? " or do not match their SHA-256" : "",
                errno ? strerror(errno) : "no errno");
        return 1;
    }
    double cached_after = wio_cached(cfd, file_bytes, &pass);
    qsort(pass.lat, n, sizeof(double), wio_cmp);
    printf("{\"path\": \"%s\", \"direct\": %d, \"threads\": %u, \"depth\": %u, \"split\": %u, \"advise\": \"%s\", \"ahead\": %d, "
           "\"warm\": %d, \"verified\": %d, \"records\": %u, \"bytes\": %llu, \"seconds\": %.6f, \"mb_s\": %.1f, "
           "\"record_ms_p50\": %.3f, \"record_ms_max\": %.3f, \"cached_before\": %.4f, \"cached_after\": %.4f}\n",
           path, direct, uring ? 1u : threads, uring ? depth : 1u, uring ? split : 1u, advise, ahead, warm, verify, n,
           (unsigned long long)bytes, seconds, (double)bytes / seconds / 1e6, pass.lat[n / 2u] * 1e3, pass.lat[n - 1u] * 1e3,
           cached_before, cached_after);
    return 0;
}
#else
int main(void) {
    puts("{\"skipped\": \"wex_io reads through the Linux system calls; it does nothing on this system\"}");
    return 0;
}
#endif
