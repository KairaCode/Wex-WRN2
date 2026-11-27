/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

#ifdef __linux__

#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#define WEX_IO_DEFINE_COMMON_FUNCS
#include "wex_io_common.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#define WEX_HAVE_IO_URING_H 1
#endif
#else
#include <linux/io_uring.h>
#define WEX_HAVE_IO_URING_H 1
#endif

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif

#define WEX_LNX_MAX_READ    (16u << 20)
#define WEX_LNX_MAX_SLOTS   64
#define WEX_LNX_BOUNCE_CAP  (64u << 10)

#if defined(WEX_HAVE_IO_URING_H) && defined(IORING_OP_READ) && defined(IORING_FEAT_RW_CUR_POS)
#define WEX_CAN_IO_URING 1
typedef struct {
    int                   fd;
    void                 *sq;
    void                 *cq;
    void                 *sqe_map;
    size_t                sq_len;
    size_t                cq_len;
    size_t                sqe_len;
    unsigned             *sq_head;
    unsigned             *sq_tail;
    unsigned             *sq_mask;
    unsigned             *sq_array;
    unsigned             *cq_head;
    unsigned             *cq_tail;
    unsigned             *cq_mask;
    struct io_uring_sqe  *sqes;
    struct io_uring_cqe  *cqes;
    unsigned              to_submit;
} wex_uring_t;
#else
#define WEX_CAN_IO_URING 0
typedef struct { int fd; } wex_uring_t;
#endif

struct wex_io {
    int              fds[WEX_IO_MAX_REPLICAS];
    int              direct_flags[WEX_IO_MAX_REPLICAS];
    int              nrep;
    int              is_direct;
    int              qd;
    uint64_t         file_size;
    wex_io_picker_t  picker;

    /* Raw Linux io_uring state (active when use_uring == 1) */
    int              use_uring;
    volatile int     ring_busy;
    wex_uring_t      ring;

    /* Sync read_exact bounce buffer */
    pthread_mutex_t  sync_mu;
    uint8_t         *bounce_buf;
    size_t           bounce_cap;
};

static int lnx_open_fd(const char *path, int direct_mode, int *out_direct) {
    int fd = -1;
    int used_direct = 0;
    if (direct_mode != 0 && O_DIRECT != 0) {
        fd = open(path, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            used_direct = 1;
        } else if (errno == EINVAL || errno == EOPNOTSUPP || errno == EACCES) {
            /* Automatic fallback to buffered on filesystems without O_DIRECT */
            fd = open(path, O_RDONLY);
            used_direct = 0;
        }
    } else {
        fd = open(path, O_RDONLY);
        used_direct = 0;
    }
    if (out_direct) *out_direct = used_direct;
    return fd;
}

static ssize_t lnx_pread_once(int fd, void *buf, size_t n, uint64_t off) {
    for (;;) {
        ssize_t got = pread(fd, buf, n, (off_t)off);
        if (got < 0 && errno == EINTR) continue;
        return got;
    }
}

static int lnx_read_exact_impl(int fd, int is_direct, uint64_t file_size,
                               uint8_t *bounce, size_t bounce_cap,
                               uint64_t offset, void *buf, size_t len) {
    if (len == 0) return 0;
    if (!buf || offset > file_size || len > file_size - offset) return -1;

    uint8_t *dst = (uint8_t *)buf;
    int aligned_ok = (!is_direct) ||
                     (((offset & (WEX_IO_ALIGN - 1u)) == 0) &&
                      ((len & (WEX_IO_ALIGN - 1u)) == 0) &&
                      (((uintptr_t)dst & (WEX_IO_ALIGN - 1u)) == 0));

    if (aligned_ok) {
        size_t rem = len;
        uint64_t off = offset;
        while (rem > 0) {
            size_t chunk = rem > WEX_LNX_MAX_READ ? WEX_LNX_MAX_READ : rem;
            ssize_t got = lnx_pread_once(fd, dst, chunk, off);
            if (got <= 0) return -1;
            dst += (size_t)got;
            off += (uint64_t)got;
            rem -= (size_t)got;
        }
        return 0;
    }

    if (!bounce || bounce_cap < WEX_IO_ALIGN) return -1;
    uint64_t aoff = offset & ~(uint64_t)(WEX_IO_ALIGN - 1u);
    size_t skip = (size_t)(offset - aoff);
    size_t rem = len;

    while (rem > 0) {
        uint64_t need_bytes = (uint64_t)skip + (uint64_t)rem;
        uint64_t aligned_need = (need_bytes + (WEX_IO_ALIGN - 1u)) & ~(uint64_t)(WEX_IO_ALIGN - 1u);
        size_t want = (aligned_need > (uint64_t)bounce_cap) ? bounce_cap : (size_t)aligned_need;
        uint64_t max_avail = file_size - aoff;
        size_t min_expected = (max_avail < (uint64_t)want) ? (size_t)max_avail : want;

        size_t total_got = 0;
        while (total_got < min_expected) {
            size_t req_sz = want - total_got;
            ssize_t got = lnx_pread_once(fd, bounce + total_got, req_sz, aoff + total_got);
            if (got <= 0) return -1;
            total_got += (size_t)got;
            if (aoff + total_got >= file_size) break;
        }
        if (total_got <= skip) return -1;

        size_t avail = total_got - skip;
        size_t take = avail < rem ? avail : rem;
        memcpy(dst, bounce + skip, take);
        dst += take;
        rem -= take;
        aoff += (uint64_t)want;
        skip = 0;
    }
    return 0;
}

typedef struct {
    int      fd;
    int      is_direct;
    uint8_t *bounce;
    size_t   bounce_cap;
} lnx_page0_ctx_t;

static int lnx_read_page0(void *vctx, uint8_t *page_out, size_t page_len, uint64_t *size_out) {
    lnx_page0_ctx_t *c = (lnx_page0_ctx_t *)vctx;
    struct stat st;
    if (fstat(c->fd, &st) != 0 || st.st_size < 0) return -1;
    uint64_t fsz = (uint64_t)st.st_size;
    *size_out = fsz;
    if (page_len == 0) return 0;
    if (fsz < page_len) return -1;
    return lnx_read_exact_impl(c->fd, c->is_direct, fsz, c->bounce, c->bounce_cap,
                               0, page_out, page_len);
}

#if WEX_CAN_IO_URING
static void lnx_uring_close(wex_uring_t *r) {
    if (r->sq && r->sq != MAP_FAILED) munmap(r->sq, r->sq_len);
    if (r->cq && r->cq != MAP_FAILED) munmap(r->cq, r->cq_len);
    if (r->sqe_map && r->sqe_map != MAP_FAILED) munmap(r->sqe_map, r->sqe_len);
    if (r->fd >= 0) close(r->fd);
    memset(r, 0, sizeof(*r));
    r->fd = -1;
    r->sq = MAP_FAILED;
    r->cq = MAP_FAILED;
    r->sqe_map = MAP_FAILED;
}

static int lnx_uring_open(wex_uring_t *r, unsigned entries) {
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    memset(r, 0, sizeof(*r));
    r->fd = -1;
    r->sq = MAP_FAILED;
    r->cq = MAP_FAILED;
    r->sqe_map = MAP_FAILED;

    r->fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (r->fd < 0) return -1;
    if (!(p.features & IORING_FEAT_RW_CUR_POS)) {
        lnx_uring_close(r);
        return -1;
    }

    r->sq_len  = (size_t)p.sq_off.array + (size_t)p.sq_entries * sizeof(unsigned);
    r->cq_len  = (size_t)p.cq_off.cqes + (size_t)p.cq_entries * sizeof(struct io_uring_cqe);
    r->sqe_len = (size_t)p.sq_entries * sizeof(struct io_uring_sqe);

    r->sq      = mmap(NULL, r->sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      r->fd, IORING_OFF_SQ_RING);
    r->cq      = mmap(NULL, r->cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      r->fd, IORING_OFF_CQ_RING);
    r->sqe_map = mmap(NULL, r->sqe_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      r->fd, IORING_OFF_SQES);
    if (r->sq == MAP_FAILED || r->cq == MAP_FAILED || r->sqe_map == MAP_FAILED) {
        lnx_uring_close(r);
        return -1;
    }

    unsigned char *sq = (unsigned char *)r->sq;
    unsigned char *cq = (unsigned char *)r->cq;
    r->sq_head  = (unsigned *)(sq + p.sq_off.head);
    r->sq_tail  = (unsigned *)(sq + p.sq_off.tail);
    r->sq_mask  = (unsigned *)(sq + p.sq_off.ring_mask);
    r->sq_array = (unsigned *)(sq + p.sq_off.array);
    r->cq_head  = (unsigned *)(cq + p.cq_off.head);
    r->cq_tail  = (unsigned *)(cq + p.cq_off.tail);
    r->cq_mask  = (unsigned *)(cq + p.cq_off.ring_mask);
    r->cqes     = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    r->sqes     = (struct io_uring_sqe *)r->sqe_map;
    r->to_submit = 0;
    return 0;
}

static void lnx_uring_queue_read(wex_uring_t *r, int fd, void *buf,
                                 uint32_t len, uint64_t off, uint64_t tag) {
    unsigned tail = *r->sq_tail;
    unsigned at = tail & *r->sq_mask;
    struct io_uring_sqe *s = &r->sqes[at];
    memset(s, 0, sizeof(*s));
    s->opcode    = IORING_OP_READ;
    s->fd        = fd;
    s->addr      = (uint64_t)(uintptr_t)buf;
    s->len       = len;
    s->off       = off;
    s->user_data = tag;
    r->sq_array[at] = at;
    __atomic_store_n(r->sq_tail, tail + 1u, __ATOMIC_RELEASE);
    r->to_submit++;
}

static int lnx_uring_enter(wex_uring_t *r, unsigned min_complete) {
    for (;;) {
        unsigned flags = min_complete ? IORING_ENTER_GETEVENTS : 0u;
        long ret = syscall(__NR_io_uring_enter, r->fd, r->to_submit, min_complete, flags, NULL, 0);
        if (ret >= 0) {
            if ((unsigned)ret <= r->to_submit) r->to_submit -= (unsigned)ret;
            else r->to_submit = 0;
            return 0;
        }
        if (errno == EINTR) continue;
        if ((errno == EAGAIN || errno == EBUSY) &&
            *r->cq_head != __atomic_load_n(r->cq_tail, __ATOMIC_ACQUIRE)) {
            return 0;
        }
        return -1;
    }
}

typedef struct {
    int      req_idx;
    int      rep;
    uint64_t pos;
    uint64_t sha_emitted;
} lnx_slot_t;

static void lnx_uring_submit_next(wex_io_t *io, lnx_slot_t *slots, int s_idx,
                                  wex_io_req_t *reqs, wex_io_chunk_fn on_chunk) {
    lnx_slot_t *s = &slots[s_idx];
    wex_io_req_t *r = &reqs[s->req_idx];
    uint64_t rem = r->length - s->pos;
    uint64_t cap;
    if (r->stream_chunks && on_chunk) {
        uint64_t in_chunk = s->pos - s->sha_emitted;
        cap = (in_chunk < WEX_IO_SHA_CHUNK) ? (WEX_IO_SHA_CHUNK - in_chunk) : WEX_IO_SHA_CHUNK;
        if (cap > WEX_LNX_MAX_READ) cap = WEX_LNX_MAX_READ;
    } else {
        cap = WEX_LNX_MAX_READ;
    }
    uint32_t want = (rem > cap) ? (uint32_t)cap : (uint32_t)rem;
    lnx_uring_queue_read(&io->ring, io->fds[s->rep], r->buf + s->pos, want,
                         r->file_offset + s->pos, (uint64_t)s_idx);
}

static int lnx_batch_uring(wex_io_t *io, wex_io_req_t *reqs, int n,
                           wex_io_chunk_fn on_chunk, wex_io_done_fn on_done, void *ctx) {
    int max_slots = io->qd;
    if (max_slots > n) max_slots = n;
    if (max_slots > WEX_LNX_MAX_SLOTS) max_slots = WEX_LNX_MAX_SLOTS;

    lnx_slot_t slots[WEX_LNX_MAX_SLOTS];
    int free_stack[WEX_LNX_MAX_SLOTS];
    int nfree = max_slots;
    for (int s = 0; s < max_slots; s++) {
        memset(&slots[s], 0, sizeof(slots[s]));
        slots[s].req_idx = -1;
        free_stack[s] = max_slots - 1 - s;
    }

    int next_req = 0;
    int in_flight = 0;
    int failed = 0;

    while (next_req < n || in_flight > 0) {
        while (nfree > 0 && next_req < n) {
            int idx = next_req++;
            wex_io_req_t *r = &reqs[idx];
            r->status = 0;
            r->used_replica = -1;

            if (r->length == 0) {
                r->used_replica = 0;
                if (r->stream_chunks && on_chunk) on_chunk(ctx, idx, r->buf, 0, 0, 1);
                if (on_done) on_done(ctx, idx, 0);
                continue;
            }
            if (!r->buf || r->file_offset > io->file_size ||
                r->length > io->file_size - r->file_offset) {
                r->status = -1;
                failed++;
                if (on_done) on_done(ctx, idx, -1);
                continue;
            }

            int rep = wex_io_pick_rep(&io->picker, r->hint_replica, r->length);
            if (io->direct_flags[rep] &&
                (((r->file_offset & (WEX_IO_ALIGN - 1u)) != 0) ||
                 ((r->length & (WEX_IO_ALIGN - 1u)) != 0) ||
                 (((uintptr_t)r->buf & (WEX_IO_ALIGN - 1u)) != 0))) {
                wex_io_unpick_rep(&io->picker, rep, r->length);
                r->status = -1;
                failed++;
                if (on_done) on_done(ctx, idx, -1);
                continue;
            }

            int s_idx = free_stack[--nfree];
            r->used_replica = rep;
            slots[s_idx].req_idx = idx;
            slots[s_idx].rep = rep;
            slots[s_idx].pos = 0;
            slots[s_idx].sha_emitted = 0;
            lnx_uring_submit_next(io, slots, s_idx, reqs, on_chunk);
            in_flight++;
        }

        if (in_flight == 0) continue;

        if (lnx_uring_enter(&io->ring, 1u) != 0) {
            /* Fatal ring error: drain any already-posted CQEs, mark remaining failed. */
            unsigned head = *io->ring.cq_head;
            unsigned tail = __atomic_load_n(io->ring.cq_tail, __ATOMIC_ACQUIRE);
            __atomic_store_n(io->ring.cq_head, tail, __ATOMIC_RELEASE);
            (void)head;
            for (int s = 0; s < max_slots; s++) {
                if (slots[s].req_idx >= 0) {
                    int idx = slots[s].req_idx;
                    reqs[idx].status = -1;
                    failed++;
                    wex_io_unpick_rep(&io->picker, slots[s].rep, reqs[idx].length);
                    slots[s].req_idx = -1;
                    if (on_done) on_done(ctx, idx, -1);
                }
            }
            while (next_req < n) {
                int idx = next_req++;
                reqs[idx].status = -1;
                reqs[idx].used_replica = -1;
                failed++;
                if (on_done) on_done(ctx, idx, -1);
            }
            break;
        }

        unsigned head = *io->ring.cq_head;
        unsigned tail = __atomic_load_n(io->ring.cq_tail, __ATOMIC_ACQUIRE);
        while (head != tail) {
            const struct io_uring_cqe *c = &io->ring.cqes[head & *io->ring.cq_mask];
            int s_idx = (int)c->user_data;
            int32_t res = c->res;
            head++;

            if (s_idx < 0 || s_idx >= max_slots || slots[s_idx].req_idx < 0) continue;
            lnx_slot_t *s = &slots[s_idx];
            int idx = s->req_idx;
            wex_io_req_t *r = &reqs[idx];

            if (res <= 0) {
                r->status = -1;
                failed++;
                wex_io_unpick_rep(&io->picker, s->rep, r->length);
                s->req_idx = -1;
                free_stack[nfree++] = s_idx;
                in_flight--;
                if (on_done) on_done(ctx, idx, -1);
                continue;
            }

            s->pos += (uint64_t)res;
            if (s->pos > r->length) s->pos = r->length;

            if (r->stream_chunks && on_chunk) {
                uint64_t unemitted = s->pos - s->sha_emitted;
                if (unemitted >= WEX_IO_SHA_CHUNK || s->pos == r->length) {
                    uint64_t off = s->sha_emitted;
                    int is_last = (s->pos == r->length) ? 1 : 0;
                    s->sha_emitted = s->pos;
                    on_chunk(ctx, idx, r->buf, off, unemitted, is_last);
                }
            }

            if (s->pos < r->length) {
                lnx_uring_submit_next(io, slots, s_idx, reqs, on_chunk);
            } else {
                r->status = 0;
                wex_io_unpick_rep(&io->picker, s->rep, r->length);
                s->req_idx = -1;
                free_stack[nfree++] = s_idx;
                in_flight--;
                if (on_done) on_done(ctx, idx, 0);
            }
        }
        __atomic_store_n(io->ring.cq_head, head, __ATOMIC_RELEASE);
    }

    return failed;
}
#endif /* WEX_CAN_IO_URING */

/* ---- Multi-threaded pread() fallback (restricted containers / contention) - */
typedef struct {
    wex_io_t        *io;
    wex_io_req_t    *reqs;
    int              n;
    volatile int     next;
    volatile int     failed;
    wex_io_chunk_fn  on_chunk;
    wex_io_done_fn   on_done;
    void            *ctx;
} lnx_pread_batch_t;

static void *lnx_pread_worker(void *arg) {
    lnx_pread_batch_t *b = (lnx_pread_batch_t *)arg;
    wex_io_t *io = b->io;

    for (;;) {
        int idx = __atomic_fetch_add(&b->next, 1, __ATOMIC_RELAXED);
        if (idx >= b->n) break;

        wex_io_req_t *r = &b->reqs[idx];
        r->status = 0;
        r->used_replica = -1;

        if (r->length == 0) {
            r->used_replica = 0;
            if (r->stream_chunks && b->on_chunk) b->on_chunk(b->ctx, idx, r->buf, 0, 0, 1);
            if (b->on_done) b->on_done(b->ctx, idx, 0);
            continue;
        }
        if (!r->buf || r->file_offset > io->file_size ||
            r->length > io->file_size - r->file_offset) {
            r->status = -1;
            __atomic_fetch_add(&b->failed, 1, __ATOMIC_RELAXED);
            if (b->on_done) b->on_done(b->ctx, idx, -1);
            continue;
        }

        int rep = wex_io_pick_rep(&io->picker, r->hint_replica, r->length);
        if (io->direct_flags[rep] &&
            (((r->file_offset & (WEX_IO_ALIGN - 1u)) != 0) ||
             ((r->length & (WEX_IO_ALIGN - 1u)) != 0) ||
             (((uintptr_t)r->buf & (WEX_IO_ALIGN - 1u)) != 0))) {
            wex_io_unpick_rep(&io->picker, rep, r->length);
            r->status = -1;
            __atomic_fetch_add(&b->failed, 1, __ATOMIC_RELAXED);
            if (b->on_done) b->on_done(b->ctx, idx, -1);
            continue;
        }

        r->used_replica = rep;
        uint64_t pos = 0;
        uint64_t sha_emitted = 0;
        int err = 0;

        while (pos < r->length) {
            uint64_t rem = r->length - pos;
            uint64_t cap;
            if (r->stream_chunks && b->on_chunk) {
                uint64_t in_chunk = pos - sha_emitted;
                cap = (in_chunk < WEX_IO_SHA_CHUNK) ? (WEX_IO_SHA_CHUNK - in_chunk) : WEX_IO_SHA_CHUNK;
                if (cap > WEX_LNX_MAX_READ) cap = WEX_LNX_MAX_READ;
            } else {
                cap = WEX_LNX_MAX_READ;
            }
            size_t want = (rem > cap) ? (size_t)cap : (size_t)rem;
            ssize_t got = lnx_pread_once(io->fds[rep], r->buf + pos, want, r->file_offset + pos);
            if (got <= 0) {
                err = 1;
                break;
            }
            pos += (uint64_t)got;
            if (pos > r->length) pos = r->length;

            if (r->stream_chunks && b->on_chunk) {
                uint64_t unemitted = pos - sha_emitted;
                if (unemitted >= WEX_IO_SHA_CHUNK || pos == r->length) {
                    uint64_t off = sha_emitted;
                    int is_last = (pos == r->length) ? 1 : 0;
                    sha_emitted = pos;
                    b->on_chunk(b->ctx, idx, r->buf, off, unemitted, is_last);
                }
            }
        }

        wex_io_unpick_rep(&io->picker, rep, r->length);
        if (err) {
            r->status = -1;
            __atomic_fetch_add(&b->failed, 1, __ATOMIC_RELAXED);
            if (b->on_done) b->on_done(b->ctx, idx, -1);
        } else {
            r->status = 0;
            if (b->on_done) b->on_done(b->ctx, idx, 0);
        }
    }
    return NULL;
}

static int lnx_batch_pread(wex_io_t *io, wex_io_req_t *reqs, int n,
                           wex_io_chunk_fn on_chunk, wex_io_done_fn on_done, void *ctx) {
    lnx_pread_batch_t batch;
    batch.io = io;
    batch.reqs = reqs;
    batch.n = n;
    batch.next = 0;
    batch.failed = 0;
    batch.on_chunk = on_chunk;
    batch.on_done = on_done;
    batch.ctx = ctx;

    int nthreads = io->qd;
    if (nthreads > n) nthreads = n;
    if (nthreads > WEX_LNX_MAX_SLOTS) nthreads = WEX_LNX_MAX_SLOTS;
    if (nthreads <= 1) {
        lnx_pread_worker(&batch);
        return batch.failed;
    }

    pthread_t tids[WEX_LNX_MAX_SLOTS];
    int started = 0;
    for (int t = 0; t < nthreads - 1; t++) {
        if (pthread_create(&tids[started], NULL, lnx_pread_worker, &batch) == 0) {
            started++;
        }
    }
    lnx_pread_worker(&batch);
    for (int t = 0; t < started; t++) {
        pthread_join(tids[t], NULL);
    }
    return batch.failed;
}

wex_io_t *wex_io_open(const char *primary_path,
                      const char *const *mirrors, int num_mirrors,
                      const uint32_t *weights,
                      int direct_mode, int queue_depth,
                      char *err_buf, size_t err_cap) {
    if (!primary_path || !*primary_path) {
        wex_io_set_err(err_buf, err_cap, "primary_path is NULL or empty");
        return NULL;
    }

    int primary_direct = 0;
    int fd0 = lnx_open_fd(primary_path, direct_mode, &primary_direct);
    if (fd0 < 0) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "open failed (%s)", strerror(errno));
        }
        return NULL;
    }

    struct stat st;
    if (fstat(fd0, &st) != 0 || st.st_size < 0) {
        close(fd0);
        wex_io_set_err(err_buf, err_cap, "fstat failed on primary file");
        return NULL;
    }
    uint64_t file_size = (uint64_t)st.st_size;

    wex_io_t *io = (wex_io_t *)calloc(1, sizeof(*io));
    if (!io) {
        close(fd0);
        wex_io_set_err(err_buf, err_cap, "out of memory allocating wex_io_t");
        return NULL;
    }
    for (int i = 0; i < WEX_IO_MAX_REPLICAS; i++) io->fds[i] = -1;
    io->ring.fd = -1;

    io->fds[0] = fd0;
    io->direct_flags[0] = primary_direct;
    io->nrep = 1;
    io->is_direct = primary_direct;
    io->file_size = file_size;
    pthread_mutex_init(&io->sync_mu, NULL);

    io->bounce_cap = WEX_LNX_BOUNCE_CAP;
    io->bounce_buf = (uint8_t *)wex_io_common_alloc_aligned(io->bounce_cap);
    if (!io->bounce_buf) {
        wex_io_close(io);
        wex_io_set_err(err_buf, err_cap, "failed to allocate aligned bounce buffer");
        return NULL;
    }

    uint32_t accepted_weights[WEX_IO_MAX_REPLICAS];
    accepted_weights[0] = (weights && weights[0] > 0) ? weights[0] : 1u;

    if (mirrors && num_mirrors > 0) {
        size_t hdr_len = file_size < WEX_IO_ALIGN ? (size_t)file_size : (size_t)WEX_IO_ALIGN;
        uint8_t *primary_hdr = (uint8_t *)wex_io_common_alloc_aligned(WEX_IO_ALIGN);
        if (primary_hdr) {
            memset(primary_hdr, 0, WEX_IO_ALIGN);
            int hdr_ok = (hdr_len == 0) ||
                         (lnx_read_exact_impl(fd0, primary_direct, file_size,
                                              io->bounce_buf, io->bounce_cap,
                                              0, primary_hdr, hdr_len) == 0);
            if (hdr_ok) {
                for (int m = 0; m < num_mirrors && io->nrep < WEX_IO_MAX_REPLICAS; m++) {
                    if (!mirrors[m] || !*mirrors[m]) continue;
                    int m_direct = 0;
                    int mfd = lnx_open_fd(mirrors[m], direct_mode, &m_direct);
                    if (mfd < 0) continue;

                    lnx_page0_ctx_t mctx;
                    mctx.fd = mfd;
                    mctx.is_direct = m_direct;
                    mctx.bounce = io->bounce_buf;
                    mctx.bounce_cap = io->bounce_cap;

                    if (wex_io_mirror_matches(file_size, primary_hdr, hdr_len,
                                              lnx_read_page0, &mctx)) {
                        io->fds[io->nrep] = mfd;
                        io->direct_flags[io->nrep] = m_direct;
                        accepted_weights[io->nrep] =
                            (weights && weights[1 + m] > 0) ? weights[1 + m] : 1u;
                        io->nrep++;
                    } else {
                        close(mfd);
                    }
                }
            }
            wex_io_common_free_aligned(primary_hdr);
        }
    }

    wex_io_picker_init(&io->picker, io->nrep, weights ? accepted_weights : NULL);
    io->qd = wex_io_default_qd(queue_depth, io->nrep, WEX_LNX_MAX_SLOTS);

#if WEX_CAN_IO_URING
    unsigned ring_entries = 64u;
    while (ring_entries < (unsigned)io->qd * 2u) ring_entries <<= 1u;
    if (lnx_uring_open(&io->ring, ring_entries) == 0) {
        io->use_uring = 1;
    }
#endif

    return io;
}

const char *wex_io_engine_name(const wex_io_t *io) {
    return (io && io->use_uring) ? "linux-iouring" : "linux-pread";
}

int wex_io_replica_count(const wex_io_t *io) {
    return io ? io->nrep : 0;
}

uint64_t wex_io_file_size(const wex_io_t *io) {
    return io ? io->file_size : 0;
}

int wex_io_is_direct(const wex_io_t *io) {
    return io ? io->is_direct : 0;
}

int wex_io_read_exact(wex_io_t *io, uint64_t offset, void *buf, size_t len) {
    if (!io) return -1;
    if (len == 0) return 0;
    pthread_mutex_lock(&io->sync_mu);
    int rc = lnx_read_exact_impl(io->fds[0], io->direct_flags[0], io->file_size,
                                 io->bounce_buf, io->bounce_cap, offset, buf, len);
    pthread_mutex_unlock(&io->sync_mu);
    return rc;
}

int wex_io_read_batch(wex_io_t *io, wex_io_req_t *reqs, int n,
                      wex_io_chunk_fn on_chunk, wex_io_done_fn on_done, void *ctx) {
    if (n <= 0) return 0;
    if (!io || !reqs) return n;

#if WEX_CAN_IO_URING
    if (io->use_uring && __atomic_exchange_n(&io->ring_busy, 1, __ATOMIC_ACQUIRE) == 0) {
        int failed = lnx_batch_uring(io, reqs, n, on_chunk, on_done, ctx);
        __atomic_store_n(&io->ring_busy, 0, __ATOMIC_RELEASE);
        return failed;
    }
#endif

    return lnx_batch_pread(io, reqs, n, on_chunk, on_done, ctx);
}

void wex_io_close(wex_io_t *io) {
    if (!io) return;
#if WEX_CAN_IO_URING
    if (io->use_uring) lnx_uring_close(&io->ring);
#endif
    wex_io_common_free_aligned(io->bounce_buf);
    pthread_mutex_destroy(&io->sync_mu);
    for (int r = 0; r < WEX_IO_MAX_REPLICAS; r++) {
        if (io->fds[r] >= 0) close(io->fds[r]);
    }
    free(io);
}

#else
typedef int wex_io_linux_unused_t;
#endif /* __linux__ */
