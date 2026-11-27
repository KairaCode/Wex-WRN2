/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

#if !defined(_WIN32) && !defined(__linux__)

#ifdef __APPLE__
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#elif !defined(__FreeBSD__) && !defined(__OpenBSD__) && !defined(__NetBSD__)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif
#define _FILE_OFFSET_BITS 64

#define WEX_IO_DEFINE_COMMON_FUNCS
#include "wex_io_common.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

#define WEX_POSIX_MAX_READ    (16u << 20)
#define WEX_POSIX_MAX_SLOTS   64
#define WEX_POSIX_BOUNCE_CAP  (64u << 10)

struct wex_io {
    int              fds[WEX_IO_MAX_REPLICAS];
    int              direct_flags[WEX_IO_MAX_REPLICAS];
    int              nrep;
    int              is_direct;
    int              qd;
    uint64_t         file_size;
    wex_io_picker_t  picker;

    pthread_mutex_t  sync_mu;
    uint8_t         *bounce_buf;
    size_t           bounce_cap;
};

static int posix_open_fd(const char *path, int direct_mode, int *out_direct) {
    int fd = -1;
    int used_direct = 0;

#ifdef __APPLE__
    fd = open(path, O_RDONLY);
    if (fd >= 0 && direct_mode != 0) {
        if (fcntl(fd, F_NOCACHE, 1) == 0) {
            used_direct = 1;
            (void)fcntl(fd, F_RDAHEAD, 0);
        }
    }
#elif defined(O_DIRECT)
    if (direct_mode != 0) {
        fd = open(path, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            used_direct = 1;
        } else if (errno == EINVAL || errno == EOPNOTSUPP) {
            fd = open(path, O_RDONLY);
            used_direct = 0;
        }
    } else {
        fd = open(path, O_RDONLY);
    }
#else
    (void)direct_mode;
    fd = open(path, O_RDONLY);
#endif

    if (out_direct) *out_direct = used_direct;
    return fd;
}

static ssize_t posix_pread_once(int fd, void *buf, size_t n, uint64_t off) {
    for (;;) {
        ssize_t got = pread(fd, buf, n, (off_t)off);
        if (got < 0 && errno == EINTR) continue;
        return got;
    }
}

static int posix_read_exact_impl(int fd, int is_direct, uint64_t file_size,
                                 uint8_t *bounce, size_t bounce_cap,
                                 uint64_t offset, void *buf, size_t len) {
    if (len == 0) return 0;
    if (!buf || offset > file_size || len > file_size - offset) return -1;

    uint8_t *dst = (uint8_t *)buf;
    int aligned_ok = (!is_direct) ||
                     (((offset & (WEX_IO_ALIGN - 1u)) == 0) &&
                      ((len & (WEX_IO_ALIGN - 1u)) == 0) &&
                      (((uintptr_t)dst & (WEX_IO_ALIGN - 1u)) == 0));

#ifdef __APPLE__
    /* macOS F_NOCACHE permits arbitrary offsets/lengths natively */
    aligned_ok = 1;
#endif

    if (aligned_ok) {
        size_t rem = len;
        uint64_t off = offset;
        while (rem > 0) {
            size_t chunk = rem > WEX_POSIX_MAX_READ ? WEX_POSIX_MAX_READ : rem;
            ssize_t got = posix_pread_once(fd, dst, chunk, off);
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
            ssize_t got = posix_pread_once(fd, bounce + total_got, req_sz, aoff + total_got);
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
} posix_page0_ctx_t;

static int posix_read_page0(void *vctx, uint8_t *page_out, size_t page_len, uint64_t *size_out) {
    posix_page0_ctx_t *c = (posix_page0_ctx_t *)vctx;
    struct stat st;
    if (fstat(c->fd, &st) != 0 || st.st_size < 0) return -1;
    uint64_t fsz = (uint64_t)st.st_size;
    *size_out = fsz;
    if (page_len == 0) return 0;
    if (fsz < page_len) return -1;
    return posix_read_exact_impl(c->fd, c->is_direct, fsz, c->bounce, c->bounce_cap,
                                 0, page_out, page_len);
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
    int fd0 = posix_open_fd(primary_path, direct_mode, &primary_direct);
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

    io->fds[0] = fd0;
    io->direct_flags[0] = primary_direct;
    io->nrep = 1;
    io->is_direct = primary_direct;
    io->file_size = file_size;
    pthread_mutex_init(&io->sync_mu, NULL);

    io->bounce_cap = WEX_POSIX_BOUNCE_CAP;
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
                         (posix_read_exact_impl(fd0, primary_direct, file_size,
                                                io->bounce_buf, io->bounce_cap,
                                                0, primary_hdr, hdr_len) == 0);
            if (hdr_ok) {
                for (int m = 0; m < num_mirrors && io->nrep < WEX_IO_MAX_REPLICAS; m++) {
                    if (!mirrors[m] || !*mirrors[m]) continue;
                    int m_direct = 0;
                    int mfd = posix_open_fd(mirrors[m], direct_mode, &m_direct);
                    if (mfd < 0) continue;

                    posix_page0_ctx_t mctx;
                    mctx.fd = mfd;
                    mctx.is_direct = m_direct;
                    mctx.bounce = io->bounce_buf;
                    mctx.bounce_cap = io->bounce_cap;

                    if (wex_io_mirror_matches(file_size, primary_hdr, hdr_len,
                                              posix_read_page0, &mctx)) {
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
    io->qd = wex_io_default_qd(queue_depth, io->nrep, WEX_POSIX_MAX_SLOTS);

    return io;
}

const char *wex_io_engine_name(const wex_io_t *io) {
#ifdef __APPLE__
    return (io && io->is_direct) ? "macos-fnocache" : "posix-pread";
#else
    (void)io;
    return "posix-pread";
#endif
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
    int rc = posix_read_exact_impl(io->fds[0], io->direct_flags[0], io->file_size,
                                   io->bounce_buf, io->bounce_cap, offset, buf, len);
    pthread_mutex_unlock(&io->sync_mu);
    return rc;
}

typedef struct {
    wex_io_t        *io;
    wex_io_req_t    *reqs;
    int              n;
    volatile int     next;
    volatile int     failed;
    wex_io_chunk_fn  on_chunk;
    wex_io_done_fn   on_done;
    void            *ctx;
} posix_batch_t;

static void *posix_batch_worker(void *arg) {
    posix_batch_t *b = (posix_batch_t *)arg;
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
#ifndef __APPLE__
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
#endif

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
                if (cap > WEX_POSIX_MAX_READ) cap = WEX_POSIX_MAX_READ;
            } else {
                cap = WEX_POSIX_MAX_READ;
            }
            size_t want = (rem > cap) ? (size_t)cap : (size_t)rem;
            ssize_t got = posix_pread_once(io->fds[rep], r->buf + pos, want, r->file_offset + pos);
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

int wex_io_read_batch(wex_io_t *io, wex_io_req_t *reqs, int n,
                      wex_io_chunk_fn on_chunk, wex_io_done_fn on_done, void *ctx) {
    if (n <= 0) return 0;
    if (!io || !reqs) return n;

    posix_batch_t batch;
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
    if (nthreads > WEX_POSIX_MAX_SLOTS) nthreads = WEX_POSIX_MAX_SLOTS;
    if (nthreads <= 1) {
        posix_batch_worker(&batch);
        return batch.failed;
    }

    pthread_t tids[WEX_POSIX_MAX_SLOTS];
    int started = 0;
    for (int t = 0; t < nthreads - 1; t++) {
        if (pthread_create(&tids[started], NULL, posix_batch_worker, &batch) == 0) {
            started++;
        }
    }
    posix_batch_worker(&batch);
    for (int t = 0; t < started; t++) {
        pthread_join(tids[t], NULL);
    }
    return batch.failed;
}

void wex_io_close(wex_io_t *io) {
    if (!io) return;
    wex_io_common_free_aligned(io->bounce_buf);
    pthread_mutex_destroy(&io->sync_mu);
    for (int r = 0; r < WEX_IO_MAX_REPLICAS; r++) {
        if (io->fds[r] >= 0) close(io->fds[r]);
    }
    free(io);
}

#else
typedef int wex_io_posix_unused_t;
#endif /* !defined(_WIN32) && !defined(__linux__) */
