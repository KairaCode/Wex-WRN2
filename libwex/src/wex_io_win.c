/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

#ifdef _WIN32

#define WEX_IO_DEFINE_COMMON_FUNCS
#include "wex_io_common.h"

/* Cap each OVERLAPPED ReadFile to 16 MiB (#3139) so large uncached transfers
 * never starve shared chipset PCIe devices (e.g., 2.5GbE RX DMA). */
#define WEX_WIN_MAX_READ    (16u << 20)
#define WEX_WIN_MAX_SLOTS   (MAXIMUM_WAIT_OBJECTS - 1) /* 63 */
#define WEX_WIN_BOUNCE_CAP  (64u << 10)                /* 64 KiB aligned bounce */

struct wex_io {
    HANDLE           handles[WEX_IO_MAX_REPLICAS];
    int              direct_flags[WEX_IO_MAX_REPLICAS];
    int              nrep;
    int              is_direct;
    int              qd;
    uint64_t         file_size;
    wex_io_picker_t  picker;

    /* Reusable manual-reset event pool for zero-churn single-dispatcher batching */
    volatile LONG    pool_busy;
    HANDLE           events[WEX_WIN_MAX_SLOTS];

    /* Sync read_exact event + aligned bounce buffer */
    CRITICAL_SECTION sync_cs;
    HANDLE           sync_event;
    uint8_t         *bounce_buf;
    size_t           bounce_cap;
};

static HANDLE win_open_handle(const char *path, int direct_mode, int *out_direct) {
    DWORD base = FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
    HANDLE h = INVALID_HANDLE_VALUE;
    int used_direct = 0;

    if (direct_mode != 0) {
        h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                        NULL, OPEN_EXISTING, base | FILE_FLAG_NO_BUFFERING, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            used_direct = 1;
        } else {
            /* Automatic fallback to buffered on virtual/network filesystems */
            h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            NULL, OPEN_EXISTING, base, NULL);
            used_direct = 0;
        }
    } else {
        h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                        NULL, OPEN_EXISTING, base, NULL);
        used_direct = 0;
    }

    if (out_direct) *out_direct = used_direct;
    return h;
}

static int win_read_raw_sync(HANDLE h, HANDLE ev, uint64_t offset, uint8_t *dst,
                             DWORD want, DWORD *got_out) {
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset     = (DWORD)(offset & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)(offset >> 32);
    ov.hEvent     = ev;
    ResetEvent(ev);

    DWORD got = 0;
    BOOL ok = ReadFile(h, dst, want, &got, &ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err == ERROR_IO_PENDING) {
            ok = GetOverlappedResult(h, &ov, &got, TRUE);
        } else if (err == ERROR_HANDLE_EOF) {
            got = 0;
            ok = TRUE;
        }
    }
    if (!ok) return -1;
    if (got_out) *got_out = got;
    return 0;
}

static int win_read_exact_impl(HANDLE h, int is_direct, uint64_t file_size,
                               HANDLE ev, uint8_t *bounce, size_t bounce_cap,
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
            DWORD chunk = rem > WEX_WIN_MAX_READ ? (DWORD)WEX_WIN_MAX_READ : (DWORD)rem;
            DWORD got = 0;
            if (win_read_raw_sync(h, ev, off, dst, chunk, &got) != 0 || got == 0) {
                return -1;
            }
            dst += got;
            off += got;
            rem -= got;
        }
        return 0;
    }

    /* Direct I/O with unaligned offset, length, or destination buffer: read via aligned bounce. */
    if (!bounce || bounce_cap < WEX_IO_ALIGN) return -1;
    uint64_t aoff = offset & ~(uint64_t)(WEX_IO_ALIGN - 1u);
    size_t skip = (size_t)(offset - aoff);
    size_t rem = len;

    while (rem > 0) {
        uint64_t need_bytes = (uint64_t)skip + (uint64_t)rem;
        uint64_t aligned_need = (need_bytes + (WEX_IO_ALIGN - 1u)) & ~(uint64_t)(WEX_IO_ALIGN - 1u);
        DWORD want = (aligned_need > (uint64_t)bounce_cap) ? (DWORD)bounce_cap : (DWORD)aligned_need;
        uint64_t max_avail = file_size - aoff;
        DWORD min_expected = (max_avail < (uint64_t)want) ? (DWORD)max_avail : want;

        DWORD total_got = 0;
        while (total_got < min_expected) {
            DWORD req_sz = want - total_got;
            DWORD got = 0;
            if (win_read_raw_sync(h, ev, aoff + total_got, bounce + total_got, req_sz, &got) != 0 ||
                got == 0) {
                return -1;
            }
            total_got += got;
            if (aoff + total_got >= file_size) break;
        }
        if (total_got <= skip) return -1;

        size_t avail = (size_t)total_got - skip;
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
    HANDLE   h;
    int      is_direct;
    HANDLE   ev;
    uint8_t *bounce;
    size_t   bounce_cap;
} win_page0_ctx_t;

static int win_read_page0(void *vctx, uint8_t *page_out, size_t page_len, uint64_t *size_out) {
    win_page0_ctx_t *c = (win_page0_ctx_t *)vctx;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(c->h, &sz) || sz.QuadPart < 0) return -1;
    uint64_t fsz = (uint64_t)sz.QuadPart;
    *size_out = fsz;
    if (page_len == 0) return 0;
    if (fsz < page_len) return -1;
    return win_read_exact_impl(c->h, c->is_direct, fsz, c->ev, c->bounce, c->bounce_cap,
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
    HANDLE hp = win_open_handle(primary_path, direct_mode, &primary_direct);
    if (hp == INVALID_HANDLE_VALUE) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "CreateFileA failed (err %lu)",
                     (unsigned long)GetLastError());
        }
        return NULL;
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hp, &sz) || sz.QuadPart < 0) {
        CloseHandle(hp);
        wex_io_set_err(err_buf, err_cap, "GetFileSizeEx failed on primary file");
        return NULL;
    }
    uint64_t file_size = (uint64_t)sz.QuadPart;

    wex_io_t *io = (wex_io_t *)calloc(1, sizeof(*io));
    if (!io) {
        CloseHandle(hp);
        wex_io_set_err(err_buf, err_cap, "out of memory allocating wex_io_t");
        return NULL;
    }
    for (int i = 0; i < WEX_IO_MAX_REPLICAS; i++) io->handles[i] = INVALID_HANDLE_VALUE;

    io->handles[0] = hp;
    io->direct_flags[0] = primary_direct;
    io->nrep = 1;
    io->is_direct = primary_direct;
    io->file_size = file_size;
    InitializeCriticalSection(&io->sync_cs);

    io->sync_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    io->bounce_cap = WEX_WIN_BOUNCE_CAP;
    io->bounce_buf = (uint8_t *)wex_io_common_alloc_aligned(io->bounce_cap);
    if (!io->sync_event || !io->bounce_buf) {
        wex_io_close(io);
        wex_io_set_err(err_buf, err_cap, "failed to allocate sync event or bounce buffer");
        return NULL;
    }

    uint32_t accepted_weights[WEX_IO_MAX_REPLICAS];
    accepted_weights[0] = (weights && weights[0] > 0) ? weights[0] : 1u;

    /* Validate optional mirrors against primary file_size + 4096-byte header page. */
    if (mirrors && num_mirrors > 0) {
        size_t hdr_len = file_size < WEX_IO_ALIGN ? (size_t)file_size : (size_t)WEX_IO_ALIGN;
        uint8_t *primary_hdr = (uint8_t *)wex_io_common_alloc_aligned(WEX_IO_ALIGN);
        if (primary_hdr) {
            memset(primary_hdr, 0, WEX_IO_ALIGN);
            int hdr_ok = (hdr_len == 0) ||
                         (win_read_exact_impl(hp, primary_direct, file_size, io->sync_event,
                                              io->bounce_buf, io->bounce_cap,
                                              0, primary_hdr, hdr_len) == 0);
            if (hdr_ok) {
                for (int m = 0; m < num_mirrors && io->nrep < WEX_IO_MAX_REPLICAS; m++) {
                    if (!mirrors[m] || !*mirrors[m]) continue;
                    int m_direct = 0;
                    HANDLE mh = win_open_handle(mirrors[m], direct_mode, &m_direct);
                    if (mh == INVALID_HANDLE_VALUE) continue;

                    win_page0_ctx_t mctx;
                    mctx.h = mh;
                    mctx.is_direct = m_direct;
                    mctx.ev = io->sync_event;
                    mctx.bounce = io->bounce_buf;
                    mctx.bounce_cap = io->bounce_cap;

                    if (wex_io_mirror_matches(file_size, primary_hdr, hdr_len,
                                              win_read_page0, &mctx)) {
                        io->handles[io->nrep] = mh;
                        io->direct_flags[io->nrep] = m_direct;
                        accepted_weights[io->nrep] =
                            (weights && weights[1 + m] > 0) ? weights[1 + m] : 1u;
                        io->nrep++;
                    } else {
                        CloseHandle(mh);
                    }
                }
            }
            wex_io_common_free_aligned(primary_hdr);
        }
    }

    wex_io_picker_init(&io->picker, io->nrep, weights ? accepted_weights : NULL);
    io->qd = wex_io_default_qd(queue_depth, io->nrep, WEX_WIN_MAX_SLOTS);

    for (int s = 0; s < io->qd; s++) {
        io->events[s] = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!io->events[s]) {
            wex_io_close(io);
            wex_io_set_err(err_buf, err_cap, "CreateEventA failed for OVERLAPPED pool");
            return NULL;
        }
    }

    return io;
}

const char *wex_io_engine_name(const wex_io_t *io) {
    (void)io;
    return "win-overlapped";
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
    EnterCriticalSection(&io->sync_cs);
    int rc = win_read_exact_impl(io->handles[0], io->direct_flags[0], io->file_size,
                                 io->sync_event, io->bounce_buf, io->bounce_cap,
                                 offset, buf, len);
    LeaveCriticalSection(&io->sync_cs);
    return rc;
}

typedef struct {
    int        req_idx;
    int        rep;
    uint64_t   pos;
    uint64_t   sha_emitted;
    OVERLAPPED ov;
    HANDLE     ev;
    int        pending;
} win_slot_t;

static void win_slot_fail(wex_io_t *io, win_slot_t *s, wex_io_req_t *reqs,
                          wex_io_done_fn on_done, void *ctx, int *failed) {
    int idx = s->req_idx;
    int rep = s->rep;
    reqs[idx].status = -1;
    (*failed)++;
    wex_io_unpick_rep(&io->picker, rep, reqs[idx].length);
    s->req_idx = -1;
    s->pending = 0;
    if (on_done) on_done(ctx, idx, -1);
}

/* Advance slot after `got` bytes landed; emit chunk / done callbacks as ready.
 * Returns 1 if slot still needs more reads, 0 if request completed or failed. */
static int win_slot_advance(wex_io_t *io, win_slot_t *s, DWORD got,
                            wex_io_req_t *reqs, wex_io_chunk_fn on_chunk,
                            wex_io_done_fn on_done, void *ctx, int *failed) {
    int idx = s->req_idx;
    wex_io_req_t *r = &reqs[idx];
    if (got == 0) {
        win_slot_fail(io, s, reqs, on_done, ctx, failed);
        return 0;
    }

    s->pos += (uint64_t)got;
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

    if (s->pos >= r->length) {
        r->status = 0;
        wex_io_unpick_rep(&io->picker, s->rep, r->length);
        s->req_idx = -1;
        s->pending = 0;
        if (on_done) on_done(ctx, idx, 0);
        return 0;
    }
    return 1;
}

/* Issue OVERLAPPED ReadFile chunks on slot `s` until either an async read is
 * pending (returns 1) or the request completes/fails synchronously (returns 0). */
static int win_slot_pump(wex_io_t *io, win_slot_t *s, wex_io_req_t *reqs,
                         wex_io_chunk_fn on_chunk, wex_io_done_fn on_done,
                         void *ctx, int *failed) {
    while (s->req_idx >= 0) {
        wex_io_req_t *r = &reqs[s->req_idx];
        uint64_t rem = r->length - s->pos;
        uint64_t cap;
        if (r->stream_chunks && on_chunk) {
            uint64_t in_chunk = s->pos - s->sha_emitted;
            cap = (in_chunk < WEX_IO_SHA_CHUNK) ? (WEX_IO_SHA_CHUNK - in_chunk) : WEX_IO_SHA_CHUNK;
            if (cap > WEX_WIN_MAX_READ) cap = WEX_WIN_MAX_READ;
        } else {
            cap = WEX_WIN_MAX_READ;
        }
        DWORD want = (rem > cap) ? (DWORD)cap : (DWORD)rem;
        uint64_t foff = r->file_offset + s->pos;

        memset(&s->ov, 0, sizeof(s->ov));
        s->ov.Offset     = (DWORD)(foff & 0xFFFFFFFFu);
        s->ov.OffsetHigh = (DWORD)(foff >> 32);
        s->ov.hEvent     = s->ev;
        ResetEvent(s->ev);

        DWORD got = 0;
        BOOL ok = ReadFile(io->handles[s->rep], r->buf + s->pos, want, &got, &s->ov);
        if (ok) {
            /* Completed synchronously: advance and loop if more chunks remain. */
            if (!win_slot_advance(io, s, got, reqs, on_chunk, on_done, ctx, failed)) {
                return 0;
            }
            continue;
        }

        DWORD err = GetLastError();
        if (err == ERROR_IO_PENDING) {
            s->pending = 1;
            return 1;
        }

        win_slot_fail(io, s, reqs, on_done, ctx, failed);
        return 0;
    }
    return 0;
}

int wex_io_read_batch(wex_io_t *io, wex_io_req_t *reqs, int n,
                      wex_io_chunk_fn on_chunk, wex_io_done_fn on_done, void *ctx) {
    if (n <= 0) return 0;
    if (!io || !reqs) return n;

    int max_slots = io->qd;
    if (max_slots > n) max_slots = n;
    if (max_slots > WEX_WIN_MAX_SLOTS) max_slots = WEX_WIN_MAX_SLOTS;

    /* Acquire reusable event pool with zero CreateEvent churn in steady state;
     * if another thread concurrently invokes read_batch, use local events safely. */
    int own_events = 0;
    HANDLE local_events[WEX_WIN_MAX_SLOTS];
    HANDLE *ev_pool = NULL;

    if (InterlockedCompareExchange(&io->pool_busy, 1, 0) == 0) {
        ev_pool = io->events;
    } else {
        own_events = 1;
        for (int s = 0; s < max_slots; s++) {
            local_events[s] = CreateEventA(NULL, TRUE, FALSE, NULL);
            if (!local_events[s]) {
                for (int j = 0; j < s; j++) CloseHandle(local_events[j]);
                for (int i = 0; i < n; i++) {
                    reqs[i].status = -1;
                    reqs[i].used_replica = -1;
                    if (on_done) on_done(ctx, i, -1);
                }
                return n;
            }
        }
        ev_pool = local_events;
    }

    win_slot_t slots[WEX_WIN_MAX_SLOTS];
    for (int s = 0; s < max_slots; s++) {
        memset(&slots[s], 0, sizeof(slots[s]));
        slots[s].req_idx = -1;
        slots[s].ev = ev_pool[s];
    }

    int next_req = 0;
    int in_flight = 0;
    int failed = 0;

    while (next_req < n || in_flight > 0) {
        /* 1. Fill idle slots to saturate NVMe queue depth */
        for (int s = 0; s < max_slots && next_req < n; s++) {
            while (slots[s].req_idx < 0 && next_req < n) {
                int idx = next_req++;
                wex_io_req_t *r = &reqs[idx];
                r->status = 0;
                r->used_replica = -1;

                if (r->length == 0) {
                    r->used_replica = 0;
                    if (r->stream_chunks && on_chunk) {
                        on_chunk(ctx, idx, r->buf, 0, 0, 1);
                    }
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

                r->used_replica = rep;
                slots[s].req_idx = idx;
                slots[s].rep = rep;
                slots[s].pos = 0;
                slots[s].sha_emitted = 0;
                slots[s].pending = 0;

                if (win_slot_pump(io, &slots[s], reqs, on_chunk, on_done, ctx, &failed)) {
                    in_flight++;
                }
            }
        }

        if (in_flight == 0) continue;

        /* 2. Wait for ANY in-flight OVERLAPPED transfer to complete */
        HANDLE wait_handles[WEX_WIN_MAX_SLOTS];
        win_slot_t *wait_slots[WEX_WIN_MAX_SLOTS];
        int nw = 0;
        for (int s = 0; s < max_slots; s++) {
            if (slots[s].pending) {
                wait_handles[nw] = slots[s].ev;
                wait_slots[nw] = &slots[s];
                nw++;
            }
        }
        if (nw == 0) {
            in_flight = 0;
            continue;
        }

        DWORD wr = WaitForMultipleObjects((DWORD)nw, wait_handles, FALSE, INFINITE);
        if ((wr - WAIT_OBJECT_0) < (DWORD)nw) {
            int first_k = (int)(wr - WAIT_OBJECT_0);
            /* Reap the signaled slot plus any other already-finished slots to prevent starvation */
            for (int k = first_k; k < nw; k++) {
                win_slot_t *s = wait_slots[k];
                if (k > first_k && !HasOverlappedIoCompleted(&s->ov)) {
                    continue;
                }
                DWORD got = 0;
                BOOL ok = GetOverlappedResult(io->handles[s->rep], &s->ov, &got, FALSE);
                if (!ok && GetLastError() == ERROR_IO_INCOMPLETE) {
                    continue;
                }
                s->pending = 0;
                in_flight--;

                if (!ok) {
                    win_slot_fail(io, s, reqs, on_done, ctx, &failed);
                } else if (win_slot_advance(io, s, got, reqs, on_chunk, on_done, ctx, &failed)) {
                    if (win_slot_pump(io, s, reqs, on_chunk, on_done, ctx, &failed)) {
                        in_flight++;
                    }
                }
            }
        } else {
            /* Fatal WaitForMultipleObjects failure: cancel & drain all pending I/Os
             * before returning so caller buffers are never touched after return. */
            for (int k = 0; k < nw; k++) {
                win_slot_t *s = wait_slots[k];
                if (s->pending) {
                    CancelIoEx(io->handles[s->rep], &s->ov);
                    DWORD got = 0;
                    GetOverlappedResult(io->handles[s->rep], &s->ov, &got, TRUE);
                    s->pending = 0;
                    in_flight--;
                    win_slot_fail(io, s, reqs, on_done, ctx, &failed);
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
    }

    if (own_events) {
        for (int s = 0; s < max_slots; s++) CloseHandle(local_events[s]);
    } else {
        InterlockedExchange(&io->pool_busy, 0);
    }

    return failed;
}

void wex_io_close(wex_io_t *io) {
    if (!io) return;
    for (int s = 0; s < WEX_WIN_MAX_SLOTS; s++) {
        if (io->events[s]) CloseHandle(io->events[s]);
    }
    if (io->sync_event) CloseHandle(io->sync_event);
    wex_io_common_free_aligned(io->bounce_buf);
    DeleteCriticalSection(&io->sync_cs);
    for (int r = 0; r < WEX_IO_MAX_REPLICAS; r++) {
        if (io->handles[r] != INVALID_HANDLE_VALUE) {
            CloseHandle(io->handles[r]);
        }
    }
    free(io);
}

#else
typedef int wex_io_win_unused_t;
#endif /* _WIN32 */
