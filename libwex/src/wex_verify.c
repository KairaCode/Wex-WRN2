/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

/* Streaming chunked SHA-256 batch verifier overlapping SHA-NI / ARMv8 Crypto
 * incremental hashing with async 2 MiB NVMe chunk completions. */

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#elif defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#elif !defined(__FreeBSD__) && !defined(__OpenBSD__) && !defined(__NetBSD__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <pthread.h>
#endif

#include "wex_io_common.h"
#include "../../c/wex4.h"

#define WEX_SHA_MAX_WORKERS 16

typedef struct {
    uint64_t off;
    uint64_t len;
    int      is_last;
} sha_ooo_chunk_t;

typedef struct {
    const uint8_t   *expected;
    const uint8_t   *buf;
    Wex4Sha          sha;
    uint64_t         ready;
    uint64_t         hashed;
    int              got_last;
    int              order_err;
    int              in_worker;
    int              queued;
    int              done;
    int              ok;
    /* Optional small out-of-order staging list (zero allocations in normal ordered stream) */
    sha_ooo_chunk_t *ooo;
    int              ooo_count;
    int              ooo_cap;
} sha_item_t;

struct wex_sha_batch {
    int            n;
    int            active;       /* 0 when n == 0 or all expected_sha == NULL */
    int            nworkers;
    int            finishing;
    double         hash_seconds;
    sha_item_t    *items;

    /* Bounded ring queue of item indices ready for hashing (capacity n + 1) */
    int           *q;
    int            q_cap;
    int            q_head;
    int            q_tail;

#ifdef _WIN32
    CRITICAL_SECTION   mu;
    CONDITION_VARIABLE cv;
    HANDLE             threads[WEX_SHA_MAX_WORKERS];
#else
    pthread_mutex_t    mu;
    pthread_cond_t     cv;
    pthread_t          threads[WEX_SHA_MAX_WORKERS];
#endif
};

static inline void sha_lock(wex_sha_batch_t *b) {
#ifdef _WIN32
    EnterCriticalSection(&b->mu);
#else
    pthread_mutex_lock(&b->mu);
#endif
}

static inline void sha_unlock(wex_sha_batch_t *b) {
#ifdef _WIN32
    LeaveCriticalSection(&b->mu);
#else
    pthread_mutex_unlock(&b->mu);
#endif
}

static inline void sha_wait(wex_sha_batch_t *b) {
#ifdef _WIN32
    SleepConditionVariableCS(&b->cv, &b->mu, INFINITE);
#else
    pthread_cond_wait(&b->cv, &b->mu);
#endif
}

static inline void sha_signal(wex_sha_batch_t *b) {
#ifdef _WIN32
    WakeConditionVariable(&b->cv);
#else
    pthread_cond_signal(&b->cv);
#endif
}

static inline void sha_broadcast(wex_sha_batch_t *b) {
#ifdef _WIN32
    WakeAllConditionVariable(&b->cv);
#else
    pthread_cond_broadcast(&b->cv);
#endif
}

/* Caller must hold b->mu. */
static inline void sha_enqueue_locked(wex_sha_batch_t *b, int idx) {
    sha_item_t *it = &b->items[idx];
    if (it->queued || it->in_worker || it->done) return;
    b->q[b->q_tail] = idx;
    b->q_tail = (b->q_tail + 1) % b->q_cap;
    it->queued = 1;
    sha_signal(b);
}

/* Caller must hold b->mu. Drains any staged OOO chunks now contiguous with it->ready. */
static void sha_drain_ooo_locked(sha_item_t *it) {
    int progress = 1;
    while (progress && it->ooo_count > 0) {
        progress = 0;
        for (int k = 0; k < it->ooo_count; k++) {
            if (it->ooo[k].off == it->ready) {
                it->ready += it->ooo[k].len;
                if (it->ooo[k].is_last) it->got_last = 1;
                it->ooo[k] = it->ooo[it->ooo_count - 1];
                it->ooo_count--;
                progress = 1;
                break;
            }
        }
    }
}

/* Process available contiguous bytes for item `idx` (inline fallback or worker). */
static void sha_worker_loop(wex_sha_batch_t *b) {
    for (;;) {
        sha_lock(b);
        int idx = -1;
        while (idx < 0) {
            if (b->q_head != b->q_tail) {
                idx = b->q[b->q_head];
                b->q_head = (b->q_head + 1) % b->q_cap;
                sha_item_t *it = &b->items[idx];
                it->queued = 0;
                if (it->in_worker || it->done) {
                    idx = -1;
                    continue;
                }
                if (!(it->ready > it->hashed || it->got_last || it->order_err)) {
                    idx = -1;
                    continue;
                }
                it->in_worker = 1;
                break;
            }
            if (b->finishing) break;
            sha_wait(b);
        }

        if (idx < 0) {
            sha_unlock(b);
            break;
        }

        sha_item_t *it = &b->items[idx];
        const uint8_t *buf = it->buf;
        uint64_t start = it->hashed;
        uint64_t end = it->ready;
        int got_last = it->got_last;
        int order_err = it->order_err;
        int has_ooo = (it->ooo_count > 0);
        sha_unlock(b);

        /* Strictly ordered incremental SHA-256 over [start, end) without holding lock. */
        double t0 = wex_io_common_now_s();
        if (!order_err && end > start) {
            if (buf) {
                wex4_sha_update(&it->sha, buf + start, (size_t)(end - start));
            } else {
                order_err = 1;
            }
        }

        int finished_now = 0;
        int match_ok = 0;
        if (order_err) {
            finished_now = 1;
            match_ok = 0;
        } else if (got_last && !has_ooo) {
            unsigned char digest[32];
            wex4_sha_final(&it->sha, digest);
            match_ok = (memcmp(digest, it->expected, 32u) == 0) ? 1 : 0;
            finished_now = 1;
        }
        double dt = wex_io_common_now_s() - t0;

        sha_lock(b);
        b->hash_seconds += dt;
        if (!order_err) it->hashed = end;
        it->in_worker = 0;
        if (finished_now) {
            it->done = 1;
            it->ok = match_ok;
        } else if (it->ready > it->hashed || it->got_last || it->order_err) {
            sha_enqueue_locked(b, idx);
        }
        sha_unlock(b);
    }
}

#ifdef _WIN32
static DWORD WINAPI sha_thread_entry(LPVOID arg) {
    sha_worker_loop((wex_sha_batch_t *)arg);
    return 0;
}
#else
static void *sha_thread_entry(void *arg) {
    sha_worker_loop((wex_sha_batch_t *)arg);
    return NULL;
}
#endif

wex_sha_batch_t *wex_sha_batch_begin(int n, const uint8_t *const *expected_sha, int workers) {
    wex_sha_batch_t *b = (wex_sha_batch_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->n = (n > 0) ? n : 0;

    int need_verify = 0;
    if (b->n > 0 && expected_sha) {
        for (int i = 0; i < b->n; i++) {
            if (expected_sha[i]) need_verify++;
        }
    }

    /* Fast path: nothing to verify -> zero threads, zero locking overhead. */
    if (need_verify == 0) {
        b->active = 0;
        return b;
    }

    b->items = (sha_item_t *)calloc((size_t)b->n, sizeof(sha_item_t));
    b->q_cap = b->n + 1;
    b->q = (int *)malloc((size_t)b->q_cap * sizeof(int));
    if (!b->items || !b->q) {
        free(b->items);
        free(b->q);
        free(b);
        return NULL;
    }

    for (int i = 0; i < b->n; i++) {
        b->items[i].expected = expected_sha[i];
        if (expected_sha[i]) {
            wex4_sha_init(&b->items[i].sha);
        } else {
            b->items[i].done = 1;
            b->items[i].ok = 1;
        }
    }

    b->active = 1;
#ifdef _WIN32
    InitializeCriticalSection(&b->mu);
    InitializeConditionVariable(&b->cv);
#else
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
#endif

    int want_w = workers;
    if (want_w <= 0) want_w = 4;
    if (want_w > need_verify) want_w = need_verify;
    if (want_w > WEX_SHA_MAX_WORKERS) want_w = WEX_SHA_MAX_WORKERS;

    for (int w = 0; w < want_w; w++) {
#ifdef _WIN32
        HANDLE th = CreateThread(NULL, 0, sha_thread_entry, b, 0, NULL);
        if (th) {
            b->threads[b->nworkers++] = th;
        }
#else
        if (pthread_create(&b->threads[b->nworkers], NULL, sha_thread_entry, b) == 0) {
            b->nworkers++;
        }
#endif
    }

    return b;
}

void wex_sha_batch_feed(wex_sha_batch_t *b, int req_idx, const uint8_t *buf,
                        uint64_t chunk_off, uint64_t chunk_len, int is_last) {
    if (!b || !b->active || req_idx < 0 || req_idx >= b->n) return;
    sha_item_t *it = &b->items[req_idx];
    if (!it->expected) return;

    /* Inline fallback only if zero worker threads could be spawned */
    if (b->nworkers == 0) {
        sha_lock(b);
        if (!it->done) {
            if (!it->buf) it->buf = buf;
            if (chunk_off == it->ready && (!buf || buf == it->buf)) {
                double t0 = wex_io_common_now_s();
                if (chunk_len > 0) {
                    if (it->buf) {
                        wex4_sha_update(&it->sha, it->buf + chunk_off, (size_t)chunk_len);
                    } else {
                        it->order_err = 1;
                    }
                }
                it->ready += chunk_len;
                it->hashed = it->ready;
                if (is_last) it->got_last = 1;
                if (it->order_err) {
                    it->done = 1;
                    it->ok = 0;
                } else if (it->got_last) {
                    unsigned char digest[32];
                    wex4_sha_final(&it->sha, digest);
                    it->ok = (memcmp(digest, it->expected, 32u) == 0) ? 1 : 0;
                    it->done = 1;
                }
                b->hash_seconds += wex_io_common_now_s() - t0;
            } else {
                it->order_err = 1;
                it->done = 1;
                it->ok = 0;
            }
        }
        sha_unlock(b);
        return;
    }

    sha_lock(b);
    if (!it->done) {
        if (!it->buf) it->buf = buf;
        if (buf && it->buf && buf != it->buf) {
            it->order_err = 1;
        } else if (chunk_off == it->ready) {
            it->ready += chunk_len;
            if (is_last) it->got_last = 1;
            if (it->ooo_count > 0) sha_drain_ooo_locked(it);
        } else if (chunk_off > it->ready) {
            /* Stage out-of-order chunk until earlier bytes arrive so hashing stays strictly ordered */
            if (it->ooo_count == it->ooo_cap) {
                int new_cap = it->ooo_cap ? it->ooo_cap * 2 : 8;
                sha_ooo_chunk_t *grown = (sha_ooo_chunk_t *)realloc(
                    it->ooo, (size_t)new_cap * sizeof(sha_ooo_chunk_t));
                if (!grown) {
                    it->order_err = 1;
                } else {
                    it->ooo = grown;
                    it->ooo_cap = new_cap;
                }
            }
            if (!it->order_err) {
                it->ooo[it->ooo_count].off = chunk_off;
                it->ooo[it->ooo_count].len = chunk_len;
                it->ooo[it->ooo_count].is_last = is_last ? 1 : 0;
                it->ooo_count++;
            }
        } else {
            /* Overlapping/duplicate chunk offset */
            it->order_err = 1;
        }

        if (it->ready > it->hashed || it->got_last || it->order_err) {
            sha_enqueue_locked(b, req_idx);
        }
    }
    sha_unlock(b);
}

double wex_sha_batch_finish(wex_sha_batch_t *b, int *ok_out) {
    if (!b) return 0.0;

    if (!b->active) {
        if (ok_out) {
            for (int i = 0; i < b->n; i++) ok_out[i] = 1;
        }
        free(b);
        return 0.0;
    }

    sha_lock(b);
    b->finishing = 1;
    sha_broadcast(b);
    sha_unlock(b);

    for (int w = 0; w < b->nworkers; w++) {
#ifdef _WIN32
        WaitForSingleObject(b->threads[w], INFINITE);
        CloseHandle(b->threads[w]);
#else
        pthread_join(b->threads[w], NULL);
#endif
    }

    double secs = b->hash_seconds;
    if (ok_out) {
        for (int i = 0; i < b->n; i++) {
            if (!b->items[i].expected) {
                ok_out[i] = 1;
            } else if (b->items[i].done) {
                ok_out[i] = b->items[i].ok;
            } else {
                /* Never received is_last (e.g., failed I/O): mark unverified cleanly */
                ok_out[i] = 0;
            }
        }
    }

    for (int i = 0; i < b->n; i++) {
        free(b->items[i].ooo);
    }
#ifdef _WIN32
    DeleteCriticalSection(&b->mu);
#else
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
#endif
    free(b->items);
    free(b->q);
    free(b);
    return secs;
}
