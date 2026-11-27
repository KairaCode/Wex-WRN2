/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

#include "wex_cache.h"
#include "wex_io.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef SRWLOCK wex_lock_t;
static inline void wex_lock_init(wex_lock_t *l)    { InitializeSRWLock(l); }
static inline void wex_lock_destroy(wex_lock_t *l) { (void)l; }
static inline void wex_lock_acquire(wex_lock_t *l) { AcquireSRWLockExclusive(l); }
static inline void wex_lock_release(wex_lock_t *l) { ReleaseSRWLockExclusive(l); }
#else
#include <pthread.h>
typedef pthread_mutex_t wex_lock_t;
static inline void wex_lock_init(wex_lock_t *l)    { pthread_mutex_init(l, NULL); }
static inline void wex_lock_destroy(wex_lock_t *l) { pthread_mutex_destroy(l); }
static inline void wex_lock_acquire(wex_lock_t *l) { pthread_mutex_lock(l); }
static inline void wex_lock_release(wex_lock_t *l) { pthread_mutex_unlock(l); }
#endif

#if defined(__GNUC__) || defined(__clang__)
static inline uint64_t wex_a_load64(const uint64_t *p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
static inline void     wex_a_store64(uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }
#elif defined(_MSC_VER)
#include <intrin.h>
static inline uint64_t wex_a_load64(const uint64_t *p) {
    return (uint64_t)InterlockedCompareExchange64((volatile LONG64 *)p, 0, 0);
}
static inline void wex_a_store64(uint64_t *p, uint64_t v) {
    InterlockedExchange64((volatile LONG64 *)p, (LONG64)v);
}
#else
static inline uint64_t wex_a_load64(const uint64_t *p) { return *p; }
static inline void     wex_a_store64(uint64_t *p, uint64_t v) { *p = v; }
#endif

#define WEX_CACHE_MAX_SHARDS 16u
#define WEX_CACHE_INIT_BUCKETS 32u

static inline size_t wex_cache_align_up(size_t n) {
    return (n + (WEX_IO_ALIGN - 1u)) & ~(size_t)(WEX_IO_ALIGN - 1u);
}

static inline uint32_t wex_cache_hash(uint32_t key) {
    return key * UINT32_C(2654435761);
}

static inline uint32_t wex_cache_next_pow2(uint32_t v) {
    uint32_t p = WEX_CACHE_INIT_BUCKETS;
    while (p < v && p < (1u << 24)) p <<= 1;
    return p;
}

typedef struct wex_cache_entry {
    uint32_t key;
    uint32_t refcount;
    uint8_t *data;
    size_t   len;        /* stored record bytes */
    size_t   cap;        /* allocated buffer capacity (arena_slot_bytes or aligned dynamic) */
    int      in_table;   /* 1 if currently indexed in hash + LRU */
    struct wex_cache_entry *hash_next;
    struct wex_cache_entry *lru_prev;
    struct wex_cache_entry *lru_next;
} wex_cache_entry_t;

typedef struct wex_cache_shard {
    wex_lock_t lock;
    uint64_t   cap_bytes;
    uint64_t   used_bytes;     /* updated under lock, also read atomically for stats */
    uint64_t   pinned_bytes;   /* sum of len for entries with refcount > 0 */
    uint64_t   entries;        /* updated under lock, also read atomically for stats */

    wex_cache_entry_t **buckets;
    uint32_t            nbuckets;
    wex_cache_entry_t  *lru_head; /* MRU */
    wex_cache_entry_t  *lru_tail; /* LRU (evict from here) */

    /* Fixed-slot arena pool (only when arena != NULL) */
    wex_cache_entry_t  *pool;
    wex_cache_entry_t  *free_list;
    size_t              pool_count;
} wex_cache_shard_t;

struct wex_cache {
    uint32_t           num_shards;
    uint64_t           capacity_bytes;
    size_t             slot_bytes;    /* > 0 in fixed-slot arena mode, 0 in dynamic mode */
    uint8_t           *arena;
    size_t             arena_bytes;
    int                is_pinned;
    wex_cache_unpin_fn unpin_fn;
    wex_cache_shard_t  shards[WEX_CACHE_MAX_SHARDS];
};

static inline uint32_t wex_cache_shard_idx(const wex_cache_t *c, uint32_t key) {
    if (c->num_shards == 1u) return 0u;
    if (c->num_shards == WEX_CACHE_MAX_SHARDS) return (wex_cache_hash(key) >> 28) & 15u;
    return (wex_cache_hash(key) >> 16) % c->num_shards;
}

static inline uint32_t wex_cache_bucket_idx(uint32_t key, uint32_t nbuckets) {
    return (wex_cache_hash(key) >> 4) & (nbuckets - 1u);
}

static void wex_lru_unlink(wex_cache_shard_t *sh, wex_cache_entry_t *e) {
    if (e->lru_prev) e->lru_prev->lru_next = e->lru_next;
    else sh->lru_head = e->lru_next;
    if (e->lru_next) e->lru_next->lru_prev = e->lru_prev;
    else sh->lru_tail = e->lru_prev;
    e->lru_prev = NULL;
    e->lru_next = NULL;
}

static void wex_lru_push_front(wex_cache_shard_t *sh, wex_cache_entry_t *e) {
    e->lru_prev = NULL;
    e->lru_next = sh->lru_head;
    if (sh->lru_head) sh->lru_head->lru_prev = e;
    else sh->lru_tail = e;
    sh->lru_head = e;
}

static void wex_lru_touch(wex_cache_shard_t *sh, wex_cache_entry_t *e) {
    if (sh->lru_head == e) return;
    wex_lru_unlink(sh, e);
    wex_lru_push_front(sh, e);
}

static wex_cache_entry_t *wex_hash_find(wex_cache_shard_t *sh, uint32_t key) {
    uint32_t b = wex_cache_bucket_idx(key, sh->nbuckets);
    wex_cache_entry_t *e = sh->buckets[b];
    while (e) {
        if (e->key == key) return e;
        e = e->hash_next;
    }
    return NULL;
}

static void wex_hash_remove(wex_cache_shard_t *sh, wex_cache_entry_t *target) {
    uint32_t b = wex_cache_bucket_idx(target->key, sh->nbuckets);
    wex_cache_entry_t **pp = &sh->buckets[b];
    while (*pp) {
        if (*pp == target) {
            *pp = target->hash_next;
            target->hash_next = NULL;
            target->in_table = 0;
            return;
        }
        pp = &(*pp)->hash_next;
    }
}

static void wex_hash_insert(wex_cache_shard_t *sh, wex_cache_entry_t *e) {
    uint32_t b = wex_cache_bucket_idx(e->key, sh->nbuckets);
    e->hash_next = sh->buckets[b];
    sh->buckets[b] = e;
    e->in_table = 1;
}

static void wex_hash_maybe_grow(wex_cache_shard_t *sh) {
    uint32_t new_n, i;
    wex_cache_entry_t **nb;
    if (sh->entries <= sh->nbuckets || sh->nbuckets >= (1u << 24)) return;
    new_n = sh->nbuckets << 1;
    nb = (wex_cache_entry_t **)calloc(new_n, sizeof(*nb));
    if (!nb) return;
    for (i = 0; i < sh->nbuckets; i++) {
        wex_cache_entry_t *e = sh->buckets[i];
        while (e) {
            wex_cache_entry_t *next = e->hash_next;
            uint32_t b = wex_cache_bucket_idx(e->key, new_n);
            e->hash_next = nb[b];
            nb[b] = e;
            e = next;
        }
    }
    free(sh->buckets);
    sh->buckets = nb;
    sh->nbuckets = new_n;
}

wex_cache_t *wex_cache_create(uint64_t capacity_bytes,
                              size_t arena_slot_bytes,
                              size_t min_rec_bytes,
                              wex_cache_pin_fn pin_fn,
                              wex_cache_unpin_fn unpin_fn) {
    wex_cache_t *c;
    uint32_t s;
    if (!capacity_bytes) return NULL;

    c = (wex_cache_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->unpin_fn = unpin_fn;

    if (arena_slot_bytes > 0) {
        size_t slot_sz = wex_cache_align_up(arena_slot_bytes);
        uint64_t total_slots = capacity_bytes / slot_sz;
        size_t arena_bytes;
        size_t assigned = 0;
        if (!total_slots) {
            free(c);
            return NULL;
        }
        /* Guard size_t multiplication */
        if (total_slots > (uint64_t)(SIZE_MAX / slot_sz)) {
            total_slots = (uint64_t)(SIZE_MAX / slot_sz);
        }
        arena_bytes = (size_t)total_slots * slot_sz;
        c->arena = (uint8_t *)wex_io_alloc_aligned(arena_bytes);
        if (!c->arena) {
            free(c);
            return NULL;
        }
        c->arena_bytes = arena_bytes;
        c->slot_bytes = slot_sz;
        c->capacity_bytes = (uint64_t)arena_bytes;

        if (pin_fn) {
            if (pin_fn(c->arena, arena_bytes) == 0) {
                c->is_pinned = 1;
            }
        }

        /* Use 16 shards once we have at least 32 slots; 1 shard for tiny caches for exact LRU */
        c->num_shards = (total_slots >= 32u) ? WEX_CACHE_MAX_SHARDS : 1u;

        for (s = 0; s < c->num_shards; s++) {
            wex_cache_shard_t *sh = &c->shards[s];
            size_t cnt = (size_t)(total_slots / c->num_shards) +
                         (s < (uint32_t)(total_slots % c->num_shards) ? 1u : 0u);
            uint32_t nb = wex_cache_next_pow2((uint32_t)(cnt * 2u));
            size_t k;
            wex_lock_init(&sh->lock);
            sh->cap_bytes = (uint64_t)cnt * (uint64_t)slot_sz;
            sh->pool_count = cnt;
            sh->nbuckets = nb;
            sh->buckets = (wex_cache_entry_t **)calloc(nb, sizeof(*sh->buckets));
            sh->pool = (wex_cache_entry_t *)calloc(cnt ? cnt : 1u, sizeof(*sh->pool));
            if (!sh->buckets || !sh->pool) {
                wex_cache_destroy(c);
                return NULL;
            }
            for (k = 0; k < cnt; k++) {
                wex_cache_entry_t *e = &sh->pool[k];
                e->data = c->arena + (assigned + k) * slot_sz;
                e->cap = slot_sz;
                e->hash_next = sh->free_list;
                sh->free_list = e;
            }
            assigned += cnt;
        }
    } else {
        /* Dynamic byte-budgeted mode */
        uint64_t est_slots = (min_rec_bytes > 0) ? (capacity_bytes / min_rec_bytes)
                                                 : (capacity_bytes / (1u << 20));
        c->num_shards = (est_slots >= 32u) ? WEX_CACHE_MAX_SHARDS : 1u;
        c->capacity_bytes = capacity_bytes;

        for (s = 0; s < c->num_shards; s++) {
            wex_cache_shard_t *sh = &c->shards[s];
            uint64_t share = capacity_bytes / c->num_shards +
                             (s < (uint32_t)(capacity_bytes % c->num_shards) ? 1u : 0u);
            wex_lock_init(&sh->lock);
            sh->cap_bytes = share;
            sh->nbuckets = WEX_CACHE_INIT_BUCKETS;
            sh->buckets = (wex_cache_entry_t **)calloc(sh->nbuckets, sizeof(*sh->buckets));
            if (!sh->buckets) {
                wex_cache_destroy(c);
                return NULL;
            }
        }
    }

    return c;
}

void wex_cache_destroy(wex_cache_t *c) {
    uint32_t s;
    if (!c) return;
    for (s = 0; s < c->num_shards; s++) {
        wex_cache_shard_t *sh = &c->shards[s];
        if (!c->arena) {
            wex_cache_entry_t *e = sh->lru_head;
            while (e) {
                wex_cache_entry_t *next = e->lru_next;
                if (e->data) wex_io_free_aligned(e->data);
                free(e);
                e = next;
            }
        }
        free(sh->pool);
        free(sh->buckets);
        wex_lock_destroy(&sh->lock);
    }
    if (c->arena) {
        if (c->is_pinned && c->unpin_fn) {
            (void)c->unpin_fn(c->arena);
        }
        wex_io_free_aligned(c->arena);
    }
    free(c);
}

int wex_cache_is_pinned(const wex_cache_t *c) {
    return c ? c->is_pinned : 0;
}

uint64_t wex_cache_capacity(const wex_cache_t *c) {
    return c ? c->capacity_bytes : 0u;
}

int wex_cache_contains(wex_cache_t *c, uint32_t rec_idx) {
    wex_cache_shard_t *sh;
    int found;
    if (!c) return 0;
    sh = &c->shards[wex_cache_shard_idx(c, rec_idx)];
    wex_lock_acquire(&sh->lock);
    found = wex_hash_find(sh, rec_idx) != NULL;
    wex_lock_release(&sh->lock);
    return found;
}

int wex_cache_get(wex_cache_t *c, uint32_t rec_idx,
                  const uint8_t **buf_out, size_t *len_out,
                  wex_cache_handle_t *handle_out) {
    uint32_t sidx;
    wex_cache_shard_t *sh;
    wex_cache_entry_t *e;
    if (!c || !buf_out || !len_out || !handle_out) return 0;
    sidx = wex_cache_shard_idx(c, rec_idx);
    sh = &c->shards[sidx];

    wex_lock_acquire(&sh->lock);
    e = wex_hash_find(sh, rec_idx);
    if (!e) {
        wex_lock_release(&sh->lock);
        handle_out->shard = 0;
        handle_out->valid = 0;
        handle_out->entry = NULL;
        return 0;
    }
    if (e->refcount == 0u) {
        sh->pinned_bytes += (uint64_t)e->len;
    }
    e->refcount++;
    wex_lru_touch(sh, e);
    *buf_out = e->data;
    *len_out = e->len;
    handle_out->shard = sidx;
    handle_out->valid = 1u;
    handle_out->entry = e;
    wex_lock_release(&sh->lock);
    return 1;
}

void wex_cache_release(wex_cache_t *c, wex_cache_handle_t handle) {
    wex_cache_shard_t *sh;
    wex_cache_entry_t *e;
    if (!c || !handle.valid || !handle.entry || handle.shard >= c->num_shards) return;
    sh = &c->shards[handle.shard];
    e = (wex_cache_entry_t *)handle.entry;

    wex_lock_acquire(&sh->lock);
    if (e->refcount > 0u) {
        e->refcount--;
        if (e->refcount == 0u) {
            if (sh->pinned_bytes >= (uint64_t)e->len) sh->pinned_bytes -= (uint64_t)e->len;
            else sh->pinned_bytes = 0u;
        }
    }
    wex_lock_release(&sh->lock);
}

int wex_cache_put(wex_cache_t *c, uint32_t rec_idx, const uint8_t *data, size_t len) {
    uint32_t sidx;
    wex_cache_shard_t *sh;
    wex_cache_entry_t *existing;
    if (!c || !data || !len) return 0;
    if (c->slot_bytes > 0 && len > c->slot_bytes) return 0;

    sidx = wex_cache_shard_idx(c, rec_idx);
    sh = &c->shards[sidx];
    if ((uint64_t)len > sh->cap_bytes) return 0;

    wex_lock_acquire(&sh->lock);
    existing = wex_hash_find(sh, rec_idx);
    if (existing) {
        /* Immutable compressed record already cached (or actively borrowed): touch LRU and return */
        if (existing->refcount > 0u || existing->len == len) {
            wex_lru_touch(sh, existing);
            wex_lock_release(&sh->lock);
            return 1;
        }
        /* Unpinned entry with changed length: remove old and fall through to insert */
        wex_hash_remove(sh, existing);
        wex_lru_unlink(sh, existing);
        wex_a_store64(&sh->used_bytes, sh->used_bytes - (uint64_t)existing->len);
        wex_a_store64(&sh->entries, sh->entries - 1u);
        if (c->arena) {
            existing->len = 0;
            existing->hash_next = sh->free_list;
            sh->free_list = existing;
        } else {
            wex_io_free_aligned(existing->data);
            free(existing);
        }
    }

    if (c->arena) {
        /* Fixed-slot arena mode */
        wex_cache_entry_t *slot = sh->free_list;
        if (slot) {
            sh->free_list = slot->hash_next;
            slot->hash_next = NULL;
        } else {
            /* Evict oldest unpinned entry from LRU tail */
            wex_cache_entry_t *vic = sh->lru_tail;
            while (vic && vic->refcount > 0u) vic = vic->lru_prev;
            if (!vic) {
                wex_lock_release(&sh->lock);
                return 0;
            }
            wex_hash_remove(sh, vic);
            wex_lru_unlink(sh, vic);
            wex_a_store64(&sh->used_bytes, sh->used_bytes - (uint64_t)vic->len);
            wex_a_store64(&sh->entries, sh->entries - 1u);
            slot = vic;
        }
        slot->key = rec_idx;
        slot->refcount = 0;
        slot->len = len;
        memcpy(slot->data, data, len);
        wex_hash_insert(sh, slot);
        wex_lru_push_front(sh, slot);
        wex_a_store64(&sh->used_bytes, sh->used_bytes + (uint64_t)len);
        wex_a_store64(&sh->entries, sh->entries + 1u);
        wex_lock_release(&sh->lock);
        return 1;
    } else {
        /* Dynamic byte-budgeted mode */
        wex_cache_entry_t *reuse = NULL;
        wex_cache_entry_t *e;
        size_t aligned_len = wex_cache_align_up(len);
        if (sh->pinned_bytes + (uint64_t)len > sh->cap_bytes) {
            wex_lock_release(&sh->lock);
            return 0;
        }
        while (sh->used_bytes + (uint64_t)len > sh->cap_bytes) {
            wex_cache_entry_t *vic = sh->lru_tail;
            while (vic && vic->refcount > 0u) vic = vic->lru_prev;
            if (!vic) {
                if (reuse) {
                    wex_io_free_aligned(reuse->data);
                    free(reuse);
                }
                wex_lock_release(&sh->lock);
                return 0;
            }
            wex_hash_remove(sh, vic);
            wex_lru_unlink(sh, vic);
            wex_a_store64(&sh->used_bytes, sh->used_bytes - (uint64_t)vic->len);
            wex_a_store64(&sh->entries, sh->entries - 1u);
            if (!reuse && vic->cap >= aligned_len &&
                vic->cap <= aligned_len + (aligned_len >> 1) + 262144u) {
                reuse = vic;
            } else {
                wex_io_free_aligned(vic->data);
                free(vic);
            }
        }
        if (reuse) {
            e = reuse;
        } else {
            e = (wex_cache_entry_t *)calloc(1, sizeof(*e));
            if (!e) {
                wex_lock_release(&sh->lock);
                return 0;
            }
            e->data = (uint8_t *)wex_io_alloc_aligned(aligned_len);
            if (!e->data) {
                free(e);
                wex_lock_release(&sh->lock);
                return 0;
            }
            e->cap = aligned_len;
        }
        e->key = rec_idx;
        e->refcount = 0;
        e->len = len;
        e->hash_next = NULL;
        e->lru_prev = NULL;
        e->lru_next = NULL;
        memcpy(e->data, data, len);
        wex_hash_insert(sh, e);
        wex_lru_push_front(sh, e);
        wex_a_store64(&sh->used_bytes, sh->used_bytes + (uint64_t)len);
        wex_a_store64(&sh->entries, sh->entries + 1u);
        wex_hash_maybe_grow(sh);
        wex_lock_release(&sh->lock);
        return 1;
    }
}

void wex_cache_stats(const wex_cache_t *c, uint64_t *used_bytes, uint64_t *entries) {
    uint64_t u = 0, n = 0;
    uint32_t s;
    if (c) {
        for (s = 0; s < c->num_shards; s++) {
            u += wex_a_load64(&c->shards[s].used_bytes);
            n += wex_a_load64(&c->shards[s].entries);
        }
    }
    if (used_bytes) *used_bytes = u;
    if (entries) *entries = n;
}
