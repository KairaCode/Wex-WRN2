/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_WEX_CACHE_H
#define LIBWEX_WEX_CACHE_H

/* Internal L2 Compressed-RAM Sharded LRU Cache.
 * Stores 4096-aligned compressed on-disk records keyed by uint32_t record_index.
 * Supports:
 *   - 16-way sharded hash + doubly-linked LRU with per-shard locking.
 *   - Safe refcounted borrow (wex_cache_get / wex_cache_release): borrowed entries
 *     cannot be evicted or mutated until released.
 *   - Fixed-slot contiguous arena mode with optional GPU host_register pin_fn / unpin_fn
 *     for zero-bounce PCIe H2D transfers, OR dynamic byte-budgeted 4096-aligned mode. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wex_cache wex_cache_t;

typedef struct wex_cache_handle {
    uint32_t shard;
    uint32_t valid;
    void    *entry;
} wex_cache_handle_t;

typedef int (*wex_cache_pin_fn)(void *ptr, size_t bytes);
typedef int (*wex_cache_unpin_fn)(void *ptr);

/* Create L2 cache with `capacity_bytes` total budget.
 * - If arena_slot_bytes > 0: allocates one contiguous 4096-aligned arena of
 *   floor(capacity_bytes / align4096(arena_slot_bytes)) fixed slots; if pin_fn != NULL,
 *   registers the entire arena once (falls back to unpinned if pin_fn fails).
 * - If arena_slot_bytes == 0: dynamic byte-budgeted mode with 4096-aligned buffers
 *   and victim buffer reuse; min_rec_bytes hints shard count for tight caches.
 * Returns NULL if capacity_bytes == 0 or allocation fails. */
wex_cache_t *wex_cache_create(uint64_t capacity_bytes,
                              size_t arena_slot_bytes,
                              size_t min_rec_bytes,
                              wex_cache_pin_fn pin_fn,
                              wex_cache_unpin_fn unpin_fn);

void     wex_cache_destroy(wex_cache_t *c);
int      wex_cache_is_pinned(const wex_cache_t *c);
uint64_t wex_cache_capacity(const wex_cache_t *c);

/* Non-borrowing existence check (does not touch LRU). Returns 1 if resident, 0 else. */
int wex_cache_contains(wex_cache_t *c, uint32_t rec_idx);

/* Look up rec_idx and acquire a refcounted borrow (bumps LRU to MRU).
 * Returns 1 on hit (setting *buf_out, *len_out, *handle_out), 0 on miss. */
int wex_cache_get(wex_cache_t *c, uint32_t rec_idx,
                  const uint8_t **buf_out, size_t *len_out,
                  wex_cache_handle_t *handle_out);

/* Release a refcounted borrow obtained from wex_cache_get. */
void wex_cache_release(wex_cache_t *c, wex_cache_handle_t handle);

/* Insert or update rec_idx with `len` bytes from `data`.
 * Evicts oldest unpinned (refcount == 0) LRU entries in the shard if needed.
 * Returns 1 if cached (or already present), 0 if skipped (too large / all pinned). */
int wex_cache_put(wex_cache_t *c, uint32_t rec_idx, const uint8_t *data, size_t len);

/* Aggregate used bytes and resident entry count across all shards. */
void wex_cache_stats(const wex_cache_t *c, uint64_t *used_bytes, uint64_t *entries);

#ifdef __cplusplus
}
#endif

#endif /* LIBWEX_WEX_CACHE_H */
