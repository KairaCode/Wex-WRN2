/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* v4_wex_crc_policy.h -- which slot of deepseek_v4.c's RAM cache of compressed WEX
 * records (V4WexCrc, COLI_WEX_RAM_CACHE_GB) a hit refreshes and a miss takes.
 * Portable on purpose: the cache itself sits in the Windows-only WEX direct path,
 * this part is unit-tested everywhere (tests/test_v4_wex_crc_policy.c).
 *
 * COLI_WEX_RAM_CACHE_POLICY (2026-11-17):
 *   lru (default)  the cache as it was: every hit, demand or prefetch, restamps
 *                  the slot, and a miss evicts the oldest unleased stamp (an
 *                  O(slots) scan under the cache mutex, as before).
 *   clock          CLOCK with use counts, from TensorFold's "fluid subgraph
 *                  residency" recipe (Apache-2.0; the idea is ported, not its
 *                  code): a demand hit adds one, saturating at 3; a new record
 *                  starts at 0; a demand miss sweeps the hand, taking one off
 *                  every unleased count it passes, and evicts the first zero,
 *                  so a record used once leaves before one used again and a
 *                  one-shot scan (a prefill) cannot flush the reused ones.
 *                  Prefetch discipline comes with it: a prefetch hit is not a
 *                  use (no count), and a prefetch miss takes only a cold slot
 *                  (never used, emptied, or count 0; first from the hand, which
 *                  moves past it; no count spent) or none -- the reader then
 *                  stages the record outside the cache. A wrong guess stays
 *                  cold, and a guess never displaces a record demand reused.
 * Which slot holds what never changes a byte the decoder reads: the cache only
 * decides whether a record comes from RAM or from disk. */
#ifndef V4_WEX_CRC_POLICY_H
#define V4_WEX_CRC_POLICY_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* count: clock's use count (0..V4_WEX_CRC_COUNT_MAX) */
typedef struct { int cell; int refs; int ready; uint32_t bytes; uint64_t clock, sum; int count; } V4WexCrcSlot;

enum { V4_WEX_CRC_LRU = 0, V4_WEX_CRC_CLOCK = 1 };
/* who asks coli_v4_wex_crc_get: decode demand, the next-layer prefetcher, the tier prewarm
 * (a demand fill for the policy; told apart in COLI_WEX_RAM_CACHE_TRACE only) */
enum { V4_WEX_CRC_DEMAND = 0, V4_WEX_CRC_PREFETCH = 1, V4_WEX_CRC_PREWARM = 2 };
#define V4_WEX_CRC_COUNT_MAX 3

/* COLI_WEX_RAM_CACHE_POLICY: "clock" or lru (unset, empty, "lru"); anything else is lru
 * with a warning from the caller (*unknown = 1) */
static inline int v4_wex_crc_policy_parse(const char *value, int *unknown) {
    *unknown = 0;
    if (!value || !*value || !strcmp(value, "lru")) return V4_WEX_CRC_LRU;
    if (!strcmp(value, "clock")) return V4_WEX_CRC_CLOCK;
    *unknown = 1;
    return V4_WEX_CRC_LRU;
}

/* A hit on v (cache mutex held) */
static inline void v4_wex_crc_policy_hit(V4WexCrcSlot *v, uint64_t *clock, int policy, int kind) {
    if (policy == V4_WEX_CRC_CLOCK) {
        if (kind != V4_WEX_CRC_PREFETCH && v->count < V4_WEX_CRC_COUNT_MAX) v->count++;
    } else v->clock = ++*clock;
}

/* A record just claimed v (cache mutex held) */
static inline void v4_wex_crc_policy_fill(V4WexCrcSlot *v, uint64_t *clock) {
    v->clock = ++*clock;
    v->count = 0;
}

/* The slot a miss takes once every slot has held a record (cache mutex held), -1 if none:
 * every slot leased or, clock prefetch, none cold. *hand: clock's hand. */
static inline int v4_wex_crc_policy_victim(V4WexCrcSlot *slot, int slots, int *hand,
                                           int policy, int kind) {
    if (slots < 1) return -1;
    if (policy == V4_WEX_CRC_CLOCK) {
        if (kind == V4_WEX_CRC_PREFETCH) {
            /* the hand moves past the slot taken, nothing it skips loses a count: the guess
             * gets the turn a demand fill gets before the hand comes back, so the next demand
             * miss does not evict a right guess before its layer routes (2026-11-17: with the
             * hand left on it, tools/residency_sim.py wex-ram showed exactly that) */
            for (int k = 0; k < slots; k++) {
                int i = (*hand + k) % slots;
                const V4WexCrcSlot *v = &slot[i];
                if (!v->refs && (v->cell < 0 || !v->count)) {
                    *hand = i + 1 < slots ? i + 1 : 0;
                    return i;
                }
            }
            return -1;
        }
        /* every unleased count reaches 0 within COUNT_MAX + 1 turns of the hand */
        for (long k = 0; k < (long)slots * (V4_WEX_CRC_COUNT_MAX + 1); k++) {
            int i = *hand;
            V4WexCrcSlot *v = &slot[i];
            *hand = i + 1 < slots ? i + 1 : 0;
            if (v->refs) continue;
            if (v->cell < 0 || !v->count) return i;
            v->count--;
        }
        return -1;
    }
    int s = -1;
    for (int i = 0; i < slots; i++) {
        const V4WexCrcSlot *v = &slot[i];
        if (v->refs) continue;
        if (v->cell < 0) return i;
        if (s < 0 || v->clock < slot[s].clock) s = i;
    }
    return s;
}
#endif
