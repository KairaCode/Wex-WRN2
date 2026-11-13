/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/deepseek_v4_internal.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: KAIRA src/colibri/c/deepseek_v4_internal.h at commit 88ddb07 (2026-11-07), as committed.
 * Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
 * Declarations of the WEX engine hooks. expert_store_wex.h includes deepseek_v4_internal.h.
 */

/* ==== deepseek_v4_internal.h:1250-1285  WEX direct-to-VRAM, store-side and RAM-cache prototypes ==== */
/* WEX direct-to-VRAM (COLI_WEX_GPU_DIRECT=1, Windows): a VRAM-first routed
 * expert whose WEX record is Neural-rANS and that the RAM cache does not hold is
 * read compressed and reconstructed by kaira_wex_cuda.dll straight into its fp4
 * mirror. 0: the view carries only device handles, like a VRAM hit; non-zero:
 * not applicable or failed, the caller takes the ordinary host lookup. */
int coli_v4_gpu_expert_attach_wex(ColiExpertStore *store, ColiExpertView *view);
/* The prefetcher's twin: same load, no routing credit (it is a guess). */
int coli_v4_gpu_expert_prefetch_wex(ColiExpertStore *store, ColiExpertView *view);
/* COLI_WEX_GPU_DIRECT=2: the next-layer prefetcher uses the direct load too. */
int coli_v4_gpu_wex_direct_prefetch(int rank);
/* A decode MoE layer finished computing: mirrors handed out before become
 * recyclable by WEX direct loads (which write outside the backend's stream). */
void coli_v4_gpu_mirror_step(ColiExpertStore *store);
#ifdef _WIN32
#include "wex_format.h"
#include "wex_decode.h"
const WexEntry *coli_v4_expert_store_wex_compressed(ColiExpertStore *store, ColiExpertKey key,
                                                    unsigned char *dst, size_t capacity);
/* the record when it is Neural-rANS, fits capacity and RAM does not hold it */
const WexEntry *coli_v4_expert_store_wex_entry(ColiExpertStore *store, ColiExpertKey key,
                                               size_t capacity);
/* read that record's compressed bytes into dst (4096-aligned); 0 on success */
int coli_v4_expert_store_wex_read(ColiExpertStore *store, ColiExpertKey key,
                                  const WexEntry *entry, unsigned char *dst);
void coli_v4_wex_direct_note(int ok, uint64_t bytes);
/* compressed record cache, the RAM tier of the direct path (COLI_WEX_RAM_CACHE_GB) */
int coli_v4_wex_crc_get(ColiExpertStore *store, ColiExpertKey key, const WexEntry *entry,
                        unsigned char **record, int *hit);
void coli_v4_wex_crc_done(ColiExpertStore *store, int slot, int filled);
/* transient Windows flock failure (ERROR_FILE_NOT_FOUND on remote NTFS,
 * allocation-starved machines): one retry after a short wait. WEX_READ_RETRY
 * is the env knob (0/1, default 0 = upstream exact); declared here so both
 * readers (expert_store_wex.c and the direct path in deepseek_v4.c) share it. */
int coli_v4_wex_flock_retry(void);
int coli_v4_wex_crc_prewarm(ColiExpertStore *store, const uint64_t *usage, const uint8_t *vram_bits);
#endif

