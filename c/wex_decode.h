/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef COLI_WEX_DECODE_H
#define COLI_WEX_DECODE_H

/*
 * WEX record reconstruction, shared by the engine's expert store and the WEX
 * backend. Kept apart from expert_store_wex.c so that anything linking the
 * expert-store unit (the engine, the native tests) needs only this object,
 * not the backend's store and registry dependencies.
 */

#include <stddef.h>
#include <stdint.h>

#include "wex_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Neural records may use the optional sibling kaira_wex_cuda.dll on Windows.
 * COLI_WEX_CUDA_LIB selects an absolute helper path; "0" forces CPU.
 * Both paths return the same canonical host slab. Failures retain the input.
 * On the CPU a WNR3 record goes to the fast decoder of wex_fast_decode.h (AVX2
 * where CPUID says the CPU has it, NEON on ARM64) unless COLI_WEX_FAST_DECODE=0;
 * to the reference decoder of wex_neural_rans.h then, on other CPUs and for WNR2
 * records. Same bytes, same verdicts.
 */
int coli_wex_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity);
uint64_t coli_wex_cuda_decode_count(void);
/* The CPU decoder of WNR3 records now: "AVX2" or "NEON" (the fast decoder), or
 * "reference" (COLI_WEX_FAST_DECODE=0, or no kernel for this CPU). And the WNR3
 * records the fast decoder has decoded so far. */
const char *coli_wex_cpu_decoder(void);
uint64_t coli_wex_fast_decode_count(void);

/* Direct to VRAM (helper feature, optional): reconstruct a Neural-rANS record
 * held in host memory (entry->total_bytes) into caller-owned device buffers,
 * matrix m's scales to dst_scale[m] and weights to dst_weight[m]. Returns 0,
 * 1 when the helper/feature/CUDA is unavailable, -1 for invalid input; the
 * destination is untouched unless it returns 0. */
int coli_wex_cuda_device_available(void);
int coli_wex_cuda_decode_to_device(const WexEntry *entry, const unsigned char *record, int device,
                                   unsigned char *const dst_scale[3], unsigned char *const dst_weight[3]);
/* Page-lock a staging buffer through the helper's runtime (0 = registered). */
int coli_wex_cuda_host_register(void *pointer, size_t bytes);
int coli_wex_cuda_host_unregister(void *pointer);
uint64_t coli_wex_cuda_device_decode_count(void);

/* WNR2 records name a container-level model by id. A container registers its
 * model blob at open (process-wide, kept until exit; registering the same
 * model again is a no-op); the CPU decoder and the CUDA helper both use it.
 * Returns 0 and the model id, or -1 for an invalid blob or a full registry. */
int coli_wex_register_model(const unsigned char *blob, size_t bytes, uint32_t *id);
/* 1 when the CUDA helper decodes WNR2 records (direct to VRAM included). */
int coli_wex_cuda_nr2_available(void);
/* 1 when it also decodes WNR3 records (codec 5, helper feature 16). WNR3
 * records go to the batch API only when this is 1. */
int coli_wex_cuda_nr3_available(void);
/* Every container open path calls this once: 1 (and a stderr line, prefixed
 * by `who`) when a canonical-zero (header flag 64) container must be refused
 * because DSV4_CUDA_MOE_MASKED != 0 selects the masked DeepGEMM MoE path, which
 * is not proven value-exact; otherwise 0 (logging "value-exact" for flag 64). */
int coli_wex_container_refused(uint32_t header_flags, const char *who);
/* Batched direct-to-VRAM WNR2 decode (helper feature 8): n records, record
 * i's matrix m to dst_scale[3*i+m] / dst_weight[3*i+m], one host wait for the
 * whole batch. status[i]: 0 written, -1 invalid (untouched), 1 unavailable
 * (correct data or untouched). Returns the count of non-zero statuses; when
 * the helper lacks the feature every status is 1 and the caller falls back to
 * coli_wex_cuda_decode_to_device per record. */
int coli_wex_cuda_batch_available(void);
int coli_wex_cuda_decode_batch_to_device(int n, const WexEntry *const *entries,
                                         const unsigned char *const *records, int device,
                                         unsigned char *const *dst_scale, unsigned char *const *dst_weight,
                                         int *status);

/* Batched decode to host (helper feature 32): n WNR2/WNR3 records decoded on the
 * GPU, record i's canonical raw slab written to the host buffer dst[i] (the layout
 * coli_wex_decode_record leaves at the slab start; dst[i] may overlap record i
 * itself -- the in-place decode -- never another record). status[i]: 0 written, -1
 * invalid record (dst[i] untouched), 1 unavailable (dst[i] untouched, except after a
 * CUDA failure during the copy back itself, which may leave part of a correct slab);
 * on a non-zero status the caller treats the slot as empty and decodes that record
 * again on its own. Returns the count of non-zero statuses (every one 1 when the
 * helper lacks the feature). */
int coli_wex_cuda_batch_host_available(void);
int coli_wex_cuda_decode_batch_to_host(int n, const WexEntry *const *entries,
                                       const unsigned char *const *records, int device,
                                       unsigned char *const *dst, int *status);

/* Batched decode to host scratch buffers (helper feature 256; bits 64 and 128 are
 * reserved for WNR4): like coli_wex_cuda_decode_batch_to_host, but on error
 * (status[i] != 0) dst[i] may be overwritten, and dst[i] must not overlap any
 * records[j]. Avoids the second host sync when the helper exports feature 256;
 * gracefully falls back to coli_wex_cuda_decode_batch_to_host on older helpers. */
#define COLI_WEX_CUDA_BATCH_HOST_SCRATCH 256u
int coli_wex_cuda_batch_host_scratch_available(void);
int coli_wex_cuda_decode_batch_to_host_scratch(int n, const WexEntry *const *entries,
                                               const unsigned char *const *records, int device,
                                               unsigned char *const *dst, int *status);

#ifdef __cplusplus
}
#endif

#endif /* COLI_WEX_DECODE_H */
