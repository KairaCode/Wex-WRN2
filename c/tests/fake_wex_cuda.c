/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* CPU stand-in for kaira_wex_cuda.dll (tests/test_v4_wex_nr3.c, tests/test_qwen38_mxfp4.py):
 * helper ABI 1, features 1 (direct decode), 4 (WNR2 models), 16 (WNR3), 8 (batch decode
 * to device, with FAKE_WEX_BATCH_DEVICE=1), 32 (batch decode to host) and 256 (batch decode
 * to host scratch; bits 64 and 128 reserved for WNR4; FAKE_WEX_BATCH_FAIL=<i> reports record
 * i unavailable after scribbling over its dst, as a real half-written slab would). Its "device"
 * pointers are host memory. FAKE_WEX_UNAVAILABLE=1 makes the direct decode
 * report 1 (CUDA unavailable), as the real helper does without a device.
 * Built only for tests and the fuzz replay; never shipped. */
#include "../wex_format.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

#define FAKE_MAX_MODELS 16   /* as the helper's WEX_CUDA_NR2_MAX_MODELS */
static WexNr2Model models[FAKE_MAX_MODELS];
static int model_count;

static const WexNr2Model *fake_model(const unsigned char *record) {
    for (int i = 0; i < model_count; i++)
        if (models[i].id == wex_read_u32_le(record + 4u)) return &models[i];
    return NULL;
}

FAKE_EXPORT unsigned int wex_cuda_helper_abi(void) { return 1u; }
/* FAKE_WEX_BATCH_DEVICE=1 adds feature 8 (batch decode to "device") */
FAKE_EXPORT unsigned int wex_cuda_helper_features(void) {
    const char *batch = getenv("FAKE_WEX_BATCH_DEVICE");
    return 1u | 4u | 16u | 32u | 256u | (batch && *batch == '1' ? 8u : 0u);
}

/* The contract of the helper's wex_cuda_register_model (tools/wex_cuda_bridge.cu): 0 when registered, or
 * already registered with the same bytes; -1 for an invalid blob or another blob with the same id; 1 when the
 * table is full or memory is short. A process that opens the same container again registers its model again. */
FAKE_EXPORT int wex_cuda_register_model(const unsigned char *blob, size_t bytes) {
    WexNr2Model probe;
    unsigned char *copy;
    if (wex_nr2_model_open(blob, bytes, &probe)) return -1;
    for (int i = 0; i < model_count; i++)
        if (models[i].id == probe.id) return memcmp(models[i].blob, blob, WEX_NR2_MODEL_BYTES) ? -1 : 0;
    copy = model_count < FAKE_MAX_MODELS ? malloc(bytes) : NULL;
    if (!copy) return 1;
    memcpy(copy, blob, bytes);
    if (wex_nr2_model_open(copy, bytes, &models[model_count])) { free(copy); return -1; }
    model_count++;
    return 0;
}

FAKE_EXPORT int wex_cuda_decode_to_host(const WexEntry *entry, unsigned char *slab, size_t capacity) {
    size_t at = wex_record_read_offset(entry);
    const WexNr2Model *model = at && at < capacity ? fake_model(slab + at) : NULL;
    return model ? wex_decode_nr2_record(entry, slab, capacity, model) : -1;
}

FAKE_EXPORT int wex_cuda_decode_to_device(const WexEntry *entry, const unsigned char *record, int device,
                                          unsigned char *const scale[3], unsigned char *const weight[3]) {
    const char *off = getenv("FAKE_WEX_UNAVAILABLE");
    const WexNr2Model *model = fake_model(record);
    uint64_t raw, aligned;
    unsigned char *slab;
    int status;
    (void)device;
    if (off && *off == '1') return 1;
    if (!model || wex_entry_raw_sizes(entry, &raw, &aligned) || entry->total_bytes >= aligned) return -1;
    slab = malloc((size_t)aligned);
    if (!slab) return 1;
    memcpy(slab + (aligned - entry->total_bytes), record, entry->total_bytes);
    status = wex_decode_nr2_record(entry, slab, (size_t)aligned, model);
    for (int m = 0; !status && m < 3; m++) {   /* the destination is untouched unless 0 */
        memcpy(scale[m], slab + entry->scale_offsets[m],
               (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1]);
        memcpy(weight[m], slab + entry->weight_offsets[m],
               (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1]);
    }
    free(slab);
    return status;
}

FAKE_EXPORT int wex_cuda_decode_batch_to_device(int n, const WexEntry *const *entries,
                                                const unsigned char *const *records, int device,
                                                unsigned char *const *scale, unsigned char *const *weight,
                                                int *status) {
    int failed = 0;
    for (int i = 0; i < n; i++) {
        status[i] = wex_cuda_decode_to_device(entries[i], records[i], device, scale + 3 * i, weight + 3 * i);
        failed += status[i] != 0;
    }
    return failed;
}

FAKE_EXPORT int wex_cuda_decode_batch_to_host(int n, const WexEntry *const *entries, const unsigned char *const *records,
                                              int device, unsigned char *const *dst, int *status) {
    const char *fail = getenv("FAKE_WEX_BATCH_FAIL");
    int failed = 0;
    (void)device;
    for (int i = 0; i < n; i++) {
        const WexNr2Model *model = fake_model(records[i]);
        uint64_t raw, aligned;
        unsigned char *slab;
        status[i] = -1;
        if (!model || wex_entry_raw_sizes(entries[i], &raw, &aligned) || entries[i]->total_bytes >= aligned ||
            !(slab = malloc((size_t)aligned))) { failed++; continue; }
        memcpy(slab + (aligned - entries[i]->total_bytes), records[i], entries[i]->total_bytes);
        status[i] = wex_decode_nr2_record(entries[i], slab, (size_t)aligned, model) ? -1 : 0;
        if (!status[i]) memcpy(dst[i], slab, (size_t)raw);
        free(slab);
        if (fail && *fail && atoi(fail) == i) { memset(dst[i], 0x5a, (size_t)raw); status[i] = 1; }
        if (status[i]) failed++;
    }
    return failed;
}

FAKE_EXPORT int wex_cuda_decode_batch_to_host_scratch(int n, const WexEntry *const *entries,
                                                      const unsigned char *const *records, int device,
                                                      unsigned char *const *dst, int *status) {
    const char *fail = getenv("FAKE_WEX_BATCH_FAIL");
    int failed = 0;
    (void)device;
    if (n <= 0) return 0;
    if (!status) return n;
    for (int i = 0; i < n; i++) {
        const WexNr2Model *model = entries && records && records[i] ? fake_model(records[i]) : NULL;
        uint64_t raw = 0, aligned = 0;
        unsigned char *slab;
        status[i] = -1;
        if (!entries || !records || !dst || !entries[i] || !records[i] || !dst[i] ||
            wex_entry_raw_sizes(entries[i], &raw, &aligned) || entries[i]->total_bytes >= aligned) {
            failed++;
            continue;
        }
        if (!model || !(slab = malloc((size_t)aligned))) {
            memset(dst[i], 0x5a, (size_t)raw);
            status[i] = 1;
            failed++;
            continue;
        }
        memcpy(slab + (aligned - entries[i]->total_bytes), records[i], entries[i]->total_bytes);
        status[i] = wex_decode_nr2_record(entries[i], slab, (size_t)aligned, model) ? -1 : 0;
        if (!status[i]) memcpy(dst[i], slab, (size_t)raw);
        else memset(dst[i], 0x5a, (size_t)raw);   /* scratch contract: dst[i] may be overwritten on error */
        free(slab);
        if (fail && *fail && atoi(fail) == i) { memset(dst[i], 0x5a, (size_t)raw); status[i] = 1; }
        if (status[i]) failed++;
    }
    return failed;
}
