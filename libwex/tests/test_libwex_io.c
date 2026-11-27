/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */

/* Standalone unit + throughput test for libwex async I/O & streaming SHA-256. */

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#define TEST_GETPID() ((unsigned long)_getpid())
#else
#include <unistd.h>
#define TEST_GETPID() ((unsigned long)getpid())
#endif

#include "../src/wex_io.h"
#include "../src/wex_io_common.h"
#include "../../c/wex4.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
        return 1; \
    } \
} while (0)

static void fill_pattern(uint8_t *buf, size_t len, uint64_t seed) {
    uint64_t x = seed ^ UINT64_C(0x9E3779B97F4A7C15);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        uint64_t v = x * UINT64_C(0x2545F4914F6CDD1D);
        memcpy(buf + i, &v, 8);
    }
    for (; i < len; i++) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        buf[i] = (uint8_t)(x & 0xFFu);
    }
}

static int write_file_bytes(const char *path, const uint8_t *data, size_t len) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t off = 0;
    while (off < len) {
        size_t chunk = (len - off > (4u << 20)) ? (4u << 20) : (len - off);
        if (fwrite(data + off, 1, chunk, fp) != chunk) {
            fclose(fp);
            return -1;
        }
        off += chunk;
    }
    if (fflush(fp) != 0 || fclose(fp) != 0) return -1;
    return 0;
}

static void get_scratch_dir(char *out, size_t cap) {
#ifdef _WIN32
    DWORD n = GetTempPathA((DWORD)cap, out);
    if (n == 0 || n >= cap) {
        snprintf(out, cap, ".");
    } else if (out[n - 1] == '\\' || out[n - 1] == '/') {
        out[n - 1] = '\0';
    }
#else
    const char *d = getenv("TMPDIR");
    if (d && *d) {
        snprintf(out, cap, "%s", d);
    } else {
        snprintf(out, cap, ".");
    }
#endif
}

/* ---- Test 1: Utility helpers & replica picker ---------------------------- */
static int test_utils_and_picker(void) {
    void *p1 = wex_io_alloc_aligned(13);
    void *p2 = wex_io_alloc_aligned(WEX_IO_ALIGN * 7u);
    void *p3 = wex_io_alloc_aligned(0);
    CHECK(p1 && ((uintptr_t)p1 & (WEX_IO_ALIGN - 1u)) == 0, "p1 4096-aligned");
    CHECK(p2 && ((uintptr_t)p2 & (WEX_IO_ALIGN - 1u)) == 0, "p2 4096-aligned");
    CHECK(p3 && ((uintptr_t)p3 & (WEX_IO_ALIGN - 1u)) == 0, "p3 4096-aligned");
    memset(p2, 0x5A, WEX_IO_ALIGN * 7u);
    wex_io_free_aligned(p1);
    wex_io_free_aligned(p2);
    wex_io_free_aligned(p3);
    wex_io_free_aligned(NULL);

    uint64_t ram = wex_io_available_ram();
    CHECK(ram > 0, "available_ram > 0");
    double t0 = wex_io_now_s();
    double t1 = wex_io_now_s();
    CHECK(t1 >= t0, "monotonic clock non-decreasing");

    /* Picker: nrep == 1 fast path */
    wex_io_picker_t pk1;
    wex_io_picker_init(&pk1, 1, NULL);
    for (int i = 0; i < 16; i++) {
        int r = wex_io_pick_rep(&pk1, i, 4u << 20);
        CHECK(r == 0, "single-NVMe always picks replica 0");
        wex_io_unpick_rep(&pk1, r, 4u << 20);
    }

    /* Picker: nrep == 2 weighted 75:25 sequential */
    wex_io_picker_t pk2;
    uint32_t w2[2] = { 75, 25 };
    wex_io_picker_init(&pk2, 2, w2);
    int cnt2[2] = { 0, 0 };
    for (int i = 0; i < 100; i++) {
        int r = wex_io_pick_rep(&pk2, 0, 1u << 20);
        CHECK(r == 0 || r == 1, "valid replica index");
        cnt2[r]++;
        wex_io_unpick_rep(&pk2, r, 1u << 20);
    }
    CHECK(cnt2[0] >= 73 && cnt2[0] <= 77, "weighted 75:25 replica 0 share");
    CHECK(cnt2[1] >= 23 && cnt2[1] <= 27, "weighted 75:25 replica 1 share");

    /* Picker: nrep == 2 concurrent active queue steering */
    wex_io_picker_t pk3;
    uint32_t w3[2] = { 60, 40 };
    wex_io_picker_init(&pk3, 2, w3);
    int r0 = wex_io_pick_rep(&pk3, 0, 4u << 20);
    int r1 = wex_io_pick_rep(&pk3, 0, 4u << 20);
    CHECK(r0 == 0 && r1 == 1, "2 simultaneous requests spread across both drives");
    wex_io_unpick_rep(&pk3, r0, 4u << 20);
    wex_io_unpick_rep(&pk3, r1, 4u << 20);

    printf("[PASS] utils & picker (avail RAM = %.2f GiB, 75:25 split = %d:%d)\n",
           (double)ram / (1024.0 * 1024.0 * 1024.0), cnt2[0], cnt2[1]);
    return 0;
}

typedef struct {
    wex_sha_batch_t *sha;
    int              chunk_calls[32];
    uint64_t         chunk_next_off[32];
    int              chunk_saw_last[32];
    int              chunk_order_bad;
    int              done_calls[32];
    int              done_before_last;
    int              done_status[32];
} batch_cb_ctx_t;

static void test_on_chunk(void *vctx, int req_idx, const uint8_t *buf,
                          uint64_t chunk_off, uint64_t chunk_len, int done) {
    batch_cb_ctx_t *c = (batch_cb_ctx_t *)vctx;
    if (req_idx >= 0 && req_idx < 32) {
        if (chunk_off != c->chunk_next_off[req_idx]) {
            c->chunk_order_bad = 1;
        }
        c->chunk_next_off[req_idx] = chunk_off + chunk_len;
        c->chunk_calls[req_idx]++;
        if (done) c->chunk_saw_last[req_idx] = 1;
    }
    if (c->sha) {
        wex_sha_batch_feed(c->sha, req_idx, buf, chunk_off, chunk_len, done);
    }
}

static void test_on_done(void *vctx, int req_idx, int status) {
    batch_cb_ctx_t *c = (batch_cb_ctx_t *)vctx;
    if (req_idx >= 0 && req_idx < 32) {
        if (c->chunk_calls[req_idx] > 0 && !c->chunk_saw_last[req_idx]) {
            c->done_before_last = 1;
        }
        c->done_calls[req_idx]++;
        c->done_status[req_idx] = status;
    }
}

int main(void) {
    if (test_utils_and_picker() != 0) return 1;

    const size_t FILE_SIZE = 48u << 20; /* 48 MiB, 4096-aligned */
    uint8_t *ref = (uint8_t *)wex_io_alloc_aligned(FILE_SIZE);
    CHECK(ref != NULL, "alloc 48 MiB reference buffer");
    fill_pattern(ref, FILE_SIZE, UINT64_C(0x123456789ABCDEF0));

    char dir[512];
    get_scratch_dir(dir, sizeof(dir));
    unsigned long pid = TEST_GETPID();

    char p_primary[600], p_good_mirror[600], p_bad_hdr[600], p_bad_size[600];
    snprintf(p_primary,     sizeof(p_primary),     "%s/wex_io_test_%lu_primary.bin",  dir, pid);
    snprintf(p_good_mirror, sizeof(p_good_mirror), "%s/wex_io_test_%lu_mirror.bin",   dir, pid);
    snprintf(p_bad_hdr,     sizeof(p_bad_hdr),     "%s/wex_io_test_%lu_badhdr.bin",   dir, pid);
    snprintf(p_bad_size,    sizeof(p_bad_size),    "%s/wex_io_test_%lu_badsize.bin",  dir, pid);

    int rc = 1;
    wex_io_t *io1 = NULL;
    wex_io_t *io_buf = NULL;
    wex_io_t *io_mir = NULL;
    uint8_t *read_buf = NULL;
    uint8_t *unaligned_raw = NULL;

    if (write_file_bytes(p_primary, ref, FILE_SIZE) != 0) {
        fprintf(stderr, "FAIL: cannot write primary test file\n");
        goto cleanup;
    }
    if (write_file_bytes(p_good_mirror, ref, FILE_SIZE) != 0) {
        fprintf(stderr, "FAIL: cannot write good mirror test file\n");
        goto cleanup;
    }
    /* Bad header mirror: flip 1 bit inside first 4096-byte page */
    ref[4065] ^= 0x01u;
    if (write_file_bytes(p_bad_hdr, ref, FILE_SIZE) != 0) {
        fprintf(stderr, "FAIL: cannot write bad-header mirror\n");
        goto cleanup;
    }
    ref[4065] ^= 0x01u;
    /* Bad size mirror: truncated by 4096 bytes */
    if (write_file_bytes(p_bad_size, ref, FILE_SIZE - WEX_IO_ALIGN) != 0) {
        fprintf(stderr, "FAIL: cannot write bad-size mirror\n");
        goto cleanup;
    }

    char err[256] = { 0 };

    /* ---- Test 2: Single-drive open + metadata ---------------------------- */
    io1 = wex_io_open(p_primary, NULL, 0, NULL, 1, 0, err, sizeof(err));
    if (!io1) {
        fprintf(stderr, "FAIL: wex_io_open primary failed: %s\n", err);
        goto cleanup;
    }
    if (wex_io_replica_count(io1) != 1 || wex_io_file_size(io1) != FILE_SIZE) {
        fprintf(stderr, "FAIL: primary replica_count=%d size=%llu\n",
                wex_io_replica_count(io1), (unsigned long long)wex_io_file_size(io1));
        goto cleanup;
    }
    printf("[PASS] wex_io_open single-drive (engine=%s, direct=%d, size=%llu MiB)\n",
           wex_io_engine_name(io1), wex_io_is_direct(io1),
           (unsigned long long)(wex_io_file_size(io1) >> 20));

    /* ---- Test 3: Mirror validation rejects mismatched mirrors ------------ */
    const char *mirrors[3] = { p_bad_size, p_good_mirror, p_bad_hdr };
    uint32_t mweights[4] = { 60, 10, 40, 10 };
    io_mir = wex_io_open(p_primary, mirrors, 3, mweights, 1, 0, err, sizeof(err));
    if (!io_mir || wex_io_replica_count(io_mir) != 2) {
        fprintf(stderr, "FAIL: mirror validation expected 2 replicas, got %d (%s)\n",
                io_mir ? wex_io_replica_count(io_mir) : -1, err);
        goto cleanup;
    }
    printf("[PASS] mirror validation (accepted 1 identical mirror, rejected bad-size + bad-hdr)\n");

    /* ---- Test 4: wex_io_read_exact (aligned + unaligned + >16 MiB) ------- */
    read_buf = (uint8_t *)wex_io_alloc_aligned(FILE_SIZE);
    unaligned_raw = (uint8_t *)malloc(16384u);
    if (!read_buf || !unaligned_raw) {
        fprintf(stderr, "FAIL: OOM read buffers\n");
        goto cleanup;
    }

    /* 4a: aligned 4096 bytes at offset 0 */
    if (wex_io_read_exact(io1, 0, read_buf, 4096) != 0 ||
        memcmp(read_buf, ref, 4096) != 0) {
        fprintf(stderr, "FAIL: read_exact aligned 4096\n");
        goto cleanup;
    }
    /* 4b: tiny unaligned (off=7, len=13) under O_DIRECT / NO_BUFFERING */
    if (wex_io_read_exact(io1, 7, read_buf, 13) != 0 ||
        memcmp(read_buf, ref + 7, 13) != 0) {
        fprintf(stderr, "FAIL: read_exact unaligned off=7 len=13\n");
        goto cleanup;
    }
    /* 4c: multi-bounce unaligned (off=4090, len=100000) */
    if (wex_io_read_exact(io1, 4090, read_buf, 100000) != 0 ||
        memcmp(read_buf, ref + 4090, 100000) != 0) {
        fprintf(stderr, "FAIL: read_exact unaligned cross-bounce 100000B\n");
        goto cleanup;
    }
    /* 4d: tail near EOF (off=FILE_SIZE-137, len=137) */
    if (wex_io_read_exact(io1, FILE_SIZE - 137u, read_buf, 137u) != 0 ||
        memcmp(read_buf, ref + FILE_SIZE - 137u, 137u) != 0) {
        fprintf(stderr, "FAIL: read_exact tail 137B\n");
        goto cleanup;
    }
    /* 4e: unaligned destination buffer pointer with aligned offset & len */
    if (wex_io_read_exact(io1, 8192u, unaligned_raw + 3u, 8192u) != 0 ||
        memcmp(unaligned_raw + 3u, ref + 8192u, 8192u) != 0) {
        fprintf(stderr, "FAIL: read_exact unaligned dst pointer\n");
        goto cleanup;
    }
    /* 4f: > 16 MiB cap read (20 MiB aligned) */
    if (wex_io_read_exact(io1, 4096u, read_buf, 20u << 20) != 0 ||
        memcmp(read_buf, ref + 4096u, 20u << 20) != 0) {
        fprintf(stderr, "FAIL: read_exact 20 MiB (>16 MiB cap)\n");
        goto cleanup;
    }
    /* 4g: out-of-bounds rejected */
    if (wex_io_read_exact(io1, FILE_SIZE - 16u, read_buf, 32u) == 0) {
        fprintf(stderr, "FAIL: read_exact out-of-bounds should fail\n");
        goto cleanup;
    }
    /* 4h: unaligned file size (5000 bytes) under auto direct mode */
    {
        char p_small[600];
        snprintf(p_small, sizeof(p_small), "%s/wex_io_test_%lu_small.bin", dir, pid);
        if (write_file_bytes(p_small, ref, 5000u) == 0) {
            wex_io_t *io_sm = wex_io_open(p_small, NULL, 0, NULL, -1, 0, err, sizeof(err));
            int sm_ok = 0;
            if (io_sm && wex_io_file_size(io_sm) == 5000u) {
                memset(read_buf, 0, 5000u);
                if (wex_io_read_exact(io_sm, 0, read_buf, 5000u) == 0 &&
                    memcmp(read_buf, ref, 5000u) == 0 &&
                    wex_io_read_exact(io_sm, 4090u, read_buf, 910u) == 0 &&
                    memcmp(read_buf, ref + 4090u, 910u) == 0) {
                    sm_ok = 1;
                }
            }
            wex_io_close(io_sm);
            remove(p_small);
            if (!sm_ok) {
                fprintf(stderr, "FAIL: read_exact on unaligned 5000B file\n");
                goto cleanup;
            }
        }
    }
    printf("[PASS] wex_io_read_exact (aligned, unaligned, cross-bounce, EOF tail, 20 MiB, OOB, 5000B)\n");

    /* ---- Test 5: wex_io_read_batch whole-read + >16 MiB + dual mirror ---- */
    {
        wex_io_req_t reqs[12];
        batch_cb_ctx_t cb;
        memset(&cb, 0, sizeof(cb));
        memset(read_buf, 0, FILE_SIZE);
        for (int i = 0; i < 12; i++) {
            memset(&reqs[i], 0, sizeof(reqs[i]));
            reqs[i].file_offset = (uint64_t)i * (4u << 20);
            reqs[i].length      = (4u << 20);
            reqs[i].buf         = read_buf + (size_t)i * (4u << 20);
        }
        int bad = wex_io_read_batch(io1, reqs, 12, NULL, test_on_done, &cb);
        if (bad != 0 || memcmp(read_buf, ref, FILE_SIZE) != 0) {
            fprintf(stderr, "FAIL: whole-read batch failed (bad=%d)\n", bad);
            goto cleanup;
        }
        for (int i = 0; i < 12; i++) {
            if (cb.done_calls[i] != 1 || reqs[i].status != 0 || reqs[i].used_replica != 0) {
                fprintf(stderr, "FAIL: whole-read req[%d] state mismatch\n", i);
                goto cleanup;
            }
        }

        /* Single > 16 MiB record in batch (24 MiB) */
        wex_io_req_t big;
        memset(&big, 0, sizeof(big));
        memset(read_buf, 0, 24u << 20);
        big.file_offset = 4u << 20;
        big.length      = 24u << 20;
        big.buf         = read_buf;
        if (wex_io_read_batch(io1, &big, 1, NULL, NULL, NULL) != 0 ||
            memcmp(read_buf, ref + (4u << 20), 24u << 20) != 0) {
            fprintf(stderr, "FAIL: 24 MiB single batch read (>16 MiB cap)\n");
            goto cleanup;
        }

        /* Dual-mirror weighted distribution check (60:40) */
        int rep_cnt[2] = { 0, 0 };
        memset(read_buf, 0, FILE_SIZE);
        for (int i = 0; i < 12; i++) {
            reqs[i].status = 0;
            reqs[i].used_replica = -1;
        }
        if (wex_io_read_batch(io_mir, reqs, 12, NULL, NULL, NULL) != 0 ||
            memcmp(read_buf, ref, FILE_SIZE) != 0) {
            fprintf(stderr, "FAIL: dual-mirror batch read failed\n");
            goto cleanup;
        }
        for (int i = 0; i < 12; i++) {
            if (reqs[i].used_replica == 0 || reqs[i].used_replica == 1) {
                rep_cnt[reqs[i].used_replica]++;
            }
        }
        if (rep_cnt[0] == 0 || rep_cnt[1] == 0) {
            fprintf(stderr, "FAIL: dual-mirror did not distribute across both drives (%d:%d)\n",
                    rep_cnt[0], rep_cnt[1]);
            goto cleanup;
        }
        printf("[PASS] wex_io_read_batch whole-read + 24 MiB cap + dual-mirror (%d:%d split)\n",
               rep_cnt[0], rep_cnt[1]);
    }

    /* ---- Test 6: Buffered mode (exercises synchronous ReadFile path) ----- */
    io_buf = wex_io_open(p_primary, NULL, 0, NULL, 0, 4, err, sizeof(err));
    if (!io_buf || wex_io_is_direct(io_buf) != 0) {
        fprintf(stderr, "FAIL: buffered open failed\n");
        goto cleanup;
    }
    {
        wex_io_req_t reqs[8];
        memset(read_buf, 0, 16u << 20);
        for (int i = 0; i < 8; i++) {
            memset(&reqs[i], 0, sizeof(reqs[i]));
            reqs[i].file_offset = (uint64_t)i * (2u << 20);
            reqs[i].length      = (2u << 20);
            reqs[i].buf         = read_buf + (size_t)i * (2u << 20);
        }
        if (wex_io_read_batch(io_buf, reqs, 8, NULL, NULL, NULL) != 0 ||
            memcmp(read_buf, ref, 16u << 20) != 0) {
            fprintf(stderr, "FAIL: buffered batch read failed\n");
            goto cleanup;
        }
        printf("[PASS] buffered batch read (sync completion handling)\n");
    }

    /* ---- Test 7: Streaming 2 MiB chunks + concurrent SHA-256 verifier ---- */
    {
        /* Mixed sizes: 4 KiB, 2 MiB, 3 MiB, 4 MiB, 6 MiB, 20 MiB */
        const uint64_t lens[6] = {
            4096u, 2u << 20, 3u << 20, 4u << 20, 6u << 20, 20u << 20
        };
        uint64_t offs[6];
        uint8_t expected_digests[6][32];
        const uint8_t *exp_ptrs[6];
        wex_io_req_t reqs[6];

        uint64_t cur = 0;
        memset(read_buf, 0, FILE_SIZE);
        for (int i = 0; i < 6; i++) {
            offs[i] = cur;
            wex4_sha256(ref + cur, (size_t)lens[i], expected_digests[i]);
            exp_ptrs[i] = expected_digests[i];
            memset(&reqs[i], 0, sizeof(reqs[i]));
            reqs[i].file_offset   = cur;
            reqs[i].length        = lens[i];
            reqs[i].buf           = read_buf + cur;
            reqs[i].stream_chunks = 1;
            cur += lens[i];
        }

        /* 7a: All valid records match */
        wex_sha_batch_t *sb = wex_sha_batch_begin(6, exp_ptrs, 4);
        if (!sb) { fprintf(stderr, "FAIL: sha_batch_begin\n"); goto cleanup; }
        batch_cb_ctx_t cb;
        memset(&cb, 0, sizeof(cb));
        cb.sha = sb;

        int bad = wex_io_read_batch(io1, reqs, 6, test_on_chunk, test_on_done, &cb);
        int ok_out[6] = { 0 };
        double hash_s = wex_sha_batch_finish(sb, ok_out);
        if (bad != 0 || cb.chunk_order_bad || cb.done_before_last) {
            fprintf(stderr, "FAIL: streaming batch bad=%d order_bad=%d done_early=%d\n",
                    bad, cb.chunk_order_bad, cb.done_before_last);
            goto cleanup;
        }
        for (int i = 0; i < 6; i++) {
            int expected_chunks = (int)((lens[i] + WEX_IO_SHA_CHUNK - 1u) / WEX_IO_SHA_CHUNK);
            if (!ok_out[i] || cb.chunk_calls[i] != expected_chunks ||
                !cb.chunk_saw_last[i] || cb.done_calls[i] != 1) {
                fprintf(stderr, "FAIL: streaming item %d ok=%d chunks=%d/%d\n",
                        i, ok_out[i], cb.chunk_calls[i], expected_chunks);
                goto cleanup;
            }
        }

        /* 7b: 1-bit corruption detection + NULL skip entry */
        expected_digests[3][17] ^= 0x04u; /* flip 1 bit in item 3 expected digest */
        exp_ptrs[0] = NULL;               /* skip item 0 */
        wex_sha_batch_t *sb2 = wex_sha_batch_begin(6, exp_ptrs, 4);
        memset(&cb, 0, sizeof(cb));
        cb.sha = sb2;
        bad = wex_io_read_batch(io1, reqs, 6, test_on_chunk, test_on_done, &cb);
        memset(ok_out, 0, sizeof(ok_out));
        (void)wex_sha_batch_finish(sb2, ok_out);
        if (bad != 0 || ok_out[0] != 1 || ok_out[1] != 1 || ok_out[2] != 1 ||
            ok_out[3] != 0 || ok_out[4] != 1 || ok_out[5] != 1) {
            fprintf(stderr, "FAIL: 1-bit corrupt/skip detection got [%d,%d,%d,%d,%d,%d]\n",
                    ok_out[0], ok_out[1], ok_out[2], ok_out[3], ok_out[4], ok_out[5]);
            goto cleanup;
        }
        expected_digests[3][17] ^= 0x04u;
        exp_ptrs[0] = expected_digests[0];

        /* 7c: 1-bit corruption in actual data buffer caught on streaming feed */
        read_buf[offs[4] + (3u << 20) + 77u] ^= 0x80u;
        wex_sha_batch_t *sb3 = wex_sha_batch_begin(6, exp_ptrs, 2);
        for (int i = 0; i < 6; i++) {
            uint64_t pos = 0;
            while (pos < lens[i]) {
                uint64_t ch = (lens[i] - pos > WEX_IO_SHA_CHUNK) ? WEX_IO_SHA_CHUNK : (lens[i] - pos);
                wex_sha_batch_feed(sb3, i, read_buf + offs[i], pos, ch, (pos + ch == lens[i]) ? 1 : 0);
                pos += ch;
            }
        }
        memset(ok_out, 0, sizeof(ok_out));
        (void)wex_sha_batch_finish(sb3, ok_out);
        if (ok_out[4] != 0 || ok_out[0] != 1 || ok_out[5] != 1) {
            fprintf(stderr, "FAIL: 1-bit data corruption not caught (ok[4]=%d)\n", ok_out[4]);
            goto cleanup;
        }

        /* 7d: Out-of-order chunk feeding still hashes in strict offset order */
        wex_sha_batch_t *sb_ooo = wex_sha_batch_begin(1, &exp_ptrs[3], 2);
        /* lens[3] == 4 MiB: feed second 2 MiB chunk (is_last=1) BEFORE first 2 MiB chunk */
        wex_sha_batch_feed(sb_ooo, 0, ref + offs[3], WEX_IO_SHA_CHUNK, WEX_IO_SHA_CHUNK, 1);
        wex_sha_batch_feed(sb_ooo, 0, ref + offs[3], 0, WEX_IO_SHA_CHUNK, 0);
        int ok_ooo[1] = { 0 };
        (void)wex_sha_batch_finish(sb_ooo, ok_ooo);
        if (!ok_ooo[0]) {
            fprintf(stderr, "FAIL: out-of-order chunk staging failed\n");
            goto cleanup;
        }

        /* 7e: All-NULL fast path + unfed item safety (no deadlock) */
        const uint8_t *null_ptrs[4] = { NULL, NULL, NULL, NULL };
        wex_sha_batch_t *sb_noop = wex_sha_batch_begin(4, null_ptrs, 4);
        int ok4[4] = { 0 };
        double noop_s = wex_sha_batch_finish(sb_noop, ok4);
        if (noop_s != 0.0 || !ok4[0] || !ok4[1] || !ok4[2] || !ok4[3]) {
            fprintf(stderr, "FAIL: all-NULL sha fast path\n");
            goto cleanup;
        }
        wex_sha_batch_t *sb_unfed = wex_sha_batch_begin(2, exp_ptrs, 2);
        /* Only feed item 0, leave item 1 unfed (simulating failed read) */
        wex_sha_batch_feed(sb_unfed, 0, ref + offs[0], 0, lens[0], 1);
        int ok2[2] = { 1, 1 };
        (void)wex_sha_batch_finish(sb_unfed, ok2);
        if (ok2[0] != 1 || ok2[1] != 0) {
            fprintf(stderr, "FAIL: unfed item handling got [%d,%d]\n", ok2[0], ok2[1]);
            goto cleanup;
        }

        printf("[PASS] streaming 2 MiB SHA-256 overlap + 1-bit corruption + skip/unfed (hash_s=%.4fs)\n",
               hash_s);
    }

    /* ---- Test 8: Single-NVMe Async OVERLAPPED Throughput Benchmark ------- */
    {
        const int NREC = 12;
        const uint64_t REC_BYTES = 4u << 20; /* 12 x 4 MiB = 48 MiB */
        wex_io_req_t reqs[12];
        uint8_t shas[12][32];
        const uint8_t *sha_ptrs[12];
        for (int i = 0; i < NREC; i++) {
            wex4_sha256(ref + (size_t)i * REC_BYTES, (size_t)REC_BYTES, shas[i]);
            sha_ptrs[i] = shas[i];
            memset(&reqs[i], 0, sizeof(reqs[i]));
            reqs[i].file_offset = (uint64_t)i * REC_BYTES;
            reqs[i].length      = REC_BYTES;
            reqs[i].buf         = read_buf + (size_t)i * REC_BYTES;
        }

        /* 8a: Raw async OVERLAPPED throughput (3 passes) */
        const int PASSES = 3;
        double best_raw_mbs = 0.0;
        for (int p = 0; p < PASSES; p++) {
            for (int i = 0; i < NREC; i++) reqs[i].stream_chunks = 0;
            double t0 = wex_io_now_s();
            int bad = wex_io_read_batch(io1, reqs, NREC, NULL, NULL, NULL);
            double dt = wex_io_now_s() - t0;
            if (bad != 0) { fprintf(stderr, "FAIL: bench raw read\n"); goto cleanup; }
            double mbs = ((double)FILE_SIZE / (1024.0 * 1024.0)) / (dt > 1e-9 ? dt : 1e-9);
            if (mbs > best_raw_mbs) best_raw_mbs = mbs;
        }

        /* 8b: Streaming 2 MiB chunked async OVERLAPPED + live SHA-256 overlap */
        double best_sha_mbs = 0.0;
        double last_hash_s = 0.0;
        for (int p = 0; p < PASSES; p++) {
            for (int i = 0; i < NREC; i++) reqs[i].stream_chunks = 1;
            wex_sha_batch_t *sb = wex_sha_batch_begin(NREC, sha_ptrs, 4);
            batch_cb_ctx_t cb;
            memset(&cb, 0, sizeof(cb));
            cb.sha = sb;
            double t0 = wex_io_now_s();
            int bad = wex_io_read_batch(io1, reqs, NREC, test_on_chunk, test_on_done, &cb);
            int ok_out[12] = { 0 };
            last_hash_s = wex_sha_batch_finish(sb, ok_out);
            double dt = wex_io_now_s() - t0;
            if (bad != 0) { fprintf(stderr, "FAIL: bench sha read\n"); goto cleanup; }
            for (int i = 0; i < NREC; i++) {
                if (!ok_out[i]) { fprintf(stderr, "FAIL: bench sha verify %d\n", i); goto cleanup; }
            }
            double mbs = ((double)FILE_SIZE / (1024.0 * 1024.0)) / (dt > 1e-9 ? dt : 1e-9);
            if (mbs > best_sha_mbs) best_sha_mbs = mbs;
        }

        printf("[BENCH] Single-NVMe %s (direct=%d, 48 MiB = 12x4 MiB, qd=6):\n"
               "        Raw async OVERLAPPED:          %.1f MiB/s\n"
               "        Streaming 2 MiB + SHA-256:     %.1f MiB/s (cumul hash %.2f ms)\n",
               wex_io_engine_name(io1), wex_io_is_direct(io1),
               best_raw_mbs, best_sha_mbs, last_hash_s * 1e3);
    }

    printf("ALL TESTS PASSED!\n");
    rc = 0;

cleanup:
    wex_io_close(io1);
    wex_io_close(io_buf);
    wex_io_close(io_mir);
    wex_io_free_aligned(read_buf);
    wex_io_free_aligned(ref);
    free(unaligned_raw);
    remove(p_primary);
    remove(p_good_mirror);
    remove(p_bad_hdr);
    remove(p_bad_size);
    return rc;
}
