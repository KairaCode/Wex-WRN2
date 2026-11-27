/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef LIBWEX_WEX_IO_H
#define LIBWEX_WEX_IO_H

/* Internal single-NVMe-first async I/O & streaming SHA-256 contract.
 * Implemented per platform by:
 *   - wex_io_win.c   (Windows OVERLAPPED async batch + 16 MiB cap #3139 + FLS)
 *   - wex_io_linux.c (Linux io_uring + O_DIRECT with pread fallback)
 *   - wex_io_posix.c (macOS F_NOCACHE + portable POSIX pread fallback)
 *   - wex_verify.c   (2 MiB chunked streaming SHA-256 overlap) */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WEX_IO_MAX_REPLICAS 16
#define WEX_IO_ALIGN        4096u
#define WEX_IO_SHA_CHUNK    (2u << 20)   /* 2 MiB streaming SHA-256 chunk */

typedef struct wex_io wex_io_t;

/* Callback invoked on chunk progress during streaming reads (optional).
 * `off` = total contiguous bytes valid in `buf` so far (multiple of 4096). */
typedef void (*wex_io_chunk_fn)(void *ctx, int req_idx, const uint8_t *buf,
                                uint64_t chunk_off, uint64_t chunk_len, int done);

typedef struct {
    uint64_t file_offset;    /* 4096-aligned offset in container */
    uint64_t length;         /* 4096-aligned byte count to read */
    uint8_t *buf;            /* 4096-aligned destination buffer */
    int      hint_replica;   /* deterministic hint (0 for single-NVMe) */
    int      stream_chunks;  /* 1 = invoke chunk_cb every WEX_IO_SHA_CHUNK bytes */
    int      status;         /* output: 0 = OK, <0 = error */
    int      used_replica;   /* output: replica index that served this request */
} wex_io_req_t;

/* Aligned allocation helpers (4096-byte page aligned, portable Windows/POSIX) */
void *wex_io_alloc_aligned(size_t bytes);
void  wex_io_free_aligned(void *p);

/* Available system RAM in bytes (0 if unknown), used for L2 auto-sizing */
uint64_t wex_io_available_ram(void);

/* Monotonic wall clock in seconds */
double wex_io_now_s(void);

/* Open primary file + optional verified mirror paths (mirrors/num_mirrors may be NULL/0).
 * Validates mirror file size and 4096-byte header page identity against primary.
 * direct_mode: -1 = auto, 0 = buffered, 1 = O_DIRECT / FILE_FLAG_NO_BUFFERING.
 * weights: optional array of (1 + num_mirrors) positive weights (NULL = equal/dynamic). */
wex_io_t *wex_io_open(const char *primary_path,
                      const char *const *mirrors, int num_mirrors,
                      const uint32_t *weights,
                      int direct_mode, int queue_depth,
                      char *err_buf, size_t err_cap);

const char *wex_io_engine_name(const wex_io_t *io);
int         wex_io_replica_count(const wex_io_t *io);
uint64_t    wex_io_file_size(const wex_io_t *io);
int         wex_io_is_direct(const wex_io_t *io);

/* Read exact byte range synchronously from primary (used for header/index/model blobs). */
int wex_io_read_exact(wex_io_t *io, uint64_t offset, void *buf, size_t len);

/* Execute a batch of n page-aligned record reads using the platform's optimal async
 * mechanism. On a single NVMe, issues up to queue_depth concurrent transfers and
 * invokes optional `on_complete(ctx, req_idx)` as soon as each request lands
 * (first-finished-first-decoded). Optional `on_chunk` fires per 2 MiB chunk when
 * reqs[i].stream_chunks != 0. Returns count of failed requests (0 = all OK). */
typedef void (*wex_io_done_fn)(void *ctx, int req_idx, int status);

int wex_io_read_batch(wex_io_t *io, wex_io_req_t *reqs, int n,
                      wex_io_chunk_fn on_chunk, wex_io_done_fn on_done, void *ctx);

void wex_io_close(wex_io_t *io);

/* ---- Streaming chunked SHA-256 verifier (wex_verify.c) --------------------- */
typedef struct wex_sha_batch wex_sha_batch_t;

/* Create a batch verifier for n items; expected_sha[i] may be NULL (skip verify)
 * or point to the 32-byte expected SHA-256 digest of item i. */
wex_sha_batch_t *wex_sha_batch_begin(int n, const uint8_t *const *expected_sha, int workers);

/* Feed chunk [chunk_off, chunk_off + chunk_len) for item req_idx (called from I/O
 * thread as chunks land; queues async incremental SHA-256 on worker threads). */
void wex_sha_batch_feed(wex_sha_batch_t *b, int req_idx, const uint8_t *buf,
                        uint64_t chunk_off, uint64_t chunk_len, int is_last);

/* Wait for all fed items to finish hashing and write per-item result into ok_out[i]
 * (1 = matched or skipped, 0 = mismatch). Returns cumulative hash seconds. */
double wex_sha_batch_finish(wex_sha_batch_t *b, int *ok_out);

#ifdef __cplusplus
}
#endif

#endif /* LIBWEX_WEX_IO_H */
