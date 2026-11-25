/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/qwen38_core.h, c/st.h and c/compat.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
#ifndef WEX_FUZZ_ENGINE_Q38_H
#define WEX_FUZZ_ENGINE_Q38_H
/* STAND-IN, for wex4/fuzz/fuzz_q38_wex.c only. c/qwen38_wex.h is not a header that stands alone: the engine
 * includes it once from qwen38_core.h, in the middle of qwen38.c, and it uses the engine's model, its expert
 * slots, its safetensors reader (st.h) and its compat.h. This file declares what c/qwen38_wex.h names.
 *
 * Checked against colibri dev 98327913f980514970173b6df7bf67210ad0dc16 and against the open colibri pull request
 * #2825 (refs/pull/2825/head 5e0a0b5b2170ceeff3e46578f0003846eee031f9), which adds MXFP4 experts read from a
 * sidecar to qwen38_core.h. No public tree has c/qwen38_wex.h or the members it adds to the expert index. Three
 * kinds of declaration below:
 *  - COPIED, with the line they come from: st_tensor (st.h), and what c/qwen38_wex.h takes from compat.h
 *    (Windows: pread, posix_memalign, compat_aligned_free, compat_fsize, compat_pread_lasterr; macOS:
 *    compat_open_direct; elsewhere: compat_aligned_free). With -DWEX_FUZZ_COLIBRI and the c/ folder of a
 *    colibri checkout on the include path (run.sh, WEX_FUZZ_COLIBRI) the harness includes colibri's compat.h
 *    and st.h instead, and the Quark functions of st.h are the real ones;
 *  - ALIGNED: Cfg, Q38WeightKind, Q38Weight, Slot, Model and the expert index Q38Int4Experts. The members
 *    c/qwen38_wex.h uses that colibri has are declared with colibri's types and in colibri's order; the
 *    members it does not use are left out (Slot keeps all of its members); q38_weight_free and q38_name do
 *    what colibri's do with the members declared here;
 *  - INFERRED, in no public tree: the expert index's mx and wex_* members, the Q38Weight member e8, the value
 *    of Q38_WEIGHT_MXFP4, q38_slab_unpin. Their types come from how c/qwen38_wex.h uses them; where that use
 *    leaves a choice (a width, a signedness, a const) the comment says so, and a difference there from the
 *    engine's type is one this harness cannot see. Pull request #2825 holds the scales of an MXFP4 weight in
 *    the member x2q (const uint8_t *, qwen38_core.h:130 there), where c/qwen38_wex.h writes e8: the two
 *    trees name that member differently.
 *
 * What is real: every line of c/qwen38_wex.h, c/wex_decode.c, c/wex_format.h and c/wex4.h. The Quark path
 * (Q38_MXFP4_DIR) is not a container and is not fuzzed: its stand-ins below stop the process. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <malloc.h>
#else
#include <strings.h>
#include <unistd.h>
#endif

#ifdef WEX_FUZZ_COLIBRI
/* <>: from the include path (the tree's c/), never the stand-in compat.h next to this file */
#include <compat.h>
#include <st.h>
#else
/* ---- COPIED from colibri c/compat.h (dev 98327913) ---- */
#ifdef _WIN32
/* compat.h:89-91 */
#if !defined(_FILE_OFFSET_BITS) || _FILE_OFFSET_BITS < 64
#error "_FILE_OFFSET_BITS=64 required on Windows (add -D_FILE_OFFSET_BITS=64 to CFLAGS)"
#endif
/* compat.h:143: the GetLastError of a failed read, per thread */
static __thread DWORD compat_pread_lasterr __attribute__((unused));
/* compat.h:150-182: a positional read through ReadFile with an OVERLAPPED and an event of this call, on a
 * synchronous or an overlapped handle; past the end of the file it returns the bytes read (0 if none) */
static inline ssize_t compat_pread(int fd, void *buf, size_t n, off_t off){
    intptr_t osfh = _get_osfhandle(fd);
    if(osfh == -1 || osfh == -2){ errno = EBADF; return -1; }
    HANDLE h = (HANDLE)osfh;
    HANDLE done = CreateEventA(NULL, TRUE, FALSE, NULL);
    if(!done){ compat_pread_lasterr = GetLastError(); errno = EIO; return -1; }
    size_t total = 0;
    while(total < n){
        size_t chunk = n - total;
        DWORD chunk32 = (chunk > 0x7FFFFFFF) ? 0x7FFFFFFF : (DWORD)chunk;
        OVERLAPPED ov = {0};
        ov.Offset     = (DWORD)( (off + (off_t)total)        & 0xFFFFFFFFULL);
        ov.OffsetHigh = (DWORD)(((off + (off_t)total) >> 32) & 0xFFFFFFFFULL);
        ov.hEvent     = done;
        DWORD rd = 0;
        BOOL ok = ReadFile(h, (char*)buf + total, chunk32, &rd, &ov);
        if(!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(h, &ov, &rd, TRUE);
        if(!ok){
            DWORD err = GetLastError();
            if(err == ERROR_HANDLE_EOF) break;
            compat_pread_lasterr = err;
            if(err == ERROR_INVALID_HANDLE || err == ERROR_INVALID_FUNCTION) errno = EBADF;
            else errno = EIO;
            CloseHandle(done);
            return -1;
        }
        total += rd;
        if(rd == 0 || rd < chunk32) break;
    }
    CloseHandle(done);
    return (ssize_t)total;
}
#define pread(fd,buf,n,off) compat_pread(fd,buf,n,off)
/* compat.h:212-220 */
static inline int compat_posix_memalign(void **memptr, size_t alignment, size_t size){
    if(alignment < sizeof(void*)) alignment = sizeof(void*);
    *memptr = _aligned_malloc(size, alignment);
    return *memptr ? 0 : ENOMEM;
}
#define posix_memalign(memptr,alignment,size) compat_posix_memalign(memptr,alignment,size)
#define compat_aligned_free _aligned_free
/* compat.h:330-336 */
static inline off_t compat_fsize(int fd){
    intptr_t osfh = _get_osfhandle(fd);
    if(osfh == -1 || osfh == -2) return -1;
    LARGE_INTEGER li;
    if(!GetFileSizeEx((HANDLE)osfh, &li)) return -1;
    return (off_t)li.QuadPart;
}
#endif /* _WIN32 */
#ifdef __APPLE__
/* compat.h:59-63: macOS has no O_DIRECT; F_NOCACHE on the descriptor bypasses the unified buffer cache, and the
 * caller's 4 KiB-aligned reads stay valid. Issue #22 asked for this definition: the stand-in of 2026-11-14 did
 * the same (it also checked that F_NOCACHE is defined) */
static inline int compat_open_direct(const char *path){
    int fd = open(path, O_RDONLY);
    if(fd>=0) fcntl(fd, F_NOCACHE, 1);
    return fd;
}
#endif
/* compat.h:420-424 */
#ifndef compat_aligned_free
#define compat_aligned_free free
#endif

/* ---- COPIED from colibri c/st.h (dev 98327913) ---- */
/* st.h:30 and 32-41 */
#define ST_MAX_RANK 8
typedef struct {
    char   *name;
    int     fd;
    int64_t off;
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 3=U8/I8 4=F8_E4M3 5=F8_E8M0 6=I64 7=U32/I32 */
    int64_t numel;
    int     rank;
    int64_t shape[ST_MAX_RANK];
} st_tensor;
/* st.h:43-74 is the reader of a set of shards; c/qwen38_wex.h only passes it to the functions below */
typedef struct { int unused; } shards;
/* st.h:963, 990 and 1291-1293. The Quark shards of Q38_MXFP4_DIR are not a container: not fuzzed, so these
 * stop the process (colibri's read the shards; its st_read_range_raw_cap stops the engine on a range outside
 * the shard or the destination, and with drop = 1 tells the kernel to drop the pages it read) */
static void st_init(shards *S, const char *snap_dir) { (void)S; (void)snap_dir; abort(); }
static st_tensor *st_find(shards *S, const char *name) { (void)S; (void)name; return NULL; }
static void st_read_range_raw_cap(shards *S, int fd, int64_t off, int64_t nbytes, void *out, int64_t cap, int drop,
                                  const char *tag) {
    (void)S; (void)fd; (void)off; (void)nbytes; (void)out; (void)cap; (void)drop; (void)tag;
    abort();
}
static int64_t st_dtype_esz(int d) { return d == 2 ? 4 : (d == 0 || d == 1) ? 2 : 1; }
#endif /* WEX_FUZZ_COLIBRI */

/* ---- ALIGNED with colibri c/qwen38_core.h (dev 98327913) ---- */
/* qwen38_core.h:48-68: the four members c/qwen38_wex.h reads, all int there, in that order (colibri's Cfg has
 * 48 more) */
typedef struct {
    int hidden, layers, experts, inter;
} Cfg;

/* qwen38_core.h:70-81. Q38_WEIGHT_MXFP4 is INFERRED: dev has no such kind; pull request #2825 appends it after
 * Q38_WEIGHT_VQ1, as here (its qwen38_core.h:81). Until 2026-11-15 this enum was {F32 = 0, MXFP4} */
typedef enum {
    Q38_WEIGHT_NONE = 0,
    Q38_WEIGHT_F32,
    Q38_WEIGHT_BF16,
    Q38_WEIGHT_FP8,
    Q38_WEIGHT_INT4G64,
    Q38_WEIGHT_GGML,
    Q38_WEIGHT_INT2G32,
    Q38_WEIGHT_LOWBIT16,
    Q38_WEIGHT_VQ2,
    Q38_WEIGHT_VQ1,
    Q38_WEIGHT_MXFP4
} Q38WeightKind;

/* qwen38_core.h:112-135, its first nine members (colibri's has 30). e8 is INFERRED: c/qwen38_wex.h stores
 * there a pointer into the record (one E8M0 exponent per row and 32 inputs); a const uint8_t * like x2q, the
 * member pull request #2825 uses for the same bytes */
typedef struct {
    void *data;
    float *scales;
    int rows, cols;
    int64_t elements, scale_count;
    Q38WeightKind kind;
    unsigned owns_data:1, owns_scales:1;
    const uint8_t *e8;
} Q38Weight;

/* qwen38_core.h:182-191, every member */
typedef struct {
    int eid;
    Q38Weight gate, up, down;
    uint64_t used;
    void *fp8_slab;
    int64_t fp8_slab_bytes;
    void *int4_slab;
    int64_t int4_slab_bytes;
    int8_t *t8; float *t8s;
} Slot;

/* qwen38_core.h:202-214: S (line 203) and record_bytes (line 207) are colibri's; colibri's other members are
 * left out (c/qwen38_wex.h does not use them). The rest is INFERRED from c/qwen38_wex.h. Pull request #2825 has
 * none of these members: it marks an MXFP4 sidecar by the member mx of its Q38Lowbit (qwen38_core.h:2271
 * there), and keeps its replicas in mir_n and mir_cut (lines 215-216 there), where c/qwen38_wex.h has wex_n and
 * wex_cut */
typedef struct {
    shards S;
    int64_t record_bytes;
    int64_t code_bytes[3], scale_count[3], scale_off;
    int mx;                                   /* 1: MXFP4 records; any integer type would do */
    int64_t mx_raw, mx_scale_off[3], mx_weight_off[3];   /* int64_t: passed as int64_t * to q38_mx4_layout */
    int wex4;                                 /* int: passed as int * to q38_wex_manifest */
    int nr3, wex_n, wex_stripe;               /* any integer type would do */
    int wex_fd[4];                            /* descriptors; 4: the primary and up to three mirrors */
    int wex_cut[4];                           /* int: passed as const int * to q38_wex_stripe_cut */
    uint32_t wex_flags;                       /* the header flags; any type that holds 32 bits would do */
    WexEntry *wex_entry;                      /* not const: compat_aligned_free frees it */
    unsigned char *wex_seen;                  /* calloc'd, atomically or'ed */
    unsigned char *wex_sha;                   /* points into the table; a const would do too */
} Q38Int4Experts;

typedef struct {
    int fd, dfd, scale_dtype;
    int64_t size;
    int64_t *off;
} Q38Fp8Wex;

/* qwen38_core.h:241-335: c (line 242), prefix (244), x4 (253), range_begin and range_end (276), mtp_prefix
 * (307); colibri's Model has many more members */
typedef struct {
    Cfg c;
    char prefix[32];
    int native_fp8;
    Q38Int4Experts *x4;
    Q38Fp8Wex *fw;
    int range_begin, range_end;
    char mtp_prefix[32];
} Model;

#define Q38_INT4_FORMAT "colibri.qwen38.experts-int4g64"
static int q38_expert_int4_wanted(void) {
    const char *e = getenv("Q38_EXPERT_INT4");
    return (e && !strcmp(e, "1")) ? 1 : 0;
}
static int64_t q38_int4_row_bytes(int cols) { return ((int64_t)cols + 1) / 2; }
static int64_t q38_int4_groups(int cols) { return ((int64_t)cols + 63) / 64; }
static int64_t fp8_nblk(int64_t n) { return (n + 127) / 128; }
static void q38_bind_int4_record(Model *m, Slot *slot, uint8_t *record) { (void)m; (void)slot; (void)record; }

/* qwen38_core.h:396-407, for the members declared here: what the weight owns is freed, then it is cleared.
 * The MXFP4 weights c/qwen38_wex.h binds own nothing */
static void q38_weight_free(Q38Weight *weight) {
    if(!weight)return;
    if(weight->owns_data)free(weight->data);
    if(weight->owns_scales)free(weight->scales);
    memset(weight,0,sizeof(*weight));
}
/* INFERRED: in no public tree. The engine unpins a record slab that its CUDA tier pinned before it resizes it;
 * nothing is pinned here */
static void q38_slab_unpin(Slot *slot) { (void)slot; }
/* qwen38_core.h:1462-1465 (the Quark path only) */
static void q38_name(Model *m,char *out,size_t cap,int layer,const char *suffix) {
    if(layer==m->c.layers&&m->mtp_prefix[0]) snprintf(out,cap,"%s.layers.0.%s",m->mtp_prefix,suffix);
    else snprintf(out,cap,"%s.layers.%d.%s",m->prefix,layer,suffix);
}

#endif /* WEX_FUZZ_ENGINE_Q38_H */
