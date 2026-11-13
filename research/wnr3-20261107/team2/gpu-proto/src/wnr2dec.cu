// WNR2-style weight-stream GPU decoder prototype (team2/gpu-proto). Compiled with NVRTC from CuPy.
// One thread per stream (2048 symbols = 64 blocks of 32), one CUDA block serves one matrix,
// the matrix type's CDF rows live in shared memory. Payload bytes come from global memory through a
// per-thread register window (aligned 32-bit __ldg loads, one word prefetched ahead): this is the
// "register prefetch" style, NOT KAIRA's cp.async shared ring.
//
// Template parameters:
//   ALPHA 16 = FP4 nibble alphabet (WNR2 today); 8 = magnitude alphabet with raw sign plane (variant D)
//   CTX   0 = 567-row table, row = (srel*9+pm)*9+mx               (KAIRA layout, 18,144 B)
//         1 = 259 reachable rows, row = srel*37 + tri(pm,mx)        (same model, compacted)
//         2 = 1036 rows: ((srel*2+bm7)*2+half)*37 + tri, bm7 bit decoded in-stream at each block start
//   ILV   0 = one rANS state; 1 = two states alternating per symbol (B1);
//         2 = two states, even blocks / odd blocks, steps interleaved (B2, two independent chains)
//   NT    threads per CUDA block
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
#define RL (1u << 23)

// effective SM clock probe: cycles (clock64) vs nanoseconds (%globaltimer) over a dependent loop
extern "C" __global__ void clkprobe(long long* o, int iters) {
    long long c0 = clock64();
    u64 t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    u32 v = threadIdx.x;
    for (int i = 0; i < iters; i++) v = v * 1664525u + 1013904223u;
    long long c1 = clock64();
    u64 t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) { o[0] = c1 - c0; o[1] = (long long)(t1 - t0); o[2] = v; }
}

__device__ __forceinline__ u32 bswap32(u32 v) { return __byte_perm(v, 0, 0x0123); }

// payload load flavour (compile-time -DLDM=...): 0 = __ldg (L1 cached), 1 = __ldcg (L2 only), 2 = __ldcs (streaming)
#ifndef LDM
#define LDM 0
#endif
__device__ __forceinline__ u32 ldp(const u32* p) {
#if LDM == 1
    return __ldcg(p);
#elif LDM == 2
    return __ldcs(p);
#else
    return __ldg(p);
#endif
}

// -DNOLB removes __launch_bounds__ (ptxas then picks registers freely; -maxrregcount applies)
#ifdef NOLB
#define LBND(nt)
#else
#define LBND(nt) __launch_bounds__(nt)
#endif

// output store mode for wdec (diagnostics): 0 = plain uint4 store (canonical layout), 1 = no stores
// (XOR of all output folded into one uint4 per stream), 2 = plain stores into a 1 MB sink,
// 3 = __stcs streaming (evict-first) stores (canonical layout, verifiable)
#ifndef OUTMODE
#define OUTMODE 0
#endif
#if OUTMODE == 1
#define OSTORE(o, i, v) do { uint4 _v = (v); xacc.x ^= _v.x; xacc.y ^= _v.y; xacc.z ^= _v.z; xacc.w ^= _v.w; } while (0)
#elif OUTMODE == 3
#define OSTORE(o, i, v) __stcs((o) + (i), (v))
#elif OUTMODE == 4
// stage 8 blocks (128 B = one full cache line) per lane in shared memory, then write the line at once
#define OSTORE(o, i, v) do { stg[(i) & 7] = (v); if (((i) & 7) == 7) { _Pragma("unroll") for (int _k = 0; _k < 8; _k++) (o)[(i) - 7 + _k] = stg[_k]; } } while (0)
#else
#define OSTORE(o, i, v) ((o)[(i)] = (v))
#endif

struct Win {
    u64 buf;      // unread bytes, most significant first
    int cnt;      // valid bytes in buf
    u32 nxt;      // next 4 stream bytes, already loaded (byte-swapped)
    const u32* wp;
};

__device__ __forceinline__ void win_init(Win& w, const u32* __restrict__ base, u32 off) {
    const u32* p = base + (off >> 2);
    u32 sk = off & 3u;
    u64 v = ((u64)bswap32(ldp(p)) << 32) | (u64)bswap32(ldp(p + 1));
    w.buf = v << (8 * sk);
    w.cnt = 8 - (int)sk;
    w.nxt = bswap32(ldp(p + 2));
    w.wp = p + 3;
}

__device__ __forceinline__ void win_refill(Win& w) {
    if (w.cnt <= 4) {
        w.buf |= (u64)w.nxt << (32 - 8 * w.cnt);
        w.cnt += 4;
        w.nxt = bswap32(ldp(w.wp));
        w.wp++;
    }
}

__device__ __forceinline__ u32 win_take4(Win& w) {
    u32 r = (u32)(w.buf >> 32);
    w.buf <<= 32;
    w.cnt -= 4;
    win_refill(w);
    return r;
}

// branch-free byte renormalisation: 0, 1 or 2 bytes
__device__ __forceinline__ void renorm(u32& x, Win& w) {
    u32 b0 = (u32)(w.buf >> 56);
    u32 b1 = (u32)(w.buf >> 48) & 255u;
    bool c1 = x < RL;
    u32 x1 = c1 ? ((x << 8) | b0) : x;
    bool c2 = x1 < RL;
    x = c2 ? ((x1 << 8) | b1) : x1;
    int k = (int)c1 + (int)c2;
    w.buf <<= (8 * k);
    w.cnt -= k;
    win_refill(w);
}

// rows hold cum[1..16] as u16 (cum[16] = 16384). Sign trick: per u16 lane (0x8000+slot) - cum has
// bit 15 set iff cum <= slot; no borrow crosses lanes because cum <= 0x4000.
__device__ __forceinline__ u32 dsym16(const u16* __restrict__ tab, u32 row, u32& x) {
    const uint4* r = reinterpret_cast<const uint4*>(tab + row * 16);
    uint4 a = r[0], b = r[1];
    u32 slot = x & 16383u;
    u32 t = (slot * 0x10001u) | 0x80008000u;
    const u32 M = 0x80008000u;
    u32 c = ((t - a.x) & M) | (((t - a.y) & M) >> 1) | (((t - a.z) & M) >> 2) | (((t - a.w) & M) >> 3) |
            (((t - b.x) & M) >> 4) | (((t - b.y) & M) >> 5) | (((t - b.z) & M) >> 6) | (((t - b.w) & M) >> 7);
    u32 s = __popc(c);
    u32 st = s ? (u32)tab[row * 16 + s - 1] : 0u;
    u32 en = tab[row * 16 + s];
    x = (en - st) * (x >> 14) + slot - st;
    return s;
}

__device__ __forceinline__ u32 dsym8(const u16* __restrict__ tab, u32 row, u32& x) {
    uint4 a = *reinterpret_cast<const uint4*>(tab + row * 8);
    u32 slot = x & 16383u;
    u32 t = (slot * 0x10001u) | 0x80008000u;
    const u32 M = 0x80008000u;
    u32 c = ((t - a.x) & M) | (((t - a.y) & M) >> 1) | (((t - a.z) & M) >> 2) | (((t - a.w) & M) >> 3);
    u32 s = __popc(c);
    u32 st = s ? (u32)tab[row * 8 + s - 1] : 0u;
    u32 en = tab[row * 8 + s];
    x = (en - st) * (x >> 14) + slot - st;
    return s;
}

__device__ __forceinline__ u32 dbit(u32 c1, u32& x) {
    u32 slot = x & 16383u;
    u32 bit = slot >= c1;
    u32 st = bit ? c1 : 0u;
    u32 f = bit ? 16384u - c1 : c1;
    x = f * (x >> 14) + slot - st;
    return bit;
}

template <int ALPHA>
__device__ __forceinline__ u32 dsym(const u16* __restrict__ tab, u32 row, u32& x) {
    if (ALPHA == 16) return dsym16(tab, row, x);
    return dsym8(tab, row, x);
}

template <int CTX>
__device__ __forceinline__ u32 rowof(u32 pm, u32 mx) {
    if (CTX == 0) return pm * 9 + mx;
    return ((mx * (mx + 1)) >> 1) + pm;
}

template <int CTX>
__device__ __forceinline__ u32 srow_base(u32 srel, u32 bit) {
    if (CTX == 0) return srel * 81;
    if (CTX == 1) return srel * 37;
    return (srel * 2 + bit) * 74;
}

template <int ALPHA, int CTX, int ILV, int NT>
__global__ void LBND(NT) wdec(const u32* __restrict__ pay, const u32* __restrict__ offs,
                                           const u8* __restrict__ scales, const u32* __restrict__ signs,
                                           const u16* __restrict__ tabs, const u16* __restrict__ bittabs,
                                           const u8* __restrict__ mtype, const u8* __restrict__ med,
                                           int m0, int spm, int bpm, uint4* __restrict__ out,
                                           int* __restrict__ err) {
    constexpr int ROWS = CTX == 0 ? 567 : (CTX == 1 ? 259 : 1036);
    constexpr int TABU16 = ROWS * ALPHA;
    constexpr u32 NONE = CTX == 0 ? 80u : 36u;
    extern __shared__ uint4 smem[];
    u16* tab = reinterpret_cast<u16*>(smem + 1);   // smem[0] is a zero guard (tab[-1])
    u16* bt = tab + TABU16;
    const int m = m0 + blockIdx.x / bpm;
    const int ls = (blockIdx.x % bpm) * NT + threadIdx.x;
    const int mt = mtype[m];
    {
        const uint4* src = reinterpret_cast<const uint4*>(tabs + (size_t)mt * TABU16);
        for (int i = threadIdx.x; i < TABU16 / 8; i += NT) smem[1 + i] = src[i];
        if (threadIdx.x == 0) smem[0] = make_uint4(0u, 0u, 0u, 0u);
        if (CTX == 2 && threadIdx.x < 8) bt[threadIdx.x] = bittabs[mt * 8 + threadIdx.x];
    }
    __syncthreads();
    if (ls >= spm) return;
    const int gs = m * spm + ls;
    // ILV 3: two separate sub-streams per thread (even blocks / odd blocks), offsets 2*gs .. 2*gs+2
    const u32 off = (ILV == 3) ? offs[2 * gs] : offs[gs];
    const u32 mid = (ILV == 3) ? offs[2 * gs + 1] : 0u;
    const u32 endo = (ILV == 3) ? offs[2 * gs + 2] : offs[gs + 1];
    Win w;
    win_init(w, pay, off);
    Win w1;
    if (ILV == 3) win_init(w1, pay, mid);
    Win& wb_ = (ILV == 3) ? w1 : w;
    const u32* p0 = pay + (off >> 2);
    const u32* p1 = pay + (mid >> 2);
    const int md = med[m];
    const u8* sc = scales + (size_t)gs * 64;
    const u32* sg = signs + (size_t)gs * 64;
#if OUTMODE == 2
    uint4* o = out + (size_t)(gs & 1023) * 64;   // diagnostic sink: all streams write into 1 MB
#else
    uint4* o = out + (size_t)gs * 64;
#endif
#if OUTMODE == 1
    uint4 xacc = make_uint4(0u, 0u, 0u, 0u);
#endif
#if OUTMODE == 4
    uint4* stg = smem + 1 + TABU16 / 8 + (CTX == 2 ? 1 : 0) + threadIdx.x * 9;   // 144-B stride per lane
#endif
    int bad = 0;

    if (ILV == 0 || ILV == 1) {
        u32 x0 = win_take4(w);
        u32 x1 = RL;
        bad |= (x0 < RL);
        if (ILV == 1) { x1 = win_take4(w); bad |= (x1 < RL); }
        u32 scn = __ldg(sc);
        u32 sgn = (ALPHA == 8) ? __ldg(sg) : 0u;
        for (int b = 0; b < 64; b++) {
            int sv = (int)scn - md;
            sv = max(-3, min(3, sv)) + 3;
            u32 sgw = sgn;
            if (b + 1 < 64) {
                scn = __ldg(sc + b + 1);
                if (ALPHA == 8) sgn = __ldg(sg + b + 1);
            }
            u32 bit = 0;
            if (CTX == 2) { bit = dbit(bt[sv], x0); renorm(x0, w); }
            const u32 base = srow_base<CTX>((u32)sv, bit);
            u32 wd[4];
            u32 pm, mx;
            {
                u32 s = dsym<ALPHA>(tab, base + NONE, x0);
                renorm(x0, w);
                pm = s & 7u; mx = pm;
                wd[0] = (ALPHA == 16) ? s : (s | ((sgw & 1u) << 3));
            }
#pragma unroll
            for (int j = 1; j < 32; j++) {
                const u32 hb = (CTX == 2 && j >= 16) ? 37u : 0u;
                const u32 row = base + hb + rowof<CTX>(pm, mx);
                u32 s;
                if (ILV == 1 && (j & 1)) { s = dsym<ALPHA>(tab, row, x1); renorm(x1, w); }
                else { s = dsym<ALPHA>(tab, row, x0); renorm(x0, w); }
                pm = s & 7u; mx = max(mx, pm);
                const u32 nib = (ALPHA == 16) ? s : (s | (((sgw >> j) & 1u) << 3));
                if ((j & 7) == 0) wd[j >> 3] = nib; else wd[j >> 3] |= nib << (4 * (j & 7));
            }
            OSTORE(o, b, make_uint4(wd[0], wd[1], wd[2], wd[3]));
        }
        bad |= (x0 != RL) | (x1 != RL);
    } else {
        u32 x0 = win_take4(w);
        u32 x1 = win_take4(wb_);
        bad |= (x0 < RL) | (x1 < RL);
        const u16* sc2 = reinterpret_cast<const u16*>(sc);
        const uint2* sg2 = reinterpret_cast<const uint2*>(sg);
        u32 scn = __ldg(sc2);
        uint2 sgn = (ALPHA == 8) ? __ldg(sg2) : make_uint2(0u, 0u);
        for (int p = 0; p < 32; p++) {
            int sv0 = (int)(scn & 255u) - md, sv1 = (int)(scn >> 8) - md;
            sv0 = max(-3, min(3, sv0)) + 3;
            sv1 = max(-3, min(3, sv1)) + 3;
            uint2 sgw = sgn;
            if (p + 1 < 32) {
                scn = __ldg(sc2 + p + 1);
                if (ALPHA == 8) sgn = __ldg(sg2 + p + 1);
            }
            u32 bit0 = 0, bit1 = 0;
            if (CTX == 2) {
                bit0 = dbit(bt[sv0], x0); renorm(x0, w);
                bit1 = dbit(bt[sv1], x1); renorm(x1, wb_);
            }
            const u32 base0 = srow_base<CTX>((u32)sv0, bit0), base1 = srow_base<CTX>((u32)sv1, bit1);
            u32 wa[4], wb[4];
            u32 pm0, mx0, pm1, mx1;
            {
                u32 s0 = dsym<ALPHA>(tab, base0 + NONE, x0);
                renorm(x0, w);
                u32 s1 = dsym<ALPHA>(tab, base1 + NONE, x1);
                renorm(x1, wb_);
                pm0 = s0 & 7u; mx0 = pm0; pm1 = s1 & 7u; mx1 = pm1;
                wa[0] = (ALPHA == 16) ? s0 : (s0 | ((sgw.x & 1u) << 3));
                wb[0] = (ALPHA == 16) ? s1 : (s1 | ((sgw.y & 1u) << 3));
            }
#pragma unroll
            for (int j = 1; j < 32; j++) {
                const u32 hb = (CTX == 2 && j >= 16) ? 37u : 0u;
                u32 s0 = dsym<ALPHA>(tab, base0 + hb + rowof<CTX>(pm0, mx0), x0);
                renorm(x0, w);
                u32 s1 = dsym<ALPHA>(tab, base1 + hb + rowof<CTX>(pm1, mx1), x1);
                renorm(x1, wb_);
                pm0 = s0 & 7u; mx0 = max(mx0, pm0);
                pm1 = s1 & 7u; mx1 = max(mx1, pm1);
                const u32 n0 = (ALPHA == 16) ? s0 : (s0 | (((sgw.x >> j) & 1u) << 3));
                const u32 n1 = (ALPHA == 16) ? s1 : (s1 | (((sgw.y >> j) & 1u) << 3));
                if ((j & 7) == 0) { wa[j >> 3] = n0; wb[j >> 3] = n1; }
                else { wa[j >> 3] |= n0 << (4 * (j & 7)); wb[j >> 3] |= n1 << (4 * (j & 7)); }
            }
            OSTORE(o, 2 * p, make_uint4(wa[0], wa[1], wa[2], wa[3]));
            OSTORE(o, 2 * p + 1, make_uint4(wb[0], wb[1], wb[2], wb[3]));
        }
        bad |= (x0 != RL) | (x1 != RL);
    }
#if OUTMODE == 1
    o[0] = xacc;
#endif
    const long long consumed = (long long)(8 - (int)(off & 3u)) + 4LL * (long long)(w.wp - (p0 + 3)) - w.cnt;
    bad |= (consumed != (long long)(((ILV == 3) ? mid : endo) - off));
    if (ILV == 3) {
        const long long c1 = (long long)(8 - (int)(mid & 3u)) + 4LL * (long long)(w1.wp - (p1 + 3)) - w1.cnt;
        bad |= (c1 != (long long)(endo - mid));
    }
    if (bad) atomicAdd(err, 1);
}

// ------------------------------------------------------------------------------------------------
// wdec2: same formats (ILV 0 or 2), but the 32-symbol block is decoded as 4 groups of 8 symbols and
// the group loop is unrolled only UNRG times (1 = rolled: ~8 symbol steps of code instead of 32).
// First symbol of a block is detected with the pm == 8 sentinel (mx starts at 0).
template <int ALPHA, int CTX, int ILV, int NT, int UNRG>
__global__ void LBND(NT) wdec2(const u32* __restrict__ pay, const u32* __restrict__ offs,
                                            const u8* __restrict__ scales, const u32* __restrict__ signs,
                                            const u16* __restrict__ tabs, const u16* __restrict__ bittabs,
                                            const u8* __restrict__ mtype, const u8* __restrict__ med,
                                            int m0, int spm, int bpm, uint4* __restrict__ out,
                                            int* __restrict__ err) {
    constexpr int ROWS = CTX == 0 ? 567 : (CTX == 1 ? 259 : 1036);
    constexpr int TABU16 = ROWS * ALPHA;
    constexpr u32 NONE = CTX == 0 ? 80u : 36u;
    extern __shared__ uint4 smem[];
    u16* tab = reinterpret_cast<u16*>(smem + 1);
    u16* bt = tab + TABU16;
    const int m = m0 + blockIdx.x / bpm;
    const int ls = (blockIdx.x % bpm) * NT + threadIdx.x;
    const int mt = mtype[m];
    {
        const uint4* src = reinterpret_cast<const uint4*>(tabs + (size_t)mt * TABU16);
        for (int i = threadIdx.x; i < TABU16 / 8; i += NT) smem[1 + i] = src[i];
        if (threadIdx.x == 0) smem[0] = make_uint4(0u, 0u, 0u, 0u);
        if (CTX == 2 && threadIdx.x < 8) bt[threadIdx.x] = bittabs[mt * 8 + threadIdx.x];
    }
    __syncthreads();
    if (ls >= spm) return;
    const int gs = m * spm + ls;
    const u32 off = offs[gs], endo = offs[gs + 1];
    Win w;
    win_init(w, pay, off);
    const u32* p0 = pay + (off >> 2);
    const int md = med[m];
    const u8* sc = scales + (size_t)gs * 64;
    const u32* sg = signs + (size_t)gs * 64;
    uint4* o = out + (size_t)gs * 64;
    int bad = 0;
    if (ILV == 0) {
        u32 x0 = win_take4(w);
        bad |= (x0 < RL);
        u32 scn = __ldg(sc);
        u32 sgn = (ALPHA == 8) ? __ldg(sg) : 0u;
#pragma unroll 1
        for (int b = 0; b < 64; b++) {
            int sv = (int)scn - md;
            sv = max(-3, min(3, sv)) + 3;
            const u32 sgw = sgn;
            if (b + 1 < 64) {
                scn = __ldg(sc + b + 1);
                if (ALPHA == 8) sgn = __ldg(sg + b + 1);
            }
            u32 bit = 0;
            if (CTX == 2) { bit = dbit(bt[sv], x0); renorm(x0, w); }
            const u32 base = srow_base<CTX>((u32)sv, bit);
            u32 pm = 8u, mx = 0u;
            u32 w0 = 0, w1 = 0, w2 = 0, w3 = 0;
#pragma unroll UNRG
            for (int g = 0; g < 4; g++) {
                const u32 hb = (CTX == 2 && g >= 2) ? 37u : 0u;
                u32 acc = 0;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const u32 row = (pm == 8u) ? base + NONE : base + hb + rowof<CTX>(pm, mx);
                    u32 s = dsym<ALPHA>(tab, row, x0);
                    renorm(x0, w);
                    pm = s & 7u; mx = max(mx, pm);
                    const u32 nib = (ALPHA == 16) ? s : (s | (((sgw >> (g * 8 + k)) & 1u) << 3));
                    acc |= nib << (4 * k);
                }
                w0 = w1; w1 = w2; w2 = w3; w3 = acc;
            }
            o[b] = make_uint4(w0, w1, w2, w3);
        }
        bad |= (x0 != RL);
    } else {
        u32 x0 = win_take4(w);
        u32 x1 = win_take4(w);
        bad |= (x0 < RL) | (x1 < RL);
        const u16* sc2 = reinterpret_cast<const u16*>(sc);
        const uint2* sg2 = reinterpret_cast<const uint2*>(sg);
        u32 scn = __ldg(sc2);
        uint2 sgn = (ALPHA == 8) ? __ldg(sg2) : make_uint2(0u, 0u);
#pragma unroll 1
        for (int p = 0; p < 32; p++) {
            int sv0 = (int)(scn & 255u) - md, sv1 = (int)(scn >> 8) - md;
            sv0 = max(-3, min(3, sv0)) + 3;
            sv1 = max(-3, min(3, sv1)) + 3;
            const uint2 sgw = sgn;
            if (p + 1 < 32) {
                scn = __ldg(sc2 + p + 1);
                if (ALPHA == 8) sgn = __ldg(sg2 + p + 1);
            }
            u32 bit0 = 0, bit1 = 0;
            if (CTX == 2) {
                bit0 = dbit(bt[sv0], x0); renorm(x0, w);
                bit1 = dbit(bt[sv1], x1); renorm(x1, w);
            }
            const u32 base0 = srow_base<CTX>((u32)sv0, bit0), base1 = srow_base<CTX>((u32)sv1, bit1);
            u32 pm0 = 8u, mx0 = 0u, pm1 = 8u, mx1 = 0u;
            u32 a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll UNRG
            for (int g = 0; g < 4; g++) {
                const u32 hb = (CTX == 2 && g >= 2) ? 37u : 0u;
                u32 acc0 = 0, acc1 = 0;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const u32 r0 = (pm0 == 8u) ? base0 + NONE : base0 + hb + rowof<CTX>(pm0, mx0);
                    u32 s0 = dsym<ALPHA>(tab, r0, x0);
                    renorm(x0, w);
                    const u32 r1 = (pm1 == 8u) ? base1 + NONE : base1 + hb + rowof<CTX>(pm1, mx1);
                    u32 s1 = dsym<ALPHA>(tab, r1, x1);
                    renorm(x1, w);
                    pm0 = s0 & 7u; mx0 = max(mx0, pm0);
                    pm1 = s1 & 7u; mx1 = max(mx1, pm1);
                    const u32 n0 = (ALPHA == 16) ? s0 : (s0 | (((sgw.x >> (g * 8 + k)) & 1u) << 3));
                    const u32 n1 = (ALPHA == 16) ? s1 : (s1 | (((sgw.y >> (g * 8 + k)) & 1u) << 3));
                    acc0 |= n0 << (4 * k);
                    acc1 |= n1 << (4 * k);
                }
                a0 = a1; a1 = a2; a2 = a3; a3 = acc0;
                c0 = c1; c1 = c2; c2 = c3; c3 = acc1;
            }
            o[2 * p] = make_uint4(a0, a1, a2, a3);
            o[2 * p + 1] = make_uint4(c0, c1, c2, c3);
        }
        bad |= (x0 != RL) | (x1 != RL);
    }
    const long long consumed = (long long)(8 - (int)(off & 3u)) + 4LL * (long long)(w.wp - (p0 + 3)) - w.cnt;
    bad |= (consumed != (long long)(endo - off));
    if (bad) atomicAdd(err, 1);
}
