/* Calibration on this machine: core cycles per TSC tick (dependent add chain) and throughput of AVX-512 gathers
 * (independent, L1-resident and 1.5 MB tables). Prints TSC ticks per operation. */
#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
static uint64_t tab[1u << 18];   /* 2 MB */
int main(void) {
    for (unsigned i = 0; i < (1u << 18); i++) tab[i] = i * 0x9e3779b97f4a7c15ull;
    uint64_t t0, t1, n = 200000000;
    uint64_t a = 1;
    t0 = __rdtsc();
    for (uint64_t i = 0; i < n; i++) { __asm__ volatile("imul %0, %0\n\timul %0, %0\n\timul %0, %0\n\timul %0, %0" : "+r"(a)); }
    t1 = __rdtsc();
    printf("dependent imul (3 core cycles each): %.3f TSC ticks per imul => core clock = %.2f x TSC\n", (double)(t1 - t0) / (4.0 * n), 3.0 * 4.0 * n / (double)(t1 - t0));
    for (unsigned span = 64; span <= (1u << 18); span <<= 6) {
        __m512i idx = _mm512_set_epi32(15*7,14*7,13*7,12*7,11*7,10*7,9*7,8*7,7*7,6*7,5*7,4*7,3*7,2*7,7,0), acc = _mm512_setzero_si512();
        const __m512i step = _mm512_set1_epi32(97), mask = _mm512_set1_epi32((int)span - 1);
        uint64_t reps = 20000000;
        t0 = __rdtsc();
        for (uint64_t i = 0; i < reps; i++) {
            idx = _mm512_and_si512(_mm512_add_epi32(idx, step), mask);
            __m512i q0 = _mm512_i32gather_epi64(_mm512_castsi512_si256(idx), (const void *)tab, 8);
            __m512i q1 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx, 1), (const void *)tab, 8);
            acc = _mm512_add_epi64(acc, _mm512_add_epi64(q0, q1));
        }
        t1 = __rdtsc();
        printf("2 x vpgatherdq zmm (16 qwords), table %u KB: %.2f TSC ticks per pair\n", span * 8 / 1024, (double)(t1 - t0) / reps);
        t0 = __rdtsc();
        for (uint64_t i = 0; i < reps; i++) {
            idx = _mm512_and_si512(_mm512_add_epi32(idx, step), mask);
            acc = _mm512_add_epi32(acc, _mm512_i32gather_epi32(idx, (const void *)tab, 4));
        }
        t1 = __rdtsc();
        printf("1 x vpgatherdd zmm (16 dwords), table %u KB: %.2f TSC ticks\n", span * 4 / 1024, (double)(t1 - t0) / reps);
        if (_mm512_reduce_add_epi64(acc) == 42) printf("!");
    }
    return 0;
}
