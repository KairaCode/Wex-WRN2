/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/quant.h and c/deepseek_v4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* zero-safety kernel DLL (freestanding: no CRT; driven by zs_drive.py via ctypes).
 * Does canonicalising FP4 E2M1 code 8 (-0) to code 0 (+0) change any output bit?
 *
 * Kernels copied verbatim from the colibri dev tree (OpenMP pragmas dropped: they only
 * split rows/tiles across threads and never change a row's operation order):
 *   0  K1   quant.h:1280-1305  matmul_mxfp4 AVX2 arm (V4.1 CPU expert path deepseek_v41.c:3088)
 *   1  K2   quant.h:1307-1317  matmul_mxfp4 scalar arm, mx4_lut code 8 = -0.f (quant.h:1266-1267)
 *   2  K2c  K2 with explicit FMA contraction (what gcc -ffp-contract=fast may emit)
 *   3  K3   quant.h:1339-1389  mxfp4_qx + matmul_mxfp4_i8_pre AVX2 (int8 IDOT)
 *   4  K4   quant.h:1391-1404  matmul_mxfp4_i8_pre scalar
 *   5  K5   deepseek_v4.c:20602-20695 coli_fp4_matvec_rows16_order AVX2
 *   6  K6   deepseek_v4.c:20760-20808 coli_fp4_matvec_rows16_order scalar
 *   7  K7   deepseek_v4.c:20396-20492 coli_fp4_matmul_batch_rows16_order AVX2 (S rows)
 *   8  K8   deepseek_v4.c:21540-21558,21594-21616,21727-21764 rows16 pack + v10 AVX2 matvec
 * IEEE-exact C emulations of the GPU kernels' arithmetic order (NOT the GPU binaries):
 *   9  K9a  backend_cuda_dsv4.cu:99,127-143 mv_fp4_grouped, f4(8) = -0.0f, fma(x*f4, e8, sum)
 *   10 K9b  same, uncontracted sum + (x*f4)*e8
 *   11 K10  backend_cuda_dsv4.cu:2250-2257,2279-2289,2313 v4f_dot_fp4 + v4f_tree
 *   12 K11  shaders/chain_moe_gv.comp:97-114 e2/dotmx/stepmx, row acc = 0.0 (rowpart :131)
 *   13 K12  NEGATIVE CONTROL: K11 with the row accumulator seeded by the first step
 *           (the fold 0.0 + x -> x a driver may do when signed zeros are not preserved)
 */
#include <stdint.h>
#include <immintrin.h>

int _fltused = 0;
#pragma function(memset)
void *memset(void *d, int c, unsigned long long n){ unsigned char *p=(unsigned char*)d; while(n--) *p++=(unsigned char)c; return d; }
#pragma function(memcpy)
void *memcpy(void *d, const void *s, unsigned long long n){ unsigned char *p=(unsigned char*)d; const unsigned char *q=(const unsigned char*)s; while(n--) *p++=*q++; return d; }

#define EXPORT __declspec(dllexport)
static float e8lut[256];   /* deepseek_v4.c:20100-20103 coli_e8m0_decode (filled by the driver) */
static float e8cuda[256];  /* backend_cuda_dsv4.cu:1605 dsv4_cuda_init e8_table (filled by the driver) */
EXPORT void zs_set_tables(const float *a, const float *b){ for(int i=0;i<256;i++){ e8lut[i]=a[i]; e8cuda[i]=b[i]; } }

static float coli_e2m1_decode(uint8_t nibble){   /* deepseek_v4.c:20123-20129 */
    static const float values[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                                     0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
    return values[nibble & 15];
}
static inline float hsum256(__m256 v){           /* idot.h:25-29 */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
static const float mx4_lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,      /* quant.h:1266-1267 */
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){ union { uint32_t u; float f; } b; b.u=(uint32_t)s<<23; return b.f; }
static inline float fma_hw(float a,float b,float c){ return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a),_mm_set_ss(b),_mm_set_ss(c))); }
static inline float fabs_bits(float x){ union { uint32_t u; float f; } b; b.f=x; b.u&=0x7fffffffu; return b.f; }
static inline int lrint_sse(float x){ return _mm_cvtss_si32(_mm_set_ss(x)); }   /* lrintf, round-to-nearest-even MXCSR */
static inline uint32_t ld32(const uint8_t *p){ return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }

/* ---------------- K1 / K2 / K2c ---------------- */
static void k1(float *y,const float *x,const uint8_t *q4,const uint8_t *e8s,int S,int I,int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I;
            const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
            const __m128i m4=_mm_set1_epi8(0x0F);
            __m256 acc=_mm256_setzero_ps();
            for(int g=0;g<ng;g++){
                __m128i by=_mm_loadu_si128((const __m128i*)(w+g*16));
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi));
                __m128i n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));
                const float *xg=xs+g*32;
                __m256 ga=_mm256_mul_ps(_mm256_loadu_ps(xg),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n0)));
                ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+8),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n0,8))),ga);
                ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+16),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n1)),ga);
                ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+24),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n1,8))),ga);
                acc=_mm256_fmadd_ps(ga,_mm256_set1_ps(mx4_scale(scl[g])*0.5f),acc);
            }
            y[(int64_t)s*O+o]=hsum256(acc);
        }
    }
}
static void k2(float *y,const float *x,const uint8_t *q4,const uint8_t *e8s,int S,int I,int O,int contract){
    int rb=(I+1)/2, ng=(I+31)/32;
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0;g<ng;g++){ int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float sc=mx4_scale(scl[g]), ga=0;
                for(int i=base;i<base+glen;i+=2){ uint8_t byte=w[i>>1];
                    if(contract){ ga=fma_hw(xs[i],mx4_lut[byte&0xF],ga); if(i+1<base+glen) ga=fma_hw(xs[i+1],mx4_lut[byte>>4],ga); }
                    else { ga+=xs[i]*mx4_lut[byte&0xF]; if(i+1<base+glen) ga+=xs[i+1]*mx4_lut[byte>>4]; } }
                if(contract) a=fma_hw(ga,sc,a); else a+=ga*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}
/* ---------------- K3 / K4 ---------------- */
static void mxfp4_qx(const float *x,int S,int I,int8_t *xq,float *xsc){   /* quant.h:1339-1353 */
    int ng=I/32;
    for(int s=0;s<S;s++) for(int g=0;g<ng;g++){ const float *xg=x+(int64_t)s*I+g*32;
        float am=0; for(int i=0;i<32;i++){ float a=fabs_bits(xg[i]); if(a>am)am=a; }
        float sc=am/127.f; if(sc<1e-20f)sc=1e-20f;
        xsc[(int64_t)s*ng+g]=sc; float inv=1.f/sc; int8_t *qg=xq+(int64_t)s*I+g*32;
        for(int i=0;i<32;i++){ int v=lrint_sse(xg[i]*inv); if(v>127)v=127; if(v<-127)v=-127; qg[i]=(int8_t)v; } }
}
static void k3k4(float *y,const int8_t *xq,const float *xsc,const uint8_t *q4,const uint8_t *e8s,int S,int I,int O,int simd){
    int rb=I/2, ng=I/32;
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){ const int8_t *xr=xq+(int64_t)s*I; const float *xsr=xsc+(int64_t)s*ng;
            if(simd){
                const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i ones=_mm256_set1_epi16(1);
                __m256 acc=_mm256_setzero_ps();
                for(int g=0;g<ng;g++){
                    __m128i by=_mm_loadu_si128((const __m128i*)(w+g*16));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi));
                    __m128i n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));
                    __m256i wv=_mm256_set_m128i(n1,n0);
                    __m256i xv=_mm256_loadu_si256((const __m256i*)(xr+g*32));
                    __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
                    acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p,ones)),
                                        _mm256_set1_ps(mx4_scale(scl[g])*0.5f*xsr[g]),acc);
                }
                y[(int64_t)s*O+o]=hsum256(acc);
            } else {
                static const int8_t l2[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
                float a=0;
                for(int g=0;g<ng;g++){ const int8_t *qg=xr+g*32; int32_t gi=0;
                    for(int i=0;i<32;i+=2){ uint8_t byte=w[(g*32+i)>>1];
                        gi+=(int32_t)l2[byte&0xF]*qg[i]+(int32_t)l2[byte>>4]*qg[i+1]; }
                    a+=(float)gi*(mx4_scale(scl[g])*0.5f*xsr[g]); }
                y[(int64_t)s*O+o]=a;
            }
        }
    }
}
/* ---------------- K5 ---------------- */
static void k5(float *y,const uint8_t *q4,const uint8_t *e8s,const float *x,int I,int O){
    int rb=I/2, ng=I/32;
    for(int64_t tile=0;tile<O/16;tile++){
        const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
        const __m128i m4=_mm_set1_epi8(0x0F); const __m256 half=_mm256_set1_ps(0.5f);
        __m256 sum0=_mm256_setzero_ps(), sum1=_mm256_setzero_ps();
        for(int base=0;base<I;base+=32){
            for(int half_rows=0;half_rows<2;half_rows++){
                float sc[8]; __m256 rowv[8][4];
                for(int r8=0;r8<8;r8++){ int64_t row=tile*16+half_rows*8+r8;
                    sc[r8]=e8lut[e8s[row*ng+base/32]];
                    __m128i by=_mm_loadu_si128((const __m128i*)(q4+row*rb+base/2));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi));
                    __m128i n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));
                    rowv[r8][0]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n0)),half);
                    rowv[r8][1]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n0,8))),half);
                    rowv[r8][2]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n1)),half);
                    rowv[r8][3]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n1,8))),half);
                }
                __m256 scale=_mm256_loadu_ps(sc); __m256 partial=half_rows?sum1:sum0;
                for(int cb=0;cb<4;cb++){ __m256 blk[8]; for(int r8=0;r8<8;r8++) blk[r8]=rowv[r8][cb];
                    __m256 t0=_mm256_unpacklo_ps(blk[0],blk[1]), t1=_mm256_unpackhi_ps(blk[0],blk[1]);
                    __m256 t2=_mm256_unpacklo_ps(blk[2],blk[3]), t3=_mm256_unpackhi_ps(blk[2],blk[3]);
                    __m256 t4=_mm256_unpacklo_ps(blk[4],blk[5]), t5=_mm256_unpackhi_ps(blk[4],blk[5]);
                    __m256 t6=_mm256_unpacklo_ps(blk[6],blk[7]), t7=_mm256_unpackhi_ps(blk[6],blk[7]);
                    __m256 s0=_mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0)), s1=_mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));
                    __m256 s2=_mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0)), s3=_mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));
                    __m256 s4=_mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0)), s5=_mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));
                    __m256 s6=_mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0)), s7=_mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));
                    __m256 weights[8];
                    weights[0]=_mm256_permute2f128_ps(s0,s4,0x20); weights[1]=_mm256_permute2f128_ps(s1,s5,0x20);
                    weights[2]=_mm256_permute2f128_ps(s2,s6,0x20); weights[3]=_mm256_permute2f128_ps(s3,s7,0x20);
                    weights[4]=_mm256_permute2f128_ps(s0,s4,0x31); weights[5]=_mm256_permute2f128_ps(s1,s5,0x31);
                    weights[6]=_mm256_permute2f128_ps(s2,s6,0x31); weights[7]=_mm256_permute2f128_ps(s3,s7,0x31);
                    for(int column=0;column<8;column++){ __m256 xv=_mm256_set1_ps(x[base+cb*8+column]);
                        partial=_mm256_add_ps(partial,_mm256_mul_ps(_mm256_mul_ps(xv,weights[column]),scale)); }
                }
                if(half_rows) sum1=partial; else sum0=partial;
            }
        }
        _mm256_storeu_ps(y+tile*16,sum0); _mm256_storeu_ps(y+tile*16+8,sum1);
    }
}
/* ---------------- K6 ---------------- */
static void k6(float *y,const uint8_t *q4,const uint8_t *e8s,const float *x,int I,int O){
    int rb=I/2, ng=I/32;
    for(int o=0;o<O;o+=4){ int o1=o+1<O?o+1:o, o2=o+2<O?o+2:o, o3=o+3<O?o+3:o;
        const uint8_t *w0=q4+(int64_t)o*rb,*w1=q4+(int64_t)o1*rb,*w2=q4+(int64_t)o2*rb,*w3=q4+(int64_t)o3*rb;
        const uint8_t *scl0=e8s+(int64_t)o*ng,*scl1=e8s+(int64_t)o1*ng,*scl2=e8s+(int64_t)o2*ng,*scl3=e8s+(int64_t)o3*ng;
        float sum0=0.0f,sum1=0.0f,sum2=0.0f,sum3=0.0f;
        for(int c=0;c<I;c++){ float xc=x[c];
            uint8_t b0=w0[c>>1],b1=w1[c>>1],b2=w2[c>>1],b3=w3[c>>1];
            float wv0=coli_e2m1_decode((c&1)?(uint8_t)(b0>>4):(uint8_t)(b0&0xF));
            float wv1=coli_e2m1_decode((c&1)?(uint8_t)(b1>>4):(uint8_t)(b1&0xF));
            float wv2=coli_e2m1_decode((c&1)?(uint8_t)(b2>>4):(uint8_t)(b2&0xF));
            float wv3=coli_e2m1_decode((c&1)?(uint8_t)(b3>>4):(uint8_t)(b3&0xF));
            float t0=xc*wv0, t1=xc*wv1, t2=xc*wv2, t3=xc*wv3;
            t0=t0*e8lut[scl0[c/32]]; t1=t1*e8lut[scl1[c/32]]; t2=t2*e8lut[scl2[c/32]]; t3=t3*e8lut[scl3[c/32]];
            sum0=sum0+t0; sum1=sum1+t1; sum2=sum2+t2; sum3=sum3+t3; }
        y[o]=sum0; if(o1!=o)y[o1]=sum1; if(o2!=o)y[o2]=sum2; if(o3!=o)y[o3]=sum3;
    }
}
/* ---------------- K7 (large arrays static: no __chkstk without a CRT) ---------------- */
static __m256 k7sum0[128], k7sum1[128]; static float k7tmp[2][32][8];
static void k7(float *y,const uint8_t *q4,const uint8_t *e8s,const float *x,int S,int I,int O){
    int rb=I/2, ng=I/32;
    for(int64_t tile=0;tile<O/16;tile++){
        const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
        const __m128i m4=_mm_set1_epi8(0x0F); const __m256 half=_mm256_set1_ps(0.5f);
        __m256 *sum0=k7sum0, *sum1=k7sum1;
        for(int s=0;s<S;s++){ sum0[s]=_mm256_setzero_ps(); sum1[s]=_mm256_setzero_ps(); }
        for(int base=0;base<I;base+=32){ float sc[16];
            for(int half_rows=0;half_rows<2;half_rows++){ __m256 rowv[8][4];
                for(int r8=0;r8<8;r8++){ int64_t row=tile*16+half_rows*8+r8;
                    sc[half_rows*8+r8]=e8lut[e8s[row*ng+base/32]];
                    __m128i by=_mm_loadu_si128((const __m128i*)(q4+row*rb+base/2));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi));
                    __m128i n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));
                    rowv[r8][0]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n0)),half);
                    rowv[r8][1]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n0,8))),half);
                    rowv[r8][2]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n1)),half);
                    rowv[r8][3]=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n1,8))),half);
                }
                for(int cb=0;cb<4;cb++){ __m256 blk[8]; for(int r8=0;r8<8;r8++) blk[r8]=rowv[r8][cb];
                    __m256 t0=_mm256_unpacklo_ps(blk[0],blk[1]), t1=_mm256_unpackhi_ps(blk[0],blk[1]);
                    __m256 t2=_mm256_unpacklo_ps(blk[2],blk[3]), t3=_mm256_unpackhi_ps(blk[2],blk[3]);
                    __m256 t4=_mm256_unpacklo_ps(blk[4],blk[5]), t5=_mm256_unpackhi_ps(blk[4],blk[5]);
                    __m256 t6=_mm256_unpacklo_ps(blk[6],blk[7]), t7=_mm256_unpackhi_ps(blk[6],blk[7]);
                    __m256 s0=_mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0)), s1=_mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));
                    __m256 s2=_mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0)), s3=_mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));
                    __m256 s4=_mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0)), s5=_mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));
                    __m256 s6=_mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0)), s7=_mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+0],_mm256_permute2f128_ps(s0,s4,0x20));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+1],_mm256_permute2f128_ps(s1,s5,0x20));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+2],_mm256_permute2f128_ps(s2,s6,0x20));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+3],_mm256_permute2f128_ps(s3,s7,0x20));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+4],_mm256_permute2f128_ps(s0,s4,0x31));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+5],_mm256_permute2f128_ps(s1,s5,0x31));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+6],_mm256_permute2f128_ps(s2,s6,0x31));
                    _mm256_storeu_ps(k7tmp[half_rows][cb*8+7],_mm256_permute2f128_ps(s3,s7,0x31));
                }
            }
            __m256 sc0=_mm256_loadu_ps(sc), sc1=_mm256_loadu_ps(sc+8);
            for(int c=0;c<32;c++){ __m256 weight0=_mm256_loadu_ps(k7tmp[0][c]), weight1=_mm256_loadu_ps(k7tmp[1][c]);
                for(int s=0;s<S;s++){ __m256 xv=_mm256_set1_ps(x[(int64_t)s*I+base+c]);
                    sum0[s]=_mm256_add_ps(sum0[s],_mm256_mul_ps(_mm256_mul_ps(xv,weight0),sc0));
                    sum1[s]=_mm256_add_ps(sum1[s],_mm256_mul_ps(_mm256_mul_ps(xv,weight1),sc1)); } }
        }
        for(int s=0;s<S;s++){ _mm256_storeu_ps(y+(int64_t)s*O+tile*16,sum0[s]); _mm256_storeu_ps(y+(int64_t)s*O+tile*16+8,sum1[s]); }
    }
}
/* ---------------- K8 ---------------- */
EXPORT void zs_pack_rows16(uint8_t *pd,uint8_t *ps,const uint8_t *data,const uint8_t *scales,int rows_,int columns_){
    uint64_t rows=(uint64_t)rows_, columns=(uint64_t)columns_, data_stride=columns/2, scale_stride=columns/32;
    for(uint64_t row=0;row<rows;row++){ uint64_t tile=row/16, lane=row%16;
        for(uint64_t c=0;c<data_stride;c++) pd[(tile*data_stride+c)*16+lane]=data[row*data_stride+c];
        for(uint64_t c=0;c<scale_stride;c++) ps[(tile*scale_stride+c)*16+lane]=scales[row*scale_stride+c]; }
}
static float k8fp4[16];
static void k8(float *output,const uint8_t *data,const uint8_t *scales,const float *activation,uint64_t rows,uint64_t columns){
    for(int i=0;i<16;i++) k8fp4[i]=coli_e2m1_decode((uint8_t)i);   /* avx2_rows16_tables, :21594-21598 */
    uint64_t data_stride=columns/2, scale_stride=columns/32;
    for(int64_t tile=0;tile<(int64_t)rows/16;tile++){
        __m256 sum[2]; sum[0]=_mm256_setzero_ps(); sum[1]=_mm256_setzero_ps();
        for(uint64_t base=0;base<columns;base+=32){ __m256 scale[2];
            { __m128i bytes=_mm_loadu_si128((const __m128i*)(scales+((uint64_t)tile*scale_stride+base/32)*16));
              __m256i lo=_mm256_cvtepu8_epi32(bytes), hi=_mm256_cvtepu8_epi32(_mm_srli_si128(bytes,8));
              scale[0]=_mm256_i32gather_ps(e8lut,lo,4); scale[1]=_mm256_i32gather_ps(e8lut,hi,4); }
            for(uint64_t offset=0;offset<32;offset++){ uint64_t column=base+offset;
                const unsigned char *codes=data+((uint64_t)tile*data_stride+column/2)*16;
                __m256 values[2];
                { __m128i bytes=_mm_loadu_si128((const __m128i*)codes);
                  __m256i lo=_mm256_cvtepu8_epi32(bytes), hi=_mm256_cvtepu8_epi32(_mm_srli_si128(bytes,8));
                  __m256i mask=_mm256_set1_epi32(15);
                  if(column&1){ lo=_mm256_srli_epi32(lo,4); hi=_mm256_srli_epi32(hi,4); }
                  else { lo=_mm256_and_si256(lo,mask); hi=_mm256_and_si256(hi,mask); }
                  values[0]=_mm256_i32gather_ps(k8fp4,lo,4); values[1]=_mm256_i32gather_ps(k8fp4,hi,4); }
                __m256 xx=_mm256_set1_ps(activation[column]);
                for(int h=0;h<2;h++) sum[h]=_mm256_add_ps(sum[h],_mm256_mul_ps(_mm256_mul_ps(xx,values[h]),scale[h]));
            }
        }
        _mm256_storeu_ps(output+(uint64_t)tile*16,sum[0]); _mm256_storeu_ps(output+(uint64_t)tile*16+8,sum[1]);
    }
}
/* ---------------- K9 (CUDA mv_fp4_grouped arithmetic order) ---------------- */
static float f4cuda(uint8_t q){ static const float a[8]={0,.5f,1,1.5f,2,3,4,6}; return q&8?-a[q&7]:a[q]; }
static float k9part[256];
static void k9(float *y,const float *x,const uint8_t *w,const uint8_t *sc,int I,int O,int contract){
    float *part=k9part;
    for(int o=0;o<O;o++){
        for(int j=0;j<256;j++){ float sum=0;
            for(int i=j;i<I;i+=256){ uint8_t p=w[((long long)o*I+i)>>1], q=(i&1)?p>>4:p&15;
                float t=x[i]*f4cuda(q); float e=e8cuda[sc[(long long)o*(I/32)+i/32]];
                if(contract) sum=fma_hw(t,e,sum); else { float u=t*e; sum=sum+u; } }
            part[j]=sum; }
        for(int n=128;n;n>>=1) for(int j=0;j<n;j++) part[j]+=part[j+n];
        y[o]=part[0];
    }
}
/* ---------------- K10 (CUDA v4f_dot_fp4 + v4f_tree) ---------------- */
static float v4f_e8f(unsigned b){ union{uint32_t u; float f;} v; v.u = b ? (b<<23) : 0x00400000u; return v.f; }
static float k10P[32][8], k10Q[32][8];
static void k10(float *y,const float *xs,const uint8_t *w,const uint8_t *s,int I,int O){
    static const float lut[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
    for(int o=0;o<O;o++){ const uint8_t *wr=w+(uint64_t)o*(I/2), *srow=s+(uint64_t)o*(I/32);
        for(int l=0;l<32;l++) for(int r=0;r<8;r++) k10P[l][r]=0.f;
        for(int m=0;m<I;m+=256) for(int l=0;l<32;l++){ int i=m+8*l; uint32_t u=ld32(wr+(i>>1));
            float sc=v4f_e8f(srow[i>>5]);
            for(int r=0;r<8;r++){ float a=xs[i+r]*lut[(u>>(4*r))&15]; float b=a*sc; k10P[l][r]=k10P[l][r]+b; } }
        for(int k=16;k;k>>=1){ for(int l=0;l<32;l++) for(int r=0;r<8;r++) k10Q[l][r]=k10P[l][r];
            for(int l=0;l<32;l++) for(int r=0;r<8;r++){ float other=(l+k<32)?k10Q[l+k][r]:k10Q[l][r]; k10P[l][r]=k10Q[l][r]+other; } }
        float *p=k10P[0]; p[0]=p[0]+p[4]; p[1]=p[1]+p[5]; p[2]=p[2]+p[6]; p[3]=p[3]+p[7];
        p[0]=p[0]+p[2]; p[1]=p[1]+p[3]; y[o]=p[0]+p[1];
    }
}
/* ---------------- K11 / K12 (Vulkan chain_moe_gv) ---------------- */
static float e2glsl(uint32_t w,int off){ uint32_t n=(w>>off)&15u; float v=(float)((0xC8643210u>>((n&7u)*4u))&15u)*0.5f; return (n&8u)!=0u?-v:v; }
static float dot4(const float *a,float b0,float b1,float b2,float b3){ float r=a[0]*b0; r=r+a[1]*b1; r=r+a[2]*b2; r=r+a[3]*b3; return r; }
static void k11(float *y,const float *x,const uint8_t *w,const uint8_t *s,int I,int O,int firstterm){
    int ng=I/32;
    for(int o=0;o<O;o++){ const uint8_t *wr=w+(uint64_t)o*(I/2); float acc=0.0f;
        for(int q=0;q<I/32;q++){ uint32_t v0=ld32(wr+q*16),v1=ld32(wr+q*16+4),v2=ld32(wr+q*16+8),v3=ld32(wr+q*16+12);
            const float *xb=x+q*32; float d;
            d =dot4(xb+0 ,e2glsl(v0,0),e2glsl(v0,4),e2glsl(v0,8),e2glsl(v0,12));
            d =d+dot4(xb+4 ,e2glsl(v0,16),e2glsl(v0,20),e2glsl(v0,24),e2glsl(v0,28));
            d =d+dot4(xb+8 ,e2glsl(v1,0),e2glsl(v1,4),e2glsl(v1,8),e2glsl(v1,12));
            d =d+dot4(xb+12,e2glsl(v1,16),e2glsl(v1,20),e2glsl(v1,24),e2glsl(v1,28));
            d =d+dot4(xb+16,e2glsl(v2,0),e2glsl(v2,4),e2glsl(v2,8),e2glsl(v2,12));
            d =d+dot4(xb+20,e2glsl(v2,16),e2glsl(v2,20),e2glsl(v2,24),e2glsl(v2,28));
            d =d+dot4(xb+24,e2glsl(v3,0),e2glsl(v3,4),e2glsl(v3,8),e2glsl(v3,12));
            d =d+dot4(xb+28,e2glsl(v3,16),e2glsl(v3,20),e2glsl(v3,24),e2glsl(v3,28));
            float step=d*mx4_scale(s[(uint64_t)o*ng+q]);
            if(firstterm && q==0) acc=step; else acc=acc+step; }
        y[o]=acc;
    }
}
/* ---------------- entry ---------------- */
EXPORT int zs_run(int k,float *y,const uint8_t *w,const uint8_t *s,const uint8_t *pd,const uint8_t *ps,
                  const float *x,int S,int I,int O,int8_t *xq,float *xsc){
    switch(k){
    case 0: k1(y,x,w,s,S,I,O); return 0;
    case 1: k2(y,x,w,s,S,I,O,0); return 0;
    case 2: k2(y,x,w,s,S,I,O,1); return 0;
    case 3: mxfp4_qx(x,S,I,xq,xsc); k3k4(y,xq,xsc,w,s,S,I,O,1); return 0;
    case 4: mxfp4_qx(x,S,I,xq,xsc); k3k4(y,xq,xsc,w,s,S,I,O,0); return 0;
    case 5: k5(y,w,s,x,I,O); return 0;
    case 6: k6(y,w,s,x,I,O); return 0;
    case 7: if(S>128) return -1; k7(y,w,s,x,S,I,O); return 0;
    case 8: k8(y,pd,ps,x,(uint64_t)O,(uint64_t)I); return 0;
    case 9: k9(y,x,w,s,I,O,1); return 0;
    case 10: k9(y,x,w,s,I,O,0); return 0;
    case 11: k10(y,x,w,s,I,O); return 0;
    case 12: k11(y,x,w,s,I,O,0); return 0;
    case 13: k11(y,x,w,s,I,O,1); return 0;
    }
    return -1;
}
