/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/quant.h and c/deepseek_v4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* zero-safety: does canonicalising FP4 E2M1 code 8 (-0) to code 0 (+0) change any
 * output bit of the MXFP4 expert matvec kernels?
 *
 * Kernels are copied verbatim from the colibri dev tree (OpenMP pragmas dropped:
 * they split rows/tiles across threads and never change a row's operation order).
 *   K1  quant.h:1280-1305  matmul_mxfp4, AVX2 arm   (V4.1 CPU expert path: deepseek_v41.c:3088)
 *   K2  quant.h:1307-1317  matmul_mxfp4, scalar arm (mx4_lut with -0.f, quant.h:1266-1267)
 *   K2c K2 with explicit FMA contraction (what gcc -ffp-contract=fast may emit)
 *   K3  quant.h:1339-1389  mxfp4_qx + matmul_mxfp4_i8_pre, AVX2 arm (int8 IDOT)
 *   K4  quant.h:1391-1404  matmul_mxfp4_i8_pre, scalar arm
 *   K5  deepseek_v4.c:20602-20695 coli_fp4_matvec_rows16_order, AVX2 arm
 *   K6  deepseek_v4.c:20760-20808 coli_fp4_matvec_rows16_order, scalar arm
 *   K7  deepseek_v4.c:20396-20492 coli_fp4_matmul_batch_rows16_order, AVX2 arm, S=4
 *   K8  deepseek_v4.c:21540-21558 + 21594-21616 + 21727-21764 rows16 pack + AVX2 matvec
 * IEEE-exact C emulations of GPU kernels' arithmetic order (not the GPU binaries):
 *   K9a backend_cuda_dsv4.cu:99,127-143 mv_fp4_grouped, f4() code 8 = -0.0f, nvcc-contracted fma
 *   K9b same, uncontracted (sum + (x*f4)*e8)
 *   K10 backend_cuda_dsv4.cu:2250-2257,2279-2289,2313 v4f_dot_fp4 + v4f_tree (-0.f LUT, __fadd_rn/__fmul_rn)
 *   K11 shaders/chain_moe_gv.comp:97-114 + qmatmul rowpart with acc = 0.0 (e2(): code 8 = -0.0)
 * Negative control (shows the test sees sign-of-zero effects when they exist):
 *   K12 K11 but the row accumulator starts from the first step (what a driver that does not
 *       preserve signed zeros may legally fold 0.0 + x -> x into)
 * Usage: zs_cpu <sample_dir> <full|quick> <file.bin>...
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define NK 14
static const char *KN[NK] = {"K1_mxfp4_avx2","K2_mxfp4_scalar","K2c_mxfp4_scalar_fma","K3_idot_avx2","K4_idot_scalar",
    "K5_rows16order_avx2","K6_rows16order_scalar","K7_batch_rows16order_avx2_S4","K8_rows16_v10_avx2",
    "K9a_cuda_grouped_fma","K9b_cuda_grouped_nofma","K10_cuda_v4f_dot_fp4","K11_vk_gv_acc0","K12_NEGCTRL_vk_firstterm"};

/* ---------------- shared decoders ---------------- */
static float e8lut[256];                 /* deepseek_v4.c:20100-20103 coli_e8m0_decode */
static float e8cuda[256];                /* backend_cuda_dsv4.cu:1605 e8_table */
static void init_tables(void){
    for(int v=0;v<256;v++){ e8lut[v] = v==0xff ? NAN : ldexpf(1.0f,v-127);
        e8cuda[v] = v ? ldexpf(1.f,v-127) : ldexpf(1.f,-127); }
}
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
static const float mx4_lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,      /* quant.h:1266 */
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){ union { uint32_t u; float f; } b; b.u=(uint32_t)s<<23; return b.f; }
static inline float fma_hw(float a,float b,float c){ return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a),_mm_set_ss(b),_mm_set_ss(c))); }

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
        float am=0; for(int i=0;i<32;i++){ float a=fabsf(xg[i]); if(a>am)am=a; }
        float sc=am/127.f; if(sc<1e-20f)sc=1e-20f;
        xsc[(int64_t)s*ng+g]=sc; float inv=1.f/sc; int8_t *qg=xq+(int64_t)s*I+g*32;
        for(int i=0;i<32;i++){ int v=(int)lrintf(xg[i]*inv); if(v>127)v=127; if(v<-127)v=-127; qg[i]=(int8_t)v; } }
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
/* ---------------- K5 (rows16 order, AVX2, batch one) ---------------- */
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
                    __m256 weights[8]={_mm256_permute2f128_ps(s0,s4,0x20),_mm256_permute2f128_ps(s1,s5,0x20),
                        _mm256_permute2f128_ps(s2,s6,0x20),_mm256_permute2f128_ps(s3,s7,0x20),
                        _mm256_permute2f128_ps(s0,s4,0x31),_mm256_permute2f128_ps(s1,s5,0x31),
                        _mm256_permute2f128_ps(s2,s6,0x31),_mm256_permute2f128_ps(s3,s7,0x31)};
                    for(int column=0;column<8;column++){ __m256 xv=_mm256_set1_ps(x[base+cb*8+column]);
                        partial=_mm256_add_ps(partial,_mm256_mul_ps(_mm256_mul_ps(xv,weights[column]),scale)); }
                }
                if(half_rows) sum1=partial; else sum0=partial;
            }
        }
        _mm256_storeu_ps(y+tile*16,sum0); _mm256_storeu_ps(y+tile*16+8,sum1);
    }
}
/* ---------------- K6 (rows16 order, scalar) ---------------- */
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
/* ---------------- K7 (batch rows16 order, AVX2, S rows) ---------------- */
static void k7(float *y,const uint8_t *q4,const uint8_t *e8s,const float *x,int S,int I,int O){
    int rb=I/2, ng=I/32;
    for(int64_t tile=0;tile<O/16;tile++){
        const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
        const __m128i m4=_mm_set1_epi8(0x0F); const __m256 half=_mm256_set1_ps(0.5f);
        __m256 sum0[128], sum1[128];
        for(int s=0;s<S;s++){ sum0[s]=_mm256_setzero_ps(); sum1[s]=_mm256_setzero_ps(); }
        for(int base=0;base<I;base+=32){ float sc[16], tmp[2][32][8];
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
                    _mm256_storeu_ps(tmp[half_rows][cb*8+0],_mm256_permute2f128_ps(s0,s4,0x20));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+1],_mm256_permute2f128_ps(s1,s5,0x20));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+2],_mm256_permute2f128_ps(s2,s6,0x20));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+3],_mm256_permute2f128_ps(s3,s7,0x20));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+4],_mm256_permute2f128_ps(s0,s4,0x31));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+5],_mm256_permute2f128_ps(s1,s5,0x31));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+6],_mm256_permute2f128_ps(s2,s6,0x31));
                    _mm256_storeu_ps(tmp[half_rows][cb*8+7],_mm256_permute2f128_ps(s3,s7,0x31));
                }
            }
            __m256 sc0=_mm256_loadu_ps(sc), sc1=_mm256_loadu_ps(sc+8);
            for(int c=0;c<32;c++){ __m256 weight0=_mm256_loadu_ps(tmp[0][c]), weight1=_mm256_loadu_ps(tmp[1][c]);
                for(int s=0;s<S;s++){ __m256 xv=_mm256_set1_ps(x[(int64_t)s*I+base+c]);
                    sum0[s]=_mm256_add_ps(sum0[s],_mm256_mul_ps(_mm256_mul_ps(xv,weight0),sc0));
                    sum1[s]=_mm256_add_ps(sum1[s],_mm256_mul_ps(_mm256_mul_ps(xv,weight1),sc1)); } }
        }
        for(int s=0;s<S;s++){ _mm256_storeu_ps(y+(int64_t)s*O+tile*16,sum0[s]); _mm256_storeu_ps(y+(int64_t)s*O+tile*16+8,sum1[s]); }
    }
}
/* ---------------- K8 (rows16 pack + AVX2 v10 matvec) ---------------- */
static void pack_rows16(uint8_t *pd,uint8_t *ps,const uint8_t *data,const uint8_t *scales,size_t rows,size_t columns){
    size_t data_stride=columns/2, scale_stride=columns/32;
    for(size_t row=0;row<rows;row++){ size_t tile=row/16, lane=row%16;
        for(size_t c=0;c<data_stride;c++) pd[(tile*data_stride+c)*16+lane]=data[row*data_stride+c];
        for(size_t c=0;c<scale_stride;c++) ps[(tile*scale_stride+c)*16+lane]=scales[row*scale_stride+c]; }
}
typedef struct { float fp4[16]; float e8[256]; } Avx2Rows16Tables;
static void k8(float *output,const uint8_t *data,const uint8_t *scales,const float *activation,size_t rows,size_t columns){
    Avx2Rows16Tables tables; for(int i=0;i<16;i++) tables.fp4[i]=coli_e2m1_decode((uint8_t)i);
    memcpy(tables.e8,e8lut,sizeof(tables.e8));
    size_t data_stride=columns/2, scale_stride=columns/32;
    for(int64_t tile=0;tile<(int64_t)rows/16;tile++){
        __m256 sum[2]={_mm256_setzero_ps(),_mm256_setzero_ps()};
        for(size_t base=0;base<columns;base+=32){ __m256 scale[2];
            { __m128i bytes=_mm_loadu_si128((const __m128i*)(scales+((size_t)tile*scale_stride+base/32)*16));
              __m256i lo=_mm256_cvtepu8_epi32(bytes), hi=_mm256_cvtepu8_epi32(_mm_srli_si128(bytes,8));
              scale[0]=_mm256_i32gather_ps(tables.e8,lo,4); scale[1]=_mm256_i32gather_ps(tables.e8,hi,4); }
            for(size_t offset=0;offset<32;offset++){ size_t column=base+offset;
                const unsigned char *codes=data+((size_t)tile*data_stride+column/2)*16;
                __m256 values[2];
                { __m128i bytes=_mm_loadu_si128((const __m128i*)codes);
                  __m256i lo=_mm256_cvtepu8_epi32(bytes), hi=_mm256_cvtepu8_epi32(_mm_srli_si128(bytes,8));
                  __m256i mask=_mm256_set1_epi32(15);
                  if(column&1){ lo=_mm256_srli_epi32(lo,4); hi=_mm256_srli_epi32(hi,4); }
                  else { lo=_mm256_and_si256(lo,mask); hi=_mm256_and_si256(hi,mask); }
                  values[0]=_mm256_i32gather_ps(tables.fp4,lo,4); values[1]=_mm256_i32gather_ps(tables.fp4,hi,4); }
                __m256 xx=_mm256_set1_ps(activation[column]);
                for(int h=0;h<2;h++) sum[h]=_mm256_add_ps(sum[h],_mm256_mul_ps(_mm256_mul_ps(xx,values[h]),scale[h]));
            }
        }
        _mm256_storeu_ps(output+(size_t)tile*16,sum[0]); _mm256_storeu_ps(output+(size_t)tile*16+8,sum[1]);
    }
}
/* ---------------- K9 (CUDA mv_fp4_grouped order) ---------------- */
static float f4cuda(uint8_t q){ const float a[8]={0,.5f,1,1.5f,2,3,4,6}; return q&8?-a[q&7]:a[q]; } /* backend_cuda_dsv4.cu:99 */
static void k9(float *y,const float *x,const uint8_t *w,const uint8_t *sc,int I,int O,int contract){
    float part[256];
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
static void k10(float *y,const float *xs,const uint8_t *w,const uint8_t *s,int I,int O){
    static const float lut[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};   /* :2313 */
    float P[32][8], Q[32][8];
    for(int o=0;o<O;o++){ const uint8_t *wr=w+(size_t)o*(I/2), *srow=s+(size_t)o*(I/32);
        for(int l=0;l<32;l++) for(int r=0;r<8;r++) P[l][r]=0.f;
        for(int m=0;m<I;m+=256) for(int l=0;l<32;l++){ int i=m+8*l; uint32_t u; memcpy(&u,wr+(i>>1),4);
            float sc=v4f_e8f(srow[i>>5]);
            for(int r=0;r<8;r++){ float a=xs[i+r]*lut[(u>>(4*r))&15]; float b=a*sc; P[l][r]=P[l][r]+b; } }
        for(int k=16;k;k>>=1){ memcpy(Q,P,sizeof(P));
            for(int l=0;l<32;l++) for(int r=0;r<8;r++){ float other=(l+k<32)?Q[l+k][r]:Q[l][r]; P[l][r]=Q[l][r]+other; } }
        float *p=P[0]; p[0]=p[0]+p[4]; p[1]=p[1]+p[5]; p[2]=p[2]+p[6]; p[3]=p[3]+p[7];
        p[0]=p[0]+p[2]; p[1]=p[1]+p[3]; y[o]=p[0]+p[1];
    }
}
/* ---------------- K11 / K12 (Vulkan chain_moe_gv e2/dotmx/stepmx) ---------------- */
static float e2glsl(uint32_t w,int off){ uint32_t n=(w>>off)&15u; float v=(float)((0xC8643210u>>((n&7u)*4u))&15u)*0.5f; return (n&8u)!=0u?-v:v; }
static float dot4(const float *a,float b0,float b1,float b2,float b3){ float r=a[0]*b0; r=r+a[1]*b1; r=r+a[2]*b2; r=r+a[3]*b3; return r; }
static void k11(float *y,const float *x,const uint8_t *w,const uint8_t *s,int I,int O,int firstterm){
    int ng=I/32;
    for(int o=0;o<O;o++){ const uint8_t *wr=w+(size_t)o*(I/2); float acc=0.0f;
        for(int q=0;q<I/32;q++){ uint32_t v[4]; memcpy(v,wr+q*16,16); const float *xb=x+q*32; float d;
            d =dot4(xb+0 ,e2glsl(v[0],0),e2glsl(v[0],4),e2glsl(v[0],8),e2glsl(v[0],12));
            d =d+dot4(xb+4 ,e2glsl(v[0],16),e2glsl(v[0],20),e2glsl(v[0],24),e2glsl(v[0],28));
            d =d+dot4(xb+8 ,e2glsl(v[1],0),e2glsl(v[1],4),e2glsl(v[1],8),e2glsl(v[1],12));
            d =d+dot4(xb+12,e2glsl(v[1],16),e2glsl(v[1],20),e2glsl(v[1],24),e2glsl(v[1],28));
            d =d+dot4(xb+16,e2glsl(v[2],0),e2glsl(v[2],4),e2glsl(v[2],8),e2glsl(v[2],12));
            d =d+dot4(xb+20,e2glsl(v[2],16),e2glsl(v[2],20),e2glsl(v[2],24),e2glsl(v[2],28));
            d =d+dot4(xb+24,e2glsl(v[3],0),e2glsl(v[3],4),e2glsl(v[3],8),e2glsl(v[3],12));
            d =d+dot4(xb+28,e2glsl(v[3],16),e2glsl(v[3],20),e2glsl(v[3],24),e2glsl(v[3],28));
            float step=d*mx4_scale(s[(size_t)o*ng+q]);
            if(firstterm && q==0) acc=step; else acc=acc+step; }
        y[o]=acc;
    }
}
/* ---------------- harness ---------------- */
static uint64_t rng=0x9E3779B97F4A7C15ull;
static uint32_t nxt(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>16); }
static float gauss(void){ float u=(nxt()+1.0f)/4294967296.0f*65536.0f; u=(float)((nxt()&0xffffff)+1)/16777217.0f; float v=(float)(nxt()&0xffffff)/16777216.0f; return sqrtf(-2.f*logf(u))*cosf(6.2831853f*v); }
#define NA 9
static const char *AN[NA]={"gauss","gauss_25pct+0_25pct-0","all-0","all+0","one_nonzero_rest-0","denormals","inf_at_100","nan_at_200","huge_1e30"};
static void make_act(float *x,int I,int a){
    for(int i=0;i<I;i++) x[i]=gauss();
    switch(a){
    case 1: for(int i=0;i<I;i++){ uint32_t r=nxt()%4; if(r==0)x[i]=0.0f; else if(r==1)x[i]=-0.0f; } break;
    case 2: for(int i=0;i<I;i++) x[i]=-0.0f; break;
    case 3: for(int i=0;i<I;i++) x[i]=0.0f; break;
    case 4: for(int i=0;i<I;i++) x[i]=-0.0f; x[7]=1.0f; break;
    case 5: for(int i=0;i<I;i++) if(nxt()%10==0) x[i]=(nxt()&1)?1e-41f:-1e-41f; break;
    case 6: x[100]=INFINITY; break;
    case 7: x[200]=NAN; break;
    case 8: for(int i=0;i<I;i++) x[i]*=1e30f; break;
    }
}
typedef struct { long long cmp, mis_val, mis_sz, mis_nan, zero_out, nan_out; } St;
static St st[2][NK];   /* [0]=real matrices, [1]=synthetic all-zero rows */
static void cmp(St *s,const float *a,const float *b,long long n){
    for(long long i=0;i<n;i++){ uint32_t ua,ub; memcpy(&ua,a+i,4); memcpy(&ub,b+i,4); s->cmp++;
        if((ua&0x7fffffffu)==0) s->zero_out++; if(a[i]!=a[i]) s->nan_out++;
        if(ua!=ub){ if(((ua|ub)&0x7fffffffu)==0) s->mis_sz++; else if(a[i]!=a[i]&&b[i]!=b[i]) s->mis_nan++; else s->mis_val++; } }
}
static void canon(uint8_t *d,const uint8_t *s,size_t n,long long *c8){
    for(size_t i=0;i<n;i++){ uint8_t lo=s[i]&15, hi=s[i]>>4; if(lo==8){lo=0;(*c8)++;} if(hi==8){hi=0;(*c8)++;} d[i]=(uint8_t)(lo|(hi<<4)); }
}
static float *ya,*yb,*xs4; static int8_t *xq; static float *xsc; static uint8_t *pdA,*psA,*pdB,*psB;
static void run_one(int k,float *y,const uint8_t *w,const uint8_t *s,const uint8_t *pd,const uint8_t *ps,const float *x,int I,int O){
    switch(k){
    case 0: k1(y,x,w,s,1,I,O); break;
    case 1: k2(y,x,w,s,1,I,O,0); break;
    case 2: k2(y,x,w,s,1,I,O,1); break;
    case 3: mxfp4_qx(x,1,I,xq,xsc); k3k4(y,xq,xsc,w,s,1,I,O,1); break;
    case 4: mxfp4_qx(x,1,I,xq,xsc); k3k4(y,xq,xsc,w,s,1,I,O,0); break;
    case 5: k5(y,w,s,x,I,O); break;
    case 6: k6(y,w,s,x,I,O); break;
    case 7: for(int i=0;i<I;i++){ xs4[i]=x[i]; xs4[I+i]=-x[i]; xs4[2*I+i]=(i&1)?-0.0f:x[i]; xs4[3*I+i]=x[i]*1e-30f; }
            k7(y,w,s,xs4,4,I,O); break;
    case 8: k8(y,pd,ps,x,O,I); break;
    case 9: k9(y,x,w,s,I,O,1); break;
    case 10: k9(y,x,w,s,I,O,0); break;
    case 11: k10(y,x,w,s,I,O); break;
    case 12: k11(y,x,w,s,I,O,0); break;
    case 13: k11(y,x,w,s,I,O,1); break;
    }
}
static void test_matrix(int which,const uint8_t *wA,const uint8_t *wB,const uint8_t *s,int I,int O,const int *ks,int nk,const int *as,int na){
    float x[5120];
    pack_rows16(pdA,psA,wA,s,O,I); pack_rows16(pdB,psB,wB,s,O,I);
    for(int ai=0;ai<na;ai++){ make_act(x,I,as[ai]);
        for(int ki=0;ki<nk;ki++){ int k=ks[ki]; long long n=(k==7)?4LL*O:O;
            run_one(k,ya,wA,s,pdA,psA,x,I,O); run_one(k,yb,wB,s,pdB,psB,x,I,O);
            cmp(&st[which][k],ya,yb,n); } }
}
int main(int argc,char **argv){
    if(argc<4){ fprintf(stderr,"usage: zs_cpu <dir> <full|quick> files...\n"); return 2; }
    init_tables(); int full=!strcmp(argv[2],"full");
    const size_t SC=368640, WB=5898240, REC=3*SC+3*WB;
    uint8_t *rec=malloc(REC), *can=malloc(3*WB); ya=malloc(4*5120*4); yb=malloc(4*5120*4); xs4=malloc(4*5120*4);
    xq=malloc(5120); xsc=malloc(160*4); pdA=malloc(WB); psA=malloc(SC); pdB=malloc(WB); psB=malloc(SC);
    int ks_full[NK]; for(int i=0;i<NK;i++) ks_full[i]=i;
    int ks_quick[]={0,5,8,9,11,12}; int as_full[NA]; for(int i=0;i<NA;i++) as_full[i]=i; int as_quick[]={0,1};
    const int *ks=full?ks_full:ks_quick; int nk=full?NK:(int)(sizeof ks_quick/sizeof *ks_quick);
    const int *as=full?as_full:as_quick; int na=full?NA:2;
    long long c8=0, nib=0;
    for(int f=3;f<argc;f++){
        char path[1024]; snprintf(path,sizeof path,"%s\\%s",argv[1],argv[f]);
        FILE *fp=fopen(path,"rb"); if(!fp){ fprintf(stderr,"open %s failed\n",path); return 1; }
        if(fread(rec,1,REC,fp)!=REC){ fprintf(stderr,"short %s\n",path); return 1; } fclose(fp);
        canon(can,rec+3*SC,3*WB,&c8); nib+=2LL*3*WB;
        const uint8_t *sc1=rec, *sc2=rec+SC, *sc3=rec+2*SC, *w1=rec+3*SC, *w2=rec+3*SC+WB, *w3=rec+3*SC+2*WB;
        test_matrix(0,w1,can,sc1,5120,2304,ks,nk,as,na);
        test_matrix(0,w2,can+WB,sc2,2304,5120,ks,nk,as,na);
        test_matrix(0,w3,can+2*WB,sc3,5120,2304,ks,nk,as,na);
        /* synthetic: first 64 rows, rows 0..31 all zero codes (random mix of 0 and 8; canonical all 0) */
        for(int m=0;m<2;m++){ int I=m?2304:5120; const uint8_t *wsrc=m?w2:w1; const uint8_t *ssrc=m?sc2:sc1;
            size_t rb=I/2; uint8_t *sa=malloc(64*rb), *sb=malloc(64*rb); long long dummy=0;
            memcpy(sa,wsrc,64*rb);
            for(size_t i=0;i<32*rb;i++){ uint8_t lo=(nxt()&1)?8:0, hi=(nxt()&1)?8:0; sa[i]=(uint8_t)(lo|(hi<<4)); }
            canon(sb,sa,64*rb,&dummy);
            test_matrix(1,sa,sb,ssrc,I,64,ks_full,NK,as_full,NA);
            free(sa); free(sb); }
        fprintf(stderr,"done %s\n",argv[f]);
    }
    printf("{\"mode\":\"%s\",\"files\":%d,\"code8_fraction\":%.6f,\"act_sets\":[",argv[2],argc-3,(double)c8/(double)nib);
    for(int a=0;a<na;a++) printf("%s\"%s\"",a?",":"",AN[as[a]]);
    printf("],\"results\":{");
    for(int w=0;w<2;w++){ printf("%s\"%s\":{",w?",":"",w?"synthetic_zero_rows_all_act_sets":"real_matrices");
        int first=1;
        for(int k=0;k<NK;k++){ St *s=&st[w][k]; if(!s->cmp) continue;
            printf("%s\"%s\":{\"compared\":%lld,\"mismatch_value\":%lld,\"mismatch_signzero_only\":%lld,\"mismatch_nan_payload\":%lld,\"zero_outputs\":%lld,\"nan_outputs\":%lld}",
                first?"":",",KN[k],s->cmp,s->mis_val,s->mis_sz,s->mis_nan,s->zero_out,s->nan_out); first=0; }
        printf("}"); }
    printf("}}\n");
    return 0;
}
