/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/backend_cuda_dsv4.cu (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
// Device code copied from colibri dev tree backend_cuda_dsv4.cu (FP4 paths only).
// Changes: the fmt 8/9 (fp8/bf16) branches of mv/mv_rows/mm_batch are removed (FP4 never
// takes them; they need cuda_bf16.h), and a warp-per-row wrapper around v4f_dot_fp4 is added
// for V4.1 shapes (the engine calls it only inside v4f_moe_kernel at V4's H=4096/I=2048).
typedef unsigned char uint8_t;
typedef unsigned long long size_t_;
struct MvDesc { const uint8_t *w,*scale; const float *x; float *y; };          // :51
__constant__ float e8_table[256];                                                // :93
__device__ float e8(uint8_t b){return e8_table[b];}                              // :98
__device__ float f4(uint8_t q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}  // :99
template<int F> __global__ void mv(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){   // :100-108
    int o=blockIdx.x;if(o>=O)return;float sum=0;
    const float *xi=x+(long long)(o/(O/groups))*I;for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        sum+=xi[i]*v;
    }
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];
}
template<int F,int R> __global__ void mv_rows(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){  // :109-126
    int base=blockIdx.x*R;float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){const float *xi=x+(long long)(o/(O/groups))*I;float v;
            {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}sum[r]+=xi[i]*v;}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)y[base+r]=s[r][0];}
}
template<int R> __global__ void mv_fp4_grouped(const MvDesc *desc,int count,int O,int I){   // :127-143
    int matrix=blockIdx.y,base=blockIdx.x*R;if(matrix>=count)return;MvDesc d=desc[matrix];float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t p=d.w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;
            sum[r]+=d.x[i]*f4(q)*e8(d.scale[(long long)o*(I/32)+i/32]);}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)d.y[base+r]=s[r][0];}
}
template<int TB> __global__ void __launch_bounds__(256) mv_fp4_expert_major(const uint8_t*bank_w,const uint8_t*bank_s,size_t_ wstride,size_t_ sstride,   // :156-190
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    int slot=blockIdx.y,lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);
    if(o>=O)return;
    int e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    const uint8_t*w=bank_w+(size_t_)e*wstride+(size_t_)o*(I/2);const uint8_t*sc=bank_s+(size_t_)e*sstride+(size_t_)o*(I/32);
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;const float*xs[TB];float p[TB][8];
        #pragma unroll
        for(int t=0;t<TB;t++){xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
            #pragma unroll
            for(int q=0;q<8;q++)p[t][q]=0.f;}
        for(int c0=0;c0<I/32;c0+=8){
            #pragma unroll
            for(int q=0;q<8;q++){int i=lane+32*(c0+q);
                float xv[TB];
                #pragma unroll
                for(int t=0;t<TB;t++)xv[t]=t<n?xs[t][i]:0.f;
                uint8_t pk=w[i>>1],qq=(i&1)?pk>>4:pk&15;float f=f4(qq),es=e8(sc[i/32]);
                #pragma unroll
                for(int t=0;t<TB;t++)if(t<n)p[t][q]+=xv[t]*f*es;
            }
        }
        #pragma unroll
        for(int t=0;t<TB;t++){
            float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
            v0=v0+v2;v1=v1+v3;float v=v0+v1;
            #pragma unroll
            for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
            if(!lane&&t<n){long long route=routes[g+t];
                if(o<split)y0[route*ystride+o]=v;else y1[route*ystride+(o-split)]=v;}
        }
    }
}
template<int F> __global__ void mm_batch(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){   // :226-243
    int o=blockIdx.x;if(o>=O)return;int tbase=blockIdx.y*16,tcount=T-tbase<16?T-tbase:16;
    float sum[16]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)sum[t]+=x[(long long)(tbase+t)*I+i]*v;
    }
    __shared__ float s[16][256];
    #pragma unroll
    for(int t=0;t<16;t++)s[t][threadIdx.x]=sum[t];
    __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int t=0;t<16;t++)s[t][threadIdx.x]+=s[t][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x)for(int t=0;t<tcount;t++)y[(long long)(tbase+t)*O+o]=s[t][0];
}
__device__ __forceinline__ float v4f_tree(float p[8]){                         // :2250-2257
    #pragma unroll
    for(int k=16;k;k>>=1){
        #pragma unroll
        for(int r=0;r<8;r++)p[r]=__fadd_rn(p[r],__shfl_down_sync(0xffffffffu,p[r],k));}
    p[0]=__fadd_rn(p[0],p[4]);p[1]=__fadd_rn(p[1],p[5]);p[2]=__fadd_rn(p[2],p[6]);p[3]=__fadd_rn(p[3],p[7]);
    p[0]=__fadd_rn(p[0],p[2]);p[1]=__fadd_rn(p[1],p[3]);
    return __fadd_rn(p[0],p[1]);}
__device__ __forceinline__ float v4f_e8f(unsigned b){ return b? __uint_as_float(b<<23) : __uint_as_float(0x00400000u); }   // :2242
__device__ __forceinline__ float v4f_dot_fp4(const uint8_t* __restrict__ w,const uint8_t* __restrict__ srow,const float* xs,const float* lut,int I,int lane){  // :2279-2289
    float p[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(int m=0;m<I;m+=256){
        int i=m+8*lane;
        unsigned u=__ldcs((const unsigned*)(w+(i>>1)));
        float sc=v4f_e8f(__ldg(srow+(i>>5)));
        const float4 *xp=(const float4*)(xs+i); float4 xa=xp[0],xb=xp[1];
        const float xv[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int r=0;r<8;r++)p[r]=__fadd_rn(p[r],__fmul_rn(__fmul_rn(xv[r],lut[(u>>(4*r))&15]),sc));}
    return v4f_tree(p);}
// wrapper (added): one warp per row, lut and x staged in shared memory like v4f_moe_kernel (:2311-2315)
extern "C" __global__ void v4f_rows(const uint8_t* w,const uint8_t* s,const float* x,float* y,int I,int O){
    __shared__ __align__(16) float xs[5120]; __shared__ float lut[16];
    if(threadIdx.x<16){ const float t[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6}; lut[threadIdx.x]=t[threadIdx.x]; }
    for(int i=threadIdx.x;i<I;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    int lane=threadIdx.x&31, wid=threadIdx.x>>5; int o=blockIdx.x*(blockDim.x/32)+wid; if(o>=O) return;
    float p=v4f_dot_fp4(w+(size_t_)o*(I/2), s+(size_t_)o*(I/32), xs, lut, I, lane);
    if(!lane) y[o]=p;
}
