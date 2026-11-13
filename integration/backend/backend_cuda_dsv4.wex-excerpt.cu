/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/backend_cuda_dsv4.cu (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: KAIRA src/colibri/c/backend_cuda_dsv4.cu at commit 88ddb07 (2026-11-07), as committed.
 * Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
 * DSV4 CUDA backend exports GPU-direct needs, and DSV4_CUDA_STREAM_PRIORITY.
 * Also apply backend-upload-race-fix.patch (next to this file) on a tree that lacks it.
 */

/* ==== backend_cuda_dsv4.cu:1935-1941  inference_stream_priority (DSV4_CUDA_STREAM_PRIORITY) and its use in dsv4_cuda_init ==== */
/* DSV4_CUDA_STREAM_PRIORITY=1: the inference streams (stream, aux) take the current
 * device's highest priority. Background kernels on default-priority streams (the WEX
 * helper's WNR2 decodes, the copy stream) then yield the SMs to attention/mHC/MoE as
 * their blocks retire instead of running whole grids ahead of them. Unset or 0 keeps
 * the default priority (the historical behaviour). */
static int inference_stream_priority(){const char*s=getenv("DSV4_CUDA_STREAM_PRIORITY");int least=0,greatest=0;if(!s||!atoi(s)||cudaDeviceGetStreamPriorityRange(&least,&greatest)!=cudaSuccess){(void)cudaGetLastError();return 0;}static int said;if(!said){said=1;fprintf(stderr,"[DSV4 CUDA] inference streams at priority %d (range %d..%d)\n",greatest,least,greatest);}return greatest;}
extern "C" int dsv4_cuda_init(const int *dev,int n){if(n<1||n>16)return 0;float scale[256];for(int b=0;b<256;b++)scale[b]=b?ldexpf(1.f,b-127):ldexpf(1.f,-127);ng=0;for(int i=0;i<n;i++){cudaDeviceProp p{};if(!ok(cudaSetDevice(dev[i]),"select device")||!ok(cudaMemcpyToSymbol(e8_table,scale,sizeof(scale)),"scale table upload")||!ok(cudaGetDeviceProperties(&p,dev[i]),"device properties")||!ok(cudaStreamCreateWithPriority(&g[i].stream,cudaStreamNonBlocking,inference_stream_priority()),"stream")||!ok(cudaStreamCreateWithPriority(&g[i].aux,cudaStreamNonBlocking,inference_stream_priority()),"aux stream")||!ok(cudaStreamCreateWithFlags(&g[i].copy,cudaStreamNonBlocking),"copy stream")||!ok(cudaEventCreateWithFlags(&g[i].last_read,cudaEventDisableTiming),"last-read event")||!ok(cudaEventCreateWithFlags(&g[i].ready,cudaEventDisableTiming),"device event")||!ok(cudaEventCreateWithFlags(&g[i].fork,cudaEventDisableTiming),"fork event")||!ok(cudaEventCreateWithFlags(&g[i].join,cudaEventDisableTiming),"join event")||!tc_dev_init(&g[i])||!ok(cudaMalloc(&g[i].decode_state,2*sizeof(int)),"decode state allocation"))return 0;g[i].id=dev[i];g[i].sms=p.multiProcessorCount;ng++;fprintf(stderr,"[DSV4 CUDA] device %d: %s %.1f GB sm_%d%d\n",dev[i],p.name,p.totalGlobalMem/1e9,p.major,p.minor);}for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(i!=j){int can=0;if(!ok(cudaDeviceCanAccessPeer(&can,dev[i],dev[j]),"peer query"))return 0;if(can){if(!ok(cudaSetDevice(dev[i]),"select peer device"))return 0;cudaError_t e=cudaDeviceEnablePeerAccess(dev[j],0);if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)return ok(e,"enable peer access");}}

/* ==== backend_cuda_dsv4.cu:2369-2384  dsv4_cuda_tensor_alloc_fp4, dsv4_cuda_wex_direct_available, dsv4_cuda_tensor_fp4_buffers ==== */
/* WEX direct: an fp4 mirror whose bytes another producer writes on the device
 * (kaira_wex_cuda.dll reconstructs a compressed expert straight into it), so a
 * GPU-destined expert never goes through a host slab. Owned buffers, like a
 * refillable upload; fp4 keeps no derived data, so writing w and scale is all. */
extern "C" int dsv4_cuda_tensor_alloc_fp4(Dsv4CudaTensor **pt,int O,int I,int d){
    if(!pt||*pt||O<1||I<32||I%32)return 0;Dev*c=ctx(d);if(!c||!ok(cudaSetDevice(d),"select fp4 alloc device"))return 0;
    Dsv4CudaTensor*t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));if(!t)return 0;
    size_t wb=(size_t)O*I/2,sb=(size_t)O*I/32;t->O=O;t->I=I;t->device=d;t->fmt=4;t->bytes=wb+sb;
    if(!ok(cudaMalloc(&t->w,wb),"fp4 weight allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_w=1;
    if(!ok(cudaMalloc(&t->scale,sb),"fp4 scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_scale=1;
    *pt=t;return 1;}
/* Linked directly (no Windows loader): the exports above are always present. */
extern "C" int dsv4_cuda_wex_direct_available(void){return 1;}
extern "C" int dsv4_cuda_tensor_fp4_buffers(const Dsv4CudaTensor *t,void **w,void **scale,int *O,int *I){
    if(!t||t->fmt!=4||!t->w||!t->scale)return 0;
    if(w)*w=t->w;if(scale)*scale=t->scale;if(O)*O=t->O;if(I)*I=t->I;return 1;}

