/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/backend_loader_dsv4.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: KAIRA src/colibri/c/backend_loader_dsv4.c at commit 88ddb07 (2026-11-07), as committed.
 * Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
 * Windows loader of coli_cuda_dsv4*.dll: optional symbols, so an older DLL falls back to the host slab path.
 */

/* ==== backend_loader_dsv4.c:181-183  typedefs ==== */
typedef int             (*fn_stream_drain)(int device);
typedef int             (*fn_tensor_alloc_fp4)(Dsv4CudaTensor **t,int O,int I,int device);
typedef int             (*fn_tensor_fp4_buffers)(const Dsv4CudaTensor *t,void **w,void **scale,int *O,int *I);

/* ==== backend_loader_dsv4.c:349-351  g_dsv4 fields ==== */
    fn_stream_drain stream_drain;                /* optional (older DLLs) */
    fn_tensor_alloc_fp4 tensor_alloc_fp4;        /* optional: WEX direct-to-VRAM */
    fn_tensor_fp4_buffers tensor_fp4_buffers;    /* optional: WEX direct-to-VRAM */

/* ==== backend_loader_dsv4.c:583-586  GetProcAddress ==== */
    g_dsv4.stream_drain = (fn_stream_drain)GetProcAddress(g_dsv4.dll, "dsv4_cuda_stream_drain");
    /* optional: WEX3 experts decoded straight into a mirror; absent = host slab path */
    g_dsv4.tensor_alloc_fp4 = (fn_tensor_alloc_fp4)GetProcAddress(g_dsv4.dll, "dsv4_cuda_tensor_alloc_fp4");
    g_dsv4.tensor_fp4_buffers = (fn_tensor_fp4_buffers)GetProcAddress(g_dsv4.dll, "dsv4_cuda_tensor_fp4_buffers");

/* ==== backend_loader_dsv4.c:1001-1023  wrappers: alloc_fp4, fp4_buffers, wex_direct_available (needs all three exports), stream_drain ==== */
int dsv4_cuda_tensor_alloc_fp4(Dsv4CudaTensor **t,int O,int I,int device){
    if(!g_dsv4.available || !g_dsv4.tensor_alloc_fp4) return 0;
    return g_dsv4.tensor_alloc_fp4(t,O,I,device);
}

int dsv4_cuda_tensor_fp4_buffers(const Dsv4CudaTensor *t,void **w,void **scale,int *O,int *I){
    if(!g_dsv4.available || !g_dsv4.tensor_fp4_buffers) return 0;
    return g_dsv4.tensor_fp4_buffers(t,w,scale,O,I);
}

int dsv4_cuda_wex_direct_available(void){
    return g_dsv4.available && g_dsv4.tensor_alloc_fp4 && g_dsv4.tensor_fp4_buffers && g_dsv4.stream_drain;
}

int dsv4_cuda_stream_drain(int device){
    /* Without the DLL there is never in-flight DMA to wait for, so a drain
     * trivially succeeds. With an older DLL that lacks the export, report
     * failure instead: the engine probes this once and keeps every attach
     * synchronous, so nothing is ever enqueued that could not be drained. */
    if(!g_dsv4.available) return 1;
    if(!g_dsv4.stream_drain) return 0;
    return g_dsv4.stream_drain(device);
}

