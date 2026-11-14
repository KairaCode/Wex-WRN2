/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/backend_cuda_dsv4.h (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* REFERENCE EXCERPT, not compilable on its own.
 * Source: the maintainer's engine tree, src/colibri/c/backend_cuda_dsv4.h as of 2026-11-07, as committed.
 * Line numbers are of that version; the live tree moves them. Code between ranges is omitted.
 * Declarations of the GPU-direct backend exports.
 */

/* ==== backend_cuda_dsv4.h:33-39  dsv4_cuda_tensor_alloc_fp4, dsv4_cuda_tensor_fp4_buffers, dsv4_cuda_wex_direct_available ==== */
int dsv4_cuda_copy_stats(double *out);
/* WEX direct (optional, newer DLLs): allocate an fp4 mirror without copying, and
 * expose its device buffers to another producer (kaira_wex_cuda.dll). Return 1
 * on success, 0 when unavailable; dsv4_cuda_wex_direct_available reports both. */
int dsv4_cuda_tensor_alloc_fp4(Dsv4CudaTensor **t,int O,int I,int device);
int dsv4_cuda_tensor_fp4_buffers(const Dsv4CudaTensor *t,void **w,void **scale,int *O,int *I);
int dsv4_cuda_wex_direct_available(void);

