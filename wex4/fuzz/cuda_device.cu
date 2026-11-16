/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* The kernels of c/wex_cuda.cuh as PTX, for the GPU mode of fuzz_wnr3_cuda.cpp (cuda_gpu.h): compiled by clang
 * without the CUDA toolkit (cuda_device.h says how), once for each target the harness loads:
 *
 *   clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_86 -nocudainc -nocudalib -O2 -S \
 *       -I OUT -I c -I wex4/fuzz wex4/fuzz/cuda_device.cu -o OUT/sm86.ptx
 *
 * where OUT holds the stub cuda_runtime.h that includes cuda_device.h (wex4/fuzz/run.sh --gpu writes both). The
 * kernels are those of c/wex_cuda.cuh, unchanged; this file only adds a kernel that reports the layout of the
 * structures the host passes by value, and takes the address of every kernel the harness launches, so that
 * clang emits them. */
#include "wex_cuda.cuh"

/* sizes and offsets of the structures passed by value, as the device sees them; the harness compares them with
 * its host build before it launches anything */
static __global__ void wex_gpu_layout(uint32_t *out) {
    out[0] = (uint32_t)sizeof(WexCudaNr2Plan);
    out[1] = (uint32_t)offsetof(WexCudaNr2Plan, model_id);
    out[2] = (uint32_t)offsetof(WexCudaNr2Plan, word24);
    out[3] = (uint32_t)sizeof(WexCudaNr2Tables);
    out[4] = (uint32_t)offsetof(WexCudaNr2Tables, scale_symbol);
    out[5] = (uint32_t)sizeof(WexCudaCommit);
    out[6] = (uint32_t)offsetof(WexCudaCommit, bytes);
    out[7] = (uint32_t)__CUDA_ARCH__;
}

void *wex_gpu_kernels[] = {
    (void *)wex_cuda_nr2_build_rows, (void *)wex_cuda_nr2_build_scales, (void *)wex_cuda_nr2_build_buckets,
    (void *)wex_cuda_nr2_scales, (void *)wex_cuda_nr2_weights<0>, (void *)wex_cuda_nr2_weights<1>,
    (void *)wex_cuda_commit_regions, (void *)wex_gpu_layout,
};
