/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef WEX_FUZZ_CUDA_DEVICE_H
#define WEX_FUZZ_CUDA_DEVICE_H
/* What c/wex_cuda.cuh takes from the CUDA toolkit, for clang's CUDA mode without the toolkit, so that its
 * kernels compile to PTX on a machine that has an NVIDIA driver and no CUDA toolkit:
 *
 *   clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_86 -nocudainc -nocudalib -O2 -S cuda_device.cu
 *
 * (cuda_device.cu includes c/wex_cuda.cuh, whose `#include <cuda_runtime.h>` finds a one-line stub that includes
 * this file; wex4/fuzz/run.sh --gpu writes it). The NVIDIA driver compiles the PTX for the GPU when the harness
 * loads it (cuda_gpu.h). clang defines __CUDA_ARCH__ from --cuda-gpu-arch, as nvcc does from -arch.
 *
 * Device side, construct by construct:
 *   - the qualifiers are clang's CUDA attributes; threadIdx, blockIdx, blockDim and gridDim are clang's own
 *     (__clang_cuda_builtin_vars.h, which reads the PTX special registers); __syncthreads() is clang's builtin
 *     (bar.sync 0);
 *   - __ldg is ld.global.nc through clang's __nvvm_ldg builtins, for the three types the kernels read with it
 *     (uint32_t, uint16_t, uint4);
 *   - __byte_perm is prmt (clang's __nvvm_prmt), __funnelshift_l is shf.l.wrap.b32, __vmaxu2 and __vminu2 are
 *     vmax2 and vmin2 on unsigned 16-bit halves, __popc is popc: PTX instructions that compute what the CUDA
 *     documentation says of each (the code nvcc makes of them was not compared);
 *   - atomicOr and atomicExch are atom.or and atom.exch on a 32-bit word (clang's __nvvm_atom_or_gen_i and
 *     __nvvm_atom_xchg_gen_i; atomicExch is used by the WNR1 kernel only); __cvta_generic_to_shared is the
 *     address-space conversion to shared memory; min and max are the overloads of CUDA's math API;
 *   - the inline PTX of wex_cuda.cuh (prmt with the sign selector, cp.async, its commit and wait) is not
 *     touched: clang passes it to the PTX as nvcc does.
 * Host side: only declarations, so that the host functions of wex_cuda.cuh parse; a device-only compile emits
 * none of them. The harness computes the plan, the offsets and the table layout with the host build of the
 * same source (cuda_host.h) and launches the kernels through the CUDA driver API (cuda_gpu.h).
 *
 * What this does not give: nvcc's front end and its device compiler (the production helper is built by nvcc;
 * here LLVM's NVPTX back end writes the PTX and the driver's JIT compiles it), and the CUDA runtime that the
 * bridge c/tools/wex_cuda_bridge.cu calls. */
#ifndef __CUDA__
#error "cuda_device.h is for clang's CUDA mode (clang++ -x cuda --cuda-device-only)"
#endif
#ifndef __CUDACC__
#define __CUDACC__ 1   /* c/wex_neural_rans.h makes its hash helpers __host__ __device__ under it */
#endif
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#include <stddef.h>
#include <stdint.h>
#include <__clang_cuda_builtin_vars.h>

/* ---- types ------------------------------------------------------------------------------------- */
struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    __host__ __device__ dim3(unsigned int a = 1u, unsigned int b = 1u, unsigned int c = 1u) : x(a), y(b), z(c) {}
};
struct __attribute__((aligned(16))) uint4 { unsigned int x, y, z, w; };
static __host__ __device__ __forceinline__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z,
                                                            unsigned int w) {
    uint4 r;
    r.x = x; r.y = y; r.z = z; r.w = w;
    return r;
}

/* ---- host: declarations only ------------------------------------------------------------------- */
enum cudaError { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
typedef enum cudaError cudaError_t;
typedef struct CUstream_st *cudaStream_t;
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor = 75 };
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
cudaError_t cudaGetLastError(void);
cudaError_t cudaMemsetAsync(void *to, int value, size_t bytes, cudaStream_t stream);
template<class F> cudaError_t cudaFuncSetAttribute(F *kernel, cudaFuncAttribute attr, int value);
/* what clang calls for a <<<...>>> launch in host code */
extern "C" unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared = 0, void *stream = 0);
extern "C" int cudaConfigureCall(dim3 grid, dim3 block, size_t shared = 0, cudaStream_t stream = 0);

/* ---- device intrinsics --------------------------------------------------------------------------- */
static __device__ __forceinline__ uint32_t __ldg(const uint32_t *p) { return __nvvm_ldg_ui(p); }
static __device__ __forceinline__ uint16_t __ldg(const uint16_t *p) { return __nvvm_ldg_us(p); }
static __device__ __forceinline__ uint4 __ldg(const uint4 *p) {
    typedef unsigned int v4 __attribute__((ext_vector_type(4)));
    v4 v = __nvvm_ldg_ui4((const v4 *)p);
    return make_uint4(v.x, v.y, v.z, v.w);
}
static __device__ __forceinline__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    return __nvvm_prmt(x, y, s);
}
static __device__ __forceinline__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
    unsigned int r;
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(shift));
    return r;
}
static __device__ __forceinline__ unsigned int __vmaxu2(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("vmax2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));
    return r;
}
static __device__ __forceinline__ unsigned int __vminu2(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("vmin2.u32.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0u));
    return r;
}
static __device__ __forceinline__ int __popc(unsigned int x) { return __builtin_popcount(x); }
static __device__ __forceinline__ uint32_t atomicOr(uint32_t *address, uint32_t value) {
    return (uint32_t)__nvvm_atom_or_gen_i((int *)address, (int)value);
}
static __device__ __forceinline__ uint32_t atomicExch(uint32_t *address, uint32_t value) {   /* the WNR1 kernel */
    return (uint32_t)__nvvm_atom_xchg_gen_i((int *)address, (int)value);
}
static __device__ __forceinline__ size_t __cvta_generic_to_shared(const void *p) {
    return (size_t)(const void __attribute__((address_space(3))) *)p;
}
static __device__ __forceinline__ int min(int a, int b) { return a < b ? a : b; }
static __device__ __forceinline__ int max(int a, int b) { return a > b ? a : b; }
static __device__ __forceinline__ unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }
static __device__ __forceinline__ unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }
static __device__ __forceinline__ unsigned int min(int a, unsigned int b) { return min((unsigned int)a, b); }
static __device__ __forceinline__ unsigned int min(unsigned int a, int b) { return min(a, (unsigned int)b); }
static __device__ __forceinline__ unsigned int max(int a, unsigned int b) { return max((unsigned int)a, b); }
static __device__ __forceinline__ unsigned int max(unsigned int a, int b) { return max(a, (unsigned int)b); }

#endif /* WEX_FUZZ_CUDA_DEVICE_H */
