"""Generate lab-only wex_cuda_ceil.cuh and wex_cuda_bridge_ceil.cu in lab/wnr3-5090-20261107/ceil-kaira."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r'<KAIRA>\src\colibri\c'

with open(os.path.join(SRC, 'wex_cuda.cuh'), 'r', encoding='utf-8') as f:
    cuh = f.read()

ceil_helpers = """
/* ---- WNR3 ceil-form inverse (lab prototype) -----------------------------
 * Transformed blocks are identified by decoded running max magnitude == 5;
 * their magnitudes 0..5 map back to 0,2,4,5,6,7 (sign bit 8 kept) and their
 * scale byte is decremented by 1 into the final scale slab. */
static __device__ __forceinline__ uint32_t wex_cuda_ceil_dbl8(uint32_t w) {
    const uint32_t m = w & 0x77777777u;
    const uint32_t g = ((m >> 1) | (m >> 2)) & 0x11111111u;          /* 1 per nibble iff m >= 2 */
    const uint32_t mn = (g << 1) | (m & 0x11111111u & ~g);            /* min(m, 2) */
    return (w & 0x88888888u) | (m + mn);
}
static __device__ __forceinline__ void wex_cuda_ceil_fix_scales(
    const unsigned char *sc_in, unsigned char *sc_out, uint64_t emask, uint32_t blocks) {
    const uint4 *s4 = (const uint4 *)sc_in;
    uint4 *o4 = (uint4 *)sc_out;
    uint32_t qmax = blocks >> 4;
#pragma unroll
    for (uint32_t q = 0; q < 4u; q++) {
        if (q < qmax) {
            const uint32_t m16 = (uint32_t)(emask >> (16u * q)) & 0xffffu;
            uint4 v = s4[q];
            if (m16) {
                v.x -= ((m16 & 15u) * 0x00204081u) & 0x01010101u;
                v.y -= (((m16 >> 4) & 15u) * 0x00204081u) & 0x01010101u;
                v.z -= (((m16 >> 8) & 15u) * 0x00204081u) & 0x01010101u;
                v.w -= ((m16 >> 12) * 0x00204081u) & 0x01010101u;
            }
            o4[q] = v;
        }
    }
    for (uint32_t b = qmax * 16u; b < blocks; b++) {
        sc_out[b] = (unsigned char)(sc_in[b] - ((uint32_t)(emask >> b) & 1u));
    }
}
"""

target_plan = "    plan->weight_base = (entry->scale_bytes + 15u) & ~15u;   /* uint4 stores */"
repl_plan = """    uint32_t sc_aligned = (entry->scale_bytes + 15u) & ~15u;
    plan->weight_base = sc_aligned * 2u;   /* [0, sc_aligned) = final scales, [sc_aligned, 2*sc_aligned) = coded ceil scales */"""
assert target_plan in cuh
cuh = cuh.replace(target_plan, repl_plan, 1)

target_scales_launch = """    wex_cuda_nr2_scales<<<plan->block_base[0][3], WEX_CUDA_NR2_THREADS, WEX_CUDA_NR2_SCALE_SMEM, stream>>>(
        record, offsets, *plan, *tables, raw, error);"""
repl_scales_launch = """    wex_cuda_nr2_scales<<<plan->block_base[0][3], WEX_CUDA_NR2_THREADS, WEX_CUDA_NR2_SCALE_SMEM, stream>>>(
        record, offsets, *plan, *tables, raw + plan->weight_base / 2u, error);"""
assert target_scales_launch in cuh
cuh = cuh.replace(target_scales_launch, repl_scales_launch, 1)

target_weights = """    const unsigned char *scales = raw + plan.scale_offset[m] + first / 32u;
    uint4 *out = (uint4 *)(raw + plan.weight_offset[m] + first / 2u);"""
repl_weights = """    const unsigned char *scales = raw + plan.weight_base / 2u + plan.scale_offset[m] + first / 32u;
    unsigned char *scales_out = raw + plan.scale_offset[m] + first / 32u;
    uint4 *out = (uint4 *)(raw + plan.weight_offset[m] + first / 2u);"""
assert target_weights in cuh
cuh = cuh.replace(target_weights, repl_weights, 1)

target_loop = """    uint32_t scale = scales[0];
#pragma unroll 1
    for (uint32_t block = 0; block < blocks; block++) {"""
repl_loop = """    uint32_t scale = scales[0];
    uint64_t emask = 0;
#pragma unroll 1
    for (uint32_t block = 0; block < blocks; block++) {"""
assert target_loop in cuh
cuh = cuh.replace(target_loop, repl_loop, 1)

target_store = """        out[block] = make_uint4(word[0], word[1], word[2], word[3]);
        scale = next_scale;
    }
    if (s.pos != end || state != WEX_NR_L) atomicOr(error, 1u);"""
repl_store = """        bool ev = (maximum == 5u);
        if (ev) {
#pragma unroll
            for (int w_ = 0; w_ < 4; w_++) word[w_] = wex_cuda_ceil_dbl8(word[w_]);
            emask |= (uint64_t)1u << block;
        }
        out[block] = make_uint4(word[0], word[1], word[2], word[3]);
        scale = next_scale;
    }
    wex_cuda_ceil_fix_scales(scales, scales_out, emask, blocks);
    if (s.pos != end || state != WEX_NR_L) atomicOr(error, 1u);"""
assert target_store in cuh
cuh = cuh.replace(target_store, repl_store, 1)

target_kb = "/* Kernel B: weight streams."
assert target_kb in cuh
cuh = cuh.replace(target_kb, ceil_helpers + "\n" + target_kb, 1)

out_cuh = os.path.join(HERE, 'wex_cuda_ceil.cuh')
with open(out_cuh, 'w', encoding='utf-8') as f:
    f.write(cuh)

with open(os.path.join(SRC, 'tools', 'wex_cuda_bridge.cu'), 'r', encoding='utf-8') as f:
    br = f.read()

br = br.replace('#include "../wex_cuda.cuh"', '#include "wex_cuda_ceil.cuh"')
br = br.replace('workspace.ensure((size_t)aligned,',
                'workspace.ensure((size_t)aligned + ((entry->scale_bytes + 4095u) & ~4095u),')
br = br.replace('if((size_t)aligned>max_aligned) max_aligned=(size_t)aligned;',
                'size_t need=(size_t)aligned+((entry->scale_bytes+4095u)&~4095u); if(need>max_aligned) max_aligned=need;')

out_br = os.path.join(HERE, 'wex_cuda_bridge_ceil.cu')
with open(out_br, 'w', encoding='utf-8') as f:
    f.write(br)

print("Generated:", out_cuh, out_br)
