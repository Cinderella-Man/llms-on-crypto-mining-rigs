// The kernels that one Flint MoE layer runs at one token in this project's llama.cpp fork (branch pascal-p106-v2) on a
// compute 6.1 card, written as device functions that both the multi-launch baseline and the persistent kernel call, so
// the two paths do the same arithmetic in the same order.
//
// Copied from the fork (MIT, Copyright (c) 2023-2026 The ggml authors), paths of a compute 6.1 card with one token:
//   ggml/src/ggml-cuda/norm.cu                    rms_norm_f32<512, 1, vec, multiply>  (RMS_NORM+MUL, 2,048 columns)
//   ggml/src/ggml-cuda/mmvf.cu                    mul_mat_vec_f<T, float, 1, 256>      (router; compute 6.1 accumulates
//                                                                                       F16 weights in FP32)
//   ggml/src/ggml-cuda/topk-moe.cu                topk_moe_cuda<256, false>            (sigmoid, top-6, normalisation)
//   ggml/src/ggml-cuda/quantize.cu                quantize_q8_1
//   ggml/src/ggml-cuda/mmvq-pascal.cu             mul_mat_vec_q_pascal                 (fork's Pascal kernels with FP32
//                                                                                       activation sums)
//   ggml/src/ggml-cuda/moe-weighted-reduction.cu  moe_weighted_reduction_f32
//   ggml/src/ggml-cuda/binbcast.cu                op_add
// Only the code these paths execute is kept, unchanged in arithmetic and order. Compile with -use_fast_math, as
// llama.cpp does.
#pragma once

#include "../../mmvq_pascal/src/common.cuh"

#include <cfloat>
#include <type_traits>

namespace pk {

using mb::block_q8_1;

// ---- common.cuh ------------------------------------------------------------------------------------------------------

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// unary.cuh
static __device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

// binbcast.cu
static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

// ---- norm.cu: rms_norm_f32 with do_vec and do_multiply ------------------------------------------------------------

// One thread's partial sum of squares over its float4 (vpt = 1).
static __device__ __forceinline__ float rms_norm_partial(const float4 xs) {
    float tmp = 0.0f;
    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    return tmp;
}

static __device__ __forceinline__ float rms_norm_scale(const float tmp, const int ncols, const float eps) {
    return rsqrtf(tmp / ncols + eps);
}

// One output value: scale * x, then times the norm weight (two roundings, as the kernel's float4 code does per component).
static __device__ __forceinline__ float rms_norm_apply1(const float scale, const float x, const float m) {
    float r = scale*x;
    r *= m;
    return r;
}

static __device__ __forceinline__ float4 rms_norm_apply(const float scale, const float4 xs, const float4 m) {
    return make_float4(rms_norm_apply1(scale, xs.x, m.x), rms_norm_apply1(scale, xs.y, m.y), rms_norm_apply1(scale, xs.z, m.z),
                       rms_norm_apply1(scale, xs.w, m.w));
}

// rms_norm_f32<block_size = 512, vpt = 1, do_vec = true, do_multiply = true> for one row of ncols = 4 * 512 floats.
// block_reduce<SUM, 512>: warp sums, then one warp sums the 16 warp sums.
static __global__ void k_rms_norm_mul(const float * x, float * dst, const int ncols, const float eps, const float * mul) {
    constexpr int block_size = 512;
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
    const float4 * x4   = (const float4 *) x;
    const float4 * mul4 = (const float4 *) mul;
    float4       * dst4 = (float4 *) dst;
    const float4 xs = x4[tid];
    float tmp = rms_norm_partial(xs);
    tmp = warp_reduce_sum(tmp);
    if (tid % 32 == 0) {
        s_sum[tid / 32] = tmp;
    }
    __syncthreads();
    tmp = 0.0f;
    if (tid % 32 < block_size / 32) {
        tmp = s_sum[tid % 32];
    }
    tmp = warp_reduce_sum(tmp);
    const float scale = rms_norm_scale(tmp, ncols, eps);
    dst4[tid] = rms_norm_apply(scale, xs, mul4[tid]);
}

// ---- mmvf.cu: mul_mat_vec_f<T, type_acc = float, ncols_dst = 1, block_size = 256> ------------------------------------

// One thread's partial dot product of a weight row (T = half or float) with the F32 activations, which y2(col2) returns
// as a float2: columns 2 * col2 and 2 * col2 + 1 for col2 = tid, tid + block_size, ... NC2 > 0 makes the column count a
// compile-time constant (the persistent kernel), so the loop unrolls and its loads can be issued together, and
// LDG reads the weights through the read-only path; the arithmetic is the same.
template <typename T, int NC2 = 0, bool LDG = false, typename Y2>
static __device__ __forceinline__ float mmvf_partial(const T * x, const Y2 & y2, const int ncols2_rt, const int tid, const int block_size) {
    const int ncols2 = NC2 > 0 ? NC2 : ncols2_rt;
    float sumf = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
#pragma unroll
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = LDG ? __ldg(x2 + col2) : x2[col2];
            const float2 tmpy = y2(col2);
            sumf += tmpx.x*tmpy.x;
            sumf += tmpx.y*tmpy.y;
        }
    } else {
        const half2 * x2 = (const half2 *) x;
#pragma unroll
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = __half22float2(LDG ? __ldg(x2 + col2) : x2[col2]);
            const float2 tmpy = y2(col2);
            sumf += tmpx.x*tmpy.x;
            sumf += tmpx.y*tmpy.y;
        }
    }
    return sumf;
}

// One block of 256 threads per row; dst[row] = sum.
template <typename T>
static __global__ void k_router(const T * x, const float * y, float * dst, const int ncols2, const int stride_row) {
    constexpr int block_size = 256;
    __shared__ float buf_iw[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x += (int64_t) row * stride_row;
    if (tid < 32) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();
    const float2 * y2 = (const float2 *) y;
    float sumf = mmvf_partial<T>(x, [&](const int c) { return y2[c]; }, ncols2, tid, block_size);
    sumf = warp_reduce_sum(sumf);
    buf_iw[tid / 32] = sumf;
    __syncthreads();
    if (tid < 32) {
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum(sumf);
    }
    __syncthreads();
    if (tid >= 1) {
        return;
    }
    dst[row] = sumf;
}

// ---- topk-moe.cu: topk_moe_cuda<n_experts = 256, has_bias = false>, sigmoid, with_norm -----------------------------

// One warp, one token. wt[i] holds the logit of expert lane + 32 i. Writes ids[0..n_expert_used) and weights.
template <int n_experts>
static __device__ __forceinline__ void topk_moe_warp(float (&wt)[n_experts / 32], float * weights, int32_t * ids,
                                                     const int n_expert_used, const float clamp_val, const float scale_val) {
    constexpr int experts_per_thread = n_experts / 32;
    const int lane = threadIdx.x % 32;

    // sigmoid_warp_inplace
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = 1.f / (1.f + expf(-wt[i]));
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * 32;
            if (wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, 32);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, 32);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }
        if ((max_expert & 31) == lane) {
            wt[max_expert / 32] = -INFINITY;
        }
        if ((k & 31) == lane) {
            output_weights[k / 32] = max_val;
        }
        if ((max_expert & 31) == lane) {
            ids[k] = max_expert;
            wt_sum += max_val;
        }
    }

    wt_sum              = warp_reduce_sum(wt_sum);
    wt_sum              = max(wt_sum, clamp_val);
    const float inv_sum = 1.0f / wt_sum;
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] *= inv_sum;
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * 32 + lane;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// Launched as llama.cpp does: blocks of (32, TOPK_MOE_ROWS_PER_BLOCK = 8), one warp per row.
static __global__ void k_topk_moe(const float * logits, float * weights, int32_t * ids, const int n_rows, const int n_expert_used,
                                  const float clamp_val, const float scale_val) {
    constexpr int n_experts = 256;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }
    logits  += n_experts * row;
    weights += n_expert_used * row;
    ids     += n_experts * row;
    float wt[n_experts / 32];
#pragma unroll
    for (int i = 0; i < n_experts; i += 32) {
        wt[i / 32] = logits[i + threadIdx.x];
    }
    __syncthreads();
    topk_moe_warp<n_experts>(wt, weights, ids, n_expert_used, clamp_val, scale_val);
}

// ---- quantize.cu: quantize_q8_1 -------------------------------------------------------------------------------------

// One value per lane; the 32 lanes of a warp hold Q8_1 block ib. y may point to global or shared memory.
static __device__ __forceinline__ void quantize_q8_1_lane(const float xi, block_q8_1 * y, const int64_t ib, const int iqs) {
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max(amax);
    sum  = warp_reduce_sum(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, sum);
}

// ne0 values per row (a multiple of 32, no padding needed at Flint's widths), rows along grid.y.
static __global__ void k_quantize_q8_1(const float * x, block_q8_1 * y, const int64_t ne00, const int64_t s01, const int64_t ne0) {
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    const int64_t i1     = blockIdx.y;
    const int64_t i_cont = i1 * ne0 + i0;
    const int64_t ib     = i_cont / 32;
    const int64_t iqs    = i_cont % 32;
    const float xi = i0 < ne00 ? x[i1 * s01 + i0] : 0.0f;
    quantize_q8_1_lane(xi, y, ib, (int) iqs);
}

// ---- moe-weighted-reduction.cu ---------------------------------------------------------------------------------------

// sum over the used experts of (expert output * expert scale) * routing weight, expert 0 first.
template <int N>
static __device__ __forceinline__ float moe_weighted_sum(const float (&v)[N], const float (&w)[N], const float * expert_scale,
                                                         const uint64_t first_row) {
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float sum = (v[0] * first_scale) * w[0];
#pragma unroll
    for (int expert = 1; expert < N; ++expert) {
        const float scale = expert_scale != nullptr ? expert_scale[first_row + expert] : 1.0f;
        sum += (v[expert] * scale) * w[expert];
    }
    return sum;
}

template <int N>
static __global__ void k_moe_weighted_reduction(const float * experts, const float * expert_scale, const float * weights, float * dst,
                                                const int64_t n_embd) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    const uint64_t first_row = (uint64_t) token * N;
    float v[N], w[N];
#pragma unroll
    for (int e = 0; e < N; ++e) {
        v[e] = experts[(first_row + e) * n_embd + col];
        w[e] = weights[first_row + e];
    }
    dst[token * n_embd + col] = moe_weighted_sum<N>(v, w, expert_scale, first_row);
}

// binbcast.cu, both operands contiguous with the same shape
static __global__ void k_add(const float * src0, const float * src1, float * dst, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = op_add(src0[i], src1[i]);
    }
}

// ---- mmvq-pascal.cu --------------------------------------------------------------------------------------------------

// Bytes and Q8_1 activation blocks per weight block (mb::T_* numbering).
template <int T> struct mmvqp_traits;
template <> struct mmvqp_traits<mb::T_Q4_0> { static constexpr int bs = 18; static constexpr int na = 1; };
template <> struct mmvqp_traits<mb::T_Q4_1> { static constexpr int bs = 20; static constexpr int na = 1; };
template <> struct mmvqp_traits<mb::T_Q5_0> { static constexpr int bs = 22; static constexpr int na = 1; };
template <> struct mmvqp_traits<mb::T_Q8_0> { static constexpr int bs = 34; static constexpr int na = 1; };

constexpr __host__ __device__ int mmvqp_gcd(const int a, const int b) {
    return b == 0 ? a : mmvqp_gcd(b, a % b);
}

// a: 4 unsigned bytes, b: 4 signed bytes
static __device__ __forceinline__ int mmvqp_dp4a_us(const uint32_t a, const int b, const int c) {
#if __CUDA_ARCH__ >= 610
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    const int8_t * b8 = (const int8_t *) &b;
    return c + (int) (a & 0xff) * b8[0] + (int) ((a >> 8) & 0xff) * b8[1] + (int) ((a >> 16) & 0xff) * b8[2] + (int) (a >> 24) * b8[3];
#endif
}

static __device__ __forceinline__ float mmvqp_h2f_lo(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w & 0xffff)));
}

static __device__ __forceinline__ float mmvqp_h2f_hi(const uint32_t w) {
    return __half2float(__ushort_as_half((unsigned short) (w >> 16)));
}

// Where the activations come from: global memory through the read-only path (as the fork reads them), shared memory,
// or global memory through L2 only (data written earlier by the same kernel, which the read-only path may not see).
enum act_src { SRC_LDG = 0, SRC_SHARED = 1, SRC_LDCG = 2 };

// NB bytes (a multiple of 4) aligned to AL bytes into 32-bit words.
template <int NB, int AL, int SRC>
static __device__ __forceinline__ void mmvqp_load_words(uint32_t * w, const char * p) {
    static_assert(NB % 4 == 0 && AL >= 4, "chunk must be 4-byte aligned");
    if constexpr (AL >= 16 && NB % 16 == 0) {
#pragma unroll
        for (int i = 0; i < NB/16; ++i) {
            const uint4 v = SRC == SRC_SHARED ? ((const uint4 *) p)[i] : SRC == SRC_LDCG ? __ldcg((const uint4 *) p + i) : __ldg((const uint4 *) p + i);
            w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
        }
    } else if constexpr (AL >= 8 && NB % 8 == 0) {
#pragma unroll
        for (int i = 0; i < NB/8; ++i) {
            const uint2 v = SRC == SRC_SHARED ? ((const uint2 *) p)[i] : SRC == SRC_LDCG ? __ldcg((const uint2 *) p + i) : __ldg((const uint2 *) p + i);
            w[2*i + 0] = v.x; w[2*i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            w[i] = SRC == SRC_SHARED ? ((const uint32_t *) p)[i] : SRC == SRC_LDCG ? __ldcg((const unsigned int *) p + i) : __ldg((const unsigned int *) p + i);
        }
    }
}

// NB bytes (a multiple of 4) from shared memory aligned to AL bytes.
template <int NB, int AL>
static __device__ __forceinline__ void mmvqp_lds_words(uint32_t * w, const char * p) {
    mmvqp_load_words<NB, AL, SRC_SHARED>(w, p);
}

// 32-bit word / 16-bit value at an even byte offset of a word array (offsets are compile-time constants after unrolling).
static __device__ __forceinline__ uint32_t mmvqp_word_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? w[off >> 2] : __byte_perm(w[off >> 2], w[(off >> 2) + 1], 0x5432);
}

static __device__ __forceinline__ uint32_t mmvqp_u16_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? (w[off >> 2] & 0xffff) : (w[off >> 2] >> 16);
}

// NA Q8_1 activation blocks, split into quants and float scales.
template <int NA>
struct mmvqp_act {
    int   u[NA][8];
    int   t[NA][8];  // Q5_0 only: t[j] = bytes a[j], a[8 + j], a[16 + j], a[24 + j] of the block
    float d[NA];     // d8 (Q8_0) or d8 / 16 (4- and 5-bit legacy types)
    float s[NA];     // with S = d8 * the sum of the int8 values: S (Q4_1), -8 * S (Q4_0), -16 * S (Q5_0)
};

// 4 x 4 byte transpose: out[j] = bytes j of x0, x1, x2, x3.
static __device__ __forceinline__ void mmvqp_transpose4(
        const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, int * out) {
    const uint32_t a = __byte_perm(x0, x1, 0x5140);
    const uint32_t b = __byte_perm(x0, x1, 0x7362);
    const uint32_t c = __byte_perm(x2, x3, 0x5140);
    const uint32_t d = __byte_perm(x2, x3, 0x7362);
    out[0] = (int) __byte_perm(a, c, 0x5410);
    out[1] = (int) __byte_perm(a, c, 0x7632);
    out[2] = (int) __byte_perm(b, d, 0x5410);
    out[3] = (int) __byte_perm(b, d, 0x7632);
}

template <int type, int NA, int SRC>
static __device__ __forceinline__ void mmvqp_load_act(mmvqp_act<NA> & a, const char * p) {
    constexpr int NB = NA * sizeof(block_q8_1);
    uint32_t w[NB / 4];
    mmvqp_load_words<NB, mmvqp_gcd(NB, 16), SRC>(w, p);
#pragma unroll
    for (int b = 0; b < NA; ++b) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a.u[b][k] = (int) w[9*b + 1 + k];
        }
        const float d8 = mmvqp_h2f_lo(w[9*b]);
        // The block sum the offsets and mins need, from the int8 values in FP32 (block_q8_1's FP16 sum overflows on
        // Flint's largest activations).
        float S = 0.0f;
        if constexpr (type == mb::T_Q4_0 || type == mb::T_Q4_1 || type == mb::T_Q5_0) {
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = ggml_cuda_dp4a(0x01010101, a.u[b][k], sum);
            }
            S = d8 * (float) sum;
        }
        if constexpr (type == mb::T_Q8_0) {
            a.d[b] = d8;
            a.s[b] = 0.0f;
        } else if constexpr (type == mb::T_Q4_0) {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = -8.0f * S;
        } else if constexpr (type == mb::T_Q5_0) {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = -16.0f * S;
            mmvqp_transpose4(a.u[b][0], a.u[b][2], a.u[b][4], a.u[b][6], a.t[b] + 0);
            mmvqp_transpose4(a.u[b][1], a.u[b][3], a.u[b][5], a.u[b][7], a.t[b] + 4);
        } else {
            a.d[b] = d8 * 0.0625f;
            a.s[b] = S;
        }
    }
}

// 16 * sum(code * activation) over one 32-weight block of 4-bit codes q0..q3.
static __device__ __forceinline__ int mmvqp_q4_isum16(
        const uint32_t q0, const uint32_t q1, const uint32_t q2, const uint32_t q3, const int * u) {
    int lo = 0;
    int hi = 0;
    lo = ggml_cuda_dp4a((int) (q0 & 0x0F0F0F0F), u[0], lo);
    hi = mmvqp_dp4a_us(q0 & 0xF0F0F0F0, u[4], hi);
    lo = ggml_cuda_dp4a((int) (q1 & 0x0F0F0F0F), u[1], lo);
    hi = mmvqp_dp4a_us(q1 & 0xF0F0F0F0, u[5], hi);
    lo = ggml_cuda_dp4a((int) (q2 & 0x0F0F0F0F), u[2], lo);
    hi = mmvqp_dp4a_us(q2 & 0xF0F0F0F0, u[6], hi);
    lo = ggml_cuda_dp4a((int) (q3 & 0x0F0F0F0F), u[3], lo);
    hi = mmvqp_dp4a_us(q3 & 0xF0F0F0F0, u[7], hi);
    return (lo << 4) + hi;
}

// C weight blocks (raw words) times the matching activation blocks, added to acc.
template <int type, int C>
static __device__ __forceinline__ float mmvqp_dot(const uint32_t * w, const mmvqp_act<C * mmvqp_traits<type>::na> & a, float acc) {
    constexpr int BS = mmvqp_traits<type>::bs;
#pragma unroll
    for (int b = 0; b < C; ++b) {
        const int o = b * BS;
        if constexpr (type == mb::T_Q4_1) {
            const uint32_t dm = mmvqp_word_at(w, o);
            const int isum = mmvqp_q4_isum16(mmvqp_word_at(w, o + 4), mmvqp_word_at(w, o + 8), mmvqp_word_at(w, o + 12),
                                             mmvqp_word_at(w, o + 16), a.u[b]);
            acc = fmaf((float) isum, mmvqp_h2f_lo(dm) * a.d[b], acc);
            acc = fmaf(mmvqp_h2f_hi(dm), a.s[b], acc);
        } else if constexpr (type == mb::T_Q4_0) {
            const float d = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
            const int isum = mmvqp_q4_isum16(mmvqp_word_at(w, o + 2), mmvqp_word_at(w, o + 6), mmvqp_word_at(w, o + 10),
                                             mmvqp_word_at(w, o + 14), a.u[b]);
            acc = fmaf(d, fmaf((float) isum, a.d[b], a.s[b]), acc);
        } else if constexpr (type == mb::T_Q5_0) {
            const float    d  = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
            const uint32_t qh = mmvqp_word_at(w, o + 2);
            int hsum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                hsum = ggml_cuda_dp4a((int) ((qh >> j) & 0x01010101), a.t[b][j], hsum);
            }
            const int isum = mmvqp_q4_isum16(mmvqp_word_at(w, o + 6), mmvqp_word_at(w, o + 10), mmvqp_word_at(w, o + 14),
                                             mmvqp_word_at(w, o + 18), a.u[b]) + (hsum << 8);
            acc = fmaf(d, fmaf((float) isum, a.d[b], a.s[b]), acc);
        } else { // Q8_0
            const float d = mmvqp_h2f_lo(mmvqp_u16_at(w, o));
            int isum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                isum = ggml_cuda_dp4a((int) mmvqp_word_at(w, o + 2 + 4*k), a.u[b][k], isum);
            }
            acc = fmaf((float) isum, d * a.d[b], acc);
        }
    }
    return acc;
}

// The launch shape the fork picks per type (mmvqp_choose): rows per lane group R, blocks per lane C; staged loads.
template <int type, bool gate> struct mmvqp_shape;
template <> struct mmvqp_shape<mb::T_Q4_1, false> { static constexpr int R = 2; static constexpr int C = 4; };
template <> struct mmvqp_shape<mb::T_Q4_1, true>  { static constexpr int R = 2; static constexpr int C = 2; };
template <> struct mmvqp_shape<mb::T_Q4_0, false> { static constexpr int R = 2; static constexpr int C = 2; };
template <> struct mmvqp_shape<mb::T_Q4_0, true>  { static constexpr int R = 2; static constexpr int C = 2; };
template <> struct mmvqp_shape<mb::T_Q5_0, false> { static constexpr int R = 2; static constexpr int C = 2; };
template <> struct mmvqp_shape<mb::T_Q5_0, true>  { static constexpr int R = 1; static constexpr int C = 2; };
template <> struct mmvqp_shape<mb::T_Q8_0, false> { static constexpr int R = 2; static constexpr int C = 2; };
template <> struct mmvqp_shape<mb::T_Q8_0, true>  { static constexpr int R = 1; static constexpr int C = 2; };

// lanes per row (log2) for K weights per row, as the fork chooses it
constexpr __host__ __device__ int mmvqp_lpr_log2(const int type, const int K, const int C) {
    const int chunks = K / 32 / C;
    int l = 0;
    while (l < 5 && chunks % (2 << l) == 0) {
        l++;
    }
    return l;
}

// Staging buffer of one warp, in bytes.
template <int type, int R, int M>
constexpr __host__ __device__ int mmvqp_stage_bytes() {
    return 32 * R * M * mmvqp_shape<type, M == 2>::C * mmvqp_traits<type>::bs;
}

// The rows of one warp of mul_mat_vec_q_pascal, as the fork computes them. The lane group grp (lpr lanes) owns rows
// row0 .. row0 + R - 1 of matrix x0 (and of x1 when M == 2: up in x0, gate in x1), with
// row0 = ((warp_index << (5 - lpr_log2)) + grp) * R; y is this channel's activations. On return every lane of a group
// holds the group's sums in acc; the function returns row0.
template <int type, int R, int C, bool staged, int M, int SRC>
static __device__ __forceinline__ int mmvqp_rows(
        const char * x0, const char * x1, const int nrows, const int row_bytes, const int lpr_log2, const int niter,
        const char * y, const int warp_index, const int lane, char * smw_warp, float (&acc)[M][R]) {
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int NA = C * mmvqp_traits<type>::na; // activation blocks per lane and iteration
    constexpr int YB = NA * sizeof(block_q8_1);    // activation bytes per lane and iteration
    constexpr int NU = (WB + 15) / 16;             // staged: 16-byte units per lane and row, the last one predicated

    const int lpr  = 1 << lpr_log2;
    const int sl   = lane & (lpr - 1);
    const int grp  = lane >> lpr_log2;
    const int row0 = ((warp_index << (5 - lpr_log2)) + grp) * R;

    const char * rw[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rw[m][i] = (m == 0 ? x0 : x1) + (int64_t) min(row0 + i, nrows - 1) * row_bytes + (staged ? 0 : sl * WB);
        }
    }
    const char * yp = y + sl * YB;
    const int wstep = WB << lpr_log2;
    const int ystep = YB << lpr_log2;
    [[maybe_unused]] char * smw = smw_warp + (staged ? grp * R * M * WB << lpr_log2 : 0);
    [[maybe_unused]] const int seg_units = (WB << lpr_log2) >> 4;

#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    for (int it = 0; it < niter; ++it) {
        if constexpr (staged) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    uint4       * dsm = (uint4 *) (smw + ((m*R + i) * WB << lpr_log2));
                    const uint4 * src = (const uint4 *) rw[m][i];
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + (j << lpr_log2);
                        if (j < NU - 1 || u < seg_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                }
            }
            __syncwarp();
        }
        mmvqp_act<NA> a;
        mmvqp_load_act<type, NA, SRC>(a, yp);
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                uint32_t w[WB / 4];
                if constexpr (staged) {
                    mmvqp_lds_words<WB, mmvqp_gcd(WB, 16)>(w, smw + ((m*R + i) * WB << lpr_log2) + sl * WB);
                } else {
                    mmvqp_load_words<WB, mmvqp_gcd(WB, 16), SRC_LDG>(w, rw[m][i]);
                }
                acc[m][i] = mmvqp_dot<type, C>(w, a, acc[m][i]);
                rw[m][i] += wstep;
            }
        }
        if constexpr (staged) {
            __syncwarp();
        }
        yp += ystep;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < lpr) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    acc[m][i] += __shfl_xor_sync(0xFFFFFFFF, acc[m][i], off, 32);
                }
            }
        }
    }
    return row0;
}

enum mmvqp_fusion { MMVQP_PLAIN = 0, MMVQP_BIAS = 1, MMVQP_GATE = 2 };

// The fork's epilogue for one lane: the lane with sl == i % lpr writes row row0 + i. Plain, + bias, or up * silu(gate).
template <int R, int M, int fk>
static __device__ __forceinline__ void mmvqp_store(const float (&acc)[M][R], const int row0, const int lane, const int lpr_log2,
                                                   const int nrows, float * dst, const float * bias) {
    const int sl = lane & ((1 << lpr_log2) - 1);
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = row0 + i;
        if ((i & ((1 << lpr_log2) - 1)) != sl || row >= nrows) {
            continue;
        }
        float result = acc[0][i];
        if constexpr (fk == MMVQP_BIAS) {
            result += bias[row];
        }
        if constexpr (fk == MMVQP_GATE) {
            const float gate_value = acc[M - 1][i];
            result *= ggml_cuda_op_silu_single(gate_value);
        }
        dst[row] = result;
    }
}

// Byte strides for x and the activations, float strides for dst.
struct mmvqp_args {
    const char    * vx;
    const char    * gate;       // MMVQP_GATE: gate weights, same layout as vx
    const char    * vy;
    const int32_t * ids;        // MUL_MAT_ID: expert of each channel
    const float   * bias;       // MMVQP_BIAS
    float         * dst;
    int             nrows;
    int             row_bytes;
    int             lpr_log2;
    int             niter;
    int             nchannels_y; // MUL_MAT_ID: channel_y = channel % nchannels_y
    int64_t         channel_bytes_x;
    int64_t         channel_bytes_y;
    int64_t         stride_channel_dst;
};

// mul_mat_vec_q_pascal: blocks of 4 warps, grid (row blocks, channels).
template <int type, int R, int C, bool staged, int fk>
__launch_bounds__(4 * 32, 1)
static __global__ void k_mmvq_pascal(const mmvqp_args p) {
    constexpr int NW = 4;
    constexpr int M  = fk == MMVQP_GATE ? 2 : 1;
    constexpr int WB = C * mmvqp_traits<type>::bs;
    __shared__ uint4 sm[staged ? NW : 1][staged ? 32 * R * M * WB / 16 : 1];

    const int channel_dst = blockIdx.y;
    const int channel_x   = p.ids ? p.ids[channel_dst] : channel_dst;
    const int channel_y   = p.ids ? channel_dst % p.nchannels_y : channel_dst;
    const int64_t xoff    = channel_x * p.channel_bytes_x;

    float acc[M][R];
    const int row0 = mmvqp_rows<type, R, C, staged, M, SRC_LDG>(
        p.vx + xoff, M == 2 ? p.gate + xoff : nullptr, p.nrows, p.row_bytes, p.lpr_log2, p.niter,
        p.vy + channel_y * p.channel_bytes_y, blockIdx.x * NW + threadIdx.y, threadIdx.x,
        (char *) sm[staged ? threadIdx.y : 0], acc);
    mmvqp_store<R, M, fk>(acc, row0, threadIdx.x, p.lpr_log2, p.nrows, p.dst + channel_dst * p.stride_channel_dst, p.bias);
}

} // namespace pk
