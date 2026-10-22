// Baseline: llama.cpp's single-token quantized matrix-vector kernel (MMVQ) and its Q8_1 activation quantizer.
//
// Copied from llama.cpp (https://github.com/ggml-org/llama.cpp) at commit d77256999 (2026-10-22):
//   ggml/src/ggml-cuda/mmvq.cu       mul_mat_vec_q, calc_nwarps, calc_rows_per_block, the small-K rule
//   ggml/src/ggml-cuda/vecdotq.cuh   get_int_b2/b4, vec_dot_q4_0/q4_1/q8_0_q8_1 and their _impl functions
//   ggml/src/ggml-cuda/quantize.cu   quantize_q8_1
//   ggml/src/ggml-cuda/common.cuh    ggml_cuda_dp4a, warp reductions, fastdiv, FAST_FP16_AVAILABLE
//   ggml/src/ggml-cuda/unary.cuh     ggml_cuda_op_silu_single
// plus vec_dot_q4_1_64_q8_1 from this repo's patches/q4_1_64-type.diff, and the Pascal parameter table of
// llama.cpp PR #25479 (2 warps per block at one token on compute capability 6.1/6.2).
// llama.cpp is MIT-licensed, Copyright (c) 2023-2026 The ggml authors.
//
// Only the paths a Pascal (compute 6.1) card takes for one token (ncols_dst == 1) with Q4_0, Q4_1, Q8_0 and
// Q4_1_64 weights are kept. Code for other architectures (HIP, MUSA, DGX Spark prefetch, NVFP4 scales, PDL) and
// GLU variants other than SwiGLU are left out; everything kept is unchanged, so the arithmetic, the summation order
// and the launch shapes are the same as llama.cpp's. Compile with -use_fast_math, as llama.cpp does.
#pragma once

#include "common.cuh"

#include <utility>

namespace mb {
namespace llama {

// ---- common.cuh ----------------------------------------------------------------------------------------------

#define GGML_CUDA_CC_PASCAL 600
#define GGML_CUDA_CC_DP4A   610
#define GGML_CUDA_CC_VOLTA  700

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define MB_FP16_AVAILABLE
#endif
#if defined(MB_FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define MB_FAST_FP16_AVAILABLE
#endif

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

template <int width = 32>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template <int width = 32>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

static uint3 init_fastdiv_values(uint64_t d_64) {
    uint32_t d = (uint32_t) d_64;
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }
    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

static __device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

// ---- vecdotq.cuh ---------------------------------------------------------------------------------------------

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_64_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMVQ 2

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef MB_FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // MB_FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q4_1_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    // 64 weights per block: iqs in [0, QI4_1_64) selects the 32-weight half (one Q8_1 block) and the ints within it
    const block_q4_1_64 * bq = (const block_q4_1_64 *) vbq + kbx;
    const int h = iqs / QI4_1;
    const int j = iqs % QI4_1;
    const block_q8_1 * b8 = bq8_1 + h;

    int v[VDR_Q4_1_64_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_64_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_64_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq->qs + 16*h, j + i);
        u[2*i+0] = get_int_b4(b8->qs, j + i);
        u[2*i+1] = get_int_b4(b8->qs, j + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_64_Q8_1_MMVQ>(v, u, bq->dm, b8->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

// ---- mmvq.cu -------------------------------------------------------------------------------------------------

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(int type) {
    switch (type) {
        case T_Q4_0:    return vec_dot_q4_0_q8_1;
        case T_Q4_1:    return vec_dot_q4_1_q8_1;
        case T_Q4_1_64: return vec_dot_q4_1_64_q8_1;
        case T_Q8_0:    return vec_dot_q8_0_q8_1;
        default:        return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(int type) {
    switch (type) {
        case T_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case T_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case T_Q4_1_64: return VDR_Q4_1_64_Q8_1_MMVQ;
        case T_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        default:        return 1;
    }
}

// MMVQ_PARAMETERS_GENERIC is the table every NVIDIA card from Pascal to Ada (except Volta/Turing) uses.
// MMVQ_PARAMETERS_PASCAL_DP4A is the table PR #25479 adds for compute capability 6.1/6.2.
enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_PASCAL_DP4A,
};

static constexpr __host__ __device__ int calc_nwarps(int type, int ncols_dst, int table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_PASCAL_DP4A) {
        // PR #25479: Pascal cc 6.1/6.2, 2 warps for single-token decode.
        if (table_id == MMVQ_PARAMETERS_PASCAL_DP4A && ncols_dst == 1) {
            return 2;
        }
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
    (void) type;
}

// d77256999: GENERIC uses 2 rows per block at one token only for Q1_0/Q2_K/Q3_K/IQ1_S/IQ1_M (mmvq_two_rows_ncols1),
// none of which is benchmarked here.
static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_PASCAL_DP4A) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// Device-side fusion arguments (ggml_cuda_mm_fusion_args_device without the NVFP4 scales).
struct fusion_args {
    const void * gate        = nullptr;
    const void * x_bias      = nullptr;
    const void * gate_bias   = nullptr;
    const void * shared_up   = nullptr;
    const void * shared_gate = nullptr;
    float      * shared_dst  = nullptr;
    uint32_t     shared_stride_col_dst = 0;
    int          glu_op      = 0; // 0 = SwiGLU
    float        glu_limit   = 0.0f;
};

template <int type, int ncols_dst, bool has_fusion, bool small_k, int table_id>
__launch_bounds__(calc_nwarps(type, ncols_dst, table_id)*32, 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const fusion_args fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * __restrict__ vx  = vx_ptr;
    const void    * __restrict__ vy  = vy_ptr;
    const int32_t * __restrict__ ids = ids_ptr;
    float         * __restrict__ dst = dst_ptr;

    constexpr int qk  = traits<type>::qk;
    constexpr int qi  = traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = 32;

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    int active_glu = 0;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }

    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case 0: // GGML_GLU_OP_SWIGLU
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }
    (void) ids_stride;
}

// ---- quantize.cu ---------------------------------------------------------------------------------------------

#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define MATRIX_ROW_PADDING 512

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * x_ptr, void * vy_ptr,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const float * __restrict__ x  = x_ptr;
    void        * __restrict__ vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// Host launcher for a contiguous [ncols, nvec] float matrix (quantize_row_q8_1_cuda with ne2 = ne3 = 1).
static void quantize_row_q8_1_cuda(const float * x, void * vy, int64_t ne00, int64_t nvec, cudaStream_t stream) {
    const int64_t ne0 = ((ne00 + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    const uint3 ne2_fastdiv = init_fastdiv_values(1);
    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, nvec, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, ne00, ne00*nvec, ne00*nvec, ne0, (uint32_t) nvec, ne2_fastdiv);
}

// ---- host-side launch (mul_mat_vec_q_switch_ncols_dst, ncols_dst == 1) -----------------------------------------

// One MUL_MAT (ids == nullptr) or MUL_MAT_ID (ids != nullptr) call for a single token, as ggml_cuda_mul_mat_vec_q
// sets it up. Strides are in blocks (weights), Q8_1 blocks (activations) or floats (dst).
struct call_args {
    const void       * vx          = nullptr; // weights (all experts for MUL_MAT_ID)
    const void       * vgate       = nullptr; // gate weights of a fused gate/up/SwiGLU call, else nullptr
    const block_q8_1 * vy          = nullptr; // quantized activations
    const int32_t    * ids         = nullptr; // expert ids [n_used], MUL_MAT_ID only
    float            * dst         = nullptr;
    int ncols_x = 0;            // K
    int nrows_x = 0;            // rows per matrix (per expert)
    int nchannels_dst = 1;      // MUL_MAT_ID: n_used
    int nchannels_y = 1;        // MUL_MAT_ID: 1 for a broadcast input (gate/up), n_used for per-expert inputs (down)
    int stride_row_x = 0;       // blocks per row
    int stride_col_y = 0;
    int stride_col_dst = 0;
    int stride_channel_x = 0;   // blocks per expert
    int stride_channel_y = 0;
    int stride_channel_dst = 0;
};

struct launch_shape {
    const void * func;
    dim3 grid;
    dim3 block;
};

// Picks the same kernel instance and launch dimensions as ggml_cuda_mul_mat_vec_q does on a compute 6.1 card.
template <int type, int table_id>
static launch_shape plan(const call_args & a) {
    constexpr int qk = traits<type>::qk;
    constexpr int qi = traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int blocks_per_row_x      = a.ncols_x / qk;
    const int blocks_per_iter_1warp = vdr * 32 / qi;
    const int nwarps0 = calc_nwarps(type, 1, table_id);
    // should_use_small_k: none of Q4_0/Q4_1/Q8_0/Q4_1_64 is on the Pascal or IQ exclusion lists.
    const bool small_k = nwarps0 > 1 && blocks_per_row_x < nwarps0 * blocks_per_iter_1warp;
    const int nwarps = calc_nwarps(type, 1, table_id);
    const int rpb = calc_rows_per_block(1, table_id, small_k, nwarps);
    const int64_t nblocks = (a.nrows_x + rpb - 1) / rpb;
    launch_shape s;
    s.grid  = dim3((unsigned) nblocks, a.nchannels_dst, 1);
    s.block = dim3(32, nwarps, 1);
    const bool fused = a.vgate != nullptr;
    if (small_k) {
        s.func = fused ? (const void *) mul_mat_vec_q<type, 1, true,  true, table_id>
                       : (const void *) mul_mat_vec_q<type, 1, false, true, table_id>;
    } else {
        s.func = fused ? (const void *) mul_mat_vec_q<type, 1, true,  false, table_id>
                       : (const void *) mul_mat_vec_q<type, 1, false, false, table_id>;
    }
    return s;
}

template <int type, int table_id>
static void launch(const call_args & a, cudaStream_t stream) {
    constexpr int qk = traits<type>::qk;
    constexpr int qi = traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int blocks_per_row_x      = a.ncols_x / qk;
    const int blocks_per_iter_1warp = vdr * 32 / qi;
    const int nwarps0 = calc_nwarps(type, 1, table_id);
    const bool small_k = nwarps0 > 1 && blocks_per_row_x < nwarps0 * blocks_per_iter_1warp;
    const launch_shape s = plan<type, table_id>(a);

    const bool has_ids = a.ids != nullptr;
    const uint3 nchannels_y_fd   = has_ids ? init_fastdiv_values(a.nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = has_ids ? make_uint3(0, 0, 0) : init_fastdiv_values(a.nchannels_dst / 1);
    const uint3 sample_ratio_fd  = init_fastdiv_values(1);
    const uint32_t stride_sample_x = a.stride_channel_x, stride_sample_y = a.stride_channel_y, stride_sample_dst = a.stride_channel_dst;

    fusion_args f;
    f.gate = a.vgate;

#define MB_LAUNCH(FUSED, SMALLK)                                                                                     \
    mul_mat_vec_q<type, 1, FUSED, SMALLK, table_id><<<s.grid, s.block, 0, stream>>>(                                \
        a.vx, a.vy, a.ids, f, a.dst, a.ncols_x, nchannels_y_fd, a.stride_row_x, a.stride_col_y, a.stride_col_dst,     \
        channel_ratio_fd, a.stride_channel_x, a.stride_channel_y, a.stride_channel_dst, sample_ratio_fd,             \
        stride_sample_x, stride_sample_y, stride_sample_dst, has_ids ? 1u : 0u)

    if (a.vgate) {
        if (small_k) { MB_LAUNCH(true, true); } else { MB_LAUNCH(true, false); }
    } else {
        if (small_k) { MB_LAUNCH(false, true); } else { MB_LAUNCH(false, false); }
    }
#undef MB_LAUNCH
}

} // namespace llama
} // namespace mb
