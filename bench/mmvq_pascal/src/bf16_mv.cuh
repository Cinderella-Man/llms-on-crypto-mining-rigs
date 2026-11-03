// BF16 matrix-vector products of 1 to 8 tokens with FP32 activations (--bf16): the BF16 tensors of Qwen3.8-Flash-Next
// (hyper-connection projections 320 -> 10240, 10240 -> 320 and 10240 -> 4, the indexer's query projection
// 2560 -> 512, the recurrent layers' alpha and beta 2560 -> 48).
//
// "mmvf": llama.cpp's mul_mat_vec_f<nv_bfloat16, float, T, block_size> as a compute 6.1 card runs it (ggml/src/ggml-cuda/
//   mmvf.cu; MIT, the ggml authors), without fusion or expert ids: one block per row, block size chosen from the row
//   width (160 threads for K = 320, 256 for K >= 512), each thread taking every block_size-th pair of weights
//   (4 bytes), converted to FP32 and accumulated in FP32, reduced per warp and then over the warps through shared memory.
// "gG_rR": Miner Llama's Pascal BF16 kernel (mul_mat_vec_bf16_pascal, copied below from the fork's
//   ggml/src/ggml-cuda/mmvf-pascal-impl.cuh, branch pascal-p106-v3-b12076-kcol) at a fixed launch shape:
//   G lanes per row group (8, 32, or 128 over the block's 4 warps), R rows per group (1, 2, 4, 8) sharing every
//   activation load,
//   8 weights per lane per iteration with one 128-bit load, each weight converted exactly (BF16 is the upper half of
//   an FP32) and added with one FMA.
// "pascal": the same kernel at the launch shape the fork picks for this card (mmvfp_pick_shape, copied below).
#pragma once

#include "common.cuh"
#include "data.cuh"

#include <cuda_bf16.h>

namespace mb {
namespace bf16mv {

// n random BF16 weights, uniform in (-0.05, 0.05)
static __global__ void gen_bf16(nv_bfloat16 * __restrict__ dst, const size_t n, const uint64_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = __float2bfloat16((u01(hash_u32(seed * 0x2545F4914F6CDD1Dull + i)) - 0.5f) * 0.1f);
    }
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// mul_mat_vec_f<nv_bfloat16, float, T, block_size> without fusion or ids; ncols2 = K / 2, stride_row in elements,
// stride_col_y2 in float2, stride_col_dst in floats
template <int T, int block_size>
__global__ void mmvf_stock(const nv_bfloat16 * x, const float * y, float * dst, const int ncols2, const int stride_row,
                           const int stride_col_y2, const int stride_col_dst) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x += (int64_t) row * stride_row;
    const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[32];
    if (block_size > 32) {
        if (tid < 32) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[T] = { 0.0f };
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float2 tmpy = y2[j * stride_col_y2 + col2];
            sumf[j] += (float) tmpx.x * tmpy.x;
            sumf[j] += (float) tmpx.y * tmpy.y;
        }
    }

#pragma unroll
    for (int j = 0; j < T; ++j) {
        sumf[j] = warp_reduce_sum(sumf[j]);
        if (block_size > 32) {
            buf_iw[tid / 32] = sumf[j];
            __syncthreads();
            if (tid < 32) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum(sumf[j]);
            }
            __syncthreads();
        }
    }
    if (tid >= T) {
        return;
    }
    dst[tid * stride_col_dst + row] = sumf[tid];
}

// ---- copied from the fork's ggml/src/ggml-cuda/mmvf-pascal-impl.cuh -------------------------------------------------

#define MMVF_PASCAL_NT 128 // threads per block (4 warps)

// 8 BF16 weights (one 16-byte load) times 8 FP32 activations, added to acc one weight at a time in column order. A
// BF16 value is the upper half of an FP32 one, so moving it into place converts it exactly.
static __device__ __forceinline__ float mmvfp_dot8(const uint4 w, const float4 a, const float4 b, float acc) {
    acc = fmaf(__uint_as_float(w.x << 16),         a.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xFFFF0000u), a.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16),         a.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xFFFF0000u), a.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16),         b.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xFFFF0000u), b.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16),         b.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xFFFF0000u), b.w, acc);
    return acc;
}

// G lanes per row group (8, 32, or 128, which spans the block's 4 warps), R rows per group, T tokens.
// x: nrows rows of nchunks 16-byte chunks (8 weights each), stride_row chunks apart. y: T columns of FP32 activations,
// stride_col_y4 float4 apart. dst, and bias if has_bias: T columns, stride_col_dst floats apart.
// Lane sl of a group takes chunks sl, sl + G, ... of each of its R rows, so every weight load of a warp is a coalesced
// 128-bit load; the activations of a chunk are loaded once for all R rows. Partial sums are reduced with shuffles in
// the warp, then (G = 128) over the warps through shared memory, always in the same order; after the shuffles every lane
// of a group holds all T * R sums, and lane lg writes those whose index t * R + i is lg modulo the group's lanes in the
// warp, so a lane writes several outputs where T * R exceeds them.
template <int G, int R, int T, bool has_bias>
__launch_bounds__(MMVF_PASCAL_NT)
static __global__ void mul_mat_vec_bf16_pascal(
        const uint4 * __restrict__ x, const float4 * __restrict__ y, const float * __restrict__ bias, float * __restrict__ dst,
        const int nrows, const int nchunks, const int stride_row, const int stride_col_y4, const int stride_col_dst) {
    constexpr int warp = 32;
    constexpr int NW   = MMVF_PASCAL_NT / warp;
    constexpr int GW   = G < warp ? G : warp;     // lanes of a group within one warp
    constexpr int WPG  = G > warp ? G / warp : 1; // warps per group
    static_assert(G == 8 || G == 32 || G == 128, "unsupported group size");

    const int tid  = threadIdx.x;
    const int sl   = tid % G;
    const int lg   = tid % GW;
    const int row0 = (blockIdx.x * (MMVF_PASCAL_NT / G) + tid / G) * R;

    const uint4 * xr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        xr[i] = x + (int64_t) min(row0 + i, nrows - 1) * stride_row + sl;
    }
    const float4 * yp = y + 2 * sl;

    float acc[T][R];
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[t][i] = 0.0f;
        }
    }

#pragma unroll 2
    for (int c = sl; c < nchunks; c += G) {
        uint4 w[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            w[i] = __ldg(xr[i]);
            xr[i] += G;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 a = __ldg(yp + t * stride_col_y4);
            const float4 b = __ldg(yp + t * stride_col_y4 + 1);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[t][i] = mmvfp_dot8(w[i], a, b, acc[t][i]);
            }
        }
        yp += 2 * G;
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int off = GW / 2; off > 0; off >>= 1) {
                acc[t][i] += __shfl_xor_sync(0xFFFFFFFF, acc[t][i], off, warp);
            }
        }
    }

    if constexpr (WPG > 1) {
        __shared__ float buf[NW][T * R];
        const int wid = tid / warp;
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if ((t * R + i) % GW == lg) {
                    buf[wid][t * R + i] = acc[t][i];
                }
            }
        }
        __syncthreads();
        if (wid % WPG != 0) {
            return;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if ((t * R + i) % GW == lg) {
#pragma unroll
                    for (int k = 1; k < WPG; ++k) {
                        acc[t][i] += buf[wid + k][t * R + i];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = row0 + i;
            if ((t * R + i) % GW != lg || row >= nrows) {
                continue;
            }
            float v = acc[t][i];
            if constexpr (has_bias) {
                v += bias[t * stride_col_dst + row];
            }
            dst[t * stride_col_dst + row] = v;
        }
    }
}

// ---- copied from the fork's ggml/src/ggml-cuda/mmvf-pascal.cu ---------------------------------------------------------

inline int mmvfp_blocks(const int nrows, const int G, const int R) {
    const int rows_per_block = (MMVF_PASCAL_NT / G) * R;
    return (nrows + rows_per_block - 1) / rows_per_block;
}

inline void mmvfp_pick_shape(const int nchunks, const int nrows, const int ncols_dst, const int nsm, int & G, int & R) {
    const int rmax = ncols_dst <= 4 ? 4 : 8;
    const int target = 4 * nsm;
    // 3 and 4 tokens: at most 32 lanes from the width, and a short grid gets more lanes before fewer rows
    const bool mid = ncols_dst == 3 || ncols_dst == 4;
    G = 8;
    while (G < (mid ? 32 : 128) && nchunks > 16 * G) {
        G *= 4;
    }
    R = rmax;
    if (mid) {
        while (G < 128 && mmvfp_blocks(nrows, G, R) < target && nchunks >= 8 * G) {
            G *= 4;
        }
    }
    while (R > 1 && mmvfp_blocks(nrows, G, R) < target) {
        R /= 2;
    }
    while (G < 128 && mmvfp_blocks(nrows, G, R) < target && nchunks >= 8 * G) {
        G *= 4;
    }
}

// ---- kernel pointers for the benchmark ------------------------------------------------------------------------------

template <int T>
inline const void * stock_func(const int block_size) {
    switch (block_size) {
        case  32: return (const void *) mmvf_stock<T,  32>;
        case  64: return (const void *) mmvf_stock<T,  64>;
        case  96: return (const void *) mmvf_stock<T,  96>;
        case 128: return (const void *) mmvf_stock<T, 128>;
        case 160: return (const void *) mmvf_stock<T, 160>;
        case 192: return (const void *) mmvf_stock<T, 192>;
        case 224: return (const void *) mmvf_stock<T, 224>;
        case 256: return (const void *) mmvf_stock<T, 256>;
        default:  return nullptr;
    }
}

inline const void * stock_func(const int T, const int block_size) {
    switch (T) {
        case 1: return stock_func<1>(block_size);
        case 2: return stock_func<2>(block_size);
        case 3: return stock_func<3>(block_size);
        case 4: return stock_func<4>(block_size);
        default: return nullptr;
    }
}

template <int G, int R>
inline const void * pascal_func_t(const int T) {
    switch (T) {
        case 1: return (const void *) mul_mat_vec_bf16_pascal<G, R, 1, false>;
        case 2: return (const void *) mul_mat_vec_bf16_pascal<G, R, 2, false>;
        case 3: return (const void *) mul_mat_vec_bf16_pascal<G, R, 3, false>;
        case 4: return (const void *) mul_mat_vec_bf16_pascal<G, R, 4, false>;
        case 5: return (const void *) mul_mat_vec_bf16_pascal<G, R, 5, false>;
        case 6: return (const void *) mul_mat_vec_bf16_pascal<G, R, 6, false>;
        case 7: return (const void *) mul_mat_vec_bf16_pascal<G, R, 7, false>;
        case 8: return (const void *) mul_mat_vec_bf16_pascal<G, R, 8, false>;
        default: return nullptr;
    }
}

template <int G>
inline const void * pascal_func_g(const int R, const int T) {
    switch (R) {
        case 1: return pascal_func_t<G, 1>(T);
        case 2: return pascal_func_t<G, 2>(T);
        case 4: return pascal_func_t<G, 4>(T);
        case 8: return pascal_func_t<G, 8>(T);
        default: return nullptr;
    }
}

// nullptr for shapes the kernel has no instance of
inline const void * pascal_func(const int G, const int R, const int T) {
    switch (G) {
        case   8: return pascal_func_g<  8>(R, T);
        case  32: return pascal_func_g< 32>(R, T);
        case 128: return pascal_func_g<128>(R, T);
        default:  return nullptr;
    }
}

} // namespace bf16mv
} // namespace mb
