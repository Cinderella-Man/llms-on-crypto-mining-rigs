// F16 matrix-vector products of one token with FP32 activations (--f16): the MoE router of Flint files that keep it in
// F16 (K2h-r16, G3: 256 rows of 2048 weights, 1 MiB per layer).
//
// "mmvf": llama.cpp's mul_mat_vec_f as a compute 6.1 card runs it for one token (copied from llama.cpp d77256999,
//   ggml/src/ggml-cuda/mmvf.cu; MIT, the ggml authors): F16 weights converted to FP32 and accumulated in FP32 (fast FP16
//   is unavailable on 6.1, so the precision is F32), one block per row, block size chosen from the row width (256
//   threads for K = 2048), each thread taking every block_size-th pair of weights, reduced per warp and then over the
//   warps through shared memory. No fusion, no expert ids.
// "moep": the fork's persistent MoE kernel's router rows (same sums as "mmvf", one warp per row), timed as a kernel of
//   their own: the router phase of that kernel if it had the whole card.
// "f16_rR": the Pascal kernel. A warp is cut into lane groups of LPR lanes (LPR from the row width, as in the
//   quantized kernels); a lane group owns R consecutive rows. Per iteration each lane reads 8 consecutive weights
//   (16 bytes) of each of its rows and the 8 matching activations (two 16-byte loads), which all R rows share, and does
//   one FMA per weight in FP32; the lane group's partial sums are reduced with shuffles. Weight and activation loads are
//   fully coalesced 128-bit loads, so no staging is needed.
#pragma once

#include "common.cuh"
#include "data.cuh"

namespace mb {
namespace f16mv {

// n random F16 weights, uniform in (-0.05, 0.05) like a router's
static __global__ void gen_f16(half * __restrict__ dst, const size_t n, const uint64_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = __float2half((u01(hash_u32(seed * 0x2545F4914F6CDD1Dull + i)) - 0.5f) * 0.1f);
    }
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// mul_mat_vec_f<half, float, 1, block_size> without fusion or ids; ncols2 = K / 2, stride_row in halves
template <int block_size>
__global__ void mmvf_stock(const half * x, const float * y, float * dst, const int ncols2, const int stride_row) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x += (int64_t) row * stride_row;
    const half2  * x2 = (const half2 *) x;
    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[32];
    if (block_size > 32) {
        if (tid < 32) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = __half22float2(x2[col2]);
        const float2 tmpy = y2[col2];
        sumf += tmpx.x * tmpy.x;
        sumf += tmpx.y * tmpy.y;
    }

    sumf = warp_reduce_sum(sumf);
    if (block_size > 32) {
        buf_iw[tid / 32] = sumf;
        __syncthreads();
        if (tid >= 32) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum(sumf);
    }
    if (tid != 0) {
        return;
    }
    dst[row] = sumf;
}

// The router rows of the fork's persistent MoE kernel (moe-persistent.cu, Q88), as a kernel of their own: a work item
// of 2 rows taken by one warp, which computes them one after the other; each lane stands in for 8 of mul_mat_vec_f's
// 256 threads (4-byte loads of a half2 at a stride of 256 half2, so the warp reads 128 contiguous bytes per load), the
// normed input is computed on the fly from the layer input and the norm weight (scale * x * m, two more 8-byte loads per
// pair of columns), and mul_mat_vec_f's two butterfly reductions follow. Same sums as mmvf_stock<256>.
template <int NW>
__launch_bounds__(NW * 32)
__global__ void moep_router(const half * x, const float * xin, const float * normw, const float scale, float * dst,
                            const int nrows, const int ncols2) {
    const int lane = threadIdx.x;
    const int item = blockIdx.x * NW + threadIdx.y;
    const float2 * x2 = (const float2 *) xin;
    const float2 * m2 = (const float2 *) normw;
    for (int k = 0; k < 2; ++k) {
        const int row = 2 * item + k;
        if (row >= nrows) {
            return;
        }
        const half2 * w2 = (const half2 *) (x + (size_t) row * 2 * ncols2);
        float s[8];
#pragma unroll
        for (int vw = 0; vw < 8; ++vw) {
            float sumf = 0.0f;
            for (int col2 = vw * 32 + lane; col2 < ncols2; col2 += 256) {
                const float2 tmpx = __half22float2(__ldg(w2 + col2));
                const float2 xi = __ldg(x2 + col2);
                const float2 m = __ldg(m2 + col2);
                float y0 = scale * xi.x;
                y0 *= m.x;
                float y1 = scale * xi.y;
                y1 *= m.y;
                sumf += tmpx.x * y0;
                sumf += tmpx.y * y1;
            }
            s[vw] = sumf;
        }
#pragma unroll
        for (int vw = 0; vw < 8; ++vw) {
            s[vw] = warp_reduce_sum(s[vw]);
        }
        float sumf = 0.0f;
#pragma unroll
        for (int vw = 0; vw < 8; ++vw) {
            if (lane == vw) {
                sumf = s[vw];
            }
        }
        sumf = warp_reduce_sum(sumf);
        if (lane == 0) {
            dst[row] = sumf;
        }
    }
}

// llama.cpp's block size for a row of ncols weights: the multiple of 32 up to 256 with the fewest iterations
inline int mmvf_block_size(const int ncols) {
    int best = 32;
    int niter_best = (ncols + 63) / 64;
    for (int bs = 64; bs <= 256; bs += 32) {
        const int niter = (ncols + 2 * bs - 1) / (2 * bs);
        if (niter < niter_best) {
            niter_best = niter;
            best = bs;
        }
    }
    return best;
}

static __device__ __forceinline__ float2 h2f2(const uint32_t w) {
    return __half22float2(*reinterpret_cast<const half2 *>(&w));
}

// R rows per lane group of 1 << lpr_log2 lanes, NW warps per block; row_bytes a multiple of 16, niter = K / 8 / lpr
template <int R, int NW>
__launch_bounds__(NW * 32)
__global__ void f16_rows(const uint8_t * x, const float * y, float * dst, const int nrows, const int row_bytes,
                         const int lpr_log2, const int niter) {
    const int lane = threadIdx.x;
    const int lpr  = 1 << lpr_log2;
    const int sl   = lane & (lpr - 1);
    const int row0 = (((blockIdx.x * NW + threadIdx.y) << (5 - lpr_log2)) + (lane >> lpr_log2)) * R;

    const uint4 * xr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        xr[i] = (const uint4 *) (x + (size_t) min(row0 + i, nrows - 1) * row_bytes) + sl;
    }
    const float4 * yp = (const float4 *) y + 2 * sl;

    float acc[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        acc[i] = 0.0f;
    }
    for (int it = 0; it < niter; ++it) {
        uint4 w[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            w[i] = __ldg(xr[i]);
        }
        const float4 y0 = __ldg(yp);
        const float4 y1 = __ldg(yp + 1);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const float2 a0 = h2f2(w[i].x);
            const float2 a1 = h2f2(w[i].y);
            const float2 a2 = h2f2(w[i].z);
            const float2 a3 = h2f2(w[i].w);
            acc[i] = fmaf(a0.x, y0.x, acc[i]);
            acc[i] = fmaf(a0.y, y0.y, acc[i]);
            acc[i] = fmaf(a1.x, y0.z, acc[i]);
            acc[i] = fmaf(a1.y, y0.w, acc[i]);
            acc[i] = fmaf(a2.x, y1.x, acc[i]);
            acc[i] = fmaf(a2.y, y1.y, acc[i]);
            acc[i] = fmaf(a3.x, y1.z, acc[i]);
            acc[i] = fmaf(a3.y, y1.w, acc[i]);
            xr[i] += lpr;
        }
        yp += 2 * lpr;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < lpr) {
                acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        if ((i & (lpr - 1)) == sl && row0 + i < nrows) {
            dst[row0 + i] = acc[i];
        }
    }
}

} // namespace f16mv
} // namespace mb
