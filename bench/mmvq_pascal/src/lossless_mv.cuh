// Matrix-vector products of 1 to 8 tokens over losslessly coded BF16 weights (Q621; the codes are described in
// lossless_bf16.h). The kernel is Miner Llama's Pascal BF16 kernel (bf16_mv.cuh, mul_mat_vec_bf16_pascal) with the
// weight load replaced by the decoder: G lanes per row group, R rows per group sharing every activation load, the
// partial sums reduced in the same fixed order. Each group of 8 weights is decoded in registers into the 8 BF16 values
// the plain kernel would have loaded (the same 16-byte layout), escapes are put in place, and the same FMA sequence
// runs on them: x12, whose lane takes the plain kernel's 8-weight chunks, gives the plain kernel's output bit for bit at
// the same (G, R); x11 takes 32-weight chunks (4 groups of 8), so its sums run in another order.
#pragma once

#include "bf16_mv.cuh"

namespace mb {
namespace lossless_mv {

struct lparams {
    const uint8_t  * lo;      // N rows of K low bytes
    const uint32_t * idx;     // x12: N rows of K / 8 words; x11: N rows of three planes of K / 32 words
    const uint32_t * rowinfo; // N + 1: first escape (bits 0..23) | window top pair << 24
    const uint32_t * esc;     // column | BF16 << 16, by row then column
    const float4   * y;       // T columns of K activations, stride_col_y4 float4 apart
    float          * dst;     // T columns of N outputs, stride_col_dst floats apart
    int nrows, K, stride_col_y4, stride_col_dst;
};

static __device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// 4 BF16 pairs from 8 low bytes and the 8 high bytes in H0 (weights 0..3) and H1 (4..7): byte order lo0 hi0 lo1 hi1, ...
static __device__ __forceinline__ uint4 interleave8(const uint2 L, const uint32_t H0, const uint32_t H1) {
    return make_uint4(prmt(L.x, H0, 0x5140), prmt(L.x, H0, 0x7362), prmt(L.y, H1, 0x5140), prmt(L.y, H1, 0x7362));
}

// x12: nibble n_j = bank << 3 | sign << 2 | j. prmt(t0, t1, n & 0x7777) is the bank-1 byte; prmt(4s, 4s, n) adds 4 to
// the pair where bit 3 is clear (bank 0), because bit 3 set replicates the top bit of 0x04, which is 0.
static __device__ __forceinline__ uint4 x12_decode8(const uint2 L, const uint32_t n, const uint32_t t0, const uint32_t t1) {
    const uint32_t k4 = 0x04040404u;
    const uint32_t m  = n & 0x77777777u;
    const uint32_t H0 = prmt(t0, t1, m) + prmt(k4, k4, n);
    const uint32_t H1 = prmt(t0, t1, m >> 16) + prmt(k4, k4, n >> 16);
    return interleave8(L, H0, H1);
}

// x11: the selectors of group g of a 32-weight chunk (words w0, w1, w2), bit 3 of every nibble clear
template <int g>
static __device__ __forceinline__ uint32_t x11_sel(const uint32_t w0, const uint32_t w1, const uint32_t w2) {
    if constexpr (g == 0) return w0 & 0x77777777u;
    if constexpr (g == 1) return w1 & 0x77777777u;
    if constexpr (g == 2) return w2 & 0x77777777u;
    return ((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) | ((w2 >> 1) & 0x44444444u);
}

static __device__ __forceinline__ uint4 x11_decode8(const uint2 L, const uint32_t s, const uint32_t t0, const uint32_t t1) {
    return interleave8(L, prmt(t0, t1, s), prmt(t0, t1, s >> 16));
}

// weight k (0..7) of q becomes the BF16 value v
static __device__ __forceinline__ void patch8(uint4 & q, const uint32_t k, const uint32_t v) {
    const uint32_t keep = (k & 1) ? 0x0000FFFFu : 0xFFFF0000u;
    const uint32_t put  = (k & 1) ? v << 16 : v;
    const uint32_t wd   = k >> 1;
    q.x = wd == 0 ? (q.x & keep) | put : q.x;
    q.y = wd == 1 ? (q.y & keep) | put : q.y;
    q.z = wd == 2 ? (q.z & keep) | put : q.z;
    q.w = wd == 3 ? (q.w & keep) | put : q.w;
}

// The escapes of a row in column order: next = the next entry (0xFFFF as its column past the row's last), at = its
// index, end = the row's end. Before the products of columns cb .. cb + 7: entries below cb belong to other lanes'
// chunks and are passed over, those in the 8 columns are put into q.
static __device__ __forceinline__ void escapes8(const uint32_t * __restrict__ esc, uint32_t & next, uint32_t & at,
                                                const uint32_t end, const uint32_t cb, uint4 & q) {
    while ((next & 0xFFFFu) < cb + 8) {
        const uint32_t col = next & 0xFFFFu;
        if (col >= cb) {
            patch8(q, col - cb, next >> 16);
        }
        ++at;
        next = at < end ? __ldg(esc + at) : 0xFFFFu;
    }
}

template <int CODE, int G, int R, int T>
__launch_bounds__(MMVF_PASCAL_NT)
static __global__ void mul_mat_vec_lossless(const lparams p) {
    constexpr int warp = 32;
    constexpr int NW   = MMVF_PASCAL_NT / warp;
    constexpr int GW   = G < warp ? G : warp;
    constexpr int WPG  = G > warp ? G / warp : 1;
    constexpr int CW   = CODE == 12 ? 8 : 32; // weights per chunk
    static_assert(CODE == 11 || CODE == 12, "unknown code");
    static_assert(G == 2 || G == 8 || G == 32 || G == 128, "unsupported group size");

    const int tid  = threadIdx.x;
    const int sl   = tid % G;
    const int lg   = tid % GW;
    const int row0 = (blockIdx.x * (MMVF_PASCAL_NT / G) + tid / G) * R;
    const int nchunks = p.K / CW;

    const uint8_t  * lor[R];
    const uint32_t * idr[R];
    uint32_t t0[R], t1[R], enext[R], eat[R], eend[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int r = min(row0 + i, p.nrows - 1);
        lor[i] = p.lo + (int64_t) r * p.K;
        idr[i] = p.idx + (int64_t) r * (CODE == 12 ? 1 : 3) * nchunks;
        const uint32_t ri = __ldg(p.rowinfo + r);
        const uint32_t P  = ri >> 24;
        t0[i]   = (CODE == 12 ? P - 4 : P) * 0x01010101u - 0x03020100u;
        t1[i]   = t0[i] | 0x80808080u;
        eat[i]  = ri & 0xFFFFFFu;
        eend[i] = __ldg(p.rowinfo + r + 1) & 0xFFFFFFu;
        enext[i] = eat[i] < eend[i] ? __ldg(p.esc + eat[i]) : 0xFFFFu;
    }
    const float4 * yp = p.y + (CW / 4) * sl;

    float acc[T][R];
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[t][i] = 0.0f;
        }
    }

    if constexpr (CODE == 12) {
#pragma unroll 2
        for (int c = sl; c < nchunks; c += G) {
            uint4 q[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const uint2    L = __ldg((const uint2 *) lor[i] + c);
                const uint32_t n = __ldg(idr[i] + c);
                q[i] = x12_decode8(L, n, t0[i], t1[i]);
                escapes8(p.esc, enext[i], eat[i], eend[i], 8 * c, q[i]);
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 a = __ldg(yp + t * p.stride_col_y4);
                const float4 b = __ldg(yp + t * p.stride_col_y4 + 1);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    acc[t][i] = bf16mv::mmvfp_dot8(q[i], a, b, acc[t][i]);
                }
            }
            yp += 2 * G;
        }
    } else {
        for (int c = sl; c < nchunks; c += G) {
            uint4    la[R], lb[R];
            uint32_t w0[R], w1[R], w2[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                la[i] = __ldg((const uint4 *) lor[i] + 2 * c);
                lb[i] = __ldg((const uint4 *) lor[i] + 2 * c + 1);
                w0[i] = __ldg(idr[i] + c);
                w1[i] = __ldg(idr[i] + nchunks + c);
                w2[i] = __ldg(idr[i] + 2 * nchunks + c);
            }
            // the 4 groups of 8 weights in column order
#define X11_GROUP(g, LX, LY)                                                                                \
            {                                                                                               \
                uint4 q[R];                                                                                 \
                _Pragma("unroll")                                                                           \
                for (int i = 0; i < R; ++i) {                                                               \
                    q[i] = x11_decode8(make_uint2(LX, LY), x11_sel<g>(w0[i], w1[i], w2[i]), t0[i], t1[i]);  \
                    escapes8(p.esc, enext[i], eat[i], eend[i], 32 * c + 8 * g, q[i]);                       \
                }                                                                                           \
                _Pragma("unroll")                                                                           \
                for (int t = 0; t < T; ++t) {                                                               \
                    const float4 a = __ldg(yp + t * p.stride_col_y4 + 2 * g);                               \
                    const float4 b = __ldg(yp + t * p.stride_col_y4 + 2 * g + 1);                           \
                    _Pragma("unroll")                                                                       \
                    for (int i = 0; i < R; ++i) {                                                           \
                        acc[t][i] = bf16mv::mmvfp_dot8(q[i], a, b, acc[t][i]);                              \
                    }                                                                                       \
                }                                                                                           \
            }
            X11_GROUP(0, la[i].x, la[i].y)
            X11_GROUP(1, la[i].z, la[i].w)
            X11_GROUP(2, lb[i].x, lb[i].y)
            X11_GROUP(3, lb[i].z, lb[i].w)
#undef X11_GROUP
            yp += 8 * G;
        }
    }

    // the plain kernel's reduction, unchanged
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
            if ((t * R + i) % GW != lg || row >= p.nrows) {
                continue;
            }
            p.dst[t * p.stride_col_dst + row] = acc[t][i];
        }
    }
}

// Streams n16 16-byte words and folds them into one word per thread, so the read can't be dropped: the card's read
// bandwidth for a given byte count, the floor under every kernel above.
static __global__ void read_only(const uint4 * __restrict__ x, const size_t n16, uint32_t * __restrict__ sink) {
    uint32_t v = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 w = __ldg(x + i);
        v ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    if (v == 0x9E3779B9u) {
        sink[0] = v;
    }
}

// ---- kernel pointers ------------------------------------------------------------------------------------------------

template <int CODE, int G, int R>
inline const void * func_t(const int T) {
    switch (T) {
        case 1: return (const void *) mul_mat_vec_lossless<CODE, G, R, 1>;
        case 2: return (const void *) mul_mat_vec_lossless<CODE, G, R, 2>;
        case 3: return (const void *) mul_mat_vec_lossless<CODE, G, R, 3>;
        case 4: return (const void *) mul_mat_vec_lossless<CODE, G, R, 4>;
        case 5: return (const void *) mul_mat_vec_lossless<CODE, G, R, 5>;
        case 6: return (const void *) mul_mat_vec_lossless<CODE, G, R, 6>;
        case 7: return (const void *) mul_mat_vec_lossless<CODE, G, R, 7>;
        case 8: return (const void *) mul_mat_vec_lossless<CODE, G, R, 8>;
        default: return nullptr;
    }
}

template <int CODE, int G>
inline const void * func_g(const int R, const int T) {
    switch (R) {
        case 1: return func_t<CODE, G, 1>(T);
        case 2: return func_t<CODE, G, 2>(T);
        case 4: return func_t<CODE, G, 4>(T);
        case 8: return func_t<CODE, G, 8>(T);
        default: return nullptr;
    }
}

// nullptr where there is no instance: x12 has G = 8, 32, 128 (the plain kernel's), x11 also G = 2 (5 of the 10 chunks
// of a 320-wide row per lane)
inline const void * func(const int code, const int G, const int R, const int T) {
    if (code == 12) {
        switch (G) {
            case   8: return func_g<12,   8>(R, T);
            case  32: return func_g<12,  32>(R, T);
            case 128: return func_g<12, 128>(R, T);
            default:  return nullptr;
        }
    }
    if (code == 11) {
        switch (G) {
            case   2: return func_g<11,   2>(R, T);
            case   8: return func_g<11,   8>(R, T);
            case  32: return func_g<11,  32>(R, T);
            case 128: return func_g<11, 128>(R, T);
            default:  return nullptr;
        }
    }
    return nullptr;
}

} // namespace lossless_mv
} // namespace mb
