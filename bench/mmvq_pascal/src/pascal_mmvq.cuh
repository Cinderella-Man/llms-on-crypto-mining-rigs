// Pascal-tuned single-token matrix-vector kernels.
//
// Variant (c), "p_rRcC": the same weight blocks and Q8_1 activations as llama.cpp, a different work split.
//   - A warp is cut into sub-groups of LPR lanes (LPR = lanes per row, a power of two chosen from K). Each sub-group
//     owns R consecutive output rows, so one activation load feeds R rows (2R for fused gate/up).
//   - Each lane takes C consecutive weight blocks per iteration and loads them with the widest vector loads their
//     alignment allows (C*20 bytes for Q4_1: 64-bit at C = 2, 128-bit at C = 4), then extracts fields at compile-time
//     byte offsets (one PRMT where a 2-byte-aligned field straddles two words).
//   - Row pointers are computed once; the loop advances them by a constant, so there are no per-iteration multiplies.
//   - Nibbles are unpacked with masks only: dp4a on (q & 0x0F0F0F0F) and an unsigned-by-signed dp4a on
//     (q & 0xF0F0F0F0), whose sum is 16x too large; the two sums are combined as (lo << 4) + hi and the 1/16 goes into
//     the activation scale, which every row shares.
//   - Partial sums are reduced with warp shuffles inside the sub-group; no shared memory, no __syncthreads.
//
// Variant (d), "soa_rR": weights repacked into a structure of arrays (Q4_1 with group 32 or group 64).
//   - Nibble plane: 16 bytes per 32 weights; for group 64 the two 16-byte halves of a group live in two planes, so that
//     lane l reading group g + l touches consecutive 16-byte chunks: every weight load is a fully coalesced 128-bit load.
//   - Scale plane: half2 (d, m) per group, 4 bytes, read with one coalesced 32-bit load.
//   - Activations in the same spirit: an int8 plane (32 bytes per 32 values, two 128-bit loads) and a float2 (d, s)
//     plane with the block sum kept in FP32.
#pragma once

#include "common.cuh"

namespace mb {
namespace pascal {

// a: 4 unsigned bytes, b: 4 signed bytes
static __device__ __forceinline__ int dp4a_us(uint32_t a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    const int8_t * b8 = (const int8_t *) &b;
    return c + (int) (a & 0xff) * b8[0] + (int) ((a >> 8) & 0xff) * b8[1] + (int) ((a >> 16) & 0xff) * b8[2] + (int) (a >> 24) * b8[3];
#endif
}

static __device__ __forceinline__ int dp4a_ss(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

static __device__ __forceinline__ float h2f_lo(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) (w & 0xffff))); }
static __device__ __forceinline__ float h2f_hi(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) (w >> 16))); }

// Load NB bytes (a multiple of 4) from p, which is aligned to AL bytes, into 32-bit words through the read-only path.
template <int NB, int AL>
static __device__ __forceinline__ void ldg_words(uint32_t * w, const uint8_t * p) {
    static_assert(NB % 4 == 0 && AL >= 4, "chunk must be 4-byte aligned");
    if constexpr (AL >= 16 && NB % 16 == 0) {
#pragma unroll
        for (int i = 0; i < NB/16; ++i) {
            const uint4 v = __ldg((const uint4 *) p + i);
            w[4*i+0] = v.x; w[4*i+1] = v.y; w[4*i+2] = v.z; w[4*i+3] = v.w;
        }
    } else if constexpr (AL >= 8 && NB % 8 == 0) {
#pragma unroll
        for (int i = 0; i < NB/8; ++i) {
            const uint2 v = __ldg((const uint2 *) p + i);
            w[2*i+0] = v.x; w[2*i+1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            w[i] = __ldg((const unsigned int *) p + i);
        }
    }
}

// 32-bit word at an even byte offset of a word array (offsets are compile-time constants after unrolling).
static __device__ __forceinline__ uint32_t word_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? w[off >> 2] : __byte_perm(w[off >> 2], w[(off >> 2) + 1], 0x5432);
}
static __device__ __forceinline__ uint32_t u16_at(const uint32_t * w, const int off) {
    return (off & 3) == 0 ? (w[off >> 2] & 0xffff) : (w[off >> 2] >> 16);
}

// Activation chunk: NA Q8_1 blocks, already split into quants and float scales.
template <int NA>
struct act_chunk {
    int   u[NA][8];
    float d16[NA]; // d8 / 16 (4-bit types) or d8 (Q8_0)
    float s[NA];   // s8 (Q4_1, Q4_1_64) or -8 * s8 (Q4_0)
};

template <int T, int NA>
static __device__ __forceinline__ void load_act(act_chunk<NA> & a, const uint8_t * p) {
    constexpr int NB = NA * 36;
    uint32_t w[NB / 4];
    ldg_words<NB, cgcd(NB, 16)>(w, p);
#pragma unroll
    for (int b = 0; b < NA; ++b) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a.u[b][k] = (int) w[9*b + 1 + k];
        }
        const float d8 = h2f_lo(w[9*b]);
        const float s8 = h2f_hi(w[9*b]);
        if constexpr (T == T_Q8_0) {
            a.d16[b] = d8;
            a.s[b]   = 0.0f;
        } else if constexpr (T == T_Q4_0) {
            a.d16[b] = d8 * 0.0625f;
            a.s[b]   = -8.0f * s8;
        } else {
            a.d16[b] = d8 * 0.0625f;
            a.s[b]   = s8;
        }
    }
}

// 16 * sum(nibble * activation) over one 32-weight Q4 sub-block whose 4 quant words are q[0..3].
static __device__ __forceinline__ int q4_isum16(const uint32_t q0, const uint32_t q1, const uint32_t q2, const uint32_t q3, const int * u) {
    int lo = 0;
    int hi = 0;
    lo = dp4a_ss((int) (q0 & 0x0F0F0F0F), u[0], lo);
    hi = dp4a_us(q0 & 0xF0F0F0F0, u[4], hi);
    lo = dp4a_ss((int) (q1 & 0x0F0F0F0F), u[1], lo);
    hi = dp4a_us(q1 & 0xF0F0F0F0, u[5], hi);
    lo = dp4a_ss((int) (q2 & 0x0F0F0F0F), u[2], lo);
    hi = dp4a_us(q2 & 0xF0F0F0F0, u[6], hi);
    lo = dp4a_ss((int) (q3 & 0x0F0F0F0F), u[3], lo);
    hi = dp4a_us(q3 & 0xF0F0F0F0, u[7], hi);
    return (lo << 4) + hi;
}

// Dot product of one weight chunk (C blocks as raw words) with the matching activation chunk, added to acc.
template <int T, int C>
static __device__ __forceinline__ float dot_chunk(const uint32_t * w, const act_chunk<C * traits<T>::qk / 32> & a, float acc) {
    constexpr int BS = traits<T>::bs;
#pragma unroll
    for (int b = 0; b < C; ++b) {
        const int o = b * BS;
        if constexpr (T == T_Q4_1) {
            const uint32_t dm = word_at(w, o);
            const int isum = q4_isum16(word_at(w, o + 4), word_at(w, o + 8), word_at(w, o + 12), word_at(w, o + 16), a.u[b]);
            acc = fmaf((float) isum, h2f_lo(dm) * a.d16[b], acc);
            acc = fmaf(h2f_hi(dm), a.s[b], acc);
        } else if constexpr (T == T_Q4_0) {
            const float d = h2f_lo(u16_at(w, o));
            const int isum = q4_isum16(word_at(w, o + 2), word_at(w, o + 6), word_at(w, o + 10), word_at(w, o + 14), a.u[b]);
            acc = fmaf(d, fmaf((float) isum, a.d16[b], a.s[b]), acc);
        } else if constexpr (T == T_Q8_0) {
            const float d = h2f_lo(u16_at(w, o));
            int isum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                isum = dp4a_ss((int) word_at(w, o + 2 + 4*k), a.u[b][k], isum);
            }
            acc = fmaf((float) isum, d * a.d16[b], acc);
        } else { // T_Q4_1_64
            const uint32_t dm = word_at(w, o);
            const int i0 = q4_isum16(word_at(w, o +  4), word_at(w, o +  8), word_at(w, o + 12), word_at(w, o + 16), a.u[2*b + 0]);
            const int i1 = q4_isum16(word_at(w, o + 20), word_at(w, o + 24), word_at(w, o + 28), word_at(w, o + 32), a.u[2*b + 1]);
            const float t = fmaf((float) i1, a.d16[2*b + 1], (float) i0 * a.d16[2*b + 0]);
            acc = fmaf(h2f_lo(dm), t, acc);
            acc = fmaf(h2f_hi(dm), a.s[2*b + 0] + a.s[2*b + 1], acc);
        }
    }
    return acc;
}

static __device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Arguments shared by the tuned kernels. Byte strides; channel = expert slot for MUL_MAT_ID, else 0.
struct args {
    const uint8_t * w        = nullptr; // up (or plain) weights, all experts
    const uint8_t * wg       = nullptr; // gate weights for fused gate/up/SwiGLU
    const uint8_t * y        = nullptr; // Q8_1 activations
    const int     * ids      = nullptr; // expert ids [n_used], or nullptr
    float         * dst      = nullptr;
    int    nrows             = 0;       // rows per matrix
    int    row_bytes         = 0;
    size_t expert_bytes      = 0;
    int    y_chan_bytes      = 0;       // 0: one activation vector shared by all channels
    int    dst_chan_stride   = 0;       // floats
    int    lpr_log2          = 5;       // lanes per row = 1 << lpr_log2
    int    niter             = 1;       // iterations per lane
};

template <int T, int R, int C, int NW, bool GATE>
__launch_bounds__(NW * 32)
static __global__ void mmvq_rows(const args p) {
    constexpr int QK = traits<T>::qk;
    constexpr int BS = traits<T>::bs;
    constexpr int NA = C * QK / 32;     // activation blocks per chunk
    constexpr int WB = C * BS;          // weight bytes per chunk
    constexpr int AL = cgcd(WB, 16);    // alignment of a chunk (row_bytes is a multiple of 16)
    constexpr int NWW = WB / 4;

    const int lane = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr = 1 << lpr_log2;
    const int sl  = lane & (lpr - 1);
    const int row0 = (((blockIdx.x * NW + threadIdx.y) << (5 - lpr_log2)) + (lane >> lpr_log2)) * R;

    const int ch = blockIdx.y;
    const size_t e = p.ids ? (size_t) p.ids[ch] : 0;

    const uint8_t * rw[R];
    [[maybe_unused]] const uint8_t * rg[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const size_t off = e * p.expert_bytes + (size_t) min(row0 + i, p.nrows - 1) * p.row_bytes + (size_t) sl * WB;
        rw[i] = p.w + off;
        if constexpr (GATE) {
            rg[i] = p.wg + off;
        }
    }
    const uint8_t * yp = p.y + (size_t) ch * p.y_chan_bytes + (size_t) sl * (NA * 36);
    const int wstep = WB << lpr_log2;
    const int ystep = (NA * 36) << lpr_log2;

    float acc[R];
    [[maybe_unused]] float accg[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        acc[i] = 0.0f;
        accg[i] = 0.0f;
    }

    for (int it = 0; it < p.niter; ++it) {
        uint32_t ww[R][NWW];
        [[maybe_unused]] uint32_t wg[GATE ? R : 1][NWW];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            ldg_words<WB, AL>(ww[i], rw[i]);
            if constexpr (GATE) {
                ldg_words<WB, AL>(wg[i], rg[i]);
            }
        }
        act_chunk<NA> a;
        load_act<T, NA>(a, yp);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[i] = dot_chunk<T, C>(ww[i], a, acc[i]);
            if constexpr (GATE) {
                accg[i] = dot_chunk<T, C>(wg[i], a, accg[i]);
            }
            rw[i] += wstep;
            if constexpr (GATE) {
                rg[i] += wstep;
            }
        }
        yp += ystep;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < lpr) {
                acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
                if constexpr (GATE) {
                    accg[i] += __shfl_xor_sync(0xffffffff, accg[i], off);
                }
            }
        }
    }

    float * d = p.dst + (size_t) ch * p.dst_chan_stride;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        if ((i & (lpr - 1)) == sl && row0 + i < p.nrows) {
            float r = acc[i];
            if constexpr (GATE) {
                r *= silu(accg[i]);
            }
            d[row0 + i] = r;
        }
    }
}

// ---- (d) structure-of-arrays layout -------------------------------------------------------------------------------

// S = 32-weight sub-blocks per scale group (1: Q4_1, 2: Q4_1_64). For a matrix with nrows_total rows (all experts
// stacked) and K columns, ngroups = K / (32 S) per row:
//   qs plane: chunk (row r, half h, group g) of 16 bytes at ((r*S + h)*ngroups + g) * 16
//   dm plane: half2 (d, m) of (row r, group g) at (r*ngroups + g) * 4
// Activations: aq plane int8 [nvec][K], ads plane float2 (d, s) [nvec][K/32].
struct soa_args {
    const uint8_t * qs       = nullptr;
    const uint8_t * dm       = nullptr;
    const uint8_t * gqs      = nullptr; // gate planes for fused gate/up/SwiGLU
    const uint8_t * gdm      = nullptr;
    const int8_t  * aq       = nullptr;
    const float2  * ads      = nullptr;
    const int     * ids      = nullptr;
    float         * dst      = nullptr;
    int nrows                = 0;  // rows per matrix
    int ngroups              = 0;  // groups per row
    int y_chan_vals          = 0;  // activation values per channel (0: shared)
    int dst_chan_stride      = 0;
    int lpr_log2             = 5;
    int niter                = 1;
};

template <int S, int R, int NW, bool GATE>
__launch_bounds__(NW * 32)
static __global__ void mmvq_soa(const soa_args p) {
    const int lane = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr = 1 << lpr_log2;
    const int sl  = lane & (lpr - 1);
    const int row0 = (((blockIdx.x * NW + threadIdx.y) << (5 - lpr_log2)) + (lane >> lpr_log2)) * R;

    const int ch = blockIdx.y;
    const size_t rbase = p.ids ? (size_t) p.ids[ch] * p.nrows : 0;
    const size_t plane_bytes = (size_t) p.ngroups * 16; // one half-plane of one row

    const uint8_t * rq[R][S];
    const uint8_t * rd[R];
    [[maybe_unused]] const uint8_t * gq[R][S];
    [[maybe_unused]] const uint8_t * gd[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const size_t r = rbase + (size_t) min(row0 + i, p.nrows - 1);
#pragma unroll
        for (int h = 0; h < S; ++h) {
            const size_t qoff = (r * S + h) * plane_bytes + (size_t) sl * 16;
            rq[i][h] = p.qs + qoff;
            if constexpr (GATE) {
                gq[i][h] = p.gqs + qoff;
            }
        }
        const size_t doff = (r * p.ngroups + sl) * 4;
        rd[i] = p.dm + doff;
        if constexpr (GATE) {
            gd[i] = p.gdm + doff;
        }
    }
    const int8_t * aq  = p.aq  + (size_t) ch * p.y_chan_vals + (size_t) sl * (32 * S);
    const float2 * ads = p.ads + (size_t) ch * (p.y_chan_vals / 32) + (size_t) sl * S;

    float acc[R];
    [[maybe_unused]] float accg[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        acc[i] = 0.0f;
        accg[i] = 0.0f;
    }

    for (int it = 0; it < p.niter; ++it) {
        uint4 wq[R][S];
        uint32_t wd[R];
        [[maybe_unused]] uint4 xq[GATE ? R : 1][S];
        [[maybe_unused]] uint32_t xd[GATE ? R : 1];
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int h = 0; h < S; ++h) {
                wq[i][h] = __ldg((const uint4 *) rq[i][h]);
                if constexpr (GATE) {
                    xq[i][h] = __ldg((const uint4 *) gq[i][h]);
                }
            }
            wd[i] = __ldg((const unsigned int *) rd[i]);
            if constexpr (GATE) {
                xd[i] = __ldg((const unsigned int *) gd[i]);
            }
        }
        int   u[S][8];
        float d16[S];
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) {
            const uint4 a0 = __ldg((const uint4 *) (aq + 32*h));
            const uint4 a1 = __ldg((const uint4 *) (aq + 32*h + 16));
            u[h][0] = a0.x; u[h][1] = a0.y; u[h][2] = a0.z; u[h][3] = a0.w;
            u[h][4] = a1.x; u[h][5] = a1.y; u[h][6] = a1.z; u[h][7] = a1.w;
            const float2 ds = __ldg(ads + h);
            d16[h] = ds.x * 0.0625f;
            ss += ds.y;
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float t = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) {
                const int isum = q4_isum16(wq[i][h].x, wq[i][h].y, wq[i][h].z, wq[i][h].w, u[h]);
                t = h == 0 ? (float) isum * d16[0] : fmaf((float) isum, d16[h], t);
            }
            acc[i] = fmaf(h2f_lo(wd[i]), t, acc[i]);
            acc[i] = fmaf(h2f_hi(wd[i]), ss, acc[i]);
            if constexpr (GATE) {
                float tg = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) {
                    const int isum = q4_isum16(xq[i][h].x, xq[i][h].y, xq[i][h].z, xq[i][h].w, u[h]);
                    tg = h == 0 ? (float) isum * d16[0] : fmaf((float) isum, d16[h], tg);
                }
                accg[i] = fmaf(h2f_lo(xd[i]), tg, accg[i]);
                accg[i] = fmaf(h2f_hi(xd[i]), ss, accg[i]);
            }
#pragma unroll
            for (int h = 0; h < S; ++h) {
                rq[i][h] += 16 << lpr_log2;
                if constexpr (GATE) {
                    gq[i][h] += 16 << lpr_log2;
                }
            }
            rd[i] += 4 << lpr_log2;
            if constexpr (GATE) {
                gd[i] += 4 << lpr_log2;
            }
        }
        aq  += (32 * S) << lpr_log2;
        ads += S << lpr_log2;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < lpr) {
                acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
                if constexpr (GATE) {
                    accg[i] += __shfl_xor_sync(0xffffffff, accg[i], off);
                }
            }
        }
    }

    float * d = p.dst + (size_t) ch * p.dst_chan_stride;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        if ((i & (lpr - 1)) == sl && row0 + i < p.nrows) {
            float r = acc[i];
            if constexpr (GATE) {
                r *= silu(accg[i]);
            }
            d[row0 + i] = r;
        }
    }
}

// Activation quantizer for the SoA kernels: same rounding as quantize_q8_1, but d and the block sum s are stored as
// FP32 in a separate plane. One thread per value; x is [nvec][K] contiguous.
static __global__ void quantize_soa(const float * __restrict__ x, int8_t * __restrict__ aq, float2 * __restrict__ ads, const int K, const int nvec) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K * nvec) {
        return;
    }
    const float xi = x[i];
    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    const float d = amax / 127.0f;
    aq[i] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if ((i & 31) == 0) {
        ads[i / 32] = make_float2(d, sum);
    }
}

// Repack Q4_1 (S = 1) or Q4_1_64 (S = 2) blocks of nrows_total rows into the SoA planes. One thread per block.
template <int S>
static __global__ void repack_soa(const uint8_t * __restrict__ src, uint8_t * __restrict__ qs, uint8_t * __restrict__ dm,
                                  const size_t nrows_total, const int ngroups) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows_total * ngroups) {
        return;
    }
    const size_t r = i / ngroups;
    const int    g = (int) (i % ngroups);
    constexpr int BS = 4 + 16 * S;
    const uint8_t * b = src + i * BS;
    ((uint32_t *) dm)[i] = *(const uint32_t *) b;
#pragma unroll
    for (int h = 0; h < S; ++h) {
        const uint32_t * q = (const uint32_t *) (b + 4 + 16 * h);
        uint4 v = make_uint4(q[0], q[1], q[2], q[3]);
        *(uint4 *) (qs + ((r * S + h) * ngroups + g) * 16) = v;
    }
}

} // namespace pascal
} // namespace mb
