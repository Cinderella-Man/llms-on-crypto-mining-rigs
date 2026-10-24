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
//   - Q5_0: the fifth bits are not moved next to their nibbles. The block's sum splits as
//     sum(nibble * a) + 16 * sum(bit * a); (qh >> j) & 0x01010101 holds the fifth bits of weights j, 8+j, 16+j, 24+j,
//     so the activations are regrouped the same way (two 4 x 4 byte transposes per block, shared by all rows).
//   - Q5_K and IQ4_XS: each lane takes one 256-weight super-block per iteration (C = 1). Q5_K's activation sums
//     (for the mins) are computed once per chunk and shared by all rows. IQ4_XS looks its 4-bit indices up with
//     __byte_perm in a table held in registers; the activations are pre-permuted into the order the lookup produces,
//     which saves llama.cpp's two re-ordering byte permutes per word.
//   - Q6_K (staged variant only: its 210-byte super-blocks are 2-byte aligned): one super-block per lane. The 6-bit
//     codes are the low or high nibbles of ql OR-ed with a 2-bit field of qh moved into bits 4-5, one dp4a per 4
//     weights; the offset -32 comes from the activations' half-block sums (32 x the sum of 16 int8 values), computed
//     once per chunk and shared by all rows, instead of llama.cpp's second dp4a per word with 0xE0E0E0E0.
//
// Variant (e), "rl_rRcC": row-local planes. Each row keeps its byte range but holds its codes first and its scales
// after them (Q8_0 [codes nb x 32][d nb x 2], Q4_1 [codes nb x 16][dm nb x 4], Q5_0 [codes nb x 16][qh nb x 4]
// [d nb x 2]), so every code load is a 128-bit load that the lanes of a lane group coalesce. Same work split as (c).
//
// Variant (f), "st_rRcC": the unchanged GGUF blocks, staged through shared memory. Each lane group copies the contiguous
// segment its lanes are about to read (lpr chunks of C blocks per row) with coalesced 128-bit loads, then every lane
// reads its own blocks from shared memory. This gets (e)'s coalesced loads without any repacking.
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
    int   u[NA][8];  // IQ4_XS: pre-permuted for the table lookup
    int   t[NA][8];  // Q5_0: t[j] = bytes a[j], a[8+j], a[16+j], a[24+j]; Q6_K: t[0], t[1] = 32 x the sums of a[0..15], a[16..31]
    float d16[NA];   // d8 / 16 (4- and 5-bit legacy types) or d8 (Q8_0, Q5_K, IQ4_XS)
    float s[NA];     // s8 (Q4_1, Q4_1_64), -8 * s8 (Q4_0), -16 * s8 (Q5_0), d8 * sum of the int8 values (Q5_K)
};

// 4 x 4 byte transpose: out[j] = bytes j of x0, x1, x2, x3.
static __device__ __forceinline__ void transpose4(const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3, int * out) {
    const uint32_t a = __byte_perm(x0, x1, 0x5140);
    const uint32_t b = __byte_perm(x0, x1, 0x7362);
    const uint32_t c = __byte_perm(x2, x3, 0x5140);
    const uint32_t d = __byte_perm(x2, x3, 0x7362);
    out[0] = (int) __byte_perm(a, c, 0x5410);
    out[1] = (int) __byte_perm(a, c, 0x7632);
    out[2] = (int) __byte_perm(b, d, 0x5410);
    out[3] = (int) __byte_perm(b, d, 0x7632);
}

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
        } else if constexpr (T == T_Q5_0) {
            a.d16[b] = d8 * 0.0625f;
            a.s[b]   = -16.0f * s8;
            transpose4(a.u[b][0], a.u[b][2], a.u[b][4], a.u[b][6], a.t[b] + 0);
            transpose4(a.u[b][1], a.u[b][3], a.u[b][5], a.u[b][7], a.t[b] + 4);
        } else if constexpr (T == T_Q5_K) {
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = dp4a_ss(0x01010101, a.u[b][k], sum);
            }
            a.d16[b] = d8;
            a.s[b]   = d8 * (float) sum;
        } else if constexpr (T == T_Q6_K) {
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s0 = dp4a_ss(0x01010101, a.u[b][k], s0);
                s1 = dp4a_ss(0x01010101, a.u[b][k + 4], s1);
            }
            a.t[b][0] = s0 << 5;
            a.t[b][1] = s1 << 5;
            a.d16[b]  = d8;
            a.s[b]    = 0.0f;
        } else if constexpr (T == T_IQ4_XS) {
            int p[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                p[2*k + 0] = (int) __byte_perm(a.u[b][k], a.u[b][k + 4], 0x5140);
                p[2*k + 1] = (int) __byte_perm(a.u[b][k], a.u[b][k + 4], 0x7362);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                a.u[b][k] = p[k];
            }
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

// One Q5_0 block: d (16 bits), qh, 4 nibble words; activation block b of the chunk.
template <int NA>
static __device__ __forceinline__ float dot_q5_0(const uint32_t d16bits, const uint32_t qh, const uint32_t q0, const uint32_t q1,
                                                 const uint32_t q2, const uint32_t q3, const act_chunk<NA> & a, const int b, float acc) {
    int hsum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        hsum = dp4a_ss((int) ((qh >> j) & 0x01010101), a.t[b][j], hsum);
    }
    const int isum16 = q4_isum16(q0, q1, q2, q3, a.u[b]) + (hsum << 8);
    return fmaf(h2f_lo(d16bits), fmaf((float) isum16, a.d16[b], a.s[b]), acc);
}

// One Q5_K super-block as 44 words: dm, 3 words of scales, 8 words qh, 32 words qs.
static __device__ __forceinline__ float dot_q5_K(const uint32_t * w, const act_chunk<8> & a, float acc) {
    const uint32_t sc_lo = w[1] & 0x3f3f3f3f;                                    // scales of sub-blocks 0..3
    const uint32_t m_lo  = w[2] & 0x3f3f3f3f;                                    // mins of sub-blocks 0..3
    const uint32_t sc_hi = (w[3] & 0x0f0f0f0f) | ((w[1] >> 2) & 0x30303030);    // scales of 4..7
    const uint32_t m_hi  = ((w[3] >> 4) & 0x0f0f0f0f) | ((w[2] >> 2) & 0x30303030);
    float sumd = 0.0f;
    float summ = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int isum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t q   = w[12 + 8*(s/2) + k];
            const uint32_t nib = (s & 1) ? (q >> 4) & 0x0F0F0F0F : q & 0x0F0F0F0F;
            const uint32_t hb  = s <= 4 ? (w[4 + k] << (4 - s)) & 0x10101010 : (w[4 + k] >> (s - 4)) & 0x10101010;
            isum = dp4a_ss((int) (nib | hb), a.u[s][k], isum);
        }
        const int sc = (int) (((s < 4 ? sc_lo : sc_hi) >> (8 * (s & 3))) & 0xff);
        const int m  = (int) (((s < 4 ? m_lo  : m_hi ) >> (8 * (s & 3))) & 0xff);
        sumd = fmaf(a.d16[s], (float) (isum * sc), sumd);
        summ = fmaf((float) m, a.s[s], summ);
    }
    return acc + h2f_lo(w[0]) * sumd - h2f_hi(w[0]) * summ;
}

// One IQ4_XS super-block as 34 words: d | scales_h << 16, scales_l, 32 words qs.
static __device__ __forceinline__ float dot_iq4_xs(const uint32_t * w, const act_chunk<8> & a, float acc) {
    // kvalues_iq4nl as four words
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const uint32_t sh = w[0] >> 16;
    const uint32_t sl = w[1];
    float sumf = 0.0f;
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        int isum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4  = w[2 + 4*ib + j];
            const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
            const uint32_t r0  = __byte_perm(__byte_perm(t0, t1, q4), __byte_perm(t2, t3, q4), sel);
            const uint32_t r1  = __byte_perm(__byte_perm(t0, t1, q4 >> 16), __byte_perm(t2, t3, q4 >> 16), sel >> 16);
            isum = dp4a_ss((int) r0, a.u[ib][2*j + 0], isum);
            isum = dp4a_ss((int) r1, a.u[ib][2*j + 1], isum);
        }
        const int ls = (int) (((sl >> (4 * ib)) & 0xF) | (((sh >> (2 * ib)) & 3) << 4));
        sumf = fmaf(a.d16[ib], (float) (isum * (ls - 32)), sumf);
    }
    return fmaf(h2f_lo(w[0]), sumf, acc);
}

// One Q6_K super-block as 53 words (4-byte aligned here): ql words 0..31, qh words 32..47, scale words 48..51, d in the
// low half of word 52. Sub-block s = 4h + k takes the low (k < 2) or high nibbles of ql words 16h + 8(k & 1) + j and bits
// 2k, 2k + 1 of qh words 32 + 8h + j.
static __device__ __forceinline__ float dot_q6_K(const uint32_t * w, const act_chunk<8> & a, float acc) {
    float sumf = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int h = s >> 2;
        const int k = s & 3;
        int isum0 = 0;
        int isum1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t ql = w[16*h + 8*(k & 1) + j];
            const uint32_t qh = w[32 + 8*h + j];
            const uint32_t lo = k < 2 ? ql & 0x0F0F0F0F : (ql >> 4) & 0x0F0F0F0F;
            const uint32_t hi = k == 0 ? (qh << 4) & 0x30303030 : k == 1 ? (qh << 2) & 0x30303030
                              : k == 2 ? qh & 0x30303030 : (qh >> 2) & 0x30303030;
            if (j < 4) {
                isum0 = dp4a_ss((int) (lo | hi), a.u[s][j], isum0);
            } else {
                isum1 = dp4a_ss((int) (lo | hi), a.u[s][j], isum1);
            }
        }
        const uint32_t sw  = w[48 + (s >> 1)];
        const int      sc0 = (int) (int8_t) (sw >> (16 * (s & 1)));
        const int      sc1 = (int) (int8_t) (sw >> (16 * (s & 1) + 8));
        sumf = fmaf(a.d16[s], (float) (sc0 * (isum0 - a.t[s][0]) + sc1 * (isum1 - a.t[s][1])), sumf);
    }
    return fmaf(h2f_lo(w[52]), sumf, acc);
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
        } else if constexpr (T == T_Q5_0) {
            acc = dot_q5_0(u16_at(w, o), word_at(w, o + 2), word_at(w, o + 6), word_at(w, o + 10), word_at(w, o + 14),
                           word_at(w, o + 18), a, b, acc);
        } else if constexpr (T == T_Q5_K) {
            acc = dot_q5_K(w, a, acc);
        } else if constexpr (T == T_IQ4_XS) {
            acc = dot_iq4_xs(w, a, acc);
        } else if constexpr (T == T_Q6_K) {
            acc = dot_q6_K(w, a, acc);
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

// ---- (f) staged through shared memory --------------------------------------------------------------------------------

// Read NB bytes (a multiple of 4) from shared memory aligned to AL bytes into 32-bit words.
template <int NB, int AL>
static __device__ __forceinline__ void lds_words(uint32_t * w, const uint8_t * p) {
    if constexpr (AL >= 16 && NB % 16 == 0) {
#pragma unroll
        for (int i = 0; i < NB/16; ++i) {
            const uint4 v = ((const uint4 *) p)[i];
            w[4*i+0] = v.x; w[4*i+1] = v.y; w[4*i+2] = v.z; w[4*i+3] = v.w;
        }
    } else if constexpr (AL >= 8 && NB % 8 == 0) {
#pragma unroll
        for (int i = 0; i < NB/8; ++i) {
            const uint2 v = ((const uint2 *) p)[i];
            w[2*i+0] = v.x; w[2*i+1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            w[i] = ((const uint32_t *) p)[i];
        }
    }
}

template <int SU> struct stage_unit;
template <> struct stage_unit<16> { using t = uint4; };
template <> struct stage_unit<8>  { using t = uint2; };
template <> struct stage_unit<4>  { using t = unsigned int; };

// Read one lane's Q6_K super-block (210 bytes at a 2-byte boundary in shared memory) as 53 4-byte-aligned words: the 53
// words around it, shifted by 2 bytes where it starts at an odd half-word. The last word reaches 2 bytes past the block.
static __device__ __forceinline__ void lds_q6_K(uint32_t * w, const uint8_t * p) {
    const int        mis = (int) ((uintptr_t) p & 2);
    const uint32_t * aw  = (const uint32_t *) (p - mis); // pointer arithmetic keeps the shared address space
    const uint32_t   sel = mis ? 0x5432 : 0x3210;
    uint32_t cur = aw[0];
#pragma unroll
    for (int i = 0; i < 52; ++i) {
        const uint32_t nxt = aw[i + 1];
        w[i] = __byte_perm(cur, nxt, sel);
        cur = nxt;
    }
    w[52] = __byte_perm(cur, 0, sel);
}

// Variant (f), "st_rRcC": the GGUF blocks as they are, but each lane group first copies its rows' current segment
// (lpr chunks of C blocks, contiguous in memory) into shared memory with coalesced loads of SU bytes (128-bit unless the
// rows or the segment are only 8- or 4-byte aligned, which happens for Q6_K at K = 1024 and 512), then every lane reads
// its own C blocks from there. Same work split and arithmetic as (c). Needs lpr * C * block bytes and the row bytes to
// be multiples of SU. Q6_K's fused gate/up stages the up and the gate segment one after the other in the same buffer,
// which keeps a block at 27 KB of shared memory instead of 54 KB.
template <int T, int R, int C, int NW, bool GATE, int SU = 16>
__launch_bounds__(NW * 32)
static __global__ void mmvq_staged(const args p) {
    using unit_t = typename stage_unit<SU>::t;
    constexpr int QK = traits<T>::qk;
    constexpr int BS = traits<T>::bs;
    constexpr int NA = C * QK / 32;
    constexpr int WB = C * BS;
    constexpr int M  = GATE ? 2 : 1;
    constexpr int NU = (WB + SU - 1) / SU;  // units per lane per segment (the last one predicated)
    constexpr int MS = T == T_Q6_K ? 1 : M;    // matrices staged at a time
    constexpr int PAD = T == T_Q6_K ? 16 : 0; // lds_q6_K reads 2 bytes past the last block
    __shared__ uint4 sm[NW][(32 * R * MS * WB + PAD) / 16];

    const int lane = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr = 1 << lpr_log2;
    const int sl  = lane & (lpr - 1);
    const int grp = lane >> lpr_log2;
    const int row0 = (((blockIdx.x * NW + threadIdx.y) << (5 - lpr_log2)) + grp) * R;
    const int seg_units = (lpr * WB) >> (SU == 16 ? 4 : SU == 8 ? 3 : 2);

    const int ch = blockIdx.y;
    const size_t e = p.ids ? (size_t) p.ids[ch] : 0;

    const uint8_t * rw[M][R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const size_t off = e * p.expert_bytes + (size_t) min(row0 + i, p.nrows - 1) * p.row_bytes;
        rw[0][i] = p.w + off;
        if constexpr (GATE) {
            rw[M - 1][i] = p.wg + off;
        }
    }
    uint8_t * smw = (uint8_t *) sm[threadIdx.y] + (size_t) grp * R * MS * lpr * WB;
    const uint8_t * yp = p.y + (size_t) ch * p.y_chan_bytes + (size_t) sl * (NA * 36);
    const int ystep = (NA * 36) << lpr_log2;

    float acc[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    int seg_off = 0;
    for (int it = 0; it < p.niter; ++it) {
        act_chunk<NA> a;
#pragma unroll
        for (int g = 0; g < M / MS; ++g) {
#pragma unroll
            for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    unit_t * dsm = (unit_t *) (smw + (ms * R + i) * lpr * WB);
                    const unit_t * src = (const unit_t *) (rw[g * MS + ms][i] + seg_off);
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + j * lpr;
                        if (j < NU - 1 || u < seg_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                }
            }
            __syncwarp();
            if (g == 0) {
                load_act<T, NA>(a, yp);
            }
#pragma unroll
            for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    uint32_t w[(WB + 3) / 4];
                    if constexpr (T == T_Q6_K) {
                        static_assert(C == 1, "Q6_K: one super-block per lane");
                        lds_q6_K(w, smw + (ms * R + i) * lpr * WB + sl * WB);
                    } else {
                        lds_words<WB, cgcd(WB, 16)>(w, smw + (ms * R + i) * lpr * WB + sl * WB);
                    }
                    acc[g * MS + ms][i] = dot_chunk<T, C>(w, a, acc[g * MS + ms][i]);
                }
            }
            __syncwarp();
        }
        seg_off += lpr * WB;
        yp += ystep;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            if (off < lpr) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    acc[m][i] += __shfl_xor_sync(0xffffffff, acc[m][i], off);
                }
            }
        }
    }

    float * d = p.dst + (size_t) ch * p.dst_chan_stride;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        if ((i & (lpr - 1)) == sl && row0 + i < p.nrows) {
            float r = acc[0][i];
            if constexpr (GATE) {
                r *= silu(acc[M - 1][i]);
            }
            d[row0 + i] = r;
        }
    }
}

// ---- (e) row-local planes ------------------------------------------------------------------------------------------

// Plane geometry of one row of nb blocks: code bytes per block (CS), scale-plane bytes per block (S1: Q8_0 d, Q4_1 dm,
// Q5_0 qh) and a second scale plane (S2: Q5_0 d).
template <int T> struct rl_traits;
template <> struct rl_traits<T_Q8_0> { static constexpr int CS = 32; static constexpr int S1 = 2; static constexpr int S2 = 0; };
template <> struct rl_traits<T_Q4_1> { static constexpr int CS = 16; static constexpr int S1 = 4; static constexpr int S2 = 0; };
template <> struct rl_traits<T_Q5_0> { static constexpr int CS = 16; static constexpr int S1 = 4; static constexpr int S2 = 2; };

// Same arguments as (c); row_bytes and expert_bytes are unchanged by the repack, nb = blocks per row.
template <int T, int R, int C, int NW, bool GATE>
__launch_bounds__(NW * 32)
static __global__ void mmvq_rowplanes(const args p, const int nb) {
    using rt = rl_traits<T>;
    constexpr int CB  = C * rt::CS;                 // code bytes per chunk
    constexpr int SB1 = C * rt::S1;
    constexpr int SB2 = C * rt::S2;
    constexpr int NA  = C;

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
        const size_t off = e * p.expert_bytes + (size_t) min(row0 + i, p.nrows - 1) * p.row_bytes;
        rw[i] = p.w + off;
        if constexpr (GATE) {
            rg[i] = p.wg + off;
        }
    }
    // offsets inside a row, advanced per iteration
    int oc  = sl * CB;
    int os1 = nb * rt::CS + sl * SB1;
    int os2 = nb * (rt::CS + rt::S1) + sl * SB2;
    const uint8_t * yp = p.y + (size_t) ch * p.y_chan_bytes + (size_t) sl * (NA * 36);
    const int ystep = (NA * 36) << lpr_log2;

    float acc[R];
    [[maybe_unused]] float accg[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        acc[i] = 0.0f;
        accg[i] = 0.0f;
    }

    for (int it = 0; it < p.niter; ++it) {
        uint32_t wc[GATE ? 2 : 1][R][CB / 4];
        uint32_t ws1[GATE ? 2 : 1][R][SB1 >= 4 ? SB1 / 4 : 1];
        [[maybe_unused]] uint32_t ws2[GATE ? 2 : 1][R][SB2 >= 4 ? SB2 / 4 : 1];
#pragma unroll
        for (int m = 0; m < (GATE ? 2 : 1); ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const uint8_t * base = m == 0 ? rw[i] : rg[i];
                ldg_words<CB, 16>(wc[m][i], base + oc);
                if constexpr (SB1 >= 4) {
                    ldg_words<SB1, cgcd(SB1, 16)>(ws1[m][i], base + os1);
                } else {
                    ws1[m][i][0] = __ldg((const unsigned short *) (base + os1));
                }
                if constexpr (SB2 >= 4) {
                    ldg_words<SB2, cgcd(SB2, 16)>(ws2[m][i], base + os2);
                }
            }
        }
        act_chunk<NA> a;
        load_act<T, NA>(a, yp);
#pragma unroll
        for (int m = 0; m < (GATE ? 2 : 1); ++m) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                float r = m == 0 ? acc[i] : accg[i];
#pragma unroll
                for (int b = 0; b < C; ++b) {
                    if constexpr (T == T_Q8_0) {
                        const float d = h2f_lo(u16_at(ws1[m][i], 2*b));
                        int isum = 0;
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            isum = dp4a_ss((int) wc[m][i][8*b + k], a.u[b][k], isum);
                        }
                        r = fmaf((float) isum, d * a.d16[b], r);
                    } else if constexpr (T == T_Q4_1) {
                        const uint32_t dm = ws1[m][i][b];
                        const int isum = q4_isum16(wc[m][i][4*b], wc[m][i][4*b + 1], wc[m][i][4*b + 2], wc[m][i][4*b + 3], a.u[b]);
                        r = fmaf((float) isum, h2f_lo(dm) * a.d16[b], r);
                        r = fmaf(h2f_hi(dm), a.s[b], r);
                    } else { // T_Q5_0
                        r = dot_q5_0(u16_at(ws2[m][i], 2*b), ws1[m][i][b], wc[m][i][4*b], wc[m][i][4*b + 1], wc[m][i][4*b + 2],
                                     wc[m][i][4*b + 3], a, b, r);
                    }
                }
                if (m == 0) {
                    acc[i] = r;
                } else {
                    accg[i] = r;
                }
            }
        }
        oc  += CB << lpr_log2;
        os1 += SB1 << lpr_log2;
        os2 += SB2 << lpr_log2;
        yp  += ystep;
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

// Repack nrows_total rows of GGUF blocks into row-local planes (out of place). One thread per block.
template <int T>
static __global__ void repack_rowplanes(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const size_t nrows_total, const int nb) {
    using rt = rl_traits<T>;
    constexpr int BS = traits<T>::bs;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows_total * nb) {
        return;
    }
    const size_t r = i / nb;
    const int    b = (int) (i % nb);
    const uint8_t * s = src + i * BS;
    uint8_t * row = dst + r * (size_t) nb * BS;
    // GGUF field order: Q8_0 [d][qs], Q4_1 [dm][qs], Q5_0 [d][qh][qs]
    constexpr int hdr = BS - rt::CS;
    for (int k = 0; k < rt::CS; ++k) {
        row[b * rt::CS + k] = s[hdr + k];
    }
    if constexpr (T == T_Q5_0) {
        for (int k = 0; k < 4; ++k) row[nb * rt::CS + b * 4 + k] = s[2 + k];
        for (int k = 0; k < 2; ++k) row[nb * (rt::CS + 4) + b * 2 + k] = s[k];
    } else {
        for (int k = 0; k < rt::S1; ++k) row[nb * rt::CS + b * rt::S1 + k] = s[k];
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
