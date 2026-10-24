// Baseline: llama.cpp's single-token quantized matrix-vector kernel (MMVQ) and its Q8_1 activation quantizer.
//
// Copied from llama.cpp (https://github.com/ggml-org/llama.cpp) at commit d77256999 (2026-10-22):
//   ggml/src/ggml-cuda/mmvq.cu       mul_mat_vec_q, calc_nwarps, calc_rows_per_block, the small-K rule
//   ggml/src/ggml-cuda/vecdotq.cuh   get_int_b2/b4, get_int_from_table_16, vec_dot_q4_0/q4_1/q5_0/q8_0/q5_K/iq4_xs/q6_K_q8_1
//                                    and their _impl functions
//   ggml/src/ggml-common.h           kvalues_iq4nl
//   ggml/src/ggml-cuda/quantize.cu   quantize_q8_1
//   ggml/src/ggml-cuda/common.cuh    ggml_cuda_dp4a, warp reductions, fastdiv, FAST_FP16_AVAILABLE
//   ggml/src/ggml-cuda/unary.cuh     ggml_cuda_op_silu_single
// plus vec_dot_q4_1_64_q8_1 from this repo's patches/q4_1_64-type.diff, and the Pascal parameter table of
// llama.cpp PR #25479 (2 warps per block at one token on compute capability 6.1/6.2).
// llama.cpp is MIT-licensed, Copyright (c) 2023-2026 The ggml authors.
//
// Only the paths a Pascal (compute 6.1) card takes for one token (ncols_dst == 1) with Q4_0, Q4_1, Q5_0, Q8_0, Q5_K,
// IQ4_XS, Q6_K and Q4_1_64 weights are kept. Code for other architectures (HIP, MUSA, DGX Spark prefetch, NVFP4 scales, PDL) and
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
#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_IQ4_XS_Q8_1_MMVQ 4
#define VDR_Q6_K_Q8_1_MMVQ 2

static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
    const uint32_t * table32 = (const uint32_t *) table;

    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
    // However, for the result we need ints with all even/odd 4 bit indices in q4.
    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

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

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
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

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b2(bq5_0->qs, iqs + i);
        vh[i]    = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, __half2float(bq5_0->d), bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    // same as q4_K
    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));

    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            const int vil = (vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh[j] >> (4*i)) << 4) & 0x30303030;

            // (vil | vih) - 32 for each byte, the - 32 is done by the second dp4a
            sumi = ggml_cuda_dp4a(vil | vih,  u[VDR_Q6_K_Q8_1_MMVQ*i + j], sumi); // SIMD dot product
            sumi = ggml_cuda_dp4a(0xE0E0E0E0, u[VDR_Q6_K_Q8_1_MMVQ*i + j], sumi);
        }

        sumf += d8[i] * (sumi * sc);
    }

    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    int vl[VDR_Q6_K_Q8_1_MMVQ];
    int vh[VDR_Q6_K_Q8_1_MMVQ];
#pragma unroll
    for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
        vl[j] = get_int_b2(bq6_K->ql, iqs + j);
        vh[j] = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4) + j) >> vh_shift;
    }

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[VDR_Q6_K_Q8_1_MMVQ*QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            u[VDR_Q6_K_Q8_1_MMVQ*i + j] = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// ---- mmvq.cu -------------------------------------------------------------------------------------------------

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(int type) {
    switch (type) {
        case T_Q4_0:    return vec_dot_q4_0_q8_1;
        case T_Q4_1:    return vec_dot_q4_1_q8_1;
        case T_Q4_1_64: return vec_dot_q4_1_64_q8_1;
        case T_Q8_0:    return vec_dot_q8_0_q8_1;
        case T_Q5_0:    return vec_dot_q5_0_q8_1;
        case T_Q5_K:    return vec_dot_q5_K_q8_1;
        case T_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case T_Q6_K:    return vec_dot_q6_K_q8_1;
        default:        return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(int type) {
    switch (type) {
        case T_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case T_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case T_Q4_1_64: return VDR_Q4_1_64_Q8_1_MMVQ;
        case T_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case T_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case T_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case T_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        case T_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
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
    bool iq_small_k = false;    // Q47: small-K shape for IQ4_XS too
};

// should_use_small_k at one token on a compute 6.1 card: of the types here only IQ4_XS is on an exclusion list
// (iq_slow_other, which d77256999 also applies to Pascal). iq_small_k lifts that exclusion (experiment Q47).
template <int type, int table_id>
static bool use_small_k(int blocks_per_row_x, int blocks_per_iter_1warp, bool iq_small_k) {
    const int nwarps = calc_nwarps(type, 1, table_id);
    const bool use = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
    return use && (type != T_IQ4_XS || iq_small_k);
}

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
    const bool small_k = use_small_k<type, table_id>(blocks_per_row_x, blocks_per_iter_1warp, a.iq_small_k);
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
    const bool small_k = use_small_k<type, table_id>(blocks_per_row_x, blocks_per_iter_1warp, a.iq_small_k);
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
