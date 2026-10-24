// Shared definitions: weight and activation block layouts (identical to ggml's), type traits, error checks.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                                                     \
    do {                                                                                                  \
        cudaError_t err_ = (x);                                                                           \
        if (err_ != cudaSuccess) {                                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n  in %s\n", __FILE__, __LINE__,                     \
                    cudaGetErrorString(err_), #x);                                                        \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while (0)

namespace mb {

// Weight types. The numbering is this program's own; the byte layouts match ggml-common.h.
enum wtype : int { T_Q4_0 = 0, T_Q4_1 = 1, T_Q8_0 = 2, T_Q4_1_64 = 3, T_Q5_0 = 4, T_Q5_K = 5, T_IQ4_XS = 6, T_Q6_K = 7, T_Q4_K = 8,
                   T_COUNT = 9 };

// K-quants and I-quants: super-blocks of 256 weights in 8 sub-blocks of 32.
#define QK_K 256

#define QK4_0    32
#define QR4_0    2
#define QI4_0    (QK4_0 / (4 * QR4_0))
#define QK4_1    32
#define QR4_1    2
#define QI4_1    (QK4_1 / (4 * QR4_1))
#define QK4_1_64 64
#define QR4_1_64 2
#define QI4_1_64 (QK4_1_64 / (4 * QR4_1_64))
#define QK8_0    32
#define QR8_0    1
#define QI8_0    (QK8_0 / (4 * QR8_0))
#define QK5_0    32
#define QR5_0    2
#define QI5_0    (QK5_0 / (4 * QR5_0))
#define QR4_K    2
#define QI4_K    (QK_K / (4 * QR4_K))
#define QR5_K    2
#define QI5_K    (QK_K / (4 * QR5_K))
#define QR4_XS   2
#define QI4_XS   (QK_K / (4 * QR4_XS))
#define QR6_K    2
#define QI6_K    (QK_K / (4 * QR6_K))
#define QK8_1    32
#define QR8_1    1
#define QI8_1    (QK8_1 / (4 * QR8_1))

// 18 bytes per 32 weights: fp16 scale d, then 16 bytes of nibbles. Weight j = low nibble of qs[j], weight j+16 = high.
// Value = d * (q - 8).
struct block_q4_0 {
    half    d;
    uint8_t qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "q4_0 size");

// 20 bytes per 32 weights: half2 (d, m), then 16 bytes of nibbles, same nibble order as Q4_0. Value = d * q + m.
struct block_q4_1 {
    half2   dm;
    uint8_t qs[QK4_1 / 2];
};
static_assert(sizeof(block_q4_1) == 20, "q4_1 size");

// 34 bytes per 32 weights: fp16 scale d, then 32 int8. Value = d * q.
struct block_q8_0 {
    half   d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 size");

// 36 bytes per 64 weights: one half2 (d, m), then two 16-byte halves, each packed exactly like a Q4_1 block:
// qs[16*h + j] holds weight 32*h + j (low nibble) and weight 32*h + 16 + j (high nibble).
struct block_q4_1_64 {
    half2   dm;
    uint8_t qs[QK4_1_64 / 2];
};
static_assert(sizeof(block_q4_1_64) == 36, "q4_1_64 size");

// 22 bytes per 32 weights: fp16 scale d, 4 bytes qh (bit j = fifth bit of weight j), then 16 bytes of nibbles in Q4_0's
// order. Value = d * ((nibble | fifth bit << 4) - 16).
struct block_q5_0 {
    half    d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};
static_assert(sizeof(block_q5_0) == 22, "q5_0 size");

// 144 bytes per 256 weights: half2 (d, dmin), 12 bytes of 6-bit sub-block scales and mins (packed as in Q5_K), 128 bytes
// qs. Sub-block s (32 weights) takes the low (s even) or high (s odd) nibbles of qs[32 (s/2) .. 32 (s/2) + 31].
// Value = d * sc[s] * q - dmin * m[s].
struct block_q4_K {
    half2   dm;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "q4_K size");

// 176 bytes per 256 weights: half2 (d, dmin), 12 bytes of 6-bit sub-block scales and mins, 32 bytes qh, 128 bytes qs.
// Sub-block s (32 weights) takes the low (s even) or high (s odd) nibbles of qs[32 (s/2) .. 32 (s/2) + 31], and bit s of
// qh[0..31] as fifth bit. Value = d * sc[s] * q - dmin * m[s].
struct block_q5_K {
    half2   dm;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 176, "q5_K size");

// 136 bytes per 256 weights: fp16 d, 16 bits scales_h, 4 bytes scales_l, 128 bytes qs. Sub-block ib takes qs[16 ib ..],
// packed like Q4_0; its 6-bit scale is ls = low nibble of scales_l[ib/2] (high nibble for odd ib) | 2 bits of scales_h
// << 4. Value = d * (ls - 32) * kvalues_iq4nl[nibble].
struct block_iq4_xs {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs size");

// 210 bytes per 256 weights (2-byte aligned): 128 bytes ql, 64 bytes qh, 16 int8 scales, fp16 d. Weights 128 h + 32 k + l
// (half h = 0..1, k = 0..3, l = 0..31) take the low (k < 2) or high (k >= 2) nibble of ql[64 h + 32 (k & 1) + l] and bits
// 2k, 2k + 1 of qh[32 h + l]; each 16 weights share one scale, sub-block s = 4h + k (32 weights) uses scales[2s] for
// l < 16 and scales[2s + 1] for l >= 16. Value = d * scale * (q - 32).
struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    half    d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K size");

// Activation block as llama.cpp's CUDA backend builds it: half2 (d, s) with s = fp16 of the sum of the 32 unquantized
// floats, then 32 int8 values. 36 bytes per 32 activations.
struct block_q8_1 {
    half2  ds;
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 size");

template <int T> struct traits;
template <> struct traits<T_Q4_0>    { static constexpr int qk = QK4_0;    static constexpr int qi = QI4_0;    static constexpr int bs = 18; };
template <> struct traits<T_Q4_1>    { static constexpr int qk = QK4_1;    static constexpr int qi = QI4_1;    static constexpr int bs = 20; };
template <> struct traits<T_Q8_0>    { static constexpr int qk = QK8_0;    static constexpr int qi = QI8_0;    static constexpr int bs = 34; };
template <> struct traits<T_Q4_1_64> { static constexpr int qk = QK4_1_64; static constexpr int qi = QI4_1_64; static constexpr int bs = 36; };
template <> struct traits<T_Q5_0>    { static constexpr int qk = QK5_0;    static constexpr int qi = QI5_0;    static constexpr int bs = 22; };
template <> struct traits<T_Q4_K>    { static constexpr int qk = QK_K;     static constexpr int qi = QI4_K;    static constexpr int bs = 144; };
template <> struct traits<T_Q5_K>    { static constexpr int qk = QK_K;     static constexpr int qi = QI5_K;    static constexpr int bs = 176; };
template <> struct traits<T_IQ4_XS>  { static constexpr int qk = QK_K;     static constexpr int qi = QI4_XS;   static constexpr int bs = 136; };
template <> struct traits<T_Q6_K>    { static constexpr int qk = QK_K;     static constexpr int qi = QI6_K;    static constexpr int bs = 210; };

inline const char * type_name(int t) {
    switch (t) {
        case T_Q4_0:    return "q4_0";
        case T_Q4_1:    return "q4_1";
        case T_Q8_0:    return "q8_0";
        case T_Q4_1_64: return "q4_1_64";
        case T_Q5_0:    return "q5_0";
        case T_Q5_K:    return "q5_K";
        case T_IQ4_XS:  return "iq4_xs";
        case T_Q6_K:    return "q6_K";
        case T_Q4_K:    return "q4_K";
        default:        return "?";
    }
}
__host__ __device__ inline bool type_is_k(int t) { return t == T_Q5_K || t == T_IQ4_XS || t == T_Q6_K || t == T_Q4_K; }
__host__ __device__ inline int type_qk(int t) { return type_is_k(t) ? QK_K : t == T_Q4_1_64 ? 64 : 32; }
__host__ __device__ inline int type_bs(int t) {
    switch (t) {
        case T_Q4_0:    return 18;
        case T_Q4_1:    return 20;
        case T_Q8_0:    return 34;
        case T_Q4_1_64: return 36;
        case T_Q5_0:    return 22;
        case T_Q5_K:    return 176;
        case T_IQ4_XS:  return 136;
        case T_Q6_K:    return 210;
        case T_Q4_K:    return 144;
        default:        return 0;
    }
}
inline size_t row_bytes(int t, int K) { return (size_t) (K / type_qk(t)) * type_bs(t); }

constexpr __host__ __device__ int cgcd(int a, int b) { return b == 0 ? a : cgcd(b, a % b); }

} // namespace mb
