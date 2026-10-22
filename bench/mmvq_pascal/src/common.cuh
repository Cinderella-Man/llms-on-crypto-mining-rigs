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
enum wtype : int { T_Q4_0 = 0, T_Q4_1 = 1, T_Q8_0 = 2, T_Q4_1_64 = 3, T_COUNT = 4 };

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

inline const char * type_name(int t) {
    switch (t) {
        case T_Q4_0:    return "q4_0";
        case T_Q4_1:    return "q4_1";
        case T_Q8_0:    return "q8_0";
        case T_Q4_1_64: return "q4_1_64";
        default:        return "?";
    }
}
inline int type_qk(int t) { return t == T_Q4_1_64 ? 64 : 32; }
inline int type_bs(int t) {
    switch (t) {
        case T_Q4_0:    return 18;
        case T_Q4_1:    return 20;
        case T_Q8_0:    return 34;
        case T_Q4_1_64: return 36;
        default:        return 0;
    }
}
inline size_t row_bytes(int t, int K) { return (size_t) (K / type_qk(t)) * type_bs(t); }

constexpr __host__ __device__ int cgcd(int a, int b) { return b == 0 ? a : cgcd(b, a % b); }

} // namespace mb
