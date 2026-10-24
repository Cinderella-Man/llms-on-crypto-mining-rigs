// Deterministic random weights and activations (generated on the GPU), and the CPU reference dot products.
#pragma once

#include "common.cuh"

#include <cmath>
#include <vector>

namespace mb {

static __host__ __device__ __forceinline__ uint32_t hash_u32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (uint32_t) x;
}

static __host__ __device__ __forceinline__ float u01(uint32_t h) { return (h >> 8) * (1.0f / 16777216.0f); }

// Fill nblocks weight blocks of the given type. Scales are positive FP16 normals; Q4_1 minimums are negative, as in
// real affine quants (m ~ -8 d). Quants, the Q5 fifth bits and the K/I-quant sub-block scales are uniform random.
static __global__ void gen_blocks(uint8_t * __restrict__ dst, const size_t nblocks, const int type, const uint64_t seed) {
    const int bs = type_bs(type);
    for (size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x; b < nblocks; b += (size_t) gridDim.x * blockDim.x) {
        uint8_t * p = dst + b * bs;
        const uint64_t base = seed * 0x2545F4914F6CDD1Dull + b * 64;
        const float r0 = u01(hash_u32(base + 0));
        const float r1 = u01(hash_u32(base + 1));
        int qoff = 0;
        if (type == T_Q4_0 || type == T_Q8_0 || type == T_Q5_0 || type == T_IQ4_XS) {
            const float d = type == T_Q4_0 ? (0.5f + r0) * (1.0f / 64.0f) : type == T_Q8_0 ? (0.5f + r0) * (1.0f / 512.0f)
                          : type == T_Q5_0 ? (0.5f + r0) * (1.0f / 128.0f) : (0.5f + r0) * (1.0f / 8192.0f);
            *(half *) p = __float2half(d);
            qoff = 2;
        } else if (type == T_Q5_K) {
            const float d    = (0.5f + r0) * (1.0f / 2048.0f);
            const float dmin = (0.5f + r1) * (1.0f / 2048.0f);
            *(half2 *) p = __halves2half2(__float2half(d), __float2half(dmin));
            qoff = 4;
        } else if (type == T_Q6_K) {
            // d comes last; ql, qh and the int8 scales are filled with random bytes below
            const float d = (0.5f + r0) * (1.0f / 8192.0f);
            *(half *) (p + 208) = __float2half(d); // 210-byte blocks: p + 208 is 2-byte aligned
        } else {
            const float d = (0.5f + r0) * (1.0f / 64.0f);
            const float m = -d * (6.0f + 4.0f * r1);
            *(half2 *) p = __halves2half2(__float2half(d), __float2half(m));
            qoff = 4;
        }
        const int nq = type == T_Q6_K ? 208 : bs - qoff;
        for (int j = 0; j < nq; j += 4) {
            uint32_t h = hash_u32(base + 2 + j);
            for (int k = 0; k < 4 && j + k < nq; ++k) {
                uint8_t v = (uint8_t) (h >> (8 * k));
                if (type == T_Q8_0 && v == 0x80) {
                    v = 0x81; // keep int8 quants in [-127, 127] like ggml's Q8_0
                }
                p[qoff + j + k] = v;
            }
        }
    }
}

// Activations: roughly normal, sd ~1.5.
static __global__ void gen_floats(float * __restrict__ dst, const size_t n, const uint64_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint64_t base = seed * 0x2545F4914F6CDD1Dull + i * 4;
        const float s = u01(hash_u32(base)) + u01(hash_u32(base + 1)) + u01(hash_u32(base + 2)) + u01(hash_u32(base + 3));
        dst[i] = (s - 2.0f) * 2.6f;
    }
}

static inline float half_bits_to_float(uint16_t bits) {
    __half_raw r;
    r.x = bits;
    return __half2float(__half(r));
}
static inline uint16_t rd_u16(const uint8_t * p) { return (uint16_t) (p[0] | (p[1] << 8)); }

// Quantized activations as the kernels see them: int8 values and per-32 FP32 scale d and sum s
// (for Q8_1 blocks, d and s are the FP16 values converted to float).
struct act_host {
    int K = 0;
    int nvec = 0;
    std::vector<int8_t> q;
    std::vector<float> d, s;
};

static act_host act_from_q8_1(const std::vector<block_q8_1> & b, int K, int nvec) {
    act_host a;
    a.K = K;
    a.nvec = nvec;
    a.q.resize((size_t) K * nvec);
    a.d.resize((size_t) K / 32 * nvec);
    a.s.resize((size_t) K / 32 * nvec);
    for (int v = 0; v < nvec; ++v) {
        for (int ib = 0; ib < K / 32; ++ib) {
            const block_q8_1 & blk = b[(size_t) v * (K / 32) + ib];
            const uint8_t * raw = (const uint8_t *) &blk;
            a.d[(size_t) v * (K / 32) + ib] = half_bits_to_float(rd_u16(raw));
            a.s[(size_t) v * (K / 32) + ib] = half_bits_to_float(rd_u16(raw + 2));
            for (int j = 0; j < 32; ++j) {
                a.q[(size_t) v * K + ib * 32 + j] = blk.qs[j];
            }
        }
    }
    return a;
}

// Reference dot product of one weight row with activation vector `vec`, using exactly the integer quants and scales
// the GPU uses, accumulated in double. absum returns the sum of |terms| (the scale for the error bound).
static double ref_dot(int type, const uint8_t * row, const act_host & a, int vec, double * absum) {
    const int K = a.K;
    const int8_t * q8 = a.q.data() + (size_t) vec * K;
    const float  * d8 = a.d.data() + (size_t) vec * (K / 32);
    const float  * s8 = a.s.data() + (size_t) vec * (K / 32);
    double acc = 0.0, as = 0.0;
    auto q4sum = [&](const uint8_t * qs, int ib) {
        int sumi = 0;
        for (int j = 0; j < 16; ++j) {
            sumi += (qs[j] & 15) * q8[32 * ib + j] + (qs[j] >> 4) * q8[32 * ib + j + 16];
        }
        return sumi;
    };
    switch (type) {
        case T_Q4_0:
            for (int ib = 0; ib < K / 32; ++ib) {
                const uint8_t * b = row + (size_t) ib * 18;
                const double d = half_bits_to_float(rd_u16(b));
                const double t1 = d * d8[ib] * q4sum(b + 2, ib);
                const double t2 = -8.0 * d * s8[ib];
                acc += t1 + t2;
                as += fabs(t1) + fabs(t2);
            }
            break;
        case T_Q4_1:
            for (int ib = 0; ib < K / 32; ++ib) {
                const uint8_t * b = row + (size_t) ib * 20;
                const double d = half_bits_to_float(rd_u16(b));
                const double m = half_bits_to_float(rd_u16(b + 2));
                const double t1 = d * d8[ib] * q4sum(b + 4, ib);
                const double t2 = m * s8[ib];
                acc += t1 + t2;
                as += fabs(t1) + fabs(t2);
            }
            break;
        case T_Q8_0:
            for (int ib = 0; ib < K / 32; ++ib) {
                const uint8_t * b = row + (size_t) ib * 34;
                const double d = half_bits_to_float(rd_u16(b));
                int sumi = 0;
                for (int j = 0; j < 32; ++j) {
                    sumi += (int8_t) b[2 + j] * q8[32 * ib + j];
                }
                const double t = d * d8[ib] * sumi;
                acc += t;
                as += fabs(t);
            }
            break;
        case T_Q4_1_64:
            for (int ig = 0; ig < K / 64; ++ig) {
                const uint8_t * b = row + (size_t) ig * 36;
                const double d = half_bits_to_float(rd_u16(b));
                const double m = half_bits_to_float(rd_u16(b + 2));
                for (int h = 0; h < 2; ++h) {
                    const int ib = 2 * ig + h;
                    const double t1 = d * d8[ib] * q4sum(b + 4 + 16 * h, ib);
                    const double t2 = m * s8[ib];
                    acc += t1 + t2;
                    as += fabs(t1) + fabs(t2);
                }
            }
            break;
        case T_Q5_0:
            for (int ib = 0; ib < K / 32; ++ib) {
                const uint8_t * b = row + (size_t) ib * 22;
                const double d = half_bits_to_float(rd_u16(b));
                const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
                int sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    const int q0 = (b[6 + j] & 15) | (((qh >> j) & 1) << 4);
                    const int q1 = (b[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
                    sumi += q0 * q8[32 * ib + j] + q1 * q8[32 * ib + j + 16];
                }
                const double t1 = d * d8[ib] * sumi;
                const double t2 = -16.0 * d * s8[ib];
                acc += t1 + t2;
                as += fabs(t1) + fabs(t2);
            }
            break;
        case T_Q5_K:
            for (int sb = 0; sb < K / 256; ++sb) {
                const uint8_t * b = row + (size_t) sb * 176;
                const double d    = half_bits_to_float(rd_u16(b));
                const double dmin = half_bits_to_float(rd_u16(b + 2));
                const uint8_t * sc = b + 4;
                const uint8_t * qh = b + 16;
                const uint8_t * qs = b + 48;
                for (int s = 0; s < 8; ++s) {
                    int scv, mv;
                    if (s < 4) {
                        scv = sc[s] & 63;
                        mv  = sc[s + 4] & 63;
                    } else {
                        scv = (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4);
                        mv  = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
                    }
                    const int ib = 8 * sb + s;
                    int sumi = 0, suma = 0;
                    for (int l = 0; l < 32; ++l) {
                        const int q = ((qs[32 * (s / 2) + l] >> (4 * (s % 2))) & 15) | (((qh[l] >> s) & 1) << 4);
                        sumi += q * q8[32 * ib + l];
                        suma += q8[32 * ib + l];
                    }
                    const double t1 = d * scv * d8[ib] * sumi;
                    const double t2 = -dmin * mv * d8[ib] * suma;
                    acc += t1 + t2;
                    as += fabs(t1) + fabs(t2);
                }
            }
            break;
        case T_Q6_K:
            for (int sb = 0; sb < K / 256; ++sb) {
                const uint8_t * b = row + (size_t) sb * 210;
                const double d = half_bits_to_float(rd_u16(b + 208));
                const uint8_t * ql = b;
                const uint8_t * qh = b + 128;
                const int8_t  * sc = (const int8_t *) (b + 192);
                for (int s = 0; s < 8; ++s) {
                    const int h = s / 4, k = s % 4;
                    const int ib = 8 * sb + s;
                    for (int g = 0; g < 2; ++g) {
                        int sumi = 0;
                        for (int l = 16 * g; l < 16 * g + 16; ++l) {
                            const int lo = ql[64 * h + 32 * (k & 1) + l];
                            const int q  = ((k < 2 ? lo : lo >> 4) & 15) | (((qh[32 * h + l] >> (2 * k)) & 3) << 4);
                            sumi += (q - 32) * q8[32 * ib + l];
                        }
                        const double t = d * sc[2 * s + g] * d8[ib] * sumi;
                        acc += t;
                        as += fabs(t);
                    }
                }
            }
            break;
        case T_IQ4_XS: {
            static const int kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
            for (int sb = 0; sb < K / 256; ++sb) {
                const uint8_t * b = row + (size_t) sb * 136;
                const double d = half_bits_to_float(rd_u16(b));
                const int scales_h = rd_u16(b + 2);
                const uint8_t * scales_l = b + 4;
                const uint8_t * qs = b + 8;
                for (int j8 = 0; j8 < 8; ++j8) {
                    const int ls = ((scales_l[j8 / 2] >> (4 * (j8 % 2))) & 0xF) | (((scales_h >> (2 * j8)) & 3) << 4);
                    const int ib = 8 * sb + j8;
                    int sumi = 0;
                    for (int j = 0; j < 16; ++j) {
                        sumi += kv[qs[16 * j8 + j] & 15] * q8[32 * ib + j] + kv[qs[16 * j8 + j] >> 4] * q8[32 * ib + j + 16];
                    }
                    const double t = d * (ls - 32) * d8[ib] * sumi;
                    acc += t;
                    as += fabs(t);
                }
            }
            break;
        }
    }
    *absum = as;
    return acc;
}

} // namespace mb
