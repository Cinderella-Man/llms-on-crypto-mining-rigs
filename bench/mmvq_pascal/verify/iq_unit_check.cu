// iq_unit_check: the decoding of IQ2_XXS, IQ2_XS and IQ3_S in Miner Llama's Pascal matrix-vector kernels
// (ggml/src/ggml-cuda/mmvq-pascal-iq.cuh of the fork, branch pascal-p106-v3-b12076-iq2xs and later) against ggml's own
// dequantization (ggml_get_type_traits(type)->to_float, linked from the same tree's libggml-base), on the CPU, so it
// needs no GPU.
//
// Per type and trial: 4 super-blocks of random bytes (every byte pattern is a valid block of these types) with FP16
// scales of +-[1, 2) * 2^-8, random activations quantized to Q8_1 as quantize_q8_1 does it (d = amax / 127,
// q = round(x / d), d kept as FP16), the kernel's dot function over the 4 blocks, each block read as the kernels read
// it from shared memory (mmvqp_lds_h: words around a block at a 2-byte boundary, shifted by 2 bytes where it starts at
// an odd half-word; both starts are checked, with the bytes around the block set to garbage). The check passes when the
// sum is within 1e-5 of the sum of its absolute terms of a double-precision product of ggml's dequantized weights and the
// same Q8_1 activations. It covers the bit layout, the grids, the signs and the scales; the kernels' staging, lane
// groups and reductions are checked on a GPU (test-backend-ops).
//
// Build and run: bench/mmvq_pascal/verify/iq_unit_check.sh <llama.cpp tree> [trials]

#define MMVQP_IQ_HD __host__ __device__
#include "mmvq-pascal-iq.cuh"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct type_info {
    ggml_type        type;
    const char *     name;
    int              bs;    // bytes per super-block
    const uint32_t * grid;  // the grid as 32-bit words, as the kernels copy it into shared memory
};

float dot_block(const ggml_type t, const uint32_t * w, const int (*u)[8], const float * d8, const uint32_t * grid,
                const float acc) {
    switch (t) {
        case GGML_TYPE_IQ2_XXS: return mmvqp_iq::dot_iq2_xxs(w, u, d8, grid, acc);
        case GGML_TYPE_IQ2_XS:  return mmvqp_iq::dot_iq2_xs(w, u, d8, grid, acc);
        default:                return mmvqp_iq::dot_iq3_s(w, u, d8, grid, acc);
    }
}

// mmvqp_lds_h of mmvq-pascal.cu on the host: the NB / 4 + 1 words of a block at a 2-byte boundary p
void lds_h(uint32_t * w, const uint8_t * p, const int nb) {
    const int        mis = (int) ((uintptr_t) p & 2);
    const uint8_t *  a   = p - mis;
    const uint32_t   sel = mis ? 0x5432 : 0x3210;
    auto word = [&](const int i) { uint32_t v; memcpy(&v, a + 4 * i, 4); return v; };
    uint32_t cur = word(0);
    for (int i = 0; i < nb / 4; ++i) {
        const uint32_t nxt = word(i + 1);
        w[i] = mmvqp_iq::prmt(cur, nxt, sel);
        cur = nxt;
    }
    w[nb / 4] = mmvqp_iq::prmt(cur, 0, sel);
}

uint16_t f2h_bits(const float f) {
    const __half h = __float2half(f);
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}

bool check_case(const type_info & ti, std::mt19937 & rng, const float scale, const bool odd, double & worst) {
    constexpr int NB = 4;           // super-blocks per row
    constexpr int K  = NB * 256;
    std::vector<uint8_t> row((size_t) NB * ti.bs);
    for (auto & b : row) b = (uint8_t) rng();
    for (int b = 0; b < NB; ++b) {
        const uint32_t h = rng();
        const uint16_t d = (uint16_t) (0x1C00 | (h & 0x3FF) | ((h >> 10) & 1) << 15); // +-[1, 2) * 2^-8
        row[(size_t) b * ti.bs + 0] = d & 0xff;
        row[(size_t) b * ti.bs + 1] = d >> 8;
    }
    // activations: Q8_1 blocks, one all-zero block
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> x(K);
    for (auto & v : x) v = nd(rng);
    for (int i = 0; i < 32; ++i) x[32 + i] = 0.0f;
    int   u[K / 32][8];
    float d8[K / 32];
    std::vector<double> av(K);
    for (int blk = 0; blk < K / 32; ++blk) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[32 * blk + i]));
        const float d = amax / 127.0f;
        int8_t q[32];
        for (int i = 0; i < 32; ++i) q[i] = (int8_t) (amax == 0.0f ? 0 : (int) roundf(x[32 * blk + i] / d));
        memcpy(u[blk], q, 32);
        d8[blk] = __half2float(__float2half(d)); // block_q8_1 keeps d as FP16
        for (int i = 0; i < 32; ++i) av[32 * blk + i] = (double) d8[blk] * q[i];
    }
    // reference: ggml's dequantization
    std::vector<float> wf(K);
    ggml_get_type_traits(ti.type)->to_float(row.data(), wf.data(), K);
    double ref = 0.0, as = 0.0;
    for (int i = 0; i < K; ++i) {
        const double t = (double) wf[i] * av[i];
        ref += t;
        as += fabs(t);
    }
    // the kernel's decoding, each block staged at an even or an odd half-word
    alignas(16) uint8_t buf[256];
    uint32_t w[32];
    float acc = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const int off = 32 + (odd ? 2 : 0) + (b & 1) * 4;
        for (auto & c : buf) c = (uint8_t) rng();
        memcpy(buf + off, row.data() + (size_t) b * ti.bs, ti.bs);
        lds_h(w, buf + off, ti.bs);
        acc = dot_block(ti.type, w, u + 8 * b, d8 + 8 * b, ti.grid, acc);
    }
    const double e = std::isfinite(acc) ? fabs(acc - ref) / (1e-5 * as + 1e-30) : 1e30;
    worst = std::max(worst, e);
    if (e > 1.0) {
        printf("    %s odd=%d scale=%g: got %.9g ref %.9g (err/tol %.3g)\n", ti.name, (int) odd, scale, acc, ref, e);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 200;
    const type_info types[] = {
        { GGML_TYPE_IQ2_XXS, "iq2_xxs", (int) sizeof(block_iq2_xxs), (const uint32_t *) iq2xxs_grid },
        { GGML_TYPE_IQ2_XS,  "iq2_xs",  (int) sizeof(block_iq2_xs),  (const uint32_t *) iq2xs_grid  },
        { GGML_TYPE_IQ3_S,   "iq3_s",   (int) sizeof(block_iq3_s),   (const uint32_t *) iq3s_grid   },
    };
    static_assert(sizeof(block_iq2_xxs) == mmvqp_iq::BS_IQ2_XXS && sizeof(block_iq2_xs) == mmvqp_iq::BS_IQ2_XS &&
                  sizeof(block_iq3_s) == mmvqp_iq::BS_IQ3_S, "block sizes");
    static_assert(sizeof(iq2xxs_grid) == 4 * mmvqp_iq::GRID_WORDS_IQ2_XXS && sizeof(iq2xs_grid) == 4 * mmvqp_iq::GRID_WORDS_IQ2_XS &&
                  sizeof(iq3s_grid) == 4 * mmvqp_iq::GRID_WORDS_IQ3_S, "grid sizes");
    bool all = true;
    for (const auto & ti : types) {
        std::mt19937 rng(1234 + (int) ti.type);
        bool ok = true;
        double worst = 0.0;
        int n = 0;
        for (int tr = 0; tr < trials; ++tr) {
            // ordinary activations, and large ones (the grid types use no activation sums, so FP16's range is moot)
            const float scale = tr % 4 == 3 ? 3000.0f : 1.5f;
            ok &= check_case(ti, rng, scale, false, worst);
            ok &= check_case(ti, rng, scale, true, worst);
            n += 2;
        }
        printf("%-8s %5d cases  worst err/tol %.4f  %s\n", ti.name, n, worst, ok ? "PASS" : "FAIL");
        all &= ok;
    }
    printf("decoding: %s\n", all ? "every type matches ggml's dequantization" : "FAILURES");
    return all ? 0 : 1;
}
