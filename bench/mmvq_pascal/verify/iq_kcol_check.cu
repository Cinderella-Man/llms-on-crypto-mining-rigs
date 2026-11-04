// iq_kcol_check: the decoding of the grid types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) and IQ1_M in Miner Llama's
// k-column and multi-token Pascal kernels (Q713: mmvqp_iq::decode32 of ggml/src/ggml-cuda/mmvq-pascal-iq.cuh, the unit
// function unit_iq of mmvq-pascal-kcol-impl.cuh, fork branch pascal-p106-v3-b12076-iqkcol and later) against ggml's own
// dequantization (ggml_get_type_traits(type)->to_float, linked from the same tree's libggml-base), on the CPU, so it
// needs no GPU. Two parts per type:
//   kcol  the k-column unit function at 1, 3 and 8 columns and 1 and 2 rows over 4 super-blocks per row, each placed
//         at the alignments the staging buffer gives it (2-byte-aligned types at both half-word parities), random
//         activations quantized into the kernels' records by their quantizer's rule (as quantize_q8_1: d = amax / 127,
//         q = round(x / d), d kept as FP16; IQ1_M's values in its lookup order), checked within 1e-5 of the sum of the
//         absolute terms of a double-precision product of ggml's dequantized weights and the same activations;
//   mtok  the multi-token expert kernel's arithmetic per token and super-block (mmvq-pascal-mtok-iq.cuh: decode32, two
//         dp4a chains per 32 weights, sumf = fmaf(d8, ls0 i0 + ls1 i1, sumf), acc = fmaf(dd, sumf, acc)), the block
//         read as the kernel reads it from shared memory (mmvqp_lds_h for the 2-byte-aligned types), against the same
//         reference, and for IQ2_XXS, IQ2_XS and IQ3_S bit for bit against the one-token kernels' functions
//         (mmvqp_iq::dot_*), which the kernel's results must equal (IQ2_S's and IQ3_XXS's one-token functions need
//         ggml's CUDA headers; IQ1_M has none).
// Weights are random bytes (every byte pattern is a valid block of these types) with FP16 scales of +-[1, 2) * 2^-8;
// activations include an all-zero block and, every fourth trial, values in the thousands.
//
// Build and run: bench/mmvq_pascal/verify/iq_kcol_check.sh <llama.cpp tree> [trials]

#define KCOL_HD __host__ __device__
#define MMVQP_IQ_HD __host__ __device__
#include "mmvq-pascal-kcol-impl.cuh"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"
// ggml-common.h defines the GPU form of the IQ1_M grid only for the GPU backends: iq_kcol_check.sh copies it from the same
// file into iq1s_grid_gpu.h (namespace gpu_tab)
#include "iq1s_grid_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace kcol;

struct tinfo {
    int              kt;    // kcol type
    ggml_type        type;
    const char *     name;
    const uint32_t * grid;  // as the kernels copy it into shared memory
};

uint32_t f2u(const float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// Random super-blocks: random bytes, then a finite FP16 d of +-[1, 2) * 2^-8 (IQ1_M: in the top nibbles of its four
// 16-bit scale words).
void random_blocks(const ggml_type type, uint8_t * w, const int nblocks, const int bs, std::mt19937 & rng) {
    for (int b = 0; b < nblocks; ++b) {
        uint8_t * blk = w + (size_t) b * bs;
        for (int i = 0; i < bs; ++i) blk[i] = (uint8_t) rng();
        const uint32_t h = rng();
        const uint16_t d = (uint16_t) (0x1C00 | (h & 0x3FF) | ((h >> 10) & 1) << 15);
        if (type == GGML_TYPE_IQ1_M) {
            for (int k = 0; k < 4; ++k) {
                uint16_t s;
                memcpy(&s, blk + 48 + 2 * k, 2);
                s = (uint16_t) ((s & 0x0FFF) | (((d >> (4 * k)) & 0xF) << 12));
                memcpy(blk + 48 + 2 * k, &s, 2);
            }
        } else {
            blk[0] = d & 0xff;
            blk[1] = d >> 8;
        }
    }
}

// Q8_1 quantization of 32-value blocks as quantize_q8_1: int8 values q, FP16-rounded scales d8.
void quantize_q8(const float * x, const int K, int8_t * q, float * d8) {
    for (int blk = 0; blk < K / 32; ++blk) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[32 * blk + i]));
        const float d = amax / 127.0f;
        for (int i = 0; i < 32; ++i) q[32 * blk + i] = (int8_t) (amax == 0.0f ? 0 : (int) roundf(x[32 * blk + i] / d));
        d8[blk] = __half2float(__float2half(d));
    }
}

// The kernels' records (quantize_records) for one column, ACT_D: per super-block the int8 values at act_pos, (d8, 0).
template <int T>
void records(const int8_t * q, const float * d8, const int K, uint32_t * rec) {
    constexpr int UW = traits<T>::UW, NAB = UW / 32, RS = rec_words<T>();
    for (int blk = 0; blk < K / 32; ++blk) {
        const int u = blk / NAB, b = blk % NAB;
        uint32_t * r = rec + (size_t) u * RS;
        for (int i = 0; i < 32; ++i) ((int8_t *) r)[32 * b + act_pos<traits<T>::PERM>(i)] = q[32 * blk + i];
        r[UW / 4 + 2 * b]     = f2u(d8[blk]);
        r[UW / 4 + 2 * b + 1] = 0;
    }
}

bool report(const char * what, const tinfo & ti, const int nc, const int r, const int i, const int j, const double got,
            const double ref, const double e, bool & first) {
    if (first) {
        printf("    %s %s NC=%d R=%d row %d col %d: got %.9g ref %.9g (err/tol %.3g)\n", what, ti.name, nc, r, i, j, got, ref, e);
        first = false;
    }
    return false;
}

template <int T, int NC, int R>
bool check_kcol(const tinfo & ti, std::mt19937 & rng, const float scale, double & worst) {
    constexpr int UB = traits<T>::UB, AL = traits<T>::AL, RS = rec_words<T>();
    constexpr int NU = 4, K = NU * 256;
    std::vector<uint8_t> rows((size_t) R * NU * UB);
    random_blocks(ti.type, rows.data(), R * NU, UB, rng);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> x((size_t) NC * K);
    for (auto & v : x) v = nd(rng);
    for (int i = 0; i < 32; ++i) x[32 + i] = 0.0f;
    std::vector<int8_t> q((size_t) NC * K);
    std::vector<float>  d8((size_t) NC * K / 32);
    std::vector<uint32_t> rec((size_t) NC * NU * RS, 0);
    for (int j = 0; j < NC; ++j) {
        quantize_q8(x.data() + (size_t) j * K, K, q.data() + (size_t) j * K, d8.data() + (size_t) j * K / 32);
        records<T>(q.data() + (size_t) j * K, d8.data() + (size_t) j * K / 32, K, rec.data() + (size_t) j * NU * RS);
    }
    std::vector<float> wf((size_t) R * K);
    for (int i = 0; i < R; ++i) ggml_get_type_traits(ti.type)->to_float(rows.data() + (size_t) i * NU * UB, wf.data() + (size_t) i * K, K);

    float acc[R][NC];
    for (int i = 0; i < R; ++i) for (int j = 0; j < NC; ++j) acc[i][j] = 0.0f;
    alignas(16) static uint8_t buf[2][512];
    for (int u = 0; u < NU; ++u) {
        unit_ptr wu[R];
        for (int i = 0; i < R; ++i) {
            // odd units at the smallest misalignment the type allows in the staging buffer
            const int off = 64 + ((u & 1) ? AL : 0) + (u == 3 && AL == 2 ? 4 : 0);
            memset(buf[i], 0xA5, sizeof(buf[i]));
            memcpy(buf[i] + off, rows.data() + ((size_t) i * NU + u) * UB, UB);
            wu[i] = make_unit<AL>((const char *) buf[i] + off);
        }
        unit_dot<T, NC, R>(wu, rec.data() + (size_t) u * RS, NU * RS, acc, ti.grid);
    }
    bool ok = true, first = true;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < NC; ++j) {
            double ref = 0.0, as = 0.0;
            for (int c = 0; c < K; ++c) {
                const double t = (double) wf[(size_t) i * K + c] * ((double) d8[((size_t) j * K + c) / 32] * q[(size_t) j * K + c]);
                ref += t;
                as += fabs(t);
            }
            const double e = std::isfinite(acc[i][j]) ? fabs(acc[i][j] - ref) / (1e-5 * as + 1e-30) : 1e30;
            worst = std::max(worst, e);
            if (e > 1.0) ok = report("kcol", ti, NC, R, i, j, acc[i][j], ref, e, first);
        }
    }
    return ok;
}

// mmvqp_lds_h on the host: the NB / 4 + 1 words of a block at a 2-byte boundary p
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

// The multi-token kernel's sum of one token over one super-block (mmvq-pascal-mtok-iq.cuh, rows()); u: the staged
// activations of its 8 Q8_1 blocks (IQ1_M: each group of 8 values in its lookup order).
template <int G>
float mtok_sblock(const uint32_t * w, const int (*u)[8], const float * d8, const uint32_t * grid, const float acc) {
    const float dd = mmvqp_iq::sblock_scale<G>(w);
    float sumf = 0.0f;
    for (int ib = 0; ib < 8; ++ib) {
        int q[8];
        int ls0, ls1;
        mmvqp_iq::decode32<G>(w, ib, grid, q, ls0, ls1);
        int i0 = 0, i1 = 0;
        for (int k = 0; k < 4; ++k) {
            i0 = mmvqp_iq::dp4a(q[k],     u[ib][k],     i0);
            i1 = mmvqp_iq::dp4a(q[k + 4], u[ib][k + 4], i1);
        }
        sumf = fmaf(d8[ib], (float) (ls0 * i0 + ls1 * i1), sumf);
    }
    return fmaf(dd, sumf, acc);
}

template <int T>
bool check_mtok(const tinfo & ti, std::mt19937 & rng, const float scale, const bool odd, double & worst, int & nbit,
                int & nbit_ok) {
    constexpr int G = gtype_of<T>();
    constexpr int BS = mmvqp_iq::gtraits<G>::BS;
    constexpr int NB = 4, K = NB * 256;
    std::vector<uint8_t> row((size_t) NB * BS);
    random_blocks(ti.type, row.data(), NB, BS, rng);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> x(K);
    for (auto & v : x) v = nd(rng);
    for (int i = 0; i < 32; ++i) x[32 + i] = 0.0f;
    std::vector<int8_t> q(K);
    float d8[K / 32];
    quantize_q8(x.data(), K, q.data(), d8);
    int un[K / 32][8], up[K / 32][8]; // natural order (the one-token kernels), staged order (IQ1_M permuted)
    for (int blk = 0; blk < K / 32; ++blk) {
        int8_t pq[32];
        for (int i = 0; i < 32; ++i) pq[G == mmvqp_iq::G_IQ1_M ? mmvqp_iq::act_perm_iq1(i) : i] = q[32 * blk + i];
        memcpy(un[blk], q.data() + 32 * blk, 32);
        memcpy(up[blk], pq, 32);
    }
    std::vector<float> wf(K);
    ggml_get_type_traits(ti.type)->to_float(row.data(), wf.data(), K);
    double ref = 0.0, as = 0.0;
    for (int c = 0; c < K; ++c) {
        const double t = (double) wf[c] * ((double) d8[c / 32] * q[c]);
        ref += t;
        as += fabs(t);
    }
    alignas(16) uint8_t buf[256];
    uint32_t w[32];
    float acc = 0.0f, acc1 = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const int off = 32 + (BS % 4 == 2 ? (odd ? 2 : 0) + (b & 1) * 4 : 8 * (b & 1));
        for (auto & c : buf) c = (uint8_t) rng();
        memcpy(buf + off, row.data() + (size_t) b * BS, BS);
        if (BS % 4 == 2) {
            lds_h(w, buf + off, BS);
        } else {
            memcpy(w, buf + off, BS);
        }
        acc = mtok_sblock<G>(w, up + 8 * b, d8 + 8 * b, ti.grid, acc);
        if constexpr (G == mmvqp_iq::G_IQ2_XXS) acc1 = mmvqp_iq::dot_iq2_xxs(w, un + 8 * b, d8 + 8 * b, ti.grid, acc1);
        if constexpr (G == mmvqp_iq::G_IQ2_XS)  acc1 = mmvqp_iq::dot_iq2_xs(w, un + 8 * b, d8 + 8 * b, ti.grid, acc1);
        if constexpr (G == mmvqp_iq::G_IQ3_S)   acc1 = mmvqp_iq::dot_iq3_s(w, un + 8 * b, d8 + 8 * b, ti.grid, acc1);
    }
    bool ok = true, first = true;
    const double e = std::isfinite(acc) ? fabs(acc - ref) / (1e-5 * as + 1e-30) : 1e30;
    worst = std::max(worst, e);
    if (e > 1.0) ok = report("mtok", ti, 1, 1, 0, 0, acc, ref, e, first);
    if (G == mmvqp_iq::G_IQ2_XXS || G == mmvqp_iq::G_IQ2_XS || G == mmvqp_iq::G_IQ3_S) {
        nbit++;
        if (f2u(acc) == f2u(acc1)) {
            nbit_ok++;
        } else {
            if (first) printf("    mtok %s: %.9g, the one-token function %.9g (not bit-identical)\n", ti.name, acc, acc1);
            ok = false;
        }
    }
    return ok;
}

template <int T>
bool check_type(const tinfo & ti, const int trials) {
    std::mt19937 rng(4321 + T);
    bool ok = true;
    double wk = 0.0, wm = 0.0;
    int nk = 0, nm = 0, nbit = 0, nbit_ok = 0;
    for (int tr = 0; tr < trials; ++tr) {
        const float scale = tr % 4 == 3 ? 3000.0f : 1.5f;
        ok &= check_kcol<T, 1, 1>(ti, rng, scale, wk);
        ok &= check_kcol<T, 3, 1>(ti, rng, scale, wk);
        ok &= check_kcol<T, 8, 1>(ti, rng, scale, wk);
        ok &= check_kcol<T, 1, 2>(ti, rng, scale, wk);
        ok &= check_kcol<T, 3, 2>(ti, rng, scale, wk);
        ok &= check_kcol<T, 8, 2>(ti, rng, scale, wk);
        nk += 6;
        ok &= check_mtok<T>(ti, rng, scale, false, wm, nbit, nbit_ok);
        ok &= check_mtok<T>(ti, rng, scale, true, wm, nbit, nbit_ok);
        nm += 2;
    }
    char bit[64] = "";
    if (nbit > 0) snprintf(bit, sizeof(bit), ", %d/%d bit-identical to the one-token function", nbit_ok, nbit);
    printf("%-8s kcol %4d cases worst err/tol %.4f; mtok %4d cases worst %.4f%s  %s\n", ti.name, nk, wk, nm, wm, bit,
           ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 100;
    static_assert(sizeof(block_iq2_xxs) == traits<KIQ2_XXS>::UB && sizeof(block_iq2_xs) == traits<KIQ2_XS>::UB &&
                  sizeof(block_iq2_s) == traits<KIQ2_S>::UB && sizeof(block_iq3_xxs) == traits<KIQ3_XXS>::UB &&
                  sizeof(block_iq3_s) == traits<KIQ3_S>::UB && sizeof(block_iq1_m) == traits<KIQ1_M>::UB, "block sizes");
    static_assert(sizeof(iq2xxs_grid) == 4 * grid_words<KIQ2_XXS>() && sizeof(iq2xs_grid) == 4 * grid_words<KIQ2_XS>() &&
                  sizeof(iq2s_grid) == 4 * grid_words<KIQ2_S>() && sizeof(iq3xxs_grid) == 4 * grid_words<KIQ3_XXS>() &&
                  sizeof(iq3s_grid) == 4 * grid_words<KIQ3_S>() && sizeof(gpu_tab::iq1s_grid_gpu) == 4 * grid_words<KIQ1_M>(),
                  "grid sizes");
    const tinfo types[] = {
        { KIQ2_XXS, GGML_TYPE_IQ2_XXS, "iq2_xxs", (const uint32_t *) iq2xxs_grid },
        { KIQ2_XS,  GGML_TYPE_IQ2_XS,  "iq2_xs",  (const uint32_t *) iq2xs_grid  },
        { KIQ2_S,   GGML_TYPE_IQ2_S,   "iq2_s",   (const uint32_t *) iq2s_grid   },
        { KIQ3_XXS, GGML_TYPE_IQ3_XXS, "iq3_xxs", iq3xxs_grid },
        { KIQ3_S,   GGML_TYPE_IQ3_S,   "iq3_s",   iq3s_grid   },
        { KIQ1_M,   GGML_TYPE_IQ1_M,   "iq1_m",   gpu_tab::iq1s_grid_gpu },
    };
    // the GPU grid is the CPU grid (values -1, 0, 1) plus one as nibbles: weights 0..3 in the low nibbles of bytes 0..3,
    // 4..7 in the high nibbles (the layout decode32 reads)
    int ngrid_bad = 0;
    for (int i = 0; i < NGRID_IQ1S; ++i) {
        const int8_t * g = (const int8_t *) (iq1s_grid + i);
        uint32_t w = 0;
        for (int j = 0; j < 4; ++j) {
            w |= (uint32_t) (g[j] + 1) << (8 * j) | (uint32_t) (g[j + 4] + 1) << (8 * j + 4);
        }
        ngrid_bad += w != gpu_tab::iq1s_grid_gpu[i];
    }
    printf("iq1s_grid_gpu: %s\n", ngrid_bad == 0 ? "the CPU grid plus one as nibbles (weights 0-3 low, 4-7 high)" : "LAYOUT DIFFERS");
    bool ok = true;
    ok &= check_type<KIQ2_XXS>(types[0], trials);
    ok &= check_type<KIQ2_XS>(types[1], trials);
    ok &= check_type<KIQ2_S>(types[2], trials);
    ok &= check_type<KIQ3_XXS>(types[3], trials);
    ok &= check_type<KIQ3_S>(types[4], trials);
    ok &= check_type<KIQ1_M>(types[5], trials);
    ok &= ngrid_bad == 0;
    printf("decoding: %s\n", ok ? "every type matches ggml's dequantization" : "FAILURES");
    return ok ? 0 : 1;
}
