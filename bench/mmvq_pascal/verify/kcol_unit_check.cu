// kcol_unit_check: each type's decoding in the k-column products (src/kcol_mm.cuh) against ggml's reference
// dequantization, on the CPU, so it needs no GPU: random weight units placed at every alignment the staging buffer gives
// them, random activations quantized into records as the kernels' quantizer does it, and each type's unit function at
// 1, 3 and 8 columns and 1 and 2 rows, summed over 4 units per row. The check passes when every row's sum is within
// 1e-5 of the sum of its absolute terms of a double-precision product of the dequantized weights and the record's
// activations. It covers the decoding, the record layout and the activation terms; the kernels' staging, tiles and
// reductions are checked on a GPU (test-backend-ops, build/mmvq_kcol).
//
// Build and run (any x86-64 machine with Docker; no GPU): bench/mmvq_pascal/verify/kcol_unit_check.sh

#define KCOL_HD __host__ __device__
#include "../src/kcol_ref.h"

#include <cuda_fp16.h>

#include <cstdio>
#include <random>
#include <vector>

using namespace kcol;

namespace {

uint32_t f2u(const float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// The kernels' quantizer (quantize_records) for one column of K values, written for the host.
template <int T>
void quantize_host(const float * x, const int K, uint32_t * rec) {
    constexpr int UW = traits<T>::UW, NAB = UW / 32, RS = rec_words<T>();
    for (int blk = 0; blk < K / 32; ++blk) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[32 * blk + i]));
        const float d = amax / 127.0f;
        int q[32];
        int s16[2] = { 0, 0 };
        for (int i = 0; i < 32; ++i) {
            q[i] = amax == 0.0f ? 0 : (int) roundf(x[32 * blk + i] / d);
            s16[i / 16] += q[i];
        }
        const int s32 = s16[0] + s16[1];
        const int u = blk / NAB, b = blk % NAB;
        uint32_t * r = rec + (size_t) u * RS;
        for (int i = 0; i < 32; ++i) ((int8_t *) r)[32 * b + act_pos<traits<T>::PERM>(i)] = (int8_t) q[i];
        const float d8 = __half2float(__float2half(d));
        if (traits<T>::ACT == ACT_DH) {
            int f = 1, m = -32;
            if (T == KQ3_K) {
                f = 1 << (2 * std::min(b & 3, 2));
                m = -4 * f;
            }
            r[UW / 4 + 2 * b]               = f2u(d8 / (float) f);
            r[UW / 4 + 2 * b + 1]           = 0;
            r[UW / 4 + 2 * NAB + 2 * b]     = (uint32_t) (m * s16[0]);
            r[UW / 4 + 2 * NAB + 2 * b + 1] = (uint32_t) (m * s16[1]);
        } else {
            r[UW / 4 + 2 * b]     = f2u(d8);
            r[UW / 4 + 2 * b + 1] = f2u(traits<T>::ACT == ACT_DS ? d8 * (float) s32 : 0.0f);
        }
    }
}

// Random blocks with every FP16 scale set to +-[1, 2) * 2^-8 (as build/mmvq_kcol's weights).
void random_blocks(const int t, uint8_t * w, const size_t nblocks, std::mt19937 & rng) {
    const int bs = kbs(t);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t * blk = w + b * bs;
        for (int i = 0; i < bs; ++i) blk[i] = (uint8_t) rng();
        const uint32_t h = rng();
        const uint16_t d    = (uint16_t) (0x1C00 | (h & 0x3FF) | ((h >> 10) & 1) << 15);
        const uint16_t dmin = (uint16_t) (0x1C00 | ((h >> 11) & 0x3FF));
        const int off = t == KQ3_K ? 108 : t == KQ6_K ? 208 : 0;
        blk[off] = d & 0xff;
        blk[off + 1] = d >> 8;
        if (t == KQ4_K || t == KQ5_K) {
            blk[2] = dmin & 0xff;
            blk[3] = dmin >> 8;
        }
    }
}

template <int T, int NC, int R>
bool check_case(std::mt19937 & rng, const float scale, double & worst_out) {
    constexpr int UB = traits<T>::UB, UW = traits<T>::UW, AL = traits<T>::AL, RS = rec_words<T>();
    constexpr int NU = 4; // units per row
    constexpr int K = NU * UW;
    const size_t rbytes = (size_t) K / kqk(T) * kbs(T);
    std::vector<uint8_t> rows(R * rbytes);
    random_blocks(T, rows.data(), R * rbytes / kbs(T), rng);

    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> x((size_t) NC * K);
    for (auto & v : x) v = nd(rng);
    for (int i = 0; i < 32; ++i) x[i] = 0.0f; // one all-zero block
    std::vector<uint32_t> rec((size_t) NC * NU * RS, 0);
    for (int j = 0; j < NC; ++j) quantize_host<T>(x.data() + (size_t) j * K, K, rec.data() + (size_t) j * NU * RS);
    std::vector<double> av;
    act_values(T, rec, K, NC, av);
    std::vector<double> wd((size_t) R * K);
    for (int i = 0; i < R; ++i) dequant_row(T, rows.data() + i * rbytes, K, wd.data() + (size_t) i * K);

    float acc[R][NC];
    for (int i = 0; i < R; ++i) for (int j = 0; j < NC; ++j) acc[i][j] = 0.0f;
    alignas(16) static uint8_t buf[2][1024];
    for (int u = 0; u < NU; ++u) {
        unit_ptr wu[R];
        for (int i = 0; i < R; ++i) {
            // odd units at the smallest misalignment the type allows in the staging buffer
            const int off = 64 + ((u & 1) && AL < 16 ? AL : 0) + (u == 3 && AL == 2 ? 4 : 0);
            memset(buf[i], 0xA5, sizeof(buf[i]));
            memcpy(buf[i] + off, rows.data() + i * rbytes + (size_t) u * UB, UB);
            wu[i] = make_unit<AL>((const char *) buf[i] + off);
        }
        unit_dot<T, NC, R>(wu, rec.data() + (size_t) u * RS, NU * RS, acc);
    }
    bool ok = true;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < NC; ++j) {
            double ref = 0.0, as = 0.0;
            for (int c = 0; c < K; ++c) {
                const double t = wd[(size_t) i * K + c] * av[(size_t) j * K + c];
                ref += t;
                as += fabs(t);
            }
            const double e = std::isfinite(acc[i][j]) ? fabs(acc[i][j] - ref) / (1e-5 * as + 1e-30) : 1e30;
            worst_out = std::max(worst_out, e);
            if (e > 1.0) {
                if (ok) printf("    %s NC=%d R=%d row %d col %d: got %.9g ref %.9g (err/tol %.3g)\n", kname(T), NC, R, i, j,
                               acc[i][j], ref, e);
                ok = false;
            }
        }
    }
    return ok;
}

template <int T>
bool check_type(const int trials) {
    std::mt19937 rng(1234 + T);
    bool ok = true;
    double worst = 0.0;
    int n = 0;
    for (int tr = 0; tr < trials; ++tr) {
        // ordinary activations, and large ones whose 32-value sums exceed FP16's range
        const float scale = tr % 4 == 3 ? 3000.0f : 1.5f;
        ok &= check_case<T, 1, 1>(rng, scale, worst);
        ok &= check_case<T, 3, 1>(rng, scale, worst);
        ok &= check_case<T, 8, 1>(rng, scale, worst);
        ok &= check_case<T, 1, 2>(rng, scale, worst);
        ok &= check_case<T, 3, 2>(rng, scale, worst);
        ok &= check_case<T, 8, 2>(rng, scale, worst);
        n += 6;
    }
    printf("%-7s %4d cases  worst err/tol %.4f  %s\n", kname(T), n, worst, ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 40;
    bool ok = true;
    ok &= check_type<KQ8_0>(trials);
    ok &= check_type<KQ2_0>(trials);
    ok &= check_type<KQ3_K>(trials);
    ok &= check_type<KQ4_K>(trials);
    ok &= check_type<KQ5_K>(trials);
    ok &= check_type<KQ6_K>(trials);
    ok &= check_type<KIQ4_XS>(trials);
    printf("decoding: %s\n", ok ? "every type matches ggml's reference" : "FAILURES");
    return ok ? 0 : 1;
}
