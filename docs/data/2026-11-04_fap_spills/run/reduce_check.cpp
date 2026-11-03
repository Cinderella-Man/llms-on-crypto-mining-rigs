// reduce_check: the transpose-reduce of Miner Llama's Pascal decode attention kernel (fattn-pascal.cu,
// fap_transpose_reduce), in the form v2.0 shipped (a conditional on the two array elements) and the form without local
// memory (both elements read first, then the values selected), on an emulated warp: 32 lanes, __shfl_xor_sync as an
// exchange between the lanes' arrays after every lane has chosen what it sends. Checks, on random inputs:
//   1. both forms give every lane the same 32-bit result (the change must not move a single bit);
//   2. lane l ends with the sum over the 32 lanes of value l (the mapping the kernel's scores rely on: s[j4] of lane l
//      is key 4*j4 + l/8, query head l%8), against an FP64 sum;
//   3. a mutated form (keep and send swapped in the last stage) fails check 2, so the check can fail.
// Host floats add as the GPU's FADD does (round to nearest even); the kernel flushes subnormals (-use_fast_math), which
// these inputs do not produce. Build and run: g++ -O2 -std=c++17 -o reduce_check reduce_check.cpp && ./reduce_check
// Exit status 0 if every check passes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

constexpr int W = 32;

enum form { V20, VALUES, MUTATED };

// one warp's transpose-reduce: v[lane][32] in, r[lane] = v[lane][0] out
static void transpose_reduce(float v[W][32], const form f, float r[W]) {
    for (int o = 16, n = 32; o > 0; o >>= 1, n >>= 1) {
        for (int i = 0; i < n/2; ++i) {
            float send[W], keep[W];
            for (int lane = 0; lane < W; ++lane) {
                const bool upper = lane & o;
                if (f == V20) {
                    send[lane] = upper ? v[lane][i] : v[lane][i + n/2];
                    keep[lane] = upper ? v[lane][i + n/2] : v[lane][i];
                } else {
                    const float lo = v[lane][i];
                    const float hi = v[lane][i + n/2];
                    send[lane] = upper ? lo : hi;
                    keep[lane] = upper ? hi : lo;
                    if (f == MUTATED && o == 1) {
                        const float t = send[lane];
                        send[lane] = keep[lane];
                        keep[lane] = t;
                    }
                }
            }
            for (int lane = 0; lane < W; ++lane) {
                v[lane][i] = keep[lane] + send[lane ^ o]; // __shfl_xor_sync(mask, send, o)
            }
        }
    }
    for (int lane = 0; lane < W; ++lane) {
        r[lane] = v[lane][0];
    }
}

static uint32_t bits(const float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

int main() {
    std::mt19937 gen(0x5A17);
    const int n_trials = 20000;
    long bit_diffs = 0, map_fail[3] = {0, 0, 0};
    double worst_rel = 0.0;
    for (int t = 0; t < n_trials; ++t) {
        // partial dot products as the kernel makes them: 4 dimensions of a pre-scaled query times a key, per lane;
        // the scale varies so that some trials sum values of very different sizes
        const float scale = std::ldexp(1.0f, (int) (gen() % 24) - 12);
        std::uniform_real_distribution<float> d(-scale, scale);
        float in[W][32];
        for (int lane = 0; lane < W; ++lane) {
            for (int k = 0; k < 32; ++k) {
                in[lane][k] = d(gen);
            }
        }
        float r[3][W];
        const form forms[3] = { V20, VALUES, MUTATED };
        for (int f = 0; f < 3; ++f) {
            float v[W][32];
            memcpy(v, in, sizeof(v));
            transpose_reduce(v, forms[f], r[f]);
        }
        for (int lane = 0; lane < W; ++lane) {
            bit_diffs += bits(r[V20][lane]) != bits(r[VALUES][lane]);
            double ref = 0.0, mag = 0.0;
            for (int l = 0; l < W; ++l) {
                ref += in[l][lane];
                mag += std::fabs(in[l][lane]);
            }
            // 5 rounded additions per result: |error| <= 5 * 2^-24 * sum |x| (with margin)
            const double tol = 8.0*std::ldexp(1.0, -24)*mag;
            for (int f = 0; f < 3; ++f) {
                const double err = std::fabs((double) r[f][lane] - ref);
                if (err > tol) {
                    map_fail[f]++;
                }
                if (f != MUTATED && mag > 0.0) {
                    worst_rel = std::fmax(worst_rel, err/tol);
                }
            }
        }
    }
    const long n = (long) n_trials*W;
    printf("trials %d x %d lanes\n", n_trials, W);
    printf("1. v2.0 form against the value-select form: %ld of %ld results differ in any bit\n", bit_diffs, n);
    printf("2. lane l = sum over lanes of value l (FP64 reference, 5-add rounding bound): v2.0 %ld fail, "
           "value-select %ld fail (worst error %.3f of the bound)\n", map_fail[V20], map_fail[VALUES], worst_rel);
    printf("3. mutated (last stage keep/send swapped): %ld of %ld fail check 2\n", map_fail[MUTATED], n);
    const bool ok = bit_diffs == 0 && map_fail[V20] == 0 && map_fail[VALUES] == 0 && map_fail[MUTATED] > 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
