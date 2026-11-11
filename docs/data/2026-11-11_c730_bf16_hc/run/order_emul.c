// order_emul.c: C730's CPU emulation of the sums of Miner Llama's Pascal BF16 kernel (mul_mat_vec_bf16_pascal in
// ggml/src/ggml-cuda/mmvf-pascal-impl.cuh) in two launch shapes, to count the outputs whose bits differ.
//
// One row's sum in a shape of G lanes per row group, as the kernel computes it: lane sl takes the 8-weight chunks sl,
// sl + G, ... and adds each weight times its activation to its accumulator with one FMA, in column order; the lanes of
// a warp are reduced with the butterfly (xor 16, 8, 4, 2, 1 for G >= 32; 4, 2, 1 for G = 8), and for G = 128 warp 0's
// sum then adds warps 1, 2 and 3 in that order. Each output's order depends on G only: neither R (rows per group) nor
// T (tokens) changes it, so two shapes of the same G give the same bits.
//
// The weights are BF16 (a normal sample of standard deviation 0.02, rounded to BF16), the activations FP32 (standard
// normal), a fixed seed; denormals are flushed as the kernel's FTZ instructions do. Prints, per pair of shapes, the
// outputs compared, how many differ, the median difference of those that differ in units of the last place, and the
// largest difference over the sum of the absolute terms (the tolerance unit of bench/mmvq_pascal's checks).
//
// Build and run (any x86-64 CPU): gcc -O2 -ffp-contract=off -o order_emul order_emul.c -lm && ./order_emul
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double urand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((rng >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static float nrand(void) {
    return (float) (sqrt(-2.0 * log(urand())) * cos(6.283185307179586 * urand()));
}
static float bf16_round(float f) { // FP32 to BF16 (round to nearest even), back as FP32
    uint32_t u; memcpy(&u, &f, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&f, &u, 4);
    return f;
}

// the sum of one row (K weights w, activations a) in the kernel's order for G lanes per row group
static float row_sum(const float * w, const float * a, int K, int G) {
    const int nchunks = K / 8;
    float lane[128];
    for (int sl = 0; sl < G; ++sl) {
        float acc = 0.0f;
        for (int c = sl; c < nchunks; c += G) {
            for (int j = 0; j < 8; ++j) {
                acc = fmaf(w[8 * c + j], a[8 * c + j], acc);
            }
        }
        lane[sl] = acc;
    }
    const int GW = G < 32 ? G : 32;
    float warp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int wid = 0; wid < G / GW; ++wid) {
        float v[32];
        memcpy(v, lane + wid * GW, GW * sizeof(float));
        for (int off = GW / 2; off > 0; off >>= 1) {
            float n[32];
            for (int j = 0; j < GW; ++j) {
                n[j] = v[j] + v[j ^ off];
            }
            memcpy(v, n, GW * sizeof(float));
        }
        warp[wid] = v[0];
    }
    float s = warp[0];
    for (int k = 1; k < G / GW; ++k) {
        s += warp[k];
    }
    return s;
}

static int cmp_i64(const void * x, const void * y) {
    const int64_t a = *(const int64_t *) x, b = *(const int64_t *) y;
    return (a > b) - (a < b);
}

static void compare(const char * name, int K, int N, int G1, int G2, int ncols) {
    float * w = malloc(sizeof(float) * K * N);
    float * a = malloc(sizeof(float) * K);
    for (int i = 0; i < K * N; ++i) {
        w[i] = bf16_round(0.02f * nrand());
    }
    long n = 0, diff = 0;
    int64_t * ulps = malloc(sizeof(int64_t) * (size_t) N * ncols);
    double maxrel = 0.0;
    for (int t = 0; t < ncols; ++t) {
        for (int k = 0; k < K; ++k) {
            a[k] = nrand();
        }
        for (int r = 0; r < N; ++r) {
            const float s1 = row_sum(w + (size_t) r * K, a, K, G1);
            const float s2 = row_sum(w + (size_t) r * K, a, K, G2);
            double abs_sum = 0.0;
            for (int k = 0; k < K; ++k) {
                abs_sum += fabs((double) w[(size_t) r * K + k] * a[k]);
            }
            int32_t u1, u2;
            memcpy(&u1, &s1, 4);
            memcpy(&u2, &s2, 4);
            ++n;
            if (u1 != u2) {
                ++diff;
                // ULP distance between two finite floats of any sign
                const int64_t o1 = u1 < 0 ? (int64_t) INT32_MIN - u1 : u1;
                const int64_t o2 = u2 < 0 ? (int64_t) INT32_MIN - u2 : u2;
                ulps[diff - 1] = o1 > o2 ? o1 - o2 : o2 - o1;
                const double rel = fabs((double) s1 - (double) s2) / abs_sum;
                if (rel > maxrel) maxrel = rel;
            }
        }
    }
    qsort(ulps, diff, sizeof(int64_t), cmp_i64);
    printf("%s\t%d\t%d\tG=%d\tG=%d\t%ld\t%ld\t%lld\t%.2e\n", name, K, N, G1, G2, n, diff,
           (long long) (diff ? ulps[diff / 2] : 0), maxrel);
    free(ulps);
    free(w);
    free(a);
}

int main(void) {
    _mm_setcsr(_mm_getcsr() | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON);
    printf("shape\tK\tN\tshape A\tshape B\toutputs\tdiffer\tmedian ulp\tmax diff / sum |terms|\n");
    // hc_attn_down, hc_ffn_down (and the output's hc_head_down): G = 128 at 1-2 tokens, G = 32 at 3-4 on 5 SMs
    compare("hc_down", 10240, 320, 128, 32, 8);
    // hc_attn_up, hc_ffn_up: G = 8 at every token count (the control: 0 must differ)
    compare("hc_up", 320, 10240, 8, 8, 2);
    return 0;
}
