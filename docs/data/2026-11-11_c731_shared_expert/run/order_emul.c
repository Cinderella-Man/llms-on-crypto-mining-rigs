// order_emul.c: C731's CPU emulation of the shared expert's quantized products in Miner Llama's two kernels, to count
// the outputs whose bits differ between one token and 2-4 tokens.
//
// Under the Flash-Next preset a plain product of one token runs the one-token Pascal kernel (mul_mat_vec_q_pascal,
// ggml/src/ggml-cuda/mmvq-pascal.cu, its per-block arithmetic in mmvqp_dot of mmvq-pascal-impl.cuh) and the same
// product of 2-8 tokens the k-column kernel (mul_mat_kcol, mmvq-pascal-kcol-impl.cuh: unit_q2_0, unit_q4_0, unit_q5_0).
// For Q2_0, Q4_0 and Q5_0 both kernels give a row the same lanes (L, the largest power of two up to 32 that divides
// the row's chunks of 2 blocks) and the same order: lane sl adds its chunks sl, sl + L, ... in order, block by block,
// and the lanes are added with the butterfly xor 16, ..., 1 (only offsets below L). What differs is the FP32 arithmetic
// of one block, copied here from the two sources:
//   Q2_0 (64 weights, codes -1..2, activation blocks a0 and a1), both kernels:
//       acc = fmaf(d, fmaf((float) i1, d8_1, (float) i0 * d8_0), acc)
//   Q4_0 (32 weights, codes 0..15, offset -8):
//       one token:  acc = fmaf(d, fmaf((float) (16 * sum(code * q)), d8 * 0.0625f, -8.0f * (d8 * (float) sum(q))), acc)
//       k-column:   acc = fmaf((float) (sum(code * q) - 8 * sum(q)) * d, d8, acc)
//   Q5_0 (32 weights, codes 0..31, offset -16):
//       one token:  acc = fmaf(d, fmaf((float) (16 * sum(code * q)), d8 * 0.0625f, -16.0f * (d8 * (float) sum(q))), acc)
//       k-column:   acc = fmaf((float) (sum(code * q) - 16 * sum(q)) * d, d8, acc)
// The integer sums are exact in any order, so the emulation forms them directly. For Q4_0 and Q5_0 both forms are
// d * d8 * I + acc rounded once wherever d * I and d8 * I (I = sum((code - offset) * q)) are exact in FP32, which holds
// while |I| < 2^13 (an FP16 scale has 11 significant bits); only blocks past that can give other bits.
//
// Weights: a normal sample of standard deviation 0.02 quantized as ggml's reference quantizers do (Q4_0: d = max / -8,
// code = min(15, (int8) (x / d + 8.5)); Q5_0: d = max / -16, code = min(31, (int8) (x / d + 16.5)); Q2_0: d =
// amax / 2 rounded to FP16, codes uniform), d stored as FP16. Activations: standard normal, then uniform on [-1, 1]
// (larger integer sums), quantized per 32 values as quantize_q8_1 does (d8 = amax / 127 stored as FP16, q =
// roundf(x / d)). A fixed seed; denormals flushed as the kernels' FTZ instructions do. Prints per product and sample:
// the outputs compared (rows x tokens), how many differ, the median difference of those that differ in units of the
// last place, the largest difference over the sum of the absolute terms, and (Q4_0, Q5_0) the blocks and how many of
// them have a rounded product.
//
// Build and run (any x86-64 CPU with GCC 12 or later, for _Float16):
//   gcc -O2 -ffp-contract=off -o order_emul order_emul.c -lm && ./order_emul
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
static float h(float f) { return (float) (_Float16) f; } // FP32 to FP16 (round to nearest even) and back

enum { Q2_0 = 0, Q4_0 = 1, Q5_0 = 2 };
static const char * tname[] = { "q2_0", "q4_0", "q5_0" };

// one block of weights: its scale and its codes (Q2_0: 64 codes -1..2; Q4_0, Q5_0: 32 codes 0..15 / 0..31)
typedef struct { float d; int c[64]; } wblk;
// one 32-value block of Q8_1 activations
typedef struct { float d8; int q[32]; int sum; } ablk;

static void quant_w(int t, wblk * b) {
    if (t == Q2_0) {
        float amax = 0.0f;
        for (int i = 0; i < 64; ++i) { float x = 0.02f * nrand(); if (fabsf(x) > amax) amax = fabsf(x); }
        b->d = h(amax / 2.0f);
        for (int i = 0; i < 64; ++i) b->c[i] = (int) (urand() * 4.0) - 1;
        return;
    }
    const float off = t == Q4_0 ? 8.0f : 16.0f;
    const int   top = t == Q4_0 ? 15 : 31;
    float x[32], amax = 0.0f, max = 0.0f;
    for (int i = 0; i < 32; ++i) { x[i] = 0.02f * nrand(); if (fabsf(x[i]) > amax) { amax = fabsf(x[i]); max = x[i]; } }
    const float d  = max / -off;
    const float id = d ? 1.0f / d : 0.0f;
    b->d = h(d);
    for (int i = 0; i < 32; ++i) {
        int q = (int8_t) (x[i] * id + off + 0.5f);
        b->c[i] = q < top ? q : top;
    }
}

static int adist; // the activations: 0 standard normal, 1 uniform on [-1, 1] (larger integer sums: more rounded products)
static void quant_a(ablk * a) {
    float x[32], amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        x[i] = adist ? (float) (2.0 * urand() - 1.0) : nrand();
        if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
    }
    const float d = amax / 127.0f;
    a->d8 = h(d);
    a->sum = 0;
    for (int i = 0; i < 32; ++i) { a->q[i] = amax == 0.0f ? 0 : (int) roundf(x[i] / d); a->sum += a->q[i]; }
}

// one weight block times its activations, added to acc, in the one-token (kc = 0) or the k-column (kc = 1) arithmetic;
// *abs gets the absolute value of the block's term
static long n_blk, n_inexact; // Q4_0, Q5_0: blocks, and blocks where d * I or d8 * I (I the integer sum) rounds in FP32
static float blk(int t, int kc, const wblk * w, const ablk * a, float acc, double * abs) {
    if (t == Q2_0) {
        int i0 = 0, i1 = 0;
        for (int k = 0; k < 32; ++k) { i0 += w->c[k] * a[0].q[k]; i1 += w->c[32 + k] * a[1].q[k]; }
        *abs += fabs((double) w->d * ((double) i0 * a[0].d8 + (double) i1 * a[1].d8));
        return fmaf(w->d, fmaf((float) i1, a[1].d8, (float) i0 * a[0].d8), acc);
    }
    const int off = t == Q4_0 ? 8 : 16;
    int cq = 0;
    for (int k = 0; k < 32; ++k) cq += w->c[k] * a->q[k];
    const int I = cq - off * a->sum;
    *abs += fabs((double) w->d * a->d8 * (double) I);
    if (kc == 0) {
        ++n_blk;
        n_inexact += (double) (float) ((float) I * w->d) != (double) I * w->d ||
                     (double) (float) ((float) I * a->d8) != (double) I * a->d8;
    }
    if (kc == 0) {
        const float S = a->d8 * (float) a->sum;
        return fmaf(w->d, fmaf((float) (16 * cq), a->d8 * 0.0625f, -(float) off * S), acc);
    }
    return fmaf((float) I * w->d, a->d8, acc);
}

static uint32_t ord(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? 0x80000000u - (u & 0x7FFFFFFFu) : u + 0x80000000u; }
static int cmp_u(const void * x, const void * y) { uint32_t a = *(const uint32_t *) x, b = *(const uint32_t *) y; return a < b ? -1 : a > b; }

// rows of K weights times T tokens in the two arithmetics; L lanes per row
static void run(int t, int K, int nrows, int T) {
    const int bw  = t == Q2_0 ? 64 : 32;   // weights per block
    const int nb  = K / bw;                // blocks per row
    const int na  = K / 32;                // activation blocks per token
    const int ncu = nb / 2;                // chunks (2 blocks) per row
    int L = 1;
    while (L < 32 && ncu % (2 * L) == 0) L *= 2;
    ablk * act = malloc(sizeof(ablk) * na * T);
    for (int i = 0; i < na * T; ++i) quant_a(&act[i]);
    wblk * row = malloc(sizeof(wblk) * nb);
    uint32_t * ulp = malloc(sizeof(uint32_t) * nrows * T);
    long ndiff = 0;
    double worst = 0.0;
    for (int r = 0; r < nrows; ++r) {
        for (int b = 0; b < nb; ++b) quant_w(t, &row[b]);
        for (int j = 0; j < T; ++j) {
            float  s[2];
            double abs = 0.0;
            for (int kc = 0; kc < 2; ++kc) {
                float lane[32] = { 0 };
                double ab = 0.0;
                for (int sl = 0; sl < L; ++sl) {
                    for (int c = sl; c < ncu; c += L) {
                        for (int e = 0; e < 2; ++e) {
                            const int b = 2 * c + e;
                            lane[sl] = blk(t, kc, &row[b], &act[j * na + b * (bw / 32)], lane[sl], &ab);
                        }
                    }
                }
                for (int o = 16; o > 0; o >>= 1) { // the butterfly: each lane adds lane ^ o, offsets below L only
                    if (o >= L) continue;
                    float nl[32];
                    for (int sl = 0; sl < L; ++sl) nl[sl] = lane[sl] + lane[sl ^ o];
                    memcpy(lane, nl, sizeof(float) * L);
                }
                s[kc] = lane[0];
                abs = ab;
            }
            if (s[0] != s[1]) {
                uint32_t a = ord(s[0]), b = ord(s[1]);
                ulp[ndiff++] = a > b ? a - b : b - a;
                const double rel = fabs((double) s[0] - (double) s[1]) / abs;
                if (rel > worst) worst = rel;
            }
        }
    }
    uint32_t med = 0;
    if (ndiff) { qsort(ulp, ndiff, sizeof(uint32_t), cmp_u); med = ulp[ndiff / 2]; }
    printf("%s\t%d\t%d\t%d\t%s\t%d\t%ld\t%ld\t%u\t%.3g\t%ld\t%ld\n", tname[t], K, nrows, T, adist ? "uniform" : "normal", L,
           (long) nrows * T, ndiff, med, worst, n_blk, n_inexact);
    n_blk = n_inexact = 0;
    free(act); free(row); free(ulp);
}

int main(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    printf("type\tK\trows\ttokens\tactivations\tlanes_per_row\toutputs\tdiffer\tmedian_ulp_of_those\t"
           "max_diff_over_sum_abs\tblocks\tblocks_with_a_rounded_product\n");
    for (adist = 0; adist < 2; ++adist) {
        run(Q2_0, 2560, 640, 4);  // gate or up alone (the control: the same arithmetic in both kernels)
        run(Q2_0, 640, 2560, 4);  // down
        run(Q4_0, 640, 2560, 4);  // down
        run(Q5_0, 640, 2560, 4);  // down
    }
    return 0;
}
