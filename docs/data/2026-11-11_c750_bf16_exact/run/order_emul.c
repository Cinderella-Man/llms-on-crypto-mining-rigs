// order_emul.c: C750's CPU emulation of the sums of Miner Llama's Pascal BF16 kernels (ggml/src/ggml-cuda/
// mmvf-pascal-impl.cuh) in three orders, to count the outputs whose bits differ. It extends C730's emulation
// (docs/data/2026-11-11_c730_bf16_hc/run/order_emul.c) by the orders of GGML_CUDA_MMVF_PASCAL_EXACT's two kernels.
//
//   g128, g32  mul_mat_vec_bf16_pascal<G, R, T> at G = 128 (one token) and G = 32 (Q611's rule at 3-4 tokens on 5 SMs):
//              lane sl takes the 8-weight chunks sl, sl + G, ... and adds each weight times its activation with one FMA,
//              in column order; the lanes of a warp are added with the butterfly (xor 16, 8, 4, 2, 1), and for G = 128
//              warp 0's sum then adds warps 1, 2 and 3 in that order.
//   x4p        mul_mat_vec_bf16_pascal_x4p (EXACT=1), step by step as its code runs: 32 lanes, four passes; in pass v lane j
//              adds the chunks j + 32 v, j + 32 v + 128, ... into one accumulator, the butterfly gives the pass's sum S_v,
//              and the running sum is S0, then += S1, += S2, += S3.
//   x4i        mul_mat_vec_bf16_pascal_x4 (EXACT=2), step by step as its code runs: 32 lanes, four accumulators each; lane
//              j's loop takes chunks c + 32 v into acc[v] (v = 0..3) while c + 96 < nchunks, c = j, j + 128, ..., then the
//              at most three chunks left into acc[0], acc[1], acc[2]; each acc[v] through the butterfly; the sum
//              ((acc[0] + acc[1]) + acc[2]) + acc[3].
//   x4pfault   x4p without S3 (GGML_CUDA_MMVF_PASCAL_EXACT_FAULT=1)
//
// The weights are BF16 (a normal sample of standard deviation 0.02, rounded to BF16), the activations FP32 (standard
// normal), a fixed seed; denormals are flushed as the kernels' FTZ instructions do. Prints per comparison the outputs
// compared, how many differ, the median difference of those in units of the last place, and the largest difference over
// the sum of the absolute terms. The sweeps compare x4p and x4i with g128 at every row width of 1 to 2,048 chunks (K = 8 to
// 16,384), 4 rows and 2 activation columns each: every tail of every lane, and lanes with no chunk.
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

// one 8-weight chunk into acc, as mmvfp_dot8
static float dot8(const float * w, const float * a, float acc) {
    for (int j = 0; j < 8; ++j) {
        acc = fmaf(w[j], a[j], acc);
    }
    return acc;
}

// the butterfly of 32 lanes (xor 16, 8, 4, 2, 1); every lane ends with the same value, lane 0's is returned
static float butterfly32(const float * lane) {
    float v[32];
    memcpy(v, lane, sizeof(v));
    for (int off = 16; off > 0; off >>= 1) {
        float n[32];
        for (int j = 0; j < 32; ++j) {
            n[j] = v[j] + v[j ^ off];
        }
        memcpy(v, n, sizeof(v));
    }
    return v[0];
}

// mul_mat_vec_bf16_pascal<G, ...>'s sum of one row, G = 32 or 128
static float row_sum_g(const float * w, const float * a, int nchunks, int G) {
    float lane[128];
    for (int sl = 0; sl < G; ++sl) {
        float acc = 0.0f;
        for (int c = sl; c < nchunks; c += G) {
            acc = dot8(w + 8 * c, a + 8 * c, acc);
        }
        lane[sl] = acc;
    }
    float s = butterfly32(lane);
    for (int k = 1; k < G / 32; ++k) {
        s += butterfly32(lane + 32 * k);
    }
    return s;
}

// mul_mat_vec_bf16_pascal_x4p's sum of one row, as its code runs
static float row_sum_x4p(const float * w, const float * a, int nchunks, int fault) {
    enum { warp = 32, V = 4 };
    float s = 0.0f;
    for (int v = 0; v < V; ++v) {
        float acc[warp];
        for (int j = 0; j < warp; ++j) {
            acc[j] = 0.0f;
            for (int c = v * warp + j; c < nchunks; c += V * warp) {
                acc[j] = dot8(w + 8 * c, a + 8 * c, acc[j]);
            }
        }
        const float sv = butterfly32(acc);
        if (v == 0) {
            s = sv;
        } else if (!fault || v < V - 1) {
            s += sv;
        }
    }
    return s;
}

// mul_mat_vec_bf16_pascal_x4's sum of one row, as its code runs
static float row_sum_x4(const float * w, const float * a, int nchunks, int fault) {
    enum { warp = 32, V = 4 };
    float acc[V][warp];
    for (int j = 0; j < warp; ++j) {
        for (int v = 0; v < V; ++v) {
            acc[v][j] = 0.0f;
        }
        int c = j;
        for (; c + (V - 1) * warp < nchunks; c += V * warp) {
            for (int v = 0; v < V; ++v) {
                acc[v][j] = dot8(w + 8 * (c + v * warp), a + 8 * (c + v * warp), acc[v][j]);
            }
        }
        for (int v = 0; v < V - 1; ++v) {
            if (c + v * warp < nchunks) {
                acc[v][j] = dot8(w + 8 * (c + v * warp), a + 8 * (c + v * warp), acc[v][j]);
            }
        }
    }
    float s = butterfly32(acc[0]);
    s += butterfly32(acc[1]);
    s += butterfly32(acc[2]);
    if (!fault) {
        s += butterfly32(acc[3]);
    }
    return s;
}

enum order { G128, G32, X4P, X4I, X4PFAULT };
static const char * order_name[] = { "g128", "g32", "x4p", "x4i", "x4pfault" };

static float row_sum(enum order o, const float * w, const float * a, int nchunks) {
    switch (o) {
        case G128:     return row_sum_g(w, a, nchunks, 128);
        case G32:      return row_sum_g(w, a, nchunks, 32);
        case X4P:      return row_sum_x4p(w, a, nchunks, 0);
        case X4I:      return row_sum_x4(w, a, nchunks, 0);
        case X4PFAULT: return row_sum_x4p(w, a, nchunks, 1);
    }
    return 0.0f;
}

static int cmp_i64(const void * x, const void * y) {
    const int64_t a = *(const int64_t *) x, b = *(const int64_t *) y;
    return (a > b) - (a < b);
}

struct tally { long n, diff; int64_t * ulps; double maxrel; };

static void compare_rows(struct tally * t, int K, int N, int ncols, enum order o1, enum order o2) {
    float * w = malloc(sizeof(float) * (size_t) K * N);
    float * a = malloc(sizeof(float) * K);
    for (size_t i = 0; i < (size_t) K * N; ++i) {
        w[i] = bf16_round(0.02f * nrand());
    }
    for (int c = 0; c < ncols; ++c) {
        for (int k = 0; k < K; ++k) {
            a[k] = nrand();
        }
        for (int r = 0; r < N; ++r) {
            const float * wr = w + (size_t) r * K;
            const float s1 = row_sum(o1, wr, a, K / 8);
            const float s2 = row_sum(o2, wr, a, K / 8);
            double abs_sum = 0.0;
            for (int k = 0; k < K; ++k) {
                abs_sum += fabs((double) wr[k] * a[k]);
            }
            int32_t u1, u2;
            memcpy(&u1, &s1, 4);
            memcpy(&u2, &s2, 4);
            ++t->n;
            if (u1 != u2) {
                // ULP distance between two finite floats of any sign
                const int64_t q1 = u1 < 0 ? (int64_t) INT32_MIN - u1 : u1;
                const int64_t q2 = u2 < 0 ? (int64_t) INT32_MIN - u2 : u2;
                t->ulps[t->diff++] = q1 > q2 ? q1 - q2 : q2 - q1;
                const double rel = fabs((double) s1 - (double) s2) / abs_sum;
                if (rel > t->maxrel) t->maxrel = rel;
            }
        }
    }
    free(w);
    free(a);
}

static void print_tally(const char * name, const char * K, const char * N, int ncols, enum order o1, enum order o2,
                        struct tally * t) {
    qsort(t->ulps, t->diff, sizeof(int64_t), cmp_i64);
    printf("%s\t%s\t%s\t%d\t%s\t%s\t%ld\t%ld\t%lld\t%.2e\n", name, K, N, ncols, order_name[o1], order_name[o2], t->n,
           t->diff, (long long) (t->diff ? t->ulps[t->diff / 2] : 0), t->maxrel);
}

static void compare(const char * name, int K, int N, int ncols, enum order o1, enum order o2) {
    struct tally t = { 0, 0, malloc(sizeof(int64_t) * (size_t) N * ncols), 0.0 };
    compare_rows(&t, K, N, ncols, o1, o2);
    char ks[16], ns[16];
    snprintf(ks, sizeof(ks), "%d", K);
    snprintf(ns, sizeof(ns), "%d", N);
    print_tally(name, ks, ns, ncols, o1, o2, &t);
    free(t.ulps);
}

int main(void) {
    _mm_setcsr(_mm_getcsr() | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON);
    printf("shape\tK\tN\tcolumns\torder A\torder B\toutputs\tdiffer\tmedian ulp\tmax diff / sum |terms|\n");
    // hc_attn_down, hc_ffn_down, hc_head_down: G = 128 at 1-2 tokens; at 3-4 tokens G = 32 by default, x4p or x4i with
    // the switch
    compare("hc_down", 10240, 320, 8, G128, X4P);
    compare("hc_down", 10240, 320, 8, G128, X4I);
    compare("hc_down", 10240, 320, 8, G128, G32);
    compare("hc_down", 10240, 320, 8, G128, X4PFAULT);
    // test-backend-ops' three widths whose 32-lane chunks end in each of x4i's tails (513, 549, 592 chunks)
    const struct { const char * name; int K, N; } tails[] = { { "tail1", 4104, 320 }, { "tail21", 4392, 320 },
                                                               { "tail32", 4736, 333 } };
    for (int i = 0; i < 3; ++i) {
        compare(tails[i].name, tails[i].K, tails[i].N, 4, G128, X4P);
        compare(tails[i].name, tails[i].K, tails[i].N, 4, G128, X4I);
    }
    // every row width of 1 to 2,048 chunks
    const int nmax = 2048, rows = 4, cols = 2;
    const enum order sweep[] = { X4P, X4I };
    for (int s = 0; s < 2; ++s) {
        struct tally t = { 0, 0, malloc(sizeof(int64_t) * (size_t) nmax * rows * cols), 0.0 };
        for (int nc = 1; nc <= nmax; ++nc) {
            compare_rows(&t, 8 * nc, rows, cols, G128, sweep[s]);
        }
        print_tally("sweep", "8-16384", "4", cols, G128, sweep[s], &t);
        free(t.ulps);
    }
    return 0;
}
