// order_emul.c: C767's CPU emulation of a one-token Q5_K product in Miner Llama's two kernels, to count the outputs
// whose bits differ between them.
//
// With GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1 a plain one-token Q5_K product runs mul_mat_vec_q5_K_pascal_mt<1, R, PF>
// (ggml/src/ggml-cuda/mmvq-pascal-q5k.cu: q5kmt_superblock) instead of the one-token Pascal kernel
// (mul_mat_vec_q_pascal<Q5_K, 1, 1, staged>, mmvq-pascal.cu, its super-block in mmvqp_dot_q5_K of
// mmvq-pascal-impl.cuh). Both give a row the same lanes (L, the largest power of two up to 32 that divides the row's
// super-blocks; lane sl adds super-blocks sl, sl + L, ... in order) and add the lanes with the butterfly xor 16, ...,
// 1 (offsets below L only). Per super-block, sub-block s (32 weights, codes 0..31, 6-bit scale sc and min m), with the
// integer sum isum of code times int8 activation and S = d8 * (float) sum(int8):
//   one token:  sumd = fmaf(d8, (float) (isum * sc), sumd);   summ = fmaf((float) m, S, summ)
//   1R kernel:  x = fmaf(as_float(0x4B400000 + isum), scf, -12582912 * scf);   sumd = fmaf(d8, x, sumd);
//               summ = fmaf(mf, S, summ),  with scf = as_float(0x4B4000sc) - 12582912 and mf likewise from m
// and per super-block, in both (the SASS: t = FFMA(sumd, d, acc); acc = FFMA(summ, -dmin, t)):
//   acc = fmaf(-dmin, summ, fmaf(d, sumd, acc))
// The emulation forms the integer sums once (exact in any order) and runs both FP32 arithmetics, plus two controls
// that must differ: the one-token arithmetic at one lane per row (another order), and with the super-block's last
// expression not contracted (d * sumd rounded before the add). It also counts the sub-blocks where the 1R kernel's x,
// scf or mf is not the one-token kernel's (float) (isum * sc), (float) sc or (float) m.
//
// Weights: random codes 0..31, scales and mins uniform 0..63, d and dmin |N(0, 0.002)| rounded to FP16 (the 1R
// kernel takes any bits the one-token kernel takes). Activations: standard normal, or uniform on [-1, 1] times
// 250,000 (test-backend-ops' large-activation cases: larger integer sums and S), quantized per 32 values as
// quantize_q8_1 does (d8 = amax / 127 stored as FP16, q = roundf(x / d8)). A fixed seed; denormals flushed as the
// kernels' FTZ instructions do. Prints one line per product and activation sample.
//
// Build and run (any x86-64 CPU with GCC 12 or later, for _Float16):
//   gcc -O2 -ffp-contract=off -o order_emul order_emul.c -lm && ./order_emul > order_emul.tsv
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double urand(void) { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static float nrand(void) { return (float) (sqrt(-2.0 * log(urand())) * cos(6.283185307179586 * urand())); }
static float h(float f) { return (float) (_Float16) f; } // FP32 to FP16 (round to nearest even) and back
static float as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t as_u32(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

typedef struct { float d, dmin; int sc[8], m[8]; int8_t c[256]; } sblk; // one super-block of a row
typedef struct { float d8; int8_t q[32]; int sum; } ablk;                // one 32-value block of Q8_1 activations

static void make_w(sblk * b) {
    b->d    = h(fabsf(0.002f * nrand()));
    b->dmin = h(fabsf(0.002f * nrand()));
    for (int s = 0; s < 8; ++s) { b->sc[s] = (int) (next() >> 58); b->m[s] = (int) (next() >> 58); }
    for (int i = 0; i < 256; ++i) b->c[i] = (int8_t) (next() >> 59);
}

static int big; // the activations: 0 standard normal, 1 uniform on [-1, 1] times 250,000
static void quant_a(ablk * a) {
    float x[32], amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        x[i] = big ? (float) ((2.0 * urand() - 1.0) * 250000.0) : nrand();
        if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
    }
    const float d = amax / 127.0f;
    a->d8 = h(d);
    a->sum = 0;
    for (int i = 0; i < 32; ++i) { a->q[i] = amax == 0.0f ? 0 : (int8_t) roundf(x[i] / d); a->sum += a->q[i]; }
}

static long n_sub, n_conv_differ;

// one super-block of a row times its 8 activation blocks, added to acc: kind 0 the one-token kernel, 1 the 1R kernel,
// 2 the one-token arithmetic with the last expression not contracted (a control)
static float superblock(int kind, const sblk * w, const ablk * a, const int * isum, float acc) {
    float sumd = 0.0f, summ = 0.0f;
    for (int s = 0; s < 8; ++s) {
        const float S = a[s].d8 * (float) a[s].sum;
        if (kind == 1) {
            const float scf  = as_float(0x4B400000u | (uint32_t) w->sc[s]) - 12582912.0f;
            const float mf   = as_float(0x4B400000u | (uint32_t) w->m[s]) - 12582912.0f;
            const float cneg = -12582912.0f * scf;
            const float x    = fmaf(as_float((uint32_t) (0x4B400000 + isum[s])), scf, cneg);
            ++n_sub;
            n_conv_differ += as_u32(x) != as_u32((float) (isum[s] * w->sc[s])) || scf != (float) w->sc[s] ||
                             mf != (float) w->m[s];
            sumd = fmaf(a[s].d8, x, sumd);
            summ = fmaf(mf, S, summ);
        } else {
            sumd = fmaf(a[s].d8, (float) (isum[s] * w->sc[s]), sumd);
            summ = fmaf((float) w->m[s], S, summ);
        }
    }
    if (kind == 2) {
        const float t = acc + w->d * sumd;
        return t - w->dmin * summ;
    }
    return fmaf(-w->dmin, summ, fmaf(w->d, sumd, acc));
}

// lanes L: lane sl adds super-blocks sl, sl + L, ... in order; then the butterfly over offsets below L
static float row_sum(int kind, int L, int nsb, const sblk * row, const ablk * act, int (*isum)[8]) {
    float lane[32] = { 0 };
    for (int sl = 0; sl < L; ++sl) {
        for (int b = sl; b < nsb; b += L) {
            lane[sl] = superblock(kind, &row[b], &act[8 * b], isum[b], lane[sl]);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        if (o >= L) continue;
        float nl[32];
        for (int sl = 0; sl < L; ++sl) nl[sl] = lane[sl] + lane[sl ^ o];
        memcpy(lane, nl, sizeof(float) * L);
    }
    return lane[0];
}

// nrows rows of K weights times one token
static void run(const char * name, int K, int nrows) {
    const int nsb = K / 256;
    int L = 1;
    while (L < 32 && nsb % (2 * L) == 0) L *= 2;
    ablk * act = malloc(sizeof(ablk) * (K / 32));
    for (int i = 0; i < K / 32; ++i) quant_a(&act[i]);
    sblk * row = malloc(sizeof(sblk) * nsb);
    int (*isum)[8] = malloc(sizeof(int[8]) * nsb);
    long differ = 0, ctrl_order = 0, ctrl_contract = 0, zero = 0;
    n_sub = n_conv_differ = 0;
    for (int r = 0; r < nrows; ++r) {
        for (int b = 0; b < nsb; ++b) {
            make_w(&row[b]);
            for (int s = 0; s < 8; ++s) {
                int v = 0;
                for (int i = 0; i < 32; ++i) v += row[b].c[32 * s + i] * act[8 * b + s].q[i];
                isum[b][s] = v;
            }
        }
        const float one = row_sum(0, L, nsb, row, act, isum);
        const float r1  = row_sum(1, L, nsb, row, act, isum);
        differ += as_u32(one) != as_u32(r1);
        zero += one == 0.0f;
        ctrl_order += as_u32(one) != as_u32(row_sum(0, L == 1 ? 2 : 1, nsb, row, act, isum));
        ctrl_contract += as_u32(one) != as_u32(row_sum(2, L, nsb, row, act, isum));
    }
    printf("%s\t%d\t%d\t%s\t%d\t%d\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n", name, K, nrows, big ? "uniform x 250000" : "normal",
           L, nrows, differ, n_sub, n_conv_differ, ctrl_order, ctrl_contract, zero);
    free(act); free(row); free(isum);
}

int main(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    printf("product\tK\trows\tactivations\tlanes_per_row\toutputs\tdiffer_1R_vs_one_token\tsub_blocks\t"
           "sub_blocks_x_or_scales_differ\tcontrol_other_lanes_differ\tcontrol_not_contracted_differ\toutputs_zero\n");
    for (big = 0; big < 2; ++big) {
        run("output (the head)", 2560, 248320);
        run("attn_output, ssm_out", 6144, 2560);
        run("attn_k, attn_v", 2560, 512);
        run("ffn_gate_shexp, ffn_up_shexp", 2560, 640);
        run("K = 256 (1 lane)", 256, 4096);
        run("K = 768 (1 lane)", 768, 4096);
        run("K = 1024 (4 lanes)", 1024, 4096);
        run("K = 2048 (8 lanes)", 2048, 4096);
        run("K = 4096 (16 lanes)", 4096, 4096);
        run("K = 8192 (32 lanes)", 8192, 4096);
        run("K = 12288 (16 lanes)", 12288, 1024);
    }
    return 0;
}
