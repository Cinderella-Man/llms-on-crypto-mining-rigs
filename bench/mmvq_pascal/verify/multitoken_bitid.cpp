// Bit-identity check of the fork's multi-token Pascal kernels (mmvq-pascal.cu, branch pascal-p106-v2-multicol; Q6_K from
// pascal-p106-v2-q6k): each product of n = 2..4 tokens must equal, bit for bit, the n single-token products of the same
// columns, on CUDA0. Covers MUL_MAT at Flint's dense shapes, MUL_MAT_ID (6 of 32 experts, tokens that share experts) and
// the fused MUL_MAT_ID gate/up + SWIGLU, for Q8_0, Q5_0, Q4_1, Q4_0, Q5_K, IQ4_XS and Q6_K. Run with
// GGML_CUDA_MMVQ_PASCAL=2 on a newer GPU (add CUDA_FORCE_PTX_JIT=1 for the compute 6.1 code path) or as is on a P106.
// With GGML_CUDA_MMVQ_PASCAL=0 the stock kernels run, whose multi-token and single-token kernels sum in different
// orders: most comparisons then differ. Build: verify/build_multitoken_bitid.sh. Exit status 1 if any comparison
// differs.
// `multitoken_bitid ornith` checks the types and shapes of Ornith-1.5-122B's files instead (fork branch
// pascal-p106-v3-mtpk, the kernels that decode each weight once for all tokens): IQ2_S, IQ3_XXS, N2P, N3P, Q8_0 and
// Q5_0; dense products at K = 3072, 1024 and 8192, experts of width 1024 at hidden 3072 (8 of 16 used), with three
// routings: neighbours sharing 2 of 8 experts, every token on the same 8 experts in other slots, and 4 shared plus 4 of
// their own. The weights are rows drawn from a pool of 256 quantized rows per shape, so the slow IQ quantizers run on
// few rows while every expert's rows differ.
// `multitoken_bitid q5k` checks Q5_K's dense products alone (C690: the kernel of mmvq-pascal-q5k.cu with
// GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 or 2), at Qwen3.8-Flash-Next's output layer (2560 -> 248,320) and at row widths that
// give 1 to 32 lanes per row, weights drawn from a row pool as for ornith; with the switch at 0 the k-column kernel takes
// these products and its sums differ from the one-token kernel's.
// `multitoken_bitid dmt` checks IQ4_XS's and Q4_K's dense products the same way (C722: the kernel of
// mmvq-pascal-dmt.cu with GGML_CUDA_MMVQ_PASCAL_DMT=1 or 2), at Qwen3.8-Flash-Next's attention and recurrent projections
// (2560 -> 10240, 6144 and 12288; 6144 -> 2560) and at row widths that give 1 to 32 lanes per row.
// `multitoken_bitid dmtkt` checks the same products with their K tiles (C728: GGML_CUDA_MMVQ_PASCAL_DMT=1 with
// GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 or 2): dmt's shapes, K = 12288, and K = 6144 at 333 and 4097 rows (a partial last round
// and row block). GGML_CUDA_MMVQ_PASCAL_DMT_KT_FAULT=1 (each iteration takes the next one's activations) must give
// differences at K = 6144.
// `multitoken_bitid bf16` checks BF16's dense products the same way (C750: the Pascal BF16 kernel of mmvf-pascal.cu with
// GGML_CUDA_MMVF_PASCAL_EXACT=1 or 2), at Qwen3.8-Flash-Next's BF16 shapes (the hyper-connections 10240 -> 320 and
// 320 -> 10240, 10240 -> 4, the indexer's 2560 -> 512, the recurrent layers' 2560 -> 48) and at row widths whose
// 3-4-token launch shape on 5 SMs has other lanes than one token's (K = 1000: 32 against 8; K = 4104, 4392 and 4736, the
// 32-lane kernel's tails: 32 against 128); with the switch at 0 those differ at 3 and 4 tokens. Run as is on a P106, or
// with GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVF_PASCAL_NSM=5 on a newer GPU (the P106-090's launch shapes).
// `multitoken_bitid f32` checks F32's dense products the same way, at 2 to 8 tokens (C754: llama.cpp's mul_mat_vec_f
// with GGML_CUDA_MMVF_F32_MT=1 at 4-8 tokens), at Qwen3.8-Flash-Next's F32 shapes (the shared expert's gate 2560 -> 1,
// the router 2560 -> 512) and at row widths that take mul_mat_vec_f's other block sizes (64, 100, 300) and a wide row
// (6144); with the switch at 0, cuBLAS takes 4-8 tokens and those differ. Run as is on a P106, or with
// GGML_CUDA_MMVF_F32_MT=2 on a newer GPU. From C807 on the switch takes at most GGML_CUDA_MMVF_F32_MT_ROWS rows (default
// 32), so this mode gives 0 differences with GGML_CUDA_MMVF_F32_MT_ROWS=0 (C754's rule); at the default its shapes of
// more than 32 rows (the router, 48, 33 and 129 rows) go to cuBLAS at 4-8 tokens and differ.
// `multitoken_bitid f32r` (C807) is the same check at shapes the switch takes at its default row limit: the shared
// expert's gate (2560 -> 1), the limit's edge at each block size (32 rows of 2560 and 64, 31 of 100, 17 of 300) and 7 of
// 6144; with the switch on, 0 differences.
// `multitoken_bitid q3k` checks Q3_K's dense products at 2 to 8 tokens (C726: the kernel of mmvq-pascal-q3kmt.cu with
// GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 or 2, whose one-token kernel is the k-column kernel), at Qwen3.8-Flash-Next's Q3_K
// projections (2560 -> 10240, 6144 and 12288; 6144 -> 2560; the shared expert's 2560 -> 640), at row widths that give 1
// to 32 lanes per row, and at widths whose activations pass the kernel's shared memory from 3 or 4 tokens on. With the
// switch at 0 the k-column kernel takes every product of 1 to 8 tokens, whose sums are already the one-token ones, so
// that run also gives 0 differences; GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT=1 (each token takes the next token's
// activations) must give differences.
// `multitoken_bitid glu` checks a gate and an up product of their own types and their SWIGLU, 2 to 8 tokens (C757: the
// one launch of mmvq-pascal-glu.cu with GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 or 2), against the same two products run apart
// and the GLU op run apart, at the same tokens ("GLU/apart") and token by token ("GLU/1-tok"): the 15 type pairs of
// Qwen3.8-Flash-Next's shared experts at 2560 -> 640 that the kernel takes (its two Q6_K pairs are left to the separate
// products), and every ordered pair of Q2_0, Q3_K, Q4_K, Q5_K and IQ4_XS at 77 x 1024 (a partial row block), 33 x 8192
// (32 lanes per row) and 200 x 512; 2 to 8 tokens for every pair. Run it with the switches whose separate kernels it
// repeats (GGML_CUDA_MMVQ_PASCAL_DMT=1, GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1, the k-column kernels on, and for 5 to 8 tokens
// of IQ4_XS, Q4_K and Q5_K GGML_CUDA_MMVQ_PASCAL_RO8=1 with GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 or 4, C793). A pair with an
// IQ4_XS, Q4_K or Q5_K product of 5 to 8 tokens that the read-once kernels do not take (the build says which, as in
// dmt8) runs on the k-column kernel, whose sums are not the one-token ones: its "GLU/1-tok" line is printed as
// "declined" and not counted, unless BITID_ALL=1; its "GLU/apart" line is counted. GGML_CUDA_MMVQ_PASCAL_GLU_MT_FAULT=1
// (each token takes the next token's activations) must give differences.
// `multitoken_bitid q5k1` checks Q5_K's one-token products (C767: the kernel of mmvq-pascal-q5k.cu with
// GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1) against the one-token kernel's, which no single process can run beside it (the
// switch is read once): with BITID_REF=<file> missing, the run writes every product's values to it; with the file
// there, it compares every product with it bit for bit. So: once with GGML_CUDA_MMVQ_PASCAL_Q5K_1R=0 (the one-token
// kernel writes the file), then with =1 and its settings. The weights and inputs come from the fixed seed, so both runs
// build the same ones. Shapes: q5k's, Qwen3.8-Flash-Next's other Q5_K products (6144 -> 2560, 2560 -> 512 and 640) and
// K = 256 (one super-block a row), each with inputs of standard deviation 1 and 250,000.
// GGML_CUDA_MMVQ_PASCAL_Q5K_1R_FAULT=1 (each row's last iteration left out) must give differences.
// `multitoken_bitid dmt8` and `multitoken_bitid q5k8` check dmt's and q5k's products at 5 to 8 tokens (C729: the same
// kernels with GGML_CUDA_MMVQ_PASCAL_RO8=1 or 2), at dmt's and q5k's shapes, q5k8 with Qwen3.8-Flash-Next's other Q5_K
// products (6144 -> 2560, 2560 -> 512 and 640). The kernels leave a product whose activations and one warp's staging pass
// 48 KB of shared memory to the k-column kernel, whose sums are not the one-token kernel's (6144-wide rows from 6 or 7
// tokens on, 8192-wide ones at every count): the build itself says which it takes (ggml_cuda_mmvq_pascal_dmt_takes,
// ggml_cuda_mmvq_pascal_q5k_mt_takes), and those it leaves are printed as "declined" and not counted, unless BITID_ALL=1
// (the control: with GGML_CUDA_MMVQ_PASCAL_RO8=0 and BITID_ALL=1 the k-column kernel's sums must differ).
// GGML_CUDA_MMVQ_PASCAL_RO8_FAULT=1 (every token stages the first token's activations) must give differences.
// `multitoken_bitid kt8` checks dmtkt's products at 5 to 8 tokens (C794: the K tiles at 5 to 8 tokens with
// GGML_CUDA_MMVQ_PASCAL_RO8=1 GGML_CUDA_MMVQ_PASCAL_RO8_KT=1 or 2), declined products counted as in dmt8: at K = 6144 the
// tiles take every product (6 warps a block) where whole staging holds at most 3 warps or none.
// GGML_CUDA_MMVQ_PASCAL_DMT_KT_FAULT=1 (each iteration takes the next one's activations) must give differences.
// `multitoken_bitid q6k` checks Q6_K's dense products at 2 to 4 tokens (C727: the kernel of mmvq-pascal-dmt.cu with
// GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 or 2), at Qwen3.8-Flash-Next's Q6_K attn_output and ssm_out (6144 -> 2560) and at the
// other row widths the kernel takes at 2 to 4 tokens (8 and 16 lanes per row), with partial last row blocks. With the
// switch at 0 the k-column kernel takes these products and its sums differ from the one-token kernel's;
// GGML_CUDA_MMVQ_PASCAL_Q6K_MT_FAULT=1 (every token stages the first token's activations) must give differences.
// `multitoken_bitid shdown` checks Q4_0's and Q5_0's dense products at 2 to 4 tokens against their one-token products
// (C755: Qwen3.8-Flash-Next's shared-expert down projections, 640 -> 2560, which the preset's
// GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 gives the k-column kernel; GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0 gives them to
// mul_mat_vec_q_pascal_mc, which runs the one-token kernel's block arithmetic for every token). Shapes: 2560 x 640,
// 4097 x 640 (a partial last row block), 640 x 2560 and 333 x 4096. Three inputs per shape and token count: standard
// normal activations ("MUL_MAT"), uniform on [-1, 1] ("MUL_MAT/unif"), and weights with one sign per column for every
// row times activations of that sign ("MUL_MAT/big"), whose blocks' integer sums I pass 2^13, where the k-column
// kernel's round(I * d) * d8 and the one-token kernel's d * round(d8 * I) part; each product line is followed by the
// count of such blocks (est., the activations quantized on the CPU as quantize_q8_1 does). The controls: with the
// preset's switches and no skip (the k-column kernel) the "big" products differ, and with the skip and
// GGML_CUDA_MMVQ_PASCAL_MC=0 (llama.cpp's mul_mat_vec_q at 2-4 tokens) products differ.
// BITID_CPU=1 runs any mode on ggml's CPU backend instead of CUDA0 (the harness's own check: no GPU needed; the CPU's
// products are computed column by column, so every comparison must be bit-identical).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(42);

// the fork's tests of whether its read-once kernels take a product (mmvq-pascal-dmt.cu, mmvq-pascal-q5k.cu), for dmt8
// and q5k8; weak, so that the other modes still link against builds without them
bool ggml_cuda_mmvq_pascal_dmt_takes(ggml_type type, int64_t K, int64_t nrows, int64_t ncols, int64_t row_bytes,
                                     const void * x) __attribute__((weak));
bool ggml_cuda_mmvq_pascal_q5k_mt_takes(int64_t K, int64_t nrows, int64_t ncols, int64_t row_bytes,
                                        const void * x) __attribute__((weak));
// whether the k-column hook takes a type at all (mmvq-pascal-kcol.cu: GGML_CUDA_MMVQ_PASCAL_KCOL not 0, the type one of
// its kernels', not in GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP), printed by shdown
bool ggml_cuda_mmvq_pascal_kcol_on(ggml_type type) __attribute__((weak));

static void fill_weights(ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t nrows = ggml_nrows(t);
    std::vector<float> f(k * nrows);
    std::normal_distribution<float> d(0.0f, 0.02f);
    for (auto & x : f) x = d(rng);
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, nrows, k, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

// rows drawn at random from 256 quantized rows of the tensor's width (the IQ quantizers are slow on whole experts)
static void fill_weights_pool(ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t nrows = ggml_nrows(t);
    const int64_t npool = 256;
    const size_t  rb = ggml_row_size(t->type, k);
    std::vector<float> f(k * npool);
    std::normal_distribution<float> d(0.0f, 0.02f);
    for (auto & x : f) x = d(rng);
    std::vector<uint8_t> pool(rb * npool);
    ggml_quantize_chunk(t->type, f.data(), pool.data(), 0, npool, k, nullptr);
    std::vector<uint8_t> q(ggml_nbytes(t));
    for (int64_t r = 0; r < nrows; ++r) {
        memcpy(q.data() + r * rb, pool.data() + (rng() % npool) * rb, rb);
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void fill_f32(ggml_tensor * t, float s) {
    std::vector<float> f(ggml_nelements(t));
    std::normal_distribution<float> d(0.0f, s);
    for (auto & x : f) x = d(rng);
    ggml_backend_tensor_set(t, f.data(), 0, f.size() * sizeof(float));
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> f(ggml_nelements(t));
    ggml_backend_tensor_get(t, f.data(), 0, f.size() * sizeof(float));
    return f;
}

// run one graph whose output is `out` in a fresh allocation
static std::vector<float> run(ggml_backend_t be, ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> r = get(out);
    ggml_gallocr_free(ga);
    return r;
}

static int n_checked = 0, n_diff = 0, n_declined = 0;

// counted = false: printed, but not counted (a product the kernel under test declines)
static void report(const char * what, ggml_type type, int n, int64_t m, int64_t k, const std::vector<float> & a,
                   const std::vector<std::vector<float>> & cols, int64_t col_elems, bool counted = true) {
    int64_t ndiff = 0;
    double maxrel = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int64_t i = 0; i < col_elems; ++i) {
            const float x = a[j * col_elems + i], y = cols[j][i];
            if (memcmp(&x, &y, 4) != 0) {
                ndiff++;
                maxrel = std::max(maxrel, (double) std::abs(x - y) / (std::abs(y) + 1e-30));
            }
        }
    }
    if (counted) {
        n_checked++;
        if (ndiff) n_diff++;
    } else {
        n_declined++;
    }
    printf("%-14s %-7s n=%d m=%-6lld k=%-5lld: %s (%lld of %lld values differ, max rel %.2e)\n", what, ggml_type_name(type), n,
           (long long) m, (long long) k, ndiff ? "DIFFERENT" : "bit-identical", (long long) ndiff,
           (long long) (n * col_elems), maxrel);
}

struct mm { int64_t m, k; };

// as report, for a pair of types (gate+up)
static void report_label(const char * what, const char * types, int n, int64_t m, int64_t k, const std::vector<float> & a,
                         const std::vector<std::vector<float>> & cols, int64_t col_elems, bool counted = true) {
    int64_t ndiff = 0;
    double maxrel = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int64_t i = 0; i < col_elems; ++i) {
            const float x = a[j * col_elems + i], y = cols[j][i];
            if (memcmp(&x, &y, 4) != 0) {
                ndiff++;
                maxrel = std::max(maxrel, (double) std::abs(x - y) / (std::abs(y) + 1e-30));
            }
        }
    }
    if (counted) {
        n_checked++;
        if (ndiff) n_diff++;
    } else {
        n_declined++;
    }
    printf("%-14s %-13s n=%d m=%-6lld k=%-5lld: %s (%lld of %lld values differ, max rel %.2e)\n", what, types, n,
           (long long) m, (long long) k, ndiff ? "DIFFERENT" : "bit-identical", (long long) ndiff,
           (long long) (n * col_elems), maxrel);
}

// gate (tg) and up (tu) of m rows of k weights, SWIGLU, at 2..n_max tokens: one graph (the fused launch where the
// backend takes it) against the products and the GLU run apart at n tokens and token by token
static void glu_check(ggml_backend_t be, ggml_type tg, ggml_type tu, int64_t m, int64_t k, int n_max) {
    ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * G = ggml_new_tensor_2d(wctx, tg, k, m);
    ggml_tensor * U = ggml_new_tensor_2d(wctx, tu, k, m);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    fill_weights_pool(G);
    fill_weights_pool(U);
    char label[48];
    snprintf(label, sizeof(label), "%s+%s", ggml_type_name(tg), ggml_type_name(tu));
    for (int n = 2; n <= n_max; ++n) {
        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * ictx = ggml_init(ip);
        ggml_tensor * x   = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, k, n);
        ggml_tensor * gn  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, m, n); // the products apart, as inputs of the GLU
        ggml_tensor * un  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, m, n);
        ggml_tensor * g1  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, m, 1);
        ggml_tensor * u1  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, m, 1);
        ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, be);
        fill_f32(x, 1.0f);

        ggml_init_params gp = { 4096 * ggml_tensor_overhead() + 64 * ggml_graph_overhead_custom(64, false), nullptr, true };
        ggml_context * g = ggml_init(gp);
        const std::vector<float> fused = run(be, g, ggml_swiglu_split(g, ggml_mul_mat(g, G, x), ggml_mul_mat(g, U, x)));
        // apart at n tokens: each product alone, then the GLU op alone on their results
        {
            const std::vector<float> gv = run(be, g, ggml_mul_mat(g, G, x));
            const std::vector<float> uv = run(be, g, ggml_mul_mat(g, U, x));
            ggml_backend_tensor_set(gn, gv.data(), 0, gv.size() * sizeof(float));
            ggml_backend_tensor_set(un, uv.data(), 0, uv.size() * sizeof(float));
            const std::vector<float> apart = run(be, g, ggml_swiglu_split(g, gn, un));
            std::vector<std::vector<float>> cols;
            for (int j = 0; j < n; ++j) {
                cols.emplace_back(apart.begin() + j * m, apart.begin() + (j + 1) * m);
            }
            report_label("GLU/apart", label, n, m, k, fused, cols, m);
        }
        // token by token: each one-token product alone, then the GLU op alone
        {
            std::vector<std::vector<float>> cols;
            for (int j = 0; j < n; ++j) {
                ggml_tensor * xj = ggml_view_2d(g, x, k, 1, x->nb[1], j * x->nb[1]);
                const std::vector<float> gv = run(be, g, ggml_mul_mat(g, G, xj));
                const std::vector<float> uv = run(be, g, ggml_mul_mat(g, U, xj));
                ggml_backend_tensor_set(g1, gv.data(), 0, gv.size() * sizeof(float));
                ggml_backend_tensor_set(u1, uv.data(), 0, uv.size() * sizeof(float));
                cols.push_back(run(be, g, ggml_swiglu_split(g, g1, u1)));
            }
            // 5 to 8 tokens (C793): an IQ4_XS, Q4_K or Q5_K product is on the one-token order only where the read-once
            // kernels take it (else the k-column kernel's sums, separate and fused alike)
            bool one_tok = true;
            if (n > 4) {
                for (const ggml_tensor * w : { G, U }) {
                    if (w->type == GGML_TYPE_Q2_0 || w->type == GGML_TYPE_Q3_K) {
                        continue;
                    }
                    one_tok = one_tok &&
                        (w->type == GGML_TYPE_Q5_K
                             ? ggml_cuda_mmvq_pascal_q5k_mt_takes && ggml_cuda_mmvq_pascal_q5k_mt_takes(k, m, n, w->nb[1], w->data)
                             : ggml_cuda_mmvq_pascal_dmt_takes && ggml_cuda_mmvq_pascal_dmt_takes(w->type, k, m, n, w->nb[1], w->data));
                }
            }
            const char * all = getenv("BITID_ALL");
            report_label(one_tok ? "GLU/1-tok" : "GLU/1-tok/decl", label, n, m, k, fused, cols, m,
                         one_tok || (all != nullptr && atoi(all) == 1));
        }
        ggml_free(g);
        ggml_backend_buffer_free(ibuf);
        ggml_free(ictx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}

static int glu_main(ggml_backend_t be) {
    const ggml_type Q2 = GGML_TYPE_Q2_0, Q3 = GGML_TYPE_Q3_K, Q4 = GGML_TYPE_Q4_K, Q5 = GGML_TYPE_Q5_K,
                    I4 = GGML_TYPE_IQ4_XS;
    // 2 to 8 tokens for every pair (C793; C757 checked IQ4_XS, Q4_K and Q5_K pairs at 2 to 4)
    const auto n_max = [](ggml_type, ggml_type) { return 8; };
    // Qwen3.8-Flash-Next's shared experts (Swift GSQ-RCO Q2_0), gate+up, by layer count
    const std::vector<std::pair<ggml_type, ggml_type>> fn = {
        { Q2, Q3 }, { Q3, Q3 }, { Q2, Q2 }, { Q3, Q2 }, { Q2, Q4 }, { Q2, I4 }, { I4, Q3 }, { Q4, Q4 }, { Q4, Q3 },
        { Q4, Q2 }, { Q3, I4 }, { I4, Q2 }, { Q5, Q3 }, { Q3, Q5 }, { I4, Q4 },
    };
    for (const auto & [tg, tu] : fn) {
        glu_check(be, tg, tu, 640, 2560, n_max(tg, tu));
    }
    const ggml_type types[] = { Q2, Q3, Q4, Q5, I4 };
    for (ggml_type tg : types) {
        for (ggml_type tu : types) {
            for (const mm s : { mm{ 77, 1024 }, mm{ 33, 8192 }, mm{ 200, 512 } }) {
                glu_check(be, tg, tu, s.m, s.k, n_max(tg, tu));
            }
        }
    }
    printf("%d comparisons, %d with differences; %d one-token comparisons declined (not on the read-once kernels), %s\n",
           n_checked, n_diff, n_declined, getenv("BITID_ALL") && atoi(getenv("BITID_ALL")) == 1 ? "counted (BITID_ALL=1)" : "not counted");
    ggml_backend_free(be);
    return n_diff != 0;
}

// Q5_K's one-token products, written to or compared with the file BITID_REF (see the top)
static int q5k1_main(ggml_backend_t be) {
    const char * ref = getenv("BITID_REF");
    if (ref == nullptr) {
        fprintf(stderr, "q5k1: set BITID_REF=<file>\n");
        return 2;
    }
    FILE * f = fopen(ref, "rb");
    const bool write = f == nullptr;
    if (write) {
        f = fopen(ref, "wb");
    }
    GGML_ASSERT(f);
    int n_written = 0;
    const std::vector<mm> shapes = { { 248320, 2560 }, { 2560, 6144 }, { 512, 2560 }, { 640, 2560 }, { 4097, 2560 },
                                     { 200, 768 }, { 1000, 1024 }, { 129, 2048 }, { 77, 4096 }, { 64, 8192 },
                                     { 96, 12288 }, { 16, 256 } };
    for (const mm & s : shapes) {
        ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_Q5_K, s.k, s.m);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        fill_weights_pool(W);
        for (const float sigma : { 1.0f, 250000.0f }) {
            ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * ictx = ggml_init(ip);
            ggml_tensor * x = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, s.k, 1);
            ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, be);
            fill_f32(x, sigma);
            ggml_init_params gp = { 64 * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(64, false), nullptr, true };
            ggml_context * g = ggml_init(gp);
            const std::vector<float> out = run(be, g, ggml_mul_mat(g, W, x));
            if (write) {
                GGML_ASSERT(fwrite(out.data(), sizeof(float), out.size(), f) == out.size());
                n_written++;
            } else {
                std::vector<std::vector<float>> cols(1, std::vector<float>(out.size(), NAN));
                if (fread(cols[0].data(), sizeof(float), out.size(), f) != out.size()) {
                    printf("q5k1: %s ends before m=%lld k=%lld\n", ref, (long long) s.m, (long long) s.k);
                    n_checked++;
                    n_diff++;
                } else {
                    report(sigma > 1.0f ? "MUL_MAT/big" : "MUL_MAT", GGML_TYPE_Q5_K, 1, s.m, s.k, out, cols, s.m);
                }
            }
            ggml_free(g);
            ggml_backend_buffer_free(ibuf);
            ggml_free(ictx);
        }
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }
    fclose(f);
    if (write) {
        printf("wrote %d products to %s (the reference)\n", n_written, ref);
    } else {
        printf("%d comparisons, %d with differences\n", n_checked, n_diff);
    }
    ggml_backend_free(be);
    return n_diff != 0;
}

// shdown: the blocks of 32 weights and one token whose integer sum I = sum (code - offset) * q reaches 2^13, with the
// activations x (n columns of k) quantized as quantize_q8_1 does (est.: the GPU's division, done on the CPU)
static int64_t shdown_big_blocks(ggml_type type, const std::vector<uint8_t> & w, int64_t m, int64_t k,
                                 const std::vector<float> & x, int n) {
    const size_t  bs = ggml_type_size(type);
    const int64_t nb = k / 32;
    std::vector<int8_t> q(x.size());
    for (size_t b = 0; b < x.size() / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(x[b * 32 + i]));
        const float d = amax / 127;
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = amax == 0.0f ? 0 : (int8_t) roundf(x[b * 32 + i] / d);
    }
    int64_t big = 0;
    int c[32];
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t * blk = w.data() + (r * nb + b) * bs;
            if (type == GGML_TYPE_Q4_0) {
                for (int j = 0; j < 16; ++j) {
                    c[j]      = (blk[2 + j] & 15) - 8;
                    c[j + 16] = (blk[2 + j] >> 4) - 8;
                }
            } else {
                uint32_t qh;
                memcpy(&qh, blk + 2, 4);
                for (int j = 0; j < 16; ++j) {
                    c[j]      = ((blk[6 + j] & 15) | (((qh >> j) << 4) & 0x10)) - 16;
                    c[j + 16] = ((blk[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
                }
            }
            for (int t = 0; t < n; ++t) {
                const int8_t * qt = q.data() + t * k + b * 32;
                int s = 0;
                for (int i = 0; i < 32; ++i) s += c[i] * qt[i];
                big += std::abs(s) >= 8192;
            }
        }
    }
    return big;
}

static std::vector<uint8_t> set_quantized(ggml_tensor * t, const std::vector<float> & f) {
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
    return q;
}

// Q4_0's and Q5_0's products of 2 to 4 tokens against their one-token products (see the top)
static int shdown_main(ggml_backend_t be) {
    const std::vector<mm> shapes = { { 2560, 640 }, { 4097, 640 }, { 640, 2560 }, { 333, 4096 } };
    static const char * what[3] = { "MUL_MAT", "MUL_MAT/unif", "MUL_MAT/big" };
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q5_0 }) {
        if (ggml_cuda_mmvq_pascal_kcol_on) {
            printf("%s: the k-column hook %s this type\n", ggml_type_name(type),
                   ggml_cuda_mmvq_pascal_kcol_on(type) ? "takes" : "leaves");
        }
        for (const mm & s : shapes) {
            // W: weights of standard deviation 0.02; S: one sign per column for every row, magnitudes 0.01 to 0.02
            std::vector<float> sign(s.k), wf(s.k * s.m), sf(s.k * s.m);
            for (auto & v : sign) v = (rng() & 1) ? 1.0f : -1.0f;
            for (auto & v : wf) v = 0.02f * nd(rng);
            for (int64_t i = 0; i < s.k * s.m; ++i) sf[i] = sign[i % s.k] * 0.01f * (1.0f + ud(rng));
            ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * wctx = ggml_init(wp);
            ggml_tensor * W = ggml_new_tensor_2d(wctx, type, s.k, s.m);
            ggml_tensor * S = ggml_new_tensor_2d(wctx, type, s.k, s.m);
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
            const std::vector<uint8_t> wq = set_quantized(W, wf);
            const std::vector<uint8_t> sq = set_quantized(S, sf);
            for (int n = 2; n <= 4; ++n) {
                ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
                ggml_context * ictx = ggml_init(ip);
                ggml_tensor * B = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, s.k, n);
                ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, be);
                for (int c = 0; c < 3; ++c) {
                    // activations: standard normal, uniform on [-1, 1], or the column's sign times 0.75 to 1
                    std::vector<float> x(s.k * n);
                    for (int64_t i = 0; i < s.k * n; ++i) {
                        x[i] = c == 0 ? nd(rng) : c == 1 ? 2.0f * ud(rng) - 1.0f : sign[i % s.k] * (1.0f - 0.25f * ud(rng));
                    }
                    ggml_backend_tensor_set(B, x.data(), 0, x.size() * sizeof(float));
                    ggml_tensor * A = c == 2 ? S : W;
                    ggml_init_params gp = { 64 * ggml_tensor_overhead() + 8 * ggml_graph_overhead_custom(64, false), nullptr, true };
                    ggml_context * g = ggml_init(gp);
                    const std::vector<float> all = run(be, g, ggml_mul_mat(g, A, B));
                    std::vector<std::vector<float>> cols;
                    for (int j = 0; j < n; ++j) {
                        ggml_tensor * bj = ggml_view_2d(g, B, s.k, 1, B->nb[1], j * B->nb[1]);
                        cols.push_back(run(be, g, ggml_mul_mat(g, A, bj)));
                    }
                    report(what[c], type, n, s.m, s.k, all, cols, s.m);
                    printf("%-14s %-7s n=%d m=%-6lld k=%-5lld: %lld of %lld blocks with |I| >= 2^13 (est.)\n", "", ggml_type_name(type),
                           n, (long long) s.m, (long long) s.k,
                           (long long) shdown_big_blocks(type, c == 2 ? sq : wq, s.m, s.k, x, n),
                           (long long) (s.m * (s.k / 32) * n));
                    ggml_free(g);
                }
                ggml_backend_buffer_free(ibuf);
                ggml_free(ictx);
            }
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
        }
    }
    printf("%d comparisons, %d with differences\n", n_checked, n_diff);
    ggml_backend_free(be);
    return n_diff != 0;
}

// one model's types and shapes: dense products, E experts of which U used, experts' hidden width H and FFN width F
struct mode_cfg {
    std::vector<ggml_type> types;
    std::vector<mm> dense;
    int E, U;
    int64_t H, F;
    bool pool;      // weights drawn from a pool of quantized rows
    int n_routes;   // routings (route())
};

// slot i of token j: 0 = neighbours share some experts ((3i + 2j) % E), 1 = every token on experts 0..U-1 in rotated
// slots, 2 = experts 0..U/2-1 shared, the others the token's own
static int32_t route(int r, int i, int j, int E, int U) {
    if (r == 1) return (i + j) % U;
    if (r == 2) return i < U/2 ? i : U/2 + ((i - U/2) + (U - U/2) * j) % (E - U/2);
    return (3 * i + 2 * j) % E;
}

int main(int argc, char ** argv) {
    const bool on_cpu = getenv("BITID_CPU") != nullptr && atoi(getenv("BITID_CPU")) == 1;
    ggml_backend_t be = on_cpu ? ggml_backend_cpu_init() : ggml_backend_cuda_init(0);
    GGML_ASSERT(be);
    printf("backend: %s\n", ggml_backend_name(be));
    if (argc > 1 && strcmp(argv[1], "shdown") == 0) {
        return shdown_main(be);
    }
    if (argc > 1 && strcmp(argv[1], "glu") == 0) {
        return glu_main(be);
    }
    if (argc > 1 && strcmp(argv[1], "q5k1") == 0) {
        return q5k1_main(be);
    }
    const bool ornith = argc > 1 && strcmp(argv[1], "ornith") == 0;
    const bool q5k    = argc > 1 && strcmp(argv[1], "q5k") == 0;
    const bool dmt    = argc > 1 && strcmp(argv[1], "dmt") == 0;
    const bool dmtkt  = argc > 1 && strcmp(argv[1], "dmtkt") == 0;
    const bool bf16   = argc > 1 && strcmp(argv[1], "bf16") == 0;
    const bool f32    = argc > 1 && strcmp(argv[1], "f32") == 0;
    const bool f32r   = argc > 1 && strcmp(argv[1], "f32r") == 0;
    const bool q3k    = argc > 1 && strcmp(argv[1], "q3k") == 0;
    const bool q6k    = argc > 1 && strcmp(argv[1], "q6k") == 0;
    const bool dmt8   = argc > 1 && strcmp(argv[1], "dmt8") == 0;
    const bool q5k8   = argc > 1 && strcmp(argv[1], "q5k8") == 0;
    const bool kt8    = argc > 1 && strcmp(argv[1], "kt8") == 0;
    const bool count_all = getenv("BITID_ALL") != nullptr && atoi(getenv("BITID_ALL")) == 1;
    const int  n_min  = dmt8 || q5k8 || kt8 ? 5 : 2;
    const int  n_max  = f32 || f32r || q3k || dmt8 || q5k8 || kt8 ? 8 : 4;
    if (((dmt8 || kt8) && !ggml_cuda_mmvq_pascal_dmt_takes) || (q5k8 && !ggml_cuda_mmvq_pascal_q5k_mt_takes)) {
        fprintf(stderr, "%s: this build has no read-once kernels (ggml_cuda_mmvq_pascal_*_takes)\n", argv[1]);
        return 2;
    }
    const mode_cfg cfg = q6k
        ? mode_cfg{ { GGML_TYPE_Q6_K },
                    { { 2560, 6144 }, { 1001, 6144 }, { 4097, 2048 }, { 200, 2048 }, { 77, 4096 }, { 333, 4096 } },
                    1, 1, 256, 256, true, 0 }
        : q3k
        ? mode_cfg{ { GGML_TYPE_Q3_K },
                    { { 10240, 2560 }, { 6144, 2560 }, { 12288, 2560 }, { 2560, 6144 }, { 640, 2560 }, { 4097, 2560 },
                      { 200, 512 }, { 1000, 1024 }, { 129, 2048 }, { 77, 4096 }, { 64, 8192 }, { 96, 12288 },
                      { 333, 768 } },
                    1, 1, 256, 256, true, 0 }
        : ornith
        ? mode_cfg{ { GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_N2P, GGML_TYPE_N3P, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0 },
                    { { 8192, 3072 }, { 3072, 8192 }, { 3072, 1024 }, { 1024, 3072 }, { 4097, 3072 } },
                    16, 8, 3072, 1024, true, 3 }
        : q5k
        ? mode_cfg{ { GGML_TYPE_Q5_K },
                    { { 248320, 2560 }, { 4097, 2560 }, { 200, 768 }, { 1000, 1024 }, { 129, 2048 }, { 77, 4096 },
                      { 64, 8192 } },
                    1, 1, 256, 256, true, 0 }
        : q5k8
        ? mode_cfg{ { GGML_TYPE_Q5_K },
                    { { 248320, 2560 }, { 4097, 2560 }, { 200, 768 }, { 1000, 1024 }, { 129, 2048 }, { 77, 4096 },
                      { 64, 8192 }, { 2560, 6144 }, { 512, 2560 }, { 640, 2560 } },
                    1, 1, 256, 256, true, 0 }
        : dmtkt || kt8
        ? mode_cfg{ { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K },
                    { { 2560, 6144 }, { 333, 6144 }, { 4097, 6144 }, { 10240, 2560 }, { 6144, 2560 }, { 12288, 2560 },
                      { 4097, 2560 }, { 200, 512 }, { 1000, 1024 }, { 129, 2048 }, { 77, 4096 }, { 64, 8192 },
                      { 96, 12288 } },
                    1, 1, 256, 256, true, 0 }
        : dmt || dmt8
        ? mode_cfg{ { GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K },
                    { { 10240, 2560 }, { 6144, 2560 }, { 12288, 2560 }, { 2560, 6144 }, { 4097, 2560 }, { 200, 512 },
                      { 1000, 1024 }, { 129, 2048 }, { 77, 4096 }, { 64, 8192 } },
                    1, 1, 256, 256, true, 0 }
        : bf16
        ? mode_cfg{ { GGML_TYPE_BF16 },
                    { { 320, 10240 }, { 10240, 320 }, { 4, 10240 }, { 512, 2560 }, { 48, 2560 }, { 333, 1000 }, { 320, 4104 },
                      { 320, 4392 }, { 333, 4736 } },
                    1, 1, 256, 256, false, 0 }
        : f32
        ? mode_cfg{ { GGML_TYPE_F32 },
                    { { 1, 2560 }, { 512, 2560 }, { 48, 64 }, { 33, 100 }, { 129, 300 }, { 7, 6144 } },
                    1, 1, 256, 256, false, 0 }
        : f32r
        ? mode_cfg{ { GGML_TYPE_F32 },
                    { { 1, 2560 }, { 32, 2560 }, { 32, 64 }, { 31, 100 }, { 17, 300 }, { 7, 6144 } },
                    1, 1, 256, 256, false, 0 }
        : mode_cfg{ { GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS,
                      GGML_TYPE_Q6_K },
                    { { 4096, 2048 }, { 2048, 4096 }, { 512, 2048 }, { 2048, 1024 }, { 4097, 2048 } },
                    32, 6, 2048, 512, false, 1 };
    const int     E = cfg.E, U = cfg.U;
    const int64_t H = cfg.H, F = cfg.F;

    for (ggml_type type : cfg.types) {
        // weights live in their own context and buffer
        ggml_init_params wp = { 64 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        const std::vector<mm> & dense = cfg.dense;
        std::vector<ggml_tensor *> W;
        for (auto s : dense) W.push_back(ggml_new_tensor_2d(wctx, type, s.k, s.m));
        ggml_tensor * ups   = ggml_new_tensor_3d(wctx, type, H, F, E);
        ggml_tensor * gates = ggml_new_tensor_3d(wctx, type, H, F, E);
        ggml_tensor * downs = ggml_new_tensor_3d(wctx, type, F, H, E);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (cfg.pool) fill_weights_pool(t); else fill_weights(t);
        }

        for (int n = n_min; n <= n_max; ++n) {
            ggml_init_params ip = { 256 * ggml_tensor_overhead() + 64 * ggml_graph_overhead_custom(64, false), nullptr, true };
            // inputs
            ggml_context * ictx = ggml_init(ip);
            std::vector<ggml_tensor *> B;
            for (auto s : dense) B.push_back(ggml_new_tensor_2d(ictx, GGML_TYPE_F32, s.k, n));
            ggml_tensor * xb   = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, H, 1, n);   // broadcast input of gate/up
            ggml_tensor * xd   = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, F, U, n);   // one input per used expert
            ggml_tensor * ids  = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, U, n);
            ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, be);
            for (auto * b : B) fill_f32(b, 1.0f);
            fill_f32(xb, 1.0f);
            fill_f32(xd, 1.0f);

            ggml_init_params gp = { 4096 * ggml_tensor_overhead() + 64 * ggml_graph_overhead_custom(64, false), nullptr, true };
            for (size_t s = 0; s < W.size(); ++s) {
                ggml_context * g = ggml_init(gp);
                std::vector<float> all = run(be, g, ggml_mul_mat(g, W[s], B[s]));
                std::vector<std::vector<float>> cols;
                for (int j = 0; j < n; ++j) {
                    ggml_tensor * bj = ggml_view_2d(g, B[s], B[s]->ne[0], 1, B[s]->nb[1], j * B[s]->nb[1]);
                    cols.push_back(run(be, g, ggml_mul_mat(g, W[s], bj)));
                }
                // dmt8, kt8, q5k8: whether the read-once kernel takes the product (else the k-column kernel does)
                const int64_t rb = W[s]->nb[1];
                const bool taken = dmt8 || kt8 ? ggml_cuda_mmvq_pascal_dmt_takes(type, dense[s].k, dense[s].m, n, rb, W[s]->data)
                                 : q5k8 ? ggml_cuda_mmvq_pascal_q5k_mt_takes(dense[s].k, dense[s].m, n, rb, W[s]->data)
                                 : true;
                report(taken ? "MUL_MAT" : "MUL_MAT/declined", type, n, dense[s].m, dense[s].k, all, cols, dense[s].m,
                       taken || count_all);
                ggml_free(g);
            }
            for (int r = 0; r < cfg.n_routes; ++r) {
                std::vector<int32_t> idv(U * n);
                for (int j = 0; j < n; ++j) for (int i = 0; i < U; ++i) idv[j * U + i] = route(r, i, j, E, U);
                ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * 4);
                char what_id[32], what_glu[32];
                snprintf(what_id, sizeof(what_id), cfg.n_routes > 1 ? "MUL_MAT_ID/r%d" : "MUL_MAT_ID", r);
                snprintf(what_glu, sizeof(what_glu), cfg.n_routes > 1 ? "MMID+GLU/r%d" : "MUL_MAT_ID+GLU", r);
                // routed experts: down (one input per slot) and fused gate/up + SWIGLU (broadcast input)
                {
                    ggml_context * g = ggml_init(gp);
                    std::vector<float> all = run(be, g, ggml_mul_mat_id(g, downs, xd, ids));
                    std::vector<std::vector<float>> cols;
                    for (int j = 0; j < n; ++j) {
                        ggml_tensor * xj = ggml_view_3d(g, xd, F, U, 1, xd->nb[1], xd->nb[2], j * xd->nb[2]);
                        ggml_tensor * ij = ggml_view_2d(g, ids, U, 1, ids->nb[1], j * ids->nb[1]);
                        cols.push_back(run(be, g, ggml_mul_mat_id(g, downs, xj, ij)));
                    }
                    report(what_id, type, n, H, F, all, cols, H * U);
                    ggml_free(g);
                }
                {
                    ggml_context * g = ggml_init(gp);
                    std::vector<float> all = run(be, g, ggml_swiglu_split(g, ggml_mul_mat_id(g, gates, xb, ids), ggml_mul_mat_id(g, ups, xb, ids)));
                    std::vector<std::vector<float>> cols;
                    for (int j = 0; j < n; ++j) {
                        ggml_tensor * xj = ggml_view_3d(g, xb, H, 1, 1, xb->nb[1], xb->nb[2], j * xb->nb[2]);
                        ggml_tensor * ij = ggml_view_2d(g, ids, U, 1, ids->nb[1], j * ids->nb[1]);
                        cols.push_back(run(be, g, ggml_swiglu_split(g, ggml_mul_mat_id(g, gates, xj, ij), ggml_mul_mat_id(g, ups, xj, ij))));
                    }
                    report(what_glu, type, n, F, H, all, cols, F * U);
                    ggml_free(g);
                }
            }
            ggml_backend_buffer_free(ibuf);
            ggml_free(ictx);
        }
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }
    if (dmt8 || q5k8 || kt8) {
        printf("%d declined by the read-once kernel, %s\n", n_declined, count_all ? "counted (BITID_ALL=1)" : "not counted");
    }
    printf("%d comparisons, %d with differences\n", n_checked, n_diff);
    ggml_backend_free(be);
    return n_diff != 0;
}
