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
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(42);

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

static int n_checked = 0, n_diff = 0;

static void report(const char * what, ggml_type type, int n, int64_t m, int64_t k, const std::vector<float> & a,
                   const std::vector<std::vector<float>> & cols, int64_t col_elems) {
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
    n_checked++;
    if (ndiff) n_diff++;
    printf("%-14s %-7s n=%d m=%-6lld k=%-5lld: %s (%lld of %lld values differ, max rel %.2e)\n", what, ggml_type_name(type), n,
           (long long) m, (long long) k, ndiff ? "DIFFERENT" : "bit-identical", (long long) ndiff,
           (long long) (n * col_elems), maxrel);
}

struct mm { int64_t m, k; };

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
    ggml_backend_t be = ggml_backend_cuda_init(0);
    GGML_ASSERT(be);
    const bool ornith = argc > 1 && strcmp(argv[1], "ornith") == 0;
    const mode_cfg cfg = ornith
        ? mode_cfg{ { GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_N2P, GGML_TYPE_N3P, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0 },
                    { { 8192, 3072 }, { 3072, 8192 }, { 3072, 1024 }, { 1024, 3072 }, { 4097, 3072 } },
                    16, 8, 3072, 1024, true, 3 }
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

        for (int n = 2; n <= 4; ++n) {
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
                report("MUL_MAT", type, n, dense[s].m, dense[s].k, all, cols, dense[s].m);
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
    printf("%d comparisons, %d with differences\n", n_checked, n_diff);
    ggml_backend_free(be);
    return n_diff != 0;
}
