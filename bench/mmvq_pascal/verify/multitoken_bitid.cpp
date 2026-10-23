// Bit-identity check of the fork's multi-token Pascal kernels (mmvq-pascal.cu, branch pascal-p106-v2-multicol): each
// product of n = 2..4 tokens must equal, bit for bit, the n single-token products of the same columns, on CUDA0.
// Covers MUL_MAT at Flint's dense shapes, MUL_MAT_ID (6 of 32 experts, tokens that share experts) and the fused
// MUL_MAT_ID gate/up + SWIGLU, for Q8_0, Q5_0, Q4_1, Q4_0, Q5_K and IQ4_XS. Run with GGML_CUDA_MMVQ_PASCAL=2 on a newer
// GPU (add CUDA_FORCE_PTX_JIT=1 for the compute 6.1 code path) or as is on a P106. With GGML_CUDA_MMVQ_PASCAL=0 the
// stock kernels run, whose multi-token and single-token kernels sum in different orders: most comparisons then differ.
// Build: verify/build_multitoken_bitid.sh. Exit status 1 if any comparison differs.
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

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    GGML_ASSERT(be);
    const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS };

    for (ggml_type type : types) {
        // weights live in their own context and buffer
        ggml_init_params wp = { 64 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        struct mm { int64_t m, k; };
        const mm dense[] = { { 4096, 2048 }, { 2048, 4096 }, { 512, 2048 }, { 2048, 1024 }, { 4097, 2048 } };
        std::vector<ggml_tensor *> W;
        for (auto s : dense) W.push_back(ggml_new_tensor_2d(wctx, type, s.k, s.m));
        const int E = 32, U = 6;
        ggml_tensor * ups   = ggml_new_tensor_3d(wctx, type, 2048, 512, E);
        ggml_tensor * gates = ggml_new_tensor_3d(wctx, type, 2048, 512, E);
        ggml_tensor * downs = ggml_new_tensor_3d(wctx, type, 512, 2048, E);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) fill_weights(t);

        for (int n = 2; n <= 4; ++n) {
            ggml_init_params ip = { 256 * ggml_tensor_overhead() + 64 * ggml_graph_overhead_custom(64, false), nullptr, true };
            // inputs
            ggml_context * ictx = ggml_init(ip);
            std::vector<ggml_tensor *> B;
            for (auto s : dense) B.push_back(ggml_new_tensor_2d(ictx, GGML_TYPE_F32, s.k, n));
            ggml_tensor * xb   = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, 2048, 1, n);   // broadcast input of gate/up
            ggml_tensor * xd   = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, 512, U, n);    // one input per used expert
            ggml_tensor * ids  = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, U, n);
            ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, be);
            for (auto * b : B) fill_f32(b, 1.0f);
            fill_f32(xb, 1.0f);
            fill_f32(xd, 1.0f);
            // tokens share some experts: token j picks (3*i + 2*j) % E, so neighbours overlap in 4 of 6
            std::vector<int32_t> idv(U * n);
            for (int j = 0; j < n; ++j) for (int i = 0; i < U; ++i) idv[j * U + i] = (3 * i + 2 * j) % E;
            ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * 4);

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
            // routed experts: down (one input per slot) and fused gate/up + SWIGLU (broadcast input)
            {
                ggml_context * g = ggml_init(gp);
                std::vector<float> all = run(be, g, ggml_mul_mat_id(g, downs, xd, ids));
                std::vector<std::vector<float>> cols;
                for (int j = 0; j < n; ++j) {
                    ggml_tensor * xj = ggml_view_3d(g, xd, 512, U, 1, xd->nb[1], xd->nb[2], j * xd->nb[2]);
                    ggml_tensor * ij = ggml_view_2d(g, ids, U, 1, ids->nb[1], j * ids->nb[1]);
                    cols.push_back(run(be, g, ggml_mul_mat_id(g, downs, xj, ij)));
                }
                report("MUL_MAT_ID", type, n, 2048, 512, all, cols, 2048 * U);
                ggml_free(g);
            }
            {
                ggml_context * g = ggml_init(gp);
                std::vector<float> all = run(be, g, ggml_swiglu_split(g, ggml_mul_mat_id(g, gates, xb, ids), ggml_mul_mat_id(g, ups, xb, ids)));
                std::vector<std::vector<float>> cols;
                for (int j = 0; j < n; ++j) {
                    ggml_tensor * xj = ggml_view_3d(g, xb, 2048, 1, 1, xb->nb[1], xb->nb[2], j * xb->nb[2]);
                    ggml_tensor * ij = ggml_view_2d(g, ids, U, 1, ids->nb[1], j * ids->nb[1]);
                    cols.push_back(run(be, g, ggml_swiglu_split(g, ggml_mul_mat_id(g, gates, xj, ij), ggml_mul_mat_id(g, ups, xj, ij))));
                }
                report("MUL_MAT_ID+GLU", type, n, 512, 2048, all, cols, 512 * U);
                ggml_free(g);
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
