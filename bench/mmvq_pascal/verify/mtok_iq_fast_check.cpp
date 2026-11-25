// Old against new for a multi-token expert kernel of the fork (C1138: IQ2_XXS's fast kernel, GGML_CUDA_MTOK_IQ_FAST):
// runs a fixed set of MUL_MAT_ID products on CUDA0 and writes every output value, so that two runs, one with the switch
// off and one with it on, can be compared byte for byte (`cmp`). Each case also prints a line with an FNV-1a hash of
// its outputs, so the two logs can be compared with `diff`.
//
// The products are Qwen3.8-Flash-Next's routed gate/up shape in Swift Qwen3.8-Flash-Next Pascal's gmix-q2 file: 512
// experts of 640 rows of 2,560 weights, 10 used per token, at 1 to 8 tokens, as
//   - fused: SWIGLU(gate, up) of two MUL_MAT_ID, which the CUDA backend runs as one gate/up launch;
//   - plain: one MUL_MAT_ID;
//   - shared: the fused pair plus a shared expert of the same type and shape added to it (the fork's routed + shared
//     launch);
// with five routings: distinct random experts per token; neighbours sharing 4 of their 10; every token on the same 10
// experts in other slots (groups of up to 8 members); an expert twice within a token; 3 slots of each token skipped
// (id -1, which the fork's kernels write as zeros).
// The weights: experts 0-255 random blocks (any grid index, sign pattern and scale, a random FP16 d of 2^-12..2^-4),
// experts 256-511 rows drawn from 256 rows quantized from normal values (ggml's quantizer, a unit importance matrix).
// Activations are normal values (the backend quantizes them to Q8_1).
//
// Usage: mtok_iq_fast_check <out.bin> [type]   (type: iq2_xxs, the default)
// Run it with the fork's Pascal kernels forced on a newer GPU ($PASCAL_FORCED of scripts/pascal_forced.sh), once with
// the switch off and once on; GGML_CUDA_MTOK_IQ_FAST_FAULT=1 (the fast kernel takes up and gate the wrong way round)
// must make every fused and shared case of 2 to 8 tokens differ.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(1138);

static const int64_t K = 2560, M = 640, NEXP = 512, NUSED = 10;

static void fill_iq_experts(ggml_tensor * t) {
    const int64_t rb = ggml_row_size(t->type, K);
    const int64_t bs = ggml_type_size(t->type);
    const int64_t nrows = ggml_nrows(t);
    std::vector<uint8_t> q(ggml_nbytes(t));
    // a pool of 256 quantized rows
    const int64_t npool = 256;
    std::vector<float> f(K * npool);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    for (auto & x : f) x = nd(rng);
    std::vector<float> imatrix(K, 1.0f);
    std::vector<uint8_t> pool(rb * npool);
    ggml_quantize_chunk(t->type, f.data(), pool.data(), 0, npool, K, ggml_quantize_requires_imatrix(t->type) ? imatrix.data() : nullptr);
    std::uniform_int_distribution<int> ex(-12, -4);
    for (int64_t r = 0; r < nrows; ++r) {
        uint8_t * row = q.data() + r * rb;
        if (r < nrows / 2) {
            for (int64_t i = 0; i < rb; i += 4) {
                const uint32_t v = rng();
                memcpy(row + i, &v, std::min<int64_t>(4, rb - i));
            }
            for (int64_t b = 0; b < rb / bs; ++b) {
                const float d = std::ldexp((rng() & 1) ? 1.0f : -1.0f, ex(rng)) * (1.0f + (rng() % 1000) / 1000.0f);
                const ggml_fp16_t h = ggml_fp32_to_fp16(d);
                memcpy(row + b * bs, &h, 2);
            }
        } else {
            memcpy(row, pool.data() + (rng() % npool) * rb, rb);
        }
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void fill_f32(ggml_tensor * t) {
    std::vector<float> f(ggml_nelements(t));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & x : f) x = nd(rng);
    ggml_backend_tensor_set(t, f.data(), 0, f.size() * sizeof(float));
}

// ids [NUSED, n] for routing r
static std::vector<int32_t> make_ids(int r, int n) {
    std::vector<int32_t> ids(NUSED * n);
    std::uniform_int_distribution<int> e(0, NEXP - 1);
    auto distinct = [&](int32_t * s, int k) {
        for (int i = 0; i < k; ++i) {
            for (;;) {
                const int32_t v = e(rng);
                bool dup = false;
                for (int j = 0; j < i; ++j) dup = dup || s[j] == v;
                if (!dup) { s[i] = v; break; }
            }
        }
    };
    for (int t = 0; t < n; ++t) {
        int32_t * s = ids.data() + NUSED * t;
        switch (r) {
            case 0: distinct(s, NUSED); break;
            case 1: // 4 of the previous token's experts, in other slots, and 6 new ones
                distinct(s, NUSED);
                if (t > 0) {
                    for (int i = 0; i < 4; ++i) s[(i * 3 + t) % NUSED] = ids[NUSED * (t - 1) + i];
                    // keep them distinct
                    for (int i = 0; i < NUSED; ++i) for (int j = 0; j < i; ++j) if (s[i] == s[j]) s[i] = (s[i] + 1 + i) % NEXP;
                }
                break;
            case 2: // every token on the same 10 experts, rotated
                if (t == 0) distinct(s, NUSED);
                else for (int i = 0; i < NUSED; ++i) s[i] = ids[(i + t) % NUSED];
                break;
            case 3: // an expert twice within the token
                distinct(s, NUSED);
                s[NUSED - 1] = s[t % (NUSED - 1)];
                break;
            default: // 3 slots skipped
                distinct(s, NUSED);
                for (int i = 0; i < 3; ++i) s[(i * 4 + t) % NUSED] = -1;
                break;
        }
    }
    return ids;
}

static uint64_t fnv(const std::vector<float> & v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *) v.data();
    for (size_t i = 0; i < v.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <out.bin> [iq2_xxs]\n", argv[0]); return 2; }
    const std::string tname = argc > 2 ? argv[2] : "iq2_xxs";
    ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && tname == nm) type = (ggml_type) t;
    }
    if (type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", tname.c_str()); return 2; }
    FILE * out = fopen(argv[1], "wb");
    if (!out) { perror(argv[1]); return 2; }

    ggml_backend_t be = ggml_backend_cuda_init(0);
    GGML_ASSERT(be);

    ggml_init_params wp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * U  = ggml_new_tensor_3d(wctx, type, K, M, NEXP);
    ggml_tensor * G  = ggml_new_tensor_3d(wctx, type, K, M, NEXP);
    ggml_tensor * SU = ggml_new_tensor_2d(wctx, type, K, M);
    ggml_tensor * SG = ggml_new_tensor_2d(wctx, type, K, M);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    fill_iq_experts(U);
    fill_iq_experts(G);
    fill_iq_experts(SU);
    fill_iq_experts(SG);

    const char * rname[] = { "distinct", "neighbours", "same10", "repeat", "skip3" };
    int ncases = 0;
    for (int n = 1; n <= 8; ++n) {
        for (int r = 0; r < 5; ++r) {
            const std::vector<int32_t> idv = make_ids(r, n);
            for (int g = 0; g < 3; ++g) {
                if (g == 2 && (r == 2 || r == 3)) continue; // shared: three routings
                ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUSED, n);
                ggml_tensor * x3  = ggml_reshape_3d(ctx, x, K, 1, n);
                ggml_tensor * o;
                if (g == 0) {
                    o = ggml_swiglu_split(ctx, ggml_mul_mat_id(ctx, G, x3, ids), ggml_mul_mat_id(ctx, U, x3, ids));
                } else if (g == 1) {
                    o = ggml_mul_mat_id(ctx, U, x3, ids);
                } else {
                    ggml_tensor * routed = ggml_swiglu_split(ctx, ggml_mul_mat_id(ctx, G, x3, ids), ggml_mul_mat_id(ctx, U, x3, ids));
                    ggml_tensor * shared = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, SG, x), ggml_mul_mat(ctx, SU, x));
                    o = ggml_add(ctx, routed, ggml_reshape_3d(ctx, shared, M, 1, n));
                }
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
                ggml_build_forward_expand(gf, o);
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
                fill_f32(x);
                ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * sizeof(int32_t));
                GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
                std::vector<float> v(ggml_nelements(o));
                ggml_backend_tensor_get(o, v.data(), 0, v.size() * sizeof(float));
                // skipped slots' rows: the fork's kernels write zeros there, llama.cpp's may leave them; compare the rest
                size_t nan = 0;
                for (float f : v) nan += std::isnan(f) || std::isinf(f);
                fwrite(v.data(), sizeof(float), v.size(), out);
                printf("case %-6s %-10s n=%d: %zu values, %zu not finite, fnv %016llx\n", g == 0 ? "fused" : g == 1 ? "plain" : "shared",
                       rname[r], n, v.size(), nan, (unsigned long long) fnv(v));
                ncases++;
                ggml_gallocr_free(ga);
                ggml_free(ctx);
            }
        }
    }
    fclose(out);
    printf("%d cases, type %s\n", ncases, ggml_type_name(type));
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
