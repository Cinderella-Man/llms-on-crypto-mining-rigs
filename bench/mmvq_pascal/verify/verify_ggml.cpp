// The ggml side of the bit-exactness check: run one MUL_MAT / MUL_MAT_ID (optionally fused with a SwiGLU gate) through
// llama.cpp's real CUDA backend and return its output. Built against a llama.cpp tree that has the Q4_1_64 patch
// (patches/q4_1_64-type.diff), because the benchmark includes that type.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "verify_case.h"

#include <cstdio>

static ggml_type to_ggml(int t) {
    switch (t) {
        case 0:  return GGML_TYPE_Q4_0;
        case 1:  return GGML_TYPE_Q4_1;
        case 2:  return GGML_TYPE_Q8_0;
        default: return GGML_TYPE_Q4_1_64;
    }
}

int vg_run(const vg_case * c) {
    static ggml_backend_t be = nullptr;
    if (!be) {
        be = ggml_backend_cuda_init(0);
        if (!be) {
            fprintf(stderr, "ggml_backend_cuda_init failed\n");
            return 1;
        }
    }
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const ggml_type t = to_ggml(c->type);

    ggml_tensor * w   = nullptr;
    ggml_tensor * wg  = nullptr;
    ggml_tensor * x   = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out = nullptr;
    if (c->moe) {
        w  = ggml_new_tensor_3d(ctx, t, c->K, c->N, c->n_expert);
        x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c->K, c->nvec, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c->n_used, 1);
        ggml_tensor * up = ggml_mul_mat_id(ctx, w, x, ids);
        if (c->glu) {
            wg = ggml_new_tensor_3d(ctx, t, c->K, c->N, c->n_expert);
            ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, x, ids);
            out = ggml_swiglu_split(ctx, gate, up);
        } else {
            out = up;
        }
    } else {
        w = ggml_new_tensor_2d(ctx, t, c->K, c->N);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->K, 1);
        ggml_tensor * up = ggml_mul_mat(ctx, w, x);
        if (c->glu) {
            wg = ggml_new_tensor_2d(ctx, t, c->K, c->N);
            ggml_tensor * gate = ggml_mul_mat(ctx, wg, x);
            out = ggml_swiglu_split(ctx, gate, up);
        } else {
            out = up;
        }
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(w, c->w, 0, ggml_nbytes(w));
    if (wg) ggml_backend_tensor_set(wg, c->wg, 0, ggml_nbytes(wg));
    ggml_backend_tensor_set(x, c->x, 0, ggml_nbytes(x));
    if (ids) ggml_backend_tensor_set(ids, c->ids, 0, ggml_nbytes(ids));

    const int st = ggml_backend_graph_compute(be, gf);
    if (st != 0) {
        fprintf(stderr, "ggml_backend_graph_compute failed: %d\n", st);
        return 1;
    }
    ggml_backend_tensor_get(out, c->out, 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}
