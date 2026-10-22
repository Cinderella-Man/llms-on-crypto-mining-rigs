// Interface between the benchmark's kernels (verify_main.cu) and llama.cpp's ggml (verify_ggml.cpp).
#pragma once

struct vg_case {
    int type;          // 0 q4_0, 1 q4_1, 2 q8_0, 3 q4_1_64 (mb::wtype)
    int K, N;          // input width, rows per matrix
    int n_expert;      // experts in the weight tensor (MUL_MAT_ID)
    int n_used;        // experts selected (MUL_MAT_ID)
    int nvec;          // activation vectors: 1, or n_used for a per-expert input (down projection)
    int moe, glu;      // MUL_MAT_ID? fused gate/up/SwiGLU?
    const void  * w;   // up (or plain) weights, host
    const void  * wg;  // gate weights, host (glu only)
    const float * x;   // activations [nvec][K], host
    const int   * ids; // [n_used], host (moe only)
    float       * out; // [n_used or 1][N], host
};

int vg_run(const vg_case * c);
