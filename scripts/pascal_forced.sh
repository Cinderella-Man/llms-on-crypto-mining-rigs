# shellcheck shell=bash
# The fork's compute-6.1-only card paths forced on for any NVIDIA card, with a P106's launch shapes. Sourced by the
# GPU-check drivers that run the rig build on newer cards (its compute 6.1 PTX through the driver's JIT), so that a check
# there computes with the kernels a P106 computes with:
#   . "$REPO/scripts/pascal_forced.sh"
#   env $PASCAL_FORCED test-backend-ops test -b CUDA0 ...
#   env $PRESET_ENV $(pascal_force_preset "$PRESET_ENV") llama-server ...
#
# Each of these paths reads a switch whose 1 means "compute capability 6.1 only" and 2 "any NVIDIA card" (the
# GGML_CUDA_CC_DP4A gates in ggml/src/ggml-cuda/). At 1 a newer card takes llama.cpp's own kernel in its place, so a
# check there without these settings checks llama.cpp's kernel, not the fork's (C947, C968, C991). The rig build holds
# only compute 6.1 code, so llama.cpp's own choices by compiled architecture (no tensor-core kernels) are a P106's on
# any card already. Two choices read the card itself instead: cuBLAS's arithmetic (PASCAL_CUBLAS below) and the
# 64-token limit above which a dense quantized product leaves MMQ for cuBLAS on a card with FP16 tensor cores
# (ggml_cuda_should_use_mmq; registry Q1901, not forced yet).
#
# PASCAL_CUBLAS: cuBLAS's matrix products computed as on a P106: in F32, without TF32 (C1108).
#   NVIDIA_TF32_OVERRIDE=0: ggml's CUDA backend sets CUBLAS_TF32_TENSOR_OP_MATH on every cuBLAS handle
#     (ggml/src/ggml-cuda/common.cuh), so on an Ampere or newer card cuBLAS rounds the inputs of each product it
#     computes in F32 to TF32's 10-bit mantissa (an F32 weight's product with a prompt of more than 8 tokens, for one);
#     a P106 has no TF32. This NVIDIA switch overrides the library's own setting.
#   GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32: ggml_cuda_mul_mat_cublas computes F16 and quantized sources in F16 on a card with
#     fast F16 (fast_fp16_hardware_available, false for compute 6.1 only), in F32 on a P106.
#   With TF32 on, a random model's card-against-CPU nmse sits near 1e-7 instead of a P106's 1e-14 and hides a change a
#   P106 shows (C1039); with TF32 off alone, 13 of 148 random models stay more than 10 times above the P106's number
#   (kimi-k3 8,990 times); with both, all 148 are within 4 times of it (C1108). It describes the card, not a kernel choice, so every run of
#   the rig build on a newer card passes it, a run with no switch too (env $PASCAL_CUBLAS ...), and both forced sets
#   below carry it.
PASCAL_CUBLAS="NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32"
# PASCAL_FORCED_MMV: the matrix-vector kernels for products of 1 to 8 tokens (mul_mat_vec_q_pascal, its multi-token,
#   read-once, k-column and expert kernels, the BF16 and F16 kernels) at a P106's shapes: 5 SMs, 96 KB of shared memory
#   per SM, and cuBLAS as on a P106 (PASCAL_CUBLAS, since 2026-11-26). The kernels are the set C953's and C968's
#   drivers spelled out until 2026-11-25.
PASCAL_FORCED_MMV="$PASCAL_CUBLAS GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 \
GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
# PASCAL_FORCED_PATHS: the other paths a P106 takes with every switch at its default:
#   GGML_CUDA_ATTN_BLOCK=2         one token's attention block (Q/K/V products, norms, RoPE, cache writes) in two launches
#   GGML_CUDA_PERSISTENT_MOE=2     a MoE layer at one token as one persistent kernel (k_moe_layer)
#   GGML_CUDA_FATTN_TILE_PASCAL=2  the Pascal flash-attention tile kernel for prompt attention (flash_attn_tile_pascal)
PASCAL_FORCED_PATHS="GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_TILE_PASCAL=2"
# PASCAL_FORCED_GRAPH: the graph choice llama.cpp makes when every card is compute 6.1 (the backend's COMPUTE_6_1
#   feature): the MoE down-projection rescale dropped at one token, since the Pascal kernels sum in FP32. Without it a
#   model's MoE layers keep the rescale on a newer card and the persistent MoE kernel leaves them to the per-op path.
#   Read when a model's graph is built; test-backend-ops builds its own graphs and does not read it.
PASCAL_FORCED_GRAPH="LLAMA_NO_MOE_DOWN_RESCALE=tg"
# PASCAL_FORCED: all three (PASCAL_CUBLAS with the first), what a driver passes for "as a P106 with no preset switch"
PASCAL_FORCED="$PASCAL_FORCED_MMV $PASCAL_FORCED_PATHS $PASCAL_FORCED_GRAPH"
# PASCAL_GATES: every switch with such a 1/2 gate. The last four are off by default on a P106 too; a preset that sets
# one to 1 turns it on for the P106 only, so on a newer card it needs its own 2 (pascal_raise_gates):
#   GGML_CUDA_MMVF_F32_MT          F32 products of 4 to 8 tokens on mul_mat_vec_f (the Flash-Next preset sets 1)
#   GGML_CUDA_MMQ_KQ_FP32SCALE     the K-quant prompt products' scales in FP32
#   GGML_CUDA_FATTN_PASCAL         the older Pascal decode attention kernel
#   GGML_CUDA_FATTN_SPARSE_PASCAL  sparse decode attention over a QSA layer's kept cells (the Flash-Next preset sets 1);
#                                  its 1 means "a build without tensor-core code", which the rig build is on any card,
#                                  and 2 keeps it on in a build that has such code
PASCAL_GATES="GGML_CUDA_MMVQ_PASCAL GGML_CUDA_MMVF_PASCAL GGML_CUDA_ATTN_BLOCK GGML_CUDA_PERSISTENT_MOE \
GGML_CUDA_FATTN_TILE_PASCAL GGML_CUDA_MMVF_F32_MT GGML_CUDA_MMQ_KQ_FP32SCALE GGML_CUDA_FATTN_PASCAL \
GGML_CUDA_FATTN_SPARSE_PASCAL"

# pascal_raise_gates "<NAME=value ...>": each gate the list sets to 1, at 2 ("GGML_CUDA_MMVF_F32_MT=2" for the
# Flash-Next preset's list); nothing for a list that sets none.
pascal_raise_gates() {
    local kv g out=""
    for kv in $1; do
        for g in $PASCAL_GATES; do
            [ "$kv" = "$g=1" ] && out+="$g=2 "
        done
    done
    echo "${out% }"
}

# pascal_force_preset "<NAME=value ...>": PASCAL_FORCED without the names the list sets, then pascal_raise_gates of
# the list, so that `env <list> $(pascal_force_preset "<list>")` runs the list as a P106 runs it (later settings win in
# env): the list's own values stand (a gate at 0 stays 0, its launch shapes and LLAMA_NO_MOE_DOWN_RESCALE are its own),
# except a gate at 1, which becomes 2.
pascal_force_preset() {
    local kv f base="" set=" "
    for kv in $1; do set+="${kv%%=*} "; done
    for f in $PASCAL_FORCED; do
        [[ $set == *" ${f%%=*} "* ]] || base+="$f "
    done
    echo "$base$(pascal_raise_gates "$1")" | sed 's/ *$//'
}
