# N2P/N3P correctness on a V100 (build 5fd8ffeec, 2026-10-27)

`run/correctness.sh` on one V100-PCIE-32GB: `test-backend-ops` (CUDA against the CPU) from the CUDA build of fork
branch `pascal-p106-v3-lowbit` at `5fd8ffeec` (tarball md5 `16b226cf28506009c46d4192fafe805d`). Each cell is the test
count that passed for that operation.

| Code, kernels | Operation | N2P | N3P | Q6_K (regression) |
|---|---|---|---|---|
| sm_70, Pascal kernels forced | MUL_MAT (1-128 tokens; MMQ from 9) | 40/40 | 40/40 | 106/106 |
| | MUL_MAT_ID | 24/24 | 24/24 | 51/51 |
| | MUL_MAT_VEC_FUSION (fused gate/up) | 5/5 | 5/5 | 9/9 |
| | MUL_MAT_ID_SHARED (shared expert in the routed launch) | 3/3 | 3/3 | |
| | GET_ROWS | 8/8 | 8/8 | |
| compute 6.1 (`CUDA_FORCE_PTX_JIT=1`), Pascal kernels forced | MUL_MAT, MUL_MAT_ID at 1-8 tokens | no count; the memcheck cases below pass | no count; the memcheck cases below pass | no count |
| | MUL_MAT_VEC_FUSION | 5/5 | 5/5 | 9/9 |
| | MUL_MAT_ID_SHARED | 3/3 | 3/3 | |
| | GET_ROWS | 8/8 | 8/8 | |
| | MUL_MAT, MUL_MAT_ID with MMQ (9 tokens and more) | not testable | not testable | not testable |
| sm_70, llama.cpp's matrix-vector kernel (`GGML_CUDA_MMVQ_PASCAL=0`) | MUL_MAT | 36/36 | 36/36 | |
| | MUL_MAT_ID | 20/20 | 20/20 | |

compute-sanitizer memcheck through the compute 6.1 code, 0 errors for each type: MUL_MAT_ID 3072 -> 1024 at 1 token and
1024 -> 3072 at 4 tokens (8 of 32 experts), the fused gate/up at 3072 -> 1024, MUL_MAT 512 -> 2048 at 1 token.

MMQ through the compute 6.1 code cannot be tested on a newer card: the host picks the MMQ tile configuration for the
card it sees (Volta), while the forced compute 6.1 code is built with Pascal's, so the kernels read out of bounds and
return wrong results for every type, untouched Q6_K included. This run's compute 6.1 pass included those cases (wrong
results for N2P, N3P and Q6_K at 9, 33 and 128 tokens, out-of-bounds reads under memcheck at 33 tokens), and both
operations stopped without a summary, so their 1-8-token cases have no count. `run/correctness.sh` now runs that pass at
1-8 tokens only and checks MMQ with memcheck natively.
The P106 checks its own MMQ in `run/rig_kernels.sh` step 2.
