# Correctness of the IQ2_S / IQ3_XXS Pascal kernels (test-backend-ops against the CPU)

One V100, `GGML_CUDA_MMVQ_PASCAL=2` (the Pascal kernels on any NVIDIA GPU), builds A (`6b880e344`) and B (`baf2e8ded`);
both gave the same counts. "Native" runs the sm_70 code; "6.1 code" runs the compute_61 PTX through
`CUDA_FORCE_PTX_JIT=1`, the code a P106 runs, at 1 to 4 tokens ([run/correctness.sh](run/correctness.sh)).

| Operation | Type | Native | 6.1 code, 1–4 tokens |
|---|---|---|---|
| MUL_MAT | IQ2_S | 73 / 73 | 40 / 40 |
| MUL_MAT | IQ3_XXS | 73 / 73 | 40 / 40 |
| MUL_MAT_ID | IQ2_S | 22 / 22 | 19 / 19 |
| MUL_MAT_ID | IQ3_XXS | 22 / 22 | 19 / 19 |
| MUL_MAT_VEC_FUSION (fused gate/up and GLU) | IQ2_S | 7 / 7 | 7 / 7 |
| MUL_MAT_VEC_FUSION | IQ3_XXS | 5 / 5 | 5 / 5 |
| MUL_MAT_ID_SHARED (routed + shared expert) | IQ2_S | 3 / 3 | 3 / 3 |
| MUL_MAT_ID_SHARED | IQ3_XXS | 3 / 3 | 3 / 3 |
| MUL_MAT | Q6_K (shares the staged read) | 106 / 106 | 64 / 64 |
| MUL_MAT_ID | Q6_K | 51 / 51 | 39 / 39 |
| MUL_MAT_VEC_FUSION | Q6_K | 9 / 9 | 9 / 9 |

- **compute-sanitizer memcheck** (6.1 code): 0 errors on expert gate/up (3072 → 1024, 1 token), expert down
  (1024 → 3072, 4 tokens), the fused gate/up (2 tokens) and a K = 512 product (super-blocks staged with 4-byte loads),
  for each type: 10 runs.
- The new cases (in `test-backend-ops`, Pascal section): 1 to 4 tokens at Ornith-1.5-122B's shapes (1024 × 3072,
  3072 × 1024, 4097 × 3072, 2048 × 512, 1024 × 2048), expert products of 8 of 32 experts, the fused gate/up, products whose
  activation blocks sum past FP16's range, routed + shared expert launches, and tokens that pick the same expert.
- Under the JIT, prompt-sized batches (9 to 512 tokens) run llama.cpp's own batched kernels, which this branch does
  not change: there Q6_K's MUL_MAT_ID cases fail and the IQ runs abort; they pass natively.
