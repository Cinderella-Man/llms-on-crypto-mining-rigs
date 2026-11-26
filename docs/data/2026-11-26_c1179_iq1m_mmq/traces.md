# Which kernels one IQ1_M expert product launches (C1179)

One `test-backend-ops` case under `nsys profile -t cuda` ([run/checks.sh](run/checks.sh), step `trace`):
`MUL_MAT_ID(type_a=iq1_m, n_mats=32, n_used=10, b=0, m=640, n=512, k=2560)`, Qwen3.8-Flash-Next's routed gate/up
shape at 512 tokens, on one V100. Kernel names and counts only; nothing here is a timing. Each run passed its case
(1/1). The CUDA call counts cover the whole test run, so the test's own copies back to the host are in them.

| Run | Build | `GGML_CUDA_MMQ_IQ1_M` | Kernels (launches) | Kernel launches | `cudaStreamSynchronize` |
|---|---|---|---|---|---|
| `a_trace_off` | A (sm_70 machine code) | unset | `dequantize_block_iq1_m<half>` (32), `convert_unary_cont_vec4<float, half>` (32), cuBLAS `volta_s884gemm_fp16_128x64_ldg8_tn` (32) and `splitKreduce_kernel` (32), `k_get_rows_float_vec` (2), `k_skip_ids_fix` (1), `k_skip_ids_zero_rows` (1) | 132 | 549 |
| `a_trace_1` | A | `1` | the same as unset: a V100 is not compute 6.1 | 132 | 549 |
| `a_trace_on` | A | `2` | `mul_mat_q<IQ1_M, 80, ...>` (1), `quantize_mmq_q8_1<D4>` (1), `mm_ids_helper<10>` (1), `k_skip_ids_fix` (1), `k_skip_ids_zero_rows` (1) | 5 | 547 |
| `b_trace_off` | B (compute-6.1 PTX, MMQ forced) | unset | as `a_trace_off` | 132 | 549 |
| `b_trace_on` | B | `2` | `mul_mat_q<IQ1_M, 64, ...>` (1), `quantize_mmq_q8_1<D4>` (1), `mm_ids_helper<10>` (1), `k_skip_ids_fix` (1), `k_skip_ids_zero_rows` (1) | 5 | 547 |

- **Unset**, the product takes `ggml_cuda_mul_mat_id`'s generic path: two more stream synchronizations (549 against
  547) and one product per used expert, 32 here, each a dequantization and a cuBLAS product. On a V100 that product is
  FP16 on tensor cores; a P106 has no FP16 tensor cores and takes F32 and SGEMM there (C1139).
- **`=2`**, one `mul_mat_q` kernel for IQ1_M covers every expert, after the activations' Q8_1 quantization and the
  expert-id tile list, with no synchronization of its own. Build A takes the tile of the newer cards' table (80
  columns), build B the P106's DP4A table's 64-column tile.
