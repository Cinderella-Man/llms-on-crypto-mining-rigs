| Tensor | rows x n | Hessian (rows) | as in the mix | RTN Q5_0 (imatrix) | = mix | GPTQ Q5_0 | GPTQ / RTN |
|---|---|---|---|---|---|---|---|
| blk.0.attn_output.weight | 2048 x 4096 | kqv_out-0 (255500) | 2.25e-05 (Q8_0) | 1.27e-03 |  | 3.75e-04 | 0.296 |
| blk.1.attn_output.weight | 2048 x 4096 | kqv_out-1 (255500) | 2.13e-05 (Q8_0) | 1.19e-03 |  | 2.50e-04 | 0.210 |
| blk.2.attn_output.weight | 2048 x 4096 | kqv_out-2 (255500) | 3.28e-05 (Q8_0) | 1.76e-03 |  | 2.50e-04 | 0.142 |
| blk.3.attn_output.weight | 2048 x 4096 | kqv_out-3 (255500) | 2.96e-05 (Q8_0) | 1.54e-03 |  | 4.89e-04 | 0.317 |
| blk.4.attn_output.weight | 2048 x 4096 | kqv_out-4 (255500) | 2.32e-05 (Q8_0) | 1.25e-03 |  | 4.26e-04 | 0.342 |
| blk.5.attn_output.weight | 2048 x 4096 | kqv_out-5 (255500) | 2.28e-05 (Q8_0) | 1.20e-03 |  | 1.56e-04 | 0.130 |
| blk.6.attn_output.weight | 2048 x 4096 | kqv_out-6 (255500) | 2.86e-05 (Q8_0) | 1.50e-03 |  | 5.21e-04 | 0.347 |
| blk.7.attn_output.weight | 2048 x 4096 | kqv_out-7 (255500) | 2.85e-05 (Q8_0) | 1.50e-03 |  | 5.44e-04 | 0.363 |
| blk.8.attn_output.weight | 2048 x 4096 | kqv_out-8 (255500) | 2.72e-05 (Q8_0) | 1.45e-03 |  | 4.81e-04 | 0.331 |

group o08: GPTQ / RTN error, output-weighted over the tensors: 0.294
donor: donor-o08.gguf (9 tensors, Q5_0)
