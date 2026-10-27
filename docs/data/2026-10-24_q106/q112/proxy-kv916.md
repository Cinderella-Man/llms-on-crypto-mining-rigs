| Tensor | rows x n | Hessian (rows) | as in the mix | RTN Q5_0 (imatrix) | GPTQ Q5_0 | GPTQ / RTN |
|---|---|---|---|---|---|---|
| blk.9.attn_k.weight | 512 x 2048 | attn_norm-9 (255500) | 1.27e-05 (Q8_0) | 6.91e-04 | 3.91e-04 | 0.566 |
| blk.9.attn_v.weight | 512 x 2048 | attn_norm-9 (255500) | 2.61e-05 (Q8_0) | 1.49e-03 | 8.21e-04 | 0.552 |
| blk.10.attn_k.weight | 512 x 2048 | attn_norm-10 (255500) | 5.82e-06 (Q8_0) | 3.18e-04 | 1.78e-04 | 0.562 |
| blk.10.attn_v.weight | 512 x 2048 | attn_norm-10 (255500) | 2.19e-05 (Q8_0) | 1.24e-03 | 6.88e-04 | 0.554 |
| blk.11.attn_k.weight | 512 x 2048 | attn_norm-11 (255500) | 5.57e-06 (Q8_0) | 3.00e-04 | 1.67e-04 | 0.558 |
| blk.11.attn_v.weight | 512 x 2048 | attn_norm-11 (255500) | 2.13e-05 (Q8_0) | 1.20e-03 | 6.55e-04 | 0.544 |
| blk.12.attn_k.weight | 512 x 2048 | attn_norm-12 (255500) | 6.14e-06 (Q8_0) | 3.41e-04 | 1.85e-04 | 0.544 |
| blk.12.attn_v.weight | 512 x 2048 | attn_norm-12 (255500) | 2.05e-05 (Q8_0) | 1.17e-03 | 6.37e-04 | 0.545 |
| blk.13.attn_k.weight | 512 x 2048 | attn_norm-13 (255500) | 1.15e-05 (Q8_0) | 6.60e-04 | 3.51e-04 | 0.531 |
| blk.13.attn_v.weight | 512 x 2048 | attn_norm-13 (255500) | 2.20e-05 (Q8_0) | 1.26e-03 | 6.70e-04 | 0.531 |
| blk.14.attn_k.weight | 512 x 2048 | attn_norm-14 (255500) | 8.18e-06 (Q8_0) | 4.49e-04 | 2.46e-04 | 0.548 |
| blk.14.attn_v.weight | 512 x 2048 | attn_norm-14 (255500) | 1.91e-05 (Q8_0) | 1.10e-03 | 5.88e-04 | 0.536 |
| blk.15.attn_k.weight | 512 x 2048 | attn_norm-15 (255500) | 7.68e-06 (Q8_0) | 4.24e-04 | 2.37e-04 | 0.558 |
| blk.15.attn_v.weight | 512 x 2048 | attn_norm-15 (255500) | 2.25e-05 (Q8_0) | 1.28e-03 | 7.06e-04 | 0.550 |
| blk.16.attn_k.weight | 512 x 2048 | attn_norm-16 (255500) | 6.79e-06 (Q8_0) | 3.74e-04 | 2.06e-04 | 0.551 |
| blk.16.attn_v.weight | 512 x 2048 | attn_norm-16 (255500) | 2.18e-05 (Q8_0) | 1.24e-03 | 6.83e-04 | 0.549 |

group kv916: GPTQ / RTN error, output-weighted over the tensors: 0.546
donor: donor-kv916.gguf (16 tensors, Q5_0)
