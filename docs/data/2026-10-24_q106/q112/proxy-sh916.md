| Tensor | rows x n | Hessian (rows) | as in the mix | RTN Q5_0 (imatrix) | GPTQ Q5_0 | GPTQ / RTN |
|---|---|---|---|---|---|---|
| blk.9.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-9 (255500) | 7.23e-06 (Q8_0) | 3.79e-04 | 2.11e-04 | 0.556 |
| blk.9.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-9 (255500) | 1.80e-05 (Q8_0) | 1.01e-03 | 5.47e-04 | 0.542 |
| blk.9.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-9 (255500) | 2.79e-05 (Q8_0) | 1.47e-03 | 9.97e-04 | 0.676 |
| blk.10.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-10 (255500) | 6.32e-06 (Q8_0) | 3.40e-04 | 1.86e-04 | 0.549 |
| blk.10.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-10 (255500) | 1.68e-05 (Q8_0) | 9.26e-04 | 5.14e-04 | 0.555 |
| blk.10.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-10 (255500) | 2.50e-05 (Q8_0) | 1.35e-03 | 9.20e-04 | 0.683 |
| blk.11.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-11 (255500) | 6.75e-06 (Q8_0) | 3.51e-04 | 1.96e-04 | 0.560 |
| blk.11.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-11 (255500) | 1.74e-05 (Q8_0) | 9.79e-04 | 5.37e-04 | 0.549 |
| blk.11.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-11 (255500) | 1.92e-05 (Q8_0) | 1.07e-03 | 7.23e-04 | 0.678 |
| blk.12.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-12 (255500) | 7.74e-06 (Q8_0) | 3.96e-04 | 2.17e-04 | 0.549 |
| blk.12.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-12 (255500) | 1.29e-05 (Q8_0) | 7.15e-04 | 3.92e-04 | 0.548 |
| blk.12.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-12 (255500) | 2.48e-05 (Q8_0) | 1.40e-03 | 8.48e-04 | 0.606 |
| blk.13.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-13 (255500) | 6.84e-06 (Q8_0) | 3.45e-04 | 1.94e-04 | 0.563 |
| blk.13.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-13 (255500) | 1.78e-05 (Q8_0) | 9.91e-04 | 5.47e-04 | 0.553 |
| blk.13.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-13 (255500) | 2.70e-05 (Q8_0) | 1.50e-03 | 1.03e-03 | 0.688 |
| blk.14.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-14 (255500) | 7.85e-06 (Q8_0) | 4.04e-04 | 2.26e-04 | 0.559 |
| blk.14.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-14 (255500) | 1.49e-05 (Q8_0) | 8.26e-04 | 4.69e-04 | 0.568 |
| blk.14.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-14 (255500) | 2.53e-05 (Q8_0) | 1.39e-03 | 9.46e-04 | 0.683 |
| blk.15.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-15 (255500) | 7.27e-06 (Q8_0) | 3.64e-04 | 2.07e-04 | 0.568 |
| blk.15.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-15 (255500) | 1.56e-05 (Q8_0) | 8.90e-04 | 4.85e-04 | 0.545 |
| blk.15.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-15 (255500) | 2.21e-05 (Q8_0) | 1.20e-03 | 8.26e-04 | 0.687 |
| blk.16.ffn_gate_shexp.weight | 1024 x 2048 | ffn_norm-16 (255500) | 8.41e-06 (Q8_0) | 4.23e-04 | 2.30e-04 | 0.543 |
| blk.16.ffn_up_shexp.weight | 1024 x 2048 | ffn_norm-16 (255500) | 1.36e-05 (Q8_0) | 7.61e-04 | 4.17e-04 | 0.548 |
| blk.16.ffn_down_shexp.weight | 2048 x 1024 | ffn_swiglu-16 (255500) | 2.19e-05 (Q8_0) | 1.19e-03 | 7.25e-04 | 0.611 |

group sh916: GPTQ / RTN error, output-weighted over the tensors: 0.552
donor: donor-sh916.gguf (24 tensors, Q5_0)
