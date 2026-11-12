2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 87, 85; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 87 | 3 | 83 | 4: 83 | 80 | 103.26 | 140.09 |
| r2 | 85 | 0 | 84 | 4: 84 | 81 | 103.23 | 139.55 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.59 | 19.63 | +0.04 |
| shared expert | 3.68 | 3.69 | +0.01 |
| hyper-connections (BF16) | 12.34 | 12.31 | -0.03 |
| router | 3.90 | 3.90 | +0.00 |
| attention and recurrent projections | 22.72 | 22.76 | +0.04 |
| Gated DeltaNet state and conv | 8.67 | 8.69 | +0.01 |
| attention (FLASH_ATTN_EXT) | 0.94 | 0.94 | -0.00 |
| other ops without weights | 6.03 | 5.99 | -0.05 |
| other weights | 0.74 | 0.74 | -0.00 |
| **sum** | **78.62** | **78.64** | **+0.02** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert | 0.18 | 0.18 | +0.00 |
| hyper-connections (BF16) | 0.55 | 0.55 | +0.00 |
| router | 0.17 | 0.17 | -0.00 |
| attention and recurrent projections | 0.80 | 0.78 | -0.02 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 3.17 | 3.17 | +0.00 |
| other ops without weights | 1.14 | 1.15 | +0.02 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **7.35** | **7.36** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.45 | 20.50 | +0.05 |
| shared expert | 3.86 | 3.87 | +0.01 |
| hyper-connections (BF16) | 12.89 | 12.87 | -0.03 |
| router | 4.07 | 4.07 | +0.00 |
| attention and recurrent projections | 23.52 | 23.54 | +0.02 |
| Gated DeltaNet state and conv | 8.92 | 8.94 | +0.01 |
| attention (FLASH_ATTN_EXT) | 1.03 | 1.02 | -0.00 |
| output layer | 3.17 | 3.17 | +0.00 |
| other ops without weights | 7.17 | 7.14 | -0.03 |
| other weights | 0.89 | 0.88 | -0.00 |
| **sum** | **85.97** | **86.00** | **+0.03** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.67 | 8.69 | +0.01 |
| GDN_CONV | 0.72 | 0.73 | +0.00 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.08 | +0.01 |
| routed experts | 19.59 | 19.63 | +0.04 |
| shared expert, router, hyper-connections | 19.93 | 19.91 | -0.02 |
| everything else | 29.63 | 29.61 | -0.03 |
| **sum** | **78.62** | **78.64** | **+0.02** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 3.17 | 3.17 | +0.00 |
| the sampling chain after the product | 0.73 | 0.74 | +0.01 |
| the output's hyper-connection products | 0.10 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert, router, hyper-connections | 0.90 | 0.90 | +0.00 |
| everything else | 1.31 | 1.31 | -0.00 |
| **sum** | **7.35** | **7.36** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 80 of 80, 200 launch groups, verify-r2 81 of 81, 200 launch groups)

| Position | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 3.46 | 3.46 | +0.00 |
| the head's product | 3.17 | 3.17 | +0.00 |
| after the head's product | 0.73 | 0.74 | +0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.87 (134) | 0.77 (134) | -0.10 |
| MUL_MAT_ID | 19.59 (92) | 19.63 (92) | +0.04 |
| RMS_NORM | 0.90 (174) | 0.93 (174) | +0.03 |
| MUL_MAT | 41.63 (699) | 41.66 (699) | +0.03 |
| SCALE | 1.46 (197) | 1.44 (197) | -0.02 |
| SOFTPLUS | 0.21 (35) | 0.24 (35) | +0.02 |
| SIGMOID | 0.22 (93) | 0.24 (93) | +0.02 |
| ADD | 0.40 (119) | 0.39 (119) | -0.01 |
| GATED_DELTA_NET | 8.67 (35) | 8.69 (35) | +0.01 |
| SET_ROWS | 0.13 (34) | 0.14 (34) | +0.01 |
| SOFT_MAX | 0.52 (46) | 0.53 (46) | +0.01 |
| GET_ROWS | 0.18 (23) | 0.17 (23) | -0.01 |
| SILU | 0.17 (93) | 0.16 (93) | -0.01 |
| CONT | 0.54 (31) | 0.54 (31) | +0.01 |
| the other 18 ops | 3.12 | 3.11 | -0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| ADD | 0.03 (13) | 0.06 (13) | +0.03 |
| MUL | 0.08 (5) | 0.05 (5) | -0.03 |
| SOFTPLUS | 0.06 (1) | 0.07 (1) | +0.02 |
| MUL_MAT | 4.94 (34) | 4.93 (34) | -0.01 |
| SCALE | 0.10 (19) | 0.09 (19) | -0.01 |
| SET_ROWS | 0.03 (8) | 0.02 (8) | -0.01 |
| SUM | 0.01 (4) | 0.02 (4) | +0.01 |
| STEP | 0.01 (4) | 0.01 (4) | -0.01 |
| CLAMP | 0.01 (4) | 0.01 (4) | +0.01 |
| SOFT_MAX | 0.03 (6) | 0.04 (6) | +0.00 |
| ARGSORT | 0.01 (4) | 0.01 (4) | +0.00 |
| GET_ROWS | 0.08 (27) | 0.08 (27) | +0.00 |
| ARGMAX | 0.01 (8) | 0.01 (8) | -0.00 |
| SIGMOID | 0.01 (5) | 0.01 (5) | -0.00 |
| the other 18 ops | 1.94 | 1.94 | +0.01 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 110 | `ADD node_6214` | 0.002 |
| 111 | `ADD node_6216` | 0.001 |
| 112 | `SCALE result_norm` | 0.003 |
| 113 | `MUL_MAT output:q5_K:2560x248320` | 3.166 |
| 114 | `PAD node_6219` | 0.216 |
| 115 | `TOP_K top_k` | 0.077 |
| 116 | `GET_ROWS top_k_rows` | 0.007 |
| 117 | `ARGSORT top_p_sorted_idx` | 0.008 |
| 118 | `GET_ROWS top_p_candidates` | 0.007 |
| 119 | `GET_ROWS top_p_sorted_logits` | 0.011 |
| 120 | `SOFT_MAX top_p_softmax` | 0.013 |
| 121 | `CUMSUM top_p_cdf` | 0.012 |
| 122 | `SCALE top_p_cdf_scaled` | 0.007 |
| 123 | `STEP top_p_mask` | 0.005 |
| 124 | `SUM top_p_index_f32` | 0.018 |
| 125 | `CLAMP node_6238` | 0.011 |
| 126 | `SCALE top_p_ones` | 0.010 |
| 127 | `CPY _(copy)` | 0.012 |
| 128 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.004 |
| 129 | `LOG top_p_bias` | 0.007 |
| 130 | `ADD top_p_logits` | 0.008 |
| 131 | `ARGMAX temp_max_idx` | 0.003 |
| 132 | `GET_ROWS node_6248` | 0.003 |
| 133 | `GET_ROWS node_6251` | 0.003 |
| 134 | `ARGMAX greedy_argmax` | 0.003 |
| 135 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 136 | `TOP_K top_k` | 0.073 |
| 137 | `GET_ROWS top_k_rows` | 0.003 |
| 138 | `ARGSORT top_p_sorted_idx` | 0.002 |
| 139 | `GET_ROWS top_p_candidates` | 0.003 |
| 140 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 141 | `SOFT_MAX top_p_softmax` | 0.003 |
| 142 | `CUMSUM top_p_cdf` | 0.002 |
| 143 | `SCALE top_p_cdf_scaled` | 0.003 |
| 144 | `STEP top_p_mask` | 0.003 |
| 145 | `SUM top_p_index_f32` | 0.003 |
| 146 | `CLAMP node_6273` | 0.002 |
| 147 | `SCALE top_p_ones` | 0.003 |
| 148 | `CPY _(copy)` | 0.003 |
| 149 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 150 | `LOG top_p_bias` | 0.003 |
| 151 | `ADD top_p_logits` | 0.003 |
| 152 | `ARGMAX temp_max_idx` | 0.003 |
| 153 | `GET_ROWS node_6283` | 0.004 |
| 154 | `GET_ROWS node_6286` | 0.003 |
| 155 | `ARGMAX greedy_argmax` | 0.002 |
| 156 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 157 | `TOP_K top_k` | 0.071 |
| 158 | `GET_ROWS top_k_rows` | 0.001 |
| 159 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 160 | `GET_ROWS top_p_candidates` | 0.001 |
| 161 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 162 | `SOFT_MAX top_p_softmax` | -0.000 |
| 163 | `CUMSUM top_p_cdf` | 0.000 |
| 164 | `SCALE top_p_cdf_scaled` | 0.000 |
| 165 | `STEP top_p_mask` | 0.000 |
| 166 | `SUM top_p_index_f32` | 0.000 |
| 167 | `CLAMP node_6308` | -0.000 |
| 168 | `SCALE top_p_ones` | 0.000 |
| 169 | `CPY _(copy)` | 0.000 |
| 170 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 171 | `LOG top_p_bias` | 0.000 |
| 172 | `ADD top_p_logits` | 0.000 |
| 173 | `ARGMAX temp_max_idx` | -0.000 |
| 174 | `GET_ROWS node_6318` | 0.001 |
| 175 | `GET_ROWS node_6321` | 0.001 |
| 176 | `ARGMAX greedy_argmax` | 0.000 |
| 177 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 178 | `TOP_K top_k` | 0.071 |
| 179 | `GET_ROWS top_k_rows` | 0.001 |
| 180 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 181 | `GET_ROWS top_p_candidates` | 0.000 |
| 182 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 183 | `SOFT_MAX top_p_softmax` | 0.000 |
| 184 | `CUMSUM top_p_cdf` | 0.000 |
| 185 | `SCALE top_p_cdf_scaled` | 0.000 |
| 186 | `STEP top_p_mask` | -0.000 |
| 187 | `SUM top_p_index_f32` | 0.000 |
| 188 | `CLAMP node_6343` | -0.000 |
| 189 | `SCALE top_p_ones` | 0.000 |
| 190 | `CPY _(copy)` | 0.000 |
| 191 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 192 | `LOG top_p_bias` | -0.000 |
| 193 | `ADD top_p_logits` | 0.000 |
| 194 | `ARGMAX temp_max_idx` | -0.000 |
| 195 | `GET_ROWS node_6353` | 0.001 |
| 196 | `GET_ROWS node_6356` | 0.001 |
| 197 | `ARGMAX greedy_argmax` | -0.000 |
| 198 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 199 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 10.95 | 11.00 | +0.04 |
| 1 | 10.43 | 10.41 | -0.02 |
| 2 | 10.29 | 10.31 | +0.02 |
| 3 | 10.39 | 10.40 | +0.01 |
| 4 | 8.24 | 8.23 | -0.01 |
| 5 | 10.03 | 10.03 | -0.00 |
| 6 | 8.50 | 8.49 | -0.00 |
| 7 | 9.79 | 9.78 | -0.01 |
| 8 | 7.35 | 7.36 | +0.01 |
| **all** | **85.97** | **86.00** | **+0.03** |

