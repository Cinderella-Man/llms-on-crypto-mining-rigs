2848 evaluations on cards 0 to 8: the model's graph 0: 170, 1: 170, 2: 170, 3: 170, 4: 170, 5: 170, 6: 170, 7: 170, 8: 170; the draft's graphs 1318 (card 4: 659, card 8: 659), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 85, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 85 | 3 | 81 | 1: 1, 4: 80 | 77 | 105.40 | 142.27 |
| r2 | 84 | 0 | 83 | 1: 1, 4: 82 | 79 | 105.41 | 141.79 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.53 | 19.53 | +0.01 |
| shared expert | 4.31 | 4.31 | -0.00 |
| hyper-connections (BF16) | 12.34 | 12.37 | +0.03 |
| router | 2.30 | 2.30 | -0.00 |
| attention and recurrent projections | 25.26 | 25.25 | -0.01 |
| Gated DeltaNet state and conv | 8.65 | 8.65 | -0.01 |
| attention (FLASH_ATTN_EXT) | 0.96 | 0.96 | -0.00 |
| other ops without weights | 6.13 | 6.11 | -0.02 |
| other weights | 0.74 | 0.74 | -0.00 |
| **sum** | **80.22** | **80.22** | **-0.01** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.88 | 0.88 | +0.00 |
| shared expert | 0.24 | 0.24 | +0.00 |
| hyper-connections (BF16) | 0.55 | 0.55 | +0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 0.82 | 0.82 | +0.00 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 3.12 | 3.12 | -0.00 |
| other ops without weights | 1.11 | 1.11 | +0.00 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **7.30** | **7.31** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.40 | 20.41 | +0.01 |
| shared expert | 4.55 | 4.55 | -0.00 |
| hyper-connections (BF16) | 12.89 | 12.93 | +0.04 |
| router | 2.40 | 2.40 | -0.00 |
| attention and recurrent projections | 26.08 | 26.07 | -0.01 |
| Gated DeltaNet state and conv | 8.90 | 8.90 | -0.00 |
| attention (FLASH_ATTN_EXT) | 1.05 | 1.05 | -0.00 |
| output layer | 3.12 | 3.12 | -0.00 |
| other ops without weights | 7.23 | 7.22 | -0.02 |
| other weights | 0.89 | 0.88 | -0.00 |
| **sum** | **87.52** | **87.53** | **+0.00** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.65 | 8.65 | -0.01 |
| GDN_CONV | 0.74 | 0.73 | -0.01 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | -0.00 |
| routed experts | 19.53 | 19.53 | +0.01 |
| shared expert, router, hyper-connections | 18.96 | 18.98 | +0.03 |
| everything else | 32.28 | 32.25 | -0.03 |
| **sum** | **80.22** | **80.22** | **-0.01** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 3.12 | 3.12 | -0.00 |
| the sampling chain after the product | 0.72 | 0.72 | -0.00 |
| the output's hyper-connection products | 0.10 | 0.10 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | -0.00 |
| routed experts | 0.88 | 0.88 | +0.00 |
| shared expert, router, hyper-connections | 0.89 | 0.89 | +0.01 |
| everything else | 1.32 | 1.33 | +0.01 |
| **sum** | **7.30** | **7.31** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 77 of 77, 204 launch groups, verify-r2 79 of 79, 204 launch groups)

| Position | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 3.46 | 3.48 | +0.01 |
| the head's product | 3.12 | 3.12 | -0.00 |
| after the head's product | 0.72 | 0.72 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| RMS_NORM | 0.98 (174) | 0.93 (174) | -0.05 |
| MUL | 0.82 (134) | 0.78 (134) | -0.04 |
| SIGMOID | 0.26 (93) | 0.30 (93) | +0.04 |
| SOFTPLUS | 0.15 (35) | 0.17 (35) | +0.02 |
| SCALE | 1.47 (197) | 1.49 (197) | +0.02 |
| SILU | 0.19 (93) | 0.21 (93) | +0.02 |
| CONT | 0.51 (31) | 0.52 (31) | +0.01 |
| MUL_MAT_ID | 19.53 (92) | 19.53 (92) | +0.01 |
| GET_ROWS | 0.19 (23) | 0.18 (23) | -0.01 |
| ARANGE | 0.02 (1) | 0.01 (1) | -0.01 |
| ADD | 0.39 (119) | 0.38 (119) | -0.01 |
| REPEAT | 0.04 (2) | 0.04 (2) | +0.01 |
| GDN_CONV | 0.74 (35) | 0.73 (35) | -0.01 |
| SET_ROWS | 0.15 (34) | 0.15 (34) | -0.01 |
| the other 18 ops | 54.80 | 54.79 | -0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| ARGMAX | 0.02 (8) | 0.03 (8) | +0.02 |
| GET_ROWS | 0.08 (27) | 0.09 (27) | +0.01 |
| SCALE | 0.09 (19) | 0.08 (19) | -0.01 |
| LOG | 0.01 (4) | 0.02 (4) | +0.01 |
| CLAMP | 0.01 (4) | 0.01 (4) | -0.01 |
| CUMSUM | 0.01 (4) | 0.00 (4) | -0.01 |
| MUL | 0.08 (5) | 0.08 (5) | +0.01 |
| SUM | 0.01 (4) | 0.01 (4) | -0.00 |
| SOFTPLUS | 0.02 (1) | 0.02 (1) | -0.00 |
| ARGSORT | 0.01 (4) | 0.00 (4) | -0.00 |
| SET_ROWS | 0.03 (8) | 0.03 (8) | -0.00 |
| MUL_MAT | 4.90 (36) | 4.91 (36) | +0.00 |
| GATED_DELTA_NET | 0.25 (1) | 0.25 (1) | +0.00 |
| SOFT_MAX | 0.03 (6) | 0.03 (6) | +0.00 |
| the other 19 ops | 1.75 | 1.75 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.000 |
| 115 | `ADD node_6216` | 0.000 |
| 116 | `SCALE result_norm` | 0.000 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 3.118 |
| 118 | `PAD node_6219` | 0.213 |
| 119 | `TOP_K top_k` | 0.074 |
| 120 | `GET_ROWS top_k_rows` | 0.004 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 122 | `GET_ROWS top_p_candidates` | 0.003 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 124 | `SOFT_MAX top_p_softmax` | 0.005 |
| 125 | `CUMSUM top_p_cdf` | 0.001 |
| 126 | `SCALE top_p_cdf_scaled` | 0.004 |
| 127 | `STEP top_p_mask` | 0.005 |
| 128 | `SUM top_p_index_f32` | 0.004 |
| 129 | `CLAMP node_6238` | 0.003 |
| 130 | `SCALE top_p_ones` | 0.005 |
| 131 | `CPY _(copy)` | 0.008 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.009 |
| 133 | `LOG top_p_bias` | 0.014 |
| 134 | `ADD top_p_logits` | 0.013 |
| 135 | `ARGMAX temp_max_idx` | 0.016 |
| 136 | `GET_ROWS node_6248` | 0.014 |
| 137 | `GET_ROWS node_6251` | 0.009 |
| 138 | `ARGMAX greedy_argmax` | 0.012 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.015 |
| 140 | `TOP_K top_k` | 0.078 |
| 141 | `GET_ROWS top_k_rows` | 0.003 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 143 | `GET_ROWS top_p_candidates` | 0.003 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 145 | `SOFT_MAX top_p_softmax` | 0.003 |
| 146 | `CUMSUM top_p_cdf` | 0.003 |
| 147 | `SCALE top_p_cdf_scaled` | 0.003 |
| 148 | `STEP top_p_mask` | 0.003 |
| 149 | `SUM top_p_index_f32` | 0.003 |
| 150 | `CLAMP node_6273` | 0.003 |
| 151 | `SCALE top_p_ones` | 0.003 |
| 152 | `CPY _(copy)` | 0.003 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 154 | `LOG top_p_bias` | 0.003 |
| 155 | `ADD top_p_logits` | 0.003 |
| 156 | `ARGMAX temp_max_idx` | 0.003 |
| 157 | `GET_ROWS node_6283` | 0.003 |
| 158 | `GET_ROWS node_6286` | 0.003 |
| 159 | `ARGMAX greedy_argmax` | 0.002 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.001 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.000 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | -0.000 |
| 169 | `STEP top_p_mask` | 0.000 |
| 170 | `SUM top_p_index_f32` | 0.000 |
| 171 | `CLAMP node_6308` | 0.000 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.000 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 175 | `LOG top_p_bias` | -0.000 |
| 176 | `ADD top_p_logits` | 0.000 |
| 177 | `ARGMAX temp_max_idx` | 0.000 |
| 178 | `GET_ROWS node_6318` | 0.001 |
| 179 | `GET_ROWS node_6321` | 0.000 |
| 180 | `ARGMAX greedy_argmax` | -0.000 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.001 |
| 182 | `TOP_K top_k` | 0.071 |
| 183 | `GET_ROWS top_k_rows` | 0.000 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 185 | `GET_ROWS top_p_candidates` | 0.000 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 187 | `SOFT_MAX top_p_softmax` | -0.000 |
| 188 | `CUMSUM top_p_cdf` | 0.000 |
| 189 | `SCALE top_p_cdf_scaled` | -0.000 |
| 190 | `STEP top_p_mask` | -0.000 |
| 191 | `SUM top_p_index_f32` | -0.000 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | -0.000 |
| 194 | `CPY _(copy)` | -0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 196 | `LOG top_p_bias` | 0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.000 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 10.81 | 10.83 | +0.02 |
| 1 | 10.77 | 10.77 | -0.00 |
| 2 | 10.67 | 10.66 | -0.01 |
| 3 | 10.74 | 10.73 | -0.01 |
| 4 | 8.41 | 8.42 | +0.01 |
| 5 | 10.19 | 10.19 | -0.00 |
| 6 | 8.65 | 8.63 | -0.02 |
| 7 | 9.98 | 9.99 | +0.01 |
| 8 | 7.30 | 7.31 | +0.01 |
| **all** | **87.52** | **87.53** | **+0.00** |

