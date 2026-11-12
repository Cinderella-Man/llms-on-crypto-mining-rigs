2848 evaluations on cards 0 to 8: the model's graph 0: 170, 1: 170, 2: 170, 3: 170, 4: 170, 5: 170, 6: 170, 7: 170, 8: 170; the draft's graphs 1318 (card 4: 659, card 8: 659), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 85, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 85 | 3 | 81 | 1: 1, 4: 80 | 77 | 105.58 | 142.47 |
| r2 | 84 | 0 | 83 | 1: 1, 4: 82 | 79 | 105.77 | 142.13 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.62 | 19.65 | +0.03 |
| shared expert | 4.34 | 4.35 | +0.01 |
| hyper-connections (BF16) | 12.40 | 12.43 | +0.03 |
| router | 2.31 | 2.30 | -0.01 |
| attention and recurrent projections | 25.26 | 25.22 | -0.04 |
| Gated DeltaNet state and conv | 8.68 | 8.70 | +0.02 |
| attention (FLASH_ATTN_EXT) | 0.97 | 0.96 | -0.00 |
| other ops without weights | 6.10 | 6.15 | +0.05 |
| other weights | 0.74 | 0.74 | +0.00 |
| **sum** | **80.42** | **80.51** | **+0.09** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.88 | 0.89 | +0.00 |
| shared expert | 0.24 | 0.24 | +0.00 |
| hyper-connections (BF16) | 0.55 | 0.55 | +0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 0.82 | 0.81 | -0.01 |
| Gated DeltaNet state and conv | 0.25 | 0.26 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | +0.00 |
| output layer | 3.15 | 3.16 | +0.01 |
| other ops without weights | 1.11 | 1.12 | +0.02 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **7.34** | **7.37** | **+0.02** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.50 | 20.54 | +0.03 |
| shared expert | 4.58 | 4.59 | +0.01 |
| hyper-connections (BF16) | 12.95 | 12.98 | +0.03 |
| router | 2.41 | 2.41 | -0.01 |
| attention and recurrent projections | 26.08 | 26.03 | -0.05 |
| Gated DeltaNet state and conv | 8.93 | 8.96 | +0.03 |
| attention (FLASH_ATTN_EXT) | 1.06 | 1.05 | -0.00 |
| output layer | 3.15 | 3.16 | +0.01 |
| other ops without weights | 7.21 | 7.28 | +0.07 |
| other weights | 0.89 | 0.89 | +0.00 |
| **sum** | **87.76** | **87.87** | **+0.11** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.68 | 8.70 | +0.02 |
| GDN_CONV | 0.75 | 0.74 | -0.00 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.62 | 19.65 | +0.03 |
| shared expert, router, hyper-connections | 19.05 | 19.08 | +0.03 |
| everything else | 32.25 | 32.27 | +0.01 |
| **sum** | **80.42** | **80.51** | **+0.09** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 3.15 | 3.16 | +0.01 |
| the sampling chain after the product | 0.73 | 0.73 | +0.00 |
| the output's hyper-connection products | 0.10 | 0.10 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.26 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.88 | 0.89 | +0.00 |
| shared expert, router, hyper-connections | 0.89 | 0.90 | +0.00 |
| everything else | 1.32 | 1.32 | +0.00 |
| **sum** | **7.34** | **7.37** | **+0.02** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 77 of 77, 204 launch groups, verify-r2 79 of 79, 204 launch groups)

| Position | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 3.47 | 3.48 | +0.02 |
| the head's product | 3.15 | 3.16 | +0.01 |
| after the head's product | 0.73 | 0.73 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.85 (134) | 0.81 (134) | -0.05 |
| SOFTPLUS | 0.16 (35) | 0.19 (35) | +0.04 |
| SCALE | 1.46 (197) | 1.49 (197) | +0.03 |
| MUL_MAT | 43.28 (737) | 43.25 (737) | -0.03 |
| MUL_MAT_ID | 19.62 (92) | 19.65 (92) | +0.03 |
| GATED_DELTA_NET | 8.68 (35) | 8.70 (35) | +0.02 |
| RMS_NORM | 0.95 (174) | 0.97 (174) | +0.02 |
| SIGMOID | 0.24 (93) | 0.25 (93) | +0.02 |
| CONT | 0.51 (31) | 0.53 (31) | +0.01 |
| DSV4_HC_PRE | 0.51 (92) | 0.52 (92) | +0.01 |
| ADD | 0.36 (119) | 0.36 (119) | -0.01 |
| GET_ROWS | 0.20 (23) | 0.19 (23) | -0.01 |
| SET_ROWS | 0.15 (34) | 0.16 (34) | +0.01 |
| SWIGLU | 0.08 (40) | 0.08 (40) | +0.01 |
| the other 18 ops | 3.36 | 3.36 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SIGMOID | 0.02 (5) | 0.04 (5) | +0.02 |
| GET_ROWS | 0.08 (27) | 0.09 (27) | +0.01 |
| CPY | 0.04 (5) | 0.03 (5) | -0.01 |
| ARGMAX | 0.02 (8) | 0.02 (8) | +0.01 |
| ADD | 0.03 (13) | 0.03 (13) | -0.00 |
| SUM | 0.01 (4) | 0.01 (4) | -0.00 |
| LOG | 0.01 (4) | 0.01 (4) | -0.00 |
| SOFTPLUS | 0.03 (1) | 0.02 (1) | -0.00 |
| GATED_DELTA_NET | 0.25 (1) | 0.26 (1) | +0.00 |
| MUL_MAT_ID | 0.88 (4) | 0.89 (4) | +0.00 |
| CUMSUM | 0.01 (4) | 0.01 (4) | +0.00 |
| CLAMP | 0.01 (4) | 0.01 (4) | -0.00 |
| MUL_MAT | 4.94 (36) | 4.94 (36) | -0.00 |
| SCALE | 0.08 (19) | 0.08 (19) | +0.00 |
| the other 19 ops | 0.93 | 0.94 | +0.01 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.000 |
| 115 | `ADD node_6216` | 0.000 |
| 116 | `SCALE result_norm` | 0.000 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 3.158 |
| 118 | `PAD node_6219` | 0.216 |
| 119 | `TOP_K top_k` | 0.075 |
| 120 | `GET_ROWS top_k_rows` | 0.003 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 122 | `GET_ROWS top_p_candidates` | 0.005 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 124 | `SOFT_MAX top_p_softmax` | 0.007 |
| 125 | `CUMSUM top_p_cdf` | 0.006 |
| 126 | `SCALE top_p_cdf_scaled` | 0.005 |
| 127 | `STEP top_p_mask` | 0.009 |
| 128 | `SUM top_p_index_f32` | 0.006 |
| 129 | `CLAMP node_6238` | 0.005 |
| 130 | `SCALE top_p_ones` | 0.008 |
| 131 | `CPY _(copy)` | 0.007 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.011 |
| 133 | `LOG top_p_bias` | 0.006 |
| 134 | `ADD top_p_logits` | 0.007 |
| 135 | `ARGMAX temp_max_idx` | 0.009 |
| 136 | `GET_ROWS node_6248` | 0.014 |
| 137 | `GET_ROWS node_6251` | 0.008 |
| 138 | `ARGMAX greedy_argmax` | 0.008 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.013 |
| 140 | `TOP_K top_k` | 0.078 |
| 141 | `GET_ROWS top_k_rows` | 0.002 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.004 |
| 143 | `GET_ROWS top_p_candidates` | 0.005 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.004 |
| 145 | `SOFT_MAX top_p_softmax` | 0.003 |
| 146 | `CUMSUM top_p_cdf` | 0.003 |
| 147 | `SCALE top_p_cdf_scaled` | 0.003 |
| 148 | `STEP top_p_mask` | 0.004 |
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
| 159 | `ARGMAX greedy_argmax` | 0.003 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 161 | `TOP_K top_k` | 0.072 |
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
| 173 | `CPY _(copy)` | 0.001 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 175 | `LOG top_p_bias` | -0.000 |
| 176 | `ADD top_p_logits` | 0.000 |
| 177 | `ARGMAX temp_max_idx` | 0.000 |
| 178 | `GET_ROWS node_6318` | 0.001 |
| 179 | `GET_ROWS node_6321` | 0.000 |
| 180 | `ARGMAX greedy_argmax` | -0.000 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 182 | `TOP_K top_k` | 0.072 |
| 183 | `GET_ROWS top_k_rows` | 0.000 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 185 | `GET_ROWS top_p_candidates` | 0.001 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 187 | `SOFT_MAX top_p_softmax` | -0.000 |
| 188 | `CUMSUM top_p_cdf` | 0.000 |
| 189 | `SCALE top_p_cdf_scaled` | -0.000 |
| 190 | `STEP top_p_mask` | -0.000 |
| 191 | `SUM top_p_index_f32` | 0.001 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | -0.000 |
| 194 | `CPY _(copy)` | -0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 196 | `LOG top_p_bias` | 0.000 |
| 197 | `ADD top_p_logits` | 0.001 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.000 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 10.86 | 10.94 | +0.07 |
| 1 | 10.80 | 10.80 | -0.00 |
| 2 | 10.68 | 10.69 | +0.01 |
| 3 | 10.74 | 10.74 | +0.00 |
| 4 | 8.47 | 8.47 | +0.00 |
| 5 | 10.18 | 10.18 | -0.00 |
| 6 | 8.69 | 8.70 | +0.01 |
| 7 | 9.99 | 9.99 | -0.00 |
| 8 | 7.34 | 7.37 | +0.02 |
| **all** | **87.76** | **87.87** | **+0.11** |

