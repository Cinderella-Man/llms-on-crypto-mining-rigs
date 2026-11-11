2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.52 | 154.56 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 118.58 | 153.94 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.57 | 19.56 | -0.00 |
| shared expert | 4.62 | 4.62 | -0.00 |
| hyper-connections (BF16) | 12.42 | 12.42 | -0.00 |
| router | 2.30 | 2.30 | +0.00 |
| attention and recurrent projections | 32.74 | 32.75 | +0.02 |
| Gated DeltaNet state and conv | 8.69 | 8.69 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.95 | 0.94 | -0.01 |
| other ops without weights | 6.13 | 6.15 | +0.02 |
| other weights | 0.73 | 0.74 | +0.00 |
| **sum** | **88.15** | **88.18** | **+0.03** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.86 | 0.86 | +0.01 |
| shared expert | 0.28 | 0.28 | +0.00 |
| hyper-connections (BF16) | 0.57 | 0.57 | +0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.42 | 1.43 | +0.01 |
| Gated DeltaNet state and conv | 0.25 | 0.26 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.07 | 8.08 | +0.02 |
| other ops without weights | 0.74 | 0.73 | -0.02 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **12.52** | **12.55** | **+0.03** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.42 | 20.42 | +0.00 |
| shared expert | 4.90 | 4.90 | +0.00 |
| hyper-connections (BF16) | 12.99 | 12.99 | +0.00 |
| router | 2.40 | 2.41 | +0.00 |
| attention and recurrent projections | 34.16 | 34.18 | +0.03 |
| Gated DeltaNet state and conv | 8.94 | 8.95 | +0.01 |
| attention (FLASH_ATTN_EXT) | 1.04 | 1.03 | -0.01 |
| output layer | 8.07 | 8.08 | +0.02 |
| other ops without weights | 6.87 | 6.88 | +0.01 |
| other weights | 0.89 | 0.89 | +0.00 |
| **sum** | **100.67** | **100.73** | **+0.06** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.69 | 8.69 | +0.00 |
| GDN_CONV | 0.74 | 0.75 | +0.02 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.57 | 19.56 | -0.00 |
| shared expert, router, hyper-connections | 19.34 | 19.34 | +0.00 |
| everything else | 39.74 | 39.76 | +0.01 |
| **sum** | **88.15** | **88.18** | **+0.03** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.07 | 8.08 | +0.02 |
| the sampling chain after the product | 0.32 | 0.32 | -0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.26 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.86 | 0.86 | +0.01 |
| shared expert, router, hyper-connections | 0.94 | 0.95 | +0.01 |
| everything else | 1.96 | 1.95 | -0.01 |
| **sum** | **12.52** | **12.55** | **+0.03** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 197 launch groups, verify-r2 80 of 80, 197 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.14 | 4.15 | +0.01 |
| the head's product | 8.07 | 8.08 | +0.02 |
| after the head's product | 0.32 | 0.32 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.81 (134) | 0.76 (134) | -0.05 |
| SIGMOID | 0.27 (93) | 0.31 (93) | +0.04 |
| DSV4_HC_PRE | 0.50 (92) | 0.52 (92) | +0.02 |
| GDN_CONV | 0.74 (35) | 0.75 (35) | +0.02 |
| GET_ROWS | 0.20 (23) | 0.18 (23) | -0.02 |
| SCALE | 1.48 (197) | 1.47 (197) | -0.01 |
| FLASH_ATTN_EXT | 0.95 (11) | 0.94 (11) | -0.01 |
| SOFTPLUS | 0.18 (35) | 0.18 (35) | +0.01 |
| ADD | 0.37 (119) | 0.38 (119) | +0.01 |
| ARANGE | 0.01 (1) | 0.02 (1) | +0.01 |
| MUL_MAT | 51.05 (737) | 51.05 (737) | +0.01 |
| SET_ROWS | 0.15 (34) | 0.15 (34) | +0.01 |
| SOFT_MAX | 0.52 (46) | 0.51 (46) | -0.01 |
| MUL_MAT_ID | 19.57 (92) | 19.56 (92) | -0.00 |
| the other 18 ops | 11.37 | 11.39 | +0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 10.50 (36) | 10.53 (36) | +0.03 |
| SOFTPLUS | 0.03 (1) | 0.01 (1) | -0.02 |
| MUL_MAT_ID | 0.86 (4) | 0.86 (4) | +0.01 |
| GATED_DELTA_NET | 0.25 (1) | 0.26 (1) | +0.00 |
| MUL | 0.08 (5) | 0.08 (5) | -0.00 |
| TOP_K | 0.24 (2) | 0.23 (2) | -0.00 |
| SIGMOID | 0.03 (5) | 0.03 (5) | +0.00 |
| SUM | 0.01 (4) | 0.00 (4) | -0.00 |
| LOG | 0.00 (4) | 0.01 (4) | +0.00 |
| SCALE | 0.08 (19) | 0.09 (19) | +0.00 |
| ARGMAX | 0.00 (8) | 0.01 (8) | +0.00 |
| SET_ROWS | 0.02 (8) | 0.02 (8) | +0.00 |
| STEP | 0.01 (4) | 0.00 (4) | -0.00 |
| GET_ROWS | 0.05 (24) | 0.05 (24) | +0.00 |
| the other 18 ops | 0.37 | 0.37 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.002 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.085 |
| 118 | `TOP_K top_k` | 0.219 |
| 119 | `GET_ROWS top_k_rows` | 0.005 |
| 120 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 121 | `GET_ROWS top_p_candidates` | 0.005 |
| 122 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 123 | `SOFT_MAX top_p_softmax` | 0.004 |
| 124 | `CUMSUM top_p_cdf` | 0.003 |
| 125 | `SCALE top_p_cdf_scaled` | 0.003 |
| 126 | `STEP top_p_mask` | 0.003 |
| 127 | `SUM top_p_index_f32` | 0.003 |
| 128 | `CLAMP node_6237` | 0.003 |
| 129 | `SCALE top_p_ones` | 0.003 |
| 130 | `CPY _(copy)` | 0.003 |
| 131 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.002 |
| 132 | `LOG top_p_bias` | 0.003 |
| 133 | `ADD top_p_logits` | 0.003 |
| 134 | `ARGMAX temp_max_idx` | 0.002 |
| 135 | `GET_ROWS node_6247` | 0.002 |
| 136 | `GET_ROWS node_6250` | 0.002 |
| 137 | `ARGMAX greedy_argmax` | 0.002 |
| 138 | `GET_ROWS greedy_sampled_token` | 0.001 |
| 139 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 140 | `GET_ROWS top_p_candidates` | 0.001 |
| 141 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 142 | `SOFT_MAX top_p_softmax` | 0.001 |
| 143 | `CUMSUM top_p_cdf` | 0.001 |
| 144 | `SCALE top_p_cdf_scaled` | 0.000 |
| 145 | `STEP top_p_mask` | 0.000 |
| 146 | `SUM top_p_index_f32` | -0.000 |
| 147 | `CLAMP node_6269` | 0.000 |
| 148 | `SCALE top_p_ones` | 0.000 |
| 149 | `CPY _(copy)` | 0.000 |
| 150 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 151 | `LOG top_p_bias` | 0.000 |
| 152 | `ADD top_p_logits` | 0.000 |
| 153 | `ARGMAX temp_max_idx` | 0.000 |
| 154 | `GET_ROWS node_6279` | 0.001 |
| 155 | `GET_ROWS node_6282` | 0.001 |
| 156 | `ARGMAX greedy_argmax` | 0.000 |
| 157 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 158 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 159 | `GET_ROWS top_p_candidates` | 0.000 |
| 160 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 161 | `SOFT_MAX top_p_softmax` | 0.001 |
| 162 | `CUMSUM top_p_cdf` | 0.000 |
| 163 | `SCALE top_p_cdf_scaled` | -0.000 |
| 164 | `STEP top_p_mask` | 0.000 |
| 165 | `SUM top_p_index_f32` | 0.000 |
| 166 | `CLAMP node_6301` | 0.000 |
| 167 | `SCALE top_p_ones` | 0.000 |
| 168 | `CPY _(copy)` | 0.000 |
| 169 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 170 | `LOG top_p_bias` | 0.000 |
| 171 | `ADD top_p_logits` | 0.000 |
| 172 | `ARGMAX temp_max_idx` | 0.000 |
| 173 | `GET_ROWS node_6311` | 0.001 |
| 174 | `GET_ROWS node_6314` | 0.000 |
| 175 | `ARGMAX greedy_argmax` | -0.000 |
| 176 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 177 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 178 | `GET_ROWS top_p_candidates` | 0.001 |
| 179 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 180 | `SOFT_MAX top_p_softmax` | -0.000 |
| 181 | `CUMSUM top_p_cdf` | 0.001 |
| 182 | `SCALE top_p_cdf_scaled` | 0.001 |
| 183 | `STEP top_p_mask` | 0.000 |
| 184 | `SUM top_p_index_f32` | 0.001 |
| 185 | `CLAMP node_6333` | 0.003 |
| 186 | `SCALE top_p_ones` | 0.001 |
| 187 | `CPY _(copy)` | 0.001 |
| 188 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 189 | `LOG top_p_bias` | 0.003 |
| 190 | `ADD top_p_logits` | 0.001 |
| 191 | `ARGMAX temp_max_idx` | 0.001 |
| 192 | `GET_ROWS node_6343` | 0.001 |
| 193 | `GET_ROWS node_6346` | 0.001 |
| 194 | `ARGMAX greedy_argmax` | -0.000 |
| 195 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 196 | `SET_ROWS h_nextn_dev_(view)` | 0.008 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.41 | 12.43 | +0.02 |
| 1 | 11.19 | 11.17 | -0.02 |
| 2 | 11.34 | 11.36 | +0.01 |
| 3 | 11.49 | 11.50 | +0.01 |
| 4 | 9.26 | 9.26 | +0.00 |
| 5 | 11.43 | 11.43 | +0.00 |
| 6 | 9.58 | 9.59 | +0.00 |
| 7 | 11.44 | 11.44 | -0.00 |
| 8 | 12.52 | 12.55 | +0.03 |
| **all** | **100.67** | **100.73** | **+0.06** |

