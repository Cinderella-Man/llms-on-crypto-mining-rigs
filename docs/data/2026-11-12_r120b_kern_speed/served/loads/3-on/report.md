2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 87, 85; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 87 | 3 | 83 | 4: 83 | 80 | 103.21 | 139.95 |
| r2 | 85 | 0 | 84 | 4: 84 | 81 | 103.35 | 139.41 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.60 | 19.63 | +0.03 |
| shared expert | 3.69 | 3.67 | -0.02 |
| hyper-connections (BF16) | 12.34 | 12.35 | +0.02 |
| router | 3.90 | 3.91 | +0.01 |
| attention and recurrent projections | 22.75 | 22.74 | -0.01 |
| Gated DeltaNet state and conv | 8.68 | 8.68 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.94 | 0.94 | -0.00 |
| other ops without weights | 5.97 | 5.97 | -0.00 |
| other weights | 0.74 | 0.73 | -0.00 |
| **sum** | **78.60** | **78.62** | **+0.02** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert | 0.18 | 0.18 | +0.00 |
| hyper-connections (BF16) | 0.56 | 0.56 | +0.00 |
| router | 0.17 | 0.17 | +0.00 |
| attention and recurrent projections | 0.80 | 0.81 | +0.01 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | -0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 3.16 | 3.17 | +0.01 |
| other ops without weights | 1.13 | 1.12 | -0.02 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **7.36** | **7.37** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.46 | 20.50 | +0.03 |
| shared expert | 3.87 | 3.85 | -0.02 |
| hyper-connections (BF16) | 12.89 | 12.91 | +0.02 |
| router | 4.07 | 4.08 | +0.01 |
| attention and recurrent projections | 23.55 | 23.55 | +0.00 |
| Gated DeltaNet state and conv | 8.93 | 8.93 | +0.00 |
| attention (FLASH_ATTN_EXT) | 1.03 | 1.03 | -0.00 |
| output layer | 3.16 | 3.17 | +0.01 |
| other ops without weights | 7.10 | 7.08 | -0.02 |
| other weights | 0.88 | 0.88 | -0.00 |
| **sum** | **85.96** | **85.99** | **+0.03** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.68 | 8.68 | +0.00 |
| GDN_CONV | 0.74 | 0.75 | +0.01 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.60 | 19.63 | +0.03 |
| shared expert, router, hyper-connections | 19.93 | 19.93 | +0.01 |
| everything else | 29.59 | 29.56 | -0.03 |
| **sum** | **78.60** | **78.62** | **+0.02** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 3.16 | 3.17 | +0.01 |
| the sampling chain after the product | 0.72 | 0.72 | -0.00 |
| the output's hyper-connection products | 0.10 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | -0.00 |
| GDN_CONV | 0.02 | 0.02 | -0.00 |
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert, router, hyper-connections | 0.91 | 0.91 | +0.00 |
| everything else | 1.32 | 1.32 | +0.00 |
| **sum** | **7.36** | **7.37** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 80 of 80, 200 launch groups, verify-r2 81 of 81, 200 launch groups)

| Position | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 3.47 | 3.48 | +0.01 |
| the head's product | 3.16 | 3.17 | +0.01 |
| after the head's product | 0.72 | 0.72 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SOFTPLUS | 0.18 (35) | 0.23 (35) | +0.05 |
| ADD | 0.40 (119) | 0.35 (119) | -0.04 |
| MUL | 0.82 (134) | 0.79 (134) | -0.04 |
| MUL_MAT_ID | 19.60 (92) | 19.63 (92) | +0.03 |
| RMS_NORM | 0.92 (174) | 0.89 (174) | -0.03 |
| MUL_MAT | 41.67 (699) | 41.65 (699) | -0.02 |
| GET_ROWS | 0.18 (23) | 0.19 (23) | +0.01 |
| DSV4_HC_PRE | 0.49 (92) | 0.50 (92) | +0.01 |
| SIGMOID | 0.25 (93) | 0.27 (93) | +0.01 |
| GDN_CONV | 0.74 (35) | 0.75 (35) | +0.01 |
| SOFT_MAX | 0.53 (46) | 0.54 (46) | +0.01 |
| CONT | 0.54 (31) | 0.54 (31) | +0.00 |
| CPY | 0.29 (20) | 0.30 (20) | +0.00 |
| TOP_K | 0.13 (11) | 0.13 (11) | -0.00 |
| the other 18 ops | 11.87 | 11.88 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 4.95 (34) | 4.97 (34) | +0.02 |
| MUL | 0.09 (5) | 0.08 (5) | -0.01 |
| SET_ROWS | 0.03 (8) | 0.03 (8) | -0.01 |
| SOFTPLUS | 0.04 (1) | 0.04 (1) | +0.01 |
| ADD | 0.03 (13) | 0.03 (13) | +0.01 |
| LOG | 0.01 (4) | 0.01 (4) | +0.01 |
| SIGMOID | 0.02 (5) | 0.02 (5) | -0.00 |
| CLAMP | 0.01 (4) | 0.01 (4) | -0.00 |
| ARGSORT | 0.01 (4) | 0.01 (4) | +0.00 |
| SOFT_MAX | 0.03 (6) | 0.03 (6) | -0.00 |
| MUL_MAT_ID | 0.87 (4) | 0.87 (4) | +0.00 |
| GET_ROWS | 0.09 (27) | 0.09 (27) | +0.00 |
| SCALE | 0.08 (19) | 0.08 (19) | -0.00 |
| ARGMAX | 0.03 (8) | 0.03 (8) | +0.00 |
| the other 18 ops | 1.08 | 1.07 | -0.01 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 110 | `ADD node_6214` | 0.000 |
| 111 | `ADD node_6216` | 0.000 |
| 112 | `SCALE result_norm` | 0.001 |
| 113 | `MUL_MAT output:q5_K:2560x248320` | 3.168 |
| 114 | `PAD node_6219` | 0.216 |
| 115 | `TOP_K top_k` | 0.074 |
| 116 | `GET_ROWS top_k_rows` | 0.003 |
| 117 | `ARGSORT top_p_sorted_idx` | 0.002 |
| 118 | `GET_ROWS top_p_candidates` | 0.003 |
| 119 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 120 | `SOFT_MAX top_p_softmax` | 0.003 |
| 121 | `CUMSUM top_p_cdf` | 0.001 |
| 122 | `SCALE top_p_cdf_scaled` | 0.000 |
| 123 | `STEP top_p_mask` | 0.003 |
| 124 | `SUM top_p_index_f32` | 0.003 |
| 125 | `CLAMP node_6238` | 0.003 |
| 126 | `SCALE top_p_ones` | 0.006 |
| 127 | `CPY _(copy)` | 0.006 |
| 128 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.006 |
| 129 | `LOG top_p_bias` | 0.011 |
| 130 | `ADD top_p_logits` | 0.017 |
| 131 | `ARGMAX temp_max_idx` | 0.012 |
| 132 | `GET_ROWS node_6248` | 0.012 |
| 133 | `GET_ROWS node_6251` | 0.012 |
| 134 | `ARGMAX greedy_argmax` | 0.012 |
| 135 | `GET_ROWS greedy_sampled_token` | 0.007 |
| 136 | `TOP_K top_k` | 0.083 |
| 137 | `GET_ROWS top_k_rows` | 0.007 |
| 138 | `ARGSORT top_p_sorted_idx` | 0.006 |
| 139 | `GET_ROWS top_p_candidates` | 0.006 |
| 140 | `GET_ROWS top_p_sorted_logits` | 0.004 |
| 141 | `SOFT_MAX top_p_softmax` | 0.003 |
| 142 | `CUMSUM top_p_cdf` | 0.002 |
| 143 | `SCALE top_p_cdf_scaled` | 0.003 |
| 144 | `STEP top_p_mask` | 0.003 |
| 145 | `SUM top_p_index_f32` | 0.003 |
| 146 | `CLAMP node_6273` | 0.003 |
| 147 | `SCALE top_p_ones` | 0.003 |
| 148 | `CPY _(copy)` | 0.003 |
| 149 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 150 | `LOG top_p_bias` | 0.003 |
| 151 | `ADD top_p_logits` | 0.002 |
| 152 | `ARGMAX temp_max_idx` | 0.003 |
| 153 | `GET_ROWS node_6283` | 0.003 |
| 154 | `GET_ROWS node_6286` | 0.003 |
| 155 | `ARGMAX greedy_argmax` | 0.002 |
| 156 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 157 | `TOP_K top_k` | 0.071 |
| 158 | `GET_ROWS top_k_rows` | 0.001 |
| 159 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 160 | `GET_ROWS top_p_candidates` | 0.000 |
| 161 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 162 | `SOFT_MAX top_p_softmax` | 0.001 |
| 163 | `CUMSUM top_p_cdf` | 0.001 |
| 164 | `SCALE top_p_cdf_scaled` | 0.000 |
| 165 | `STEP top_p_mask` | -0.000 |
| 166 | `SUM top_p_index_f32` | 0.000 |
| 167 | `CLAMP node_6308` | 0.000 |
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
| 178 | `TOP_K top_k` | 0.070 |
| 179 | `GET_ROWS top_k_rows` | 0.000 |
| 180 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 181 | `GET_ROWS top_p_candidates` | 0.000 |
| 182 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 183 | `SOFT_MAX top_p_softmax` | 0.000 |
| 184 | `CUMSUM top_p_cdf` | -0.000 |
| 185 | `SCALE top_p_cdf_scaled` | -0.000 |
| 186 | `STEP top_p_mask` | -0.000 |
| 187 | `SUM top_p_index_f32` | 0.000 |
| 188 | `CLAMP node_6343` | 0.000 |
| 189 | `SCALE top_p_ones` | 0.000 |
| 190 | `CPY _(copy)` | -0.000 |
| 191 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 192 | `LOG top_p_bias` | -0.000 |
| 193 | `ADD top_p_logits` | 0.000 |
| 194 | `ARGMAX temp_max_idx` | -0.000 |
| 195 | `GET_ROWS node_6353` | 0.001 |
| 196 | `GET_ROWS node_6356` | 0.000 |
| 197 | `ARGMAX greedy_argmax` | -0.000 |
| 198 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 199 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 10.95 | 11.00 | +0.05 |
| 1 | 10.43 | 10.44 | +0.01 |
| 2 | 10.29 | 10.30 | +0.00 |
| 3 | 10.38 | 10.37 | -0.01 |
| 4 | 8.24 | 8.23 | -0.01 |
| 5 | 10.02 | 10.02 | -0.00 |
| 6 | 8.50 | 8.50 | +0.00 |
| 7 | 9.78 | 9.77 | -0.02 |
| 8 | 7.36 | 7.37 | +0.01 |
| **all** | **85.96** | **85.99** | **+0.03** |

