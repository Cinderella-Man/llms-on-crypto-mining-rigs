2945 evaluations on cards 0 to 8: the model's graph 0: 175, 1: 175, 2: 175, 3: 175, 4: 175, 5: 175, 6: 175, 7: 175, 8: 175; the draft's graphs 1370 (card 4: 685, card 8: 685), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 86; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 110.91 | 148.11 |
| r2 | 86 | 0 | 85 | 4: 85 | 82 | 110.92 | 146.81 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.52 | 19.53 | +0.02 |
| shared expert | 4.39 | 4.39 | -0.01 |
| hyper-connections (BF16) | 12.38 | 12.35 | -0.03 |
| router | 2.30 | 2.31 | +0.00 |
| attention and recurrent projections | 25.62 | 25.59 | -0.03 |
| Gated DeltaNet state and conv | 8.66 | 8.68 | +0.02 |
| attention (FLASH_ATTN_EXT) | 0.94 | 0.94 | -0.00 |
| other ops without weights | 6.14 | 6.16 | +0.03 |
| other weights | 0.74 | 0.74 | -0.00 |
| **sum** | **80.69** | **80.69** | **+0.00** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert | 0.27 | 0.27 | -0.00 |
| hyper-connections (BF16) | 0.56 | 0.55 | -0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 0.90 | 0.89 | -0.00 |
| Gated DeltaNet state and conv | 0.26 | 0.26 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.11 | 8.12 | +0.01 |
| other ops without weights | 1.11 | 1.12 | +0.01 |
| other weights | 0.15 | 0.15 | -0.00 |
| **sum** | **12.41** | **12.43** | **+0.02** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.39 | 20.41 | +0.02 |
| shared expert | 4.66 | 4.66 | -0.01 |
| hyper-connections (BF16) | 12.93 | 12.90 | -0.03 |
| router | 2.40 | 2.41 | +0.00 |
| attention and recurrent projections | 26.52 | 26.49 | -0.04 |
| Gated DeltaNet state and conv | 8.92 | 8.93 | +0.02 |
| attention (FLASH_ATTN_EXT) | 1.03 | 1.03 | -0.00 |
| output layer | 8.11 | 8.12 | +0.01 |
| other ops without weights | 7.24 | 7.29 | +0.04 |
| other weights | 0.89 | 0.89 | -0.00 |
| **sum** | **93.11** | **93.12** | **+0.02** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.66 | 8.68 | +0.02 |
| GDN_CONV | 0.72 | 0.75 | +0.04 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.52 | 19.53 | +0.02 |
| shared expert, router, hyper-connections | 19.07 | 19.04 | -0.03 |
| everything else | 32.66 | 32.62 | -0.04 |
| **sum** | **80.69** | **80.69** | **+0.00** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.11 | 8.12 | +0.01 |
| the sampling chain after the product | 0.73 | 0.73 | -0.00 |
| the output's hyper-connection products | 0.10 | 0.10 | -0.00 |
| GATED_DELTA_NET | 0.26 | 0.26 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert, router, hyper-connections | 0.93 | 0.92 | -0.00 |
| everything else | 1.39 | 1.40 | +0.01 |
| **sum** | **12.41** | **12.43** | **+0.02** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 204 launch groups, verify-r2 82 of 82, 204 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 3.57 | 3.58 | +0.01 |
| the head's product | 8.11 | 8.12 | +0.01 |
| after the head's product | 0.73 | 0.73 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.88 (134) | 0.81 (134) | -0.07 |
| MUL_MAT | 43.67 (737) | 43.63 (737) | -0.04 |
| GDN_CONV | 0.72 (35) | 0.75 (35) | +0.04 |
| SOFTPLUS | 0.20 (35) | 0.23 (35) | +0.03 |
| GET_ROWS | 0.20 (23) | 0.18 (23) | -0.02 |
| MUL_MAT_ID | 19.52 (92) | 19.53 (92) | +0.02 |
| GATED_DELTA_NET | 8.66 (35) | 8.68 (35) | +0.02 |
| ADD | 0.35 (119) | 0.37 (119) | +0.02 |
| SIGMOID | 0.24 (93) | 0.25 (93) | +0.01 |
| SET_ROWS | 0.17 (34) | 0.16 (34) | -0.01 |
| SILU | 0.19 (93) | 0.20 (93) | +0.01 |
| RMS_NORM | 0.95 (174) | 0.94 (174) | -0.01 |
| SWIGLU | 0.08 (40) | 0.08 (40) | -0.00 |
| TOP_K_MASK | 0.08 (11) | 0.08 (11) | +0.00 |
| the other 18 ops | 4.79 | 4.80 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SOFTPLUS | 0.01 (1) | 0.03 (1) | +0.01 |
| MUL | 0.07 (5) | 0.06 (5) | -0.01 |
| CLAMP | 0.01 (4) | 0.01 (4) | -0.01 |
| SET_ROWS | 0.02 (8) | 0.03 (8) | +0.00 |
| MUL_MAT | 10.01 (36) | 10.02 (36) | +0.00 |
| STEP | 0.01 (4) | 0.01 (4) | +0.00 |
| TOP_K | 0.31 (5) | 0.31 (5) | -0.00 |
| MUL_MAT_ID | 0.87 (4) | 0.87 (4) | +0.00 |
| ADD | 0.03 (13) | 0.03 (13) | +0.00 |
| GDN_CONV | 0.02 (1) | 0.02 (1) | +0.00 |
| ARGSORT | 0.01 (4) | 0.00 (4) | -0.00 |
| GET_ROWS | 0.09 (27) | 0.09 (27) | +0.00 |
| DSV4_HC_PRE | 0.02 (4) | 0.02 (4) | -0.00 |
| CPY | 0.03 (5) | 0.04 (5) | +0.00 |
| the other 19 ops | 0.90 | 0.90 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.000 |
| 115 | `ADD node_6216` | 0.002 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.125 |
| 118 | `PAD node_6219` | 0.217 |
| 119 | `TOP_K top_k` | 0.075 |
| 120 | `GET_ROWS top_k_rows` | 0.003 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 122 | `GET_ROWS top_p_candidates` | 0.003 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 124 | `SOFT_MAX top_p_softmax` | 0.003 |
| 125 | `CUMSUM top_p_cdf` | 0.004 |
| 126 | `SCALE top_p_cdf_scaled` | 0.007 |
| 127 | `STEP top_p_mask` | 0.008 |
| 128 | `SUM top_p_index_f32` | 0.009 |
| 129 | `CLAMP node_6238` | 0.004 |
| 130 | `SCALE top_p_ones` | 0.007 |
| 131 | `CPY _(copy)` | 0.010 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.008 |
| 133 | `LOG top_p_bias` | 0.009 |
| 134 | `ADD top_p_logits` | 0.009 |
| 135 | `ARGMAX temp_max_idx` | 0.011 |
| 136 | `GET_ROWS node_6248` | 0.013 |
| 137 | `GET_ROWS node_6251` | 0.010 |
| 138 | `ARGMAX greedy_argmax` | 0.008 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.010 |
| 140 | `TOP_K top_k` | 0.079 |
| 141 | `GET_ROWS top_k_rows` | 0.004 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.004 |
| 143 | `GET_ROWS top_p_candidates` | 0.005 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.004 |
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
| 159 | `ARGMAX greedy_argmax` | 0.003 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 161 | `TOP_K top_k` | 0.072 |
| 162 | `GET_ROWS top_k_rows` | 0.001 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.000 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.000 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | -0.000 |
| 169 | `STEP top_p_mask` | 0.000 |
| 170 | `SUM top_p_index_f32` | 0.000 |
| 171 | `CLAMP node_6308` | -0.000 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.000 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 175 | `LOG top_p_bias` | -0.000 |
| 176 | `ADD top_p_logits` | 0.000 |
| 177 | `ARGMAX temp_max_idx` | -0.000 |
| 178 | `GET_ROWS node_6318` | 0.001 |
| 179 | `GET_ROWS node_6321` | 0.001 |
| 180 | `ARGMAX greedy_argmax` | 0.000 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 182 | `TOP_K top_k` | 0.071 |
| 183 | `GET_ROWS top_k_rows` | 0.000 |
| 184 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 185 | `GET_ROWS top_p_candidates` | 0.000 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 187 | `SOFT_MAX top_p_softmax` | 0.000 |
| 188 | `CUMSUM top_p_cdf` | -0.000 |
| 189 | `SCALE top_p_cdf_scaled` | 0.000 |
| 190 | `STEP top_p_mask` | -0.000 |
| 191 | `SUM top_p_index_f32` | -0.000 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | -0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | 0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | 0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (82 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 10.93 | 10.92 | -0.01 |
| 1 | 10.72 | 10.73 | +0.00 |
| 2 | 10.77 | 10.75 | -0.01 |
| 3 | 10.70 | 10.71 | +0.01 |
| 4 | 8.54 | 8.53 | -0.01 |
| 5 | 10.26 | 10.28 | +0.01 |
| 6 | 8.81 | 8.82 | +0.01 |
| 7 | 9.96 | 9.96 | -0.01 |
| 8 | 12.41 | 12.43 | +0.02 |
| **all** | **93.11** | **93.12** | **+0.02** |

