2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.90 | 157.30 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 119.04 | 158.69 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.57 | 19.54 | -0.02 |
| shared expert | 4.61 | 4.62 | +0.01 |
| hyper-connections (BF16) | 12.43 | 12.41 | -0.02 |
| router | 2.30 | 2.30 | -0.00 |
| attention and recurrent projections | 32.77 | 32.69 | -0.08 |
| Gated DeltaNet state and conv | 8.68 | 8.68 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.95 | 0.94 | -0.01 |
| other ops without weights | 6.09 | 6.19 | +0.10 |
| other weights | 0.74 | 0.73 | -0.00 |
| **sum** | **88.13** | **88.12** | **-0.02** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert | 0.28 | 0.28 | +0.00 |
| hyper-connections (BF16) | 0.57 | 0.57 | -0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.42 | 1.44 | +0.01 |
| Gated DeltaNet state and conv | 0.25 | 0.26 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.10 | 8.12 | +0.02 |
| other ops without weights | 1.07 | 1.06 | -0.01 |
| other weights | 0.15 | 0.15 | -0.00 |
| **sum** | **12.90** | **12.93** | **+0.03** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.43 | 20.41 | -0.02 |
| shared expert | 4.88 | 4.89 | +0.01 |
| hyper-connections (BF16) | 13.00 | 12.98 | -0.02 |
| router | 2.41 | 2.40 | -0.00 |
| attention and recurrent projections | 34.19 | 34.13 | -0.06 |
| Gated DeltaNet state and conv | 8.93 | 8.94 | +0.00 |
| attention (FLASH_ATTN_EXT) | 1.04 | 1.03 | -0.01 |
| output layer | 8.10 | 8.12 | +0.02 |
| other ops without weights | 7.16 | 7.26 | +0.10 |
| other weights | 0.89 | 0.89 | -0.00 |
| **sum** | **101.03** | **101.05** | **+0.01** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.68 | 8.68 | +0.00 |
| GDN_CONV | 0.73 | 0.75 | +0.03 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.57 | 19.54 | -0.02 |
| shared expert, router, hyper-connections | 19.34 | 19.33 | -0.01 |
| everything else | 39.75 | 39.74 | -0.01 |
| **sum** | **88.13** | **88.12** | **-0.02** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.10 | 8.12 | +0.02 |
| the sampling chain after the product | 0.65 | 0.65 | +0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | -0.00 |
| GATED_DELTA_NET | 0.25 | 0.26 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | -0.00 |
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert, router, hyper-connections | 0.95 | 0.95 | -0.00 |
| everything else | 1.96 | 1.96 | +0.00 |
| **sum** | **12.90** | **12.93** | **+0.03** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 204 launch groups, verify-r2 80 of 80, 204 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.14 | 4.15 | +0.01 |
| the head's product | 8.10 | 8.12 | +0.02 |
| after the head's product | 0.65 | 0.65 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SOFTPLUS | 0.14 (35) | 0.22 (35) | +0.07 |
| MUL_MAT | 51.05 (737) | 51.00 (737) | -0.06 |
| DSV4_HC_PRE | 0.52 (92) | 0.50 (92) | -0.03 |
| GDN_CONV | 0.73 (35) | 0.75 (35) | +0.03 |
| MUL_MAT_ID | 19.57 (92) | 19.54 (92) | -0.02 |
| RMS_NORM | 0.98 (174) | 0.96 (174) | -0.02 |
| ADD | 0.36 (119) | 0.37 (119) | +0.01 |
| CONT | 0.54 (31) | 0.52 (31) | -0.01 |
| SILU | 0.19 (93) | 0.20 (93) | +0.01 |
| SIGMOID | 0.26 (93) | 0.27 (93) | +0.01 |
| CONCAT | 0.13 (12) | 0.14 (12) | +0.01 |
| FLASH_ATTN_EXT | 0.95 (11) | 0.94 (11) | -0.01 |
| MUL | 0.82 (134) | 0.81 (134) | -0.01 |
| SCALE | 1.48 (197) | 1.48 (197) | -0.01 |
| the other 18 ops | 10.42 | 10.41 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 10.55 (36) | 10.58 (36) | +0.03 |
| SOFTPLUS | 0.02 (1) | 0.00 (1) | -0.02 |
| SIGMOID | 0.04 (5) | 0.04 (5) | +0.00 |
| TOP_K | 0.31 (5) | 0.31 (5) | +0.00 |
| MUL_MAT_ID | 0.86 (4) | 0.86 (4) | +0.00 |
| CUMSUM | 0.01 (4) | 0.01 (4) | +0.00 |
| GATED_DELTA_NET | 0.25 (1) | 0.26 (1) | +0.00 |
| MUL | 0.07 (5) | 0.07 (5) | +0.00 |
| ARGMAX | 0.01 (8) | 0.01 (8) | -0.00 |
| ARGSORT | 0.01 (4) | 0.01 (4) | +0.00 |
| RMS_NORM | 0.05 (10) | 0.06 (10) | +0.00 |
| SUM | 0.01 (4) | 0.00 (4) | -0.00 |
| DSV4_HC_PRE | 0.03 (4) | 0.02 (4) | -0.00 |
| SET_ROWS | 0.02 (8) | 0.02 (8) | -0.00 |
| the other 19 ops | 0.67 | 0.67 | -0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.125 |
| 118 | `PAD node_6219` | 0.217 |
| 119 | `TOP_K top_k` | 0.076 |
| 120 | `GET_ROWS top_k_rows` | 0.004 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 122 | `GET_ROWS top_p_candidates` | 0.005 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 124 | `SOFT_MAX top_p_softmax` | 0.004 |
| 125 | `CUMSUM top_p_cdf` | 0.003 |
| 126 | `SCALE top_p_cdf_scaled` | 0.003 |
| 127 | `STEP top_p_mask` | 0.003 |
| 128 | `SUM top_p_index_f32` | 0.003 |
| 129 | `CLAMP node_6238` | 0.002 |
| 130 | `SCALE top_p_ones` | 0.003 |
| 131 | `CPY _(copy)` | 0.003 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 133 | `LOG top_p_bias` | 0.002 |
| 134 | `ADD top_p_logits` | 0.003 |
| 135 | `ARGMAX temp_max_idx` | 0.002 |
| 136 | `GET_ROWS node_6248` | 0.003 |
| 137 | `GET_ROWS node_6251` | 0.003 |
| 138 | `ARGMAX greedy_argmax` | 0.003 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 140 | `TOP_K top_k` | 0.073 |
| 141 | `GET_ROWS top_k_rows` | 0.002 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.002 |
| 143 | `GET_ROWS top_p_candidates` | 0.002 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.001 |
| 147 | `SCALE top_p_cdf_scaled` | 0.001 |
| 148 | `STEP top_p_mask` | 0.001 |
| 149 | `SUM top_p_index_f32` | 0.001 |
| 150 | `CLAMP node_6273` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.000 |
| 152 | `CPY _(copy)` | 0.000 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 154 | `LOG top_p_bias` | 0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | -0.000 |
| 157 | `GET_ROWS node_6283` | 0.001 |
| 158 | `GET_ROWS node_6286` | 0.001 |
| 159 | `ARGMAX greedy_argmax` | 0.000 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.000 |
| 165 | `GET_ROWS top_p_sorted_logits` | -0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.000 |
| 167 | `CUMSUM top_p_cdf` | 0.003 |
| 168 | `SCALE top_p_cdf_scaled` | 0.000 |
| 169 | `STEP top_p_mask` | 0.001 |
| 170 | `SUM top_p_index_f32` | 0.000 |
| 171 | `CLAMP node_6308` | 0.001 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.000 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 175 | `LOG top_p_bias` | 0.000 |
| 176 | `ADD top_p_logits` | 0.001 |
| 177 | `ARGMAX temp_max_idx` | 0.001 |
| 178 | `GET_ROWS node_6318` | 0.005 |
| 179 | `GET_ROWS node_6321` | 0.002 |
| 180 | `ARGMAX greedy_argmax` | 0.004 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.006 |
| 182 | `TOP_K top_k` | 0.077 |
| 183 | `GET_ROWS top_k_rows` | 0.005 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.004 |
| 185 | `GET_ROWS top_p_candidates` | 0.006 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 187 | `SOFT_MAX top_p_softmax` | 0.002 |
| 188 | `CUMSUM top_p_cdf` | 0.003 |
| 189 | `SCALE top_p_cdf_scaled` | 0.001 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | 0.001 |
| 192 | `CLAMP node_6343` | 0.001 |
| 193 | `SCALE top_p_ones` | -0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.001 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.41 | 12.40 | -0.01 |
| 1 | 11.18 | 11.18 | -0.00 |
| 2 | 11.34 | 11.35 | +0.01 |
| 3 | 11.48 | 11.48 | +0.00 |
| 4 | 9.26 | 9.25 | -0.01 |
| 5 | 11.42 | 11.42 | +0.00 |
| 6 | 9.59 | 9.58 | -0.01 |
| 7 | 11.45 | 11.45 | -0.00 |
| 8 | 12.90 | 12.93 | +0.03 |
| **all** | **101.03** | **101.05** | **+0.01** |

