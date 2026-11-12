2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.70 | 161.41 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 118.71 | 155.48 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.52 | 19.51 | -0.01 |
| shared expert | 4.62 | 4.64 | +0.02 |
| hyper-connections (BF16) | 12.40 | 12.41 | +0.01 |
| router | 2.30 | 2.30 | -0.00 |
| attention and recurrent projections | 32.68 | 32.66 | -0.02 |
| Gated DeltaNet state and conv | 8.65 | 8.65 | -0.00 |
| attention (FLASH_ATTN_EXT) | 0.95 | 0.94 | -0.01 |
| other ops without weights | 6.10 | 6.09 | -0.01 |
| other weights | 0.74 | 0.73 | -0.01 |
| **sum** | **87.95** | **87.91** | **-0.04** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.85 | 0.86 | +0.00 |
| shared expert | 0.28 | 0.28 | -0.00 |
| hyper-connections (BF16) | 0.56 | 0.57 | +0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.41 | 1.42 | +0.00 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.10 | 8.10 | +0.01 |
| other ops without weights | 1.07 | 1.07 | -0.00 |
| other weights | 0.15 | 0.15 | -0.00 |
| **sum** | **12.87** | **12.89** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.38 | 20.36 | -0.01 |
| shared expert | 4.89 | 4.92 | +0.02 |
| hyper-connections (BF16) | 12.96 | 12.97 | +0.01 |
| router | 2.40 | 2.40 | -0.00 |
| attention and recurrent projections | 34.09 | 34.08 | -0.01 |
| Gated DeltaNet state and conv | 8.90 | 8.90 | -0.00 |
| attention (FLASH_ATTN_EXT) | 1.04 | 1.02 | -0.01 |
| output layer | 8.10 | 8.10 | +0.01 |
| other ops without weights | 7.17 | 7.16 | -0.01 |
| other weights | 0.89 | 0.89 | -0.01 |
| **sum** | **100.82** | **100.80** | **-0.02** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.65 | 8.65 | -0.00 |
| GDN_CONV | 0.72 | 0.74 | +0.02 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.52 | 19.51 | -0.01 |
| shared expert, router, hyper-connections | 19.32 | 19.35 | +0.03 |
| everything else | 39.67 | 39.60 | -0.07 |
| **sum** | **87.95** | **87.91** | **-0.04** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.10 | 8.10 | +0.01 |
| the sampling chain after the product | 0.65 | 0.64 | -0.01 |
| the output's hyper-connection products | 0.11 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | -0.00 |
| routed experts | 0.85 | 0.86 | +0.00 |
| shared expert, router, hyper-connections | 0.94 | 0.94 | +0.00 |
| everything else | 1.95 | 1.96 | +0.01 |
| **sum** | **12.87** | **12.89** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 204 launch groups, verify-r2 80 of 80, 204 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.13 | 4.14 | +0.01 |
| the head's product | 8.10 | 8.10 | +0.01 |
| after the head's product | 0.65 | 0.64 | -0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.83 (134) | 0.78 (134) | -0.05 |
| SOFTPLUS | 0.17 (35) | 0.21 (35) | +0.04 |
| RMS_NORM | 0.98 (174) | 0.95 (174) | -0.03 |
| GET_ROWS | 0.19 (23) | 0.17 (23) | -0.03 |
| ARANGE | 0.00 (1) | 0.03 (1) | +0.03 |
| SCALE | 1.47 (197) | 1.49 (197) | +0.02 |
| GDN_CONV | 0.72 (35) | 0.74 (35) | +0.02 |
| MUL_MAT_ID | 19.52 (92) | 19.51 (92) | -0.01 |
| FLASH_ATTN_EXT | 0.95 (11) | 0.94 (11) | -0.01 |
| SILU | 0.19 (93) | 0.20 (93) | +0.01 |
| ADD | 0.38 (119) | 0.37 (119) | -0.01 |
| DSV4_HC_PRE | 0.50 (92) | 0.51 (92) | +0.01 |
| CONCAT | 0.13 (12) | 0.14 (12) | +0.01 |
| MUL_MAT | 50.97 (737) | 50.97 (737) | -0.01 |
| the other 18 ops | 10.95 | 10.93 | -0.02 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SOFTPLUS | 0.04 (1) | 0.01 (1) | -0.02 |
| SIGMOID | 0.03 (5) | 0.04 (5) | +0.01 |
| GET_ROWS | 0.07 (27) | 0.08 (27) | +0.01 |
| MUL | 0.08 (5) | 0.09 (5) | +0.01 |
| TOP_K | 0.30 (5) | 0.31 (5) | +0.01 |
| MUL_MAT | 10.53 (36) | 10.53 (36) | +0.01 |
| ARGMAX | 0.02 (8) | 0.01 (8) | -0.01 |
| LOG | 0.01 (4) | 0.00 (4) | -0.00 |
| DSV4_HC_PRE | 0.02 (4) | 0.02 (4) | +0.00 |
| CLAMP | 0.00 (4) | 0.00 (4) | -0.00 |
| ADD | 0.04 (13) | 0.03 (13) | -0.00 |
| SET_ROWS | 0.02 (8) | 0.02 (8) | -0.00 |
| MUL_MAT_ID | 0.85 (4) | 0.86 (4) | +0.00 |
| SILU | 0.01 (5) | 0.01 (5) | +0.00 |
| the other 19 ops | 0.86 | 0.86 | -0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.103 |
| 118 | `PAD node_6219` | 0.216 |
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
| 137 | `GET_ROWS node_6251` | 0.002 |
| 138 | `ARGMAX greedy_argmax` | 0.002 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.002 |
| 140 | `TOP_K top_k` | 0.072 |
| 141 | `GET_ROWS top_k_rows` | 0.002 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 143 | `GET_ROWS top_p_candidates` | 0.001 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.001 |
| 147 | `SCALE top_p_cdf_scaled` | 0.001 |
| 148 | `STEP top_p_mask` | 0.000 |
| 149 | `SUM top_p_index_f32` | 0.001 |
| 150 | `CLAMP node_6273` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.000 |
| 152 | `CPY _(copy)` | 0.000 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 154 | `LOG top_p_bias` | 0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | 0.000 |
| 157 | `GET_ROWS node_6283` | 0.001 |
| 158 | `GET_ROWS node_6286` | 0.001 |
| 159 | `ARGMAX greedy_argmax` | -0.000 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.070 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 164 | `GET_ROWS top_p_candidates` | 0.001 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 166 | `SOFT_MAX top_p_softmax` | 0.001 |
| 167 | `CUMSUM top_p_cdf` | -0.000 |
| 168 | `SCALE top_p_cdf_scaled` | 0.000 |
| 169 | `STEP top_p_mask` | 0.001 |
| 170 | `SUM top_p_index_f32` | 0.003 |
| 171 | `CLAMP node_6308` | 0.000 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.001 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 175 | `LOG top_p_bias` | 0.000 |
| 176 | `ADD top_p_logits` | 0.002 |
| 177 | `ARGMAX temp_max_idx` | 0.003 |
| 178 | `GET_ROWS node_6318` | 0.002 |
| 179 | `GET_ROWS node_6321` | 0.007 |
| 180 | `ARGMAX greedy_argmax` | 0.003 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.008 |
| 182 | `TOP_K top_k` | 0.079 |
| 183 | `GET_ROWS top_k_rows` | 0.002 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 185 | `GET_ROWS top_p_candidates` | 0.003 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 187 | `SOFT_MAX top_p_softmax` | 0.002 |
| 188 | `CUMSUM top_p_cdf` | 0.002 |
| 189 | `SCALE top_p_cdf_scaled` | 0.001 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | 0.000 |
| 192 | `CLAMP node_6343` | -0.000 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.39 | 12.37 | -0.02 |
| 1 | 11.16 | 11.14 | -0.02 |
| 2 | 11.31 | 11.30 | -0.01 |
| 3 | 11.44 | 11.44 | -0.00 |
| 4 | 9.26 | 9.28 | +0.03 |
| 5 | 11.41 | 11.41 | -0.00 |
| 6 | 9.55 | 9.54 | -0.01 |
| 7 | 11.43 | 11.44 | +0.01 |
| 8 | 12.87 | 12.89 | +0.01 |
| **all** | **100.82** | **100.80** | **-0.02** |

