2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.86 | 156.47 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 118.92 | 159.36 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| routed experts | 19.27 | 19.54 | 19.52 | +0.27 | +0.25 |
| shared expert | 4.66 | 4.66 | 4.63 | -0.00 | -0.03 |
| hyper-connections (BF16) | 12.33 | 12.40 | 12.40 | +0.07 | +0.07 |
| router | 2.30 | 2.29 | 2.30 | -0.00 | +0.00 |
| attention and recurrent projections | 32.72 | 32.66 | 32.65 | -0.05 | -0.07 |
| Gated DeltaNet state and conv | 3.68 | 8.66 | 8.65 | +4.97 | +4.97 |
| attention (FLASH_ATTN_EXT) | 0.93 | 0.95 | 0.94 | +0.02 | +0.01 |
| other ops without weights | 5.86 | 6.23 | 6.28 | +0.37 | +0.42 |
| other weights | 0.74 | 0.74 | 0.74 | +0.00 | +0.01 |
| **sum** | **82.48** | **88.13** | **88.13** | **+5.64** | **+5.65** |

**GPU time of a pass by part, ms: card 8**

| Part | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| routed experts | 0.41 | 0.85 | 0.86 | +0.45 | +0.45 |
| shared expert | 0.13 | 0.27 | 0.28 | +0.14 | +0.14 |
| hyper-connections (BF16) | 0.40 | 0.56 | 0.56 | +0.15 | +0.16 |
| router | 0.05 | 0.10 | 0.10 | +0.05 | +0.05 |
| attention and recurrent projections | 1.41 | 1.43 | 1.42 | +0.02 | +0.01 |
| Gated DeltaNet state and conv | 0.11 | 0.25 | 0.25 | +0.14 | +0.14 |
| attention (FLASH_ATTN_EXT) | 0.08 | 0.09 | 0.09 | +0.00 | +0.00 |
| output layer | 0.00 | 8.05 | 8.06 | +8.05 | +8.06 |
| other ops without weights | 0.24 | 1.09 | 1.10 | +0.85 | +0.86 |
| other weights | 0.04 | 0.15 | 0.15 | +0.11 | +0.11 |
| **sum** | **2.88** | **12.86** | **12.87** | **+9.98** | **+9.99** |

**GPU time of a pass by part, ms: all cards**

| Part | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| routed experts | 19.68 | 20.39 | 20.38 | +0.71 | +0.69 |
| shared expert | 4.79 | 4.93 | 4.91 | +0.14 | +0.12 |
| hyper-connections (BF16) | 12.73 | 12.96 | 12.96 | +0.23 | +0.23 |
| router | 2.35 | 2.40 | 2.40 | +0.05 | +0.05 |
| attention and recurrent projections | 34.13 | 34.09 | 34.07 | -0.04 | -0.06 |
| Gated DeltaNet state and conv | 3.79 | 8.91 | 8.91 | +5.12 | +5.12 |
| attention (FLASH_ATTN_EXT) | 1.01 | 1.04 | 1.03 | +0.03 | +0.02 |
| output layer | 0.00 | 8.05 | 8.06 | +8.05 | +8.06 |
| other ops without weights | 6.10 | 7.33 | 7.38 | +1.22 | +1.28 |
| other weights | 0.78 | 0.89 | 0.90 | +0.11 | +0.12 |
| **sum** | **85.36** | **100.99** | **101.00** | **+15.63** | **+15.64** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| GATED_DELTA_NET | 3.68 | 8.66 | 8.65 | +4.97 | +4.97 |
| GDN_CONV | 0.47 | 0.72 | 0.74 | +0.25 | +0.27 |
| copies of the PLE conv history (cache_ple_r) | 0.05 | 0.19 | 0.19 | +0.15 | +0.15 |
| routed experts | 19.27 | 19.54 | 19.52 | +0.27 | +0.25 |
| shared expert, router, hyper-connections | 19.29 | 19.35 | 19.34 | +0.06 | +0.05 |
| everything else | 39.72 | 39.66 | 39.68 | -0.06 | -0.04 |
| **sum** | **82.48** | **88.13** | **88.13** | **+5.64** | **+5.65** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| the head's product | 0.00 | 8.05 | 8.06 | +8.05 | +8.06 |
| the sampling chain after the product | 0.00 | 0.68 | 0.68 | +0.68 | +0.68 |
| the output's hyper-connection products | 0.00 | 0.11 | 0.11 | +0.11 | +0.11 |
| GATED_DELTA_NET | 0.11 | 0.25 | 0.25 | +0.14 | +0.14 |
| GDN_CONV | 0.01 | 0.02 | 0.02 | +0.01 | +0.01 |
| routed experts | 0.41 | 0.85 | 0.86 | +0.45 | +0.45 |
| shared expert, router, hyper-connections | 0.58 | 0.93 | 0.94 | +0.35 | +0.35 |
| everything else | 1.77 | 1.96 | 1.96 | +0.20 | +0.19 |
| **sum** | **2.88** | **12.86** | **12.87** | **+9.98** | **+9.99** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: plain 82 of 82, 84 launch groups, verify-r1 81 of 81, 203 launch groups, verify-r2 80 of 80, 203 launch groups)

| Position | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| before the head's product | 2.88 | 4.13 | 4.13 | +1.25 | +1.25 |
| the head's product | 0.00 | 8.05 | 8.06 | +8.05 | +8.06 |
| after the head's product | 0.00 | 0.68 | 0.68 | +0.68 | +0.68 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| GATED_DELTA_NET | 3.68 (35) | 8.66 (35) | 8.65 (35) | +4.97 | +4.97 |
| GDN_CONV | 0.47 (35) | 0.72 (35) | 0.74 (35) | +0.25 | +0.27 |
| MUL_MAT_ID | 19.27 (92) | 19.54 (92) | 19.52 (92) | +0.27 | +0.25 |
| CPY | 0.34 (20) | 0.48 (23) | 0.48 (23) | +0.14 | +0.14 |
| MUL | 0.79 (134) | 0.86 (134) | 0.86 (134) | +0.07 | +0.08 |
| ADD | 0.44 (119) | 0.39 (119) | 0.37 (119) | -0.05 | -0.07 |
| SOFTPLUS | 0.18 (35) | 0.20 (35) | 0.22 (35) | +0.03 | +0.05 |
| SIGMOID | 0.26 (93) | 0.21 (93) | 0.22 (93) | -0.05 | -0.04 |
| GET_ROWS | 0.17 (23) | 0.18 (23) | 0.20 (23) | +0.02 | +0.03 |
| RMS_NORM | 0.94 (174) | 0.95 (174) | 0.96 (174) | +0.01 | +0.03 |
| SWIGLU | 0.10 (40) | 0.08 (40) | 0.08 (40) | -0.02 | -0.03 |
| MUL_MAT | 51.00 (737) | 51.00 (737) | 50.98 (737) | +0.00 | -0.02 |
| SILU | 0.20 (93) | 0.19 (93) | 0.18 (93) | -0.01 | -0.02 |
| REPEAT | 0.02 (2) | 0.03 (2) | 0.03 (2) | +0.01 | +0.02 |
| the other 17 ops | 4.63 | 4.64 | 4.63 | +0.01 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| MUL_MAT | 1.99 (25) | 10.49 (36) | 10.49 (36) | +8.50 | +8.50 |
| MUL_MAT_ID | 0.41 (2) | 0.85 (4) | 0.86 (4) | +0.45 | +0.45 |
| TOP_K | 0.01 (1) | 0.34 (5) | 0.34 (5) | +0.33 | +0.33 |
| PAD | 0.00 (0) | 0.22 (1) | 0.22 (1) | +0.22 | +0.22 |
| GATED_DELTA_NET | 0.11 (1) | 0.25 (1) | 0.25 (1) | +0.14 | +0.14 |
| GET_ROWS | 0.01 (2) | 0.07 (27) | 0.07 (27) | +0.06 | +0.06 |
| SCALE | 0.03 (7) | 0.09 (19) | 0.09 (19) | +0.06 | +0.05 |
| SIGMOID | 0.01 (3) | 0.04 (5) | 0.05 (5) | +0.03 | +0.03 |
| ADD | 0.01 (5) | 0.04 (13) | 0.03 (13) | +0.03 | +0.03 |
| RMS_NORM | 0.03 (8) | 0.05 (10) | 0.05 (10) | +0.02 | +0.03 |
| SET_ROWS | 0.00 (3) | 0.02 (7) | 0.02 (7) | +0.02 | +0.02 |
| SOFT_MAX | 0.01 (1) | 0.03 (6) | 0.03 (6) | +0.02 | +0.02 |
| ARGMAX | 0.00 (0) | 0.01 (8) | 0.01 (8) | +0.01 | +0.01 |
| ROPE | 0.00 (4) | 0.01 (4) | 0.01 (4) | +0.01 | +0.01 |
| the other 19 ops | 0.25 | 0.35 | 0.35 | +0.09 | +0.09 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6219` | 0.003 |
| 115 | `ADD node_6221` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.065 |
| 118 | `PAD node_6224` | 0.216 |
| 119 | `TOP_K top_k` | 0.085 |
| 120 | `GET_ROWS top_k_rows` | 0.005 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 122 | `GET_ROWS top_p_candidates` | 0.005 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 124 | `SOFT_MAX top_p_softmax` | 0.003 |
| 125 | `CUMSUM top_p_cdf` | 0.003 |
| 126 | `SCALE top_p_cdf_scaled` | 0.003 |
| 127 | `STEP top_p_mask` | 0.002 |
| 128 | `SUM top_p_index_f32` | 0.002 |
| 129 | `CLAMP node_6243` | 0.003 |
| 130 | `SCALE top_p_ones` | 0.002 |
| 131 | `CPY _(copy)` | 0.002 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.002 |
| 133 | `LOG top_p_bias` | 0.002 |
| 134 | `ADD top_p_logits` | 0.003 |
| 135 | `ARGMAX temp_max_idx` | 0.002 |
| 136 | `GET_ROWS node_6253` | 0.002 |
| 137 | `GET_ROWS node_6256` | 0.002 |
| 138 | `ARGMAX greedy_argmax` | 0.002 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.002 |
| 140 | `TOP_K top_k` | 0.082 |
| 141 | `GET_ROWS top_k_rows` | 0.001 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 143 | `GET_ROWS top_p_candidates` | 0.001 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.000 |
| 147 | `SCALE top_p_cdf_scaled` | 0.000 |
| 148 | `STEP top_p_mask` | 0.000 |
| 149 | `SUM top_p_index_f32` | 0.000 |
| 150 | `CLAMP node_6278` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.000 |
| 152 | `CPY _(copy)` | 0.000 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 154 | `LOG top_p_bias` | -0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | 0.000 |
| 157 | `GET_ROWS node_6288` | 0.001 |
| 158 | `GET_ROWS node_6291` | 0.000 |
| 159 | `ARGMAX greedy_argmax` | 0.000 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.079 |
| 162 | `GET_ROWS top_k_rows` | 0.001 |
| 163 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 164 | `GET_ROWS top_p_candidates` | 0.000 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.002 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | 0.001 |
| 169 | `STEP top_p_mask` | 0.003 |
| 170 | `SUM top_p_index_f32` | 0.000 |
| 171 | `CLAMP node_6313` | 0.002 |
| 172 | `SCALE top_p_ones` | 0.003 |
| 173 | `CPY _(copy)` | 0.005 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.010 |
| 175 | `LOG top_p_bias` | 0.003 |
| 176 | `ADD top_p_logits` | 0.005 |
| 177 | `ARGMAX temp_max_idx` | 0.004 |
| 178 | `GET_ROWS node_6323` | 0.008 |
| 179 | `GET_ROWS node_6326` | 0.007 |
| 180 | `ARGMAX greedy_argmax` | 0.004 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.005 |
| 182 | `TOP_K top_k` | 0.082 |
| 183 | `GET_ROWS top_k_rows` | 0.001 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 185 | `GET_ROWS top_p_candidates` | 0.001 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 187 | `SOFT_MAX top_p_softmax` | 0.000 |
| 188 | `CUMSUM top_p_cdf` | 0.001 |
| 189 | `SCALE top_p_cdf_scaled` | 0.000 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | -0.000 |
| 192 | `CLAMP node_6348` | -0.000 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | 0.000 |
| 197 | `ADD top_p_logits` | -0.000 |
| 198 | `ARGMAX temp_max_idx` | 0.000 |
| 199 | `GET_ROWS node_6358` | 0.001 |
| 200 | `GET_ROWS node_6361` | 0.001 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | plain (82 passes) | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r1 − plain | verify-r2 − plain |
|---|---:|---:|---:|---:|---:|
| 0 | 11.56 | 12.49 | 12.49 | +0.93 | +0.93 |
| 1 | 10.45 | 11.18 | 11.19 | +0.73 | +0.74 |
| 2 | 10.54 | 11.37 | 11.37 | +0.82 | +0.82 |
| 3 | 10.75 | 11.44 | 11.44 | +0.69 | +0.69 |
| 4 | 8.64 | 9.26 | 9.25 | +0.62 | +0.61 |
| 5 | 10.70 | 11.40 | 11.41 | +0.71 | +0.71 |
| 6 | 9.12 | 9.56 | 9.55 | +0.44 | +0.43 |
| 7 | 10.72 | 11.43 | 11.44 | +0.71 | +0.72 |
| 8 | 2.88 | 12.86 | 12.87 | +9.98 | +9.99 |
| **all** | **85.36** | **100.99** | **101.00** | **+15.63** | **+15.64** |

