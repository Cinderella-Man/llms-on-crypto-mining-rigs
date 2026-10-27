12 chunks, 49140 scored tokens; reference for differences: ud; dump reference: Q8_0 (its log p agrees across the dumps to 0.0e+00 nats)

| Positions | All | produced by g3 | produced by ud-q4kxl |
|---|---|---|---|
| stops (</think> closing a non-empty block) | 132 in 12 chunks | 62 | 70 |
| empty blocks (left out) | 0 in 0 chunks | 0 | 0 |
| hesitation markers in reasoning | 663 in 11 chunks | 347 | 316 |

Stops: log p of the actual token

| File | mean log p | mean p | log p - ud (95 % bootstrap over chunks) | produced by g3 | produced by ud-q4kxl | log p - Q8_0 |
|---|---|---|---|---|---|---|
| ud | -0.1834 | 0.8805 |  |  |  | +0.0064 (-0.0238 to +0.0383) |
| g3 | -0.1959 | 0.8741 | -0.0125 (-0.0666 to +0.0238) | -0.0434 (-0.1256 to +0.0202) | +0.0149 (-0.0174 to +0.0469) | -0.0060 (-0.0316 to +0.0120) |
| g3g | -0.2020 | 0.8710 | -0.0187 (-0.0706 to +0.0147) | -0.0496 (-0.1286 to +0.0121) | +0.0087 (-0.0236 to +0.0314) | -0.0122 (-0.0370 to +0.0084) |
| k2h | -0.1860 | 0.8801 | -0.0026 (-0.0548 to +0.0317) | -0.0263 (-0.1185 to +0.0431) | +0.0183 (-0.0005 to +0.0451) | +0.0038 (-0.0220 to +0.0258) |
| k2hg | -0.1892 | 0.8767 | -0.0058 (-0.0545 to +0.0283) | -0.0379 (-0.1088 to +0.0204) | +0.0226 (-0.0042 to +0.0544) | +0.0006 (-0.0200 to +0.0166) |

Hesitation markers: log p of the actual token

| File | mean log p | mean p | log p - ud (95 % bootstrap over chunks) | produced by g3 | produced by ud-q4kxl | log p - Q8_0 |
|---|---|---|---|---|---|---|
| ud | -0.7168 | 0.6342 |  |  |  | -0.0023 (-0.0151 to +0.0093) |
| g3 | -0.7167 | 0.6348 | +0.0001 (-0.0158 to +0.0203) | +0.0177 (-0.0020 to +0.0318) | -0.0192 (-0.0341 to -0.0060) | -0.0022 (-0.0147 to +0.0162) |
| g3g | -0.7172 | 0.6346 | -0.0004 (-0.0150 to +0.0160) | +0.0155 (-0.0030 to +0.0303) | -0.0178 (-0.0646 to -0.0045) | -0.0027 (-0.0136 to +0.0138) |
| k2h | -0.7282 | 0.6306 | -0.0115 (-0.0290 to +0.0083) | +0.0099 (-0.0048 to +0.0227) | -0.0349 (-0.0902 to -0.0201) | -0.0138 (-0.0287 to +0.0036) |
| k2hg | -0.7339 | 0.6287 | -0.0172 (-0.0294 to -0.0042) | -0.0038 (-0.0201 to +0.0066) | -0.0318 (-0.0649 to -0.0280) | -0.0195 (-0.0279 to -0.0098) |

Kill rule: |log p - ud| < 0.05 nats with an interval covering 0, at stops and markers, for every file

- g3: stops -0.0125 (-0.0666 to +0.0238); hesitation markers +0.0001 (-0.0158 to +0.0203) -> within
- g3g: stops -0.0187 (-0.0706 to +0.0147); hesitation markers -0.0004 (-0.0150 to +0.0160) -> within
- k2h: stops -0.0026 (-0.0548 to +0.0317); hesitation markers -0.0115 (-0.0290 to +0.0083) -> within
- k2hg: stops -0.0058 (-0.0545 to +0.0283); hesitation markers -0.0172 (-0.0294 to -0.0042) -> outside

Quant choice as a reasoning-length lever: open
