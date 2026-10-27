12 chunks, 49140 scored tokens; reference for differences: ud; dump reference: BF16 (its log p agrees across the dumps to 0.0e+00 nats)

| Positions | All | produced by g3 | produced by ud-q4kxl |
|---|---|---|---|
| stops (</think> closing a non-empty block) | 132 in 12 chunks | 62 | 70 |
| empty blocks (left out) | 0 in 0 chunks | 0 | 0 |
| hesitation markers in reasoning | 663 in 11 chunks | 347 | 316 |

Stops: log p of the actual token

| File | mean log p | mean p | log p - ud (95 % bootstrap over chunks) | produced by g3 | produced by ud-q4kxl | log p - BF16 |
|---|---|---|---|---|---|---|
| ud | -0.1834 | 0.8805 |  |  |  | -0.0020 (-0.0320 to +0.0278) |
| q8 | -0.1898 | 0.8754 | -0.0064 (-0.0377 to +0.0231) | -0.0272 (-0.0663 to +0.0052) | +0.0119 (-0.0168 to +0.0563) | -0.0084 (-0.0145 to -0.0025) |
| g3 | -0.1959 | 0.8741 | -0.0125 (-0.0668 to +0.0236) | -0.0434 (-0.1266 to +0.0205) | +0.0149 (-0.0182 to +0.0466) | -0.0145 (-0.0443 to +0.0044) |
| g3g | -0.2020 | 0.8710 | -0.0187 (-0.0691 to +0.0146) | -0.0496 (-0.1320 to +0.0129) | +0.0087 (-0.0229 to +0.0323) | -0.0207 (-0.0496 to +0.0007) |
| k2h | -0.1860 | 0.8801 | -0.0026 (-0.0550 to +0.0315) | -0.0263 (-0.1136 to +0.0423) | +0.0183 (-0.0005 to +0.0452) | -0.0046 (-0.0363 to +0.0194) |
| k2hg | -0.1892 | 0.8767 | -0.0058 (-0.0532 to +0.0285) | -0.0379 (-0.1090 to +0.0220) | +0.0226 (-0.0039 to +0.0530) | -0.0078 (-0.0328 to +0.0094) |

Hesitation markers: log p of the actual token

| File | mean log p | mean p | log p - ud (95 % bootstrap over chunks) | produced by g3 | produced by ud-q4kxl | log p - BF16 |
|---|---|---|---|---|---|---|
| ud | -0.7168 | 0.6342 |  |  |  | +0.0008 (-0.0139 to +0.0119) |
| q8 | -0.7144 | 0.6338 | +0.0023 (-0.0090 to +0.0150) | +0.0132 (-0.0026 to +0.0295) | -0.0096 (-0.0467 to -0.0029) | +0.0031 (-0.0053 to +0.0092) |
| g3 | -0.7167 | 0.6348 | +0.0001 (-0.0157 to +0.0195) | +0.0177 (-0.0020 to +0.0328) | -0.0192 (-0.0341 to -0.0060) | +0.0009 (-0.0106 to +0.0146) |
| g3g | -0.7172 | 0.6346 | -0.0004 (-0.0149 to +0.0172) | +0.0155 (-0.0038 to +0.0303) | -0.0178 (-0.0646 to -0.0018) | +0.0004 (-0.0123 to +0.0120) |
| k2h | -0.7282 | 0.6306 | -0.0115 (-0.0292 to +0.0080) | +0.0099 (-0.0050 to +0.0234) | -0.0349 (-0.0902 to -0.0201) | -0.0106 (-0.0221 to +0.0012) |
| k2hg | -0.7339 | 0.6287 | -0.0172 (-0.0291 to -0.0040) | -0.0038 (-0.0185 to +0.0066) | -0.0318 (-0.0669 to -0.0281) | -0.0163 (-0.0264 to -0.0113) |

Kill rule: |log p - ud| < 0.05 nats with an interval covering 0, at stops and markers, for every file

- q8: stops -0.0064 (-0.0377 to +0.0231); hesitation markers +0.0023 (-0.0090 to +0.0150) -> within
- g3: stops -0.0125 (-0.0668 to +0.0236); hesitation markers +0.0001 (-0.0157 to +0.0195) -> within
- g3g: stops -0.0187 (-0.0691 to +0.0146); hesitation markers -0.0004 (-0.0149 to +0.0172) -> within
- k2h: stops -0.0026 (-0.0550 to +0.0315); hesitation markers -0.0115 (-0.0292 to +0.0080) -> within
- k2hg: stops -0.0058 (-0.0532 to +0.0285); hesitation markers -0.0172 (-0.0291 to -0.0040) -> outside

Quant choice as a reasoning-length lever: open
