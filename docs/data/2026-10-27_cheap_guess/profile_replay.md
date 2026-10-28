traces: rel/bugfix-javascript-complex-numbers/a1/t1, rel/bugfix-javascript-phone-number/a1/t1, rel/bugfix-javascript-space-age/a1/t1, rel/bugfix-python-affine-cipher/a1/t1, rel/bugfix-python-list-ops/a1/t1

| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | TTFT p50 s | wall min | same-work min |
|---|---|---|---|---|---|---|---|---|
| base | 29 | 5.57 | 49.38 | 0.15 | 3604 | 4.65 | 8.6 | 8.6 |
| taps | 29 | 8.46 | 51.16 | 0.20 | 3604 | 6.33 | 10.2 | 10.2 |
| stair | 29 | 8.42 | 53.30 | 0.17 | 3604 | 6.32 | 10.4 | 10.4 |
| live:rel | 29 | 5.18 | 21.34 | 0.15 | 3934 | 2.74 | 5.8 | 5.0 |

Ratios to base (95 % bootstrap over traces, 2000 resamples):

| label | ms per prompt token | ms per generated token | same-work minutes |
|---|---|---|---|
| taps | 1.519 (1.459–1.585) | 1.036 (1.005–1.081) | 1.178 (1.156–1.205) |
| stair | 1.513 (1.448–1.584) | 1.079 (1.041–1.124) | 1.205 (1.179–1.231) |
| live:rel | 0.930 (0.845–1.007) | 0.432 (0.392–0.484) | 0.581 (0.540–0.640) |
