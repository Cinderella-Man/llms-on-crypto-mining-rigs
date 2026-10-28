traces: rel/bugfix-javascript-complex-numbers/a1/t1, rel/bugfix-javascript-phone-number/a1/t1, rel/bugfix-javascript-space-age/a1/t1, rel/bugfix-python-affine-cipher/a1/t1, rel/bugfix-python-list-ops/a1/t1, rel/bugfix-python-scale-generator/a1/t1

| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | TTFT p50 s | wall min | same-work min |
|---|---|---|---|---|---|---|---|---|
| q8 | 36 | 5.14 | 22.92 | 0.15 | 4443 | 4.29 | 11.8 | 11.8 |
| p065 | 36 | 5.15 | 21.58 | 0.17 | 4443 | 4.25 | 11.6 | 11.4 |
| live:rel | 36 | 5.32 | 22.02 | 0.15 | 4788 | 2.84 | 13.0 | 11.6 |

Ratios to q8 (95 % bootstrap over traces, 2000 resamples):

| label | ms per prompt token | ms per generated token | same-work minutes |
|---|---|---|---|
| p065 | 1.002 (0.997–1.008) | 0.941 (0.930–0.980) | 0.963 (0.950–0.992) |
| live:rel | 1.034 (0.956–1.103) | 0.961 (0.923–0.985) | 0.985 (0.969–1.002) |
