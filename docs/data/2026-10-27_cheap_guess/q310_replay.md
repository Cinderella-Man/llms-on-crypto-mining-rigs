traces: rel/bugfix-javascript-complex-numbers/a1/t1, rel/bugfix-javascript-phone-number/a1/t1, rel/bugfix-javascript-space-age/a1/t1, rel/bugfix-python-affine-cipher/a1/t1, rel/bugfix-python-list-ops/a1/t1, rel/bugfix-python-scale-generator/a1/t1

| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | TTFT p50 s | wall min | same-work min |
|---|---|---|---|---|---|---|---|---|
| base | 36 | 0.43 | 8.09 | 0.03 | 4442 | 0.40 | 2.6 | 2.6 |
| gen | 36 | 0.56 | 8.33 | 0.03 | 4442 | 0.50 | 2.8 | 2.8 |
| nine | 36 | 0.55 | 9.04 | 0.03 | 4442 | 0.47 | 3.0 | 3.0 |
| live:rel | 36 | 5.32 | 22.02 | 0.15 | 4788 | 2.84 | 13.0 | 10.3 |

Ratios to base (95 % bootstrap over traces, 2000 resamples):

| label | ms per prompt token | ms per generated token | same-work minutes |
|---|---|---|---|
| gen | 1.315 (1.230–1.369) | 1.031 (1.020–1.038) | 1.065 (1.053–1.073) |
| nine | 1.295 (1.205–1.353) | 1.118 (1.110–1.134) | 1.138 (1.133–1.148) |
| live:rel | 12.442 (10.429–13.656) | 2.723 (2.604–2.766) | 3.925 (3.619–4.350) |
PAIR_CHECK base vs gen: 36 requests, 0 differ; 17036 generated tokens in base
PAIR_CHECK base vs nine: 36 requests, 0 differ; 17036 generated tokens in base
