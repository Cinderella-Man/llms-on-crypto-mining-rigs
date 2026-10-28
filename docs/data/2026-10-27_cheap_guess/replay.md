traces: rel/bugfix-javascript-complex-numbers/a1/t1, rel/bugfix-javascript-phone-number/a1/t1, rel/bugfix-javascript-space-age/a1/t1, rel/bugfix-python-affine-cipher/a1/t1, rel/bugfix-python-list-ops/a1/t1, rel/bugfix-python-scale-generator/a1/t1

| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | TTFT p50 s | wall min | same-work min |
|---|---|---|---|---|---|---|---|---|
| base | 36 | 5.13 | 22.85 | 0.16 | 4443 | 4.30 | 11.8 | 11.8 |
| stair | 36 | 5.46 | 23.59 | 0.15 | 4443 | 4.37 | 12.3 | 12.3 |
| live:rel | 36 | 5.32 | 22.02 | 0.15 | 4788 | 2.84 | 13.0 | 11.6 |

Ratios to base (95 % bootstrap over traces, 2000 resamples):

| label | ms per prompt token | ms per generated token | same-work minutes |
|---|---|---|---|
| stair | 1.063 (1.030–1.152) | 1.032 (1.028–1.047) | 1.042 (1.029–1.089) |
| live:rel | 1.036 (0.957–1.101) | 0.963 (0.930–0.988) | 0.987 (0.972–1.004) |
