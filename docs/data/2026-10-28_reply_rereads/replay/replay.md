traces: rel/bugfix-javascript-complex-numbers/a1/t1, rel/bugfix-javascript-phone-number/a1/t1, rel/bugfix-javascript-space-age/a1/t1, rel/bugfix-python-affine-cipher/a1/t1, rel/bugfix-python-list-ops/a1/t1, rel/bugfix-python-scale-generator/a1/t1

| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | TTFT p50 s | wall min | same-work min |
|---|---|---|---|---|---|---|---|---|
| base | 36 | 5.13 | 22.72 | 0.15 | 4443 | 4.26 | 11.7 | 11.7 |
| mask | 36 | 5.26 | 22.81 | 0.15 | 4443 | 4.34 | 11.9 | 11.9 |
| live:rel | 36 | 5.32 | 22.02 | 0.15 | 4788 | 2.84 | 13.0 | 11.6 |

Ratios to base (95 % bootstrap over traces, 2000 resamples):

| label | ms per prompt token | ms per generated token | same-work minutes |
|---|---|---|---|
| mask | 1.025 (1.002–1.093) | 1.004 (1.002–1.013) | 1.011 (1.003–1.048) |
| live:rel | 1.037 (0.959–1.105) | 0.969 (0.935–0.993) | 0.992 (0.975–1.008) |
