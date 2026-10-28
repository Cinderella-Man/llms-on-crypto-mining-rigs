traces: rel/bugfix-javascript-complex-numbers/a1/t1, rel/bugfix-javascript-phone-number/a1/t1, rel/bugfix-javascript-space-age/a1/t1, rel/bugfix-python-affine-cipher/a1/t1, rel/bugfix-python-list-ops/a1/t1, rel/bugfix-python-scale-generator/a1/t1

| label | requests | ms per prompt token | ms per generated token | fixed s per request | mean cache_n | TTFT p50 s | wall min | same-work min |
|---|---|---|---|---|---|---|---|---|
| store | 36 | 5.12 | 22.55 | 0.17 | 4443 | 4.32 | 11.7 | 11.7 |
| base | 36 | 5.13 | 22.87 | 0.16 | 4443 | 4.28 | 11.8 | 11.8 |
| live:rel | 36 | 5.32 | 22.02 | 0.15 | 4788 | 2.84 | 13.0 | 11.6 |

Ratios to base (95 % bootstrap over traces, 2000 resamples):

| label | ms per prompt token | ms per generated token | same-work minutes |
|---|---|---|---|
| store | 0.998 (0.994–1.001) | 0.986 (0.981–0.996) | 0.991 (0.988–0.996) |
| live:rel | 1.037 (0.959–1.100) | 0.963 (0.931–0.987) | 0.987 (0.973–1.005) |
