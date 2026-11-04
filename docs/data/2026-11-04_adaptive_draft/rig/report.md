### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | nodraft: result, min, turns, tool calls | dflash2b: result, min, turns, tool calls | fix2: result, min, turns, tool calls | adapt5: result, min, turns, tool calls | time nodraft / dflash2b | time nodraft / fix2 | time nodraft / adapt5 |
|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 8.2, 8, 8 / 1 / 0 | pass, 6.4, 7, 7 / 1 / 0 | pass, 5.1, 8, 8 / 2 / 0 | pass, 5.3, 8, 8 / 1 / 0 | 1.29× | 1.61× | 1.56× |
| bugfix-javascript-space-age | pass, 4.1, 6, 5 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | 2.37× | 2.44× | 2.77× |
| bugfix-python-phone-number | pass, 3.0, 5, 5 / 0 / 0 | pass, 2.1, 5, 4 / 0 / 0 | pass, 2.5, 5, 5 / 0 / 0 | pass, 2.1, 6, 5 / 0 / 0 | 1.42× | 1.22× | 1.45× |
| bugfix-javascript-triangle | pass, 2.4, 5, 4 / 0 / 0 | pass, 2.2, 5, 4 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | pass, 1.6, 4, 3 / 0 / 0 | 1.13× | 1.42× | 1.52× |
| bugfix-python-variable-length-quantity | pass, 6.9, 7, 7 / 1 / 0 | pass, 6.8, 6, 6 / 1 / 0 | pass, 5.9, 5, 5 / 0 / 0 | pass, 2.9, 5, 6 / 0 / 0 | 1.02× | 1.17× | 2.43× |
| bugfix-javascript-phone-number | pass, 2.5, 5, 4 / 0 / 0 | pass, 2.9, 5, 4 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | pass, 1.9, 5, 4 / 0 / 0 | 0.85× | 1.09× | 1.31× |
| bugfix-python-list-ops | pass, 4.3, 5, 5 / 0 / 0 | pass, 7.0, 5, 5 / 0 / 0 | pass, 10.9, 5, 5 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | 0.61× | 0.39× | 1.61× |
| bugfix-javascript-grade-school | pass, 2.8, 5, 4 / 0 / 0 | pass, 2.1, 5, 4 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | 1.35× | 1.41× | 1.43× |
| bugfix-python-scale-generator | pass, 8.7, 5, 5 / 0 / 0 | pass, 3.6, 5, 5 / 0 / 0 | pass, 4.5, 5, 5 / 0 / 0 | pass, 4.6, 5, 5 / 0 / 0 | 2.40× | 1.93× | 1.92× |
| bugfix-javascript-complex-numbers | — | pass, 2.4, 7, 6 / 0 / 0 | pass, 5.2, 8, 8 / 0 / 0 | pass, 5.7, 6, 5 / 0 / 0 | — | — | — |
| bugfix-python-pig-latin | — | pass, 3.8, 5, 4 / 0 / 0 | — | pass, 4.3, 5, 5 / 0 / 0 | — | — | — |
| bugfix-javascript-variable-length-quantity | — | — | — | runaway (time), 12.0, 3, 3 / 0 / 0 | — | — | — |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| dflash2b | 11 / 11 | 41.0 | 2.9 | 8.38, 26.46 | 47 | 9.5 | 53 / 2 / 0 | 0 |
| fix2 | 10 / 10 | 41.8 | 3.5 | 8.46, 31.60 | 51 | 12.0 | 52 / 2 / 0 | 0 |
| adapt5 | 11 / 12 | 46.4 | 2.8 | 9.23, 27.73 | 50 | 12.5 | 57 / 1 / 0 | 1 |

Tasks scored on every arm: 9.
- dflash2b against nodraft: 34.8 against 43.0 minutes in all (1.24× faster), per task 1.27× (geometric mean, range 0.61–2.40); passed 9 against 9
- fix2 against nodraft: 36.6 against 43.0 minutes in all (1.18× faster), per task 1.28× (geometric mean, range 0.39–2.44); passed 9 against 9
- adapt5 against nodraft: 24.4 against 43.0 minutes in all (1.77× faster), per task 1.72× (geometric mean, range 1.31–2.77); passed 9 against 9

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| dflash2b | 11 / 11 | 41.0 | 2.9 | 8.38, 26.46 | 47 | 9.5 | 53 / 2 / 0 | 0 |
| fix2 | 10 / 10 | 41.8 | 3.5 | 8.46, 31.60 | 51 | 12.0 | 52 / 2 / 0 | 0 |
| adapt5 | 11 / 12 | 46.4 | 2.8 | 9.23, 27.73 | 50 | 12.5 | 57 / 1 / 0 | 1 |

Tasks scored on every arm: 9.
