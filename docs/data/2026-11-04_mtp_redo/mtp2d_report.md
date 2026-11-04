### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| mtp2 | 10 / 11 | 43.7 | 2.7 | 7.85, 21.09 | 55 | 8.8 | 61 / 1 / 0 | 1 |
| dflash2b | 11 / 11 | 41.0 | 2.9 | 8.38, 26.46 | 47 | 9.5 | 53 / 2 / 0 | 0 |
| mtp2d | 11 / 12 | 47.4 | 2.9 | 9.17, 21.05 | 56 | 12.2 | 55 / 0 / 0 | 1 |

Tasks scored on every arm: 9.
### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | nodraft: result, min, turns, tool calls | mtp2: result, min, turns, tool calls | dflash2b: result, min, turns, tool calls | mtp2d: result, min, turns, tool calls | time nodraft / mtp2 | time nodraft / dflash2b | time nodraft / mtp2d |
|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 8.2, 8, 8 / 1 / 0 | pass, 4.4, 5, 5 / 0 / 0 | pass, 6.4, 7, 7 / 1 / 0 | pass, 5.4, 6, 5 / 0 / 0 | 1.89× | 1.29× | 1.54× |
| bugfix-javascript-space-age | pass, 4.1, 6, 5 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | 2.06× | 2.37× | 1.78× |
| bugfix-python-phone-number | pass, 3.0, 5, 5 / 0 / 0 | pass, 2.7, 5, 5 / 0 / 0 | pass, 2.1, 5, 4 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | 1.13× | 1.42× | 1.18× |
| bugfix-javascript-triangle | pass, 2.4, 5, 4 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | pass, 2.2, 5, 4 / 0 / 0 | pass, 2.3, 7, 6 / 0 / 0 | 1.42× | 1.13× | 1.05× |
| bugfix-python-variable-length-quantity | pass, 6.9, 7, 7 / 1 / 0 | pass, 3.1, 6, 6 / 0 / 0 | pass, 6.8, 6, 6 / 1 / 0 | pass, 3.1, 5, 5 / 0 / 0 | 2.28× | 1.02× | 2.22× |
| bugfix-javascript-phone-number | pass, 2.5, 5, 4 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | pass, 2.9, 5, 4 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | 0.97× | 0.85× | 1.16× |
| bugfix-python-list-ops | pass, 4.3, 5, 5 / 0 / 0 | pass, 4.6, 5, 5 / 0 / 0 | pass, 7.0, 5, 5 / 0 / 0 | pass, 3.3, 5, 5 / 0 / 0 | 0.94× | 0.61× | 1.28× |
| bugfix-javascript-grade-school | pass, 2.8, 5, 4 / 0 / 0 | pass, 1.9, 5, 4 / 0 / 0 | pass, 2.1, 5, 4 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | 1.49× | 1.35× | 1.19× |
| bugfix-python-scale-generator | pass, 8.7, 5, 5 / 0 / 0 | pass, 6.2, 5, 5 / 0 / 0 | pass, 3.6, 5, 5 / 0 / 0 | pass, 5.7, 5, 5 / 0 / 0 | 1.40× | 2.40× | 1.54× |
| bugfix-javascript-complex-numbers | — | pass, 2.7, 5, 5 / 0 / 0 | pass, 2.4, 7, 6 / 0 / 0 | pass, 2.7, 6, 5 / 0 / 0 | — | — | — |
| bugfix-python-pig-latin | — | runaway (time), 12.0, 14, 13 / 1 / 0 | pass, 3.8, 5, 4 / 0 / 0 | pass, 3.6, 5, 4 / 0 / 0 | — | — | — |
| bugfix-javascript-variable-length-quantity | — | — | — | runaway (time), 12.0, 2, 2 / 0 / 0 | — | — | — |

