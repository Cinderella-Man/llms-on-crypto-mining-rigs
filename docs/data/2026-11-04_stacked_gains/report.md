### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | nodraft: result, min, turns, tool calls | mtp2d: result, min, turns, tool calls | s34nd: result, min, turns, tool calls | s34mtp: result, min, turns, tool calls | s34mtps: result, min, turns, tool calls | time nodraft / mtp2d | time nodraft / s34nd | time nodraft / s34mtp | time nodraft / s34mtps |
|---|---|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 8.2, 8, 8 / 1 / 0 | pass, 5.4, 6, 5 / 0 / 0 | pass, 3.4, 5, 6 / 1 / 0 | pass, 3.3, 5, 5 / 0 / 0 | pass, 3.6, 7, 7 / 1 / 0 | 1.54× | 2.40× | 2.49× | 2.26× |
| bugfix-javascript-space-age | pass, 4.1, 6, 5 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | pass, 1.3, 5, 4 / 0 / 0 | pass, 1.5, 5, 6 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | 1.78× | 3.09× | 2.68× | 2.74× |
| bugfix-python-phone-number | pass, 3.0, 5, 5 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | pass, 2.1, 5, 5 / 0 / 0 | pass, 3.1, 6, 6 / 0 / 0 | 1.18× | 1.38× | 1.48× | 0.99× |
| bugfix-javascript-triangle | pass, 2.4, 5, 4 / 0 / 0 | pass, 2.3, 7, 6 / 0 / 0 | pass, 1.1, 5, 4 / 0 / 0 | pass, 1.6, 5, 4 / 0 / 0 | pass, 1.2, 5, 4 / 0 / 0 | 1.05× | 2.23× | 1.49× | 2.09× |
| bugfix-python-variable-length-quantity | pass, 6.9, 7, 7 / 1 / 0 | pass, 3.1, 5, 5 / 0 / 0 | pass, 2.7, 6, 6 / 0 / 0 | pass, 4.1, 5, 5 / 0 / 0 | pass, 2.5, 5, 6 / 0 / 0 | 2.22× | 2.58× | 1.70× | 2.75× |
| bugfix-javascript-phone-number | pass, 2.5, 5, 4 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | 1.16× | 1.51× | 1.47× | 1.32× |
| bugfix-python-list-ops | pass, 4.3, 5, 5 / 0 / 0 | pass, 3.3, 5, 5 / 0 / 0 | pass, 2.4, 5, 5 / 0 / 0 | pass, 3.1, 5, 5 / 0 / 0 | pass, 2.1, 5, 5 / 0 / 0 | 1.28× | 1.79× | 1.39× | 2.06× |
| bugfix-javascript-grade-school | pass, 2.8, 5, 4 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | pass, 2.0, 8, 7 / 1 / 0 | 1.19× | 1.90× | 1.84× | 1.41× |
| bugfix-python-scale-generator | pass, 8.7, 5, 5 / 0 / 0 | pass, 5.7, 5, 5 / 0 / 0 | pass, 10.6, 6, 6 / 1 / 0 | pass, 4.5, 6, 6 / 0 / 0 | pass, 4.1, 6, 6 / 0 / 0 | 1.54× | 0.82× | 1.94× | 2.11× |
| bugfix-javascript-complex-numbers | — | pass, 2.7, 6, 5 / 0 / 0 | pass, 2.5, 8, 8 / 2 / 0 | pass, 1.9, 7, 7 / 1 / 0 | pass, 2.2, 5, 5 / 0 / 0 | — | — | — | — |
| bugfix-python-pig-latin | — | pass, 3.6, 5, 4 / 0 / 0 | pass, 6.0, 7, 7 / 0 / 0 | pass, 5.6, 5, 5 / 0 / 0 | pass, 3.3, 5, 5 / 0 / 0 | — | — | — | — |
| bugfix-javascript-variable-length-quantity | — | runaway (time), 12.0, 2, 2 / 0 / 0 | runaway (time), 12.0, 3, 3 / 0 / 0 | runaway (time), 12.0, 2, 2 / 0 / 0 | pass, 4.9, 7, 7 / 1 / 0 | — | — | — | — |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| mtp2d | 11 / 12 | 47.4 | 2.9 | 9.17, 21.05 | 56 | 12.2 | 55 / 0 / 0 | 1 |
| s34nd | 11 / 12 | 47.3 | 2.4 | 7.84, 17.83 | 63 | 14.3 | 63 / 4 / 0 | 1 |
| s34mtp | 11 / 12 | 43.0 | 2.6 | 8.06, 20.56 | 57 | 14.6 | 59 / 1 / 0 | 1 |
| s34mtps | 12 / 12 | 32.4 | 2.3 | 8.35, 19.99 | 58 | 16.0 | 67 / 3 / 0 | 0 |

Tasks scored on every arm: 9.
- mtp2d against nodraft: 29.2 against 43.0 minutes in all (1.48× faster), per task 1.40× (geometric mean, range 1.05–2.22); passed 9 against 9
- s34nd against nodraft: 26.9 against 43.0 minutes in all (1.60× faster), per task 1.85× (geometric mean, range 0.82–3.09); passed 9 against 9
- s34mtp against nodraft: 23.4 against 43.0 minutes in all (1.84× faster), per task 1.78× (geometric mean, range 1.39–2.68); passed 9 against 9
- s34mtps against nodraft: 22.0 against 43.0 minutes in all (1.96× faster), per task 1.87× (geometric mean, range 0.99–2.75); passed 9 against 9

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| mtp2d | 11 / 12 | 47.4 | 2.9 | 9.17, 21.05 | 56 | 12.2 | 55 / 0 / 0 | 1 |
| s34nd | 11 / 12 | 47.3 | 2.4 | 7.84, 17.83 | 63 | 14.3 | 63 / 4 / 0 | 1 |
| s34mtp | 11 / 12 | 43.0 | 2.6 | 8.06, 20.56 | 57 | 14.6 | 59 / 1 / 0 | 1 |
| s34mtps | 12 / 12 | 32.4 | 2.3 | 8.35, 19.99 | 58 | 16.0 | 67 / 3 / 0 | 0 |

Tasks scored on every arm: 9.
