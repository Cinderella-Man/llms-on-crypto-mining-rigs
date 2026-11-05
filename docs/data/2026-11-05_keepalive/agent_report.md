48 scored attempts, arms s35kaoff, s35ka, s35kareq, s35kaidle; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | s35kaoff: result, min, turns, tool calls | s35ka: result, min, turns, tool calls | s35kareq: result, min, turns, tool calls | s35kaidle: result, min, turns, tool calls | time s35kaoff / s35ka | time s35kaoff / s35kareq | time s35kaoff / s35kaidle |
|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 2.4, 6, 6 / 0 / 0 | pass, 3.4, 7, 7 / 1 / 0 | pass, 2.5, 5, 5 / 0 / 0 | runaway (time), 12.0, 3, 3 / 0 / 0 | 0.71× | 0.97× | 0.20× |
| bugfix-javascript-space-age | pass, 1.1, 5, 4 / 0 / 0 | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.3, 5, 6 / 0 / 0 | 0.94× | 0.97× | 0.90× |
| bugfix-python-phone-number | pass, 2.1, 5, 5 / 0 / 0 | pass, 1.6, 5, 5 / 0 / 0 | pass, 1.6, 5, 5 / 0 / 0 | pass, 1.6, 6, 6 / 0 / 0 | 1.29× | 1.29× | 1.33× |
| bugfix-javascript-triangle | pass, 1.1, 5, 5 / 0 / 0 | pass, 1.4, 5, 4 / 0 / 0 | pass, 1.1, 5, 4 / 0 / 0 | pass, 0.9, 4, 3 / 0 / 0 | 0.77× | 0.98× | 1.19× |
| bugfix-python-variable-length-quantity | pass, 4.8, 5, 5 / 0 / 0 | pass, 2.3, 5, 5 / 0 / 0 | pass, 2.4, 6, 6 / 0 / 0 | pass, 1.6, 5, 5 / 0 / 0 | 2.05× | 2.03× | 3.03× |
| bugfix-javascript-phone-number | pass, 1.5, 5, 4 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | 1.03× | 1.09× | 1.03× |
| bugfix-python-list-ops | pass, 2.8, 5, 5 / 0 / 0 | pass, 1.8, 5, 5 / 0 / 0 | pass, 2.0, 5, 5 / 0 / 0 | pass, 2.4, 5, 5 / 0 / 0 | 1.56× | 1.39× | 1.16× |
| bugfix-javascript-grade-school | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.6, 8, 7 / 0 / 0 | pass, 1.2, 5, 5 / 0 / 0 | 1.01× | 0.78× | 1.06× |
| bugfix-python-scale-generator | pass, 3.7, 5, 5 / 0 / 0 | pass, 3.9, 5, 5 / 0 / 0 | pass, 5.7, 7, 7 / 0 / 0 | runaway (time), 12.0, 3, 3 / 0 / 0 | 0.94× | 0.65× | 0.31× |
| bugfix-javascript-complex-numbers | pass, 2.2, 5, 5 / 0 / 0 | pass, 1.8, 6, 6 / 1 / 0 | pass, 1.7, 5, 5 / 0 / 0 | pass, 1.3, 6, 6 / 1 / 0 | 1.20× | 1.33× | 1.68× |
| bugfix-python-pig-latin | pass, 2.3, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | pass, 2.9, 6, 6 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | 1.18× | 0.79× | 0.87× |
| bugfix-javascript-variable-length-quantity | pass, 4.6, 9, 10 / 1 / 0 | pass, 7.4, 5, 5 / 0 / 0 | pass, 2.1, 5, 5 / 0 / 0 | pass, 2.5, 5, 5 / 0 / 0 | 0.62× | 2.18× | 1.82× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| s35kaoff | 12 / 12 | 29.9 | 2.3 | 7.27, 15.81 | 69 | 19.2 | 63 / 1 / 0 | 0 |
| s35ka | 12 / 12 | 29.6 | 1.8 | 6.41, 16.86 | 77 | 20.8 | 59 / 2 / 0 | 0 |
| s35kareq | 12 / 12 | 26.2 | 1.8 | 6.23, 16.21 | 72 | 20.8 | 64 / 0 / 0 | 0 |
| s35kaidle | 10 / 12 | 40.9 | 1.6 | 7.60, 15.93 | 36 | 18.1 | 56 / 1 / 0 | 2 |

Tasks scored on every arm: 12.
- s35ka against s35kaoff: 29.6 against 29.9 minutes in all (1.01× faster), per task 1.05× (geometric mean, range 0.62–2.05); passed 12 against 12
- s35kareq against s35kaoff: 26.2 against 29.9 minutes in all (1.14× faster), per task 1.13× (geometric mean, range 0.65–2.18); passed 12 against 12
- s35kaidle against s35kaoff: 40.9 against 29.9 minutes in all (0.73× faster), per task 0.99× (geometric mean, range 0.20–3.03); passed 10 against 12

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | s35kaoff: tasks, TTFT s, new prompt tokens | s35ka: tasks, TTFT s, new prompt tokens | s35kareq: tasks, TTFT s, new prompt tokens | s35kaidle: tasks, TTFT s, new prompt tokens |
|---|---|---|---|---|
| 1 | 12, 13.58, 789 | 12, 13.36, 789 | 12, 13.50, 789 | 12, 13.37, 789 |
| 2 | 12, 15.39, 1445 | 12, 15.82, 1669 | 12, 15.74, 1608 | 12, 15.53, 1550 |
| 3 | 12, 7.54, 355 | 12, 6.81, 341 | 12, 7.57, 396 | 12, 6.45, 298 |
| 4 | 12, 2.75, 31 | 12, 2.56, 30 | 12, 2.56, 31 | 10, 2.65, 32 |
| 5 | 12, 3.43, 91 | 12, 3.48, 90 | 12, 3.05, 44 | 9, 3.63, 92 |
| 6 | 2, 3.25, 72 | 2, 3.06, 61 | 4, 2.58, 32 | 2, 5.53, 253 |
| 7 | 1, 2.48, 29 | 1, 4.82, 156 | 2, 3.08, 66 | 0, —, — |
| 8 | 1, 2.87, 29 | 0, —, — | 1, 4.24, 145 | 0, —, — |
| 9 | 1, 3.74, 86 | 0, —, — | 0, —, — | 0, —, — |
