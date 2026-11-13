36 scored attempts, arms v4c2, v4c22, v4c3; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4c2: result, min, turns, tool calls | v4c22: result, min, turns, tool calls | v4c3: result, min, turns, tool calls | time v4c2 / v4c22 | time v4c2 / v4c3 |
|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 2.5, 6, 6 / 0 / 0 | pass, 2.4, 5, 5 / 0 / 0 | pass, 1.8, 6, 5 / 0 / 0 | 1.02× | 1.41× |
| bugfix-javascript-space-age | pass, 0.8, 5, 4 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | 0.93× | 0.97× |
| bugfix-python-phone-number | pass, 1.1, 6, 5 / 0 / 0 | pass, 1.3, 5, 5 / 0 / 0 | pass, 0.9, 6, 5 / 0 / 0 | 0.82× | 1.14× |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | 1.02× | 0.97× |
| bugfix-python-variable-length-quantity | pass, 1.7, 5, 5 / 0 / 0 | pass, 1.1, 5, 6 / 1 / 0 | pass, 2.2, 6, 6 / 1 / 0 | 1.51× | 0.77× |
| bugfix-javascript-phone-number | pass, 0.9, 5, 5 / 0 / 0 | pass, 1.1, 5, 4 / 0 / 0 | pass, 1.5, 7, 6 / 0 / 0 | 0.89× | 0.63× |
| bugfix-python-list-ops | pass, 2.3, 5, 5 / 0 / 0 | pass, 1.3, 5, 5 / 0 / 0 | pass, 1.3, 5, 5 / 0 / 0 | 1.80× | 1.74× |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | 0.99× | 0.93× |
| bugfix-python-scale-generator | pass, 2.0, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | 1.06× | 1.09× |
| bugfix-javascript-complex-numbers | runaway (time), 12.0, 35, 34 / 2 / 0 | pass, 1.0, 5, 5 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | 11.74× | 13.37× |
| bugfix-python-pig-latin | pass, 1.4, 5, 5 / 0 / 0 | pass, 2.3, 9, 8 / 1 / 0 | pass, 1.9, 5, 5 / 1 / 0 | 0.61× | 0.76× |
| bugfix-javascript-variable-length-quantity | pass, 5.0, 5, 5 / 0 / 0 | pass, 5.1, 6, 6 / 0 / 0 | pass, 1.6, 6, 6 / 0 / 0 | 0.99× | 3.06× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4c2 | 11 / 12 | 31.3 | 1.6 | 2.42, 9.29 | 136 | 26.1 | 87 / 2 / 0 | 1 |
| v4c22 | 12 / 12 | 19.9 | 1.2 | 2.81, 8.58 | 148 | 27.5 | 61 / 2 / 0 | 0 |
| v4c3 | 12 / 12 | 16.4 | 1.4 | 2.45, 8.39 | 145 | 28.0 | 60 / 2 / 0 | 0 |

Tasks scored on every arm: 12.
- v4c22 against v4c2: 19.9 against 31.3 minutes in all (1.58× faster), per task 1.25× (geometric mean, range 0.61–11.74); passed 12 against 11
- v4c3 against v4c2: 16.4 against 31.3 minutes in all (1.91× faster), per task 1.36× (geometric mean, range 0.63–13.37); passed 12 against 11

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4c2: graph reallocations 0 of 91 requests (0 same size, 0 changed)
- v4c22: graph reallocations 0 of 65 requests (0 same size, 0 changed)
- v4c3: graph reallocations 0 of 66 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4c2: tasks, TTFT s, new prompt tokens | v4c22: tasks, TTFT s, new prompt tokens | v4c3: tasks, TTFT s, new prompt tokens |
|---|---|---|---|
| 1 | 12, 6.65, 809 | 12, 6.67, 789 | 12, 6.64, 789 |
| 2 | 12, 7.97, 1550 | 12, 8.17, 1608 | 12, 8.09, 1608 |
| 3 | 12, 3.25, 410 | 12, 3.37, 422 | 12, 3.12, 392 |
| 4 | 12, 1.05, 31 | 12, 1.03, 30 | 12, 1.08, 31 |
| 5 | 12, 1.72, 109 | 12, 1.71, 100 | 12, 1.69, 88 |
| 6 | 3, 1.25, 46 | 2, 1.88, 160 | 5, 1.47, 56 |
| 7 | 1, 1.82, 179 | 1, 1.07, 32 | 1, 2.16, 186 |
| 8 | 1, 2.44, 324 | 1, 1.72, 110 | 0, —, — |
| 9 | 1, 0.90, 22 | 1, 1.65, 106 | 0, —, — |
| 10 | 1, 1.16, 37 | 0, —, — | 0, —, — |
| 11 | 1, 1.67, 79 | 0, —, — | 0, —, — |
| 12 | 1, 2.40, 308 | 0, —, — | 0, —, — |
| 13 | 1, 2.35, 287 | 0, —, — | 0, —, — |
| 14 | 1, 1.31, 46 | 0, —, — | 0, —, — |
| 15 | 1, 2.47, 309 | 0, —, — | 0, —, — |
| 16 | 1, 9.49, 1082 | 0, —, — | 0, —, — |
| 17 | 1, 1.01, 24 | 0, —, — | 0, —, — |
| 18 | 1, 3.93, 570 | 0, —, — | 0, —, — |
| 19 | 1, 1.07, 25 | 0, —, — | 0, —, — |
| 20 | 1, 9.04, 680 | 0, —, — | 0, —, — |
| 21 | 1, 1.41, 36 | 0, —, — | 0, —, — |
| 22 | 1, 2.13, 206 | 0, —, — | 0, —, — |
| 23 | 1, 2.28, 239 | 0, —, — | 0, —, — |
| 24 | 1, 0.92, 19 | 0, —, — | 0, —, — |
| 25 | 1, 1.51, 60 | 0, —, — | 0, —, — |
| 26 | 1, 1.13, 29 | 0, —, — | 0, —, — |
| 27 | 1, 3.87, 548 | 0, —, — | 0, —, — |
| 28 | 1, 9.79, 851 | 0, —, — | 0, —, — |
| 29 | 1, 0.93, 19 | 0, —, — | 0, —, — |
| 30 | 1, 1.88, 130 | 0, —, — | 0, —, — |
| 31 | 1, 11.48, 1258 | 0, —, — | 0, —, — |
| 32 | 1, 10.17, 727 | 0, —, — | 0, —, — |
| 33 | 1, 19.98, 2920 | 0, —, — | 0, —, — |
| 34 | 1, 1.14, 25 | 0, —, — | 0, —, — |
| 35 | 1, —, — | 0, —, — | 0, —, — |
