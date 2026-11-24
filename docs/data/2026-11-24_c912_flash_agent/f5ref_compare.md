12 scored attempts, arms f5ref; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f5ref: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.6, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | runaway (turns), 9.2, 41, 41 / 2 / 0 |
| bugfix-python-pig-latin | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 0.6, 4, 4 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f5ref | 11 / 12 | 24.0 | 1.1 | 2.59, 8.34 | 144 | 26.8 | 92 / 2 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f5ref: graph reallocations 0 of 95 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f5ref: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 5.15, 809 |
| 2 | 12, 7.53, 1396 |
| 3 | 12, 2.66, 348 |
| 4 | 12, 1.06, 31 |
| 5 | 11, 1.75, 92 |
| 6 | 2, 1.81, 180 |
| 7 | 1, 1.66, 81 |
| 8 | 1, 3.85, 563 |
| 9 | 1, 2.08, 184 |
| 10 | 1, 3.45, 433 |
| 11 | 1, 2.18, 208 |
| 12 | 1, 0.97, 25 |
| 13 | 1, 4.87, 669 |
| 14 | 1, 0.96, 26 |
| 15 | 1, 3.44, 467 |
| 16 | 1, 7.17, 1123 |
| 17 | 1, 7.34, 1208 |
| 18 | 1, 7.42, 1112 |
| 19 | 1, 2.67, 342 |
| 20 | 1, 2.34, 244 |
| 21 | 1, 1.64, 62 |
| 22 | 1, 7.63, 1210 |
| 23 | 1, 7.14, 1030 |
| 24 | 1, 3.41, 429 |
| 25 | 1, 7.20, 1144 |
| 26 | 1, 7.62, 1201 |
| 27 | 1, 5.45, 726 |
| 28 | 1, 2.12, 162 |
| 29 | 1, 1.61, 55 |
| 30 | 1, 15.42, 1812 |
| 31 | 1, 8.77, 1227 |
| 32 | 1, 0.93, 19 |
| 33 | 1, 0.94, 19 |
| 34 | 1, 2.30, 215 |
| 35 | 1, 0.93, 19 |
| 36 | 1, 1.48, 50 |
| 37 | 1, 2.45, 242 |
| 38 | 1, 0.96, 18 |
| 39 | 1, 2.09, 155 |
| 40 | 1, 2.04, 150 |
| 41 | 1, —, — |
