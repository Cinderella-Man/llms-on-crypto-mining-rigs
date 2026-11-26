12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 8, 8 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 6, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | runaway (time), 12.0, 25, 25 / 1 / 0 |
| bugfix-python-pig-latin | pass, 1.9, 6, 5 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 7.1, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 11 / 12 | 35.3 | 1.3 | 2.36, 8.22 | 162 | 29.1 | 80 / 2 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 84 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.94, 809 |
| 2 | 12, 7.78, 1648 |
| 3 | 12, 2.25, 328 |
| 4 | 12, 0.98, 30 |
| 5 | 12, 1.58, 90 |
| 6 | 4, 1.63, 121 |
| 7 | 2, 2.04, 201 |
| 8 | 2, 2.23, 298 |
| 9 | 1, 1.39, 56 |
| 10 | 1, 10.57, 1738 |
| 11 | 1, 6.62, 1160 |
| 12 | 1, 2.99, 388 |
| 13 | 1, 0.99, 26 |
| 14 | 1, 3.54, 558 |
| 15 | 1, 3.16, 437 |
| 16 | 1, 2.67, 357 |
| 17 | 1, 2.23, 242 |
| 18 | 1, 1.06, 30 |
| 19 | 1, 1.28, 44 |
| 20 | 1, 2.23, 230 |
| 21 | 1, 2.12, 222 |
| 22 | 1, 4.68, 695 |
| 23 | 1, 2.01, 163 |
| 24 | 1, 15.09, 2091 |
| 25 | 1, —, — |
