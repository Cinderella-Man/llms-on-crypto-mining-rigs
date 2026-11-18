12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.3, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 4, 3 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.1, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | runaway (time), 12.0, 34, 34 / 2 / 0 |
| bugfix-python-pig-latin | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 6.8, 6, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 11 / 12 | 32.6 | 1.3 | 2.52, 9.54 | 139 | 27.0 | 90 / 3 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 91 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 6.69, 809 |
| 2 | 12, 8.24, 1648 |
| 3 | 12, 3.33, 436 |
| 4 | 12, 1.09, 31 |
| 5 | 11, 1.80, 143 |
| 6 | 4, 2.03, 178 |
| 7 | 2, 1.93, 176 |
| 8 | 1, 1.32, 49 |
| 9 | 1, 2.46, 320 |
| 10 | 1, 9.92, 1319 |
| 11 | 1, 3.44, 474 |
| 12 | 1, 9.01, 899 |
| 13 | 1, 10.24, 1185 |
| 14 | 1, 0.93, 20 |
| 15 | 1, 3.72, 502 |
| 16 | 1, 10.25, 1013 |
| 17 | 1, 1.82, 73 |
| 18 | 1, 9.93, 889 |
| 19 | 1, 2.32, 221 |
| 20 | 1, 2.38, 267 |
| 21 | 1, 12.72, 1478 |
| 22 | 1, 1.03, 24 |
| 23 | 1, 2.30, 271 |
| 24 | 1, 2.23, 206 |
| 25 | 1, 1.16, 30 |
| 26 | 1, 2.08, 154 |
| 27 | 1, 2.36, 235 |
| 28 | 1, 2.57, 291 |
| 29 | 1, 2.54, 329 |
| 30 | 1, 1.81, 110 |
| 31 | 1, 2.29, 209 |
| 32 | 1, 2.55, 278 |
| 33 | 1, 2.05, 138 |
| 34 | 1, —, — |
