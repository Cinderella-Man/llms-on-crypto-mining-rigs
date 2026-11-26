12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 7, 8 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 4, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.5, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 4.1, 12, 12 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 1.6, 7, 8 / 2 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 6, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.0, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.9, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 21.6 | 1.6 | 3.16, 11.86 | 105 | 27.7 | 76 / 4 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 73 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.96, 789 |
| 2 | 12, 12.18, 1386 |
| 3 | 12, 4.21, 370 |
| 4 | 12, 1.12, 32 |
| 5 | 11, 1.80, 94 |
| 6 | 6, 2.20, 148 |
| 7 | 3, 1.18, 36 |
| 8 | 1, 4.97, 457 |
| 9 | 1, 1.58, 66 |
| 10 | 1, 2.13, 130 |
| 11 | 1, 1.14, 29 |
| 12 | 1, 1.22, 35 |
