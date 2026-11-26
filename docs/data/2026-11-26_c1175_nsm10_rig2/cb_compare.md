12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.7, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 8.4, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.7, 14, 13 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 1.9, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.6, 5, 5 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.0, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 22.1 | 1.4 | 2.18, 7.88 | 155 | 31.6 | 67 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 71 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.65, 789 |
| 2 | 12, 7.91, 1574 |
| 3 | 12, 2.29, 321 |
| 4 | 12, 0.98, 31 |
| 5 | 12, 1.55, 85 |
| 6 | 3, 1.39, 61 |
| 7 | 1, 3.24, 474 |
| 8 | 1, 0.95, 28 |
| 9 | 1, 1.67, 100 |
| 10 | 1, 1.61, 104 |
| 11 | 1, 0.89, 27 |
| 12 | 1, 3.11, 392 |
| 13 | 1, 0.90, 28 |
| 14 | 1, 1.90, 187 |
