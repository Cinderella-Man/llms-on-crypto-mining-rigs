9 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 6.5, 9, 9 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.1, 8, 8 / 0 / 0 |
| bugfix-python-phone-number | pass, 3.8, 7, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.7, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 3.1, 6, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 6.5, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.3, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 3, 2 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 8 / 9 | 37.6 | 3.1 | 3.79, 7.96 | 145 | 14.0 | 48 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 6.13, 685 |
| 2 | 9, 7.65, 1319 |
| 3 | 9, 4.19, 400 |
| 4 | 8, 1.80, 106 |
| 5 | 7, 1.14, 20 |
| 6 | 4, 2.29, 170 |
| 7 | 3, 1.12, 20 |
| 8 | 2, 1.73, 77 |
| 9 | 1, 4.18, 303 |
