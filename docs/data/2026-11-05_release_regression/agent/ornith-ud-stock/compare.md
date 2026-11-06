12 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.7, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.1, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 3.3, 4, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 2.1, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.2, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 2.9, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 4.5, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.4, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 2.7, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | runaway (time), 12.0, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 6.5, 8, 8 / 2 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 11 / 12 | 56.5 | 3.1 | 12.57, 20.55 | 55 | 6.0 | 56 / 2 / 0 | 2 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 16.67, 789 |
| 2 | 12, 18.20, 1452 |
| 3 | 12, 3.79, 30 |
| 4 | 12, 8.74, 220 |
| 5 | 4, 3.96, 35 |
| 6 | 2, 6.63, 91 |
| 7 | 1, 4.13, 37 |
| 8 | 1, 5.67, 92 |
