9 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 10.8, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.0, 7, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 4.5, 8, 8 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.5, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.9, 7, 8 / 2 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 6, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.1, 7, 6 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 8 / 9 | 38.2 | 2.9 | 3.67, 7.94 | 144 | 14.4 | 53 / 3 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 5.99, 685 |
| 2 | 9, 3.24, 286 |
| 3 | 9, 6.29, 774 |
| 4 | 9, 3.40, 367 |
| 5 | 8, 3.23, 98 |
| 6 | 6, 1.90, 104 |
| 7 | 4, 1.78, 76 |
| 8 | 1, 1.14, 20 |
