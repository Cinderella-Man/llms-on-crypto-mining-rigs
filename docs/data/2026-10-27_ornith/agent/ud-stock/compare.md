9 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 5.5, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 4.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 2.1, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.7, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 3.6, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.5, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.9, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 3, 3 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 8 / 9 | 38.8 | 3.5 | 9.80, 20.36 | 53 | 6.2 | 42 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 18.49, 803 |
| 2 | 9, 19.51, 1448 |
| 3 | 9, 3.56, 28 |
| 4 | 8, 5.77, 122 |
| 5 | 4, 3.61, 36 |
| 6 | 2, 4.05, 49 |
| 7 | 1, 3.79, 35 |
