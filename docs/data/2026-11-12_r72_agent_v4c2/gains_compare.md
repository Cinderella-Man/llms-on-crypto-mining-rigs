12 scored attempts, arms gains; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | gains: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 4.8, 9, 8 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.0, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.1, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.9, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.3, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.1, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.5, 7, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| gains | 12 / 12 | 26.1 | 2.0 | 4.23, 24.54 | 73 | 24.8 | 63 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | gains: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 10.75, 789 |
| 2 | 12, 21.02, 1668 |
| 3 | 12, 5.84, 404 |
| 4 | 12, 1.21, 30 |
| 5 | 12, 2.07, 84 |
| 6 | 5, 3.00, 209 |
| 7 | 2, 1.73, 92 |
| 8 | 1, 1.81, 67 |
| 9 | 1, 5.59, 344 |
