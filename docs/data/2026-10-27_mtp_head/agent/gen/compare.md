9 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 6.9, 12, 12 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.1, 6, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.7, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.5, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.5, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.5, 8, 8 / 2 / 0 |
| bugfix-python-list-ops | pass, 4.1, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 8 / 9 | 30.2 | 1.5 | 2.78, 7.91 | 177 | 30.7 | 53 / 3 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 4.25, 684 |
| 2 | 9, 5.81, 1187 |
| 3 | 9, 2.96, 528 |
| 4 | 9, 1.00, 29 |
| 5 | 9, 0.92, 22 |
| 6 | 4, 1.62, 84 |
| 7 | 2, 1.00, 19 |
| 8 | 2, 2.26, 119 |
| 9 | 1, 1.21, 38 |
| 10 | 1, 3.24, 285 |
| 11 | 1, 1.03, 20 |
| 12 | 1, 1.11, 19 |
