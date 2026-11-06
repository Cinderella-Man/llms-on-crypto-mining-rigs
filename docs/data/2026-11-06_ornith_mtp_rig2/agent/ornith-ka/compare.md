12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.4, 7, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.4, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 5.6, 8, 8 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.1, 7, 7 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 30.2 | 2.1 | 9.34, 19.86 | 60 | 16.4 | 63 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 16.39, 789 |
| 2 | 12, 18.79, 1452 |
| 3 | 12, 8.08, 320 |
| 4 | 12, 3.28, 32 |
| 5 | 8, 4.00, 70 |
| 6 | 3, 7.62, 282 |
| 7 | 3, 5.29, 132 |
| 8 | 1, 3.32, 36 |
