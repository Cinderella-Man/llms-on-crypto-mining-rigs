10 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 5.8, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.7, 10, 10 / 1 / 0 |
| bugfix-python-phone-number | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.4, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 10.9, 9, 9 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 10 / 10 | 25.8 | 1.0 | 2.43, 7.86 | 179 | 40.1 | 57 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 10, 4.63, 693 |
| 2 | 10, 5.94, 1272 |
| 3 | 10, 2.54, 438 |
| 4 | 10, 1.58, 202 |
| 5 | 9, 1.58, 140 |
| 6 | 4, 1.00, 20 |
| 7 | 2, 27.89, 4028 |
| 8 | 2, 1.66, 78 |
| 9 | 2, 1.13, 20 |
| 10 | 1, 1.86, 132 |
