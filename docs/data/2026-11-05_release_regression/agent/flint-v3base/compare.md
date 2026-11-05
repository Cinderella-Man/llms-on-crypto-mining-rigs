12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.2, 7, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.4, 10, 10 / 1 / 0 |
| bugfix-python-phone-number | pass, 1.9, 8, 7 / 1 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.9, 7, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.2, 6, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.3, 8, 8 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 19.2 | 1.2 | 2.67, 6.48 | 168 | 45.9 | 71 / 3 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.23, 693 |
| 2 | 12, 4.72, 1013 |
| 3 | 12, 3.47, 518 |
| 4 | 12, 2.48, 257 |
| 5 | 11, 1.03, 25 |
| 6 | 8, 1.00, 20 |
| 7 | 5, 1.18, 20 |
| 8 | 3, 1.10, 28 |
| 9 | 1, 0.95, 19 |
| 10 | 1, 2.05, 135 |
