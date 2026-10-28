12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.4, 8, 8 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.1, 9, 10 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.5, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.7, 10, 10 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 4.3, 14, 14 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.0, 6, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 24.3 | 1.2 | 2.48, 6.14 | 171 | 43.7 | 78 / 0 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.11, 693 |
| 2 | 12, 5.50, 1266 |
| 3 | 12, 2.71, 418 |
| 4 | 12, 1.66, 135 |
| 5 | 11, 0.95, 20 |
| 6 | 6, 1.51, 158 |
| 7 | 4, 1.35, 109 |
| 8 | 4, 1.83, 156 |
| 9 | 3, 1.65, 135 |
| 10 | 2, 2.55, 236 |
| 11 | 1, 6.14, 906 |
| 12 | 1, 2.29, 345 |
| 13 | 1, 1.10, 19 |
| 14 | 1, 3.92, 366 |
