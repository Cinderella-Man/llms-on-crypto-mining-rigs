12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.3, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.3, 10, 10 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 0.9, 5, 5 / 1 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.8, 10, 10 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.3, 7, 7 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.3, 8, 8 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 24.3 | 1.1 | 2.25, 5.97 | 179 | 45.8 | 72 / 2 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.29, 693 |
| 2 | 12, 5.95, 1288 |
| 3 | 12, 2.88, 484 |
| 4 | 12, 0.68, 40 |
| 5 | 12, 0.94, 68 |
| 6 | 6, 0.87, 19 |
| 7 | 4, 1.03, 23 |
| 8 | 3, 1.27, 62 |
| 9 | 2, 0.79, 20 |
| 10 | 2, 2.26, 230 |
