12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.8, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 6, 6 / 1 / 0 |
| bugfix-python-phone-number | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.7, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 6, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.5, 7, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.4, 6, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.5, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.7, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 16.7 | 1.2 | 2.62, 6.61 | 178 | 53.8 | 62 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.29, 693 |
| 2 | 12, 4.66, 1014 |
| 3 | 12, 3.21, 556 |
| 4 | 12, 1.92, 230 |
| 5 | 12, 1.21, 20 |
| 6 | 8, 1.28, 49 |
| 7 | 1, 1.11, 20 |
