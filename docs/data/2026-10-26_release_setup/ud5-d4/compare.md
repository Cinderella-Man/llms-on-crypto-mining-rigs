12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.4, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.6, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.5, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.8, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 5.2, 7, 7 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.2, 7, 7 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 24.8 | 1.0 | 2.63, 7.88 | 185 | 44.2 | 67 / 2 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.23, 693 |
| 2 | 12, 5.69, 1253 |
| 3 | 12, 2.61, 380 |
| 4 | 12, 0.92, 24 |
| 5 | 11, 1.43, 107 |
| 6 | 5, 1.02, 20 |
| 7 | 3, 1.57, 125 |
| 8 | 1, 1.18, 20 |
