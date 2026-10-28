12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.0, 9, 9 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 8, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.7, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.2, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.9, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 6 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.6, 7, 6 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 7, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.3, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.2, 6, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 0.9, 5, 4 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 19.4 | 1.1 | 2.38, 5.99 | 180 | 45.7 | 66 / 0 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.23, 693 |
| 2 | 12, 4.47, 784 |
| 3 | 12, 3.26, 506 |
| 4 | 12, 1.75, 256 |
| 5 | 11, 0.93, 21 |
| 6 | 7, 0.58, 20 |
| 7 | 4, 0.69, 36 |
| 8 | 2, 1.11, 72 |
| 9 | 1, 2.20, 291 |
