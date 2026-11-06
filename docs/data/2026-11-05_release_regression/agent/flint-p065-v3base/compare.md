12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.5, 6, 7 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.5, 6, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.5, 7, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.9, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 7, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 4.1, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.8, 8, 7 / 0 / 0 |
| bugfix-python-pig-latin | pass, 5.0, 8, 8 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.7, 9, 9 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 22.3 | 1.0 | 2.66, 6.75 | 175 | 47.4 | 74 / 2 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.23, 693 |
| 2 | 12, 5.66, 1122 |
| 3 | 12, 3.55, 545 |
| 4 | 12, 2.53, 260 |
| 5 | 11, 1.61, 154 |
| 6 | 10, 1.28, 65 |
| 7 | 5, 1.04, 20 |
| 8 | 3, 1.90, 177 |
| 9 | 1, 1.80, 122 |
