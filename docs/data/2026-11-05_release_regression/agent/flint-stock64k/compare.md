12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | runaway (time), 12.0, 4, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.1, 9, 8 / 0 / 0 |
| bugfix-python-phone-number | pass, 4.2, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.1, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 2.1, 7, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 2.0, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.9, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 3, 4 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 2.7, 5, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 6.7, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.2, 8, 7 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 10 / 12 | 52.6 | 2.9 | 4.13, 8.04 | 148 | 14.4 | 59 / 2 / 0 | 2 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 13, 7.31, 704 |
| 2 | 13, 7.63, 1357 |
| 3 | 12, 4.39, 417 |
| 4 | 11, 1.35, 24 |
| 5 | 9, 1.30, 32 |
| 6 | 3, 2.25, 94 |
| 7 | 3, 1.18, 21 |
| 8 | 2, 1.12, 22 |
| 9 | 1, 2.57, 132 |
