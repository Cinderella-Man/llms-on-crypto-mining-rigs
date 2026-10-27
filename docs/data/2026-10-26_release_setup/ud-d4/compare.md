11 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.5, 8, 7 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 7, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.4, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 6, 5 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 6, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 3.1, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 7, 6 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 9, 10 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.1, 7, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 10 / 11 | 26.9 | 1.1 | 2.46, 6.50 | 191 | 50.7 | 65 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 11, 4.41, 702 |
| 2 | 11, 5.39, 1187 |
| 3 | 11, 2.73, 505 |
| 4 | 11, 2.44, 306 |
| 5 | 10, 1.13, 30 |
| 6 | 8, 1.16, 52 |
| 7 | 5, 2.38, 276 |
| 8 | 2, 0.90, 19 |
| 9 | 1, —, — |
