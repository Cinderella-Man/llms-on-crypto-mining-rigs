10 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.0, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 7, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.9, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 6, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 6.1, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 15, 15 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.3, 4, 4 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 10 / 10 | 24.9 | 1.0 | 2.67, 6.87 | 180 | 48.6 | 60 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 10, 4.35, 693 |
| 2 | 10, 5.69, 1256 |
| 3 | 10, 2.56, 377 |
| 4 | 10, 2.63, 350 |
| 5 | 8, 1.10, 20 |
| 6 | 6, 1.58, 78 |
| 7 | 2, 1.90, 132 |
| 8 | 1, 1.85, 100 |
| 9 | 1, 2.42, 175 |
| 10 | 1, 4.19, 289 |
| 11 | 1, 2.25, 143 |
| 12 | 1, 2.87, 323 |
| 13 | 1, 2.91, 149 |
| 14 | 1, 2.57, 149 |
| 15 | 1, —, — |
