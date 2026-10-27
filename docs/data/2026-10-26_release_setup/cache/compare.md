10 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.6, 9, 9 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 8, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 7, 7 / 2 / 0 |
| bugfix-python-list-ops | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 7, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 7.2, 25, 25 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 9 / 10 | 29.5 | 1.0 | 2.69, 9.65 | 173 | 42.2 | 78 / 2 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 10, 4.48, 693 |
| 2 | 10, 5.93, 1336 |
| 3 | 10, 2.74, 418 |
| 4 | 10, 1.79, 220 |
| 5 | 10, 1.22, 136 |
| 6 | 6, 1.34, 114 |
| 7 | 5, 1.16, 28 |
| 8 | 3, 1.87, 135 |
| 9 | 2, 1.31, 58 |
| 10 | 1, 3.31, 334 |
| 11 | 1, 4.59, 622 |
| 12 | 1, 3.42, 434 |
| 13 | 1, 1.84, 76 |
| 14 | 1, 2.06, 104 |
| 15 | 1, 11.24, 1739 |
| 16 | 1, 1.45, 39 |
| 17 | 1, 2.49, 257 |
| 18 | 1, 2.34, 154 |
| 19 | 1, 4.33, 580 |
| 20 | 1, 2.20, 93 |
| 21 | 1, 1.67, 58 |
| 22 | 1, 11.15, 1305 |
| 23 | 1, 13.70, 1864 |
| 24 | 1, 1.44, 16 |
| 25 | 1, 4.90, 429 |
