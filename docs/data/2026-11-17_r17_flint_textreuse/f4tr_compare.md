12 scored attempts, arms f4tr; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f4tr: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.7, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 8, 8 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.2, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.9, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.9, 9, 8 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 6, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 11.1, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.0, 7, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.5, 9, 9 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | prompt tokens read (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f4tr | 12 / 12 | 26.7 | 1.3 | 1.76, 5.31 | 210 | 46.7 | 71 / 0 / 0 | 0 | 35635 (2970) | 0 (0; 0.0 of 169.7) | 11.68 (bugfix-javascript-complex-numbers, 2) |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f4tr: graph reallocations 76 of 77 requests (200 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f4tr: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 3.21, 693 |
| 2 | 12, 3.66, 772 |
| 3 | 12, 2.44, 503 |
| 4 | 12, 1.71, 283 |
| 5 | 11, 0.69, 25 |
| 6 | 7, 1.01, 19 |
| 7 | 5, 1.11, 20 |
| 8 | 4, 0.82, 72 |
| 9 | 2, 1.31, 90 |
