12 scored attempts, arms f4tr; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f4tr: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 7, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.3, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.5, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 7, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 1.8, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.0, 6, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.7, 7, 7 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.9, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | prompt tokens read (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f4tr | 12 / 12 | 19.1 | 1.1 | 1.81, 6.18 | 208 | 49.4 | 64 / 1 / 0 | 0 | 35309 (2942) | 0 (0; 0.0 of 170.1) | 12.57 (bugfix-javascript-complex-numbers, 2) |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f4tr: graph reallocations 69 of 70 requests (191 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f4tr: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 3.26, 693 |
| 2 | 12, 5.43, 1230 |
| 3 | 12, 2.65, 494 |
| 4 | 12, 0.83, 44 |
| 5 | 11, 0.80, 20 |
| 6 | 8, 0.75, 44 |
| 7 | 3, 1.00, 132 |
