12 scored attempts, arms f5pst; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f5pst: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.4, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 10, 9 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.3, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.2, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.4, 4, 3 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 8, 8 / 2 / 0 |
| bugfix-python-list-ops | pass, 1.1, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.5, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.0, 10, 10 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.5, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | prompt tokens read (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f5pst | 12 / 12 | 19.5 | 1.1 | 1.30, 2.74 | 348 | 49.2 | 66 / 2 / 0 | 0 | 35454 (2954) | 0 (0; 0.0 of 101.8) | 9.22 (bugfix-python-affine-cipher, 1) |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f5pst: graph reallocations 0 of 72 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f5pst: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 1.72, 693 |
| 2 | 12, 2.69, 1256 |
| 3 | 12, 1.62, 410 |
| 4 | 12, 0.70, 20 |
| 5 | 10, 0.78, 74 |
| 6 | 4, 0.93, 73 |
| 7 | 3, 1.06, 173 |
| 8 | 3, 1.10, 107 |
| 9 | 2, 0.72, 20 |
| 10 | 2, 1.62, 230 |
