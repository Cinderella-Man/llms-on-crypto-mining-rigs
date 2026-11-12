24 scored attempts, arms gains, v4c2; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | gains: result, min, turns, tool calls | v4c2: result, min, turns, tool calls | time gains / v4c2 |
|---|---|---|---|
| bugfix-python-affine-cipher | pass, 4.8, 9, 8 / 1 / 0 | pass, 2.5, 6, 6 / 0 / 0 | 1.94× |
| bugfix-javascript-space-age | pass, 1.0, 5, 4 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | 1.33× |
| bugfix-python-phone-number | pass, 2.0, 6, 6 / 0 / 0 | pass, 1.1, 6, 5 / 0 / 0 | 1.89× |
| bugfix-javascript-triangle | pass, 1.1, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | 1.57× |
| bugfix-python-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | 1.50× |
| bugfix-javascript-phone-number | pass, 1.2, 5, 4 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | 1.22× |
| bugfix-python-list-ops | pass, 1.9, 5, 5 / 0 / 0 | pass, 2.3, 5, 5 / 0 / 0 | 0.81× |
| bugfix-javascript-grade-school | pass, 1.3, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | 1.73× |
| bugfix-python-scale-generator | pass, 3.1, 5, 5 / 0 / 0 | pass, 2.0, 5, 5 / 0 / 0 | 1.52× |
| bugfix-javascript-complex-numbers | pass, 2.0, 5, 5 / 0 / 0 | runaway (time), 12.0, 35, 34 / 2 / 0 | 0.16× |
| bugfix-python-pig-latin | pass, 2.5, 7, 6 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | 1.75× |
| bugfix-javascript-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 | pass, 5.0, 5, 5 / 0 / 0 | 0.52× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| gains | 12 / 12 | 26.1 | 2.0 | 4.23, 24.54 | 73 | 24.8 | 63 / 1 / 0 | 0 |
| v4c2 | 11 / 12 | 31.3 | 1.6 | 2.42, 9.29 | 136 | 26.1 | 87 / 2 / 0 | 1 |

Tasks scored on every arm: 12.
- v4c2 against gains: 31.3 against 26.1 minutes in all (0.83× faster), per task 1.13× (geometric mean, range 0.16–1.94); passed 11 against 12

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | gains: tasks, TTFT s, new prompt tokens | v4c2: tasks, TTFT s, new prompt tokens |
|---|---|---|
| 1 | 12, 10.75, 789 | 12, 6.65, 809 |
| 2 | 12, 21.02, 1668 | 12, 7.97, 1550 |
| 3 | 12, 5.84, 404 | 12, 3.25, 410 |
| 4 | 12, 1.21, 30 | 12, 1.05, 31 |
| 5 | 12, 2.07, 84 | 12, 1.72, 109 |
| 6 | 5, 3.00, 209 | 3, 1.25, 46 |
| 7 | 2, 1.73, 92 | 1, 1.82, 179 |
| 8 | 1, 1.81, 67 | 1, 2.44, 324 |
| 9 | 1, 5.59, 344 | 1, 0.90, 22 |
| 10 | 0, —, — | 1, 1.16, 37 |
| 11 | 0, —, — | 1, 1.67, 79 |
| 12 | 0, —, — | 1, 2.40, 308 |
| 13 | 0, —, — | 1, 2.35, 287 |
| 14 | 0, —, — | 1, 1.31, 46 |
| 15 | 0, —, — | 1, 2.47, 309 |
| 16 | 0, —, — | 1, 9.49, 1082 |
| 17 | 0, —, — | 1, 1.01, 24 |
| 18 | 0, —, — | 1, 3.93, 570 |
| 19 | 0, —, — | 1, 1.07, 25 |
| 20 | 0, —, — | 1, 9.04, 680 |
| 21 | 0, —, — | 1, 1.41, 36 |
| 22 | 0, —, — | 1, 2.13, 206 |
| 23 | 0, —, — | 1, 2.28, 239 |
| 24 | 0, —, — | 1, 0.92, 19 |
| 25 | 0, —, — | 1, 1.51, 60 |
| 26 | 0, —, — | 1, 1.13, 29 |
| 27 | 0, —, — | 1, 3.87, 548 |
| 28 | 0, —, — | 1, 9.79, 851 |
| 29 | 0, —, — | 1, 0.93, 19 |
| 30 | 0, —, — | 1, 1.88, 130 |
| 31 | 0, —, — | 1, 11.48, 1258 |
| 32 | 0, —, — | 1, 10.17, 727 |
| 33 | 0, —, — | 1, 19.98, 2920 |
| 34 | 0, —, — | 1, 1.14, 25 |
| 35 | 0, —, — | 1, —, — |
