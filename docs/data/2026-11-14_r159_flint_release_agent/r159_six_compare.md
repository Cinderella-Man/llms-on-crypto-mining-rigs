72 scored attempts, arms f3a, f4a, f3b, f4b, f4ta, f4tb; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f3a: result, min, turns, tool calls | f4a: result, min, turns, tool calls | f3b: result, min, turns, tool calls | f4b: result, min, turns, tool calls | f4ta: result, min, turns, tool calls | f4tb: result, min, turns, tool calls | time f3a / f4a | time f3a / f3b | time f3a / f4b | time f3a / f4ta | time f3a / f4tb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 1.7, 7, 6 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | pass, 1.8, 5, 4 / 0 / 0 | pass, 2.9, 5, 5 / 0 / 0 | pass, 1.7, 6, 5 / 0 / 0 | 1.16× | 0.76× | 0.94× | 0.58× | 1.00× |
| bugfix-javascript-space-age | pass, 0.4, 4, 4 / 0 / 0 | pass, 0.7, 6, 6 / 0 / 0 | pass, 0.8, 7, 6 / 0 / 0 | pass, 0.5, 5, 5 / 0 / 0 | pass, 0.8, 7, 6 / 0 / 0 | pass, 1.0, 8, 8 / 1 / 0 | 0.63× | 0.54× | 0.85× | 0.53× | 0.42× |
| bugfix-python-phone-number | pass, 1.2, 6, 5 / 0 / 0 | pass, 1.6, 7, 7 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | pass, 1.0, 6, 5 / 0 / 0 | pass, 1.0, 5, 5 / 0 / 0 | pass, 1.6, 8, 8 / 0 / 0 | 0.79× | 0.81× | 1.26× | 1.26× | 0.76× |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 | pass, 0.4, 5, 4 / 0 / 0 | pass, 0.3, 4, 4 / 0 / 0 | pass, 0.3, 4, 3 / 0 / 0 | pass, 0.3, 4, 3 / 0 / 0 | pass, 0.3, 4, 4 / 0 / 0 | 0.87× | 1.10× | 1.48× | 1.30× | 1.14× |
| bugfix-python-variable-length-quantity | pass, 0.7, 4, 3 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.8, 5, 5 / 0 / 0 | pass, 0.6, 6, 5 / 0 / 0 | pass, 0.6, 5, 5 / 0 / 0 | pass, 0.9, 6, 6 / 0 / 0 | 0.90× | 0.83× | 1.07× | 1.05× | 0.74× |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 | pass, 1.0, 5, 5 / 0 / 0 | pass, 1.0, 5, 4 / 0 / 0 | pass, 0.9, 6, 5 / 0 / 0 | pass, 0.8, 10, 9 / 2 / 0 | pass, 0.8, 5, 5 / 0 / 0 | 0.80× | 0.82× | 0.87× | 0.95× | 1.02× |
| bugfix-python-list-ops | pass, 1.7, 6, 5 / 0 / 0 | pass, 3.5, 5, 5 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | pass, 2.1, 5, 5 / 0 / 0 | pass, 1.8, 6, 5 / 0 / 0 | pass, 1.2, 6, 5 / 0 / 0 | 0.50× | 1.03× | 0.84× | 0.97× | 1.41× |
| bugfix-javascript-grade-school | pass, 1.1, 10, 9 / 0 / 0 | pass, 0.7, 5, 5 / 0 / 0 | pass, 0.8, 8, 8 / 0 / 0 | pass, 0.6, 6, 6 / 0 / 0 | pass, 0.6, 5, 5 / 0 / 0 | pass, 0.6, 5, 5 / 0 / 0 | 1.59× | 1.44× | 1.82× | 1.78× | 1.87× |
| bugfix-python-scale-generator | runaway (time), 12.0, 4, 3 / 0 / 0 | pass, 11.5, 9, 9 / 0 / 0 | pass, 5.7, 6, 6 / 0 / 0 | pass, 7.8, 9, 8 / 0 / 0 | runaway (time), 12.0, 5, 4 / 0 / 0 | pass, 10.2, 6, 5 / 0 / 0 | 1.04× | 2.11× | 1.53× | 1.00× | 1.17× |
| bugfix-javascript-complex-numbers | pass, 1.1, 5, 4 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | pass, 1.2, 5, 5 / 0 / 0 | pass, 1.0, 6, 6 / 0 / 0 | 1.15× | 1.44× | 1.21× | 0.88× | 1.05× |
| bugfix-python-pig-latin | pass, 2.3, 6, 5 / 0 / 0 | pass, 4.4, 11, 10 / 1 / 0 | pass, 2.2, 7, 6 / 0 / 0 | pass, 4.0, 10, 9 / 0 / 0 | pass, 3.1, 8, 8 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | 0.53× | 1.05× | 0.57× | 0.75× | 1.35× |
| bugfix-javascript-variable-length-quantity | pass, 1.3, 5, 5 / 0 / 0 | pass, 4.5, 5, 5 / 0 / 0 | pass, 2.0, 6, 5 / 0 / 0 | pass, 8.7, 13, 12 / 0 / 0 | pass, 3.7, 8, 8 / 1 / 0 | pass, 2.2, 6, 6 / 0 / 0 | 0.30× | 0.67× | 0.15× | 0.37× | 0.61× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f3a | 11 / 12 | 24.7 | 1.2 | 2.65, 7.20 | 179 | 48.5 | 57 / 0 / 0 | 1 |
| f4a | 12 / 12 | 31.3 | 1.2 | 2.85, 7.84 | 177 | 47.4 | 70 / 1 / 0 | 0 |
| f3b | 12 / 12 | 19.7 | 1.2 | 2.61, 7.97 | 181 | 48.0 | 61 / 0 / 0 | 0 |
| f4b | 12 / 12 | 29.2 | 1.0 | 2.56, 6.76 | 179 | 48.2 | 72 / 0 / 0 | 0 |
| f4ta | 11 / 12 | 28.8 | 1.1 | 1.74, 6.61 | 220 | 49.3 | 68 / 3 / 0 | 1 |
| f4tb | 12 / 12 | 23.3 | 1.1 | 1.82, 6.35 | 204 | 47.0 | 68 / 1 / 0 | 0 |

Tasks scored on every arm: 12.
- f4a against f3a: 31.3 against 24.7 minutes in all (0.79× faster), per task 0.78× (geometric mean, range 0.30–1.59); passed 12 against 11
- f3b against f3a: 19.7 against 24.7 minutes in all (1.26× faster), per task 0.98× (geometric mean, range 0.54–2.11); passed 12 against 11
- f4b against f3a: 29.2 against 24.7 minutes in all (0.85× faster), per task 0.91× (geometric mean, range 0.15–1.82); passed 12 against 11
- f4ta against f3a: 28.8 against 24.7 minutes in all (0.86× faster), per task 0.88× (geometric mean, range 0.37–1.78); passed 11 against 11
- f4tb against f3a: 23.3 against 24.7 minutes in all (1.06× faster), per task 0.97× (geometric mean, range 0.42–1.87); passed 12 against 11

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f3a: graph reallocations 64 of 65 requests (188 same size, 0 changed)
- f4a: graph reallocations 71 of 73 requests (323 same size, 0 changed)
- f3b: graph reallocations 66 of 68 requests (237 same size, 0 changed)
- f4b: graph reallocations 79 of 80 requests (308 same size, 0 changed)
- f4ta: graph reallocations 71 of 72 requests (250 same size, 0 changed)
- f4tb: graph reallocations 70 of 71 requests (290 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f3a: tasks, TTFT s, new prompt tokens | f4a: tasks, TTFT s, new prompt tokens | f3b: tasks, TTFT s, new prompt tokens | f4b: tasks, TTFT s, new prompt tokens | f4ta: tasks, TTFT s, new prompt tokens | f4tb: tasks, TTFT s, new prompt tokens |
|---|---|---|---|---|---|---|
| 1 | 12, 4.17, 693 | 12, 4.29, 693 | 12, 4.18, 693 | 12, 4.16, 693 | 12, 3.24, 693 | 12, 3.18, 693 |
| 2 | 12, 5.58, 1196 | 12, 6.24, 1334 | 12, 5.75, 1291 | 12, 2.48, 372 | 12, 5.23, 1230 | 12, 4.23, 1013 |
| 3 | 12, 2.76, 463 | 12, 2.93, 496 | 12, 2.81, 400 | 12, 3.71, 774 | 12, 2.38, 508 | 12, 3.29, 676 |
| 4 | 12, 2.22, 193 | 12, 1.15, 44 | 12, 1.57, 156 | 12, 2.08, 290 | 12, 0.77, 29 | 12, 1.82, 394 |
| 5 | 8, 1.85, 190 | 12, 1.70, 142 | 11, 1.14, 36 | 11, 1.04, 20 | 11, 1.30, 100 | 11, 0.90, 20 |
| 6 | 5, 1.04, 20 | 4, 1.87, 250 | 5, 1.31, 68 | 7, 1.56, 218 | 5, 0.66, 29 | 8, 1.10, 58 |
| 7 | 2, 1.43, 59 | 3, 1.21, 34 | 3, 0.99, 20 | 3, 1.92, 157 | 4, 0.62, 17 | 2, 0.56, 16 |
| 8 | 1, 3.13, 473 | 2, 1.52, 63 | 1, 1.77, 121 | 3, 3.79, 379 | 3, 0.65, 25 | 2, 0.82, 76 |
| 9 | 1, 0.91, 19 | 2, 14.66, 2662 | 0, —, — | 3, 1.28, 20 | 1, 0.58, 18 | 0, —, — |
| 10 | 1, 1.73, 245 | 1, 1.76, 106 | 0, —, — | 2, 2.16, 130 | 1, 0.93, 84 | 0, —, — |
| 11 | 0, —, — | 1, 1.02, 20 | 0, —, — | 1, 1.19, 14 | 0, —, — | 0, —, — |
| 12 | 0, —, — | 0, —, — | 0, —, — | 1, 1.33, 21 | 0, —, — | 0, —, — |
| 13 | 0, —, — | 0, —, — | 0, —, — | 1, 2.55, 99 | 0, —, — | 0, —, — |
