12 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways | prompt tokens read | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) | card energy kJ (J a task) | J a generated token | W generating / reading the prompt / idle (all cards) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre | 12 | 100.0 % | 1.1, 2.1 | 21 / 53 / 0 / 2 | 5.0 | 3177 | 87.6 % | 2.51 | 0/59 | 0 | 38127 | 0 (0; 0.0 of 248.0) | 19.86 (bugfix-python-affine-cipher, 1) | 321.9 (26827) | 17.93 | 335.0 / 393.0 / 309.5 |

Card energy (FN101): all cards' power from the rig's card log, over each arm's session window (its first request to its last request's end, as the proxy logged them); J a task and J a generated token divide that energy; the watts are the cards' sum while generating (first token to end), reading the prompt (start to first token) and idle between turns (the rest of the window).
