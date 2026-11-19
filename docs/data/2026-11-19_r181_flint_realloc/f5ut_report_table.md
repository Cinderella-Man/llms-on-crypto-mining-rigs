12 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways | prompt tokens read | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f5ut | 12 | 100.0 % | 0.9, 3.3 | 14 / 74 / 1 / 2 | 6.0 | 2875 | 89.5 % | 1.77 | 0/71 | 0 | 34502 | 0 (0; 0.0 of 162.1) | 12.53 (bugfix-javascript-complex-numbers, 2) |
