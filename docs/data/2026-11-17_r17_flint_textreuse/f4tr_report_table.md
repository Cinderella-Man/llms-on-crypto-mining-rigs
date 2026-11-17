12 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways | prompt tokens read | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f4tr | 12 | 100.0 % | 1.3, 3.9 | 14 / 117 / 1 / 2 | 6.0 | 2970 | 90.4 % | 1.81 | 0/71 | 0 | 35635 | 0 (0; 0.0 of 169.7) | 11.68 (bugfix-javascript-complex-numbers, 2) |
