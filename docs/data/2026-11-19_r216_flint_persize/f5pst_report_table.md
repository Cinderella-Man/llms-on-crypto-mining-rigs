12 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways | prompt tokens read | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f5pst | 12 | 100.0 % | 1.1, 2.5 | 8 / 86 / 1 / 2 | 5.0 | 2954 | 89.3 % | 1.31 | 0/66 | 0 | 35454 | 0 (0; 0.0 of 101.8) | 9.22 (bugfix-python-affine-cipher, 1) |
