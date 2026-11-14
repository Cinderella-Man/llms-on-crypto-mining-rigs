60 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| o3a | 12 | 100.0 % | 2.5, 4.7 | 59 / 117 / 1 / 2 | 5.0 | 3618 | 86.2 % | 9.66 | 0/71 | 0 |
| o4a | 12 | 100.0 % | 2.1, 4.5 | 53 / 93 / 1 / 2 | 5.0 | 2991 | 86.5 % | 8.54 | 0/62 | 0 |
| o4b | 12 | 100.0 % | 2.0, 4.1 | 54 / 99 / 0 / 2 | 4.5 | 3506 | 83.7 % | 9.95 | 0/57 | 0 |
| o4sa | 12 | 100.0 % | 1.9, 3.7 | 29 / 107 / 0 / 2 | 5.0 | 3316 | 86.1 % | 3.50 | 0/64 | 0 |
| o4sb | 12 | 100.0 % | 1.5, 4.7 | 30 / 97 / 1 / 2 | 4.5 | 3439 | 85.1 % | 3.77 | 0/65 | 0 |
