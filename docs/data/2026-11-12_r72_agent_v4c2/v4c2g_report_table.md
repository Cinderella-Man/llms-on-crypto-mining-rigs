36 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| gains | 12 | 100.0 % | 2.0, 3.1 | 50 / 77 / 0 / 4 | 5.0 | 3634 | 87.2 % | 4.40 | 0/63 | 0 |
| v4c2 | 12 | 91.7 % | 1.6, 5.0 | 33 / 109 / 6 / 9 | 5.0 | 4456 | 86.7 % | 3.17 | 0/87 | 1 |
| v4c2g | 12 | 100.0 % | 1.6, 2.4 | 30 / 76 / 0 / 8 | 5.0 | 3153 | 87.5 % | 4.87 | 0/65 | 0 |
