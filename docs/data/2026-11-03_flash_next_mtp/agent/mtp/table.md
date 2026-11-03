10 scored attempts -> res/attempts.csv; 2 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| mtp | 1 | 100.0 % | 8.1, 8.1 | 115 / 272 / 62 / 37 | 8 | 3388 | 91.9 % | 13.96 | 0/8 | 0 |
| nodraft | 9 | 100.0 % | 4.1, 8.7 | 54 / 222 / 8 / 3 | 5 | 3185 | 88.0 % | 8.09 | 0/47 | 0 |
