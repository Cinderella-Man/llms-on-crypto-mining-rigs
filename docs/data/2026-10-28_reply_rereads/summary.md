| run | turn pairs | re-reads (checkpoint restores) | where the reply diverged | re-read tokens | re-read s | task time s (12 tasks) | share of task time | share of prompt time | worst first token s | same, if exact |
|---|---|---|---|---|---|---|---|---|---|---|
| d4 | 69 | 4 | reasoning | 16,954 | 94.4 | 1599 | 5.9 % | 31 % | 59.8 | 0.5 |
| q8-d4b | 65 | 5 | reasoning | 29,083 | 150.3 | 1818 | 8.3 % | 43 % | 80.9 | 0.9 |
| base | 61 | 2 | reasoning | 17,294 | 90.8 | 1360 | 6.7 % | 32 % | 82.5 | 3.5 |

| reply length (tokens) | turns | generated tokens | ids change in a text round trip | first change ≥ 256 tokens before the end | in the reasoning | in a tool call |
|---|---|---|---|---|---|---|
| 0–255 | 730 | 53,927 | 5 (0.7 %) | 0 | 0 | 5 |
| 256–1023 | 249 | 135,141 | 9 (3.6 %) | 7 | 5 | 3 |
| 1024–4095 | 199 | 409,707 | 31 (15.6 %) | 28 | 26 | 5 |
| ≥ 4096 | 69 | 527,055 | 31 (44.9 %) | 30 | 29 | 2 |
| all | 1,247 | 1,125,830 | 76 (6.1 %) | 65 | 60 | 15 |
