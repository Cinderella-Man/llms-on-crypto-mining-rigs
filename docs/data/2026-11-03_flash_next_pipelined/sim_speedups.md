# Q550: pipelined speculation for Flash-Next in the event model (est.)

`run/sim.sh`: [bench/ngram_replay/pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py) on Q51's 38 Flint
transcripts (65,558 generated tokens), with Flash-Next's per-card GPU time from Q215's decode profile scaled to a plain
token of 112, 140 and 185 ms, 0.6 ms per hop, 3 ms of host time per sample. Speed-ups over plain decoding per prompt
class; "Right" and "Wrong guesses" are shares of the generated tokens; the last column is the card time spent on
dropped guesses. Guessers: `copy, match >= 2` is llama-server's `copy`; `copy >= 4, else request counts (no table)` is
its `cont`; `copy >= 4, else table + request counts, share >= 0.3` is `staircase` with `--pipe-store` (a store of other
transcripts' answers, never the transcript's own); the oracles guess each token right with probability p. The rows at
depth 8 are each setup's full depth (one less than the cards).

| Setup | Stage ms per card | Plain tokens/s | One token per slowest stage + hop |
|---|---|---|---|
| fn-112ms | 11.87 / 11.26 / 10.64 / 13.79 / 10.88 / 11.42 / 12.37 / 9.14 / 12.86 | 8.93 | 69.5 |
| fn-140ms | 15.13 / 14.36 / 13.57 / 17.58 / 13.87 / 14.56 / 15.77 / 11.65 / 15.71 | 7.14 | 55.0 |
| fn-185ms | 20.38 / 19.34 / 18.27 / 23.68 / 18.68 / 19.61 / 21.25 / 15.70 / 20.29 | 5.41 | 41.2 |

| Setup | Guesser | Depth | Chat | Code | Edits | Agent | All | File rewrites | Right guesses | Wrong guesses | Wasted card-ms per token |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fn-112ms | copy, match >= 2 | depth 8 | 1.13 | 1.31 | 1.95 | 1.24 | 1.41 | 6.88 | 33.6 % | 16.4 % | 69.18 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 8 | 1.21 | 1.41 | 2.10 | 1.34 | 1.52 | 6.99 | 39.2 % | 40.3 % | 169.42 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 8 | 1.18 | 1.38 | 2.06 | 1.42 | 1.49 | 7.09 | 38.0 % | 32.5 % | 130.90 |
| fn-112ms | oracle p 0.5 | depth 8 | 1.75 | 1.77 | 1.76 | 1.76 | 1.76 | 1.75 | 49.7 % | 50.2 % | 212.46 |
| fn-112ms | oracle p 0.7 | depth 8 | 2.54 | 2.57 | 2.56 | 2.56 | 2.56 | 2.58 | 70.0 % | 30.0 % | 128.35 |
| fn-112ms | oracle p 0.9 | depth 8 | 4.62 | 4.61 | 4.65 | 4.62 | 4.63 | 4.72 | 90.0 % | 10.0 % | 43.45 |
| fn-112ms | copy >= 4, else oracle p 0.5 | depth 8 | 1.76 | 2.00 | 2.63 | 1.85 | 2.10 | 6.95 | 60.1 % | 39.8 % | 168.46 |
| fn-112ms | copy >= 4, else oracle p 0.7 | depth 8 | 2.46 | 2.75 | 3.35 | 2.57 | 2.84 | 7.13 | 74.3 % | 25.6 % | 109.71 |
| fn-112ms | oracle p 0.7 | depth 1 | 1.67 | 1.67 | 1.68 | 1.67 | 1.67 | 1.69 | 70.0 % | 30.0 % | 18.93 |
| fn-112ms | oracle p 0.7 | depth 2 | 2.06 | 2.08 | 2.07 | 2.08 | 2.07 | 2.08 | 70.0 % | 30.0 % | 38.61 |
| fn-112ms | oracle p 0.7 | depth 3 | 2.29 | 2.31 | 2.31 | 2.30 | 2.30 | 2.32 | 70.0 % | 30.0 % | 57.96 |
| fn-112ms | oracle p 0.7 | depth 4 | 2.42 | 2.44 | 2.44 | 2.43 | 2.43 | 2.46 | 70.0 % | 30.0 % | 76.62 |
| fn-112ms | oracle p 0.7 | depth 5 | 2.49 | 2.52 | 2.51 | 2.51 | 2.51 | 2.53 | 70.0 % | 30.0 % | 93.77 |
| fn-112ms | oracle p 0.7 | depth 6 | 2.53 | 2.56 | 2.55 | 2.55 | 2.55 | 2.57 | 70.0 % | 30.0 % | 108.79 |
| fn-112ms | oracle p 0.7 | depth 7 | 2.54 | 2.58 | 2.56 | 2.56 | 2.56 | 2.58 | 70.0 % | 30.0 % | 121.65 |
| fn-112ms | oracle p 0.7 | depth 8 | hop 0.1 | 2.54 | 2.57 | 2.56 | 2.56 | 2.56 | 2.58 | 70.0 % | 30.0 % | 133.40 |
| fn-112ms | oracle p 0.7 | depth 8 | hop 0.6 | 2.54 | 2.57 | 2.56 | 2.56 | 2.56 | 2.58 | 70.0 % | 30.0 % | 128.35 |
| fn-112ms | oracle p 0.7 | depth 8 | hop 1.0 | 2.54 | 2.57 | 2.56 | 2.56 | 2.56 | 2.58 | 70.0 % | 30.0 % | 124.31 |
| fn-112ms | oracle p 0.7 | depth 8 | stages x1.1 | 2.32 | 2.35 | 2.34 | 2.34 | 2.34 | 2.36 | 70.0 % | 30.0 % | 141.19 |
| fn-112ms | copy, match >= 2 | depth 1 | 1.09 | 1.18 | 1.43 | 1.15 | 1.23 | 2.01 | 33.6 % | 16.4 % | 12.12 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 1 | 1.15 | 1.25 | 1.49 | 1.22 | 1.29 | 2.01 | 39.2 % | 40.3 % | 32.17 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 1 | 1.13 | 1.23 | 1.47 | 1.26 | 1.28 | 2.02 | 38.0 % | 32.5 % | 25.53 |
| fn-112ms | copy, match >= 2 | depth 2 | 1.12 | 1.24 | 1.64 | 1.20 | 1.32 | 2.97 | 33.6 % | 16.4 % | 24.12 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 2 | 1.18 | 1.33 | 1.74 | 1.29 | 1.40 | 2.98 | 39.2 % | 40.3 % | 63.48 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 2 | 1.16 | 1.31 | 1.72 | 1.35 | 1.38 | 3.00 | 38.0 % | 32.5 % | 49.62 |
| fn-112ms | copy, match >= 2 | depth 3 | 1.12 | 1.27 | 1.76 | 1.22 | 1.36 | 3.88 | 33.6 % | 16.4 % | 35.49 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 3 | 1.20 | 1.37 | 1.88 | 1.31 | 1.45 | 3.90 | 39.2 % | 40.3 % | 91.73 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 3 | 1.18 | 1.34 | 1.85 | 1.38 | 1.43 | 3.94 | 38.0 % | 32.5 % | 71.14 |
| fn-112ms | copy, match >= 2 | depth 4 | 1.13 | 1.29 | 1.84 | 1.23 | 1.38 | 4.76 | 33.6 % | 16.4 % | 45.55 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 4 | 1.20 | 1.39 | 1.98 | 1.33 | 1.48 | 4.81 | 39.2 % | 40.3 % | 116.03 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 4 | 1.18 | 1.36 | 1.94 | 1.40 | 1.46 | 4.86 | 38.0 % | 32.5 % | 89.74 |
| fn-112ms | copy, match >= 2 | depth 5 | 1.13 | 1.30 | 1.90 | 1.24 | 1.40 | 5.60 | 33.6 % | 16.4 % | 54.26 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 5 | 1.21 | 1.40 | 2.04 | 1.34 | 1.50 | 5.67 | 39.2 % | 40.3 % | 136.61 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 5 | 1.18 | 1.37 | 2.00 | 1.41 | 1.48 | 5.74 | 38.0 % | 32.5 % | 105.47 |
| fn-112ms | copy, match >= 2 | depth 6 | 1.13 | 1.30 | 1.93 | 1.24 | 1.41 | 6.40 | 33.6 % | 16.4 % | 61.13 |
| fn-112ms | copy >= 4, else request counts (no table) | depth 6 | 1.21 | 1.41 | 2.08 | 1.34 | 1.51 | 6.50 | 39.2 % | 40.3 % | 152.19 |
| fn-112ms | copy >= 4, else table + request counts, share >= 0.3 | depth 6 | 1.18 | 1.38 | 2.04 | 1.42 | 1.49 | 6.59 | 38.0 % | 32.5 % | 117.48 |
| fn-140ms | copy, match >= 2 | depth 8 | 1.13 | 1.30 | 1.95 | 1.24 | 1.41 | 6.81 | 33.6 % | 16.4 % | 88.23 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 8 | 1.20 | 1.40 | 2.09 | 1.33 | 1.51 | 6.92 | 39.2 % | 40.3 % | 216.06 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 8 | 1.18 | 1.38 | 2.05 | 1.41 | 1.49 | 7.02 | 38.0 % | 32.5 % | 166.94 |
| fn-140ms | oracle p 0.5 | depth 8 | 1.74 | 1.76 | 1.75 | 1.75 | 1.75 | 1.74 | 49.7 % | 50.2 % | 270.96 |
| fn-140ms | oracle p 0.7 | depth 8 | 2.53 | 2.56 | 2.55 | 2.55 | 2.55 | 2.57 | 70.0 % | 30.0 % | 163.69 |
| fn-140ms | oracle p 0.9 | depth 8 | 4.59 | 4.58 | 4.62 | 4.59 | 4.60 | 4.69 | 90.0 % | 10.0 % | 55.42 |
| fn-140ms | copy >= 4, else oracle p 0.5 | depth 8 | 1.75 | 1.99 | 2.62 | 1.84 | 2.09 | 6.89 | 60.1 % | 39.8 % | 214.84 |
| fn-140ms | copy >= 4, else oracle p 0.7 | depth 8 | 2.45 | 2.73 | 3.33 | 2.56 | 2.82 | 7.06 | 74.3 % | 25.6 % | 139.91 |
| fn-140ms | oracle p 0.7 | depth 1 | 1.67 | 1.67 | 1.67 | 1.67 | 1.67 | 1.68 | 70.0 % | 30.0 % | 24.15 |
| fn-140ms | oracle p 0.7 | depth 2 | 2.05 | 2.07 | 2.07 | 2.07 | 2.06 | 2.07 | 70.0 % | 30.0 % | 49.24 |
| fn-140ms | oracle p 0.7 | depth 3 | 2.29 | 2.30 | 2.30 | 2.30 | 2.30 | 2.32 | 70.0 % | 30.0 % | 73.92 |
| fn-140ms | oracle p 0.7 | depth 4 | 2.41 | 2.44 | 2.43 | 2.42 | 2.43 | 2.45 | 70.0 % | 30.0 % | 97.72 |
| fn-140ms | oracle p 0.7 | depth 5 | 2.49 | 2.51 | 2.50 | 2.50 | 2.50 | 2.53 | 70.0 % | 30.0 % | 119.59 |
| fn-140ms | oracle p 0.7 | depth 6 | 2.53 | 2.56 | 2.54 | 2.55 | 2.54 | 2.56 | 70.0 % | 30.0 % | 138.75 |
| fn-140ms | oracle p 0.7 | depth 7 | 2.54 | 2.57 | 2.56 | 2.56 | 2.56 | 2.58 | 70.0 % | 30.0 % | 155.14 |
| fn-140ms | copy, match >= 2 | depth 1 | 1.09 | 1.18 | 1.43 | 1.15 | 1.23 | 2.00 | 33.6 % | 16.4 % | 15.45 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 1 | 1.14 | 1.25 | 1.49 | 1.22 | 1.29 | 2.01 | 39.2 % | 40.3 % | 41.03 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 1 | 1.13 | 1.23 | 1.47 | 1.26 | 1.28 | 2.01 | 38.0 % | 32.5 % | 32.55 |
| fn-140ms | copy, match >= 2 | depth 2 | 1.12 | 1.24 | 1.64 | 1.20 | 1.32 | 2.96 | 33.6 % | 16.4 % | 30.76 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 2 | 1.18 | 1.33 | 1.74 | 1.29 | 1.40 | 2.97 | 39.2 % | 40.3 % | 80.96 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 2 | 1.16 | 1.31 | 1.71 | 1.35 | 1.38 | 2.99 | 38.0 % | 32.5 % | 63.29 |
| fn-140ms | copy, match >= 2 | depth 3 | 1.12 | 1.27 | 1.76 | 1.22 | 1.36 | 3.86 | 33.6 % | 16.4 % | 45.26 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 3 | 1.20 | 1.37 | 1.88 | 1.31 | 1.45 | 3.89 | 39.2 % | 40.3 % | 116.98 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 3 | 1.18 | 1.34 | 1.85 | 1.38 | 1.43 | 3.92 | 38.0 % | 32.5 % | 90.73 |
| fn-140ms | copy, match >= 2 | depth 4 | 1.13 | 1.29 | 1.84 | 1.23 | 1.38 | 4.74 | 33.6 % | 16.4 % | 58.09 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 4 | 1.20 | 1.39 | 1.97 | 1.33 | 1.48 | 4.79 | 39.2 % | 40.3 % | 147.97 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 4 | 1.18 | 1.36 | 1.94 | 1.40 | 1.46 | 4.84 | 38.0 % | 32.5 % | 114.45 |
| fn-140ms | copy, match >= 2 | depth 5 | 1.13 | 1.30 | 1.89 | 1.24 | 1.40 | 5.58 | 33.6 % | 16.4 % | 69.20 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 5 | 1.20 | 1.40 | 2.04 | 1.34 | 1.50 | 5.65 | 39.2 % | 40.3 % | 174.23 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 5 | 1.18 | 1.37 | 2.00 | 1.41 | 1.48 | 5.71 | 38.0 % | 32.5 % | 134.50 |
| fn-140ms | copy, match >= 2 | depth 6 | 1.13 | 1.30 | 1.93 | 1.24 | 1.41 | 6.37 | 33.6 % | 16.4 % | 77.96 |
| fn-140ms | copy >= 4, else request counts (no table) | depth 6 | 1.21 | 1.41 | 2.08 | 1.34 | 1.51 | 6.47 | 39.2 % | 40.3 % | 194.09 |
| fn-140ms | copy >= 4, else table + request counts, share >= 0.3 | depth 6 | 1.18 | 1.38 | 2.04 | 1.42 | 1.49 | 6.56 | 38.0 % | 32.5 % | 149.83 |
| fn-185ms | copy, match >= 2 | depth 8 | 1.13 | 1.30 | 1.94 | 1.24 | 1.41 | 6.75 | 33.6 % | 16.4 % | 118.58 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 8 | 1.19 | 1.39 | 2.08 | 1.33 | 1.50 | 6.86 | 39.2 % | 40.3 % | 290.65 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 8 | 1.17 | 1.37 | 2.04 | 1.41 | 1.48 | 6.96 | 38.0 % | 32.5 % | 224.54 |
| fn-185ms | oracle p 0.5 | depth 8 | 1.73 | 1.75 | 1.74 | 1.74 | 1.74 | 1.73 | 49.7 % | 50.2 % | 363.77 |
| fn-185ms | oracle p 0.7 | depth 8 | 2.52 | 2.55 | 2.54 | 2.54 | 2.54 | 2.56 | 70.0 % | 30.0 % | 219.65 |
| fn-185ms | oracle p 0.9 | depth 8 | 4.57 | 4.55 | 4.59 | 4.56 | 4.57 | 4.66 | 90.0 % | 10.0 % | 74.49 |
| fn-185ms | copy >= 4, else oracle p 0.5 | depth 8 | 1.74 | 1.98 | 2.60 | 1.83 | 2.07 | 6.82 | 60.1 % | 39.8 % | 288.46 |
| fn-185ms | copy >= 4, else oracle p 0.7 | depth 8 | 2.44 | 2.72 | 3.31 | 2.54 | 2.81 | 7.00 | 74.3 % | 25.6 % | 187.74 |
| fn-185ms | oracle p 0.7 | depth 1 | 1.66 | 1.66 | 1.67 | 1.66 | 1.66 | 1.68 | 70.0 % | 30.0 % | 32.53 |
| fn-185ms | oracle p 0.7 | depth 2 | 2.05 | 2.07 | 2.06 | 2.07 | 2.06 | 2.07 | 70.0 % | 30.0 % | 66.33 |
| fn-185ms | oracle p 0.7 | depth 3 | 2.28 | 2.30 | 2.30 | 2.30 | 2.29 | 2.31 | 70.0 % | 30.0 % | 99.57 |
| fn-185ms | oracle p 0.7 | depth 4 | 2.41 | 2.43 | 2.43 | 2.42 | 2.42 | 2.45 | 70.0 % | 30.0 % | 131.62 |
| fn-185ms | oracle p 0.7 | depth 5 | 2.48 | 2.51 | 2.50 | 2.50 | 2.50 | 2.52 | 70.0 % | 30.0 % | 161.08 |
| fn-185ms | oracle p 0.7 | depth 6 | 2.52 | 2.55 | 2.54 | 2.54 | 2.54 | 2.56 | 70.0 % | 30.0 % | 186.89 |
| fn-185ms | oracle p 0.7 | depth 7 | 2.53 | 2.56 | 2.55 | 2.55 | 2.55 | 2.57 | 70.0 % | 30.0 % | 208.96 |
| fn-185ms | copy, match >= 2 | depth 1 | 1.09 | 1.18 | 1.42 | 1.15 | 1.23 | 2.00 | 33.6 % | 16.4 % | 20.81 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 1 | 1.14 | 1.24 | 1.48 | 1.22 | 1.29 | 2.00 | 39.2 % | 40.3 % | 55.27 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 1 | 1.13 | 1.23 | 1.47 | 1.26 | 1.28 | 2.00 | 38.0 % | 32.5 % | 43.85 |
| fn-185ms | copy, match >= 2 | depth 2 | 1.12 | 1.24 | 1.64 | 1.20 | 1.32 | 2.94 | 33.6 % | 16.4 % | 41.43 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 2 | 1.18 | 1.33 | 1.73 | 1.29 | 1.40 | 2.96 | 39.2 % | 40.3 % | 109.05 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 2 | 1.16 | 1.31 | 1.71 | 1.35 | 1.38 | 2.98 | 38.0 % | 32.5 % | 85.24 |
| fn-185ms | copy, match >= 2 | depth 3 | 1.12 | 1.27 | 1.76 | 1.22 | 1.36 | 3.85 | 33.6 % | 16.4 % | 60.97 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 3 | 1.20 | 1.37 | 1.88 | 1.31 | 1.45 | 3.87 | 39.2 % | 40.3 % | 157.57 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 3 | 1.18 | 1.34 | 1.85 | 1.38 | 1.43 | 3.91 | 38.0 % | 32.5 % | 122.21 |
| fn-185ms | copy, match >= 2 | depth 4 | 1.13 | 1.29 | 1.84 | 1.23 | 1.38 | 4.72 | 33.6 % | 16.4 % | 78.25 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 4 | 1.20 | 1.39 | 1.97 | 1.33 | 1.48 | 4.77 | 39.2 % | 40.3 % | 199.31 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 4 | 1.18 | 1.36 | 1.94 | 1.40 | 1.46 | 4.82 | 38.0 % | 32.5 % | 154.16 |
| fn-185ms | copy, match >= 2 | depth 5 | 1.13 | 1.30 | 1.89 | 1.24 | 1.40 | 5.55 | 33.6 % | 16.4 % | 93.21 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 5 | 1.20 | 1.40 | 2.04 | 1.34 | 1.50 | 5.62 | 39.2 % | 40.3 % | 234.67 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 5 | 1.18 | 1.37 | 1.99 | 1.41 | 1.48 | 5.69 | 38.0 % | 32.5 % | 181.17 |
| fn-185ms | copy, match >= 2 | depth 6 | 1.13 | 1.30 | 1.93 | 1.24 | 1.41 | 6.35 | 33.6 % | 16.4 % | 105.00 |
| fn-185ms | copy >= 4, else request counts (no table) | depth 6 | 1.21 | 1.40 | 2.08 | 1.34 | 1.51 | 6.45 | 39.2 % | 40.3 % | 261.43 |
| fn-185ms | copy >= 4, else table + request counts, share >= 0.3 | depth 6 | 1.18 | 1.38 | 2.04 | 1.42 | 1.49 | 6.53 | 38.0 % | 32.5 % | 201.81 |
