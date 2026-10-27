| Setup | Stage ms per card | Plain tokens/s | One token per slowest stage + hop |
|---|---|---|---|
| 6 cards, 9/8/8/8/8/8 layers | 3.91 / 3.48 / 3.48 / 3.48 / 3.48 / 5.98 | 37.31 | 159.3 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | 3.05 / 3.05 / 3.05 / 3.05 / 2.54 / 2.54 / 2.54 / 2.54 / 5.04 | 31.95 | 187.2 |
| 9 cards, head + 2 layers on the last | 3.05 / 3.05 / 3.05 / 3.05 / 3.05 / 3.05 / 3.05 / 2.54 / 3.52 | 31.95 | 262.0 |

| Cards | Guesser | Guess | Depth | Chat | Code | Edits | Agent | All | File rewrites | Right guesses | Wrong guesses | Wasted card-ms per token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | sampled top guess, expected | depth 5 | 1.11 | 1.26 | 1.74 | 1.21 | 1.34 | 4.04 | 33.2 % | 13.1 % | 3.90 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | Gumbel-coupled | depth 5 | 1.11 | 1.25 | 1.73 | 1.21 | 1.34 | 4.04 | 33.1 % | 13.2 % | 3.97 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else table | sampled top guess, expected | depth 5 | 1.12 | 1.25 | 1.72 | 1.33 | 1.35 | 4.02 | 33.6 % | 53.7 % | 28.94 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else table | Gumbel-coupled | depth 5 | 1.10 | 1.24 | 1.71 | 1.32 | 1.33 | 4.03 | 32.4 % | 55.0 % | 29.66 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else table + request counts | sampled top guess, expected | depth 5 | 1.18 | 1.35 | 1.85 | 1.37 | 1.44 | 4.08 | 39.7 % | 53.8 % | 32.46 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else table + request counts | Gumbel-coupled | depth 5 | 1.16 | 1.33 | 1.84 | 1.37 | 1.42 | 4.08 | 38.6 % | 55.0 % | 33.18 |
| 6 cards, 9/8/8/8/8/8 layers | table + request counts | sampled top guess, expected | depth 5 | 1.18 | 1.34 | 1.84 | 1.37 | 1.43 | 4.01 | 39.4 % | 54.1 % | 32.65 |
| 6 cards, 9/8/8/8/8/8 layers | table + request counts | Gumbel-coupled | depth 5 | 1.16 | 1.33 | 1.82 | 1.37 | 1.41 | 4.03 | 38.3 % | 55.2 % | 33.38 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | sampled top guess, expected | depth 8 | 1.12 | 1.28 | 1.85 | 1.23 | 1.38 | 5.41 | 33.2 % | 13.1 % | 5.41 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | Gumbel-coupled | depth 8 | 1.12 | 1.28 | 1.85 | 1.23 | 1.38 | 5.41 | 33.1 % | 13.2 % | 5.51 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else table | sampled top guess, expected | depth 8 | 1.14 | 1.28 | 1.83 | 1.37 | 1.39 | 5.37 | 33.6 % | 53.7 % | 44.23 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else table | Gumbel-coupled | depth 8 | 1.11 | 1.26 | 1.82 | 1.36 | 1.37 | 5.38 | 32.4 % | 55.0 % | 45.26 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else table + request counts | sampled top guess, expected | depth 8 | 1.20 | 1.39 | 1.99 | 1.42 | 1.49 | 5.48 | 39.7 % | 53.8 % | 51.88 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else table + request counts | Gumbel-coupled | depth 8 | 1.18 | 1.37 | 1.97 | 1.41 | 1.47 | 5.49 | 38.6 % | 55.0 % | 52.92 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | table + request counts | sampled top guess, expected | depth 8 | 1.20 | 1.38 | 1.97 | 1.41 | 1.49 | 5.35 | 39.4 % | 54.1 % | 52.22 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | table + request counts | Gumbel-coupled | depth 8 | 1.17 | 1.37 | 1.96 | 1.41 | 1.47 | 5.38 | 38.3 % | 55.2 % | 53.28 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | sampled top guess, expected | depth 8 | 1.13 | 1.31 | 1.94 | 1.25 | 1.41 | 7.30 | 33.2 % | 13.1 % | 5.52 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | Gumbel-coupled | depth 8 | 1.13 | 1.30 | 1.94 | 1.25 | 1.41 | 7.30 | 33.1 % | 13.2 % | 5.62 |
| 9 cards, head + 2 layers on the last | copy >= 4, else table | sampled top guess, expected | depth 8 | 1.15 | 1.30 | 1.92 | 1.40 | 1.42 | 7.21 | 33.6 % | 53.7 % | 43.56 |
| 9 cards, head + 2 layers on the last | copy >= 4, else table | Gumbel-coupled | depth 8 | 1.12 | 1.28 | 1.91 | 1.39 | 1.40 | 7.24 | 32.4 % | 55.0 % | 44.52 |
| 9 cards, head + 2 layers on the last | copy >= 4, else table + request counts | sampled top guess, expected | depth 8 | 1.22 | 1.42 | 2.11 | 1.45 | 1.54 | 7.43 | 39.7 % | 53.8 % | 50.93 |
| 9 cards, head + 2 layers on the last | copy >= 4, else table + request counts | Gumbel-coupled | depth 8 | 1.19 | 1.40 | 2.09 | 1.45 | 1.51 | 7.46 | 38.6 % | 55.0 % | 51.91 |
| 9 cards, head + 2 layers on the last | table + request counts | sampled top guess, expected | depth 8 | 1.22 | 1.42 | 2.09 | 1.45 | 1.53 | 7.19 | 39.4 % | 54.1 % | 51.26 |
| 9 cards, head + 2 layers on the last | table + request counts | Gumbel-coupled | depth 8 | 1.19 | 1.40 | 2.07 | 1.45 | 1.51 | 7.24 | 38.3 % | 55.2 % | 52.25 |

| Guesser | Expected right, top guess: all / fresh / copied | Gumbel-coupled: all / fresh / copied | Fresh, coupled: chat / code / edit / agent |
|---|---|---|---|
| copy >= 4, else ngram-cache | 33.2 / 12.0 / 88.7 % | 33.1 / 11.8 / 88.6 % | 10.4 / 10.4 / 15.7 / 13.4 |
| copy >= 4, else table | 33.6 / 13.9 / 85.0 % | 32.4 / 12.3 / 84.7 % | 10.1 / 10.0 / 16.3 / 24.5 |
| copy >= 4, else table + request counts | 39.7 / 20.5 / 89.9 % | 38.6 / 18.9 / 89.7 % | 15.8 / 17.5 / 23.8 / 27.6 |
| table + request counts | 39.5 / 20.5 / 89.0 % | 38.3 / 18.9 / 88.7 % | 15.7 / 17.5 / 23.9 / 27.7 |
