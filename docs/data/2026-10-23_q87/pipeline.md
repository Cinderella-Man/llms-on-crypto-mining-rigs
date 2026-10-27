| Setup | Stage ms per card | Plain tokens/s | One token per slowest stage + hop |
|---|---|---|---|
| 6 cards, 9/8/8/8/8/8 layers | 3.91 / 3.48 / 3.48 / 3.48 / 3.48 / 5.98 | 37.31 | 159.3 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | 3.05 / 3.05 / 3.05 / 3.05 / 2.54 / 2.54 / 2.54 / 2.54 / 5.04 | 31.95 | 187.2 |
| 9 cards, head + 2 layers on the last | 3.05 / 3.05 / 3.05 / 3.05 / 3.05 / 3.05 / 3.05 / 2.54 / 3.52 | 31.95 | 262.0 |

| Setup | Guesser | Depth | Chat | Code | Edits | Agent | All | File rewrites | Right guesses | Wrong guesses | Wasted card-ms per token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 cards, 9/8/8/8/8/8 layers | copy, match >= 2 | depth 5 | 1.11 | 1.26 | 1.75 | 1.21 | 1.35 | 4.01 | 33.6 % | 16.4 % | 11.42 |
| 6 cards, 9/8/8/8/8/8 layers | copy, match >= 3 | depth 5 | 1.06 | 1.20 | 1.65 | 1.15 | 1.28 | 3.97 | 28.6 % | 7.8 % | 5.49 |
| 6 cards, 9/8/8/8/8/8 layers | copy, match >= 4 | depth 5 | 1.03 | 1.17 | 1.58 | 1.11 | 1.24 | 3.94 | 25.2 % | 4.6 % | 3.30 |
| 6 cards, 9/8/8/8/8/8 layers | copy, match >= 6 | depth 5 | 1.01 | 1.13 | 1.49 | 1.07 | 1.19 | 3.86 | 21.2 % | 2.2 % | 1.58 |
| 6 cards, 9/8/8/8/8/8 layers | copy, match >= 8 | depth 5 | 1.00 | 1.11 | 1.43 | 1.05 | 1.17 | 3.78 | 18.7 % | 1.4 % | 1.04 |
| 6 cards, 9/8/8/8/8/8 layers | copy, match = 12 | depth 5 | 1.00 | 1.08 | 1.36 | 1.03 | 1.14 | 3.66 | 15.6 % | 0.9 % | 0.64 |
| 6 cards, 9/8/8/8/8/8 layers | fixed key 4 (ngram-simple/map-k lookup) | depth 5 | 1.03 | 1.16 | 1.53 | 1.11 | 1.23 | 3.34 | 24.0 % | 5.8 % | 4.14 |
| 6 cards, 9/8/8/8/8/8 layers | ngram-cache counts | depth 5 | 1.11 | 1.25 | 1.68 | 1.21 | 1.33 | 3.46 | 32.2 % | 13.3 % | 8.84 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 5 | 1.11 | 1.26 | 1.74 | 1.21 | 1.34 | 4.04 | 33.3 % | 13.0 % | 8.80 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.3 | depth 5 | 1.29 | 1.30 | 1.29 | 1.29 | 1.30 | 1.29 | 29.8 % | 70.1 % | 48.26 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.4 | depth 5 | 1.43 | 1.45 | 1.44 | 1.44 | 1.44 | 1.43 | 39.8 % | 60.1 % | 41.65 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.5 | depth 5 | 1.61 | 1.62 | 1.61 | 1.61 | 1.61 | 1.61 | 49.7 % | 50.2 % | 35.11 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.6 | depth 5 | 1.84 | 1.85 | 1.85 | 1.87 | 1.85 | 1.87 | 59.9 % | 40.0 % | 28.30 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 5 | 2.14 | 2.16 | 2.15 | 2.16 | 2.15 | 2.17 | 70.0 % | 30.0 % | 21.45 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.8 | depth 5 | 2.57 | 2.59 | 2.60 | 2.55 | 2.59 | 2.61 | 80.1 % | 19.8 % | 14.40 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.9 | depth 5 | 3.21 | 3.21 | 3.22 | 3.21 | 3.21 | 3.25 | 90.0 % | 10.0 % | 7.37 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.95 | depth 5 | 3.67 | 3.66 | 3.67 | 3.68 | 3.66 | 3.69 | 94.9 % | 5.0 % | 3.72 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else oracle p 0.5 | depth 5 | 1.61 | 1.79 | 2.20 | 1.68 | 1.85 | 4.04 | 60.1 % | 39.8 % | 27.86 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else oracle p 0.7 | depth 5 | 2.09 | 2.27 | 2.61 | 2.16 | 2.32 | 4.09 | 74.3 % | 25.6 % | 18.33 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 1 | 1.09 | 1.18 | 1.43 | 1.16 | 1.23 | 2.06 | 33.3 % | 13.0 % | 1.93 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 1 | 1.65 | 1.66 | 1.66 | 1.65 | 1.65 | 1.67 | 70.0 % | 30.0 % | 4.05 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 2 | 1.11 | 1.23 | 1.62 | 1.20 | 1.30 | 3.04 | 33.3 % | 13.0 % | 3.94 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 2 | 1.98 | 1.99 | 1.99 | 1.99 | 1.98 | 1.99 | 70.0 % | 30.0 % | 8.60 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 3 | 1.11 | 1.26 | 1.73 | 1.21 | 1.34 | 3.97 | 33.3 % | 13.0 % | 5.79 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 3 | 2.14 | 2.15 | 2.15 | 2.15 | 2.15 | 2.16 | 70.0 % | 30.0 % | 13.03 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 4 | 1.11 | 1.26 | 1.74 | 1.21 | 1.34 | 4.04 | 33.3 % | 13.0 % | 7.45 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 4 | 2.14 | 2.16 | 2.15 | 2.16 | 2.15 | 2.17 | 70.0 % | 30.0 % | 17.51 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 5 | hop 0.1 | 1.11 | 1.26 | 1.74 | 1.21 | 1.34 | 4.06 | 33.3 % | 13.0 % | 9.22 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 5 | hop 0.1 | 2.15 | 2.17 | 2.16 | 2.16 | 2.16 | 2.17 | 70.0 % | 30.0 % | 22.46 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 5 | hop 0.6 | 1.11 | 1.26 | 1.73 | 1.21 | 1.34 | 4.01 | 33.3 % | 13.0 % | 7.97 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 5 | hop 0.6 | 2.14 | 2.16 | 2.15 | 2.15 | 2.15 | 2.16 | 70.0 % | 30.0 % | 19.65 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 5 | hop 1.0 | 1.11 | 1.25 | 1.73 | 1.21 | 1.34 | 3.97 | 33.3 % | 13.0 % | 7.07 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 5 | hop 1.0 | 2.13 | 2.15 | 2.14 | 2.14 | 2.14 | 2.15 | 70.0 % | 30.0 % | 17.24 |
| 6 cards, 9/8/8/8/8/8 layers | copy >= 4, else ngram-cache | depth 5 | stages x1.1 | 1.02 | 1.15 | 1.59 | 1.11 | 1.23 | 3.69 | 33.3 % | 13.0 % | 9.68 |
| 6 cards, 9/8/8/8/8/8 layers | oracle p 0.7 | depth 5 | stages x1.1 | 1.96 | 1.98 | 1.97 | 1.98 | 1.97 | 1.99 | 70.0 % | 30.0 % | 23.60 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy, match >= 2 | depth 8 | 1.12 | 1.29 | 1.87 | 1.23 | 1.39 | 5.35 | 33.6 % | 16.4 % | 19.70 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy, match >= 3 | depth 8 | 1.07 | 1.22 | 1.74 | 1.16 | 1.31 | 5.28 | 28.6 % | 7.8 % | 9.64 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy, match >= 4 | depth 8 | 1.04 | 1.19 | 1.66 | 1.12 | 1.26 | 5.21 | 25.2 % | 4.6 % | 5.85 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy, match >= 6 | depth 8 | 1.01 | 1.14 | 1.55 | 1.08 | 1.21 | 5.06 | 21.2 % | 2.2 % | 2.86 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy, match >= 8 | depth 8 | 1.00 | 1.12 | 1.48 | 1.05 | 1.18 | 4.92 | 18.7 % | 1.4 % | 1.91 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy, match = 12 | depth 8 | 1.00 | 1.08 | 1.40 | 1.03 | 1.15 | 4.70 | 15.6 % | 0.9 % | 1.17 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | fixed key 4 (ngram-simple/map-k lookup) | depth 8 | 1.04 | 1.18 | 1.60 | 1.12 | 1.25 | 4.15 | 24.0 % | 5.8 % | 7.41 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | ngram-cache counts | depth 8 | 1.12 | 1.27 | 1.78 | 1.23 | 1.36 | 4.35 | 32.2 % | 13.3 % | 15.11 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 8 | 1.12 | 1.28 | 1.85 | 1.24 | 1.38 | 5.41 | 33.3 % | 13.0 % | 15.26 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.3 | depth 8 | 1.32 | 1.34 | 1.33 | 1.32 | 1.33 | 1.32 | 29.8 % | 70.1 % | 81.99 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.4 | depth 8 | 1.49 | 1.50 | 1.49 | 1.49 | 1.49 | 1.48 | 39.8 % | 60.1 % | 71.60 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.5 | depth 8 | 1.69 | 1.71 | 1.70 | 1.70 | 1.70 | 1.69 | 49.7 % | 50.2 % | 61.13 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.6 | depth 8 | 1.97 | 1.99 | 1.99 | 2.02 | 1.99 | 2.01 | 59.9 % | 40.0 % | 49.94 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 8 | 2.37 | 2.39 | 2.38 | 2.38 | 2.38 | 2.40 | 70.0 % | 30.0 % | 38.41 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.8 | depth 8 | 2.96 | 2.99 | 3.00 | 2.93 | 2.98 | 3.01 | 80.1 % | 19.8 % | 26.22 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.9 | depth 8 | 3.93 | 3.93 | 3.95 | 3.93 | 3.94 | 4.00 | 90.0 % | 10.0 % | 13.66 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.95 | depth 8 | 4.71 | 4.69 | 4.71 | 4.72 | 4.71 | 4.76 | 94.9 % | 5.0 % | 6.96 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else oracle p 0.5 | depth 8 | 1.70 | 1.91 | 2.44 | 1.78 | 1.99 | 5.40 | 60.1 % | 39.8 % | 48.59 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else oracle p 0.7 | depth 8 | 2.30 | 2.53 | 3.01 | 2.39 | 2.61 | 5.50 | 74.3 % | 25.6 % | 32.78 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 1 | 1.09 | 1.18 | 1.43 | 1.16 | 1.23 | 2.05 | 33.3 % | 13.0 % | 2.31 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 1 | 1.67 | 1.67 | 1.67 | 1.67 | 1.67 | 1.69 | 70.0 % | 30.0 % | 4.69 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 2 | 1.11 | 1.24 | 1.63 | 1.21 | 1.31 | 3.03 | 33.3 % | 13.0 % | 4.71 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 2 | 2.03 | 2.05 | 2.04 | 2.05 | 2.04 | 2.05 | 70.0 % | 30.0 % | 9.97 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 3 | 1.12 | 1.26 | 1.75 | 1.22 | 1.35 | 3.96 | 33.3 % | 13.0 % | 7.07 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 3 | 2.23 | 2.25 | 2.24 | 2.24 | 2.24 | 2.26 | 70.0 % | 30.0 % | 15.29 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 4 | 1.12 | 1.28 | 1.82 | 1.23 | 1.37 | 4.86 | 33.3 % | 13.0 % | 9.25 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 4 | 2.33 | 2.35 | 2.35 | 2.34 | 2.34 | 2.36 | 70.0 % | 30.0 % | 20.57 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 5 | 1.12 | 1.28 | 1.85 | 1.24 | 1.38 | 5.41 | 33.3 % | 13.0 % | 11.13 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 5 | 2.37 | 2.39 | 2.38 | 2.38 | 2.38 | 2.40 | 70.0 % | 30.0 % | 25.57 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 6 | 1.12 | 1.28 | 1.85 | 1.24 | 1.38 | 5.41 | 33.3 % | 13.0 % | 12.81 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 6 | 2.37 | 2.39 | 2.38 | 2.38 | 2.38 | 2.40 | 70.0 % | 30.0 % | 30.46 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else ngram-cache | depth 7 | 1.12 | 1.28 | 1.85 | 1.24 | 1.38 | 5.41 | 33.3 % | 13.0 % | 14.20 |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | oracle p 0.7 | depth 7 | 2.37 | 2.39 | 2.38 | 2.38 | 2.38 | 2.40 | 70.0 % | 30.0 % | 34.80 |
| 9 cards, head + 2 layers on the last | copy, match >= 2 | depth 8 | 1.13 | 1.31 | 1.97 | 1.25 | 1.42 | 7.19 | 33.6 % | 16.4 % | 19.21 |
| 9 cards, head + 2 layers on the last | copy, match >= 3 | depth 8 | 1.07 | 1.24 | 1.82 | 1.17 | 1.34 | 7.05 | 28.6 % | 7.8 % | 9.36 |
| 9 cards, head + 2 layers on the last | copy, match >= 4 | depth 8 | 1.04 | 1.20 | 1.72 | 1.13 | 1.28 | 6.93 | 25.2 % | 4.6 % | 5.64 |
| 9 cards, head + 2 layers on the last | copy, match >= 6 | depth 8 | 1.01 | 1.15 | 1.60 | 1.08 | 1.23 | 6.65 | 21.2 % | 2.2 % | 2.72 |
| 9 cards, head + 2 layers on the last | copy, match >= 8 | depth 8 | 1.01 | 1.12 | 1.53 | 1.06 | 1.20 | 6.39 | 18.7 % | 1.4 % | 1.80 |
| 9 cards, head + 2 layers on the last | copy, match = 12 | depth 8 | 1.00 | 1.09 | 1.43 | 1.03 | 1.16 | 6.00 | 15.6 % | 0.9 % | 1.10 |
| 9 cards, head + 2 layers on the last | fixed key 4 (ngram-simple/map-k lookup) | depth 8 | 1.04 | 1.19 | 1.66 | 1.13 | 1.27 | 5.09 | 24.0 % | 5.8 % | 7.10 |
| 9 cards, head + 2 layers on the last | ngram-cache counts | depth 8 | 1.13 | 1.30 | 1.87 | 1.25 | 1.39 | 5.42 | 32.2 % | 13.3 % | 14.73 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 8 | 1.13 | 1.31 | 1.95 | 1.25 | 1.41 | 7.30 | 33.3 % | 13.0 % | 14.88 |
| 9 cards, head + 2 layers on the last | oracle p 0.3 | depth 8 | 1.35 | 1.36 | 1.35 | 1.34 | 1.35 | 1.35 | 29.8 % | 70.1 % | 80.89 |
| 9 cards, head + 2 layers on the last | oracle p 0.4 | depth 8 | 1.53 | 1.55 | 1.54 | 1.54 | 1.54 | 1.52 | 39.8 % | 60.1 % | 70.58 |
| 9 cards, head + 2 layers on the last | oracle p 0.5 | depth 8 | 1.77 | 1.78 | 1.77 | 1.77 | 1.77 | 1.77 | 49.7 % | 50.2 % | 60.04 |
| 9 cards, head + 2 layers on the last | oracle p 0.6 | depth 8 | 2.09 | 2.12 | 2.11 | 2.15 | 2.11 | 2.14 | 59.9 % | 40.0 % | 48.68 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 8 | 2.57 | 2.61 | 2.59 | 2.59 | 2.59 | 2.61 | 70.0 % | 30.0 % | 37.06 |
| 9 cards, head + 2 layers on the last | oracle p 0.8 | depth 8 | 3.34 | 3.38 | 3.39 | 3.31 | 3.37 | 3.42 | 80.1 % | 19.8 % | 24.93 |
| 9 cards, head + 2 layers on the last | oracle p 0.9 | depth 8 | 4.75 | 4.74 | 4.78 | 4.75 | 4.76 | 4.85 | 90.0 % | 10.0 % | 12.76 |
| 9 cards, head + 2 layers on the last | oracle p 0.95 | depth 8 | 6.03 | 5.99 | 6.02 | 6.05 | 6.01 | 6.10 | 94.9 % | 5.0 % | 6.44 |
| 9 cards, head + 2 layers on the last | copy >= 4, else oracle p 0.5 | depth 8 | 1.77 | 2.02 | 2.67 | 1.87 | 2.12 | 7.28 | 60.1 % | 39.8 % | 47.61 |
| 9 cards, head + 2 layers on the last | copy >= 4, else oracle p 0.7 | depth 8 | 2.49 | 2.78 | 3.41 | 2.60 | 2.88 | 7.48 | 74.3 % | 25.6 % | 31.63 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 1 | 1.09 | 1.18 | 1.43 | 1.16 | 1.23 | 2.05 | 33.3 % | 13.0 % | 2.45 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 1 | 1.69 | 1.69 | 1.69 | 1.69 | 1.69 | 1.70 | 70.0 % | 30.0 % | 4.96 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 2 | 1.12 | 1.24 | 1.64 | 1.21 | 1.32 | 3.03 | 33.3 % | 13.0 % | 4.78 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 2 | 2.07 | 2.09 | 2.09 | 2.09 | 2.09 | 2.10 | 70.0 % | 30.0 % | 10.10 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 3 | 1.12 | 1.27 | 1.76 | 1.23 | 1.36 | 3.97 | 33.3 % | 13.0 % | 7.08 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 3 | 2.31 | 2.33 | 2.33 | 2.32 | 2.32 | 2.34 | 70.0 % | 30.0 % | 15.32 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 4 | 1.13 | 1.29 | 1.84 | 1.24 | 1.38 | 4.88 | 33.3 % | 13.0 % | 9.15 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 4 | 2.44 | 2.46 | 2.46 | 2.45 | 2.45 | 2.48 | 70.0 % | 30.0 % | 20.50 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 5 | 1.13 | 1.30 | 1.89 | 1.25 | 1.40 | 5.75 | 33.3 % | 13.0 % | 11.02 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 5 | 2.52 | 2.54 | 2.53 | 2.53 | 2.53 | 2.56 | 70.0 % | 30.0 % | 25.36 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 6 | 1.13 | 1.30 | 1.92 | 1.25 | 1.41 | 6.59 | 33.3 % | 13.0 % | 12.59 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 6 | 2.55 | 2.59 | 2.57 | 2.58 | 2.57 | 2.59 | 70.0 % | 30.0 % | 29.79 |
| 9 cards, head + 2 layers on the last | copy >= 4, else ngram-cache | depth 7 | 1.13 | 1.31 | 1.95 | 1.25 | 1.41 | 7.30 | 33.3 % | 13.0 % | 13.86 |
| 9 cards, head + 2 layers on the last | oracle p 0.7 | depth 7 | 2.57 | 2.61 | 2.59 | 2.59 | 2.59 | 2.61 | 70.0 % | 30.0 % | 33.66 |
