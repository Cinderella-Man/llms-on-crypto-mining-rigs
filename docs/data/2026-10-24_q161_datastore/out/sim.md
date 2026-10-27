# Q161: nine-card pipeline simulation of the store chains (old stores, 2026-10-24)

pipeline_sim.py on Q51 held out: speed-up over plain decoding at depths 1, 2, 3, 4 and 8, for both nine-card splits.

Every store named here was built from the first answer set of the staircase drafter (Q20), which was deleted on
licence grounds on 2026-10-25; the stores went with it and none was released. Store names: [README.md](../README.md).

| Cards | Guesser | Depth | Chat | Code | Edits | Agent | All | File rewrites | Tool calls | Right guesses | Wrong guesses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | cont (copy >= 4, else request counts) | depth 1 | 1.14 | 1.24 | 1.49 | 1.22 | 1.29 | 2.05 | 1.15 | 39.2 % | 40.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.17 | 1.27 | 1.51 | 1.28 | 1.32 | 2.06 | 1.45 | 40.9 % | 47.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.28 | 1.53 | 1.29 | 1.33 | 2.06 | 1.45 | 42.5 % | 46.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.27 | 1.51 | 1.29 | 1.32 | 2.06 | 1.48 | 41.1 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.29 | 1.53 | 1.31 | 1.33 | 2.06 | 1.48 | 42.9 % | 46.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.28 | 1.51 | 1.29 | 1.33 | 2.06 | 1.48 | 41.7 % | 48.7 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.30 | 1.53 | 1.31 | 1.34 | 2.06 | 1.46 | 43.5 % | 47.0 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.28 | 1.52 | 1.29 | 1.33 | 2.06 | 1.48 | 42.0 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.30 | 1.54 | 1.31 | 1.34 | 2.06 | 1.46 | 43.6 % | 46.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.28 | 1.51 | 1.24 | 1.32 | 2.06 | 1.19 | 41.4 % | 48.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.30 | 1.53 | 1.26 | 1.34 | 2.06 | 1.19 | 43.2 % | 47.2 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | cont (copy >= 4, else request counts) | depth 2 | 1.18 | 1.32 | 1.73 | 1.28 | 1.40 | 3.03 | 1.18 | 39.2 % | 40.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.21 | 1.35 | 1.75 | 1.37 | 1.43 | 3.05 | 1.68 | 40.9 % | 47.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.22 | 1.37 | 1.79 | 1.39 | 1.45 | 3.05 | 1.66 | 42.5 % | 46.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.21 | 1.36 | 1.75 | 1.38 | 1.43 | 3.05 | 1.72 | 41.1 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.38 | 1.79 | 1.40 | 1.45 | 3.05 | 1.68 | 42.9 % | 46.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.22 | 1.37 | 1.76 | 1.39 | 1.44 | 3.05 | 1.72 | 41.7 % | 48.7 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.80 | 1.41 | 1.46 | 3.06 | 1.67 | 43.5 % | 47.0 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.22 | 1.37 | 1.77 | 1.39 | 1.44 | 3.06 | 1.72 | 42.0 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.81 | 1.41 | 1.47 | 3.06 | 1.67 | 43.6 % | 46.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.22 | 1.37 | 1.76 | 1.30 | 1.43 | 3.06 | 1.23 | 41.4 % | 48.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.80 | 1.33 | 1.46 | 3.06 | 1.24 | 43.2 % | 47.2 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | cont (copy >= 4, else request counts) | depth 3 | 1.19 | 1.36 | 1.87 | 1.30 | 1.44 | 3.96 | 1.18 | 39.2 % | 40.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.22 | 1.39 | 1.89 | 1.41 | 1.48 | 3.99 | 1.75 | 40.9 % | 47.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.24 | 1.41 | 1.94 | 1.42 | 1.50 | 4.00 | 1.72 | 42.5 % | 46.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.22 | 1.39 | 1.89 | 1.42 | 1.48 | 4.00 | 1.81 | 41.1 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.24 | 1.42 | 1.94 | 1.44 | 1.51 | 4.01 | 1.76 | 42.9 % | 46.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.23 | 1.41 | 1.90 | 1.42 | 1.49 | 4.01 | 1.80 | 41.7 % | 48.7 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.25 | 1.43 | 1.95 | 1.45 | 1.52 | 4.02 | 1.74 | 43.5 % | 47.0 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.23 | 1.41 | 1.92 | 1.43 | 1.49 | 4.02 | 1.80 | 42.0 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.25 | 1.43 | 1.97 | 1.45 | 1.52 | 4.02 | 1.74 | 43.6 % | 46.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.23 | 1.41 | 1.89 | 1.32 | 1.49 | 4.02 | 1.24 | 41.4 % | 48.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.25 | 1.43 | 1.95 | 1.36 | 1.52 | 4.02 | 1.24 | 43.2 % | 47.2 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | cont (copy >= 4, else request counts) | depth 4 | 1.19 | 1.37 | 1.96 | 1.32 | 1.47 | 4.87 | 1.18 | 39.2 % | 40.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.22 | 1.41 | 1.97 | 1.42 | 1.50 | 4.92 | 1.80 | 40.9 % | 47.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.24 | 1.43 | 2.03 | 1.44 | 1.53 | 4.93 | 1.76 | 42.5 % | 46.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.23 | 1.41 | 1.97 | 1.43 | 1.51 | 4.92 | 1.87 | 41.1 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.44 | 2.04 | 1.46 | 1.54 | 4.94 | 1.81 | 42.9 % | 46.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.23 | 1.43 | 1.98 | 1.44 | 1.52 | 4.94 | 1.86 | 41.7 % | 48.7 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.45 | 2.05 | 1.46 | 1.55 | 4.97 | 1.78 | 43.5 % | 47.0 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.23 | 1.42 | 2.01 | 1.44 | 1.52 | 4.95 | 1.86 | 42.0 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.45 | 2.06 | 1.47 | 1.55 | 4.97 | 1.78 | 43.6 % | 46.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.23 | 1.43 | 1.98 | 1.33 | 1.51 | 4.96 | 1.24 | 41.4 % | 48.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.45 | 2.05 | 1.37 | 1.55 | 4.97 | 1.24 | 43.2 % | 47.2 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | cont (copy >= 4, else request counts) | depth 8 | 1.19 | 1.38 | 2.00 | 1.32 | 1.48 | 5.42 | 1.18 | 39.2 % | 40.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.22 | 1.41 | 2.01 | 1.43 | 1.51 | 5.48 | 1.81 | 40.9 % | 47.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.24 | 1.44 | 2.07 | 1.45 | 1.54 | 5.50 | 1.76 | 42.5 % | 46.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.23 | 1.42 | 2.01 | 1.44 | 1.52 | 5.49 | 1.88 | 41.1 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.25 | 1.45 | 2.08 | 1.47 | 1.55 | 5.51 | 1.81 | 42.9 % | 46.8 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.24 | 1.43 | 2.02 | 1.45 | 1.53 | 5.51 | 1.87 | 41.7 % | 48.7 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.26 | 1.46 | 2.09 | 1.47 | 1.56 | 5.55 | 1.79 | 43.5 % | 47.0 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.24 | 1.43 | 2.04 | 1.45 | 1.53 | 5.52 | 1.87 | 42.0 % | 48.6 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.26 | 1.46 | 2.11 | 1.47 | 1.57 | 5.55 | 1.79 | 43.6 % | 46.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.24 | 1.43 | 2.02 | 1.34 | 1.52 | 5.54 | 1.24 | 41.4 % | 48.9 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.26 | 1.46 | 2.09 | 1.38 | 1.56 | 5.55 | 1.25 | 43.2 % | 47.2 % |
| 9 cards, head + 2 layers on the last | cont (copy >= 4, else request counts) | depth 1 | 1.15 | 1.25 | 1.50 | 1.22 | 1.30 | 2.05 | 1.15 | 39.2 % | 40.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.28 | 1.52 | 1.29 | 1.32 | 2.06 | 1.47 | 40.9 % | 47.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.29 | 1.54 | 1.30 | 1.34 | 2.06 | 1.46 | 42.5 % | 46.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.18 | 1.28 | 1.52 | 1.30 | 1.33 | 2.06 | 1.49 | 41.1 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.29 | 1.54 | 1.32 | 1.34 | 2.06 | 1.49 | 42.9 % | 46.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.29 | 1.52 | 1.30 | 1.33 | 2.06 | 1.49 | 41.7 % | 48.7 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.20 | 1.30 | 1.54 | 1.32 | 1.35 | 2.06 | 1.48 | 43.5 % | 47.0 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.29 | 1.53 | 1.30 | 1.34 | 2.06 | 1.49 | 42.0 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.20 | 1.30 | 1.55 | 1.32 | 1.35 | 2.06 | 1.48 | 43.6 % | 46.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.19 | 1.29 | 1.52 | 1.25 | 1.33 | 2.06 | 1.20 | 41.4 % | 48.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 1 | 1.20 | 1.30 | 1.54 | 1.27 | 1.35 | 2.06 | 1.20 | 43.2 % | 47.2 % |
| 9 cards, head + 2 layers on the last | cont (copy >= 4, else request counts) | depth 2 | 1.19 | 1.33 | 1.75 | 1.29 | 1.41 | 3.03 | 1.19 | 39.2 % | 40.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.22 | 1.37 | 1.77 | 1.39 | 1.44 | 3.05 | 1.72 | 40.9 % | 47.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.81 | 1.41 | 1.46 | 3.06 | 1.69 | 42.5 % | 46.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.22 | 1.37 | 1.77 | 1.40 | 1.44 | 3.05 | 1.75 | 41.1 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.24 | 1.39 | 1.81 | 1.42 | 1.47 | 3.06 | 1.71 | 42.9 % | 46.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.78 | 1.40 | 1.45 | 3.06 | 1.76 | 41.7 % | 48.7 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.25 | 1.41 | 1.82 | 1.43 | 1.48 | 3.07 | 1.71 | 43.5 % | 47.0 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.79 | 1.41 | 1.46 | 3.06 | 1.76 | 42.0 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.25 | 1.41 | 1.83 | 1.43 | 1.48 | 3.07 | 1.71 | 43.6 % | 46.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.23 | 1.39 | 1.78 | 1.32 | 1.45 | 3.06 | 1.25 | 41.4 % | 48.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 2 | 1.25 | 1.41 | 1.82 | 1.35 | 1.47 | 3.07 | 1.25 | 43.2 % | 47.2 % |
| 9 cards, head + 2 layers on the last | cont (copy >= 4, else request counts) | depth 3 | 1.20 | 1.37 | 1.89 | 1.32 | 1.46 | 3.97 | 1.19 | 39.2 % | 40.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.23 | 1.40 | 1.91 | 1.42 | 1.49 | 4.00 | 1.79 | 40.9 % | 47.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.25 | 1.43 | 1.96 | 1.45 | 1.52 | 4.01 | 1.77 | 42.5 % | 46.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.24 | 1.41 | 1.91 | 1.44 | 1.50 | 4.01 | 1.85 | 41.1 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.26 | 1.44 | 1.97 | 1.46 | 1.53 | 4.02 | 1.81 | 42.9 % | 46.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.24 | 1.43 | 1.92 | 1.44 | 1.51 | 4.02 | 1.84 | 41.7 % | 48.7 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.26 | 1.45 | 1.98 | 1.47 | 1.54 | 4.03 | 1.79 | 43.5 % | 47.0 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.24 | 1.42 | 1.94 | 1.45 | 1.51 | 4.03 | 1.84 | 42.0 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.26 | 1.45 | 1.99 | 1.47 | 1.54 | 4.03 | 1.79 | 43.6 % | 46.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.24 | 1.43 | 1.92 | 1.34 | 1.50 | 4.03 | 1.25 | 41.4 % | 48.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 3 | 1.26 | 1.45 | 1.98 | 1.38 | 1.53 | 4.03 | 1.26 | 43.2 % | 47.2 % |
| 9 cards, head + 2 layers on the last | cont (copy >= 4, else request counts) | depth 4 | 1.21 | 1.39 | 1.99 | 1.33 | 1.49 | 4.89 | 1.20 | 39.2 % | 40.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.24 | 1.42 | 2.00 | 1.45 | 1.52 | 4.94 | 1.86 | 40.9 % | 47.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.26 | 1.45 | 2.06 | 1.47 | 1.55 | 4.96 | 1.83 | 42.5 % | 46.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.24 | 1.43 | 2.00 | 1.46 | 1.53 | 4.95 | 1.93 | 41.1 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.26 | 1.46 | 2.07 | 1.49 | 1.56 | 4.97 | 1.88 | 42.9 % | 46.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.45 | 2.01 | 1.46 | 1.54 | 4.96 | 1.91 | 41.7 % | 48.7 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.27 | 1.47 | 2.08 | 1.49 | 1.57 | 4.99 | 1.84 | 43.5 % | 47.0 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.44 | 2.04 | 1.47 | 1.54 | 4.97 | 1.91 | 42.0 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.27 | 1.47 | 2.10 | 1.49 | 1.58 | 4.99 | 1.84 | 43.6 % | 46.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.25 | 1.45 | 2.01 | 1.35 | 1.53 | 4.98 | 1.26 | 41.4 % | 48.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 4 | 1.27 | 1.47 | 2.08 | 1.39 | 1.57 | 4.99 | 1.26 | 43.2 % | 47.2 % |
| 9 cards, head + 2 layers on the last | cont (copy >= 4, else request counts) | depth 8 | 1.21 | 1.41 | 2.12 | 1.35 | 1.53 | 7.32 | 1.20 | 39.2 % | 40.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.24 | 1.45 | 2.13 | 1.46 | 1.56 | 7.43 | 1.90 | 40.9 % | 47.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.26 | 1.48 | 2.21 | 1.49 | 1.60 | 7.48 | 1.85 | 42.5 % | 46.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.25 | 1.45 | 2.13 | 1.48 | 1.57 | 7.46 | 1.99 | 41.1 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.27 | 1.48 | 2.22 | 1.51 | 1.60 | 7.50 | 1.90 | 42.9 % | 46.8 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.25 | 1.47 | 2.15 | 1.48 | 1.58 | 7.50 | 1.98 | 41.7 % | 48.7 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.27 | 1.50 | 2.23 | 1.51 | 1.62 | 7.57 | 1.87 | 43.5 % | 47.0 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.25 | 1.47 | 2.18 | 1.49 | 1.58 | 7.52 | 1.98 | 42.0 % | 48.6 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [all-14M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.27 | 1.50 | 2.25 | 1.51 | 1.62 | 7.57 | 1.87 | 43.6 % | 46.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.25 | 1.47 | 2.15 | 1.37 | 1.57 | 7.55 | 1.26 | 41.4 % | 48.9 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | depth 8 | 1.27 | 1.50 | 2.23 | 1.41 | 1.61 | 7.57 | 1.26 | 43.2 % | 47.2 % |
