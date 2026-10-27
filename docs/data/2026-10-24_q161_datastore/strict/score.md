# Q161: the strict store against the old store (2026-10-25)

Q51 held out, the engine's chain, nine cards at depth 3.
Top table: one token ahead, % of positions. Bottom table: pipeline_sim.py speed-up over plain decoding.

Every store named here was built from the first answer set of the staircase drafter (Q20), which was deleted on
licence grounds on 2026-10-25; the stores went with it and none was released. Store names: [README.md](../README.md).

| Guesser | Right: chat / code / edit / agent / all | Fresh right: chat / code / edit / agent / all | File rewrites right | Tool calls right | Guessed (all) | Datastore guesses: share / right |
|---|---|---|---|---|---|---|
| cont (copy >= 4, else request counts) | 19.6 / 33.3 / 60.2 / 29.4 / 39.2 | 17.1 / 17.8 / 23.3 / 19.9 / 19.0 | 98.4 | 20.3 | 79.6 % | – |
| copy >= 4, else request counts (L >= 2), else datastore [old, L >= 2, share >= 0.3], else request counts | 24.5 / 38.0 / 62.9 / 38.7 / 43.5 | 22.3 / 24.0 / 29.2 / 31.0 / 25.0 | 98.9 | 54.7 | 90.5 % | 26.0 % / 27.9 % |
| copy >= 4, else request counts (L >= 2), else datastore [strict, L >= 2, share >= 0.3], else request counts | 21.0 / 34.9 / 63.1 / 39.0 / 41.6 | 18.5 / 20.0 / 29.6 / 31.3 / 22.4 | 98.9 | 58.8 | 88.1 % | 21.6 % / 22.8 % |

| Cards | Guesser | Depth | Chat | Code | Edits | Agent | All | File rewrites | Tool calls | Right guesses | Wrong guesses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | cont (copy >= 4, else request counts) | depth 3 | 1.19 | 1.36 | 1.87 | 1.30 | 1.44 | 3.96 | 1.18 | 39.2 % | 40.3 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [old, L >= 2, share >= 0.3], else request counts | depth 3 | 1.25 | 1.43 | 1.95 | 1.45 | 1.52 | 4.02 | 1.74 | 43.5 % | 47.0 % |
| 9 cards, 6/6/6/6/5/5/5/5/5 layers | copy >= 4, else request counts (L >= 2), else datastore [strict, L >= 2, share >= 0.3], else request counts | depth 3 | 1.21 | 1.38 | 1.96 | 1.45 | 1.49 | 4.02 | 1.84 | 41.6 % | 46.4 % |
| 9 cards, head + 2 layers on the last | cont (copy >= 4, else request counts) | depth 3 | 1.20 | 1.37 | 1.89 | 1.32 | 1.46 | 3.97 | 1.19 | 39.2 % | 40.3 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [old, L >= 2, share >= 0.3], else request counts | depth 3 | 1.26 | 1.45 | 1.98 | 1.47 | 1.54 | 4.03 | 1.79 | 43.5 % | 47.0 % |
| 9 cards, head + 2 layers on the last | copy >= 4, else request counts (L >= 2), else datastore [strict, L >= 2, share >= 0.3], else request counts | depth 3 | 1.22 | 1.40 | 1.99 | 1.47 | 1.50 | 4.03 | 1.90 | 41.6 % | 46.4 % |
