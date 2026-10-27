# Q161: chain orders and the oracle bound (old store clean-14M, 2026-10-24)

Q51 held out (38 transcripts), one token ahead on the true prefix; numbers are % of positions.

Every store named here was built from the first answer set of the staircase drafter (Q20), which was deleted on
licence grounds on 2026-10-25; the stores went with it and none was released. Store names: [README.md](../README.md).

| Guesser | Right: chat / code / edit / agent / all | Fresh right: chat / code / edit / agent / all | File rewrites right | Tool calls right | Guessed (all) | Datastore guesses: share / right |
|---|---|---|---|---|---|---|
| cont (copy >= 4, else request counts) | 19.6 / 33.3 / 60.2 / 29.4 / 39.2 | 17.1 / 17.8 / 23.3 / 19.9 / 19.0 | 98.4 | 20.3 | 79.6 % | – |
| copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | 23.0 / 36.4 / 60.9 / 37.4 / 41.8 | 20.9 / 22.7 / 27.5 / 30.3 / 23.6 | 98.7 | 57.8 | 90.5 % | 37.3 % / 28.9 % |
| copy >= 4, else request counts (L >= 3), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | 23.6 / 37.2 / 62.0 / 38.1 / 42.6 | 21.4 / 23.3 / 28.5 / 30.7 / 24.3 | 98.9 | 58.1 | 90.5 % | 33.5 % / 28.4 % |
| copy >= 4, else request counts (L >= 2), else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | 24.5 / 38.0 / 62.9 / 38.7 / 43.5 | 22.3 / 24.0 / 29.2 / 31.0 / 25.0 | 98.9 | 54.7 | 90.5 % | 26.0 % / 27.9 % |
| copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts (L >= 2), else datastore [clean-14M, longest], else request counts | 23.1 / 35.8 / 60.2 / 37.4 / 41.4 | 21.1 / 22.2 / 26.5 / 30.5 / 23.3 | 98.7 | 59.4 | 99.8 % | 37.3 % / 28.9 % |
| copy >= 4, else oracle choice of datastore [clean-14M, longest] and request counts (bound) | 30.3 / 43.4 / 66.5 / 44.9 / 48.4 | 28.1 / 30.5 / 35.9 / 37.8 / 31.3 | 98.9 | 63.7 | 99.8 % | – |
