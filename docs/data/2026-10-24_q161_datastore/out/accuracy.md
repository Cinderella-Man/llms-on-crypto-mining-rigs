# Q161: store chains one token ahead (old stores, 2026-10-24)

Q51 held out (38 transcripts). Each row is a guesser chain over the true prefix; numbers are % of positions.

Every store named here was built from the first answer set of the staircase drafter (Q20), which was deleted on
licence grounds on 2026-10-25; the stores went with it and none was released. Store names: [README.md](../README.md).

| Guesser | Right: chat / code / edit / agent / all | Fresh right: chat / code / edit / agent / all | File rewrites right | Tool calls right | Guessed (all) | Datastore guesses: share / right |
|---|---|---|---|---|---|---|
| cont (copy >= 4, else request counts) | 19.6 / 33.3 / 60.2 / 29.4 / 39.2 | 17.1 / 17.8 / 23.3 / 19.9 / 19.0 | 98.4 | 20.3 | 79.6 % | – |
| copy >= 4, else datastore [clean-3M, longest], else request counts | 19.3 / 31.5 / 57.0 / 32.8 / 37.5 | 17.5 / 17.7 / 22.1 / 26.0 / 19.1 | 98.4 | 56.2 | 99.4 % | 68.9 % / 17.5 % |
| copy >= 4, else datastore [clean-3M, longest, L >= 2], else request counts | 20.9 / 34.3 / 59.7 / 35.5 / 40.0 | 18.9 / 20.3 / 25.6 / 28.2 / 21.5 | 98.7 | 57.5 | 91.9 % | 47.4 % / 21.2 % |
| copy >= 4, else datastore [clean-3M, longest, L >= 3], else request counts | 21.6 / 35.2 / 61.4 / 35.8 / 41.1 | 19.3 / 20.7 / 27.1 / 27.9 / 22.1 | 98.7 | 54.4 | 85.3 % | 22.2 % / 27.6 % |
| copy >= 4, else datastore [clean-3M, occurs >= 2], else request counts | 19.4 / 31.7 / 57.1 / 32.8 / 37.7 | 17.7 / 17.9 / 22.2 / 26.0 / 19.3 | 98.4 | 56.9 | 99.1 % | 68.3 % / 17.6 % |
| copy >= 4, else datastore [clean-3M, occurs >= 4], else request counts | 19.2 / 31.5 / 57.0 / 32.4 / 37.5 | 17.4 / 17.7 / 22.2 / 25.5 / 19.1 | 98.4 | 54.7 | 98.5 % | 67.2 % / 17.4 % |
| copy >= 4, else datastore [clean-3M, share >= 0.3], else request counts | 21.8 / 34.5 / 59.8 / 35.7 / 40.3 | 19.7 / 20.5 / 25.3 / 28.2 / 21.8 | 98.6 | 55.6 | 91.2 % | 37.7 % / 25.5 % |
| copy >= 4, else datastore [clean-3M, share >= 0.5], else request counts | 21.8 / 35.1 / 60.3 / 35.6 / 40.7 | 19.7 / 20.8 / 25.6 / 27.9 / 21.9 | 98.5 | 55.6 | 88.4 % | 28.3 % / 28.5 % |
| copy >= 4, else datastore [clean-3M, L >= 2, share >= 0.3], else request counts | 22.1 / 35.2 / 60.6 / 36.3 / 40.9 | 19.9 / 21.1 / 26.3 / 28.7 / 22.3 | 98.6 | 55.6 | 88.8 % | 32.4 % / 27.2 % |
| copy >= 4, else datastore [clean-7M, longest], else request counts | 20.0 / 32.1 / 57.4 / 34.1 / 38.1 | 18.3 / 18.4 / 22.8 / 27.5 / 19.9 | 98.4 | 59.1 | 99.7 % | 69.6 % / 18.4 % |
| copy >= 4, else datastore [clean-7M, longest, L >= 2], else request counts | 21.3 / 34.5 / 59.6 / 36.3 / 40.1 | 19.3 / 20.6 / 25.8 / 29.2 / 21.8 | 98.6 | 60.3 | 93.4 % | 52.1 % / 21.3 % |
| copy >= 4, else datastore [clean-7M, longest, L >= 3], else request counts | 21.8 / 35.6 / 61.5 / 36.6 / 41.4 | 19.6 / 21.2 / 27.6 / 28.8 / 22.6 | 98.6 | 58.1 | 86.5 % | 26.5 % / 27.2 % |
| copy >= 4, else datastore [clean-7M, occurs >= 2], else request counts | 20.1 / 32.1 / 57.3 / 33.7 / 38.1 | 18.4 / 18.5 / 22.8 / 27.1 / 19.9 | 98.4 | 57.5 | 99.5 % | 69.2 % / 18.4 % |
| copy >= 4, else datastore [clean-7M, occurs >= 4], else request counts | 20.0 / 32.1 / 57.4 / 33.7 / 38.1 | 18.2 / 18.3 / 23.0 / 27.0 / 19.9 | 98.5 | 57.5 | 99.2 % | 68.7 % / 18.3 % |
| copy >= 4, else datastore [clean-7M, share >= 0.3], else request counts | 22.2 / 35.0 / 60.0 / 36.5 / 40.7 | 20.2 / 21.0 / 25.9 / 29.4 / 22.4 | 98.7 | 58.8 | 91.5 % | 38.9 % / 26.4 % |
| copy >= 4, else datastore [clean-7M, share >= 0.5], else request counts | 22.2 / 35.5 / 60.7 / 36.7 / 41.2 | 20.0 / 21.4 / 26.7 / 29.0 / 22.5 | 98.7 | 57.5 | 88.6 % | 29.3 % / 29.8 % |
| copy >= 4, else datastore [clean-7M, L >= 2, share >= 0.3], else request counts | 22.4 / 35.5 / 60.5 / 37.0 / 41.2 | 20.3 / 21.4 / 26.7 / 29.8 / 22.7 | 98.7 | 58.4 | 89.8 % | 35.2 % / 27.4 % |
| copy >= 4, else datastore [clean-14M, longest], else request counts | 21.3 / 33.2 / 58.0 / 35.5 / 39.1 | 19.6 / 19.9 / 23.8 / 29.0 / 21.2 | 98.6 | 58.8 | 99.8 % | 69.8 % / 19.8 % |
| copy >= 4, else datastore [clean-14M, longest, L >= 2], else request counts | 22.1 / 35.4 / 59.8 / 36.9 / 40.8 | 20.2 / 21.9 / 26.2 / 30.1 / 22.8 | 98.7 | 59.1 | 94.6 % | 55.5 % / 22.4 % |
| copy >= 4, else datastore [clean-14M, longest, L >= 3], else request counts | 22.3 / 36.2 / 61.7 / 37.1 / 41.8 | 20.1 / 22.2 / 28.2 / 29.6 / 23.3 | 98.7 | 56.9 | 87.6 % | 30.6 % / 27.5 % |
| copy >= 4, else datastore [clean-14M, occurs >= 2], else request counts | 21.4 / 33.2 / 58.1 / 34.5 / 39.1 | 19.7 / 19.8 / 24.0 / 27.9 / 21.2 | 98.5 | 57.8 | 99.7 % | 69.6 % / 19.8 % |
| copy >= 4, else datastore [clean-14M, occurs >= 4], else request counts | 21.3 / 33.0 / 58.1 / 34.1 / 39.0 | 19.5 / 19.5 / 24.0 / 27.4 / 21.0 | 98.5 | 58.4 | 99.6 % | 69.3 % / 19.6 % |
| copy >= 4, else datastore [clean-14M, share >= 0.3], else request counts | 23.0 / 35.9 / 60.5 / 37.2 / 41.4 | 20.9 / 22.4 / 27.0 / 30.2 / 23.4 | 98.7 | 58.1 | 91.8 % | 40.2 % / 28.2 % |
| copy >= 4, else datastore [clean-14M, share >= 0.5], else request counts | 22.9 / 36.6 / 61.0 / 37.1 / 41.8 | 20.7 / 22.7 / 27.3 / 29.8 / 23.5 | 98.8 | 57.2 | 89.0 % | 30.5 % / 31.8 % |
| copy >= 4, else datastore [clean-14M, L >= 2, share >= 0.3], else request counts | 23.0 / 36.4 / 60.9 / 37.4 / 41.8 | 20.9 / 22.7 / 27.5 / 30.3 / 23.6 | 98.7 | 57.8 | 90.5 % | 37.3 % / 28.9 % |
| copy >= 4, else datastore [all-14M, longest], else request counts | 21.2 / 33.2 / 59.0 / 35.3 / 39.5 | 19.5 / 19.9 / 25.7 / 28.9 / 21.6 | 98.6 | 58.8 | 99.9 % | 69.8 % / 20.3 % |
| copy >= 4, else datastore [all-14M, longest, L >= 2], else request counts | 22.1 / 35.4 / 60.7 / 36.9 / 41.1 | 20.2 / 21.9 / 27.9 / 30.1 / 23.2 | 98.7 | 59.1 | 94.6 % | 55.8 % / 23.0 % |
| copy >= 4, else datastore [all-14M, longest, L >= 3], else request counts | 22.3 / 36.2 / 62.4 / 37.1 / 42.1 | 20.1 / 22.2 / 29.6 / 29.6 / 23.6 | 98.7 | 56.9 | 87.7 % | 31.3 % / 28.4 % |
| copy >= 4, else datastore [all-14M, occurs >= 2], else request counts | 21.4 / 33.2 / 59.0 / 34.2 / 39.5 | 19.6 / 19.9 / 25.7 / 27.6 / 21.6 | 98.6 | 57.8 | 99.8 % | 69.6 % / 20.3 % |
| copy >= 4, else datastore [all-14M, occurs >= 4], else request counts | 21.3 / 33.0 / 59.0 / 33.9 / 39.3 | 19.5 / 19.5 / 25.8 / 27.2 / 21.4 | 98.5 | 58.4 | 99.6 % | 69.3 % / 20.1 % |
| copy >= 4, else datastore [all-14M, share >= 0.3], else request counts | 22.9 / 35.9 / 61.3 / 37.4 / 41.7 | 20.8 / 22.4 / 28.5 / 30.4 / 23.7 | 98.8 | 58.4 | 91.8 % | 40.6 % / 28.9 % |
| copy >= 4, else datastore [all-14M, share >= 0.5], else request counts | 22.8 / 36.5 / 61.6 / 37.3 / 42.0 | 20.6 / 22.6 / 28.4 / 29.9 / 23.7 | 98.8 | 57.2 | 89.0 % | 30.9 % / 32.6 % |
| copy >= 4, else datastore [all-14M, L >= 2, share >= 0.3], else request counts | 23.0 / 36.4 / 61.6 / 37.6 / 42.0 | 20.8 / 22.7 / 28.9 / 30.5 / 23.9 | 98.8 | 57.8 | 90.6 % | 37.8 % / 29.6 % |
| copy >= 4, else datastore [noagent-13M, longest], else request counts | 21.4 / 33.2 / 57.9 / 26.8 / 38.7 | 19.6 / 19.9 / 23.5 / 19.1 / 20.6 | 98.6 | 19.1 | 99.8 % | 69.7 % / 19.2 % |
| copy >= 4, else datastore [noagent-13M, longest, L >= 2], else request counts | 22.2 / 35.5 / 59.8 / 29.4 / 40.5 | 20.3 / 21.9 / 26.2 / 21.5 / 22.3 | 98.7 | 25.3 | 94.4 % | 54.9 % / 21.8 % |
| copy >= 4, else datastore [noagent-13M, longest, L >= 3], else request counts | 22.4 / 36.3 / 61.8 / 30.8 / 41.6 | 20.2 / 22.2 / 28.2 / 22.2 / 22.9 | 98.7 | 23.4 | 87.4 % | 29.8 % / 26.8 % |
| copy >= 4, else datastore [noagent-13M, occurs >= 2], else request counts | 21.5 / 33.2 / 58.0 / 26.7 / 38.8 | 19.7 / 19.9 / 23.8 / 18.9 / 20.7 | 98.5 | 18.8 | 99.7 % | 69.5 % / 19.3 % |
| copy >= 4, else datastore [noagent-13M, occurs >= 4], else request counts | 21.3 / 32.9 / 58.0 / 26.7 / 38.6 | 19.5 / 19.5 / 23.9 / 18.9 / 20.5 | 98.5 | 17.5 | 99.6 % | 69.1 % / 19.1 % |
| copy >= 4, else datastore [noagent-13M, share >= 0.3], else request counts | 23.0 / 35.9 / 60.4 / 30.4 / 41.1 | 20.9 / 22.4 / 26.8 / 22.2 / 22.9 | 98.8 | 24.7 | 91.7 % | 39.9 % / 27.5 % |
| copy >= 4, else datastore [noagent-13M, share >= 0.5], else request counts | 22.9 / 36.7 / 61.0 / 30.6 / 41.5 | 20.7 / 22.7 / 27.4 / 22.0 / 23.1 | 98.8 | 24.4 | 88.8 % | 30.0 % / 31.0 % |
| copy >= 4, else datastore [noagent-13M, L >= 2, share >= 0.3], else request counts | 23.0 / 36.4 / 60.8 / 30.6 / 41.5 | 20.9 / 22.7 / 27.4 / 22.3 / 23.2 | 98.8 | 25.0 | 90.4 % | 36.8 % / 28.1 % |
