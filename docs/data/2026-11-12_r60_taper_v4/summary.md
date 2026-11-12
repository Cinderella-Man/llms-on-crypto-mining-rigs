Base: a, taper off, 62 turns in every pass of every arm.

- a off: 492.68, 480.70 s per pass over 62 turns against the base's 492.68, 480.70 s: +0.00 s (+0.00 %), spread 11.98 s → base; greedy replies as the base's on 62 of 62 turns
- a on: 486.04 s per pass over 62 turns against the base's 492.68, 480.70 s: -0.65 s (-0.13 %), spread 11.98 s → negligible; greedy replies as the base's on 62 of 62 turns
- a short=600:64: 469.17 s per pass over 62 turns against the base's 492.68, 480.70 s: -17.52 s (-3.60 %), spread 11.98 s → gain; greedy replies as the base's on 62 of 62 turns
- a tail=512:128,short=600:64: 473.82 s per pass over 62 turns against the base's 492.68, 480.70 s: -12.87 s (-2.64 %), spread 11.98 s → gain; greedy replies as the base's on 61 of 62 turns
- ro8 off: 483.93 s per pass over 62 turns against the base's 492.68, 480.70 s: -2.76 s (-0.57 %), spread 11.98 s → negligible; greedy replies as the base's on 62 of 62 turns
- ro8 on: 497.08 s per pass over 62 turns against the base's 492.68, 480.70 s: +10.39 s (+2.14 %), spread 11.98 s → negligible; greedy replies as the base's on 61 of 62 turns

taper_replay.py's summary, a:

```
off: 2 passes, 124 turns, 73586 prompt tokens in 973.4 s (75.6 tok/s); per pass 486.7 s
on: 1 passes, 62 turns, 36793 prompt tokens in 486.0 s (75.7 tok/s); per pass 486.0 s
short=600:64: 1 passes, 62 turns, 36793 prompt tokens in 469.2 s (78.4 tok/s); per pass 469.2 s
tail=512:128,short=600:64: 1 passes, 62 turns, 36793 prompt tokens in 473.8 s (77.7 tok/s); per pass 473.8 s
on against off, paired by task and turn: 62 turns, speed-up median 1.01x, 0.6 s saved per pass
  1-64 new tokens: 18 turns, off median 1.09 s, on median 1.09 s, sum 20.7 -> 20.0 s
  65-255 new tokens: 8 turns, off median 2.47 s, on median 1.79 s, sum 20.0 -> 15.9 s
  256-511 new tokens: 12 turns, off median 5.36 s, on median 4.57 s, sum 63.1 -> 53.2 s
  512-1023 new tokens: 10 turns, off median 9.01 s, on median 9.72 s, sum 93.3 -> 98.9 s
  1024-1999 new tokens: 11 turns, off median 18.43 s, on median 19.58 s, sum 193.0 -> 200.0 s
  2000-up new tokens: 3 turns, off median 29.03 s, on median 27.97 s, sum 96.6 -> 98.1 s
short=600:64 against off, paired by task and turn: 62 turns, speed-up median 1.01x, 17.5 s saved per pass
  1-64 new tokens: 18 turns, off median 1.09 s, short=600:64 median 1.09 s, sum 20.7 -> 20.0 s
  65-255 new tokens: 8 turns, off median 2.47 s, short=600:64 median 1.79 s, sum 20.0 -> 15.2 s
  256-511 new tokens: 12 turns, off median 5.36 s, short=600:64 median 4.57 s, sum 63.1 -> 53.2 s
  512-1023 new tokens: 10 turns, off median 9.01 s, short=600:64 median 9.27 s, sum 93.3 -> 93.4 s
  1024-1999 new tokens: 11 turns, off median 18.43 s, short=600:64 median 18.18 s, sum 193.0 -> 190.8 s
  2000-up new tokens: 3 turns, off median 29.03 s, short=600:64 median 29.16 s, sum 96.6 -> 96.7 s
tail=512:128,short=600:64 against off, paired by task and turn: 62 turns, speed-up median 1.01x, 12.9 s saved per pass
  1-64 new tokens: 18 turns, off median 1.09 s, tail=512:128,short=600:64 median 1.09 s, sum 20.7 -> 20.0 s
  65-255 new tokens: 8 turns, off median 2.47 s, tail=512:128,short=600:64 median 1.79 s, sum 20.0 -> 15.2 s
  256-511 new tokens: 12 turns, off median 5.36 s, tail=512:128,short=600:64 median 4.57 s, sum 63.1 -> 53.2 s
  512-1023 new tokens: 10 turns, off median 9.01 s, tail=512:128,short=600:64 median 9.05 s, sum 93.3 -> 94.3 s
  1024-1999 new tokens: 11 turns, off median 18.43 s, tail=512:128,short=600:64 median 19.52 s, sum 193.0 -> 194.7 s
  2000-up new tokens: 3 turns, off median 29.03 s, tail=512:128,short=600:64 median 28.43 s, sum 96.6 -> 96.5 s
greedy output identical to off's on 185 of 186 turns of the other arms
fresh 2049-token prompts, off: median 23.14 s (23.36, 22.94, 23.32, 22.97)
fresh 2049-token prompts, on: median 23.44 s (23.27, 23.80, 23.33, 23.55)
```

taper_replay.py's summary, ro8:

```
off: 1 passes, 62 turns, 36793 prompt tokens in 483.9 s (76.0 tok/s); per pass 483.9 s
on: 1 passes, 62 turns, 36793 prompt tokens in 497.1 s (74.0 tok/s); per pass 497.1 s
on against off, paired by task and turn: 62 turns, speed-up median 1.00x, -13.2 s saved per pass
  1-64 new tokens: 18 turns, off median 1.10 s, on median 1.13 s, sum 20.0 -> 21.7 s
  65-255 new tokens: 8 turns, off median 2.25 s, on median 1.82 s, sum 18.8 -> 16.0 s
  256-511 new tokens: 12 turns, off median 5.39 s, on median 4.78 s, sum 62.9 -> 56.5 s
  512-1023 new tokens: 10 turns, off median 8.96 s, on median 10.10 s, sum 92.6 -> 100.5 s
  1024-1999 new tokens: 11 turns, off median 18.35 s, on median 19.69 s, sum 193.0 -> 203.7 s
  2000-up new tokens: 3 turns, off median 29.36 s, on median 28.43 s, sum 96.6 -> 98.5 s
greedy output identical to off's on 61 of 62 turns of the other arms
fresh 2049-token prompts, off: median 23.15 s (23.34, 23.07, 23.22, 23.06)
fresh 2049-token prompts, on: median 23.50 s (23.33, 23.65, 23.51, 23.50)
```

