Base: v, taper off, 62 turns in every pass of every arm.

- v off: 416.56, 410.60 s per pass over 62 turns against the base's 416.56, 410.60 s: +0.00 s (+0.00 %), spread 5.96 s → base; greedy replies as the base's on 62 of 62 turns
- v on: 344.00, 343.60 s per pass over 62 turns against the base's 416.56, 410.60 s: -69.78 s (-16.87 %), spread 5.96 s → gain; greedy replies as the base's on 61 of 62 turns
- i off: 338.96, 331.09 s per pass over 62 turns against the base's 416.56, 410.60 s: -78.55 s (-18.99 %), spread 7.87 s → gain; greedy replies as the base's on 62 of 62 turns
- i on: 321.40, 321.09 s per pass over 62 turns against the base's 416.56, 410.60 s: -92.33 s (-22.32 %), spread 5.96 s → gain; greedy replies as the base's on 62 of 62 turns

taper_replay.py's summary, v:

```
off: 2 passes, 124 turns, 73586 prompt tokens in 827.2 s (89.0 tok/s); per pass 413.6 s
on: 2 passes, 124 turns, 73586 prompt tokens in 687.6 s (107.0 tok/s); per pass 343.8 s
on against off, paired by task and turn: 62 turns, speed-up median 1.26x, 69.8 s saved per pass
  1-64 new tokens: 18 turns, off median 1.11 s, on median 1.11 s, sum 21.0 -> 20.4 s
  65-255 new tokens: 8 turns, off median 2.51 s, on median 1.84 s, sum 20.3 -> 15.1 s
  256-511 new tokens: 12 turns, off median 5.56 s, on median 3.63 s, sum 65.1 -> 43.9 s
  512-1023 new tokens: 10 turns, off median 9.37 s, on median 6.98 s, sum 95.7 -> 69.8 s
  1024-1999 new tokens: 11 turns, off median 13.83 s, on median 13.04 s, sum 152.2 -> 137.3 s
  2000-up new tokens: 3 turns, off median 17.05 s, on median 17.02 s, sum 59.2 -> 57.2 s
greedy output identical to off's on 61 of 62 turns of the other arms
fresh 2049-token prompts, off: median 14.36 s (14.38, 14.51, 14.33, 14.33)
fresh 2049-token prompts, on: median 15.26 s (15.18, 15.19, 15.32, 15.44)
```

taper_replay.py's summary, i:

```
off: 2 passes, 124 turns, 73586 prompt tokens in 670.1 s (109.8 tok/s); per pass 335.0 s
on: 2 passes, 124 turns, 73586 prompt tokens in 642.5 s (114.5 tok/s); per pass 321.2 s
on against off, paired by task and turn: 62 turns, speed-up median 1.02x, 13.8 s saved per pass
  1-64 new tokens: 18 turns, off median 0.99 s, on median 0.99 s, sum 18.8 -> 18.2 s
  65-255 new tokens: 8 turns, off median 2.39 s, on median 1.72 s, sum 19.3 -> 14.2 s
  256-511 new tokens: 12 turns, off median 5.44 s, on median 3.53 s, sum 63.7 -> 43.9 s
  512-1023 new tokens: 10 turns, off median 6.70 s, on median 6.22 s, sum 69.3 -> 63.9 s
  1024-1999 new tokens: 11 turns, off median 8.71 s, on median 12.43 s, sum 112.1 -> 126.9 s
  2000-up new tokens: 3 turns, off median 17.06 s, on median 16.99 s, sum 51.8 -> 54.1 s
greedy output identical to off's on 62 of 62 turns of the other arms
fresh 2049-token prompts, off: median 12.33 s (12.27, 12.38, 12.41, 12.14)
fresh 2049-token prompts, on: median 12.23 s (12.07, 12.23, 12.30, 12.24)
```

