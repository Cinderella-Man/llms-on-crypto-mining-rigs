Base: A, taper off, 62 turns in every pass of every arm.

- A off: 509.16, 499.93 s per pass over 62 turns against the base's 509.16, 499.93 s: +0.00 s (+0.00 %), spread 9.23 s → base; greedy replies as the base's on 62 of 62 turns
- A on: 434.46, 430.50 s per pass over 62 turns against the base's 509.16, 499.93 s: -72.06 s (-14.28 %), spread 9.23 s → gain; greedy replies as the base's on 61 of 62 turns
- B off: 349.57 s per pass over 62 turns against the base's 509.16, 499.93 s: -154.98 s (-30.72 %), spread 9.23 s → gain; greedy replies as the base's on 61 of 62 turns
- B on: 318.08 s per pass over 62 turns against the base's 509.16, 499.93 s: -186.47 s (-36.96 %), spread 9.23 s → gain; greedy replies as the base's on 62 of 62 turns
- vram4 off: 437.54, 416.17 s per pass over 62 turns against the base's 509.16, 499.93 s: -77.69 s (-15.40 %), spread 21.37 s → gain; greedy replies as the base's on 62 of 62 turns
- vram4 on: 358.94, 349.97 s per pass over 62 turns against the base's 509.16, 499.93 s: -150.09 s (-29.75 %), spread 9.23 s → gain; greedy replies as the base's on 62 of 62 turns

taper_replay.py's summary, A:

```
off: 2 passes, 124 turns, 73586 prompt tokens in 1009.1 s (72.9 tok/s); per pass 504.5 s
on: 2 passes, 124 turns, 73586 prompt tokens in 865.0 s (85.1 tok/s); per pass 432.5 s
on against off, paired by task and turn: 62 turns, speed-up median 1.16x, 72.1 s saved per pass
  1-64 new tokens: 18 turns, off median 2.41 s, on median 2.40 s, sum 44.0 -> 43.7 s
  65-255 new tokens: 8 turns, off median 3.60 s, on median 3.11 s, sum 29.8 -> 25.5 s
  256-511 new tokens: 12 turns, off median 6.96 s, on median 5.01 s, sum 81.9 -> 60.4 s
  512-1023 new tokens: 10 turns, off median 10.75 s, on median 8.39 s, sum 110.7 -> 84.1 s
  1024-1999 new tokens: 11 turns, off median 15.83 s, on median 14.59 s, sum 171.6 -> 155.6 s
  2000-up new tokens: 3 turns, off median 19.09 s, on median 18.82 s, sum 66.5 -> 63.2 s
greedy output identical to off's on 61 of 62 turns of the other arms
fresh 2049-token prompts, off: median 16.04 s (16.27, 15.96, 16.11, 15.97)
fresh 2049-token prompts, on: median 16.81 s (16.71, 16.91, 16.61, 17.09)
```

taper_replay.py's summary, B:

```
off: 1 passes, 62 turns, 37826 prompt tokens in 349.6 s (108.2 tok/s); per pass 349.6 s
on: 1 passes, 62 turns, 37826 prompt tokens in 318.1 s (118.9 tok/s); per pass 318.1 s
on against off, paired by task and turn: 62 turns, speed-up median 1.08x, 31.5 s saved per pass
  1-64 new tokens: 18 turns, off median 1.04 s, on median 1.03 s, sum 20.3 -> 18.8 s
  65-255 new tokens: 8 turns, off median 2.60 s, on median 1.68 s, sum 21.1 -> 13.7 s
  256-511 new tokens: 12 turns, off median 5.61 s, on median 3.78 s, sum 66.0 -> 46.0 s
  512-1023 new tokens: 10 turns, off median 6.96 s, on median 5.88 s, sum 73.8 -> 59.2 s
  1024-1999 new tokens: 10 turns, off median 8.94 s, on median 11.33 s, sum 92.4 -> 108.7 s
  2000-up new tokens: 4 turns, off median 19.19 s, on median 16.97 s, sum 76.0 -> 71.7 s
greedy output identical to off's on 61 of 62 turns of the other arms
fresh 2049-token prompts, off: median 12.77 s (12.78, 12.98, 12.76, 12.42)
fresh 2049-token prompts, on: median 11.76 s (11.61, 11.91, 11.60, 11.96)
```

taper_replay.py's summary, vram4:

```
off: 2 passes, 124 turns, 73586 prompt tokens in 853.7 s (86.2 tok/s); per pass 426.9 s
on: 2 passes, 124 turns, 73586 prompt tokens in 708.9 s (103.8 tok/s); per pass 354.5 s
on against off, paired by task and turn: 62 turns, speed-up median 1.28x, 72.4 s saved per pass
  1-64 new tokens: 18 turns, off median 1.12 s, on median 1.13 s, sum 21.2 -> 20.5 s
  65-255 new tokens: 8 turns, off median 2.54 s, on median 1.87 s, sum 20.6 -> 15.4 s
  256-511 new tokens: 12 turns, off median 5.65 s, on median 3.76 s, sum 66.7 -> 45.3 s
  512-1023 new tokens: 10 turns, off median 9.60 s, on median 7.15 s, sum 98.6 -> 71.7 s
  1024-1999 new tokens: 11 turns, off median 14.54 s, on median 13.49 s, sum 157.7 -> 142.1 s
  2000-up new tokens: 3 turns, off median 17.77 s, on median 17.76 s, sum 62.0 -> 59.4 s
greedy output identical to off's on 62 of 62 turns of the other arms
fresh 2049-token prompts, off: median 14.84 s (14.98, 14.78, 14.90, 14.71)
fresh 2049-token prompts, on: median 15.61 s (15.51, 15.71, 15.40, 15.88)
```

