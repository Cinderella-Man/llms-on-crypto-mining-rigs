62 replayed turns in every pass of every load. pp1_2 and pp1_3 against embd1 and embd4: 241.49 s a pass against 240.11 (+1.39 s, +0.58 %), the reference's spread 0.21, the test's 3.51 → loss.

By new tokens:
- 0-511: 64.19 against 64.20 (-0.01 s, -0.02 %), spread 0.04 → negligible
- 512-1023: 56.38 against 56.15 (+0.23 s, +0.40 %), spread 0.06 → loss
- 1024-1999: 86.39 against 85.32 (+1.07 s, +1.25 %), spread 0.01 → loss
- 2000+: 34.54 against 34.44 (+0.10 s, +0.30 %), spread 0.10 → loss
