62 replayed turns in every pass; the control is short=600:64, passes 0, 3, 6.

- short=2048:128 (passes 1, 5): 216.09, 213.98 s a pass against the control's 230.86, 220.09, 220.28 (-8.71 s, -3.89 %, spread 10.77) -> negligible (R60's rule: negligible); by new tokens: under 600 loss (+4.97 s, spread 3.52), 600-1023 gain (-10.13 s, spread 4.96), 1024-1999 gain (-4.24 s, spread 1.86), 2000+ loss (+0.69 s, spread 0.51)
- short=2048:64 (passes 2, 4): 213.78, 213.49 s a pass against the control's 230.86, 220.09, 220.28 (-10.11 s, -4.52 %, spread 10.77) -> negligible (R60's rule: negligible); by new tokens: under 600 negligible (-1.25 s, spread 3.52), 600-1023 gain (-11.19 s, spread 4.96), 1024-1999 negligible (+0.72 s, spread 1.86), 2000+ loss (+1.61 s, spread 0.51)
