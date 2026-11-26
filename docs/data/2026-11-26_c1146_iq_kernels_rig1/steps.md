# The step pair: ms a step, every load

Arms: p0, u0, u1, u2; control `u0`. A step is one pass of the main model (its generated tokens minus the accepted draft tokens); ms a step = generation time / steps.

| Depth | Pass | Arm | Per load (ms a step, steps) | Mean | Spread | Against the control | % | Class |
|---:|---|---|---|---:|---:|---:|---:|---|
| 0 | 1 | u0 | u0: 115.22 (105); u0b: 115.27 (105) | 115.25 | 0.05 |  |  | control |
| 0 | 1 | u1 | u1: 105.74 (105) | 105.74 | — | -9.51 | -8.25 | no class (one load) |
| 0 | 1 | u2 | u2: 103.05 (105); u2b: 103.59 (105) | 103.32 | 0.55 | -11.93 | -10.35 | gain |
| 0 | 1 | p0 | p0: 102.16 (89) | 102.16 | — | -13.09 | -11.36 | no class (one load) |
| 0 | 2 | u0 | u0: 111.31 (104); u0b: 111.37 (104) | 111.34 | 0.05 |  |  | control |
| 0 | 2 | u1 | u1: 101.91 (104) | 101.91 | — | -9.43 | -8.47 | no class (one load) |
| 0 | 2 | u2 | u2: 99.61 (104); u2b: 99.75 (104) | 99.68 | 0.14 | -11.66 | -10.47 | gain |
| 0 | 2 | p0 | p0: 95.00 (88) | 95.00 | — | -16.34 | -14.67 | no class (one load) |
| 0 | s | u0 | u0: 115.90 (119); u0b: 123.06 (119) | 119.48 | 7.16 |  |  | control |
| 0 | s | u1 | u1: 106.40 (119) | 106.40 | — | -13.09 | -10.95 | no class (one load) |
| 0 | s | u2 | u2: 103.88 (119); u2b: 104.52 (119) | 104.20 | 0.64 | -15.29 | -12.79 | gain |
| 0 | s | p0 | p0: 95.80 (85) | 95.80 | — | -23.68 | -19.82 | no class (one load) |
| 8,192 | 1 | u0 | u0: 144.80 (97); u0b: 130.42 (97) | 137.61 | 14.38 |  |  | control |
| 8,192 | 1 | u1 | u1: 121.13 (97) | 121.13 | — | -16.49 | -11.98 | no class (one load) |
| 8,192 | 1 | u2 | u2: 118.93 (97); u2b: 127.69 (97) | 123.31 | 8.76 | -14.30 | -10.39 | negligible |
| 8,192 | 1 | p0 | p0: 113.32 (87) | 113.32 | — | -24.29 | -17.65 | no class (one load) |
| 8,192 | 2 | u0 | u0: 125.87 (97); u0b: 125.95 (97) | 125.91 | 0.08 |  |  | control |
| 8,192 | 2 | u1 | u1: 116.12 (97) | 116.12 | — | -9.79 | -7.77 | no class (one load) |
| 8,192 | 2 | u2 | u2: 114.01 (97); u2b: 114.25 (97) | 114.13 | 0.25 | -11.78 | -9.36 | gain |
| 8,192 | 2 | p0 | p0: 109.01 (84) | 109.01 | — | -16.90 | -13.42 | no class (one load) |
| 8,192 | s | u0 | u0: 129.29 (86); u0b: 129.71 (86) | 129.50 | 0.41 |  |  | control |
| 8,192 | s | u1 | u1: 120.11 (86) | 120.11 | — | -9.39 | -7.25 | no class (one load) |
| 8,192 | s | u2 | u2: 118.32 (86); u2b: 128.49 (86) | 123.40 | 10.17 | -6.10 | -4.71 | negligible |
| 8,192 | s | p0 | p0: 111.85 (88) | 111.85 | — | -17.65 | -13.63 | no class (one load) |

## Replies

- the main file's arms, depth 0, pass 1: the same reply on every load
- the main file's arms, depth 0, pass 2: the same reply on every load
- the main file's arms, depth 0, pass s: the same reply on every load
- the main file's arms, depth 8,192, pass 1: the same reply on every load
- the main file's arms, depth 8,192, pass 2: the same reply on every load
- the main file's arms, depth 8,192, pass s: the same reply on every load
- `p0`, depth 0, pass 1: the same reply on every load
- `p0`, depth 0, pass 2: the same reply on every load
- `p0`, depth 0, pass s: the same reply on every load
- `p0`, depth 8,192, pass 1: the same reply on every load
- `p0`, depth 8,192, pass 2: the same reply on every load
- `p0`, depth 8,192, pass s: the same reply on every load
