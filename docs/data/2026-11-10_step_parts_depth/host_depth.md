**Load `host`:** 6774 evaluations in 11 pieces (cut at gaps over 3000 ms), 8 after joining those with fewer than 10 main passes to the next; the first is the load's own, the others are its 7 requests.

| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0 | (the load) | 9 | 1 | 0 | 0 | | | |
| 1 | warmup 0 | 211 | 13 | 47 | 0 | no | 0 | 0 |
| 2 | prime 16384 | 534 | 32 | 123 | 0 | no | 0 | 0 |
| 3 | decode 16384 p1 | 1214 | 72 | 283 | 0 | no | 69 | 66 |
| 4 | decode 16384 p2 | 1184 | 70 | 277 | 0 | no | 69 | 66 |
| 5 | prime 49152 | 1097 | 65 | 256 | 0 | no | 0 | 0 |
| 6 | decode 49152 p1 | 1282 | 76 | 299 | 0 | no | 75 | 72 |
| 7 | decode 49152 p2 | 1243 | 73 | 293 | 0 | no | 72 | 69 |


**The step by part, ms** (the mean over each decode's typical steps with 4 draft passes; every column is one request of the one load):

| part | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|
| typical steps | 66 | 66 | 72 | 69 |
| **the step** | 196.35 | 153.26 | 188.27 | 183.58 |
| GPU time, every evaluation | 135.32 | 135.00 | 150.32 | 149.64 |
| **outside the kernels** (the step less that) | 61.03 | 18.26 | 37.95 | 33.94 |
| verify pass: GPU time of the nine cards | 119.96 | 119.84 | 131.38 | 131.04 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.68 | 6.65 | 11.17 | 11.05 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 9.86 | 8.74 | 19.86 | 18.72 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 2.04 | 2.04 | 3.26 | 3.28 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 2.08 | 2.07 | 3.29 | 3.31 |
| draft pass 0: GPU time, card 4 and card 8 | 5.10 | 5.10 | 6.03 | 6.02 |
| draft pass 0: turn | 0.36 | 0.36 | 2.10 | 1.93 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 1.04 | 1.03 | 2.00 | 1.99 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 1.06 | 1.06 | 2.02 | 2.02 |
| draft pass 1: GPU time, card 4 and card 8 | 3.47 | 3.40 | 4.34 | 4.24 |
| draft pass 1: turn | 0.59 | 0.59 | 0.74 | 0.73 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 1.03 | 1.03 | 2.01 | 1.99 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 1.05 | 1.04 | 2.03 | 2.01 |
| draft pass 2: GPU time, card 4 and card 8 | 3.40 | 3.33 | 4.30 | 4.17 |
| draft pass 2: turn | 0.59 | 0.58 | 0.74 | 0.73 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 1.04 | 1.03 | 2.00 | 1.99 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 1.05 | 1.05 | 2.02 | 2.01 |
| draft pass 3: GPU time, card 4 and card 8 | 3.39 | 3.33 | 4.27 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 47.65 | 4.93 | 13.91 | 10.23 |
| the draft's crossings, all passes | 5.15 | 5.13 | 9.27 | 9.26 |
| the draft's turns before its last pass | 1.55 | 1.53 | 3.59 | 3.40 |
| verify pass: card 0's host span | 0.28 | 0.21 | 0.44 | 0.21 |
| verify pass: card 1's host gap before entry | 1.01 | 0.92 | 2.20 | 2.15 |
| verify pass: card 2's host gap before entry | 1.12 | 0.89 | 2.15 | 2.11 |
| verify pass: card 3's host gap before entry | 0.96 | 0.89 | 2.14 | 2.11 |
| verify pass: card 4's host gap before entry | 0.88 | 0.82 | 2.03 | 2.03 |
| verify pass: card 5's host gap before entry | 0.94 | 0.88 | 2.13 | 2.11 |
| verify pass: card 6's host gap before entry | 0.94 | 0.88 | 2.12 | 2.16 |
| verify pass: card 7's host gap before entry | 0.94 | 0.89 | 2.11 | 2.11 |
| verify pass: card 8's host gap before entry | 0.99 | 0.94 | 2.16 | 2.15 |
| the server's step, this load | 211.80 | 151.44 | 185.71 | 181.48 |
| the server's step, load `warm` | 203.68 | 150.33 | 184.41 | 180.87 |

Medians of the same:

| part | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|
| **the step** | 154.84 | 153.20 | 187.03 | 184.02 |
| GPU time, every evaluation | 135.35 | 135.11 | 149.80 | 149.82 |
| **outside the kernels** (the step less that) | 18.37 | 18.06 | 37.52 | 34.30 |
| verify pass: GPU time of the nine cards | 120.07 | 119.93 | 131.17 | 131.17 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.67 | 6.64 | 11.06 | 11.02 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 8.80 | 8.73 | 18.70 | 18.62 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 2.04 | 2.04 | 3.26 | 3.26 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 2.07 | 2.07 | 3.29 | 3.29 |
| draft pass 0: GPU time, card 4 and card 8 | 5.10 | 5.10 | 6.01 | 6.03 |
| draft pass 0: turn | 0.36 | 0.36 | 2.43 | 2.38 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 1.03 | 1.03 | 1.99 | 1.99 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 1.06 | 1.06 | 2.02 | 2.02 |
| draft pass 1: GPU time, card 4 and card 8 | 3.41 | 3.41 | 4.24 | 4.24 |
| draft pass 1: turn | 0.59 | 0.59 | 0.73 | 0.73 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 1.03 | 1.03 | 1.99 | 1.99 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 1.05 | 1.04 | 2.01 | 2.01 |
| draft pass 2: GPU time, card 4 and card 8 | 3.32 | 3.32 | 4.17 | 4.16 |
| draft pass 2: turn | 0.59 | 0.58 | 0.73 | 0.72 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 1.03 | 1.03 | 1.99 | 1.99 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 1.05 | 1.05 | 2.01 | 2.01 |
| draft pass 3: GPU time, card 4 and card 8 | 3.34 | 3.32 | 4.18 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 4.87 | 4.75 | 13.93 | 10.21 |
| the draft's crossings, all passes | 5.14 | 5.13 | 9.24 | 9.23 |
| the draft's turns before its last pass | 1.55 | 1.53 | 3.89 | 3.84 |
| verify pass: card 0's host span | 0.22 | 0.21 | 0.22 | 0.21 |
| verify pass: card 1's host gap before entry | 0.93 | 0.92 | 2.15 | 2.15 |
| verify pass: card 2's host gap before entry | 0.89 | 0.88 | 2.12 | 2.11 |
| verify pass: card 3's host gap before entry | 0.90 | 0.89 | 2.11 | 2.11 |
| verify pass: card 4's host gap before entry | 0.83 | 0.82 | 1.99 | 1.99 |
| verify pass: card 5's host gap before entry | 0.89 | 0.88 | 2.11 | 2.11 |
| verify pass: card 6's host gap before entry | 0.88 | 0.88 | 2.11 | 2.11 |
| verify pass: card 7's host gap before entry | 0.89 | 0.88 | 2.11 | 2.11 |
| verify pass: card 8's host gap before entry | 0.94 | 0.93 | 2.15 | 2.15 |

**From 16384 to 49152 tokens, ms a step** (the deeper request less the shallower, pass against pass; the share is of the step's growth):

| part | p1: 16384 | p1: 49152 | p1: growth | p1: share | p2: 16384 | p2: 49152 | p2: growth | p2: share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **the step** | 196.35 | 188.27 | -8.08 | 100.0 % | 153.26 | 183.58 | +30.33 | 100.0 % |
| GPU time, every evaluation | 135.32 | 150.32 | +15.01 | -185.7 % | 135.00 | 149.64 | +14.64 | 48.3 % |
| **outside the kernels** (the step less that) | 61.03 | 37.95 | -23.09 | 285.7 % | 18.26 | 33.94 | +15.69 | 51.7 % |
| verify pass: GPU time of the nine cards | 119.96 | 131.38 | +11.43 | -141.4 % | 119.84 | 131.04 | +11.20 | 36.9 % |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.68 | 11.17 | +4.49 | -55.6 % | 6.65 | 11.05 | +4.41 | 14.5 % |
| verify pass: the host's launch, card 0's entry to card 8's exit | 9.86 | 19.86 | +10.00 | -123.8 % | 8.74 | 18.72 | +9.98 | 32.9 % |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 2.04 | 3.26 | +1.22 | -15.1 % | 2.04 | 3.28 | +1.24 | 4.1 % |
| draft pass 0: lead (card 4's entry to card 8's entry) | 2.08 | 3.29 | +1.22 | -15.1 % | 2.07 | 3.31 | +1.24 | 4.1 % |
| draft pass 0: GPU time, card 4 and card 8 | 5.10 | 6.03 | +0.93 | -11.5 % | 5.10 | 6.02 | +0.92 | 3.0 % |
| draft pass 0: turn | 0.36 | 2.10 | +1.74 | -21.5 % | 0.36 | 1.93 | +1.57 | 5.2 % |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 1.04 | 2.00 | +0.96 | -11.9 % | 1.03 | 1.99 | +0.96 | 3.2 % |
| draft pass 1: lead (card 4's entry to card 8's entry) | 1.06 | 2.02 | +0.96 | -11.9 % | 1.06 | 2.02 | +0.96 | 3.2 % |
| draft pass 1: GPU time, card 4 and card 8 | 3.47 | 4.34 | +0.87 | -10.8 % | 3.40 | 4.24 | +0.83 | 2.8 % |
| draft pass 1: turn | 0.59 | 0.74 | +0.15 | -1.8 % | 0.59 | 0.73 | +0.14 | 0.5 % |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 1.03 | 2.01 | +0.98 | -12.1 % | 1.03 | 1.99 | +0.96 | 3.2 % |
| draft pass 2: lead (card 4's entry to card 8's entry) | 1.05 | 2.03 | +0.98 | -12.2 % | 1.04 | 2.01 | +0.97 | 3.2 % |
| draft pass 2: GPU time, card 4 and card 8 | 3.40 | 4.30 | +0.90 | -11.1 % | 3.33 | 4.17 | +0.84 | 2.8 % |
| draft pass 2: turn | 0.59 | 0.74 | +0.15 | -1.8 % | 0.58 | 0.73 | +0.15 | 0.5 % |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 1.04 | 2.00 | +0.96 | -11.9 % | 1.03 | 1.99 | +0.96 | 3.2 % |
| draft pass 3: lead (card 4's entry to card 8's entry) | 1.05 | 2.02 | +0.96 | -11.9 % | 1.05 | 2.01 | +0.96 | 3.2 % |
| draft pass 3: GPU time, card 4 and card 8 | 3.39 | 4.27 | +0.88 | -10.9 % | 3.33 | 4.17 | +0.84 | 2.8 % |
| draft pass 3: turn (the server before the next verify pass) | 47.65 | 13.91 | -33.74 | 417.5 % | 4.93 | 10.23 | +5.30 | 17.5 % |
| the draft's crossings, all passes | 5.15 | 9.27 | +4.12 | -51.0 % | 5.13 | 9.26 | +4.12 | 13.6 % |
| the draft's turns before its last pass | 1.55 | 3.59 | +2.04 | -25.2 % | 1.53 | 3.40 | +1.86 | 6.1 % |
| verify pass: card 0's host span | 0.28 | 0.44 | +0.16 | -2.0 % | 0.21 | 0.21 | +0.00 | 0.0 % |
| verify pass: card 1's host gap before entry | 1.01 | 2.20 | +1.19 | -14.7 % | 0.92 | 2.15 | +1.23 | 4.0 % |
| verify pass: card 2's host gap before entry | 1.12 | 2.15 | +1.02 | -12.6 % | 0.89 | 2.11 | +1.23 | 4.0 % |
| verify pass: card 3's host gap before entry | 0.96 | 2.14 | +1.19 | -14.7 % | 0.89 | 2.11 | +1.22 | 4.0 % |
| verify pass: card 4's host gap before entry | 0.88 | 2.03 | +1.15 | -14.2 % | 0.82 | 2.03 | +1.21 | 4.0 % |
| verify pass: card 5's host gap before entry | 0.94 | 2.13 | +1.19 | -14.7 % | 0.88 | 2.11 | +1.23 | 4.1 % |
| verify pass: card 6's host gap before entry | 0.94 | 2.12 | +1.18 | -14.6 % | 0.88 | 2.16 | +1.28 | 4.2 % |
| verify pass: card 7's host gap before entry | 0.94 | 2.11 | +1.17 | -14.5 % | 0.89 | 2.11 | +1.23 | 4.0 % |
| verify pass: card 8's host gap before entry | 0.99 | 2.16 | +1.17 | -14.5 % | 0.94 | 2.15 | +1.21 | 4.0 % |

