**Load `hostdev`:** 11628 evaluations in 14 pieces (cut at gaps over 3000 ms), 10 after joining those with fewer than 10 main passes to the next; the first is the load's own, the others are its 9 requests.

| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0 | (the load) | 9 | 1 | 0 | 2 | | | |
| 1 | warmup 0 | 258 | 13 | 49 | 59 | no | 0 | 0 |
| 2 | decode 0 p1 | 1641 | 79 | 310 | 389 | yes | 77 | 74 |
| 3 | decode 0 p2 | 1566 | 75 | 297 | 372 | yes | 74 | 71 |
| 4 | prime 16384 | 615 | 31 | 113 | 162 | no | 0 | 0 |
| 5 | decode 16384 p1 | 1575 | 75 | 301 | 356 | no | 71 | 67 |
| 6 | decode 16384 p2 | 1494 | 72 | 282 | 354 | yes | 71 | 67 |
| 7 | prime 49152 | 1353 | 65 | 258 | 327 | no | 0 | 0 |
| 8 | decode 49152 p1 | 1581 | 76 | 300 | 369 | no | 75 | 72 |
| 9 | decode 49152 p2 | 1536 | 73 | 294 | 367 | yes | 72 | 69 |

**The k-pool inputs by request, pass and micro-batch size** (one line per evaluated micro-batch; the median, and in brackets the lowest and highest):

| request | pass | tokens | micro-batches | n_pool | host us | KiB a card |
|---|---|---:|---:|---|---|---|
| (load) | ? | 2 | 2 | 64 | 25.0 (8.0, 42.0) | 3.3 (3.3, 3.3) |
| warmup 0 | ? | 60 | 2 | 64 | 40.5 (22.0, 59.0) | 12.3 (12.3, 12.3) |
| warmup 0 | ? | 4 | 24 | 64 | 10.5 (9.0, 44.0) | 3.7 (3.7, 3.7) |
| warmup 0 | ? | 1 | 33 | 64 | 7.0 (6.0, 78.0) | 3.2 (3.2, 3.2) |
| decode 0 p1 | main | 60 | 1 | 64 | 29.0 (29.0, 29.0) | 12.3 (12.3, 12.3) |
| decode 0 p1 | draft | 60 | 1 | 64 | 21.0 (21.0, 21.0) | 12.3 (12.3, 12.3) |
| decode 0 p1 | main | 4 | 77 | 64 to 128 | 10.0 (8.0, 23.0) | 3.7 (3.7, 4.2) |
| decode 0 p1 | draft | 4 | 77 | 64 to 128 | 11.0 (8.0, 13.0) | 3.7 (3.7, 4.2) |
| decode 0 p1 | draft | 1 | 231 | 64 to 128 | 7.0 (5.0, 16.0) | 3.2 (3.2, 3.3) |
| decode 0 p1 | main | 2 | 1 | 128 | 12.0 (12.0, 12.0) | 3.6 (3.6, 3.6) |
| decode 0 p1 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.6 (3.6, 3.6) |
| decode 0 p2 | main | 4 | 74 | 64 to 128 | 9.0 (8.0, 23.0) | 3.7 (3.7, 4.2) |
| decode 0 p2 | draft | 4 | 74 | 64 to 128 | 10.0 (9.0, 15.0) | 3.7 (3.7, 4.2) |
| decode 0 p2 | draft | 1 | 222 | 64 to 128 | 7.0 (5.0, 17.0) | 3.2 (3.2, 3.3) |
| decode 0 p2 | main | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.6 (3.6, 3.6) |
| decode 0 p2 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.6 (3.6, 3.6) |
| prime 16384 | ? | 512 | 31 | 192 to 4096 | 803.0 (132.0, 2796.0) | 2133.0 (213.0, 4117.0) |
| prime 16384 | ? | 128 | 127 | 64 to 4096 | 139.0 (37.0, 271.0) | 534.8 (22.8, 1030.8) |
| prime 16384 | ? | 444 | 1 | 3968 | 909.0 (909.0, 909.0) | 3458.5 (3458.5, 3458.5) |
| prime 16384 | ? | 60 | 1 | 3968 | 133.0 (133.0, 133.0) | 469.8 (469.8, 469.8) |
| prime 16384 | ? | 4 | 2 | 4096 | 71.5 (49.0, 94.0) | 35.2 (35.2, 35.2) |
| decode 16384 p1 | ? | 60 | 2 | 4096 | 137.5 (119.0, 156.0) | 484.8 (484.8, 484.8) |
| decode 16384 p1 | ? | 4 | 142 | 4160 | 49.0 (46.0, 77.0) | 35.7 (35.7, 35.7) |
| decode 16384 p1 | ? | 1 | 212 | 4160 | 40.0 (33.0, 82.0) | 11.2 (11.2, 11.2) |
| decode 16384 p2 | main | 4 | 71 | 4160 | 47.0 (46.0, 83.0) | 35.7 (35.7, 35.7) |
| decode 16384 p2 | draft | 4 | 71 | 4160 | 49.0 (48.0, 69.0) | 35.7 (35.7, 35.7) |
| decode 16384 p2 | draft | 1 | 211 | 4160 | 40.0 (33.0, 184.0) | 11.2 (11.2, 11.2) |
| decode 16384 p2 | main | 1 | 1 | 4160 | 66.0 (66.0, 66.0) | 11.2 (11.2, 11.2) |
| prime 49152 | ? | 512 | 65 | 4096 to 12288 | 2451.0 (1266.0, 3610.0) | 8213.0 (4117.0, 12309.0) |
| prime 49152 | ? | 128 | 260 | 4032 to 12288 | 515.0 (206.0, 956.0) | 2046.8 (1014.8, 3078.8) |
| prime 49152 | ? | 4 | 2 | 12288 | 157.5 (144.0, 171.0) | 99.2 (99.2, 99.2) |
| decode 49152 p1 | ? | 60 | 2 | 12288 | 682.0 (447.0, 917.0) | 1444.8 (1444.8, 1444.8) |
| decode 49152 p1 | ? | 4 | 146 | 12352 | 127.0 (123.0, 831.0) | 99.7 (99.7, 99.7) |
| decode 49152 p1 | ? | 1 | 219 | 12352 | 110.0 (92.0, 260.0) | 27.2 (27.2, 27.2) |
| decode 49152 p1 | ? | 3 | 2 | 12352 | 151.0 (124.0, 178.0) | 75.5 (75.5, 75.5) |
| decode 49152 p2 | main | 4 | 73 | 12352 | 123.0 (121.0, 178.0) | 99.7 (99.7, 99.7) |
| decode 49152 p2 | draft | 4 | 73 | 12352 | 127.0 (125.0, 155.0) | 99.7 (99.7, 99.7) |
| decode 49152 p2 | draft | 1 | 219 | 12352 | 110.0 (93.0, 154.0) | 27.2 (27.2, 27.2) |
| decode 49152 p2 | draft | 3 | 2 | 12352 | 137.5 (124.0, 151.0) | 75.5 (75.5, 75.5) |

**The step by part, ms** (the mean over each decode's typical steps with 4 draft passes; every column is one request of the one load):

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| typical steps | 74 | 71 | 67 | 67 | 72 | 69 |
| **the step** | 127.50 | 123.44 | 151.53 | 149.36 | 220.68 | 176.10 |
| GPU time, every evaluation | 114.37 | 113.03 | 135.28 | 135.01 | 153.75 | 149.11 |
| **outside the kernels** (the step less that) | 13.13 | 10.41 | 16.25 | 14.34 | 66.93 | 26.99 |
| verify pass: GPU time of the nine cards | 101.64 | 101.68 | 119.88 | 119.80 | 130.72 | 130.43 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.29 | 4.25 | 6.15 | 6.10 | 16.67 | 10.54 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 4.76 | 4.71 | 5.22 | 5.22 | 9.90 | 8.13 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.54 | 0.53 | 0.70 | 0.70 | 2.57 | 1.05 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.65 | 0.63 | 0.79 | 0.79 | 2.83 | 1.18 |
| draft pass 0: GPU time, card 4 and card 8 | 3.64 | 3.67 | 5.14 | 5.14 | 7.86 | 6.08 |
| draft pass 0: turn | 0.36 | 0.32 | 0.39 | 0.37 | 6.24 | 2.72 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.52 | 0.52 | 0.56 | 0.56 | 3.78 | 0.65 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 0.64 | 0.64 | 5.53 | 0.73 |
| draft pass 1: GPU time, card 4 and card 8 | 3.03 | 2.57 | 3.46 | 3.40 | 4.36 | 4.22 |
| draft pass 1: turn | 0.55 | 0.55 | 0.66 | 0.66 | 1.13 | 0.90 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.52 | 0.52 | 0.56 | 0.56 | 0.66 | 0.65 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.59 | 0.58 | 0.63 | 0.63 | 0.81 | 0.72 |
| draft pass 2: GPU time, card 4 and card 8 | 3.10 | 2.55 | 3.41 | 3.33 | 4.31 | 4.17 |
| draft pass 2: turn | 0.54 | 0.52 | 0.66 | 0.66 | 1.02 | 0.87 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.52 | 0.51 | 0.56 | 0.56 | 0.70 | 0.65 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.63 | 0.63 | 3.02 | 0.75 |
| draft pass 3: GPU time, card 4 and card 8 | 2.96 | 2.57 | 3.40 | 3.34 | 6.50 | 4.21 |
| draft pass 3: turn (the server before the next verify pass) | 5.19 | 2.57 | 5.91 | 4.07 | 32.33 | 8.85 |
| the draft's crossings, all passes | 2.09 | 2.08 | 2.37 | 2.37 | 7.72 | 2.99 |
| the draft's turns before its last pass | 1.45 | 1.40 | 1.71 | 1.69 | 8.39 | 4.49 |
| k-pool inputs: host us a step, all passes (set_input) | 43.19 | 42.37 | nan | 222.79 | nan | 583.96 |
| k-pool inputs: host us, the verify pass | 9.92 | 9.42 | nan | 48.13 | nan | 123.81 |
| k-pool inputs: host us, the draft's passes | 33.27 | 32.94 | nan | 174.66 | nan | 460.14 |
| k-pool inputs: KiB a card, the verify pass | 3.81 | 3.82 | nan | 35.70 | nan | 99.70 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.22 | 3.22 | nan | 11.20 | nan | 27.20 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 47.82 | 47.87 | nan | 390.60 | nan | 1078.60 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 17.30 | 17.31 | nan | 105.00 | nan | 281.00 |
| k-pool inputs: n_pool of the verify pass | 78.70 | 79.32 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.34 | 0.34 | 0.22 | 0.21 | 0.43 | 0.21 |
| verify pass: card 1's host gap before entry | 0.30 | 0.30 | 0.48 | 0.48 | 0.88 | 0.83 |
| verify pass: card 2's host gap before entry | 0.32 | 0.28 | 0.45 | 0.45 | 0.88 | 0.84 |
| verify pass: card 3's host gap before entry | 0.28 | 0.27 | 0.44 | 0.44 | 0.92 | 0.79 |
| verify pass: card 4's host gap before entry | 0.24 | 0.24 | 0.40 | 0.40 | 0.82 | 0.82 |
| verify pass: card 5's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.84 | 0.79 |
| verify pass: card 6's host gap before entry | 0.27 | 0.26 | 0.44 | 0.44 | 0.83 | 0.79 |
| verify pass: card 7's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.83 | 0.79 |
| verify pass: card 8's host gap before entry | 0.32 | 0.32 | 0.50 | 0.50 | 0.89 | 0.84 |
| the server's step, this load | 125.94 | 121.97 | 165.60 | 146.97 | 247.25 | 174.15 |
| the server's step, load `warm` | 125.34 | 124.43 | 166.84 | 148.21 | 225.22 | 179.96 |

Medians of the same:

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| **the step** | 125.60 | 122.70 | 150.89 | 149.43 | 182.57 | 175.93 |
| GPU time, every evaluation | 112.97 | 112.78 | 135.31 | 135.00 | 149.57 | 149.20 |
| **outside the kernels** (the step less that) | 13.10 | 9.87 | 14.50 | 14.32 | 32.43 | 27.23 |
| verify pass: GPU time of the nine cards | 101.62 | 101.46 | 120.02 | 119.86 | 130.64 | 130.51 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.28 | 4.24 | 6.12 | 6.07 | 10.58 | 10.52 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 3.85 | 3.85 | 5.21 | 5.20 | 8.34 | 7.99 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.53 | 0.53 | 0.70 | 0.70 | 1.05 | 1.04 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.63 | 0.63 | 0.79 | 0.79 | 1.14 | 1.14 |
| draft pass 0: GPU time, card 4 and card 8 | 3.62 | 3.64 | 5.13 | 5.15 | 6.05 | 6.04 |
| draft pass 0: turn | 0.32 | 0.32 | 0.36 | 0.36 | 3.06 | 2.99 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.56 | 0.55 | 0.65 | 0.65 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 0.64 | 0.64 | 0.73 | 0.73 |
| draft pass 1: GPU time, card 4 and card 8 | 2.58 | 2.55 | 3.40 | 3.40 | 4.23 | 4.23 |
| draft pass 1: turn | 0.55 | 0.55 | 0.66 | 0.66 | 0.88 | 0.88 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.56 | 0.56 | 0.65 | 0.65 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.62 | 0.62 | 0.71 | 0.71 |
| draft pass 2: GPU time, card 4 and card 8 | 2.57 | 2.54 | 3.33 | 3.33 | 4.17 | 4.16 |
| draft pass 2: turn | 0.52 | 0.52 | 0.66 | 0.66 | 0.87 | 0.87 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.56 | 0.56 | 0.65 | 0.65 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.57 | 0.63 | 0.63 | 0.72 | 0.71 |
| draft pass 3: GPU time, card 4 and card 8 | 2.56 | 2.58 | 3.35 | 3.35 | 4.17 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 5.26 | 2.03 | 4.16 | 4.07 | 13.50 | 8.84 |
| the draft's crossings, all passes | 2.08 | 2.07 | 2.37 | 2.37 | 2.99 | 2.99 |
| the draft's turns before its last pass | 1.40 | 1.39 | 1.69 | 1.69 | 4.80 | 4.74 |
| k-pool inputs: host us a step, all passes (set_input) | 43.00 | 42.00 | nan | 219.00 | nan | 584.00 |
| k-pool inputs: host us, the verify pass | 10.00 | 9.00 | nan | 47.00 | nan | 123.00 |
| k-pool inputs: host us, the draft's passes | 33.00 | 33.00 | nan | 171.00 | nan | 460.00 |
| k-pool inputs: KiB a card, the verify pass | 3.70 | 3.70 | nan | 35.70 | nan | 99.70 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.20 | 3.20 | nan | 11.20 | nan | 27.20 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 46.60 | 46.60 | nan | 390.60 | nan | 1078.60 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 17.00 | 17.00 | nan | 105.00 | nan | 281.00 |
| k-pool inputs: n_pool of the verify pass | 64.00 | 64.00 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.24 | 0.24 | 0.22 | 0.21 | 0.22 | 0.21 |
| verify pass: card 1's host gap before entry | 0.30 | 0.29 | 0.48 | 0.48 | 0.83 | 0.83 |
| verify pass: card 2's host gap before entry | 0.27 | 0.27 | 0.45 | 0.44 | 0.80 | 0.80 |
| verify pass: card 3's host gap before entry | 0.27 | 0.27 | 0.44 | 0.44 | 0.80 | 0.79 |
| verify pass: card 4's host gap before entry | 0.23 | 0.23 | 0.40 | 0.40 | 0.75 | 0.73 |
| verify pass: card 5's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.80 | 0.79 |
| verify pass: card 6's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.80 | 0.79 |
| verify pass: card 7's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.79 | 0.79 |
| verify pass: card 8's host gap before entry | 0.31 | 0.31 | 0.49 | 0.49 | 0.85 | 0.84 |

**From 16384 to 49152 tokens, ms a step** (the deeper request less the shallower, pass against pass; the share is of the step's growth):

| part | p1: 16384 | p1: 49152 | p1: growth | p1: share | p2: 16384 | p2: 49152 | p2: growth | p2: share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **the step** | 151.53 | 220.68 | +69.14 | 100.0 % | 149.36 | 176.10 | +26.74 | 100.0 % |
| GPU time, every evaluation | 135.28 | 153.75 | +18.47 | 26.7 % | 135.01 | 149.11 | +14.10 | 52.7 % |
| **outside the kernels** (the step less that) | 16.25 | 66.93 | +50.68 | 73.3 % | 14.34 | 26.99 | +12.64 | 47.3 % |
| verify pass: GPU time of the nine cards | 119.88 | 130.72 | +10.85 | 15.7 % | 119.80 | 130.43 | +10.64 | 39.8 % |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.15 | 16.67 | +10.52 | 15.2 % | 6.10 | 10.54 | +4.44 | 16.6 % |
| verify pass: the host's launch, card 0's entry to card 8's exit | 5.22 | 9.90 | +4.68 | 6.8 % | 5.22 | 8.13 | +2.91 | 10.9 % |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.70 | 2.57 | +1.87 | 2.7 % | 0.70 | 1.05 | +0.35 | 1.3 % |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.79 | 2.83 | +2.04 | 3.0 % | 0.79 | 1.18 | +0.39 | 1.5 % |
| draft pass 0: GPU time, card 4 and card 8 | 5.14 | 7.86 | +2.72 | 3.9 % | 5.14 | 6.08 | +0.94 | 3.5 % |
| draft pass 0: turn | 0.39 | 6.24 | +5.85 | 8.5 % | 0.37 | 2.72 | +2.35 | 8.8 % |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.56 | 3.78 | +3.22 | 4.7 % | 0.56 | 0.65 | +0.09 | 0.3 % |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.64 | 5.53 | +4.88 | 7.1 % | 0.64 | 0.73 | +0.09 | 0.3 % |
| draft pass 1: GPU time, card 4 and card 8 | 3.46 | 4.36 | +0.90 | 1.3 % | 3.40 | 4.22 | +0.82 | 3.1 % |
| draft pass 1: turn | 0.66 | 1.13 | +0.47 | 0.7 % | 0.66 | 0.90 | +0.23 | 0.9 % |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.56 | 0.66 | +0.11 | 0.2 % | 0.56 | 0.65 | +0.09 | 0.3 % |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.63 | 0.81 | +0.18 | 0.3 % | 0.63 | 0.72 | +0.09 | 0.3 % |
| draft pass 2: GPU time, card 4 and card 8 | 3.41 | 4.31 | +0.91 | 1.3 % | 3.33 | 4.17 | +0.84 | 3.1 % |
| draft pass 2: turn | 0.66 | 1.02 | +0.36 | 0.5 % | 0.66 | 0.87 | +0.21 | 0.8 % |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.56 | 0.70 | +0.15 | 0.2 % | 0.56 | 0.65 | +0.09 | 0.3 % |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.63 | 3.02 | +2.39 | 3.5 % | 0.63 | 0.75 | +0.12 | 0.5 % |
| draft pass 3: GPU time, card 4 and card 8 | 3.40 | 6.50 | +3.10 | 4.5 % | 3.34 | 4.21 | +0.87 | 3.2 % |
| draft pass 3: turn (the server before the next verify pass) | 5.91 | 32.33 | +26.42 | 38.2 % | 4.07 | 8.85 | +4.78 | 17.9 % |
| the draft's crossings, all passes | 2.37 | 7.72 | +5.34 | 7.7 % | 2.37 | 2.99 | +0.62 | 2.3 % |
| the draft's turns before its last pass | 1.71 | 8.39 | +6.67 | 9.7 % | 1.69 | 4.49 | +2.80 | 10.5 % |
| k-pool inputs: host us a step, all passes (set_input) | nan | nan | +nan |  | 222.79 | 583.96 | +361.17 |  |
| k-pool inputs: host us, the verify pass | nan | nan | +nan |  | 48.13 | 123.81 | +75.68 |  |
| k-pool inputs: host us, the draft's passes | nan | nan | +nan |  | 174.66 | 460.14 | +285.49 |  |
| k-pool inputs: KiB a card, the verify pass | nan | nan | +nan |  | 35.70 | 99.70 | +64.00 |  |
| k-pool inputs: KiB, one draft pass (card 8) | nan | nan | +nan |  | 11.20 | 27.20 | +16.00 |  |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | nan | nan | +nan |  | 390.60 | 1078.60 | +688.00 |  |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | nan | nan | +nan |  | 105.00 | 281.00 | +176.00 |  |
| k-pool inputs: n_pool of the verify pass | nan | nan | +nan |  | 4160.00 | 12352.00 | +8192.00 |  |
| verify pass: card 0's host span | 0.22 | 0.43 | +0.21 | 0.3 % | 0.21 | 0.21 | +0.00 | 0.0 % |
| verify pass: card 1's host gap before entry | 0.48 | 0.88 | +0.40 | 0.6 % | 0.48 | 0.83 | +0.35 | 1.3 % |
| verify pass: card 2's host gap before entry | 0.45 | 0.88 | +0.43 | 0.6 % | 0.45 | 0.84 | +0.39 | 1.5 % |
| verify pass: card 3's host gap before entry | 0.44 | 0.92 | +0.47 | 0.7 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 4's host gap before entry | 0.40 | 0.82 | +0.41 | 0.6 % | 0.40 | 0.82 | +0.42 | 1.6 % |
| verify pass: card 5's host gap before entry | 0.44 | 0.84 | +0.40 | 0.6 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 6's host gap before entry | 0.44 | 0.83 | +0.39 | 0.6 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 7's host gap before entry | 0.44 | 0.83 | +0.39 | 0.6 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 8's host gap before entry | 0.50 | 0.89 | +0.39 | 0.6 % | 0.50 | 0.84 | +0.35 | 1.3 % |

