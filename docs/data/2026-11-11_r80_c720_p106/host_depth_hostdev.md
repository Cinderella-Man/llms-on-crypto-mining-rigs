**Load `hostdev`:** 11628 evaluations in 14 pieces (cut at gaps over 3000 ms), 10 after joining those with fewer than 10 main passes to the next; the first is the load's own, the others are its 9 requests.

| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0 | (the load) | 9 | 1 | 0 | 2 | | | |
| 1 | warmup 0 | 258 | 13 | 49 | 59 | no | 0 | 0 |
| 2 | decode 0 p1 | 1641 | 79 | 310 | 389 | yes | 77 | 74 |
| 3 | decode 0 p2 | 1566 | 75 | 297 | 372 | yes | 74 | 71 |
| 4 | prime 16384 | 615 | 31 | 115 | 162 | no | 0 | 0 |
| 5 | decode 16384 p1 | 1575 | 75 | 302 | 356 | no | 71 | 67 |
| 6 | decode 16384 p2 | 1494 | 72 | 282 | 354 | yes | 71 | 67 |
| 7 | prime 49152 | 1353 | 65 | 259 | 327 | no | 0 | 0 |
| 8 | decode 49152 p1 | 1581 | 76 | 300 | 369 | no | 75 | 72 |
| 9 | decode 49152 p2 | 1536 | 73 | 294 | 367 | yes | 72 | 69 |

**The k-pool inputs by request, pass and micro-batch size** (one line per evaluated micro-batch; the median, and in brackets the lowest and highest):

| request | pass | tokens | micro-batches | n_pool | host us | KiB a card |
|---|---|---:|---:|---|---|---|
| (load) | ? | 2 | 2 | 64 | 20.0 (9.0, 31.0) | 3.3 (3.3, 3.3) |
| warmup 0 | ? | 60 | 2 | 64 | 32.5 (21.0, 44.0) | 12.3 (12.3, 12.3) |
| warmup 0 | ? | 4 | 24 | 64 | 9.0 (8.0, 11.0) | 3.7 (3.7, 3.7) |
| warmup 0 | ? | 1 | 33 | 64 | 7.0 (5.0, 22.0) | 3.2 (3.2, 3.2) |
| decode 0 p1 | main | 60 | 1 | 64 | 55.0 (55.0, 55.0) | 12.3 (12.3, 12.3) |
| decode 0 p1 | draft | 60 | 1 | 64 | 21.0 (21.0, 21.0) | 12.3 (12.3, 12.3) |
| decode 0 p1 | main | 4 | 77 | 64 to 128 | 9.0 (7.0, 14.0) | 3.7 (3.7, 4.2) |
| decode 0 p1 | draft | 4 | 77 | 64 to 128 | 10.0 (9.0, 34.0) | 3.7 (3.7, 4.2) |
| decode 0 p1 | draft | 1 | 231 | 64 to 128 | 7.0 (5.0, 15.0) | 3.2 (3.2, 3.3) |
| decode 0 p1 | main | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.6 (3.6, 3.6) |
| decode 0 p1 | draft | 2 | 1 | 128 | 11.0 (11.0, 11.0) | 3.6 (3.6, 3.6) |
| decode 0 p2 | main | 4 | 74 | 64 to 128 | 9.0 (7.0, 32.0) | 3.7 (3.7, 4.2) |
| decode 0 p2 | draft | 4 | 74 | 64 to 128 | 10.0 (8.0, 38.0) | 3.7 (3.7, 4.2) |
| decode 0 p2 | draft | 1 | 222 | 64 to 128 | 7.0 (5.0, 21.0) | 3.2 (3.2, 3.3) |
| decode 0 p2 | main | 2 | 1 | 128 | 11.0 (11.0, 11.0) | 3.6 (3.6, 3.6) |
| decode 0 p2 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.6 (3.6, 3.6) |
| prime 16384 | ? | 512 | 31 | 192 to 4096 | 781.0 (147.0, 1389.0) | 2133.0 (213.0, 4117.0) |
| prime 16384 | ? | 128 | 127 | 64 to 4096 | 137.0 (35.0, 354.0) | 534.8 (22.8, 1030.8) |
| prime 16384 | ? | 444 | 1 | 3968 | 1323.0 (1323.0, 1323.0) | 3458.5 (3458.5, 3458.5) |
| prime 16384 | ? | 60 | 1 | 3968 | 138.0 (138.0, 138.0) | 469.8 (469.8, 469.8) |
| prime 16384 | ? | 4 | 2 | 4096 | 71.0 (48.0, 94.0) | 35.2 (35.2, 35.2) |
| decode 16384 p1 | ? | 60 | 2 | 4096 | 268.0 (180.0, 356.0) | 484.8 (484.8, 484.8) |
| decode 16384 p1 | ? | 4 | 142 | 4160 | 49.0 (46.0, 390.0) | 35.7 (35.7, 35.7) |
| decode 16384 p1 | ? | 1 | 212 | 4160 | 40.0 (33.0, 73.0) | 11.2 (11.2, 11.2) |
| decode 16384 p2 | main | 4 | 71 | 4160 | 47.0 (46.0, 84.0) | 35.7 (35.7, 35.7) |
| decode 16384 p2 | draft | 4 | 71 | 4160 | 49.0 (48.0, 73.0) | 35.7 (35.7, 35.7) |
| decode 16384 p2 | draft | 1 | 211 | 4160 | 40.0 (32.0, 62.0) | 11.2 (11.2, 11.2) |
| decode 16384 p2 | main | 1 | 1 | 4160 | 68.0 (68.0, 68.0) | 11.2 (11.2, 11.2) |
| prime 49152 | ? | 512 | 65 | 4096 to 12288 | 2618.0 (1401.0, 8253.0) | 8213.0 (4117.0, 12309.0) |
| prime 49152 | ? | 128 | 260 | 4032 to 12288 | 514.5 (218.0, 1018.0) | 2046.8 (1014.8, 3078.8) |
| prime 49152 | ? | 4 | 2 | 12288 | 175.5 (127.0, 224.0) | 99.2 (99.2, 99.2) |
| decode 49152 p1 | ? | 60 | 2 | 12288 | 354.5 (341.0, 368.0) | 1444.8 (1444.8, 1444.8) |
| decode 49152 p1 | ? | 4 | 146 | 12352 | 127.0 (121.0, 193.0) | 99.7 (99.7, 99.7) |
| decode 49152 p1 | ? | 1 | 219 | 12352 | 110.0 (92.0, 186.0) | 27.2 (27.2, 27.2) |
| decode 49152 p1 | ? | 3 | 2 | 12352 | 262.0 (124.0, 400.0) | 75.5 (75.5, 75.5) |
| decode 49152 p2 | main | 4 | 73 | 12352 | 123.0 (121.0, 167.0) | 99.7 (99.7, 99.7) |
| decode 49152 p2 | draft | 4 | 73 | 12352 | 127.0 (126.0, 146.0) | 99.7 (99.7, 99.7) |
| decode 49152 p2 | draft | 1 | 219 | 12352 | 110.0 (92.0, 152.0) | 27.2 (27.2, 27.2) |
| decode 49152 p2 | draft | 3 | 2 | 12352 | 135.0 (123.0, 147.0) | 75.5 (75.5, 75.5) |

**The step by part, ms** (the mean over each decode's typical steps with 4 draft passes; every column is one request of the one load):

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| typical steps | 74 | 71 | 67 | 67 | 72 | 69 |
| **the step** | 127.86 | 123.64 | 154.32 | 149.59 | 191.37 | 176.16 |
| GPU time, every evaluation | 114.43 | 113.07 | 135.33 | 135.11 | 149.88 | 149.33 |
| **outside the kernels** (the step less that) | 13.43 | 10.56 | 18.99 | 14.48 | 41.49 | 26.83 |
| verify pass: GPU time of the nine cards | 101.71 | 101.72 | 119.93 | 119.89 | 130.93 | 130.74 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.35 | 4.26 | 6.14 | 6.17 | 10.58 | 10.55 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 4.67 | 4.70 | 5.32 | 5.21 | 9.26 | 7.99 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.53 | 0.53 | 0.70 | 0.70 | 1.04 | 1.05 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.64 | 0.63 | 0.79 | 0.79 | 1.14 | 1.14 |
| draft pass 0: GPU time, card 4 and card 8 | 3.65 | 3.66 | 5.14 | 5.14 | 6.06 | 6.04 |
| draft pass 0: turn | 0.37 | 0.34 | 0.38 | 0.45 | 2.72 | 2.57 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.52 | 0.52 | 0.56 | 0.56 | 0.65 | 0.65 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 0.64 | 0.64 | 0.74 | 0.73 |
| draft pass 1: GPU time, card 4 and card 8 | 3.04 | 2.56 | 3.46 | 3.41 | 4.33 | 4.22 |
| draft pass 1: turn | 0.55 | 0.55 | 0.66 | 0.65 | 0.88 | 0.87 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.52 | 0.51 | 0.56 | 0.56 | 0.65 | 0.65 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.59 | 0.58 | 0.62 | 0.62 | 0.72 | 0.72 |
| draft pass 2: GPU time, card 4 and card 8 | 3.09 | 2.56 | 3.40 | 3.34 | 4.29 | 4.17 |
| draft pass 2: turn | 0.54 | 0.52 | 0.65 | 0.65 | 0.86 | 0.86 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.52 | 0.51 | 0.56 | 0.56 | 0.65 | 0.65 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.63 | 0.63 | 0.72 | 0.72 |
| draft pass 3: GPU time, card 4 and card 8 | 2.95 | 2.57 | 3.39 | 3.35 | 4.27 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 5.42 | 2.70 | 8.67 | 4.07 | 23.34 | 8.86 |
| the draft's crossings, all passes | 2.09 | 2.08 | 2.37 | 2.37 | 2.99 | 2.99 |
| the draft's turns before its last pass | 1.46 | 1.42 | 1.70 | 1.75 | 4.46 | 4.31 |
| k-pool inputs: host us a step, all passes (set_input) | 42.11 | 41.28 | nan | 219.54 | nan | 584.74 |
| k-pool inputs: host us, the verify pass | 9.23 | 8.94 | nan | 47.78 | nan | 124.71 |
| k-pool inputs: host us, the draft's passes | 32.88 | 32.34 | nan | 171.76 | nan | 460.03 |
| k-pool inputs: KiB a card, the verify pass | 3.81 | 3.82 | nan | 35.70 | nan | 99.70 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.22 | 3.22 | nan | 11.20 | nan | 27.20 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 47.82 | 47.87 | nan | 390.60 | nan | 1078.60 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 17.30 | 17.31 | nan | 105.00 | nan | 281.00 |
| k-pool inputs: n_pool of the verify pass | 78.70 | 79.32 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.33 | 0.33 | 0.23 | 0.22 | 0.33 | 0.21 |
| verify pass: card 1's host gap before entry | 0.30 | 0.30 | 0.49 | 0.48 | 0.84 | 0.83 |
| verify pass: card 2's host gap before entry | 0.28 | 0.27 | 0.45 | 0.44 | 0.81 | 0.79 |
| verify pass: card 3's host gap before entry | 0.28 | 0.28 | 0.45 | 0.44 | 0.82 | 0.80 |
| verify pass: card 4's host gap before entry | 0.23 | 0.24 | 0.41 | 0.40 | 0.74 | 0.73 |
| verify pass: card 5's host gap before entry | 0.27 | 0.27 | 0.45 | 0.44 | 0.80 | 0.79 |
| verify pass: card 6's host gap before entry | 0.27 | 0.26 | 0.45 | 0.44 | 0.81 | 0.79 |
| verify pass: card 7's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.80 | 0.79 |
| verify pass: card 8's host gap before entry | 0.32 | 0.32 | 0.50 | 0.50 | 0.86 | 0.84 |
| the server's step, this load | 126.31 | 122.15 | 172.55 | 147.05 | 221.29 | 174.24 |
| the server's step, load `warm` | 125.46 | 121.42 | 167.38 | 148.57 | 225.44 | 180.37 |

Medians of the same:

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| **the step** | 126.09 | 122.87 | 150.98 | 149.52 | 179.54 | 176.02 |
| GPU time, every evaluation | 113.07 | 112.85 | 135.32 | 135.23 | 149.62 | 149.37 |
| **outside the kernels** (the step less that) | 13.40 | 9.90 | 14.50 | 14.31 | 30.14 | 27.22 |
| verify pass: GPU time of the nine cards | 101.62 | 101.54 | 120.08 | 120.08 | 130.93 | 130.77 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.28 | 4.25 | 6.12 | 6.09 | 10.56 | 10.53 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 3.84 | 3.83 | 5.21 | 5.19 | 8.11 | 7.99 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.53 | 0.53 | 0.70 | 0.70 | 1.04 | 1.04 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.64 | 0.63 | 0.79 | 0.79 | 1.14 | 1.13 |
| draft pass 0: GPU time, card 4 and card 8 | 3.62 | 3.64 | 5.14 | 5.14 | 6.04 | 6.03 |
| draft pass 0: turn | 0.34 | 0.33 | 0.38 | 0.38 | 3.04 | 3.00 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.55 | 0.56 | 0.65 | 0.64 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 0.64 | 0.64 | 0.73 | 0.73 |
| draft pass 1: GPU time, card 4 and card 8 | 2.60 | 2.55 | 3.41 | 3.41 | 4.22 | 4.22 |
| draft pass 1: turn | 0.55 | 0.54 | 0.65 | 0.65 | 0.87 | 0.87 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.55 | 0.55 | 0.65 | 0.65 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.62 | 0.62 | 0.71 | 0.72 |
| draft pass 2: GPU time, card 4 and card 8 | 2.55 | 2.57 | 3.33 | 3.34 | 4.17 | 4.17 |
| draft pass 2: turn | 0.52 | 0.52 | 0.65 | 0.65 | 0.86 | 0.86 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.56 | 0.56 | 0.65 | 0.65 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.62 | 0.62 | 0.72 | 0.72 |
| draft pass 3: GPU time, card 4 and card 8 | 2.55 | 2.57 | 3.35 | 3.35 | 4.18 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 5.46 | 2.03 | 4.14 | 4.05 | 12.35 | 8.85 |
| the draft's crossings, all passes | 2.08 | 2.07 | 2.37 | 2.37 | 2.98 | 2.98 |
| the draft's turns before its last pass | 1.41 | 1.40 | 1.69 | 1.69 | 4.79 | 4.74 |
| k-pool inputs: host us a step, all passes (set_input) | 41.00 | 40.00 | nan | 218.00 | nan | 584.00 |
| k-pool inputs: host us, the verify pass | 9.00 | 9.00 | nan | 47.00 | nan | 123.00 |
| k-pool inputs: host us, the draft's passes | 32.00 | 31.00 | nan | 170.00 | nan | 459.00 |
| k-pool inputs: KiB a card, the verify pass | 3.70 | 3.70 | nan | 35.70 | nan | 99.70 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.20 | 3.20 | nan | 11.20 | nan | 27.20 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 46.60 | 46.60 | nan | 390.60 | nan | 1078.60 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 17.00 | 17.00 | nan | 105.00 | nan | 281.00 |
| k-pool inputs: n_pool of the verify pass | 64.00 | 64.00 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.24 | 0.24 | 0.22 | 0.22 | 0.22 | 0.21 |
| verify pass: card 1's host gap before entry | 0.30 | 0.29 | 0.48 | 0.48 | 0.83 | 0.83 |
| verify pass: card 2's host gap before entry | 0.27 | 0.27 | 0.44 | 0.44 | 0.80 | 0.79 |
| verify pass: card 3's host gap before entry | 0.27 | 0.27 | 0.44 | 0.44 | 0.80 | 0.80 |
| verify pass: card 4's host gap before entry | 0.23 | 0.23 | 0.40 | 0.40 | 0.73 | 0.73 |
| verify pass: card 5's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.79 | 0.79 |
| verify pass: card 6's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.79 | 0.79 |
| verify pass: card 7's host gap before entry | 0.26 | 0.26 | 0.44 | 0.44 | 0.79 | 0.79 |
| verify pass: card 8's host gap before entry | 0.32 | 0.32 | 0.49 | 0.49 | 0.85 | 0.84 |

**From 16384 to 49152 tokens, ms a step** (the deeper request less the shallower, pass against pass; the share is of the step's growth):

| part | p1: 16384 | p1: 49152 | p1: growth | p1: share | p2: 16384 | p2: 49152 | p2: growth | p2: share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **the step** | 154.32 | 191.37 | +37.04 | 100.0 % | 149.59 | 176.16 | +26.57 | 100.0 % |
| GPU time, every evaluation | 135.33 | 149.88 | +14.54 | 39.3 % | 135.11 | 149.33 | +14.22 | 53.5 % |
| **outside the kernels** (the step less that) | 18.99 | 41.49 | +22.50 | 60.7 % | 14.48 | 26.83 | +12.35 | 46.5 % |
| verify pass: GPU time of the nine cards | 119.93 | 130.93 | +11.00 | 29.7 % | 119.89 | 130.74 | +10.85 | 40.8 % |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.14 | 10.58 | +4.44 | 12.0 % | 6.17 | 10.55 | +4.38 | 16.5 % |
| verify pass: the host's launch, card 0's entry to card 8's exit | 5.32 | 9.26 | +3.94 | 10.6 % | 5.21 | 7.99 | +2.79 | 10.5 % |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.70 | 1.04 | +0.35 | 0.9 % | 0.70 | 1.05 | +0.35 | 1.3 % |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.79 | 1.14 | +0.35 | 0.9 % | 0.79 | 1.14 | +0.35 | 1.3 % |
| draft pass 0: GPU time, card 4 and card 8 | 5.14 | 6.06 | +0.92 | 2.5 % | 5.14 | 6.04 | +0.90 | 3.4 % |
| draft pass 0: turn | 0.38 | 2.72 | +2.34 | 6.3 % | 0.45 | 2.57 | +2.12 | 8.0 % |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.56 | 0.65 | +0.09 | 0.2 % | 0.56 | 0.65 | +0.09 | 0.3 % |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.64 | 0.74 | +0.09 | 0.3 % | 0.64 | 0.73 | +0.09 | 0.3 % |
| draft pass 1: GPU time, card 4 and card 8 | 3.46 | 4.33 | +0.86 | 2.3 % | 3.41 | 4.22 | +0.81 | 3.1 % |
| draft pass 1: turn | 0.66 | 0.88 | +0.22 | 0.6 % | 0.65 | 0.87 | +0.22 | 0.8 % |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.56 | 0.65 | +0.09 | 0.2 % | 0.56 | 0.65 | +0.09 | 0.3 % |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.62 | 0.72 | +0.09 | 0.2 % | 0.62 | 0.72 | +0.09 | 0.4 % |
| draft pass 2: GPU time, card 4 and card 8 | 3.40 | 4.29 | +0.89 | 2.4 % | 3.34 | 4.17 | +0.83 | 3.1 % |
| draft pass 2: turn | 0.65 | 0.86 | +0.21 | 0.6 % | 0.65 | 0.86 | +0.21 | 0.8 % |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.56 | 0.65 | +0.09 | 0.2 % | 0.56 | 0.65 | +0.09 | 0.3 % |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.63 | 0.72 | +0.09 | 0.2 % | 0.63 | 0.72 | +0.09 | 0.3 % |
| draft pass 3: GPU time, card 4 and card 8 | 3.39 | 4.27 | +0.87 | 2.4 % | 3.35 | 4.17 | +0.82 | 3.1 % |
| draft pass 3: turn (the server before the next verify pass) | 8.67 | 23.34 | +14.67 | 39.6 % | 4.07 | 8.86 | +4.79 | 18.0 % |
| the draft's crossings, all passes | 2.37 | 2.99 | +0.62 | 1.7 % | 2.37 | 2.99 | +0.62 | 2.3 % |
| the draft's turns before its last pass | 1.70 | 4.46 | +2.77 | 7.5 % | 1.75 | 4.31 | +2.55 | 9.6 % |
| k-pool inputs: host us a step, all passes (set_input) | nan | nan | +nan |  | 219.54 | 584.74 | +365.20 |  |
| k-pool inputs: host us, the verify pass | nan | nan | +nan |  | 47.78 | 124.71 | +76.93 |  |
| k-pool inputs: host us, the draft's passes | nan | nan | +nan |  | 171.76 | 460.03 | +288.27 |  |
| k-pool inputs: KiB a card, the verify pass | nan | nan | +nan |  | 35.70 | 99.70 | +64.00 |  |
| k-pool inputs: KiB, one draft pass (card 8) | nan | nan | +nan |  | 11.20 | 27.20 | +16.00 |  |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | nan | nan | +nan |  | 390.60 | 1078.60 | +688.00 |  |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | nan | nan | +nan |  | 105.00 | 281.00 | +176.00 |  |
| k-pool inputs: n_pool of the verify pass | nan | nan | +nan |  | 4160.00 | 12352.00 | +8192.00 |  |
| verify pass: card 0's host span | 0.23 | 0.33 | +0.10 | 0.3 % | 0.22 | 0.21 | -0.00 | -0.0 % |
| verify pass: card 1's host gap before entry | 0.49 | 0.84 | +0.35 | 1.0 % | 0.48 | 0.83 | +0.35 | 1.3 % |
| verify pass: card 2's host gap before entry | 0.45 | 0.81 | +0.36 | 1.0 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 3's host gap before entry | 0.45 | 0.82 | +0.36 | 1.0 % | 0.44 | 0.80 | +0.36 | 1.3 % |
| verify pass: card 4's host gap before entry | 0.41 | 0.74 | +0.33 | 0.9 % | 0.40 | 0.73 | +0.33 | 1.2 % |
| verify pass: card 5's host gap before entry | 0.45 | 0.80 | +0.35 | 1.0 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 6's host gap before entry | 0.45 | 0.81 | +0.36 | 1.0 % | 0.44 | 0.79 | +0.34 | 1.3 % |
| verify pass: card 7's host gap before entry | 0.44 | 0.80 | +0.36 | 1.0 % | 0.44 | 0.79 | +0.35 | 1.3 % |
| verify pass: card 8's host gap before entry | 0.50 | 0.86 | +0.36 | 1.0 % | 0.50 | 0.84 | +0.35 | 1.3 % |

