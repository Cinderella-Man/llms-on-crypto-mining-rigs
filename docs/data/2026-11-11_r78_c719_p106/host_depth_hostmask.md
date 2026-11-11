**Load `hostmask`:** 11628 evaluations in 14 pieces (cut at gaps over 3000 ms), 10 after joining those with fewer than 10 main passes to the next; the first is the load's own, the others are its 9 requests.

| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0 | (the load) | 9 | 1 | 0 | 2 | | | |
| 1 | warmup 0 | 258 | 13 | 49 | 59 | no | 0 | 0 |
| 2 | decode 0 p1 | 1641 | 79 | 310 | 389 | yes | 77 | 74 |
| 3 | decode 0 p2 | 1566 | 75 | 297 | 372 | yes | 74 | 71 |
| 4 | prime 16384 | 615 | 31 | 113 | 162 | no | 0 | 0 |
| 5 | decode 16384 p1 | 1575 | 75 | 301 | 356 | no | 71 | 67 |
| 6 | decode 16384 p2 | 1494 | 72 | 282 | 354 | yes | 71 | 67 |
| 7 | prime 49152 | 1353 | 65 | 257 | 327 | no | 0 | 0 |
| 8 | decode 49152 p1 | 1581 | 76 | 299 | 369 | no | 75 | 72 |
| 9 | decode 49152 p2 | 1536 | 73 | 294 | 367 | yes | 72 | 69 |

**The k-pool inputs by request, pass and micro-batch size** (one line per evaluated micro-batch; the median, and in brackets the lowest and highest):

| request | pass | tokens | micro-batches | n_pool | host us | KiB a card |
|---|---|---:|---:|---|---|---|
| (load) | ? | 2 | 2 | 64 | 20.5 (9.0, 32.0) | 3.1 (3.1, 3.1) |
| warmup 0 | ? | 60 | 2 | 64 | 21.5 (19.0, 24.0) | 5.3 (5.3, 5.3) |
| warmup 0 | ? | 4 | 24 | 64 | 10.0 (8.0, 12.0) | 3.2 (3.2, 3.2) |
| warmup 0 | ? | 1 | 33 | 64 | 7.0 (5.0, 18.0) | 3.1 (3.1, 3.1) |
| decode 0 p1 | main | 60 | 1 | 64 | 40.0 (40.0, 40.0) | 5.3 (5.3, 5.3) |
| decode 0 p1 | draft | 60 | 1 | 64 | 20.0 (20.0, 20.0) | 5.3 (5.3, 5.3) |
| decode 0 p1 | main | 4 | 77 | 64 to 128 | 9.0 (7.0, 20.0) | 3.2 (3.2, 3.2) |
| decode 0 p1 | draft | 4 | 77 | 64 to 128 | 10.0 (8.0, 33.0) | 3.2 (3.2, 3.2) |
| decode 0 p1 | draft | 1 | 231 | 64 to 128 | 7.0 (5.0, 11.0) | 3.1 (3.1, 3.1) |
| decode 0 p1 | main | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.1 (3.1, 3.1) |
| decode 0 p1 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.1 (3.1, 3.1) |
| decode 0 p2 | main | 4 | 74 | 64 to 128 | 9.0 (7.0, 11.0) | 3.2 (3.2, 3.2) |
| decode 0 p2 | draft | 4 | 74 | 64 to 128 | 10.0 (9.0, 38.0) | 3.2 (3.2, 3.2) |
| decode 0 p2 | draft | 1 | 222 | 64 to 128 | 7.0 (5.0, 17.0) | 3.1 (3.1, 3.1) |
| decode 0 p2 | main | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.1 (3.1, 3.1) |
| decode 0 p2 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.1 (3.1, 3.1) |
| prime 16384 | ? | 512 | 31 | 192 to 4096 | 177.0 (104.0, 365.0) | 25.0 (25.0, 25.0) |
| prime 16384 | ? | 128 | 127 | 64 to 4096 | 51.0 (30.0, 84.0) | 7.8 (7.8, 7.8) |
| prime 16384 | ? | 444 | 1 | 3968 | 141.0 (141.0, 141.0) | 21.0 (21.0, 21.0) |
| prime 16384 | ? | 60 | 1 | 3968 | 54.0 (54.0, 54.0) | 5.3 (5.3, 5.3) |
| prime 16384 | ? | 4 | 2 | 4096 | 72.0 (43.0, 101.0) | 3.2 (3.2, 3.2) |
| decode 16384 p1 | ? | 60 | 2 | 4096 | 78.0 (54.0, 102.0) | 5.3 (5.3, 5.3) |
| decode 16384 p1 | ? | 4 | 142 | 4160 | 44.0 (42.0, 76.0) | 3.2 (3.2, 3.2) |
| decode 16384 p1 | ? | 1 | 212 | 4160 | 39.0 (32.0, 150.0) | 3.1 (3.1, 3.1) |
| decode 16384 p2 | main | 4 | 71 | 4160 | 43.0 (42.0, 77.0) | 3.2 (3.2, 3.2) |
| decode 16384 p2 | draft | 4 | 71 | 4160 | 44.0 (43.0, 66.0) | 3.2 (3.2, 3.2) |
| decode 16384 p2 | draft | 1 | 211 | 4160 | 39.0 (32.0, 72.0) | 3.1 (3.1, 3.1) |
| decode 16384 p2 | main | 1 | 1 | 4160 | 79.0 (79.0, 79.0) | 3.1 (3.1, 3.1) |
| prime 49152 | ? | 512 | 65 | 4096 to 12288 | 255.0 (142.0, 587.0) | 25.0 (25.0, 25.0) |
| prime 49152 | ? | 128 | 260 | 4032 to 12288 | 101.0 (59.0, 253.0) | 7.8 (7.8, 7.8) |
| prime 49152 | ? | 4 | 2 | 12288 | 133.5 (112.0, 155.0) | 3.2 (3.2, 3.2) |
| decode 49152 p1 | ? | 60 | 2 | 12288 | 143.0 (140.0, 146.0) | 5.3 (5.3, 5.3) |
| decode 49152 p1 | ? | 4 | 146 | 12352 | 114.0 (109.0, 193.0) | 3.2 (3.2, 3.2) |
| decode 49152 p1 | ? | 1 | 219 | 12352 | 106.0 (89.0, 158.0) | 3.1 (3.1, 3.1) |
| decode 49152 p1 | ? | 3 | 2 | 12352 | 128.0 (115.0, 141.0) | 3.1 (3.1, 3.1) |
| decode 49152 p2 | main | 4 | 73 | 12352 | 112.0 (110.0, 161.0) | 3.2 (3.2, 3.2) |
| decode 49152 p2 | draft | 4 | 73 | 12352 | 114.0 (113.0, 281.0) | 3.2 (3.2, 3.2) |
| decode 49152 p2 | draft | 1 | 219 | 12352 | 106.0 (89.0, 144.0) | 3.1 (3.1, 3.1) |
| decode 49152 p2 | draft | 3 | 2 | 12352 | 147.5 (146.0, 149.0) | 3.1 (3.1, 3.1) |

**The step by part, ms** (the mean over each decode's typical steps with 4 draft passes; every column is one request of the one load):

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| typical steps | 74 | 71 | 67 | 67 | 72 | 69 |
| **the step** | 127.73 | 123.43 | 150.26 | 148.52 | 179.72 | 174.64 |
| GPU time, every evaluation | 114.34 | 113.02 | 134.87 | 134.70 | 149.32 | 148.68 |
| **outside the kernels** (the step less that) | 13.39 | 10.41 | 15.39 | 13.82 | 30.40 | 25.97 |
| verify pass: GPU time of the nine cards | 101.57 | 101.60 | 119.59 | 119.60 | 130.51 | 130.23 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.26 | 4.23 | 6.05 | 6.04 | 10.55 | 10.53 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 4.68 | 4.69 | 3.79 | 3.78 | 4.49 | 3.80 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.63 | 0.63 | 0.61 | 0.61 | 0.61 | 0.61 |
| draft pass 0: GPU time, card 4 and card 8 | 3.65 | 3.68 | 5.07 | 5.07 | 6.00 | 5.97 |
| draft pass 0: turn | 0.35 | 0.32 | 0.37 | 0.36 | 2.67 | 2.61 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |
| draft pass 1: GPU time, card 4 and card 8 | 3.06 | 2.62 | 3.40 | 3.34 | 4.27 | 4.16 |
| draft pass 1: turn | 0.56 | 0.55 | 0.66 | 0.66 | 0.87 | 0.87 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.52 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.58 | 0.58 | 0.59 | 0.59 |
| draft pass 2: GPU time, card 4 and card 8 | 3.09 | 2.56 | 3.41 | 3.34 | 4.29 | 4.15 |
| draft pass 2: turn | 0.53 | 0.53 | 0.66 | 0.66 | 0.87 | 0.87 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.58 | 0.59 | 0.58 | 0.58 |
| draft pass 3: GPU time, card 4 and card 8 | 2.95 | 2.55 | 3.40 | 3.34 | 4.26 | 4.16 |
| draft pass 3: turn (the server before the next verify pass) | 5.50 | 2.61 | 5.48 | 3.93 | 13.26 | 8.91 |
| the draft's crossings, all passes | 2.06 | 2.06 | 2.05 | 2.06 | 2.06 | 2.06 |
| the draft's turns before its last pass | 1.44 | 1.39 | 1.68 | 1.67 | 4.41 | 4.35 |
| k-pool inputs: host us a step, all passes (set_input) | 41.42 | 40.97 | nan | 209.40 | nan | 547.20 |
| k-pool inputs: host us, the verify pass | 9.34 | 8.58 | nan | 44.03 | nan | 112.93 |
| k-pool inputs: host us, the draft's passes | 32.08 | 32.39 | nan | 165.37 | nan | 434.28 |
| k-pool inputs: KiB a card, the verify pass | 3.20 | 3.20 | nan | 3.20 | nan | 3.20 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.10 | 3.10 | nan | 3.10 | nan | 3.10 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 41.30 | 41.30 | nan | 41.30 | nan | 41.30 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 15.70 | 15.70 | nan | 15.70 | nan | 15.70 |
| k-pool inputs: n_pool of the verify pass | 78.70 | 79.32 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.34 | 0.34 | 0.22 | 0.22 | 0.31 | 0.23 |
| verify pass: card 1's host gap before entry | 0.29 | 0.29 | 0.29 | 0.29 | 0.30 | 0.29 |
| verify pass: card 2's host gap before entry | 0.28 | 0.27 | 0.26 | 0.26 | 0.27 | 0.26 |
| verify pass: card 3's host gap before entry | 0.28 | 0.27 | 0.26 | 0.26 | 0.27 | 0.27 |
| verify pass: card 4's host gap before entry | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| verify pass: card 5's host gap before entry | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| verify pass: card 6's host gap before entry | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| verify pass: card 7's host gap before entry | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| verify pass: card 8's host gap before entry | 0.32 | 0.31 | 0.32 | 0.32 | 0.32 | 0.32 |
| the server's step, this load | 126.15 | 121.96 | 164.44 | 146.11 | 207.07 | 172.82 |
| the server's step, load `warm` | 125.41 | 121.40 | 167.35 | 149.33 | 226.26 | 180.38 |

Medians of the same:

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| **the step** | 125.62 | 122.68 | 149.72 | 148.64 | 179.00 | 174.63 |
| GPU time, every evaluation | 113.00 | 112.84 | 134.78 | 134.83 | 148.98 | 148.75 |
| **outside the kernels** (the step less that) | 13.25 | 9.81 | 13.84 | 13.75 | 30.63 | 25.86 |
| verify pass: GPU time of the nine cards | 101.53 | 101.51 | 119.70 | 119.67 | 130.43 | 130.30 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.25 | 4.22 | 6.03 | 6.02 | 10.52 | 10.51 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 3.83 | 3.80 | 3.78 | 3.77 | 3.80 | 3.77 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.52 | 0.52 | 0.51 | 0.51 | 0.52 | 0.52 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.63 | 0.62 | 0.61 | 0.61 | 0.61 | 0.61 |
| draft pass 0: GPU time, card 4 and card 8 | 3.63 | 3.67 | 5.07 | 5.06 | 5.98 | 5.98 |
| draft pass 0: turn | 0.31 | 0.31 | 0.36 | 0.35 | 3.05 | 3.05 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.59 | 0.60 | 0.60 | 0.60 | 0.60 |
| draft pass 1: GPU time, card 4 and card 8 | 2.61 | 2.61 | 3.33 | 3.34 | 4.16 | 4.15 |
| draft pass 1: turn | 0.55 | 0.54 | 0.66 | 0.65 | 0.87 | 0.87 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 0.58 | 0.58 | 0.58 | 0.58 |
| draft pass 2: GPU time, card 4 and card 8 | 2.56 | 2.55 | 3.34 | 3.34 | 4.16 | 4.16 |
| draft pass 2: turn | 0.52 | 0.52 | 0.65 | 0.65 | 0.86 | 0.86 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.57 | 0.58 | 0.58 | 0.58 | 0.58 |
| draft pass 3: GPU time, card 4 and card 8 | 2.55 | 2.55 | 3.35 | 3.35 | 4.18 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 5.47 | 2.02 | 3.93 | 3.89 | 13.34 | 8.38 |
| the draft's crossings, all passes | 2.06 | 2.05 | 2.05 | 2.05 | 2.05 | 2.05 |
| the draft's turns before its last pass | 1.39 | 1.38 | 1.67 | 1.66 | 4.81 | 4.78 |
| k-pool inputs: host us a step, all passes (set_input) | 40.00 | 40.00 | nan | 206.00 | nan | 544.00 |
| k-pool inputs: host us, the verify pass | 9.00 | 9.00 | nan | 43.00 | nan | 112.00 |
| k-pool inputs: host us, the draft's passes | 31.00 | 31.00 | nan | 162.00 | nan | 430.00 |
| k-pool inputs: KiB a card, the verify pass | 3.20 | 3.20 | nan | 3.20 | nan | 3.20 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.10 | 3.10 | nan | 3.10 | nan | 3.10 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 41.30 | 41.30 | nan | 41.30 | nan | 41.30 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 15.70 | 15.70 | nan | 15.70 | nan | 15.70 |
| k-pool inputs: n_pool of the verify pass | 64.00 | 64.00 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.25 | 0.25 | 0.22 | 0.22 | 0.22 | 0.22 |
| verify pass: card 1's host gap before entry | 0.29 | 0.28 | 0.29 | 0.29 | 0.29 | 0.29 |
| verify pass: card 2's host gap before entry | 0.27 | 0.27 | 0.26 | 0.26 | 0.26 | 0.26 |
| verify pass: card 3's host gap before entry | 0.27 | 0.27 | 0.26 | 0.26 | 0.26 | 0.26 |
| verify pass: card 4's host gap before entry | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| verify pass: card 5's host gap before entry | 0.26 | 0.25 | 0.26 | 0.26 | 0.26 | 0.26 |
| verify pass: card 6's host gap before entry | 0.26 | 0.25 | 0.26 | 0.25 | 0.25 | 0.25 |
| verify pass: card 7's host gap before entry | 0.26 | 0.25 | 0.25 | 0.25 | 0.26 | 0.25 |
| verify pass: card 8's host gap before entry | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 |

**From 16384 to 49152 tokens, ms a step** (the deeper request less the shallower, pass against pass; the share is of the step's growth):

| part | p1: 16384 | p1: 49152 | p1: growth | p1: share | p2: 16384 | p2: 49152 | p2: growth | p2: share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **the step** | 150.26 | 179.72 | +29.47 | 100.0 % | 148.52 | 174.64 | +26.13 | 100.0 % |
| GPU time, every evaluation | 134.87 | 149.32 | +14.46 | 49.1 % | 134.70 | 148.68 | +13.98 | 53.5 % |
| **outside the kernels** (the step less that) | 15.39 | 30.40 | +15.01 | 50.9 % | 13.82 | 25.97 | +12.15 | 46.5 % |
| verify pass: GPU time of the nine cards | 119.59 | 130.51 | +10.92 | 37.1 % | 119.60 | 130.23 | +10.63 | 40.7 % |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.05 | 10.55 | +4.50 | 15.3 % | 6.04 | 10.53 | +4.48 | 17.2 % |
| verify pass: the host's launch, card 0's entry to card 8's exit | 3.79 | 4.49 | +0.70 | 2.4 % | 3.78 | 3.80 | +0.02 | 0.1 % |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.52 | 0.52 | -0.00 | -0.0 % | 0.52 | 0.52 | -0.00 | -0.0 % |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.61 | 0.61 | +0.00 | 0.0 % | 0.61 | 0.61 | -0.00 | -0.0 % |
| draft pass 0: GPU time, card 4 and card 8 | 5.07 | 6.00 | +0.93 | 3.2 % | 5.07 | 5.97 | +0.90 | 3.5 % |
| draft pass 0: turn | 0.37 | 2.67 | +2.30 | 7.8 % | 0.36 | 2.61 | +2.26 | 8.6 % |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | +0.00 | 0.0 % | 0.51 | 0.51 | -0.00 | -0.0 % |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | -0.00 | -0.0 % | 0.60 | 0.60 | -0.00 | -0.0 % |
| draft pass 1: GPU time, card 4 and card 8 | 3.40 | 4.27 | +0.87 | 2.9 % | 3.34 | 4.16 | +0.82 | 3.1 % |
| draft pass 1: turn | 0.66 | 0.87 | +0.21 | 0.7 % | 0.66 | 0.87 | +0.21 | 0.8 % |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | +0.00 | 0.0 % | 0.51 | 0.52 | +0.01 | 0.0 % |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.59 | +0.01 | 0.0 % | 0.58 | 0.59 | +0.01 | 0.0 % |
| draft pass 2: GPU time, card 4 and card 8 | 3.41 | 4.29 | +0.88 | 3.0 % | 3.34 | 4.15 | +0.81 | 3.1 % |
| draft pass 2: turn | 0.66 | 0.87 | +0.21 | 0.7 % | 0.66 | 0.87 | +0.21 | 0.8 % |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.51 | 0.51 | +0.00 | 0.0 % | 0.51 | 0.51 | -0.00 | -0.0 % |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | +0.00 | 0.0 % | 0.59 | 0.58 | -0.00 | -0.0 % |
| draft pass 3: GPU time, card 4 and card 8 | 3.40 | 4.26 | +0.86 | 2.9 % | 3.34 | 4.16 | +0.82 | 3.1 % |
| draft pass 3: turn (the server before the next verify pass) | 5.48 | 13.26 | +7.77 | 26.4 % | 3.93 | 8.91 | +4.98 | 19.1 % |
| the draft's crossings, all passes | 2.05 | 2.06 | +0.00 | 0.0 % | 2.06 | 2.06 | -0.00 | -0.0 % |
| the draft's turns before its last pass | 1.68 | 4.41 | +2.73 | 9.3 % | 1.67 | 4.35 | +2.68 | 10.3 % |
| k-pool inputs: host us a step, all passes (set_input) | nan | nan | +nan |  | 209.40 | 547.20 | +337.80 |  |
| k-pool inputs: host us, the verify pass | nan | nan | +nan |  | 44.03 | 112.93 | +68.90 |  |
| k-pool inputs: host us, the draft's passes | nan | nan | +nan |  | 165.37 | 434.28 | +268.90 |  |
| k-pool inputs: KiB a card, the verify pass | nan | nan | +nan |  | 3.20 | 3.20 | +0.00 |  |
| k-pool inputs: KiB, one draft pass (card 8) | nan | nan | +nan |  | 3.10 | 3.10 | +0.00 |  |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | nan | nan | +nan |  | 41.30 | 41.30 | +0.00 |  |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | nan | nan | +nan |  | 15.70 | 15.70 | +0.00 |  |
| k-pool inputs: n_pool of the verify pass | nan | nan | +nan |  | 4160.00 | 12352.00 | +8192.00 |  |
| verify pass: card 0's host span | 0.22 | 0.31 | +0.09 | 0.3 % | 0.22 | 0.23 | +0.00 | 0.0 % |
| verify pass: card 1's host gap before entry | 0.29 | 0.30 | +0.00 | 0.0 % | 0.29 | 0.29 | +0.00 | 0.0 % |
| verify pass: card 2's host gap before entry | 0.26 | 0.27 | +0.00 | 0.0 % | 0.26 | 0.26 | +0.00 | 0.0 % |
| verify pass: card 3's host gap before entry | 0.26 | 0.27 | +0.00 | 0.0 % | 0.26 | 0.27 | +0.00 | 0.0 % |
| verify pass: card 4's host gap before entry | 0.23 | 0.23 | +0.00 | 0.0 % | 0.23 | 0.23 | +0.00 | 0.0 % |
| verify pass: card 5's host gap before entry | 0.26 | 0.26 | +0.00 | 0.0 % | 0.26 | 0.26 | +0.00 | 0.0 % |
| verify pass: card 6's host gap before entry | 0.26 | 0.26 | +0.00 | 0.0 % | 0.26 | 0.26 | +0.00 | 0.0 % |
| verify pass: card 7's host gap before entry | 0.26 | 0.26 | +0.00 | 0.0 % | 0.26 | 0.26 | -0.00 | -0.0 % |
| verify pass: card 8's host gap before entry | 0.32 | 0.32 | +0.00 | 0.0 % | 0.32 | 0.32 | +0.00 | 0.0 % |

