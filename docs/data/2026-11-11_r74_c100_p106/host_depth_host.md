**Load `host`:** 9432 evaluations in 14 pieces (cut at gaps over 3000 ms), 10 after joining those with fewer than 10 main passes to the next; the first is the load's own, the others are its 9 requests.

| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0 | (the load) | 9 | 1 | 0 | 2 | | | |
| 1 | warmup 0 | 211 | 13 | 47 | 59 | no | 0 | 0 |
| 2 | decode 0 p1 | 1331 | 79 | 310 | 389 | yes | 77 | 74 |
| 3 | decode 0 p2 | 1269 | 75 | 297 | 372 | yes | 74 | 71 |
| 4 | prime 16384 | 503 | 31 | 112 | 162 | no | 0 | 0 |
| 5 | decode 16384 p1 | 1275 | 75 | 300 | 356 | no | 71 | 67 |
| 6 | decode 16384 p2 | 1212 | 72 | 282 | 354 | yes | 71 | 67 |
| 7 | prime 49152 | 1097 | 65 | 256 | 327 | no | 0 | 0 |
| 8 | decode 49152 p1 | 1282 | 76 | 299 | 369 | no | 75 | 72 |
| 9 | decode 49152 p2 | 1243 | 73 | 293 | 367 | no | 72 | 69 |

**The k-pool inputs by request, pass and micro-batch size** (one line per evaluated micro-batch; the median, and in brackets the lowest and highest):

| request | pass | tokens | micro-batches | n_pool | host us | KiB a card |
|---|---|---:|---:|---|---|---|
| (load) | ? | 2 | 2 | 64 | 15.5 (9.0, 22.0) | 1.6 (1.6, 1.6) |
| warmup 0 | ? | 60 | 2 | 64 | 21.5 (19.0, 24.0) | 10.5 (10.5, 10.5) |
| warmup 0 | ? | 4 | 24 | 64 | 9.0 (7.0, 9.0) | 1.9 (1.9, 1.9) |
| warmup 0 | ? | 1 | 33 | 64 | 6.0 (4.0, 9.0) | 1.4 (1.4, 1.4) |
| decode 0 p1 | main | 60 | 1 | 64 | 25.0 (25.0, 25.0) | 10.5 (10.5, 10.5) |
| decode 0 p1 | draft | 60 | 1 | 64 | 21.0 (21.0, 21.0) | 10.5 (10.5, 10.5) |
| decode 0 p1 | main | 4 | 77 | 64 to 128 | 8.0 (7.0, 10.0) | 1.9 (1.9, 3.7) |
| decode 0 p1 | draft | 4 | 77 | 64 to 128 | 9.0 (8.0, 11.0) | 1.9 (1.9, 3.7) |
| decode 0 p1 | draft | 1 | 231 | 64 to 128 | 6.0 (4.0, 17.0) | 1.4 (1.4, 2.8) |
| decode 0 p1 | main | 2 | 1 | 128 | 9.0 (9.0, 9.0) | 3.1 (3.1, 3.1) |
| decode 0 p1 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.1 (3.1, 3.1) |
| decode 0 p2 | main | 4 | 74 | 64 to 128 | 8.0 (7.0, 23.0) | 1.9 (1.9, 3.7) |
| decode 0 p2 | draft | 4 | 74 | 64 to 128 | 9.0 (8.0, 18.0) | 1.9 (1.9, 3.7) |
| decode 0 p2 | draft | 1 | 222 | 64 to 128 | 6.0 (4.0, 17.0) | 1.4 (1.4, 2.8) |
| decode 0 p2 | main | 2 | 1 | 128 | 9.0 (9.0, 9.0) | 3.1 (3.1, 3.1) |
| decode 0 p2 | draft | 2 | 1 | 128 | 9.0 (9.0, 9.0) | 3.1 (3.1, 3.1) |
| prime 16384 | ? | 512 | 31 | 192 to 4096 | 803.0 (232.0, 1869.0) | 2168.3 (210.8, 4191.0) |
| prime 16384 | ? | 128 | 127 | 64 to 4096 | 151.0 (34.0, 284.0) | 573.0 (21.0, 1107.8) |
| prime 16384 | ? | 444 | 1 | 3968 | 995.0 (995.0, 995.0) | 3531.5 (3531.5, 3531.5) |
| prime 16384 | ? | 60 | 1 | 3968 | 157.0 (157.0, 157.0) | 544.3 (544.3, 544.3) |
| prime 16384 | ? | 4 | 2 | 4096 | 131.5 (73.0, 190.0) | 112.2 (112.2, 112.2) |
| decode 16384 p1 | ? | 60 | 2 | 4096 | 304.5 (192.0, 417.0) | 561.8 (561.8, 561.8) |
| decode 16384 p1 | ? | 4 | 142 | 4160 | 75.0 (72.0, 114.0) | 113.9 (113.9, 113.9) |
| decode 16384 p1 | ? | 1 | 212 | 4160 | 69.0 (63.0, 116.0) | 89.4 (89.4, 89.4) |
| decode 16384 p2 | main | 4 | 71 | 4160 | 74.0 (72.0, 203.0) | 113.9 (113.9, 113.9) |
| decode 16384 p2 | draft | 4 | 71 | 4160 | 75.0 (73.0, 101.0) | 113.9 (113.9, 113.9) |
| decode 16384 p2 | draft | 1 | 211 | 4160 | 70.0 (62.0, 244.0) | 89.4 (89.4, 89.4) |
| decode 16384 p2 | main | 1 | 1 | 4160 | 103.0 (103.0, 103.0) | 89.4 (89.4, 89.4) |
| prime 49152 | ? | 512 | 65 | 4096 to 12288 | 2414.0 (1484.0, 8244.0) | 8367.0 (4191.0, 12543.0) |
| prime 49152 | ? | 128 | 260 | 4032 to 12288 | 607.0 (260.0, 1040.0) | 2203.2 (1090.5, 3315.8) |
| prime 49152 | ? | 4 | 2 | 12288 | 231.5 (205.0, 258.0) | 336.2 (336.2, 336.2) |
| decode 49152 p1 | ? | 60 | 2 | 12288 | 434.5 (427.0, 442.0) | 1681.8 (1681.8, 1681.8) |
| decode 49152 p1 | ? | 4 | 146 | 12352 | 206.0 (202.0, 277.0) | 337.9 (337.9, 337.9) |
| decode 49152 p1 | ? | 1 | 219 | 12352 | 194.0 (176.0, 318.0) | 265.4 (265.4, 265.4) |
| decode 49152 p1 | ? | 3 | 2 | 12352 | 297.5 (218.0, 377.0) | 313.7 (313.7, 313.7) |
| decode 49152 p2 | ? | 4 | 146 | 12352 | 206.0 (201.0, 261.0) | 337.9 (337.9, 337.9) |
| decode 49152 p2 | ? | 1 | 219 | 12352 | 194.0 (177.0, 323.0) | 265.4 (265.4, 265.4) |
| decode 49152 p2 | ? | 3 | 2 | 12352 | 220.5 (203.0, 238.0) | 313.7 (313.7, 313.7) |

**The step by part, ms** (the mean over each decode's typical steps with 4 draft passes; every column is one request of the one load):

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| typical steps | 74 | 71 | 67 | 67 | 72 | 69 |
| **the step** | 127.25 | 123.52 | 153.63 | 151.27 | 211.13 | 182.36 |
| GPU time, every evaluation | 114.20 | 112.88 | 135.18 | 134.90 | 150.21 | 149.60 |
| **outside the kernels** (the step less that) | 13.05 | 10.64 | 18.45 | 16.37 | 60.93 | 32.76 |
| verify pass: GPU time of the nine cards | 101.60 | 101.62 | 119.98 | 119.88 | 131.31 | 131.05 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.26 | 4.31 | 6.12 | 6.01 | 10.51 | 10.42 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 4.74 | 4.76 | 8.87 | 8.74 | 20.28 | 18.68 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.58 | 0.58 | 1.18 | 1.18 | 2.40 | 2.40 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.62 | 0.62 | 1.22 | 1.22 | 2.43 | 2.43 |
| draft pass 0: GPU time, card 4 and card 8 | 3.62 | 3.62 | 5.08 | 5.07 | 6.06 | 6.02 |
| draft pass 0: turn | 0.33 | 0.29 | 0.37 | 0.37 | 2.52 | 2.40 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.57 | 0.57 | 1.04 | 1.04 | 2.00 | 2.00 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 1.08 | 1.08 | 2.04 | 2.03 |
| draft pass 1: GPU time, card 4 and card 8 | 2.99 | 2.55 | 3.38 | 3.32 | 4.30 | 4.21 |
| draft pass 1: turn | 0.53 | 0.52 | 0.60 | 0.60 | 0.74 | 0.74 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.56 | 0.56 | 1.04 | 1.04 | 2.00 | 2.00 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 1.06 | 1.06 | 2.02 | 2.02 |
| draft pass 2: GPU time, card 4 and card 8 | 3.06 | 2.55 | 3.40 | 3.33 | 4.29 | 4.16 |
| draft pass 2: turn | 0.51 | 0.50 | 0.60 | 0.60 | 0.73 | 0.72 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.56 | 0.56 | 1.04 | 1.04 | 2.00 | 2.00 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 1.06 | 1.06 | 2.02 | 2.02 |
| draft pass 3: GPU time, card 4 and card 8 | 2.93 | 2.53 | 3.34 | 3.29 | 4.25 | 4.16 |
| draft pass 3: turn (the server before the next verify pass) | 5.14 | 2.74 | 6.45 | 4.49 | 38.03 | 10.07 |
| the draft's crossings, all passes | 2.27 | 2.27 | 4.30 | 4.30 | 8.40 | 8.39 |
| the draft's turns before its last pass | 1.36 | 1.32 | 1.57 | 1.56 | 3.98 | 3.86 |
| k-pool inputs: host us a step, all passes (set_input) | 37.62 | 38.03 | nan | 363.43 | nan | nan |
| k-pool inputs: host us, the verify pass | 8.43 | 8.86 | nan | 76.64 | nan | nan |
| k-pool inputs: host us, the draft's passes | 29.19 | 29.17 | nan | 286.79 | nan | nan |
| k-pool inputs: KiB a card, the verify pass | 2.31 | 2.33 | nan | 113.90 | nan | nan |
| k-pool inputs: KiB, one draft pass (card 8) | 1.72 | 1.74 | nan | 89.40 | nan | nan |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 28.30 | 28.52 | nan | 1407.20 | nan | nan |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 9.79 | 9.87 | nan | 496.00 | nan | nan |
| k-pool inputs: n_pool of the verify pass | 78.70 | 79.32 | nan | 4160.00 | nan | nan |
| verify pass: card 0's host span | 0.33 | 0.34 | 0.22 | 0.21 | 0.41 | 0.21 |
| verify pass: card 1's host gap before entry | 0.31 | 0.30 | 0.93 | 0.92 | 2.18 | 2.14 |
| verify pass: card 2's host gap before entry | 0.29 | 0.28 | 0.89 | 0.89 | 2.15 | 2.11 |
| verify pass: card 3's host gap before entry | 0.29 | 0.28 | 0.93 | 0.89 | 2.15 | 2.11 |
| verify pass: card 4's host gap before entry | 0.24 | 0.24 | 0.83 | 0.82 | 2.02 | 2.04 |
| verify pass: card 5's host gap before entry | 0.27 | 0.28 | 0.89 | 0.88 | 2.14 | 2.12 |
| verify pass: card 6's host gap before entry | 0.27 | 0.27 | 0.89 | 0.88 | 2.14 | 2.11 |
| verify pass: card 7's host gap before entry | 0.27 | 0.27 | 0.89 | 0.88 | 2.14 | 2.11 |
| verify pass: card 8's host gap before entry | 0.33 | 0.33 | 0.95 | 0.94 | 2.20 | 2.17 |
| the server's step, this load | 125.67 | 122.04 | 168.12 | 148.80 | 227.48 | 180.31 |
| the server's step, load `warm` | 125.34 | 124.43 | 166.84 | 148.21 | 225.22 | 179.96 |

Medians of the same:

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| **the step** | 125.47 | 122.61 | 152.77 | 151.30 | 185.35 | 182.34 |
| GPU time, every evaluation | 112.88 | 112.67 | 135.16 | 134.97 | 149.82 | 149.72 |
| **outside the kernels** (the step less that) | 13.20 | 9.84 | 16.56 | 16.35 | 36.25 | 33.10 |
| verify pass: GPU time of the nine cards | 101.51 | 101.42 | 120.14 | 120.00 | 131.18 | 131.20 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.25 | 4.22 | 6.04 | 5.99 | 10.46 | 10.40 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 3.89 | 3.89 | 8.82 | 8.73 | 18.74 | 18.62 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.58 | 0.58 | 1.18 | 1.18 | 2.40 | 2.40 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.62 | 0.62 | 1.21 | 1.21 | 2.43 | 2.43 |
| draft pass 0: GPU time, card 4 and card 8 | 3.60 | 3.60 | 5.07 | 5.08 | 6.03 | 6.03 |
| draft pass 0: turn | 0.29 | 0.29 | 0.37 | 0.36 | 2.85 | 2.81 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.56 | 0.56 | 1.04 | 1.04 | 2.00 | 2.00 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.60 | 0.60 | 1.08 | 1.08 | 2.04 | 2.03 |
| draft pass 1: GPU time, card 4 and card 8 | 2.53 | 2.52 | 3.32 | 3.32 | 4.21 | 4.22 |
| draft pass 1: turn | 0.52 | 0.52 | 0.59 | 0.59 | 0.73 | 0.73 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.56 | 0.56 | 1.04 | 1.03 | 1.99 | 1.99 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 1.05 | 1.05 | 2.02 | 2.02 |
| draft pass 2: GPU time, card 4 and card 8 | 2.54 | 2.55 | 3.35 | 3.35 | 4.17 | 4.16 |
| draft pass 2: turn | 0.50 | 0.50 | 0.60 | 0.60 | 0.72 | 0.72 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.56 | 0.56 | 1.04 | 1.04 | 2.00 | 2.00 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.58 | 0.58 | 1.06 | 1.06 | 2.02 | 2.02 |
| draft pass 3: GPU time, card 4 and card 8 | 2.54 | 2.54 | 3.28 | 3.28 | 4.17 | 4.16 |
| draft pass 3: turn (the server before the next verify pass) | 5.37 | 2.02 | 4.55 | 4.47 | 13.55 | 10.04 |
| the draft's crossings, all passes | 2.26 | 2.26 | 4.30 | 4.29 | 8.39 | 8.39 |
| the draft's turns before its last pass | 1.32 | 1.31 | 1.56 | 1.56 | 4.30 | 4.27 |
| k-pool inputs: host us a step, all passes (set_input) | 37.00 | 37.00 | nan | 357.00 | nan | nan |
| k-pool inputs: host us, the verify pass | 8.00 | 8.00 | nan | 74.00 | nan | nan |
| k-pool inputs: host us, the draft's passes | 29.00 | 29.00 | nan | 282.00 | nan | nan |
| k-pool inputs: KiB a card, the verify pass | 1.90 | 1.90 | nan | 113.90 | nan | nan |
| k-pool inputs: KiB, one draft pass (card 8) | 1.40 | 1.40 | nan | 89.40 | nan | nan |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 23.20 | 23.20 | nan | 1407.20 | nan | nan |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 8.00 | 8.00 | nan | 496.00 | nan | nan |
| k-pool inputs: n_pool of the verify pass | 64.00 | 64.00 | nan | 4160.00 | nan | nan |
| verify pass: card 0's host span | 0.24 | 0.24 | 0.22 | 0.21 | 0.22 | 0.21 |
| verify pass: card 1's host gap before entry | 0.31 | 0.31 | 0.92 | 0.92 | 2.15 | 2.14 |
| verify pass: card 2's host gap before entry | 0.28 | 0.28 | 0.89 | 0.89 | 2.12 | 2.11 |
| verify pass: card 3's host gap before entry | 0.28 | 0.28 | 0.89 | 0.89 | 2.12 | 2.11 |
| verify pass: card 4's host gap before entry | 0.24 | 0.24 | 0.83 | 0.82 | 1.99 | 2.00 |
| verify pass: card 5's host gap before entry | 0.27 | 0.27 | 0.89 | 0.88 | 2.12 | 2.11 |
| verify pass: card 6's host gap before entry | 0.27 | 0.26 | 0.89 | 0.88 | 2.11 | 2.11 |
| verify pass: card 7's host gap before entry | 0.27 | 0.27 | 0.89 | 0.88 | 2.11 | 2.11 |
| verify pass: card 8's host gap before entry | 0.32 | 0.32 | 0.95 | 0.94 | 2.17 | 2.17 |

**From 16384 to 49152 tokens, ms a step** (the deeper request less the shallower, pass against pass; the share is of the step's growth):

| part | p1: 16384 | p1: 49152 | p1: growth | p1: share | p2: 16384 | p2: 49152 | p2: growth | p2: share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **the step** | 153.63 | 211.13 | +57.51 | 100.0 % | 151.27 | 182.36 | +31.09 | 100.0 % |
| GPU time, every evaluation | 135.18 | 150.21 | +15.03 | 26.1 % | 134.90 | 149.60 | +14.71 | 47.3 % |
| **outside the kernels** (the step less that) | 18.45 | 60.93 | +42.48 | 73.9 % | 16.37 | 32.76 | +16.38 | 52.7 % |
| verify pass: GPU time of the nine cards | 119.98 | 131.31 | +11.33 | 19.7 % | 119.88 | 131.05 | +11.17 | 35.9 % |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.12 | 10.51 | +4.39 | 7.6 % | 6.01 | 10.42 | +4.41 | 14.2 % |
| verify pass: the host's launch, card 0's entry to card 8's exit | 8.87 | 20.28 | +11.41 | 19.8 % | 8.74 | 18.68 | +9.94 | 32.0 % |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 1.18 | 2.40 | +1.22 | 2.1 % | 1.18 | 2.40 | +1.22 | 3.9 % |
| draft pass 0: lead (card 4's entry to card 8's entry) | 1.22 | 2.43 | +1.22 | 2.1 % | 1.22 | 2.43 | +1.22 | 3.9 % |
| draft pass 0: GPU time, card 4 and card 8 | 5.08 | 6.06 | +0.98 | 1.7 % | 5.07 | 6.02 | +0.95 | 3.0 % |
| draft pass 0: turn | 0.37 | 2.52 | +2.15 | 3.7 % | 0.37 | 2.40 | +2.03 | 6.5 % |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 1.04 | 2.00 | +0.96 | 1.7 % | 1.04 | 2.00 | +0.96 | 3.1 % |
| draft pass 1: lead (card 4's entry to card 8's entry) | 1.08 | 2.04 | +0.96 | 1.7 % | 1.08 | 2.03 | +0.96 | 3.1 % |
| draft pass 1: GPU time, card 4 and card 8 | 3.38 | 4.30 | +0.92 | 1.6 % | 3.32 | 4.21 | +0.90 | 2.9 % |
| draft pass 1: turn | 0.60 | 0.74 | +0.13 | 0.2 % | 0.60 | 0.74 | +0.14 | 0.5 % |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 1.04 | 2.00 | +0.96 | 1.7 % | 1.04 | 2.00 | +0.96 | 3.1 % |
| draft pass 2: lead (card 4's entry to card 8's entry) | 1.06 | 2.02 | +0.96 | 1.7 % | 1.06 | 2.02 | +0.96 | 3.1 % |
| draft pass 2: GPU time, card 4 and card 8 | 3.40 | 4.29 | +0.89 | 1.6 % | 3.33 | 4.16 | +0.83 | 2.7 % |
| draft pass 2: turn | 0.60 | 0.73 | +0.13 | 0.2 % | 0.60 | 0.72 | +0.12 | 0.4 % |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 1.04 | 2.00 | +0.96 | 1.7 % | 1.04 | 2.00 | +0.96 | 3.1 % |
| draft pass 3: lead (card 4's entry to card 8's entry) | 1.06 | 2.02 | +0.96 | 1.7 % | 1.06 | 2.02 | +0.96 | 3.1 % |
| draft pass 3: GPU time, card 4 and card 8 | 3.34 | 4.25 | +0.91 | 1.6 % | 3.29 | 4.16 | +0.87 | 2.8 % |
| draft pass 3: turn (the server before the next verify pass) | 6.45 | 38.03 | +31.58 | 54.9 % | 4.49 | 10.07 | +5.58 | 17.9 % |
| the draft's crossings, all passes | 4.30 | 8.40 | +4.10 | 7.1 % | 4.30 | 8.39 | +4.09 | 13.2 % |
| the draft's turns before its last pass | 1.57 | 3.98 | +2.41 | 4.2 % | 1.56 | 3.86 | +2.30 | 7.4 % |
| k-pool inputs: host us a step, all passes (set_input) | nan | nan | +nan |  | 363.43 | nan | +nan |  |
| k-pool inputs: host us, the verify pass | nan | nan | +nan |  | 76.64 | nan | +nan |  |
| k-pool inputs: host us, the draft's passes | nan | nan | +nan |  | 286.79 | nan | +nan |  |
| k-pool inputs: KiB a card, the verify pass | nan | nan | +nan |  | 113.90 | nan | +nan |  |
| k-pool inputs: KiB, one draft pass (card 8) | nan | nan | +nan |  | 89.40 | nan | +nan |  |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | nan | nan | +nan |  | 1407.20 | nan | +nan |  |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | nan | nan | +nan |  | 496.00 | nan | +nan |  |
| k-pool inputs: n_pool of the verify pass | nan | nan | +nan |  | 4160.00 | nan | +nan |  |
| verify pass: card 0's host span | 0.22 | 0.41 | +0.20 | 0.3 % | 0.21 | 0.21 | -0.00 | -0.0 % |
| verify pass: card 1's host gap before entry | 0.93 | 2.18 | +1.25 | 2.2 % | 0.92 | 2.14 | +1.22 | 3.9 % |
| verify pass: card 2's host gap before entry | 0.89 | 2.15 | +1.26 | 2.2 % | 0.89 | 2.11 | +1.22 | 3.9 % |
| verify pass: card 3's host gap before entry | 0.93 | 2.15 | +1.22 | 2.1 % | 0.89 | 2.11 | +1.22 | 3.9 % |
| verify pass: card 4's host gap before entry | 0.83 | 2.02 | +1.19 | 2.1 % | 0.82 | 2.04 | +1.22 | 3.9 % |
| verify pass: card 5's host gap before entry | 0.89 | 2.14 | +1.25 | 2.2 % | 0.88 | 2.12 | +1.23 | 4.0 % |
| verify pass: card 6's host gap before entry | 0.89 | 2.14 | +1.25 | 2.2 % | 0.88 | 2.11 | +1.23 | 4.0 % |
| verify pass: card 7's host gap before entry | 0.89 | 2.14 | +1.25 | 2.2 % | 0.88 | 2.11 | +1.23 | 3.9 % |
| verify pass: card 8's host gap before entry | 0.95 | 2.20 | +1.26 | 2.2 % | 0.94 | 2.17 | +1.23 | 3.9 % |

