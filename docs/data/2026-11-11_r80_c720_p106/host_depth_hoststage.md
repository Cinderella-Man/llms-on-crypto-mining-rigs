**Load `hoststage`:** 11628 evaluations in 14 pieces (cut at gaps over 3000 ms), 10 after joining those with fewer than 10 main passes to the next; the first is the load's own, the others are its 9 requests.

| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0 | (the load) | 9 | 1 | 0 | 2 | | | |
| 1 | warmup 0 | 258 | 13 | 49 | 59 | no | 0 | 0 |
| 2 | decode 0 p1 | 1641 | 79 | 310 | 389 | yes | 77 | 74 |
| 3 | decode 0 p2 | 1566 | 75 | 297 | 372 | yes | 74 | 71 |
| 4 | prime 16384 | 615 | 31 | 112 | 162 | no | 0 | 0 |
| 5 | decode 16384 p1 | 1575 | 75 | 301 | 356 | no | 71 | 67 |
| 6 | decode 16384 p2 | 1494 | 72 | 282 | 354 | yes | 71 | 67 |
| 7 | prime 49152 | 1353 | 65 | 259 | 327 | no | 0 | 0 |
| 8 | decode 49152 p1 | 1581 | 76 | 299 | 369 | no | 75 | 72 |
| 9 | decode 49152 p2 | 1536 | 73 | 294 | 367 | yes | 72 | 69 |

**The k-pool inputs by request, pass and micro-batch size** (one line per evaluated micro-batch; the median, and in brackets the lowest and highest):

| request | pass | tokens | micro-batches | n_pool | host us | KiB a card |
|---|---|---:|---:|---|---|---|
| (load) | ? | 2 | 2 | 64 | 20.0 (10.0, 30.0) | 3.3 (3.3, 3.3) |
| warmup 0 | ? | 60 | 2 | 64 | 22.5 (22.0, 23.0) | 12.0 (12.0, 12.0) |
| warmup 0 | ? | 4 | 24 | 64 | 10.5 (10.0, 12.0) | 3.6 (3.6, 3.6) |
| warmup 0 | ? | 1 | 33 | 64 | 7.0 (6.0, 10.0) | 3.2 (3.2, 3.2) |
| decode 0 p1 | main | 60 | 1 | 64 | 128.0 (128.0, 128.0) | 12.0 (12.0, 12.0) |
| decode 0 p1 | draft | 60 | 1 | 64 | 42.0 (42.0, 42.0) | 12.0 (12.0, 12.0) |
| decode 0 p1 | main | 4 | 77 | 64 to 128 | 10.0 (9.0, 28.0) | 3.6 (3.6, 4.1) |
| decode 0 p1 | draft | 4 | 77 | 64 to 128 | 11.0 (9.0, 39.0) | 3.6 (3.6, 4.1) |
| decode 0 p1 | draft | 1 | 231 | 64 to 128 | 7.0 (5.0, 16.0) | 3.2 (3.2, 3.3) |
| decode 0 p1 | main | 2 | 1 | 128 | 12.0 (12.0, 12.0) | 3.6 (3.6, 3.6) |
| decode 0 p1 | draft | 2 | 1 | 128 | 12.0 (12.0, 12.0) | 3.6 (3.6, 3.6) |
| decode 0 p2 | main | 4 | 74 | 64 to 128 | 10.0 (8.0, 12.0) | 3.6 (3.6, 4.1) |
| decode 0 p2 | draft | 4 | 74 | 64 to 128 | 11.0 (9.0, 17.0) | 3.6 (3.6, 4.1) |
| decode 0 p2 | draft | 1 | 222 | 64 to 128 | 7.0 (5.0, 17.0) | 3.2 (3.2, 3.3) |
| decode 0 p2 | main | 2 | 1 | 128 | 12.0 (12.0, 12.0) | 3.6 (3.6, 3.6) |
| decode 0 p2 | draft | 2 | 1 | 128 | 10.0 (10.0, 10.0) | 3.6 (3.6, 3.6) |
| prime 16384 | ? | 512 | 31 | 192 to 4096 | 905.0 (162.0, 1609.0) | 2130.5 (210.5, 4114.5) |
| prime 16384 | ? | 128 | 127 | 64 to 4096 | 136.0 (37.0, 279.0) | 534.2 (22.2, 1030.2) |
| prime 16384 | ? | 444 | 1 | 3968 | 1475.0 (1475.0, 1475.0) | 3456.4 (3456.4, 3456.4) |
| prime 16384 | ? | 60 | 1 | 3968 | 135.0 (135.0, 135.0) | 469.5 (469.5, 469.5) |
| prime 16384 | ? | 4 | 2 | 4096 | 74.0 (49.0, 99.0) | 35.1 (35.1, 35.1) |
| decode 16384 p1 | ? | 60 | 2 | 4096 | 243.0 (160.0, 326.0) | 484.5 (484.5, 484.5) |
| decode 16384 p1 | ? | 4 | 142 | 4160 | 50.0 (47.0, 86.0) | 35.6 (35.6, 35.6) |
| decode 16384 p1 | ? | 1 | 212 | 4160 | 41.0 (33.0, 74.0) | 11.2 (11.2, 11.2) |
| decode 16384 p2 | main | 4 | 71 | 4160 | 48.0 (46.0, 108.0) | 35.6 (35.6, 35.6) |
| decode 16384 p2 | draft | 4 | 71 | 4160 | 50.0 (48.0, 76.0) | 35.6 (35.6, 35.6) |
| decode 16384 p2 | draft | 1 | 211 | 4160 | 41.0 (34.0, 54.0) | 11.2 (11.2, 11.2) |
| decode 16384 p2 | main | 1 | 1 | 4160 | 67.0 (67.0, 67.0) | 11.2 (11.2, 11.2) |
| prime 49152 | ? | 512 | 65 | 4096 to 12288 | 2418.0 (1220.0, 4144.0) | 8210.5 (4114.5, 12306.5) |
| prime 49152 | ? | 128 | 260 | 4032 to 12288 | 522.0 (235.0, 980.0) | 2046.2 (1014.2, 3078.2) |
| prime 49152 | ? | 4 | 2 | 12288 | 192.5 (127.0, 258.0) | 99.1 (99.1, 99.1) |
| decode 49152 p1 | ? | 60 | 2 | 12288 | 621.0 (388.0, 854.0) | 1444.5 (1444.5, 1444.5) |
| decode 49152 p1 | ? | 4 | 146 | 12352 | 128.0 (124.0, 469.0) | 99.6 (99.6, 99.6) |
| decode 49152 p1 | ? | 1 | 219 | 12352 | 110.0 (92.0, 231.0) | 27.2 (27.2, 27.2) |
| decode 49152 p1 | ? | 3 | 2 | 12352 | 148.5 (125.0, 172.0) | 75.5 (75.5, 75.5) |
| decode 49152 p2 | main | 4 | 73 | 12352 | 125.0 (121.0, 183.0) | 99.6 (99.6, 99.6) |
| decode 49152 p2 | draft | 4 | 73 | 12352 | 127.0 (126.0, 166.0) | 99.6 (99.6, 99.6) |
| decode 49152 p2 | draft | 1 | 219 | 12352 | 109.0 (92.0, 147.0) | 27.2 (27.2, 27.2) |
| decode 49152 p2 | draft | 3 | 2 | 12352 | 138.0 (125.0, 151.0) | 75.5 (75.5, 75.5) |

**The step by part, ms** (the mean over each decode's typical steps with 4 draft passes; every column is one request of the one load):

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| typical steps | 74 | 71 | 67 | 67 | 72 | 69 |
| **the step** | 128.24 | 123.47 | 151.30 | 149.22 | 181.14 | 175.89 |
| GPU time, every evaluation | 114.66 | 113.28 | 135.20 | 135.05 | 150.00 | 149.36 |
| **outside the kernels** (the step less that) | 13.59 | 10.19 | 16.10 | 14.16 | 31.14 | 26.53 |
| verify pass: GPU time of the nine cards | 101.79 | 101.78 | 119.82 | 119.87 | 131.10 | 130.82 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.32 | 4.34 | 6.15 | 6.13 | 10.62 | 10.59 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 4.17 | 4.19 | 4.69 | 4.68 | 8.29 | 7.45 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.46 | 0.46 | 0.63 | 0.63 | 0.97 | 0.99 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.56 | 0.56 | 0.72 | 0.72 | 1.07 | 1.09 |
| draft pass 0: GPU time, card 4 and card 8 | 3.70 | 3.68 | 5.09 | 5.09 | 6.02 | 6.00 |
| draft pass 0: turn | 0.36 | 0.33 | 0.37 | 0.37 | 2.72 | 2.62 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.45 | 0.44 | 0.49 | 0.49 | 0.58 | 0.58 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.53 | 0.53 | 0.57 | 0.57 | 0.66 | 0.66 |
| draft pass 1: GPU time, card 4 and card 8 | 3.07 | 2.64 | 3.45 | 3.39 | 4.31 | 4.21 |
| draft pass 1: turn | 0.55 | 0.56 | 0.69 | 0.66 | 0.89 | 0.88 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.44 | 0.44 | 0.49 | 0.48 | 0.57 | 0.57 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.51 | 0.51 | 0.59 | 0.55 | 0.64 | 0.64 |
| draft pass 2: GPU time, card 4 and card 8 | 3.12 | 2.60 | 3.42 | 3.35 | 4.30 | 4.17 |
| draft pass 2: turn | 0.55 | 0.52 | 0.66 | 0.65 | 0.87 | 0.87 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.44 | 0.44 | 0.49 | 0.48 | 0.57 | 0.57 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.51 | 0.51 | 0.55 | 0.55 | 0.64 | 0.64 |
| draft pass 3: GPU time, card 4 and card 8 | 2.98 | 2.59 | 3.41 | 3.35 | 4.26 | 4.17 |
| draft pass 3: turn (the server before the next verify pass) | 5.91 | 2.55 | 6.01 | 4.17 | 13.22 | 8.74 |
| the draft's crossings, all passes | 1.79 | 1.79 | 2.08 | 2.08 | 2.70 | 2.72 |
| the draft's turns before its last pass | 1.46 | 1.41 | 1.71 | 1.68 | 4.49 | 4.37 |
| k-pool inputs: host us a step, all passes (set_input) | 45.32 | 43.55 | nan | 224.91 | nan | 582.94 |
| k-pool inputs: host us, the verify pass | 10.57 | 9.70 | nan | 48.58 | nan | 125.84 |
| k-pool inputs: host us, the draft's passes | 34.76 | 33.85 | nan | 176.33 | nan | 457.10 |
| k-pool inputs: KiB a card, the verify pass | 3.71 | 3.72 | nan | 35.60 | nan | 99.60 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.22 | 3.22 | nan | 11.20 | nan | 27.20 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 46.82 | 46.87 | nan | 389.60 | nan | 1077.60 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 17.10 | 17.11 | nan | 104.80 | nan | 280.80 |
| k-pool inputs: n_pool of the verify pass | 78.70 | 79.32 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.35 | 0.34 | 0.22 | 0.22 | 0.31 | 0.22 |
| verify pass: card 1's host gap before entry | 0.22 | 0.22 | 0.41 | 0.41 | 0.75 | 0.75 |
| verify pass: card 2's host gap before entry | 0.20 | 0.20 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 3's host gap before entry | 0.20 | 0.20 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 4's host gap before entry | 0.18 | 0.18 | 0.34 | 0.34 | 0.68 | 0.67 |
| verify pass: card 5's host gap before entry | 0.19 | 0.19 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 6's host gap before entry | 0.19 | 0.19 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 7's host gap before entry | 0.19 | 0.19 | 0.37 | 0.37 | 0.72 | 0.71 |
| verify pass: card 8's host gap before entry | 0.25 | 0.25 | 0.43 | 0.43 | 0.78 | 0.78 |
| the server's step, this load | 126.72 | 122.01 | 165.44 | 146.64 | 208.64 | 173.71 |
| the server's step, load `warm` | 125.46 | 121.42 | 167.38 | 148.57 | 225.44 | 180.37 |

Medians of the same:

| part | decode 0 p1 | decode 0 p2 | decode 16384 p1 | decode 16384 p2 | decode 49152 p1 | decode 49152 p2 |
|---|---:|---:|---:|---:|---:|---:|
| **the step** | 126.53 | 122.63 | 150.57 | 149.30 | 180.75 | 175.80 |
| GPU time, every evaluation | 113.34 | 113.03 | 135.20 | 135.21 | 149.68 | 149.49 |
| **outside the kernels** (the step less that) | 13.71 | 9.51 | 14.24 | 14.03 | 31.55 | 26.90 |
| verify pass: GPU time of the nine cards | 101.70 | 101.59 | 120.01 | 120.08 | 131.10 | 130.92 |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 4.31 | 4.27 | 6.13 | 6.11 | 10.61 | 10.56 |
| verify pass: the host's launch, card 0's entry to card 8's exit | 3.32 | 3.31 | 4.69 | 4.66 | 7.62 | 7.45 |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.45 | 0.45 | 0.63 | 0.62 | 0.97 | 0.97 |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.56 | 0.55 | 0.72 | 0.71 | 1.06 | 1.06 |
| draft pass 0: GPU time, card 4 and card 8 | 3.68 | 3.65 | 5.09 | 5.09 | 6.00 | 6.00 |
| draft pass 0: turn | 0.33 | 0.32 | 0.37 | 0.36 | 3.10 | 3.07 |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.44 | 0.44 | 0.48 | 0.48 | 0.57 | 0.57 |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.53 | 0.52 | 0.57 | 0.57 | 0.66 | 0.66 |
| draft pass 1: GPU time, card 4 and card 8 | 2.63 | 2.63 | 3.40 | 3.40 | 4.20 | 4.20 |
| draft pass 1: turn | 0.55 | 0.54 | 0.66 | 0.66 | 0.88 | 0.87 |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.44 | 0.44 | 0.48 | 0.48 | 0.57 | 0.57 |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.51 | 0.51 | 0.55 | 0.55 | 0.64 | 0.64 |
| draft pass 2: GPU time, card 4 and card 8 | 2.57 | 2.58 | 3.36 | 3.36 | 4.17 | 4.18 |
| draft pass 2: turn | 0.52 | 0.52 | 0.65 | 0.65 | 0.87 | 0.86 |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.44 | 0.44 | 0.48 | 0.48 | 0.57 | 0.57 |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.50 | 0.50 | 0.55 | 0.55 | 0.64 | 0.64 |
| draft pass 3: GPU time, card 4 and card 8 | 2.58 | 2.58 | 3.36 | 3.36 | 4.18 | 4.16 |
| draft pass 3: turn (the server before the next verify pass) | 6.13 | 1.96 | 4.09 | 4.03 | 13.55 | 8.73 |
| the draft's crossings, all passes | 1.78 | 1.78 | 2.08 | 2.08 | 2.69 | 2.69 |
| the draft's turns before its last pass | 1.40 | 1.39 | 1.68 | 1.67 | 4.86 | 4.81 |
| k-pool inputs: host us a step, all passes (set_input) | 44.00 | 43.00 | nan | 222.00 | nan | 579.00 |
| k-pool inputs: host us, the verify pass | 10.00 | 10.00 | nan | 48.00 | nan | 125.00 |
| k-pool inputs: host us, the draft's passes | 34.00 | 33.00 | nan | 173.00 | nan | 451.00 |
| k-pool inputs: KiB a card, the verify pass | 3.60 | 3.60 | nan | 35.60 | nan | 99.60 |
| k-pool inputs: KiB, one draft pass (card 8) | 3.20 | 3.20 | nan | 11.20 | nan | 27.20 |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | 45.60 | 45.60 | nan | 389.60 | nan | 1077.60 |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | 16.80 | 16.80 | nan | 104.80 | nan | 280.80 |
| k-pool inputs: n_pool of the verify pass | 64.00 | 64.00 | nan | 4160.00 | nan | 12352.00 |
| verify pass: card 0's host span | 0.25 | 0.25 | 0.22 | 0.22 | 0.22 | 0.22 |
| verify pass: card 1's host gap before entry | 0.22 | 0.22 | 0.41 | 0.41 | 0.75 | 0.75 |
| verify pass: card 2's host gap before entry | 0.20 | 0.19 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 3's host gap before entry | 0.20 | 0.19 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 4's host gap before entry | 0.17 | 0.17 | 0.34 | 0.34 | 0.67 | 0.67 |
| verify pass: card 5's host gap before entry | 0.19 | 0.19 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 6's host gap before entry | 0.19 | 0.19 | 0.37 | 0.37 | 0.72 | 0.72 |
| verify pass: card 7's host gap before entry | 0.19 | 0.19 | 0.37 | 0.37 | 0.72 | 0.71 |
| verify pass: card 8's host gap before entry | 0.24 | 0.25 | 0.43 | 0.43 | 0.78 | 0.77 |

**From 16384 to 49152 tokens, ms a step** (the deeper request less the shallower, pass against pass; the share is of the step's growth):

| part | p1: 16384 | p1: 49152 | p1: growth | p1: share | p2: 16384 | p2: 49152 | p2: growth | p2: share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **the step** | 151.30 | 181.14 | +29.84 | 100.0 % | 149.22 | 175.89 | +26.67 | 100.0 % |
| GPU time, every evaluation | 135.20 | 150.00 | +14.80 | 49.6 % | 135.05 | 149.36 | +14.30 | 53.6 % |
| **outside the kernels** (the step less that) | 16.10 | 31.14 | +15.04 | 50.4 % | 14.16 | 26.53 | +12.37 | 46.4 % |
| verify pass: GPU time of the nine cards | 119.82 | 131.10 | +11.28 | 37.8 % | 119.87 | 130.82 | +10.95 | 41.0 % |
| verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft) | 6.15 | 10.62 | +4.47 | 15.0 % | 6.13 | 10.59 | +4.46 | 16.7 % |
| verify pass: the host's launch, card 0's entry to card 8's exit | 4.69 | 8.29 | +3.59 | 12.0 % | 4.68 | 7.45 | +2.77 | 10.4 % |
| draft pass 0 (catch-up): crossing (card 4's exit to card 8's entry) | 0.63 | 0.97 | +0.35 | 1.2 % | 0.63 | 0.99 | +0.37 | 1.4 % |
| draft pass 0: lead (card 4's entry to card 8's entry) | 0.72 | 1.07 | +0.35 | 1.2 % | 0.72 | 1.09 | +0.37 | 1.4 % |
| draft pass 0: GPU time, card 4 and card 8 | 5.09 | 6.02 | +0.93 | 3.1 % | 5.09 | 6.00 | +0.91 | 3.4 % |
| draft pass 0: turn | 0.37 | 2.72 | +2.36 | 7.9 % | 0.37 | 2.62 | +2.26 | 8.5 % |
| draft pass 1 (draft 1): crossing (card 4's exit to card 8's entry) | 0.49 | 0.58 | +0.09 | 0.3 % | 0.49 | 0.58 | +0.09 | 0.3 % |
| draft pass 1: lead (card 4's entry to card 8's entry) | 0.57 | 0.66 | +0.09 | 0.3 % | 0.57 | 0.66 | +0.09 | 0.3 % |
| draft pass 1: GPU time, card 4 and card 8 | 3.45 | 4.31 | +0.86 | 2.9 % | 3.39 | 4.21 | +0.82 | 3.1 % |
| draft pass 1: turn | 0.69 | 0.89 | +0.21 | 0.7 % | 0.66 | 0.88 | +0.22 | 0.8 % |
| draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry) | 0.49 | 0.57 | +0.09 | 0.3 % | 0.48 | 0.57 | +0.09 | 0.3 % |
| draft pass 2: lead (card 4's entry to card 8's entry) | 0.59 | 0.64 | +0.06 | 0.2 % | 0.55 | 0.64 | +0.09 | 0.3 % |
| draft pass 2: GPU time, card 4 and card 8 | 3.42 | 4.30 | +0.87 | 2.9 % | 3.35 | 4.17 | +0.81 | 3.0 % |
| draft pass 2: turn | 0.66 | 0.87 | +0.22 | 0.7 % | 0.65 | 0.87 | +0.22 | 0.8 % |
| draft pass 3 (draft 3): crossing (card 4's exit to card 8's entry) | 0.49 | 0.57 | +0.09 | 0.3 % | 0.48 | 0.57 | +0.09 | 0.3 % |
| draft pass 3: lead (card 4's entry to card 8's entry) | 0.55 | 0.64 | +0.09 | 0.3 % | 0.55 | 0.64 | +0.09 | 0.3 % |
| draft pass 3: GPU time, card 4 and card 8 | 3.41 | 4.26 | +0.85 | 2.9 % | 3.35 | 4.17 | +0.82 | 3.1 % |
| draft pass 3: turn (the server before the next verify pass) | 6.01 | 13.22 | +7.21 | 24.2 % | 4.17 | 8.74 | +4.57 | 17.1 % |
| the draft's crossings, all passes | 2.08 | 2.70 | +0.61 | 2.1 % | 2.08 | 2.72 | +0.64 | 2.4 % |
| the draft's turns before its last pass | 1.71 | 4.49 | +2.78 | 9.3 % | 1.68 | 4.37 | +2.69 | 10.1 % |
| k-pool inputs: host us a step, all passes (set_input) | nan | nan | +nan |  | 224.91 | 582.94 | +358.03 |  |
| k-pool inputs: host us, the verify pass | nan | nan | +nan |  | 48.58 | 125.84 | +77.26 |  |
| k-pool inputs: host us, the draft's passes | nan | nan | +nan |  | 176.33 | 457.10 | +280.77 |  |
| k-pool inputs: KiB a card, the verify pass | nan | nan | +nan |  | 35.60 | 99.60 | +64.00 |  |
| k-pool inputs: KiB, one draft pass (card 8) | nan | nan | +nan |  | 11.20 | 27.20 | +16.00 |  |
| k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass) | nan | nan | +nan |  | 389.60 | 1077.60 | +688.00 |  |
| k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's) | nan | nan | +nan |  | 104.80 | 280.80 | +176.00 |  |
| k-pool inputs: n_pool of the verify pass | nan | nan | +nan |  | 4160.00 | 12352.00 | +8192.00 |  |
| verify pass: card 0's host span | 0.22 | 0.31 | +0.09 | 0.3 % | 0.22 | 0.22 | +0.00 | 0.0 % |
| verify pass: card 1's host gap before entry | 0.41 | 0.75 | +0.35 | 1.2 % | 0.41 | 0.75 | +0.35 | 1.3 % |
| verify pass: card 2's host gap before entry | 0.37 | 0.72 | +0.35 | 1.2 % | 0.37 | 0.72 | +0.35 | 1.3 % |
| verify pass: card 3's host gap before entry | 0.37 | 0.72 | +0.35 | 1.2 % | 0.37 | 0.72 | +0.35 | 1.3 % |
| verify pass: card 4's host gap before entry | 0.34 | 0.68 | +0.33 | 1.1 % | 0.34 | 0.67 | +0.33 | 1.2 % |
| verify pass: card 5's host gap before entry | 0.37 | 0.72 | +0.36 | 1.2 % | 0.37 | 0.72 | +0.35 | 1.3 % |
| verify pass: card 6's host gap before entry | 0.37 | 0.72 | +0.36 | 1.2 % | 0.37 | 0.72 | +0.35 | 1.3 % |
| verify pass: card 7's host gap before entry | 0.37 | 0.72 | +0.36 | 1.2 % | 0.37 | 0.71 | +0.35 | 1.3 % |
| verify pass: card 8's host gap before entry | 0.43 | 0.78 | +0.35 | 1.2 % | 0.43 | 0.78 | +0.35 | 1.3 % |

